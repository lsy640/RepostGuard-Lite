from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from repostguard.checkpoint import (
    atomic_text,
    atomic_torch_save,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
)
from repostguard.config import config_digest, load_config, save_resolved_config
from repostguard.data.dataset import build_eval_loader, build_train_loader
from repostguard.losses import compute_training_loss
from repostguard.metrics import binary_metrics, select_balanced_threshold
from repostguard.models import build_model, count_parameters


_STOP_REQUESTED = False


def _request_safe_stop(signum: int, frame: object) -> None:
    del signum, frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _set_seed(seed: int, deterministic: bool) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def _scheduler(optimizer: AdamW, total_steps: int, warmup_ratio: float, minimum: float) -> LambdaLR:
    warmup_steps = max(1, round(total_steps * warmup_ratio))

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return minimum + (1.0 - minimum) * cosine

    return LambdaLR(optimizer, multiplier)


def _checkpoint_payload(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    config: dict[str, Any],
    epoch: int,
    batch_in_epoch: int,
    global_step: int,
    best_metric: float,
    sampler_state: torch.Tensor | None,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "rng_state": capture_rng_state(),
        "sampler_state": sampler_state,
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "config_sha256": config_digest(config),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "saved_at_unix": time.time(),
    }


def _predict_clean(
    model: nn.Module,
    config: dict[str, Any],
    device: torch.device,
    amp_enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    loader = build_eval_loader(config, {"name": "clean", "params": {}})
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)["logits"]
            labels.append(batch["label"].numpy())
            probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def train(config_path: str, resume_override: str | None = None) -> int:
    config = load_config(config_path)
    output_directory = Path(config["output"]["directory"]).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    done_marker = output_directory / "DONE"
    if done_marker.exists():
        print(f"DONE marker exists; refusing to retrain: {done_marker}", flush=True)
        return 0
    save_resolved_config(config, output_directory / "resolved_config.yaml")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this SLURM pilot job")
    device = torch.device("cuda")
    seed = int(config["seed"])
    _set_seed(seed, bool(config.get("deterministic", True)))
    signal.signal(signal.SIGUSR1, _request_safe_stop)

    model = build_model(config).to(device)
    parameter_counts = count_parameters(model)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(
        trainable_parameters,
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    train_loader = build_train_loader(config)
    accumulation = int(config["train"]["gradient_accumulation"])
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = optimizer_steps_per_epoch * int(config["train"]["epochs"])
    scheduler = _scheduler(
        optimizer,
        total_steps,
        float(config["train"]["warmup_ratio"]),
        float(config["train"]["min_learning_rate_ratio"]),
    )
    amp_enabled = bool(config["train"]["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 0
    resume_batch_in_epoch = 0
    global_step = 0
    best_metric = float("-inf")
    resume_value = resume_override or str(config["train"].get("resume", "auto"))
    resume_path = output_directory / "latest.pt" if resume_value == "auto" else Path(resume_value)
    if resume_value != "none" and resume_path.is_file():
        checkpoint = load_checkpoint(resume_path)
        if checkpoint.get("config_sha256") != config_digest(config):
            raise ValueError("Resume checkpoint config does not match the requested config")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint["rng_state"])
        sampler_generator = getattr(train_loader.sampler, "generator", None)
        if sampler_generator is not None and checkpoint.get("sampler_state") is not None:
            sampler_generator.set_state(checkpoint["sampler_state"])
        start_epoch = int(checkpoint["epoch"])
        resume_batch_in_epoch = int(checkpoint.get("batch_in_epoch", 0))
        global_step = int(checkpoint["global_step"])
        best_metric = float(checkpoint["best_metric"])
        print(
            f"Resumed {resume_path} at epoch={start_epoch} "
            f"batch_in_epoch={resume_batch_in_epoch} global_step={global_step}",
            flush=True,
        )

    experiment = str(config["model"]["experiment"]).lower()
    print(
        json.dumps(
            {
                "event": "start",
                "experiment": experiment,
                "device": torch.cuda.get_device_name(device),
                "parameters": parameter_counts,
                "train_samples": len(train_loader.dataset),
                "optimizer_steps": total_steps,
                "config_sha256": config_digest(config),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, int(config["train"]["epochs"])):
        model.train()
        running: dict[str, float] = {}
        batches_since_log = 0
        sampler_generator = getattr(train_loader.sampler, "generator", None)
        epoch_sampler_state = (
            sampler_generator.get_state() if sampler_generator is not None else None
        )
        for batch_index, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < resume_batch_in_epoch:
                continue
            labels = batch["label"].to(device, non_blocking=True)
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                clean_output = model(images)
                augmented_output = None
                if experiment in {"m2", "m3"}:
                    augmented_images = batch["image_aug"].to(device, non_blocking=True)
                    augmented_output = model(augmented_images)
                loss, components = compute_training_loss(
                    experiment,
                    labels,
                    clean_output,
                    augmented_output,
                    lambda_kl=float(config["train"]["lambda_kl"]),
                    lambda_feature=float(config["train"]["lambda_feature"]),
                )
                scaled_loss = loss / accumulation
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(
                    "Non-finite training loss at "
                    f"epoch={epoch} batch={batch_index} global_step={global_step}: "
                    f"{json.dumps(components, sort_keys=True)}"
                )
            scaler.scale(scaled_loss).backward()

            should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader)
            optimizer_step_succeeded = False
            if should_step:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, float(config["train"]["grad_clip_norm"])
                )
                gradient_is_finite = bool(torch.isfinite(gradient_norm).item())
                if not gradient_is_finite and not amp_enabled:
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        "Non-finite gradient norm at "
                        f"epoch={epoch} batch={batch_index} global_step={global_step}"
                    )
                scale_before = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                scale_after = float(scaler.get_scale())
                optimizer.zero_grad(set_to_none=True)
                if gradient_is_finite:
                    scheduler.step()
                    global_step += 1
                    optimizer_step_succeeded = True
                else:
                    print(
                        json.dumps(
                            {
                                "event": "amp_gradient_overflow",
                                "epoch": epoch,
                                "batch": batch_index,
                                "global_step": global_step,
                                "scale_before": scale_before,
                                "scale_after": scale_after,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

            for key, value in components.items():
                running[key] = running.get(key, 0.0) + value
            batches_since_log += 1
            if optimizer_step_succeeded and global_step % int(config["train"]["log_every_steps"]) == 0:
                payload = {
                    "event": "train",
                    "epoch": epoch,
                    "global_step": global_step,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                payload.update({key: value / batches_since_log for key, value in running.items()})
                print(json.dumps(payload, sort_keys=True), flush=True)
                running.clear()
                batches_since_log = 0

            checkpoint_due = (
                optimizer_step_succeeded
                and global_step % int(config["train"]["checkpoint_every_steps"]) == 0
            )
            safe_stop_due = _STOP_REQUESTED and should_step
            if checkpoint_due or safe_stop_due:
                payload = _checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    config,
                    epoch,
                    batch_index + 1,
                    global_step,
                    best_metric,
                    epoch_sampler_state,
                )
                atomic_torch_save(payload, output_directory / "latest.pt")
                if safe_stop_due:
                    print("SIGUSR1 checkpoint completed at a safe batch boundary", flush=True)
                    return 75

        labels, probabilities = _predict_clean(model, config, device, amp_enabled)
        threshold = select_balanced_threshold(labels, probabilities)
        validation = binary_metrics(labels, probabilities, threshold)
        validation["event"] = "validation"
        validation["epoch"] = epoch
        print(json.dumps(validation, sort_keys=True), flush=True)
        current_metric = float(validation["auroc"])
        if current_metric > best_metric:
            best_metric = current_metric
            sampler_generator = getattr(train_loader.sampler, "generator", None)
            sampler_state = sampler_generator.get_state() if sampler_generator is not None else None
            best_payload = _checkpoint_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                config,
                epoch + 1,
                0,
                global_step,
                best_metric,
                sampler_state,
            )
            best_payload["validation_metrics"] = validation
            atomic_torch_save(best_payload, output_directory / "best.pt")

        sampler_generator = getattr(train_loader.sampler, "generator", None)
        sampler_state = sampler_generator.get_state() if sampler_generator is not None else None
        latest_payload = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            config,
            epoch + 1,
            0,
            global_step,
            best_metric,
            sampler_state,
        )
        latest_payload["validation_metrics"] = validation
        atomic_torch_save(latest_payload, output_directory / "latest.pt")
        resume_batch_in_epoch = 0

    atomic_text(
        json.dumps(
            {
                "experiment": experiment,
                "best_clean_auroc": best_metric,
                "global_step": global_step,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
            sort_keys=True,
        )
        + "\n",
        done_marker,
    )
    print(f"Training complete; best clean AUROC={best_metric:.6f}", flush=True)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one RepostGuard pilot experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="auto, none, or checkpoint path")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    sys.exit(train(arguments.config, arguments.resume))


if __name__ == "__main__":
    main()
