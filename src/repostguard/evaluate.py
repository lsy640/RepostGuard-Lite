from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import random
import socket
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
import yaml
from sklearn.metrics import roc_auc_score
from torch import nn

from repostguard.checkpoint import atomic_text, load_checkpoint
from repostguard.config import config_digest, load_config
from repostguard.data.dataset import build_eval_loader
from repostguard.metrics import binary_metrics, select_balanced_threshold
from repostguard.models import build_model, count_parameters
from repostguard.models.quality_gate import QUALITY_FEATURE_NAMES


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


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _transform_key(specification: dict[str, Any]) -> str:
    name = str(specification["name"])
    parameters = specification.get("params", {})
    if not parameters:
        return name
    if "transforms" in parameters:
        parts = []
        for transform in parameters["transforms"]:
            parts.append(_transform_key(transform))
        return "__".join(parts)
    suffix = "_".join(f"{key}={parameters[key]}" for key in sorted(parameters) if key != "seed")
    return f"{name}__{suffix}" if suffix else name


def _load_matrix(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    matrix = payload.get("evaluation", [])
    if not matrix or matrix[0].get("name") != "clean":
        raise ValueError("Evaluation matrix must start with a clean condition")
    return matrix


def _build_summary(
    experiment: str,
    threshold: float,
    metrics_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize clean and robustness metrics, including clean-only evaluations."""
    clean = metrics_rows[0]
    robust = metrics_rows[1:]
    summary: dict[str, Any] = {
        "experiment": experiment,
        "threshold_from_clean_validation": threshold,
        "clean_auroc": clean["auroc"],
        "clean_balanced_accuracy": clean["balanced_accuracy"],
        "robust_mean_auroc": None,
        "robust_worst_auroc": None,
        "robust_worst_transform": None,
        "delta_auroc": None,
        "robust_mean_balanced_accuracy": None,
        "conditions": len(metrics_rows),
    }
    if not robust:
        return summary

    robust_aurocs = [float(row["auroc"]) for row in robust]
    robust_balanced = [float(row["balanced_accuracy"]) for row in robust]
    worst_index = int(np.argmin(robust_aurocs))
    robust_mean_auroc = float(np.mean(robust_aurocs))
    summary.update(
        {
            "robust_mean_auroc": robust_mean_auroc,
            "robust_worst_auroc": robust_aurocs[worst_index],
            "robust_worst_transform": robust[worst_index]["transform"],
            "delta_auroc": float(clean["auroc"] - robust_mean_auroc),
            "robust_mean_balanced_accuracy": float(np.mean(robust_balanced)),
        }
    )
    return summary


def _predict(
    model: nn.Module,
    config: dict[str, Any],
    transform_spec: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    loader = build_eval_loader(config, transform_spec)
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    amp_enabled = bool(config["train"]["amp"])
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                output = model(images)
                logits = output["logits"]
            probabilities = torch.sigmoid(logits).float().cpu().numpy()
            gate_fractions = output.get("gate_fractions")
            quality_features = output.get("quality_features")
            if gate_fractions is not None:
                gate_fractions = gate_fractions.float().cpu().numpy()
            if quality_features is not None:
                quality_features = quality_features.float().cpu().numpy()
            labels = batch["label"].numpy()
            all_labels.append(labels)
            all_probabilities.append(probabilities)
            for index, probability in enumerate(probabilities):
                record: dict[str, Any] = {
                    "sample_id": batch["sample_id"][index],
                    "image_path": batch["path"][index],
                    "label": int(labels[index]),
                    "pred": float(probability),
                    "transform": _transform_key(transform_spec),
                    "transform_params": transform_spec.get("params", {}),
                }
                if gate_fractions is not None and quality_features is not None:
                    record["semantic_gate_fraction"] = float(gate_fractions[index, 0])
                    record["forensic_gate_fraction"] = float(gate_fractions[index, 1])
                    record["quality_features"] = {
                        name: float(value)
                        for name, value in zip(
                            QUALITY_FEATURE_NAMES, quality_features[index], strict=True
                        )
                    }
                records.append(record)
    return np.concatenate(all_labels), np.concatenate(all_probabilities), records


def evaluate(
    config_path: str,
    checkpoint_path: str,
    *,
    matrix_path: str | None = None,
    output_directory_override: str | None = None,
    val_manifest_override: str | None = None,
    threshold_override: float | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_config_sha256 = config_digest(config)
    if checkpoint.get("config_sha256") != checkpoint_config_sha256:
        raise ValueError("Checkpoint and evaluation config digests differ")
    evaluation_overrides: dict[str, Any] = {}
    if val_manifest_override is not None or threshold_override is not None:
        config = copy.deepcopy(config)
    if val_manifest_override is not None:
        config["data"]["val_manifest"] = str(val_manifest_override)
        evaluation_overrides["val_manifest"] = str(Path(val_manifest_override).resolve())
    if threshold_override is not None:
        config["evaluation"]["threshold"] = float(threshold_override)
        evaluation_overrides["threshold"] = float(threshold_override)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this SLURM evaluation job")
    device = torch.device("cuda")
    _set_seed(int(config["seed"]), bool(config.get("deterministic", True)))
    model = build_model(config, load_pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)

    resolved_matrix_path = matrix_path or config["evaluation"]["matrix"]
    matrix = _load_matrix(resolved_matrix_path)
    output_directory = Path(
        output_directory_override or config["output"]["directory"]
    ).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    prediction_path = output_directory / "predictions.jsonl"
    metrics_rows: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    threshold: float | None = None
    for transform_spec in matrix:
        labels, probabilities, records = _predict(model, config, transform_spec, device)
        if threshold is None:
            configured_threshold = config["evaluation"].get("threshold", "auto")
            threshold = (
                select_balanced_threshold(labels, probabilities)
                if configured_threshold == "auto"
                else float(configured_threshold)
            )
        metrics = binary_metrics(labels, probabilities, threshold)
        metrics["transform"] = _transform_key(transform_spec)
        metrics["transform_name"] = transform_spec["name"]
        metrics["transform_params"] = json.dumps(
            transform_spec.get("params", {}), sort_keys=True, separators=(",", ":")
        )
        if records and "semantic_gate_fraction" in records[0]:
            metrics["mean_semantic_gate_fraction"] = float(
                np.mean([record["semantic_gate_fraction"] for record in records])
            )
            metrics["mean_forensic_gate_fraction"] = float(
                np.mean([record["forensic_gate_fraction"] for record in records])
            )
        metrics_rows.append(metrics)
        all_records.extend(records)
        print(json.dumps({"event": "evaluation", **metrics}, sort_keys=True), flush=True)

    with prediction_path.open("w", encoding="utf-8") as handle:
        for record in all_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    if all_records and "quality_features" in all_records[0]:
        clean_records = [record for record in all_records if record["transform"] == "clean"]
        clean_labels = np.asarray([record["label"] for record in clean_records])
        quality_audit: dict[str, Any] = {
            "n": len(clean_records),
            "features": {},
            "note": "Quality features control only the gate and never enter the classifier directly.",
        }
        for feature_name in QUALITY_FEATURE_NAMES:
            values = np.asarray(
                [record["quality_features"][feature_name] for record in clean_records]
            )
            raw_auroc = float(roc_auc_score(clean_labels, values))
            quality_audit["features"][feature_name] = {
                "raw_label_auroc": raw_auroc,
                "label_separability_auroc": max(raw_auroc, 1.0 - raw_auroc),
                "real_mean": float(values[clean_labels == 0].mean()),
                "aigc_mean": float(values[clean_labels == 1].mean()),
            }
        atomic_text(
            json.dumps(quality_audit, indent=2, sort_keys=True) + "\n",
            output_directory / "quality_gate_audit.json",
        )

    metric_path = output_directory / "metrics_by_transform.csv"
    fieldnames = list(metrics_rows[0].keys())
    with metric_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)

    summary = _build_summary(config["model"]["experiment"], threshold, metrics_rows)
    atomic_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", output_directory / "summary.json")

    run_card = {
        "experiment": config["model"]["experiment"],
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": checkpoint_config_sha256,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "evaluation_matrix_path": str(Path(resolved_matrix_path).resolve()),
        "evaluation_matrix_sha256": _sha256(resolved_matrix_path),
        "output_directory": str(output_directory),
        "evaluation_overrides": evaluation_overrides,
        "train_manifest_sha256": _sha256(config["data"]["train_manifest"]),
        "val_manifest_sha256": _sha256(config["data"]["val_manifest"]),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "parameters": count_parameters(model),
        "summary": summary,
    }
    atomic_text(json.dumps(run_card, indent=2, sort_keys=True) + "\n", output_directory / "run_card.json")
    print(json.dumps({"event": "summary", **summary}, sort_keys=True), flush=True)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RepostGuard across fixed transforms")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--matrix",
        help="Optional evaluation matrix override; does not alter checkpoint config validation",
    )
    parser.add_argument(
        "--output-directory",
        help="Optional output directory override used to preserve earlier evaluation artifacts",
    )
    parser.add_argument(
        "--val-manifest",
        help="Optional validation/test manifest override applied after checkpoint digest validation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Optional fixed probability threshold applied after checkpoint digest validation",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    evaluate(
        arguments.config,
        arguments.checkpoint,
        matrix_path=arguments.matrix,
        output_directory_override=arguments.output_directory,
        val_manifest_override=arguments.val_manifest,
        threshold_override=arguments.threshold,
    )


if __name__ == "__main__":
    main()
