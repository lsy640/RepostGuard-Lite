from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import socket
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
import yaml
from sklearn.metrics import roc_auc_score

from repostguard.checkpoint import atomic_text, load_checkpoint
from repostguard.config import config_digest, load_config
from repostguard.data.dataset import build_eval_loader
from repostguard.evaluate import _load_matrix, _set_seed, _sha256, _transform_key
from repostguard.metrics import binary_metrics
from repostguard.models import build_model, count_parameters
from repostguard.models.detectors import RepostGuardM3


SUPPORTED_MODES = ("learned", "fixed_equal", "fixed_clean_mean", "shuffled")


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    fieldnames = list(rows[0])
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _manifest_counts(path: str | Path) -> tuple[int, int, int]:
    rows = _read_csv(path)
    real = sum(int(row["label"]) == 0 for row in rows)
    aigi = sum(int(row["label"]) == 1 for row in rows)
    return len(rows), real, aigi


def _gate_statistics(gates: torch.Tensor) -> dict[str, float]:
    semantic = gates[:, 0].float().numpy()
    clipped = gates.float().clamp_min(1e-12)
    entropy = -(clipped * clipped.log()).sum(dim=1).numpy()
    return {
        "semantic_gate_mean": float(semantic.mean()),
        "semantic_gate_std": float(semantic.std(ddof=0)),
        "semantic_gate_min": float(semantic.min()),
        "semantic_gate_max": float(semantic.max()),
        "gate_entropy_mean": float(entropy.mean()),
    }


def _extract_branches(
    model: RepostGuardM3,
    config: dict[str, Any],
    transform_spec: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, list[str], list[str], np.ndarray]:
    loader = build_eval_loader(config, transform_spec)
    semantic_batches: list[torch.Tensor] = []
    forensic_batches: list[torch.Tensor] = []
    gate_batches: list[torch.Tensor] = []
    learned_probability_batches: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sample_ids: list[str] = []
    paths: list[str] = []
    amp_enabled = bool(config["train"]["amp"])
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                branches = model.extract_branch_features(images)
                learned_output = model.fuse_branch_features(
                    branches["semantic_features"],
                    branches["forensic_features"],
                    branches["gate_fractions"],
                )
            semantic_batches.append(branches["semantic_features"].detach().cpu())
            forensic_batches.append(branches["forensic_features"].detach().cpu())
            gate_batches.append(branches["gate_fractions"].detach().cpu())
            learned_probability_batches.append(
                torch.sigmoid(learned_output["logits"]).float().cpu().numpy()
            )
            labels.append(batch["label"].numpy())
            sample_ids.extend(str(value) for value in batch["sample_id"])
            paths.extend(str(value) for value in batch["path"])
    return (
        torch.cat(semantic_batches),
        torch.cat(forensic_batches),
        torch.cat(gate_batches),
        np.concatenate(labels),
        sample_ids,
        paths,
        np.concatenate(learned_probability_batches),
    )


def _probabilities_from_gates(
    model: RepostGuardM3,
    semantic: torch.Tensor,
    forensic: torch.Tensor,
    gates: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
    amp_enabled: bool,
) -> np.ndarray:
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, semantic.shape[0], batch_size):
            stop = min(start + batch_size, semantic.shape[0])
            semantic_batch = semantic[start:stop].to(device, non_blocking=True)
            forensic_batch = forensic[start:stop].to(device, non_blocking=True)
            gate_batch = gates[start:stop].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                output = model.fuse_branch_features(
                    semantic_batch,
                    forensic_batch,
                    gate_batch,
                )
            probabilities.append(torch.sigmoid(output["logits"]).float().cpu().numpy())
    return np.concatenate(probabilities)


def _paired_bootstrap(
    labels: np.ndarray,
    learned: np.ndarray,
    ablated: np.ndarray,
    threshold: float,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    real_indices = np.flatnonzero(labels == 0)
    aigi_indices = np.flatnonzero(labels == 1)
    if not len(real_indices) or not len(aigi_indices):
        raise ValueError("paired bootstrap requires both classes")
    rng = np.random.default_rng(seed)
    auroc_deltas = np.empty(replicates, dtype=np.float64)
    accuracy_deltas = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = np.concatenate(
            (
                rng.choice(real_indices, size=len(real_indices), replace=True),
                rng.choice(aigi_indices, size=len(aigi_indices), replace=True),
            )
        )
        sampled_labels = labels[sampled]
        learned_sample = learned[sampled]
        ablated_sample = ablated[sampled]
        auroc_deltas[replicate] = roc_auc_score(sampled_labels, learned_sample) - roc_auc_score(
            sampled_labels, ablated_sample
        )
        learned_accuracy = np.mean((learned_sample >= threshold) == sampled_labels)
        ablated_accuracy = np.mean((ablated_sample >= threshold) == sampled_labels)
        accuracy_deltas[replicate] = learned_accuracy - ablated_accuracy
    return {
        "replicates": replicates,
        "seed": seed,
        "auroc_delta_ci_low": float(np.quantile(auroc_deltas, 0.025)),
        "auroc_delta_ci_high": float(np.quantile(auroc_deltas, 0.975)),
        "accuracy_delta_ci_low": float(np.quantile(accuracy_deltas, 0.025)),
        "accuracy_delta_ci_high": float(np.quantile(accuracy_deltas, 0.975)),
    }


def _reconcile_learned(
    rows: list[dict[str, Any]],
    reference_path: str | Path,
    tolerance: float,
) -> dict[str, Any]:
    reference = _read_csv(reference_path)
    learned = [row for row in rows if row["mode"] == "learned"]
    if len(reference) != len(learned):
        raise RuntimeError("learned/reference condition count differs")
    fields = ("auroc", "average_precision", "balanced_accuracy", "aigc_recall", "real_specificity")
    maximum = 0.0
    per_condition: list[dict[str, Any]] = []
    for index, (candidate, expected) in enumerate(zip(learned, reference, strict=True)):
        if candidate["transform"] != expected["transform"]:
            raise RuntimeError(f"learned/reference transform mismatch at condition {index}")
        differences = {field: abs(float(candidate[field]) - float(expected[field])) for field in fields}
        maximum = max(maximum, *differences.values())
        per_condition.append({"condition_index": index, "transform": candidate["transform"], **differences})
    if maximum > tolerance:
        raise RuntimeError(f"learned-mode reconciliation exceeded tolerance: {maximum} > {tolerance}")
    return {"reference_metrics": str(reference_path), "tolerance": tolerance, "maximum_absolute_difference": maximum, "conditions": per_condition}


def evaluate(protocol_path: str) -> None:
    protocol = _read_yaml(protocol_path)
    modes = tuple(protocol["evaluation"]["modes"])
    if modes != SUPPORTED_MODES:
        raise ValueError(f"modes must be exactly {SUPPORTED_MODES}")
    config_path = Path(protocol["model"]["config"])
    checkpoint_path = Path(protocol["model"]["checkpoint"])
    manifest_path = Path(protocol["dataset"]["manifest"])
    matrix_path = Path(protocol["evaluation"]["matrix"])
    output_directory = Path(protocol["output"]["directory"])
    output_directory.mkdir(parents=True, exist_ok=True)

    expected_counts = (
        int(protocol["dataset"]["expected_rows"]),
        int(protocol["dataset"]["expected_real"]),
        int(protocol["dataset"]["expected_aigi"]),
    )
    if _manifest_counts(manifest_path) != expected_counts:
        raise RuntimeError("selected manifest count/class balance differs from protocol")
    matrix = _load_matrix(matrix_path)
    if len(matrix) != int(protocol["evaluation"]["expected_conditions"]):
        raise RuntimeError("evaluation matrix condition count differs from protocol")
    threshold_summary = _read_json(protocol["model"]["threshold_summary"])
    threshold = float(threshold_summary["threshold_from_clean_validation"])

    config = load_config(config_path)
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_config_sha256 = config_digest(config)
    if checkpoint.get("config_sha256") != checkpoint_config_sha256:
        raise ValueError("checkpoint and evaluation config digests differ")
    config = copy.deepcopy(config)
    config["data"]["val_manifest"] = str(manifest_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the M3 gate ablation")
    device = torch.device("cuda")
    _set_seed(int(config["seed"]), bool(config.get("deterministic", True)))
    built_model = build_model(config).to(device)
    if not isinstance(built_model, RepostGuardM3):
        raise TypeError("gate ablation requires a RepostGuardM3 checkpoint")
    model = built_model
    model.load_state_dict(checkpoint["model"], strict=True)

    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    fixed_clean_semantic_fraction: float | None = None
    bootstrap_conditions = {int(value) for value in protocol["bootstrap"]["conditions"]}
    bootstrap_replicates = int(protocol["bootstrap"]["replicates"])
    bootstrap_seed = int(protocol["bootstrap"]["seed"])
    shuffle_seed = int(protocol["evaluation"]["shuffle_seed"])

    for condition_index, transform_spec in enumerate(matrix):
        semantic, forensic, learned_gates, labels, sample_ids, paths, learned_probabilities = _extract_branches(
            model, config, transform_spec, device
        )
        if condition_index == 0:
            fixed_clean_semantic_fraction = float(learned_gates[:, 0].float().mean())
        if fixed_clean_semantic_fraction is None:
            raise RuntimeError("clean condition must establish the fixed mean before other conditions")
        fixed_equal = torch.full_like(learned_gates, 0.5)
        fixed_clean_mean = torch.empty_like(learned_gates)
        fixed_clean_mean[:, 0] = fixed_clean_semantic_fraction
        fixed_clean_mean[:, 1] = 1.0 - fixed_clean_semantic_fraction
        generator = torch.Generator(device="cpu")
        generator.manual_seed(shuffle_seed + condition_index)
        permutation = torch.randperm(learned_gates.shape[0], generator=generator)
        shuffled = learned_gates[permutation]
        gates_by_mode = {
            "learned": learned_gates,
            "fixed_equal": fixed_equal,
            "fixed_clean_mean": fixed_clean_mean,
            "shuffled": shuffled,
        }
        probabilities_by_mode = {"learned": learned_probabilities}
        for mode in SUPPORTED_MODES[1:]:
            probabilities_by_mode[mode] = _probabilities_from_gates(
                model,
                semantic,
                forensic,
                gates_by_mode[mode],
                device,
                batch_size=int(config["evaluation"]["batch_size"]),
                amp_enabled=bool(config["train"]["amp"]),
            )

        transform_key = _transform_key(transform_spec)
        learned_decisions = learned_probabilities >= threshold
        for mode in SUPPORTED_MODES:
            probabilities = probabilities_by_mode[mode]
            metrics = binary_metrics(labels, probabilities, threshold)
            metrics_rows.append(
                {
                    "mode": mode,
                    "condition_index": condition_index,
                    "transform": transform_key,
                    "transform_name": transform_spec["name"],
                    "transform_params": json.dumps(transform_spec.get("params", {}), sort_keys=True, separators=(",", ":")),
                    **metrics,
                    **_gate_statistics(gates_by_mode[mode]),
                    "mean_abs_probability_delta_from_learned": float(np.mean(np.abs(probabilities - learned_probabilities))),
                    "max_abs_probability_delta_from_learned": float(np.max(np.abs(probabilities - learned_probabilities))),
                    "threshold_flip_rate_from_learned": float(np.mean((probabilities >= threshold) != learned_decisions)),
                }
            )
        if condition_index in bootstrap_conditions:
            for mode in SUPPORTED_MODES[1:]:
                learned_metrics = binary_metrics(labels, learned_probabilities, threshold)
                ablated_metrics = binary_metrics(labels, probabilities_by_mode[mode], threshold)
                bootstrap_rows.append(
                    {
                        "condition_index": condition_index,
                        "transform": transform_key,
                        "comparison": f"learned_minus_{mode}",
                        "auroc_delta": float(learned_metrics["auroc"] - ablated_metrics["auroc"]),
                        "accuracy_delta": float(
                            np.mean(learned_decisions == labels)
                            - np.mean((probabilities_by_mode[mode] >= threshold) == labels)
                        ),
                        **_paired_bootstrap(
                            labels,
                            learned_probabilities,
                            probabilities_by_mode[mode],
                            threshold,
                            replicates=bootstrap_replicates,
                            seed=bootstrap_seed + condition_index,
                        ),
                    }
                )
        permutation_array = permutation.numpy()
        for sample_index, sample_id in enumerate(sample_ids):
            prediction_rows.append(
                {
                    "condition_index": condition_index,
                    "transform": transform_key,
                    "sample_id": sample_id,
                    "image_path": paths[sample_index],
                    "label": int(labels[sample_index]),
                    "learned_probability": float(probabilities_by_mode["learned"][sample_index]),
                    "fixed_equal_probability": float(probabilities_by_mode["fixed_equal"][sample_index]),
                    "fixed_clean_mean_probability": float(probabilities_by_mode["fixed_clean_mean"][sample_index]),
                    "shuffled_probability": float(probabilities_by_mode["shuffled"][sample_index]),
                    "learned_semantic_gate": float(learned_gates[sample_index, 0]),
                    "shuffled_semantic_gate": float(shuffled[sample_index, 0]),
                    "shuffled_gate_source_sample_id": sample_ids[int(permutation_array[sample_index])],
                }
            )
        print(
            json.dumps(
                {
                    "event": "m3_gate_ablation_condition",
                    "condition_index": condition_index,
                    "transform": transform_key,
                    "learned_auroc": metrics_rows[-4]["auroc"],
                    "fixed_equal_auroc": metrics_rows[-3]["auroc"],
                    "fixed_clean_mean_auroc": metrics_rows[-2]["auroc"],
                    "shuffled_auroc": metrics_rows[-1]["auroc"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    reconciliation = _reconcile_learned(
        metrics_rows,
        protocol["evaluation"]["reference_metrics"],
        float(protocol["evaluation"]["reconciliation_tolerance"]),
    )
    _atomic_csv(output_directory / "metrics_by_mode_and_transform.csv", metrics_rows)
    _atomic_csv(output_directory / "paired_bootstrap_key_conditions.csv", bootstrap_rows)
    _atomic_jsonl(output_directory / "predictions_by_mode.jsonl", prediction_rows)

    def rows_for_mode(mode: str) -> list[dict[str, Any]]:
        return [row for row in metrics_rows if row["mode"] == mode]

    mode_summary: dict[str, Any] = {}
    for mode in SUPPORTED_MODES:
        rows = rows_for_mode(mode)
        mode_summary[mode] = {
            "clean_auroc": rows[0]["auroc"],
            "clean_accuracy": (rows[0]["tp"] + rows[0]["tn"]) / rows[0]["n"],
            "nonclean_mean_auroc": float(np.mean([row["auroc"] for row in rows[1:]])),
            "new_three_mean_auroc": float(np.mean([row["auroc"] for row in rows[18:21]])),
            "six_stage_auroc": rows[20]["auroc"],
            "worst_nonclean_auroc": min(row["auroc"] for row in rows[1:]),
        }
    adaptive_comparisons = [
        row for row in bootstrap_rows
        if row["comparison"] in {"learned_minus_fixed_clean_mean", "learned_minus_shuffled"}
    ]
    summary = {
        "protocol_id": protocol["protocol_id"],
        "checkpoint_sha256": _sha256(checkpoint_path),
        "manifest_sha256": _sha256(manifest_path),
        "matrix_sha256": _sha256(matrix_path),
        "threshold": threshold,
        "fixed_clean_semantic_fraction": fixed_clean_semantic_fraction,
        "fixed_clean_forensic_fraction": 1.0 - fixed_clean_semantic_fraction,
        "modes": mode_summary,
        "key_condition_bootstrap_comparisons": bootstrap_rows,
        "adaptive_gate_max_abs_auroc_delta": max(abs(row["auroc_delta"]) for row in adaptive_comparisons),
        "adaptive_gate_all_auroc_intervals_include_zero": all(
            row["auroc_delta_ci_low"] <= 0.0 <= row["auroc_delta_ci_high"]
            for row in adaptive_comparisons
        ),
        "learned_reconciliation": reconciliation,
    }
    atomic_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", output_directory / "summary.json")
    run_card = {
        "protocol_id": protocol["protocol_id"],
        "protocol_config": str(Path(protocol_path).resolve()),
        "protocol_config_sha256": _sha256(protocol_path),
        "model_config": str(config_path.resolve()),
        "model_config_sha256": checkpoint_config_sha256,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "training_complete": str(Path(protocol["model"]["training_complete"]).resolve()),
        "threshold_summary": str(Path(protocol["model"]["threshold_summary"]).resolve()),
        "threshold": threshold,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "matrix": str(matrix_path.resolve()),
        "matrix_sha256": _sha256(matrix_path),
        "mode_definitions": {
            "learned": "checkpoint learned per-sample gate",
            "fixed_equal": "constant fractions 0.5/0.5; branch scales 1.0/1.0",
            "fixed_clean_mean": "constant label-free mean learned gate from the clean selected manifest",
            "shuffled": "learned gate vectors globally permuted across all 4,000 samples with a deterministic per-condition seed",
        },
        "fixed_clean_semantic_fraction": fixed_clean_semantic_fraction,
        "shuffle_seed": shuffle_seed,
        "bootstrap": protocol["bootstrap"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "parameters": count_parameters(model),
        "artifacts": {
            "metrics": "metrics_by_mode_and_transform.csv",
            "bootstrap": "paired_bootstrap_key_conditions.csv",
            "predictions": "predictions_by_mode.jsonl",
            "summary": "summary.json",
        },
    }
    atomic_text(json.dumps(run_card, indent=2, sort_keys=True) + "\n", output_directory / "run_card.json")
    atomic_text(
        json.dumps(
            {
                "completed_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                "protocol_id": protocol["protocol_id"],
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
            sort_keys=True,
        )
        + "\n",
        output_directory / "COMPLETE",
    )
    print(json.dumps({"event": "m3_gate_ablation_complete", **summary}, sort_keys=True), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate four M3 gate counterfactuals on a frozen checkpoint")
    parser.add_argument("--protocol", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(_parse_args().protocol)
