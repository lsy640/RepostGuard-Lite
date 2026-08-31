from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from repostguard.checkpoint import atomic_text
from repostguard.data.distillation import sha256_file
from repostguard.metrics import binary_metrics, select_balanced_threshold


def _fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float, float]:
    logits = logits.float()
    labels = labels.float()
    before = float(F.binary_cross_entropy_with_logits(logits, labels))
    log_temperature = nn.Parameter(torch.zeros((), dtype=torch.float32))
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(0.05, 20.0))
    after = float(F.binary_cross_entropy_with_logits(logits / temperature, labels))
    return temperature, before, after


def _fit_affine_platt(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float, float, float]:
    """Fit ``sigmoid(a * logit + b)`` while constraining ``a`` to be positive."""

    logits = logits.float().reshape(-1)
    labels = labels.float().reshape(-1)
    if logits.shape != labels.shape:
        raise ValueError("Platt calibration logits and labels must align")
    if labels.unique().numel() != 2:
        raise ValueError("Platt calibration requires both real and AIGC labels")
    before = float(F.binary_cross_entropy_with_logits(logits, labels))
    log_a = nn.Parameter(torch.zeros((), dtype=torch.float32))
    b = nn.Parameter(torch.zeros((), dtype=torch.float32))
    optimizer = torch.optim.LBFGS(
        [log_a, b], lr=0.1, max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        a = log_a.exp().clamp(1e-4, 100.0)
        loss = F.binary_cross_entropy_with_logits(a * logits + b, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    a = float(log_a.detach().exp().clamp(1e-4, 100.0))
    bias = float(b.detach())
    after = float(F.binary_cross_entropy_with_logits(a * logits + bias, labels))
    if not math.isfinite(a) or not math.isfinite(bias) or a <= 0.0:
        raise FloatingPointError("Affine Platt calibration produced invalid parameters")
    return a, bias, before, after


def _fit_per_view_affine_platt(
    logits: torch.Tensor,
    labels: torch.Tensor,
    view_ids: list[str],
) -> dict[str, dict[str, float | int]]:
    if logits.ndim != 2 or logits.shape[1] != len(view_ids):
        raise ValueError("Per-view Platt logits and view ids must align")
    fitted: dict[str, dict[str, float | int]] = {}
    for view_index, view_id in enumerate(view_ids):
        a, b, before, after = _fit_affine_platt(logits[:, view_index], labels)
        probability = torch.sigmoid(a * logits[:, view_index].float() + b)
        threshold_metrics = binary_metrics(
            labels.cpu().numpy(), probability.cpu().numpy(), 0.5
        )
        real_mask = labels == 0
        aigi_mask = labels == 1
        fitted[view_id] = {
            "view_index": int(view_index),
            "a": a,
            "b": b,
            "raw_logit_center": -b / a,
            "binary_cross_entropy_before": before,
            "binary_cross_entropy_after": after,
            "brier_after": float(torch.mean((probability - labels.float()) ** 2)),
            "auroc": float(threshold_metrics["auroc"]),
            "balanced_accuracy_at_0_5": float(
                threshold_metrics["balanced_accuracy"]
            ),
            "positive_rate_at_0_5": float((probability >= 0.5).float().mean()),
            "real_probability_mean": float(probability[real_mask].mean()),
            "aigi_probability_mean": float(probability[aigi_mask].mean()),
        }
    return fitted


def _apply_per_view_affine(
    logits: torch.Tensor,
    view_ids: list[str],
    parameters: dict[str, dict[str, float | int]],
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != len(view_ids):
        raise ValueError("Affine calibration logits and view ids must align")
    missing = [view_id for view_id in view_ids if view_id not in parameters]
    if missing:
        raise KeyError(f"Affine calibration is missing views: {missing}")
    a = logits.new_tensor([float(parameters[view_id]["a"]) for view_id in view_ids])
    b = logits.new_tensor([float(parameters[view_id]["b"]) for view_id in view_ids])
    return logits.float() * a + b


def _parse_m2_weight_grid(raw: str) -> list[float]:
    values = sorted({float(value.strip()) for value in raw.split(",") if value.strip()})
    if not values:
        raise ValueError("M2 weight grid must contain at least one value")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("M2 weight grid values must be finite and in [0, 1]")
    if 0.0 not in values or 1.0 not in values:
        raise ValueError("M2 weight grid must include M3-only (0) and M2-only (1)")
    return values


def _candidate_metrics(
    labels: torch.Tensor,
    m2_probabilities: torch.Tensor,
    m3_probabilities: torch.Tensor,
    *,
    m2_weight: float,
) -> dict[str, object]:
    m3_weight = 1.0 - float(m2_weight)
    mixture = (
        float(m2_weight) * m2_probabilities
        + m3_weight * m3_probabilities
    ).clamp(1e-7, 1.0 - 1e-7)
    if mixture.ndim != 2 or mixture.shape[1] < 1:
        raise ValueError("Teacher calibration cache must contain at least one view")
    view_labels = labels[:, None].expand_as(mixture)
    per_view: list[dict[str, float | int]] = []
    for view_index in range(mixture.shape[1]):
        view_probability = mixture[:, view_index]
        view_label = view_labels[:, view_index]
        label_array = view_label.cpu().numpy()
        probability_array = view_probability.cpu().numpy()
        threshold = select_balanced_threshold(label_array, probability_array)
        metrics = binary_metrics(label_array, probability_array, threshold)
        per_view.append(
            {
                "view_index": int(view_index),
                "binary_cross_entropy": float(
                    F.binary_cross_entropy(view_probability, view_label)
                ),
                "brier": float(torch.mean((view_probability - view_label) ** 2)),
                "auroc": float(metrics["auroc"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "ece_15": float(metrics["ece_15"]),
                "threshold": float(threshold),
            }
        )
    return {
        "m2_weight": float(m2_weight),
        "m3_weight": m3_weight,
        "clean_auroc": float(per_view[0]["auroc"]),
        "mean_view_auroc": float(np.mean([row["auroc"] for row in per_view])),
        "clean_binary_cross_entropy": float(
            per_view[0]["binary_cross_entropy"]
        ),
        "mean_view_binary_cross_entropy": float(
            np.mean([row["binary_cross_entropy"] for row in per_view])
        ),
        "clean_brier": float(per_view[0]["brier"]),
        "mean_view_brier": float(np.mean([row["brier"] for row in per_view])),
        "clean_balanced_accuracy": float(per_view[0]["balanced_accuracy"]),
        "mean_view_balanced_accuracy": float(
            np.mean([row["balanced_accuracy"] for row in per_view])
        ),
        "clean_ece_15": float(per_view[0]["ece_15"]),
        "mean_view_ece_15": float(np.mean([row["ece_15"] for row in per_view])),
        "per_view": per_view,
    }


def _select_mixture(
    labels: torch.Tensor,
    m2_logits: torch.Tensor,
    m3_logits: torch.Tensor,
    *,
    m2_temperature: float,
    m3_temperature: float,
    m2_weight_grid: list[float],
    minimum_dual_auroc_gain: float,
    maximum_clean_auroc_regression: float,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    m2_probabilities = torch.sigmoid(m2_logits.float() / float(m2_temperature))
    m3_probabilities = torch.sigmoid(m3_logits.float() / float(m3_temperature))
    return _select_mixture_from_probabilities(
        labels,
        m2_probabilities,
        m3_probabilities,
        m2_weight_grid=m2_weight_grid,
        minimum_dual_auroc_gain=minimum_dual_auroc_gain,
        maximum_clean_auroc_regression=maximum_clean_auroc_regression,
    )


def _select_mixture_from_probabilities(
    labels: torch.Tensor,
    m2_probabilities: torch.Tensor,
    m3_probabilities: torch.Tensor,
    *,
    m2_weight_grid: list[float],
    minimum_dual_auroc_gain: float,
    maximum_clean_auroc_regression: float,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    if m2_probabilities.shape != m3_probabilities.shape:
        raise ValueError("M2 and M3 calibrated probabilities must align")
    candidates = [
        _candidate_metrics(
            labels,
            m2_probabilities,
            m3_probabilities,
            m2_weight=m2_weight,
        )
        for m2_weight in m2_weight_grid
    ]
    m3_only = next(row for row in candidates if float(row["m2_weight"]) == 0.0)
    dual_candidates = [
        row for row in candidates if 0.0 < float(row["m2_weight"]) < 1.0
    ]
    if dual_candidates:
        best_dual = max(
            dual_candidates,
            key=lambda row: (
                float(row["mean_view_auroc"]),
                -float(row["mean_view_binary_cross_entropy"]),
                -abs(float(row["m2_weight"]) - 0.3),
            ),
        )
        robust_gain = float(best_dual["mean_view_auroc"]) - float(
            m3_only["mean_view_auroc"]
        )
        clean_regression = float(m3_only["clean_auroc"]) - float(
            best_dual["clean_auroc"]
        )
        dual_accepted = (
            robust_gain >= float(minimum_dual_auroc_gain)
            and clean_regression <= float(maximum_clean_auroc_regression)
        )
    else:
        best_dual = None
        robust_gain = float("-inf")
        clean_regression = float("inf")
        dual_accepted = False
    selected = best_dual if dual_accepted else m3_only
    decision = {
        "policy": "accept dual only when mean-view AUROC gain clears the gate without excessive clean regression",
        "minimum_dual_auroc_gain": float(minimum_dual_auroc_gain),
        "maximum_clean_auroc_regression": float(maximum_clean_auroc_regression),
        "best_dual_m2_weight": (
            None if best_dual is None else float(best_dual["m2_weight"])
        ),
        "best_dual_mean_view_auroc_gain_vs_m3": robust_gain,
        "best_dual_clean_auroc_regression_vs_m3": clean_regression,
        "dual_accepted": bool(dual_accepted),
        "selected_reason": "dual_gate_passed" if dual_accepted else "m3_primary_fallback",
    }
    return dict(selected), candidates, decision


def calibrate(
    cache_directory: str,
    output_path: str,
    *,
    m2_weight_grid: list[float] | None = None,
    minimum_dual_auroc_gain: float = 0.002,
    maximum_clean_auroc_regression: float = 0.001,
    require_preprocessing_digest: bool = False,
    method: str = "temperature",
) -> dict[str, object]:
    if method not in {"temperature", "per_view_affine_platt"}:
        raise ValueError("Calibration method must be temperature or per_view_affine_platt")
    cache_files = sorted(Path(cache_directory).expanduser().resolve().glob("teacher_cache_*-of-*.pt"))
    if not cache_files:
        raise FileNotFoundError(f"No teacher cache shards under {cache_directory}")
    labels: list[torch.Tensor] = []
    m2_logits: list[torch.Tensor] = []
    m3_logits: list[torch.Tensor] = []
    manifest_hashes: set[str] = set()
    checkpoint_hashes: dict[str, set[str]] = {"m2": set(), "m3": set()}
    cached_views: set[str] = set()
    preprocessing_hashes: set[str] = set()
    for cache_file in cache_files:
        payload = torch.load(cache_file, map_location="cpu", weights_only=False)
        metadata = payload["metadata"]
        manifest_hashes.add(str(metadata["manifest_sha256"]))
        cached_views.add(json.dumps(metadata["views"], sort_keys=True))
        preprocessing_hashes.add(str(metadata.get("preprocessing_sha256", "")))
        checkpoint_hashes["m2"].add(str(metadata["m2_checkpoint_sha256"]))
        checkpoint_hashes["m3"].add(str(metadata["m3_checkpoint_sha256"]))
        labels.append(torch.as_tensor(payload["labels"]).float())
        m2_logits.append(torch.as_tensor(payload["m2_logits"]).float())
        m3_logits.append(torch.as_tensor(payload["m3_logits"]).float())
    if (
        len(manifest_hashes) != 1
        or len(cached_views) != 1
        or len(preprocessing_hashes) != 1
        or (require_preprocessing_digest and preprocessing_hashes == {""})
        or any(len(values) != 1 for values in checkpoint_hashes.values())
    ):
        raise ValueError("Calibration cache shards do not share one manifest and teacher lineage")
    merged_labels = torch.cat(labels)
    if merged_labels.unique().numel() != 2:
        raise ValueError("Teacher calibration requires both real and AIGC labels")
    merged_m2_logits = torch.cat(m2_logits)
    merged_m3_logits = torch.cat(m3_logits)
    if merged_m2_logits.shape != merged_m3_logits.shape:
        raise ValueError("M2 and M3 calibration logits must have identical shapes")
    if merged_m2_logits.ndim != 2 or merged_m2_logits.shape[0] != merged_labels.numel():
        raise ValueError("Calibration logits must have shape [samples, views]")
    view_specs = json.loads(next(iter(cached_views)))
    view_ids = [str(view["id"]) for view in view_specs]
    if len(view_ids) != merged_m2_logits.shape[1] or len(set(view_ids)) != len(view_ids):
        raise ValueError("Calibration cache view metadata does not match logits")
    m2_temperature, m2_before, m2_after = _fit_temperature(
        merged_m2_logits[:, 0], merged_labels
    )
    m3_temperature, m3_before, m3_after = _fit_temperature(
        merged_m3_logits[:, 0], merged_labels
    )
    affine_calibration = {
        "m2": _fit_per_view_affine_platt(
            merged_m2_logits, merged_labels, view_ids
        ),
        "m3": _fit_per_view_affine_platt(
            merged_m3_logits, merged_labels, view_ids
        ),
    }
    if method == "per_view_affine_platt":
        m2_probabilities = torch.sigmoid(
            _apply_per_view_affine(
                merged_m2_logits, view_ids, affine_calibration["m2"]
            )
        )
        m3_probabilities = torch.sigmoid(
            _apply_per_view_affine(
                merged_m3_logits, view_ids, affine_calibration["m3"]
            )
        )
        selected_mixture, mixture_candidates, mixture_decision = (
            _select_mixture_from_probabilities(
                merged_labels,
                m2_probabilities,
                m3_probabilities,
                m2_weight_grid=m2_weight_grid
                or [0.0, 0.25, 0.3, 0.5, 0.75, 1.0],
                minimum_dual_auroc_gain=minimum_dual_auroc_gain,
                maximum_clean_auroc_regression=maximum_clean_auroc_regression,
            )
        )
    else:
        selected_mixture, mixture_candidates, mixture_decision = _select_mixture(
            merged_labels,
            merged_m2_logits,
            merged_m3_logits,
            m2_temperature=m2_temperature,
            m3_temperature=m3_temperature,
            m2_weight_grid=m2_weight_grid
            or [0.0, 0.25, 0.3, 0.5, 0.75, 1.0],
            minimum_dual_auroc_gain=minimum_dual_auroc_gain,
            maximum_clean_auroc_regression=maximum_clean_auroc_regression,
        )
    result: dict[str, object] = {
        "schema_version": 2,
        "calibration_method": method,
        "samples": int(merged_labels.numel()),
        "views": len(view_specs),
        "view_ids": view_ids,
        "manifest_sha256": next(iter(manifest_hashes)),
        "preprocessing_sha256": next(iter(preprocessing_hashes)) or None,
        "teacher_checkpoint_sha256": {
            "m2": next(iter(checkpoint_hashes["m2"])),
            "m3": next(iter(checkpoint_hashes["m3"])),
        },
        "temperatures": {"m2": m2_temperature, "m3": m3_temperature},
        "affine_calibration": affine_calibration,
        "binary_cross_entropy": {
            "m2_before": m2_before,
            "m2_after": m2_after,
            "m3_before": m3_before,
            "m3_after": m3_after,
        },
        "selected_mixture": {
            "m2_weight": float(selected_mixture["m2_weight"]),
            "m3_weight": float(selected_mixture["m3_weight"]),
        },
        "selected_mixture_metrics": selected_mixture,
        "mixture_decision": mixture_decision,
        "mixture_candidates": mixture_candidates,
        "cache_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in cache_files
        ],
    }
    atomic_text(json.dumps(result, indent=2, sort_keys=True) + "\n", output_path)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit temperature or per-view affine calibration for cached teachers"
    )
    parser.add_argument("--cache-directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--m2-weight-grid",
        default="0,0.25,0.3,0.5,0.75,1",
        help="Comma-separated M2 weights; must include 0 and 1",
    )
    parser.add_argument("--minimum-dual-auroc-gain", type=float, default=0.002)
    parser.add_argument(
        "--maximum-clean-auroc-regression", type=float, default=0.001
    )
    parser.add_argument("--require-preprocessing-digest", action="store_true")
    parser.add_argument(
        "--method",
        choices=("temperature", "per_view_affine_platt"),
        default="temperature",
    )
    arguments = parser.parse_args()
    calibrate(
        arguments.cache_directory,
        arguments.output,
        m2_weight_grid=_parse_m2_weight_grid(arguments.m2_weight_grid),
        minimum_dual_auroc_gain=arguments.minimum_dual_auroc_gain,
        maximum_clean_auroc_regression=arguments.maximum_clean_auroc_regression,
        require_preprocessing_digest=arguments.require_preprocessing_digest,
        method=arguments.method,
    )


if __name__ == "__main__":
    main()
