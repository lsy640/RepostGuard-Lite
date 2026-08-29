from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import yaml
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from repostguard.metrics import expected_calibration_error


MODELS = ("b0", "b1", "b2", "m2", "m3")
MODEL_ARCHITECTURES = {
    "b0": "EfficientNet-B0 clean baseline",
    "b1": "EfficientNet-B0 robust-augmentation baseline",
    "b2": "Frozen OpenCLIP ViT-B/32 + linear head",
    "m2": "Frozen CLIP semantic + forensic branch",
    "m3": "M2 + quality-aware branch gate",
}
CONDITION_LABELS = (
    "Clean",
    "JPEG Q90",
    "JPEG Q70",
    "JPEG Q50",
    "JPEG Q30",
    "Gaussian blur sigma=0.5",
    "Gaussian blur sigma=1.0",
    "Gaussian blur sigma=2.0",
    "Resize 0.5 bicubic",
    "Resize 0.25 bilinear",
    "Gaussian noise sigma=0.02",
    "Gaussian noise sigma=0.05",
    "Gaussian noise sigma=0.10",
    "Color jitter 0.8/0.8/0.8",
    "Color jitter 1.2/1.2/1.2",
    "Center crop ratio=0.8",
    "2-stage resize 0.5 + JPEG Q70",
    "2-stage crop 0.8 + JPEG Q50",
    "4-stage A platform repost",
    "4-stage B edit repost",
    "6-stage random composition",
)
CHART_METRICS = (
    ("Accuracy", "accuracy"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("Specificity", "specificity"),
    ("F1", "f1"),
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from error
    return rows


def _atomic_text(path: str | Path, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, target)


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    columns = list(rows[0])
    if any(set(row) != set(columns) for row in rows):
        raise RuntimeError(f"Inconsistent CSV columns: {path}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target: float) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    eligible = np.flatnonzero(fpr <= target)
    return float(tpr[eligible[-1]]) if eligible.size else 0.0


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    accuracy = _safe_divide(tp + tn, labels.size)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    npv = _safe_divide(tn, tn + fn)
    f1 = _safe_divide(2 * tp, 2 * tp + fp + fn)
    return {
        "n": int(labels.size),
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "npv": npv,
        "f1": f1,
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "predicted_positive_rate": float(predictions.mean()),
        "false_positive_rate": _safe_divide(fp, fp + tn),
        "false_negative_rate": _safe_divide(fn, fn + tp),
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
        "ece_15": float(expected_calibration_error(labels, scores, bins=15)),
        "tpr_at_fpr_1pct": _tpr_at_fpr(labels, scores, 0.01),
        "tpr_at_fpr_5pct": _tpr_at_fpr(labels, scores, 0.05),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _bootstrap_clean(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    replicates: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if positive.size == 0 or negative.size == 0:
        raise RuntimeError("Stratified bootstrap requires both classes")
    rng = np.random.default_rng(seed)
    fields = ("accuracy", "precision", "recall", "specificity", "f1", "mcc", "auroc", "average_precision")
    samples = {field: np.empty(replicates, dtype=np.float64) for field in fields}
    for index in range(replicates):
        sampled = np.concatenate(
            [rng.choice(negative, negative.size, replace=True), rng.choice(positive, positive.size, replace=True)]
        )
        result = _metrics(labels[sampled], scores[sampled], threshold)
        for field in fields:
            samples[field][index] = float(result[field])
    return {
        field: (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
        for field, values in samples.items()
    }


def _common_curve_grid(points: int = 201) -> np.ndarray:
    """Return a renderer-safe shared x grid for multi-series ROC/PR charts."""
    if points < 2:
        raise ValueError("Curve grid must include at least two points")
    return np.linspace(0.0, 1.0, points, dtype=np.float64)


def _interpolate_roc_for_display(
    labels: np.ndarray,
    scores: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    interpolated = np.interp(grid, false_positive_rate, true_positive_rate)
    # A ROC curve starts at (0, 0) and ends at (1, 1). Duplicate FPR=0
    # thresholds otherwise make numpy.interp select the top of the vertical step.
    interpolated[0] = 0.0
    interpolated[-1] = 1.0
    return np.clip(interpolated, 0.0, 1.0)


def _interpolate_pr_for_display(
    labels: np.ndarray,
    scores: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    precision_values, recall_values, _ = precision_recall_curve(labels, scores)
    # sklearn returns recall in decreasing order. The report renderer expects a
    # shared increasing x grid across all color series. At duplicate recall
    # values retain the best precision on the vertical threshold segment.
    ascending_recall = recall_values[::-1]
    ascending_precision = precision_values[::-1]
    unique_recall = np.unique(ascending_recall)
    precision_at_recall = np.asarray(
        [ascending_precision[ascending_recall == value].max() for value in unique_recall],
        dtype=np.float64,
    )
    interpolated = np.interp(grid, unique_recall, precision_at_recall)
    interpolated[0] = 1.0
    interpolated[-1] = float(labels.mean())
    return np.clip(interpolated, 0.0, 1.0)


def _round_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: round(value, 8) if isinstance(value, float) and math.isfinite(value) else value
        for key, value in row.items()
    }


def _condition_group(index: int) -> str:
    if index == 0:
        return "clean"
    if index <= 15:
        return "single_stage"
    if index <= 17:
        return "two_stage"
    if index <= 19:
        return "four_stage"
    if index == 20:
        return "six_stage"
    raise ValueError(index)


def _verify_existing_metrics(recomputed: dict[str, Any], existing: dict[str, str], model: str, condition: str) -> None:
    direct = {
        "n": "n",
        "threshold": "threshold",
        "auroc": "auroc",
        "average_precision": "average_precision",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "macro_f1",
        "recall": "aigc_recall",
        "specificity": "real_specificity",
        "false_positive_rate": "false_positive_rate",
        "tpr_at_fpr_1pct": "tpr_at_fpr_1pct",
        "tpr_at_fpr_5pct": "tpr_at_fpr_5pct",
        "brier": "brier",
        "ece_15": "ece_15",
        "tn": "tn",
        "fp": "fp",
        "fn": "fn",
        "tp": "tp",
    }
    for computed_key, existing_key in direct.items():
        expected = float(existing[existing_key])
        observed = float(recomputed[computed_key])
        tolerance = 1e-6 if computed_key not in {"n", "tn", "fp", "fn", "tp"} else 0.0
        if abs(observed - expected) > tolerance:
            raise RuntimeError(
                f"Metric drift {model}/{condition}/{computed_key}: recomputed={observed} existing={expected}"
            )


def _load_and_compute(arguments: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    manifest_path = Path(arguments.manifest)
    manifest_rows = _read_csv(manifest_path)
    if len(manifest_rows) != 2000:
        raise RuntimeError(f"Expected 2,000 manifest rows, found {len(manifest_rows)}")
    manifest_by_id = {row["sample_id"]: row for row in manifest_rows}
    if len(manifest_by_id) != len(manifest_rows):
        raise RuntimeError("Duplicate sample_id in unseen-generator manifest")
    labels_by_id = {sample_id: int(row["label"]) for sample_id, row in manifest_by_id.items()}
    positive_count = sum(labels_by_id.values())
    negative_count = len(labels_by_id) - positive_count
    if positive_count != 1000 or negative_count != 1000:
        raise RuntimeError("Unseen-generator manifest is not 1,000 Real / 1,000 AIGI")

    matrix = yaml.safe_load(Path(arguments.matrix).read_text(encoding="utf-8"))["evaluation"]
    if len(matrix) != len(CONDITION_LABELS):
        raise RuntimeError("Unexpected perturbation matrix length")

    all_condition_metrics: list[dict[str, Any]] = []
    clean_metrics: list[dict[str, Any]] = []
    clean_metric_chart: list[dict[str, Any]] = []
    roc_rows: list[dict[str, Any]] = []
    pr_rows: list[dict[str, Any]] = []
    generator_recall: list[dict[str, Any]] = []
    real_specificity: list[dict[str, Any]] = []
    robustness_summary: list[dict[str, Any]] = []
    robustness_chart: list[dict[str, Any]] = []
    model_lineage: list[dict[str, Any]] = []
    input_audit: dict[str, Any] = {}
    display_grid = _common_curve_grid()

    for model_order, model in enumerate(MODELS):
        output = Path(arguments.evaluation_root) / model / "unseen_generator"
        for required in ("COMPLETE", "metrics_by_transform.csv", "predictions.jsonl", "run_card.json", "summary.json"):
            if not (output / required).is_file():
                raise RuntimeError(f"Missing {output / required}")
        existing_metrics = _read_csv(output / "metrics_by_transform.csv")
        if len(existing_metrics) != len(matrix):
            raise RuntimeError(f"Unexpected metric row count for {model}")
        run_card = _read_json(output / "run_card.json")
        summary = _read_json(output / "summary.json")
        source_summary = _read_json(Path("outputs/community_forensics") / model / "summary.json")
        source_run_card = _read_json(Path("outputs/community_forensics") / model / "run_card.json")
        threshold = float(source_summary["threshold_from_clean_validation"])
        if abs(threshold - float(summary["threshold_from_clean_validation"])) > 1e-12:
            raise RuntimeError(f"Frozen threshold mismatch for {model}")
        if run_card["val_manifest_sha256"] != _sha256(manifest_path):
            raise RuntimeError(f"Manifest SHA256 mismatch for {model}")
        if run_card["checkpoint_sha256"] != source_run_card["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint lineage mismatch for {model}")

        predictions = _read_jsonl(output / "predictions.jsonl")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in predictions:
            grouped[str(row["transform"])].append(row)
        if len(predictions) != len(matrix) * len(manifest_rows) or len(grouped) != len(matrix):
            raise RuntimeError(f"Prediction cardinality mismatch for {model}")

        clean_labels: np.ndarray | None = None
        clean_scores: np.ndarray | None = None
        per_model: list[dict[str, Any]] = []
        for condition_index, (existing, specification) in enumerate(zip(existing_metrics, matrix, strict=True)):
            transform = existing["transform"]
            if existing["transform_name"] != specification["name"]:
                raise RuntimeError(f"Matrix order mismatch for {model}/{condition_index}")
            rows = grouped[transform]
            if len(rows) != len(manifest_rows):
                raise RuntimeError(f"Condition sample count mismatch for {model}/{transform}")
            row_by_id = {str(row["sample_id"]): row for row in rows}
            if set(row_by_id) != set(manifest_by_id):
                raise RuntimeError(f"Sample identity mismatch for {model}/{transform}")
            ordered = [row_by_id[row["sample_id"]] for row in manifest_rows]
            labels = np.asarray([int(row["label"]) for row in ordered], dtype=np.int64)
            expected_labels = np.asarray([labels_by_id[row["sample_id"]] for row in manifest_rows], dtype=np.int64)
            if not np.array_equal(labels, expected_labels):
                raise RuntimeError(f"Label mismatch for {model}/{transform}")
            scores = np.asarray([float(row["pred"]) for row in ordered], dtype=np.float64)
            if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
                raise RuntimeError(f"Invalid probability for {model}/{transform}")
            computed = _metrics(labels, scores, threshold)
            _verify_existing_metrics(computed, existing, model, transform)
            enriched = _round_row({
                "model": model.upper(),
                "model_order": model_order,
                "condition_index": condition_index,
                "condition_group": _condition_group(condition_index),
                "condition": CONDITION_LABELS[condition_index],
                "transform": transform,
                **computed,
            })
            all_condition_metrics.append(enriched)
            per_model.append(enriched)
            if condition_index == 0:
                clean_labels = labels
                clean_scores = scores

        if clean_labels is None or clean_scores is None:
            raise RuntimeError(f"Missing clean predictions for {model}")
        clean = dict(per_model[0])
        intervals = _bootstrap_clean(
            clean_labels,
            clean_scores,
            threshold,
            arguments.bootstrap_replicates,
            arguments.bootstrap_seed + model_order,
        )
        for field, (lower, upper) in intervals.items():
            clean[f"{field}_ci_low"] = round(lower, 8)
            clean[f"{field}_ci_high"] = round(upper, 8)
            clean[f"{field}_ci_95"] = f"{100.0 * lower:.2f}%–{100.0 * upper:.2f}%"
        clean_metrics.append(clean)
        for metric_order, (metric_label, metric_field) in enumerate(CHART_METRICS):
            clean_metric_chart.append({
                "model": model.upper(),
                "model_order": model_order,
                "metric": metric_label,
                "metric_order": metric_order,
                "value": clean[metric_field],
            })

        display_tpr = _interpolate_roc_for_display(clean_labels, clean_scores, display_grid)
        roc_rows.extend([
            {
                "model": model.upper(),
                "model_order": model_order,
                "point_order": index,
                "fpr": round(float(x), 6),
                "tpr": round(float(y), 8),
                "auroc": clean["auroc"],
                "line_style": "solid",
            }
            for index, (x, y) in enumerate(zip(display_grid, display_tpr, strict=True))
        ])

        display_precision = _interpolate_pr_for_display(clean_labels, clean_scores, display_grid)
        pr_rows.extend([
            {
                "model": model.upper(),
                "model_order": model_order,
                "point_order": index,
                "recall": round(float(x), 6),
                "precision": round(float(y), 8),
                "average_precision": clean["average_precision"],
                "line_style": "solid",
            }
            for index, (x, y) in enumerate(zip(display_grid, display_precision, strict=True))
        ])

        clean_predictions = (clean_scores >= threshold).astype(np.int64)
        generator_groups: dict[str, list[int]] = defaultdict(list)
        real_groups: dict[str, list[int]] = defaultdict(list)
        for index, manifest_row in enumerate(manifest_rows):
            if int(manifest_row["label"]) == 1:
                generator_groups[manifest_row["canonical_generator_id"]].append(index)
            else:
                real_groups[manifest_row["real_source"]].append(index)
        for generator in sorted(generator_groups):
            indices = np.asarray(generator_groups[generator], dtype=np.int64)
            tp = int(clean_predictions[indices].sum())
            generator_recall.append({
                "model": model.upper(),
                "model_order": model_order,
                "generator": generator,
                "n": int(indices.size),
                "tp": tp,
                "fn": int(indices.size - tp),
                "recall": round(_safe_divide(tp, indices.size), 8),
                "mean_score": round(float(clean_scores[indices].mean()), 8),
            })
        for source in sorted(real_groups):
            indices = np.asarray(real_groups[source], dtype=np.int64)
            fp = int(clean_predictions[indices].sum())
            tn = int(indices.size - fp)
            real_specificity.append({
                "model": model.upper(),
                "model_order": model_order,
                "real_source": source,
                "n": int(indices.size),
                "tn": tn,
                "fp": fp,
                "specificity": round(_safe_divide(tn, indices.size), 8),
                "false_positive_rate": round(_safe_divide(fp, indices.size), 8),
                "mean_score": round(float(clean_scores[indices].mean()), 8),
            })

        perturbed = per_model[1:]
        worst_accuracy = min(perturbed, key=lambda row: float(row["accuracy"]))
        worst_f1 = min(perturbed, key=lambda row: float(row["f1"]))
        worst_auroc = min(perturbed, key=lambda row: float(row["auroc"]))
        robust = {
            "model": model.upper(),
            "model_order": model_order,
            "clean_accuracy": clean["accuracy"],
            "mean_20_accuracy": round(fmean(float(row["accuracy"]) for row in perturbed), 8),
            "worst_20_accuracy": worst_accuracy["accuracy"],
            "worst_accuracy_condition": worst_accuracy["condition"],
            "mean_20_f1": round(fmean(float(row["f1"]) for row in perturbed), 8),
            "worst_20_f1": worst_f1["f1"],
            "worst_f1_condition": worst_f1["condition"],
            "mean_20_auroc": round(fmean(float(row["auroc"]) for row in perturbed), 8),
            "worst_20_auroc": worst_auroc["auroc"],
            "worst_auroc_condition": worst_auroc["condition"],
            "mean_20_recall": round(fmean(float(row["recall"]) for row in perturbed), 8),
            "mean_20_specificity": round(fmean(float(row["specificity"]) for row in perturbed), 8),
        }
        robustness_summary.append(robust)
        for group_order, (label, field) in enumerate((
            ("Clean", "clean_accuracy"),
            ("20 perturbations mean", "mean_20_accuracy"),
            ("20 perturbations worst", "worst_20_accuracy"),
        )):
            robustness_chart.append({
                "model": model.upper(),
                "model_order": model_order,
                "condition_group": label,
                "group_order": group_order,
                "accuracy": robust[field],
            })

        model_lineage.append({
            "model": model.upper(),
            "model_order": model_order,
            "architecture": MODEL_ARCHITECTURES[model],
            "frozen_threshold": round(threshold, 8),
            "checkpoint_sha256": run_card["checkpoint_sha256"][:12],
            "evaluation_job": str(run_card["slurm_job_id"]),
            "prediction_rows": len(predictions),
        })
        input_audit[model] = {
            "evaluation_job": str(run_card["slurm_job_id"]),
            "checkpoint_sha256": run_card["checkpoint_sha256"],
            "metrics_csv": str(output / "metrics_by_transform.csv"),
            "metrics_csv_sha256": _sha256(output / "metrics_by_transform.csv"),
            "predictions_jsonl": str(output / "predictions.jsonl"),
            "predictions_jsonl_sha256": _sha256(output / "predictions.jsonl"),
            "run_card_sha256": _sha256(output / "run_card.json"),
            "summary_sha256": _sha256(output / "summary.json"),
        }

    prevalence = positive_count / len(manifest_rows)
    roc_rows.extend([
        {
            "model": "Chance",
            "model_order": 5,
            "point_order": index,
            "fpr": round(float(value), 6),
            "tpr": round(float(value), 8),
            "auroc": 0.5,
            "line_style": "dashed",
        }
        for index, value in enumerate(display_grid)
    ])
    pr_rows.extend([
        {
            "model": "Prevalence baseline",
            "model_order": 5,
            "point_order": index,
            "recall": round(float(value), 6),
            "precision": round(prevalence, 8),
            "average_precision": prevalence,
            "line_style": "dashed",
        }
        for index, value in enumerate(display_grid)
    ])

    expected_grid = [round(float(value), 6) for value in display_grid]
    for dataset_name, rows, x_field in (
        ("ROC", roc_rows, "fpr"),
        ("PR", pr_rows, "recall"),
    ):
        for model_name in [model.upper() for model in MODELS] + (["Chance"] if dataset_name == "ROC" else ["Prevalence baseline"]):
            model_grid = [row[x_field] for row in rows if row["model"] == model_name]
            if model_grid != expected_grid:
                raise RuntimeError(f"{dataset_name} display grid mismatch for {model_name}")

    condition_matrix = []
    for index, specification in enumerate(matrix):
        condition_matrix.append({
            "condition_index": index,
            "condition": CONDITION_LABELS[index],
            "condition_group": _condition_group(index),
            "transform_name": specification["name"],
            "parameters": json.dumps(specification.get("params", {}), sort_keys=True, separators=(",", ":")),
        })

    metric_definitions = [
        {"metric": "Accuracy", "formula": "(TP + TN) / N", "interpretation": "全部样本中分类正确的比例", "threshold_dependency": "固定阈值"},
        {"metric": "Precision (PPV)", "formula": "TP / (TP + FP)", "interpretation": "预测为 AIGI 的样本中实际为 AIGI 的比例", "threshold_dependency": "固定阈值"},
        {"metric": "Recall / Sensitivity", "formula": "TP / (TP + FN)", "interpretation": "实际 AIGI 被检出的比例", "threshold_dependency": "固定阈值"},
        {"metric": "Specificity (TNR)", "formula": "TN / (TN + FP)", "interpretation": "真实图片被正确放行的比例", "threshold_dependency": "固定阈值"},
        {"metric": "NPV", "formula": "TN / (TN + FN)", "interpretation": "预测为真实的样本中实际为真实的比例", "threshold_dependency": "固定阈值"},
        {"metric": "F1", "formula": "2TP / (2TP + FP + FN)", "interpretation": "AIGI Precision 与 Recall 的调和平均", "threshold_dependency": "固定阈值"},
        {"metric": "Macro-F1", "formula": "(F1_real + F1_AIGI) / 2", "interpretation": "两类 F1 等权平均", "threshold_dependency": "固定阈值"},
        {"metric": "Balanced accuracy", "formula": "(Recall + Specificity) / 2", "interpretation": "两类召回等权平均", "threshold_dependency": "固定阈值"},
        {"metric": "MCC", "formula": "(TP*TN-FP*FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))", "interpretation": "综合四格混淆矩阵，范围 -1 至 1", "threshold_dependency": "固定阈值"},
        {"metric": "AUROC", "formula": "ROC 曲线下面积", "interpretation": "随机 AIGI 得分高于随机真实图的概率解释", "threshold_dependency": "阈值无关"},
        {"metric": "Average precision", "formula": "Precision-recall 阶梯积分", "interpretation": "正类为 AIGI 的排序质量", "threshold_dependency": "阈值无关"},
        {"metric": "FPR", "formula": "FP / (FP + TN)", "interpretation": "真实图片被误报为 AIGI 的比例", "threshold_dependency": "固定阈值"},
        {"metric": "FNR", "formula": "FN / (FN + TP)", "interpretation": "AIGI 被漏检为真实的比例", "threshold_dependency": "固定阈值"},
        {"metric": "Brier", "formula": "mean((p-y)^2)", "interpretation": "概率误差，越低越好", "threshold_dependency": "阈值无关"},
        {"metric": "ECE-15", "formula": "15 个等宽概率桶的加权 |accuracy-confidence|", "interpretation": "概率校准误差，越低越好", "threshold_dependency": "阈值无关"},
        {"metric": "TPR@1%/5% FPR", "formula": "FPR 不超过目标时的最大 TPR", "interpretation": "低误报操作区的 AIGI 检出率", "threshold_dependency": "操作点约束"},
    ]

    datasets = {
        "headline": [],
        "clean_metrics": clean_metrics,
        "clean_metric_chart": clean_metric_chart,
        "roc_curve": [_round_row(row) for row in roc_rows],
        "pr_curve": [_round_row(row) for row in pr_rows],
        "robustness_summary": robustness_summary,
        "robustness_chart": robustness_chart,
        "condition_metrics": all_condition_metrics,
        "generator_recall": generator_recall,
        "real_specificity": real_specificity,
        "condition_matrix": condition_matrix,
        "metric_definitions": metric_definitions,
        "model_lineage": model_lineage,
    }
    best_accuracy = max(clean_metrics, key=lambda row: float(row["accuracy"]))
    best_f1 = max(clean_metrics, key=lambda row: float(row["f1"]))
    best_auroc = max(clean_metrics, key=lambda row: float(row["auroc"]))
    best_robust = max(robustness_summary, key=lambda row: float(row["mean_20_accuracy"]))
    datasets["headline"] = [{
        "best_accuracy_model": best_accuracy["model"],
        "best_accuracy": best_accuracy["accuracy"],
        "best_f1_model": best_f1["model"],
        "best_f1": best_f1["f1"],
        "best_auroc_model": best_auroc["model"],
        "best_auroc": best_auroc["auroc"],
        "best_robust_accuracy_model": best_robust["model"],
        "best_robust_accuracy": best_robust["mean_20_accuracy"],
        "sample_count": len(manifest_rows),
        "real_count": negative_count,
        "aigi_count": positive_count,
        "generator_count": len({row["canonical_generator_id"] for row in manifest_rows if int(row["label"]) == 1}),
        "condition_count": len(matrix),
        "prediction_count": len(MODELS) * len(matrix) * len(manifest_rows),
    }]
    audit_context = {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "matrix": str(arguments.matrix),
        "matrix_sha256": _sha256(arguments.matrix),
        "input_artifacts": input_audit,
        "bootstrap_replicates": arguments.bootstrap_replicates,
        "bootstrap_seed": arguments.bootstrap_seed,
    }
    return datasets, audit_context


def _sqlite_type(values: list[Any]) -> str:
    concrete = [value for value in values if value is not None]
    if concrete and all(isinstance(value, int) and not isinstance(value, bool) for value in concrete):
        return "INTEGER"
    if concrete and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in concrete):
        return "REAL"
    return "TEXT"


def _materialize_sql_snapshot(staged: dict[str, list[dict[str, Any]]], queries: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        for dataset, rows in staged.items():
            if not rows:
                raise RuntimeError(f"Empty report dataset: {dataset}")
            columns = list(rows[0])
            if any(set(row) != set(columns) for row in rows):
                raise RuntimeError(f"Inconsistent report dataset: {dataset}")
            definitions = ", ".join(
                f'"{column}" {_sqlite_type([row[column] for row in rows])}' for column in columns
            )
            connection.execute(f'CREATE TABLE "{dataset}" ({definitions})')
            placeholders = ", ".join("?" for _ in columns)
            connection.executemany(
                f'INSERT INTO "{dataset}" VALUES ({placeholders})',
                [[row[column] for column in columns] for row in rows],
            )
        snapshot: dict[str, list[dict[str, Any]]] = {}
        for dataset, query in queries.items():
            snapshot[dataset] = [dict(row) for row in connection.execute(query).fetchall()]
            if len(snapshot[dataset]) != len(staged[dataset]):
                raise RuntimeError(f"SQL row-count drift for {dataset}")
        return snapshot
    finally:
        connection.close()


def _source(dataset: str, query: str, generated_at: str) -> dict[str, Any]:
    return {
        "id": f"{dataset}_sql",
        "label": f"Reviewed {dataset.replace('_', ' ')} snapshot",
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": query,
            "description": "Executed over reviewed rows after manifest identity, label, threshold, lineage, cardinality, probability-range, and published-metric reconciliation checks.",
            "executed_at": generated_at,
            "tables_used": [dataset],
            "filters": ["split = test_external_unseen_generator", "positive class = AIGI", "threshold frozen from internal Small clean validation"],
        },
    }


def _columns(*values: tuple[str, str, str | None]) -> list[dict[str, Any]]:
    result = []
    for field, label, format_name in values:
        item: dict[str, Any] = {"field": field, "label": label}
        if format_name:
            item["format"] = format_name
        result.append(item)
    return result


def _fmt_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _build_artifact(snapshot: dict[str, list[dict[str, Any]]], queries: dict[str, str], generated_at: str) -> dict[str, Any]:
    title = "Community Forensics Unseen-generator 详细准确率与分类指标报告"
    sources = [_source(dataset, query, generated_at) for dataset, query in queries.items()]
    source_ids = {dataset: f"{dataset}_sql" for dataset in queries}
    headline = snapshot["headline"][0]
    clean = {row["model"]: row for row in snapshot["clean_metrics"]}
    robust = {row["model"]: row for row in snapshot["robustness_summary"]}
    best_accuracy_model = headline["best_accuracy_model"]
    best_auroc_model = headline["best_auroc_model"]
    best_robust_model = headline["best_robust_accuracy_model"]

    cards = [
        {
            "id": "best_accuracy_card",
            "description": f"Clean、冻结阈值；最佳模型 {best_accuracy_model}。",
            "dataset": "headline",
            "sourceId": source_ids["headline"],
            "metrics": [{"label": "最佳 Clean Accuracy", "field": "best_accuracy", "format": "percent"}],
        },
        {
            "id": "best_f1_card",
            "description": f"正类为 AIGI；最佳模型 {headline['best_f1_model']}。",
            "dataset": "headline",
            "sourceId": source_ids["headline"],
            "metrics": [{"label": "最佳 Clean F1", "field": "best_f1", "format": "percent"}],
        },
        {
            "id": "best_auroc_card",
            "description": f"阈值无关；最佳模型 {best_auroc_model}。",
            "dataset": "headline",
            "sourceId": source_ids["headline"],
            "metrics": [{"label": "最佳 Clean AUROC", "field": "best_auroc", "format": "percent"}],
        },
        {
            "id": "best_robust_card",
            "description": f"排除 Clean 后 20 条件等权平均；最佳模型 {best_robust_model}。",
            "dataset": "headline",
            "sourceId": source_ids["headline"],
            "metrics": [{"label": "最佳扰动平均 Accuracy", "field": "best_robust_accuracy", "format": "percent"}],
        },
    ]

    charts = [
        {
            "id": "clean_metric_chart",
            "title": "Clean 固定阈值分类指标",
            "subtitle": "Accuracy、Precision、Recall、Specificity 与 AIGI F1；测试集不重新调阈值。",
            "type": "bar",
            "intent": "comparison",
            "question": "五个模型在 Clean Unseen-generator 上的阈值相关指标如何比较？",
            "rationale": "五个模型与五项同尺度比例适合使用分组柱状图。",
            "comparisonContext": {"unit": "rate", "grain": "model by metric", "baseline": "frozen internal-validation threshold"},
            "dataset": "clean_metric_chart",
            "sourceId": source_ids["clean_metric_chart"],
            "encodings": {
                "x": {"field": "model", "type": "nominal", "label": "Model"},
                "y": {"field": "value", "type": "quantitative", "label": "Rate", "format": "percent"},
                "color": {"field": "metric", "type": "nominal", "label": "Metric"},
                "tooltip": [
                    {"field": "metric", "type": "nominal", "label": "Metric"},
                    {"field": "value", "type": "quantitative", "label": "Value", "format": "percent"},
                ],
            },
            "palette": {"kind": "categorical"},
            "legend": {"position": "bottom", "interactive": True},
            "labels": {"values": "none"},
            "valueFormat": "percent",
            "layout": "full",
        },
        {
            "id": "roc_chart",
            "title": "Clean ROC 曲线",
            "subtitle": "正类为 AIGI；各模型统一插值到 201 点 FPR 网格，Chance 虚线表示随机排序基线。",
            "type": "line",
            "intent": "comparison",
            "question": "在不同误报率操作点上，各模型的 AIGI 检出率如何变化？",
            "rationale": "ROC 是阈值扫描形成的连续路径，使用多序列折线。",
            "comparisonContext": {"unit": "rate", "grain": "threshold point by model", "baseline": "chance diagonal"},
            "dataset": "roc_curve",
            "sourceId": source_ids["roc_curve"],
            "encodings": {
                "x": {"field": "fpr", "type": "quantitative", "label": "False positive rate", "format": "percent"},
                "y": {"field": "tpr", "type": "quantitative", "label": "True positive rate", "format": "percent"},
                "color": {"field": "model", "type": "nominal", "label": "Model"},
                "lineStyle": {"field": "line_style", "type": "nominal", "label": "Line style"},
                "tooltip": [
                    {"field": "model", "type": "nominal", "label": "Model"},
                    {"field": "fpr", "type": "quantitative", "label": "FPR", "format": "percent"},
                    {"field": "tpr", "type": "quantitative", "label": "TPR", "format": "percent"},
                    {"field": "auroc", "type": "quantitative", "label": "Full-data AUROC", "format": "percent"},
                ],
            },
            "palette": {"kind": "categorical"},
            "legend": {"position": "bottom", "interactive": True},
            "labels": {"values": "none"},
            "valueFormat": "percent",
            "layout": "full",
        },
        {
            "id": "pr_chart",
            "title": "Clean Precision–Recall 曲线",
            "subtitle": "Recall 按 0→1 统一插值到 201 点网格；AIGI 占比为 50%，虚线基线为 50% Precision。",
            "type": "line",
            "intent": "comparison",
            "question": "各模型在 AIGI Precision 与 Recall 之间提供怎样的阈值权衡？",
            "rationale": "PR 是阈值扫描形成的连续路径，且直接强调正类检测质量。",
            "comparisonContext": {"unit": "rate", "grain": "threshold point by model", "baseline": "AIGI prevalence = 0.5"},
            "dataset": "pr_curve",
            "sourceId": source_ids["pr_curve"],
            "encodings": {
                "x": {"field": "recall", "type": "quantitative", "label": "Recall", "format": "percent"},
                "y": {"field": "precision", "type": "quantitative", "label": "Precision", "format": "percent"},
                "color": {"field": "model", "type": "nominal", "label": "Model"},
                "lineStyle": {"field": "line_style", "type": "nominal", "label": "Line style"},
                "tooltip": [
                    {"field": "model", "type": "nominal", "label": "Model"},
                    {"field": "recall", "type": "quantitative", "label": "Recall", "format": "percent"},
                    {"field": "precision", "type": "quantitative", "label": "Precision", "format": "percent"},
                    {"field": "average_precision", "type": "quantitative", "label": "Full-data AP", "format": "percent"},
                ],
            },
            "palette": {"kind": "categorical"},
            "legend": {"position": "bottom", "interactive": True},
            "labels": {"values": "none"},
            "valueFormat": "percent",
            "layout": "full",
        },
        {
            "id": "robustness_chart",
            "title": "Clean 与 20 种扰动的 Accuracy",
            "subtitle": "扰动均值和最坏值均按模型独立计算；同一冻结阈值贯穿全部条件。",
            "type": "bar",
            "intent": "comparison",
            "question": "Clean 表现能否在 20 个扰动条件下保持？",
            "rationale": "每个模型的 Clean、扰动均值、扰动最坏值适合分组柱状图。",
            "comparisonContext": {"unit": "accuracy", "grain": "model by condition summary", "normalization": "equal weight across 20 perturbations"},
            "dataset": "robustness_chart",
            "sourceId": source_ids["robustness_chart"],
            "encodings": {
                "x": {"field": "model", "type": "nominal", "label": "Model"},
                "y": {"field": "accuracy", "type": "quantitative", "label": "Accuracy", "format": "percent"},
                "color": {"field": "condition_group", "type": "nominal", "label": "Condition summary"},
                "tooltip": [{"field": "accuracy", "type": "quantitative", "label": "Accuracy", "format": "percent"}],
            },
            "palette": {"kind": "categorical"},
            "legend": {"position": "bottom", "interactive": True},
            "labels": {"values": "none"},
            "valueFormat": "percent",
            "layout": "full",
        },
        {
            "id": "generator_recall_chart",
            "title": "Clean AIGI Recall by exact unseen generator",
            "subtitle": "每个精确生成器 83–84 张；该切片只衡量正类检出率，不定义 Precision。",
            "type": "bar",
            "intent": "comparison",
            "question": "模型性能是否被个别未见生成器拖累？",
            "rationale": "12 个离散生成器与五个模型适合分组柱状图定位召回差异。",
            "comparisonContext": {"unit": "recall", "grain": "model by exact generator", "sample_size": "83 or 84 AIGI images per generator"},
            "dataset": "generator_recall",
            "sourceId": source_ids["generator_recall"],
            "encodings": {
                "x": {"field": "generator", "type": "nominal", "label": "Exact generator"},
                "y": {"field": "recall", "type": "quantitative", "label": "AIGI Recall", "format": "percent"},
                "color": {"field": "model", "type": "nominal", "label": "Model"},
                "tooltip": [
                    {"field": "n", "type": "quantitative", "label": "N", "format": "number"},
                    {"field": "tp", "type": "quantitative", "label": "TP", "format": "number"},
                    {"field": "fn", "type": "quantitative", "label": "FN", "format": "number"},
                ],
            },
            "palette": {"kind": "categorical"},
            "legend": {"position": "bottom", "interactive": True},
            "labels": {"values": "none"},
            "valueFormat": "percent",
            "layout": "full",
        },
    ]

    tables = [
        {
            "id": "clean_metrics_table",
            "title": "Clean 完整分类指标",
            "subtitle": "所有阈值相关指标均使用内部验证集冻结阈值；区间另见下一表。",
            "dataset": "clean_metrics",
            "sourceId": source_ids["clean_metrics"],
            "defaultSort": {"field": "accuracy", "direction": "desc"},
            "density": "dense",
            "columns": _columns(
                ("model", "模型", None), ("threshold", "冻结阈值", "number"),
                ("accuracy", "Accuracy", "percent"),
                ("precision", "Precision", "percent"), ("recall", "Recall", "percent"), ("specificity", "Specificity", "percent"),
                ("npv", "NPV", "percent"), ("f1", "F1", "percent"), ("macro_f1", "Macro-F1", "percent"),
                ("balanced_accuracy", "Balanced Acc", "percent"), ("mcc", "MCC", "number"),
                ("auroc", "AUROC", "percent"),
                ("average_precision", "AP", "percent"), ("brier", "Brier", "number"), ("ece_15", "ECE-15", "number"),
                ("tpr_at_fpr_1pct", "TPR@1%FPR", "percent"), ("tpr_at_fpr_5pct", "TPR@5%FPR", "percent"),
            ),
        },
        {
            "id": "clean_ci_table",
            "title": "Clean 指标的 95% bootstrap 区间",
            "subtitle": "1,000 次 Real/AIGI 分层 bootstrap；区间仅量化当前 2,000 张测试样本的抽样不确定性。",
            "dataset": "clean_metrics",
            "sourceId": source_ids["clean_metrics"],
            "defaultSort": {"field": "model_order", "direction": "asc"},
            "columns": _columns(
                ("model_order", "顺序", "number"), ("model", "模型", None),
                ("accuracy_ci_95", "Accuracy 95% CI", None), ("precision_ci_95", "Precision 95% CI", None),
                ("recall_ci_95", "Recall 95% CI", None), ("specificity_ci_95", "Specificity 95% CI", None),
                ("f1_ci_95", "F1 95% CI", None), ("mcc_ci_95", "MCC 95% CI", None),
                ("auroc_ci_95", "AUROC 95% CI", None), ("average_precision_ci_95", "AP 95% CI", None),
            ),
        },
        {
            "id": "confusion_table",
            "title": "Clean 混淆矩阵计数",
            "subtitle": "正类为 AIGI；每个模型的四格计数总和均为 2,000。",
            "dataset": "clean_metrics",
            "sourceId": source_ids["clean_metrics"],
            "defaultSort": {"field": "fn", "direction": "asc"},
            "columns": _columns(
                ("model", "模型", None), ("tn", "TN", "number"), ("fp", "FP", "number"),
                ("fn", "FN", "number"), ("tp", "TP", "number"),
                ("false_positive_rate", "FPR", "percent"), ("false_negative_rate", "FNR", "percent"),
            ),
        },
        {
            "id": "robustness_summary_table",
            "title": "20 种扰动条件汇总",
            "subtitle": "排除 Clean；均值对 20 个条件等权，最坏值为条件最小值。",
            "dataset": "robustness_summary",
            "sourceId": source_ids["robustness_summary"],
            "defaultSort": {"field": "mean_20_accuracy", "direction": "desc"},
            "columns": _columns(
                ("model", "模型", None), ("clean_accuracy", "Clean Acc", "percent"),
                ("mean_20_accuracy", "扰动平均 Acc", "percent"), ("worst_20_accuracy", "扰动最坏 Acc", "percent"),
                ("worst_accuracy_condition", "Acc 最坏条件", None),
                ("mean_20_f1", "扰动平均 F1", "percent"), ("worst_20_f1", "扰动最坏 F1", "percent"),
                ("mean_20_auroc", "扰动平均 AUROC", "percent"), ("worst_20_auroc", "扰动最坏 AUROC", "percent"),
                ("worst_auroc_condition", "AUROC 最坏条件", None),
                ("mean_20_recall", "扰动平均 Recall", "percent"), ("mean_20_specificity", "扰动平均 Specificity", "percent"),
            ),
        },
        {
            "id": "condition_metrics_table",
            "title": "完整 105 条模型 × 条件指标",
            "subtitle": "5 个模型 × 21 个条件；含新增 Accuracy、Precision、NPV、F1、MCC 与 FNR。",
            "dataset": "condition_metrics",
            "sourceId": source_ids["condition_metrics"],
            "defaultSort": {"field": "accuracy", "direction": "asc"},
            "density": "dense",
            "columns": _columns(
                ("model", "模型", None), ("condition_index", "序号", "number"), ("condition_group", "分组", None), ("condition", "条件", None),
                ("accuracy", "Accuracy", "percent"), ("precision", "Precision", "percent"), ("recall", "Recall", "percent"),
                ("specificity", "Specificity", "percent"), ("npv", "NPV", "percent"), ("f1", "F1", "percent"),
                ("macro_f1", "Macro-F1", "percent"), ("balanced_accuracy", "Balanced Acc", "percent"), ("mcc", "MCC", "number"),
                ("auroc", "AUROC", "percent"), ("average_precision", "AP", "percent"),
                ("false_positive_rate", "FPR", "percent"), ("false_negative_rate", "FNR", "percent"),
                ("brier", "Brier", "number"), ("ece_15", "ECE-15", "number"),
                ("tn", "TN", "number"), ("fp", "FP", "number"), ("fn", "FN", "number"), ("tp", "TP", "number"),
            ),
        },
        {
            "id": "generator_recall_table",
            "title": "12 个未见精确生成器的 Clean Recall",
            "subtitle": "生成器切片只有 AIGI 正类，因此报告 Recall、TP、FN，不报告 Precision/Specificity。",
            "dataset": "generator_recall",
            "sourceId": source_ids["generator_recall"],
            "defaultSort": {"field": "recall", "direction": "asc"},
            "density": "dense",
            "columns": _columns(
                ("model", "模型", None), ("generator", "精确生成器", None), ("n", "N", "number"),
                ("tp", "TP", "number"), ("fn", "FN", "number"), ("recall", "Recall", "percent"), ("mean_score", "平均 AIGI 分数", "percent"),
            ),
        },
        {
            "id": "real_specificity_table",
            "title": "4 个真实来源的 Clean Specificity",
            "subtitle": "每个真实来源 250 张；Specificity 衡量正确放行，FPR 衡量误报。",
            "dataset": "real_specificity",
            "sourceId": source_ids["real_specificity"],
            "defaultSort": {"field": "specificity", "direction": "asc"},
            "columns": _columns(
                ("model", "模型", None), ("real_source", "真实来源", None), ("n", "N", "number"),
                ("tn", "TN", "number"), ("fp", "FP", "number"), ("specificity", "Specificity", "percent"),
                ("false_positive_rate", "FPR", "percent"), ("mean_score", "平均 AIGI 分数", "percent"),
            ),
        },
        {
            "id": "metric_definition_table",
            "title": "指标定义与解释",
            "subtitle": "正类统一定义为 AIGI；TP/FP/TN/FN 均以该定义展开。",
            "dataset": "metric_definitions",
            "sourceId": source_ids["metric_definitions"],
            "defaultSort": {"field": "metric", "direction": "asc"},
            "columns": _columns(
                ("metric", "指标", None), ("formula", "公式", None), ("interpretation", "解释", None), ("threshold_dependency", "阈值关系", None),
            ),
        },
        {
            "id": "condition_matrix_table",
            "title": "21 个评测条件",
            "subtitle": "Clean + 17 个既有条件 + 2 个四阶段条件 + 1 个六阶段随机共同扰动。",
            "dataset": "condition_matrix",
            "sourceId": source_ids["condition_matrix"],
            "defaultSort": {"field": "condition_index", "direction": "asc"},
            "columns": _columns(
                ("condition_index", "序号", "number"), ("condition_group", "分组", None), ("condition", "条件", None),
                ("transform_name", "变换名", None), ("parameters", "参数", None),
            ),
        },
        {
            "id": "lineage_table",
            "title": "模型、冻结阈值与评测谱系",
            "subtitle": "本报告直接读取已完成评测的逐样本预测，不重新训练或推理。",
            "dataset": "model_lineage",
            "sourceId": source_ids["model_lineage"],
            "defaultSort": {"field": "model_order", "direction": "asc"},
            "columns": _columns(
                ("model_order", "顺序", "number"), ("model", "模型", None), ("architecture", "架构", None),
                ("frozen_threshold", "冻结阈值", "number"), ("checkpoint_sha256", "Checkpoint SHA256", None),
                ("evaluation_job", "评测 Job", None), ("prediction_rows", "预测记录", "number"),
            ),
        },
    ]

    accuracy_ci = f"{_fmt_percent(clean[best_accuracy_model]['accuracy_ci_low'])}–{_fmt_percent(clean[best_accuracy_model]['accuracy_ci_high'])}"
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": source_ids["headline"],
            "body": (
                "## 技术摘要\n\n"
                f"- **Clean Accuracy 最优为 {best_accuracy_model}：{_fmt_percent(headline['best_accuracy'])}**（分层 bootstrap 95% CI {accuracy_ci}）。\n"
                f"- **Clean AUROC 最优为 {best_auroc_model}：{_fmt_percent(headline['best_auroc'])}**；AUROC 与冻结阈值无关。\n"
                f"- **20 个扰动条件的平均 Accuracy 最优为 {best_robust_model}：{_fmt_percent(headline['best_robust_accuracy'])}**。\n"
                f"- 证据范围为 2,000 张平衡测试图（1,000 Real / 1,000 AIGI）、12 个训练未见精确生成器、21 个条件、5 个冻结模型，共 210,000 条逐样本预测。\n\n"
                "Accuracy 受 1:1 类别比例影响；真实部署比例变化时，Precision 与 NPV 会随先验改变。因此模型选择不能只看 Accuracy，还需联合 Recall、Specificity、低 FPR TPR、PR 曲线与校准误差。"
            ),
        },
        {"id": "headline_cards", "type": "metric-strip", "cardIds": ["best_accuracy_card", "best_f1_card", "best_auroc_card", "best_robust_card"]},
        {
            "id": "clean_finding",
            "type": "markdown",
            "sourceId": source_ids["clean_metrics"],
            "body": (
                "## M2/M3 在 Clean 的综合分类质量领先，但误报与漏报仍需分开审计\n\n"
                f"{best_accuracy_model} 的 Accuracy、Precision、Recall、Specificity 与 F1 不能由单一 AUC 代替。"
                "下图比较固定阈值指标，完整表同时给出 NPV、Macro-F1、Balanced Accuracy、MCC、Brier、ECE 与低误报区 TPR。"
                "测试阈值来自内部 Small clean validation，未在本测试集上重新选择。"
            ),
        },
        {"id": "clean_chart_block", "type": "chart", "chartId": "clean_metric_chart", "layout": "full"},
        {"id": "clean_table_block", "type": "table", "tableId": "clean_metrics_table", "layout": "full"},
        {"id": "clean_ci_table_block", "type": "table", "tableId": "clean_ci_table", "layout": "full"},
        {
            "id": "confusion_finding",
            "type": "markdown",
            "sourceId": source_ids["clean_metrics"],
            "body": (
                "## 混淆矩阵揭示同一 Accuracy 背后的错误结构\n\n"
                "FP 是真实图片被误判为 AIGI，FN 是 AIGI 被漏判为真实。二者的业务代价通常不对称；因此本表保留四格计数及 FPR/FNR，方便后续按部署成本重新选择操作阈值。"
            ),
        },
        {"id": "confusion_table_block", "type": "table", "tableId": "confusion_table", "layout": "full"},
        {
            "id": "roc_finding",
            "type": "markdown",
            "sourceId": source_ids["roc_curve"],
            "body": (
                "## ROC 展示全阈值排序能力，不能单独回答正类命中质量\n\n"
                f"Clean AUROC 最高的是 **{best_auroc_model}**。ROC 对类别比例不敏感，但在低误报部署中应优先查看 TPR@1%FPR 与 TPR@5%FPR，而不是只比较整条曲线的面积。"
            ),
        },
        {"id": "roc_chart_block", "type": "chart", "chartId": "roc_chart", "layout": "full"},
        {
            "id": "pr_finding",
            "type": "markdown",
            "sourceId": source_ids["pr_curve"],
            "body": (
                "## PR 曲线补充了 AIGI Precision–Recall 的操作权衡\n\n"
                "PR 曲线以 AIGI 为正类，更直接反映提高召回时需要付出的误报代价。本测试集 AIGI 占比固定为 50%，所以随机基线 Precision 为 50%；部署先验更低时，实际 Precision 通常也会下降。"
            ),
        },
        {"id": "pr_chart_block", "type": "chart", "chartId": "pr_chart", "layout": "full"},
        {
            "id": "robust_finding",
            "type": "markdown",
            "sourceId": source_ids["robustness_summary"],
            "body": (
                "## 扰动均值与最坏值比 Clean 单点更接近部署风险\n\n"
                f"排除 Clean 后，{best_robust_model} 的 20 条件平均 Accuracy 最高。"
                "均值用于概括总体稳健性，最坏值用于暴露单一失效模式；二者均为描述性统计，不是未来扰动分布的概率保证。"
            ),
        },
        {"id": "robust_chart_block", "type": "chart", "chartId": "robustness_chart", "layout": "full"},
        {"id": "robust_table_block", "type": "table", "tableId": "robustness_summary_table", "layout": "full"},
        {
            "id": "generator_finding",
            "type": "markdown",
            "sourceId": source_ids["generator_recall"],
            "body": (
                "## 精确生成器切片用于定位 AIGI 漏检来源\n\n"
                "Unseen-generator 的 1,000 张 AIGI 来自 12 个训练未见精确生成器，每类 83–84 张。"
                "该切片没有配对真实负类，因此只能解释 Recall/TP/FN；不能从单生成器行推导 Precision、Specificity 或 Accuracy。"
            ),
        },
        {"id": "generator_chart_block", "type": "chart", "chartId": "generator_recall_chart", "layout": "full"},
        {"id": "generator_table_block", "type": "table", "tableId": "generator_recall_table", "layout": "full"},
        {
            "id": "real_source_finding",
            "type": "markdown",
            "sourceId": source_ids["real_specificity"],
            "body": (
                "## 真实来源切片用于定位误报来源\n\n"
                "1,000 张真实图片由 COCO、FFHQ、LAION、RAISE 各 250 张组成。"
                "Specificity 越低表示该来源越容易被误报为 AIGI；来源间差异可能混合内容、格式、分辨率和采集管线效应，不应直接解释为内容因果效应。"
            ),
        },
        {"id": "real_source_table_block", "type": "table", "tableId": "real_specificity_table", "layout": "full"},
        {
            "id": "all_conditions_finding",
            "type": "markdown",
            "sourceId": source_ids["condition_metrics"],
            "body": (
                "## 完整条件明细保留每个模型的失效证据\n\n"
                "下表覆盖 105 个模型-条件单元，并将原始 AUROC/AP/BA/Macro-F1/Recall/Specificity 扩展为 Accuracy、Precision、NPV、AIGI F1、MCC、FNR 与混淆计数。"
                "默认按 Accuracy 从低到高排序，便于优先检查最差条件。"
            ),
        },
        {"id": "condition_metrics_block", "type": "table", "tableId": "condition_metrics_table", "layout": "full"},
        {
            "id": "scope_definitions",
            "type": "markdown",
            "body": (
                "## 范围、标签与指标定义\n\n"
                "评测对象仅为冻结的 B0/B1/B2/M2/M3 checkpoint 在 `test_external_unseen_generator` 上的既有预测。"
                "正类固定为 AIGI（label=1），负类为 Real（label=0）。Exact generator 与 architecture family 均未在 Small 训练集出现；"
                "这是一项外部分布描述性评测，不代表所有未来生成器或真实图片来源。"
            ),
        },
        {"id": "metric_definition_block", "type": "table", "tableId": "metric_definition_table", "layout": "full"},
        {
            "id": "methodology",
            "type": "markdown",
            "sourceId": source_ids["condition_matrix"],
            "body": (
                "## 计算方法与扰动设计\n\n"
                "逐样本预测按 sample_id 与冻结 manifest 一一对齐，并核对标签、概率范围、每条件 2,000 行、每模型 42,000 行、run card manifest/checkpoint 摘要及 COMPLETE 标记。"
                "随后从概率和冻结阈值重新计算所有指标，并与既有 `metrics_by_transform.csv` 在 1e-6 容差内逐字段核对。"
                "Clean 的 95% 区间使用 Real/AIGI 分层非参数 bootstrap 1,000 次；ROC/PR 图为满足多序列渲染器的共同横轴要求，分别插值到 201 点递增 FPR/Recall 网格。"
                "表中 AUROC/AP 始终使用完整原始预测计算，不使用显示插值点重新估计。"
            ),
        },
        {"id": "condition_matrix_block", "type": "table", "tableId": "condition_matrix_table", "layout": "full"},
        {"id": "lineage_block", "type": "table", "tableId": "lineage_table", "layout": "full"},
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## 局限性、不确定性与稳健性边界\n\n"
                "- Bootstrap 区间只反映当前样本的抽样波动，不覆盖数据集构建偏差、生成器覆盖不足、训练随机性或 checkpoint 选择不确定性。\n"
                "- 五个模型共享同一测试集，模型差异的严格显著性应使用配对 bootstrap/置换检验并进行多重比较校正；本报告未据此声称显著性。\n"
                "- 测试集人为保持 1:1 类别平衡，Accuracy、Precision、NPV 不能直接外推到真实 AIGI 流行率。\n"
                "- 生成器与真实来源切片样本较小；83–84 张生成器图和 250 张来源图的点估计不应过度解读。\n"
                "- 固定阈值保证无测试泄漏，但阈值接近 1 也意味着概率校准与排序能力需分开判断。"
            ),
        },
        {
            "id": "recommendations",
            "type": "markdown",
            "sourceId": source_ids["robustness_summary"],
            "body": (
                "## 建议的模型选择与下一步验证\n\n"
                f"1. 若目标是当前平衡测试集上的综合准确率，优先审查 **{best_accuracy_model}**；若目标是跨阈值排序，优先审查 **{best_auroc_model}**。\n"
                f"2. 部署前以可接受 FPR 定义操作点，并在该约束下比较 Recall；不要直接沿用平衡验证集阈值。\n"
                f"3. 对 {best_robust_model} 及候选次优模型执行配对 bootstrap，并针对最坏扰动、最低召回生成器和最低 Specificity 真实来源补充样本。\n"
                "4. 使用接近实际流行率的回放集重新报告 Precision、NPV 与成本加权指标，并进行温度缩放等校准验证。"
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## 后续问题\n\n"
                "- 在 FPR ≤ 0.1%、1%、5% 三个部署约束下，哪个模型的配对置信下界最高？\n"
                "- 最坏生成器的漏检来自内容语义、图像格式、分辨率，还是生成后处理？\n"
                "- 真实流量中的 AIGI 比例和误报/漏报成本分别是多少，应如何设定阈值？\n"
                "- 增加新生成器后，模型排序和校准误差是否稳定？"
            ),
        },
    ]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Detailed threshold-dependent and threshold-free evaluation of B0/B1/B2/M2/M3 on the frozen Community Forensics unseen-generator test split.",
            "generatedAt": generated_at,
            "blocks": blocks,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": [{"id": source["id"], "label": source["label"]} for source in sources],
        },
        "snapshot": {
            "version": 1,
            "status": "ready",
            "generatedAt": generated_at,
            "datasets": snapshot,
        },
        "sources": sources,
    }


def generate(arguments: argparse.Namespace) -> None:
    generated_at = datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds")
    staged, audit_context = _load_and_compute(arguments)
    queries = {
        "headline": "SELECT * FROM headline",
        "clean_metrics": "SELECT * FROM clean_metrics ORDER BY model_order",
        "clean_metric_chart": "SELECT * FROM clean_metric_chart ORDER BY metric_order, model_order",
        "roc_curve": "SELECT * FROM roc_curve ORDER BY model_order, point_order",
        "pr_curve": "SELECT * FROM pr_curve ORDER BY model_order, point_order",
        "robustness_summary": "SELECT * FROM robustness_summary ORDER BY model_order",
        "robustness_chart": "SELECT * FROM robustness_chart ORDER BY group_order, model_order",
        "condition_metrics": "SELECT * FROM condition_metrics ORDER BY model_order, condition_index",
        "generator_recall": "SELECT * FROM generator_recall ORDER BY generator, model_order",
        "real_specificity": "SELECT * FROM real_specificity ORDER BY real_source, model_order",
        "condition_matrix": "SELECT * FROM condition_matrix ORDER BY condition_index",
        "metric_definitions": "SELECT * FROM metric_definitions ORDER BY metric",
        "model_lineage": "SELECT * FROM model_lineage ORDER BY model_order",
    }
    snapshot = _materialize_sql_snapshot(staged, queries)
    _write_csv(arguments.metrics_csv, snapshot["condition_metrics"])
    _write_csv(arguments.clean_metrics_csv, snapshot["clean_metrics"])
    slice_rows = [
        {
            "slice_type": "exact_generator",
            "model": row["model"],
            "model_order": row["model_order"],
            "slice_name": row["generator"],
            "n": row["n"],
            "correct": row["tp"],
            "errors": row["fn"],
            "metric": "recall",
            "metric_value": row["recall"],
            "mean_aigc_score": row["mean_score"],
        }
        for row in snapshot["generator_recall"]
    ] + [
        {
            "slice_type": "real_source",
            "model": row["model"],
            "model_order": row["model_order"],
            "slice_name": row["real_source"],
            "n": row["n"],
            "correct": row["tn"],
            "errors": row["fp"],
            "metric": "specificity",
            "metric_value": row["specificity"],
            "mean_aigc_score": row["mean_score"],
        }
        for row in snapshot["real_specificity"]
    ]
    _write_csv(arguments.slice_metrics_csv, slice_rows)
    artifact = _build_artifact(snapshot, queries, generated_at)
    _atomic_text(arguments.artifact_json, json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    audit = {
        "schema_version": 1,
        "generated_at_asia_singapore": generated_at,
        "report_job_id": os.environ.get("SLURM_JOB_ID"),
        "report_contract": {
            "delivery_mode": "portable_html",
            "audience": "technical",
            "scope": "Frozen Community Forensics test_external_unseen_generator predictions only",
            "positive_class": "AIGI (label=1)",
            "threshold_policy": "Per-model threshold frozen from internal Small clean validation; no test retuning",
            "uncertainty": f"{arguments.bootstrap_replicates} stratified bootstrap replicates for clean metrics",
        },
        **audit_context,
        "sql_snapshot_queries": queries,
        "output_files": {
            "metrics_csv": str(arguments.metrics_csv),
            "clean_metrics_csv": str(arguments.clean_metrics_csv),
            "slice_metrics_csv": str(arguments.slice_metrics_csv),
            "artifact_json": str(arguments.artifact_json),
        },
        "dataset_rows": {key: len(rows) for key, rows in snapshot.items()},
        "chart_map": [
            {"chart": "clean_metric_chart", "type": "grouped bar", "question": "Compare fixed-threshold clean classification metrics."},
            {"chart": "roc_chart", "type": "line", "question": "Compare TPR across FPR operating points."},
            {"chart": "pr_chart", "type": "line", "question": "Compare AIGI precision-recall tradeoffs."},
            {"chart": "robustness_chart", "type": "grouped bar", "question": "Compare clean, perturbation mean, and worst accuracy."},
            {"chart": "generator_recall_chart", "type": "grouped bar", "question": "Locate exact-generator recall failures."},
        ],
    }
    _atomic_text(arguments.audit_json, json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "event": "community_forensics_unseen_accuracy_report_complete",
        "models": len(MODELS),
        "conditions": len(CONDITION_LABELS),
        "prediction_rows": snapshot["headline"][0]["prediction_count"],
        "artifact_json": str(arguments.artifact_json),
        "audit_json": str(arguments.audit_json),
    }, sort_keys=True), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the detailed Community Forensics unseen-generator accuracy report")
    parser.add_argument("--evaluation-root", default="outputs/community_forensics_robustness_v2")
    parser.add_argument("--manifest", default="data/manifests/community_forensics_test_external_unseen_generator.csv")
    parser.add_argument("--matrix", default="configs/community_forensics_robustness_v2.yaml")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260829)
    parser.add_argument("--metrics-csv", default="reports/evaluations/unseen_generator/community_forensics_unseen_generator_all_metrics.csv")
    parser.add_argument("--clean-metrics-csv", default="reports/evaluations/unseen_generator/community_forensics_unseen_generator_clean_metrics.csv")
    parser.add_argument("--slice-metrics-csv", default="reports/evaluations/unseen_generator/community_forensics_unseen_generator_slice_metrics.csv")
    parser.add_argument("--artifact-json", default="reports/evaluations/unseen_generator/community_forensics_unseen_generator_accuracy_artifact.json")
    parser.add_argument("--audit-json", default="reports/evaluations/unseen_generator/community_forensics_unseen_generator_accuracy_notes.json")
    return parser.parse_args()


def main() -> None:
    generate(_parse_args())


if __name__ == "__main__":
    main()
