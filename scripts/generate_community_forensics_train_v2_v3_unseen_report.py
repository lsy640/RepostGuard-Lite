from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import yaml

import generate_community_forensics_unseen_accuracy_report as unseen


MODELS = unseen.MODELS
MODEL_ORDER = {model: index for index, model in enumerate(MODELS)}
CONDITION_LABELS = unseen.CONDITION_LABELS
DELTA_FIELDS = ("accuracy", "precision", "recall", "specificity", "f1", "mcc", "auroc", "average_precision")
OPERATING_FIELDS = (
    ("Accuracy", "accuracy"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("Specificity", "specificity"),
    ("F1", "f1"),
)
MULTISTAGE_INDICES = (18, 19, 20)


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


def _round(value: float) -> float:
    return round(float(value), 8)


def _profile_training(path: str | Path) -> dict[str, Any]:
    rows = _read_csv(path)
    real = [row for row in rows if int(row["label"]) == 0]
    aigi = [row for row in rows if int(row["label"]) == 1]
    architectures: dict[str, int] = defaultdict(int)
    for row in aigi:
        architectures[row["architecture"]] += 1
    return {
        "rows": rows,
        "count": len(rows),
        "real": len(real),
        "aigi": len(aigi),
        "generators": sorted({row["canonical_generator_id"] for row in aigi}),
        "architectures": dict(architectures),
        "sha256": _sha256(path),
    }


def _identity_overlap(training_rows: list[dict[str, str]], test_rows: list[dict[str, str]]) -> dict[str, int]:
    overlap: dict[str, int] = {}
    for field in ("path", "sample_id", "sha256", "source_locator"):
        training_values = {row[field] for row in training_rows if row.get(field)}
        test_values = {row[field] for row in test_rows if row.get(field)}
        overlap[field] = len(training_values & test_values)
    return overlap


def _paired_delta_intervals(
    labels: np.ndarray,
    v2_scores: np.ndarray,
    v3_scores: np.ndarray,
    v2_threshold: float,
    v3_threshold: float,
    replicates: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(seed)
    samples = {field: np.empty(replicates, dtype=np.float64) for field in DELTA_FIELDS}
    for replicate in range(replicates):
        selected = np.concatenate(
            (rng.choice(negative, negative.size, replace=True), rng.choice(positive, positive.size, replace=True))
        )
        v2_metrics = unseen._metrics(labels[selected], v2_scores[selected], v2_threshold)
        v3_metrics = unseen._metrics(labels[selected], v3_scores[selected], v3_threshold)
        for field in DELTA_FIELDS:
            samples[field][replicate] = float(v3_metrics[field]) - float(v2_metrics[field])
    return {
        field: (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))
        for field, values in samples.items()
    }


def _load_protocol(
    protocol: str,
    evaluation_root: str | Path,
    source_root: str | Path,
    split_directory: str,
    comparison_manifest_path: str | Path,
    evaluation_manifest_path: str | Path,
    matrix_path: str | Path,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    evaluation_root = Path(evaluation_root)
    source_root = Path(source_root)
    manifest_path = Path(comparison_manifest_path)
    evaluation_manifest_path = Path(evaluation_manifest_path)
    matrix_path = Path(matrix_path)
    manifest_rows = _read_csv(manifest_path)
    manifest_by_id = {row["sample_id"]: row for row in manifest_rows}
    if len(manifest_rows) != 2000 or len(manifest_by_id) != 2000:
        raise RuntimeError("Unseen intersection manifest must contain 2,000 unique rows")
    labels_by_id = {sample_id: int(row["label"]) for sample_id, row in manifest_by_id.items()}
    if sum(labels_by_id.values()) != 1000:
        raise RuntimeError("Unseen intersection manifest must contain 1,000 Real and 1,000 AIGI")
    evaluation_manifest_rows = _read_csv(evaluation_manifest_path)
    evaluation_manifest_by_id = {row["sample_id"]: row for row in evaluation_manifest_rows}
    if len(evaluation_manifest_by_id) != len(evaluation_manifest_rows):
        raise RuntimeError(f"Duplicate sample_id in evaluation manifest: {evaluation_manifest_path}")
    if not set(manifest_by_id).issubset(evaluation_manifest_by_id):
        raise RuntimeError(f"Comparison manifest is not a subset of {evaluation_manifest_path}")
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))["evaluation"]
    if len(matrix) != 21 or len(matrix) != len(CONDITION_LABELS):
        raise RuntimeError("Expected the frozen 21-condition robustness matrix")

    condition_metrics: list[dict[str, Any]] = []
    clean_metrics: list[dict[str, Any]] = []
    robustness: list[dict[str, Any]] = []
    generators: list[dict[str, Any]] = []
    real_sources: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    clean_arrays: dict[str, dict[str, Any]] = {}
    input_audit: dict[str, Any] = {}

    generator_indices: dict[str, list[int]] = defaultdict(list)
    real_source_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(manifest_rows):
        if int(row["label"]) == 1:
            generator_indices[row["canonical_generator_id"]].append(index)
        else:
            real_source_indices[row["real_source"]].append(index)

    for model in MODELS:
        model_order = MODEL_ORDER[model]
        output = evaluation_root / model / split_directory
        for required in ("COMPLETE", "metrics_by_transform.csv", "predictions.jsonl", "run_card.json", "summary.json"):
            if not (output / required).is_file():
                raise RuntimeError(f"Missing completed evaluation artifact: {output / required}")
        existing_metrics = _read_csv(output / "metrics_by_transform.csv")
        if len(existing_metrics) != 21:
            raise RuntimeError(f"Expected 21 metric rows for {protocol}/{model}")
        run_card = _read_json(output / "run_card.json")
        summary = _read_json(output / "summary.json")
        source_summary = _read_json(source_root / model / "summary.json")
        source_run_card = _read_json(source_root / model / "run_card.json")
        threshold = float(source_summary["threshold_from_clean_validation"])
        if abs(threshold - float(summary["threshold_from_clean_validation"])) > 1e-12:
            raise RuntimeError(f"Frozen threshold mismatch for {protocol}/{model}")
        if run_card["val_manifest_sha256"] != _sha256(evaluation_manifest_path):
            raise RuntimeError(f"Manifest mismatch for {protocol}/{model}")
        if run_card["checkpoint_sha256"] != source_run_card["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint lineage mismatch for {protocol}/{model}")
        if run_card["evaluation_matrix_sha256"] != _sha256(matrix_path):
            raise RuntimeError(f"Matrix mismatch for {protocol}/{model}")

        predictions = _read_jsonl(output / "predictions.jsonl")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in predictions:
            grouped[str(row["transform"])].append(row)
        expected_prediction_rows = len(evaluation_manifest_rows) * 21
        if len(predictions) != expected_prediction_rows or len(grouped) != 21:
            raise RuntimeError(f"Prediction cardinality mismatch for {protocol}/{model}")

        per_model: list[dict[str, Any]] = []
        clean_labels: np.ndarray | None = None
        clean_scores: np.ndarray | None = None
        for condition_index, (existing, specification) in enumerate(zip(existing_metrics, matrix, strict=True)):
            transform = existing["transform"]
            if existing["transform_name"] != specification["name"]:
                raise RuntimeError(f"Condition-order drift for {protocol}/{model}/{condition_index}")
            row_by_id = {str(row["sample_id"]): row for row in grouped[transform]}
            if len(row_by_id) != len(evaluation_manifest_rows) or not set(manifest_by_id).issubset(row_by_id):
                raise RuntimeError(f"Intersection sample identity mismatch for {protocol}/{model}/{transform}")
            ordered = [row_by_id[row["sample_id"]] for row in manifest_rows]
            labels = np.asarray([int(row["label"]) for row in ordered], dtype=np.int64)
            expected_labels = np.asarray([labels_by_id[row["sample_id"]] for row in manifest_rows], dtype=np.int64)
            if not np.array_equal(labels, expected_labels):
                raise RuntimeError(f"Label mismatch for {protocol}/{model}/{transform}")
            scores = np.asarray([float(row["pred"]) for row in ordered], dtype=np.float64)
            if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
                raise RuntimeError(f"Invalid probability for {protocol}/{model}/{transform}")
            computed = unseen._metrics(labels, scores, threshold)
            if len(evaluation_manifest_rows) == len(manifest_rows):
                unseen._verify_existing_metrics(computed, existing, f"{protocol}/{model}", transform)
            row = unseen._round_row({
                "protocol": protocol,
                "protocol_order": 0 if protocol == "train-v2" else 1,
                "model": model.upper(),
                "model_order": model_order,
                "condition_index": condition_index,
                "condition_group": unseen._condition_group(condition_index),
                "condition": CONDITION_LABELS[condition_index],
                "transform": transform,
                **computed,
            })
            condition_metrics.append(row)
            per_model.append(row)
            if condition_index == 0:
                clean_labels = labels
                clean_scores = scores

        if clean_labels is None or clean_scores is None:
            raise RuntimeError(f"Missing clean predictions for {protocol}/{model}")
        clean = dict(per_model[0])
        intervals = unseen._bootstrap_clean(
            clean_labels,
            clean_scores,
            threshold,
            bootstrap_replicates,
            bootstrap_seed + model_order + (100 if protocol == "train-v3" else 0),
        )
        for field, (lower, upper) in intervals.items():
            clean[f"{field}_ci_low"] = _round(lower)
            clean[f"{field}_ci_high"] = _round(upper)
            clean[f"{field}_ci_95"] = f"{100 * lower:.2f}%–{100 * upper:.2f}%"
        clean_metrics.append(clean)
        clean_arrays[model] = {"labels": clean_labels, "scores": clean_scores, "threshold": threshold}

        clean_predictions = (clean_scores >= threshold).astype(np.int64)
        for generator in sorted(generator_indices):
            indices = np.asarray(generator_indices[generator], dtype=np.int64)
            tp = int(clean_predictions[indices].sum())
            generators.append({
                "protocol": protocol,
                "protocol_order": 0 if protocol == "train-v2" else 1,
                "model": model.upper(),
                "model_order": model_order,
                "generator": generator,
                "n": int(indices.size),
                "tp": tp,
                "fn": int(indices.size - tp),
                "recall": _round(tp / indices.size),
                "mean_score": _round(clean_scores[indices].mean()),
            })
        for source in sorted(real_source_indices):
            indices = np.asarray(real_source_indices[source], dtype=np.int64)
            fp = int(clean_predictions[indices].sum())
            tn = int(indices.size - fp)
            real_sources.append({
                "protocol": protocol,
                "protocol_order": 0 if protocol == "train-v2" else 1,
                "model": model.upper(),
                "model_order": model_order,
                "real_source": source,
                "n": int(indices.size),
                "tn": tn,
                "fp": fp,
                "specificity": _round(tn / indices.size),
                "false_positive_rate": _round(fp / indices.size),
                "mean_score": _round(clean_scores[indices].mean()),
            })

        perturbed = per_model[1:]
        worst_accuracy = min(perturbed, key=lambda row: float(row["accuracy"]))
        worst_auroc = min(perturbed, key=lambda row: float(row["auroc"]))
        robustness.append({
            "protocol": protocol,
            "protocol_order": 0 if protocol == "train-v2" else 1,
            "model": model.upper(),
            "model_order": model_order,
            "clean_accuracy": clean["accuracy"],
            "clean_auroc": clean["auroc"],
            "mean_20_accuracy": _round(fmean(float(row["accuracy"]) for row in perturbed)),
            "worst_20_accuracy": worst_accuracy["accuracy"],
            "worst_accuracy_condition": worst_accuracy["condition"],
            "mean_20_auroc": _round(fmean(float(row["auroc"]) for row in perturbed)),
            "worst_20_auroc": worst_auroc["auroc"],
            "worst_auroc_condition": worst_auroc["condition"],
            "mean_20_recall": _round(fmean(float(row["recall"]) for row in perturbed)),
            "mean_20_specificity": _round(fmean(float(row["specificity"]) for row in perturbed)),
        })
        lineage.append({
            "protocol": protocol,
            "protocol_order": 0 if protocol == "train-v2" else 1,
            "model": model.upper(),
            "model_order": model_order,
            "architecture": unseen.MODEL_ARCHITECTURES[model],
            "threshold": _round(threshold),
            "checkpoint_sha256": run_card["checkpoint_sha256"],
            "train_manifest_sha256": run_card["train_manifest_sha256"],
            "test_manifest_sha256": run_card["val_manifest_sha256"],
            "matrix_sha256": run_card["evaluation_matrix_sha256"],
            "evaluation_job": str(run_card["slurm_job_id"]),
            "prediction_rows": len(predictions),
        })
        input_audit[model] = {
            "evaluation_directory": str(output),
            "evaluation_manifest": str(evaluation_manifest_path),
            "evaluation_manifest_sha256": _sha256(evaluation_manifest_path),
            "original_prediction_rows": len(predictions),
            "intersection_prediction_rows": len(manifest_rows) * 21,
            "checkpoint_sha256": run_card["checkpoint_sha256"],
            "metrics_sha256": _sha256(output / "metrics_by_transform.csv"),
            "predictions_sha256": _sha256(output / "predictions.jsonl"),
            "run_card_sha256": _sha256(output / "run_card.json"),
            "summary_sha256": _sha256(output / "summary.json"),
        }

    return {
        "condition_metrics": condition_metrics,
        "clean_metrics": clean_metrics,
        "robustness": robustness,
        "generators": generators,
        "real_sources": real_sources,
        "lineage": lineage,
        "clean_arrays": clean_arrays,
        "input_audit": input_audit,
    }


def _wide_comparisons(
    v2: dict[str, Any],
    v3: dict[str, Any],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, list[dict[str, Any]]]:
    clean_v2 = {row["model"].lower(): row for row in v2["clean_metrics"]}
    clean_v3 = {row["model"].lower(): row for row in v3["clean_metrics"]}
    clean_comparison: list[dict[str, Any]] = []
    clean_ranking_chart: list[dict[str, Any]] = []
    clean_operating_chart: list[dict[str, Any]] = []
    for model in MODELS:
        row2 = clean_v2[model]
        row3 = clean_v3[model]
        arrays2 = v2["clean_arrays"][model]
        arrays3 = v3["clean_arrays"][model]
        if not np.array_equal(arrays2["labels"], arrays3["labels"]):
            raise RuntimeError(f"Paired labels differ for {model}")
        intervals = _paired_delta_intervals(
            arrays2["labels"], arrays2["scores"], arrays3["scores"],
            arrays2["threshold"], arrays3["threshold"], bootstrap_replicates,
            bootstrap_seed + MODEL_ORDER[model],
        )
        row: dict[str, Any] = {
            "model": model.upper(),
            "model_order": MODEL_ORDER[model],
            "v2_threshold": row2["threshold"],
            "v3_threshold": row3["threshold"],
        }
        for field in ("accuracy", "precision", "recall", "specificity", "npv", "f1", "macro_f1", "balanced_accuracy", "mcc", "auroc", "average_precision", "brier", "ece_15", "tpr_at_fpr_1pct", "tpr_at_fpr_5pct"):
            row[f"v2_{field}"] = row2[field]
            row[f"v3_{field}"] = row3[field]
            row[f"delta_{field}"] = _round(float(row3[field]) - float(row2[field]))
        for field, (lower, upper) in intervals.items():
            row[f"delta_{field}_ci_low"] = _round(lower)
            row[f"delta_{field}_ci_high"] = _round(upper)
            row[f"delta_{field}_ci_95"] = f"{100 * lower:+.2f}–{100 * upper:+.2f} pp"
        row.update({
            "v2_tn": row2["tn"], "v2_fp": row2["fp"], "v2_fn": row2["fn"], "v2_tp": row2["tp"],
            "v3_tn": row3["tn"], "v3_fp": row3["fp"], "v3_fn": row3["fn"], "v3_tp": row3["tp"],
        })
        clean_comparison.append(row)
        for protocol, source in (("train-v2", row2), ("train-v3", row3)):
            for metric in ("auroc", "average_precision"):
                clean_ranking_chart.append({
                    "model": model.upper(), "model_order": MODEL_ORDER[model],
                    "series": f"{protocol} {metric.upper() if metric == 'auroc' else 'AP'}",
                    "value": source[metric],
                })
            for metric_label, metric_field in OPERATING_FIELDS:
                clean_operating_chart.append({
                    "model_version": f"{model.upper()} {protocol[-2:]}",
                    "model_order": MODEL_ORDER[model],
                    "protocol": protocol,
                    "metric": metric_label,
                    "value": source[metric_field],
                })

    condition_v2 = {(row["model"], row["condition_index"]): row for row in v2["condition_metrics"]}
    condition_v3 = {(row["model"], row["condition_index"]): row for row in v3["condition_metrics"]}
    condition_comparison: list[dict[str, Any]] = []
    condition_delta_chart: list[dict[str, Any]] = []
    multistage: list[dict[str, Any]] = []
    multistage_chart: list[dict[str, Any]] = []
    for model in MODELS:
        display_model = model.upper()
        for condition_index in range(21):
            row2 = condition_v2[(display_model, condition_index)]
            row3 = condition_v3[(display_model, condition_index)]
            if row2["transform"] != row3["transform"]:
                raise RuntimeError(f"Transform drift for {model}/{condition_index}")
            row = {
                "model": display_model,
                "model_order": MODEL_ORDER[model],
                "condition_index": condition_index,
                "condition_group": row2["condition_group"],
                "condition": row2["condition"],
                "transform": row2["transform"],
            }
            for field in ("accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy", "mcc", "auroc", "average_precision", "brier", "ece_15", "tpr_at_fpr_1pct", "tpr_at_fpr_5pct"):
                row[f"v2_{field}"] = row2[field]
                row[f"v3_{field}"] = row3[field]
                row[f"delta_{field}"] = _round(float(row3[field]) - float(row2[field]))
            condition_comparison.append(row)
            condition_delta_chart.append({
                "model": display_model,
                "model_order": MODEL_ORDER[model],
                "condition_index": condition_index,
                "condition": row2["condition"],
                "delta_auroc": row["delta_auroc"],
            })
            if condition_index in MULTISTAGE_INDICES:
                multistage.append(row)
                for protocol, source in (("train-v2", row2), ("train-v3", row3)):
                    multistage_chart.append({
                        "model": display_model,
                        "model_order": MODEL_ORDER[model],
                        "series": f"{protocol} · {row2['condition']}",
                        "auroc": source["auroc"],
                    })

    robust_v2 = {row["model"].lower(): row for row in v2["robustness"]}
    robust_v3 = {row["model"].lower(): row for row in v3["robustness"]}
    robustness_comparison: list[dict[str, Any]] = []
    robustness_chart: list[dict[str, Any]] = []
    for model in MODELS:
        row2 = robust_v2[model]
        row3 = robust_v3[model]
        row = {"model": model.upper(), "model_order": MODEL_ORDER[model]}
        for field in ("clean_accuracy", "clean_auroc", "mean_20_accuracy", "worst_20_accuracy", "mean_20_auroc", "worst_20_auroc", "mean_20_recall", "mean_20_specificity"):
            row[f"v2_{field}"] = row2[field]
            row[f"v3_{field}"] = row3[field]
            row[f"delta_{field}"] = _round(float(row3[field]) - float(row2[field]))
        row["v2_worst_accuracy_condition"] = row2["worst_accuracy_condition"]
        row["v3_worst_accuracy_condition"] = row3["worst_accuracy_condition"]
        row["v2_worst_auroc_condition"] = row2["worst_auroc_condition"]
        row["v3_worst_auroc_condition"] = row3["worst_auroc_condition"]
        robustness_comparison.append(row)
        for protocol, source in (("train-v2", row2), ("train-v3", row3)):
            for label, field in (("Clean AUROC", "clean_auroc"), ("20-condition mean AUROC", "mean_20_auroc"), ("Worst AUROC", "worst_20_auroc")):
                robustness_chart.append({
                    "model": model.upper(), "model_order": MODEL_ORDER[model],
                    "series": f"{protocol} {label}", "auroc": source[field],
                })

    generator_v2 = {(row["model"], row["generator"]): row for row in v2["generators"]}
    generator_v3 = {(row["model"], row["generator"]): row for row in v3["generators"]}
    generator_comparison: list[dict[str, Any]] = []
    generator_delta_chart: list[dict[str, Any]] = []
    for key in sorted(generator_v2, key=lambda item: (MODEL_ORDER[item[0].lower()], item[1])):
        row2 = generator_v2[key]
        row3 = generator_v3[key]
        row = {
            "model": key[0], "model_order": MODEL_ORDER[key[0].lower()], "generator": key[1], "n": row2["n"],
            "v2_tp": row2["tp"], "v2_fn": row2["fn"], "v2_recall": row2["recall"], "v2_mean_score": row2["mean_score"],
            "v3_tp": row3["tp"], "v3_fn": row3["fn"], "v3_recall": row3["recall"], "v3_mean_score": row3["mean_score"],
            "delta_recall": _round(float(row3["recall"]) - float(row2["recall"])),
        }
        generator_comparison.append(row)
        generator_delta_chart.append({
            "model": key[0], "model_order": MODEL_ORDER[key[0].lower()], "generator": key[1],
            "delta_recall": row["delta_recall"],
        })

    real_v2 = {(row["model"], row["real_source"]): row for row in v2["real_sources"]}
    real_v3 = {(row["model"], row["real_source"]): row for row in v3["real_sources"]}
    real_comparison: list[dict[str, Any]] = []
    for key in sorted(real_v2, key=lambda item: (MODEL_ORDER[item[0].lower()], item[1])):
        row2 = real_v2[key]
        row3 = real_v3[key]
        real_comparison.append({
            "model": key[0], "model_order": MODEL_ORDER[key[0].lower()], "real_source": key[1], "n": row2["n"],
            "v2_tn": row2["tn"], "v2_fp": row2["fp"], "v2_specificity": row2["specificity"], "v2_fpr": row2["false_positive_rate"],
            "v3_tn": row3["tn"], "v3_fp": row3["fp"], "v3_specificity": row3["specificity"], "v3_fpr": row3["false_positive_rate"],
            "delta_specificity": _round(float(row3["specificity"]) - float(row2["specificity"])),
        })

    return {
        "clean_comparison": clean_comparison,
        "clean_ranking_chart": clean_ranking_chart,
        "clean_operating_chart": clean_operating_chart,
        "condition_comparison": condition_comparison,
        "condition_delta_chart": condition_delta_chart,
        "multistage_comparison": multistage,
        "multistage_chart": multistage_chart,
        "robustness_comparison": robustness_comparison,
        "robustness_chart": robustness_chart,
        "generator_comparison": generator_comparison,
        "generator_delta_chart": generator_delta_chart,
        "real_source_comparison": real_comparison,
    }


def _columns(*values: tuple[str, str, str | None]) -> list[dict[str, Any]]:
    columns: list[dict[str, Any]] = []
    for field, label, format_name in values:
        column: dict[str, Any] = {"field": field, "label": label}
        if format_name:
            column["format"] = format_name
        columns.append(column)
    return columns


def _source(dataset: str, query: str, generated_at: str) -> dict[str, Any]:
    return {
        "id": f"{dataset}_sql",
        "label": f"Reviewed {dataset.replace('_', ' ')} snapshot",
        "query": {
            "engine": "sqlite", "language": "sql", "sql": query,
            "description": "Generated only after shared-manifest identity, label, matrix, threshold, checkpoint-lineage, cardinality and published-metric reconciliation checks.",
            "executed_at": generated_at, "tables_used": [dataset],
            "filters": ["split = v2/v3 strict unseen sample-id intersection", "positive class = AIGI", "threshold = protocol-specific frozen internal-validation threshold"],
        },
    }


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _build_markdown(snapshot: dict[str, list[dict[str, Any]]], audit: dict[str, Any]) -> str:
    headline = snapshot["headline"][0]
    clean = snapshot["clean_comparison"]
    robust = snapshot["robustness_comparison"]
    multistage = snapshot["multistage_comparison"]
    lines = [
        "# Community Forensics train-v2 / train-v3：Strict Unseen-generator 交集对比报告",
        "",
        f"> 生成时间：{audit['generated_at_asia_singapore']}  ",
        f"> 评测 manifest SHA256：`{audit['test_manifest_sha256']}`  ",
        f"> 扰动矩阵 SHA256：`{audit['matrix_sha256']}`",
        "",
        "## 结论摘要",
        "",
        f"- train-v3 的 Clean AUROC 最优模型为 **{headline['best_v3_clean_auroc_model']}**（{_pct(headline['best_v3_clean_auroc'])}）。",
        f"- train-v3 的 Clean Accuracy 最优模型为 **{headline['best_v3_clean_accuracy_model']}**（{_pct(headline['best_v3_clean_accuracy'])}，使用内部验证集冻结阈值）。",
        f"- train-v3 的 20 个非 Clean 条件平均 AUROC 最优模型为 **{headline['best_v3_robust_auroc_model']}**（{_pct(headline['best_v3_robust_auroc'])}）。",
        f"- 五个模型中有 **{headline['positive_clean_auroc_deltas']}/5** 个在相同 unseen 交集上的 Clean AUROC 高于 train-v2；有 **{headline['positive_robust_auroc_deltas']}/5** 个在扰动平均 AUROC 上提高。",
        "- 这是 v2/v3 评价集 2,000 张 sample-id 交集、同一 21 条件矩阵下的公平模型版本比较；阈值仍分别来自各训练版本的内部验证集，没有使用 external test 标签调参。",
        "",
        "## 评测合同与数据范围",
        "",
        "- 测试角色：外部 strict unseen-generator，仅用于最终评测，不参与 checkpoint、阈值或模型选择。",
        "- 样本：2,000 张交集图片，1,000 Real / 1,000 AIGI；12 个训练未见精确生成器；COCO、FFHQ、LAION、RAISE 各 250 张真实图片。",
        "- 条件：Clean + 17 个既有单/双阶段条件 + 两组 4-stage + 一组随机 6-stage，共 21 条件。",
        "- 预测总量：2 个训练版本 × 5 个模型 × 2,000 张 × 21 条件 = 420,000 条。",
        "- 正类为 AIGI（label=1），负类为 Real（label=0）。Accuracy、Precision 和 NPV 基于人为 50% AIGI 比例，不能直接外推到生产流量。",
        "",
        "## 训练集变化",
        "",
        "| 指标 | train-v2 | train-v3 | 变化 |",
        "|---|---:|---:|---:|",
    ]
    for row in snapshot["training_comparison"]:
        lines.append(f"| {row['metric']} | {row['train_v2']} | {row['train_v3']} | {row['delta']:+d} |")
    lines.extend([
        "",
        "train-v3 在 train-v2 的 20,000 张基础上增加 1,000 张 GAN AIGI、1,000 张 pixel-diffusion AIGI 和 2,000 张均衡真实图片。多个因素同时变化，因此结果不能唯一归因于某一种新增生成器类别。",
        "",
        "## Clean 指标对比",
        "",
        "| 模型 | v2 Acc | v3 Acc | ΔAcc | v2 Recall | v3 Recall | v2 Spec | v3 Spec | v2 AUROC | v3 AUROC | ΔAUROC | ΔAUROC 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in clean:
        lines.append(
            f"| {row['model']} | {_pct(row['v2_accuracy'])} | {_pct(row['v3_accuracy'])} | {100 * row['delta_accuracy']:+.2f} pp | "
            f"{_pct(row['v2_recall'])} | {_pct(row['v3_recall'])} | {_pct(row['v2_specificity'])} | {_pct(row['v3_specificity'])} | "
            f"{row['v2_auroc']:.4f} | {row['v3_auroc']:.4f} | {row['delta_auroc']:+.4f} | {row['delta_auroc_ci_95']} |"
        )
    lines.extend([
        "",
        "AUROC/AP 衡量跨阈值排序能力；Accuracy、Recall、Specificity、F1 和 MCC 衡量各版本自身冻结阈值下的操作点。高 AUROC 不能自动修复不合适的冻结阈值。",
        "",
        "## 20 个非 Clean 条件汇总",
        "",
        "| 模型 | v2 mean AUROC | v3 mean AUROC | Δ | v2 worst AUROC | v3 worst AUROC | v2 mean Acc | v3 mean Acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in robust:
        lines.append(
            f"| {row['model']} | {row['v2_mean_20_auroc']:.4f} | {row['v3_mean_20_auroc']:.4f} | {row['delta_mean_20_auroc']:+.4f} | "
            f"{row['v2_worst_20_auroc']:.4f} | {row['v3_worst_20_auroc']:.4f} | {_pct(row['v2_mean_20_accuracy'])} | {_pct(row['v3_mean_20_accuracy'])} |"
        )
    lines.extend([
        "",
        "## 多阶段共同扰动",
        "",
        "| 模型 | 条件 | v2 AUROC | v3 AUROC | ΔAUROC | v2 Acc | v3 Acc |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in multistage:
        lines.append(
            f"| {row['model']} | {row['condition']} | {row['v2_auroc']:.4f} | {row['v3_auroc']:.4f} | {row['delta_auroc']:+.4f} | {_pct(row['v2_accuracy'])} | {_pct(row['v3_accuracy'])} |"
        )
    lines.extend([
        "",
        "## 精确生成器与真实来源切片",
        "",
        "完整 CSV/HTML 表分别给出 12 个精确生成器的 Clean Recall/TP/FN 与四个真实来源的 Clean Specificity/TN/FP。生成器切片只有正类，不能单独定义 Precision、Specificity 或 Accuracy；真实来源切片只有负类，不能单独定义 Recall。每个生成器比较复用同一真实面板时，其 ROC 估计彼此相关，本报告仅作诊断。",
        "",
        "## 谱系与完整性检查",
        "",
        f"- train-v2/test 精确身份重叠：`{audit['identity_overlap']['train_v2']}`。",
        f"- train-v3/test 精确身份重叠：`{audit['identity_overlap']['train_v3']}`。",
        "- 每个模型均核对 COMPLETE、run card、checkpoint SHA256、评价 manifest SHA256、matrix SHA256、交集 sample_id 一一对齐、标签一致和概率范围；v2 的 42,000 条交集预测还与既有 metrics_by_transform.csv 复算一致，v3 指标则从 84,000 条扩展集预测中筛选 42,000 条交集预测后重新计算。",
        "- Clean 差值区间使用共享样本的 Real/AIGI 分层配对 bootstrap；区间不覆盖训练随机种子、checkpoint 选择或数据集构建偏差。",
        "",
        "## 局限性与下一步",
        "",
        "1. 当前每个训练版本只有一个随机种子；小幅差值不应被称为架构层面的统计优势。",
        "2. 训练集同时改变生成器、真实来源与样本规模，无法从本实验唯一识别因果来源。",
        "3. 部署前应在独立 calibration set 上定义可接受 FPR，并报告 TPR@0.1%、1%、5% FPR；不能利用本 strict-unseen test 重新选阈值。",
        "4. 建议至少训练三个种子，并进行样本配对、生成器分层的 bootstrap/置换检验及多重比较校正。",
        "5. 使用贴近部署 AIGI 流行率的回放流量重新估计 Precision、NPV 和成本加权指标。",
        "",
        "## 结构化产物",
        "",
        "- `reports/evaluations/train_v2_v3_unseen_intersection_comparison/community_forensics_train_v2_v3_unseen_intersection_clean.csv`",
        "- `reports/evaluations/train_v2_v3_unseen_intersection_comparison/community_forensics_train_v2_v3_unseen_intersection_all_conditions.csv`",
        "- `reports/evaluations/train_v2_v3_unseen_intersection_comparison/community_forensics_train_v2_v3_unseen_intersection_multistage.csv`",
        "- `reports/evaluations/train_v2_v3_unseen_intersection_comparison/community_forensics_train_v2_v3_unseen_intersection_generator.csv`",
        "- `reports/evaluations/train_v2_v3_unseen_intersection_comparison/community_forensics_train_v2_v3_unseen_intersection_real_source.csv`",
        "- `reports/evaluations/train_v2_v3_unseen_intersection_comparison/community_forensics_train_v2_v3_unseen_intersection_artifact.json`",
        "- `reports/evaluations/train_v2_v3_unseen_intersection_comparison/community_forensics_train_v2_v3_unseen_intersection_audit.json`",
        "",
    ])
    return "\n".join(lines)


def _build_artifact(snapshot: dict[str, list[dict[str, Any]]], queries: dict[str, str], generated_at: str) -> dict[str, Any]:
    title = "Community Forensics train-v2 / train-v3 Strict Unseen-generator 交集对比"
    sources = [_source(dataset, query, generated_at) for dataset, query in queries.items()]
    source_ids = {dataset: f"{dataset}_sql" for dataset in queries}
    headline = snapshot["headline"][0]
    cards = [
        {"id": "best_v3_auc", "description": f"最佳模型 {headline['best_v3_clean_auroc_model']}。", "dataset": "headline", "sourceId": source_ids["headline"], "metrics": [{"label": "v3 Clean AUROC", "field": "best_v3_clean_auroc", "format": "percent"}]},
        {"id": "best_v3_acc", "description": f"冻结阈值；最佳模型 {headline['best_v3_clean_accuracy_model']}。", "dataset": "headline", "sourceId": source_ids["headline"], "metrics": [{"label": "v3 Clean Accuracy", "field": "best_v3_clean_accuracy", "format": "percent"}]},
        {"id": "best_v3_robust", "description": f"20 个非 Clean 条件；最佳模型 {headline['best_v3_robust_auroc_model']}。", "dataset": "headline", "sourceId": source_ids["headline"], "metrics": [{"label": "v3 扰动平均 AUROC", "field": "best_v3_robust_auroc", "format": "percent"}]},
        {"id": "positive_deltas", "description": "相同 unseen 交集上的 Clean AUROC。", "dataset": "headline", "sourceId": source_ids["headline"], "metrics": [{"label": "AUROC 提升模型数", "field": "positive_clean_auroc_deltas", "format": "number"}]},
    ]
    charts = [
        {
            "id": "clean_ranking_chart", "title": "Clean AUROC 与 Average Precision", "subtitle": "同一 2,000 张 v2/v3 unseen 交集；阈值无关指标。", "type": "bar", "intent": "comparison",
            "question": "train-v3 是否改善各模型的 Clean 排序能力？", "rationale": "模型和版本为离散类别，使用分组柱状图。",
            "comparisonContext": {"unit": "rate", "grain": "model by protocol and metric"}, "dataset": "clean_ranking_chart", "sourceId": source_ids["clean_ranking_chart"],
            "encodings": {"x": {"field": "model", "type": "nominal", "label": "Model"}, "y": {"field": "value", "type": "quantitative", "label": "Score", "format": "percent"}, "color": {"field": "series", "type": "nominal", "label": "Protocol / metric"}, "tooltip": [{"field": "value", "type": "quantitative", "label": "Value", "format": "percent"}]},
            "palette": {"kind": "categorical"}, "legend": {"position": "bottom", "interactive": True}, "labels": {"values": "none"}, "valueFormat": "percent", "layout": "full",
        },
        {
            "id": "clean_operating_chart", "title": "Clean 冻结阈值分类指标", "subtitle": "每个版本使用自身内部验证集冻结阈值。", "type": "bar", "intent": "comparison",
            "question": "v2/v3 的误报与漏报平衡如何变化？", "rationale": "同尺度操作指标适合分组柱状图。",
            "comparisonContext": {"unit": "rate", "grain": "model-version by metric"}, "dataset": "clean_operating_chart", "sourceId": source_ids["clean_operating_chart"],
            "encodings": {"x": {"field": "model_version", "type": "nominal", "label": "Model / version"}, "y": {"field": "value", "type": "quantitative", "label": "Rate", "format": "percent"}, "color": {"field": "metric", "type": "nominal", "label": "Metric"}, "tooltip": [{"field": "value", "type": "quantitative", "label": "Value", "format": "percent"}]},
            "palette": {"kind": "categorical"}, "legend": {"position": "bottom", "interactive": True}, "labels": {"values": "none"}, "valueFormat": "percent", "layout": "full",
        },
        {
            "id": "robustness_chart", "title": "Clean、扰动均值与最坏 AUROC", "subtitle": "20 个非 Clean 条件等权；最坏值为条件最小 AUROC。", "type": "bar", "intent": "comparison",
            "question": "train-v3 的改善能否保持到整体和最坏扰动？", "rationale": "版本和汇总类型为离散系列。",
            "comparisonContext": {"unit": "AUROC", "grain": "model by protocol and summary"}, "dataset": "robustness_chart", "sourceId": source_ids["robustness_chart"],
            "encodings": {"x": {"field": "model", "type": "nominal", "label": "Model"}, "y": {"field": "auroc", "type": "quantitative", "label": "AUROC", "format": "percent"}, "color": {"field": "series", "type": "nominal", "label": "Protocol / summary"}, "tooltip": [{"field": "auroc", "type": "quantitative", "label": "AUROC", "format": "percent"}]},
            "palette": {"kind": "categorical"}, "legend": {"position": "bottom", "interactive": True}, "labels": {"values": "none"}, "valueFormat": "percent", "layout": "full",
        },
        {
            "id": "multistage_chart", "title": "4-stage / 6-stage 共同扰动 AUROC", "subtitle": "两组四阶段与一组六阶段随机组合。", "type": "bar", "intent": "comparison",
            "question": "多阶段重发与编辑链下 v2/v3 如何比较？", "rationale": "三个离散压力条件适合分组柱状图。",
            "comparisonContext": {"unit": "AUROC", "grain": "model by protocol and chain"}, "dataset": "multistage_chart", "sourceId": source_ids["multistage_chart"],
            "encodings": {"x": {"field": "model", "type": "nominal", "label": "Model"}, "y": {"field": "auroc", "type": "quantitative", "label": "AUROC", "format": "percent"}, "color": {"field": "series", "type": "nominal", "label": "Protocol / chain"}, "tooltip": [{"field": "auroc", "type": "quantitative", "label": "AUROC", "format": "percent"}]},
            "palette": {"kind": "categorical"}, "legend": {"position": "bottom", "interactive": True}, "labels": {"values": "none"}, "valueFormat": "percent", "layout": "full",
        },
        {
            "id": "condition_delta_chart", "title": "21 条件 AUROC 变化（v3 − v2）", "subtitle": "正值表示 train-v3 在相同测试样本和条件上的 AUROC 更高。", "type": "line", "intent": "trend",
            "question": "改进集中在哪些扰动，哪些条件发生退化？", "rationale": "有序条件索引用折线展示每个模型的变化轨迹。",
            "comparisonContext": {"unit": "AUROC delta", "grain": "condition by model", "baseline": "zero delta"}, "dataset": "condition_delta_chart", "sourceId": source_ids["condition_delta_chart"],
            "encodings": {"x": {"field": "condition_index", "type": "quantitative", "label": "Condition index"}, "y": {"field": "delta_auroc", "type": "quantitative", "label": "AUROC delta", "format": "percent"}, "color": {"field": "model", "type": "nominal", "label": "Model"}, "tooltip": [{"field": "condition", "type": "nominal", "label": "Condition"}, {"field": "delta_auroc", "type": "quantitative", "label": "Delta", "format": "percent"}]},
            "palette": {"kind": "categorical"}, "legend": {"position": "bottom", "interactive": True}, "labels": {"values": "none"}, "valueFormat": "percent", "layout": "full",
        },
        {
            "id": "generator_delta_chart", "title": "12 个未见精确生成器的 Clean Recall 变化", "subtitle": "正类切片，仅展示 Recall 的 v3−v2 变化。", "type": "bar", "intent": "comparison",
            "question": "哪些精确生成器从 train-v3 获益或退化？", "rationale": "生成器为离散类别，使用分组柱状图。",
            "comparisonContext": {"unit": "recall delta", "grain": "generator by model"}, "dataset": "generator_delta_chart", "sourceId": source_ids["generator_delta_chart"],
            "encodings": {"x": {"field": "generator", "type": "nominal", "label": "Exact generator"}, "y": {"field": "delta_recall", "type": "quantitative", "label": "Recall delta", "format": "percent"}, "color": {"field": "model", "type": "nominal", "label": "Model"}, "tooltip": [{"field": "delta_recall", "type": "quantitative", "label": "Delta", "format": "percent"}]},
            "palette": {"kind": "categorical"}, "legend": {"position": "bottom", "interactive": True}, "labels": {"values": "none"}, "valueFormat": "percent", "layout": "full",
        },
    ]
    tables = [
        {
            "id": "training_table", "title": "训练集规模与覆盖变化", "subtitle": "train-v3 在 train-v2 上追加 4,000 张并保持类别平衡。", "dataset": "training_comparison", "sourceId": source_ids["training_comparison"], "defaultSort": {"field": "metric_order", "direction": "asc"},
            "columns": _columns(("metric_order", "顺序", "number"), ("metric", "指标", None), ("train_v2", "train-v2", "number"), ("train_v3", "train-v3", "number"), ("delta", "变化", "number")),
        },
        {
            "id": "clean_table", "title": "Clean 完整 v2/v3 指标与差值", "subtitle": "AUROC/AP 阈值无关；其余指标使用各版本冻结阈值。", "dataset": "clean_comparison", "sourceId": source_ids["clean_comparison"], "defaultSort": {"field": "v3_auroc", "direction": "desc"}, "density": "dense",
            "columns": _columns(("model", "模型", None), ("v2_threshold", "v2 阈值", "number"), ("v3_threshold", "v3 阈值", "number"), ("v2_accuracy", "v2 Acc", "percent"), ("v3_accuracy", "v3 Acc", "percent"), ("delta_accuracy", "ΔAcc", "percent"), ("v2_precision", "v2 Precision", "percent"), ("v3_precision", "v3 Precision", "percent"), ("v2_recall", "v2 Recall", "percent"), ("v3_recall", "v3 Recall", "percent"), ("v2_specificity", "v2 Spec", "percent"), ("v3_specificity", "v3 Spec", "percent"), ("v2_f1", "v2 F1", "percent"), ("v3_f1", "v3 F1", "percent"), ("v2_mcc", "v2 MCC", "number"), ("v3_mcc", "v3 MCC", "number"), ("v2_auroc", "v2 AUROC", "percent"), ("v3_auroc", "v3 AUROC", "percent"), ("delta_auroc", "ΔAUROC", "percent"), ("v2_average_precision", "v2 AP", "percent"), ("v3_average_precision", "v3 AP", "percent"), ("v2_tpr_at_fpr_1pct", "v2 TPR@1%", "percent"), ("v3_tpr_at_fpr_1pct", "v3 TPR@1%", "percent"), ("v2_tpr_at_fpr_5pct", "v2 TPR@5%", "percent"), ("v3_tpr_at_fpr_5pct", "v3 TPR@5%", "percent")),
        },
        {
            "id": "delta_ci_table", "title": "Clean v3−v2 配对 bootstrap 95% 区间", "subtitle": "共享样本、Real/AIGI 分层、1,000 次；区间未做多重比较校正。", "dataset": "clean_comparison", "sourceId": source_ids["clean_comparison"], "defaultSort": {"field": "model_order", "direction": "asc"},
            "columns": _columns(("model_order", "顺序", "number"), ("model", "模型", None), ("delta_accuracy_ci_95", "ΔAccuracy 95% CI", None), ("delta_precision_ci_95", "ΔPrecision 95% CI", None), ("delta_recall_ci_95", "ΔRecall 95% CI", None), ("delta_specificity_ci_95", "ΔSpecificity 95% CI", None), ("delta_f1_ci_95", "ΔF1 95% CI", None), ("delta_mcc_ci_95", "ΔMCC 95% CI", None), ("delta_auroc_ci_95", "ΔAUROC 95% CI", None), ("delta_average_precision_ci_95", "ΔAP 95% CI", None)),
        },
        {
            "id": "confusion_table", "title": "Clean 混淆矩阵", "subtitle": "正类为 AIGI；每行每版本合计 2,000。", "dataset": "clean_comparison", "sourceId": source_ids["clean_comparison"], "defaultSort": {"field": "model_order", "direction": "asc"},
            "columns": _columns(("model_order", "顺序", "number"), ("model", "模型", None), ("v2_tn", "v2 TN", "number"), ("v2_fp", "v2 FP", "number"), ("v2_fn", "v2 FN", "number"), ("v2_tp", "v2 TP", "number"), ("v3_tn", "v3 TN", "number"), ("v3_fp", "v3 FP", "number"), ("v3_fn", "v3 FN", "number"), ("v3_tp", "v3 TP", "number")),
        },
        {
            "id": "robust_table", "title": "20 个非 Clean 条件汇总", "subtitle": "均值对条件等权，最坏值为最小条件值。", "dataset": "robustness_comparison", "sourceId": source_ids["robustness_comparison"], "defaultSort": {"field": "v3_mean_20_auroc", "direction": "desc"},
            "columns": _columns(("model", "模型", None), ("v2_mean_20_accuracy", "v2 Mean Acc", "percent"), ("v3_mean_20_accuracy", "v3 Mean Acc", "percent"), ("delta_mean_20_accuracy", "ΔMean Acc", "percent"), ("v2_worst_20_accuracy", "v2 Worst Acc", "percent"), ("v3_worst_20_accuracy", "v3 Worst Acc", "percent"), ("v2_mean_20_auroc", "v2 Mean AUROC", "percent"), ("v3_mean_20_auroc", "v3 Mean AUROC", "percent"), ("delta_mean_20_auroc", "ΔMean AUROC", "percent"), ("v2_worst_20_auroc", "v2 Worst AUROC", "percent"), ("v3_worst_20_auroc", "v3 Worst AUROC", "percent"), ("v2_worst_auroc_condition", "v2 最坏条件", None), ("v3_worst_auroc_condition", "v3 最坏条件", None)),
        },
        {
            "id": "multistage_table", "title": "4-stage / 6-stage 详细指标", "subtitle": "15 个模型-共同扰动单元。", "dataset": "multistage_comparison", "sourceId": source_ids["multistage_comparison"], "defaultSort": {"field": "v3_auroc", "direction": "desc"}, "density": "dense",
            "columns": _columns(("model", "模型", None), ("condition", "条件", None), ("v2_accuracy", "v2 Acc", "percent"), ("v3_accuracy", "v3 Acc", "percent"), ("delta_accuracy", "ΔAcc", "percent"), ("v2_recall", "v2 Recall", "percent"), ("v3_recall", "v3 Recall", "percent"), ("v2_specificity", "v2 Spec", "percent"), ("v3_specificity", "v3 Spec", "percent"), ("v2_auroc", "v2 AUROC", "percent"), ("v3_auroc", "v3 AUROC", "percent"), ("delta_auroc", "ΔAUROC", "percent"), ("v2_average_precision", "v2 AP", "percent"), ("v3_average_precision", "v3 AP", "percent")),
        },
        {
            "id": "condition_table", "title": "完整 105 条模型 × 条件 v2/v3 对比", "subtitle": "默认按 ΔAUROC 升序，优先显示 train-v3 退化条件。", "dataset": "condition_comparison", "sourceId": source_ids["condition_comparison"], "defaultSort": {"field": "delta_auroc", "direction": "asc"}, "density": "dense",
            "columns": _columns(("model", "模型", None), ("condition_index", "序号", "number"), ("condition_group", "分组", None), ("condition", "条件", None), ("v2_accuracy", "v2 Acc", "percent"), ("v3_accuracy", "v3 Acc", "percent"), ("delta_accuracy", "ΔAcc", "percent"), ("v2_recall", "v2 Recall", "percent"), ("v3_recall", "v3 Recall", "percent"), ("v2_specificity", "v2 Spec", "percent"), ("v3_specificity", "v3 Spec", "percent"), ("v2_auroc", "v2 AUROC", "percent"), ("v3_auroc", "v3 AUROC", "percent"), ("delta_auroc", "ΔAUROC", "percent"), ("v2_average_precision", "v2 AP", "percent"), ("v3_average_precision", "v3 AP", "percent"), ("v2_tpr_at_fpr_1pct", "v2 TPR@1%", "percent"), ("v3_tpr_at_fpr_1pct", "v3 TPR@1%", "percent")),
        },
        {
            "id": "generator_table", "title": "12 个未见精确生成器 Clean Recall", "subtitle": "正类切片只支持 Recall/TP/FN。", "dataset": "generator_comparison", "sourceId": source_ids["generator_comparison"], "defaultSort": {"field": "delta_recall", "direction": "asc"}, "density": "dense",
            "columns": _columns(("model", "模型", None), ("generator", "精确生成器", None), ("n", "N", "number"), ("v2_tp", "v2 TP", "number"), ("v2_fn", "v2 FN", "number"), ("v2_recall", "v2 Recall", "percent"), ("v3_tp", "v3 TP", "number"), ("v3_fn", "v3 FN", "number"), ("v3_recall", "v3 Recall", "percent"), ("delta_recall", "ΔRecall", "percent")),
        },
        {
            "id": "real_table", "title": "四个真实来源 Clean Specificity", "subtitle": "负类切片只支持 Specificity/TN/FP/FPR。", "dataset": "real_source_comparison", "sourceId": source_ids["real_source_comparison"], "defaultSort": {"field": "delta_specificity", "direction": "asc"},
            "columns": _columns(("model", "模型", None), ("real_source", "真实来源", None), ("n", "N", "number"), ("v2_tn", "v2 TN", "number"), ("v2_fp", "v2 FP", "number"), ("v2_specificity", "v2 Spec", "percent"), ("v3_tn", "v3 TN", "number"), ("v3_fp", "v3 FP", "number"), ("v3_specificity", "v3 Spec", "percent"), ("delta_specificity", "ΔSpec", "percent")),
        },
        {
            "id": "lineage_table", "title": "Checkpoint、阈值与评测谱系", "subtitle": "每个版本/模型独立冻结阈值；测试集与扰动矩阵完全共享。", "dataset": "lineage", "sourceId": source_ids["lineage"], "defaultSort": {"field": "protocol_order", "direction": "asc"}, "density": "dense",
            "columns": _columns(("protocol_order", "版本顺序", "number"), ("protocol", "训练版本", None), ("model", "模型", None), ("architecture", "架构", None), ("threshold", "冻结阈值", "number"), ("checkpoint_sha256", "Checkpoint SHA256", None), ("train_manifest_sha256", "Train manifest SHA256", None), ("evaluation_job", "评测 Job", None), ("prediction_rows", "预测数", "number")),
        },
    ]
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {"id": "summary", "type": "markdown", "sourceId": source_ids["headline"], "body": (
            "## 技术摘要\n\n"
            f"- **v3 Clean AUROC 最优：{headline['best_v3_clean_auroc_model']}，{_pct(headline['best_v3_clean_auroc'])}。**\n"
            f"- **v3 Clean Accuracy 最优：{headline['best_v3_clean_accuracy_model']}，{_pct(headline['best_v3_clean_accuracy'])}。**\n"
            f"- **v3 20 扰动平均 AUROC 最优：{headline['best_v3_robust_auroc_model']}，{_pct(headline['best_v3_robust_auroc'])}。**\n"
            f"- Clean AUROC 提升模型数：{headline['positive_clean_auroc_deltas']}/5；扰动平均 AUROC 提升模型数：{headline['positive_robust_auroc_deltas']}/5。\n\n"
            "主比较仅使用 v2/v3 unseen 评价集的 2,000 张 sample-id 交集和同一 21 条件矩阵；各版本仅使用内部验证集冻结阈值。"
        )},
        {"id": "cards", "type": "metric-strip", "cardIds": ["best_v3_auc", "best_v3_acc", "best_v3_robust", "positive_deltas"]},
        {"id": "contract", "type": "markdown", "body": "## 评测合同\n\n正类为 AIGI，负类为 Real。比较面板是 v2/v3 unseen 评价集的 2,000 张 sample-id 交集且类别平衡，含 12 个训练未见精确生成器；21 条件包括 Clean、17 个既有单/双阶段扰动、两组 4-stage 和一组随机 6-stage。两个训练版本共核验 420,000 条交集逐样本预测。External test 标签未用于 checkpoint、阈值或模型选择。"},
        {"id": "training", "type": "table", "tableId": "training_table", "layout": "full"},
        {"id": "clean_text", "type": "markdown", "body": "## Clean 综合指标\n\nAUROC/AP 描述阈值无关排序能力；Accuracy、Precision、Recall、Specificity、F1 与 MCC 描述各版本自身冻结阈值的操作点。Accuracy、Precision 与 NPV受 50% AIGI 测试先验影响。"},
        {"id": "clean_rank", "type": "chart", "chartId": "clean_ranking_chart", "layout": "full"},
        {"id": "clean_operating", "type": "chart", "chartId": "clean_operating_chart", "layout": "full"},
        {"id": "clean_table_block", "type": "table", "tableId": "clean_table", "layout": "full"},
        {"id": "delta_ci", "type": "table", "tableId": "delta_ci_table", "layout": "full"},
        {"id": "confusion", "type": "table", "tableId": "confusion_table", "layout": "full"},
        {"id": "robust_text", "type": "markdown", "body": "## 扰动鲁棒性\n\n均值用于概括 20 个非 Clean 条件，最坏值用于暴露单条件失效。二者均是选定矩阵上的描述性统计，不是未来变换分布的概率保证。"},
        {"id": "robust_chart", "type": "chart", "chartId": "robustness_chart", "layout": "full"},
        {"id": "robust_table_block", "type": "table", "tableId": "robust_table", "layout": "full"},
        {"id": "multistage_chart_block", "type": "chart", "chartId": "multistage_chart", "layout": "full"},
        {"id": "multistage_table_block", "type": "table", "tableId": "multistage_table", "layout": "full"},
        {"id": "condition_text", "type": "markdown", "body": "## 21 条件变化轨迹\n\n正差值表示 train-v3 更高；默认表格按 ΔAUROC 升序，优先定位退化条件。小差值在单随机种子下不构成统计优越性结论。"},
        {"id": "condition_delta", "type": "chart", "chartId": "condition_delta_chart", "layout": "full"},
        {"id": "condition_table_block", "type": "table", "tableId": "condition_table", "layout": "full"},
        {"id": "generator_text", "type": "markdown", "body": "## 精确生成器与真实来源切片\n\n单生成器切片只有 AIGI 正类，只支持 Recall/TP/FN；单真实来源切片只有 Real 负类，只支持 Specificity/TN/FP。来源差异可能混合内容、格式、分辨率和处理链效应。"},
        {"id": "generator_chart_block", "type": "chart", "chartId": "generator_delta_chart", "layout": "full"},
        {"id": "generator_table_block", "type": "table", "tableId": "generator_table", "layout": "full"},
        {"id": "real_table_block", "type": "table", "tableId": "real_table", "layout": "full"},
        {"id": "lineage", "type": "table", "tableId": "lineage_table", "layout": "full"},
        {"id": "limits", "type": "markdown", "body": "## 局限性与建议\n\n- 每个训练版本只有一个随机种子；配对 bootstrap 只覆盖当前测试样本抽样，不覆盖训练随机性或 checkpoint 选择。\n- train-v3 同时改变样本量、生成器覆盖和真实来源，结果不能唯一归因于单一新增类别。\n- 部署前应在独立 calibration set 上按 FPR 约束设阈值，并至少运行三个训练种子及生成器分层的配对检验。\n- 使用贴近部署 AIGI 流行率的回放集重新估计 Precision、NPV 和成本加权指标。"},
    ]
    return {
        "surface": "report",
        "manifest": {"version": 1, "surface": "report", "title": title, "description": "Fair train-v2/train-v3 comparison on the sample-id intersection of their strict unseen-generator evaluation sets under the same 21-condition perturbation matrix.", "generatedAt": generated_at, "blocks": blocks, "cards": cards, "charts": charts, "tables": tables, "sources": [{"id": source["id"], "label": source["label"]} for source in sources]},
        "snapshot": {"version": 1, "status": "ready", "generatedAt": generated_at, "datasets": snapshot},
        "sources": sources,
    }


def generate(arguments: argparse.Namespace) -> None:
    generated_at = datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds")
    manifest_rows = _read_csv(arguments.manifest)
    train_v2 = _profile_training(arguments.train_v2_manifest)
    train_v3 = _profile_training(arguments.train_v3_manifest)
    test_generators = {row["canonical_generator_id"] for row in manifest_rows if int(row["label"]) == 1}
    if test_generators & set(train_v2["generators"]) or test_generators & set(train_v3["generators"]):
        raise RuntimeError("Exact-generator exposure violation in strict unseen intersection")
    overlap = {
        "train_v2": _identity_overlap(train_v2["rows"], manifest_rows),
        "train_v3": _identity_overlap(train_v3["rows"], manifest_rows),
    }
    if any(value for protocol in overlap.values() for value in protocol.values()):
        raise RuntimeError(f"Training/test identity overlap detected: {overlap}")

    v2 = _load_protocol(
        "train-v2", arguments.v2_evaluation_root, arguments.v2_source_root, arguments.v2_split_directory,
        arguments.manifest, arguments.v2_evaluation_manifest, arguments.matrix,
        arguments.bootstrap_replicates, arguments.bootstrap_seed,
    )
    v3 = _load_protocol(
        "train-v3", arguments.v3_evaluation_root, arguments.v3_source_root, arguments.v3_split_directory,
        arguments.manifest, arguments.v3_evaluation_manifest, arguments.matrix,
        arguments.bootstrap_replicates, arguments.bootstrap_seed,
    )
    datasets = _wide_comparisons(v2, v3, arguments.bootstrap_replicates, arguments.bootstrap_seed + 1000)
    training_comparison = [
        {"metric_order": 0, "metric": "Total images", "train_v2": train_v2["count"], "train_v3": train_v3["count"], "delta": train_v3["count"] - train_v2["count"]},
        {"metric_order": 1, "metric": "Real images", "train_v2": train_v2["real"], "train_v3": train_v3["real"], "delta": train_v3["real"] - train_v2["real"]},
        {"metric_order": 2, "metric": "AIGI images", "train_v2": train_v2["aigi"], "train_v3": train_v3["aigi"], "delta": train_v3["aigi"] - train_v2["aigi"]},
        {"metric_order": 3, "metric": "Exact AIGI generators", "train_v2": len(train_v2["generators"]), "train_v3": len(train_v3["generators"]), "delta": len(train_v3["generators"]) - len(train_v2["generators"])},
        {"metric_order": 4, "metric": "GAN AIGI", "train_v2": train_v2["architectures"].get("GAN", 0), "train_v3": train_v3["architectures"].get("GAN", 0), "delta": train_v3["architectures"].get("GAN", 0) - train_v2["architectures"].get("GAN", 0)},
        {"metric_order": 5, "metric": "Pixel-diffusion AIGI", "train_v2": train_v2["architectures"].get("PixDiff", 0), "train_v3": train_v3["architectures"].get("PixDiff", 0), "delta": train_v3["architectures"].get("PixDiff", 0) - train_v2["architectures"].get("PixDiff", 0)},
    ]
    clean = datasets["clean_comparison"]
    robust = datasets["robustness_comparison"]
    best_auc = max(clean, key=lambda row: row["v3_auroc"])
    best_accuracy = max(clean, key=lambda row: row["v3_accuracy"])
    best_robust = max(robust, key=lambda row: row["v3_mean_20_auroc"])
    headline = [{
        "best_v3_clean_auroc_model": best_auc["model"], "best_v3_clean_auroc": best_auc["v3_auroc"],
        "best_v3_clean_accuracy_model": best_accuracy["model"], "best_v3_clean_accuracy": best_accuracy["v3_accuracy"],
        "best_v3_robust_auroc_model": best_robust["model"], "best_v3_robust_auroc": best_robust["v3_mean_20_auroc"],
        "positive_clean_auroc_deltas": sum(row["delta_auroc"] > 0 for row in clean),
        "positive_robust_auroc_deltas": sum(row["delta_mean_20_auroc"] > 0 for row in robust),
        "test_images": len(manifest_rows), "conditions": 21, "models": 5, "prediction_rows": 420000,
    }]
    matrix = yaml.safe_load(Path(arguments.matrix).read_text(encoding="utf-8"))["evaluation"]
    condition_matrix = [
        {"condition_index": index, "condition_group": unseen._condition_group(index), "condition": CONDITION_LABELS[index], "transform_name": specification["name"], "parameters": json.dumps(specification.get("params", {}), ensure_ascii=False, sort_keys=True)}
        for index, specification in enumerate(matrix)
    ]
    staged = {
        "headline": headline,
        "training_comparison": training_comparison,
        **datasets,
        "lineage": v2["lineage"] + v3["lineage"],
        "condition_matrix": condition_matrix,
    }
    queries = {dataset: f"SELECT * FROM {dataset}" for dataset in staged}
    snapshot = unseen._materialize_sql_snapshot(staged, queries)

    audit = {
        "schema_version": 1,
        "generated_at_asia_singapore": generated_at,
        "report_job_id": os.environ.get("SLURM_JOB_ID"),
        "protocol_id": "community_forensics_train_v2_v3_strict_unseen_intersection_21_conditions",
        "intersection_manifest": str(arguments.manifest), "test_manifest_sha256": _sha256(arguments.manifest),
        "v2_evaluation_manifest": str(arguments.v2_evaluation_manifest), "v2_evaluation_manifest_sha256": _sha256(arguments.v2_evaluation_manifest),
        "v3_evaluation_manifest": str(arguments.v3_evaluation_manifest), "v3_evaluation_manifest_sha256": _sha256(arguments.v3_evaluation_manifest),
        "matrix": str(arguments.matrix), "matrix_sha256": _sha256(arguments.matrix),
        "train_v2_manifest": str(arguments.train_v2_manifest), "train_v2_manifest_sha256": train_v2["sha256"],
        "train_v3_manifest": str(arguments.train_v3_manifest), "train_v3_manifest_sha256": train_v3["sha256"],
        "identity_overlap": overlap,
        "exact_generator_overlap": {"train_v2": sorted(test_generators & set(train_v2["generators"])), "train_v3": sorted(test_generators & set(train_v3["generators"]))},
        "threshold_policy": "Per-model and per-training-version threshold frozen from internal clean validation; no external test retuning",
        "bootstrap": {"method": "paired Real/AIGI-stratified nonparametric bootstrap on shared clean samples", "replicates": arguments.bootstrap_replicates, "seed": arguments.bootstrap_seed},
        "input_audit": {"train_v2": v2["input_audit"], "train_v3": v3["input_audit"]},
        "dataset_rows": {key: len(rows) for key, rows in snapshot.items()},
    }
    _write_csv(arguments.clean_csv, snapshot["clean_comparison"])
    _write_csv(arguments.conditions_csv, snapshot["condition_comparison"])
    _write_csv(arguments.multistage_csv, snapshot["multistage_comparison"])
    _write_csv(arguments.generator_csv, snapshot["generator_comparison"])
    _write_csv(arguments.real_source_csv, snapshot["real_source_comparison"])
    _write_csv(arguments.lineage_csv, snapshot["lineage"])
    _atomic_text(arguments.markdown, _build_markdown(snapshot, audit) + "\n")
    _atomic_text(arguments.artifact_json, json.dumps(_build_artifact(snapshot, queries, generated_at), ensure_ascii=False, indent=2) + "\n")
    _atomic_text(arguments.audit_json, json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "train_v2_v3_unseen_intersection_report_complete", "artifact": str(arguments.artifact_json), "markdown": str(arguments.markdown), "prediction_rows": 420000}, sort_keys=True), flush=True)


def _parse_args() -> argparse.Namespace:
    root = "reports/evaluations/train_v2_v3_unseen_intersection_comparison"
    parser = argparse.ArgumentParser(description="Compare train-v2/train-v3 on their strict unseen-generator sample-id intersection")
    parser.add_argument("--v2-evaluation-root", default="outputs/community_forensics_v2_robustness_v2")
    parser.add_argument("--v3-evaluation-root", default="outputs/community_forensics_v3_robustness_v2")
    parser.add_argument("--v2-source-root", default="outputs/community_forensics_v2")
    parser.add_argument("--v3-source-root", default="outputs/community_forensics_v3")
    parser.add_argument("--v2-split-directory", default="unseen_generator")
    parser.add_argument("--v3-split-directory", default="unseen_generator_expanded")
    parser.add_argument("--manifest", default="data/manifests/community_forensics_test_external_unseen_generator.csv")
    parser.add_argument("--v2-evaluation-manifest", default="data/manifests/community_forensics_test_external_unseen_generator.csv")
    parser.add_argument("--v3-evaluation-manifest", default="data/manifests/community_forensics_test_external_unseen_generator_v3_expanded.csv")
    parser.add_argument("--matrix", default="configs/community_forensics_robustness_v2.yaml")
    parser.add_argument("--train-v2-manifest", default="data/manifests/community_forensics_train_v2.csv")
    parser.add_argument("--train-v3-manifest", default="data/manifests/community_forensics_train_v3.csv")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260830)
    parser.add_argument("--markdown", default="reports/summaries/COMMUNITY_FORENSICS_TRAIN_V2_V3_UNSEEN_INTERSECTION_COMPARISON.md")
    parser.add_argument("--artifact-json", default=f"{root}/community_forensics_train_v2_v3_unseen_intersection_artifact.json")
    parser.add_argument("--audit-json", default=f"{root}/community_forensics_train_v2_v3_unseen_intersection_audit.json")
    parser.add_argument("--clean-csv", default=f"{root}/community_forensics_train_v2_v3_unseen_intersection_clean.csv")
    parser.add_argument("--conditions-csv", default=f"{root}/community_forensics_train_v2_v3_unseen_intersection_all_conditions.csv")
    parser.add_argument("--multistage-csv", default=f"{root}/community_forensics_train_v2_v3_unseen_intersection_multistage.csv")
    parser.add_argument("--generator-csv", default=f"{root}/community_forensics_train_v2_v3_unseen_intersection_generator.csv")
    parser.add_argument("--real-source-csv", default=f"{root}/community_forensics_train_v2_v3_unseen_intersection_real_source.csv")
    parser.add_argument("--lineage-csv", default=f"{root}/community_forensics_train_v2_v3_unseen_intersection_lineage.csv")
    return parser.parse_args()


if __name__ == "__main__":
    generate(_parse_args())
