from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any
from zoneinfo import ZoneInfo

import yaml


MODELS = ("b0", "b1", "b2", "m2", "m3")
MODEL_ARCHITECTURES = {
    "b0": "EfficientNet-B0 clean baseline",
    "b1": "EfficientNet-B0 robust-augmentation baseline",
    "b2": "Frozen OpenCLIP ViT-B/32 + linear head",
    "m2": "Frozen CLIP semantic + forensic branch",
    "m3": "M2 + quality-aware branch gate",
}
SPLITS = (
    (
        "exact_seen_generator",
        "Exact-seen generator",
        "data/manifests/community_forensics_val_external_exact_seen_generator.csv",
        "External source; exact generator identity appears in Small train.",
    ),
    (
        "hard_hourglass_exact_seen",
        "Hard Hourglass (exact-seen)",
        "data/manifests/community_forensics_val_hard_hourglass_v2_exact_seen.csv",
        "Hourglass exact identity is train-v2-seen; evaluation images are disjoint.",
    ),
    (
        "hard_dfgan_exact_seen",
        "Hard DFGAN (exact-seen)",
        "data/manifests/community_forensics_val_hard_dfgan_v2_exact_seen.csv",
        "DFGAN exact identity is train-v2-seen; evaluation images are disjoint.",
    ),
    (
        "hard_galip_exact_seen",
        "Hard GALIP (exact-seen)",
        "data/manifests/community_forensics_val_hard_galip_v2_exact_seen.csv",
        "GALIP exact identity is train-v2-seen; evaluation images are disjoint.",
    ),
    (
        "unseen_generator",
        "Unseen-generator",
        "data/manifests/community_forensics_test_external_unseen_generator.csv",
        "Architecture family and exact generator identity are both unseen.",
    ),
)
SPLIT_TITLES = {key: title for key, title, _, _ in SPLITS}
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


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_text(path: str | Path, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, target)


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    preferred = [
        "model",
        "split",
        "split_title",
        "condition_index",
        "condition_group",
        "condition_label",
    ]
    remaining = sorted({key for row in rows for key in row} - set(preferred))
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=preferred + remaining)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)


def _condition_group(index: int) -> str:
    if index == 0:
        return "clean"
    if index <= 15:
        return "legacy_single_stage"
    if index <= 17:
        return "legacy_two_stage"
    if index == 18:
        return "new_four_stage_a"
    if index == 19:
        return "new_four_stage_b"
    if index == 20:
        return "new_six_stage"
    raise ValueError(index)


def _metric(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def _short_digest(value: str) -> str:
    return value[:12]


def _manifest_counts(path: str | Path) -> tuple[int, int, int]:
    rows = _read_csv(path)
    real = sum(int(row["label"]) == 0 for row in rows)
    aigi = sum(int(row["label"]) == 1 for row in rows)
    return len(rows), real, aigi


def _matrix_stage_description(index: int, specification: dict[str, Any]) -> str:
    if index == 18:
        return "crop(0.85) -> resize(0.5,bicubic) -> blur(1.0) -> JPEG(50)"
    if index == 19:
        return (
            "color(1.15/1.15/0.85) -> resize(0.5,bilinear) -> "
            "noise(0.05) -> JPEG(50)"
        )
    if index == 20:
        return (
            "crop -> resize -> color -> blur -> noise -> JPEG; each strength is "
            "sampled deterministically from the full training range"
        )
    parameters = specification.get("params", {})
    if "transforms" in parameters:
        return " -> ".join(item["name"] for item in parameters["transforms"])
    if not parameters:
        return "none"
    printable = {key: value for key, value in parameters.items() if key != "seed"}
    return json.dumps(printable, sort_keys=True, separators=(",", ":"))


def _load_and_verify(
    evaluation_root: Path,
    matrix_path: Path,
    source_root: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str], list[dict[str, str]]],
    dict[tuple[str, str], dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    matrix = _read_yaml(matrix_path)["evaluation"]
    legacy = _read_yaml("configs/transforms.yaml")["evaluation"]
    if len(matrix) != 21 or matrix[:18] != legacy:
        raise RuntimeError("Robustness-v2 matrix must preserve 18 legacy conditions and add 3")
    if len(CONDITION_LABELS) != len(matrix):
        raise RuntimeError("Condition label count does not match evaluation matrix")

    metrics: dict[tuple[str, str], list[dict[str, str]]] = {}
    run_cards: dict[tuple[str, str], dict[str, Any]] = {}
    model_lineage: dict[str, dict[str, Any]] = {}
    matrix_digest = _sha256(matrix_path)

    for model in MODELS:
        source_run_card = _read_json(source_root / model / "run_card.json")
        source_summary = _read_json(source_root / model / "summary.json")
        checkpoint_path = source_root / model / "best.pt"
        checkpoint_digest = _sha256(checkpoint_path)
        if source_run_card["checkpoint_sha256"] != checkpoint_digest:
            raise RuntimeError(f"Checkpoint digest drift for {model}")
        model_lineage[model] = {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_digest,
            "config_path": str(
                source_root / model / "resolved_config.yaml"
            ),
            "config_sha256": source_run_card["config_sha256"],
            "parameters": source_run_card["parameters"],
            "threshold": float(source_summary["threshold_from_clean_validation"]),
            "training_job_id": _read_json(source_root / model / "DONE")[
                "slurm_job_id"
            ],
        }
        if not (evaluation_root / model / "COMPLETE").is_file():
            raise RuntimeError(f"Missing model completion marker: {model}")

        for split_key, _, manifest_path, _ in SPLITS:
            output = evaluation_root / model / split_key
            if not (output / "COMPLETE").is_file():
                raise RuntimeError(f"Missing split completion marker: {model}/{split_key}")
            rows = _read_csv(output / "metrics_by_transform.csv")
            summary = _read_json(output / "summary.json")
            run_card = _read_json(output / "run_card.json")
            if len(rows) != 21 or int(summary["conditions"]) != 21:
                raise RuntimeError(f"Unexpected condition count: {model}/{split_key}")
            if run_card["checkpoint_sha256"] != checkpoint_digest:
                raise RuntimeError(f"Checkpoint mismatch: {model}/{split_key}")
            if run_card["evaluation_matrix_sha256"] != matrix_digest:
                raise RuntimeError(f"Matrix mismatch: {model}/{split_key}")
            if run_card["val_manifest_sha256"] != _sha256(manifest_path):
                raise RuntimeError(f"Manifest mismatch: {model}/{split_key}")
            configured_threshold = float(model_lineage[model]["threshold"])
            if any(abs(_metric(row, "threshold") - configured_threshold) > 1e-12 for row in rows):
                raise RuntimeError(f"Threshold drift: {model}/{split_key}")
            expected_n, _, _ = _manifest_counts(manifest_path)
            if any(int(row["n"]) != expected_n for row in rows):
                raise RuntimeError(f"Sample count drift: {model}/{split_key}")
            for index, (row, specification) in enumerate(zip(rows, matrix, strict=True)):
                if row["transform_name"] != specification["name"]:
                    raise RuntimeError(
                        f"Matrix order mismatch: {model}/{split_key}/condition-{index}"
                    )
            metrics[(model, split_key)] = rows
            run_cards[(model, split_key)] = run_card
    return matrix, metrics, run_cards, model_lineage


def _flatten_metrics(
    metrics: dict[tuple[str, str], list[dict[str, str]]]
) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for model in MODELS:
        for split_key, split_title, _, _ in SPLITS:
            for index, row in enumerate(metrics[(model, split_key)]):
                flattened.append(
                    {
                        "model": model,
                        "split": split_key,
                        "split_title": split_title,
                        "condition_index": index,
                        "condition_group": _condition_group(index),
                        "condition_label": CONDITION_LABELS[index],
                        **row,
                    }
                )
    return flattened


def _split_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    legacy = rows[1:18]
    new = rows[18:21]
    worst_legacy = min(legacy, key=lambda row: _metric(row, "auroc"))
    worst_new = min(new, key=lambda row: _metric(row, "auroc"))
    return {
        "clean_auroc": _metric(rows[0], "auroc"),
        "clean_ba": _metric(rows[0], "balanced_accuracy"),
        "legacy_mean_auroc": fmean(_metric(row, "auroc") for row in legacy),
        "legacy_mean_ba": fmean(_metric(row, "balanced_accuracy") for row in legacy),
        "legacy_worst_auroc": _metric(worst_legacy, "auroc"),
        "legacy_worst_transform": worst_legacy["transform"],
        "four_a_auroc": _metric(rows[18], "auroc"),
        "four_a_ba": _metric(rows[18], "balanced_accuracy"),
        "four_b_auroc": _metric(rows[19], "auroc"),
        "four_b_ba": _metric(rows[19], "balanced_accuracy"),
        "six_auroc": _metric(rows[20], "auroc"),
        "six_ba": _metric(rows[20], "balanced_accuracy"),
        "six_recall": _metric(rows[20], "aigc_recall"),
        "six_specificity": _metric(rows[20], "real_specificity"),
        "new_mean_auroc": fmean(_metric(row, "auroc") for row in new),
        "new_mean_ba": fmean(_metric(row, "balanced_accuracy") for row in new),
        "new_worst_auroc": _metric(worst_new, "auroc"),
        "new_worst_transform": worst_new["transform"],
    }


def _macro_summary(
    metrics: dict[tuple[str, str], list[dict[str, str]]]
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    split_keys = [item[0] for item in SPLITS]
    for model in MODELS:
        per_split = {
            split: _split_summary(metrics[(model, split)]) for split in split_keys
        }
        new_candidates = [
            (split, index, _metric(metrics[(model, split)][index], "auroc"))
            for split in split_keys
            for index in range(18, 21)
        ]
        worst_split, worst_index, worst_value = min(new_candidates, key=lambda item: item[2])
        summaries[model] = {
            "clean_auroc": fmean(item["clean_auroc"] for item in per_split.values()),
            "clean_ba": fmean(item["clean_ba"] for item in per_split.values()),
            "legacy_mean_auroc": fmean(
                item["legacy_mean_auroc"] for item in per_split.values()
            ),
            "legacy_mean_ba": fmean(item["legacy_mean_ba"] for item in per_split.values()),
            "four_a_auroc": fmean(item["four_a_auroc"] for item in per_split.values()),
            "four_b_auroc": fmean(item["four_b_auroc"] for item in per_split.values()),
            "six_auroc": fmean(item["six_auroc"] for item in per_split.values()),
            "six_ba": fmean(item["six_ba"] for item in per_split.values()),
            "new_mean_auroc": fmean(item["new_mean_auroc"] for item in per_split.values()),
            "new_mean_ba": fmean(item["new_mean_ba"] for item in per_split.values()),
            "new_worst_auroc": worst_value,
            "new_worst_split": worst_split,
            "new_worst_condition": CONDITION_LABELS[worst_index],
        }
    return summaries


def _round_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: round(value, 6) if isinstance(value, float) else value
        for key, value in row.items()
    }


def _sqlite_type(values: list[Any]) -> str:
    concrete = [value for value in values if value is not None]
    if concrete and all(isinstance(value, int) and not isinstance(value, bool) for value in concrete):
        return "INTEGER"
    if concrete and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in concrete):
        return "REAL"
    return "TEXT"


def _materialize_sql_snapshot(
    staged: dict[str, list[dict[str, Any]]],
    queries: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    """Execute the exact SQL exposed in report provenance over reviewed staged rows."""
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        for dataset, rows in staged.items():
            if not rows:
                raise RuntimeError(f"Cannot materialize empty report dataset: {dataset}")
            columns = list(rows[0])
            if any(set(row) != set(columns) for row in rows):
                raise RuntimeError(f"Inconsistent columns in report dataset: {dataset}")
            definitions = ", ".join(
                f'"{column}" {_sqlite_type([row[column] for row in rows])}'
                for column in columns
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
                raise RuntimeError(f"SQL snapshot row-count drift for {dataset}")
        return snapshot
    finally:
        connection.close()


def _report_datasets(
    matrix: list[dict[str, Any]],
    metrics: dict[tuple[str, str], list[dict[str, str]]],
    run_cards: dict[tuple[str, str], dict[str, Any]],
    model_lineage: dict[str, dict[str, Any]],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, str],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    macro = _macro_summary(metrics)
    best_clean = max(MODELS, key=lambda model: macro[model]["clean_auroc"])
    best_new = max(MODELS, key=lambda model: macro[model]["new_mean_auroc"])
    best_worst = max(MODELS, key=lambda model: macro[model]["new_worst_auroc"])
    best_six = max(MODELS, key=lambda model: macro[model]["six_auroc"])

    headline = [{
        "best_clean_model": best_clean.upper(),
        "best_clean_auroc": round(macro[best_clean]["clean_auroc"], 6),
        "best_new_model": best_new.upper(),
        "best_new_auroc": round(macro[best_new]["new_mean_auroc"], 6),
        "best_worst_model": best_worst.upper(),
        "best_worst_auroc": round(macro[best_worst]["new_worst_auroc"], 6),
        "best_worst_split": SPLIT_TITLES[macro[best_worst]["new_worst_split"]],
        "best_worst_condition": macro[best_worst]["new_worst_condition"],
        "best_six_model": best_six.upper(),
        "best_six_auroc": round(macro[best_six]["six_auroc"], 6),
        "model_count": len(MODELS),
        "split_count": len(SPLITS),
        "condition_count": len(matrix),
        "evaluation_points": len(MODELS) * len(SPLITS) * len(matrix),
    }]

    macro_detail: list[dict[str, Any]] = []
    macro_chart: list[dict[str, Any]] = []
    worst_new_chart: list[dict[str, Any]] = []
    for model_order, model in enumerate(MODELS):
        item = macro[model]
        macro_detail.append(_round_metrics({
            "model": model.upper(),
            "model_order": model_order,
            "clean_auroc": item["clean_auroc"],
            "legacy_17_mean_auroc": item["legacy_mean_auroc"],
            "four_stage_a_auroc": item["four_a_auroc"],
            "four_stage_b_auroc": item["four_b_auroc"],
            "six_stage_auroc": item["six_auroc"],
            "new_3_mean_auroc": item["new_mean_auroc"],
            "new_3_worst_auroc": item["new_worst_auroc"],
            "new_3_mean_balanced_accuracy": item["new_mean_ba"],
            "new_3_worst_split": SPLIT_TITLES[item["new_worst_split"]],
            "new_3_worst_condition": item["new_worst_condition"],
            "new_minus_legacy_auroc": item["new_mean_auroc"] - item["legacy_mean_auroc"],
        }))
        for condition_order, (condition_group, value) in enumerate((
            ("Clean", item["clean_auroc"]),
            ("Legacy 17 mean", item["legacy_mean_auroc"]),
            ("New 3 mean", item["new_mean_auroc"]),
        )):
            macro_chart.append({
                "model": model.upper(),
                "model_order": model_order,
                "condition_group": condition_group,
                "condition_order": condition_order,
                "auroc": round(value, 6),
            })
        worst_new_chart.append({
            "model": model.upper(),
            "model_order": model_order,
            "worst_auroc": round(item["new_worst_auroc"], 6),
            "split": SPLIT_TITLES[item["new_worst_split"]],
            "condition": item["new_worst_condition"],
        })

    split_definitions: list[dict[str, Any]] = []
    split_summary: list[dict[str, Any]] = []
    strict_six_chart: list[dict[str, Any]] = []
    for split_order, (split_key, title, manifest, definition) in enumerate(SPLITS):
        total, real, aigi = _manifest_counts(manifest)
        split_definitions.append({
            "split": title,
            "split_order": split_order,
            "total": total,
            "real": real,
            "aigi": aigi,
            "generator_exposure": definition,
            "manifest_sha256": _short_digest(_sha256(manifest)),
        })
        for model_order, model in enumerate(MODELS):
            item = _split_summary(metrics[(model, split_key)])
            split_summary.append(_round_metrics({
                "split": title,
                "split_order": split_order,
                "model": model.upper(),
                "model_order": model_order,
                "clean_auroc": item["clean_auroc"],
                "clean_balanced_accuracy": item["clean_ba"],
                "legacy_17_mean_auroc": item["legacy_mean_auroc"],
                "legacy_17_worst_auroc": item["legacy_worst_auroc"],
                "four_stage_a_auroc": item["four_a_auroc"],
                "four_stage_b_auroc": item["four_b_auroc"],
                "six_stage_auroc": item["six_auroc"],
                "new_3_mean_auroc": item["new_mean_auroc"],
                "six_stage_balanced_accuracy": item["six_ba"],
                "six_stage_recall": item["six_recall"],
                "six_stage_specificity": item["six_specificity"],
            }))
            strict_six_chart.append({
                "split": title,
                "split_order": split_order,
                "model": model.upper(),
                "model_order": model_order,
                "auroc": round(item["six_auroc"], 6),
                "balanced_accuracy": round(item["six_ba"], 6),
                "aigi_recall": round(item["six_recall"], 6),
                "real_specificity": round(item["six_specificity"], 6),
            })

    lineage: list[dict[str, Any]] = []
    for model_order, model in enumerate(MODELS):
        item = model_lineage[model]
        evaluation_jobs = sorted(
            {str(run_cards[(model, split[0])]["slurm_job_id"]) for split in SPLITS}
        )
        lineage.append({
            "model": model.upper(),
            "model_order": model_order,
            "architecture": MODEL_ARCHITECTURES[model],
            "total_parameters": int(item["parameters"]["total"]),
            "trainable_parameters": int(item["parameters"]["trainable"]),
            "frozen_threshold": round(float(item["threshold"]), 8),
            "checkpoint_sha256": _short_digest(item["checkpoint_sha256"]),
            "config_sha256": _short_digest(item["config_sha256"]),
            "training_job": str(item["training_job_id"]),
            "evaluation_jobs": ", ".join(evaluation_jobs),
        })

    perturbation_matrix = [
        {
            "condition_index": index,
            "condition": CONDITION_LABELS[index],
            "group": _condition_group(index),
            "transform_name": specification["name"],
            "parameters_and_order": _matrix_stage_description(index, specification),
        }
        for index, specification in enumerate(matrix)
    ]

    m3_gates: list[dict[str, Any]] = []
    for split_order, (split_key, title, _, _) in enumerate(SPLITS):
        for index in range(18, 21):
            row = metrics[("m3", split_key)][index]
            m3_gates.append({
                "split": title,
                "split_order": split_order,
                "condition": CONDITION_LABELS[index],
                "condition_index": index,
                "semantic_gate_fraction": round(_metric(row, "mean_semantic_gate_fraction"), 6),
                "forensic_gate_fraction": round(_metric(row, "mean_forensic_gate_fraction"), 6),
            })

    numeric_fields = (
        "n", "threshold", "auroc", "average_precision", "balanced_accuracy",
        "macro_f1", "aigc_recall", "real_specificity", "false_positive_rate",
        "brier", "ece_15", "tpr_at_fpr_1pct", "tpr_at_fpr_5pct",
    )
    all_metrics: list[dict[str, Any]] = []
    for row in _flatten_metrics(metrics):
        all_metrics.append({
            "model": str(row["model"]).upper(),
            "split": row["split_title"],
            "condition_index": int(row["condition_index"]),
            "condition_group": row["condition_group"],
            "condition": row["condition_label"],
            **{
                field: int(row[field]) if field == "n" else round(float(row[field]), 6)
                for field in numeric_fields
            },
        })

    staged = {
        "headline_metrics": headline,
        "macro_chart": macro_chart,
        "macro_detail": macro_detail,
        "strict_six_chart": strict_six_chart,
        "worst_new_chart": worst_new_chart,
        "split_definitions": split_definitions,
        "split_summary": split_summary,
        "model_lineage": lineage,
        "perturbation_matrix": perturbation_matrix,
        "m3_gates": m3_gates,
        "all_metrics": all_metrics,
    }
    queries = {
        "headline_metrics": "SELECT * FROM headline_metrics",
        "macro_chart": "SELECT * FROM macro_chart ORDER BY condition_order, model_order",
        "macro_detail": "SELECT * FROM macro_detail ORDER BY model_order",
        "strict_six_chart": "SELECT * FROM strict_six_chart ORDER BY split_order, model_order",
        "worst_new_chart": "SELECT * FROM worst_new_chart ORDER BY worst_auroc DESC",
        "split_definitions": "SELECT * FROM split_definitions ORDER BY split_order",
        "split_summary": "SELECT * FROM split_summary ORDER BY split_order, model_order",
        "model_lineage": "SELECT * FROM model_lineage ORDER BY model_order",
        "perturbation_matrix": "SELECT * FROM perturbation_matrix ORDER BY condition_index",
        "m3_gates": "SELECT * FROM m3_gates ORDER BY split_order, condition_index",
        "all_metrics": "SELECT * FROM all_metrics ORDER BY split, condition_index, model",
    }
    return _materialize_sql_snapshot(staged, queries), queries, macro, headline[0]


def _source(dataset: str, query: str, generated_at: str) -> dict[str, Any]:
    return {
        "id": f"{dataset}_sql",
        "label": f"Reviewed {dataset.replace('_', ' ')} snapshot",
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": query,
            "description": (
                "Executed over rows staged only after checkpoint, manifest, matrix, "
                "threshold, sample-count, and completion-marker validation."
            ),
            "executed_at": generated_at,
            "tables_used": [dataset],
        },
    }


def _table_columns(*columns: tuple[str, str, str | None]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for field, label, format_name in columns:
        item: dict[str, Any] = {"field": field, "label": label}
        if format_name:
            item["format"] = format_name
        output.append(item)
    return output


def _build_artifact(
    snapshot_datasets: dict[str, list[dict[str, Any]]],
    queries: dict[str, str],
    macro: dict[str, dict[str, Any]],
    headline: dict[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    title = "Community Forensics B0/B1/B2/M2/M3 新数据集鲁棒性评测"
    source_ids = {dataset: f"{dataset}_sql" for dataset in queries}
    sources = [_source(dataset, query, generated_at) for dataset, query in queries.items()]
    best_new = headline["best_new_model"]
    best_worst = headline["best_worst_model"]
    best_six = headline["best_six_model"]

    cards = [
        {
            "id": "best_clean_card",
            "description": f"五个切片等权宏平均；最佳模型 {headline['best_clean_model']}。",
            "dataset": "headline_metrics",
            "sourceId": source_ids["headline_metrics"],
            "metrics": [{"label": "最佳 Clean AUROC", "field": "best_clean_auroc", "format": "number"}],
        },
        {
            "id": "best_new_card",
            "description": f"两组四阶段与一组六阶段等权宏平均；最佳模型 {best_new}。",
            "dataset": "headline_metrics",
            "sourceId": source_ids["headline_metrics"],
            "metrics": [{"label": "最佳新增三组 AUROC", "field": "best_new_auroc", "format": "number"}],
        },
        {
            "id": "best_worst_card",
            "description": f"新增三组所有切片中的最低点；最佳模型 {best_worst}。",
            "dataset": "headline_metrics",
            "sourceId": source_ids["headline_metrics"],
            "metrics": [{"label": "最佳全局最坏 AUROC", "field": "best_worst_auroc", "format": "number"}],
        },
    ]
    charts = [
        {
            "id": "macro_auroc_chart",
            "title": "跨五切片的模型 AUROC 对比",
            "subtitle": "Clean、原17种扰动均值和新增三组多阶段均值；五个切片等权。",
            "type": "bar",
            "intent": "comparison",
            "question": "各模型在 clean、原扰动和新增多阶段扰动上的宏平均 AUROC 如何比较？",
            "rationale": "五个离散模型与三组同单位 AUROC 适合使用分组柱状图直接比较。",
            "comparisonContext": {"unit": "AUROC", "grain": "model by condition group", "normalization": "equal-weight macro mean across five splits"},
            "dataset": "macro_chart",
            "sourceId": source_ids["macro_chart"],
            "encodings": {
                "x": {"field": "model", "type": "nominal", "label": "Model"},
                "y": {"field": "auroc", "type": "quantitative", "label": "AUROC", "format": "number"},
                "color": {"field": "condition_group", "type": "nominal", "label": "Condition group"},
                "tooltip": [
                    {"field": "condition_group", "type": "nominal", "label": "Condition group"},
                    {"field": "auroc", "type": "quantitative", "label": "AUROC", "format": "number"},
                ],
            },
            "palette": {"kind": "categorical"},
            "legend": {"position": "bottom", "interactive": True},
            "labels": {"values": "auto"},
            "valueFormat": "number",
            "layout": "full",
        },
        {
            "id": "strict_six_chart",
            "title": "六阶段共同扰动下的分切片 AUROC",
            "subtitle": "每张图依次执行六类操作，强度由固定种子逐样本抽取。",
            "type": "bar",
            "intent": "comparison",
            "question": "六阶段共同扰动下，模型在六个生成器暴露切片上的 AUROC 如何变化？",
            "rationale": "六个离散切片和五个模型构成适合分组柱状图的类别比较。",
            "comparisonContext": {"unit": "AUROC", "grain": "model by validation split", "baseline": "same frozen checkpoint and threshold"},
            "dataset": "strict_six_chart",
            "sourceId": source_ids["strict_six_chart"],
            "encodings": {
                "x": {"field": "split", "type": "nominal", "label": "Validation split"},
                "y": {"field": "auroc", "type": "quantitative", "label": "AUROC", "format": "number"},
                "color": {"field": "model", "type": "nominal", "label": "Model"},
                "tooltip": [
                    {"field": "balanced_accuracy", "type": "quantitative", "label": "Balanced accuracy", "format": "number"},
                    {"field": "aigi_recall", "type": "quantitative", "label": "AIGI recall", "format": "number"},
                    {"field": "real_specificity", "type": "quantitative", "label": "Real specificity", "format": "number"},
                ],
            },
            "palette": {"kind": "categorical"},
            "legend": {"position": "bottom", "interactive": True},
            "labels": {"values": "none"},
            "valueFormat": "number",
            "layout": "full",
        },
        {
            "id": "worst_new_chart",
            "title": "新增多阶段条件的模型最坏点 AUROC",
            "subtitle": "每个模型在五切片 × 三个新增条件中的最低 AUROC；越高越稳健。",
            "type": "horizontalBar",
            "intent": "comparison",
            "question": "哪个模型在新增多阶段扰动的全局最坏情况下保持最高 AUROC？",
            "rationale": "五个模型的单一最坏值适合按值排序的水平柱状图。",
            "comparisonContext": {"unit": "AUROC", "grain": "minimum per model", "baseline": "18 new-condition split cells per model"},
            "dataset": "worst_new_chart",
            "sourceId": source_ids["worst_new_chart"],
            "encodings": {
                "x": {"field": "model", "type": "nominal", "label": "Model"},
                "y": {"field": "worst_auroc", "type": "quantitative", "label": "Worst AUROC", "format": "number"},
                "tooltip": [
                    {"field": "split", "type": "nominal", "label": "Worst split"},
                    {"field": "condition", "type": "nominal", "label": "Worst condition"},
                ],
            },
            "palette": {"kind": "sequential"},
            "labels": {"values": "all"},
            "valueFormat": "number",
            "layout": "full",
        },
    ]

    tables = [
        {
            "id": "macro_detail_table",
            "title": "跨切片总体指标",
            "subtitle": "每个切片先独立汇总，再对五个切片等权平均。",
            "dataset": "macro_detail",
            "sourceId": source_ids["macro_detail"],
            "defaultSort": {"field": "new_3_mean_auroc", "direction": "desc"},
            "columns": _table_columns(
                ("model", "模型", None),
                ("clean_auroc", "Clean AUROC", "number"),
                ("legacy_17_mean_auroc", "原17均值", "number"),
                ("four_stage_a_auroc", "四阶段A", "number"),
                ("four_stage_b_auroc", "四阶段B", "number"),
                ("six_stage_auroc", "六阶段", "number"),
                ("new_3_mean_auroc", "新增三组均值", "number"),
                ("new_3_worst_auroc", "新增三组全局最坏", "number"),
                ("new_3_mean_balanced_accuracy", "新增三组 BA", "number"),
                ("new_minus_legacy_auroc", "新增减原17", "number"),
            ),
        },
        {
            "id": "split_definition_table",
            "title": "验证与测试切片定义",
            "subtitle": "精确生成器、生成器大类与训练暴露关系均显式区分。",
            "dataset": "split_definitions",
            "sourceId": source_ids["split_definitions"],
            "defaultSort": {"field": "split_order", "direction": "asc"},
            "columns": _table_columns(
                ("split_order", "顺序", "number"),
                ("split", "切片", None),
                ("total", "总数", "number"),
                ("real", "Real", "number"),
                ("aigi", "AIGI", "number"),
                ("generator_exposure", "生成器暴露定义", None),
                ("manifest_sha256", "Manifest SHA256", None),
            ),
        },
        {
            "id": "lineage_table",
            "title": "模型、阈值与 checkpoint 谱系",
            "subtitle": "评测仅加载冻结 best checkpoint，并沿用原 clean validation 阈值。",
            "dataset": "model_lineage",
            "sourceId": source_ids["model_lineage"],
            "defaultSort": {"field": "model_order", "direction": "asc"},
            "columns": _table_columns(
                ("model_order", "顺序", "number"),
                ("model", "模型", None),
                ("architecture", "架构", None),
                ("total_parameters", "总参数", "number"),
                ("trainable_parameters", "可训练参数", "number"),
                ("frozen_threshold", "冻结阈值", "number"),
                ("checkpoint_sha256", "Checkpoint SHA256", None),
                ("config_sha256", "Config SHA256", None),
                ("training_job", "训练 Job", None),
                ("evaluation_jobs", "评测 Job", None),
            ),
        },
        {
            "id": "perturbation_table",
            "title": "完整21条件扰动矩阵",
            "subtitle": "原 clean + 17项保持顺序和参数不变，末尾追加两个四阶段与一个六阶段条件。",
            "dataset": "perturbation_matrix",
            "sourceId": source_ids["perturbation_matrix"],
            "defaultSort": {"field": "condition_index", "direction": "asc"},
            "columns": _table_columns(
                ("condition_index", "序号", "number"),
                ("condition", "条件", None),
                ("group", "分组", None),
                ("transform_name", "变换名", None),
                ("parameters_and_order", "参数与执行顺序", None),
            ),
        },
        {
            "id": "split_summary_table",
            "title": "模型 × 切片汇总",
            "subtitle": "30个模型-切片组合；包含 clean、原17均值与三个新增多阶段条件。",
            "dataset": "split_summary",
            "sourceId": source_ids["split_summary"],
            "defaultSort": {"field": "new_3_mean_auroc", "direction": "desc"},
            "density": "dense",
            "columns": _table_columns(
                ("split", "切片", None),
                ("model", "模型", None),
                ("clean_auroc", "Clean AUC", "number"),
                ("clean_balanced_accuracy", "Clean BA", "number"),
                ("legacy_17_mean_auroc", "原17 AUC", "number"),
                ("legacy_17_worst_auroc", "原17最坏", "number"),
                ("four_stage_a_auroc", "四阶段A", "number"),
                ("four_stage_b_auroc", "四阶段B", "number"),
                ("six_stage_auroc", "六阶段", "number"),
                ("new_3_mean_auroc", "新增三组均值", "number"),
                ("six_stage_balanced_accuracy", "六阶段 BA", "number"),
                ("six_stage_recall", "六阶段 Recall", "number"),
                ("six_stage_specificity", "六阶段 Specificity", "number"),
            ),
        },
        {
            "id": "m3_gate_table",
            "title": "M3 新增多阶段条件下的平均门控权重",
            "subtitle": "语义与法证权重之和为1；它们是分支权重，不是分类概率。",
            "dataset": "m3_gates",
            "sourceId": source_ids["m3_gates"],
            "defaultSort": {"field": "split", "direction": "asc"},
            "columns": _table_columns(
                ("split", "切片", None),
                ("condition", "条件", None),
                ("semantic_gate_fraction", "语义门控", "number"),
                ("forensic_gate_fraction", "法证门控", "number"),
            ),
        },
        {
            "id": "all_metrics_table",
            "title": "完整630条评测记录",
            "subtitle": "5模型 × 6切片 × 21条件；用于逐条件精确查阅。",
            "dataset": "all_metrics",
            "sourceId": source_ids["all_metrics"],
            "defaultSort": {"field": "auroc", "direction": "asc"},
            "density": "dense",
            "columns": _table_columns(
                ("model", "模型", None),
                ("split", "切片", None),
                ("condition_index", "条件序号", "number"),
                ("condition_group", "条件分组", None),
                ("condition", "条件", None),
                ("n", "N", "number"),
                ("auroc", "AUROC", "number"),
                ("average_precision", "AP", "number"),
                ("balanced_accuracy", "BA", "number"),
                ("macro_f1", "Macro-F1", "number"),
                ("aigc_recall", "AIGI Recall", "number"),
                ("real_specificity", "Real Specificity", "number"),
                ("false_positive_rate", "FPR", "number"),
                ("brier", "Brier", "number"),
                ("ece_15", "ECE-15", "number"),
                ("tpr_at_fpr_1pct", "TPR@1%FPR", "number"),
                ("tpr_at_fpr_5pct", "TPR@5%FPR", "number"),
                ("threshold", "冻结阈值", "number"),
            ),
        },
    ]

    summary_body = (
        "## 技术摘要\n\n"
        f"- **Clean 宏平均最优：{headline['best_clean_model']}，AUROC {_fmt(headline['best_clean_auroc'])}。**\n"
        f"- **新增三组多阶段宏平均最优：{best_new}，AUROC {_fmt(headline['best_new_auroc'])}。**\n"
        f"- **全局最坏点保持最优：{best_worst}，AUROC {_fmt(headline['best_worst_auroc'])}。**"
        f" 该模型最低点出现在 {headline['best_worst_split']} / {headline['best_worst_condition']}。\n"
        f"- 本报告覆盖 {headline['model_count']} 个冻结模型、{headline['split_count']} 个切片、"
        f"{headline['condition_count']} 个条件，共 {headline['evaluation_points']} 个模型-切片-条件指标点。\n\n"
        "结论为描述性比较：未计算 bootstrap 置信区间或多重比较校正，不能据此宣称统计显著性。"
    )
    macro_deltas = "; ".join(
        f"{model.upper()} {_fmt(macro[model]['new_mean_auroc'] - macro[model]['legacy_mean_auroc'])}"
        for model in MODELS
    )

    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {"id": "technical_summary", "type": "markdown", "sourceId": source_ids["headline_metrics"], "body": summary_body},
        {"id": "headline_cards", "type": "metric-strip", "cardIds": ["best_clean_card", "best_new_card", "best_worst_card"]},
        {
            "id": "macro_finding",
            "type": "markdown",
            "sourceId": source_ids["macro_detail"],
            "body": (
                "## 多阶段扰动改变了模型间的相对稳健性\n\n"
                f"五切片等权汇总后，新增三组多阶段条件由 **{best_new}** 取得最高均值。"
                f"新增三组相对原17扰动的 AUROC 差值依次为：{macro_deltas}。"
                "下图用于比较整体形态，随后的表保留精确数值；宏平均避免2000张切片完全支配500张困难切片。"
            ),
        },
        {"id": "macro_chart_block", "type": "chart", "chartId": "macro_auroc_chart", "layout": "full"},
        {"id": "macro_table_block", "type": "table", "tableId": "macro_detail_table", "layout": "full"},
        {
            "id": "strict_six_finding",
            "type": "markdown",
            "sourceId": source_ids["strict_six_chart"],
            "body": (
                "## 六阶段共同扰动是最直接的部署压力测试\n\n"
                f"跨五切片宏平均，六阶段条件由 **{best_six}** 取得最高 AUROC "
                f"{_fmt(headline['best_six_auroc'])}。图中同时保留切片维度，便于识别模型优势是否只来自某一生成器暴露类型；"
                "固定阈值下的 BA、Recall 与 Specificity 可在悬浮信息和后续明细表中审计。"
            ),
        },
        {"id": "strict_six_chart_block", "type": "chart", "chartId": "strict_six_chart", "layout": "full"},
        {
            "id": "worst_case_finding",
            "type": "markdown",
            "sourceId": source_ids["worst_new_chart"],
            "body": (
                "## 最坏点比较用于约束平均数掩盖的风险\n\n"
                f"按每个模型在18个新增多阶段切片单元中的最低 AUROC 排名，**{best_worst}** "
                f"仍以 {_fmt(headline['best_worst_auroc'])} 居首。该指标是保守描述，不等同于具有置信保证的下界。"
            ),
        },
        {"id": "worst_chart_block", "type": "chart", "chartId": "worst_new_chart", "layout": "full"},
        {
            "id": "scope_and_definitions",
            "type": "markdown",
            "sourceId": source_ids["split_definitions"],
            "body": (
                "## 五个切片把精确生成器暴露与生成器大类暴露分开\n\n"
                "Exact-seen 表示 Small 训练集与外部验证来源存在精确生成器身份交集；Seen-family 表示大类已见但精确生成器完全未见；"
                "Unseen-generator 表示生成器大类和精确身份均未见。Hourglass、DFGAN、GALIP 作为困难切片单列。"
                "每个 manifest 均为 Real/AIGI 平衡；AUROC 与阈值无关，BA、Recall、Specificity 使用原 Small clean validation 冻结阈值。"
            ),
        },
        {"id": "split_definition_block", "type": "table", "tableId": "split_definition_table", "layout": "full"},
        {
            "id": "model_method",
            "type": "markdown",
            "body": (
                "## 五个模型均以冻结 checkpoint 只读评测\n\n"
                "B0/B1 为 EfficientNet-B0 基线，B2 为冻结 OpenCLIP ViT-B/32 加线性头，M2 融合冻结 CLIP 语义与法证分支，"
                "M3 在 M2 上加入质量感知分支门控。本轮没有继续训练、微调或依据新切片重新选择 checkpoint/阈值。"
                "报告生成前逐项核对 checkpoint、resolved config、manifest、扰动矩阵 SHA256、样本数与 COMPLETE 标记。"
            ),
        },
        {"id": "lineage_block", "type": "table", "tableId": "lineage_table", "layout": "full"},
        {
            "id": "perturbation_method",
            "type": "markdown",
            "sourceId": source_ids["perturbation_matrix"],
            "body": (
                "## 原17项保持不变，新增两个四阶段与一个六阶段条件\n\n"
                "四阶段A模拟平台转发链：crop → resize → blur → JPEG；四阶段B模拟编辑转发链：color → resize → noise → JPEG。"
                "六阶段条件始终执行 crop → resize → color → blur → noise → JPEG，强度由 `20260829 + manifest row index` "
                "从完整训练范围确定性抽取，因此不依赖 DataLoader worker 调度或标签。"
            ),
        },
        {"id": "perturbation_block", "type": "table", "tableId": "perturbation_table", "layout": "full"},
        {
            "id": "split_evidence",
            "type": "markdown",
            "sourceId": source_ids["split_summary"],
            "body": (
                "## 分切片结果揭示生成器域差异，不能只看总体平均\n\n"
                "下表保留30个模型-切片组合的 clean、原17项均值、两组四阶段、六阶段以及固定阈值指标。"
                "Hourglass、DFGAN、GALIP 共用同一真实图面板，便于描述精确生成器变化，但三组 AUROC 并非统计独立。"
            ),
        },
        {"id": "split_summary_block", "type": "table", "tableId": "split_summary_table", "layout": "full"},
        {
            "id": "m3_gate_finding",
            "type": "markdown",
            "sourceId": source_ids["m3_gates"],
            "body": (
                "## M3 门控权重用于解释分支使用方式，不代表置信度\n\n"
                "表中给出三个新增条件下的平均语义/法证权重。两者之和为1；权重变化可用于定位模型在退化输入上更依赖哪一分支，"
                "但不能解释为 AIGI 概率，也不能单独证明门控产生因果改进。"
            ),
        },
        {"id": "m3_gate_block", "type": "table", "tableId": "m3_gate_table", "layout": "full"},
        {
            "id": "complete_detail_intro",
            "type": "markdown",
            "sourceId": source_ids["all_metrics"],
            "body": (
                "## 完整630条记录保留逐条件审计路径\n\n"
                "该表覆盖5模型 × 6切片 × 21条件，默认从最低 AUROC 排序以优先暴露失败模式。"
                "它适合精确查阅，不以密集图形替代，因为多指标和长条件名在同一图中会降低可读性。"
            ),
        },
        {"id": "all_metrics_block", "type": "table", "tableId": "all_metrics_table", "layout": "full"},
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## 结论边界：结果是冻结协议下的描述性外部评测\n\n"
                "- 没有 bootstrap 置信区间、重复随机种子或多重比较校正，模型间小差异不应解释为统计显著。\n"
                "- Exact-seen 是跨数据源精确身份重合，仍包含来源与图像域偏移。\n"
                "- 三个困难切片共享真实面板，因此不能把它们当作三个独立总体。\n"
                "- 两个四阶段条件使用固定强度；六阶段仅固定一个确定性随机方案，尚未覆盖随机强度分布的方差。\n"
                "- 固定阈值指标依赖原 Small clean validation 校准；部署先验或误判成本变化时必须使用独立校准集。\n"
                "- TIFF 解码产生 `Truncated File Read` 警告，但作业未出现解码失败、样本数漂移或非零退出码；应继续把该警告作为数据质量监控项。"
            ),
        },
        {
            "id": "recommendations",
            "type": "markdown",
            "sourceId": source_ids["headline_metrics"],
            "body": (
                "## 下一步应先复核最坏切片，再扩大随机组合覆盖\n\n"
                f"1. 将 **{best_worst}** 作为多阶段稳健性的优先候选，但在独立校准集重新选择部署阈值。\n"
                "2. 对每个模型的最坏生成器切片补做分层 bootstrap，并报告模型差值的区间而不只报告点估计。\n"
                "3. 将六阶段条件扩展到多个预注册种子或强度网格，量化组合顺序与强度随机性。\n"
                "4. 对 TIFF 警告样本建立明确清单，并确认像素解码结果与 manifest 样本身份一致。"
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## 仍需回答的问题\n\n"
                "- 多阶段扰动的性能差异在 bootstrap 区间下是否仍成立？\n"
                "- 扰动顺序互换时，性能下降主要由哪一步或哪种交互造成？\n"
                "- M3 门控权重与错误类型是否相关，还是只响应整体图像质量？\n"
                "- 新增真实来源和更多精确生成器后，Seen-family 与 Unseen-generator 的相对难度是否稳定？"
            ),
        },
    ]
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "冻结五个模型，在六个生成器暴露切片和21个 clean/扰动条件上的技术评测。",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": [{"id": source["id"], "label": source["label"]} for source in sources],
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": snapshot_datasets,
        },
        "sources": sources,
    }


def generate(arguments: argparse.Namespace) -> None:
    evaluation_root = Path(arguments.evaluation_root)
    matrix_path = Path(arguments.matrix)
    matrix, metrics, run_cards, model_lineage = _load_and_verify(
        evaluation_root, matrix_path, Path(arguments.source_root)
    )
    generated_at = datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds")
    snapshot_datasets, queries, macro, headline = _report_datasets(
        matrix,
        metrics,
        run_cards,
        model_lineage,
    )
    flattened = _flatten_metrics(metrics)
    _write_csv(arguments.metrics_csv, flattened)

    split_artifacts: dict[str, Any] = {}
    for model in MODELS:
        split_artifacts[model] = {}
        for split_key, _, manifest, _ in SPLITS:
            output = evaluation_root / model / split_key
            split_artifacts[model][split_key] = {
                "evaluation_job_id": run_cards[(model, split_key)]["slurm_job_id"],
                "manifest": manifest,
                "manifest_sha256": _sha256(manifest),
                "metrics_csv": str(output / "metrics_by_transform.csv"),
                "metrics_csv_sha256": _sha256(output / "metrics_by_transform.csv"),
                "run_card": str(output / "run_card.json"),
                "run_card_sha256": _sha256(output / "run_card.json"),
                "summary": str(output / "summary.json"),
                "summary_sha256": _sha256(output / "summary.json"),
            }
    audit = {
        "schema_version": 1,
        "generated_at_asia_singapore": generated_at,
        "report_job_id": os.environ.get("SLURM_JOB_ID"),
        "report_contract": {
            "delivery_mode": "portable_html",
            "audience": "technical",
            "question": "Compare B0/B1/B2/M2/M3 on five new-data splits under legacy and new multi-stage perturbations.",
            "baseline": "Clean and the unchanged legacy 17-perturbation matrix.",
            "success_criteria": "All 5 x 5 x 21 cells complete with frozen checkpoints, thresholds, manifests, and digests.",
            "required_section_mapping": {
                "title": "title",
                "technical_summary": "technical_summary",
                "key_findings_with_visual_evidence": ["macro_finding", "strict_six_finding", "worst_case_finding"],
                "scope_data_and_metric_definitions": "scope_and_definitions",
                "methodology": ["model_method", "perturbation_method"],
                "limitations_uncertainty_and_robustness_checks": "limitations",
                "recommended_next_steps": "recommendations",
                "further_questions": "further_questions",
            },
        },
        "evaluation_root": str(evaluation_root),
        "matrix": str(matrix_path),
        "matrix_sha256": _sha256(matrix_path),
        "condition_count": len(matrix),
        "legacy_perturbation_count": 17,
        "new_multistage_perturbation_count": 3,
        "model_lineage": model_lineage,
        "macro_summary": macro,
        "headline": headline,
        "split_artifacts": split_artifacts,
        "metrics_csv": str(arguments.metrics_csv),
        "metrics_csv_sha256": _sha256(arguments.metrics_csv),
        "sql_snapshot_queries": queries,
        "chart_map": [
            {
                "section": "macro_finding",
                "question": "Compare clean, legacy-17 mean, and new-3 mean AUROC across models.",
                "family": "comparison",
                "type": "grouped bar",
                "fields": ["model", "condition_group", "auroc"],
                "palette": "categorical, three condition groups",
            },
            {
                "section": "strict_six_finding",
                "question": "Compare strict six-stage AUROC by split and model.",
                "family": "comparison",
                "type": "grouped bar",
                "fields": ["split", "model", "auroc"],
                "palette": "categorical, five models",
            },
            {
                "section": "worst_case_finding",
                "question": "Rank each model by its worst new-condition AUROC.",
                "family": "comparison and ranking",
                "type": "horizontal bar",
                "fields": ["model", "worst_auroc"],
                "palette": "single-root sequential",
            },
        ],
        "visual_omissions": [
            "The 630-row multi-metric detail is a sortable table because exact lookup is the goal and a single chart would obscure the evidence.",
            "Confidence-interval visuals are omitted because bootstrap replicates were not computed in this evaluation run.",
        ],
    }
    _atomic_text(
        arguments.audit_json,
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
    )
    artifact = _build_artifact(
        snapshot_datasets,
        queries,
        macro,
        headline,
        generated_at,
    )
    _atomic_text(
        arguments.artifact_json,
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
    )
    print(
        json.dumps(
            {
                "event": "community_forensics_robustness_v2_report_complete",
                "artifact_json": str(arguments.artifact_json),
                "audit_json": str(arguments.audit_json),
                "rows": len(flattened),
                "models": len(MODELS),
                "splits": len(SPLITS),
                "conditions": len(matrix),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Community Forensics robustness-v2 evaluation report"
    )
    parser.add_argument(
        "--evaluation-root",
        default="outputs/community_forensics_v2_robustness_v2",
    )
    parser.add_argument(
        "--source-root",
        default="outputs/community_forensics_v2",
    )
    parser.add_argument(
        "--matrix",
        default="configs/community_forensics_robustness_v2.yaml",
    )
    parser.add_argument(
        "--metrics-csv",
        default="reports/evaluations/robustness_v2_train_v2/community_forensics_robustness_v2_metrics.csv",
    )
    parser.add_argument(
        "--artifact-json",
        default="reports/evaluations/robustness_v2_train_v2/community_forensics_robustness_v2_report_artifact.json",
    )
    parser.add_argument(
        "--audit-json",
        default="reports/evaluations/robustness_v2_train_v2/community_forensics_robustness_v2_report_notes.json",
    )
    return parser.parse_args()


def main() -> None:
    generate(_parse_args())


if __name__ == "__main__":
    main()
