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
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

import generate_community_forensics_unseen_accuracy_report as unseen


MODELS = unseen.MODELS
MODEL_ORDER = {model: index for index, model in enumerate(MODELS)}
SPLITS = (
    ("exact_seen_generator", "Exact-seen generator", "exact_seen_generator", "exact_seen_generator"),
    ("hard_hourglass", "Hard Hourglass", "hard_hourglass", "hard_hourglass_exact_seen"),
    ("hard_dfgan", "Hard DFGAN", "hard_dfgan", "hard_dfgan_exact_seen"),
    ("hard_galip", "Hard GALIP", "hard_galip", "hard_galip_exact_seen"),
    ("unseen_generator", "Strict unseen-generator", "unseen_generator", "unseen_generator"),
)
MULTISTAGE = (
    (18, "Four-stage platform repost"),
    (19, "Four-stage edit repost"),
    (20, "Random six-stage"),
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


def _atomic_text(path: str | Path, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, target)


def _training_profile(path: str | Path) -> dict[str, Any]:
    rows = _read_csv(path)
    real = [row for row in rows if int(row["label"]) == 0]
    aigi = [row for row in rows if int(row["label"]) == 1]
    architectures: dict[str, int] = {}
    formats: dict[str, int] = {}
    sources: dict[str, int] = {}
    for row in aigi:
        architectures[row["architecture"]] = architectures.get(row["architecture"], 0) + 1
    for row in rows:
        formats[row["format"]] = formats.get(row["format"], 0) + 1
        sources[row["source_dataset"]] = sources.get(row["source_dataset"], 0) + 1
    return {
        "rows": len(rows),
        "real": len(real),
        "aigi": len(aigi),
        "exact_generators": len({row["canonical_generator_id"] for row in aigi}),
        "architectures": architectures,
        "formats": formats,
        "sources": sources,
        "sha256": _sha256(path),
    }


def _load_protocol_metrics(
    evaluation_root: Path,
    split_directory_index: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    loaded: dict[tuple[str, str], dict[str, Any]] = {}
    for model in MODELS:
        if not (evaluation_root / model / "COMPLETE").is_file():
            raise RuntimeError(f"Missing model COMPLETE marker: {evaluation_root / model}")
        for split_key, _, v1_directory, v2_directory in SPLITS:
            directory = (v1_directory, v2_directory)[split_directory_index]
            output = evaluation_root / model / directory
            for required in ("COMPLETE", "summary.json", "metrics_by_transform.csv"):
                if not (output / required).is_file():
                    raise RuntimeError(f"Missing {output / required}")
            summary = _read_json(output / "summary.json")
            metrics = _read_csv(output / "metrics_by_transform.csv")
            if int(summary["conditions"]) != 21 or len(metrics) != 21:
                raise RuntimeError(f"Expected 21 conditions: {output}")
            loaded[(model, split_key)] = {"summary": summary, "metrics": metrics}
    return loaded


def _comparison_datasets(arguments: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    v1_training = _training_profile(arguments.train_v1_manifest)
    v2_training = _training_profile(arguments.train_v2_manifest)
    v1 = _load_protocol_metrics(Path(arguments.v1_evaluation_root), 0)
    v2 = _load_protocol_metrics(Path(arguments.v2_evaluation_root), 1)

    for key in v1:
        v1_transforms = [row["transform"] for row in v1[key]["metrics"]]
        v2_transforms = [row["transform"] for row in v2[key]["metrics"]]
        if v1_transforms != v2_transforms:
            raise RuntimeError(f"Perturbation order drift for {key}")

    training_summary = [
        {
            "metric": "Total images",
            "train_v1": v1_training["rows"],
            "train_v2": v2_training["rows"],
            "delta": v2_training["rows"] - v1_training["rows"],
            "interpretation": "External seen-family promoted into training",
        },
        {
            "metric": "Real images",
            "train_v1": v1_training["real"],
            "train_v2": v2_training["real"],
            "delta": v2_training["real"] - v1_training["real"],
            "interpretation": "Class balance retained",
        },
        {
            "metric": "AIGI images",
            "train_v1": v1_training["aigi"],
            "train_v2": v2_training["aigi"],
            "delta": v2_training["aigi"] - v1_training["aigi"],
            "interpretation": "Class balance retained",
        },
        {
            "metric": "Exact AIGI generators",
            "train_v1": v1_training["exact_generators"],
            "train_v2": v2_training["exact_generators"],
            "delta": v2_training["exact_generators"] - v1_training["exact_generators"],
            "interpretation": "Nine exact generators promoted",
        },
    ]
    training_class_chart = [
        {"protocol": protocol, "class": class_name, "images": profile[field], "class_order": class_order}
        for protocol, profile in (("train-v1", v1_training), ("train-v2", v2_training))
        for class_order, (class_name, field) in enumerate((("Real", "real"), ("AIGI", "aigi")))
    ]
    training_architecture_chart = [
        {
            "protocol": protocol,
            "architecture": architecture,
            "images": profile["architectures"].get(architecture, 0),
            "architecture_order": architecture_order,
        }
        for protocol, profile in (("train-v1", v1_training), ("train-v2", v2_training))
        for architecture_order, architecture in enumerate(("GAN", "LatDiff", "PixDiff"))
    ]

    model_macro: list[dict[str, Any]] = []
    macro_chart: list[dict[str, Any]] = []
    split_comparison: list[dict[str, Any]] = []
    split_delta_chart: list[dict[str, Any]] = []
    strict_multistage: list[dict[str, Any]] = []
    strict_multistage_chart: list[dict[str, Any]] = []
    for model in MODELS:
        clean_v1 = fmean(float(v1[(model, split)]["summary"]["clean_auroc"]) for split, *_ in SPLITS)
        clean_v2 = fmean(float(v2[(model, split)]["summary"]["clean_auroc"]) for split, *_ in SPLITS)
        robust_v1 = fmean(float(v1[(model, split)]["summary"]["robust_mean_auroc"]) for split, *_ in SPLITS)
        robust_v2 = fmean(float(v2[(model, split)]["summary"]["robust_mean_auroc"]) for split, *_ in SPLITS)
        ba_v1 = fmean(float(v1[(model, split)]["summary"]["clean_balanced_accuracy"]) for split, *_ in SPLITS)
        ba_v2 = fmean(float(v2[(model, split)]["summary"]["clean_balanced_accuracy"]) for split, *_ in SPLITS)
        model_macro.append({
            "model": model.upper(),
            "model_order": MODEL_ORDER[model],
            "v1_clean_macro_auroc": round(clean_v1, 8),
            "v2_clean_macro_auroc": round(clean_v2, 8),
            "delta_clean_macro_auroc": round(clean_v2 - clean_v1, 8),
            "v1_perturbation_macro_auroc": round(robust_v1, 8),
            "v2_perturbation_macro_auroc": round(robust_v2, 8),
            "delta_perturbation_macro_auroc": round(robust_v2 - robust_v1, 8),
            "v1_clean_macro_balanced_accuracy": round(ba_v1, 8),
            "v2_clean_macro_balanced_accuracy": round(ba_v2, 8),
            "delta_clean_macro_balanced_accuracy": round(ba_v2 - ba_v1, 8),
        })
        for protocol, clean, robust in (
            ("train-v1", clean_v1, robust_v1),
            ("train-v2", clean_v2, robust_v2),
        ):
            macro_chart.extend((
                {
                    "model": model.upper(),
                    "model_order": MODEL_ORDER[model],
                    "series": f"{protocol} clean",
                    "series_order": 0 if protocol == "train-v1" else 1,
                    "auroc": round(clean, 8),
                },
                {
                    "model": model.upper(),
                    "model_order": MODEL_ORDER[model],
                    "series": f"{protocol} 20-perturbation mean",
                    "series_order": 2 if protocol == "train-v1" else 3,
                    "auroc": round(robust, 8),
                },
            ))

        for split_order, (split_key, split_title, _, _) in enumerate(SPLITS):
            s1 = v1[(model, split_key)]["summary"]
            s2 = v2[(model, split_key)]["summary"]
            row = {
                "model": model.upper(),
                "model_order": MODEL_ORDER[model],
                "split": split_title,
                "split_order": split_order,
                "v1_clean_auroc": round(float(s1["clean_auroc"]), 8),
                "v2_clean_auroc": round(float(s2["clean_auroc"]), 8),
                "delta_clean_auroc": round(float(s2["clean_auroc"]) - float(s1["clean_auroc"]), 8),
                "v1_perturbation_mean_auroc": round(float(s1["robust_mean_auroc"]), 8),
                "v2_perturbation_mean_auroc": round(float(s2["robust_mean_auroc"]), 8),
                "delta_perturbation_mean_auroc": round(float(s2["robust_mean_auroc"]) - float(s1["robust_mean_auroc"]), 8),
                "v1_clean_balanced_accuracy": round(float(s1["clean_balanced_accuracy"]), 8),
                "v2_clean_balanced_accuracy": round(float(s2["clean_balanced_accuracy"]), 8),
                "delta_clean_balanced_accuracy": round(float(s2["clean_balanced_accuracy"]) - float(s1["clean_balanced_accuracy"]), 8),
            }
            split_comparison.append(row)
            split_delta_chart.append({
                "model": model.upper(),
                "model_order": MODEL_ORDER[model],
                "split": split_title,
                "split_order": split_order,
                "delta_clean_auroc": row["delta_clean_auroc"],
            })

        for condition_order, (condition_index, condition) in enumerate(MULTISTAGE):
            r1 = v1[(model, "unseen_generator")]["metrics"][condition_index]
            r2 = v2[(model, "unseen_generator")]["metrics"][condition_index]
            strict_multistage.append({
                "model": model.upper(),
                "model_order": MODEL_ORDER[model],
                "condition": condition,
                "condition_order": condition_order,
                "v1_auroc": round(float(r1["auroc"]), 8),
                "v2_auroc": round(float(r2["auroc"]), 8),
                "delta_auroc": round(float(r2["auroc"]) - float(r1["auroc"]), 8),
                "v1_balanced_accuracy": round(float(r1["balanced_accuracy"]), 8),
                "v2_balanced_accuracy": round(float(r2["balanced_accuracy"]), 8),
                "delta_balanced_accuracy": round(float(r2["balanced_accuracy"]) - float(r1["balanced_accuracy"]), 8),
            })
            for protocol_order, (protocol, row) in enumerate((("train-v1", r1), ("train-v2", r2))):
                strict_multistage_chart.append({
                    "model": model.upper(),
                    "model_order": MODEL_ORDER[model],
                    "series": f"{protocol} · {condition}",
                    "series_order": condition_order * 2 + protocol_order,
                    "auroc": round(float(row["auroc"]), 8),
                })

    best_macro = max(model_macro, key=lambda row: row["v2_clean_macro_auroc"])
    strict_rows = [row for row in split_comparison if row["split"] == "Strict unseen-generator"]
    best_strict = max(strict_rows, key=lambda row: row["v2_clean_auroc"])
    comparison_headline = [{
        "best_v2_macro_model": best_macro["model"],
        "best_v2_macro_auroc": best_macro["v2_clean_macro_auroc"],
        "best_v2_strict_model": best_strict["model"],
        "best_v2_strict_auroc": best_strict["v2_clean_auroc"],
        "models_with_positive_macro_delta": sum(row["delta_clean_macro_auroc"] > 0 for row in model_macro),
        "train_v2_images": v2_training["rows"],
        "inference_samples": len(MODELS) * 5500 * 21,
    }]

    datasets = {
        "comparison_headline": comparison_headline,
        "training_summary": training_summary,
        "training_class_chart": training_class_chart,
        "training_architecture_chart": training_architecture_chart,
        "model_macro_comparison": model_macro,
        "macro_protocol_chart": macro_chart,
        "split_comparison": split_comparison,
        "split_delta_chart": split_delta_chart,
        "strict_multistage_comparison": strict_multistage,
        "strict_multistage_chart": strict_multistage_chart,
    }
    audit = {
        "train_v1_manifest": str(arguments.train_v1_manifest),
        "train_v1_manifest_sha256": v1_training["sha256"],
        "train_v2_manifest": str(arguments.train_v2_manifest),
        "train_v2_manifest_sha256": v2_training["sha256"],
        "v1_evaluation_root": str(arguments.v1_evaluation_root),
        "v2_evaluation_root": str(arguments.v2_evaluation_root),
        "split_count": len(SPLITS),
        "model_count": len(MODELS),
        "condition_count": 21,
    }
    return datasets, audit


def _m3_generator_roc_datasets(arguments: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    manifest_rows = _read_csv(arguments.unseen_manifest)
    manifest_by_id = {row["sample_id"]: row for row in manifest_rows}
    if len(manifest_by_id) != 2000:
        raise RuntimeError("M3 generator ROC analysis requires 2,000 unique manifest rows")
    real_ids = [row["sample_id"] for row in manifest_rows if int(row["label"]) == 0]
    generator_ids: dict[str, list[str]] = defaultdict(list)
    for row in manifest_rows:
        if int(row["label"]) == 1:
            generator_ids[row["canonical_generator_id"]].append(row["sample_id"])
    if len(real_ids) != 1000 or len(generator_ids) != 12:
        raise RuntimeError("Expected 1,000 Real images and 12 exact unseen generators")

    output = Path(arguments.v2_evaluation_root) / "m3" / "unseen_generator"
    metrics_rows = _read_csv(output / "metrics_by_transform.csv")
    predictions = unseen._read_jsonl(output / "predictions.jsonl")
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in predictions:
        transform = str(row["transform"])
        sample_id = str(row["sample_id"])
        if sample_id in grouped[transform]:
            raise RuntimeError(f"Duplicate M3 prediction: {transform}/{sample_id}")
        grouped[transform][sample_id] = row
    threshold = float(_read_json(Path(arguments.v2_source_root) / "m3" / "summary.json")["threshold_from_clean_validation"])
    targets = (
        (0, "Clean"),
        (18, "4-stage A platform repost"),
        (19, "4-stage B edit repost"),
    )
    detail: list[dict[str, Any]] = []
    generators = sorted(generator_ids)
    for condition_order, (condition_index, condition) in enumerate(targets):
        transform = metrics_rows[condition_index]["transform"]
        row_by_id = grouped.get(transform, {})
        if set(row_by_id) != set(manifest_by_id):
            raise RuntimeError(f"M3 prediction identity mismatch for {condition}")
        real_scores = np.asarray([float(row_by_id[sample_id]["pred"]) for sample_id in real_ids], dtype=np.float64)
        for generator_order, generator in enumerate(generators):
            positive_ids = generator_ids[generator]
            positive_scores = np.asarray(
                [float(row_by_id[sample_id]["pred"]) for sample_id in positive_ids],
                dtype=np.float64,
            )
            labels = np.concatenate((
                np.zeros(real_scores.size, dtype=np.int64),
                np.ones(positive_scores.size, dtype=np.int64),
            ))
            scores = np.concatenate((real_scores, positive_scores))
            computed = unseen._metrics(labels, scores, threshold)
            detail.append({
                "model": "M3",
                "generator": generator,
                "generator_order": generator_order,
                "condition": condition,
                "condition_order": condition_order,
                "n_real": int(real_scores.size),
                "n_aigi": int(positive_scores.size),
                "threshold": round(threshold, 8),
                "auroc": round(float(computed["auroc"]), 8),
                "average_precision": round(float(computed["average_precision"]), 8),
                "accuracy": round(float(computed["accuracy"]), 8),
                "balanced_accuracy": round(float(computed["balanced_accuracy"]), 8),
                "aigi_recall": round(float(computed["recall"]), 8),
                "real_specificity": round(float(computed["specificity"]), 8),
                "false_positive_rate": round(float(computed["false_positive_rate"]), 8),
                "tpr_at_fpr_1pct": round(float(computed["tpr_at_fpr_1pct"]), 8),
                "tpr_at_fpr_5pct": round(float(computed["tpr_at_fpr_5pct"]), 8),
                "tn": int(computed["tn"]),
                "fp": int(computed["fp"]),
                "fn": int(computed["fn"]),
                "tp": int(computed["tp"]),
                "mean_real_score": round(float(real_scores.mean()), 8),
                "mean_aigi_score": round(float(positive_scores.mean()), 8),
            })

    by_key = {(row["generator"], row["condition"]): row for row in detail}
    wide: list[dict[str, Any]] = []
    for generator_order, generator in enumerate(generators):
        clean = by_key[(generator, targets[0][1])]
        four_a = by_key[(generator, targets[1][1])]
        four_b = by_key[(generator, targets[2][1])]
        wide.append({
            "model": "M3",
            "generator": generator,
            "generator_order": generator_order,
            "n_aigi": clean["n_aigi"],
            "clean_auroc": clean["auroc"],
            "four_stage_a_auroc": four_a["auroc"],
            "four_stage_a_delta": round(four_a["auroc"] - clean["auroc"], 8),
            "four_stage_b_auroc": four_b["auroc"],
            "four_stage_b_delta": round(four_b["auroc"] - clean["auroc"], 8),
            "clean_recall": clean["aigi_recall"],
            "four_stage_a_recall": four_a["aigi_recall"],
            "four_stage_b_recall": four_b["aigi_recall"],
        })

    condition_means = {
        condition: fmean(row["auroc"] for row in detail if row["condition"] == condition)
        for _, condition in targets
    }
    condition_worst = {
        condition: min(
            (row for row in detail if row["condition"] == condition),
            key=lambda row: row["auroc"],
        )
        for _, condition in targets
    }
    headline = [{
        "model": "M3",
        "generator_count": len(generators),
        "real_reference_count": len(real_ids),
        "clean_mean_auroc": round(condition_means[targets[0][1]], 8),
        "four_stage_a_mean_auroc": round(condition_means[targets[1][1]], 8),
        "four_stage_b_mean_auroc": round(condition_means[targets[2][1]], 8),
        "clean_worst_generator": condition_worst[targets[0][1]]["generator"],
        "clean_worst_auroc": condition_worst[targets[0][1]]["auroc"],
        "four_stage_a_worst_generator": condition_worst[targets[1][1]]["generator"],
        "four_stage_a_worst_auroc": condition_worst[targets[1][1]]["auroc"],
        "four_stage_b_worst_generator": condition_worst[targets[2][1]]["generator"],
        "four_stage_b_worst_auroc": condition_worst[targets[2][1]]["auroc"],
    }]
    return {
        "m3_generator_roc_headline": headline,
        "m3_generator_roc_detail": detail,
        "m3_generator_roc_wide": wide,
    }


def _columns(*values: tuple[str, str, str | None]) -> list[dict[str, Any]]:
    columns: list[dict[str, Any]] = []
    for field, label, format_name in values:
        item: dict[str, Any] = {"field": field, "label": label}
        if format_name:
            item["format"] = format_name
        columns.append(item)
    return columns


def _comparison_charts(source_ids: dict[str, str]) -> list[dict[str, Any]]:
    common = {
        "palette": {"kind": "categorical"},
        "legend": {"position": "bottom", "interactive": True},
        "labels": {"values": "none"},
        "layout": "full",
    }
    return [
        {
            "id": "training_class_chart",
            "title": "train-v1 与 train-v2 类别规模",
            "subtitle": "两轮均保持 Real/AIGI 1:1；train-v2 各增加 1,000 张。",
            "type": "bar",
            "intent": "comparison",
            "question": "训练集扩展是否保持类别平衡？",
            "rationale": "两个协议和两个类别适合分组柱状图。",
            "comparisonContext": {"unit": "images", "grain": "protocol by class"},
            "dataset": "training_class_chart",
            "sourceId": source_ids["training_class_chart"],
            "encodings": {
                "x": {"field": "protocol", "type": "nominal", "label": "Protocol"},
                "y": {"field": "images", "type": "quantitative", "label": "Images", "format": "number"},
                "color": {"field": "class", "type": "nominal", "label": "Class"},
                "tooltip": [{"field": "images", "type": "quantitative", "label": "Images", "format": "number"}],
            },
            "valueFormat": "number",
            **common,
        },
        {
            "id": "training_architecture_chart",
            "title": "AIGI 生成器大类构成变化",
            "subtitle": "新增 222 GAN、667 Latent Diffusion 与 111 Pixel Diffusion 图像。",
            "type": "bar",
            "intent": "comparison",
            "question": "train-v2 增加了哪些生成器大类？",
            "rationale": "架构类别数量适合分组柱状图。",
            "comparisonContext": {"unit": "AIGI images", "grain": "protocol by architecture"},
            "dataset": "training_architecture_chart",
            "sourceId": source_ids["training_architecture_chart"],
            "encodings": {
                "x": {"field": "architecture", "type": "nominal", "label": "Architecture"},
                "y": {"field": "images", "type": "quantitative", "label": "AIGI images", "format": "number"},
                "color": {"field": "protocol", "type": "nominal", "label": "Protocol"},
                "tooltip": [{"field": "images", "type": "quantitative", "label": "Images", "format": "number"}],
            },
            "valueFormat": "number",
            **common,
        },
        {
            "id": "macro_protocol_chart",
            "title": "五切片宏平均 AUROC：train-v1 对比 train-v2",
            "subtitle": "五个保留切片等权；扰动均值排除 clean。",
            "type": "bar",
            "intent": "comparison",
            "question": "扩展训练集后各模型的 clean 与扰动宏平均如何变化？",
            "rationale": "同模型四个协议/条件汇总值适合分组柱状图。",
            "comparisonContext": {"unit": "AUROC", "grain": "model by protocol summary"},
            "dataset": "macro_protocol_chart",
            "sourceId": source_ids["macro_protocol_chart"],
            "encodings": {
                "x": {"field": "model", "type": "nominal", "label": "Model"},
                "y": {"field": "auroc", "type": "quantitative", "label": "AUROC", "format": "percent"},
                "color": {"field": "series", "type": "nominal", "label": "Protocol / condition"},
                "tooltip": [{"field": "auroc", "type": "quantitative", "label": "AUROC", "format": "percent"}],
            },
            "valueFormat": "percent",
            **common,
        },
        {
            "id": "split_delta_chart",
            "title": "各切片 clean AUROC 的 train-v2 增量",
            "subtitle": "正值表示 train-v2 提升；相同测试图片按模型与切片对齐。",
            "type": "bar",
            "intent": "comparison",
            "question": "性能增益集中在哪些生成器暴露切片？",
            "rationale": "带正负号的切片增量适合以零为基线的分组柱状图。",
            "comparisonContext": {"unit": "AUROC delta", "grain": "split by model", "baseline": 0},
            "dataset": "split_delta_chart",
            "sourceId": source_ids["split_delta_chart"],
            "encodings": {
                "x": {"field": "split", "type": "nominal", "label": "Split"},
                "y": {"field": "delta_clean_auroc", "type": "quantitative", "label": "AUROC delta", "format": "percent"},
                "color": {"field": "model", "type": "nominal", "label": "Model"},
                "tooltip": [{"field": "delta_clean_auroc", "type": "quantitative", "label": "Delta", "format": "percent"}],
            },
            "valueFormat": "percent",
            **common,
        },
        {
            "id": "strict_multistage_chart",
            "title": "Strict unseen 多阶段扰动 AUROC",
            "subtitle": "两组四阶段和一组随机六阶段；每个模型同时展示 train-v1/v2。",
            "type": "bar",
            "intent": "comparison",
            "question": "train-v2 是否改善 strict unseen 的复合扰动鲁棒性？",
            "rationale": "五个模型与六条协议/条件序列适合分组柱状图。",
            "comparisonContext": {"unit": "AUROC", "grain": "model by protocol and condition"},
            "dataset": "strict_multistage_chart",
            "sourceId": source_ids["strict_multistage_chart"],
            "encodings": {
                "x": {"field": "model", "type": "nominal", "label": "Model"},
                "y": {"field": "auroc", "type": "quantitative", "label": "AUROC", "format": "percent"},
                "color": {"field": "series", "type": "nominal", "label": "Protocol / condition"},
                "tooltip": [{"field": "auroc", "type": "quantitative", "label": "AUROC", "format": "percent"}],
            },
            "valueFormat": "percent",
            **common,
        },
        {
            "id": "m3_generator_roc_chart",
            "title": "[M3] 12 个未见精确生成器的 AUROC 对比",
            "subtitle": "每个生成器的 AIGI 与同一组 1,000 张 Real 比较；AUROC 为 ROC 曲线的标量汇总。",
            "type": "bar",
            "intent": "comparison",
            "question": "M3 在 Clean、4-stage A 和 4-stage B 下对各精确生成器的 ROC 排序能力如何变化？",
            "rationale": "12 个离散生成器与三个同尺度 AUROC 条件适合分组柱状图；36 条完整 ROC 曲线会过度拥挤。",
            "comparisonContext": {
                "unit": "AUROC",
                "grain": "exact generator by condition",
                "negative_reference": "the same 1,000 strict-unseen Real images",
            },
            "dataset": "m3_generator_roc_detail",
            "sourceId": source_ids["m3_generator_roc_detail"],
            "encodings": {
                "x": {"field": "generator", "type": "nominal", "label": "Exact unseen generator"},
                "y": {"field": "auroc", "type": "quantitative", "label": "AUROC", "format": "percent"},
                "color": {"field": "condition", "type": "nominal", "label": "Condition"},
                "tooltip": [
                    {"field": "model", "type": "nominal", "label": "Model"},
                    {"field": "n_aigi", "type": "quantitative", "label": "AIGI N", "format": "number"},
                    {"field": "n_real", "type": "quantitative", "label": "Real N", "format": "number"},
                    {"field": "auroc", "type": "quantitative", "label": "AUROC", "format": "percent"},
                    {"field": "aigi_recall", "type": "quantitative", "label": "Frozen-threshold recall", "format": "percent"},
                ],
            },
            "valueFormat": "percent",
            **common,
        },
    ]


def _comparison_tables(source_ids: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {
            "id": "training_summary_table",
            "title": "训练集变化",
            "subtitle": "seen-family 的 2,000 张图片被提升为训练数据，不再作为测试。",
            "dataset": "training_summary",
            "sourceId": source_ids["training_summary"],
            "defaultSort": {"field": "metric", "direction": "asc"},
            "columns": _columns(
                ("metric", "指标", None), ("train_v1", "train-v1", "number"),
                ("train_v2", "train-v2", "number"), ("delta", "变化", "number"),
                ("interpretation", "说明", None),
            ),
        },
        {
            "id": "model_macro_comparison_table",
            "title": "五切片宏平均对比",
            "subtitle": "五切片等权，避免 2,000 张切片压过 500 张 hard 切片。",
            "dataset": "model_macro_comparison",
            "sourceId": source_ids["model_macro_comparison"],
            "defaultSort": {"field": "v2_clean_macro_auroc", "direction": "desc"},
            "columns": _columns(
                ("model", "模型", None),
                ("v1_clean_macro_auroc", "v1 Clean AUC", "percent"),
                ("v2_clean_macro_auroc", "v2 Clean AUC", "percent"),
                ("delta_clean_macro_auroc", "Clean Δ", "percent"),
                ("v1_perturbation_macro_auroc", "v1 扰动 AUC", "percent"),
                ("v2_perturbation_macro_auroc", "v2 扰动 AUC", "percent"),
                ("delta_perturbation_macro_auroc", "扰动 Δ", "percent"),
                ("v2_clean_macro_balanced_accuracy", "v2 Clean BA", "percent"),
                ("delta_clean_macro_balanced_accuracy", "BA Δ", "percent"),
            ),
        },
        {
            "id": "split_comparison_table",
            "title": "模型 × 切片 train-v1/v2 完整对比",
            "subtitle": "25 个对齐单元，含 clean、20 扰动均值与冻结阈值 BA。",
            "dataset": "split_comparison",
            "sourceId": source_ids["split_comparison"],
            "defaultSort": {"field": "delta_clean_auroc", "direction": "desc"},
            "density": "dense",
            "columns": _columns(
                ("model", "模型", None), ("split", "切片", None),
                ("v1_clean_auroc", "v1 Clean AUC", "percent"),
                ("v2_clean_auroc", "v2 Clean AUC", "percent"),
                ("delta_clean_auroc", "Clean Δ", "percent"),
                ("v1_perturbation_mean_auroc", "v1 扰动 AUC", "percent"),
                ("v2_perturbation_mean_auroc", "v2 扰动 AUC", "percent"),
                ("delta_perturbation_mean_auroc", "扰动 Δ", "percent"),
                ("v1_clean_balanced_accuracy", "v1 Clean BA", "percent"),
                ("v2_clean_balanced_accuracy", "v2 Clean BA", "percent"),
                ("delta_clean_balanced_accuracy", "BA Δ", "percent"),
            ),
        },
        {
            "id": "strict_multistage_comparison_table",
            "title": "Strict unseen 多阶段扰动对比",
            "subtitle": "5 模型 × 3 复合扰动；阈值由各协议内部验证冻结。",
            "dataset": "strict_multistage_comparison",
            "sourceId": source_ids["strict_multistage_comparison"],
            "defaultSort": {"field": "v2_auroc", "direction": "desc"},
            "columns": _columns(
                ("model", "模型", None), ("condition", "条件", None),
                ("v1_auroc", "v1 AUROC", "percent"), ("v2_auroc", "v2 AUROC", "percent"),
                ("delta_auroc", "AUROC Δ", "percent"),
                ("v1_balanced_accuracy", "v1 BA", "percent"),
                ("v2_balanced_accuracy", "v2 BA", "percent"),
                ("delta_balanced_accuracy", "BA Δ", "percent"),
            ),
        },
        {
            "id": "m3_generator_roc_wide_table",
            "title": "[M3] 12 个精确生成器 AUROC 与 Recall 摘要",
            "subtitle": "三种条件使用相同的 1,000 张 Real 负类；Δ 相对各生成器 Clean AUROC。",
            "dataset": "m3_generator_roc_wide",
            "sourceId": source_ids["m3_generator_roc_wide"],
            "defaultSort": {"field": "clean_auroc", "direction": "asc"},
            "columns": _columns(
                ("model", "模型", None), ("generator", "精确生成器", None),
                ("n_aigi", "AIGI N", "number"),
                ("clean_auroc", "Clean AUROC", "percent"),
                ("four_stage_a_auroc", "4-stage A AUROC", "percent"),
                ("four_stage_a_delta", "A Δ", "percent"),
                ("four_stage_b_auroc", "4-stage B AUROC", "percent"),
                ("four_stage_b_delta", "B Δ", "percent"),
                ("clean_recall", "Clean Recall", "percent"),
                ("four_stage_a_recall", "A Recall", "percent"),
                ("four_stage_b_recall", "B Recall", "percent"),
            ),
        },
        {
            "id": "m3_generator_roc_detail_table",
            "title": "[M3] 36 条生成器 × 条件 ROC 与固定阈值明细",
            "subtitle": "AUROC/AP 为排序指标；Accuracy、BA、Recall、Specificity 和混淆矩阵使用冻结阈值。",
            "dataset": "m3_generator_roc_detail",
            "sourceId": source_ids["m3_generator_roc_detail"],
            "defaultSort": {"field": "auroc", "direction": "asc"},
            "density": "dense",
            "columns": _columns(
                ("model", "模型", None), ("generator", "精确生成器", None),
                ("condition", "条件", None), ("n_aigi", "AIGI N", "number"),
                ("n_real", "Real N", "number"), ("auroc", "AUROC", "percent"),
                ("average_precision", "AP", "percent"), ("accuracy", "Accuracy", "percent"),
                ("balanced_accuracy", "BA", "percent"), ("aigi_recall", "Recall", "percent"),
                ("real_specificity", "Specificity", "percent"),
                ("tpr_at_fpr_1pct", "TPR@1%FPR", "percent"),
                ("tpr_at_fpr_5pct", "TPR@5%FPR", "percent"),
                ("tn", "TN", "number"), ("fp", "FP", "number"),
                ("fn", "FN", "number"), ("tp", "TP", "number"),
                ("mean_real_score", "Mean Real score", "number"),
                ("mean_aigi_score", "Mean AIGI score", "number"),
            ),
        },
    ]


def generate(arguments: argparse.Namespace) -> None:
    generated_at = datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds")
    strict_arguments = SimpleNamespace(
        evaluation_root=arguments.v2_evaluation_root,
        source_root=arguments.v2_source_root,
        manifest=arguments.unseen_manifest,
        matrix=arguments.matrix,
        bootstrap_replicates=arguments.bootstrap_replicates,
        bootstrap_seed=arguments.bootstrap_seed,
    )
    strict_datasets, strict_audit = unseen._load_and_compute(strict_arguments)
    comparison_datasets, comparison_audit = _comparison_datasets(arguments)
    m3_generator_datasets = _m3_generator_roc_datasets(arguments)
    staged = {**comparison_datasets, **m3_generator_datasets, **strict_datasets}
    queries = {
        "comparison_headline": "SELECT * FROM comparison_headline",
        "training_summary": "SELECT * FROM training_summary ORDER BY metric",
        "training_class_chart": "SELECT * FROM training_class_chart ORDER BY protocol, class_order",
        "training_architecture_chart": "SELECT * FROM training_architecture_chart ORDER BY architecture_order, protocol",
        "model_macro_comparison": "SELECT * FROM model_macro_comparison ORDER BY model_order",
        "macro_protocol_chart": "SELECT * FROM macro_protocol_chart ORDER BY series_order, model_order",
        "split_comparison": "SELECT * FROM split_comparison ORDER BY split_order, model_order",
        "split_delta_chart": "SELECT * FROM split_delta_chart ORDER BY split_order, model_order",
        "strict_multistage_comparison": "SELECT * FROM strict_multistage_comparison ORDER BY condition_order, model_order",
        "strict_multistage_chart": "SELECT * FROM strict_multistage_chart ORDER BY series_order, model_order",
        "m3_generator_roc_headline": "SELECT * FROM m3_generator_roc_headline",
        "m3_generator_roc_detail": "SELECT * FROM m3_generator_roc_detail ORDER BY generator_order, condition_order",
        "m3_generator_roc_wide": "SELECT * FROM m3_generator_roc_wide ORDER BY generator_order",
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
    snapshot = unseen._materialize_sql_snapshot(staged, queries)
    source_ids = {dataset: f"{dataset}_sql" for dataset in queries}
    artifact = unseen._build_artifact(snapshot, queries, generated_at)
    title = "Community Forensics train-v1 / train-v2 对比与 Strict unseen 详细评测"
    artifact["manifest"]["title"] = title
    artifact["manifest"]["description"] = (
        "Train-v1/train-v2 dataset and five-split robustness comparison, plus full "
        "strict unseen-generator classification, ROC/PR, perturbation and slice metrics."
    )
    artifact["manifest"]["charts"].extend(_comparison_charts(source_ids))
    artifact["manifest"]["tables"].extend(_comparison_tables(source_ids))
    comparison_cards = [
        {
            "id": "best_v2_macro_card",
            "description": "五个保留切片 clean AUROC 等权平均。",
            "dataset": "comparison_headline",
            "sourceId": source_ids["comparison_headline"],
            "metrics": [{"label": "train-v2 最佳宏平均 AUROC", "field": "best_v2_macro_auroc", "format": "percent"}],
        },
        {
            "id": "best_v2_strict_card",
            "description": "Strict unseen clean AUROC；阈值无关。",
            "dataset": "comparison_headline",
            "sourceId": source_ids["comparison_headline"],
            "metrics": [{"label": "train-v2 最佳 Strict AUROC", "field": "best_v2_strict_auroc", "format": "percent"}],
        },
        {
            "id": "train_v2_size_card",
            "description": "Real/AIGI 各 10,000 张。",
            "dataset": "comparison_headline",
            "sourceId": source_ids["comparison_headline"],
            "metrics": [{"label": "train-v2 图片数", "field": "train_v2_images", "format": "number"}],
        },
    ]
    m3_generator_cards = [
        {
            "id": "m3_generator_clean_auc_card",
            "description": "12 个精确生成器等权平均；共同 Real 参照面板。",
            "dataset": "m3_generator_roc_headline",
            "sourceId": source_ids["m3_generator_roc_headline"],
            "metrics": [{"label": "M3 Clean mean AUROC", "field": "clean_mean_auroc", "format": "percent"}],
        },
        {
            "id": "m3_generator_four_a_auc_card",
            "description": "12 个精确生成器等权平均。",
            "dataset": "m3_generator_roc_headline",
            "sourceId": source_ids["m3_generator_roc_headline"],
            "metrics": [{"label": "M3 4-stage A mean AUROC", "field": "four_stage_a_mean_auroc", "format": "percent"}],
        },
        {
            "id": "m3_generator_four_b_auc_card",
            "description": "12 个精确生成器等权平均。",
            "dataset": "m3_generator_roc_headline",
            "sourceId": source_ids["m3_generator_roc_headline"],
            "metrics": [{"label": "M3 4-stage B mean AUROC", "field": "four_stage_b_mean_auroc", "format": "percent"}],
        },
    ]
    artifact["manifest"]["cards"].extend(comparison_cards + m3_generator_cards)

    markdown = Path(arguments.markdown).read_text(encoding="utf-8")
    if not markdown.startswith("# Community Forensics train-v1 / train-v2"):
        raise RuntimeError("Unexpected comparison Markdown title")
    markdown_body = markdown.split("\n", 1)[1].lstrip()
    comparison_headline = snapshot["comparison_headline"][0]
    comparison_blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {
            "id": "comparison_summary",
            "type": "markdown",
            "sourceId": source_ids["comparison_headline"],
            "body": (
                "## 结论先行\n\n"
                f"train-v2 的五切片 clean 宏平均最优模型为 **{comparison_headline['best_v2_macro_model']}** "
                f"（AUROC {comparison_headline['best_v2_macro_auroc']:.4f}）；Strict unseen clean 最优为 "
                f"**{comparison_headline['best_v2_strict_model']}**（AUROC {comparison_headline['best_v2_strict_auroc']:.4f}）。"
                f"五个模型的宏平均 clean AUROC 均相对 train-v1 提高。完整协议覆盖 "
                f"{comparison_headline['inference_samples']:,} 个 train-v2 图像—条件推理样本。"
            ),
        },
        {
            "id": "comparison_cards",
            "type": "metric-strip",
            "cardIds": ["best_v2_macro_card", "best_v2_strict_card", "train_v2_size_card"],
        },
        {
            "id": "training_change_intro",
            "type": "markdown",
            "sourceId": source_ids["training_summary"],
            "body": (
                "## 训练数据从 18k 扩展到 20k，并保持类别平衡\n\n"
                "新增的 2,000 张图片来自原 External seen-family，其中 Real/AIGI 各 1,000 张。"
                "这批图片不再参与后续测试；train-v2 同时增加 GAN、Latent Diffusion、Pixel Diffusion 和多种文件格式。"
            ),
        },
        {"id": "training_class_chart_block", "type": "chart", "chartId": "training_class_chart", "layout": "full"},
        {"id": "training_architecture_chart_block", "type": "chart", "chartId": "training_architecture_chart", "layout": "full"},
        {"id": "training_summary_table_block", "type": "table", "tableId": "training_summary_table", "layout": "full"},
        {
            "id": "macro_comparison_intro",
            "type": "markdown",
            "sourceId": source_ids["model_macro_comparison"],
            "body": (
                "## train-v2 总体提高，但 B2 与 M3 仍承担不同角色\n\n"
                "B2 保持最高五切片宏平均排序性能；M3 在 strict unseen 上最强。"
                "宏平均对五个切片等权，避免样本量较大的 2,000 张切片完全支配三个 500 张 hard 切片。"
            ),
        },
        {"id": "macro_protocol_chart_block", "type": "chart", "chartId": "macro_protocol_chart", "layout": "full"},
        {"id": "model_macro_comparison_table_block", "type": "table", "tableId": "model_macro_comparison_table", "layout": "full"},
        {"id": "split_delta_chart_block", "type": "chart", "chartId": "split_delta_chart", "layout": "full"},
        {"id": "split_comparison_table_block", "type": "table", "tableId": "split_comparison_table", "layout": "full"},
        {
            "id": "strict_multistage_intro",
            "type": "markdown",
            "sourceId": source_ids["strict_multistage_comparison"],
            "body": (
                "## Strict unseen 的三组多阶段扰动全部改善\n\n"
                "两组四阶段和一组随机六阶段用于模拟编辑—转发链。图表同时展示 train-v1/v2，"
                "随后完整 strict-unseen 部分进一步给出 Accuracy、Precision、Recall、Specificity、F1、MCC、ROC、PR 和逐生成器结果。"
            ),
        },
        {"id": "strict_multistage_chart_block", "type": "chart", "chartId": "strict_multistage_chart", "layout": "full"},
        {"id": "strict_multistage_comparison_table_block", "type": "table", "tableId": "strict_multistage_comparison_table", "layout": "full"},
        {
            "id": "m3_generator_roc_intro",
            "type": "markdown",
            "sourceId": source_ids["m3_generator_roc_headline"],
            "body": (
                "## M3 标识：12 个未见精确生成器的 Clean 与四阶段 ROC 对比\n\n"
                "**模型标识：M3。** 对每个精确生成器，将其 83–84 张 AIGI 作为正类，并使用相同的 1,000 张 "
                "strict-unseen Real 作为负类，分别计算 Clean、4-stage A platform repost 和 4-stage B edit repost。"
                "图中展示 AUROC（ROC 曲线下面积），避免同时绘制 36 条 ROC 曲线造成遮挡；下方两张表保留 AP、"
                "冻结阈值 Recall/Specificity、TPR@1%/5%FPR、混淆矩阵和平均得分等详细数据。"
            ),
        },
        {
            "id": "m3_generator_roc_cards",
            "type": "metric-strip",
            "cardIds": ["m3_generator_clean_auc_card", "m3_generator_four_a_auc_card", "m3_generator_four_b_auc_card"],
        },
        {"id": "m3_generator_roc_chart_block", "type": "chart", "chartId": "m3_generator_roc_chart", "layout": "full"},
        {"id": "m3_generator_roc_wide_table_block", "type": "table", "tableId": "m3_generator_roc_wide_table", "layout": "full"},
        {"id": "m3_generator_roc_detail_table_block", "type": "table", "tableId": "m3_generator_roc_detail_table", "layout": "full"},
        {
            "id": "markdown_source",
            "type": "markdown",
            "sourceId": source_ids["model_macro_comparison"],
            "body": "## 原 Markdown 报告正文\n\n" + markdown_body,
        },
        {
            "id": "strict_detail_divider",
            "type": "markdown",
            "sourceId": source_ids["clean_metrics"],
            "body": (
                "# Strict unseen-generator 详细评测\n\n"
                "以下部分直接从冻结 train-v2 predictions 重算指标；clean 使用 1,000 次 Real/AIGI 分层 bootstrap，"
                "测试阈值不重新拟合。"
            ),
        },
    ]
    original_blocks = [block for block in artifact["manifest"]["blocks"] if block["id"] != "title"]
    artifact["manifest"]["blocks"] = comparison_blocks + original_blocks

    comparison_source_names = set(comparison_datasets)
    m3_generator_source_names = set(m3_generator_datasets)
    for source in artifact["sources"]:
        dataset = source["id"].removesuffix("_sql")
        if dataset in m3_generator_source_names:
            source["query"]["description"] = (
                "Executed over frozen M3 strict-unseen predictions, with each exact "
                "generator's AIGI images compared against the same 1,000 Real images."
            )
            source["query"]["filters"] = [
                "model = M3",
                "conditions = clean, four-stage A, four-stage B",
                "positive class = one exact unseen generator",
                "negative class = all 1,000 strict-unseen Real images",
                "threshold frozen from internal Small clean validation",
            ]
        elif dataset in comparison_source_names:
            source["query"]["description"] = (
                "Executed over reviewed train-v1/train-v2 manifests and aligned frozen "
                "21-condition evaluation summaries."
            )
            source["query"]["filters"] = [
                "five retained evaluation slices",
                "same test sample identities for direct comparisons",
                "external seen-family excluded from train-v2 evaluation",
            ]

    unseen._write_csv(arguments.comparison_csv, snapshot["split_comparison"])
    unseen._write_csv(arguments.strict_all_metrics_csv, snapshot["condition_metrics"])
    unseen._write_csv(arguments.strict_clean_metrics_csv, snapshot["clean_metrics"])
    unseen._write_csv(arguments.strict_generator_csv, snapshot["generator_recall"])
    unseen._write_csv(arguments.strict_real_source_csv, snapshot["real_specificity"])
    unseen._write_csv(arguments.m3_generator_roc_csv, snapshot["m3_generator_roc_detail"])
    _atomic_text(arguments.artifact_json, json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    audit = {
        "schema_version": 1,
        "generated_at_asia_singapore": generated_at,
        "report_job_id": os.environ.get("SLURM_JOB_ID"),
        "markdown": str(arguments.markdown),
        "markdown_sha256": _sha256(arguments.markdown),
        "comparison": comparison_audit,
        "strict_unseen": strict_audit,
        "bootstrap_replicates": arguments.bootstrap_replicates,
        "bootstrap_seed": arguments.bootstrap_seed,
        "dataset_rows": {name: len(rows) for name, rows in snapshot.items()},
        "chart_count": len(artifact["manifest"]["charts"]),
        "table_count": len(artifact["manifest"]["tables"]),
        "output_files": {
            "artifact_json": str(arguments.artifact_json),
            "comparison_csv": str(arguments.comparison_csv),
            "strict_all_metrics_csv": str(arguments.strict_all_metrics_csv),
            "strict_clean_metrics_csv": str(arguments.strict_clean_metrics_csv),
            "strict_generator_csv": str(arguments.strict_generator_csv),
            "strict_real_source_csv": str(arguments.strict_real_source_csv),
            "m3_generator_roc_csv": str(arguments.m3_generator_roc_csv),
        },
    }
    _atomic_text(arguments.audit_json, json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "event": "community_forensics_train_v1_v2_comparison_report_complete",
        "artifact_json": str(arguments.artifact_json),
        "audit_json": str(arguments.audit_json),
        "charts": len(artifact["manifest"]["charts"]),
        "tables": len(artifact["manifest"]["tables"]),
        "strict_condition_rows": len(snapshot["condition_metrics"]),
    }, sort_keys=True), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the combined train-v1/v2 and strict unseen report artifact")
    parser.add_argument("--markdown", default="reports/summaries/COMMUNITY_FORENSICS_TRAIN_V1_V2_COMPARISON.md")
    parser.add_argument("--train-v1-manifest", default="data/manifests/community_forensics_train.csv")
    parser.add_argument("--train-v2-manifest", default="data/manifests/community_forensics_train_v2.csv")
    parser.add_argument("--v1-evaluation-root", default="outputs/community_forensics_robustness_v2")
    parser.add_argument("--v2-evaluation-root", default="outputs/community_forensics_v2_robustness_v2")
    parser.add_argument("--v2-source-root", default="outputs/community_forensics_v2")
    parser.add_argument("--unseen-manifest", default="data/manifests/community_forensics_test_external_unseen_generator.csv")
    parser.add_argument("--matrix", default="configs/community_forensics_robustness_v2.yaml")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260830)
    parser.add_argument("--artifact-json", default="reports/evaluations/train_v1_v2_comparison/community_forensics_train_v1_v2_artifact.json")
    parser.add_argument("--audit-json", default="reports/evaluations/train_v1_v2_comparison/community_forensics_train_v1_v2_notes.json")
    parser.add_argument("--comparison-csv", default="reports/evaluations/train_v1_v2_comparison/community_forensics_train_v1_v2_split_comparison.csv")
    parser.add_argument("--strict-all-metrics-csv", default="reports/evaluations/train_v1_v2_comparison/community_forensics_strict_unseen_all_metrics.csv")
    parser.add_argument("--strict-clean-metrics-csv", default="reports/evaluations/train_v1_v2_comparison/community_forensics_strict_unseen_clean_metrics.csv")
    parser.add_argument("--strict-generator-csv", default="reports/evaluations/train_v1_v2_comparison/community_forensics_strict_unseen_generator_metrics.csv")
    parser.add_argument("--strict-real-source-csv", default="reports/evaluations/train_v1_v2_comparison/community_forensics_strict_unseen_real_source_metrics.csv")
    parser.add_argument("--m3-generator-roc-csv", default="reports/evaluations/train_v1_v2_comparison/community_forensics_m3_unseen_generator_roc_metrics.csv")
    return parser.parse_args()


def main() -> None:
    generate(_parse_args())


if __name__ == "__main__":
    main()
