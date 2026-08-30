from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

import generate_community_forensics_robustness_v2_report as report


V3_SPLITS = (
    (
        "exact_seen_generator",
        "External exact-seen generator",
        "data/manifests/community_forensics_val_external_exact_seen_generator.csv",
        "External balanced test; exact generator identities are present in train-v3, while image identities remain disjoint.",
    ),
    (
        "hard_hourglass_exact_seen",
        "Hard Hourglass",
        "data/manifests/community_forensics_val_hard_hourglass_v2_exact_seen.csv",
        "Balanced hard slice; Hourglass exact identity is train-v3-seen and evaluation images are disjoint.",
    ),
    (
        "hard_dfgan_exact_seen",
        "Hard DFGAN",
        "data/manifests/community_forensics_val_hard_dfgan_v2_exact_seen.csv",
        "Balanced hard slice; DFGAN exact identity is train-v3-seen and evaluation images are disjoint.",
    ),
    (
        "hard_galip_exact_seen",
        "Hard GALIP",
        "data/manifests/community_forensics_val_hard_galip_v2_exact_seen.csv",
        "Balanced hard slice; GALIP exact identity is train-v3-seen and evaluation images are disjoint.",
    ),
    (
        "unseen_generator_expanded",
        "Full unseen-generator (4,000)",
        "data/manifests/community_forensics_test_external_unseen_generator_v3_expanded.csv",
        "External strict-unseen test; 12 exact generators and their Commercial/Other architecture families are absent from train-v3.",
    ),
)


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _configure_v3_protocol() -> None:
    report.SPLITS = V3_SPLITS
    report.SPLIT_TITLES = {key: title for key, title, _, _ in V3_SPLITS}
    report.REPORT_TITLE = "Community Forensics train-v3 B0/B1/B2/M2/M3 独立评测报告"
    report.REPORT_DESCRIPTION = (
        "train-v3 五个冻结模型在完整4,000张 strict unseen、external exact-seen及Hourglass/DFGAN/GALIP三个困难切片上的21条件技术评测。"
    )
    report.SCOPE_BODY = (
        "Full unseen-generator 使用全部4,000张扩展外部测试图片，包含2,000 Real与2,000 AIGI，"
        "12个精确生成器及其Commercial/Other大类均未进入train-v3。External exact-seen使用训练已见精确生成器的外部不重叠图片；"
        "Hourglass、DFGAN、GALIP作为三个困难生成器切片单列。所有切片均为Real/AIGI平衡；"
        "AUROC/AP与阈值无关，Accuracy、Precision、Recall、Specificity、F1、MCC与BA使用内部clean validation冻结阈值。"
    )


def _table_header(columns: list[str]) -> list[str]:
    return ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]


def _identity_values(rows: list[dict[str, str]], field: str) -> set[str]:
    return {row[field] for row in rows if row.get(field)}


def _dataset_integrity() -> dict[str, Any]:
    train_rows = report._read_csv("data/manifests/community_forensics_train_v3.csv")
    train_aigi = [row for row in train_rows if int(row["label"]) == 1]
    train_generators = _identity_values(train_aigi, "canonical_generator_id")
    train_architectures = _identity_values(train_aigi, "architecture")
    train_sample_ids = _identity_values(train_rows, "sample_id")
    train_sha256 = _identity_values(train_rows, "sha256")
    expected_exposure = {
        "exact_seen_generator": {"compvis/stable-diffusion-v1-4"},
        "hard_hourglass_exact_seen": {"hourglass"},
        "hard_dfgan_exact_seen": {"dfgan"},
        "hard_galip_exact_seen": {"galip"},
    }
    slices: dict[str, Any] = {}
    for split_key, _, manifest, _ in V3_SPLITS:
        rows = report._read_csv(manifest)
        aigi = [row for row in rows if int(row["label"]) == 1]
        generators = _identity_values(aigi, "canonical_generator_id")
        architectures = _identity_values(aigi, "architecture")
        expected = expected_exposure.get(split_key)
        slices[split_key] = {
            "rows": len(rows),
            "real": sum(int(row["label"]) == 0 for row in rows),
            "aigi": len(aigi),
            "exact_generators": sorted(generators),
            "architectures": sorted(architectures),
            "sample_id_overlap_with_train_v3": len(_identity_values(rows, "sample_id") & train_sample_ids),
            "sha256_overlap_with_train_v3": len(_identity_values(rows, "sha256") & train_sha256),
            "exact_generator_overlap_with_train_v3": sorted(generators & train_generators),
            "architecture_overlap_with_train_v3": sorted(architectures & train_architectures),
            "expected_exact_seen_generators_present": expected is None or expected <= train_generators,
        }
    unseen = slices["unseen_generator_expanded"]
    if unseen["rows"] != 4000 or unseen["real"] != 2000 or unseen["aigi"] != 2000:
        raise RuntimeError("full unseen-generator manifest is not 4,000 rows balanced 2,000/2,000")
    if unseen["exact_generator_overlap_with_train_v3"] or unseen["architecture_overlap_with_train_v3"]:
        raise RuntimeError("strict unseen-generator exposure overlaps train-v3")
    for split_key, expected in expected_exposure.items():
        if not expected <= set(slices[split_key]["exact_generator_overlap_with_train_v3"]):
            raise RuntimeError(f"expected exact-seen generator is absent from train-v3: {split_key}")
    for split_key, item in slices.items():
        if item["sample_id_overlap_with_train_v3"] or item["sha256_overlap_with_train_v3"]:
            raise RuntimeError(f"train/evaluation image identity overlap: {split_key}")
    return {
        "train_v3_rows": len(train_rows),
        "train_v3_exact_generator_count": len(train_generators),
        "train_v3_architectures": sorted(train_architectures),
        "slices": slices,
    }


def _customize_v3_artifact(artifact: dict[str, Any], audit: dict[str, Any]) -> None:
    datasets = artifact["snapshot"]["datasets"]
    split_rows = [
        row for row in datasets["split_summary"]
        if row["split"] == "Full unseen-generator (4,000)"
    ]
    if len(split_rows) != len(report.MODELS):
        raise RuntimeError("full unseen split summary must contain exactly five model rows")
    best_auroc = max(split_rows, key=lambda row: row["clean_auroc"])
    best_accuracy = max(split_rows, key=lambda row: row["clean_accuracy"])
    best_six = max(split_rows, key=lambda row: row["six_stage_auroc"])
    datasets["full_unseen_headline"] = [{
        "best_clean_auroc_model": best_auroc["model"],
        "best_clean_auroc": best_auroc["clean_auroc"],
        "best_clean_accuracy_model": best_accuracy["model"],
        "best_clean_accuracy": best_accuracy["clean_accuracy"],
        "best_six_stage_model": best_six["model"],
        "best_six_stage_auroc": best_six["six_stage_auroc"],
        "sample_count": 4000,
    }]
    datasets["full_unseen_clean_chart"] = [{
        "model": row["model"],
        "model_order": row["model_order"],
        "auroc": row["clean_auroc"],
        "accuracy": row["clean_accuracy"],
        "average_precision": row["clean_average_precision"],
        "aigi_recall": row["clean_recall"],
        "real_specificity": row["clean_specificity"],
    } for row in split_rows]
    datasets["full_unseen_multistage_chart"] = [
        {
            "model": row["model"],
            "model_order": row["model_order"],
            "condition": label,
            "condition_order": order,
            "auroc": row[field],
        }
        for row in split_rows
        for order, (label, field) in enumerate((
            ("4-stage A", "four_stage_a_auroc"),
            ("4-stage B", "four_stage_b_auroc"),
            ("6-stage", "six_stage_auroc"),
        ))
    ]

    generated_at = artifact["snapshot"]["generatedAt"]
    new_sources = [
        {
            "id": "full_unseen_headline_sql",
            "label": "Full unseen 4,000 headline snapshot",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": "SELECT * FROM full_unseen_headline",
                "description": "Best full-unseen clean and six-stage metrics selected from the reviewed five-model split summary.",
                "executed_at": generated_at,
                "tables_used": ["full_unseen_headline"],
            },
        },
        {
            "id": "full_unseen_clean_chart_sql",
            "label": "Full unseen 4,000 clean model snapshot",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": "SELECT * FROM full_unseen_clean_chart ORDER BY model_order",
                "description": "Clean metrics for all five frozen models on the complete 4,000-image strict-unseen test.",
                "executed_at": generated_at,
                "tables_used": ["full_unseen_clean_chart"],
            },
        },
        {
            "id": "full_unseen_multistage_chart_sql",
            "label": "Full unseen 4,000 multistage model snapshot",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": "SELECT * FROM full_unseen_multistage_chart ORDER BY condition_order, model_order",
                "description": "Four-stage A, four-stage B and six-stage AUROC for all five frozen models on full strict-unseen.",
                "executed_at": generated_at,
                "tables_used": ["full_unseen_multistage_chart"],
            },
        },
    ]
    artifact["sources"].extend(new_sources)
    artifact["manifest"]["sources"].extend(
        {"id": source["id"], "label": source["label"]} for source in new_sources
    )
    artifact["manifest"]["cards"] = [
        {
            "id": "full_unseen_clean_auroc_card",
            "description": f"完整4,000张 strict unseen；最佳模型 {best_auroc['model']}。",
            "dataset": "full_unseen_headline",
            "sourceId": "full_unseen_headline_sql",
            "metrics": [{"label": "最佳 Clean AUROC", "field": "best_clean_auroc", "format": "number"}],
        },
        {
            "id": "full_unseen_clean_accuracy_card",
            "description": f"内部验证冻结阈值；最佳模型 {best_accuracy['model']}。",
            "dataset": "full_unseen_headline",
            "sourceId": "full_unseen_headline_sql",
            "metrics": [{"label": "最佳 Clean Accuracy", "field": "best_clean_accuracy", "format": "percent"}],
        },
        {
            "id": "full_unseen_six_stage_card",
            "description": f"完整4,000张 strict unseen；最佳模型 {best_six['model']}。",
            "dataset": "full_unseen_headline",
            "sourceId": "full_unseen_headline_sql",
            "metrics": [{"label": "最佳 6-stage AUROC", "field": "best_six_stage_auroc", "format": "number"}],
        },
    ]
    artifact["manifest"]["charts"] = [
        {
            "id": "full_unseen_clean_auroc_chart",
            "title": "完整 strict unseen 的 Clean AUROC",
            "subtitle": "4,000张平衡测试图片（2,000 Real / 2,000 AIGI），12个精确生成器；五个冻结模型。",
            "type": "bar",
            "intent": "comparison",
            "question": "五个train-v3模型在完整4,000张strict unseen上的Clean AUROC如何比较？",
            "rationale": "五个离散模型的单一同单位指标适合使用柱状图进行直接比较。",
            "comparisonContext": {"unit": "AUROC", "grain": "model", "population": "full 4,000-image strict unseen"},
            "dataset": "full_unseen_clean_chart",
            "sourceId": "full_unseen_clean_chart_sql",
            "encodings": {
                "x": {"field": "model", "type": "nominal", "label": "Model"},
                "y": {"field": "auroc", "type": "quantitative", "label": "Clean AUROC", "format": "number"},
                "tooltip": [
                    {"field": "accuracy", "type": "quantitative", "label": "Accuracy", "format": "percent"},
                    {"field": "average_precision", "type": "quantitative", "label": "Average precision", "format": "number"},
                    {"field": "aigi_recall", "type": "quantitative", "label": "AIGI recall", "format": "percent"},
                    {"field": "real_specificity", "type": "quantitative", "label": "Real specificity", "format": "percent"},
                ],
            },
            "palette": {"kind": "sequential"},
            "labels": {"values": "all"},
            "valueFormat": "number",
            "layout": "full",
        },
        {
            "id": "full_unseen_multistage_auroc_chart",
            "title": "完整 strict unseen 的三组多阶段 AUROC",
            "subtitle": "同一4,000张测试集；两组固定四阶段链与一组逐样本确定性随机六阶段链。",
            "type": "bar",
            "intent": "comparison",
            "question": "五个train-v3模型在三组多阶段strict-unseen条件下的AUROC如何比较？",
            "rationale": "五个模型与三个同单位扰动条件适合使用分组柱状图。",
            "comparisonContext": {"unit": "AUROC", "grain": "model by multistage condition", "population": "full 4,000-image strict unseen"},
            "dataset": "full_unseen_multistage_chart",
            "sourceId": "full_unseen_multistage_chart_sql",
            "encodings": {
                "x": {"field": "model", "type": "nominal", "label": "Model"},
                "y": {"field": "auroc", "type": "quantitative", "label": "AUROC", "format": "number"},
                "color": {"field": "condition", "type": "nominal", "label": "Condition"},
                "tooltip": [{"field": "auroc", "type": "quantitative", "label": "AUROC", "format": "number"}],
            },
            "palette": {"kind": "categorical"},
            "legend": {"position": "bottom", "interactive": True},
            "labels": {"values": "auto"},
            "valueFormat": "number",
            "layout": "full",
        },
        *artifact["manifest"]["charts"],
    ]
    blocks = artifact["manifest"]["blocks"]
    for block in blocks:
        if block["id"] == "technical_summary":
            block["sourceId"] = "full_unseen_headline_sql"
            block["body"] = (
                "## 技术摘要\n\n"
                f"- **完整4,000张 strict unseen 的 Clean AUROC 最高为 {best_auroc['model']}：{best_auroc['clean_auroc']:.4f}。**\n"
                f"- **冻结阈值 Clean Accuracy 最高为 {best_accuracy['model']}：{100 * best_accuracy['clean_accuracy']:.2f}%。**\n"
                f"- **六阶段共同扰动 AUROC 最高为 {best_six['model']}：{best_six['six_stage_auroc']:.4f}。**\n"
                "- External exact-seen与Hourglass/DFGAN/GALIP用于暴露关系和失败模式诊断；五切片等权宏平均不是主测试集加权总体。\n\n"
                "所有结论均为冻结协议下的描述性点估计；没有bootstrap区间或重复训练种子，不能表述为统计显著。"
            )
        elif block["id"] == "headline_cards":
            block["cardIds"] = [
                "full_unseen_clean_auroc_card",
                "full_unseen_clean_accuracy_card",
                "full_unseen_six_stage_card",
            ]
        elif block["id"] == "macro_finding":
            block["body"] = (
                "## 困难切片使五切片等权宏平均与主 unseen 排名不同\n\n"
                "五切片等权宏平均把每个500张困难切片与4,000张full unseen赋予相同权重，用于诊断跨切片稳定性而非估计总体准确率。"
                f"该诊断口径下新增三组多阶段均值由 **{audit['headline']['best_new_model']}** 领先；"
                "必须与前述full unseen主结果及后续分切片表联合解释。"
            )
    insertion = next(index for index, block in enumerate(blocks) if block["id"] == "headline_cards") + 1
    blocks[insertion:insertion] = [
        {
            "id": "full_unseen_clean_finding",
            "type": "markdown",
            "sourceId": "full_unseen_clean_chart_sql",
            "body": (
                "## M2与M3在完整strict unseen的Clean排序上明显领先\n\n"
                f"M2与M3的Clean AUROC分别为 {next(row['clean_auroc'] for row in split_rows if row['model']=='M2'):.4f} 和 "
                f"{next(row['clean_auroc'] for row in split_rows if row['model']=='M3'):.4f}；"
                "图表同时保留Accuracy、AP、Recall与Specificity作为悬浮信息，精确值见分切片明细表。"
            ),
        },
        {"id": "full_unseen_clean_chart_block", "type": "chart", "chartId": "full_unseen_clean_auroc_chart", "layout": "full"},
        {
            "id": "full_unseen_multistage_finding",
            "type": "markdown",
            "sourceId": "full_unseen_multistage_chart_sql",
            "body": (
                "## 六阶段组合是完整strict unseen上最严的新增压力条件\n\n"
                f"六阶段条件的最佳AUROC为 **{best_six['model']}：{best_six['six_stage_auroc']:.4f}**。"
                "两组四阶段与六阶段均在同一4,000张样本上计算，因此模型间与条件间点估计可直接对照；"
                "尚未计算配对置信区间。"
            ),
        },
        {"id": "full_unseen_multistage_chart_block", "type": "chart", "chartId": "full_unseen_multistage_auroc_chart", "layout": "full"},
    ]
    audit["report_contract"]["question"] = (
        "Evaluate train-v3 B0/B1/B2/M2/M3 primarily on the full 4,000-image strict-unseen test, with exact-seen and three hard generators as diagnostic slices."
    )
    audit["report_contract"]["required_section_mapping"]["key_findings_with_visual_evidence"] = [
        "full_unseen_clean_finding",
        "full_unseen_multistage_finding",
        "macro_finding",
        "strict_six_finding",
        "worst_case_finding",
    ]
    audit["chart_map"] = [
        {
            "section": "full_unseen_clean_finding",
            "question": "Compare clean AUROC across five models on the full 4,000-image strict-unseen test.",
            "family": "comparison",
            "type": "bar",
            "fields": ["model", "auroc"],
            "palette": "single-root sequential",
        },
        {
            "section": "full_unseen_multistage_finding",
            "question": "Compare 4-stage A, 4-stage B and 6-stage AUROC across five models on full strict-unseen.",
            "family": "comparison",
            "type": "grouped bar",
            "fields": ["model", "condition", "auroc"],
            "palette": "categorical, three perturbation conditions",
        },
        *audit["chart_map"],
    ]


def _build_markdown(artifact: dict[str, Any], audit: dict[str, Any]) -> str:
    datasets = artifact["snapshot"]["datasets"]
    headline = datasets["headline_metrics"][0]
    split_definitions = datasets["split_definitions"]
    split_summary = datasets["split_summary"]
    macro = datasets["macro_detail"]
    all_metrics = datasets["all_metrics"]
    unseen_title = "Full unseen-generator (4,000)"
    hard_titles = {"Hard Hourglass", "Hard DFGAN", "Hard GALIP"}

    unseen_rows = [row for row in split_summary if row["split"] == unseen_title]
    exact_rows = [row for row in split_summary if row["split"] == "External exact-seen generator"]
    hard_rows = [row for row in split_summary if row["split"] in hard_titles]
    best_unseen_auc = max(unseen_rows, key=lambda row: row["clean_auroc"])
    best_unseen_acc = max(unseen_rows, key=lambda row: row["clean_accuracy"])
    hard_macro = []
    for model in report.MODELS:
        display = model.upper()
        rows = [row for row in hard_rows if row["model"] == display]
        hard_macro.append({
            "model": display,
            "clean_auroc": fmean(row["clean_auroc"] for row in rows),
            "six_auroc": fmean(row["six_stage_auroc"] for row in rows),
        })
    best_hard = max(hard_macro, key=lambda row: row["clean_auroc"])

    total_images_per_model = sum(int(row["total"]) for row in split_definitions)
    prediction_rows = total_images_per_model * 21 * len(report.MODELS)
    unseen_manifest = report._read_csv(V3_SPLITS[-1][2])
    unseen_generators = sorted({row["canonical_generator_id"] for row in unseen_manifest if int(row["label"]) == 1})
    real_sources: dict[str, int] = {}
    for row in unseen_manifest:
        if int(row["label"]) == 0:
            real_sources[row["real_source"]] = real_sources.get(row["real_source"], 0) + 1

    lines = [
        "# Community Forensics train-v3 B0/B1/B2/M2/M3 独立评测报告",
        "",
        f"> 生成时间：{audit['generated_at_asia_singapore']}  ",
        f"> 评测作业：`{', '.join(sorted({str(item['evaluation_job_id']) for model in audit['split_artifacts'].values() for item in model.values()}))}`  ",
        f"> 扰动矩阵 SHA256：`{audit['matrix_sha256']}`",
        "",
        "## 结论摘要",
        "",
        f"- 完整4,000张 unseen 上，Clean AUROC最高为 **{best_unseen_auc['model']}：{best_unseen_auc['clean_auroc']:.4f}**。",
        f"- 完整4,000张 unseen 上，冻结阈值 Clean Accuracy最高为 **{best_unseen_acc['model']}：{_pct(best_unseen_acc['clean_accuracy'])}**。",
        f"- 三个困难生成器的 Clean 宏平均 AUROC最高为 **{best_hard['model']}：{best_hard['clean_auroc']:.4f}**；其六阶段宏平均AUROC为 {best_hard['six_auroc']:.4f}。",
        f"- 五切片 Clean 等权宏平均最高为 **{headline['best_clean_model']}：{headline['best_clean_auroc']:.4f}**；新增三组多阶段宏平均最高为 **{headline['best_new_model']}：{headline['best_new_auroc']:.4f}**。",
        "- 本报告只描述train-v3，不使用v2结果，也未用external/hard测试标签重新选择checkpoint或阈值。",
        "",
        "## 评测范围与数据角色",
        "",
    ]
    lines.extend(_table_header(["切片", "角色", "总数", "Real", "AIGI", "Manifest SHA256"]))
    for row in split_definitions:
        role = "strict unseen test" if row["split"] == unseen_title else ("exact-seen external test" if row["split"].startswith("External") else "hard-generator test")
        lines.append(f"| {row['split']} | {role} | {row['total']} | {row['real']} | {row['aigi']} | `{row['manifest_sha256']}` |")
    lines.extend([
        "",
        f"五个切片每模型共 {total_images_per_model:,} 张图像、21个条件；五模型共 {prediction_rows:,} 条逐样本预测。"
        f" Full unseen包含 {len(unseen_generators)} 个精确生成器：`{', '.join(unseen_generators)}`。"
        f" 真实来源为 {', '.join(f'{name}={count}' for name, count in sorted(real_sources.items()))}。",
        "",
        "## Full unseen-generator（4,000张）详细Clean指标",
        "",
    ])
    lines.extend(_table_header(["模型", "Accuracy", "Precision", "Recall", "Specificity", "F1", "MCC", "BA", "AUROC", "AP", "TPR@1%FPR", "TPR@5%FPR"]));
    clean_by_model_condition = {
        row["model"]: row for row in all_metrics if row["split"] == unseen_title and row["condition_index"] == 0
    }
    for model in (item.upper() for item in report.MODELS):
        row = clean_by_model_condition[model]
        lines.append(
            f"| {model} | {_pct(row['accuracy'])} | {_pct(row['precision'])} | {_pct(row['aigc_recall'])} | "
            f"{_pct(row['real_specificity'])} | {_pct(row['f1'])} | {row['mcc']:.4f} | {row['balanced_accuracy']:.4f} | "
            f"{row['auroc']:.4f} | {row['average_precision']:.4f} | {_pct(row['tpr_at_fpr_1pct'])} | {_pct(row['tpr_at_fpr_5pct'])} |"
        )
    lines.extend([
        "",
        "Accuracy、Precision与NPV基于人为50% AIGI测试先验，部署先验变化时不能直接外推。AUROC/AP用于排序，固定阈值指标用于当前内部验证阈值的操作点，两类指标必须联合解释。",
        "",
        "## Full unseen-generator 多阶段扰动",
        "",
    ])
    lines.extend(_table_header(["模型", "4-stage A AUROC", "4-stage B AUROC", "6-stage AUROC", "6-stage Accuracy", "6-stage Recall", "6-stage Specificity", "6-stage F1"]));
    for row in unseen_rows:
        lines.append(
            f"| {row['model']} | {row['four_stage_a_auroc']:.4f} | {row['four_stage_b_auroc']:.4f} | {row['six_stage_auroc']:.4f} | "
            f"{_pct(row['six_stage_accuracy'])} | {_pct(row['six_stage_recall'])} | {_pct(row['six_stage_specificity'])} | {_pct(row['six_stage_f1'])} |"
        )
    lines.extend([
        "",
        "## External exact-seen Clean指标",
        "",
    ])
    lines.extend(_table_header(["模型", "Accuracy", "Recall", "Specificity", "F1", "AUROC", "AP", "6-stage AUROC"]));
    for row in exact_rows:
        lines.append(f"| {row['model']} | {_pct(row['clean_accuracy'])} | {_pct(row['clean_recall'])} | {_pct(row['clean_specificity'])} | {_pct(row['clean_f1'])} | {row['clean_auroc']:.4f} | {row['clean_average_precision']:.4f} | {row['six_stage_auroc']:.4f} |")
    lines.extend([
        "",
        "## 三个困难生成器Clean指标",
        "",
    ])
    lines.extend(_table_header(["切片", "模型", "Accuracy", "Recall", "Specificity", "F1", "AUROC", "6-stage AUROC"]));
    for row in hard_rows:
        lines.append(f"| {row['split']} | {row['model']} | {_pct(row['clean_accuracy'])} | {_pct(row['clean_recall'])} | {_pct(row['clean_specificity'])} | {_pct(row['clean_f1'])} | {row['clean_auroc']:.4f} | {row['six_stage_auroc']:.4f} |")
    lines.extend([
        "",
        "三个困难切片共享真实负类面板，因此这些AUROC是相关的切片诊断，不能当作三个统计独立总体。每个困难切片同时包含Real与AIGI，因而可报告完整二分类指标；如果后续只抽取单一生成器正类，则只能解释Recall/TP/FN。",
        "",
        "## 五切片宏平均",
        "",
    ])
    lines.extend(_table_header(["模型", "Clean AUROC", "17扰动均值", "4-stage A", "4-stage B", "6-stage", "新增3组均值", "新增3组最坏"]));
    for row in macro:
        lines.append(f"| {row['model']} | {row['clean_auroc']:.4f} | {row['legacy_17_mean_auroc']:.4f} | {row['four_stage_a_auroc']:.4f} | {row['four_stage_b_auroc']:.4f} | {row['six_stage_auroc']:.4f} | {row['new_3_mean_auroc']:.4f} | {row['new_3_worst_auroc']:.4f} |")

    worst = sorted((row for row in all_metrics if row["condition_index"] > 0), key=lambda row: row["auroc"])[:10]
    lines.extend([
        "",
        "## 最低AUROC条件",
        "",
    ])
    lines.extend(_table_header(["模型", "切片", "条件", "AUROC", "Accuracy", "Recall", "Specificity"]));
    for row in worst:
        lines.append(f"| {row['model']} | {row['split']} | {row['condition']} | {row['auroc']:.4f} | {_pct(row['accuracy'])} | {_pct(row['aigc_recall'])} | {_pct(row['real_specificity'])} |")
    lines.extend([
        "",
        "## 方法与完整性",
        "",
        "- 逐模型核对冻结best checkpoint、resolved config、训练完成标记和内部clean validation阈值。",
        "- 逐切片核对COMPLETE、manifest SHA256、21条件matrix SHA256、每条件样本数与run card checkpoint SHA256。",
        "- Clean、17个既有扰动、两组4-stage和一组确定性随机6-stage均来自同一冻结矩阵。",
        "- 报告不重新训练、不重新推理、不改变阈值，也不使用test/hard标签做模型选择。",
        "",
        "## 局限性与下一步",
        "",
        "1. 当前只有单训练种子且没有模型差异的配对置信区间；小差异不应表述为统计显著。",
        "2. Full unseen只覆盖12个精确生成器及四类真实来源，不能代表所有未来生成器和真实流量。",
        "3. 固定阈值接近1的模型仍存在校准风险；部署前应在独立calibration set上按FPR约束重新确定操作点。",
        "4. 建议针对最差困难生成器、最低真实来源Specificity以及六阶段条件执行样本配对和生成器分层bootstrap。",
        "5. 使用多个预注册扰动种子和贴近部署流行率的流量回放，重新报告Precision、NPV与成本加权指标。",
        "",
        "## 结构化产物",
        "",
        "- `reports/evaluations/community_forensics_v3_evaluation/community_forensics_v3_evaluation_metrics.csv`",
        "- `reports/evaluations/community_forensics_v3_evaluation/community_forensics_v3_evaluation_artifact.json`",
        "- `reports/evaluations/community_forensics_v3_evaluation/community_forensics_v3_evaluation_audit.json`",
        "",
    ])
    return "\n".join(lines)


def generate(arguments: argparse.Namespace) -> None:
    _configure_v3_protocol()
    report.generate(arguments)
    artifact = report._read_json(arguments.artifact_json)
    audit = report._read_json(arguments.audit_json)
    _customize_v3_artifact(artifact, audit)
    audit["protocol_id"] = "community_forensics_train_v3_full_unseen_exact_seen_hard3_21_conditions"
    audit["markdown_report"] = str(arguments.markdown)
    audit["full_unseen_manifest"] = V3_SPLITS[-1][2]
    audit["full_unseen_manifest_sha256"] = report._sha256(V3_SPLITS[-1][2])
    audit["dataset_integrity"] = _dataset_integrity()
    report._atomic_text(arguments.artifact_json, json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    report._atomic_text(arguments.audit_json, json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    report._atomic_text(arguments.markdown, _build_markdown(artifact, audit) + "\n")
    print(json.dumps({"event": "community_forensics_v3_evaluation_markdown_complete", "markdown": str(arguments.markdown)}, sort_keys=True), flush=True)


def _parse_args() -> argparse.Namespace:
    root = "reports/evaluations/community_forensics_v3_evaluation"
    parser = argparse.ArgumentParser(description="Generate the independent train-v3 report for full unseen, exact-seen and three hard generators")
    parser.add_argument("--evaluation-root", default="outputs/community_forensics_v3_robustness_v2")
    parser.add_argument("--source-root", default="outputs/community_forensics_v3")
    parser.add_argument("--matrix", default="configs/community_forensics_robustness_v2.yaml")
    parser.add_argument("--metrics-csv", default=f"{root}/community_forensics_v3_evaluation_metrics.csv")
    parser.add_argument("--artifact-json", default=f"{root}/community_forensics_v3_evaluation_artifact.json")
    parser.add_argument("--audit-json", default=f"{root}/community_forensics_v3_evaluation_audit.json")
    parser.add_argument("--markdown", default="reports/summaries/COMMUNITY_FORENSICS_V3_EVALUATION_REPORT.md")
    return parser.parse_args()


if __name__ == "__main__":
    generate(_parse_args())
