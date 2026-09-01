from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from repostguard.metrics import binary_metrics, select_balanced_threshold


SLICE_GENERATORS = {
    "sd14_exact_seen": "compvis/stable-diffusion-v1-4",
    "dfgan_hard": "dfgan",
    "galip_hard": "galip",
    "hourglass_hard": "hourglass",
}
FROZEN_THRESHOLD = 0.060516357421875


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize V3.2.1 on the fixed 1500-image validation set"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_predictions(path: Path) -> dict[str, float]:
    predictions: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("transform") != "clean":
                continue
            sample_id = str(row["sample_id"])
            if sample_id in predictions:
                raise ValueError(f"Duplicate clean prediction for {sample_id}")
            predictions[sample_id] = float(row["pred"])
    return predictions


def metric_bundle(
    rows: list[dict[str, str]],
    predictions: dict[str, float],
    diagnostic_threshold: float,
) -> dict[str, object]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [predictions[row["sample_id"]] for row in rows], dtype=np.float64
    )
    return {
        "n": len(rows),
        "real": int((labels == 0).sum()),
        "aigi": int((labels == 1).sum()),
        "frozen_threshold": binary_metrics(labels, probabilities, FROZEN_THRESHOLD),
        "diagnostic_fixed1500_threshold": binary_metrics(
            labels, probabilities, diagnostic_threshold
        ),
    }


def main() -> None:
    arguments = parse_arguments()
    rows = read_manifest(arguments.manifest)
    if len(rows) != 1500:
        raise ValueError(f"Expected 1500 validation rows, found {len(rows)}")
    sample_ids = {row["sample_id"] for row in rows}
    if len(sample_ids) != len(rows):
        raise ValueError("Validation manifest contains duplicate sample_id values")

    predictions = read_predictions(arguments.predictions)
    if set(predictions) != sample_ids:
        missing = sorted(sample_ids.difference(predictions))[:5]
        extra = sorted(set(predictions).difference(sample_ids))[:5]
        raise ValueError(f"Prediction coverage mismatch: missing={missing}, extra={extra}")

    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [predictions[row["sample_id"]] for row in rows], dtype=np.float64
    )
    diagnostic_threshold = select_balanced_threshold(labels, probabilities)
    real_rows = [row for row in rows if int(row["label"]) == 0]
    aigi_rows = [row for row in rows if int(row["label"]) == 1]
    if len(real_rows) != 750 or len(aigi_rows) != 750:
        raise ValueError("Fixed validation set must contain 750 Real and 750 AIGI rows")

    slices: dict[str, object] = {}
    for slice_name, generator_id in SLICE_GENERATORS.items():
        selected_aigi = [
            row
            for row in aigi_rows
            if row["generator_id"].strip().lower() == generator_id
        ]
        if not selected_aigi:
            raise ValueError(f"No AIGI rows found for slice {slice_name}")
        slices[slice_name] = metric_bundle(
            real_rows + selected_aigi, predictions, diagnostic_threshold
        )

    result = {
        "schema_version": 1,
        "model": "v3.2.1",
        "manifest": str(arguments.manifest),
        "predictions": str(arguments.predictions),
        "rows": len(rows),
        "frozen_threshold": FROZEN_THRESHOLD,
        "diagnostic_threshold": diagnostic_threshold,
        "slice_protocol": (
            "Each generator slice reuses the same 750-image Real reference pool; "
            "this makes AUROC and balanced accuracy defined for GALIP and Hourglass, "
            "whose manifest rows contain AIGI only."
        ),
        "threshold_policy": {
            "frozen_threshold": "Original V3.2.1 family-unseen deployment threshold.",
            "diagnostic_fixed1500_threshold": (
                "Selected once on the complete fixed 1500 validation set for diagnosis "
                "only; never used for protected external evaluation."
            ),
        },
        "overall": metric_bundle(rows, predictions, diagnostic_threshold),
        "slices": slices,
    }

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    json_path = arguments.output_directory / "fixed1500_v321_metrics.json"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fieldnames = [
        "scope",
        "threshold_policy",
        "n",
        "real",
        "aigi",
        "threshold",
        "auroc",
        "average_precision",
        "balanced_accuracy",
        "aigc_recall",
        "real_specificity",
        "brier",
        "ece_15",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    csv_path = arguments.output_directory / "fixed1500_v321_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        scopes = {"overall": result["overall"], **result["slices"]}
        for scope_name, scope_result in scopes.items():
            for threshold_policy in (
                "frozen_threshold",
                "diagnostic_fixed1500_threshold",
            ):
                metrics = scope_result[threshold_policy]
                writer.writerow(
                    {
                        "scope": scope_name,
                        "threshold_policy": threshold_policy,
                        "n": scope_result["n"],
                        "real": scope_result["real"],
                        "aigi": scope_result["aigi"],
                        **{name: metrics[name] for name in fieldnames if name in metrics},
                    }
                )

    markdown_path = arguments.output_directory / "FIXED1500_EVALUATION.md"
    lines = [
        "# V3.2.1 fixed-1500 evaluation",
        "",
        "AUROC and AP are threshold-independent. BA below uses the frozen V3.2.1 threshold.",
        "",
        "| Scope | AUROC | AP | BA | AIGI recall | Real specificity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scope_name, scope_result in {
        "overall": result["overall"],
        **result["slices"],
    }.items():
        metrics = scope_result["frozen_threshold"]
        lines.append(
            f"| {scope_name} | {metrics['auroc']:.6f} | "
            f"{metrics['average_precision']:.6f} | {metrics['balanced_accuracy']:.6f} | "
            f"{metrics['aigc_recall']:.6f} | {metrics['real_specificity']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Generator slices reuse the same 750-image Real reference pool.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"accepted": True, "output": str(json_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
