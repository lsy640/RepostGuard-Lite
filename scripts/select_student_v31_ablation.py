from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def _manifest_groups(path: str) -> dict[str, dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in groups:
            raise ValueError(f"Duplicate manifest sample_id: {sample_id}")
        label = int(row["label"])
        architecture = "Real" if label == 0 else str(row.get("architecture", "")).strip()
        generator = str(
            row.get("canonical_generator_id") or row.get("generator_id") or ""
        ).strip()
        if label == 1 and architecture not in {"LatDiff", "GAN", "PixDiff"}:
            raise ValueError(f"Unknown AIGI architecture for {sample_id}: {architecture!r}")
        groups[sample_id] = {
            "label": str(label),
            "architecture": architecture,
            "generator": generator,
        }
    return groups


def _group_metrics(
    predictions_path: Path,
    manifest_groups: dict[str, dict[str, str]],
) -> dict[str, object]:
    by_transform: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            prediction = json.loads(line)
            sample_id = str(prediction["sample_id"])
            if sample_id not in manifest_groups:
                raise KeyError(f"Prediction sample is absent from manifest: {sample_id}")
            metadata = manifest_groups[sample_id]
            if int(prediction["label"]) != int(metadata["label"]):
                raise ValueError(f"Prediction label mismatch for {sample_id}")
            by_transform[str(prediction["transform"])].append(
                {
                    "sample_id": sample_id,
                    "label": int(prediction["label"]),
                    "pred": float(prediction["pred"]),
                    "architecture": metadata["architecture"],
                    "generator": metadata["generator"],
                }
            )
    if "clean" not in by_transform:
        raise ValueError(f"Predictions have no clean transform: {predictions_path}")

    architectures = ("LatDiff", "GAN", "PixDiff")
    per_transform: dict[str, object] = {}
    architecture_macro: dict[str, float] = {}
    architecture_worst: dict[str, float] = {}
    hierarchical_generator_macro: dict[str, float] = {}
    for transform, rows in sorted(by_transform.items()):
        real = [row for row in rows if int(row["label"]) == 0]
        if not real:
            raise ValueError(f"Transform {transform} has no Real comparison pool")
        architecture_auc: dict[str, float] = {}
        generator_auc_by_architecture: dict[str, dict[str, float]] = {}
        for architecture in architectures:
            aigi = [
                row
                for row in rows
                if int(row["label"]) == 1 and row["architecture"] == architecture
            ]
            if not aigi:
                raise ValueError(f"Transform {transform} has no {architecture} samples")
            comparison = real + aigi
            architecture_auc[architecture] = float(
                roc_auc_score(
                    np.asarray([row["label"] for row in comparison]),
                    np.asarray([row["pred"] for row in comparison]),
                )
            )
            generator_auc: dict[str, float] = {}
            for generator in sorted({str(row["generator"]) for row in aigi}):
                generator_rows = [
                    row for row in aigi if str(row["generator"]) == generator
                ]
                generator_comparison = real + generator_rows
                generator_auc[generator] = float(
                    roc_auc_score(
                        np.asarray([row["label"] for row in generator_comparison]),
                        np.asarray([row["pred"] for row in generator_comparison]),
                    )
                )
            generator_auc_by_architecture[architecture] = generator_auc

        architecture_macro[transform] = float(np.mean(list(architecture_auc.values())))
        architecture_worst[transform] = float(min(architecture_auc.values()))
        hierarchical_generator_macro[transform] = float(
            np.mean(
                [
                    np.mean(list(generator_auc_by_architecture[architecture].values()))
                    for architecture in architectures
                ]
            )
        )
        per_transform[transform] = {
            "architecture_auroc": architecture_auc,
            "architecture_macro_auroc": architecture_macro[transform],
            "architecture_worst_auroc": architecture_worst[transform],
            "hierarchical_generator_macro_auroc": hierarchical_generator_macro[
                transform
            ],
            "generator_auroc_by_architecture": generator_auc_by_architecture,
        }

    robust_transforms = [name for name in by_transform if name != "clean"]
    return {
        "clean_architecture_macro_auroc": architecture_macro["clean"],
        "clean_architecture_worst_auroc": architecture_worst["clean"],
        "clean_hierarchical_generator_macro_auroc": hierarchical_generator_macro[
            "clean"
        ],
        "robust_architecture_macro_auroc": float(
            np.mean([architecture_macro[name] for name in robust_transforms])
        ),
        "robust_worst_architecture_condition_auroc": float(
            min(
                float(per_transform[name]["architecture_worst_auroc"])
                for name in robust_transforms
            )
        ),
        "robust_hierarchical_generator_macro_auroc": float(
            np.mean([hierarchical_generator_macro[name] for name in robust_transforms])
        ),
        "per_transform_group_metrics": per_transform,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a Student v3.1 candidate using only family-unseen dev metrics"
    )
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument(
        "--output",
        default="outputs/community_forensics_v31/ablation_selection.json",
    )
    parser.add_argument("--maximum-clean-regression", type=float, default=0.01)
    parser.add_argument(
        "--maximum-architecture-clean-regression", type=float, default=0.01
    )
    parser.add_argument(
        "--manifest",
        default="data/manifests/community_forensics_val_family_unseen_dev_v1.csv",
    )
    args = parser.parse_args()
    manifest_groups = _manifest_groups(args.manifest)
    candidates = []
    for specification in args.candidate:
        name, config_path, output_directory = specification.split(":", 2)
        root = Path(output_directory)
        summary_path = root / "dev_family_unseen_robustness" / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        group_metrics = _group_metrics(
            root / "dev_family_unseen_robustness" / "predictions.jsonl",
            manifest_groups,
        )
        candidate = {
                "name": name,
                "config": config_path,
                "output_directory": output_directory,
                "checkpoint": str(root / "best.pt"),
                "clean_auroc": float(summary["clean_auroc"]),
                "robust_mean_auroc": float(summary["robust_mean_auroc"]),
                "robust_worst_auroc": float(summary["robust_worst_auroc"]),
            }
        candidate.update(group_metrics)
        candidates.append(candidate)
    best_clean = max(candidate["clean_auroc"] for candidate in candidates)
    best_architecture_clean = max(
        candidate["clean_architecture_macro_auroc"] for candidate in candidates
    )
    eligible = [
        candidate
        for candidate in candidates
        if best_clean - candidate["clean_auroc"] <= args.maximum_clean_regression
        and best_architecture_clean
        - candidate["clean_architecture_macro_auroc"]
        <= args.maximum_architecture_clean_regression
    ]
    winner = max(
        eligible,
        key=lambda candidate: (
            candidate["robust_architecture_macro_auroc"],
            candidate["robust_worst_architecture_condition_auroc"],
            candidate["robust_hierarchical_generator_macro_auroc"],
            candidate["robust_mean_auroc"],
            candidate["clean_auroc"],
        ),
    )
    result = {
        "event": "student_v31_ablation_selection",
        "selection_data": "family_unseen_dev_only",
        "protected_external_test_used": False,
        "maximum_clean_regression": args.maximum_clean_regression,
        "maximum_architecture_clean_regression": args.maximum_architecture_clean_regression,
        "selection_primary": "robust_architecture_macro_auroc",
        "selection_secondary": "robust_worst_architecture_condition_auroc",
        "candidates": candidates,
        "winner": winner,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
