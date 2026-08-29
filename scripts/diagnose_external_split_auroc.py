from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


COHORTS = {
    "seen_family": "community_forensics_test_external_seen_family.csv",
    "unseen_generator": "community_forensics_test_external_unseen_generator.csv",
}
MODELS = ("b0", "b1", "b2", "m2", "m3")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def counter_dict(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def describe_rows(rows: list[dict[str, str]], train_architectures: set[str]) -> dict[str, Any]:
    real = [row for row in rows if int(row["label"]) == 0]
    aigi = [row for row in rows if int(row["label"]) == 1]

    def image_stats(items: list[dict[str, str]]) -> dict[str, Any]:
        pixels = [int(row["width"]) * int(row["height"]) for row in items]
        bytes_per_image = [int(row["byte_size"]) for row in items]
        resolutions = [f'{row["width"]}x{row["height"]}' for row in items]
        top_resolutions = Counter(resolutions).most_common(8)
        return {
            "n": len(items),
            "median_pixels": float(median(pixels)),
            "median_megapixels": float(median(pixels) / 1_000_000),
            "median_bytes": float(median(bytes_per_image)),
            "formats": counter_dict([row["format"] for row in items]),
            "top_resolutions": [
                {"resolution": resolution, "n": count}
                for resolution, count in top_resolutions
            ],
        }

    generators = sorted({row["canonical_generator_id"] for row in aigi})
    architectures = sorted({row["architecture"] for row in aigi})
    family_seen = [row for row in aigi if row["architecture"] in train_architectures]
    return {
        "label_counts": counter_dict([row["label"] for row in rows]),
        "generator_count": len(generators),
        "generators": generators,
        "generator_counts": counter_dict([row["canonical_generator_id"] for row in aigi]),
        "architecture_counts": counter_dict([row["architecture"] for row in aigi]),
        "architecture_count": len(architectures),
        "architectures": architectures,
        "aigi_images_with_train_seen_architecture": len(family_seen),
        "aigi_fraction_with_train_seen_architecture": len(family_seen) / len(aigi),
        "generator_exposure_counts": counter_dict([row["generator_exposure"] for row in aigi]),
        "real_source_counts": counter_dict([row["real_source"] for row in real]),
        "aigi_prompt_source_counts": counter_dict([row["real_source"] for row in aigi]),
        "real_image_stats": image_stats(real),
        "aigi_image_stats": image_stats(aigi),
    }


def subset_auc(real_scores: np.ndarray, aigi_scores: np.ndarray) -> float:
    labels = np.concatenate(
        [np.zeros(len(real_scores), dtype=np.int64), np.ones(len(aigi_scores), dtype=np.int64)]
    )
    scores = np.concatenate([real_scores, aigi_scores])
    return float(roc_auc_score(labels, scores))


def bootstrap_difference(
    seen_real: np.ndarray,
    seen_aigi: np.ndarray,
    unseen_real: np.ndarray,
    unseen_aigi: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        seen_auc = subset_auc(
            rng.choice(seen_real, len(seen_real), replace=True),
            rng.choice(seen_aigi, len(seen_aigi), replace=True),
        )
        unseen_auc = subset_auc(
            rng.choice(unseen_real, len(unseen_real), replace=True),
            rng.choice(unseen_aigi, len(unseen_aigi), replace=True),
        )
        differences[index] = unseen_auc - seen_auc
    return {
        "replicates": replicates,
        "unseen_minus_seen_mean": float(differences.mean()),
        "ci95_low": float(np.quantile(differences, 0.025)),
        "ci95_high": float(np.quantile(differences, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/community_forensics_external_split_diagnostic.json"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()

    root = args.project_root.resolve()
    manifest_dir = root / "data" / "manifests"
    train_rows = read_csv(manifest_dir / "community_forensics_train.csv")
    train_architectures = {
        row["architecture"] for row in train_rows if int(row["label"]) == 1
    }

    manifests: dict[str, list[dict[str, str]]] = {}
    manifest_by_id: dict[str, dict[str, dict[str, str]]] = {}
    composition: dict[str, Any] = {}
    for cohort, filename in COHORTS.items():
        rows = read_csv(manifest_dir / filename)
        manifests[cohort] = rows
        manifest_by_id[cohort] = {row["sample_id"]: row for row in rows}
        if len(manifest_by_id[cohort]) != len(rows):
            raise ValueError(f"Duplicate sample_id in {filename}")
        composition[cohort] = describe_rows(rows, train_architectures)

    model_metrics: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    generator_rows: list[dict[str, Any]] = []
    architecture_rows: list[dict[str, Any]] = []
    for model_index, model in enumerate(MODELS):
        model_metrics[model] = {}
        score_groups: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for cohort in COHORTS:
            prediction_path = (
                root
                / "outputs"
                / "community_forensics"
                / model
                / f"test_external_{cohort}"
                / "predictions.jsonl"
            )
            predictions = read_jsonl(prediction_path)
            if len(predictions) != len(manifests[cohort]):
                raise ValueError(f"Prediction count mismatch: {model} {cohort}")
            joined: list[tuple[dict[str, str], float]] = []
            seen_ids: set[str] = set()
            for prediction in predictions:
                sample_id = str(prediction["sample_id"])
                row = manifest_by_id[cohort].get(sample_id)
                if row is None:
                    raise ValueError(f"Unknown sample_id: {sample_id}")
                if int(row["label"]) != int(prediction["label"]):
                    raise ValueError(f"Label mismatch: {sample_id}")
                if sample_id in seen_ids:
                    raise ValueError(f"Duplicate prediction: {sample_id}")
                seen_ids.add(sample_id)
                joined.append((row, float(prediction["pred"])))

            real_scores = np.asarray([score for row, score in joined if int(row["label"]) == 0])
            aigi_scores = np.asarray([score for row, score in joined if int(row["label"]) == 1])
            score_groups[cohort] = (real_scores, aigi_scores)
            full_auc = subset_auc(real_scores, aigi_scores)
            summary_path = prediction_path.parent / "summary.json"
            summary_auc = float(json.loads(summary_path.read_text(encoding="utf-8"))["clean_auroc"])
            if abs(full_auc - summary_auc) > 1e-12:
                raise ValueError(f"AUROC reproduction mismatch: {model} {cohort}")

            metrics = {
                "auroc": full_auc,
                "n_real": len(real_scores),
                "n_aigi": len(aigi_scores),
                "real_score_mean": float(real_scores.mean()),
                "real_score_median": float(np.median(real_scores)),
                "aigi_score_mean": float(aigi_scores.mean()),
                "aigi_score_median": float(np.median(aigi_scores)),
            }
            model_metrics[model][cohort] = metrics
            comparison_rows.append({"model": model.upper(), "cohort": cohort, **metrics})

            real_reference = real_scores
            for generator_id in composition[cohort]["generators"]:
                selected = np.asarray(
                    [
                        score
                        for row, score in joined
                        if int(row["label"]) == 1
                        and row["canonical_generator_id"] == generator_id
                    ]
                )
                generator_manifest_row = next(
                    row
                    for row in manifests[cohort]
                    if int(row["label"]) == 1
                    and row["canonical_generator_id"] == generator_id
                )
                generator_rows.append(
                    {
                        "model": model.upper(),
                        "cohort": cohort,
                        "generator_id": generator_id,
                        "architecture": generator_manifest_row["architecture"],
                        "n_aigi": len(selected),
                        "auroc_vs_cohort_real": subset_auc(real_reference, selected),
                        "aigi_score_mean": float(selected.mean()),
                    }
                )

            for architecture in composition[cohort]["architectures"]:
                selected = np.asarray(
                    [
                        score
                        for row, score in joined
                        if int(row["label"]) == 1 and row["architecture"] == architecture
                    ]
                )
                architecture_rows.append(
                    {
                        "model": model.upper(),
                        "cohort": cohort,
                        "architecture": architecture,
                        "n_aigi": len(selected),
                        "auroc_vs_cohort_real": subset_auc(real_reference, selected),
                        "aigi_score_mean": float(selected.mean()),
                    }
                )

        seen_real, seen_aigi = score_groups["seen_family"]
        unseen_real, unseen_aigi = score_groups["unseen_generator"]
        model_metrics[model]["unseen_minus_seen_auroc"] = (
            model_metrics[model]["unseen_generator"]["auroc"]
            - model_metrics[model]["seen_family"]["auroc"]
        )
        model_metrics[model]["bootstrap"] = bootstrap_difference(
            seen_real,
            seen_aigi,
            unseen_real,
            unseen_aigi,
            seed=20260829 + model_index,
            replicates=args.bootstrap_replicates,
        )

    payload = {
        "metric_definition": (
            "AUROC is the probability that a randomly selected AIGI image receives a "
            "higher score than a randomly selected real image within the same cohort."
        ),
        "comparison_definition": "unseen_generator AUROC minus seen_family AUROC",
        "bootstrap": "Independent stratified resampling within each cohort and label.",
        "composition": composition,
        "models": model_metrics,
        "comparison_rows": comparison_rows,
        "generator_rows": generator_rows,
        "architecture_rows": architecture_rows,
        "source_manifests": {
            cohort: str((manifest_dir / filename).relative_to(root))
            for cohort, filename in COHORTS.items()
        },
        "prediction_sources": {
            model: {
                cohort: str(
                    Path("outputs")
                    / "community_forensics"
                    / model
                    / f"test_external_{cohort}"
                    / "predictions.jsonl"
                )
                for cohort in COHORTS
            }
            for model in MODELS
        },
    }
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"event": "complete", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
