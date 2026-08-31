from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


IDENTITY_FIELDS = ("sample_id", "path", "sha256")
LOCATOR_FIELDS = ("source_revision", "source_file", "source_row_group", "source_row_index")


def _source_locator(row: dict[str, str]) -> tuple[str, ...]:
    values = tuple(str(row.get(field, "")) for field in LOCATOR_FIELDS)
    if not all(values):
        raise ValueError(f"Manifest row is missing source locator fields: {row.get('sample_id')}")
    return values


def _read(path: str) -> list[dict[str, str]]:
    manifest = Path(path).expanduser().resolve()
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty manifest: {manifest}")
    return rows


def validate(
    calibration_path: str,
    against_paths: list[str],
    *,
    expected_samples: int,
    expected_per_label: int,
) -> dict[str, object]:
    calibration = _read(calibration_path)
    if len(calibration) != expected_samples:
        raise ValueError(
            f"Calibration sample count mismatch: {len(calibration)} != {expected_samples}"
        )
    labels = Counter(str(row.get("label", "")) for row in calibration)
    expected_labels = {"0": expected_per_label, "1": expected_per_label}
    if dict(labels) != expected_labels:
        raise ValueError(f"Calibration label counts mismatch: {dict(labels)}")

    duplicate_counts: dict[str, int] = {}
    for field in IDENTITY_FIELDS:
        values = [str(row.get(field, "")) for row in calibration]
        if not all(values):
            raise ValueError(f"Calibration manifest is missing {field}")
        duplicate_counts[field] = len(values) - len(set(values))
    locators = [_source_locator(row) for row in calibration]
    duplicate_counts["source_locator"] = len(locators) - len(set(locators))
    if any(duplicate_counts.values()):
        raise ValueError(f"Calibration manifest has duplicate identities: {duplicate_counts}")

    calibration_values = {
        field: {str(row[field]) for row in calibration} for field in IDENTITY_FIELDS
    }
    calibration_locators = set(locators)
    overlap_report: list[dict[str, object]] = []
    for against_path in against_paths:
        against = _read(against_path)
        overlaps = {
            field: len(
                calibration_values[field].intersection(
                    str(row.get(field, "")) for row in against
                )
            )
            for field in IDENTITY_FIELDS
        }
        overlaps["source_locator"] = len(
            calibration_locators.intersection(_source_locator(row) for row in against)
        )
        overlap_report.append({"manifest": str(Path(against_path)), "overlaps": overlaps})
        if any(overlaps.values()):
            raise ValueError(
                f"Calibration leakage against {against_path}: {overlaps}"
            )

    result = {
        "event": "distillation_calibration_manifest_validated",
        "calibration_manifest": str(Path(calibration_path)),
        "samples": len(calibration),
        "labels": dict(labels),
        "duplicate_counts": duplicate_counts,
        "against": overlap_report,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a held-out teacher-calibration manifest"
    )
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--against", action="append", default=[], required=True)
    parser.add_argument("--expected-samples", type=int, default=2000)
    parser.add_argument("--expected-per-label", type=int, default=1000)
    arguments = parser.parse_args()
    validate(
        arguments.calibration,
        arguments.against,
        expected_samples=arguments.expected_samples,
        expected_per_label=arguments.expected_per_label,
    )


if __name__ == "__main__":
    main()
