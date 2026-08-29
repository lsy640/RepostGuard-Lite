from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HARD_GENERATORS = ("hourglass", "dfgan", "galip")
EXPECTED_BASE_COUNTS = {0: 9_000, 1: 9_000}
EXPECTED_PROMOTED_COUNTS = {0: 1_000, 1: 1_000}
EXPECTED_TRAIN_V2_COUNTS = {0: 10_000, 1: 10_000}
CORE_REQUIRED_FIELDS = (
    "sample_id",
    "path",
    "label",
    "split",
    "source_dataset",
    "official_split",
    "project_split",
    "generator_exposure",
    "sha256",
    "phash",
    "selection_seed",
    "source_revision",
    "source_file",
    "source_row_group",
    "source_row_index",
    "width",
    "height",
    "format",
    "byte_size",
)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"Manifest has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _counts(rows: list[dict[str, str]]) -> dict[int, int]:
    return dict(Counter(int(row["label"]) for row in rows))


def _identity_sets(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    return {
        "sample_id": {row["sample_id"] for row in rows},
        "path": {row["path"] for row in rows},
        "sha256": {row["sha256"] for row in rows},
        "source_locator": {
            "|".join(
                (
                    row["source_revision"],
                    row["source_file"],
                    row["source_row_group"],
                    row["source_row_index"],
                )
            )
            for row in rows
        },
    }


def _overlap_counts(
    left: list[dict[str, str]], right: list[dict[str, str]]
) -> dict[str, int]:
    left_sets = _identity_sets(left)
    right_sets = _identity_sets(right)
    return {
        key: len(left_sets[key].intersection(right_sets[key])) for key in left_sets
    }


def _assert_unique(rows: list[dict[str, str]], name: str) -> dict[str, int]:
    sets = _identity_sets(rows)
    counts = {key: len(rows) - len(values) for key, values in sets.items()}
    if any(counts.values()):
        raise RuntimeError(f"Duplicate identities in {name}: {counts}")
    return counts


def _verify_materialized(rows: list[dict[str, str]], data_root: Path) -> None:
    missing: list[str] = []
    size_mismatch: list[str] = []
    for row in rows:
        path = data_root / row["path"]
        if not path.is_file():
            missing.append(row["path"])
            continue
        if path.stat().st_size != int(row["byte_size"]):
            size_mismatch.append(row["path"])
    if missing or size_mismatch:
        raise RuntimeError(
            f"Materialized file audit failed: missing={len(missing)}, "
            f"size_mismatch={len(size_mismatch)}"
        )


def _quality_profile(rows: list[dict[str, str]], role: str) -> dict[str, Any]:
    missing = {
        field: sum(not str(row.get(field, "")).strip() for row in rows)
        for field in CORE_REQUIRED_FIELDS
    }
    missing = {field: count for field, count in missing.items() if count}
    conditional_missing = {
        "aigi_generator_identity": sum(
            int(row["label"]) == 1
            and not str(row.get("canonical_generator_id", "")).strip()
            for row in rows
        ),
        "aigi_architecture": sum(
            int(row["label"]) == 1
            and not str(row.get("architecture", "")).strip()
            for row in rows
        ),
        "real_source": sum(
            int(row["label"]) == 0
            and not str(row.get("real_source", "")).strip()
            for row in rows
        ),
    }
    conditional_missing = {
        field: count for field, count in conditional_missing.items() if count
    }
    invalid_labels = sum(row["label"] not in {"0", "1"} for row in rows)
    role_violations = 0
    if role == "train":
        role_violations = sum(
            row["split"] != "train"
            or row["project_split"] != "train"
            or (
                int(row["label"]) == 1
                and row["generator_exposure"] != "train_seen"
            )
            or (
                int(row["label"]) == 0
                and row["generator_exposure"] != "not_applicable"
            )
            for row in rows
        )
    if missing or conditional_missing or invalid_labels or role_violations:
        raise RuntimeError(
            f"Quality profile failed for {role}: missing={missing}, "
            f"conditional_missing={conditional_missing}, "
            f"invalid_labels={invalid_labels}, role_violations={role_violations}"
        )
    return {
        "rows": len(rows),
        "columns": len(rows[0]) if rows else 0,
        "required_field_completeness_rate": 1.0,
        "conditional_field_completeness_rate": 1.0,
        "valid_label_rate": 1.0,
        "role_consistency_rate": 1.0,
        "source_dataset_counts": dict(Counter(row["source_dataset"] for row in rows)),
        "format_counts": dict(Counter(row["format"] for row in rows)),
    }


def _promote_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    promoted: list[dict[str, str]] = []
    for source in rows:
        row = dict(source)
        row["split"] = "train"
        row["project_split"] = "train"
        row["generator_exposure"] = (
            "train_seen" if int(row["label"]) == 1 else "not_applicable"
        )
        promoted.append(row)
    return promoted


def _relabel_hard_rows(
    rows: list[dict[str, str]], generator: str
) -> list[dict[str, str]]:
    labels = _counts(rows)
    if labels != {0: 250, 1: 250}:
        raise RuntimeError(f"Unexpected {generator} hard-slice counts: {labels}")
    aigi_generators = {
        row["canonical_generator_id"] for row in rows if int(row["label"]) == 1
    }
    if aigi_generators != {generator}:
        raise RuntimeError(
            f"Unexpected {generator} hard-slice identities: {sorted(aigi_generators)}"
        )
    relabeled: list[dict[str, str]] = []
    for source in rows:
        row = dict(source)
        if int(row["label"]) == 1:
            row["generator_exposure"] = "exact_seen_hard"
        relabeled.append(row)
    return relabeled


def build(arguments: argparse.Namespace) -> None:
    base_train = Path(arguments.base_train)
    promoted_source = Path(arguments.promoted_seen_family)
    strict_unseen = Path(arguments.strict_unseen)
    output = Path(arguments.output)
    audit_path = Path(arguments.audit)
    marker = Path(arguments.complete_marker)
    data_root = Path(arguments.data_root)

    fields, base_rows = _read_csv(base_train)
    promoted_fields, promoted_source_rows = _read_csv(promoted_source)
    if fields != promoted_fields:
        raise RuntimeError("Base-train and promoted seen-family headers differ")
    if _counts(base_rows) != EXPECTED_BASE_COUNTS:
        raise RuntimeError(f"Unexpected base-train counts: {_counts(base_rows)}")
    if _counts(promoted_source_rows) != EXPECTED_PROMOTED_COUNTS:
        raise RuntimeError(
            f"Unexpected promoted seen-family counts: {_counts(promoted_source_rows)}"
        )

    input_overlap = _overlap_counts(base_rows, promoted_source_rows)
    if any(input_overlap.values()):
        raise RuntimeError(f"Base/promoted input overlap: {input_overlap}")

    promoted_rows = _promote_rows(promoted_source_rows)
    train_v2_rows = base_rows + promoted_rows
    if _counts(train_v2_rows) != EXPECTED_TRAIN_V2_COUNTS:
        raise RuntimeError(f"Unexpected train-v2 counts: {_counts(train_v2_rows)}")
    train_v2_duplicates = _assert_unique(train_v2_rows, "train-v2")
    _verify_materialized(train_v2_rows, data_root)
    train_v2_quality = _quality_profile(train_v2_rows, "train")

    base_audit_path = Path(arguments.base_audit)
    base_audit = _read_json(base_audit_path)
    if int(base_audit.get("exact_duplicate_count", -1)) != 0:
        raise RuntimeError("Frozen base audit reports exact duplicates")
    if int(base_audit.get("cross_split_phash_near_duplicate_count", -1)) != 0:
        raise RuntimeError("Frozen base audit reports cross-split pHash conflicts")
    if int(base_audit.get("phash_hamming_threshold", -1)) != 4:
        raise RuntimeError("Unexpected frozen base pHash threshold")

    promoted_generators = sorted(
        {
            row["canonical_generator_id"]
            for row in promoted_rows
            if int(row["label"]) == 1
        }
    )
    if len(promoted_generators) != 9:
        raise RuntimeError(
            f"Expected 9 promoted exact generators, found {len(promoted_generators)}"
        )

    train_generators = {
        row["canonical_generator_id"]
        for row in train_v2_rows
        if int(row["label"]) == 1
    }
    train_architectures = {
        row["architecture"] for row in train_v2_rows if int(row["label"]) == 1
    }

    hard_outputs: dict[str, dict[str, Any]] = {}
    for generator in HARD_GENERATORS:
        source_path = Path(
            arguments.manifest_dir
        ) / f"community_forensics_val_hard_{generator}.csv"
        hard_fields, hard_rows = _read_csv(source_path)
        if hard_fields != fields:
            raise RuntimeError(f"Hard-slice header differs: {source_path}")
        if generator not in train_generators:
            raise RuntimeError(f"Hard generator was not promoted to train: {generator}")
        relabeled = _relabel_hard_rows(hard_rows, generator)
        hard_duplicates = _assert_unique(relabeled, f"hard-{generator}")
        _verify_materialized(relabeled, data_root)
        train_hard_overlap = _overlap_counts(train_v2_rows, relabeled)
        if any(train_hard_overlap.values()):
            raise RuntimeError(
                f"Train-v2/hard-{generator} image overlap: {train_hard_overlap}"
            )
        hard_output = Path(arguments.manifest_dir) / (
            f"community_forensics_val_hard_{generator}_v2_exact_seen.csv"
        )
        _atomic_csv(hard_output, fields, relabeled)
        hard_outputs[generator] = {
            "source_manifest": str(source_path),
            "source_manifest_sha256": _sha256(source_path),
            "output_manifest": str(hard_output),
            "output_manifest_sha256": _sha256(hard_output),
            "rows": len(relabeled),
            "aigi_exposure": "exact_seen_hard",
            "train_v2_overlap": train_hard_overlap,
            "duplicates": hard_duplicates,
            "materialized_paths_and_sizes_verified": True,
        }

    strict_fields, strict_unseen_rows = _read_csv(strict_unseen)
    if strict_fields != fields:
        raise RuntimeError("Strict-unseen manifest header differs from train")
    strict_generators = {
        row["canonical_generator_id"]
        for row in strict_unseen_rows
        if int(row["label"]) == 1
    }
    strict_architectures = {
        row["architecture"]
        for row in strict_unseen_rows
        if int(row["label"]) == 1
    }
    generator_overlap = sorted(train_generators.intersection(strict_generators))
    architecture_overlap = sorted(
        train_architectures.intersection(strict_architectures)
    )
    if generator_overlap or architecture_overlap:
        raise RuntimeError(
            "Strict unseen protocol invalid after promotion: "
            f"generators={generator_overlap}, architectures={architecture_overlap}"
        )
    strict_identity_overlap = _overlap_counts(train_v2_rows, strict_unseen_rows)
    if any(strict_identity_overlap.values()):
        raise RuntimeError(
            f"Train-v2/strict-unseen image overlap: {strict_identity_overlap}"
        )

    _atomic_csv(output, fields, train_v2_rows)
    completed_at = datetime.now(timezone.utc).isoformat()
    audit = {
        "schema_version": 1,
        "protocol_id": "community_forensics_train_v2_seen_family_promoted",
        "completed_at_utc": completed_at,
        "policy": {
            "promoted_manifest": str(promoted_source),
            "promoted_manifest_original_role": "external seen-family test",
            "promoted_manifest_new_role": "training only",
            "excluded_from_future_evaluation": True,
            "original_frozen_manifests_preserved": True,
            "physical_images_copied": False,
            "path_note": (
                "Promoted rows retain their original materialized relative paths and "
                "sample IDs for lineage; path text is never a model input."
            ),
        },
        "inputs": {
            "base_audit": {
                "path": str(base_audit_path),
                "sha256": _sha256(base_audit_path),
                "exact_duplicate_count": 0,
                "cross_split_phash_near_duplicate_count": 0,
                "phash_hamming_threshold": 4,
            },
            "base_train": {
                "path": str(base_train),
                "sha256": _sha256(base_train),
                "rows": len(base_rows),
            },
            "promoted_seen_family": {
                "path": str(promoted_source),
                "sha256": _sha256(promoted_source),
                "rows": len(promoted_source_rows),
            },
            "strict_unseen_test": {
                "path": str(strict_unseen),
                "sha256": _sha256(strict_unseen),
                "rows": len(strict_unseen_rows),
            },
        },
        "output": {
            "path": str(output),
            "sha256": _sha256(output),
            "rows": len(train_v2_rows),
            "class_counts": {
                "real": _counts(train_v2_rows)[0],
                "aigi": _counts(train_v2_rows)[1],
            },
            "exact_generator_count": len(train_generators),
            "architecture_families": sorted(train_architectures),
            "promoted_exact_generators": promoted_generators,
        },
        "integrity": {
            "base_promoted_overlap": input_overlap,
            "train_v2_duplicates": train_v2_duplicates,
            "train_v2_strict_unseen_overlap": strict_identity_overlap,
            "train_v2_strict_unseen_generator_overlap": generator_overlap,
            "train_v2_strict_unseen_architecture_overlap": architecture_overlap,
            "materialized_paths_and_sizes_verified": True,
            "quality_profile": train_v2_quality,
            "uniqueness_rates": {
                "sample_id": 1.0,
                "path": 1.0,
                "sha256": 1.0,
                "source_locator": 1.0,
            },
        },
        "hard_slices": hard_outputs,
    }
    _atomic_json(audit_path, audit)
    _atomic_json(
        marker,
        {
            "completed_at_utc": completed_at,
            "protocol_id": audit["protocol_id"],
            "train_manifest": str(output),
            "train_manifest_sha256": _sha256(output),
            "audit": str(audit_path),
            "audit_sha256": _sha256(audit_path),
        },
    )
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Promote the frozen Community Forensics seen-family test into a "
            "versioned training manifest without copying image files."
        )
    )
    parser.add_argument(
        "--base-train", default="data/manifests/community_forensics_train.csv"
    )
    parser.add_argument(
        "--promoted-seen-family",
        default="data/manifests/community_forensics_test_external_seen_family.csv",
    )
    parser.add_argument(
        "--strict-unseen",
        default="data/manifests/community_forensics_test_external_unseen_generator.csv",
    )
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument(
        "--base-audit", default="data/manifests/community_forensics_audit.json"
    )
    parser.add_argument("--data-root", default="data/raw/community_forensics")
    parser.add_argument(
        "--output", default="data/manifests/community_forensics_train_v2.csv"
    )
    parser.add_argument(
        "--audit",
        default="data/manifests/community_forensics_train_v2_audit.json",
    )
    parser.add_argument(
        "--complete-marker",
        default="data/raw/community_forensics/TRAIN_V2_COMPLETE",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(_parse_args())
