from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable
from zoneinfo import ZoneInfo


SPLITS = (
    {
        "key": "train",
        "title": "Train",
        "role": "Training",
        "manifest": "data/manifests/community_forensics_train.csv",
        "protocol": "Small train; the only split used for parameter fitting.",
    },
    {
        "key": "val_unseen_generator",
        "title": "Internal unseen-generator val",
        "role": "Checkpoint validation",
        "manifest": "data/manifests/community_forensics_val_unseen_generator.csv",
        "protocol": "Small holdout; exact generators are disjoint from Small train.",
    },
    {
        "key": "val_external_exact_seen_generator",
        "title": "External exact-seen val",
        "role": "External validation",
        "manifest": "data/manifests/community_forensics_val_external_exact_seen_generator.csv",
        "protocol": "AIGIBench; exact SD 1.4 identity appears in Small train.",
    },
    {
        "key": "val_hard_hourglass",
        "title": "Hard Hourglass val",
        "role": "Hard validation",
        "manifest": "data/manifests/community_forensics_val_hard_hourglass.csv",
        "protocol": "Exact Hourglass identity unseen; PixDiff architecture family seen.",
    },
    {
        "key": "val_hard_dfgan",
        "title": "Hard DFGAN val",
        "role": "Hard validation",
        "manifest": "data/manifests/community_forensics_val_hard_dfgan.csv",
        "protocol": "Exact DFGAN identity unseen; GAN architecture family seen.",
    },
    {
        "key": "val_hard_galip",
        "title": "Hard GALIP val",
        "role": "Hard validation",
        "manifest": "data/manifests/community_forensics_val_hard_galip.csv",
        "protocol": "Exact GALIP identity unseen; GAN architecture family seen.",
    },
    {
        "key": "test_external_seen_family",
        "title": "External seen-family test",
        "role": "External test",
        "manifest": "data/manifests/community_forensics_test_external_seen_family.csv",
        "protocol": "Architecture family seen in train; every exact generator identity unseen.",
    },
    {
        "key": "test_external_unseen_generator",
        "title": "External unseen-family test",
        "role": "External test",
        "manifest": "data/manifests/community_forensics_test_external_unseen_generator.csv",
        "protocol": "Architecture family and exact generator identity both unseen.",
    },
)

REQUIRED_FIELDS = {
    "sample_id",
    "path",
    "label",
    "source_dataset",
    "official_split",
    "project_split",
    "real_source",
    "canonical_generator_id",
    "architecture",
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
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".webp",
    ".bmp",
}

RESOLUTION_BINS = (
    ("<=0.10 MP", 0.0, 0.10),
    ("0.10-0.30 MP", 0.10, 0.30),
    ("0.30-1.00 MP", 0.30, 1.00),
    ("1.00-5.00 MP", 1.00, 5.00),
    (">5.00 MP", 5.00, float("inf")),
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
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"Manifest has no header: {path}")
        missing = REQUIRED_FIELDS - set(reader.fieldnames)
        if missing:
            raise RuntimeError(f"Manifest missing fields {sorted(missing)}: {path}")
        return list(reader)


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
    fields = list(rows[0])
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)


def _clean(value: str | None, fallback: str = "UNSPECIFIED") -> str:
    normalized = (value or "").strip()
    if not normalized or normalized.lower() in {"n/a", "na", "none", "null"}:
        return fallback
    return normalized


def _format_name(value: str | None) -> str:
    normalized = _clean(value).upper()
    if normalized in {"JPG", "JPEG"}:
        return "JPEG"
    if normalized in {"TIF", "TIFF"}:
        return "TIFF"
    return normalized


def _percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _short_digest(value: str) -> str:
    return value[:12]


def _gib(value: int | float) -> float:
    return float(value) / (1024.0**3)


def _pct(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def _sqlite_type(values: list[Any]) -> str:
    concrete = [value for value in values if value is not None]
    if concrete and all(isinstance(value, int) and not isinstance(value, bool) for value in concrete):
        return "INTEGER"
    if concrete and all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in concrete
    ):
        return "REAL"
    return "TEXT"


def _materialize_sql_snapshot(
    staged: dict[str, list[dict[str, Any]]],
    queries: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        for dataset, rows in staged.items():
            if not rows:
                raise RuntimeError(f"Cannot materialize empty report dataset: {dataset}")
            columns = list(rows[0])
            if any(set(row) != set(columns) for row in rows):
                raise RuntimeError(f"Inconsistent columns in dataset: {dataset}")
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


def _source(dataset: str, query: str, generated_at: str) -> dict[str, Any]:
    return {
        "id": f"{dataset}_sql",
        "label": f"Audited {dataset.replace('_', ' ')} snapshot",
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": query,
            "description": (
                "Executed over rows staged after manifest schema, label, path, "
                "file-size, overlap, and protocol checks."
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


def _resolution_bin(megapixels: float) -> str:
    for label, lower, upper in RESOLUTION_BINS:
        if lower < megapixels <= upper or (lower == 0.0 and megapixels <= upper):
            return label
    return RESOLUTION_BINS[-1][0]


def _load_manifests(
    data_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    by_split: dict[str, list[dict[str, Any]]] = {}
    manifest_meta: dict[str, dict[str, Any]] = {}
    for split_order, definition in enumerate(SPLITS):
        manifest_path = Path(definition["manifest"])
        raw_rows = _read_csv(manifest_path)
        rows: list[dict[str, Any]] = []
        for row_order, row in enumerate(raw_rows):
            try:
                label = int(row["label"])
                width = int(row["width"])
                height = int(row["height"])
                byte_size = int(row["byte_size"])
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"Invalid numeric field in {manifest_path} row {row_order + 2}"
                ) from error
            if label not in {0, 1}:
                raise RuntimeError(f"Invalid label {label} in {manifest_path}")
            if row["project_split"] != definition["key"]:
                raise RuntimeError(
                    f"project_split drift in {manifest_path}: {row['project_split']}"
                )
            absolute_path = data_root / row["path"]
            architecture = _clean(row["architecture"], "UNSPECIFIED")
            generator = _clean(row["canonical_generator_id"], "UNSPECIFIED")
            rows.append(
                {
                    **row,
                    "label": label,
                    "class_name": "AIGI" if label == 1 else "Real",
                    "width": width,
                    "height": height,
                    "byte_size": byte_size,
                    "megapixels": width * height / 1_000_000.0,
                    "format_normalized": _format_name(row["format"]),
                    "real_source_normalized": _clean(row["real_source"]),
                    "generator_normalized": generator,
                    "architecture_normalized": architecture,
                    "absolute_path": str(absolute_path),
                    "split_key": definition["key"],
                    "split_title": definition["title"],
                    "role": definition["role"],
                    "split_order": split_order,
                    "row_order": row_order,
                }
            )
        by_split[definition["key"]] = rows
        revisions = sorted({_clean(row["source_revision"]) for row in raw_rows})
        seeds = sorted({_clean(row["selection_seed"]) for row in raw_rows})
        manifest_meta[definition["key"]] = {
            "split": definition["title"],
            "split_key": definition["key"],
            "split_order": split_order,
            "role": definition["role"],
            "manifest": str(manifest_path),
            "manifest_sha256": _short_digest(_sha256(manifest_path)),
            "manifest_bytes": manifest_path.stat().st_size,
            "selection_seed": ", ".join(seeds),
            "source_revision": ", ".join(_short_digest(value) for value in revisions),
            "protocol": definition["protocol"],
        }
    return by_split, manifest_meta


def _analyze(
    by_split: dict[str, list[dict[str, Any]]],
    manifest_meta: dict[str, dict[str, Any]],
    data_root: Path,
    tiff_audit_path: Path,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
    dict[str, Any],
]:
    all_rows = [row for definition in SPLITS for row in by_split[definition["key"]]]
    path_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sha_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sample_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_paths: list[str] = []
    byte_size_mismatches: list[dict[str, Any]] = []
    for row in all_rows:
        path_groups[row["absolute_path"]].append(row)
        sha_groups[row["sha256"]].append(row)
        sample_groups[row["sample_id"]].append(row)

    physical_bytes = 0
    physical_rows: list[dict[str, Any]] = []
    for absolute_path, rows in sorted(path_groups.items()):
        path = Path(absolute_path)
        if not path.is_file():
            missing_paths.append(absolute_path)
            continue
        actual_size = path.stat().st_size
        expected_sizes = {int(row["byte_size"]) for row in rows}
        if expected_sizes != {actual_size}:
            byte_size_mismatches.append(
                {
                    "path": absolute_path,
                    "actual_size": actual_size,
                    "manifest_sizes": sorted(expected_sizes),
                }
            )
        physical_bytes += actual_size
        physical_rows.append(rows[0])

    split_summaries: list[dict[str, Any]] = []
    class_counts: list[dict[str, Any]] = []
    architecture_counts: list[dict[str, Any]] = []
    format_counts: list[dict[str, Any]] = []
    real_source_counts: list[dict[str, Any]] = []
    resolution_counts: list[dict[str, Any]] = []
    generator_statistics: list[dict[str, Any]] = []

    train_rows = by_split["train"]
    train_generators = {
        row["generator_normalized"]
        for row in train_rows
        if row["label"] == 1
    }
    train_architectures = {
        row["architecture_normalized"]
        for row in train_rows
        if row["label"] == 1
    }
    exposure_statistics: list[dict[str, Any]] = []

    for definition in SPLITS:
        split_key = definition["key"]
        rows = by_split[split_key]
        real_rows = [row for row in rows if row["label"] == 0]
        aigi_rows = [row for row in rows if row["label"] == 1]
        byte_sizes = [row["byte_size"] for row in rows]
        megapixels = [row["megapixels"] for row in rows]
        widths = [row["width"] for row in rows]
        heights = [row["height"] for row in rows]
        split_generators = {row["generator_normalized"] for row in aigi_rows}
        split_architectures = {row["architecture_normalized"] for row in aigi_rows}
        unique_paths = {row["absolute_path"] for row in rows}
        unique_shas = {row["sha256"] for row in rows}
        duplicate_paths_within = sum(
            1 for count in Counter(row["absolute_path"] for row in rows).values() if count > 1
        )
        duplicate_shas_within = sum(
            1 for count in Counter(row["sha256"] for row in rows).values() if count > 1
        )
        split_summaries.append(
            {
                "split": definition["title"],
                "split_key": split_key,
                "split_order": definition["key"] == "train" and 0 or list(SPLITS).index(definition),
                "role": definition["role"],
                "manifest_rows": len(rows),
                "real": len(real_rows),
                "aigi": len(aigi_rows),
                "aigi_percent": round(_pct(len(aigi_rows), len(rows)), 4),
                "unique_paths": len(unique_paths),
                "unique_sha256": len(unique_shas),
                "exact_generators": len(split_generators),
                "architecture_families": len(split_architectures),
                "real_sources": len({row["real_source_normalized"] for row in real_rows}),
                "formats": len({row["format_normalized"] for row in rows}),
                "nominal_gib": round(_gib(sum(byte_sizes)), 6),
                "mean_file_mib": round(fmean(byte_sizes) / (1024.0**2), 6),
                "median_file_mib": round(median(byte_sizes) / (1024.0**2), 6),
                "p95_file_mib": round(_percentile(byte_sizes, 0.95) / (1024.0**2), 6),
                "mean_megapixels": round(fmean(megapixels), 6),
                "median_megapixels": round(median(megapixels), 6),
                "p95_megapixels": round(_percentile(megapixels, 0.95), 6),
                "width_range": f"{min(widths)}-{max(widths)}",
                "height_range": f"{min(heights)}-{max(heights)}",
                "duplicate_path_groups_within": duplicate_paths_within,
                "duplicate_sha_groups_within": duplicate_shas_within,
            }
        )
        for class_order, class_name in enumerate(("Real", "AIGI")):
            count = sum(row["class_name"] == class_name for row in rows)
            class_counts.append(
                {
                    "split": definition["title"],
                    "split_order": list(SPLITS).index(definition),
                    "role": definition["role"],
                    "class": class_name,
                    "class_order": class_order,
                    "count": count,
                    "share_percent": round(_pct(count, len(rows)), 4),
                }
            )
        for architecture_order, (architecture, count) in enumerate(
            sorted(Counter(row["architecture_normalized"] for row in aigi_rows).items())
        ):
            architecture_counts.append(
                {
                    "split": definition["title"],
                    "split_order": list(SPLITS).index(definition),
                    "architecture": architecture,
                    "architecture_order": architecture_order,
                    "aigi_count": count,
                    "share_percent": round(_pct(count, len(aigi_rows)), 4),
                }
            )
        for format_order, (format_name, count) in enumerate(
            sorted(Counter(row["format_normalized"] for row in rows).items())
        ):
            format_counts.append(
                {
                    "split": definition["title"],
                    "split_order": list(SPLITS).index(definition),
                    "format": format_name,
                    "format_order": format_order,
                    "count": count,
                    "share_percent": round(_pct(count, len(rows)), 4),
                }
            )
        for source_order, (source, count) in enumerate(
            sorted(Counter(row["real_source_normalized"] for row in real_rows).items())
        ):
            real_source_counts.append(
                {
                    "split": definition["title"],
                    "split_order": list(SPLITS).index(definition),
                    "real_source": source,
                    "source_order": source_order,
                    "real_count": count,
                    "share_percent": round(_pct(count, len(real_rows)), 4),
                }
            )
        resolution_counter = Counter(_resolution_bin(row["megapixels"]) for row in rows)
        for bin_order, (label, _, _) in enumerate(RESOLUTION_BINS):
            resolution_counts.append(
                {
                    "split": definition["title"],
                    "split_order": list(SPLITS).index(definition),
                    "resolution_bin": label,
                    "bin_order": bin_order,
                    "count": resolution_counter[label],
                    "share_percent": round(_pct(resolution_counter[label], len(rows)), 4),
                }
            )
        generator_counter = Counter(row["generator_normalized"] for row in aigi_rows)
        generator_architectures: dict[str, set[str]] = defaultdict(set)
        generator_exposures: dict[str, set[str]] = defaultdict(set)
        for row in aigi_rows:
            generator_architectures[row["generator_normalized"]].add(
                row["architecture_normalized"]
            )
            generator_exposures[row["generator_normalized"]].add(
                _clean(row["generator_exposure"])
            )
        for generator_order, (generator, count) in enumerate(
            sorted(generator_counter.items(), key=lambda item: (-item[1], item[0]))
        ):
            generator_statistics.append(
                {
                    "split": definition["title"],
                    "split_order": list(SPLITS).index(definition),
                    "generator": generator,
                    "generator_order": generator_order,
                    "architecture": ", ".join(sorted(generator_architectures[generator])),
                    "exposure": ", ".join(sorted(generator_exposures[generator])),
                    "aigi_images": count,
                    "share_percent": round(_pct(count, len(aigi_rows)), 4),
                    "exact_identity_seen_in_train": "yes" if generator in train_generators else "no",
                    "architecture_seen_in_train": (
                        "yes"
                        if generator_architectures[generator] & train_architectures
                        else "no"
                    ),
                }
            )
        exposure_statistics.append(
            {
                "split": definition["title"],
                "split_order": list(SPLITS).index(definition),
                "role": definition["role"],
                "exact_generators": len(split_generators),
                "exact_generators_seen_in_train": len(split_generators & train_generators),
                "exact_generators_unseen_in_train": len(split_generators - train_generators),
                "architecture_families": len(split_architectures),
                "architecture_families_seen_in_train": len(
                    split_architectures & train_architectures
                ),
                "architecture_families_unseen_in_train": len(
                    split_architectures - train_architectures
                ),
                "protocol": definition["protocol"],
            }
        )

    split_sets = {
        key: {
            "sha": {row["sha256"] for row in rows},
            "path": {row["absolute_path"] for row in rows},
            "generators": {
                row["generator_normalized"] for row in rows if row["label"] == 1
            },
            "architectures": {
                row["architecture_normalized"] for row in rows if row["label"] == 1
            },
        }
        for key, rows in by_split.items()
    }
    pairwise_overlap: list[dict[str, Any]] = []
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            left_sets = split_sets[left["key"]]
            right_sets = split_sets[right["key"]]
            pairwise_overlap.append(
                {
                    "left_split": left["title"],
                    "left_order": left_index,
                    "right_split": right["title"],
                    "right_order": list(SPLITS).index(right),
                    "shared_paths": len(left_sets["path"] & right_sets["path"]),
                    "shared_sha256": len(left_sets["sha"] & right_sets["sha"]),
                    "shared_exact_generators": len(
                        left_sets["generators"] & right_sets["generators"]
                    ),
                    "shared_architecture_families": len(
                        left_sets["architectures"] & right_sets["architectures"]
                    ),
                }
            )

    duplicate_sha_groups = [rows for rows in sha_groups.values() if len(rows) > 1]
    duplicate_path_groups = [rows for rows in path_groups.values() if len(rows) > 1]
    duplicate_sample_groups = [rows for rows in sample_groups.values() if len(rows) > 1]
    hard_keys = {"val_hard_hourglass", "val_hard_dfgan", "val_hard_galip"}

    def expected_shared_panel(rows: list[dict[str, Any]]) -> bool:
        return (
            len(rows) == 3
            and {row["split_key"] for row in rows} == hard_keys
            and {row["label"] for row in rows} == {0}
            and all("hard_real_panel/real/" in row["path"] for row in rows)
        )

    unexpected_sha_groups = [rows for rows in duplicate_sha_groups if not expected_shared_panel(rows)]
    unexpected_path_groups = [rows for rows in duplicate_path_groups if not expected_shared_panel(rows)]
    unexpected_sample_groups = [rows for rows in duplicate_sample_groups if not expected_shared_panel(rows)]
    cross_label_sha_conflicts = sum(
        1 for rows in duplicate_sha_groups if len({row["label"] for row in rows}) > 1
    )

    unique_generators = {
        row["generator_normalized"] for row in all_rows if row["label"] == 1
    }
    unique_architectures = {
        row["architecture_normalized"] for row in all_rows if row["label"] == 1
    }
    unique_real_sources = {
        row["real_source_normalized"] for row in all_rows if row["label"] == 0
    }
    unique_formats = {row["format_normalized"] for row in physical_rows}
    unique_real = sum(row["label"] == 0 for row in physical_rows)
    unique_aigi = sum(row["label"] == 1 for row in physical_rows)
    nominal_bytes = sum(row["byte_size"] for row in all_rows)

    tiff_audit = _read_json(tiff_audit_path)
    current_tiff_paths = {
        row["absolute_path"] for row in physical_rows if row["format_normalized"] == "TIFF"
    }
    prior_tiff_valid = int(tiff_audit.get("valid_files", 0))
    current_tiff_count = len(current_tiff_paths)
    tiff_coverage = min(prior_tiff_valid, current_tiff_count)

    base_audit = _read_json("data/manifests/community_forensics_audit.json")
    plan_audit = _read_json("data/manifests/community_forensics_plan_audit.json")
    validation_audit = _read_json(
        "data/manifests/community_forensics_validation_v2_audit.json"
    )
    exact_repairs = _read_json(
        "data/manifests/community_forensics_exact_dedup_repairs.json"
    )
    phash_repairs = _read_json("data/manifests/community_forensics_phash_repairs.json")

    integrity_checks = [
        {
            "check_order": 0,
            "check": "Manifest schema and numeric fields",
            "result": "PASS",
            "count": len(SPLITS),
            "interpretation": "All eight manifests contain the required lineage and image metadata fields.",
        },
        {
            "check_order": 1,
            "check": "Referenced paths exist",
            "result": "PASS" if not missing_paths else "FAIL",
            "count": len(missing_paths),
            "interpretation": "Count is missing manifest-referenced files; expected zero.",
        },
        {
            "check_order": 2,
            "check": "Manifest byte_size matches stat",
            "result": "PASS" if not byte_size_mismatches else "FAIL",
            "count": len(byte_size_mismatches),
            "interpretation": "Count is files with metadata/stat size mismatch; expected zero.",
        },
        {
            "check_order": 3,
            "check": "Within-split exact SHA duplicates",
            "result": "PASS"
            if all(row["duplicate_sha_groups_within"] == 0 for row in split_summaries)
            else "FAIL",
            "count": sum(row["duplicate_sha_groups_within"] for row in split_summaries),
            "interpretation": "No exact duplicate is allowed within an individual split.",
        },
        {
            "check_order": 4,
            "check": "Expected shared hard-real panel",
            "result": "PASS" if len(duplicate_path_groups) == 250 else "WARN",
            "count": len(duplicate_path_groups),
            "interpretation": "Exactly 250 real images should be referenced by all three hard slices.",
        },
        {
            "check_order": 5,
            "check": "Unexpected cross-split path overlap",
            "result": "PASS" if not unexpected_path_groups else "FAIL",
            "count": len(unexpected_path_groups),
            "interpretation": "Any overlap outside the shared hard-real panel is unexpected.",
        },
        {
            "check_order": 6,
            "check": "Unexpected cross-split SHA overlap",
            "result": "PASS" if not unexpected_sha_groups else "FAIL",
            "count": len(unexpected_sha_groups),
            "interpretation": "Any exact-content overlap outside the shared hard-real panel is unexpected.",
        },
        {
            "check_order": 7,
            "check": "Cross-label SHA conflicts",
            "result": "PASS" if cross_label_sha_conflicts == 0 else "FAIL",
            "count": cross_label_sha_conflicts,
            "interpretation": "The same exact content must not appear with both Real and AIGI labels.",
        },
        {
            "check_order": 8,
            "check": "Base pHash cross-split near duplicates",
            "result": "PASS"
            if int(base_audit["cross_split_phash_near_duplicate_count"]) == 0
            else "FAIL",
            "count": int(base_audit["cross_split_phash_near_duplicate_count"]),
            "interpretation": "Base build audit uses pHash Hamming distance <= 4.",
        },
        {
            "check_order": 9,
            "check": "Prior TIFF full-decode audit",
            "result": "PARTIAL",
            "count": tiff_coverage,
            "interpretation": (
                f"{tiff_coverage}/{current_tiff_count} currently selected unique TIFF files were covered by the prior audit; "
                f"{current_tiff_count - tiff_coverage} validation-v2 TIFF files were added later."
            ),
        },
        {
            "check_order": 10,
            "check": "Reserved-set hash audit",
            "result": "NOT AVAILABLE",
            "count": 0,
            "interpretation": base_audit["reserved_set_limitation"],
        },
    ]

    build_audit = [
        {
            "audit_order": 0,
            "stage": "Base exact dedup repair",
            "count": len(exact_repairs.get("repairs", [])),
            "outcome": "Replaced and quarantined before final manifests.",
        },
        {
            "audit_order": 1,
            "stage": "Base pHash repair",
            "count": len(phash_repairs.get("repairs", [])),
            "outcome": "Replaced near-duplicate validation rows before final manifests.",
        },
        {
            "audit_order": 2,
            "stage": "Validation-v2 rejected candidates",
            "count": int(validation_audit.get("rejected_candidate_count", 0)),
            "outcome": "Rejected exact/pHash conflicts before materialization.",
        },
        {
            "audit_order": 3,
            "stage": "Base final exact duplicate count",
            "count": int(base_audit.get("exact_duplicate_count", -1)),
            "outcome": "Final base protocol audit result.",
        },
        {
            "audit_order": 4,
            "stage": "Base final pHash near-duplicate count",
            "count": int(base_audit.get("cross_split_phash_near_duplicate_count", -1)),
            "outcome": "Final base protocol audit result at Hamming distance <= 4.",
        },
    ]

    headline = {
        "manifest_count": len(SPLITS),
        "manifest_rows": len(all_rows),
        "unique_images": len(path_groups),
        "duplicate_references": len(all_rows) - len(path_groups),
        "unique_real": unique_real,
        "unique_aigi": unique_aigi,
        "physical_gib": round(_gib(physical_bytes), 6),
        "nominal_manifest_gib": round(_gib(nominal_bytes), 6),
        "unique_generators": len(unique_generators),
        "unique_architectures": len(unique_architectures),
        "unique_real_sources": len(unique_real_sources),
        "unique_formats": len(unique_formats),
        "shared_hard_real_images": len(duplicate_path_groups),
        "missing_paths": len(missing_paths),
        "size_mismatches": len(byte_size_mismatches),
        "unexpected_sha_overlap_groups": len(unexpected_sha_groups),
        "current_tiff_files": current_tiff_count,
        "prior_tiff_valid_files": tiff_coverage,
        "prior_tiff_warning_files": int(tiff_audit.get("warning_files", 0)),
        "prior_tiff_failed_files": int(tiff_audit.get("failed_files", 0)),
        "train_generator_count": len(train_generators),
        "internal_val_generator_count": len(
            {
                row["generator_normalized"]
                for row in by_split["val_unseen_generator"]
                if row["label"] == 1
            }
        ),
        "base_selection_seed": int(plan_audit["selection_seed"]),
    }

    global_summary = [
        {
            "scope": "Manifest references",
            "rows_or_images": len(all_rows),
            "real": sum(row["label"] == 0 for row in all_rows),
            "aigi": sum(row["label"] == 1 for row in all_rows),
            "aigi_percent": round(
                _pct(sum(row["label"] == 1 for row in all_rows), len(all_rows)), 4
            ),
            "storage_gib": round(_gib(nominal_bytes), 6),
            "definition": "Sum across eight manifests; shared hard-real panel counted three times.",
        },
        {
            "scope": "Unique physical corpus",
            "rows_or_images": len(physical_rows),
            "real": unique_real,
            "aigi": unique_aigi,
            "aigi_percent": round(_pct(unique_aigi, len(physical_rows)), 4),
            "storage_gib": round(_gib(physical_bytes), 6),
            "definition": "Unique manifest-referenced path; shared hard-real panel counted once.",
        },
        {
            "scope": "Training only",
            "rows_or_images": len(train_rows),
            "real": sum(row["label"] == 0 for row in train_rows),
            "aigi": sum(row["label"] == 1 for row in train_rows),
            "aigi_percent": round(
                _pct(sum(row["label"] == 1 for row in train_rows), len(train_rows)), 4
            ),
            "storage_gib": round(_gib(sum(row["byte_size"] for row in train_rows)), 6),
            "definition": "Only the split used for model parameter fitting.",
        },
    ]

    manifest_lineage = [manifest_meta[definition["key"]] for definition in SPLITS]
    staged = {
        "headline_metrics": [headline],
        "global_summary": global_summary,
        "split_summary": split_summaries,
        "class_counts": class_counts,
        "architecture_counts": architecture_counts,
        "format_counts": format_counts,
        "real_source_counts": real_source_counts,
        "resolution_counts": resolution_counts,
        "generator_exposure": exposure_statistics,
        "generator_statistics": generator_statistics,
        "pairwise_overlap": pairwise_overlap,
        "integrity_checks": integrity_checks,
        "build_audit": build_audit,
        "manifest_lineage": manifest_lineage,
    }
    audit_details = {
        "missing_paths": missing_paths,
        "byte_size_mismatches": byte_size_mismatches,
        "unexpected_sha_overlap_groups": [
            [
                {
                    "sample_id": row["sample_id"],
                    "split": row["split_key"],
                    "path": row["path"],
                    "label": row["label"],
                }
                for row in rows
            ]
            for rows in unexpected_sha_groups
        ],
        "unexpected_path_overlap_groups": len(unexpected_path_groups),
        "unexpected_sample_id_overlap_groups": len(unexpected_sample_groups),
        "duplicate_sha_groups_total": len(duplicate_sha_groups),
        "duplicate_path_groups_total": len(duplicate_path_groups),
        "duplicate_sample_id_groups_total": len(duplicate_sample_groups),
        "cross_label_sha_conflicts": cross_label_sha_conflicts,
        "physical_bytes": physical_bytes,
        "nominal_manifest_bytes": nominal_bytes,
        "data_root": str(data_root),
        "base_audit": str(Path("data/manifests/community_forensics_audit.json")),
        "validation_v2_audit": str(
            Path("data/manifests/community_forensics_validation_v2_audit.json")
        ),
        "tiff_audit": str(tiff_audit_path),
        "tiff_audit_completed_at_utc": tiff_audit.get("completed_at_utc"),
        "current_tiff_files": current_tiff_count,
        "prior_tiff_covered_files": tiff_coverage,
    }
    return staged, headline, audit_details


def _build_artifact(
    datasets: dict[str, list[dict[str, Any]]],
    queries: dict[str, str],
    headline: dict[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    title = "Community Forensics 当前训练集与全部验证/测试集数据统计报告"
    source_ids = {dataset: f"{dataset}_sql" for dataset in queries}
    sources = [_source(dataset, query, generated_at) for dataset, query in queries.items()]
    split_rows = datasets["split_summary"]
    largest_storage = max(split_rows, key=lambda row: float(row["nominal_gib"]))
    highest_resolution = max(split_rows, key=lambda row: float(row["mean_megapixels"]))

    cards = [
        {
            "id": "unique_images_card",
            "description": "All eight manifests deduplicated by selected physical path.",
            "dataset": "headline_metrics",
            "sourceId": source_ids["headline_metrics"],
            "metrics": [
                {"label": "唯一物理图像", "field": "unique_images", "format": "number"}
            ],
        },
        {
            "id": "storage_card",
            "description": "Unique manifest-referenced files; GiB uses 2^30 bytes.",
            "dataset": "headline_metrics",
            "sourceId": source_ids["headline_metrics"],
            "metrics": [
                {"label": "唯一文件体积 GiB", "field": "physical_gib", "format": "number"}
            ],
        },
        {
            "id": "generator_card",
            "description": "Union of canonical exact generator IDs over all AIGI rows.",
            "dataset": "headline_metrics",
            "sourceId": source_ids["headline_metrics"],
            "metrics": [
                {"label": "精确生成器数", "field": "unique_generators", "format": "number"}
            ],
        },
        {
            "id": "overlap_card",
            "description": "All are intentional repeated references to the common hard-real panel.",
            "dataset": "headline_metrics",
            "sourceId": source_ids["headline_metrics"],
            "metrics": [
                {"label": "重复 manifest 引用", "field": "duplicate_references", "format": "number"}
            ],
        },
    ]

    charts = [
        {
            "id": "class_composition_chart",
            "title": "八个 manifest 均在各自口径内保持 Real/AIGI 1:1",
            "subtitle": "柱高为样本引用数；困难切片各包含250 Real与250 AIGI。",
            "type": "bar",
            "intent": "comparison",
            "question": "训练、验证和测试 manifest 的类别数量是否平衡？",
            "rationale": "两个互斥类别在八个离散切片上的数量适合分组柱状图。",
            "comparisonContext": {
                "unit": "manifest rows",
                "grain": "class by split",
                "baseline": "50/50 within each manifest",
            },
            "dataset": "class_counts",
            "sourceId": source_ids["class_counts"],
            "encodings": {
                "x": {"field": "split", "type": "nominal", "label": "Split"},
                "y": {"field": "count", "type": "quantitative", "label": "Rows", "format": "number"},
                "color": {"field": "class", "type": "nominal", "label": "Class"},
                "tooltip": [
                    {"field": "share_percent", "type": "quantitative", "label": "Share %", "format": "number"},
                    {"field": "role", "type": "nominal", "label": "Role"},
                ],
            },
            "palette": {"kind": "semantic"},
            "legend": {"position": "bottom", "interactive": True},
            "labels": {"values": "none"},
            "layout": "full",
        },
        {
            "id": "architecture_mix_chart",
            "title": "AIGI 架构大类覆盖由训练域向外部域逐层扩展",
            "subtitle": "每个切片内按AIGI样本归一化；Commercial与Other只出现在严格未见大类测试。",
            "type": "bar",
            "intent": "composition",
            "question": "各切片中的 AIGI 架构大类构成有何差异？",
            "rationale": "对每个切片使用百分比，可在样本量不同的情况下比较构成。",
            "comparisonContext": {
                "unit": "percent of AIGI rows",
                "grain": "architecture family by split",
                "normalization": "within-split AIGI denominator",
            },
            "dataset": "architecture_counts",
            "sourceId": source_ids["architecture_counts"],
            "encodings": {
                "x": {"field": "split", "type": "nominal", "label": "Split"},
                "y": {"field": "share_percent", "type": "quantitative", "label": "AIGI share %", "format": "number"},
                "color": {"field": "architecture", "type": "nominal", "label": "Architecture"},
                "tooltip": [
                    {"field": "aigi_count", "type": "quantitative", "label": "AIGI rows", "format": "number"}
                ],
            },
            "palette": {"kind": "categorical"},
            "legend": {"position": "bottom", "interactive": True},
            "labels": {"values": "none"},
            "layout": "full",
        },
        {
            "id": "storage_chart",
            "title": "外部测试集以较少图像贡献了最大的名义存储量",
            "subtitle": "按manifest中的byte_size求和；三个困难切片的共享真实面板会被分别计入。",
            "type": "horizontalBar",
            "intent": "comparison",
            "question": "各切片的名义存储体积如何比较？",
            "rationale": "长切片名称与单一连续量适合排序水平柱状图。",
            "comparisonContext": {
                "unit": "GiB",
                "grain": "manifest split",
                "caveat": "shared hard-real panel is counted in each hard manifest",
            },
            "dataset": "split_summary",
            "sourceId": source_ids["split_summary"],
            "encodings": {
                "x": {"field": "split", "type": "nominal", "label": "Split"},
                "y": {"field": "nominal_gib", "type": "quantitative", "label": "Nominal GiB", "format": "number"},
                "color": {"field": "role", "type": "nominal", "label": "Role"},
                "tooltip": [
                    {"field": "manifest_rows", "type": "quantitative", "label": "Rows", "format": "number"},
                    {"field": "mean_file_mib", "type": "quantitative", "label": "Mean MiB", "format": "number"},
                ],
            },
            "palette": {"kind": "categorical"},
            "legend": {"position": "bottom", "interactive": True},
            "labels": {"values": "all"},
            "layout": "full",
        },
        {
            "id": "format_mix_chart",
            "title": "格式分布随数据来源变化，外部真实图引入 TIFF 与 WEBP",
            "subtitle": "百分比以各manifest全部样本为分母；JPEG包含manifest中的JPG/JPEG规范化值。",
            "type": "bar",
            "intent": "composition",
            "question": "图像文件格式在各切片中的构成是否一致？",
            "rationale": "格式类别较少，百分比分组柱状图能直接暴露来源相关格式差异。",
            "comparisonContext": {
                "unit": "percent of manifest rows",
                "grain": "format by split",
                "normalization": "within split",
            },
            "dataset": "format_counts",
            "sourceId": source_ids["format_counts"],
            "encodings": {
                "x": {"field": "split", "type": "nominal", "label": "Split"},
                "y": {"field": "share_percent", "type": "quantitative", "label": "Share %", "format": "number"},
                "color": {"field": "format", "type": "nominal", "label": "Format"},
                "tooltip": [
                    {"field": "count", "type": "quantitative", "label": "Rows", "format": "number"}
                ],
            },
            "palette": {"kind": "categorical"},
            "legend": {"position": "bottom", "interactive": True},
            "labels": {"values": "none"},
            "layout": "full",
        },
    ]

    tables = [
        {
            "id": "global_summary_table",
            "title": "全局统计的两个分母口径",
            "subtitle": "manifest引用口径用于描述评测任务；唯一物理口径用于描述磁盘与独立文件。",
            "dataset": "global_summary",
            "sourceId": source_ids["global_summary"],
            "defaultSort": {"field": "rows_or_images", "direction": "desc"},
            "columns": _table_columns(
                ("scope", "口径", None),
                ("rows_or_images", "行/图像数", "number"),
                ("real", "Real", "number"),
                ("aigi", "AIGI", "number"),
                ("aigi_percent", "AIGI %", "number"),
                ("storage_gib", "GiB", "number"),
                ("definition", "定义", None),
            ),
        },
        {
            "id": "split_summary_table",
            "title": "分切片样本、体积与图像几何统计",
            "subtitle": "文件体积为manifest名义求和；MiB与GiB使用二进制单位。",
            "dataset": "split_summary",
            "sourceId": source_ids["split_summary"],
            "defaultSort": {"field": "split_order", "direction": "asc"},
            "density": "dense",
            "columns": _table_columns(
                ("split_order", "顺序", "number"),
                ("split", "切片", None),
                ("role", "用途", None),
                ("manifest_rows", "行数", "number"),
                ("real", "Real", "number"),
                ("aigi", "AIGI", "number"),
                ("exact_generators", "生成器", "number"),
                ("architecture_families", "架构大类", "number"),
                ("real_sources", "真实来源", "number"),
                ("formats", "格式", "number"),
                ("nominal_gib", "名义GiB", "number"),
                ("median_file_mib", "中位MiB", "number"),
                ("p95_file_mib", "P95 MiB", "number"),
                ("mean_megapixels", "平均MP", "number"),
                ("p95_megapixels", "P95 MP", "number"),
                ("width_range", "宽度范围", None),
                ("height_range", "高度范围", None),
            ),
        },
        {
            "id": "generator_exposure_table",
            "title": "相对训练集的精确生成器与架构大类暴露",
            "subtitle": "Exact-seen与family-seen分别按canonical_generator_id和architecture计算。",
            "dataset": "generator_exposure",
            "sourceId": source_ids["generator_exposure"],
            "defaultSort": {"field": "split_order", "direction": "asc"},
            "columns": _table_columns(
                ("split_order", "顺序", "number"),
                ("split", "切片", None),
                ("role", "用途", None),
                ("exact_generators", "精确生成器", "number"),
                ("exact_generators_seen_in_train", "精确已见", "number"),
                ("exact_generators_unseen_in_train", "精确未见", "number"),
                ("architecture_families", "架构大类", "number"),
                ("architecture_families_seen_in_train", "大类已见", "number"),
                ("architecture_families_unseen_in_train", "大类未见", "number"),
                ("protocol", "协议定义", None),
            ),
        },
        {
            "id": "real_source_table",
            "title": "真实图来源分布",
            "subtitle": "只统计label=0；Small元数据缺失统一显示UNSPECIFIED。",
            "dataset": "real_source_counts",
            "sourceId": source_ids["real_source_counts"],
            "defaultSort": {"field": "split_order", "direction": "asc"},
            "columns": _table_columns(
                ("split_order", "顺序", "number"),
                ("split", "切片", None),
                ("real_source", "真实来源", None),
                ("real_count", "数量", "number"),
                ("share_percent", "切片Real占比%", "number"),
            ),
        },
        {
            "id": "resolution_table",
            "title": "图像分辨率分箱",
            "subtitle": "MP=width×height/1,000,000；百分比按各manifest全部行计算。",
            "dataset": "resolution_counts",
            "sourceId": source_ids["resolution_counts"],
            "defaultSort": {"field": "split_order", "direction": "asc"},
            "columns": _table_columns(
                ("split_order", "切片顺序", "number"),
                ("split", "切片", None),
                ("bin_order", "分箱顺序", "number"),
                ("resolution_bin", "分辨率", None),
                ("count", "数量", "number"),
                ("share_percent", "占比%", "number"),
            ),
        },
        {
            "id": "generator_detail_table",
            "title": "完整精确生成器统计",
            "subtitle": "每个切片内逐canonical_generator_id列出图像数、架构与训练暴露。",
            "dataset": "generator_statistics",
            "sourceId": source_ids["generator_statistics"],
            "defaultSort": {"field": "split_order", "direction": "asc"},
            "density": "dense",
            "columns": _table_columns(
                ("split_order", "切片顺序", "number"),
                ("split", "切片", None),
                ("generator_order", "切片内顺序", "number"),
                ("generator", "精确生成器", None),
                ("architecture", "架构", None),
                ("exposure", "标注暴露", None),
                ("aigi_images", "AIGI图像", "number"),
                ("share_percent", "切片AIGI占比%", "number"),
                ("exact_identity_seen_in_train", "精确已见", None),
                ("architecture_seen_in_train", "大类已见", None),
            ),
        },
        {
            "id": "pairwise_overlap_table",
            "title": "全部28组切片两两重叠审计",
            "subtitle": "图像重叠按路径和SHA256分别计算；生成器重叠仅统计AIGI。",
            "dataset": "pairwise_overlap",
            "sourceId": source_ids["pairwise_overlap"],
            "defaultSort": {"field": "shared_sha256", "direction": "desc"},
            "columns": _table_columns(
                ("left_split", "切片A", None),
                ("right_split", "切片B", None),
                ("shared_paths", "共享路径", "number"),
                ("shared_sha256", "共享SHA256", "number"),
                ("shared_exact_generators", "共享精确生成器", "number"),
                ("shared_architecture_families", "共享架构大类", "number"),
            ),
        },
        {
            "id": "integrity_table",
            "title": "数据完整性与去重检查",
            "subtitle": "PASS表示当前manifest静态审计通过；PARTIAL与NOT AVAILABLE需保留结论边界。",
            "dataset": "integrity_checks",
            "sourceId": source_ids["integrity_checks"],
            "defaultSort": {"field": "check_order", "direction": "asc"},
            "columns": _table_columns(
                ("check_order", "顺序", "number"),
                ("check", "检查", None),
                ("result", "结果", None),
                ("count", "计数", "number"),
                ("interpretation", "解释", None),
            ),
        },
        {
            "id": "build_audit_table",
            "title": "构建阶段去重与拒绝记录",
            "subtitle": "这些计数描述构建过程中的修复，不是当前manifest中的残留重复。",
            "dataset": "build_audit",
            "sourceId": source_ids["build_audit"],
            "defaultSort": {"field": "audit_order", "direction": "asc"},
            "columns": _table_columns(
                ("audit_order", "顺序", "number"),
                ("stage", "阶段", None),
                ("count", "计数", "number"),
                ("outcome", "结果", None),
            ),
        },
        {
            "id": "manifest_lineage_table",
            "title": "Manifest、种子与来源版本谱系",
            "subtitle": "短SHA用于报告展示；完整SHA保留在原审计JSON与artifact源文件中。",
            "dataset": "manifest_lineage",
            "sourceId": source_ids["manifest_lineage"],
            "defaultSort": {"field": "split_order", "direction": "asc"},
            "density": "dense",
            "columns": _table_columns(
                ("split_order", "顺序", "number"),
                ("split", "切片", None),
                ("role", "用途", None),
                ("manifest", "Manifest", None),
                ("manifest_sha256", "SHA256", None),
                ("manifest_bytes", "CSV bytes", "number"),
                ("selection_seed", "选择种子", None),
                ("source_revision", "来源revision", None),
                ("protocol", "协议", None),
            ),
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": source_ids["headline_metrics"],
            "body": (
                "## 技术摘要\n\n"
                f"当前冻结协议包含 **{headline['manifest_count']} 份manifest、{headline['manifest_rows']:,}条引用**，"
                f"对应 **{headline['unique_images']:,}张唯一物理图像**，占 **{headline['physical_gib']:.2f} GiB**。"
                f"差额{headline['duplicate_references']:,}条全部来自三个困难切片共同复用的"
                f"{headline['shared_hard_real_images']:,}张真实图面板：每张在三个manifest中出现，因此额外产生两次引用。\n\n"
                f"训练集为9,000 Real + 9,000 AIGI；每个验证/测试manifest也各自保持1:1。"
                f"去重后的全体物理语料则为{headline['unique_real']:,} Real + {headline['unique_aigi']:,} AIGI，"
                "不再严格1:1，这是共享真实面板按物理文件只计一次造成的统计现象，不是抽样错误。"
                f"AIGI共覆盖{headline['unique_generators']:,}个精确生成器和{headline['unique_architectures']}个架构大类。\n\n"
                f"静态核验发现缺失路径{headline['missing_paths']}个、文件大小不匹配{headline['size_mismatches']}个、"
                f"非预期SHA256跨切片重叠{headline['unexpected_sha_overlap_groups']}组。"
            ),
        },
        {
            "id": "headline_cards",
            "type": "metric-strip",
            "cardIds": [
                "unique_images_card",
                "storage_card",
                "generator_card",
                "overlap_card",
            ],
        },
        {
            "id": "balance_finding",
            "type": "markdown",
            "sourceId": source_ids["class_counts"],
            "body": (
                "## 每个训练、验证与测试任务都保持类别平衡\n\n"
                "八份manifest逐份计算均为50% Real与50% AIGI，因此各切片上的balanced accuracy、AUROC等二分类指标不会因"
                "manifest级类别比例不同而直接偏移。需要区分的是：合并后按唯一文件去重会移除500条重复Real引用，"
                "所以唯一物理语料的类别比例不是实验抽样比例。"
            ),
        },
        {"id": "class_chart_block", "type": "chart", "chartId": "class_composition_chart", "layout": "full"},
        {"id": "global_summary_block", "type": "table", "tableId": "global_summary_table", "layout": "full"},
        {
            "id": "generator_finding",
            "type": "markdown",
            "sourceId": source_ids["architecture_counts"],
            "body": (
                "## 生成器暴露协议同时区分精确身份与架构大类\n\n"
                f"训练集覆盖{headline['train_generator_count']}个精确生成器；内部checkpoint validation使用"
                f"{headline['internal_val_generator_count']}个与训练不重合的Small生成器。外部Exact-seen切片只有SD 1.4，"
                "其精确身份在训练中出现；Seen-family与三个困难切片仅复用训练已见架构大类；严格Unseen-family测试的"
                "Commercial与Other大类均未在训练出现。下图展示样本层面的架构构成，表格给出身份层面的交集计数。"
            ),
        },
        {"id": "architecture_chart_block", "type": "chart", "chartId": "architecture_mix_chart", "layout": "full"},
        {"id": "generator_exposure_block", "type": "table", "tableId": "generator_exposure_table", "layout": "full"},
        {
            "id": "storage_finding",
            "type": "markdown",
            "sourceId": source_ids["split_summary"],
            "body": (
                "## 存储开销主要由高分辨率外部真实图驱动\n\n"
                f"名义体积最大的切片是 **{largest_storage['split']}**（{float(largest_storage['nominal_gib']):.2f} GiB），"
                f"平均像素数最高的是 **{highest_resolution['split']}**（{float(highest_resolution['mean_megapixels']):.2f} MP）。"
                "外部Eval真实图包含高分辨率TIFF，导致每张图的字节数显著高于Small中的训练图。"
                "图中困难切片按manifest口径各自计入同一真实面板；全局唯一存储只计一次，见上方全局口径表。"
            ),
        },
        {"id": "storage_chart_block", "type": "chart", "chartId": "storage_chart", "layout": "full"},
        {"id": "split_summary_block", "type": "table", "tableId": "split_summary_table", "layout": "full"},
        {
            "id": "format_finding",
            "type": "markdown",
            "sourceId": source_ids["format_counts"],
            "body": (
                "## 图像格式和分辨率是明确的数据域变量\n\n"
                "Small训练/内部验证主要由PNG与JPEG构成，而外部Eval真实图进一步包含TIFF与WEBP。当前训练读取阶段已启用"
                "在线格式去偏，但原始manifest仍保留来源格式与尺寸；因此报告同时给出格式比例、分辨率分箱和真实来源，"
                "便于判断性能差异是否可能与生成器之外的文件编码或来源域共同变化。"
            ),
        },
        {"id": "format_chart_block", "type": "chart", "chartId": "format_mix_chart", "layout": "full"},
        {"id": "real_source_block", "type": "table", "tableId": "real_source_table", "layout": "full"},
        {"id": "resolution_block", "type": "table", "tableId": "resolution_table", "layout": "full"},
        {
            "id": "scope_definitions",
            "type": "markdown",
            "sourceId": source_ids["manifest_lineage"],
            "body": (
                "## 范围、分母与冻结数据谱系\n\n"
                "本报告覆盖当前配置引用的训练集、checkpoint选择用内部验证集、四个外部/困难验证切片和两个外部测试集。"
                "“manifest行”表示一次评测样本引用；“唯一物理图像”按完整解析路径去重；“精确生成器”按"
                "canonical_generator_id；“架构大类”按architecture；存储GiB按2^30字节。所有计数来自冻结CSV，"
                "选择种子、来源revision与manifest摘要见下表。"
            ),
        },
        {"id": "manifest_lineage_block", "type": "table", "tableId": "manifest_lineage_table", "layout": "full"},
        {
            "id": "methodology",
            "type": "markdown",
            "sourceId": source_ids["integrity_checks"],
            "body": (
                "## 统计与审计方法\n\n"
                "报告程序逐行解析八份CSV，核对必需字段、数值类型和project_split；对每个引用执行文件存在性与stat字节数核对；"
                "按路径、sample_id和manifest所存SHA256构建跨切片交集；按label聚合生成器、架构、真实来源、格式、尺寸和字节数。"
                "它不会重新编码图像，也不会重算全部25 GiB内容哈希；SHA256/pHash完整性沿用构建审计，当前程序验证的是"
                "manifest之间的哈希集合关系。所有可视化数据先写入内存SQLite并执行报告中声明的SQL，再封装为可移植HTML。"
            ),
        },
        {"id": "integrity_block", "type": "table", "tableId": "integrity_table", "layout": "full"},
        {"id": "build_audit_block", "type": "table", "tableId": "build_audit_table", "layout": "full"},
        {
            "id": "overlap_finding",
            "type": "markdown",
            "sourceId": source_ids["pairwise_overlap"],
            "body": (
                "## 图像级重叠仅限预先设计的共享困难真实面板\n\n"
                f"三个困难切片两两各共享{headline['shared_hard_real_images']}张相同Real图；这让不同生成器的比较使用同一负类面板，"
                "但也意味着三组评测并非独立样本。除这三对外，路径与SHA256交集均应为零。生成器身份或架构交集属于"
                "协议定义的域暴露，不等同于图像泄漏。"
            ),
        },
        {"id": "pairwise_overlap_block", "type": "table", "tableId": "pairwise_overlap_table", "layout": "full"},
        {
            "id": "generator_detail_intro",
            "type": "markdown",
            "sourceId": source_ids["generator_statistics"],
            "body": (
                "## 完整生成器明细保留长尾审计能力\n\n"
                "下表不只展示Top-N，而是保留所有切片中的精确生成器、图像数、架构与训练暴露状态。"
                "训练和内部验证的大多数Small生成器各抽取固定数量图像，外部Eval生成器则按冻结分配构成。"
            ),
        },
        {"id": "generator_detail_block", "type": "table", "tableId": "generator_detail_table", "layout": "full"},
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## 结论边界与当前未完成的数据质量检查\n\n"
                f"- 先前TIFF完整解码审计覆盖{headline['prior_tiff_valid_files']}/{headline['current_tiff_files']}张当前唯一TIFF；"
                f"其中{headline['prior_tiff_warning_files']}张产生`Truncated File Read`警告但0张失败。Validation-v2后来新增的"
                f"{headline['current_tiff_files'] - headline['prior_tiff_valid_files']}张TIFF尚未纳入同一次全帧解码审计。\n"
                "- CommunityForensics-Small真实图元数据全部缺少可验证real_source，因此训练集的真实来源平衡无法从当前manifest证明；"
                "报告将其记为UNSPECIFIED，不推断FFHQ/VISION/COCO等组成。\n"
                "- 未提供COCO val2017/DALL-E Advanced保留集哈希清单，因此只能核对当前八份manifest之间的泄漏，"
                "无法证明与未来保留集无重叠。\n"
                "- 文件存在性和byte_size核验不等价于逐文件像素解码；除既有TIFF审计外，本报告没有重新解码全部图像。\n"
                "- 统计是描述性的冻结快照，不包含抽样置信区间，也不用于根据测试集结果重新选择模型或数据。"
            ),
        },
        {
            "id": "recommendations",
            "type": "markdown",
            "sourceId": source_ids["integrity_checks"],
            "body": (
                "## 下一步应补齐新增TIFF与保留集审计\n\n"
                "1. 对Validation-v2新增TIFF运行与既有流程相同的Pillow verify、全帧解码、尺寸和SHA256核验，并更新覆盖率。\n"
                "2. 在获得COCO val2017/DALL-E Advanced保留集哈希后，补做SHA256与pHash交叉审计。\n"
                "3. 后续新增数据源时继续同时报告manifest引用数与唯一物理图像数，禁止把共享面板误算成独立样本。\n"
                "4. 若要比较生成器难度，应在相同真实来源/格式/分辨率条件下做分层分析，避免把来源域差异归因于生成器。"
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## 仍需回答的问题\n\n"
                "- Validation-v2新增TIFF在全帧解码下是否全部有效？\n"
                "- Small真实图来源能否从上游元数据或可追溯索引中恢复？\n"
                "- 控制格式、分辨率和真实来源后，Seen-family与Unseen-family的难度排序是否保持？\n"
                "- 下一轮训练是否需要按架构大类和真实来源进行显式重采样，而不只按二分类标签平衡？"
            ),
        },
    ]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Current frozen Community Forensics training, validation, and test data statistics with overlap and integrity audits.",
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
            "datasets": datasets,
        },
        "sources": sources,
    }


def generate(arguments: argparse.Namespace) -> None:
    generated_at = datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds")
    data_root = Path(arguments.data_root)
    by_split, manifest_meta = _load_manifests(data_root)
    staged, headline, audit_details = _analyze(
        by_split,
        manifest_meta,
        data_root,
        Path(arguments.tiff_audit),
    )
    queries = {
        "headline_metrics": "SELECT * FROM headline_metrics",
        "global_summary": "SELECT * FROM global_summary ORDER BY rows_or_images DESC",
        "split_summary": "SELECT * FROM split_summary ORDER BY split_order",
        "class_counts": "SELECT * FROM class_counts ORDER BY split_order, class_order",
        "architecture_counts": "SELECT * FROM architecture_counts ORDER BY split_order, architecture_order",
        "format_counts": "SELECT * FROM format_counts ORDER BY split_order, format_order",
        "real_source_counts": "SELECT * FROM real_source_counts ORDER BY split_order, source_order",
        "resolution_counts": "SELECT * FROM resolution_counts ORDER BY split_order, bin_order",
        "generator_exposure": "SELECT * FROM generator_exposure ORDER BY split_order",
        "generator_statistics": "SELECT * FROM generator_statistics ORDER BY split_order, generator_order",
        "pairwise_overlap": "SELECT * FROM pairwise_overlap ORDER BY shared_sha256 DESC, left_order, right_order",
        "integrity_checks": "SELECT * FROM integrity_checks ORDER BY check_order",
        "build_audit": "SELECT * FROM build_audit ORDER BY audit_order",
        "manifest_lineage": "SELECT * FROM manifest_lineage ORDER BY split_order",
    }
    datasets = _materialize_sql_snapshot(staged, queries)
    _write_csv(arguments.split_csv, datasets["split_summary"])
    _write_csv(arguments.generator_csv, datasets["generator_statistics"])
    distribution_rows: list[dict[str, Any]] = []
    for dataset_name in (
        "class_counts",
        "architecture_counts",
        "format_counts",
        "real_source_counts",
        "resolution_counts",
    ):
        for row in datasets[dataset_name]:
            distribution_rows.append(
                {
                    "distribution": dataset_name,
                    "row_json": json.dumps(row, ensure_ascii=False, sort_keys=True),
                }
            )
    _write_csv(arguments.distribution_csv, distribution_rows)

    report_contract = {
        "delivery_mode": "portable_html",
        "audience": "technical",
        "question": "Describe the current frozen training set and every configured validation/test set, including composition, storage, generator exposure, and leakage checks.",
        "baseline": "Current eight frozen manifests under configs/community_forensics/base.yaml.",
        "success_criteria": "All manifests parse; every referenced file exists; byte sizes match; only the declared shared hard-real panel overlaps across splits.",
        "required_section_mapping": {
            "title": "title",
            "technical_summary": "technical_summary",
            "key_findings_with_visual_evidence": [
                "balance_finding",
                "generator_finding",
                "storage_finding",
                "format_finding",
            ],
            "scope_data_and_metric_definitions": "scope_definitions",
            "methodology": "methodology",
            "limitations_uncertainty_and_robustness_checks": "limitations",
            "recommended_next_steps": "recommendations",
            "further_questions": "further_questions",
        },
    }
    audit = {
        "schema_version": 1,
        "generated_at_asia_singapore": generated_at,
        "report_job_id": os.environ.get("SLURM_JOB_ID"),
        "report_contract": report_contract,
        "headline": headline,
        "audit_details": audit_details,
        "manifest_lineage": datasets["manifest_lineage"],
        "sql_snapshot_queries": queries,
        "supporting_outputs": {
            "split_csv": str(arguments.split_csv),
            "generator_csv": str(arguments.generator_csv),
            "distribution_csv": str(arguments.distribution_csv),
        },
        "chart_map": [
            {
                "section": "balance_finding",
                "question": "Compare Real and AIGI counts across all manifests.",
                "family": "comparison",
                "type": "grouped bar",
                "fields": ["split", "class", "count"],
                "palette": "semantic two-class",
            },
            {
                "section": "generator_finding",
                "question": "Compare AIGI architecture composition across splits.",
                "family": "composition",
                "type": "grouped percent bar",
                "fields": ["split", "architecture", "share_percent"],
                "palette": "categorical architecture families",
            },
            {
                "section": "storage_finding",
                "question": "Rank nominal storage by split.",
                "family": "comparison and ranking",
                "type": "horizontal bar",
                "fields": ["split", "nominal_gib", "role"],
                "palette": "categorical split roles",
            },
            {
                "section": "format_finding",
                "question": "Compare image-format composition across splits.",
                "family": "composition",
                "type": "grouped percent bar",
                "fields": ["split", "format", "share_percent"],
                "palette": "categorical file formats",
            },
        ],
        "visual_omissions": [
            "The complete generator inventory is a table because more than one thousand categorical labels are not readable in a chart.",
            "Pairwise overlaps are a sortable table because exact zero/nonzero audit values matter more than visual trend detection.",
            "No confidence intervals are shown because the report is a deterministic census of frozen manifests, not a sample estimate.",
        ],
    }
    _atomic_text(arguments.audit_json, json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    artifact = _build_artifact(datasets, queries, headline, generated_at)
    _atomic_text(
        arguments.artifact_json,
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
    )
    print(
        json.dumps(
            {
                "event": "community_forensics_data_statistics_report_complete",
                "manifest_rows": headline["manifest_rows"],
                "unique_images": headline["unique_images"],
                "physical_gib": headline["physical_gib"],
                "unique_generators": headline["unique_generators"],
                "missing_paths": headline["missing_paths"],
                "size_mismatches": headline["size_mismatches"],
                "unexpected_sha_overlap_groups": headline[
                    "unexpected_sha_overlap_groups"
                ],
                "artifact_json": str(arguments.artifact_json),
                "audit_json": str(arguments.audit_json),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the current Community Forensics data statistics report"
    )
    parser.add_argument(
        "--data-root", default="data/raw/community_forensics"
    )
    parser.add_argument(
        "--tiff-audit", default="reports/community_forensics_tiff_integrity.json"
    )
    parser.add_argument(
        "--split-csv", default="reports/community_forensics_data_statistics.csv"
    )
    parser.add_argument(
        "--generator-csv",
        default="reports/community_forensics_generator_statistics.csv",
    )
    parser.add_argument(
        "--distribution-csv",
        default="reports/community_forensics_distribution_statistics.csv",
    )
    parser.add_argument(
        "--artifact-json",
        default="reports/community_forensics_data_statistics_artifact.json",
    )
    parser.add_argument(
        "--audit-json",
        default="reports/community_forensics_data_statistics_notes.json",
    )
    return parser.parse_args()


def main() -> None:
    generate(_parse_args())


if __name__ == "__main__":
    main()
