from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import signal
import sqlite3
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = False

SELECTION_SEED = 20260828
SMALL_DATASET_ID = "OwensLab/CommunityForensics-Small"
EVAL_DATASET_ID = "OwensLab/CommunityForensics-Eval"
SMALL_REVISION = "6c539a534c07917307c381f5af4053c6091b5278"
EVAL_REVISION = "7d4a74a88d2cac93b513c0853bf92c260eaceea0"

SEEN_FAMILY_DEFINITION = (
    "architecture_family_seen_exact_generator_unseen_in_small_train_and_validation"
)
UNSEEN_GENERATOR_DEFINITION = (
    "architecture_family_unseen_exact_generator_unseen_in_small_train_and_validation"
)

DATASET_SPECS = {
    "small": {
        "dataset_id": SMALL_DATASET_ID,
        "revision": SMALL_REVISION,
        "prefix": "data/",
        "source_dataset": "community-forensics-small",
    },
    "eval": {
        "dataset_id": EVAL_DATASET_ID,
        "revision": EVAL_REVISION,
        "prefix": "data/CompEval-",
        "source_dataset": "community-forensics-eval",
    },
}

METADATA_COLUMNS = (
    "image_name",
    "format",
    "resolution",
    "mode",
    "model_name",
    "nsfw_flag",
    "prompt",
    "real_source",
    "subset",
    "split",
    "label",
    "architecture",
)

PROJECT_SPLITS = (
    "train",
    "val_unseen_generator",
    "test_external_seen_family",
    "test_external_unseen_generator",
)

EXPECTED_COUNTS = {
    "train": {0: 9_000, 1: 9_000},
    "val_unseen_generator": {0: 1_000, 1: 1_000},
    "test_external_seen_family": {0: 1_000, 1: 1_000},
    "test_external_unseen_generator": {0: 1_000, 1: 1_000},
}

MANIFEST_FIELDS = (
    "sample_id",
    "path",
    "label",
    "split",
    "source_dataset",
    "generator_id",
    "official_split",
    "project_split",
    "real_source",
    "model_name_raw",
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
)

FORMAT_EXTENSIONS = {
    "BMP": ".bmp",
    "GIF": ".gif",
    "JPEG": ".jpg",
    "JPG": ".jpg",
    "PNG": ".png",
    "TIFF": ".tiff",
    "WEBP": ".webp",
}

_STOP_REQUESTED = False


def _request_safe_stop(signum: int, frame: object) -> None:
    del signum, frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _stable_digest(seed: int, *values: object) -> str:
    payload = "\x1f".join((str(seed), *(str(value) for value in values)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_generator_id(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    text = re.sub(r"^https?://huggingface\.co/", "", text)
    text = text.replace("\\", "/")
    parts = []
    for part in text.split("/"):
        normalized = re.sub(r"[^a-z0-9.+-]+", "-", part).strip("-.")
        if normalized:
            parts.append(normalized)
    return "/".join(parts)


def canonical_real_source(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    compact = re.sub(r"[^a-z0-9]+", "", text)
    if "landscape" in compact:
        return "Landscapes HQ"
    if "ffhq" in compact:
        return "FFHQ"
    if "vision" in compact:
        return "VISION"
    if "raise" in compact:
        return "RAISE"
    if "laion" in compact:
        return "LAION"
    if "coco" in compact:
        return "COCO"
    return ""


def _safe_stem(value: object) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    return stem[:80] or "image"


def _extension(value: object) -> str:
    return FORMAT_EXTENSIONS.get(str(value or "").upper(), ".img")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(payload: bytes, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _atomic_text(text: str, destination: Path) -> None:
    _atomic_bytes(text.encode("utf-8"), destination)


def _atomic_json(payload: dict[str, Any], destination: Path) -> None:
    _atomic_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", destination)


def _atomic_csv(rows: Sequence[dict[str, Any]], fields: Sequence[str], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _inspect_image(payload: bytes) -> tuple[int, int, str]:
    with Image.open(io.BytesIO(payload)) as image:
        image.verify()
    with Image.open(io.BytesIO(payload)) as image:
        width, height = image.size
        image_format = str(image.format or "").upper()
        image.convert("RGB").getpixel((0, 0))
    if width <= 0 or height <= 0 or not image_format:
        raise ValueError("Invalid decoded image")
    return width, height, image_format


def perceptual_hash(payload: bytes, size: int = 8, high_frequency_factor: int = 4) -> str:
    sample_size = int(size) * int(high_frequency_factor)
    with Image.open(io.BytesIO(payload)) as image:
        gray = image.convert("L").resize((sample_size, sample_size), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.float64)
    positions = np.arange(sample_size, dtype=np.float64)
    frequencies = positions[:, None]
    basis = np.cos(np.pi * (2.0 * positions + 1.0) * frequencies / (2.0 * sample_size))
    basis[0] *= math.sqrt(1.0 / sample_size)
    basis[1:] *= math.sqrt(2.0 / sample_size)
    coefficients = basis @ pixels @ basis.T
    low = coefficients[:size, :size]
    median = float(np.median(low.flatten()[1:]))
    bits = (low > median).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return f"{value:0{size * size // 4}x}"


def phash_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=120)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS dataset_versions (
            dataset_key TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            revision TEXT NOT NULL,
            parquet_files INTEGER NOT NULL,
            parquet_bytes INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS source_rows (
            dataset_key TEXT NOT NULL,
            source_file TEXT NOT NULL,
            row_group INTEGER NOT NULL,
            row_index INTEGER NOT NULL,
            image_name TEXT NOT NULL,
            source_format TEXT NOT NULL,
            resolution_json TEXT NOT NULL,
            mode TEXT NOT NULL,
            model_name_raw TEXT NOT NULL,
            canonical_generator_id TEXT NOT NULL,
            nsfw_flag TEXT NOT NULL,
            prompt TEXT NOT NULL,
            real_source_raw TEXT NOT NULL,
            canonical_real_source TEXT NOT NULL,
            subset_name TEXT NOT NULL,
            official_split TEXT NOT NULL,
            label INTEGER NOT NULL,
            architecture TEXT NOT NULL,
            PRIMARY KEY (dataset_key, source_file, row_group, row_index)
        );
        CREATE TABLE IF NOT EXISTS scanned_files (
            dataset_key TEXT NOT NULL,
            source_file TEXT NOT NULL,
            rows INTEGER NOT NULL,
            completed_at_utc TEXT NOT NULL,
            PRIMARY KEY (dataset_key, source_file)
        );
        CREATE TABLE IF NOT EXISTS selection (
            sample_id TEXT PRIMARY KEY,
            dataset_key TEXT NOT NULL,
            source_file TEXT NOT NULL,
            row_group INTEGER NOT NULL,
            row_index INTEGER NOT NULL,
            image_name TEXT NOT NULL,
            source_format TEXT NOT NULL,
            model_name_raw TEXT NOT NULL,
            canonical_generator_id TEXT NOT NULL,
            real_source TEXT NOT NULL,
            official_split TEXT NOT NULL,
            project_split TEXT NOT NULL,
            label INTEGER NOT NULL,
            architecture TEXT NOT NULL,
            generator_exposure TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'planned',
            sha256 TEXT,
            phash TEXT,
            width INTEGER,
            height INTEGER,
            actual_format TEXT,
            byte_size INTEGER,
            UNIQUE (dataset_key, source_file, row_group, row_index),
            UNIQUE (relative_path)
        );
        CREATE INDEX IF NOT EXISTS source_label_generator
            ON source_rows(dataset_key, label, canonical_generator_id);
        CREATE INDEX IF NOT EXISTS source_label_real
            ON source_rows(dataset_key, label, canonical_real_source);
        CREATE INDEX IF NOT EXISTS selection_status_group
            ON selection(status, dataset_key, source_file, row_group);
        """
    )
    return connection


def _resolved_files(dataset_key: str) -> tuple[str, list[tuple[str, int]]]:
    specification = DATASET_SPECS[dataset_key]
    info = HfApi().dataset_info(
        specification["dataset_id"],
        revision=specification["revision"],
        files_metadata=True,
    )
    revision = str(info.sha or "")
    if revision != specification["revision"]:
        raise RuntimeError(
            f"Resolved {dataset_key} revision changed: {revision} != {specification['revision']}"
        )
    files = sorted(
        (
            str(sibling.rfilename),
            int(sibling.size or 0),
        )
        for sibling in (info.siblings or [])
        if str(sibling.rfilename).startswith(str(specification["prefix"]))
        and str(sibling.rfilename).endswith(".parquet")
    )
    if not files:
        raise RuntimeError(f"No Parquet files found for {dataset_key}")
    return revision, files


def _source_value(value: Any) -> Any:
    return value.as_py() if hasattr(value, "as_py") else value


def scan_metadata(connection: sqlite3.Connection) -> int:
    filesystem = HfFileSystem()
    for dataset_key in ("small", "eval"):
        specification = DATASET_SPECS[dataset_key]
        revision, files = _resolved_files(dataset_key)
        expected = connection.execute(
            "SELECT * FROM dataset_versions WHERE dataset_key=?", (dataset_key,)
        ).fetchone()
        version_values = (
            dataset_key,
            specification["dataset_id"],
            revision,
            len(files),
            sum(size for _, size in files),
        )
        if expected is None:
            connection.execute(
                "INSERT INTO dataset_versions VALUES (?, ?, ?, ?, ?)", version_values
            )
            connection.commit()
        elif tuple(expected) != version_values:
            raise RuntimeError(f"Dataset version state changed for {dataset_key}: {dict(expected)}")

        completed = {
            str(row[0])
            for row in connection.execute(
                "SELECT source_file FROM scanned_files WHERE dataset_key=?", (dataset_key,)
            )
        }
        for file_index, (source_file, _) in enumerate(files, start=1):
            if source_file in completed:
                continue
            remote_path = (
                f"datasets/{specification['dataset_id']}@{revision}/{source_file}"
            )
            inserted = 0
            rows_to_insert: list[tuple[Any, ...]] = []
            with filesystem.open(remote_path, "rb") as handle:
                parquet = pq.ParquetFile(handle)
                missing = set(METADATA_COLUMNS).difference(parquet.schema_arrow.names)
                if missing:
                    raise RuntimeError(f"{source_file} missing columns: {sorted(missing)}")
                for row_group in range(parquet.metadata.num_row_groups):
                    table = parquet.read_row_group(row_group, columns=list(METADATA_COLUMNS))
                    for row_index in range(len(table)):
                        values = {
                            column: _source_value(table[column][row_index])
                            for column in METADATA_COLUMNS
                        }
                        label = int(values["label"])
                        if label not in (0, 1):
                            continue
                        model_name = str(values["model_name"] or "")
                        real_source_raw = str(values["real_source"] or "")
                        rows_to_insert.append(
                            (
                                dataset_key,
                                source_file,
                                row_group,
                                row_index,
                                str(values["image_name"] or ""),
                                str(values["format"] or ""),
                                json.dumps(values["resolution"]),
                                str(values["mode"] or ""),
                                model_name,
                                canonical_generator_id(model_name) if label == 1 else "",
                                str(values["nsfw_flag"]),
                                str(values["prompt"] or ""),
                                real_source_raw,
                                canonical_real_source(real_source_raw),
                                str(values["subset"] or ""),
                                str(values["split"] or ""),
                                label,
                                str(values["architecture"] or "unknown"),
                            )
                        )
                    inserted += len(table)
            with connection:
                connection.executemany(
                    """
                    INSERT INTO source_rows VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    rows_to_insert,
                )
                connection.execute(
                    "INSERT INTO scanned_files VALUES (?, ?, ?, ?)",
                    (dataset_key, source_file, inserted, datetime.now(timezone.utc).isoformat()),
                )
            print(
                json.dumps(
                    {
                        "event": "community_forensics_metadata_progress",
                        "dataset": dataset_key,
                        "file": source_file,
                        "file_index": file_index,
                        "file_count": len(files),
                        "rows": inserted,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if _STOP_REQUESTED:
                return 75
    return 0


def _row_key(row: sqlite3.Row, seed: int) -> str:
    return _stable_digest(
        seed,
        row["dataset_key"],
        row["source_file"],
        row["row_group"],
        row["row_index"],
        row["image_name"],
    )


def _unique_real_rows(
    rows: Iterable[sqlite3.Row], seed: int, *, allow_unspecified: bool = False
) -> list[sqlite3.Row]:
    by_identity: dict[tuple[str, str], sqlite3.Row] = {}
    for row in rows:
        source = str(row["canonical_real_source"])
        if not source:
            if not allow_unspecified:
                continue
            source = "UNSPECIFIED"
        identity = (source, str(row["image_name"]).strip().lower())
        current = by_identity.get(identity)
        if current is None or _row_key(row, seed) < _row_key(current, seed):
            by_identity[identity] = row
    return list(by_identity.values())


def _allocate_proportional(sizes: dict[str, int], total: int) -> dict[str, int]:
    available = sum(sizes.values())
    if total < 0 or total > available:
        raise ValueError(f"Cannot allocate {total} from {sizes}")
    if total == 0:
        return {key: 0 for key in sizes}
    exact = {key: total * value / available for key, value in sizes.items()}
    allocation = {key: min(sizes[key], math.floor(value)) for key, value in exact.items()}
    remaining = total - sum(allocation.values())
    order = sorted(
        sizes,
        key=lambda key: (exact[key] - math.floor(exact[key]), sizes[key], key),
        reverse=True,
    )
    while remaining:
        progressed = False
        for key in order:
            if allocation[key] < sizes[key]:
                allocation[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("Proportional allocation stalled")
    return allocation


def _stratified_pick(
    candidates: set[str],
    architecture: dict[str, str],
    count: int,
    seed: int,
    forced: set[str] | None = None,
) -> set[str]:
    forced = set(forced or ())
    if not forced.issubset(candidates):
        raise ValueError("Forced generators are not all candidates")
    if len(forced) > count or len(candidates) < count:
        raise ValueError(f"Cannot pick {count} generators from {len(candidates)}")
    picked = set(forced)
    remaining_candidates = candidates.difference(picked)
    groups: dict[str, list[str]] = defaultdict(list)
    for generator in remaining_candidates:
        groups[architecture.get(generator, "unknown")].append(generator)
    for group, values in groups.items():
        values.sort(key=lambda value: _stable_digest(seed, group, value))
    allocation = _allocate_proportional(
        {group: len(values) for group, values in groups.items()}, count - len(picked)
    )
    for group, amount in allocation.items():
        picked.update(groups[group][:amount])
    if len(picked) != count:
        raise RuntimeError(f"Stratified selection produced {len(picked)} != {count}")
    return picked


def _balanced_group_rows(
    rows_by_group: dict[str, list[sqlite3.Row]], total: int, seed: int
) -> list[sqlite3.Row]:
    ordered_groups = sorted(rows_by_group, key=lambda value: _stable_digest(seed, value))
    queues: dict[str, list[sqlite3.Row]] = {}
    for group in ordered_groups:
        queues[group] = sorted(rows_by_group[group], key=lambda row: _row_key(row, seed))
    selected: list[sqlite3.Row] = []
    offsets = {group: 0 for group in ordered_groups}
    while len(selected) < total:
        progressed = False
        for group in ordered_groups:
            offset = offsets[group]
            if offset < len(queues[group]):
                selected.append(queues[group][offset])
                offsets[group] += 1
                progressed = True
                if len(selected) == total:
                    break
        if not progressed:
            raise ValueError(f"Only {len(selected)} rows available for balanced target {total}")
    contributions = Counter(
        str(row["canonical_generator_id"]) for row in selected
    )
    if contributions and max(contributions.values()) - min(contributions.values()) > 1:
        raise RuntimeError(f"Generator contributions are not balanced: {contributions}")
    return selected


def classify_external_eval_generators(
    eval_generators: set[str],
    eval_architecture: dict[str, str],
    train_generators: set[str],
    validation_generators: set[str],
    train_architectures: set[str],
) -> tuple[set[str], set[str]]:
    """Return exact-unseen Eval identities split by train-seen architecture.

    The first set is the seen-family cohort: its exact generator identities are
    absent from both Small train and Small validation, while its architecture
    families occur in Small train.  The second set is the strict external
    cohort: both exact identities and architecture families are absent.
    """

    exact_unseen = eval_generators.difference(
        train_generators.union(validation_generators)
    )
    seen_family = {
        generator
        for generator in exact_unseen
        if eval_architecture[generator] in train_architectures
    }
    family_unseen = {
        generator
        for generator in exact_unseen
        if eval_architecture[generator] not in train_architectures
    }
    return seen_family, family_unseen


def _selection_tuple(
    row: sqlite3.Row,
    project_split: str,
    exposure: str,
) -> tuple[Any, ...]:
    label = int(row["label"])
    source = str(row["canonical_real_source"])
    if label == 0 and not source:
        source = "UNSPECIFIED"
    generator = str(row["canonical_generator_id"])
    locator_hash = _stable_digest(
        SELECTION_SEED,
        row["dataset_key"],
        row["source_file"],
        row["row_group"],
        row["row_index"],
    )[:16]
    class_name = "aigi" if label == 1 else "real"
    sample_id = f"cf_{project_split}_{class_name}_{locator_hash}"
    relative_path = (
        Path(project_split)
        / class_name
        / f"{sample_id}_{_safe_stem(row['image_name'])}{_extension(row['source_format'])}"
    ).as_posix()
    return (
        sample_id,
        row["dataset_key"],
        row["source_file"],
        row["row_group"],
        row["row_index"],
        row["image_name"],
        row["source_format"],
        row["model_name_raw"],
        generator,
        source,
        row["official_split"],
        project_split,
        label,
        row["architecture"],
        exposure,
        relative_path,
    )


def build_selection_plan(connection: sqlite3.Connection, manifest_dir: Path) -> None:
    existing = int(connection.execute("SELECT COUNT(*) FROM selection").fetchone()[0])
    if existing:
        if existing != 24_000:
            raise RuntimeError(f"Incomplete frozen selection already exists: {existing}")
        return

    small_fake_rows = list(
        connection.execute(
            """
            SELECT * FROM source_rows
            WHERE dataset_key='small' AND label=1 AND canonical_generator_id!=''
            """
        )
    )
    fake_by_generator: dict[str, list[sqlite3.Row]] = defaultdict(list)
    architecture: dict[str, str] = {}
    for row in small_fake_rows:
        generator = str(row["canonical_generator_id"])
        fake_by_generator[generator].append(row)
        architecture.setdefault(generator, str(row["architecture"] or "unknown"))

    generator_count = 0
    per_generator = 0
    eligible: set[str] = set()
    for candidate_count in range(1_000, 9, -10):
        if 10_000 % candidate_count:
            continue
        candidate_per_generator = 10_000 // candidate_count
        candidate_eligible = {
            generator
            for generator, rows in fake_by_generator.items()
            if len(rows) >= candidate_per_generator
        }
        if len(candidate_eligible) >= candidate_count:
            generator_count = candidate_count
            per_generator = candidate_per_generator
            eligible = candidate_eligible
            break
    if not generator_count:
        raise RuntimeError("No feasible equal-contribution Small generator allocation")

    eval_fake_rows = list(
        connection.execute(
            """
            SELECT * FROM source_rows
            WHERE dataset_key='eval' AND label=1 AND canonical_generator_id!=''
            """
        )
    )
    eval_by_generator: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in eval_fake_rows:
        eval_by_generator[str(row["canonical_generator_id"])].append(row)
    eval_generators = set(eval_by_generator)
    overlap = sorted(
        eligible.intersection(eval_generators),
        key=lambda value: _stable_digest(SELECTION_SEED, "overlap", value),
    )
    # The pinned CommunityForensics Small/Eval revisions currently have no exact
    # eligible generator intersection.  The dataset plan explicitly requires the
    # fallback group to be named seen_family rather than pretending that models
    # from the same architecture are the same generator.
    seen_forced: set[str] = set()
    unseen_reserved = set(overlap)

    train_generator_count = generator_count * 9 // 10
    validation_generator_count = generator_count - train_generator_count
    train_generators = _stratified_pick(
        eligible.difference(unseen_reserved),
        architecture,
        train_generator_count,
        SELECTION_SEED,
        forced=seen_forced,
    )
    validation_candidates = eligible.difference(unseen_reserved, train_generators)
    validation_generators = _stratified_pick(
        validation_candidates,
        architecture,
        validation_generator_count,
        SELECTION_SEED + 1,
    )
    if train_generators.intersection(validation_generators):
        raise RuntimeError("Small train/validation generator overlap")

    selection_rows: list[tuple[Any, ...]] = []
    for generator in sorted(train_generators):
        rows = sorted(
            fake_by_generator[generator],
            key=lambda row: _row_key(row, SELECTION_SEED),
        )[:per_generator]
        selection_rows.extend(
            _selection_tuple(row, "train", "train_seen") for row in rows
        )
    for generator in sorted(validation_generators):
        rows = sorted(
            fake_by_generator[generator],
            key=lambda row: _row_key(row, SELECTION_SEED + 1),
        )[:per_generator]
        selection_rows.extend(
            _selection_tuple(row, "val_unseen_generator", "strict_unseen")
            for row in rows
        )

    small_real_rows = _unique_real_rows(
        connection.execute(
            """
            SELECT * FROM source_rows
            WHERE dataset_key='small' AND label=0
              AND lower(official_split)='train'
            """
        ),
        SELECTION_SEED,
        allow_unspecified=True,
    )
    small_real_groups: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in small_real_rows:
        if row["canonical_real_source"] in {"FFHQ", "VISION", "COCO", "Landscapes HQ"}:
            small_real_groups[str(row["canonical_real_source"])].append(row)
    selected_real_groups: dict[str, list[sqlite3.Row]] = {}
    small_real_source_mode = "balanced_declared_sources"
    if not any(small_real_groups.values()):
        # CommunityForensics-Small's pinned revision labels every real row's
        # real_source as N/A.  Preserve that limitation instead of fabricating
        # FFHQ/VISION/COCO/Landscapes HQ identities.
        ordered = sorted(
            small_real_rows,
            key=lambda row: _row_key(row, SELECTION_SEED + 100),
        )
        if len(ordered) < 10_000:
            raise RuntimeError(f"Small has only {len(ordered)} unique real rows")
        selected_real_groups["UNSPECIFIED"] = ordered[:10_000]
        small_real_source_mode = "deterministic_global_fallback_metadata_unavailable"
    else:
        for source in ("FFHQ", "VISION", "COCO", "Landscapes HQ"):
            ordered = sorted(
                small_real_groups[source],
                key=lambda row: _row_key(row, SELECTION_SEED + 100),
            )
            selected_real_groups[source] = ordered[: min(2_500, len(ordered))]
        shortage = 10_000 - sum(len(rows) for rows in selected_real_groups.values())
        if shortage:
            remaining: dict[str, list[sqlite3.Row]] = {}
            for source, all_rows in small_real_groups.items():
                selected_ids = {
                    (row["source_file"], row["row_group"], row["row_index"])
                    for row in selected_real_groups.get(source, [])
                }
                remaining[source] = [
                    row
                    for row in sorted(all_rows, key=lambda item: _row_key(item, SELECTION_SEED + 101))
                    if (row["source_file"], row["row_group"], row["row_index"])
                    not in selected_ids
                ]
            offsets = Counter()
            while shortage:
                progressed = False
                for source in sorted(remaining):
                    if len(selected_real_groups[source]) >= 4_000:
                        continue
                    offset = offsets[source]
                    if offset < len(remaining[source]):
                        selected_real_groups[source].append(remaining[source][offset])
                        offsets[source] += 1
                        shortage -= 1
                        progressed = True
                        if shortage == 0:
                            break
                if not progressed:
                    raise RuntimeError("Small real-source shortages cannot be supplemented")

    validation_allocation = _allocate_proportional(
        {source: len(rows) for source, rows in selected_real_groups.items()}, 1_000
    )
    for source, rows in selected_real_groups.items():
        ordered = sorted(rows, key=lambda row: _row_key(row, SELECTION_SEED + 102))
        validation_count = validation_allocation[source]
        validation_rows = ordered[:validation_count]
        training_rows = ordered[validation_count:]
        selection_rows.extend(
            _selection_tuple(row, "train", "not_applicable") for row in training_rows
        )
        selection_rows.extend(
            _selection_tuple(row, "val_unseen_generator", "not_applicable")
            for row in validation_rows
        )

    train_architectures = {architecture[generator] for generator in train_generators}
    eval_architecture = {
        generator: str(rows[0]["architecture"] or "unknown")
        for generator, rows in eval_by_generator.items()
    }
    seen_family_eval_generators, unseen_eval_generators = (
        classify_external_eval_generators(
            eval_generators,
            eval_architecture,
            train_generators,
            validation_generators,
            train_architectures,
        )
    )
    if not seen_family_eval_generators:
        raise RuntimeError("External seen-family generator group unavailable")
    if not unseen_eval_generators:
        raise RuntimeError(
            "External unseen-generator group unavailable: an unseen architecture "
            "family is required and same-family fallback is forbidden"
        )
    if seen_family_eval_generators.intersection(unseen_eval_generators):
        raise RuntimeError("External seen-family/strict-unseen generator overlap")
    seen_rows = _balanced_group_rows(
        {
            generator: eval_by_generator[generator]
            for generator in seen_family_eval_generators
        },
        1_000,
        SELECTION_SEED + 200,
    )
    unseen_rows = _balanced_group_rows(
        {generator: eval_by_generator[generator] for generator in unseen_eval_generators},
        1_000,
        SELECTION_SEED + 201,
    )
    selection_rows.extend(
        _selection_tuple(row, "test_external_seen_family", "family_seen")
        for row in seen_rows
    )
    selection_rows.extend(
        _selection_tuple(row, "test_external_unseen_generator", "family_unseen")
        for row in unseen_rows
    )

    selected_small_real_identities = {
        (str(row["canonical_real_source"]), str(row["image_name"]).lower())
        for rows in selected_real_groups.values()
        for row in rows
    }
    eval_real_rows = _unique_real_rows(
        connection.execute(
            "SELECT * FROM source_rows WHERE dataset_key='eval' AND label=0"
        ),
        SELECTION_SEED + 300,
    )
    eval_real_groups: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in eval_real_rows:
        source = str(row["canonical_real_source"])
        identity = (source, str(row["image_name"]).lower())
        if source in {"RAISE", "COCO", "FFHQ", "LAION"} and identity not in selected_small_real_identities:
            eval_real_groups[source].append(row)
    for source in ("RAISE", "COCO", "FFHQ", "LAION"):
        ordered = sorted(
            eval_real_groups[source],
            key=lambda row: _row_key(row, SELECTION_SEED + 301),
        )
        if len(ordered) < 500:
            raise RuntimeError(f"Eval real source {source} has only {len(ordered)} unique rows")
        chosen = ordered[:500]
        selection_rows.extend(
            _selection_tuple(row, "test_external_seen_family", "not_applicable")
            for row in chosen[:250]
        )
        selection_rows.extend(
            _selection_tuple(row, "test_external_unseen_generator", "not_applicable")
            for row in chosen[250:]
        )

    if len(selection_rows) != 24_000:
        raise RuntimeError(f"Selection plan has {len(selection_rows)} rows, expected 24000")
    with connection:
        connection.executemany(
            """
            INSERT INTO selection (
                sample_id, dataset_key, source_file, row_group, row_index,
                image_name, source_format, model_name_raw, canonical_generator_id,
                real_source, official_split, project_split, label, architecture,
                generator_exposure, relative_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            selection_rows,
        )

    plan_rows = [dict(row) for row in connection.execute(
        """
        SELECT sample_id, dataset_key, source_file, row_group, row_index,
               image_name, source_format, model_name_raw, canonical_generator_id,
               real_source, official_split, project_split, label, architecture,
               generator_exposure, relative_path
        FROM selection ORDER BY sample_id
        """
    )]
    _atomic_csv(
        plan_rows,
        tuple(plan_rows[0].keys()),
        manifest_dir / "community_forensics_selection_plan.csv",
    )
    plan_audit = {
        "selection_seed": SELECTION_SEED,
        "generator_count": generator_count,
        "train_generator_count": train_generator_count,
        "validation_generator_count": validation_generator_count,
        "aigi_images_per_small_generator": per_generator,
        "small_real_source_selection": small_real_source_mode,
        "small_real_source_counts": {
            source: len(rows) for source, rows in sorted(selected_real_groups.items())
        },
        "small_real_source_limitation": (
            "Pinned CommunityForensics-Small metadata reports real_source=N/A for all real rows; "
            "the requested FFHQ/VISION/COCO/Landscapes HQ balance cannot be verified."
            if small_real_source_mode == "deterministic_global_fallback_metadata_unavailable"
            else None
        ),
        "small_eval_exact_generator_intersection": len(overlap),
        "external_seen_definition": SEEN_FAMILY_DEFINITION,
        "external_unseen_definition": UNSEEN_GENERATOR_DEFINITION,
        "external_seen_family_generators": sorted(seen_family_eval_generators),
        "external_seen_families": sorted(
            {eval_architecture[generator] for generator in seen_family_eval_generators}
        ),
        "external_unseen_generators": sorted(unseen_eval_generators),
        "external_unseen_families": sorted(
            {eval_architecture[generator] for generator in unseen_eval_generators}
        ),
        "reserved_overlap_generators": sorted(unseen_reserved),
        "counts": {
            split: {
                str(label): int(connection.execute(
                    "SELECT COUNT(*) FROM selection WHERE project_split=? AND label=?",
                    (split, label),
                ).fetchone()[0])
                for label in (0, 1)
            }
            for split in PROJECT_SPLITS
        },
    }
    _atomic_json(plan_audit, manifest_dir / "community_forensics_plan_audit.json")


def _restore_existing_file(
    connection: sqlite3.Connection, row: sqlite3.Row, data_root: Path
) -> bool:
    destination = data_root / str(row["relative_path"])
    if not destination.is_file():
        return False
    payload = destination.read_bytes()
    width, height, image_format = _inspect_image(payload)
    with connection:
        connection.execute(
            """
            UPDATE selection SET status='complete', sha256=?, phash=?, width=?,
                height=?, actual_format=?, byte_size=? WHERE sample_id=?
            """,
            (
                _sha256_bytes(payload),
                perceptual_hash(payload),
                width,
                height,
                image_format,
                len(payload),
                row["sample_id"],
            ),
        )
    return True


def materialize_selection(
    connection: sqlite3.Connection,
    data_root: Path,
    max_materialized_bytes: int,
) -> int:
    filesystem = HfFileSystem()
    groups = list(connection.execute(
        """
        SELECT dataset_key, source_file, row_group
        FROM selection WHERE status!='complete'
        GROUP BY dataset_key, source_file, row_group
        ORDER BY dataset_key, source_file, row_group
        """
    ))
    total_written = int(
        connection.execute("SELECT COALESCE(SUM(byte_size), 0) FROM selection").fetchone()[0]
    )
    for group_index, group in enumerate(groups, start=1):
        pending = list(connection.execute(
            """
            SELECT * FROM selection
            WHERE status!='complete' AND dataset_key=? AND source_file=? AND row_group=?
            ORDER BY row_index
            """,
            (group["dataset_key"], group["source_file"], group["row_group"]),
        ))
        pending = [
            row
            for row in pending
            if not _restore_existing_file(connection, row, data_root)
        ]
        if not pending:
            continue
        specification = DATASET_SPECS[str(group["dataset_key"])]
        remote_path = (
            f"datasets/{specification['dataset_id']}@{specification['revision']}/"
            f"{group['source_file']}"
        )
        with filesystem.open(remote_path, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            image_column = parquet.read_row_group(
                int(group["row_group"]), columns=["image_data"]
            )["image_data"]
            for row in pending:
                payload_value = image_column[int(row["row_index"])]
                payload = payload_value.as_py()
                if not isinstance(payload, (bytes, bytearray, memoryview)):
                    raise TypeError(f"Unsupported image_data for {row['sample_id']}")
                payload = bytes(payload)
                if total_written + len(payload) > max_materialized_bytes:
                    raise RuntimeError(
                        f"Materialized byte cap exceeded: {total_written + len(payload)} "
                        f"> {max_materialized_bytes}"
                    )
                width, height, image_format = _inspect_image(payload)
                digest = _sha256_bytes(payload)
                phash = perceptual_hash(payload)
                destination = data_root / str(row["relative_path"])
                _atomic_bytes(payload, destination)
                with connection:
                    connection.execute(
                        """
                        UPDATE selection SET status='complete', sha256=?, phash=?,
                            width=?, height=?, actual_format=?, byte_size=?
                        WHERE sample_id=?
                        """,
                        (
                            digest,
                            phash,
                            width,
                            height,
                            image_format,
                            len(payload),
                            row["sample_id"],
                        ),
                    )
                total_written += len(payload)
        completed = int(
            connection.execute(
                "SELECT COUNT(*) FROM selection WHERE status='complete'"
            ).fetchone()[0]
        )
        print(
            json.dumps(
                {
                    "event": "community_forensics_download_progress",
                    "group_index": group_index,
                    "group_count": len(groups),
                    "dataset": group["dataset_key"],
                    "source_file": group["source_file"],
                    "row_group": group["row_group"],
                    "completed_images": completed,
                    "materialized_bytes": total_written,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if _STOP_REQUESTED:
            return 75
    return 0


def _current_selection_plan_rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(
        """
        SELECT sample_id, dataset_key, source_file, row_group, row_index,
               image_name, source_format, model_name_raw, canonical_generator_id,
               real_source, official_split, project_split, label, architecture,
               generator_exposure, relative_path
        FROM selection ORDER BY sample_id
        """
    )]


def repair_exact_duplicates(
    connection: sqlite3.Connection,
    data_root: Path,
    manifest_dir: Path,
    repair_round: int,
) -> int:
    duplicate_hashes = [
        str(row["sha256"])
        for row in connection.execute(
            """
            SELECT sha256 FROM selection
            WHERE status='complete' AND sha256 IS NOT NULL
            GROUP BY sha256 HAVING COUNT(*) > 1
            ORDER BY sha256
            """
        )
    ]
    if not duplicate_hashes:
        return 0

    repairs: list[dict[str, Any]] = []
    quarantine_root = data_root.parent.parent / "quarantine" / "community_forensics_duplicates"
    for duplicate_hash in duplicate_hashes:
        rows = list(
            connection.execute(
                "SELECT * FROM selection WHERE sha256=? ORDER BY sample_id",
                (duplicate_hash,),
            )
        )
        keeper = min(
            rows,
            key=lambda row: _stable_digest(
                SELECTION_SEED, "exact-duplicate-keeper", duplicate_hash, row["sample_id"]
            ),
        )
        for old in rows:
            if old["sample_id"] == keeper["sample_id"]:
                continue
            parameters: list[Any] = [
                old["dataset_key"],
                old["label"],
                old["official_split"],
            ]
            identity_clause: str
            if int(old["label"]) == 1:
                identity_clause = "candidate.canonical_generator_id=?"
                parameters.append(old["canonical_generator_id"])
            elif old["real_source"] == "UNSPECIFIED":
                identity_clause = "candidate.canonical_real_source=''"
            else:
                identity_clause = "candidate.canonical_real_source=?"
                parameters.append(old["real_source"])
            candidates = list(
                connection.execute(
                    f"""
                    SELECT candidate.* FROM source_rows AS candidate
                    WHERE candidate.dataset_key=? AND candidate.label=?
                      AND candidate.official_split=? AND {identity_clause}
                      AND NOT EXISTS (
                          SELECT 1 FROM selection AS chosen
                          WHERE chosen.dataset_key=candidate.dataset_key
                            AND chosen.source_file=candidate.source_file
                            AND chosen.row_group=candidate.row_group
                            AND chosen.row_index=candidate.row_index
                      )
                    """,
                    parameters,
                )
            )
            if not candidates:
                raise RuntimeError(
                    f"No exact-dedup replacement for {old['sample_id']} "
                    f"in generator/source identity {old['canonical_generator_id'] or old['real_source']}"
                )
            replacement = min(
                candidates,
                key=lambda row: _stable_digest(
                    SELECTION_SEED,
                    "exact-dedup-replacement",
                    repair_round,
                    duplicate_hash,
                    old["sample_id"],
                    row["source_file"],
                    row["row_group"],
                    row["row_index"],
                ),
            )
            replacement_values = _selection_tuple(
                replacement,
                str(old["project_split"]),
                str(old["generator_exposure"]),
            )
            with connection:
                connection.execute("DELETE FROM selection WHERE sample_id=?", (old["sample_id"],))
                connection.execute(
                    """
                    INSERT INTO selection (
                        sample_id, dataset_key, source_file, row_group, row_index,
                        image_name, source_format, model_name_raw, canonical_generator_id,
                        real_source, official_split, project_split, label, architecture,
                        generator_exposure, relative_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    replacement_values,
                )
            old_path = data_root / str(old["relative_path"])
            quarantined_path: Path | None = None
            if old_path.is_file():
                quarantined_path = quarantine_root / f"round_{repair_round}" / str(old["relative_path"])
                quarantined_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(old_path, quarantined_path)
            repairs.append(
                {
                    "repair_round": repair_round,
                    "duplicate_sha256": duplicate_hash,
                    "keeper_sample_id": keeper["sample_id"],
                    "removed_sample_id": old["sample_id"],
                    "removed_path": str(old_path),
                    "quarantined_path": str(quarantined_path) if quarantined_path else None,
                    "replacement_sample_id": replacement_values[0],
                    "replacement_source_file": replacement["source_file"],
                    "replacement_row_group": replacement["row_group"],
                    "replacement_row_index": replacement["row_index"],
                    "project_split": old["project_split"],
                    "label": old["label"],
                    "canonical_generator_id": old["canonical_generator_id"],
                    "real_source": old["real_source"],
                }
            )

    report_path = manifest_dir / "community_forensics_exact_dedup_repairs.json"
    history: list[dict[str, Any]] = []
    if report_path.is_file():
        with report_path.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
        history = list(previous.get("repairs", []))
    history.extend(repairs)
    _atomic_json(
        {
            "selection_seed": SELECTION_SEED,
            "repair_policy": "same_split_same_generator_or_real_source_deterministic_replacement",
            "repairs": history,
        },
        report_path,
    )
    original_plan = manifest_dir / "community_forensics_selection_plan.csv"
    pre_repair_plan = manifest_dir / "community_forensics_selection_plan_pre_dedup.csv"
    if original_plan.is_file() and not pre_repair_plan.exists():
        _atomic_bytes(original_plan.read_bytes(), pre_repair_plan)
    plan_rows = _current_selection_plan_rows(connection)
    _atomic_csv(plan_rows, tuple(plan_rows[0].keys()), original_plan)
    print(
        json.dumps(
            {
                "event": "community_forensics_exact_duplicates_repaired",
                "repair_round": repair_round,
                "duplicate_clusters": len(duplicate_hashes),
                "replacement_samples": len(repairs),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return len(repairs)


def repair_cross_split_phash_duplicates(
    connection: sqlite3.Connection,
    data_root: Path,
    manifest_dir: Path,
    repair_round: int,
    phash_threshold: int,
) -> int:
    complete_rows = list(
        connection.execute(
            "SELECT * FROM selection WHERE status='complete' ORDER BY sample_id"
        )
    )
    tree: _BKNode | None = None
    rows_by_id = {str(row["sample_id"]): row for row in complete_rows}
    conflicts: list[dict[str, Any]] = []
    for row in complete_rows:
        value = int(str(row["phash"]), 16)
        split = str(row["project_split"])
        sample_id = str(row["sample_id"])
        if tree is None:
            tree = _BKNode(value, split, sample_id)
            continue
        for other_split, other_sample_id, distance in _bk_search(
            tree, value, phash_threshold
        ):
            if other_split != split:
                conflicts.append(
                    {
                        "sample_id": sample_id,
                        "split": split,
                        "other_sample_id": other_sample_id,
                        "other_split": other_split,
                        "distance": distance,
                    }
                )
        _bk_insert(tree, value, split, sample_id)
    if not conflicts:
        return 0

    preserve_priority = {
        "train": 0,
        "val_unseen_generator": 1,
        "test_external_seen_family": 2,
        "test_external_unseen_generator": 2,
    }
    victims: set[str] = set()
    conflict_by_victim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for conflict in conflicts:
        first = str(conflict["sample_id"])
        second = str(conflict["other_sample_id"])
        first_priority = preserve_priority[str(conflict["split"])]
        second_priority = preserve_priority[str(conflict["other_split"])]
        if first_priority > second_priority:
            victim = first
        elif second_priority > first_priority:
            victim = second
        else:
            victim = max(
                (first, second),
                key=lambda sample_id: _stable_digest(
                    SELECTION_SEED, "phash-victim", repair_round, sample_id
                ),
            )
        victims.add(victim)
        conflict_by_victim[victim].append(conflict)

    repairs: list[dict[str, Any]] = []
    quarantine_root = data_root.parent.parent / "quarantine" / "community_forensics_phash"
    for victim_id in sorted(victims):
        old = rows_by_id[victim_id]
        parameters: list[Any] = [
            old["dataset_key"],
            old["label"],
            old["official_split"],
        ]
        if int(old["label"]) == 1:
            identity_clause = "candidate.canonical_generator_id=?"
            parameters.append(old["canonical_generator_id"])
        elif old["real_source"] == "UNSPECIFIED":
            identity_clause = "candidate.canonical_real_source=''"
        else:
            identity_clause = "candidate.canonical_real_source=?"
            parameters.append(old["real_source"])
        candidates = list(
            connection.execute(
                f"""
                SELECT candidate.* FROM source_rows AS candidate
                WHERE candidate.dataset_key=? AND candidate.label=?
                  AND candidate.official_split=? AND {identity_clause}
                  AND NOT EXISTS (
                      SELECT 1 FROM selection AS chosen
                      WHERE chosen.dataset_key=candidate.dataset_key
                        AND chosen.source_file=candidate.source_file
                        AND chosen.row_group=candidate.row_group
                        AND chosen.row_index=candidate.row_index
                  )
                """,
                parameters,
            )
        )
        if not candidates:
            raise RuntimeError(
                f"No pHash replacement for {old['sample_id']} in "
                f"generator/source identity {old['canonical_generator_id'] or old['real_source']}"
            )
        replacement = min(
            candidates,
            key=lambda row: _stable_digest(
                SELECTION_SEED,
                "phash-replacement",
                repair_round,
                old["sample_id"],
                row["source_file"],
                row["row_group"],
                row["row_index"],
            ),
        )
        replacement_values = _selection_tuple(
            replacement,
            str(old["project_split"]),
            str(old["generator_exposure"]),
        )
        with connection:
            connection.execute("DELETE FROM selection WHERE sample_id=?", (old["sample_id"],))
            connection.execute(
                """
                INSERT INTO selection (
                    sample_id, dataset_key, source_file, row_group, row_index,
                    image_name, source_format, model_name_raw, canonical_generator_id,
                    real_source, official_split, project_split, label, architecture,
                    generator_exposure, relative_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                replacement_values,
            )
        old_path = data_root / str(old["relative_path"])
        quarantined_path: Path | None = None
        if old_path.is_file():
            quarantined_path = quarantine_root / f"round_{repair_round}" / str(old["relative_path"])
            quarantined_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(old_path, quarantined_path)
        repairs.append(
            {
                "repair_round": repair_round,
                "removed_sample_id": old["sample_id"],
                "removed_phash": old["phash"],
                "removed_path": str(old_path),
                "quarantined_path": str(quarantined_path) if quarantined_path else None,
                "replacement_sample_id": replacement_values[0],
                "replacement_source_file": replacement["source_file"],
                "replacement_row_group": replacement["row_group"],
                "replacement_row_index": replacement["row_index"],
                "project_split": old["project_split"],
                "label": old["label"],
                "canonical_generator_id": old["canonical_generator_id"],
                "real_source": old["real_source"],
                "conflicts": conflict_by_victim[victim_id],
            }
        )

    report_path = manifest_dir / "community_forensics_phash_repairs.json"
    history: list[dict[str, Any]] = []
    if report_path.is_file():
        with report_path.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
        history = list(previous.get("repairs", []))
    history.extend(repairs)
    _atomic_json(
        {
            "selection_seed": SELECTION_SEED,
            "phash_hamming_threshold": phash_threshold,
            "repair_policy": "preserve_train_then_validation_deterministic_same_identity_replacement",
            "repairs": history,
        },
        report_path,
    )
    plan_rows = _current_selection_plan_rows(connection)
    _atomic_csv(
        plan_rows,
        tuple(plan_rows[0].keys()),
        manifest_dir / "community_forensics_selection_plan.csv",
    )
    print(
        json.dumps(
            {
                "event": "community_forensics_cross_split_phash_repaired",
                "repair_round": repair_round,
                "conflicts": len(conflicts),
                "replacement_samples": len(repairs),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return len(repairs)


class _BKNode:
    def __init__(self, value: int, split: str, sample_id: str) -> None:
        self.value = value
        self.entries = [(split, sample_id)]
        self.children: dict[int, _BKNode] = {}


def _bk_insert(root: _BKNode, value: int, split: str, sample_id: str) -> None:
    node = root
    while True:
        distance = (node.value ^ value).bit_count()
        if distance == 0:
            node.entries.append((split, sample_id))
            return
        child = node.children.get(distance)
        if child is None:
            node.children[distance] = _BKNode(value, split, sample_id)
            return
        node = child


def _bk_search(root: _BKNode, value: int, radius: int) -> list[tuple[str, str, int]]:
    found: list[tuple[str, str, int]] = []
    stack = [root]
    while stack:
        node = stack.pop()
        distance = (node.value ^ value).bit_count()
        if distance <= radius:
            found.extend((split, sample_id, distance) for split, sample_id in node.entries)
        lower = distance - radius
        upper = distance + radius
        stack.extend(
            child
            for edge, child in node.children.items()
            if lower <= edge <= upper
        )
    return found


def _reserved_hashes(path: Path | None) -> tuple[set[str], set[str]]:
    if path is None:
        return set(), set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if "sha256" not in fields and "phash" not in fields:
            raise ValueError("Reserved hash manifest needs sha256 and/or phash")
        rows = list(reader)
    return (
        {str(row.get("sha256", "")).lower() for row in rows if row.get("sha256")},
        {str(row.get("phash", "")).lower() for row in rows if row.get("phash")},
    )


def finalize(
    connection: sqlite3.Connection,
    data_root: Path,
    manifest_dir: Path,
    reserved_hash_manifest: Path | None,
    phash_threshold: int,
) -> None:
    incomplete = int(
        connection.execute("SELECT COUNT(*) FROM selection WHERE status!='complete'").fetchone()[0]
    )
    if incomplete:
        raise RuntimeError(f"Cannot finalize with {incomplete} incomplete images")
    all_rows = list(connection.execute("SELECT * FROM selection ORDER BY sample_id"))
    if len(all_rows) != 24_000:
        raise RuntimeError(f"Downloaded {len(all_rows)} rows, expected 24000")
    for split, expected in EXPECTED_COUNTS.items():
        actual = Counter(
            int(row["label"]) for row in all_rows if row["project_split"] == split
        )
        if actual != Counter(expected):
            raise RuntimeError(f"Split {split} counts {actual} != {expected}")

    duplicate_hashes = list(connection.execute(
        """
        SELECT sha256, COUNT(*) AS count FROM selection
        GROUP BY sha256 HAVING COUNT(*) > 1 LIMIT 20
        """
    ))
    if duplicate_hashes:
        raise RuntimeError(
            f"Exact duplicates in frozen selection: {[dict(row) for row in duplicate_hashes]}"
        )

    reserved_sha, reserved_phash = _reserved_hashes(reserved_hash_manifest)
    selected_sha = {str(row["sha256"]) for row in all_rows}
    exact_reserved_overlap = selected_sha.intersection(reserved_sha)
    if exact_reserved_overlap:
        raise RuntimeError(f"Reserved-set SHA-256 overlap: {len(exact_reserved_overlap)}")

    tree: _BKNode | None = None
    cross_split_near: list[dict[str, Any]] = []
    reserved_near: list[dict[str, Any]] = []
    reserved_values = [(int(value, 16), value) for value in reserved_phash]
    for row in all_rows:
        value = int(str(row["phash"]), 16)
        split = str(row["project_split"])
        sample_id = str(row["sample_id"])
        if tree is None:
            tree = _BKNode(value, split, sample_id)
        else:
            for other_split, other_sample, distance in _bk_search(
                tree, value, phash_threshold
            ):
                if other_split != split:
                    cross_split_near.append(
                        {
                            "sample_id": sample_id,
                            "other_sample_id": other_sample,
                            "distance": distance,
                        }
                    )
                    if len(cross_split_near) >= 20:
                        break
            _bk_insert(tree, value, split, sample_id)
        for reserved_value, reserved_text in reserved_values:
            distance = (value ^ reserved_value).bit_count()
            if distance <= phash_threshold:
                reserved_near.append(
                    {
                        "sample_id": sample_id,
                        "reserved_phash": reserved_text,
                        "distance": distance,
                    }
                )
                break
    if cross_split_near:
        raise RuntimeError(f"Cross-split pHash near duplicates: {cross_split_near}")
    if reserved_near:
        raise RuntimeError(f"Reserved-set pHash near duplicates: {reserved_near[:20]}")

    train_generators = {
        str(row["canonical_generator_id"])
        for row in all_rows
        if row["project_split"] == "train" and int(row["label"]) == 1
    }
    validation_generators = {
        str(row["canonical_generator_id"])
        for row in all_rows
        if row["project_split"] == "val_unseen_generator" and int(row["label"]) == 1
    }
    if train_generators.intersection(validation_generators):
        raise RuntimeError("Train/validation AIGI generator leakage")
    seen_family_test_generators = {
        str(row["canonical_generator_id"])
        for row in all_rows
        if row["project_split"] == "test_external_seen_family"
        and int(row["label"]) == 1
    }
    unseen_test_generators = {
        str(row["canonical_generator_id"])
        for row in all_rows
        if row["project_split"] == "test_external_unseen_generator"
        and int(row["label"]) == 1
    }
    train_architectures = {
        str(row["architecture"])
        for row in all_rows
        if row["project_split"] == "train" and int(row["label"]) == 1
    }
    seen_family_test_architectures = {
        str(row["architecture"])
        for row in all_rows
        if row["project_split"] == "test_external_seen_family"
        and int(row["label"]) == 1
    }
    if not seen_family_test_architectures.issubset(train_architectures):
        raise RuntimeError("External seen-family architectures are not all in train")
    if seen_family_test_generators.intersection(
        train_generators.union(validation_generators)
    ):
        raise RuntimeError("Seen-family test contains a Small generator identity")
    if seen_family_test_generators.intersection(unseen_test_generators):
        raise RuntimeError("External seen-family/strict-unseen generator overlap")
    if unseen_test_generators.intersection(train_generators.union(validation_generators)):
        raise RuntimeError("External strict-unseen generator leakage")
    unseen_test_architectures = {
        str(row["architecture"])
        for row in all_rows
        if row["project_split"] == "test_external_unseen_generator"
        and int(row["label"]) == 1
    }
    if unseen_test_architectures.intersection(train_architectures):
        raise RuntimeError(
            "External unseen-generator contains an architecture family seen in train"
        )

    manifest_paths: dict[str, Path] = {}
    for split in PROJECT_SPLITS:
        manifest_rows: list[dict[str, Any]] = []
        for row in all_rows:
            if row["project_split"] != split:
                continue
            dataset_spec = DATASET_SPECS[str(row["dataset_key"])]
            manifest_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "path": row["relative_path"],
                    "label": row["label"],
                    "split": row["project_split"],
                    "source_dataset": dataset_spec["source_dataset"],
                    "generator_id": (
                        row["canonical_generator_id"]
                        if int(row["label"]) == 1
                        else f"real:{row['real_source']}"
                    ),
                    "official_split": row["official_split"],
                    "project_split": row["project_split"],
                    "real_source": row["real_source"],
                    "model_name_raw": row["model_name_raw"],
                    "canonical_generator_id": row["canonical_generator_id"],
                    "architecture": row["architecture"],
                    "generator_exposure": row["generator_exposure"],
                    "sha256": row["sha256"],
                    "phash": row["phash"],
                    "selection_seed": SELECTION_SEED,
                    "source_revision": dataset_spec["revision"],
                    "source_file": row["source_file"],
                    "source_row_group": row["row_group"],
                    "source_row_index": row["row_index"],
                    "width": row["width"],
                    "height": row["height"],
                    "format": row["actual_format"],
                    "byte_size": row["byte_size"],
                }
            )
        path = manifest_dir / f"community_forensics_{split}.csv"
        _atomic_csv(manifest_rows, MANIFEST_FIELDS, path)
        manifest_paths[split] = path

    versions = {
        str(row["dataset_key"]): dict(row)
        for row in connection.execute("SELECT * FROM dataset_versions")
    }
    audit = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "license": "CC-BY-NC-SA-4.0",
        "selection_seed": SELECTION_SEED,
        "datasets": versions,
        "counts": {
            split: {
                str(label): sum(
                    1
                    for row in all_rows
                    if row["project_split"] == split and int(row["label"]) == label
                )
                for label in (0, 1)
            }
            for split in PROJECT_SPLITS
        },
        "materialized_images": len(all_rows),
        "materialized_bytes": sum(int(row["byte_size"]) for row in all_rows),
        "train_generators": len(train_generators),
        "validation_generators": len(validation_generators),
        "external_seen_definition": SEEN_FAMILY_DEFINITION,
        "external_unseen_definition": UNSEEN_GENERATOR_DEFINITION,
        "external_seen_family_generators": sorted(seen_family_test_generators),
        "external_seen_families": sorted(seen_family_test_architectures),
        "external_unseen_generators": sorted(unseen_test_generators),
        "external_unseen_families": sorted(unseen_test_architectures),
        "exact_duplicate_count": 0,
        "cross_split_phash_near_duplicate_count": 0,
        "phash_hamming_threshold": phash_threshold,
        "reserved_hash_manifest": (
            str(reserved_hash_manifest.resolve()) if reserved_hash_manifest else None
        ),
        "reserved_set_dedup_verified": reserved_hash_manifest is not None,
        "reserved_set_limitation": (
            None
            if reserved_hash_manifest
            else "No COCO val2017/DALL-E Advanced reserved hash manifest was available; "
            "official split/source constraints were enforced, but reserved-image hash "
            "overlap could not be verified."
        ),
        "manifests": {
            split: {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
            }
            for split, path in manifest_paths.items()
        },
        "selection_plan_sha256": _sha256_file(
            manifest_dir / "community_forensics_selection_plan.csv"
        ),
    }
    audit_path = manifest_dir / "community_forensics_audit.json"
    _atomic_json(audit, audit_path)
    _atomic_text(
        f"audit={audit_path.resolve()}\n"
        f"audit_sha256={_sha256_file(audit_path)}\n",
        data_root / "COMPLETE",
    )
    print(json.dumps({"event": "community_forensics_complete", **audit}, sort_keys=True), flush=True)


def verify_state(database_path: Path, data_root: Path) -> None:
    connection = _connect(database_path)
    try:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")
        completed = int(
            connection.execute(
                "SELECT COUNT(*) FROM selection WHERE status='complete'"
            ).fetchone()[0]
        )
        for row in connection.execute(
            "SELECT relative_path, sha256 FROM selection WHERE status='complete'"
        ):
            path = data_root / str(row["relative_path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            if row["sha256"] and _sha256_file(path) != row["sha256"]:
                raise RuntimeError(f"Resume-state checksum mismatch: {path}")
        print(
            json.dumps(
                {
                    "event": "community_forensics_state_verified",
                    "completed_images": completed,
                    "database": str(database_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        connection.close()


def run(arguments: argparse.Namespace) -> int:
    database_path = Path(arguments.state_database).expanduser().resolve()
    data_root = Path(arguments.data_root).expanduser().resolve()
    manifest_dir = Path(arguments.manifest_dir).expanduser().resolve()
    if arguments.verify_state:
        verify_state(database_path, data_root)
        return 0
    data_root.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGUSR1, _request_safe_stop)
    connection = _connect(database_path)
    try:
        scan_status = scan_metadata(connection)
        if scan_status == 75:
            return 75
        build_selection_plan(connection, manifest_dir)
        download_status = materialize_selection(
            connection,
            data_root,
            int(arguments.max_materialized_gib * 1024**3),
        )
        if download_status == 75:
            return 75
        for repair_round in range(1, 21):
            repaired_exact = repair_exact_duplicates(
                connection, data_root, manifest_dir, repair_round
            )
            if repaired_exact:
                download_status = materialize_selection(
                    connection,
                    data_root,
                    int(arguments.max_materialized_gib * 1024**3),
                )
                if download_status == 75:
                    return 75
                continue
            repaired_phash = repair_cross_split_phash_duplicates(
                connection,
                data_root,
                manifest_dir,
                repair_round,
                int(arguments.phash_threshold),
            )
            if not repaired_phash:
                break
            download_status = materialize_selection(
                connection,
                data_root,
                int(arguments.max_materialized_gib * 1024**3),
            )
            if download_status == 75:
                return 75
        else:
            raise RuntimeError("Duplicate repair did not converge in 20 rounds")
        reserved = (
            Path(arguments.reserved_hash_manifest).expanduser().resolve()
            if arguments.reserved_hash_manifest
            else None
        )
        finalize(
            connection,
            data_root,
            manifest_dir,
            reserved,
            int(arguments.phash_threshold),
        )
        return 0
    finally:
        connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the frozen 24k Community Forensics train/validation/test dataset"
    )
    parser.add_argument("--data-root", default="data/raw/community_forensics")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument(
        "--state-database",
        default="data/state/community_forensics.sqlite3",
    )
    parser.add_argument("--reserved-hash-manifest")
    parser.add_argument("--phash-threshold", type=int, default=4)
    parser.add_argument("--max-materialized-gib", type=float, default=45.0)
    parser.add_argument("--verify-state", action="store_true")
    arguments = parser.parse_args()
    if arguments.phash_threshold < 0 or arguments.phash_threshold > 64:
        parser.error("--phash-threshold must be in [0, 64]")
    if arguments.max_materialized_gib <= 0:
        parser.error("--max-materialized-gib must be positive")
    return arguments


def main() -> None:
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
