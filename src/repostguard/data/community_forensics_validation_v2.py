from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from repostguard.data.community_forensics import (
    EVAL_DATASET_ID,
    EVAL_REVISION,
    MANIFEST_FIELDS,
    SELECTION_SEED,
    _BKNode,
    _atomic_bytes,
    _atomic_csv,
    _atomic_json,
    _atomic_text,
    _bk_insert,
    _bk_search,
    _extension,
    _inspect_image,
    _sha256_bytes,
    _sha256_file,
    _stable_digest,
    canonical_generator_id,
    perceptual_hash,
)


AIGIBENCH_DATASET_ID = "TheKernel01/AIGIBench"
AIGIBENCH_REVISION = "f125eabc5ac34a4729d74adc1aa1214540f91947"
EXACT_SEEN_RAW_ID = "SD14"
EXACT_SEEN_GENERATOR_ID = "compvis/stable-diffusion-v1-4"
HARD_GENERATORS = ("hourglass", "dfgan", "galip")

EXACT_SPLIT = "val_external_exact_seen_generator"
HARD_SPLITS = {
    generator: f"val_hard_{generator}" for generator in HARD_GENERATORS
}
TARGETS = {
    "exact_real": 1_000,
    "exact_sd14": 1_000,
    "hard_real_panel": 250,
    "hard_hourglass": 250,
    "hard_dfgan": 250,
    "hard_galip": 250,
}

_STOP_REQUESTED = False


def _request_safe_stop(signum: int, frame: object) -> None:
    del signum, frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=120)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS scanned_files (
            source_key TEXT NOT NULL,
            source_file TEXT NOT NULL,
            rows_seen INTEGER NOT NULL,
            scanned_at TEXT NOT NULL,
            PRIMARY KEY (source_key, source_file)
        );
        CREATE TABLE IF NOT EXISTS candidates (
            source_key TEXT NOT NULL,
            source_file TEXT NOT NULL,
            row_group INTEGER NOT NULL,
            row_index INTEGER NOT NULL,
            selection_group TEXT NOT NULL,
            image_name TEXT NOT NULL,
            source_format TEXT NOT NULL,
            label INTEGER NOT NULL,
            model_name_raw TEXT NOT NULL,
            canonical_generator_id TEXT NOT NULL,
            architecture TEXT NOT NULL,
            real_source TEXT NOT NULL,
            official_split TEXT NOT NULL,
            PRIMARY KEY (source_key, source_file, row_group, row_index)
        );
        CREATE INDEX IF NOT EXISTS candidates_group_index
            ON candidates(selection_group);
        CREATE TABLE IF NOT EXISTS selection (
            sample_id TEXT PRIMARY KEY,
            source_key TEXT NOT NULL,
            source_file TEXT NOT NULL,
            row_group INTEGER NOT NULL,
            row_index INTEGER NOT NULL,
            selection_group TEXT NOT NULL,
            image_name TEXT NOT NULL,
            source_format TEXT NOT NULL,
            label INTEGER NOT NULL,
            model_name_raw TEXT NOT NULL,
            canonical_generator_id TEXT NOT NULL,
            architecture TEXT NOT NULL,
            real_source TEXT NOT NULL,
            official_split TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            rejection_reason TEXT,
            sha256 TEXT,
            phash TEXT,
            width INTEGER,
            height INTEGER,
            actual_format TEXT,
            byte_size INTEGER,
            UNIQUE (source_key, source_file, row_group, row_index)
        );
        """
    )
    return connection


def _metadata(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute(
        "SELECT value FROM metadata WHERE key=?", (key,)
    ).fetchone()
    return str(row[0]) if row else None


def _set_metadata(connection: sqlite3.Connection, key: str, value: str) -> None:
    with connection:
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            (key, value),
        )


def _resolved_parquet_files(
    dataset_id: str, revision: str, prefix: str
) -> list[tuple[str, int]]:
    info = HfApi().dataset_info(
        dataset_id,
        revision=revision,
        files_metadata=True,
    )
    resolved = str(info.sha or "")
    if resolved != revision:
        raise RuntimeError(f"Resolved revision changed: {resolved} != {revision}")
    files = sorted(
        (
            str(sibling.rfilename),
            int(sibling.size or 0),
        )
        for sibling in list(info.siblings or [])
        if str(sibling.rfilename).startswith(prefix)
        and str(sibling.rfilename).endswith(".parquet")
    )
    if not files:
        raise RuntimeError(f"No Parquet files for {dataset_id} under {prefix}")
    return files


def _as_int(value: Any) -> int:
    if hasattr(value, "as_py"):
        value = value.as_py()
    return int(value)


def scan_aigibench(connection: sqlite3.Connection) -> int:
    files = _resolved_parquet_files(
        AIGIBENCH_DATASET_ID,
        AIGIBENCH_REVISION,
        "data/validation-",
    )
    version_payload = json.dumps(
        {
            "dataset_id": AIGIBENCH_DATASET_ID,
            "revision": AIGIBENCH_REVISION,
            "files": files,
        },
        sort_keys=True,
    )
    previous = _metadata(connection, "aigibench_version")
    if previous is not None and previous != version_payload:
        raise RuntimeError("Frozen AIGIBench source version changed")
    _set_metadata(connection, "aigibench_version", version_payload)

    completed = {
        str(row[0])
        for row in connection.execute(
            "SELECT source_file FROM scanned_files WHERE source_key='aigibench'"
        )
    }
    filesystem = HfFileSystem()
    for file_index, (source_file, _) in enumerate(files, start=1):
        if source_file in completed:
            continue
        remote_path = (
            f"datasets/{AIGIBENCH_DATASET_ID}@{AIGIBENCH_REVISION}/{source_file}"
        )
        candidates: list[tuple[Any, ...]] = []
        rows_seen = 0
        with filesystem.open(remote_path, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            missing = {"image", "label", "generator"}.difference(
                parquet.schema_arrow.names
            )
            if missing:
                raise RuntimeError(f"{source_file} missing columns: {sorted(missing)}")
            for row_group in range(parquet.metadata.num_row_groups):
                table = parquet.read_row_group(
                    row_group, columns=["label", "generator"]
                )
                for row_index in range(len(table)):
                    label = _as_int(table["label"][row_index])
                    generator = _as_int(table["generator"][row_index])
                    if label == 0 and generator == 0:
                        selection_group = "exact_real"
                        model_name = "Real"
                        canonical = ""
                        architecture = "not_applicable"
                        real_source = "ImageNet"
                    elif label == 1 and generator == 2:
                        selection_group = "exact_sd14"
                        model_name = EXACT_SEEN_RAW_ID
                        canonical = EXACT_SEEN_GENERATOR_ID
                        architecture = "LatDiff"
                        real_source = ""
                    else:
                        continue
                    candidates.append(
                        (
                            "aigibench",
                            source_file,
                            row_group,
                            row_index,
                            selection_group,
                            f"{source_file}:{row_group}:{row_index}",
                            "",
                            label,
                            model_name,
                            canonical,
                            architecture,
                            real_source,
                            "validation",
                        )
                    )
                rows_seen += len(table)
        with connection:
            connection.executemany(
                """
                INSERT OR IGNORE INTO candidates VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                candidates,
            )
            connection.execute(
                "INSERT INTO scanned_files VALUES (?, ?, ?, ?)",
                (
                    "aigibench",
                    source_file,
                    rows_seen,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        print(
            json.dumps(
                {
                    "event": "validation_v2_metadata_progress",
                    "source": "aigibench",
                    "file_index": file_index,
                    "file_count": len(files),
                    "file": source_file,
                    "eligible_candidates": len(candidates),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if _STOP_REQUESTED:
            return 75
    return 0


def import_eval_candidates(
    connection: sqlite3.Connection, base_database: Path
) -> None:
    if _metadata(connection, "eval_candidates_imported") == EVAL_REVISION:
        return
    base = sqlite3.connect(f"file:{base_database}?mode=ro", uri=True)
    base.row_factory = sqlite3.Row
    try:
        selected_locators = {
            (
                str(row["dataset_key"]),
                str(row["source_file"]),
                int(row["row_group"]),
                int(row["row_index"]),
            )
            for row in base.execute(
                "SELECT dataset_key, source_file, row_group, row_index FROM selection"
            )
        }
        rows = list(
            base.execute(
                """
                SELECT * FROM source_rows
                WHERE dataset_key='eval'
                  AND (
                    label=0 OR
                    (label=1 AND canonical_generator_id IN ('hourglass','dfgan','galip'))
                  )
                """
            )
        )
    finally:
        base.close()

    candidates: list[tuple[Any, ...]] = []
    seen_real_identities: set[tuple[str, str]] = set()
    for row in rows:
        locator = (
            "eval",
            str(row["source_file"]),
            int(row["row_group"]),
            int(row["row_index"]),
        )
        if locator in selected_locators:
            continue
        label = int(row["label"])
        generator = str(row["canonical_generator_id"])
        if label == 0:
            identity = (
                str(row["canonical_real_source"]),
                str(row["image_name"]).strip().lower(),
            )
            if not identity[0] or identity in seen_real_identities:
                continue
            seen_real_identities.add(identity)
            selection_group = "hard_real_panel"
            real_source = str(row["canonical_real_source"])
        else:
            selection_group = f"hard_{generator}"
            real_source = ""
        candidates.append(
            (
                "eval",
                str(row["source_file"]),
                int(row["row_group"]),
                int(row["row_index"]),
                selection_group,
                str(row["image_name"]),
                str(row["source_format"]),
                label,
                str(row["model_name_raw"]),
                generator,
                str(row["architecture"] or "unknown"),
                real_source,
                str(row["official_split"]),
            )
        )
    with connection:
        connection.executemany(
            """
            INSERT OR IGNORE INTO candidates VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            candidates,
        )
    _set_metadata(connection, "eval_candidates_imported", EVAL_REVISION)


def _candidate_sort_key(row: sqlite3.Row) -> tuple[str, str]:
    """Randomize deterministically while keeping selected row groups dense.

    Reading an embedded-image Parquet column fetches the complete row group.
    Ranking row groups first preserves seeded selection without scattering a
    small subset over hundreds of expensive remote reads.  Real and SD14 use
    the same I/O cohort so both classes preferentially reuse row groups.
    """

    selection_group = str(row["selection_group"])
    io_cohort = "exact_aigibench" if selection_group.startswith("exact_") else selection_group
    group_rank = _stable_digest(
        SELECTION_SEED + 700,
        io_cohort,
        row["source_key"],
        row["source_file"],
        row["row_group"],
    )
    row_rank = _stable_digest(
        SELECTION_SEED + 701,
        selection_group,
        row["source_key"],
        row["source_file"],
        row["row_group"],
        row["row_index"],
    )
    return group_rank, row_rank


def _group_path(group: str, sample_id: str) -> str:
    class_name = "real" if group.endswith("real") or group == "hard_real_panel" else "aigi"
    return (Path("validation_v2") / group / class_name / f"{sample_id}.img").as_posix()


def fill_selection(connection: sqlite3.Connection) -> None:
    for group, target in TARGETS.items():
        active = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM selection
                WHERE selection_group=? AND status IN ('pending','complete')
                """,
                (group,),
            ).fetchone()[0]
        )
        needed = target - active
        if needed <= 0:
            continue
        candidates = list(
            connection.execute(
                """
                SELECT candidate.* FROM candidates AS candidate
                WHERE candidate.selection_group=?
                  AND NOT EXISTS (
                    SELECT 1 FROM selection AS chosen
                    WHERE chosen.source_key=candidate.source_key
                      AND chosen.source_file=candidate.source_file
                      AND chosen.row_group=candidate.row_group
                      AND chosen.row_index=candidate.row_index
                  )
                """,
                (group,),
            )
        )
        candidates.sort(key=_candidate_sort_key)
        if len(candidates) < needed:
            raise RuntimeError(
                f"Selection group {group} needs {needed}, only {len(candidates)} remain"
            )
        values: list[tuple[Any, ...]] = []
        for row in candidates[:needed]:
            locator = _stable_digest(
                SELECTION_SEED + 701,
                row["source_key"],
                row["source_file"],
                row["row_group"],
                row["row_index"],
            )[:16]
            sample_id = f"cfv2_{group}_{locator}"
            values.append(
                (
                    sample_id,
                    row["source_key"],
                    row["source_file"],
                    row["row_group"],
                    row["row_index"],
                    row["selection_group"],
                    row["image_name"],
                    row["source_format"],
                    row["label"],
                    row["model_name_raw"],
                    row["canonical_generator_id"],
                    row["architecture"],
                    row["real_source"],
                    row["official_split"],
                    _group_path(group, sample_id),
                )
            )
        with connection:
            connection.executemany(
                """
                INSERT INTO selection (
                    sample_id, source_key, source_file, row_group, row_index,
                    selection_group, image_name, source_format, label,
                    model_name_raw, canonical_generator_id, architecture,
                    real_source, official_split, relative_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )


def _base_hashes_and_tree(
    manifest_dir: Path, phash_threshold: int
) -> tuple[set[str], _BKNode | None]:
    hashes: set[str] = set()
    tree: _BKNode | None = None
    for manifest in sorted(manifest_dir.glob("community_forensics_*.csv")):
        if "selection_plan" in manifest.name:
            continue
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                sha256 = str(row.get("sha256", ""))
                phash = str(row.get("phash", ""))
                if sha256:
                    hashes.add(sha256)
                if phash:
                    value = int(phash, 16)
                    if tree is None:
                        tree = _BKNode(value, "base", str(row["sample_id"]))
                    else:
                        _bk_insert(tree, value, "base", str(row["sample_id"]))
    del phash_threshold
    return hashes, tree


def _payload(cell: Any) -> bytes:
    value = cell.as_py() if hasattr(cell, "as_py") else cell
    if isinstance(value, dict):
        value = value.get("bytes")
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError("Parquet image cell does not contain embedded bytes")
    return bytes(value)


def _remote_source(row: sqlite3.Row) -> tuple[str, str]:
    if row["source_key"] == "aigibench":
        return (
            f"datasets/{AIGIBENCH_DATASET_ID}@{AIGIBENCH_REVISION}/"
            f"{row['source_file']}",
            "image",
        )
    if row["source_key"] == "eval":
        return (
            f"datasets/{EVAL_DATASET_ID}@{EVAL_REVISION}/{row['source_file']}",
            "image_data",
        )
    raise KeyError(row["source_key"])


def materialize(
    connection: sqlite3.Connection,
    data_root: Path,
    manifest_dir: Path,
    max_new_bytes: int,
    phash_threshold: int,
) -> int:
    base_hashes, tree = _base_hashes_and_tree(manifest_dir, phash_threshold)
    for row in connection.execute(
        "SELECT sample_id, sha256, phash FROM selection WHERE status='complete'"
    ):
        base_hashes.add(str(row["sha256"]))
        value = int(str(row["phash"]), 16)
        if tree is None:
            tree = _BKNode(value, "validation_v2", str(row["sample_id"]))
        else:
            _bk_insert(tree, value, "validation_v2", str(row["sample_id"]))

    new_bytes = int(
        connection.execute(
            "SELECT COALESCE(SUM(byte_size),0) FROM selection WHERE status='complete'"
        ).fetchone()[0]
    )
    filesystem = HfFileSystem()
    groups = list(
        connection.execute(
            """
            SELECT source_key, source_file, row_group
            FROM selection WHERE status='pending'
            GROUP BY source_key, source_file, row_group
            ORDER BY source_key, source_file, row_group
            """
        )
    )
    for group_index, group in enumerate(groups, start=1):
        pending = list(
            connection.execute(
                """
                SELECT * FROM selection
                WHERE status='pending' AND source_key=? AND source_file=? AND row_group=?
                ORDER BY row_index
                """,
                (group["source_key"], group["source_file"], group["row_group"]),
            )
        )
        if not pending:
            continue
        remote_path, image_column_name = _remote_source(pending[0])
        with filesystem.open(remote_path, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            image_column = parquet.read_row_group(
                int(group["row_group"]), columns=[image_column_name]
            )[image_column_name]
            for row in pending:
                payload = _payload(image_column[int(row["row_index"])])
                width, height, image_format = _inspect_image(payload)
                sha256 = _sha256_bytes(payload)
                phash = perceptual_hash(payload)
                rejection_reason: str | None = None
                if sha256 in base_hashes:
                    rejection_reason = "sha256_overlap_with_base_or_validation_v2"
                elif tree is not None and _bk_search(
                    tree, int(phash, 16), phash_threshold
                ):
                    rejection_reason = (
                        f"phash_overlap_with_base_or_validation_v2_at_distance_le_{phash_threshold}"
                    )
                if rejection_reason:
                    with connection:
                        connection.execute(
                            """
                            UPDATE selection SET status='rejected', rejection_reason=?,
                                sha256=?, phash=?, width=?, height=?, actual_format=?, byte_size=?
                            WHERE sample_id=?
                            """,
                            (
                                rejection_reason,
                                sha256,
                                phash,
                                width,
                                height,
                                image_format,
                                len(payload),
                                row["sample_id"],
                            ),
                        )
                    continue
                if new_bytes + len(payload) > max_new_bytes:
                    raise RuntimeError(
                        f"Validation-v2 byte cap exceeded: {new_bytes + len(payload)} "
                        f"> {max_new_bytes}"
                    )
                final_relative = str(
                    Path(str(row["relative_path"])).with_suffix(
                        _extension(image_format)
                    )
                )
                destination = data_root / final_relative
                _atomic_bytes(payload, destination)
                with connection:
                    connection.execute(
                        """
                        UPDATE selection SET status='complete', relative_path=?,
                            sha256=?, phash=?, width=?, height=?, actual_format=?, byte_size=?
                        WHERE sample_id=?
                        """,
                        (
                            final_relative,
                            sha256,
                            phash,
                            width,
                            height,
                            image_format,
                            len(payload),
                            row["sample_id"],
                        ),
                    )
                base_hashes.add(sha256)
                if tree is None:
                    tree = _BKNode(
                        int(phash, 16), "validation_v2", str(row["sample_id"])
                    )
                else:
                    _bk_insert(
                        tree,
                        int(phash, 16),
                        "validation_v2",
                        str(row["sample_id"]),
                    )
                new_bytes += len(payload)
        print(
            json.dumps(
                {
                    "event": "validation_v2_download_progress",
                    "group_index": group_index,
                    "group_count": len(groups),
                    "source": group["source_key"],
                    "source_file": group["source_file"],
                    "row_group": group["row_group"],
                    "complete": int(
                        connection.execute(
                            "SELECT COUNT(*) FROM selection WHERE status='complete'"
                        ).fetchone()[0]
                    ),
                    "rejected": int(
                        connection.execute(
                            "SELECT COUNT(*) FROM selection WHERE status='rejected'"
                        ).fetchone()[0]
                    ),
                    "materialized_bytes": new_bytes,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if _STOP_REQUESTED:
            return 75
    return 0


def _train_generator_sets(manifest_dir: Path) -> tuple[set[str], set[str], set[str]]:
    train: set[str] = set()
    validation: set[str] = set()
    train_architectures: set[str] = set()
    for filename, target in (
        ("community_forensics_train.csv", train),
        ("community_forensics_val_unseen_generator.csv", validation),
    ):
        with (manifest_dir / filename).open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                if int(row["label"]) != 1:
                    continue
                target.add(canonical_generator_id(row["canonical_generator_id"]))
                if filename == "community_forensics_train.csv":
                    train_architectures.add(str(row["architecture"]))
    return train, validation, train_architectures


def _fake_identities_and_architectures(path: Path) -> tuple[set[str], set[str]]:
    identities: set[str] = set()
    architectures: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["label"]) != 1:
                continue
            identities.add(canonical_generator_id(row["canonical_generator_id"]))
            architectures.add(str(row["architecture"]))
    return identities, architectures


def verify_frozen_external_protocol(
    manifest_dir: Path,
    train: set[str],
    small_validation: set[str],
    train_architectures: set[str],
) -> dict[str, Any]:
    seen_path = manifest_dir / "community_forensics_test_external_seen_family.csv"
    unseen_path = (
        manifest_dir / "community_forensics_test_external_unseen_generator.csv"
    )
    seen_ids, seen_architectures = _fake_identities_and_architectures(seen_path)
    unseen_ids, unseen_architectures = _fake_identities_and_architectures(unseen_path)
    small_ids = train.union(small_validation)
    if seen_ids.intersection(small_ids):
        raise RuntimeError("Seen-family contains a Small exact generator identity")
    if not seen_architectures.issubset(train_architectures):
        raise RuntimeError("Seen-family contains an architecture absent from Small train")
    if unseen_ids.intersection(small_ids):
        raise RuntimeError("Unseen-generator contains a Small exact generator identity")
    if unseen_architectures.intersection(train_architectures):
        raise RuntimeError("Unseen-generator contains a Small train architecture family")
    if seen_ids.intersection(unseen_ids):
        raise RuntimeError("Frozen seen-family and unseen-generator identities overlap")
    return {
        "seen_family": {
            "definition": (
                "train-seen architecture family and exact identity absent from "
                "Small train and validation"
            ),
            "generators": sorted(seen_ids),
            "architectures": sorted(seen_architectures),
        },
        "unseen_generator": {
            "definition": (
                "architecture family and exact identity both absent from Small "
                "train and validation"
            ),
            "generators": sorted(unseen_ids),
            "architectures": sorted(unseen_architectures),
        },
    }


def _manifest_row(
    row: sqlite3.Row,
    project_split: str,
    exposure: str,
) -> dict[str, Any]:
    source_dataset = (
        "aigibench-genimage"
        if row["source_key"] == "aigibench"
        else "community-forensics-eval"
    )
    revision = (
        AIGIBENCH_REVISION if row["source_key"] == "aigibench" else EVAL_REVISION
    )
    return {
        "sample_id": row["sample_id"],
        "path": row["relative_path"],
        "label": row["label"],
        "split": project_split,
        "source_dataset": source_dataset,
        "generator_id": (
            row["canonical_generator_id"]
            if int(row["label"]) == 1
            else f"real:{row['real_source']}"
        ),
        "official_split": row["official_split"],
        "project_split": project_split,
        "real_source": row["real_source"],
        "model_name_raw": row["model_name_raw"],
        "canonical_generator_id": row["canonical_generator_id"],
        "architecture": row["architecture"],
        "generator_exposure": exposure if int(row["label"]) == 1 else "not_applicable",
        "sha256": row["sha256"],
        "phash": row["phash"],
        "selection_seed": SELECTION_SEED,
        "source_revision": revision,
        "source_file": row["source_file"],
        "source_row_group": row["row_group"],
        "source_row_index": row["row_index"],
        "width": row["width"],
        "height": row["height"],
        "format": row["actual_format"],
        "byte_size": row["byte_size"],
    }


def finalize(
    connection: sqlite3.Connection,
    data_root: Path,
    manifest_dir: Path,
    complete_marker: Path,
    phash_threshold: int,
) -> None:
    counts = {
        group: int(
            connection.execute(
                """
                SELECT COUNT(*) FROM selection
                WHERE selection_group=? AND status='complete'
                """,
                (group,),
            ).fetchone()[0]
        )
        for group in TARGETS
    }
    if counts != TARGETS:
        raise RuntimeError(f"Validation-v2 counts {counts} != {TARGETS}")

    train, small_validation, train_architectures = _train_generator_sets(manifest_dir)
    if EXACT_SEEN_GENERATOR_ID not in train:
        raise RuntimeError(
            f"Exact-seen identity {EXACT_SEEN_GENERATOR_ID} is absent from Small train"
        )
    for generator in HARD_GENERATORS:
        if generator in train or generator in small_validation:
            raise RuntimeError(f"Hard generator {generator} is not exact-unseen")
    if not {"GAN", "PixDiff"}.issubset(train_architectures):
        raise RuntimeError("Small train no longer contains the hard-slice families")
    frozen_external_protocol = verify_frozen_external_protocol(
        manifest_dir,
        train,
        small_validation,
        train_architectures,
    )

    exact_rows = list(
        connection.execute(
            """
            SELECT * FROM selection
            WHERE selection_group IN ('exact_real','exact_sd14') AND status='complete'
            ORDER BY sample_id
            """
        )
    )
    exact_manifest = manifest_dir / f"community_forensics_{EXACT_SPLIT}.csv"
    _atomic_csv(
        [
            _manifest_row(row, EXACT_SPLIT, "exact_seen")
            for row in exact_rows
        ],
        MANIFEST_FIELDS,
        exact_manifest,
    )

    real_panel = list(
        connection.execute(
            """
            SELECT * FROM selection
            WHERE selection_group='hard_real_panel' AND status='complete'
            ORDER BY sample_id
            """
        )
    )
    hard_manifests: dict[str, Path] = {}
    for generator in HARD_GENERATORS:
        fake_rows = list(
            connection.execute(
                """
                SELECT * FROM selection
                WHERE selection_group=? AND status='complete'
                ORDER BY sample_id
                """,
                (f"hard_{generator}",),
            )
        )
        split = HARD_SPLITS[generator]
        path = manifest_dir / f"community_forensics_{split}.csv"
        rows = [
            _manifest_row(row, split, "not_applicable") for row in real_panel
        ] + [
            _manifest_row(row, split, "family_seen_exact_unseen_hard")
            for row in fake_rows
        ]
        rows.sort(key=lambda row: str(row["sample_id"]))
        _atomic_csv(rows, MANIFEST_FIELDS, path)
        hard_manifests[generator] = path

    rejected = [
        dict(row)
        for row in connection.execute(
            """
            SELECT sample_id, source_key, source_file, row_group, row_index,
                   selection_group, rejection_reason, sha256, phash
            FROM selection WHERE status='rejected' ORDER BY sample_id
            """
        )
    ]
    external_manifests = {
        name: _sha256_file(manifest_dir / name)
        for name in (
            "community_forensics_test_external_seen_family.csv",
            "community_forensics_test_external_unseen_generator.csv",
        )
    }
    audit = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_seed": SELECTION_SEED,
        "protocol": {
            "exact_seen_generator": (
                "same exact generator identity as Small train, different source dataset"
            ),
            "seen_family": (
                "train-seen architecture family, exact generator identity absent from "
                "Small train and validation"
            ),
            "unseen_generator": (
                "architecture family and exact generator identity both absent from "
                "Small train and validation"
            ),
        },
        "frozen_external_protocol_audit": frozen_external_protocol,
        "exact_seen": {
            "dataset_id": AIGIBENCH_DATASET_ID,
            "source_revision": AIGIBENCH_REVISION,
            "source_generator_label": EXACT_SEEN_RAW_ID,
            "canonical_generator_id": EXACT_SEEN_GENERATOR_ID,
            "identity_basis": (
                "AIGIBench declares SD14 as Stable Diffusion 1.4; Small train contains "
                "CompVis/stable-diffusion-v1-4"
            ),
            "manifest": str(exact_manifest.resolve()),
            "manifest_sha256": _sha256_file(exact_manifest),
        },
        "hard_slices": {
            generator: {
                "manifest": str(path.resolve()),
                "manifest_sha256": _sha256_file(path),
                "aigi": TARGETS[f"hard_{generator}"],
                "shared_real_panel": TARGETS["hard_real_panel"],
            }
            for generator, path in hard_manifests.items()
        },
        "counts": counts,
        "materialized_images": sum(counts.values()),
        "materialized_bytes": int(
            connection.execute(
                "SELECT COALESCE(SUM(byte_size),0) FROM selection WHERE status='complete'"
            ).fetchone()[0]
        ),
        "phash_hamming_threshold": phash_threshold,
        "rejected_candidate_count": len(rejected),
        "rejected_candidates": rejected,
        "preserved_external_manifest_sha256": external_manifests,
    }
    audit_path = manifest_dir / "community_forensics_validation_v2_audit.json"
    _atomic_json(audit, audit_path)
    _atomic_text(
        json.dumps(
            {
                "audit": str(audit_path.resolve()),
                "audit_sha256": _sha256_file(audit_path),
                "completed_at_utc": audit["completed_at_utc"],
            },
            sort_keys=True,
        )
        + "\n",
        complete_marker,
    )


def verify_state(database: Path, data_root: Path) -> None:
    connection = _connect(database)
    try:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")
        for row in connection.execute(
            """
            SELECT relative_path, sha256 FROM selection WHERE status='complete'
            """
        ):
            path = data_root / str(row["relative_path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            if _sha256_file(path) != str(row["sha256"]):
                raise RuntimeError(f"Checksum mismatch: {path}")
        print(
            json.dumps(
                {
                    "event": "community_forensics_validation_v2_state_verified",
                    "complete": int(
                        connection.execute(
                            "SELECT COUNT(*) FROM selection WHERE status='complete'"
                        ).fetchone()[0]
                    ),
                    "pending": int(
                        connection.execute(
                            "SELECT COUNT(*) FROM selection WHERE status='pending'"
                        ).fetchone()[0]
                    ),
                    "rejected": int(
                        connection.execute(
                            "SELECT COUNT(*) FROM selection WHERE status='rejected'"
                        ).fetchone()[0]
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        connection.close()


def run(arguments: argparse.Namespace) -> int:
    database = Path(arguments.state_database).expanduser().resolve()
    base_database = Path(arguments.base_state_database).expanduser().resolve()
    data_root = Path(arguments.data_root).expanduser().resolve()
    manifest_dir = Path(arguments.manifest_dir).expanduser().resolve()
    complete_marker = data_root / "VALIDATION_V2_COMPLETE"
    if arguments.verify_state:
        verify_state(database, data_root)
        return 0
    if complete_marker.is_file():
        print(complete_marker.read_text(encoding="utf-8").strip(), flush=True)
        return 0

    signal.signal(signal.SIGUSR1, _request_safe_stop)
    data_root.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    connection = _connect(database)
    try:
        if scan_aigibench(connection) == 75:
            return 75
        import_eval_candidates(connection, base_database)
        for _ in range(20):
            fill_selection(connection)
            status = materialize(
                connection,
                data_root,
                manifest_dir,
                int(arguments.max_new_gib * 1024**3),
                int(arguments.phash_threshold),
            )
            if status == 75:
                return 75
            remaining = sum(
                TARGETS[group]
                - int(
                    connection.execute(
                        """
                        SELECT COUNT(*) FROM selection
                        WHERE selection_group=? AND status='complete'
                        """,
                        (group,),
                    ).fetchone()[0]
                )
                for group in TARGETS
            )
            if remaining == 0:
                break
        else:
            raise RuntimeError("Validation-v2 duplicate replacement did not converge")
        finalize(
            connection,
            data_root,
            manifest_dir,
            complete_marker,
            int(arguments.phash_threshold),
        )
        return 0
    finally:
        connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add exact-seen and hard-generator validation slices without replacing "
            "the frozen Community Forensics external tests"
        )
    )
    parser.add_argument("--data-root", default="data/raw/community_forensics")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument(
        "--state-database",
        default="data/state/community_forensics_validation_v2.sqlite3",
    )
    parser.add_argument(
        "--base-state-database",
        default="data/state/community_forensics.sqlite3",
    )
    parser.add_argument("--phash-threshold", type=int, default=4)
    parser.add_argument("--max-new-gib", type=float, default=8.0)
    parser.add_argument("--verify-state", action="store_true")
    arguments = parser.parse_args()
    if not 0 <= arguments.phash_threshold <= 64:
        parser.error("--phash-threshold must be in [0, 64]")
    if arguments.max_new_gib <= 0:
        parser.error("--max-new-gib must be positive")
    return arguments


def main() -> None:
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
