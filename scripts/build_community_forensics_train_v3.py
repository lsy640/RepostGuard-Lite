from __future__ import annotations

import argparse
import csv
import json
import signal
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from repostguard.data.community_forensics import (
    MANIFEST_FIELDS,
    SMALL_DATASET_ID,
    SMALL_REVISION,
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
    perceptual_hash,
)


SELECTION_SEED = 20260830
EXPECTED_ADDITION_COUNTS = {
    "gan_aigi": 1_000,
    "pixdiff_aigi": 1_000,
    "gan_matched_real": 1_000,
    "pixdiff_matched_real": 1_000,
}
EXPECTED_BASE_CLASS_COUNTS = {0: 10_000, 1: 10_000}
EXPECTED_V3_CLASS_COUNTS = {0: 12_000, 1: 12_000}
EXPECTED_GENERATOR_COUNTS = {"GAN": 12, "PixDiff": 3}
STOP_REQUESTED = False


def _request_stop(signum: int, frame: object) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"Manifest has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _manifest_locator(row: dict[str, str]) -> tuple[str, str, int, int] | None:
    source_file = str(row.get("source_file", "")).strip()
    row_group = str(row.get("source_row_group", "")).strip()
    row_index = str(row.get("source_row_index", "")).strip()
    revision = str(row.get("source_revision", "")).strip()
    if not source_file or not row_group or not row_index or not revision:
        return None
    return revision, source_file, int(row_group), int(row_index)


def _all_frozen_rows(manifest_dir: Path) -> list[dict[str, str]]:
    rows_by_locator: dict[tuple[str, str, int, int], dict[str, str]] = {}
    rows_without_locator: list[dict[str, str]] = []
    for path in sorted(manifest_dir.glob("community_forensics_*.csv")):
        if any(
            token in path.name
            for token in ("selection_plan", "train_v3", "statistics")
        ):
            continue
        _, rows = _read_csv(path)
        for row in rows:
            locator = _manifest_locator(row)
            if locator is None:
                rows_without_locator.append(row)
            else:
                rows_by_locator.setdefault(locator, row)
    return list(rows_by_locator.values()) + rows_without_locator


def _counts(rows: Iterable[dict[str, str]]) -> dict[int, int]:
    return dict(Counter(int(row["label"]) for row in rows))


def _connect_state(path: Path) -> sqlite3.Connection:
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
        CREATE TABLE IF NOT EXISTS selection (
            sample_id TEXT PRIMARY KEY,
            addition_group TEXT NOT NULL,
            source_file TEXT NOT NULL,
            row_group INTEGER NOT NULL,
            row_index INTEGER NOT NULL,
            image_name TEXT NOT NULL,
            source_format TEXT NOT NULL,
            model_name_raw TEXT NOT NULL,
            canonical_generator_id TEXT NOT NULL,
            official_split TEXT NOT NULL,
            label INTEGER NOT NULL,
            architecture TEXT NOT NULL,
            relative_path TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL DEFAULT 'planned',
            sha256 TEXT,
            phash TEXT,
            width INTEGER,
            height INTEGER,
            actual_format TEXT,
            byte_size INTEGER,
            UNIQUE(source_file, row_group, row_index)
        );
        CREATE INDEX IF NOT EXISTS selection_download_group
            ON selection(status, source_file, row_group);
        """
    )
    return connection


def _source_connection(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=120)
    connection.row_factory = sqlite3.Row
    return connection


def _row_key(row: sqlite3.Row, *scope: object) -> str:
    return _stable_digest(
        SELECTION_SEED,
        *scope,
        row["source_file"],
        row["row_group"],
        row["row_index"],
        row["image_name"],
    )


def _compact_rows(
    rows: Sequence[sqlite3.Row], count: int, scope: str
) -> tuple[list[sqlite3.Row], set[tuple[str, int]]]:
    """Select deterministically while touching the fewest Parquet row groups.

    CommunityForensics-Small stores roughly 3,000 embedded images in each
    Parquet row group. Reading one selected row therefore requires reading a
    large image_data column chunk. Prefer the row groups with the most eligible
    rows, then use a stable hash for deterministic tie-breaking.
    """

    by_row_group: dict[tuple[str, int], list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        key = (str(row["source_file"]), int(row["row_group"]))
        by_row_group[key].append(row)
    ordered_groups = sorted(
        by_row_group,
        key=lambda key: (
            -len(by_row_group[key]),
            _stable_digest(SELECTION_SEED, scope, "row-group", *key),
        ),
    )
    selected: list[sqlite3.Row] = []
    touched: set[tuple[str, int]] = set()
    for key in ordered_groups:
        ordered_rows = sorted(
            by_row_group[key], key=lambda row: _row_key(row, scope, *key)
        )
        remaining = count - len(selected)
        if remaining <= 0:
            break
        selected.extend(ordered_rows[:remaining])
        touched.add(key)
    if len(selected) != count:
        raise RuntimeError(
            f"Only {len(selected)} candidates available for {scope}, need {count}"
        )
    return selected, touched


def _balanced_compact_generator_rows(
    rows_by_generator: dict[str, list[sqlite3.Row]],
    count: int,
    scope: str,
) -> list[sqlite3.Row]:
    generators = sorted(
        rows_by_generator,
        key=lambda value: _stable_digest(SELECTION_SEED, scope, "generator", value),
    )
    base_quota, remainder = divmod(count, len(generators))
    selected: list[sqlite3.Row] = []
    for index, generator in enumerate(generators):
        quota = base_quota + int(index < remainder)
        rows, _ = _compact_rows(
            rows_by_generator[generator], quota, f"{scope}:{generator}"
        )
        selected.extend(rows)
    contributions = Counter(
        str(row["canonical_generator_id"]) for row in selected
    )
    if max(contributions.values()) - min(contributions.values()) > 1:
        raise RuntimeError(f"Unbalanced {scope} contributions: {contributions}")
    return selected


def _selection_tuple(
    row: sqlite3.Row, addition_group: str
) -> tuple[Any, ...]:
    label = int(row["label"])
    class_name = "aigi" if label == 1 else "real"
    locator_hash = _stable_digest(
        SELECTION_SEED,
        addition_group,
        row["source_file"],
        row["row_group"],
        row["row_index"],
    )[:16]
    sample_id = f"cf_train_v3_{addition_group}_{locator_hash}"
    safe_name = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in str(row["image_name"])
    ).strip("._")[:80] or "image"
    relative_path = (
        Path("train_v3_additions")
        / class_name
        / f"{sample_id}_{safe_name}{_extension(row['source_format'])}"
    ).as_posix()
    return (
        sample_id,
        addition_group,
        row["source_file"],
        row["row_group"],
        row["row_index"],
        row["image_name"],
        row["source_format"],
        row["model_name_raw"],
        row["canonical_generator_id"],
        row["official_split"],
        label,
        row["architecture"],
        relative_path,
    )


def _excluded_identities(
    frozen_rows: Sequence[dict[str, str]],
) -> set[tuple[str, str, int, int]]:
    locators = {
        locator
        for row in frozen_rows
        if (locator := _manifest_locator(row)) is not None
    }
    return locators


def build_plan(
    state: sqlite3.Connection,
    source: sqlite3.Connection,
    frozen_rows: Sequence[dict[str, str]],
    plan_path: Path,
) -> None:
    existing = int(state.execute("SELECT COUNT(*) FROM selection").fetchone()[0])
    if existing:
        if existing != sum(EXPECTED_ADDITION_COUNTS.values()):
            raise RuntimeError(f"Incomplete v3 selection exists: {existing}")
        return

    excluded_locators = _excluded_identities(frozen_rows)

    aigi_selection: dict[str, list[sqlite3.Row]] = {}
    for architecture, addition_group in (
        ("GAN", "gan_aigi"),
        ("PixDiff", "pixdiff_aigi"),
    ):
        candidates = list(
            source.execute(
                """
                SELECT * FROM source_rows
                WHERE dataset_key='small' AND label=1
                  AND architecture=? AND lower(official_split)='train'
                  AND canonical_generator_id!=''
                """,
                (architecture,),
            )
        )
        by_generator: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for row in candidates:
            locator = (
                SMALL_REVISION,
                str(row["source_file"]),
                int(row["row_group"]),
                int(row["row_index"]),
            )
            if locator not in excluded_locators:
                by_generator[str(row["canonical_generator_id"])].append(row)
        expected_generators = EXPECTED_GENERATOR_COUNTS[architecture]
        if len(by_generator) != expected_generators:
            raise RuntimeError(
                f"{architecture} has {len(by_generator)} eligible generators, "
                f"expected {expected_generators}"
            )
        aigi_selection[addition_group] = _balanced_compact_generator_rows(
            by_generator,
            EXPECTED_ADDITION_COUNTS[addition_group],
            addition_group,
        )

    real_candidates = list(
        source.execute(
            """
            SELECT * FROM source_rows
            WHERE dataset_key='small' AND label=0
              AND lower(official_split)='train'
            """
        )
    )
    real_rows: list[sqlite3.Row] = []
    seen_real_names: set[str] = set()
    for row in real_candidates:
        locator = (
            SMALL_REVISION,
            str(row["source_file"]),
            int(row["row_group"]),
            int(row["row_index"]),
        )
        image_name = str(row["image_name"]).strip().lower()
        if locator in excluded_locators or image_name in seen_real_names:
            continue
        seen_real_names.add(image_name)
        real_rows.append(row)
    gan_real, gan_groups = _compact_rows(real_rows, 1_000, "gan_matched_real")
    remaining_real = [
        row
        for row in real_rows
        if (str(row["source_file"]), int(row["row_group"])) not in gan_groups
    ]
    pixdiff_real, _ = _compact_rows(
        remaining_real, 1_000, "pixdiff_matched_real"
    )
    real_selection = {
        "gan_matched_real": gan_real,
        "pixdiff_matched_real": pixdiff_real,
    }
    if any(len(rows) != 1_000 for rows in real_selection.values()):
        raise RuntimeError("Real counterpart split did not produce two 1000-row groups")

    selection_rows: list[tuple[Any, ...]] = []
    for group, rows in {**aigi_selection, **real_selection}.items():
        selection_rows.extend(_selection_tuple(row, group) for row in rows)
    if len(selection_rows) != 4_000:
        raise RuntimeError(f"v3 plan has {len(selection_rows)} rows, expected 4000")
    with state:
        state.executemany(
            """
            INSERT INTO selection (
                sample_id, addition_group, source_file, row_group, row_index,
                image_name, source_format, model_name_raw,
                canonical_generator_id, official_split, label, architecture,
                relative_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            selection_rows,
        )
        state.executemany(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            (
                ("selection_seed", str(SELECTION_SEED)),
                ("small_revision", SMALL_REVISION),
                ("plan_created_at_utc", datetime.now(timezone.utc).isoformat()),
            ),
        )
    write_plan(state, plan_path)


def write_plan(state: sqlite3.Connection, path: Path) -> None:
    rows = [
        dict(row)
        for row in state.execute(
            """
            SELECT sample_id, addition_group, source_file, row_group, row_index,
                   image_name, source_format, model_name_raw,
                   canonical_generator_id, official_split, label, architecture,
                   relative_path, status
            FROM selection ORDER BY addition_group, sample_id
            """
        )
    ]
    _atomic_csv(rows, tuple(rows[0].keys()), path)


def _restore_existing(
    state: sqlite3.Connection, row: sqlite3.Row, data_root: Path
) -> bool:
    path = data_root / str(row["relative_path"])
    if not path.is_file():
        return False
    payload = path.read_bytes()
    width, height, actual_format = _inspect_image(payload)
    with state:
        state.execute(
            """
            UPDATE selection SET status='complete', sha256=?, phash=?, width=?,
                height=?, actual_format=?, byte_size=? WHERE sample_id=?
            """,
            (
                _sha256_bytes(payload),
                perceptual_hash(payload),
                width,
                height,
                actual_format,
                len(payload),
                row["sample_id"],
            ),
        )
    return True


def materialize(
    state: sqlite3.Connection, data_root: Path, max_bytes: int
) -> int:
    filesystem = HfFileSystem()
    groups = list(
        state.execute(
            """
            SELECT source_file, row_group FROM selection
            WHERE status!='complete'
            GROUP BY source_file, row_group ORDER BY source_file, row_group
            """
        )
    )
    written = int(
        state.execute(
            "SELECT COALESCE(SUM(byte_size), 0) FROM selection"
        ).fetchone()[0]
    )
    for group_index, group in enumerate(groups, start=1):
        rows = list(
            state.execute(
                """
                SELECT * FROM selection WHERE status!='complete'
                  AND source_file=? AND row_group=? ORDER BY row_index
                """,
                (group["source_file"], group["row_group"]),
            )
        )
        rows = [row for row in rows if not _restore_existing(state, row, data_root)]
        if not rows:
            continue
        remote = (
            f"datasets/{SMALL_DATASET_ID}@{SMALL_REVISION}/"
            f"{group['source_file']}"
        )
        with filesystem.open(remote, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            images = parquet.read_row_group(
                int(group["row_group"]), columns=["image_data"]
            )["image_data"]
            for row in rows:
                payload = images[int(row["row_index"])].as_py()
                if not isinstance(payload, (bytes, bytearray, memoryview)):
                    raise TypeError(f"Invalid image_data for {row['sample_id']}")
                payload = bytes(payload)
                if written + len(payload) > max_bytes:
                    raise RuntimeError(
                        f"v3 byte cap exceeded: {written + len(payload)} > {max_bytes}"
                    )
                width, height, actual_format = _inspect_image(payload)
                destination = data_root / str(row["relative_path"])
                _atomic_bytes(payload, destination)
                with state:
                    state.execute(
                        """
                        UPDATE selection SET status='complete', sha256=?, phash=?,
                            width=?, height=?, actual_format=?, byte_size=?
                        WHERE sample_id=?
                        """,
                        (
                            _sha256_bytes(payload),
                            perceptual_hash(payload),
                            width,
                            height,
                            actual_format,
                            len(payload),
                            row["sample_id"],
                        ),
                    )
                written += len(payload)
        completed = int(
            state.execute(
                "SELECT COUNT(*) FROM selection WHERE status='complete'"
            ).fetchone()[0]
        )
        print(
            json.dumps(
                {
                    "event": "community_forensics_train_v3_download_progress",
                    "group_index": group_index,
                    "group_count": len(groups),
                    "completed_images": completed,
                    "materialized_bytes": written,
                    "source_file": group["source_file"],
                    "row_group": group["row_group"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if STOP_REQUESTED:
            return 75
    return 0


def _addition_manifest_rows(state: sqlite3.Connection) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in state.execute("SELECT * FROM selection ORDER BY sample_id"):
        label = int(row["label"])
        result.append(
            {
                "sample_id": row["sample_id"],
                "path": row["relative_path"],
                "label": label,
                "split": "train",
                "source_dataset": "community-forensics-small",
                "generator_id": (
                    row["canonical_generator_id"]
                    if label == 1
                    else "real:UNSPECIFIED"
                ),
                "official_split": row["official_split"],
                "project_split": "train",
                "real_source": "UNSPECIFIED" if label == 0 else "",
                "model_name_raw": row["model_name_raw"],
                "canonical_generator_id": row["canonical_generator_id"],
                "architecture": row["architecture"],
                "generator_exposure": "train_seen" if label == 1 else "not_applicable",
                "sha256": row["sha256"],
                "phash": row["phash"],
                "selection_seed": SELECTION_SEED,
                "source_revision": SMALL_REVISION,
                "source_file": row["source_file"],
                "source_row_group": row["row_group"],
                "source_row_index": row["row_index"],
                "width": row["width"],
                "height": row["height"],
                "format": row["actual_format"],
                "byte_size": row["byte_size"],
            }
        )
    return result


def _identity_overlap(
    base: Sequence[dict[str, str]], additions: Sequence[dict[str, Any]]
) -> dict[str, int]:
    def identities(rows: Sequence[dict[str, Any]], field: str) -> set[str]:
        if field == "source_locator":
            return {
                "|".join(
                    (
                        str(row["source_revision"]),
                        str(row["source_file"]),
                        str(row["source_row_group"]),
                        str(row["source_row_index"]),
                    )
                )
                for row in rows
            }
        return {str(row[field]) for row in rows}

    return {
        field: len(identities(base, field).intersection(identities(additions, field)))
        for field in ("sample_id", "path", "sha256", "source_locator")
    }


def _phash_conflicts(
    base: Sequence[dict[str, str]], additions: Sequence[dict[str, Any]], radius: int
) -> list[dict[str, Any]]:
    tree: _BKNode | None = None
    for row in base:
        value = str(row.get("phash", "")).strip()
        if not value:
            continue
        integer = int(value, 16)
        if tree is None:
            tree = _BKNode(integer, "base", str(row["sample_id"]))
        else:
            _bk_insert(tree, integer, "base", str(row["sample_id"]))
    conflicts: list[dict[str, Any]] = []
    for row in additions:
        integer = int(str(row["phash"]), 16)
        if tree is not None:
            for cohort, sample_id, distance in _bk_search(tree, integer, radius):
                conflicts.append(
                    {
                        "addition_sample_id": row["sample_id"],
                        "other_cohort": cohort,
                        "other_sample_id": sample_id,
                        "distance": distance,
                    }
                )
                if len(conflicts) >= 20:
                    return conflicts
        if tree is None:
            tree = _BKNode(integer, "addition", str(row["sample_id"]))
        else:
            _bk_insert(tree, integer, "addition", str(row["sample_id"]))
    return conflicts


def _all_phash_conflicts(
    frozen_rows: Sequence[dict[str, str]],
    additions: Sequence[dict[str, Any]],
    radius: int,
) -> list[dict[str, Any]]:
    """Return every addition conflict with frozen rows or earlier additions."""

    tree: _BKNode | None = None
    for row in frozen_rows:
        value = str(row.get("phash", "")).strip()
        if not value:
            continue
        integer = int(value, 16)
        if tree is None:
            tree = _BKNode(integer, "frozen", str(row["sample_id"]))
        else:
            _bk_insert(tree, integer, "frozen", str(row["sample_id"]))

    conflicts: list[dict[str, Any]] = []
    for row in sorted(additions, key=lambda item: str(item["sample_id"])):
        integer = int(str(row["phash"]), 16)
        if tree is not None:
            for cohort, other_sample_id, distance in _bk_search(
                tree, integer, radius
            ):
                conflicts.append(
                    {
                        "addition_sample_id": str(row["sample_id"]),
                        "other_cohort": cohort,
                        "other_sample_id": other_sample_id,
                        "distance": distance,
                    }
                )
        if tree is None:
            tree = _BKNode(integer, "addition", str(row["sample_id"]))
        else:
            _bk_insert(tree, integer, "addition", str(row["sample_id"]))
    return conflicts


def repair_phash_conflicts(
    state: sqlite3.Connection,
    source: sqlite3.Connection,
    frozen_rows: Sequence[dict[str, str]],
    data_root: Path,
    manifest_dir: Path,
    repair_round: int,
    radius: int,
) -> int:
    additions = _addition_manifest_rows(state)
    conflicts = _all_phash_conflicts(frozen_rows, additions, radius)
    if not conflicts:
        return 0

    conflicts_by_victim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for conflict in conflicts:
        conflicts_by_victim[str(conflict["addition_sample_id"])].append(conflict)
    selected_locators = {
        (
            SMALL_REVISION,
            str(row["source_file"]),
            int(row["row_group"]),
            int(row["row_index"]),
        )
        for row in state.execute(
            "SELECT source_file, row_group, row_index FROM selection"
        )
    }
    frozen_locators = _excluded_identities(frozen_rows)
    repairs: list[dict[str, Any]] = []
    quarantine_root = (
        data_root.parent.parent
        / "quarantine"
        / "community_forensics_train_v3_phash"
        / f"round_{repair_round}"
    )

    for victim_id in sorted(conflicts_by_victim):
        old = state.execute(
            "SELECT * FROM selection WHERE sample_id=?", (victim_id,)
        ).fetchone()
        if old is None:
            raise RuntimeError(f"Missing pHash repair victim: {victim_id}")
        parameters: list[Any] = [int(old["label"]), str(old["official_split"])]
        identity_clause = ""
        if int(old["label"]) == 1:
            identity_clause = "AND canonical_generator_id=?"
            parameters.append(str(old["canonical_generator_id"]))
        candidates = list(
            source.execute(
                f"""
                SELECT * FROM source_rows
                WHERE dataset_key='small' AND label=? AND official_split=?
                  {identity_clause}
                """,
                parameters,
            )
        )
        eligible: list[sqlite3.Row] = []
        for candidate in candidates:
            locator = (
                SMALL_REVISION,
                str(candidate["source_file"]),
                int(candidate["row_group"]),
                int(candidate["row_index"]),
            )
            if locator not in selected_locators and locator not in frozen_locators:
                eligible.append(candidate)
        if not eligible:
            raise RuntimeError(
                f"No pHash replacement candidate for {victim_id} in "
                f"group={old['addition_group']} generator={old['canonical_generator_id']}"
            )
        replacement = min(
            eligible,
            key=lambda row: (
                int(
                    str(row["source_file"]) != str(old["source_file"])
                    or int(row["row_group"]) != int(old["row_group"])
                ),
                _stable_digest(
                    SELECTION_SEED,
                    "v3-phash-replacement",
                    repair_round,
                    victim_id,
                    row["source_file"],
                    row["row_group"],
                    row["row_index"],
                ),
            ),
        )
        replacement_values = _selection_tuple(
            replacement, str(old["addition_group"])
        )
        old_locator = (
            SMALL_REVISION,
            str(old["source_file"]),
            int(old["row_group"]),
            int(old["row_index"]),
        )
        replacement_locator = (
            SMALL_REVISION,
            str(replacement["source_file"]),
            int(replacement["row_group"]),
            int(replacement["row_index"]),
        )
        with state:
            state.execute("DELETE FROM selection WHERE sample_id=?", (victim_id,))
            state.execute(
                """
                INSERT INTO selection (
                    sample_id, addition_group, source_file, row_group, row_index,
                    image_name, source_format, model_name_raw,
                    canonical_generator_id, official_split, label, architecture,
                    relative_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                replacement_values,
            )
        selected_locators.remove(old_locator)
        selected_locators.add(replacement_locator)
        old_path = data_root / str(old["relative_path"])
        quarantined_path: Path | None = None
        if old_path.is_file():
            quarantined_path = quarantine_root / str(old["relative_path"])
            quarantined_path.parent.mkdir(parents=True, exist_ok=True)
            old_path.replace(quarantined_path)
        repairs.append(
            {
                "repair_round": repair_round,
                "removed_sample_id": victim_id,
                "removed_path": str(old_path),
                "removed_phash": str(old["phash"]),
                "quarantined_path": (
                    str(quarantined_path) if quarantined_path else None
                ),
                "replacement_sample_id": replacement_values[0],
                "replacement_source_file": replacement["source_file"],
                "replacement_row_group": replacement["row_group"],
                "replacement_row_index": replacement["row_index"],
                "addition_group": old["addition_group"],
                "canonical_generator_id": old["canonical_generator_id"],
                "conflicts": conflicts_by_victim[victim_id],
            }
        )

    report_path = manifest_dir / "community_forensics_train_v3_phash_repairs.json"
    history: list[dict[str, Any]] = []
    if report_path.is_file():
        with report_path.open("r", encoding="utf-8") as handle:
            history = list(json.load(handle).get("repairs", []))
    history.extend(repairs)
    _atomic_json(
        {
            "selection_seed": SELECTION_SEED,
            "phash_hamming_threshold": radius,
            "repair_policy": (
                "replace_later_addition_conflicts_with_same_group_and_exact_"
                "generator_candidates_preferring_the_same_parquet_row_group"
            ),
            "repairs": history,
        },
        report_path,
    )
    write_plan(state, manifest_dir / "community_forensics_train_v3_selection_plan.csv")
    print(
        json.dumps(
            {
                "event": "community_forensics_train_v3_phash_repair",
                "repair_round": repair_round,
                "conflicts": len(conflicts),
                "replacement_samples": len(repairs),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return len(repairs)


def finalize(arguments: argparse.Namespace, state: sqlite3.Connection) -> None:
    incomplete = int(
        state.execute(
            "SELECT COUNT(*) FROM selection WHERE status!='complete'"
        ).fetchone()[0]
    )
    if incomplete:
        raise RuntimeError(f"Cannot finalize v3 with {incomplete} incomplete rows")

    base_fields, base_rows = _read_csv(Path(arguments.base_train))
    if tuple(base_fields) != tuple(MANIFEST_FIELDS):
        raise RuntimeError("train-v2 header does not match the canonical manifest schema")
    if _counts(base_rows) != EXPECTED_BASE_CLASS_COUNTS:
        raise RuntimeError(f"Unexpected train-v2 class counts: {_counts(base_rows)}")
    additions = _addition_manifest_rows(state)
    if _counts(additions) != {0: 2_000, 1: 2_000}:
        raise RuntimeError(f"Unexpected v3 addition class counts: {_counts(additions)}")

    group_counts = {
        str(row["addition_group"]): int(row["count"])
        for row in state.execute(
            "SELECT addition_group, COUNT(*) count FROM selection GROUP BY addition_group"
        )
    }
    if group_counts != EXPECTED_ADDITION_COUNTS:
        raise RuntimeError(f"Unexpected addition groups: {group_counts}")
    generator_counts = {
        str(row["architecture"]): int(row["count"])
        for row in state.execute(
            """
            SELECT architecture, COUNT(DISTINCT canonical_generator_id) count
            FROM selection WHERE label=1 GROUP BY architecture
            """
        )
    }
    if generator_counts != EXPECTED_GENERATOR_COUNTS:
        raise RuntimeError(f"Unexpected exact-generator coverage: {generator_counts}")

    overlap = _identity_overlap(base_rows, additions)
    if any(overlap.values()):
        raise RuntimeError(f"train-v2/v3 addition overlap: {overlap}")
    frozen_rows = _all_frozen_rows(Path(arguments.manifest_dir))
    frozen_overlap = _identity_overlap(frozen_rows, additions)
    if any(frozen_overlap.values()):
        raise RuntimeError(f"Frozen split/v3 addition overlap: {frozen_overlap}")
    phash_conflicts = _phash_conflicts(
        frozen_rows, additions, int(arguments.phash_threshold)
    )
    if phash_conflicts:
        raise RuntimeError(
            f"v3 additions contain pHash near-duplicates: {phash_conflicts}"
        )

    combined: list[dict[str, Any]] = [dict(row) for row in base_rows] + additions
    if len(combined) != 24_000 or _counts(combined) != EXPECTED_V3_CLASS_COUNTS:
        raise RuntimeError(
            f"Unexpected train-v3 counts: rows={len(combined)} classes={_counts(combined)}"
        )
    data_root = Path(arguments.data_root)
    missing = [
        str(row["path"])
        for row in combined
        if not (data_root / str(row["path"])).is_file()
    ]
    if missing:
        raise RuntimeError(f"train-v3 has {len(missing)} missing images: {missing[:10]}")

    additions_path = Path(arguments.additions_manifest)
    output_path = Path(arguments.output)
    _atomic_csv(additions, MANIFEST_FIELDS, additions_path)
    _atomic_csv(combined, MANIFEST_FIELDS, output_path)
    write_plan(state, Path(arguments.selection_plan))

    contribution_ranges: dict[str, dict[str, int]] = {}
    exact_generators: dict[str, list[str]] = {}
    for architecture in ("GAN", "PixDiff"):
        contributions = {
            str(row["canonical_generator_id"]): int(row["count"])
            for row in state.execute(
                """
                SELECT canonical_generator_id, COUNT(*) count FROM selection
                WHERE label=1 AND architecture=? GROUP BY canonical_generator_id
                ORDER BY canonical_generator_id
                """,
                (architecture,),
            )
        }
        exact_generators[architecture] = sorted(contributions)
        contribution_ranges[architecture] = {
            "minimum": min(contributions.values()),
            "maximum": max(contributions.values()),
        }
    real_shards = {
        str(row["addition_group"]): int(row["count"])
        for row in state.execute(
            """
            SELECT addition_group, COUNT(DISTINCT source_file) count
            FROM selection WHERE label=0 GROUP BY addition_group
            """
        )
    }
    materialized_bytes = int(
        state.execute("SELECT SUM(byte_size) FROM selection").fetchone()[0]
    )
    audit = {
        "protocol_id": "community_forensics_train_v3_small_gan_pixdiff_expansion",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_seed": SELECTION_SEED,
        "source": {
            "dataset_id": SMALL_DATASET_ID,
            "revision": SMALL_REVISION,
            "license": "CC-BY-NC-SA-4.0",
        },
        "base": {
            "manifest": str(Path(arguments.base_train).resolve()),
            "manifest_sha256": _sha256_file(Path(arguments.base_train)),
            "rows": len(base_rows),
            "class_counts": {"real": 10_000, "aigi": 10_000},
        },
        "additions": {
            "manifest": str(additions_path.resolve()),
            "manifest_sha256": _sha256_file(additions_path),
            "rows": len(additions),
            "group_counts": group_counts,
            "class_counts": {"real": 2_000, "aigi": 2_000},
            "exact_generator_counts": generator_counts,
            "exact_generators": exact_generators,
            "per_generator_contribution_range": contribution_ranges,
            "materialized_bytes": materialized_bytes,
        },
        "real_balance": {
            "requested_pairing": {
                "GAN counterpart": 1_000,
                "PixDiff counterpart": 1_000,
            },
            "metadata_observation": (
                "Pinned CommunityForensics-Small marks all real rows as "
                "architecture=Real, subset=Real, real_source=N/A. No factual "
                "real-source type can be assigned."
            ),
            "implemented_proxy": (
                "Two deterministic 1000-image counterpart quotas, each "
                "drawn from a distinct deterministic Parquet row group to "
                "avoid pretending that unavailable source-type labels exist."
            ),
            "distinct_source_shards": real_shards,
        },
        "output": {
            "manifest": str(output_path.resolve()),
            "manifest_sha256": _sha256_file(output_path),
            "rows": len(combined),
            "class_counts": {"real": 12_000, "aigi": 12_000},
        },
        "integrity": {
            "base_addition_identity_overlap": overlap,
            "all_frozen_manifest_addition_identity_overlap": frozen_overlap,
            "phash_hamming_threshold": int(arguments.phash_threshold),
            "phash_near_duplicate_count": 0,
            "all_materialized_paths_verified": True,
            "state_integrity": "ok",
        },
        "evaluation_policy": {
            "external_seen_family_remains_retired": True,
            "strict_unseen_generator_manifest": (
                "data/manifests/community_forensics_test_external_unseen_generator.csv"
            ),
            "strict_unseen_architectures": ["Commercial", "Other"],
        },
    }
    audit_path = Path(arguments.audit)
    _atomic_json(audit, audit_path)
    marker = Path(arguments.complete_marker)
    _atomic_text(
        f"audit={audit_path.resolve()}\n"
        f"audit_sha256={_sha256_file(audit_path)}\n"
        f"manifest={output_path.resolve()}\n"
        f"manifest_sha256={_sha256_file(output_path)}\n",
        marker,
    )
    print(
        json.dumps(
            {
                "event": "community_forensics_train_v3_complete",
                "rows": len(combined),
                "addition_rows": len(additions),
                "exact_generator_counts": generator_counts,
                "materialized_bytes": materialized_bytes,
                "manifest": str(output_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def verify_state(arguments: argparse.Namespace) -> None:
    database = Path(arguments.state_database)
    if not database.is_file():
        raise RuntimeError(f"Missing v3 state database: {database}")
    state = _connect_state(database)
    try:
        integrity = str(state.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")
        planned = int(state.execute("SELECT COUNT(*) FROM selection").fetchone()[0])
        completed = int(
            state.execute(
                "SELECT COUNT(*) FROM selection WHERE status='complete'"
            ).fetchone()[0]
        )
        missing = 0
        data_root = Path(arguments.data_root)
        for row in state.execute(
            "SELECT relative_path FROM selection WHERE status='complete'"
        ):
            if not (data_root / str(row["relative_path"])).is_file():
                missing += 1
        if planned != 4_000 or missing:
            raise RuntimeError(
                f"Invalid v3 state: planned={planned} completed={completed} missing={missing}"
            )
        print(
            json.dumps(
                {
                    "event": "community_forensics_train_v3_state_verified",
                    "planned": planned,
                    "completed": completed,
                    "missing_completed_files": missing,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        state.close()


def run(arguments: argparse.Namespace) -> int:
    if arguments.verify_state:
        verify_state(arguments)
        return 0
    signal.signal(signal.SIGUSR1, _request_stop)
    source_database = Path(arguments.source_database)
    if not source_database.is_file():
        raise RuntimeError(f"Missing frozen source database: {source_database}")
    base_train = Path(arguments.base_train)
    if not base_train.is_file():
        raise RuntimeError(f"Missing train-v2 manifest: {base_train}")
    data_root = Path(arguments.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    manifest_dir = Path(arguments.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    state = _connect_state(Path(arguments.state_database))
    source = _source_connection(source_database)
    try:
        frozen_rows = _all_frozen_rows(manifest_dir)
        build_plan(state, source, frozen_rows, Path(arguments.selection_plan))
        for repair_round in range(1, 21):
            status = materialize(
                state,
                data_root,
                int(float(arguments.max_materialized_gib) * 1024**3),
            )
            write_plan(state, Path(arguments.selection_plan))
            if status == 75:
                return 75
            repaired = repair_phash_conflicts(
                state,
                source,
                frozen_rows,
                data_root,
                manifest_dir,
                repair_round,
                int(arguments.phash_threshold),
            )
            if not repaired:
                break
        else:
            raise RuntimeError("train-v3 pHash repair did not converge in 20 rounds")
        finalize(arguments, state)
        return 0
    finally:
        source.close()
        state.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build CommunityForensics train-v3 as train-v2 plus 1000 GAN, "
            "1000 PixDiff, and 2000 matched real images from Small"
        )
    )
    parser.add_argument(
        "--source-database", default="data/state/community_forensics.sqlite3"
    )
    parser.add_argument(
        "--state-database", default="data/state/community_forensics_train_v3.sqlite3"
    )
    parser.add_argument("--data-root", default="data/raw/community_forensics")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument(
        "--base-train", default="data/manifests/community_forensics_train_v2.csv"
    )
    parser.add_argument(
        "--selection-plan",
        default="data/manifests/community_forensics_train_v3_selection_plan.csv",
    )
    parser.add_argument(
        "--additions-manifest",
        default="data/manifests/community_forensics_train_v3_additions.csv",
    )
    parser.add_argument(
        "--output", default="data/manifests/community_forensics_train_v3.csv"
    )
    parser.add_argument(
        "--audit", default="data/manifests/community_forensics_train_v3_audit.json"
    )
    parser.add_argument(
        "--complete-marker",
        default="data/raw/community_forensics/TRAIN_V3_COMPLETE",
    )
    parser.add_argument("--phash-threshold", type=int, default=4)
    parser.add_argument("--max-materialized-gib", type=float, default=12.0)
    parser.add_argument("--verify-state", action="store_true")
    arguments = parser.parse_args()
    if not 0 <= arguments.phash_threshold <= 64:
        parser.error("--phash-threshold must be in [0, 64]")
    if arguments.max_materialized_gib <= 0:
        parser.error("--max-materialized-gib must be positive")
    return arguments


if __name__ == "__main__":
    sys.exit(run(parse_args()))
