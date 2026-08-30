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
from typing import Any, Sequence

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

import build_community_forensics_train_v3 as shared
from repostguard.data.community_forensics import (
    EVAL_DATASET_ID,
    EVAL_REVISION,
    MANIFEST_FIELDS,
    _atomic_bytes,
    _atomic_csv,
    _atomic_json,
    _atomic_text,
    _extension,
    _inspect_image,
    _sha256_bytes,
    _sha256_file,
    _stable_digest,
    perceptual_hash,
)


SELECTION_SEED = 20260830
EXPECTED_BASE_COUNTS = {0: 1_000, 1: 1_000}
EXPECTED_ADDITION_COUNTS = {0: 1_000, 1: 1_000}
EXPECTED_OUTPUT_COUNTS = {0: 2_000, 1: 2_000}
REAL_SOURCES = ("COCO", "FFHQ", "LAION", "RAISE")
UNSEEN_ARCHITECTURES = ("Commercial", "Other")
STOP_REQUESTED = False


def _request_stop(signum: int, frame: object) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def _connect_state(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=120)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
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
            real_source TEXT NOT NULL,
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
    connection = sqlite3.connect(
        f"file:{path.resolve()}?mode=ro", uri=True, timeout=120
    )
    connection.row_factory = sqlite3.Row
    return connection


def _frozen_manifest_rows(manifest_dir: Path) -> list[dict[str, str]]:
    rows_by_locator: dict[tuple[str, str, int, int], dict[str, str]] = {}
    without_locator: list[dict[str, str]] = []
    for path in sorted(manifest_dir.glob("community_forensics_*.csv")):
        if "selection_plan" in path.name or "external_unseen_v3" in path.name:
            continue
        _, rows = shared._read_csv(path)
        for row in rows:
            locator = shared._manifest_locator(row)
            if locator is None:
                without_locator.append(row)
            else:
                rows_by_locator.setdefault(locator, row)
    return list(rows_by_locator.values()) + without_locator


def _selection_tuple(row: sqlite3.Row, addition_group: str) -> tuple[Any, ...]:
    label = int(row["label"])
    class_name = "aigi" if label == 1 else "real"
    locator_hash = _stable_digest(
        SELECTION_SEED,
        "external-unseen-v3",
        addition_group,
        row["source_file"],
        row["row_group"],
        row["row_index"],
    )[:16]
    sample_id = f"cf_external_unseen_v3_{addition_group}_{locator_hash}"
    safe_name = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in str(row["image_name"])
    ).strip("._")[:80] or "image"
    relative_path = (
        Path("test_external_unseen_v3_additions")
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
        row["canonical_real_source"],
        row["official_split"],
        label,
        row["architecture"],
        relative_path,
    )


def write_plan(state: sqlite3.Connection, path: Path) -> None:
    rows = [
        dict(row)
        for row in state.execute(
            """
            SELECT sample_id, addition_group, source_file, row_group, row_index,
                   image_name, source_format, model_name_raw,
                   canonical_generator_id, real_source, official_split, label,
                   architecture, relative_path, status
            FROM selection ORDER BY addition_group, sample_id
            """
        )
    ]
    _atomic_csv(rows, tuple(rows[0].keys()), path)


def build_plan(
    state: sqlite3.Connection,
    source: sqlite3.Connection,
    frozen_rows: Sequence[dict[str, str]],
    plan_path: Path,
) -> None:
    existing = int(state.execute("SELECT COUNT(*) FROM selection").fetchone()[0])
    if existing:
        if existing != 2_000:
            raise RuntimeError(f"Incomplete external-unseen v3 plan: {existing}")
        return
    excluded = {
        locator
        for row in frozen_rows
        if (locator := shared._manifest_locator(row)) is not None
    }

    fake_by_generator: dict[str, list[sqlite3.Row]] = defaultdict(list)
    architecture_by_generator: dict[str, str] = {}
    for row in source.execute(
        """
        SELECT * FROM source_rows WHERE dataset_key='eval' AND label=1
          AND architecture IN ('Commercial','Other')
          AND canonical_generator_id!=''
        """
    ):
        locator = (
            EVAL_REVISION,
            str(row["source_file"]),
            int(row["row_group"]),
            int(row["row_index"]),
        )
        if locator in excluded:
            continue
        generator = str(row["canonical_generator_id"])
        fake_by_generator[generator].append(row)
        architecture_by_generator.setdefault(generator, str(row["architecture"]))
    if len(fake_by_generator) != 12:
        raise RuntimeError(
            f"Expected all 12 Commercial/Other generators, got {len(fake_by_generator)}"
        )
    if set(architecture_by_generator.values()) != set(UNSEEN_ARCHITECTURES):
        raise RuntimeError(
            f"Unexpected unseen architectures: {set(architecture_by_generator.values())}"
        )
    selected_fake = shared._balanced_compact_generator_rows(
        fake_by_generator, 1_000, "external_unseen_v3_aigi"
    )

    selected_real: list[sqlite3.Row] = []
    for real_source in REAL_SOURCES:
        candidates = []
        for row in source.execute(
            """
            SELECT * FROM source_rows WHERE dataset_key='eval' AND label=0
              AND canonical_real_source=?
            """,
            (real_source,),
        ):
            locator = (
                EVAL_REVISION,
                str(row["source_file"]),
                int(row["row_group"]),
                int(row["row_index"]),
            )
            if locator not in excluded:
                candidates.append(row)
        chosen, _ = shared._compact_rows(
            candidates, 250, f"external_unseen_v3_real:{real_source}"
        )
        selected_real.extend(chosen)

    plan_rows = [
        _selection_tuple(row, f"aigi_{row['architecture'].lower()}")
        for row in selected_fake
    ] + [
        _selection_tuple(row, f"real_{str(row['canonical_real_source']).lower()}")
        for row in selected_real
    ]
    if len(plan_rows) != 2_000:
        raise RuntimeError(f"Expanded-test addition plan has {len(plan_rows)} rows")
    with state:
        state.executemany(
            """
            INSERT INTO selection (
                sample_id, addition_group, source_file, row_group, row_index,
                image_name, source_format, model_name_raw,
                canonical_generator_id, real_source, official_split, label,
                architecture, relative_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            plan_rows,
        )
    write_plan(state, plan_path)


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
            SELECT source_file, row_group FROM selection WHERE status!='complete'
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
            f"datasets/{EVAL_DATASET_ID}@{EVAL_REVISION}/{group['source_file']}"
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
                        f"Expanded-test byte cap exceeded: {written + len(payload)}"
                    )
                width, height, actual_format = _inspect_image(payload)
                _atomic_bytes(payload, data_root / str(row["relative_path"]))
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
                    "event": "community_forensics_external_unseen_v3_progress",
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


def _manifest_rows(state: sqlite3.Connection) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in state.execute("SELECT * FROM selection ORDER BY sample_id"):
        label = int(row["label"])
        rows.append(
            {
                "sample_id": row["sample_id"],
                "path": row["relative_path"],
                "label": label,
                "split": "test_external_unseen_generator_v3_expanded",
                "source_dataset": "community-forensics-eval",
                "generator_id": (
                    row["canonical_generator_id"]
                    if label == 1
                    else f"real:{row['real_source']}"
                ),
                "official_split": row["official_split"],
                "project_split": "test_external_unseen_generator_v3_expanded",
                "real_source": row["real_source"],
                "model_name_raw": row["model_name_raw"],
                "canonical_generator_id": row["canonical_generator_id"],
                "architecture": row["architecture"],
                "generator_exposure": "family_unseen" if label == 1 else "not_applicable",
                "sha256": row["sha256"],
                "phash": row["phash"],
                "selection_seed": SELECTION_SEED,
                "source_revision": EVAL_REVISION,
                "source_file": row["source_file"],
                "source_row_group": row["row_group"],
                "source_row_index": row["row_index"],
                "width": row["width"],
                "height": row["height"],
                "format": row["actual_format"],
                "byte_size": row["byte_size"],
            }
        )
    return rows


def repair_phash_conflicts(
    state: sqlite3.Connection,
    source: sqlite3.Connection,
    frozen_rows: Sequence[dict[str, str]],
    data_root: Path,
    manifest_dir: Path,
    repair_round: int,
    radius: int,
) -> int:
    additions = _manifest_rows(state)
    conflicts = shared._all_phash_conflicts(frozen_rows, additions, radius)
    if not conflicts:
        return 0
    conflicts_by_victim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for conflict in conflicts:
        conflicts_by_victim[str(conflict["addition_sample_id"])].append(conflict)
    selected_locators = {
        (
            EVAL_REVISION,
            str(row["source_file"]),
            int(row["row_group"]),
            int(row["row_index"]),
        )
        for row in state.execute(
            "SELECT source_file, row_group, row_index FROM selection"
        )
    }
    frozen_locators = {
        locator
        for row in frozen_rows
        if (locator := shared._manifest_locator(row)) is not None
    }
    repairs: list[dict[str, Any]] = []
    report_path = (
        manifest_dir / "community_forensics_external_unseen_v3_phash_repairs.json"
    )
    history: list[dict[str, Any]] = []
    if report_path.is_file():
        with report_path.open("r", encoding="utf-8") as handle:
            history = list(json.load(handle).get("repairs", []))
    attempted_sample_ids = {
        str(sample_id)
        for repair in history
        for sample_id in (
            repair.get("removed_sample_id"),
            repair.get("replacement_sample_id"),
        )
        if sample_id
    }
    quarantine_root = (
        data_root.parent.parent
        / "quarantine"
        / "community_forensics_external_unseen_v3_phash"
        / f"round_{repair_round}"
    )

    for victim_id in sorted(conflicts_by_victim):
        old = state.execute(
            "SELECT * FROM selection WHERE sample_id=?", (victim_id,)
        ).fetchone()
        if old is None:
            raise RuntimeError(f"Missing expanded-test repair victim: {victim_id}")
        parameters: list[Any] = [int(old["label"])]
        if int(old["label"]) == 1:
            identity_clause = "canonical_generator_id=?"
            parameters.append(str(old["canonical_generator_id"]))
        else:
            identity_clause = "canonical_real_source=?"
            parameters.append(str(old["real_source"]))
        candidates = list(
            source.execute(
                f"""
                SELECT * FROM source_rows WHERE dataset_key='eval' AND label=?
                  AND {identity_clause}
                """,
                parameters,
            )
        )
        eligible: list[sqlite3.Row] = []
        for candidate in candidates:
            locator = (
                EVAL_REVISION,
                str(candidate["source_file"]),
                int(candidate["row_group"]),
                int(candidate["row_index"]),
            )
            candidate_sample_id = _selection_tuple(
                candidate, str(old["addition_group"])
            )[0]
            if (
                locator not in selected_locators
                and locator not in frozen_locators
                and candidate_sample_id not in attempted_sample_ids
            ):
                eligible.append(candidate)
        if not eligible:
            raise RuntimeError(
                f"No expanded-test replacement for {victim_id} in "
                f"group={old['addition_group']}"
            )
        replacement = min(
            eligible,
            key=lambda row: (
                # Once a candidate conflicts, leave that row group rather than
                # oscillating among visually identical placeholder images.
                int(
                    str(row["source_file"]) == str(old["source_file"])
                    and int(row["row_group"]) == int(old["row_group"])
                ),
                _stable_digest(
                    SELECTION_SEED,
                    "external-unseen-v3-phash-replacement",
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
            EVAL_REVISION,
            str(old["source_file"]),
            int(old["row_group"]),
            int(old["row_index"]),
        )
        replacement_locator = (
            EVAL_REVISION,
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
                    canonical_generator_id, real_source, official_split, label,
                    architecture, relative_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                replacement_values,
            )
        selected_locators.remove(old_locator)
        selected_locators.add(replacement_locator)
        attempted_sample_ids.add(victim_id)
        attempted_sample_ids.add(str(replacement_values[0]))
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
                "real_source": old["real_source"],
                "conflicts": conflicts_by_victim[victim_id],
            }
        )

    history.extend(repairs)
    _atomic_json(
        {
            "selection_seed": SELECTION_SEED,
            "phash_hamming_threshold": radius,
            "repair_policy": (
                "same_exact_generator_or_real_source_replacement_blacklisting_"
                "all_prior_attempts_and_leaving_a_conflicting_eval_row_group"
            ),
            "repairs": history,
        },
        report_path,
    )
    write_plan(
        state,
        manifest_dir / "community_forensics_external_unseen_v3_selection_plan.csv",
    )
    print(
        json.dumps(
            {
                "event": "community_forensics_external_unseen_v3_phash_repair",
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
        raise RuntimeError(f"Cannot finalize expanded test with {incomplete} incomplete")
    fields, base = shared._read_csv(Path(arguments.base_manifest))
    if tuple(fields) != tuple(MANIFEST_FIELDS):
        raise RuntimeError("Frozen external-unseen manifest schema changed")
    if shared._counts(base) != EXPECTED_BASE_COUNTS:
        raise RuntimeError(f"Unexpected frozen test counts: {shared._counts(base)}")
    additions = _manifest_rows(state)
    if shared._counts(additions) != EXPECTED_ADDITION_COUNTS:
        raise RuntimeError(f"Unexpected addition counts: {shared._counts(additions)}")

    frozen = _frozen_manifest_rows(Path(arguments.manifest_dir))
    overlap = shared._identity_overlap(frozen, additions)
    if any(overlap.values()):
        raise RuntimeError(f"Expanded-test additions overlap frozen data: {overlap}")
    conflicts = shared._phash_conflicts(
        frozen, additions, int(arguments.phash_threshold)
    )
    if conflicts:
        raise RuntimeError(f"Expanded-test pHash conflicts: {conflicts}")

    derived_base: list[dict[str, Any]] = []
    for source_row in base:
        row = dict(source_row)
        row["split"] = "test_external_unseen_generator_v3_expanded"
        row["project_split"] = "test_external_unseen_generator_v3_expanded"
        derived_base.append(row)
    combined: list[dict[str, Any]] = derived_base + additions
    if len(combined) != 4_000 or shared._counts(combined) != EXPECTED_OUTPUT_COUNTS:
        raise RuntimeError(
            f"Unexpected expanded test: rows={len(combined)} counts={shared._counts(combined)}"
        )
    generator_counts = Counter(
        str(row["canonical_generator_id"])
        for row in additions
        if int(row["label"]) == 1
    )
    architecture_counts = Counter(
        str(row["architecture"])
        for row in additions
        if int(row["label"]) == 1
    )
    real_counts = Counter(
        str(row["real_source"])
        for row in additions
        if int(row["label"]) == 0
    )
    if len(generator_counts) != 12 or max(generator_counts.values()) - min(
        generator_counts.values()
    ) > 1:
        raise RuntimeError(f"Unbalanced exact-generator additions: {generator_counts}")
    if set(architecture_counts) != set(UNSEEN_ARCHITECTURES):
        raise RuntimeError(f"Unexpected architectures: {architecture_counts}")
    if real_counts != Counter({source: 250 for source in REAL_SOURCES}):
        raise RuntimeError(f"Unbalanced real-source additions: {real_counts}")

    additions_path = Path(arguments.additions_manifest)
    output_path = Path(arguments.output)
    _atomic_csv(additions, MANIFEST_FIELDS, additions_path)
    _atomic_csv(combined, MANIFEST_FIELDS, output_path)
    write_plan(state, Path(arguments.selection_plan))
    data_root = Path(arguments.data_root)
    missing = [
        str(row["path"])
        for row in combined
        if not (data_root / str(row["path"])).is_file()
    ]
    if missing:
        raise RuntimeError(f"Expanded test has {len(missing)} missing images")

    combined_architectures = Counter(
        str(row["architecture"])
        for row in combined
        if int(row["label"]) == 1
    )
    combined_real_sources = Counter(
        str(row["real_source"])
        for row in combined
        if int(row["label"]) == 0
    )
    audit = {
        "protocol_id": "community_forensics_external_unseen_v3_expanded",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_seed": SELECTION_SEED,
        "source": {
            "dataset_id": EVAL_DATASET_ID,
            "revision": EVAL_REVISION,
            "license": "CC-BY-NC-SA-4.0",
        },
        "base": {
            "manifest": str(Path(arguments.base_manifest).resolve()),
            "manifest_sha256": _sha256_file(Path(arguments.base_manifest)),
            "rows": 2_000,
            "class_counts": {"real": 1_000, "aigi": 1_000},
        },
        "additions": {
            "manifest": str(additions_path.resolve()),
            "manifest_sha256": _sha256_file(additions_path),
            "rows": 2_000,
            "class_counts": {"real": 1_000, "aigi": 1_000},
            "exact_generator_count": len(generator_counts),
            "exact_generator_counts": dict(sorted(generator_counts.items())),
            "architecture_counts": dict(sorted(architecture_counts.items())),
            "real_source_counts": dict(sorted(real_counts.items())),
            "materialized_bytes": int(
                state.execute("SELECT SUM(byte_size) FROM selection").fetchone()[0]
            ),
        },
        "output": {
            "manifest": str(output_path.resolve()),
            "manifest_sha256": _sha256_file(output_path),
            "rows": 4_000,
            "class_counts": {"real": 2_000, "aigi": 2_000},
            "architecture_counts": dict(sorted(combined_architectures.items())),
            "real_source_counts": dict(sorted(combined_real_sources.items())),
        },
        "diversity_limit": (
            "The pinned Eval revision contains exactly 12 Commercial/Other exact "
            "generators (11 Commercial and one Other); all were already present "
            "in the frozen test and all remain represented after expansion."
        ),
        "integrity": {
            "all_frozen_manifest_addition_identity_overlap": overlap,
            "phash_hamming_threshold": int(arguments.phash_threshold),
            "phash_near_duplicate_count": 0,
            "all_materialized_paths_verified": True,
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
                "event": "community_forensics_external_unseen_v3_complete",
                "rows": 4_000,
                "addition_rows": 2_000,
                "exact_generators": len(generator_counts),
                "real_sources": dict(sorted(real_counts.items())),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def verify_state(arguments: argparse.Namespace) -> None:
    state = _connect_state(Path(arguments.state_database))
    try:
        integrity = str(state.execute("PRAGMA integrity_check").fetchone()[0])
        planned = int(state.execute("SELECT COUNT(*) FROM selection").fetchone()[0])
        completed = int(
            state.execute(
                "SELECT COUNT(*) FROM selection WHERE status='complete'"
            ).fetchone()[0]
        )
        if integrity != "ok" or planned != 2_000:
            raise RuntimeError(
                f"Invalid state: integrity={integrity} planned={planned}"
            )
        print(
            json.dumps(
                {
                    "event": "community_forensics_external_unseen_v3_state_verified",
                    "planned": planned,
                    "completed": completed,
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
    state = _connect_state(Path(arguments.state_database))
    source = _source_connection(Path(arguments.source_database))
    try:
        frozen = _frozen_manifest_rows(Path(arguments.manifest_dir))
        build_plan(state, source, frozen, Path(arguments.selection_plan))
        data_root = Path(arguments.data_root)
        manifest_dir = Path(arguments.manifest_dir)
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
                frozen,
                data_root,
                manifest_dir,
                repair_round,
                int(arguments.phash_threshold),
            )
            if not repaired:
                break
        else:
            raise RuntimeError(
                "expanded external-unseen pHash repair did not converge in 20 rounds"
            )
        finalize(arguments, state)
        return 0
    finally:
        source.close()
        state.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand the frozen external unseen-generator test by 1000 AIGI "
            "and 1000 source-balanced real Eval images"
        )
    )
    parser.add_argument(
        "--source-database", default="data/state/community_forensics.sqlite3"
    )
    parser.add_argument(
        "--state-database",
        default="data/state/community_forensics_external_unseen_v3.sqlite3",
    )
    parser.add_argument("--data-root", default="data/raw/community_forensics")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument(
        "--base-manifest",
        default="data/manifests/community_forensics_test_external_unseen_generator.csv",
    )
    parser.add_argument(
        "--selection-plan",
        default="data/manifests/community_forensics_external_unseen_v3_selection_plan.csv",
    )
    parser.add_argument(
        "--additions-manifest",
        default="data/manifests/community_forensics_external_unseen_v3_additions.csv",
    )
    parser.add_argument(
        "--output",
        default="data/manifests/community_forensics_test_external_unseen_generator_v3_expanded.csv",
    )
    parser.add_argument(
        "--audit",
        default="data/manifests/community_forensics_external_unseen_v3_audit.json",
    )
    parser.add_argument(
        "--complete-marker",
        default="data/raw/community_forensics/EXTERNAL_UNSEEN_V3_COMPLETE",
    )
    parser.add_argument("--phash-threshold", type=int, default=4)
    parser.add_argument("--max-materialized-gib", type=float, default=12.0)
    parser.add_argument("--verify-state", action="store_true")
    arguments = parser.parse_args()
    if not 0 <= arguments.phash_threshold <= 64:
        parser.error("--phash-threshold must be in [0, 64]")
    return arguments


if __name__ == "__main__":
    sys.exit(run(parse_args()))
