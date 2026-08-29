from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE = PROJECT_ROOT / "data/state/community_forensics.sqlite3"
AIGIBENCH_DATASET_ID = "TheKernel01/AIGIBench"
HARD_GENERATORS = ("hourglass", "dfgan", "galip")


def _size(value: Any) -> int:
    return int(value or 0)


def main() -> None:
    api = HfApi()
    info = api.dataset_info(
        AIGIBENCH_DATASET_ID,
        revision="main",
        files_metadata=True,
    )
    revision = str(info.sha)
    parquet_files = sorted(
        sibling.rfilename
        for sibling in list(info.siblings or [])
        if sibling.rfilename.endswith(".parquet")
    )
    if not parquet_files:
        raise RuntimeError(f"No Parquet files found for {AIGIBENCH_DATASET_ID}")

    filesystem = HfFileSystem()
    representative_file = next(
        (
            source_file
            for source_file in parquet_files
            if "validation" in source_file.lower() or "valid" in source_file.lower()
        ),
        parquet_files[0],
    )
    remote_path = (
        f"datasets/{AIGIBENCH_DATASET_ID}@{revision}/{representative_file}"
    )
    with filesystem.open(remote_path, "rb") as handle:
        parquet = pq.ParquetFile(handle)
        schema = list(parquet.schema_arrow.names)
        metadata = {
            key.decode("utf-8", errors="replace"): value.decode(
                "utf-8", errors="replace"
            )
            for key, value in (parquet.schema_arrow.metadata or {}).items()
        }
        required = {"image", "label", "generator"}
        missing = required.difference(schema)
        if missing:
            raise RuntimeError(
                f"Representative AIGIBench file missing columns: {sorted(missing)}"
            )
        sample = parquet.read_row_group(0, columns=["label", "generator"])
        label_counts = Counter(str(value.as_py()) for value in sample["label"])
        generator_counts = Counter(
            str(value.as_py()) for value in sample["generator"]
        )

    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    try:
        hard_counts = {
            generator: int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM source_rows
                    WHERE dataset_key='eval' AND label=1
                      AND canonical_generator_id=?
                    """,
                    (generator,),
                ).fetchone()[0]
            )
            for generator in HARD_GENERATORS
        }
        selected_hard_counts = {
            generator: int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM selection
                    WHERE dataset_key='eval' AND label=1
                      AND canonical_generator_id=?
                    """,
                    (generator,),
                ).fetchone()[0]
            )
            for generator in HARD_GENERATORS
        }
    finally:
        connection.close()

    payload = {
        "event": "community_forensics_validation_v2_probe",
        "aigibench": {
            "dataset_id": AIGIBENCH_DATASET_ID,
            "resolved_revision": revision,
            "parquet_file_count": len(parquet_files),
            "parquet_total_bytes": sum(
                _size(sibling.size)
                for sibling in list(info.siblings or [])
                if sibling.rfilename in parquet_files
            ),
            "parquet_files": parquet_files,
            "representative_file": representative_file,
            "representative_rows": int(parquet.metadata.num_rows),
            "representative_row_groups": int(parquet.metadata.num_row_groups),
            "schema": schema,
            "schema_metadata": metadata,
            "first_row_group_label_counts": dict(label_counts),
            "first_row_group_generator_counts": dict(generator_counts),
        },
        "eval_hard_generators": {
            "available_rows": hard_counts,
            "already_selected_rows": selected_hard_counts,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
