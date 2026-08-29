from __future__ import annotations

import json
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem


DATASETS = (
    ("OwensLab/CommunityForensics-Small", "data/"),
    ("OwensLab/CommunityForensics-Eval", "data/CompEval-"),
)
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


def _json_value(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return value.as_py()
    return value


def main() -> None:
    api = HfApi()
    filesystem = HfFileSystem()
    for dataset_id, parquet_prefix in DATASETS:
        info = api.dataset_info(dataset_id, revision="main", files_metadata=True)
        revision = str(info.sha)
        siblings = list(info.siblings or [])
        parquet_files = sorted(
            sibling.rfilename
            for sibling in siblings
            if sibling.rfilename.startswith(parquet_prefix)
            and sibling.rfilename.endswith(".parquet")
        )
        if not parquet_files:
            raise RuntimeError(f"No Parquet files found for {dataset_id}")
        total_bytes = sum(int(sibling.size or 0) for sibling in siblings if sibling.rfilename in parquet_files)
        first_file = parquet_files[0]
        remote_path = f"datasets/{dataset_id}@{revision}/{first_file}"
        with filesystem.open(remote_path, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            available = set(parquet.schema_arrow.names)
            missing = set(METADATA_COLUMNS).difference(available)
            if missing:
                raise RuntimeError(f"{dataset_id} missing metadata columns: {sorted(missing)}")
            table = parquet.read_row_group(0, columns=list(METADATA_COLUMNS))
            first_row = {
                column: _json_value(table[column][0])
                for column in METADATA_COLUMNS
            }
            payload = {
                "event": "community_forensics_probe",
                "dataset_id": dataset_id,
                "resolved_revision": revision,
                "parquet_file_count": len(parquet_files),
                "parquet_total_bytes": total_bytes,
                "first_file": first_file,
                "first_file_size": next(
                    int(sibling.size or 0)
                    for sibling in siblings
                    if sibling.rfilename == first_file
                ),
                "first_file_rows": parquet.metadata.num_rows,
                "first_file_row_groups": parquet.metadata.num_row_groups,
                "first_row_group_rows": parquet.metadata.row_group(0).num_rows,
                "schema": parquet.schema_arrow.names,
                "first_row_metadata": first_row,
            }
        print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

