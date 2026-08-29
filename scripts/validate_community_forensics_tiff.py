from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = False


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate(database: Path, data_root: Path, report_path: Path) -> int:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = list(
            connection.execute(
                """
                SELECT sample_id, relative_path, sha256, width, height, byte_size
                FROM selection
                WHERE status='complete' AND actual_format='TIFF'
                ORDER BY sample_id
                """
            )
        )
    finally:
        connection.close()

    failures: list[dict[str, str]] = []
    warning_files: list[dict[str, Any]] = []
    modes: Counter[str] = Counter()
    frame_counts: Counter[int] = Counter()
    widths: list[int] = []
    heights: list[int] = []
    byte_sizes: list[int] = []

    for index, row in enumerate(rows, start=1):
        path = data_root / str(row["relative_path"])
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            size = path.stat().st_size
            if size != int(row["byte_size"]):
                raise RuntimeError(f"size mismatch: file={size} database={row['byte_size']}")
            digest = file_sha256(path)
            if digest != str(row["sha256"]):
                raise RuntimeError(
                    f"SHA256 mismatch: file={digest} database={row['sha256']}"
                )

            captured: list[str] = []
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with Image.open(path) as verify_image:
                    if verify_image.format != "TIFF":
                        raise RuntimeError(f"Pillow format is {verify_image.format!r}")
                    verify_image.verify()
                with Image.open(path) as image:
                    if image.size != (int(row["width"]), int(row["height"])):
                        raise RuntimeError(
                            f"dimension mismatch: file={image.size} "
                            f"database={(row['width'], row['height'])}"
                        )
                    frames = int(getattr(image, "n_frames", 1))
                    for frame in range(frames):
                        image.seek(frame)
                        image.load()
                    modes[image.mode] += 1
                    frame_counts[frames] += 1
                    widths.append(image.width)
                    heights.append(image.height)
                    byte_sizes.append(size)
                captured = [str(item.message) for item in caught]
            if captured:
                warning_files.append(
                    {
                        "sample_id": row["sample_id"],
                        "path": str(path),
                        "warnings": captured,
                    }
                )
        except Exception as error:  # report every damaged/unreadable file together
            failures.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "path": str(path),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        if index % 50 == 0 or index == len(rows):
            print(
                json.dumps(
                    {
                        "event": "tiff_integrity_progress",
                        "checked": index,
                        "total": len(rows),
                        "failures": len(failures),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    report = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "database": str(database),
        "data_root": str(data_root),
        "checks": [
            "file_exists",
            "byte_size_matches_sqlite",
            "sha256_matches_sqlite",
            "pillow_verify",
            "all_frames_full_decode",
            "dimensions_match_sqlite",
        ],
        "tiff_rows": len(rows),
        "valid_files": len(rows) - len(failures),
        "failed_files": len(failures),
        "warning_files": len(warning_files),
        "modes": dict(sorted(modes.items())),
        "frame_counts": {str(key): value for key, value in sorted(frame_counts.items())},
        "width_range": [min(widths), max(widths)] if widths else None,
        "height_range": [min(heights), max(heights)] if heights else None,
        "byte_size_range": [min(byte_sizes), max(byte_sizes)] if byte_sizes else None,
        "failures": failures,
        "warnings": warning_files,
    }
    atomic_json(report, report_path)
    print(json.dumps({key: report[key] for key in ("tiff_rows", "valid_files", "failed_files", "warning_files")}, sort_keys=True), flush=True)
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Fully decode and checksum Community Forensics TIFF files")
    parser.add_argument("--database", default="data/state/community_forensics.sqlite3")
    parser.add_argument("--data-root", default="data/raw/community_forensics")
    parser.add_argument(
        "--report", default="reports/data_statistics/community_forensics_tiff_integrity.json"
    )
    arguments = parser.parse_args()
    raise SystemExit(
        validate(
            Path(arguments.database).expanduser().resolve(),
            Path(arguments.data_root).expanduser().resolve(),
            Path(arguments.report).expanduser().resolve(),
        )
    )


if __name__ == "__main__":
    main()
