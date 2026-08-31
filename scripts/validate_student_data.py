from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from repostguard.checkpoint import atomic_text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(item: tuple[Path, int | None, str]) -> dict[str, object] | None:
    path, expected_bytes, expected_sha256 = item
    if not path.is_file():
        return {"path": str(path), "error": "missing"}
    observed_bytes = path.stat().st_size
    if expected_bytes is not None and observed_bytes != expected_bytes:
        return {
            "path": str(path),
            "error": "byte_size",
            "expected": expected_bytes,
            "observed": observed_bytes,
        }
    observed_sha256 = _sha256(path)
    if observed_sha256 != expected_sha256:
        return {
            "path": str(path),
            "error": "sha256",
            "expected": expected_sha256,
            "observed": observed_sha256,
        }
    return None


def validate(
    data_root: str,
    manifests: list[str],
    output_path: str,
    workers: int,
) -> dict[str, object]:
    root = Path(data_root).expanduser().resolve()
    inventory: dict[Path, tuple[int | None, str]] = {}
    manifest_summaries: list[dict[str, object]] = []
    for manifest_value in manifests:
        manifest = Path(manifest_value).expanduser().resolve()
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Empty manifest: {manifest}")
        labels = {"0": 0, "1": 0}
        for row in rows:
            labels[row["label"]] += 1
            relative = Path(row["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe manifest path: {relative}")
            absolute = (root / relative).resolve()
            if not absolute.is_relative_to(root):
                raise ValueError(f"Manifest path escapes data root: {relative}")
            expected_sha256 = str(row["sha256"]).lower()
            if len(expected_sha256) != 64:
                raise ValueError(f"Invalid SHA256 for {row['sample_id']}")
            expected_bytes = int(row["byte_size"]) if row.get("byte_size") else None
            previous = inventory.get(absolute)
            lineage = (expected_bytes, expected_sha256)
            if previous is not None and previous != lineage:
                raise ValueError(f"Conflicting manifest lineage for {absolute}")
            inventory[absolute] = lineage
        manifest_summaries.append(
            {
                "path": str(manifest),
                "sha256": _sha256(manifest),
                "rows": len(rows),
                "labels": labels,
            }
        )
    items = [(path, values[0], values[1]) for path, values in inventory.items()]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        failures = [failure for failure in executor.map(_verify, items) if failure]
    payload: dict[str, object] = {
        "schema_version": 1,
        "data_root": str(root),
        "manifests": manifest_summaries,
        "unique_files": len(inventory),
        "verified_bytes": sum(path.stat().st_size for path in inventory if path.is_file()),
        "workers": workers,
        "accepted": not failures,
        "failures": failures[:20],
        "failure_count": len(failures),
    }
    atomic_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", output_path)
    print(json.dumps(payload, sort_keys=True), flush=True)
    if failures:
        raise SystemExit(2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify every Student manifest image")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=10)
    arguments = parser.parse_args()
    if arguments.workers <= 0:
        raise ValueError("workers must be positive")
    validate(
        arguments.data_root,
        arguments.manifest,
        arguments.output,
        arguments.workers,
    )


if __name__ == "__main__":
    main()
