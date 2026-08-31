from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _keys(rows: list[dict[str, str]], field: str) -> set[str]:
    return {row[field].lower() for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--protected", type=Path, required=True)
    parser.add_argument("--expected-train-rows", type=int, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    arguments = parser.parse_args()

    train = _read(arguments.train)
    val = _read(arguments.val)
    protected = _read(arguments.protected)
    aigi_architectures: dict[str, int] = {}
    for row in val:
        if row["label"] == "1":
            architecture = row["architecture"]
            aigi_architectures[architecture] = aigi_architectures.get(architecture, 0) + 1

    checks = {
        "train_rows": len(train) == arguments.expected_train_rows,
        "train_sha256": _sha256(arguments.train) == arguments.expected_train_sha256,
        "val_rows": len(val) == 1500,
        "val_real": sum(row["label"] == "0" for row in val) == 750,
        "val_aigi": sum(row["label"] == "1" for row in val) == 750,
        "val_architectures": aigi_architectures
        == {"GAN": 250, "LatDiff": 250, "PixDiff": 250},
        "train_val_id_disjoint": not _keys(train, "sample_id").intersection(
            _keys(val, "sample_id")
        ),
        "train_val_sha_disjoint": not _keys(train, "sha256").intersection(
            _keys(val, "sha256")
        ),
        "train_protected_id_disjoint": not _keys(train, "sample_id").intersection(
            _keys(protected, "sample_id")
        ),
        "train_protected_sha_disjoint": not _keys(train, "sha256").intersection(
            _keys(protected, "sha256")
        ),
        "val_protected_id_disjoint": not _keys(val, "sample_id").intersection(
            _keys(protected, "sample_id")
        ),
        "val_protected_sha_disjoint": not _keys(val, "sha256").intersection(
            _keys(protected, "sha256")
        ),
    }
    accepted = all(checks.values())
    payload = {
        "event": "student_v32_full_refit_data_validation",
        "accepted": accepted,
        "checks": checks,
        "train_rows": len(train),
        "val_rows": len(val),
        "protected_rows": len(protected),
        "val_aigi_architectures": dict(sorted(aigi_architectures.items())),
    }
    print(json.dumps(payload, sort_keys=True))
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
