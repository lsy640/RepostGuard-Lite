from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _rank(row: dict[str, str], seed: int) -> str:
    value = f"{seed}:{row['sample_id']}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _take(
    rows: list[dict[str, str]],
    count: int,
    *,
    seed: int,
    label: str,
) -> list[dict[str, str]]:
    unique: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row["sha256"].lower()
        unique.setdefault(key, row)
    ranked = sorted(unique.values(), key=lambda row: (_rank(row, seed), row["sample_id"]))
    if len(ranked) < count:
        raise ValueError(f"{label} has only {len(ranked)} unique rows; need {count}")
    return ranked[:count]


def build(arguments: argparse.Namespace) -> dict[str, object]:
    train_fields, train_rows = _read(arguments.train)
    main_fields, main_rows = _read(arguments.exact_seen)
    if train_fields != main_fields:
        raise ValueError("Train and exact-seen manifests use different schemas")
    _, dfgan_rows = _read(arguments.dfgan)
    _, galip_rows = _read(arguments.galip)
    _, hourglass_rows = _read(arguments.hourglass)

    real_candidates = [
        row
        for row in main_rows + dfgan_rows + galip_rows + hourglass_rows
        if row["label"] == "0"
    ]
    latdiff_candidates = [
        row
        for row in main_rows
        if row["label"] == "1" and row["architecture"] == "LatDiff"
    ]
    dfgan_candidates = [row for row in dfgan_rows if row["label"] == "1"]
    galip_candidates = [row for row in galip_rows if row["label"] == "1"]
    pixdiff_candidates = [row for row in hourglass_rows if row["label"] == "1"]

    selected = (
        _take(real_candidates, 750, seed=arguments.seed + 1, label="Real")
        + _take(latdiff_candidates, 250, seed=arguments.seed + 2, label="LatDiff")
        + _take(dfgan_candidates, 125, seed=arguments.seed + 3, label="DF-GAN")
        + _take(galip_candidates, 125, seed=arguments.seed + 4, label="GALIP")
        + _take(pixdiff_candidates, 250, seed=arguments.seed + 5, label="PixDiff")
    )
    selected.sort(key=lambda row: (_rank(row, arguments.seed + 6), row["sample_id"]))

    selected_ids = [row["sample_id"] for row in selected]
    selected_hashes = [row["sha256"].lower() for row in selected]
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Selected validation rows contain duplicate sample IDs")
    if len(set(selected_hashes)) != len(selected_hashes):
        raise ValueError("Selected validation rows contain duplicate SHA-256 values")

    train_ids = {row["sample_id"] for row in train_rows}
    train_hashes = {row["sha256"].lower() for row in train_rows}
    if train_ids.intersection(selected_ids):
        raise ValueError("Validation sample IDs overlap the full training manifest")
    if train_hashes.intersection(selected_hashes):
        raise ValueError("Validation image hashes overlap the full training manifest")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=main_fields)
        writer.writeheader()
        writer.writerows(selected)

    fake_architectures: dict[str, int] = {}
    for row in selected:
        if row["label"] == "1":
            architecture = row["architecture"]
            fake_architectures[architecture] = fake_architectures.get(architecture, 0) + 1
    result = {
        "event": "full_refit_validation_manifest",
        "output": str(arguments.output),
        "rows": len(selected),
        "real": sum(row["label"] == "0" for row in selected),
        "aigi": sum(row["label"] == "1" for row in selected),
        "aigi_architectures": dict(sorted(fake_architectures.items())),
        "train_rows": len(train_rows),
        "sample_id_overlap": 0,
        "sha256_overlap": 0,
        "seed": arguments.seed,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--exact-seen", type=Path, required=True)
    parser.add_argument("--dfgan", type=Path, required=True)
    parser.add_argument("--galip", type=Path, required=True)
    parser.add_argument("--hourglass", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
