from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _rows(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _generator_groups(rows: list[dict[str, str]]) -> set[str]:
    return {
        str(row.get("canonical_generator_id") or row.get("generator_id") or "")
        for row in rows
        if int(row["label"]) == 1
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate family-unseen split isolation")
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--protected", action="append", default=[])
    args = parser.parse_args()
    train = _rows(args.train)
    dev = _rows(args.dev)
    train_ids = {row["sample_id"] for row in train}
    dev_ids = {row["sample_id"] for row in dev}
    train_hashes = {row["sha256"].lower() for row in train}
    dev_hashes = {row["sha256"].lower() for row in dev}
    checks = {
        "sample_ids_disjoint": not train_ids.intersection(dev_ids),
        "sha256_disjoint": not train_hashes.intersection(dev_hashes),
        "generator_groups_disjoint": not _generator_groups(train).intersection(
            _generator_groups(dev)
        ),
        "dev_balanced": sum(int(row["label"]) == 0 for row in dev)
        == sum(int(row["label"]) == 1 for row in dev),
        "dev_aigi_marked_family_unseen": all(
            row.get("generator_exposure") == "family_unseen_dev"
            for row in dev
            if int(row["label"]) == 1
        ),
    }
    for protected_path in args.protected:
        protected = _rows(protected_path)
        protected_ids = {row["sample_id"] for row in protected}
        protected_hashes = {row["sha256"].lower() for row in protected}
        name = Path(protected_path).stem
        checks[f"protected_{name}_sample_ids_disjoint"] = not (
            train_ids.union(dev_ids).intersection(protected_ids)
        )
        checks[f"protected_{name}_sha256_disjoint"] = not (
            train_hashes.union(dev_hashes).intersection(protected_hashes)
        )
    failures = sorted(name for name, passed in checks.items() if not passed)
    result = {
        "event": "family_unseen_dev_validation",
        "train_rows": len(train),
        "dev_rows": len(dev),
        "dev_generator_groups": len(_generator_groups(dev)),
        "checks": checks,
        "accepted": not failures,
        "failures": failures,
    }
    print(json.dumps(result, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
