from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames or [])


def _write_rows(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    """Keep audit provenance usable after the repo is copied to TC2."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _group_key(row: dict[str, str]) -> str:
    return str(row.get("canonical_generator_id") or row.get("generator_id") or "")


def _select_generator_groups(
    rows: list[dict[str, str]], *, target: int, seed: int
) -> tuple[set[str], dict[str, Any]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if int(row["label"]) != 1:
            continue
        key = _group_key(row)
        if not key:
            raise ValueError(f"AIGI row is missing generator identity: {row['sample_id']}")
        groups[key].append(row)
    by_architecture: dict[str, list[str]] = defaultdict(list)
    for key, group_rows in groups.items():
        architectures = {str(row.get("architecture", "unknown")) for row in group_rows}
        if len(architectures) != 1:
            raise ValueError(f"Generator group crosses architectures: {key}")
        by_architecture[next(iter(architectures))].append(key)
    rng = random.Random(seed)
    for keys in by_architecture.values():
        rng.shuffle(keys)

    selected: set[str] = set()
    selected_rows = 0
    architectures = sorted(by_architecture)
    for architecture in architectures:
        candidates = by_architecture[architecture]
        if not candidates:
            continue
        mandatory = min(candidates, key=lambda key: (len(groups[key]), key))
        if selected_rows + len(groups[mandatory]) <= target:
            selected.add(mandatory)
            selected_rows += len(groups[mandatory])
            candidates.remove(mandatory)
    progress = True
    while progress and selected_rows < target:
        progress = False
        for architecture in architectures:
            candidates = by_architecture[architecture]
            while candidates:
                key = candidates.pop()
                size = len(groups[key])
                if selected_rows + size <= target:
                    selected.add(key)
                    selected_rows += size
                    progress = True
                    break
            if selected_rows >= target:
                break

    if selected_rows < target:
        remaining = [key for key in groups if key not in selected]
        remaining.sort(key=lambda key: (len(groups[key]), key))
        while remaining and selected_rows < target:
            best = min(
                remaining,
                key=lambda key: (abs(target - (selected_rows + len(groups[key]))), key),
            )
            selected.add(best)
            selected_rows += len(groups[best])
            remaining.remove(best)

    selected_architectures = Counter(
        row.get("architecture", "unknown")
        for key in selected
        for row in groups[key]
    )
    return selected, {
        "target_aigi_rows": int(target),
        "selected_aigi_rows": int(selected_rows),
        "selected_generator_groups": sorted(selected),
        "selected_architectures": dict(sorted(selected_architectures.items())),
    }


def build_split(
    input_manifest: Path,
    train_output: Path,
    dev_output: Path,
    audit_output: Path,
    *,
    target_per_class: int,
    seed: int,
) -> None:
    rows, fields = _read_rows(input_manifest)
    required = {"sample_id", "label", "sha256", "canonical_generator_id", "architecture"}
    missing = sorted(required.difference(fields))
    if missing:
        raise ValueError(f"Manifest is missing fields: {missing}")
    selected_groups, selection = _select_generator_groups(
        rows, target=target_per_class, seed=seed
    )
    dev_aigi = [
        dict(row)
        for row in rows
        if int(row["label"]) == 1 and _group_key(row) in selected_groups
    ]
    rng = random.Random(seed + 1)
    real_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if int(row["label"]) == 0:
            real_by_source[str(row.get("real_source", "UNSPECIFIED"))].append(row)
    for source_rows in real_by_source.values():
        rng.shuffle(source_rows)
    all_real = sum(len(source_rows) for source_rows in real_by_source.values())
    real_target = len(dev_aigi)
    real_quotas = {
        source: int(round(real_target * len(source_rows) / all_real))
        for source, source_rows in real_by_source.items()
    }
    while sum(real_quotas.values()) > real_target:
        source = max(real_quotas, key=lambda name: (real_quotas[name], name))
        real_quotas[source] -= 1
    while sum(real_quotas.values()) < real_target:
        source = max(
            real_by_source,
            key=lambda name: (
                len(real_by_source[name]) - real_quotas[name],
                name,
            ),
        )
        real_quotas[source] += 1
    dev_real = [
        dict(row)
        for source, source_rows in sorted(real_by_source.items())
        for row in source_rows[: real_quotas[source]]
    ]
    dev_ids = {row["sample_id"] for row in dev_aigi + dev_real}
    train_rows = [dict(row) for row in rows if row["sample_id"] not in dev_ids]
    dev_rows = dev_aigi + dev_real
    rng.shuffle(dev_rows)
    for row in dev_rows:
        row["split"] = "val_family_unseen_dev_v1"
        row["project_split"] = "val_family_unseen_dev_v1"
        if int(row["label"]) == 1:
            row["generator_exposure"] = "family_unseen_dev"
    train_generator_groups = {
        _group_key(row) for row in train_rows if int(row["label"]) == 1
    }
    overlap = sorted(selected_groups.intersection(train_generator_groups))
    if overlap:
        raise ValueError(f"Held-out generator leakage into training: {overlap[:5]}")
    if len(dev_aigi) != len(dev_real):
        raise ValueError("Development split is not class balanced")
    _write_rows(train_output, train_rows, fields)
    _write_rows(dev_output, dev_rows, fields)
    audit = {
        "schema_version": 1,
        "seed": int(seed),
        "input_manifest": _portable_path(input_manifest),
        "input_sha256": _sha256(input_manifest),
        "train_manifest": _portable_path(train_output),
        "train_sha256": _sha256(train_output),
        "dev_manifest": _portable_path(dev_output),
        "dev_sha256": _sha256(dev_output),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "dev_label_counts": dict(Counter(row["label"] for row in dev_rows)),
        "dev_real_sources": dict(Counter(row.get("real_source", "") for row in dev_real)),
        **selection,
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a generator-family-disjoint Student train/dev split"
    )
    parser.add_argument(
        "--input",
        default="data/manifests/community_forensics_train_v3.csv",
    )
    parser.add_argument(
        "--train-output",
        default="data/manifests/community_forensics_train_v31_family_holdout.csv",
    )
    parser.add_argument(
        "--dev-output",
        default="data/manifests/community_forensics_val_family_unseen_dev_v1.csv",
    )
    parser.add_argument(
        "--audit-output",
        default="data/manifests/community_forensics_family_unseen_dev_v1_audit.json",
    )
    parser.add_argument("--target-per-class", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    build_split(
        Path(args.input),
        Path(args.train_output),
        Path(args.dev_output),
        Path(args.audit_output),
        target_per_class=args.target_per_class,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
