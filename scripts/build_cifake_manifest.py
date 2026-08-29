from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _locate_layout(root: Path) -> tuple[Path, Path]:
    matches: list[tuple[Path, Path]] = []
    for train_directory in root.rglob("train"):
        if not train_directory.is_dir():
            continue
        test_directory = train_directory.parent / "test"
        if not test_directory.is_dir():
            continue
        train_children = {child.name.upper() for child in train_directory.iterdir() if child.is_dir()}
        test_children = {child.name.upper() for child in test_directory.iterdir() if child.is_dir()}
        if {"REAL", "FAKE"}.issubset(train_children) and {"REAL", "FAKE"}.issubset(
            test_children
        ):
            matches.append((train_directory, test_directory))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one CIFAKE layout under {root}, found {matches}")
    return matches[0]


def _class_directory(split_directory: Path, class_name: str) -> Path:
    matches = [child for child in split_directory.iterdir() if child.name.upper() == class_name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {class_name} directory under {split_directory}")
    return matches[0]


def _shuffle(paths: list[Path], seed: int) -> list[Path]:
    generator = random.Random(seed)
    selected = sorted(paths)
    generator.shuffle(selected)
    return selected


def _atomic_csv(rows: list[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    fields = list(rows[0].keys())
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _build_rows(
    root: Path,
    split_directory: Path,
    split_name: str,
    per_class: int,
    seed: int,
    excluded_hashes: set[str],
) -> tuple[list[dict[str, Any]], set[str], int]:
    rows: list[dict[str, Any]] = []
    selected_hashes: set[str] = set()
    skipped_duplicates = 0
    for label, class_name in ((0, "REAL"), (1, "FAKE")):
        class_directory = _class_directory(split_directory, class_name)
        candidates = [
            path
            for path in class_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        accepted = 0
        for path in _shuffle(candidates, seed + label):
            content_hash = _sha256(path)
            if content_hash in excluded_hashes or content_hash in selected_hashes:
                skipped_duplicates += 1
                continue
            with Image.open(path) as image:
                width, height = image.size
                image_format = image.format or path.suffix.lstrip(".")
            rows.append(
                {
                    "sample_id": f"cifake_{split_name}_{class_name.lower()}_{accepted:05d}_{content_hash[:12]}",
                    "path": path.resolve().relative_to(root).as_posix(),
                    "label": label,
                    "split": split_name,
                    "source_dataset": "cifake",
                    "generator_id": "stable-diffusion-1.4" if label == 1 else "cifar-10",
                    "sha256": content_hash,
                    "width": width,
                    "height": height,
                    "format": image_format.upper(),
                }
            )
            selected_hashes.add(content_hash)
            accepted += 1
            if accepted == per_class:
                break
        if accepted != per_class:
            raise ValueError(
                f"Could select only {accepted}/{per_class} unique {class_name} files for {split_name}"
            )
    rows.sort(key=lambda row: row["sample_id"])
    return rows, selected_hashes, skipped_duplicates


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic balanced CIFAKE pilot manifests")
    parser.add_argument("--data-root", default="data/raw/cifake")
    parser.add_argument("--output-dir", default="data/manifests")
    parser.add_argument("--train-per-class", type=int, default=5000)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260827)
    arguments = parser.parse_args()

    root = Path(arguments.data_root).resolve()
    train_directory, test_directory = _locate_layout(root)
    train_rows, train_hashes, train_duplicates = _build_rows(
        root,
        train_directory,
        "train",
        arguments.train_per_class,
        arguments.seed,
        excluded_hashes=set(),
    )
    test_rows, test_hashes, test_duplicates = _build_rows(
        root,
        test_directory,
        "test",
        arguments.test_per_class,
        arguments.seed + 10_000,
        excluded_hashes=train_hashes,
    )
    overlap = train_hashes.intersection(test_hashes)
    if overlap:
        raise RuntimeError(f"Exact content leakage across train/test: {len(overlap)} hashes")

    output_directory = Path(arguments.output_dir).resolve()
    train_manifest = output_directory / "cifake_train_pilot.csv"
    test_manifest = output_directory / "cifake_test_pilot.csv"
    _atomic_csv(train_rows, train_manifest)
    _atomic_csv(test_rows, test_manifest)
    audit = {
        "data_root": str(root),
        "seed": arguments.seed,
        "train_manifest": str(train_manifest),
        "test_manifest": str(test_manifest),
        "train_samples": len(train_rows),
        "test_samples": len(test_rows),
        "train_labels": Counter(row["label"] for row in train_rows),
        "test_labels": Counter(row["label"] for row in test_rows),
        "train_formats": Counter(row["format"] for row in train_rows),
        "test_formats": Counter(row["format"] for row in test_rows),
        "train_sizes": Counter(f"{row['width']}x{row['height']}" for row in train_rows),
        "test_sizes": Counter(f"{row['width']}x{row['height']}" for row in test_rows),
        "train_duplicates_skipped": train_duplicates,
        "test_or_cross_split_duplicates_skipped": test_duplicates,
        "exact_train_test_overlap": 0,
        "reserved_sets_used": False,
    }
    audit_path = output_directory / "cifake_pilot_audit.json"
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
