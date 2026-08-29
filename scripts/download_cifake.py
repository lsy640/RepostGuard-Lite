from __future__ import annotations

import argparse
from pathlib import Path

import kagglehub


def _has_expected_layout(root: Path) -> bool:
    for train_directory in root.rglob("train"):
        if not train_directory.is_dir():
            continue
        children = {child.name.upper() for child in train_directory.iterdir() if child.is_dir()}
        if {"REAL", "FAKE"}.issubset(children):
            test_directory = train_directory.parent / "test"
            if test_directory.is_dir():
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the public official CIFAKE Kaggle dataset")
    parser.add_argument("--output-dir", default="data/raw/cifake")
    parser.add_argument(
        "--handle",
        default="birdy654/cifake-real-and-ai-generated-synthetic-images",
    )
    arguments = parser.parse_args()
    destination = Path(arguments.output_dir).resolve()
    if destination.exists():
        if _has_expected_layout(destination):
            print(f"CIFAKE already present: {destination}")
            return
        raise FileExistsError(
            f"Refusing to overwrite incomplete/non-CIFAKE directory: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        kagglehub.dataset_download(arguments.handle, output_dir=str(destination))
    ).resolve()
    if not _has_expected_layout(downloaded):
        raise RuntimeError(f"Downloaded dataset has no CIFAKE train/test layout: {downloaded}")
    print(f"Downloaded CIFAKE to {downloaded}")


if __name__ == "__main__":
    main()

