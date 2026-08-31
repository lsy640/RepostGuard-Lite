from __future__ import annotations

import argparse
import csv
import io
from collections import defaultdict
from pathlib import Path
from typing import Any

from repostguard.checkpoint import atomic_text


def _read(path: str) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not fields or not rows:
        raise ValueError(f"Empty manifest: {path}")
    return fields, rows


def _balanced_diverse_subset(
    rows: list[dict[str, str]], per_label: int
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for label in ("0", "1"):
        groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            if row["label"] != label:
                continue
            generator = row["generator_id"] if label == "1" else "real"
            groups[(row["source_dataset"], generator)].append(row)
        for values in groups.values():
            values.sort(key=lambda row: (row.get("sha256", ""), row["sample_id"]))
        ordered_groups = [groups[key] for key in sorted(groups)]
        label_rows: list[dict[str, str]] = []
        position = 0
        while len(label_rows) < per_label:
            added = False
            for values in ordered_groups:
                if position < len(values):
                    label_rows.append(values[position])
                    added = True
                    if len(label_rows) == per_label:
                        break
            if not added:
                raise ValueError(f"Manifest has fewer than {per_label} label={label} rows")
            position += 1
        selected.extend(label_rows)
    return sorted(selected, key=lambda row: (row["label"], row["sample_id"]))


def _write(path: str, fields: list[str], rows: list[dict[str, Any]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(buffer.getvalue(), path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic Student smoke manifests")
    parser.add_argument("--train-input", required=True)
    parser.add_argument("--validation-input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--validation-output", required=True)
    parser.add_argument("--train-per-label", type=int, default=32)
    parser.add_argument("--validation-per-label", type=int, default=16)
    arguments = parser.parse_args()
    train_fields, train_rows = _read(arguments.train_input)
    validation_fields, validation_rows = _read(arguments.validation_input)
    train_subset = _balanced_diverse_subset(train_rows, arguments.train_per_label)
    validation_subset = _balanced_diverse_subset(
        validation_rows, arguments.validation_per_label
    )
    _write(arguments.train_output, train_fields, train_subset)
    _write(arguments.validation_output, validation_fields, validation_subset)
    print(
        f"student smoke manifests: train={len(train_subset)} "
        f"validation={len(validation_subset)}"
    )


if __name__ == "__main__":
    main()
