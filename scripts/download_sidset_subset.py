from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import signal
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlopen

from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = False

LABEL_NAMES = {0: "real", 1: "full_synthetic"}
GENERATOR_IDS = {
    0: "openimages-v7",
    1: "sid-set-full-synthetic-unspecified",
}
FORMAT_EXTENSIONS = {
    "BMP": ".bmp",
    "GIF": ".gif",
    "JPEG": ".jpg",
    "PNG": ".png",
    "TIFF": ".tiff",
    "WEBP": ".webp",
}
MANIFEST_FIELDS = [
    "sample_id",
    "path",
    "label",
    "split",
    "source_dataset",
    "generator_id",
    "official_split",
    "source_image_id",
    "sha256",
    "width",
    "height",
    "format",
]

_STOP_REQUESTED = False


def _request_safe_stop(signum: int, frame: object) -> None:
    del signum, frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_stem(value: object) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return stem[:96] or "image"


def _atomic_bytes(payload: bytes, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _atomic_json(payload: dict[str, Any], destination: Path) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_bytes(encoded, destination)


def _atomic_csv(rows: list[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _image_bytes(value: object) -> bytes:
    if isinstance(value, dict):
        payload = value.get("bytes")
        if isinstance(payload, (bytes, bytearray, memoryview)):
            return bytes(payload)
        path_value = value.get("path")
        if path_value:
            path_text = str(path_value)
            if path_text.startswith(("http://", "https://")):
                with urlopen(path_text, timeout=120) as response:
                    return response.read()
            return Path(path_text).read_bytes()
    if isinstance(value, Image.Image):
        output = io.BytesIO()
        image_format = value.format or "PNG"
        value.save(output, format=image_format)
        return output.getvalue()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise TypeError(f"Unsupported SID-Set image value: {type(value)!r}")


def _inspect_image(payload: bytes) -> tuple[int, int, str, str]:
    with Image.open(io.BytesIO(payload)) as image:
        image.verify()
    with Image.open(io.BytesIO(payload)) as image:
        width, height = image.size
        image_format = (image.format or "").upper()
        image.convert("RGB").getpixel((0, 0))
    if width <= 0 or height <= 0 or not image_format:
        raise ValueError("Invalid decoded image metadata")
    extension = FORMAT_EXTENSIONS.get(image_format, f".{image_format.lower()}")
    return width, height, image_format, extension


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _initial_state(
    *,
    dataset_id: str,
    resolved_revision: str,
    official_split: str,
    target_split: str,
    per_class: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "version": 1,
        "dataset_id": dataset_id,
        "resolved_revision": resolved_revision,
        "official_split": official_split,
        "target_split": target_split,
        "per_class": per_class,
        "seed": seed,
        "source_rows_scanned": 0,
        "skipped_tampered": 0,
        "skipped_duplicate": 0,
        "skipped_invalid": 0,
        "rows": [],
    }


def _load_state(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return expected
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    for field in (
        "version",
        "dataset_id",
        "resolved_revision",
        "official_split",
        "target_split",
        "per_class",
        "seed",
    ):
        if state.get(field) != expected.get(field):
            raise ValueError(
                f"Partial state {path} has incompatible {field}: "
                f"{state.get(field)!r} != {expected.get(field)!r}"
            )
    return state


def _validate_rows(
    rows: list[dict[str, Any]], data_root: Path, per_class: int
) -> tuple[Counter[int], set[str]]:
    counts: Counter[int] = Counter()
    hashes: set[str] = set()
    sample_ids: set[str] = set()
    for row in rows:
        label = int(row["label"])
        if label not in LABEL_NAMES:
            raise ValueError(f"Unexpected label in saved state: {label}")
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe saved path: {relative}")
        image_path = (data_root / relative).resolve()
        if not image_path.is_relative_to(data_root) or not image_path.is_file():
            raise FileNotFoundError(image_path)
        digest = str(row["sha256"])
        if digest in hashes:
            raise ValueError(f"Duplicate hash within split: {digest}")
        if str(row["sample_id"]) in sample_ids:
            raise ValueError(f"Duplicate sample_id: {row['sample_id']}")
        if _sha256_file(image_path) != digest:
            raise ValueError(f"Checksum mismatch: {image_path}")
        counts[label] += 1
        hashes.add(digest)
        sample_ids.add(str(row["sample_id"]))
    if any(count > per_class for count in counts.values()):
        raise ValueError(f"Saved state exceeds requested class count: {counts}")
    return counts, hashes


def _stream_rows(
    dataset_id: str,
    official_split: str,
    resolved_revision: str,
    cache_dir: Path,
    seed: int,
    shuffle_buffer: int,
    skip_rows: int,
) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(
        dataset_id,
        split=official_split,
        revision=resolved_revision,
        streaming=True,
        cache_dir=str(cache_dir),
    )
    dataset = dataset.decode(False)
    dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    if skip_rows:
        dataset = dataset.skip(skip_rows)
    return dataset


def _materialize_split(
    *,
    dataset_id: str,
    resolved_revision: str,
    official_split: str,
    target_split: str,
    per_class: int,
    seed: int,
    shuffle_buffer: int,
    checkpoint_every: int,
    data_root: Path,
    manifest_dir: Path,
    cache_dir: Path,
    excluded_hashes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = manifest_dir / f"sidset_{target_split}.csv"
    partial_path = manifest_dir / f".sidset_{target_split}.partial.json"
    existing_manifest = _read_manifest(manifest_path)
    if existing_manifest:
        counts, hashes = _validate_rows(existing_manifest, data_root, per_class)
        if counts == Counter({0: per_class, 1: per_class}):
            if hashes.intersection(excluded_hashes):
                raise ValueError(f"Existing {target_split} manifest overlaps another split")
            return existing_manifest, {
                "source_rows_scanned": None,
                "skipped_tampered": None,
                "skipped_duplicate": None,
                "skipped_invalid": None,
                "resumed_from_complete_manifest": True,
            }
        raise ValueError(f"Incomplete existing manifest must be removed: {manifest_path}")

    expected = _initial_state(
        dataset_id=dataset_id,
        resolved_revision=resolved_revision,
        official_split=official_split,
        target_split=target_split,
        per_class=per_class,
        seed=seed,
    )
    state = _load_state(partial_path, expected)
    rows = list(state["rows"])
    counts, local_hashes = _validate_rows(rows, data_root, per_class)
    if local_hashes.intersection(excluded_hashes):
        raise ValueError(f"Partial {target_split} state overlaps another split")

    def checkpoint() -> None:
        state["rows"] = rows
        _atomic_json(state, partial_path)

    stream = _stream_rows(
        dataset_id,
        official_split,
        resolved_revision,
        cache_dir,
        seed,
        shuffle_buffer,
        int(state["source_rows_scanned"]),
    )
    since_checkpoint = 0
    for source_row in stream:
        state["source_rows_scanned"] = int(state["source_rows_scanned"]) + 1
        since_checkpoint += 1
        try:
            label = int(source_row["label"])
        except (KeyError, TypeError, ValueError):
            state["skipped_invalid"] = int(state["skipped_invalid"]) + 1
            continue
        if label == 2:
            state["skipped_tampered"] = int(state["skipped_tampered"]) + 1
            continue
        if label not in LABEL_NAMES:
            state["skipped_invalid"] = int(state["skipped_invalid"]) + 1
            continue
        if counts[label] >= per_class:
            continue

        try:
            payload = _image_bytes(source_row["image"])
            digest = _sha256_bytes(payload)
            if digest in excluded_hashes or digest in local_hashes:
                state["skipped_duplicate"] = int(state["skipped_duplicate"]) + 1
                continue
            width, height, image_format, extension = _inspect_image(payload)
        except Exception as error:
            state["skipped_invalid"] = int(state["skipped_invalid"]) + 1
            print(
                json.dumps(
                    {
                        "event": "invalid_image",
                        "official_split": official_split,
                        "img_id": str(source_row.get("img_id", "")),
                        "error": f"{type(error).__name__}: {error}",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            continue

        class_name = LABEL_NAMES[label]
        source_id = str(source_row.get("img_id", ""))
        sample_id = f"sidset_{target_split}_{class_name}_{_safe_stem(source_id)}_{digest[:12]}"
        relative_path = Path(target_split) / class_name / f"{_safe_stem(source_id)}_{digest[:12]}{extension}"
        destination = data_root / relative_path
        if destination.is_file():
            if _sha256_file(destination) != digest:
                raise ValueError(f"Existing destination has different content: {destination}")
        else:
            _atomic_bytes(payload, destination)

        rows.append(
            {
                "sample_id": sample_id,
                "path": relative_path.as_posix(),
                "label": label,
                "split": target_split,
                "source_dataset": "sid-set",
                "generator_id": GENERATOR_IDS[label],
                "official_split": official_split,
                "source_image_id": source_id,
                "sha256": digest,
                "width": width,
                "height": height,
                "format": image_format,
            }
        )
        counts[label] += 1
        local_hashes.add(digest)

        if since_checkpoint >= checkpoint_every:
            checkpoint()
            since_checkpoint = 0
            print(
                json.dumps(
                    {
                        "event": "download_progress",
                        "target_split": target_split,
                        "source_rows_scanned": state["source_rows_scanned"],
                        "class_counts": dict(sorted(counts.items())),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if _STOP_REQUESTED:
            checkpoint()
            print("SIGUSR1 progress checkpoint completed", flush=True)
            raise InterruptedError("safe stop requested")
        if counts == Counter({0: per_class, 1: per_class}):
            break

    checkpoint()
    if counts != Counter({0: per_class, 1: per_class}):
        raise ValueError(
            f"Could only select {dict(counts)} from SID-Set {official_split}; "
            f"requested {per_class} per class"
        )
    rows.sort(key=lambda row: str(row["sample_id"]))
    _atomic_csv(rows, manifest_path)
    partial_path.unlink()
    return rows, {
        "source_rows_scanned": int(state["source_rows_scanned"]),
        "skipped_tampered": int(state["skipped_tampered"]),
        "skipped_duplicate": int(state["skipped_duplicate"]),
        "skipped_invalid": int(state["skipped_invalid"]),
        "resumed_from_complete_manifest": False,
    }


def _split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    widths = [int(row["width"]) for row in rows]
    heights = [int(row["height"]) for row in rows]
    return {
        "samples": len(rows),
        "labels": dict(sorted(Counter(str(row["label"]) for row in rows).items())),
        "formats": dict(sorted(Counter(str(row["format"]) for row in rows).items())),
        "width": {"min": min(widths), "max": max(widths)},
        "height": {"min": min(heights), "max": max(heights)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream a deterministic, resumable real/full-synthetic SID-Set subset"
    )
    parser.add_argument("--dataset-id", default="saberzl/SID_Set")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--data-root", default="data/raw/sid_set")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--cache-dir", default="data/cache/huggingface/datasets")
    parser.add_argument("--train-per-class", type=int, default=10_000)
    parser.add_argument("--validation-per-class", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--shuffle-buffer", type=int, default=2_048)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    arguments = parser.parse_args()
    if arguments.train_per_class <= 0 or arguments.validation_per_class <= 0:
        parser.error("Class counts must be positive")
    if arguments.shuffle_buffer <= 0 or arguments.checkpoint_every <= 0:
        parser.error("Shuffle and checkpoint sizes must be positive")

    signal.signal(signal.SIGUSR1, _request_safe_stop)
    data_root = Path(arguments.data_root).expanduser().resolve()
    manifest_dir = Path(arguments.manifest_dir).expanduser().resolve()
    cache_dir = Path(arguments.cache_dir).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    from datasets import __version__ as datasets_version
    from huggingface_hub import HfApi

    resolved_revision = HfApi().dataset_info(
        arguments.dataset_id, revision=arguments.revision
    ).sha
    if not resolved_revision:
        raise RuntimeError("Hugging Face did not return a resolved dataset revision")
    print(
        json.dumps(
            {
                "event": "sidset_start",
                "dataset_id": arguments.dataset_id,
                "requested_revision": arguments.revision,
                "resolved_revision": resolved_revision,
                "datasets_version": datasets_version,
                "train_per_class": arguments.train_per_class,
                "validation_per_class": arguments.validation_per_class,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    try:
        train_rows, train_process = _materialize_split(
            dataset_id=arguments.dataset_id,
            resolved_revision=resolved_revision,
            official_split="train",
            target_split="train",
            per_class=arguments.train_per_class,
            seed=arguments.seed,
            shuffle_buffer=arguments.shuffle_buffer,
            checkpoint_every=arguments.checkpoint_every,
            data_root=data_root,
            manifest_dir=manifest_dir,
            cache_dir=cache_dir,
            excluded_hashes=set(),
        )
        train_hashes = {str(row["sha256"]) for row in train_rows}
        validation_rows, validation_process = _materialize_split(
            dataset_id=arguments.dataset_id,
            resolved_revision=resolved_revision,
            official_split="validation",
            target_split="validation",
            per_class=arguments.validation_per_class,
            seed=arguments.seed + 10_000,
            shuffle_buffer=arguments.shuffle_buffer,
            checkpoint_every=arguments.checkpoint_every,
            data_root=data_root,
            manifest_dir=manifest_dir,
            cache_dir=cache_dir,
            excluded_hashes=train_hashes,
        )
    except InterruptedError:
        return 75

    validation_hashes = {str(row["sha256"]) for row in validation_rows}
    overlap = train_hashes.intersection(validation_hashes)
    if overlap:
        raise RuntimeError(f"Exact SID-Set train/validation leakage: {len(overlap)} hashes")

    train_manifest = manifest_dir / "sidset_train.csv"
    validation_manifest = manifest_dir / "sidset_validation.csv"
    audit = {
        "dataset_id": arguments.dataset_id,
        "license": "CC-BY-4.0",
        "requested_revision": arguments.revision,
        "resolved_revision": resolved_revision,
        "datasets_version": datasets_version,
        "seed": arguments.seed,
        "selection_policy": "deterministic streaming shuffle; labels 0 and 1 only",
        "label_mapping": {"0": "real", "1": "full_synthetic", "2": "excluded_tampered"},
        "split_policy": {
            "train": "official SID-Set train only",
            "validation": "official SID-Set validation only",
        },
        "data_root": str(data_root),
        "train_manifest": str(train_manifest),
        "validation_manifest": str(validation_manifest),
        "train_manifest_sha256": _sha256_file(train_manifest),
        "validation_manifest_sha256": _sha256_file(validation_manifest),
        "train": _split_summary(train_rows),
        "validation": _split_summary(validation_rows),
        "train_process": train_process,
        "validation_process": validation_process,
        "exact_train_validation_overlap": 0,
        "reserved_sets_used": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    audit_path = manifest_dir / "sidset_subset_audit.json"
    _atomic_json(audit, audit_path)
    (data_root / "COMPLETE").write_text(
        f"audit={audit_path}\nresolved_revision={resolved_revision}\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
