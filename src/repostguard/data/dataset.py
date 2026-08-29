from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from repostguard.data.transforms import (
    AugmentationProbabilities,
    FormatDebiasConfig,
    SymmetricRobustAugment,
    apply_transform,
    harmonize_image_format,
    to_model_tensor,
)

ImageFile.LOAD_TRUNCATED_IMAGES = False

REQUIRED_COLUMNS = {
    "sample_id",
    "path",
    "label",
    "split",
    "source_dataset",
    "generator_id",
}


def read_manifest(path: str | Path, root: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve()
    data_root = Path(root).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest {manifest_path} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    sample_ids: set[str] = set()
    relative_paths: set[str] = set()
    for row in rows:
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError(f"Invalid label {label!r} for {row['sample_id']}")
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe manifest path: {relative}")
        resolved = (data_root / relative).resolve()
        if not resolved.is_relative_to(data_root):
            raise ValueError(f"Path escapes data root: {relative}")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        if row["sample_id"] in sample_ids:
            raise ValueError(f"Duplicate sample_id: {row['sample_id']}")
        if row["path"] in relative_paths:
            raise ValueError(f"Duplicate path: {row['path']}")
        sample_ids.add(row["sample_id"])
        relative_paths.add(row["path"])
        row["label"] = label
        row["absolute_path"] = str(resolved)
    return rows


class ManifestImageDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest: str | Path,
        root: str | Path,
        image_size: int,
        *,
        robust_augmentation: bool = False,
        return_pair: bool = False,
        eval_transform: dict[str, Any] | None = None,
        augmentation_probabilities: AugmentationProbabilities | None = None,
        format_debias: FormatDebiasConfig | None = None,
        training: bool = False,
    ) -> None:
        self.rows = read_manifest(manifest, root)
        self.image_size = int(image_size)
        self.return_pair = bool(return_pair)
        self.eval_transform = eval_transform
        self.training = bool(training)
        self.format_debias = format_debias or FormatDebiasConfig(enabled=False)
        self.format_debias.validate()
        self.augment = None
        if robust_augmentation:
            self.augment = SymmetricRobustAugment(
                augmentation_probabilities or AugmentationProbabilities()
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with Image.open(row["absolute_path"]) as image_file:
            if getattr(image_file, "is_animated", False):
                image_file.seek(0)
            clean = image_file.convert("RGB").copy()

        format_quality: int | None = None
        if self.format_debias.enabled:
            format_quality = self.format_debias.quality(training=self.training)
            clean = harmonize_image_format(
                clean,
                self.image_size,
                quality=format_quality,
                jpeg_subsampling=self.format_debias.jpeg_subsampling,
            )

        if self.eval_transform is not None:
            augmented = apply_transform(
                clean,
                str(self.eval_transform["name"]),
                dict(self.eval_transform.get("params", {})),
                seed_offset=index,
            )
        elif self.augment is not None:
            augmented = self.augment(clean)
        else:
            augmented = clean

        item: dict[str, Any] = {
            "image": to_model_tensor(clean if self.return_pair else augmented, self.image_size),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "sample_id": row["sample_id"],
            "path": row["path"],
            "source_dataset": row["source_dataset"],
            "generator_id": row["generator_id"],
        }
        if format_quality is not None:
            item["format_debias_quality"] = torch.tensor(format_quality, dtype=torch.int16)
        if self.return_pair:
            item["image_aug"] = to_model_tensor(augmented, self.image_size)
        return item


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _balanced_weights(rows: list[dict[str, Any]]) -> torch.Tensor:
    groups = [
        (
            int(row["label"]),
            row["source_dataset"],
            row["generator_id"] if int(row["label"]) == 1 else "real",
        )
        for row in rows
    ]
    counts = Counter(groups)
    return torch.as_tensor([1.0 / counts[group] for group in groups], dtype=torch.double)


def build_format_debias_config(data_config: dict[str, Any]) -> FormatDebiasConfig:
    raw = dict(data_config.get("format_debias", {}))
    return FormatDebiasConfig(
        enabled=bool(raw.get("enabled", False)),
        train_qualities=tuple(int(value) for value in raw.get("train_qualities", (70, 80, 90, 95))),
        eval_quality=int(raw.get("eval_quality", 90)),
        jpeg_subsampling=int(raw.get("jpeg_subsampling", 2)),
    )


def build_train_loader(config: dict[str, Any]) -> DataLoader[dict[str, Any]]:
    data_config = config["data"]
    experiment = config["model"]["experiment"].lower()
    robust = experiment in {"b1", "m2", "m3"}
    paired = experiment in {"m2", "m3"}
    probabilities = AugmentationProbabilities(
        clean=float(data_config["train_clean_probability"]),
        single=float(data_config["train_single_probability"]),
        double=float(data_config["train_double_probability"]),
    )
    dataset = ManifestImageDataset(
        data_config["train_manifest"],
        data_config["root"],
        int(data_config["image_size"]),
        robust_augmentation=robust,
        return_pair=paired,
        augmentation_probabilities=probabilities,
        format_debias=build_format_debias_config(data_config),
        training=True,
    )
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    sampler = WeightedRandomSampler(
        _balanced_weights(dataset.rows),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    workers = int(data_config["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        sampler=sampler,
        num_workers=workers,
        pin_memory=bool(data_config.get("pin_memory", True)),
        persistent_workers=bool(data_config.get("persistent_workers", True)) and workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
        drop_last=True,
    )


def build_eval_loader(
    config: dict[str, Any], transform_spec: dict[str, Any]
) -> DataLoader[dict[str, Any]]:
    data_config = config["data"]
    dataset = ManifestImageDataset(
        data_config["val_manifest"],
        data_config["root"],
        int(data_config["image_size"]),
        eval_transform=transform_spec,
        format_debias=build_format_debias_config(data_config),
        training=False,
    )
    workers = int(data_config["num_workers"])
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    return DataLoader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=bool(data_config.get("pin_memory", True)),
        persistent_workers=bool(data_config.get("persistent_workers", True)) and workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
