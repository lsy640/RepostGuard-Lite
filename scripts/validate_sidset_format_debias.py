from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from repostguard.checkpoint import load_checkpoint
from repostguard.config import config_digest, load_config
from repostguard.data.dataset import ManifestImageDataset, build_format_debias_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    config = load_config("configs/sidset/b0.yaml")
    data_config = config["data"]
    dataset = ManifestImageDataset(
        data_config["val_manifest"],
        data_config["root"],
        int(data_config["image_size"]),
        eval_transform={"name": "clean", "params": {}},
        format_debias=build_format_debias_config(data_config),
        training=False,
    )
    selected_indices = []
    for label in (0, 1):
        selected_indices.append(next(index for index, row in enumerate(dataset.rows) if row["label"] == label))

    records = []
    for index in selected_indices:
        row = dataset.rows[index]
        source_path = Path(row["absolute_path"])
        before = _sha256(source_path)
        first = dataset[index]
        second = dataset[index]
        after = _sha256(source_path)
        if before != after or before != row["sha256"]:
            raise AssertionError(f"Source image was modified: {source_path}")
        if not torch.equal(first["image"], second["image"]):
            raise AssertionError("Evaluation format normalisation is not deterministic")
        if tuple(first["image"].shape) != (3, 224, 224):
            raise AssertionError(f"Unexpected tensor shape: {tuple(first['image'].shape)}")
        quality = int(first["format_debias_quality"].item())
        if quality != int(data_config["format_debias"]["eval_quality"]):
            raise AssertionError(f"Unexpected evaluation JPEG quality: {quality}")
        records.append(
            {
                "label": int(row["label"]),
                "source_format": row["format"],
                "source_sha256_unchanged": True,
                "output_shape": list(first["image"].shape),
                "eval_jpeg_quality": quality,
            }
        )

    cifake_checkpoint_compatibility = {}
    for experiment in ("b0", "b1", "b2", "m2"):
        cifake_config = load_config(f"configs/{experiment}.yaml")
        checkpoint_path = Path(f"outputs/{experiment}/best.pt")
        checkpoint = load_checkpoint(checkpoint_path)
        compatible = checkpoint.get("config_sha256") == config_digest(cifake_config)
        if not compatible:
            raise AssertionError(
                f"Existing CIFAKE checkpoint config digest changed: {checkpoint_path}"
            )
        cifake_checkpoint_compatibility[experiment] = True
        del checkpoint

    print(
        json.dumps(
            {
                "event": "sidset_format_debias_validation",
                "enabled": True,
                "strategy": "RGB decode, bicubic 224x224 resize, class-independent JPEG roundtrip",
                "records": records,
                "existing_cifake_checkpoints_compatible": cifake_checkpoint_compatibility,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
