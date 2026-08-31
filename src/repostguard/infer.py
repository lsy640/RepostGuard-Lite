from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from repostguard.checkpoint import atomic_text, load_checkpoint
from repostguard.config import config_digest, load_config
from repostguard.data.dataset import build_format_debias_config
from repostguard.data.transforms import harmonize_image_format, to_model_tensor
from repostguard.models import build_model, count_parameters


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_prediction_records(records: list[dict[str, Any]]) -> None:
    previous_path = ""
    for record in records:
        if set(record) != {"image_path", "pred"}:
            raise ValueError(f"Invalid prediction keys: {sorted(record)}")
        if not isinstance(record["image_path"], str):
            raise TypeError("image_path must be a string")
        if record["image_path"] < previous_path:
            raise ValueError("Prediction records are not sorted")
        probability = record["pred"]
        if not isinstance(probability, float) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"Invalid AIGC probability: {probability!r}")
        previous_path = record["image_path"]


def infer(
    config_path: str,
    checkpoint_path: str,
    input_directory: str,
    output_path: str,
    diagnostics_path: str,
    batch_size: int,
    device_name: str,
) -> None:
    config = load_config(config_path)
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.get("config_sha256") != config_digest(config):
        raise ValueError("Checkpoint and inference config digests differ")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    model = build_model(config, load_pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    root = Path(input_directory).resolve()
    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError(f"No supported images under {root}")

    predictions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    valid_paths: list[Path] = []
    tensors: list[torch.Tensor] = []
    format_debias = build_format_debias_config(config["data"])

    def flush() -> None:
        if not tensors:
            return
        images = torch.stack(tensors).to(device)
        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda" and bool(config["train"]["amp"]),
            ):
                logits = model(images)["logits"]
        probabilities = torch.sigmoid(logits).float().cpu().tolist()
        for image_path, probability in zip(valid_paths, probabilities, strict=True):
            relative = image_path.relative_to(root).as_posix()
            predictions.append({"image_path": relative, "pred": float(probability)})
        tensors.clear()
        valid_paths.clear()

    for path in paths:
        try:
            with Image.open(path) as image_file:
                if getattr(image_file, "is_animated", False):
                    image_file.seek(0)
                image = image_file.convert("RGB").copy()
            if format_debias.enabled:
                image = harmonize_image_format(
                    image,
                    int(config["data"]["image_size"]),
                    quality=format_debias.quality(training=False),
                    jpeg_subsampling=format_debias.jpeg_subsampling,
                )
            tensors.append(to_model_tensor(image, int(config["data"]["image_size"])))
            valid_paths.append(path)
            if len(tensors) >= batch_size:
                flush()
        except Exception as error:  # a corrupt image must not abort the directory batch
            diagnostics.append(
                {
                    "image_path": path.relative_to(root).as_posix(),
                    "status": "error",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
    flush()
    predictions.sort(key=lambda record: record["image_path"])
    validate_prediction_records(predictions)
    atomic_text(json.dumps(predictions, indent=2, sort_keys=True) + "\n", output_path)
    metadata = {
        "checkpoint_sha256": _sha256(checkpoint_path),
        "config_sha256": config_digest(config),
        "device": str(device),
        "parameters": count_parameters(model),
        "processed": len(predictions),
        "errors": diagnostics,
        "format_debias": {
            "enabled": format_debias.enabled,
            "eval_quality": format_debias.eval_quality if format_debias.enabled else None,
            "jpeg_subsampling": (
                format_debias.jpeg_subsampling if format_debias.enabled else None
            ),
        },
    }
    atomic_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", diagnostics_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict AIGC probabilities for an image directory")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--diagnostics", default="diagnostics.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    infer(
        arguments.config,
        arguments.checkpoint,
        arguments.input_dir,
        arguments.output,
        arguments.diagnostics,
        arguments.batch_size,
        arguments.device,
    )


if __name__ == "__main__":
    main()
