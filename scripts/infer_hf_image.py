from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

from repostguard.config import load_config
from repostguard.data.dataset import build_format_debias_config
from repostguard.data.transforms import harmonize_image_format, to_model_tensor
from repostguard.models import build_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-image inference with a published M2/M3 safetensors model"
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_dir = Path(args.model_dir)
    config = load_config(model_dir / "resolved_config.yaml")
    threshold = json.loads(
        (model_dir / "thresholds.json").read_text(encoding="utf-8")
    )["threshold"]

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    model = build_model(config, load_pretrained=False)
    state = load_file(str(model_dir / "model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device)
    model = model.to(device).eval()

    with Image.open(args.image) as image_file:
        image = image_file.convert("RGB").copy()

    format_debias = build_format_debias_config(config["data"])
    if format_debias.enabled:
        image = harmonize_image_format(
            image,
            int(config["data"]["image_size"]),
            quality=format_debias.quality(training=False),
            jpeg_subsampling=format_debias.jpeg_subsampling,
        )

    tensor = to_model_tensor(image, int(config["data"]["image_size"]))
    with torch.inference_mode():
        logit = model(tensor.unsqueeze(0).to(device))["logits"]
        score = float(torch.sigmoid(logit).item())

    print(
        json.dumps(
            {
                "image": args.image,
                "aigc_score": score,
                "threshold": threshold,
                "prediction": "AIGC" if score >= threshold else "Real",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
