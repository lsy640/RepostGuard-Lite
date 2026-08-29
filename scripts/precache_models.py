from __future__ import annotations

import argparse
import gc

import open_clip
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pilot pretrained weights once")
    parser.add_argument("--clip-cache", default="data/cache/open_clip")
    arguments = parser.parse_args()
    cnn = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    del cnn
    gc.collect()
    clip_model, _, _ = open_clip.create_model_and_transforms(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        cache_dir=arguments.clip_cache,
    )
    del clip_model
    gc.collect()
    print("EfficientNet-B0 and OpenCLIP ViT-B-32 weights are cached")


if __name__ == "__main__":
    main()

