from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


_INTERPOLATIONS = {
    "nearest": InterpolationMode.NEAREST,
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
}


def _jpeg(image: Image.Image, quality: int, subsampling: int = 2) -> Image.Image:
    buffer = io.BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=int(quality),
        subsampling=int(subsampling),
        optimize=False,
    )
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


@dataclass(frozen=True)
class FormatDebiasConfig:
    """Class-independent, on-the-fly input codec equalisation."""

    enabled: bool = False
    train_qualities: tuple[int, ...] = (70, 80, 90, 95)
    eval_quality: int = 90
    jpeg_subsampling: int = 2

    def validate(self) -> None:
        if not self.train_qualities:
            raise ValueError("format_debias.train_qualities must not be empty")
        qualities = (*self.train_qualities, self.eval_quality)
        if any(quality < 1 or quality > 100 for quality in qualities):
            raise ValueError(f"JPEG qualities must be in [1, 100], got {qualities}")
        if self.jpeg_subsampling not in (0, 1, 2):
            raise ValueError("format_debias.jpeg_subsampling must be 0, 1, or 2")

    def quality(self, *, training: bool) -> int:
        self.validate()
        if training:
            return int(random.choice(self.train_qualities))
        return int(self.eval_quality)


def harmonize_image_format(
    image: Image.Image,
    image_size: int,
    *,
    quality: int,
    jpeg_subsampling: int = 2,
) -> Image.Image:
    """Resize then JPEG-roundtrip every class through the same pixel pipeline.

    Resizing before the common lossy codec disrupts the original 8x8 block grid
    and makes the new codec artefacts dominate. This mitigates, but cannot prove
    complete removal of, artefacts already present in a source JPEG.
    """

    if int(image_size) <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    resized = TF.resize(
        image.convert("RGB"),
        [int(image_size), int(image_size)],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    return _jpeg(resized, int(quality), int(jpeg_subsampling))


def _resize_roundtrip(image: Image.Image, scale: float, interpolation: str) -> Image.Image:
    width, height = image.size
    target = (max(1, round(height * scale)), max(1, round(width * scale)))
    mode = _INTERPOLATIONS[interpolation]
    reduced = TF.resize(image, target, interpolation=mode, antialias=True)
    return TF.resize(reduced, [height, width], interpolation=mode, antialias=True)


def _gaussian_noise(image: Image.Image, sigma: float, seed: int) -> Image.Image:
    tensor = TF.to_tensor(image)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype)
    return TF.to_pil_image((tensor + float(sigma) * noise).clamp_(0.0, 1.0))


def _center_crop_roundtrip(image: Image.Image, ratio: float) -> Image.Image:
    width, height = image.size
    crop_height = max(1, round(height * ratio))
    crop_width = max(1, round(width * ratio))
    cropped = TF.center_crop(image, [crop_height, crop_width])
    return TF.resize(
        cropped,
        [height, width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )


def _strict_random_six(image: Image.Image, seed: int) -> Image.Image:
    """Apply all six repost perturbations with deterministic random strengths.

    The ranges match the full robust-training ranges.  Strictness comes from
    composing every transform for every sample rather than sampling zero, one,
    or two transforms.  A local RNG makes the result independent of DataLoader
    worker scheduling.
    """

    rng = random.Random(int(seed))
    crop_ratio = rng.uniform(0.75, 0.95)
    resize_scale = rng.uniform(0.25, 0.75)
    resize_interpolation = rng.choice(("bilinear", "bicubic"))
    brightness = rng.uniform(0.8, 1.2)
    contrast = rng.uniform(0.8, 1.2)
    saturation = rng.uniform(0.8, 1.2)
    blur_sigma = rng.uniform(0.1, 2.5)
    noise_sigma = rng.uniform(0.005, 0.10)
    noise_seed = rng.randrange(2**31)
    jpeg_quality = rng.randint(30, 95)

    output = _center_crop_roundtrip(image, crop_ratio)
    output = _resize_roundtrip(output, resize_scale, resize_interpolation)
    output = ImageEnhance.Brightness(output).enhance(brightness)
    output = ImageEnhance.Contrast(output).enhance(contrast)
    output = ImageEnhance.Color(output).enhance(saturation)
    output = output.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    output = _gaussian_noise(output, noise_sigma, noise_seed)
    return _jpeg(output, jpeg_quality, subsampling=2)


def apply_transform(
    image: Image.Image,
    name: str,
    params: dict[str, Any] | None = None,
    *,
    seed_offset: int = 0,
) -> Image.Image:
    params = params or {}
    if name == "clean":
        return image.copy()
    if name == "jpeg":
        return _jpeg(
            image,
            int(params["quality"]),
            int(params.get("subsampling", 2)),
        )
    if name == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(params["sigma"])))
    if name == "resize":
        return _resize_roundtrip(
            image,
            float(params["scale"]),
            str(params.get("interpolation", "bicubic")),
        )
    if name == "gaussian_noise":
        return _gaussian_noise(
            image,
            float(params["sigma"]),
            int(params.get("seed", 0)) + int(seed_offset),
        )
    if name == "color_jitter":
        output = ImageEnhance.Brightness(image).enhance(float(params.get("brightness", 1.0)))
        output = ImageEnhance.Contrast(output).enhance(float(params.get("contrast", 1.0)))
        return ImageEnhance.Color(output).enhance(float(params.get("saturation", 1.0)))
    if name == "center_crop":
        return _center_crop_roundtrip(image, float(params["ratio"]))
    if name == "strict_random_six":
        profile = str(params.get("profile", "full_training_range_v1"))
        if profile != "full_training_range_v1":
            raise ValueError(f"Unknown strict_random_six profile: {profile}")
        return _strict_random_six(
            image,
            int(params.get("seed", 0)) + int(seed_offset),
        )
    if name.startswith("combo_"):
        output = image
        for transform in params["transforms"]:
            output = apply_transform(
                output,
                str(transform["name"]),
                dict(transform.get("params", {})),
                seed_offset=seed_offset,
            )
        return output
    raise ValueError(f"Unknown transform: {name}")


@dataclass(frozen=True)
class AugmentationProbabilities:
    clean: float = 0.25
    single: float = 0.50
    double: float = 0.25

    def validate(self) -> None:
        total = self.clean + self.single + self.double
        if not math.isclose(total, 1.0, abs_tol=1e-8):
            raise ValueError(f"Augmentation probabilities must sum to 1, got {total}")


class SymmetricRobustAugment:
    """Sample the same transform distribution independently of class labels."""

    transform_names = (
        "jpeg",
        "gaussian_blur",
        "resize",
        "gaussian_noise",
        "color_jitter",
        "center_crop",
    )

    def __init__(self, probabilities: AugmentationProbabilities) -> None:
        probabilities.validate()
        self.probabilities = probabilities

    @staticmethod
    def _sample_params(name: str) -> dict[str, Any]:
        if name == "jpeg":
            return {"quality": random.randint(30, 95)}
        if name == "gaussian_blur":
            return {"sigma": random.uniform(0.1, 2.5)}
        if name == "resize":
            return {
                "scale": random.uniform(0.25, 0.75),
                "interpolation": random.choice(("bilinear", "bicubic")),
            }
        if name == "gaussian_noise":
            return {"sigma": random.uniform(0.005, 0.10), "seed": random.randrange(2**31)}
        if name == "color_jitter":
            return {
                "brightness": random.uniform(0.8, 1.2),
                "contrast": random.uniform(0.8, 1.2),
                "saturation": random.uniform(0.8, 1.2),
            }
        if name == "center_crop":
            return {"ratio": random.uniform(0.75, 0.95)}
        raise ValueError(name)

    def __call__(self, image: Image.Image) -> Image.Image:
        draw = random.random()
        if draw < self.probabilities.clean:
            count = 0
        elif draw < self.probabilities.clean + self.probabilities.single:
            count = 1
        else:
            count = 2
        if count == 0:
            return image.copy()
        names = random.sample(self.transform_names, k=count)
        output = image
        for name in names:
            output = apply_transform(output, name, self._sample_params(name))
        return output


def to_model_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    resized = TF.resize(
        image,
        [image_size, image_size],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    return TF.to_tensor(resized)
