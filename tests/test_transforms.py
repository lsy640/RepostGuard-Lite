from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from repostguard.data.transforms import (
    FormatDebiasConfig,
    apply_transform,
    harmonize_image_format,
    to_model_tensor,
)


def _image() -> Image.Image:
    values = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    return Image.fromarray(values, mode="RGB")


def test_all_required_transforms_preserve_mode_and_size() -> None:
    specifications = [
        ("jpeg", {"quality": 70}),
        ("gaussian_blur", {"sigma": 1.0}),
        ("resize", {"scale": 0.5, "interpolation": "bicubic"}),
        ("gaussian_noise", {"sigma": 0.05, "seed": 17}),
        ("color_jitter", {"brightness": 1.2, "contrast": 0.8, "saturation": 1.1}),
        ("center_crop", {"ratio": 0.8}),
    ]
    source = _image()
    for name, parameters in specifications:
        transformed = apply_transform(source, name, parameters)
        assert transformed.mode == "RGB"
        assert transformed.size == source.size
        tensor = to_model_tensor(transformed, 224)
        assert tensor.shape == (3, 224, 224)
        assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0


def test_gaussian_noise_is_deterministic_per_seed_and_sample() -> None:
    source = _image()
    first = np.asarray(
        apply_transform(source, "gaussian_noise", {"sigma": 0.05, "seed": 7}, seed_offset=2)
    )
    second = np.asarray(
        apply_transform(source, "gaussian_noise", {"sigma": 0.05, "seed": 7}, seed_offset=2)
    )
    different = np.asarray(
        apply_transform(source, "gaussian_noise", {"sigma": 0.05, "seed": 7}, seed_offset=3)
    )
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)


def test_strict_random_six_is_deterministic_per_seed_and_sample() -> None:
    source = _image()
    parameters = {"seed": 20260828, "profile": "full_training_range_v1"}
    first = np.asarray(
        apply_transform(source, "strict_random_six", parameters, seed_offset=11)
    )
    second = np.asarray(
        apply_transform(source, "strict_random_six", parameters, seed_offset=11)
    )
    different = np.asarray(
        apply_transform(source, "strict_random_six", parameters, seed_offset=12)
    )
    assert first.shape == np.asarray(source).shape
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)


def test_strict_random_six_rejects_unknown_profile() -> None:
    with np.testing.assert_raises_regex(ValueError, "Unknown strict_random_six profile"):
        apply_transform(
            _image(),
            "strict_random_six",
            {"seed": 1, "profile": "unknown"},
        )


def test_community_forensics_robustness_v2_preserves_legacy_matrix() -> None:
    legacy_path = Path("configs/transforms.yaml")
    extended_path = Path("configs/community_forensics_robustness_v2.yaml")
    legacy = yaml.safe_load(legacy_path.read_text(encoding="utf-8"))["evaluation"]
    extended = yaml.safe_load(extended_path.read_text(encoding="utf-8"))["evaluation"]

    assert len(legacy) == 18
    assert len(extended) == 21
    assert extended[: len(legacy)] == legacy
    assert [item["name"] for item in extended[-3:]] == [
        "combo_four_stage_platform_repost",
        "combo_four_stage_edit_repost",
        "strict_random_six",
    ]
    assert [len(item["params"].get("transforms", [])) for item in extended[-3:]] == [
        4,
        4,
        0,
    ]


def test_new_four_stage_compositions_are_deterministic_and_preserve_size() -> None:
    matrix = yaml.safe_load(
        Path("configs/community_forensics_robustness_v2.yaml").read_text(encoding="utf-8")
    )["evaluation"]
    source = _image()
    for specification in matrix[-3:-1]:
        first = apply_transform(
            source,
            specification["name"],
            specification["params"],
            seed_offset=13,
        )
        second = apply_transform(
            source,
            specification["name"],
            specification["params"],
            seed_offset=13,
        )
        assert first.mode == "RGB"
        assert first.size == source.size
        assert np.array_equal(np.asarray(first), np.asarray(second))


def test_format_debias_is_fixed_for_evaluation_and_resizes_before_codec() -> None:
    config = FormatDebiasConfig(enabled=True, eval_quality=90)
    assert config.quality(training=False) == 90
    first = harmonize_image_format(_image(), 224, quality=config.quality(training=False))
    second = harmonize_image_format(_image(), 224, quality=config.quality(training=False))
    assert first.mode == "RGB"
    assert first.size == (224, 224)
    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_format_debias_training_quality_is_label_independent() -> None:
    config = FormatDebiasConfig(enabled=True, train_qualities=(70, 80, 90, 95))
    random_qualities = {config.quality(training=True) for _ in range(100)}
    assert random_qualities.issubset({70, 80, 90, 95})
    assert len(random_qualities) > 1
