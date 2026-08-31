from __future__ import annotations

import io
import json
import tempfile
import time
from pathlib import Path

import pytest
from PIL import Image

from server.runtime import (
    BatchManager,
    CalibrationRegistry,
    DemoInputError,
    apply_robustness,
    decode_image_bytes,
    prepare_clean_image,
    validate_relative_path,
)
from server.schemas import RobustnessRequest


def _png_bytes(color: tuple[int, int, int] = (80, 140, 210)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (96, 72), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _mpo_bytes() -> bytes:
    buffer = io.BytesIO()
    first = Image.new("RGB", (96, 72), (20, 40, 60))
    second = Image.new("RGB", (96, 72), (200, 180, 160))
    first.save(buffer, format="MPO", save_all=True, append_images=[second])
    return buffer.getvalue()


def _portrait_exif_jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    image = Image.new("RGB", (96, 72), (40, 80, 120))
    exif = Image.Exif()
    exif[274] = 6  # Rotate the stored landscape pixels 90° clockwise for display.
    image.save(buffer, format="JPEG", quality=95, exif=exif)
    return buffer.getvalue()


def test_calibration_matches_both_train_v3_checkpoints() -> None:
    calibration = CalibrationRegistry()
    m2 = calibration.describe("m2")
    m3 = calibration.describe("m3")
    assert m2["available"] is True
    assert m3["available"] is True
    assert m2["temperature"] == pytest.approx(14.50161361694336)
    assert m3["temperature"] == pytest.approx(14.526344299316406)
    assert m2["calibrated_threshold"] == pytest.approx(0.5966089452137621)
    assert m3["calibrated_threshold"] == pytest.approx(0.5990082855486655)
    for model, raw_score in (("m2", 0.999), ("m3", 0.12)):
        calibrated, threshold, entropy = calibration.transform(model, raw_score)
        assert calibrated is not None and 0 <= calibrated <= 1
        assert threshold is not None and 0 <= threshold <= 1
        assert entropy is not None and 0 <= entropy <= 1


@pytest.mark.parametrize("value", ["../x.png", "/tmp/x.png", "", ".", "a/../x.png"])
def test_relative_path_rejects_unsafe_values(value: str) -> None:
    with pytest.raises(DemoInputError):
        validate_relative_path(value)


def test_relative_path_normalizes_windows_separators() -> None:
    assert validate_relative_path("subdir\\image.png") == "subdir/image.png"


def test_image_decode_and_clean_protocol() -> None:
    image, metadata = decode_image_bytes(_png_bytes())
    clean = prepare_clean_image(image)
    assert metadata == {
        "width": 96,
        "height": 72,
        "format": "PNG",
        "bytes": len(_png_bytes()),
        "animated_first_frame": False,
    }
    assert clean.size == (224, 224)
    with pytest.raises(DemoInputError):
        decode_image_bytes(b"not-an-image")


def test_mpo_camera_photo_is_accepted_as_first_frame_jpeg() -> None:
    image, metadata = decode_image_bytes(_mpo_bytes())
    assert image.size == (96, 72)
    assert image.getpixel((0, 0)) == pytest.approx((20, 40, 60), abs=2)
    assert metadata["format"] == "JPEG"
    assert metadata["animated_first_frame"] is True


def test_exif_orientation_is_applied_before_model_preprocessing() -> None:
    image, metadata = decode_image_bytes(_portrait_exif_jpeg_bytes())
    assert image.size == (72, 96)
    assert metadata["width"] == 72
    assert metadata["height"] == 96
    assert prepare_clean_image(image).size == (224, 224)


def test_robustness_order_and_seed_are_deterministic() -> None:
    clean = prepare_clean_image(Image.new("RGB", (128, 96), (90, 130, 170)))
    request = RobustnessRequest.model_validate(
        {
            "model": "m3",
            "crop": {"enabled": True, "ratio": 0.8},
            "resize": {"enabled": True, "scale": 0.5},
            "jitter": {"enabled": True, "brightness": 1.1, "contrast": 0.9, "saturation": 1.2},
            "blur": {"enabled": True, "sigma": 1.0},
            "noise": {"enabled": True, "sigma": 0.02},
            "jpeg": {"enabled": True, "quality": 70},
        }
    )
    first, applied = apply_robustness(clean, request)
    second, _ = apply_robustness(clean, request)
    assert [item["name"] for item in applied] == [
        "center_crop",
        "resize",
        "color_jitter",
        "gaussian_blur",
        "gaussian_noise",
        "jpeg",
    ]
    assert first.tobytes() == second.tobytes()


class _FakeRegistry:
    def predict(self, image: Image.Image, model: str) -> dict:
        del image, model
        return {"raw_score": 0.93424}


def test_batch_results_keep_the_exact_two_field_schema() -> None:
    manager = BatchManager(_FakeRegistry())  # type: ignore[arg-type]
    temp_dir = Path(tempfile.mkdtemp(prefix="aigi-batch-test-"))
    good = temp_dir / "good.upload"
    bad = temp_dir / "bad.upload"
    good.write_bytes(_png_bytes())
    bad.write_bytes(b"corrupt")
    job = manager.create("m2", ["subdir/image.png", "bad.jpg"], [good, bad], temp_dir)
    deadline = time.time() + 5
    while job.status not in {"complete", "failed"} and time.time() < deadline:
        time.sleep(0.02)
    assert job.status == "complete"
    assert job.results == [{"image_path": "subdir/image.png", "pred": 0.9342}]
    assert set(job.results[0]) == {"image_path", "pred"}
    assert job.errors[0]["image_path"] == "bad.jpg"
    manager.close()
