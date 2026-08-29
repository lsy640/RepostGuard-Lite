from __future__ import annotations

import io

from PIL import Image

from scripts.download_sidset_subset import _image_bytes, _inspect_image, _safe_stem


def test_sidset_image_payload_and_metadata() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (17, 11), color=(12, 34, 56)).save(buffer, format="PNG")
    payload = _image_bytes({"bytes": buffer.getvalue(), "path": None})
    assert _inspect_image(payload) == (17, 11, "PNG", ".png")


def test_sidset_source_id_is_made_path_safe() -> None:
    assert _safe_stem("../a/b c") == "a_b_c"
    assert _safe_stem("***") == "image"

