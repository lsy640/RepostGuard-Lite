from __future__ import annotations

import io
import time

from fastapi.testclient import TestClient
from PIL import Image

from server.app import app
from server.runtime import model_registry


def _image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (80, 60), (120, 80, 180)).save(buffer, format="PNG")
    return buffer.getvalue()


def _fake_result(model: str) -> dict:
    pixel = "data:image/png;base64,iVBORw0KGgo="
    return {
        "model": model,
        "checkpoint_sha256": "a" * 64,
        "raw_score": 0.12764,
        "calibrated_score": 0.48,
        "raw_threshold": 0.99,
        "calibrated_threshold": 0.59,
        "label": "Real",
        "uncertainty_entropy": 0.9,
        "branch_evidence": {
            "kind": "ablation",
            "title": "分支消融贡献",
            "semantic": 0.4,
            "forensic": 0.6,
            "low_signal": False,
            "note": "test",
        },
        "heatmaps": {
            "srm_color": pixel,
            "srm_gray": pixel,
            "npr_color": pixel,
            "npr_gray": pixel,
        },
        "preview": pixel,
        "timing_ms": 2.0,
        "device": "cpu",
        "parameters": {"total": 1, "trainable": 1, "frozen": 0},
    }


def test_frontend_html_is_never_served_from_a_stale_localhost_cache() -> None:
    response = TestClient(app).get("/")
    assert response.headers["cache-control"] == "no-store, max-age=0"
    assert response.headers["pragma"] == "no-cache"
    assert response.headers["expires"] == "0"


def test_single_infer_and_batch_api_without_model_stub_leak(monkeypatch) -> None:
    monkeypatch.setattr(model_registry, "predict", lambda image, model: _fake_result(model))
    client = TestClient(app)
    response = client.post(
        "/api/infer",
        data={"model": "m2"},
        files={"image": ("example.png", _image_bytes(), "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "m2"
    assert payload["file"]["name"] == "example.png"
    assert payload["image_id"]
    assert payload["source_preview"].startswith("data:image/jpeg;base64,")

    response = client.post(
        "/api/batches",
        data={"paths": ["subdir/image.png"], "model": "m3"},
        files=[("files", ("image.png", _image_bytes(), "image/png"))],
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    deadline = time.time() + 5
    while time.time() < deadline:
        state = client.get(f"/api/batches/{job_id}").json()
        if state["status"] == "complete":
            break
        time.sleep(0.02)
    download = client.get(f"/api/batches/{job_id}/download")
    assert download.status_code == 200
    assert download.json() == [{"image_path": "subdir/image.png", "pred": 0.1276}]
    assert set(download.json()[0]) == {"image_path", "pred"}
