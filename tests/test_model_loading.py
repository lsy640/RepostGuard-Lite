from __future__ import annotations

from typing import Any

from repostguard.models import detectors


def test_checkpoint_model_build_skips_openclip_download(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class FakeM2:
        def __init__(self, config: dict[str, Any]) -> None:
            captured.update(config)

    monkeypatch.setattr(detectors, "RepostGuardM2", FakeM2)
    config = {
        "model": {
            "experiment": "m2",
            "clip_pretrained": "laion2b_s34b_b79k",
        }
    }

    result = detectors.build_model(config, load_pretrained=False)

    assert isinstance(result, FakeM2)
    assert captured["clip_pretrained"] is None
    assert config["model"]["clip_pretrained"] == "laion2b_s34b_b79k"


def test_checkpoint_model_build_skips_student_download(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class FakeStudent:
        def __init__(self, *, pretrained: bool, dropout: float) -> None:
            captured.update(pretrained=pretrained, dropout=dropout)

    monkeypatch.setattr(detectors, "MobileNetV3Student", FakeStudent)
    config = {
        "model": {
            "experiment": "student_mnv3",
            "student_backbone": "mobilenet_v3_large",
            "student_pretrained": True,
            "dropout": 0.2,
        }
    }

    result = detectors.build_model(config, load_pretrained=False)

    assert isinstance(result, FakeStudent)
    assert captured == {"pretrained": False, "dropout": 0.2}


def test_checkpoint_model_build_skips_forensic_student_download(
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}

    class FakeForensicStudent:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(detectors, "MobileNetV3ForensicStudent", FakeForensicStudent)
    config = {
        "model": {
            "experiment": "student_mnv3",
            "student_backbone": "mobilenet_v3_large",
            "student_pretrained": True,
            "dropout": 0.2,
            "student_forensic": {
                "enabled": True,
                "pretrained": True,
                "distill_dim": 256,
                "fusion_dim": 512,
            },
        }
    }

    result = detectors.build_model(config, load_pretrained=False)

    assert isinstance(result, FakeForensicStudent)
    assert captured == {
        "pretrained": False,
        "forensic_pretrained": False,
            "dropout": 0.2,
            "distill_dim": 256,
            "fusion_dim": 512,
            "use_npr": False,
            "quality_gate_enabled": False,
            "quality_gate_hidden_dim": 16,
            "fusion_uses_projected_semantic": False,
        }
