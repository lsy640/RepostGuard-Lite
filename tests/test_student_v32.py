from __future__ import annotations

import torch

from repostguard.models.student import (
    LightweightForensicStudentBranch,
    MobileNetV3ForensicStudent,
)


def test_npr_adds_three_residual_channels_without_a_new_backbone() -> None:
    branch = LightweightForensicStudentBranch(
        pretrained=False,
        output_dim=32,
        use_npr=True,
    )
    first_conv = branch.channel_adapter[0]
    assert isinstance(first_conv, torch.nn.Conv2d)
    assert first_conv.in_channels == 11

    images = torch.rand(2, 3, 64, 64)
    output = branch(images)
    assert output.shape == (2, 32)
    assert torch.isfinite(output).all()


def test_quality_gated_student_returns_mobile_distillation_targets() -> None:
    model = MobileNetV3ForensicStudent(
        pretrained=False,
        forensic_pretrained=False,
        dropout=0.0,
        distill_dim=32,
        fusion_dim=64,
        use_npr=True,
        quality_gate_enabled=True,
        quality_gate_hidden_dim=8,
        fusion_uses_projected_semantic=True,
    ).eval()
    images = torch.rand(2, 3, 64, 64)

    with torch.inference_mode():
        output = model(images)

    assert output["logits"].shape == (2,)
    assert output["semantic_features"].shape == (2, 32)
    assert output["forensic_features"].shape == (2, 32)
    assert output["features"].shape == (2, 32)
    assert output["gate_fractions"].shape == (2, 2)
    assert output["quality_features"].shape == (2, 6)
    assert model.fusion[0].normalized_shape == (64,)
    torch.testing.assert_close(
        output["gate_fractions"].sum(dim=1),
        torch.ones(2),
    )


def test_default_forensic_student_keeps_legacy_output_contract() -> None:
    model = MobileNetV3ForensicStudent(
        pretrained=False,
        forensic_pretrained=False,
        dropout=0.0,
        distill_dim=32,
        fusion_dim=64,
    ).eval()
    images = torch.rand(1, 3, 64, 64)

    with torch.inference_mode():
        output = model(images)

    assert "gate_fractions" not in output
    assert "quality_features" not in output
    first_conv = model.forensic.channel_adapter[0]
    assert isinstance(first_conv, torch.nn.Conv2d)
    assert first_conv.in_channels == 8
    assert model.fusion[0].normalized_shape == (992,)
