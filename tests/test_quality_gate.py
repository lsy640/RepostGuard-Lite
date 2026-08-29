from __future__ import annotations

import torch

from repostguard.models.quality_gate import QUALITY_FEATURE_NAMES, QualityAwareGate


def test_quality_gate_outputs_finite_balanced_initial_fractions() -> None:
    gate = QualityAwareGate(hidden_dim=16)
    images = torch.rand(3, 3, 224, 224, dtype=torch.float16)

    fractions, quality = gate(images)

    assert fractions.shape == (3, 2)
    assert quality.shape == (3, len(QUALITY_FEATURE_NAMES))
    assert quality.dtype == torch.float32
    assert torch.isfinite(quality).all()
    torch.testing.assert_close(fractions.sum(dim=1), torch.ones(3))
    torch.testing.assert_close(fractions, torch.full((3, 2), 0.5))


def test_quality_gate_detects_loss_of_high_frequency_content() -> None:
    gate = QualityAwareGate(hidden_dim=8)
    checkerboard = torch.arange(224).view(1, 1, 1, 224)
    checkerboard = ((checkerboard + checkerboard.transpose(2, 3)) % 2).float()
    sharp = checkerboard.expand(1, 3, 224, 224)
    flat = torch.full_like(sharp, 0.5)

    _, sharp_quality = gate(sharp)
    _, flat_quality = gate(flat)

    gradient_index = QUALITY_FEATURE_NAMES.index("gradient_energy")
    laplacian_index = QUALITY_FEATURE_NAMES.index("laplacian_energy")
    assert sharp_quality[0, gradient_index] > flat_quality[0, gradient_index]
    assert sharp_quality[0, laplacian_index] > flat_quality[0, laplacian_index]
