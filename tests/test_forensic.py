from __future__ import annotations

import torch

from repostguard.models.forensic import ForensicBranch


def test_forensic_branch_shapes_and_filter_invariants() -> None:
    branch = ForensicBranch(patch_size=56, patches_per_band=2, dct_size=16, output_dim=64)
    images = torch.rand(2, 3, 224, 224)
    features, diagnostics = branch(images)
    assert features.shape == (2, 64)
    assert diagnostics["patch_attention"].shape == (2, 4)
    assert diagnostics["frequency_ratios"].shape == (2, 16)
    assert branch.srm_filters.shape == (30, 1, 5, 5)
    assert torch.allclose(branch.srm_filters.sum(dim=(1, 2, 3)), torch.zeros(30), atol=1e-6)
    assert torch.allclose(diagnostics["patch_attention"].sum(dim=1), torch.ones(2), atol=1e-6)

