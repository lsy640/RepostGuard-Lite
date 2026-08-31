from __future__ import annotations

import copy

import pytest
import torch

from repostguard.models.forensic import DCTPatchSelector, ForensicBranch


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


@pytest.mark.skipif(
    not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available(),
    reason="MPS is not available",
)
def test_non_divisible_dct_pool_matches_cpu_on_mps() -> None:
    torch.manual_seed(20260901)
    images = torch.rand(1, 3, 224, 224)
    cpu_selector = DCTPatchSelector(patch_size=56, patches_per_band=2, dct_size=16).eval()
    mps_selector = copy.deepcopy(cpu_selector).to("mps")

    cpu_selected, cpu_ratios, cpu_types = cpu_selector(images)
    mps_selected, mps_ratios, mps_types = mps_selector(images.to("mps"))

    assert torch.allclose(mps_ratios.cpu(), cpu_ratios, atol=5e-4, rtol=5e-4)
    assert torch.equal(mps_types.cpu(), cpu_types)
    assert torch.allclose(mps_selected.cpu(), cpu_selected, atol=5e-4, rtol=5e-4)
