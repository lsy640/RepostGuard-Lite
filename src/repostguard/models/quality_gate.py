from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


QUALITY_FEATURE_NAMES = (
    "gradient_energy",
    "laplacian_energy",
    "jpeg_blockiness",
    "high_frequency_noise",
    "effective_resolution_proxy",
    "luminance_dynamic_range",
)


class QualityFeatureExtractor(nn.Module):
    """Extract label-agnostic image-quality statistics for branch gating only."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "laplacian_kernel",
            torch.tensor(
                [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3),
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Quality statistics must remain stable when the surrounding model uses
        # float16 autocast.
        images_fp32 = images.float().clamp(0.0, 1.0)
        gray = (
            0.2989 * images_fp32[:, 0:1]
            + 0.5870 * images_fp32[:, 1:2]
            + 0.1140 * images_fp32[:, 2:3]
        )
        horizontal_difference = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs()
        vertical_difference = (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs()
        gradient_energy = 0.5 * (
            horizontal_difference.mean(dim=(1, 2, 3))
            + vertical_difference.mean(dim=(1, 2, 3))
        )

        laplacian = F.conv2d(
            F.pad(gray, (1, 1, 1, 1), mode="reflect"), self.laplacian_kernel
        )
        laplacian_energy = laplacian.abs().mean(dim=(1, 2, 3))
        vertical_boundaries = horizontal_difference[:, :, :, 7::8].mean(dim=(1, 2, 3))
        horizontal_boundaries = vertical_difference[:, :, 7::8, :].mean(dim=(1, 2, 3))
        boundary_energy = 0.5 * (vertical_boundaries + horizontal_boundaries)
        jpeg_blockiness = boundary_energy / gradient_energy.clamp_min(1e-6)

        local_mean = F.avg_pool2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        high_frequency_noise = (gray - local_mean).abs().mean(dim=(1, 2, 3))
        effective_resolution_proxy = laplacian_energy / gradient_energy.clamp_min(1e-6)
        dynamic_range = gray.flatten(1).std(dim=1, unbiased=False)

        return torch.stack(
            (
                torch.log1p(100.0 * gradient_energy),
                torch.log1p(100.0 * laplacian_energy),
                torch.log1p(jpeg_blockiness),
                torch.log1p(100.0 * high_frequency_noise),
                torch.log1p(effective_resolution_proxy),
                torch.log1p(10.0 * dynamic_range),
            ),
            dim=1,
        )


class QualityAwareGate(nn.Module):
    """Map quality statistics to semantic/forensic branch fractions."""

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.extractor = QualityFeatureExtractor()
        self.network = nn.Sequential(
            nn.LayerNorm(len(QUALITY_FEATURE_NAMES)),
            nn.Linear(len(QUALITY_FEATURE_NAMES), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 2),
        )
        # M3 starts as the M2 fusion (both branch scales equal one). Softmax
        # fractions are multiplied by two by the detector before fusion.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        quality_features = self.extractor(images)
        gate_fractions = torch.softmax(self.network(quality_features), dim=1)
        return gate_fractions, quality_features
