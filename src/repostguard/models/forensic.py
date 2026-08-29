from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


def _dct_basis(size: int) -> torch.Tensor:
    positions = torch.arange(size, dtype=torch.float32)
    frequencies = positions.view(-1, 1)
    basis = torch.cos(math.pi * (2.0 * positions + 1.0) * frequencies / (2.0 * size))
    basis[0] *= math.sqrt(1.0 / size)
    basis[1:] *= math.sqrt(2.0 / size)
    return basis


def _srm_inspired_bank() -> torch.Tensor:
    """Create 30 deterministic, zero-mean 5x5 high-pass kernels."""

    vectors = [
        torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]),
        torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0,
        torch.tensor([0.0, -1.0, 0.0, 1.0, 0.0]),
        torch.tensor([0.0, 1.0, -2.0, 1.0, 0.0]),
        torch.tensor([-1.0, 2.0, 0.0, -2.0, 1.0]),
        torch.tensor([1.0, -4.0, 6.0, -4.0, 1.0]),
    ]
    kernels: list[torch.Tensor] = []
    for row_index, row in enumerate(vectors):
        for column_index, column in enumerate(vectors):
            if row_index < 2 and column_index < 2:
                continue
            kernel = torch.outer(row, column)
            kernel = kernel - kernel.mean()
            kernel = kernel / kernel.abs().sum().clamp_min(1e-8)
            kernels.append(kernel)
    if len(kernels) < 30:
        raise RuntimeError("Failed to construct the requested high-pass filter bank")
    return torch.stack(kernels[:30]).unsqueeze(1)


class DCTPatchSelector(nn.Module):
    def __init__(self, patch_size: int, patches_per_band: int, dct_size: int) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.patches_per_band = int(patches_per_band)
        self.dct_size = int(dct_size)
        self.register_buffer("dct_basis", _dct_basis(self.dct_size), persistent=False)
        coordinates = torch.arange(self.dct_size)
        frequency_sum = coordinates[:, None] + coordinates[None, :]
        self.register_buffer(
            "high_frequency_mask",
            frequency_sum >= self.dct_size,
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = images.shape
        if height < self.patch_size or width < self.patch_size:
            raise ValueError("Input is smaller than forensic patch size")
        unfolded = F.unfold(images, kernel_size=self.patch_size, stride=self.patch_size)
        patch_count = unfolded.shape[-1]
        if patch_count < 2 * self.patches_per_band:
            raise ValueError(
                f"Need at least {2 * self.patches_per_band} candidate patches, got {patch_count}"
            )
        patches = unfolded.transpose(1, 2).reshape(
            batch_size,
            patch_count,
            channels,
            self.patch_size,
            self.patch_size,
        )
        gray = patches.mean(dim=2).reshape(
            batch_size * patch_count, 1, self.patch_size, self.patch_size
        )
        gray = F.adaptive_avg_pool2d(gray, (self.dct_size, self.dct_size)).squeeze(1)
        coefficients = torch.matmul(self.dct_basis, gray)
        coefficients = torch.matmul(coefficients, self.dct_basis.transpose(0, 1))
        energy = coefficients.square().reshape(batch_size, patch_count, -1)
        high_mask = self.high_frequency_mask.flatten()
        high_energy = energy[:, :, high_mask].mean(dim=-1)
        total_energy = energy.mean(dim=-1).clamp_min(1e-8)
        frequency_ratio = high_energy / total_energy

        high_indices = frequency_ratio.topk(self.patches_per_band, dim=1, largest=True).indices
        low_indices = frequency_ratio.topk(self.patches_per_band, dim=1, largest=False).indices
        selected_indices = torch.cat((low_indices, high_indices), dim=1)
        gather_index = selected_indices[:, :, None, None, None].expand(
            -1, -1, channels, self.patch_size, self.patch_size
        )
        selected = patches.gather(1, gather_index)
        patch_types = torch.cat(
            (torch.zeros_like(low_indices), torch.ones_like(high_indices)), dim=1
        )
        return selected, frequency_ratio, patch_types


class ForensicBranch(nn.Module):
    def __init__(
        self,
        patch_size: int,
        patches_per_band: int,
        dct_size: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.selector = DCTPatchSelector(patch_size, patches_per_band, dct_size)
        self.register_buffer("srm_filters", _srm_inspired_bank(), persistent=True)
        self.channel_adapter = nn.Sequential(
            nn.Conv2d(36, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 3, kernel_size=1, bias=False),
        )
        backbone = resnet18(weights=None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, output_dim),
            nn.GELU(),
        )
        self.patch_type_embedding = nn.Embedding(2, output_dim)
        self.attention_score = nn.Linear(output_dim, 1)

    def _residual_views(self, patches: torch.Tensor) -> torch.Tensor:
        gray = patches.mean(dim=1, keepdim=True)
        srm = F.conv2d(gray, self.srm_filters, padding=2)
        srm = torch.tanh(3.0 * srm)
        reduced = F.interpolate(patches, scale_factor=0.5, mode="nearest")
        restored = F.interpolate(reduced, size=patches.shape[-2:], mode="nearest")
        npr = patches - restored
        return torch.cat((patches, srm, npr), dim=1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        selected, frequency_ratios, patch_types = self.selector(images)
        batch_size, selected_count, channels, height, width = selected.shape
        flattened = selected.reshape(batch_size * selected_count, channels, height, width)
        forensic_input = self.channel_adapter(self._residual_views(flattened))
        encoded = self.projection(self.encoder(forensic_input))
        encoded = encoded.reshape(batch_size, selected_count, -1)
        encoded = encoded + self.patch_type_embedding(patch_types)
        attention = torch.softmax(self.attention_score(encoded).squeeze(-1), dim=1)
        pooled = (encoded * attention.unsqueeze(-1)).sum(dim=1)
        return pooled, {
            "patch_attention": attention,
            "frequency_ratios": frequency_ratios,
        }

