from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    efficientnet_b0,
    mobilenet_v3_large,
)

from repostguard.models.quality_gate import QualityAwareGate


def _high_pass_bank() -> torch.Tensor:
    """Return a small deterministic bank of image-forensics residual filters."""

    kernels = torch.tensor(
        [
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
            [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]],
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]],
            [[-1.0, 2.0, -1.0], [2.0, -4.0, 2.0], [-1.0, 2.0, -1.0]],
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    kernels = kernels - kernels.mean(dim=(1, 2), keepdim=True)
    kernels = kernels / kernels.abs().sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    return kernels.unsqueeze(1)


class LightweightForensicStudentBranch(nn.Module):
    """Mobile-exportable residual branch backed by EfficientNet-B0 features."""

    def __init__(
        self,
        *,
        pretrained: bool,
        output_dim: int,
        use_npr: bool = False,
    ) -> None:
        super().__init__()
        self.use_npr = bool(use_npr)
        self.register_buffer("high_pass_filters", _high_pass_bank(), persistent=True)
        forensic_input_channels = 8 + (3 if self.use_npr else 0)
        self.channel_adapter = nn.Sequential(
            nn.Conv2d(forensic_input_channels, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(),
            nn.Conv2d(16, 3, kernel_size=1, bias=True),
        )
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.features = backbone.features
        self.pool = backbone.avgpool
        feature_dim = int(backbone.classifier[1].in_features)
        self.projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, int(output_dim)),
            nn.Hardswish(),
        )
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def _residual_input(self, images: torch.Tensor) -> torch.Tensor:
        gray = images.mean(dim=1, keepdim=True)
        high_pass = torch.tanh(
            3.0 * F.conv2d(gray, self.high_pass_filters, padding=1)
        )
        forensic_views = high_pass
        if self.use_npr:
            # Match the teacher's neighboring-pixel residual without adding a
            # second backbone. Nearest-neighbor resize is deterministic and
            # supported by the mobile ONNX/TorchScript export path.
            reduced = F.interpolate(images, scale_factor=0.5, mode="nearest")
            restored = F.interpolate(reduced, size=images.shape[-2:], mode="nearest")
            npr = images - restored
            forensic_views = torch.cat((high_pass, npr), dim=1)
        adapted = self.channel_adapter(forensic_views)
        return torch.sigmoid(adapted)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        residual = self._residual_input(images)
        normalized = (residual - self.mean) / self.std
        features = self.features(normalized)
        features = self.pool(features).flatten(1)
        return self.projection(features)


class MobileNetV3Student(nn.Module):
    """Mobile-oriented binary detector used by the first distillation stage."""

    def __init__(self, *, pretrained: bool, dropout: float) -> None:
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_large(weights=weights)
        self.features = backbone.features
        self.pool = backbone.avgpool
        feature_dim = int(backbone.classifier[0].in_features)
        hidden_dim = int(backbone.classifier[0].out_features)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Hardswish(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        normalized = (images - self.mean) / self.std
        features = self.features(normalized)
        features = self.pool(features).flatten(1)
        logits = self.classifier(features).squeeze(1)
        return {"logits": logits, "features": features}


class MobileNetV3ForensicStudent(nn.Module):
    """8-12M parameter Student with semantic and high-frequency pathways."""

    def __init__(
        self,
        *,
        pretrained: bool,
        forensic_pretrained: bool,
        dropout: float,
        distill_dim: int = 256,
        fusion_dim: int = 512,
        use_npr: bool = False,
        quality_gate_enabled: bool = False,
        quality_gate_hidden_dim: int = 16,
        fusion_uses_projected_semantic: bool = False,
    ) -> None:
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_large(weights=weights)
        self.features = backbone.features
        self.pool = backbone.avgpool
        semantic_dim = int(backbone.classifier[0].in_features)
        self.semantic_projection = nn.Sequential(
            nn.LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, int(distill_dim)),
            nn.Hardswish(),
        )
        self.forensic = LightweightForensicStudentBranch(
            pretrained=forensic_pretrained,
            output_dim=int(distill_dim),
            use_npr=bool(use_npr),
        )
        self.fusion_uses_projected_semantic = bool(fusion_uses_projected_semantic)
        self.quality_gate_enabled = bool(quality_gate_enabled)
        if self.quality_gate_enabled:
            self.quality_gate = QualityAwareGate(
                hidden_dim=int(quality_gate_hidden_dim)
            )
        semantic_fusion_dim = (
            int(distill_dim) if self.fusion_uses_projected_semantic else semantic_dim
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(semantic_fusion_dim + int(distill_dim)),
            nn.Linear(semantic_fusion_dim + int(distill_dim), int(fusion_dim)),
            nn.Hardswish(),
            nn.Dropout(float(dropout)),
        )
        self.fused_projection = nn.Sequential(
            nn.LayerNorm(int(fusion_dim)),
            nn.Linear(int(fusion_dim), int(distill_dim)),
            nn.Hardswish(),
        )
        self.classifier = nn.Linear(int(fusion_dim), 1)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        normalized = (images - self.mean) / self.std
        semantic_backbone = self.features(normalized)
        semantic_backbone = self.pool(semantic_backbone).flatten(1)
        semantic = self.semantic_projection(semantic_backbone)
        forensic = self.forensic(images)
        semantic_fusion_source = (
            semantic if self.fusion_uses_projected_semantic else semantic_backbone
        )
        gate_fractions: torch.Tensor | None = None
        quality_features: torch.Tensor | None = None
        if self.quality_gate_enabled:
            gate_fractions, quality_features = self.quality_gate(images)
            branch_scales = 2.0 * gate_fractions
            semantic_for_fusion = branch_scales[:, 0:1] * semantic_fusion_source
            forensic_for_fusion = branch_scales[:, 1:2] * forensic
        else:
            semantic_for_fusion = semantic_fusion_source
            forensic_for_fusion = forensic
        fused_hidden = self.fusion(
            torch.cat((semantic_for_fusion, forensic_for_fusion), dim=1)
        )
        fused = self.fused_projection(fused_hidden)
        logits = self.classifier(fused_hidden).squeeze(1)
        output = {
            "logits": logits,
            "features": fused,
            "semantic_features": semantic,
            "forensic_features": forensic,
        }
        if gate_fractions is not None and quality_features is not None:
            output["gate_fractions"] = gate_fractions
            output["quality_features"] = quality_features
        return output
