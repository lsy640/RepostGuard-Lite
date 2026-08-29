from __future__ import annotations

from pathlib import Path
from typing import Any

import open_clip
import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from repostguard.models.forensic import ForensicBranch
from repostguard.models.quality_gate import QualityAwareGate


class ChannelNormalize(nn.Module):
    def __init__(self, mean: list[float], std: list[float]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1), persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.mean) / self.std


class EfficientNetDetector(nn.Module):
    def __init__(self, pretrained: bool, dropout: float) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.features = backbone.features
        self.pool = backbone.avgpool
        feature_dim = int(backbone.classifier[1].in_features)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, 1))
        self.normalize = ChannelNormalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.features(self.normalize(images))
        features = self.pool(features).flatten(1)
        return {"logits": self.classifier(features).squeeze(1), "features": features}


class FrozenClipEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: str,
        cache_dir: str,
        mean: list[float],
        std: list[float],
    ) -> None:
        super().__init__()
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            cache_dir=cache_dir,
        )
        self.visual = clip_model.visual
        output_dim = getattr(self.visual, "output_dim", None)
        if output_dim is None:
            output_dim = getattr(clip_model, "embed_dim", None)
        if output_dim is None:
            raise ValueError(f"Cannot infer image feature dimension for {model_name}")
        self.output_dim = int(output_dim)
        self.normalize = ChannelNormalize(mean, std)
        self.visual.requires_grad_(False)
        self.visual.eval()
        del clip_model

    def train(self, mode: bool = True) -> "FrozenClipEncoder":
        super().train(mode)
        self.visual.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.visual.eval()
        with torch.no_grad():
            features = self.visual(self.normalize(images))
        if isinstance(features, (tuple, list)):
            features = features[0]
        if isinstance(features, dict):
            features = features.get("image_features", features.get("x"))
        if not isinstance(features, torch.Tensor):
            raise TypeError("OpenCLIP visual tower returned an unsupported feature type")
        return features.float()


class ClipLinearDetector(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.encoder = FrozenClipEncoder(
            config["clip_model"],
            config["clip_pretrained"],
            config["clip_cache_dir"],
            config["clip_mean"],
            config["clip_std"],
        )
        self.classifier = nn.Linear(self.encoder.output_dim, 1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(images)
        return {"logits": self.classifier(features).squeeze(1), "features": features}


class RepostGuardM2(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        projection_dim = int(config["projection_dim"])
        dropout = float(config["dropout"])
        self.semantic = FrozenClipEncoder(
            config["clip_model"],
            config["clip_pretrained"],
            config["clip_cache_dir"],
            config["clip_mean"],
            config["clip_std"],
        )
        self.semantic_projection = nn.Sequential(
            nn.LayerNorm(self.semantic.output_dim),
            nn.Linear(self.semantic.output_dim, projection_dim),
            nn.GELU(),
        )
        forensic_config = config["forensic"]
        self.forensic = ForensicBranch(
            patch_size=int(forensic_config["patch_size"]),
            patches_per_band=int(forensic_config["patches_per_band"]),
            dct_size=int(forensic_config["dct_size"]),
            output_dim=int(forensic_config["output_dim"]),
        )
        fused_input_dim = projection_dim + int(forensic_config["output_dim"])
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_input_dim),
            nn.Linear(fused_input_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(projection_dim, 1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        semantic = self.semantic_projection(self.semantic(images))
        forensic, forensic_diagnostics = self.forensic(images)
        fused = self.fusion(torch.cat((semantic, forensic), dim=1))
        return {
            "logits": self.classifier(fused).squeeze(1),
            "features": fused,
            "semantic_features": semantic,
            "forensic_features": forensic,
            **forensic_diagnostics,
        }


class RepostGuardM3(RepostGuardM2):
    """M2 with a label-agnostic quality gate over the two feature branches."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        gate_config = config["quality_gate"]
        self.quality_gate = QualityAwareGate(hidden_dim=int(gate_config["hidden_dim"]))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        semantic = self.semantic_projection(self.semantic(images))
        forensic, forensic_diagnostics = self.forensic(images)
        gate_fractions, quality_features = self.quality_gate(images)
        branch_scales = 2.0 * gate_fractions
        fused = self.fusion(
            torch.cat(
                (
                    branch_scales[:, 0:1] * semantic,
                    branch_scales[:, 1:2] * forensic,
                ),
                dim=1,
            )
        )
        return {
            "logits": self.classifier(fused).squeeze(1),
            "features": fused,
            "semantic_features": semantic,
            "forensic_features": forensic,
            "gate_fractions": gate_fractions,
            "quality_features": quality_features,
            **forensic_diagnostics,
        }


def build_model(config: dict[str, Any]) -> nn.Module:
    model_config = config["model"]
    experiment = str(model_config["experiment"]).lower()
    if experiment in {"b0", "b1"}:
        if model_config["cnn_backbone"] != "efficientnet_b0":
            raise ValueError("The pilot currently supports only efficientnet_b0 for B0/B1")
        return EfficientNetDetector(
            bool(model_config["cnn_pretrained"]),
            float(model_config["dropout"]),
        )
    if experiment == "b2":
        return ClipLinearDetector(model_config)
    if experiment == "m2":
        return RepostGuardM2(model_config)
    if experiment == "m3":
        return RepostGuardM3(model_config)
    raise ValueError(f"Unsupported experiment: {experiment}")


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
