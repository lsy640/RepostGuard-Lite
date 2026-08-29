from __future__ import annotations

import torch
from torch.nn import functional as F


def symmetric_bernoulli_kl(clean_logits: torch.Tensor, augmented_logits: torch.Tensor) -> torch.Tensor:
    """Return the symmetric KL between Bernoulli predictions.

    For Bernoulli parameters p=sigmoid(a) and q=sigmoid(b), the sum of
    KL(p || q) and KL(q || p) is (p - q) * (a - b).  This form avoids
    taking log(0).  Computing it in float32 is important under autocast:
    1 - 1e-6 rounds to 1 in float16, so probability clamping in float16
    does not prevent an infinite logarithm.
    """
    clean_logits_fp32 = clean_logits.float()
    augmented_logits_fp32 = augmented_logits.float()
    clean_probability = torch.sigmoid(clean_logits_fp32)
    augmented_probability = torch.sigmoid(augmented_logits_fp32)
    return 0.5 * (
        (clean_probability - augmented_probability)
        * (clean_logits_fp32 - augmented_logits_fp32)
    ).mean()


def compute_training_loss(
    experiment: str,
    labels: torch.Tensor,
    clean_output: dict[str, torch.Tensor],
    augmented_output: dict[str, torch.Tensor] | None,
    *,
    lambda_kl: float,
    lambda_feature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    clean_bce = F.binary_cross_entropy_with_logits(clean_output["logits"], labels)
    if experiment not in {"m2", "m3"}:
        return clean_bce, {"classification": float(clean_bce.detach()), "total": float(clean_bce.detach())}
    if augmented_output is None:
        raise ValueError(f"{experiment.upper()} requires paired augmented output")
    augmented_bce = F.binary_cross_entropy_with_logits(augmented_output["logits"], labels)
    consistency_kl = symmetric_bernoulli_kl(
        clean_output["logits"], augmented_output["logits"]
    )
    feature_consistency = (1.0 - F.cosine_similarity(
        clean_output["features"], augmented_output["features"], dim=1
    )).mean()
    total = (
        clean_bce
        + augmented_bce
        + float(lambda_kl) * consistency_kl
        + float(lambda_feature) * feature_consistency
    )
    return total, {
        "clean_bce": float(clean_bce.detach()),
        "augmented_bce": float(augmented_bce.detach()),
        "consistency_kl": float(consistency_kl.detach()),
        "feature_consistency": float(feature_consistency.detach()),
        "total": float(total.detach()),
    }
