from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from repostguard.distillation import (
    affine_calibrated_teacher_probabilities,
    binary_probability_distillation_loss,
    compute_dual_teacher_loss,
    relational_feature_distillation_loss,
    teacher_reliability_weights,
    validate_distillation_config,
)


def _affine_payload() -> dict[str, object]:
    views = {
        "clean": {"a": 1.0, "b": -2.0},
        "jpeg50": {"a": 0.5, "b": -1.0},
        "resize50_jpeg70": {"a": 0.75, "b": -1.5},
        "strict6": {"a": 0.25, "b": -0.5},
    }
    return {
        "schema_version": 2,
        "calibration_method": "per_view_affine_platt",
        "view_ids": ["clean", "jpeg50"],
        "teacher_checkpoint_sha256": {"m2": "2" * 64, "m3": "3" * 64},
        "affine_calibration": {"m2": views, "m3": views},
    }


def _config(calibration_path: Path, *, feature: bool = False) -> dict[str, object]:
    return {
        "distillation": {
            "m2_weight": 0.0,
            "m3_weight": 1.0,
            "temperature": 1.0,
            "teacher_checkpoint_sha256": {"m2": "2" * 64, "m3": "3" * 64},
            "teacher_calibration_temperatures": {"m2": 1.0, "m3": 1.0},
            "teacher_calibration": {
                "method": "per_view_affine_platt",
                "path": str(calibration_path),
            },
            "teacher_reliability": {
                "enabled": True,
                "confidence_power": 1.0,
                "wrong_teacher_scale": 0.1,
                "minimum_weight": 0.0,
            },
            "disagreement_threshold": 0.25,
            "disagreement_scale": 0.25,
            "distillation_warmup_epochs": 2,
            "hard_label_weight": 0.5 if feature else 0.8,
            "soft_teacher_weight": 0.15,
            "consistency_weight": 0.05,
            "feature_weight": 0.3 if feature else 0.0,
            "feature_distillation": {
                "enabled": feature,
                "teacher_dim": 2,
                "branch_weights": {
                    "semantic": 0.15,
                    "forensic": 0.35,
                    "fused": 0.50,
                },
                "component_weights": {"pointwise": 0.75, "relational": 0.25},
                "relational": {
                    "enabled": True,
                    "branch_weights": {"forensic": 0.4, "fused": 0.6},
                    "cross_view_delta_weight": 0.25,
                },
            },
            "views": [
                {"id": "clean", "name": "clean", "params": {}},
                {"id": "jpeg50", "name": "jpeg", "params": {"quality": 50}},
            ],
        }
    }


def test_per_view_affine_calibration_uses_each_samples_view() -> None:
    payload = _affine_payload()
    probabilities = affine_calibrated_teacher_probabilities(
        torch.tensor([2.0, 2.0]),
        teacher="m3",
        view_ids=["clean", "jpeg50"],
        calibration_payload=payload,
    )
    torch.testing.assert_close(probabilities, torch.tensor([0.5, 0.5]))


def test_reliability_downweights_wrong_confident_teacher() -> None:
    weights, correct = teacher_reliability_weights(
        torch.tensor([0.9, 0.9, 0.55]),
        torch.tensor([1.0, 0.0, 1.0]),
        confidence_power=1.0,
        wrong_teacher_scale=0.1,
        minimum_weight=0.0,
    )
    torch.testing.assert_close(weights, torch.tensor([0.8, 0.08, 0.1]))
    assert correct.tolist() == [True, False, True]


def test_probability_kd_reports_zero_kl_at_teacher_solution() -> None:
    targets = torch.tensor([0.2, 0.8])
    loss = binary_probability_distillation_loss(
        torch.logit(targets), targets, torch.ones_like(targets)
    )
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_relational_loss_matches_identical_batch_geometry() -> None:
    teacher = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 2.0]])
    assert float(relational_feature_distillation_loss(teacher, teacher)) == pytest.approx(0.0)
    shuffled = teacher[[1, 0, 2]]
    assert float(relational_feature_distillation_loss(shuffled, teacher)) > 0.0


def test_v32_loss_uses_affine_targets_and_warmup(tmp_path: Path) -> None:
    calibration_path = tmp_path / "affine.json"
    calibration_path.write_text(json.dumps(_affine_payload()), encoding="utf-8")
    config = _config(calibration_path)
    labels = torch.tensor([0.0, 1.0])
    clean_logits = torch.tensor([-0.5, 0.5], requires_grad=True)
    aug_logits = torch.tensor([-0.4, 0.4], requires_grad=True)
    batch = {
        "view_id": ["jpeg50", "jpeg50"],
        "teacher_m2_logit_clean": torch.zeros(2),
        "teacher_m3_logit_clean": torch.tensor([1.0, 3.0]),
        "teacher_m2_logit_aug": torch.zeros(2),
        "teacher_m3_logit_aug": torch.tensor([1.0, 3.0]),
    }
    loss, components = compute_dual_teacher_loss(
        config,
        labels,
        {"logits": clean_logits},
        {"logits": aug_logits},
        batch,
        epoch=0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert components["distillation_ramp"] == pytest.approx(0.5)
    assert components["teacher_positive_fraction"] == pytest.approx(0.5)
    assert components["mean_teacher_disagreement"] == pytest.approx(0.0)
    assert clean_logits.grad is not None
    assert aug_logits.grad is not None


def test_affine_calibration_rejects_second_temperature(tmp_path: Path) -> None:
    config = _config(tmp_path / "affine.json")
    config["distillation"]["temperature"] = 2.0
    with pytest.raises(ValueError, match="second softening"):
        validate_distillation_config(config)
