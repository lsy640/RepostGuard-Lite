from __future__ import annotations

import copy
import csv
import hashlib
from pathlib import Path

import pytest
import torch
from PIL import Image

from repostguard.data.distillation import CachedDistillationDataset, sha256_file
from repostguard.distillation import (
    calibrated_teacher_probabilities,
    compute_dual_teacher_loss,
    normalized_feature_distillation_loss,
    validate_distillation_config,
)
from repostguard.models.student import MobileNetV3ForensicStudent, MobileNetV3Student


def _logit(probability: float) -> float:
    value = torch.tensor(probability, dtype=torch.float32)
    return float(torch.logit(value))


def _config(root: Path, manifest: Path, cache_directory: Path) -> dict[str, object]:
    return {
        "seed": 7,
        "data": {
            "root": str(root),
            "train_manifest": str(manifest),
            "image_size": 32,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "format_debias": {
                "enabled": True,
                "train_qualities": [70, 80],
                "eval_quality": 90,
                "jpeg_subsampling": 2,
            },
        },
        "train": {"batch_size": 2},
        "distillation": {
            "cache_directory": str(cache_directory),
            "m2_weight": 0.3,
            "m3_weight": 0.7,
            "temperature": 3.0,
            "teacher_calibration_temperatures": {"m2": 1.0, "m3": 1.0},
            "disagreement_threshold": 0.25,
            "disagreement_scale": 0.25,
            "hard_label_weight": 0.5,
            "soft_teacher_weight": 0.4,
            "consistency_weight": 0.1,
            "feature_weight": 0.0,
            "feature_distillation": {
                "enabled": False,
                "teacher_dim": 256,
                "branch_weights": {
                    "semantic": 0.35,
                    "forensic": 0.35,
                    "fused": 0.30,
                },
            },
            "views": [
                {"id": "clean", "name": "clean", "params": {}},
                {"id": "jpeg50", "name": "jpeg", "params": {"quality": 50}},
                {
                    "id": "resize",
                    "name": "resize",
                    "params": {"scale": 0.5, "interpolation": "bicubic"},
                },
                {
                    "id": "strict6",
                    "name": "strict_random_six",
                    "params": {"seed": 11, "profile": "full_training_range_v1"},
                },
            ],
        },
    }


def test_mobile_student_returns_binary_logits_and_features() -> None:
    model = MobileNetV3Student(pretrained=False, dropout=0.2)
    output = model(torch.rand(2, 3, 64, 64))

    assert output["logits"].shape == (2,)
    assert output["features"].shape[0] == 2


def test_forensic_student_stays_in_mobile_parameter_budget() -> None:
    model = MobileNetV3ForensicStudent(
        pretrained=False,
        forensic_pretrained=False,
        dropout=0.2,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    output = model(torch.rand(2, 3, 64, 64))

    assert 8_000_000 <= parameter_count <= 12_000_000
    assert output["logits"].shape == (2,)
    for key in ("features", "semantic_features", "forensic_features"):
        assert output[key].shape == (2, 256)


def test_normalized_feature_loss_matches_identical_targets() -> None:
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    assert float(normalized_feature_distillation_loss(features, features)) == pytest.approx(0.0)
    assert float(
        normalized_feature_distillation_loss(
            features, torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        )
    ) == pytest.approx(1.0)


def test_teacher_probability_mixture_uses_m3_as_primary() -> None:
    probability, disagreement = calibrated_teacher_probabilities(
        torch.tensor([_logit(0.2)]),
        torch.tensor([_logit(0.8)]),
        m2_weight=0.3,
        m3_weight=0.7,
        m2_calibration_temperature=1.0,
        m3_calibration_temperature=1.0,
    )

    torch.testing.assert_close(probability, torch.tensor([0.62]))
    torch.testing.assert_close(disagreement, torch.tensor([0.6]))


def test_dual_teacher_loss_is_finite_and_downweights_disagreement(tmp_path: Path) -> None:
    config = _config(tmp_path, tmp_path / "manifest.csv", tmp_path / "cache")
    labels = torch.tensor([0.0, 1.0])
    clean_logits = torch.tensor([-0.5, 0.5], requires_grad=True)
    augmented_logits = torch.tensor([-0.25, 0.25], requires_grad=True)
    batch = {
        "teacher_m2_logit_clean": torch.tensor([_logit(0.1), _logit(0.9)]),
        "teacher_m3_logit_clean": torch.tensor([_logit(0.9), _logit(0.1)]),
        "teacher_m2_logit_aug": torch.tensor([_logit(0.2), _logit(0.8)]),
        "teacher_m3_logit_aug": torch.tensor([_logit(0.8), _logit(0.2)]),
    }
    loss, components = compute_dual_teacher_loss(
        config,
        labels,
        {"logits": clean_logits},
        {"logits": augmented_logits},
        batch,
    )
    full_weight_config = copy.deepcopy(config)
    full_weight_config["distillation"]["disagreement_threshold"] = 1.0
    _, full_weight_components = compute_dual_teacher_loss(
        full_weight_config,
        labels,
        {"logits": clean_logits},
        {"logits": augmented_logits},
        batch,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert components["downweighted_fraction"] == pytest.approx(1.0)
    assert components["soft_teacher"] == pytest.approx(
        0.25 * full_weight_components["soft_teacher"]
    )
    assert clean_logits.grad is not None
    assert augmented_logits.grad is not None


def test_cached_distillation_dataset_validates_hashes_and_expands_views(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    rows: list[dict[str, str]] = []
    for index, label in enumerate((0, 1)):
        image_path = image_root / f"sample_{index}.png"
        Image.new("RGB", (48, 40), color=(20 + 100 * index, 40, 60)).save(image_path)
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "path": image_path.name,
                "label": str(label),
                "split": "train",
                "source_dataset": "unit",
                "generator_id": "real" if label == 0 else "generator-a",
                "sha256": sha256_file(image_path),
            }
        )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    config = _config(image_root, manifest, cache_directory)
    payload = {
        "metadata": {
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "views": config["distillation"]["views"],
            "m2_checkpoint_sha256": "2" * 64,
            "m3_checkpoint_sha256": "3" * 64,
        },
        "sample_ids": [row["sample_id"] for row in rows],
        "sample_sha256": [row["sha256"] for row in rows],
        "labels": torch.tensor([0.0, 1.0]),
        "m2_logits": torch.zeros(2, 4),
        "m3_logits": torch.ones(2, 4),
    }
    torch.save(payload, cache_directory / "teacher_cache_000-of-001.pt")

    dataset = CachedDistillationDataset(config)
    item = dataset[2]

    assert len(dataset) == 6
    assert item["sample_id"] == "sample-0"
    assert item["view_id"] == "strict6"
    assert item["image"].shape == (3, 32, 32)
    assert item["image_aug"].shape == (3, 32, 32)
    assert float(item["teacher_m3_logit_aug"]) == 1.0

    config["distillation"].update(
        {
            "hard_label_weight": 0.5,
            "soft_teacher_weight": 0.25,
            "consistency_weight": 0.05,
            "feature_weight": 0.2,
            "feature_distillation": {
                "enabled": True,
                "teacher_dim": 256,
                "branch_weights": {
                    "semantic": 0.35,
                    "forensic": 0.35,
                    "fused": 0.30,
                },
            },
        }
    )
    payload.update(
        {
            "m3_semantic_features": torch.zeros(2, 4, 256),
            "m3_forensic_features": torch.ones(2, 4, 256),
            "m3_fused_features": torch.full((2, 4, 256), 2.0),
        }
    )
    torch.save(payload, cache_directory / "teacher_cache_000-of-001.pt")
    feature_dataset = CachedDistillationDataset(config)
    feature_item = feature_dataset[2]

    assert feature_item["teacher_m3_semantic_clean"].shape == (256,)
    assert feature_item["teacher_m3_forensic_aug"].shape == (256,)
    assert float(feature_item["teacher_m3_fused_aug"][0]) == 2.0


def test_distillation_config_rejects_non_normalized_loss_weights(tmp_path: Path) -> None:
    config = _config(tmp_path, tmp_path / "manifest.csv", tmp_path / "cache")
    config["distillation"]["consistency_weight"] = 0.2

    with pytest.raises(ValueError, match="loss weights must sum"):
        validate_distillation_config(config)


def test_distillation_config_rejects_teacher_weights_outside_unit_interval(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, tmp_path / "manifest.csv", tmp_path / "cache")
    config["distillation"]["m2_weight"] = -0.1
    config["distillation"]["m3_weight"] = 1.1

    with pytest.raises(ValueError, match="teacher weights must be in"):
        validate_distillation_config(config)
