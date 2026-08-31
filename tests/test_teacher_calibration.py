from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from scripts.calibrate_teacher_logits import (
    _apply_per_view_affine,
    _fit_per_view_affine_platt,
    _parse_m2_weight_grid,
    _select_mixture,
)
from scripts.freeze_student_distillation_config import freeze_config
from scripts.select_student_smoke_candidate import select_candidate


def test_mixture_grid_requires_both_single_teacher_baselines() -> None:
    assert _parse_m2_weight_grid("0,0.3,1") == [0.0, 0.3, 1.0]


def test_per_view_affine_platt_moves_the_binary_boundary() -> None:
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    logits = torch.tensor(
        [[3.0, 5.0], [4.0, 6.0], [7.0, 9.0], [8.0, 10.0]]
    )
    view_ids = ["clean", "jpeg50"]

    parameters = _fit_per_view_affine_platt(logits, labels, view_ids)
    calibrated = torch.sigmoid(
        _apply_per_view_affine(logits, view_ids, parameters)
    )

    assert parameters["clean"]["a"] > 0.0
    assert 4.0 < parameters["clean"]["raw_logit_center"] < 7.0
    assert 6.0 < parameters["jpeg50"]["raw_logit_center"] < 9.0
    assert torch.all(calibrated[:2] < 0.5)
    assert torch.all(calibrated[2:] > 0.5)


def test_mixture_gate_can_fall_back_to_m3_primary() -> None:
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    m3_probabilities = torch.tensor(
        [[0.1, 0.2], [0.2, 0.3], [0.8, 0.7], [0.9, 0.8]]
    )
    m2_probabilities = torch.tensor(
        [[0.2, 0.3], [0.3, 0.4], [0.7, 0.6], [0.8, 0.7]]
    )
    selected, candidates, decision = _select_mixture(
        labels,
        torch.logit(m2_probabilities),
        torch.logit(m3_probabilities),
        m2_temperature=1.0,
        m3_temperature=1.0,
        m2_weight_grid=[0.0, 0.3, 1.0],
        minimum_dual_auroc_gain=1.0,
        maximum_clean_auroc_regression=0.0,
    )

    assert len(candidates) == 3
    assert selected["m2_weight"] == 0.0
    assert selected["m3_weight"] == 1.0
    assert decision["selected_reason"] == "m3_primary_fallback"


def test_freeze_config_writes_calibrated_mixture(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "temperatures": {"m2": 1.2, "m3": 0.9},
                "teacher_checkpoint_sha256": {"m2": "2" * 64, "m3": "3" * 64},
                "selected_mixture": {"m2_weight": 0.4, "m3_weight": 0.6},
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "frozen.yaml"

    freeze_config(
        str(
            root
            / "configs"
            / "community_forensics_v3"
            / "student_mnv3_dual_teacher.yaml"
        ),
        str(calibration_path),
        str(output_path),
    )
    frozen = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    assert frozen["distillation"]["m2_weight"] == 0.4
    assert frozen["distillation"]["m3_weight"] == 0.6
    assert frozen["distillation"]["teacher_calibration_temperatures"] == {
        "m2": 1.2,
        "m3": 0.9,
    }


def test_smoke_selector_uses_actual_dual_candidate_weight() -> None:
    calibration = {"selected_mixture": {"m2_weight": 0.0, "m3_weight": 1.0}}
    dual = {"robust_mean_auroc": 0.91, "clean_auroc": 0.90}
    m3 = {"robust_mean_auroc": 0.90, "clean_auroc": 0.9005}

    decision = select_candidate(
        calibration,
        dual,
        m3,
        dual_m2_weight=0.3,
        minimum_robust_auroc_gain=0.002,
        maximum_clean_auroc_regression=0.001,
    )

    assert decision["winner"] == "dual"
    assert decision["calibration_selected_m2_weight"] == 0.0
    assert decision["dual_candidate_m2_weight"] == 0.3
