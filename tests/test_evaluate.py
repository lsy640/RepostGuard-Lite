from __future__ import annotations

import pytest

from repostguard.evaluate import _build_summary


def test_build_summary_clean_only_marks_robust_metrics_unavailable() -> None:
    summary = _build_summary(
        "b0",
        0.75,
        [{"auroc": 0.8, "balanced_accuracy": 0.7, "transform": "clean"}],
    )

    assert summary == {
        "experiment": "b0",
        "threshold_from_clean_validation": 0.75,
        "clean_auroc": 0.8,
        "clean_balanced_accuracy": 0.7,
        "robust_mean_auroc": None,
        "robust_worst_auroc": None,
        "robust_worst_transform": None,
        "delta_auroc": None,
        "robust_mean_balanced_accuracy": None,
        "conditions": 1,
    }


def test_build_summary_uses_only_non_clean_conditions_for_robustness() -> None:
    summary = _build_summary(
        "b1",
        0.5,
        [
            {"auroc": 0.9, "balanced_accuracy": 0.8, "transform": "clean"},
            {"auroc": 0.7, "balanced_accuracy": 0.6, "transform": "jpeg"},
            {"auroc": 0.5, "balanced_accuracy": 0.4, "transform": "noise"},
        ],
    )

    assert summary["robust_mean_auroc"] == 0.6
    assert summary["robust_worst_auroc"] == 0.5
    assert summary["robust_worst_transform"] == "noise"
    assert summary["delta_auroc"] == pytest.approx(0.3)
    assert summary["robust_mean_balanced_accuracy"] == 0.5
    assert summary["conditions"] == 3
