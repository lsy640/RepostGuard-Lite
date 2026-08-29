from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def select_balanced_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, probabilities)
    balanced_accuracy = 0.5 * (true_positive_rate + 1.0 - false_positive_rate)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    finite_indices = np.flatnonzero(finite)
    best = finite_indices[int(np.argmax(balanced_accuracy[finite]))]
    return float(np.clip(thresholds[best], 0.0, 1.0))


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        if upper == 1.0:
            selected = (probabilities >= lower) & (probabilities <= upper)
        else:
            selected = (probabilities >= lower) & (probabilities < upper)
        if not selected.any():
            continue
        confidence = probabilities[selected].mean()
        accuracy = labels[selected].mean()
        result += selected.mean() * abs(float(accuracy - confidence))
    return float(result)


def _tpr_at_fpr(labels: np.ndarray, probabilities: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(labels, probabilities)
    eligible = np.flatnonzero(fpr <= target_fpr)
    return float(tpr[eligible[-1]]) if eligible.size else 0.0


def binary_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    labels = labels.astype(np.int64)
    predictions = (probabilities >= threshold).astype(np.int64)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    return {
        "n": int(labels.size),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "aigc_recall": float(true_positive / max(1, true_positive + false_negative)),
        "real_specificity": float(true_negative / max(1, true_negative + false_positive)),
        "false_positive_rate": float(false_positive / max(1, false_positive + true_negative)),
        "tpr_at_fpr_1pct": _tpr_at_fpr(labels, probabilities, 0.01),
        "tpr_at_fpr_5pct": _tpr_at_fpr(labels, probabilities, 0.05),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece_15": expected_calibration_error(labels, probabilities),
        "tn": int(true_negative),
        "fp": int(false_positive),
        "fn": int(false_negative),
        "tp": int(true_positive),
    }

