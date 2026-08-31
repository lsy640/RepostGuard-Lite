from __future__ import annotations

import argparse
import json
from pathlib import Path

from repostguard.checkpoint import atomic_text


def _load(path: str) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_candidate(
    calibration: dict[str, object],
    dual: dict[str, object],
    m3: dict[str, object],
    *,
    dual_m2_weight: float | None,
    minimum_robust_auroc_gain: float,
    maximum_clean_auroc_regression: float,
) -> dict[str, object]:
    calibration_selected_m2_weight = float(
        calibration["selected_mixture"]["m2_weight"]
    )
    candidate_m2_weight = (
        calibration_selected_m2_weight
        if dual_m2_weight is None
        else float(dual_m2_weight)
    )
    if not 0.0 <= candidate_m2_weight <= 1.0:
        raise ValueError("dual M2 weight must be in [0, 1]")
    robust_gain = float(dual["robust_mean_auroc"]) - float(
        m3["robust_mean_auroc"]
    )
    clean_regression = float(m3["clean_auroc"]) - float(dual["clean_auroc"])
    dual_candidate = candidate_m2_weight > 0.0
    dual_accepted = (
        dual_candidate
        and robust_gain >= minimum_robust_auroc_gain
        and clean_regression <= maximum_clean_auroc_regression
    )
    winner = "dual" if dual_accepted else "m3_primary"
    return {
        "event": "student_smoke_selection",
        "winner": winner,
        "calibration_selected_m2_weight": calibration_selected_m2_weight,
        "dual_candidate_m2_weight": candidate_m2_weight,
        "dual_robust_mean_auroc": float(dual["robust_mean_auroc"]),
        "m3_robust_mean_auroc": float(m3["robust_mean_auroc"]),
        "dual_clean_auroc": float(dual["clean_auroc"]),
        "m3_clean_auroc": float(m3["clean_auroc"]),
        "robust_auroc_gain": robust_gain,
        "clean_auroc_regression": clean_regression,
        "minimum_robust_auroc_gain": minimum_robust_auroc_gain,
        "maximum_clean_auroc_regression": maximum_clean_auroc_regression,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Select dual or M3-primary Student after smoke")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--dual-summary", required=True)
    parser.add_argument("--m3-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dual-m2-weight",
        type=float,
        help="Actual M2 weight used by the dual-teacher smoke candidate",
    )
    parser.add_argument("--minimum-robust-auroc-gain", type=float, default=0.002)
    parser.add_argument("--maximum-clean-auroc-regression", type=float, default=0.001)
    arguments = parser.parse_args()
    calibration = _load(arguments.calibration)
    dual = _load(arguments.dual_summary)
    m3 = _load(arguments.m3_summary)
    payload = select_candidate(
        calibration,
        dual,
        m3,
        dual_m2_weight=arguments.dual_m2_weight,
        minimum_robust_auroc_gain=arguments.minimum_robust_auroc_gain,
        maximum_clean_auroc_regression=arguments.maximum_clean_auroc_regression,
    )
    atomic_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", arguments.output)
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
