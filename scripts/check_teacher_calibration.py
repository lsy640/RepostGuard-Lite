from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate frozen dual-teacher calibration")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--expected-samples", type=int, default=2000)
    parser.add_argument("--expected-views", type=int, default=4)
    parser.add_argument("--m2-checkpoint-sha256", required=True)
    parser.add_argument("--m3-checkpoint-sha256", required=True)
    arguments = parser.parse_args()
    with Path(arguments.calibration).open("r", encoding="utf-8") as handle:
        calibration = json.load(handle)

    failures: list[str] = []
    if int(calibration.get("samples", -1)) != arguments.expected_samples:
        failures.append("unexpected sample count")
    if int(calibration.get("views", -1)) != arguments.expected_views:
        failures.append("unexpected view count")
    lineage = calibration.get("teacher_checkpoint_sha256", {})
    if lineage.get("m2") != arguments.m2_checkpoint_sha256:
        failures.append("M2 checkpoint lineage mismatch")
    if lineage.get("m3") != arguments.m3_checkpoint_sha256:
        failures.append("M3 checkpoint lineage mismatch")
    losses = calibration.get("binary_cross_entropy", {})
    for teacher in ("m2", "m3"):
        before = float(losses.get(f"{teacher}_before", float("nan")))
        after = float(losses.get(f"{teacher}_after", float("nan")))
        if not math.isfinite(before) or not math.isfinite(after) or after > before + 1e-7:
            failures.append(f"{teacher.upper()} calibration BCE did not improve")
    mixture = calibration.get("selected_mixture", {})
    m2_weight = float(mixture.get("m2_weight", float("nan")))
    m3_weight = float(mixture.get("m3_weight", float("nan")))
    if (
        not math.isfinite(m2_weight)
        or not math.isfinite(m3_weight)
        or not 0.0 <= m2_weight <= 1.0
        or not 0.0 <= m3_weight <= 1.0
        or not math.isclose(m2_weight + m3_weight, 1.0, abs_tol=1e-8)
    ):
        failures.append("invalid selected mixture")
    candidate_weights = {
        round(float(row["m2_weight"]), 8)
        for row in calibration.get("mixture_candidates", [])
    }
    if not {0.0, 0.3, 1.0}.issubset(candidate_weights):
        failures.append("mixture grid omitted a required baseline")

    payload = {
        "event": "teacher_calibration_acceptance",
        "accepted": not failures,
        "selected_mixture": {"m2_weight": m2_weight, "m3_weight": m3_weight},
        "mixture_decision": calibration.get("mixture_decision"),
        "failures": failures,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
