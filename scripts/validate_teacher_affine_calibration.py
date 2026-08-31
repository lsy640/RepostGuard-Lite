from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from repostguard.config import load_config
from repostguard.data.distillation import teacher_preprocessing_sha256
from repostguard.distillation import canonical_view_specs


def validate(config_path: str) -> dict[str, object]:
    config = load_config(config_path)
    distillation = config["distillation"]
    calibration_config = distillation["teacher_calibration"]
    if calibration_config["method"] != "per_view_affine_platt":
        raise ValueError("V3.2 requires per_view_affine_platt calibration")
    calibration_path = Path(calibration_config["path"]).expanduser().resolve()
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    if payload.get("calibration_method") != "per_view_affine_platt":
        raise ValueError("Calibration artifact method mismatch")
    expected_views = [str(view["id"]) for view in canonical_view_specs(config)]
    if payload.get("view_ids") != expected_views:
        raise ValueError("Calibration artifact view ids mismatch")
    expected_lineage = distillation["teacher_checkpoint_sha256"]
    if payload.get("teacher_checkpoint_sha256") != expected_lineage:
        raise ValueError("Calibration artifact teacher lineage mismatch")
    if payload.get("preprocessing_sha256") != teacher_preprocessing_sha256(config):
        raise ValueError("Calibration artifact preprocessing lineage mismatch")

    active_teachers = [
        teacher
        for teacher in ("m2", "m3")
        if float(distillation[f"{teacher}_weight"]) > 0.0
    ]
    checks: dict[str, object] = {}
    for teacher in active_teachers:
        teacher_rows = payload["affine_calibration"][teacher]
        for view_id in expected_views:
            row = teacher_rows[view_id]
            numeric = {
                key: float(row[key])
                for key in (
                    "a",
                    "b",
                    "raw_logit_center",
                    "binary_cross_entropy_before",
                    "binary_cross_entropy_after",
                    "balanced_accuracy_at_0_5",
                    "positive_rate_at_0_5",
                    "real_probability_mean",
                    "aigi_probability_mean",
                )
            }
            if any(not math.isfinite(value) for value in numeric.values()):
                raise FloatingPointError(f"Non-finite calibration values for {teacher}/{view_id}")
            if numeric["a"] <= 0.0:
                raise ValueError(f"Non-positive calibration slope for {teacher}/{view_id}")
            if not (
                numeric["binary_cross_entropy_after"]
                < numeric["binary_cross_entropy_before"]
            ):
                raise ValueError(f"Calibration did not improve NLL for {teacher}/{view_id}")
            if numeric["balanced_accuracy_at_0_5"] < 0.70:
                raise ValueError(f"Calibration BA@0.5 gate failed for {teacher}/{view_id}")
            if not 0.35 <= numeric["positive_rate_at_0_5"] <= 0.65:
                raise ValueError(f"Calibration positive-rate gate failed for {teacher}/{view_id}")
            if not (
                numeric["real_probability_mean"] < 0.5
                < numeric["aigi_probability_mean"]
            ):
                raise ValueError(f"Calibration class-direction gate failed for {teacher}/{view_id}")
            checks[f"{teacher}/{view_id}"] = numeric
    result = {
        "event": "teacher_affine_calibration_validated",
        "path": str(calibration_path),
        "active_teachers": active_teachers,
        "checks": checks,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate V3.2 affine teacher targets")
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    validate(arguments.config)


if __name__ == "__main__":
    main()
