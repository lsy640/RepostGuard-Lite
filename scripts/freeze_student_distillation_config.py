from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from repostguard.checkpoint import atomic_text
from repostguard.config import config_digest, load_config
from repostguard.distillation import validate_distillation_config


def freeze_config(
    config_path: str,
    calibration_path: str,
    output_path: str,
    *,
    overwrite: bool = False,
    m2_weight_override: float | None = None,
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    config = copy.deepcopy(load_config(config_path))
    with Path(calibration_path).expanduser().resolve().open(
        "r", encoding="utf-8"
    ) as handle:
        calibration = json.load(handle)
    temperatures = calibration["temperatures"]
    lineage = calibration["teacher_checkpoint_sha256"]
    config["distillation"]["teacher_calibration_temperatures"] = {
        "m2": float(temperatures["m2"]),
        "m3": float(temperatures["m3"]),
    }
    config["distillation"]["teacher_checkpoint_sha256"] = {
        "m2": str(lineage["m2"]),
        "m3": str(lineage["m3"]),
    }
    selected_mixture = calibration.get("selected_mixture")
    if m2_weight_override is not None:
        m2_weight = float(m2_weight_override)
        m3_weight = 1.0 - m2_weight
    elif selected_mixture is not None:
        m2_weight = float(selected_mixture["m2_weight"])
        m3_weight = float(selected_mixture["m3_weight"])
    else:
        m2_weight = float(config["distillation"]["m2_weight"])
        m3_weight = float(config["distillation"]["m3_weight"])
    config["distillation"]["m2_weight"] = m2_weight
    config["distillation"]["m3_weight"] = m3_weight
    config.pop("_config_path", None)
    validate_distillation_config(config)
    atomic_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), destination
    )
    print(
        json.dumps(
            {
                "event": "student_config_frozen",
                "output": str(destination),
                "config_sha256": config_digest(config),
                "teacher_checkpoint_sha256": config["distillation"][
                    "teacher_checkpoint_sha256"
                ],
                "teacher_calibration_temperatures": config["distillation"][
                    "teacher_calibration_temperatures"
                ],
                "teacher_mixture_weights": {
                    "m2": config["distillation"]["m2_weight"],
                    "m3": config["distillation"]["m3_weight"],
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze calibrated teacher temperatures and lineage into a student config"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--m2-weight",
        type=float,
        help="Optional explicit M2 weight override; M3 is set to 1-M2",
    )
    arguments = parser.parse_args()
    freeze_config(
        arguments.config,
        arguments.calibration,
        arguments.output,
        overwrite=arguments.overwrite,
        m2_weight_override=arguments.m2_weight,
    )


if __name__ == "__main__":
    main()
