from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from repostguard.checkpoint import load_checkpoint
from repostguard.config import config_digest, load_config
from repostguard.models import build_model


# CPU convolution kernels used by PyTorch and ONNX Runtime can accumulate in a
# different order.  Gate both the raw decision value and the user-facing
# probability: the logit cap catches large high-confidence drift, while the
# probability cap stays strict around the 0.5 decision boundary.
MAX_LOGIT_ABS_ERROR = 5e-3
MAX_PROBABILITY_ABS_ERROR = 1e-3


class LogitsOnly(nn.Module):
    def __init__(self, detector: nn.Module) -> None:
        super().__init__()
        self.detector = detector

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.detector(images)["logits"]


def validate(config_path: str, checkpoint_path: str, onnx_path: str) -> dict[str, object]:
    config = load_config(config_path)
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.get("config_sha256") != config_digest(config):
        raise ValueError("Student checkpoint/config digest mismatch")

    detector = build_model(config, load_pretrained=False)
    detector.load_state_dict(checkpoint["model"], strict=True)
    reference = LogitsOnly(detector.eval()).cpu()
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = max(1, min(4, torch.get_num_threads()))
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(Path(onnx_path).expanduser().resolve()),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    if [item.name for item in session.get_inputs()] != ["images"]:
        raise ValueError("Unexpected ONNX input contract")
    if [item.name for item in session.get_outputs()] != ["logits"]:
        raise ValueError("Unexpected ONNX output contract")

    image_size = int(config["data"]["image_size"])
    generator = torch.Generator(device="cpu").manual_seed(20260830)
    cases = {
        "zeros": torch.zeros(1, 3, image_size, image_size),
        "ones": torch.ones(1, 3, image_size, image_size),
        "random_batch_3": torch.rand(
            3, 3, image_size, image_size, generator=generator
        ),
    }
    results: dict[str, dict[str, float | list[int]]] = {}
    worst_logit_abs_error = 0.0
    worst_probability_abs_error = 0.0
    with torch.inference_mode():
        for name, tensor in cases.items():
            expected = reference(tensor).numpy()
            actual = session.run(["logits"], {"images": tensor.numpy()})[0]
            if actual.shape != expected.shape:
                raise AssertionError(
                    f"ONNX output shape mismatch for {name}: "
                    f"actual={actual.shape}, expected={expected.shape}"
                )
            if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                raise AssertionError(f"Non-finite parity output for {name}")
            absolute_error = np.abs(actual - expected)
            denominator = np.maximum(np.abs(expected), 1e-8)
            expected_probability = 1.0 / (1.0 + np.exp(-expected))
            actual_probability = 1.0 / (1.0 + np.exp(-actual))
            probability_error = np.abs(actual_probability - expected_probability)
            case_logit_abs_error = float(np.max(absolute_error))
            case_probability_abs_error = float(np.max(probability_error))
            worst_logit_abs_error = max(worst_logit_abs_error, case_logit_abs_error)
            worst_probability_abs_error = max(
                worst_probability_abs_error, case_probability_abs_error
            )
            results[name] = {
                "shape": list(actual.shape),
                "max_abs_error": case_logit_abs_error,
                "max_rel_error": float(
                    np.max(absolute_error / denominator)
                ),
                "max_probability_abs_error": case_probability_abs_error,
            }

    thresholds = {
        "max_logit_abs_error": MAX_LOGIT_ABS_ERROR,
        "max_probability_abs_error": MAX_PROBABILITY_ABS_ERROR,
    }
    observed = {
        "max_logit_abs_error": worst_logit_abs_error,
        "max_probability_abs_error": worst_probability_abs_error,
    }
    if (
        worst_logit_abs_error > MAX_LOGIT_ABS_ERROR
        or worst_probability_abs_error > MAX_PROBABILITY_ABS_ERROR
    ):
        raise AssertionError(
            "ONNX parity thresholds exceeded: "
            + json.dumps(
                {"observed": observed, "thresholds": thresholds, "cases": results},
                sort_keys=True,
            )
        )

    report: dict[str, object] = {
        "status": "passed",
        "onnx": str(Path(onnx_path).expanduser().resolve()),
        "providers": session.get_providers(),
        "thresholds": thresholds,
        "observed": observed,
        "cases": results,
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Student ONNX parity")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", required=True)
    arguments = parser.parse_args()
    validate(arguments.config, arguments.checkpoint, arguments.onnx)


if __name__ == "__main__":
    main()
