from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from torch import nn

from repostguard.checkpoint import atomic_text, load_checkpoint
from repostguard.config import config_digest, load_config
from repostguard.models import build_model


class LogitsOnly(nn.Module):
    def __init__(self, detector: nn.Module) -> None:
        super().__init__()
        self.detector = detector

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.detector(images)["logits"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torchscript(module: torch.jit.ScriptModule, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.jit.save(module, temporary)
    os.replace(temporary, path)


def export_mobile(
    config_path: str,
    checkpoint_path: str,
    output_directory: str,
    export_format: str,
) -> dict[str, object]:
    config = load_config(config_path)
    if str(config["model"]["experiment"]).lower() != "student_mnv3":
        raise ValueError("Mobile export supports only the distilled Student")
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.get("config_sha256") != config_digest(config):
        raise ValueError("Student checkpoint/config digest mismatch")
    detector = build_model(config, load_pretrained=False)
    detector.load_state_dict(checkpoint["model"], strict=True)
    # Keep the wrapper itself in eval mode as well as its detector.  Some
    # exporters temporarily change and then restore the root module mode.
    wrapper = LogitsOnly(detector.eval()).eval().cpu()
    image_size = int(config["data"]["image_size"])
    example = torch.rand(1, 3, image_size, image_size)
    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, object]] = {}

    if export_format in {"torchscript", "both"}:
        traced = torch.jit.trace(wrapper, example, strict=True)
        traced = torch.jit.freeze(traced.eval())
        with torch.inference_mode():
            torch.testing.assert_close(traced(example), wrapper(example), rtol=1e-4, atol=1e-5)
        torchscript_path = destination / "student_mnv3_fp32.torchscript.pt"
        _atomic_torchscript(traced, torchscript_path)
        artifacts["torchscript"] = {
            "path": str(torchscript_path),
            "bytes": torchscript_path.stat().st_size,
            "sha256": _sha256(torchscript_path),
        }

    if export_format in {"onnx", "both"}:
        try:
            import onnx
        except ImportError as error:
            raise RuntimeError("Install the project mobile extra to export ONNX") from error
        onnx_path = destination / "student_mnv3_fp32.onnx"
        temporary = onnx_path.with_name(f".{onnx_path.name}.{os.getpid()}.tmp")
        torch.onnx.export(
            wrapper,
            example,
            temporary,
            input_names=["images"],
            output_names=["logits"],
            dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=18,
            do_constant_folding=True,
        )
        onnx.checker.check_model(onnx.load(temporary))
        os.replace(temporary, onnx_path)
        artifacts["onnx"] = {
            "path": str(onnx_path),
            "bytes": onnx_path.stat().st_size,
            "sha256": _sha256(onnx_path),
        }

    metadata: dict[str, object] = {
        "schema_version": 1,
        "experiment": "student_mnv3",
        "image_size": image_size,
        "input": "float32 RGB in [0,1], NCHW; normalization is embedded",
        "output": "one binary AIGI logit per image",
        "config_sha256": config_digest(config),
        "checkpoint_sha256": _sha256(Path(checkpoint_path)),
        "artifacts": artifacts,
        "quantized": False,
    }
    atomic_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        destination / "export_metadata.json",
    )
    print(json.dumps(metadata, sort_keys=True), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the accepted Student for mobile runtimes")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument(
        "--format", choices=("torchscript", "onnx", "both"), default="both"
    )
    arguments = parser.parse_args()
    export_mobile(
        arguments.config,
        arguments.checkpoint,
        arguments.output_directory,
        arguments.format,
    )


if __name__ == "__main__":
    main()
