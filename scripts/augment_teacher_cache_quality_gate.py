from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from repostguard.checkpoint import atomic_torch_save, load_checkpoint
from repostguard.config import load_config
from repostguard.data.distillation import TeacherViewDataset, sha256_file
from repostguard.models.quality_gate import QualityAwareGate


def _load_m3_quality_gate(
    config: dict[str, Any], device: torch.device
) -> tuple[QualityAwareGate, str]:
    distillation = config["distillation"]
    checkpoint_path = str(distillation["m3_checkpoint"])
    checkpoint_sha256 = sha256_file(checkpoint_path)
    expected_sha256 = str(
        distillation.get("teacher_checkpoint_sha256", {}).get("m3", "")
    )
    if expected_sha256 and checkpoint_sha256 != expected_sha256:
        raise ValueError("M3 checkpoint SHA256 does not match the Student config")
    m3_config = load_config(str(distillation["m3_config"]))
    hidden_dim = int(m3_config["model"]["quality_gate"]["hidden_dim"])
    gate = QualityAwareGate(hidden_dim=hidden_dim)
    checkpoint = load_checkpoint(checkpoint_path)
    prefix = "quality_gate."
    gate_state = {
        key[len(prefix) :]: value
        for key, value in checkpoint["model"].items()
        if key.startswith(prefix)
    }
    if not gate_state:
        raise KeyError("M3 checkpoint contains no quality_gate state")
    gate.load_state_dict(gate_state, strict=True)
    return gate.to(device).eval(), checkpoint_sha256


def augment_cache(
    config_path: str,
    source_directory: str,
    output_directory: str,
    *,
    batch_size: int,
    overwrite: bool = False,
) -> list[Path]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    config = load_config(config_path)
    dataset = TeacherViewDataset(config)
    views = dataset.views
    source_root = Path(source_directory).expanduser().resolve()
    destination_root = Path(output_directory).expanduser().resolve()
    source_files = sorted(source_root.glob("teacher_cache_*-of-*.pt"))
    if not source_files:
        raise FileNotFoundError(f"No teacher cache shards under {source_root}")
    if source_root == destination_root:
        raise ValueError("Quality-gate cache augmentation requires a new output directory")
    destination_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gate, checkpoint_sha256 = _load_m3_quality_gate(config, device)
    outputs: list[Path] = []
    for source_path in source_files:
        destination = destination_root / source_path.name
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        if "m3_gate_fractions" in payload and not overwrite:
            raise ValueError(f"Source cache already has M3 gate fractions: {source_path}")
        metadata = payload["metadata"]
        if str(metadata["m3_checkpoint_sha256"]) != checkpoint_sha256:
            raise ValueError(f"M3 cache/checkpoint lineage mismatch: {source_path}")
        if metadata["views"] != views:
            raise ValueError(f"M3 cache/view mismatch: {source_path}")
        source_start = int(metadata["source_start"])
        source_end = int(metadata["source_end"])
        sample_count = len(payload["sample_ids"])
        indices = list(range(source_start, source_end))[:sample_count]
        expected_ids = [dataset.rows[index]["sample_id"] for index in indices]
        if expected_ids != list(payload["sample_ids"]):
            raise ValueError(f"M3 cache row ordering mismatch: {source_path}")

        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(config["data"]["num_workers"]),
            pin_memory=bool(config["data"].get("pin_memory", True)),
            persistent_workers=False,
        )
        fractions: list[torch.Tensor] = []
        with torch.inference_mode():
            for batch_number, batch in enumerate(loader, start=1):
                images = batch["images"]
                batch_samples, view_count = images.shape[:2]
                flattened = images.flatten(0, 1).to(device, non_blocking=True)
                gate_fractions, _quality_features = gate(flattened)
                fractions.append(
                    gate_fractions.reshape(batch_samples, view_count, 2)
                    .cpu()
                    .to(torch.float16)
                )
                if batch_number % 50 == 0:
                    print(
                        json.dumps(
                            {
                                "event": "quality_gate_cache",
                                "source": source_path.name,
                                "batches": batch_number,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        merged = torch.cat(fractions)
        expected_shape = (sample_count, len(views), 2)
        if tuple(merged.shape) != expected_shape:
            raise RuntimeError(
                f"Quality-gate cache shape {tuple(merged.shape)} != {expected_shape}"
            )

        output_payload = dict(payload)
        output_payload["m3_gate_fractions"] = merged
        output_metadata = copy.deepcopy(metadata)
        output_metadata["schema_version"] = max(
            4, int(output_metadata.get("schema_version", 0))
        )
        output_metadata["m3_quality_gate_schema"] = {
            "fractions": 2,
            "dtype": "float16",
            "source": "m3_quality_gate_only_v1",
        }
        output_metadata["quality_gate_checkpoint_sha256"] = checkpoint_sha256
        output_payload["metadata"] = output_metadata
        atomic_torch_save(output_payload, destination)
        outputs.append(destination)
        print(
            json.dumps(
                {
                    "event": "quality_gate_cache_complete",
                    "source": str(source_path),
                    "output": str(destination),
                    "samples": sample_count,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add M3 quality-gate targets to an existing feature cache"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    augment_cache(
        arguments.config,
        arguments.source_directory,
        arguments.output_directory,
        batch_size=arguments.batch_size,
        overwrite=arguments.overwrite,
    )


if __name__ == "__main__":
    main()
