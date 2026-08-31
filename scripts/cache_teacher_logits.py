from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import socket
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from repostguard.checkpoint import atomic_torch_save, load_checkpoint
from repostguard.config import config_digest, load_config
from repostguard.data.distillation import (
    TeacherViewDataset,
    manifest_sha256,
    sha256_file,
    teacher_preprocessing_sha256,
)
from repostguard.distillation import canonical_view_specs
from repostguard.models import build_model


def _load_teacher(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    config = load_config(config_path)
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.get("config_sha256") != config_digest(config):
        raise ValueError(f"Teacher checkpoint/config digest mismatch: {checkpoint_path}")
    model = build_model(config, load_pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, config


def cache_teacher_logits(
    student_config_path: str,
    *,
    shard_index: int,
    num_shards: int,
    batch_size: int,
    output_directory_override: str | None = None,
    manifest_override: str | None = None,
    max_samples: int | None = None,
    overwrite: bool = False,
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to cache M2/M3 teacher logits")
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    config = load_config(student_config_path)
    if manifest_override is not None:
        config = copy.deepcopy(config)
        config["data"]["train_manifest"] = str(manifest_override)
    views = canonical_view_specs(config)
    dataset = TeacherViewDataset(config)
    total = len(dataset)
    shard_start = total * shard_index // num_shards
    shard_end = total * (shard_index + 1) // num_shards
    indices = list(range(shard_start, shard_end))
    if max_samples is not None:
        indices = indices[: int(max_samples)]
    if not indices:
        raise ValueError("Selected teacher-cache shard is empty")
    output_directory = Path(
        output_directory_override or config["distillation"]["cache_directory"]
    ).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / (
        f"teacher_cache_{shard_index:03d}-of-{num_shards:03d}.pt"
    )
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)

    device = torch.device("cuda")
    distillation = config["distillation"]
    m2_model, m2_config = _load_teacher(
        str(distillation["m2_config"]), str(distillation["m2_checkpoint"]), device
    )
    m3_model, m3_config = _load_teacher(
        str(distillation["m3_config"]), str(distillation["m3_checkpoint"]), device
    )
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=bool(config["data"].get("pin_memory", True)),
        persistent_workers=False,
        generator=generator,
    )
    amp_enabled = bool(config["train"].get("amp", True))
    sample_ids: list[str] = []
    sample_hashes: list[str] = []
    labels: list[torch.Tensor] = []
    m2_logits: list[torch.Tensor] = []
    m3_logits: list[torch.Tensor] = []
    feature_config = dict(distillation.get("feature_distillation", {}))
    cache_m3_features = bool(feature_config.get("enabled", False))
    gate_config = dict(feature_config.get("quality_gate_distillation", {}))
    cache_m3_gate = bool(gate_config.get("enabled", False))
    m3_semantic_features: list[torch.Tensor] = []
    m3_forensic_features: list[torch.Tensor] = []
    m3_fused_features: list[torch.Tensor] = []
    m3_gate_fractions: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, start=1):
            images = batch["images"]
            batch_samples, view_count = images.shape[:2]
            flattened = images.flatten(0, 1).to(device, non_blocking=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=amp_enabled
            ):
                m2_output = m2_model(flattened)["logits"]
                m3_outputs = m3_model(flattened)
                m3_output = m3_outputs["logits"]
            m2_logits.append(
                m2_output.float().reshape(batch_samples, view_count).cpu().to(torch.float16)
            )
            m3_logits.append(
                m3_output.float().reshape(batch_samples, view_count).cpu().to(torch.float16)
            )
            if cache_m3_features:
                required_features = {
                    "semantic": "semantic_features",
                    "forensic": "forensic_features",
                    "fused": "features",
                }
                missing = sorted(
                    output_key
                    for output_key in required_features.values()
                    if output_key not in m3_outputs
                )
                if missing:
                    raise KeyError(f"M3 output is missing feature targets: {missing}")
                feature_lists = {
                    "semantic": m3_semantic_features,
                    "forensic": m3_forensic_features,
                    "fused": m3_fused_features,
                }
                for feature_name, output_key in required_features.items():
                    values = m3_outputs[output_key].float()
                    feature_lists[feature_name].append(
                        values.reshape(batch_samples, view_count, -1)
                        .cpu()
                        .to(torch.float16)
                    )
            if cache_m3_gate:
                if "gate_fractions" not in m3_outputs:
                    raise KeyError("M3 output is missing gate_fractions")
                m3_gate_fractions.append(
                    m3_outputs["gate_fractions"]
                    .float()
                    .reshape(batch_samples, view_count, 2)
                    .cpu()
                    .to(torch.float16)
                )
            labels.append(batch["label"].float().cpu())
            sample_ids.extend(str(value) for value in batch["sample_id"])
            sample_hashes.extend(str(value) for value in batch["sample_sha256"])
            if batch_number % 25 == 0:
                print(
                    json.dumps(
                        {
                            "event": "teacher_cache",
                            "batches": batch_number,
                            "samples": len(sample_ids),
                            "shard_index": shard_index,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    payload = {
        "metadata": {
            "schema_version": 4 if cache_m3_gate else (3 if cache_m3_features else 2),
            "student_config": str(Path(student_config_path).resolve()),
            "manifest": str(Path(config["data"]["train_manifest"]).resolve()),
            "manifest_sha256": manifest_sha256(config),
            "preprocessing_sha256": teacher_preprocessing_sha256(config),
            "views": views,
            "shard_index": int(shard_index),
            "num_shards": int(num_shards),
            "source_start": int(shard_start),
            "source_end": int(shard_end),
            "cached_samples": len(sample_ids),
            "m2_config_sha256": config_digest(m2_config),
            "m3_config_sha256": config_digest(m3_config),
            "m2_checkpoint_sha256": sha256_file(distillation["m2_checkpoint"]),
            "m3_checkpoint_sha256": sha256_file(distillation["m3_checkpoint"]),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "m3_feature_schema": (
                {
                    "semantic": int(m3_semantic_features[0].shape[-1]),
                    "forensic": int(m3_forensic_features[0].shape[-1]),
                    "fused": int(m3_fused_features[0].shape[-1]),
                    "dtype": "float16",
                }
                if cache_m3_features
                else None
            ),
            "m3_quality_gate_schema": (
                {"fractions": 2, "dtype": "float16"}
                if cache_m3_gate
                else None
            ),
        },
        "sample_ids": sample_ids,
        "sample_sha256": sample_hashes,
        "labels": torch.cat(labels),
        "m2_logits": torch.cat(m2_logits),
        "m3_logits": torch.cat(m3_logits),
    }
    if cache_m3_features:
        payload.update(
            {
                "m3_semantic_features": torch.cat(m3_semantic_features),
                "m3_forensic_features": torch.cat(m3_forensic_features),
                "m3_fused_features": torch.cat(m3_fused_features),
            }
        )
    if cache_m3_gate:
        payload["m3_gate_fractions"] = torch.cat(m3_gate_fractions)
    atomic_torch_save(payload, output_path)
    print(
        json.dumps(
            {
                "event": "teacher_cache_complete",
                "output": str(output_path),
                "samples": len(sample_ids),
                "views": len(views),
                "m3_features": cache_m3_features,
                "m3_quality_gate": cache_m3_gate,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache deterministic M2/M3 teacher logits")
    parser.add_argument("--config", required=True, help="Student distillation config")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-directory")
    parser.add_argument("--manifest")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    cache_teacher_logits(
        arguments.config,
        shard_index=arguments.shard_index,
        num_shards=arguments.num_shards,
        batch_size=arguments.batch_size,
        output_directory_override=arguments.output_directory,
        manifest_override=arguments.manifest,
        max_samples=arguments.max_samples,
        overwrite=arguments.overwrite,
    )


if __name__ == "__main__":
    main()
