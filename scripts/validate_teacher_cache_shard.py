from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import torch

from repostguard.config import load_config
from repostguard.data.distillation import (
    manifest_sha256,
    sha256_file,
    teacher_preprocessing_sha256,
)
from repostguard.distillation import canonical_view_specs


def _manifest_rows(path: str) -> int:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one immutable teacher-cache shard")
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--num-shards", required=True, type=int)
    parser.add_argument("--output-directory")
    parser.add_argument("--manifest")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if arguments.manifest:
        config = copy.deepcopy(config)
        config["data"]["train_manifest"] = arguments.manifest
    output_directory = Path(
        arguments.output_directory or config["distillation"]["cache_directory"]
    ).expanduser().resolve()
    cache_path = output_directory / (
        f"teacher_cache_{arguments.shard_index:03d}-of-{arguments.num_shards:03d}.pt"
    )
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    rows = _manifest_rows(config["data"]["train_manifest"])
    expected_start = rows * arguments.shard_index // arguments.num_shards
    expected_end = rows * (arguments.shard_index + 1) // arguments.num_shards
    expected_samples = expected_end - expected_start
    views = canonical_view_specs(config)
    sample_ids = list(payload["sample_ids"])
    sample_hashes = list(payload["sample_sha256"])
    labels = torch.as_tensor(payload["labels"])
    m2_logits = torch.as_tensor(payload["m2_logits"])
    m3_logits = torch.as_tensor(payload["m3_logits"])
    expected_shape = (expected_samples, len(views))
    feature_config = dict(config["distillation"].get("feature_distillation", {}))
    require_features = bool(feature_config.get("enabled", False))
    teacher_dim = int(feature_config.get("teacher_dim", 256))
    feature_keys = (
        "m3_semantic_features",
        "m3_forensic_features",
        "m3_fused_features",
    )
    feature_shapes = {
        key: tuple(torch.as_tensor(payload[key]).shape)
        for key in feature_keys
        if key in payload
    }
    expected_feature_shape = (expected_samples, len(views), teacher_dim)
    require_gate = bool(
        feature_config.get("quality_gate_distillation", {}).get("enabled", False)
    )
    gate_shape = (
        tuple(torch.as_tensor(payload["m3_gate_fractions"]).shape)
        if "m3_gate_fractions" in payload
        else None
    )

    checks = {
        "manifest_sha256": metadata["manifest_sha256"] == manifest_sha256(config),
        "preprocessing_sha256": metadata.get("preprocessing_sha256")
        == teacher_preprocessing_sha256(config),
        "views": metadata["views"] == views,
        "shard_index": int(metadata["shard_index"]) == arguments.shard_index,
        "num_shards": int(metadata["num_shards"]) == arguments.num_shards,
        "source_bounds": (
            int(metadata["source_start"]) == expected_start
            and int(metadata["source_end"]) == expected_end
        ),
        "sample_count": len(sample_ids) == expected_samples,
        "unique_sample_ids": len(set(sample_ids)) == len(sample_ids),
        "row_counts": len(sample_hashes) == labels.numel() == len(sample_ids),
        "logit_shapes": (
            tuple(m2_logits.shape) == expected_shape
            and tuple(m3_logits.shape) == expected_shape
        ),
        "m3_feature_shapes": (
            not require_features
            or (
                len(feature_shapes) == len(feature_keys)
                and all(
                    shape == expected_feature_shape
                    for shape in feature_shapes.values()
                )
            )
        ),
        "m3_quality_gate_shape": (
            not require_gate
            or gate_shape == (expected_samples, len(views), 2)
        ),
        "m2_checkpoint_sha256": metadata["m2_checkpoint_sha256"]
        == sha256_file(config["distillation"]["m2_checkpoint"]),
        "m3_checkpoint_sha256": metadata["m3_checkpoint_sha256"]
        == sha256_file(config["distillation"]["m3_checkpoint"]),
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    result = {
        "event": "teacher_cache_shard_validation",
        "cache": str(cache_path),
        "sha256": sha256_file(cache_path),
        "samples": len(sample_ids),
        "views": len(views),
        "checks": checks,
        "accepted": not failures,
        "failures": failures,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
