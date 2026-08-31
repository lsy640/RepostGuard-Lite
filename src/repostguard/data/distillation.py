from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from repostguard.data.dataset import (
    _seed_worker,
    build_format_debias_config,
    read_manifest,
)
from repostguard.data.transforms import apply_transform, harmonize_image_format, to_model_tensor
from repostguard.distillation import canonical_view_specs

ImageFile.LOAD_TRUNCATED_IMAGES = False


LEGACY_SAMPLING_STRATEGY = "legacy_source_generator_balanced"
HIERARCHICAL_SAMPLING_STRATEGY = "class_arch_generator_hierarchical"
HIERARCHICAL_CLASS_MASS = {"real": 0.5, "aigi": 0.5}
HIERARCHICAL_AIGI_ARCHITECTURE_MASS = {
    "LatDiff": 0.5,
    "GAN": 0.25,
    "PixDiff": 0.25,
}
HIERARCHICAL_GENERATOR_ALPHA = 0.5
HIERARCHICAL_REAL_SOURCE_ALPHA = 0.5


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sha256(config: dict[str, Any]) -> str:
    return sha256_file(config["data"]["train_manifest"])


def teacher_preprocessing_sha256(config: dict[str, Any]) -> str:
    format_debias = build_format_debias_config(config["data"])
    payload = {
        "schema_version": 1,
        "image_size": int(config["data"]["image_size"]),
        "format_debias": {
            "enabled": bool(format_debias.enabled),
            "eval_quality": int(format_debias.eval_quality),
            "jpeg_subsampling": int(format_debias.jpeg_subsampling),
        },
        "views": canonical_view_specs(config),
        "tensor_pipeline": "rgb_bicubic_antialias_to_tensor_0_1_v1",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _row_sha256(row: dict[str, Any]) -> str:
    value = str(row.get("sha256", "")).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(
            f"Distillation manifest row {row['sample_id']} requires a valid sha256 column"
        )
    return value


def _load_rgb(path: str) -> Image.Image:
    with Image.open(path) as image_file:
        if getattr(image_file, "is_animated", False):
            image_file.seek(0)
        return image_file.convert("RGB").copy()


def _base_image(row: dict[str, Any], config: dict[str, Any]) -> Image.Image:
    image = _load_rgb(row["absolute_path"])
    format_debias = build_format_debias_config(config["data"])
    if format_debias.enabled:
        image = harmonize_image_format(
            image,
            int(config["data"]["image_size"]),
            quality=format_debias.quality(training=False),
            jpeg_subsampling=format_debias.jpeg_subsampling,
        )
    return image


def _view_tensor(
    image: Image.Image,
    view: dict[str, Any],
    *,
    image_size: int,
    sample_index: int,
) -> torch.Tensor:
    transformed = apply_transform(
        image,
        str(view["name"]),
        dict(view.get("params", {})),
        seed_offset=sample_index,
    )
    return to_model_tensor(transformed, image_size)


class TeacherViewDataset(Dataset[dict[str, Any]]):
    """Return all deterministic distillation views for teacher inference."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.rows = read_manifest(config["data"]["train_manifest"], config["data"]["root"])
        self.views = canonical_view_specs(config)
        self.image_size = int(config["data"]["image_size"])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = _base_image(row, self.config)
        tensors = [
            _view_tensor(
                image,
                view,
                image_size=self.image_size,
                sample_index=index,
            )
            for view in self.views
        ]
        return {
            "images": torch.stack(tensors),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "sample_id": row["sample_id"],
            "sample_sha256": _row_sha256(row),
        }


class CachedDistillationDataset(Dataset[dict[str, Any]]):
    """Pair images with immutable teacher logits and optional M3 features."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.rows = read_manifest(config["data"]["train_manifest"], config["data"]["root"])
        self.views = canonical_view_specs(config)
        self.image_size = int(config["data"]["image_size"])
        self.augmented_view_count = len(self.views) - 1
        cache_directory = Path(config["distillation"]["cache_directory"]).expanduser().resolve()
        cache_files = sorted(cache_directory.glob("teacher_cache_*-of-*.pt"))
        if not cache_files:
            raise FileNotFoundError(f"No teacher cache shards under {cache_directory}")
        expected_manifest_sha256 = manifest_sha256(config)
        expected_preprocessing_sha256 = teacher_preprocessing_sha256(config)
        require_preprocessing_digest = bool(
            config["distillation"].get("require_preprocessing_digest", False)
        )
        expected_views = json.dumps(self.views, sort_keys=True, separators=(",", ":"))
        feature_config = dict(
            config["distillation"].get("feature_distillation", {})
        )
        self.require_m3_features = bool(feature_config.get("enabled", False))
        self.require_m3_gate = bool(
            feature_config.get("quality_gate_distillation", {}).get(
                "enabled", False
            )
        )
        expected_feature_dim = int(feature_config.get("teacher_dim", 256))
        cache: dict[str, dict[str, Any]] = {}
        teacher_lineage: dict[str, set[str]] = {"m2": set(), "m3": set()}
        shard_feature_presence: set[bool] = set()
        shard_gate_presence: set[bool] = set()
        for cache_file in cache_files:
            payload = torch.load(cache_file, map_location="cpu", weights_only=False)
            metadata = payload["metadata"]
            if metadata["manifest_sha256"] != expected_manifest_sha256:
                raise ValueError(f"Teacher cache manifest mismatch: {cache_file}")
            cached_preprocessing_sha256 = str(
                metadata.get("preprocessing_sha256", "")
            )
            if cached_preprocessing_sha256:
                if cached_preprocessing_sha256 != expected_preprocessing_sha256:
                    raise ValueError(
                        f"Teacher cache preprocessing mismatch: {cache_file}"
                    )
            elif require_preprocessing_digest:
                raise ValueError(
                    f"Teacher cache is missing preprocessing lineage: {cache_file}"
                )
            cached_views = json.dumps(
                metadata["views"], sort_keys=True, separators=(",", ":")
            )
            if cached_views != expected_views:
                raise ValueError(f"Teacher cache view mismatch: {cache_file}")
            for teacher in ("m2", "m3"):
                lineage_key = f"{teacher}_checkpoint_sha256"
                lineage_value = str(metadata.get(lineage_key, ""))
                if len(lineage_value) != 64:
                    raise ValueError(
                        f"Teacher cache is missing {lineage_key}: {cache_file}"
                    )
                teacher_lineage[teacher].add(lineage_value)
            sample_ids = list(payload["sample_ids"])
            sample_hashes = list(payload["sample_sha256"])
            labels = torch.as_tensor(payload["labels"], dtype=torch.float32)
            m2_logits = torch.as_tensor(payload["m2_logits"], dtype=torch.float32)
            m3_logits = torch.as_tensor(payload["m3_logits"], dtype=torch.float32)
            expected_shape = (len(sample_ids), len(self.views))
            if tuple(m2_logits.shape) != expected_shape or tuple(m3_logits.shape) != expected_shape:
                raise ValueError(f"Invalid teacher logit shape in {cache_file}")
            if len(sample_hashes) != len(sample_ids) or labels.numel() != len(sample_ids):
                raise ValueError(f"Invalid teacher cache row counts in {cache_file}")
            feature_keys = {
                "m3_semantic_features": "semantic_features",
                "m3_forensic_features": "forensic_features",
                "m3_fused_features": "fused_features",
            }
            feature_presence = [key in payload for key in feature_keys]
            if any(feature_presence) and not all(feature_presence):
                raise ValueError(f"Incomplete M3 feature targets in {cache_file}")
            has_features = all(feature_presence)
            shard_feature_presence.add(has_features)
            has_gate = "m3_gate_fractions" in payload
            shard_gate_presence.add(has_gate)
            gate_fractions: torch.Tensor | None = None
            if has_gate:
                gate_fractions = torch.as_tensor(
                    payload["m3_gate_fractions"], dtype=torch.float32
                )
                expected_gate_shape = (len(sample_ids), len(self.views), 2)
                if tuple(gate_fractions.shape) != expected_gate_shape:
                    raise ValueError(
                        f"Invalid m3_gate_fractions shape in {cache_file}: "
                        f"{tuple(gate_fractions.shape)} != {expected_gate_shape}"
                    )
            cached_features: dict[str, torch.Tensor] = {}
            if has_features:
                expected_feature_shape = (
                    len(sample_ids),
                    len(self.views),
                    expected_feature_dim,
                )
                for payload_key, cache_key in feature_keys.items():
                    values = torch.as_tensor(payload[payload_key], dtype=torch.float32)
                    if tuple(values.shape) != expected_feature_shape:
                        raise ValueError(
                            f"Invalid {payload_key} shape in {cache_file}: "
                            f"{tuple(values.shape)} != {expected_feature_shape}"
                        )
                    cached_features[cache_key] = values
            for index, sample_id in enumerate(sample_ids):
                if sample_id in cache:
                    raise ValueError(f"Duplicate cached sample_id: {sample_id}")
                cached_row = {
                    "sample_sha256": str(sample_hashes[index]),
                    "label": float(labels[index]),
                    "m2_logits": m2_logits[index],
                    "m3_logits": m3_logits[index],
                }
                for cache_key, values in cached_features.items():
                    cached_row[cache_key] = values[index]
                if gate_fractions is not None:
                    cached_row["gate_fractions"] = gate_fractions[index]
                cache[str(sample_id)] = cached_row
        if len(shard_feature_presence) > 1:
            raise ValueError("Teacher cache shards disagree on M3 feature availability")
        if self.require_m3_features and shard_feature_presence != {True}:
            raise ValueError("Feature distillation requires cached M3 feature targets")
        if len(shard_gate_presence) > 1:
            raise ValueError("Teacher cache shards disagree on M3 quality-gate availability")
        if self.require_m3_gate and shard_gate_presence != {True}:
            raise ValueError(
                "Quality-gate distillation requires cached M3 gate fractions"
            )
        if any(len(values) != 1 for values in teacher_lineage.values()):
            raise ValueError("Teacher cache shards do not share one M2/M3 lineage")
        expected_lineage = config["distillation"].get("teacher_checkpoint_sha256", {})
        for teacher in ("m2", "m3"):
            expected_hash = str(expected_lineage.get(teacher, ""))
            if expected_hash and teacher_lineage[teacher] != {expected_hash}:
                raise ValueError(f"Teacher cache {teacher.upper()} checkpoint mismatch")
        expected_ids = {str(row["sample_id"]) for row in self.rows}
        cached_ids = set(cache)
        if cached_ids != expected_ids:
            missing = sorted(expected_ids.difference(cached_ids))[:5]
            extra = sorted(cached_ids.difference(expected_ids))[:5]
            raise ValueError(f"Teacher cache coverage mismatch; missing={missing}, extra={extra}")
        for row in self.rows:
            cached = cache[str(row["sample_id"])]
            if cached["sample_sha256"] != _row_sha256(row):
                raise ValueError(f"Teacher cache image hash mismatch: {row['sample_id']}")
            if int(cached["label"]) != int(row["label"]):
                raise ValueError(f"Teacher cache label mismatch: {row['sample_id']}")
        self.cache = cache

    def __len__(self) -> int:
        return len(self.rows) * self.augmented_view_count

    def row_index_for_item(self, index: int) -> int:
        return int(index) // self.augmented_view_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index = self.row_index_for_item(index)
        augmented_view_index = 1 + int(index) % self.augmented_view_count
        row = self.rows[row_index]
        cached = self.cache[str(row["sample_id"])]
        image = _base_image(row, self.config)
        clean = _view_tensor(
            image,
            self.views[0],
            image_size=self.image_size,
            sample_index=row_index,
        )
        augmented = _view_tensor(
            image,
            self.views[augmented_view_index],
            image_size=self.image_size,
            sample_index=row_index,
        )
        item = {
            "image": clean,
            "image_aug": augmented,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "sample_id": row["sample_id"],
            "view_id": str(self.views[augmented_view_index]["id"]),
            "architecture": str(row.get("architecture", "")),
            "generator_id": str(row.get("generator_id", "")),
            "teacher_m2_logit_clean": cached["m2_logits"][0].clone(),
            "teacher_m3_logit_clean": cached["m3_logits"][0].clone(),
            "teacher_m2_logit_aug": cached["m2_logits"][augmented_view_index].clone(),
            "teacher_m3_logit_aug": cached["m3_logits"][augmented_view_index].clone(),
        }
        if "semantic_features" in cached:
            for feature_name in ("semantic", "forensic", "fused"):
                values = cached[f"{feature_name}_features"]
                item[f"teacher_m3_{feature_name}_clean"] = values[0].clone()
                item[f"teacher_m3_{feature_name}_aug"] = values[
                    augmented_view_index
                ].clone()
        if "gate_fractions" in cached:
            gate_values = cached["gate_fractions"]
            item["teacher_m3_gate_clean"] = gate_values[0].clone()
            item["teacher_m3_gate_aug"] = gate_values[augmented_view_index].clone()
        return item


def _expanded_balanced_weights(dataset: CachedDistillationDataset) -> torch.Tensor:
    groups = [
        (
            int(row["label"]),
            row["source_dataset"],
            row["generator_id"] if int(row["label"]) == 1 else "real",
        )
        for row in dataset.rows
    ]
    counts = Counter(groups)
    weights: list[float] = []
    for group in groups:
        weights.extend([1.0 / counts[group]] * dataset.augmented_view_count)
    return torch.as_tensor(weights, dtype=torch.double)


def _require_close(
    actual: float,
    expected: float,
    *,
    name: str,
    tolerance: float = 1e-12,
) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise RuntimeError(
            f"Invalid hierarchical sampling mass for {name}: "
            f"{actual:.17g} != {expected:.17g}"
        )


def _hierarchical_sampling_plan(
    dataset: CachedDistillationDataset,
    *,
    generator_alpha: float,
    real_source_alpha: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not math.isfinite(generator_alpha) or generator_alpha < 0.0:
        raise ValueError("distillation.sampling.generator_alpha must be finite and non-negative")
    if not math.isfinite(real_source_alpha) or real_source_alpha < 0.0:
        raise ValueError(
            "distillation.sampling.real_source_alpha must be finite and non-negative"
        )
    augmented_view_count = int(dataset.augmented_view_count)
    if augmented_view_count <= 0:
        raise ValueError("Hierarchical sampling requires at least one augmented view")

    real_row_indices: list[int] = []
    real_source_counts: Counter[str] = Counter()
    aigi_group_counts: Counter[tuple[str, str]] = Counter()
    generator_architectures: defaultdict[str, set[str]] = defaultdict(set)
    aigi_rows_by_architecture: Counter[str] = Counter()
    for row_index, row in enumerate(dataset.rows):
        label = int(row["label"])
        if label == 0:
            source_dataset = str(row.get("source_dataset", "")).strip()
            if not source_dataset:
                raise ValueError(
                    "Hierarchical sampling requires non-empty source_dataset "
                    f"for Real row {row.get('sample_id', row_index)!r}"
                )
            real_row_indices.append(row_index)
            real_source_counts[source_dataset] += 1
            continue
        if label != 1:
            raise ValueError(
                f"Hierarchical sampling requires binary labels, got {label!r} "
                f"for {row.get('sample_id', row_index)!r}"
            )
        architecture = str(row.get("architecture", "")).strip()
        generator_id = str(row.get("generator_id", "")).strip()
        if not architecture or not generator_id:
            raise ValueError(
                "Hierarchical sampling requires non-empty architecture and generator_id "
                f"for AIGI row {row.get('sample_id', row_index)!r}"
            )
        aigi_group_counts[(architecture, generator_id)] += 1
        generator_architectures[generator_id].add(architecture)
        aigi_rows_by_architecture[architecture] += 1

    if not real_row_indices or not aigi_group_counts:
        raise ValueError("Hierarchical sampling requires both Real and AIGI rows")
    observed_architectures = set(aigi_rows_by_architecture)
    expected_architectures = set(HIERARCHICAL_AIGI_ARCHITECTURE_MASS)
    if observed_architectures != expected_architectures:
        missing = sorted(expected_architectures.difference(observed_architectures))
        unexpected = sorted(observed_architectures.difference(expected_architectures))
        raise ValueError(
            "Hierarchical sampling architecture mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )
    inconsistent_generators = sorted(
        generator_id
        for generator_id, architectures in generator_architectures.items()
        if len(architectures) != 1
    )
    if inconsistent_generators:
        raise ValueError(
            "Hierarchical sampling generator_id values cross architectures: "
            f"{inconsistent_generators[:5]}"
        )

    architecture_normalizers: dict[str, float] = {}
    generators_by_architecture: Counter[str] = Counter()
    for (architecture, _generator_id), count in aigi_group_counts.items():
        architecture_normalizers[architecture] = (
            architecture_normalizers.get(architecture, 0.0)
            + float(count) ** generator_alpha
        )
        generators_by_architecture[architecture] += 1

    real_source_normalizer = sum(
        float(count) ** real_source_alpha for count in real_source_counts.values()
    )
    expected_real_source_conditional_mass = {
        source_dataset: float(count) ** real_source_alpha / real_source_normalizer
        for source_dataset, count in sorted(real_source_counts.items())
    }

    row_probability_mass: list[float] = []
    for row in dataset.rows:
        if int(row["label"]) == 0:
            source_dataset = str(row["source_dataset"]).strip()
            row_probability_mass.append(
                HIERARCHICAL_CLASS_MASS["real"]
                * expected_real_source_conditional_mass[source_dataset]
                / real_source_counts[source_dataset]
            )
            continue
        architecture = str(row["architecture"]).strip()
        generator_id = str(row["generator_id"]).strip()
        generator_count = aigi_group_counts[(architecture, generator_id)]
        generator_mass_within_architecture = (
            float(generator_count) ** generator_alpha
            / architecture_normalizers[architecture]
        )
        row_probability_mass.append(
            HIERARCHICAL_CLASS_MASS["aigi"]
            * HIERARCHICAL_AIGI_ARCHITECTURE_MASS[architecture]
            * generator_mass_within_architecture
            / generator_count
        )

    expanded_weights = torch.as_tensor(
        [
            row_mass / augmented_view_count
            for row_mass in row_probability_mass
            for _ in range(augmented_view_count)
        ],
        dtype=torch.double,
    )
    expected_length = len(dataset.rows) * augmented_view_count
    if expanded_weights.numel() != expected_length:
        raise RuntimeError(
            f"Hierarchical sampling produced {expanded_weights.numel()} weights "
            f"for {expected_length} expanded items"
        )
    if not bool(torch.isfinite(expanded_weights).all()) or not bool(
        (expanded_weights > 0.0).all()
    ):
        raise RuntimeError("Hierarchical sampling weights must be finite and positive")

    class_mass = {"real": 0.0, "aigi": 0.0}
    architecture_mass = {
        architecture: 0.0 for architecture in HIERARCHICAL_AIGI_ARCHITECTURE_MASS
    }
    real_source_mass = {source_dataset: 0.0 for source_dataset in real_source_counts}
    view_position_mass = [0.0] * augmented_view_count
    for row_index, row in enumerate(dataset.rows):
        offset = row_index * augmented_view_count
        row_weights = expanded_weights[offset : offset + augmented_view_count]
        for value in row_weights[1:]:
            _require_close(
                float(value),
                float(row_weights[0]),
                name=f"row_{row_index}_augmented_view_equality",
            )
        row_mass = float(row_weights.sum())
        class_key = "aigi" if int(row["label"]) == 1 else "real"
        class_mass[class_key] += row_mass
        if class_key == "aigi":
            architecture_mass[str(row["architecture"]).strip()] += row_mass
        else:
            real_source_mass[str(row["source_dataset"]).strip()] += row_mass
        for view_index, value in enumerate(row_weights):
            view_position_mass[view_index] += float(value)

    _require_close(float(expanded_weights.sum()), 1.0, name="total")
    for class_key, expected_mass in HIERARCHICAL_CLASS_MASS.items():
        _require_close(class_mass[class_key], expected_mass, name=f"class_{class_key}")
    for architecture, conditional_mass in HIERARCHICAL_AIGI_ARCHITECTURE_MASS.items():
        _require_close(
            architecture_mass[architecture],
            HIERARCHICAL_CLASS_MASS["aigi"] * conditional_mass,
            name=f"architecture_{architecture}",
        )
    for source_dataset, conditional_mass in expected_real_source_conditional_mass.items():
        _require_close(
            real_source_mass[source_dataset],
            HIERARCHICAL_CLASS_MASS["real"] * conditional_mass,
            name=f"real_source_{source_dataset}",
        )
    for view_index, actual_mass in enumerate(view_position_mass):
        _require_close(
            actual_mass,
            1.0 / augmented_view_count,
            name=f"augmented_view_position_{view_index}",
        )

    summary: dict[str, Any] = {
        "event": "distillation_sampler",
        "strategy": HIERARCHICAL_SAMPLING_STRATEGY,
        "hierarchy": {
            "real": ["label", "source_dataset"],
            "aigi": ["label", "architecture", "generator_id"],
        },
        "source_dataset_is_flat_grouping_key": False,
        "generator_alpha": generator_alpha,
        "real_source_alpha": real_source_alpha,
        "rows": len(dataset.rows),
        "expanded_items": expected_length,
        "augmented_views_per_row": augmented_view_count,
        "expected_class_mass": dict(HIERARCHICAL_CLASS_MASS),
        "expected_aigi_architecture_conditional_mass": dict(
            HIERARCHICAL_AIGI_ARCHITECTURE_MASS
        ),
        "expected_real_source_conditional_mass": expected_real_source_conditional_mass,
        "expected_total_architecture_mass": {
            architecture: HIERARCHICAL_CLASS_MASS["aigi"] * conditional_mass
            for architecture, conditional_mass in HIERARCHICAL_AIGI_ARCHITECTURE_MASS.items()
        },
        "expected_augmented_view_position_mass": {
            str(index): 1.0 / augmented_view_count
            for index in range(augmented_view_count)
        },
        "observed_rows_by_class": {
            "real": len(real_row_indices),
            "aigi": sum(aigi_group_counts.values()),
        },
        "observed_aigi_rows_by_architecture": {
            architecture: aigi_rows_by_architecture[architecture]
            for architecture in HIERARCHICAL_AIGI_ARCHITECTURE_MASS
        },
        "observed_real_rows_by_source": dict(sorted(real_source_counts.items())),
        "observed_aigi_generators_by_architecture": {
            architecture: generators_by_architecture[architecture]
            for architecture in HIERARCHICAL_AIGI_ARCHITECTURE_MASS
        },
        "weights_sha256": hashlib.sha256(
            expanded_weights.contiguous().numpy().tobytes()
        ).hexdigest(),
    }
    return expanded_weights, summary


def build_distillation_sampling_plan(
    dataset: CachedDistillationDataset,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return expanded-item weights and a deterministic sampler summary.

    The legacy strategy remains the default.  The hierarchical strategy is
    intentionally opt-in because it changes the expected training distribution.
    """

    sampling_config = dict(config["distillation"].get("sampling", {}))
    strategy = str(sampling_config.get("strategy", LEGACY_SAMPLING_STRATEGY))
    if strategy == LEGACY_SAMPLING_STRATEGY:
        return _expanded_balanced_weights(dataset), {
            "event": "distillation_sampler",
            "strategy": LEGACY_SAMPLING_STRATEGY,
        }
    if strategy != HIERARCHICAL_SAMPLING_STRATEGY:
        raise ValueError(f"Unknown distillation sampling strategy: {strategy!r}")
    generator_alpha = float(
        sampling_config.get(
            "generator_alpha",
            sampling_config.get("alpha", HIERARCHICAL_GENERATOR_ALPHA),
        )
    )
    real_source_alpha = float(
        sampling_config.get("real_source_alpha", HIERARCHICAL_REAL_SOURCE_ALPHA)
    )
    return _hierarchical_sampling_plan(
        dataset,
        generator_alpha=generator_alpha,
        real_source_alpha=real_source_alpha,
    )


def build_distillation_train_loader(config: dict[str, Any]) -> DataLoader[dict[str, Any]]:
    dataset = CachedDistillationDataset(config)
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    weights, sampling_summary = build_distillation_sampling_plan(dataset, config)
    if sampling_summary["strategy"] == HIERARCHICAL_SAMPLING_STRATEGY:
        sampling_summary["seed"] = int(config["seed"])
        sampling_summary["replacement"] = True
        sampling_summary["num_samples"] = len(dataset)
        print(json.dumps(sampling_summary, sort_keys=True), flush=True)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    workers = int(config["data"]["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        sampler=sampler,
        num_workers=workers,
        pin_memory=bool(config["data"].get("pin_memory", True)),
        persistent_workers=bool(config["data"].get("persistent_workers", True)) and workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
        drop_last=True,
    )
