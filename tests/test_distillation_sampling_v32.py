from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch.utils.data import WeightedRandomSampler

from repostguard.data.distillation import (
    HIERARCHICAL_SAMPLING_STRATEGY,
    LEGACY_SAMPLING_STRATEGY,
    build_distillation_sampling_plan,
)


@dataclass
class _ExpandedDataset:
    rows: list[dict[str, Any]]
    augmented_view_count: int = 3

    def __len__(self) -> int:
        return len(self.rows) * self.augmented_view_count


def _row(
    sample_id: str,
    *,
    label: int,
    source_dataset: str,
    generator_id: str,
    architecture: str,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "label": label,
        "source_dataset": source_dataset,
        "generator_id": generator_id,
        "architecture": architecture,
    }


def _hierarchical_rows() -> list[dict[str, Any]]:
    rows = [
        _row(
            f"real-{index}",
            label=0,
            source_dataset=f"real-source-{index % 2}",
            generator_id="real",
            architecture="not_applicable",
        )
        for index in range(4)
    ]
    specifications = (
        ("LatDiff", "lat-big", 4),
        ("LatDiff", "lat-small", 1),
        ("GAN", "gan-big", 2),
        ("GAN", "gan-small", 1),
        ("PixDiff", "pix-only", 3),
    )
    for architecture, generator_id, count in specifications:
        for index in range(count):
            rows.append(
                _row(
                    f"{generator_id}-{index}",
                    label=1,
                    source_dataset=f"source-{index % 2}",
                    generator_id=generator_id,
                    architecture=architecture,
                )
            )
    return rows


def _config(
    *,
    hierarchical: bool,
    alpha: float = 0.5,
    real_source_alpha: float = 0.5,
) -> dict[str, Any]:
    distillation: dict[str, Any] = {}
    if hierarchical:
        distillation["sampling"] = {
            "strategy": HIERARCHICAL_SAMPLING_STRATEGY,
            "generator_alpha": alpha,
            "real_source_alpha": real_source_alpha,
        }
    return {"distillation": distillation}


def _row_mass(weights: torch.Tensor, row_index: int, views: int = 3) -> float:
    start = row_index * views
    return float(weights[start : start + views].sum())


def test_hierarchical_sampling_assigns_exact_class_arch_generator_and_view_mass() -> None:
    rows = _hierarchical_rows()
    dataset = _ExpandedDataset(rows)

    weights, summary = build_distillation_sampling_plan(
        dataset,  # type: ignore[arg-type]
        _config(hierarchical=True),
    )

    assert weights.dtype == torch.double
    assert weights.numel() == len(rows) * 3
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-12)
    assert summary["strategy"] == HIERARCHICAL_SAMPLING_STRATEGY
    assert summary["hierarchy"] == {
        "real": ["label", "source_dataset"],
        "aigi": ["label", "architecture", "generator_id"],
    }
    assert summary["source_dataset_is_flat_grouping_key"] is False
    assert summary["generator_alpha"] == 0.5
    assert summary["real_source_alpha"] == 0.5
    assert summary["expected_class_mass"] == {"real": 0.5, "aigi": 0.5}
    assert summary["expected_aigi_architecture_conditional_mass"] == {
        "LatDiff": 0.5,
        "GAN": 0.25,
        "PixDiff": 0.25,
    }
    assert summary["expected_real_source_conditional_mass"] == {
        "real-source-0": 0.5,
        "real-source-1": 0.5,
    }
    assert summary["expected_total_architecture_mass"] == {
        "LatDiff": 0.25,
        "GAN": 0.125,
        "PixDiff": 0.125,
    }
    assert summary["expected_augmented_view_position_mass"] == {
        "0": 1.0 / 3.0,
        "1": 1.0 / 3.0,
        "2": 1.0 / 3.0,
    }

    real_indices = [index for index, row in enumerate(rows) if row["label"] == 0]
    aigi_indices = [index for index, row in enumerate(rows) if row["label"] == 1]
    assert sum(_row_mass(weights, index) for index in real_indices) == pytest.approx(0.5)
    assert sum(_row_mass(weights, index) for index in aigi_indices) == pytest.approx(0.5)

    for architecture, expected_mass in {
        "LatDiff": 0.25,
        "GAN": 0.125,
        "PixDiff": 0.125,
    }.items():
        architecture_indices = [
            index
            for index, row in enumerate(rows)
            if row["label"] == 1 and row["architecture"] == architecture
        ]
        assert sum(
            _row_mass(weights, index) for index in architecture_indices
        ) == pytest.approx(expected_mass)

    generator_mass = {
        generator_id: sum(
            _row_mass(weights, index)
            for index, row in enumerate(rows)
            if row["label"] == 1 and row["generator_id"] == generator_id
        )
        for generator_id in {row["generator_id"] for row in rows if row["label"] == 1}
    }
    assert generator_mass["lat-big"] == pytest.approx(1.0 / 6.0)
    assert generator_mass["lat-small"] == pytest.approx(1.0 / 12.0)
    assert generator_mass["gan-big"] == pytest.approx(
        0.125 * math.sqrt(2.0) / (math.sqrt(2.0) + 1.0)
    )
    assert generator_mass["gan-small"] == pytest.approx(
        0.125 / (math.sqrt(2.0) + 1.0)
    )
    assert generator_mass["pix-only"] == pytest.approx(0.125)

    for row_index in range(len(rows)):
        item_weights = weights[row_index * 3 : (row_index + 1) * 3]
        torch.testing.assert_close(
            item_weights,
            torch.full((3,), float(item_weights[0]), dtype=torch.double),
            rtol=0.0,
            atol=0.0,
        )
    for view_index in range(3):
        assert float(weights[view_index::3].sum()) == pytest.approx(1.0 / 3.0)


def test_hierarchical_sampling_does_not_group_aigi_by_source_dataset() -> None:
    rows = _hierarchical_rows()
    changed_sources = copy.deepcopy(rows)
    for index, row in enumerate(changed_sources):
        if int(row["label"]) == 1:
            row["source_dataset"] = f"unique-source-{index}"

    original_weights, original_summary = build_distillation_sampling_plan(
        _ExpandedDataset(rows),  # type: ignore[arg-type]
        _config(hierarchical=True),
    )
    changed_weights, changed_summary = build_distillation_sampling_plan(
        _ExpandedDataset(changed_sources),  # type: ignore[arg-type]
        _config(hierarchical=True),
    )

    torch.testing.assert_close(original_weights, changed_weights, rtol=0.0, atol=0.0)
    assert original_summary["weights_sha256"] == changed_summary["weights_sha256"]


def test_hierarchical_sampling_uses_sqrt_mass_within_real_sources() -> None:
    rows = _hierarchical_rows()
    real_rows = [row for row in rows if int(row["label"]) == 0]
    real_rows[0]["source_dataset"] = "real-small"
    for row in real_rows[1:]:
        row["source_dataset"] = "real-large"

    weights, summary = build_distillation_sampling_plan(
        _ExpandedDataset(rows),  # type: ignore[arg-type]
        _config(hierarchical=True),
    )
    expected_small_conditional = 1.0 / (1.0 + math.sqrt(3.0))
    expected_large_conditional = math.sqrt(3.0) / (1.0 + math.sqrt(3.0))
    assert summary["expected_real_source_conditional_mass"] == pytest.approx(
        {
            "real-large": expected_large_conditional,
            "real-small": expected_small_conditional,
        }
    )
    small_index = rows.index(real_rows[0])
    large_indices = [rows.index(row) for row in real_rows[1:]]
    assert _row_mass(weights, small_index) == pytest.approx(
        0.5 * expected_small_conditional
    )
    assert sum(_row_mass(weights, index) for index in large_indices) == pytest.approx(
        0.5 * expected_large_conditional
    )


def test_hierarchical_sampler_seed_and_epoch_resume_are_reproducible() -> None:
    weights, _summary = build_distillation_sampling_plan(
        _ExpandedDataset(_hierarchical_rows()),  # type: ignore[arg-type]
        _config(hierarchical=True),
    )
    generator = torch.Generator().manual_seed(640)
    epoch_start_state = generator.get_state()
    original = list(
        WeightedRandomSampler(
            weights,
            num_samples=weights.numel(),
            replacement=True,
            generator=generator,
        )
    )

    resumed_generator = torch.Generator()
    resumed_generator.set_state(epoch_start_state)
    resumed = list(
        WeightedRandomSampler(
            weights,
            num_samples=weights.numel(),
            replacement=True,
            generator=resumed_generator,
        )
    )
    different_seed = list(
        WeightedRandomSampler(
            weights,
            num_samples=weights.numel(),
            replacement=True,
            generator=torch.Generator().manual_seed(641),
        )
    )

    assert resumed == original
    assert resumed[11:] == original[11:]
    assert different_seed != original


def test_legacy_sampling_remains_the_default_and_keeps_exact_old_weights() -> None:
    rows = _hierarchical_rows()
    dataset = _ExpandedDataset(rows)
    weights, summary = build_distillation_sampling_plan(
        dataset,  # type: ignore[arg-type]
        _config(hierarchical=False),
    )
    group_counts: dict[tuple[int, str, str], int] = {}
    groups: list[tuple[int, str, str]] = []
    for row in rows:
        group = (
            int(row["label"]),
            str(row["source_dataset"]),
            str(row["generator_id"]) if int(row["label"]) == 1 else "real",
        )
        groups.append(group)
        group_counts[group] = group_counts.get(group, 0) + 1
    expected = torch.as_tensor(
        [
            1.0 / group_counts[group]
            for group in groups
            for _ in range(dataset.augmented_view_count)
        ],
        dtype=torch.double,
    )

    assert summary["strategy"] == LEGACY_SAMPLING_STRATEGY
    torch.testing.assert_close(weights, expected, rtol=0.0, atol=0.0)


def test_hierarchical_sampling_fails_closed_on_incomplete_architecture_coverage() -> None:
    rows = [row for row in _hierarchical_rows() if row["architecture"] != "PixDiff"]

    with pytest.raises(ValueError, match=r"missing=\['PixDiff'\]"):
        build_distillation_sampling_plan(
            _ExpandedDataset(rows),  # type: ignore[arg-type]
            _config(hierarchical=True),
        )

    with pytest.raises(ValueError, match="generator_alpha must be finite"):
        build_distillation_sampling_plan(
            _ExpandedDataset(_hierarchical_rows()),  # type: ignore[arg-type]
            _config(hierarchical=True, alpha=float("nan")),
        )

    with pytest.raises(ValueError, match="real_source_alpha must be finite"):
        build_distillation_sampling_plan(
            _ExpandedDataset(_hierarchical_rows()),  # type: ignore[arg-type]
            _config(hierarchical=True, real_source_alpha=float("nan")),
        )
