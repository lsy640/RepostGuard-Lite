from __future__ import annotations

import io

import numpy as np
from PIL import Image

from repostguard.data.community_forensics import (
    _allocate_proportional,
    canonical_generator_id,
    canonical_real_source,
    classify_external_eval_generators,
    perceptual_hash,
    phash_distance,
)
from repostguard.data.community_forensics_validation_v2 import (
    EXACT_SEEN_GENERATOR_ID,
    HARD_GENERATORS,
    TARGETS,
    _candidate_sort_key,
)


def _image(offset: int = 0) -> bytes:
    values = (np.arange(64 * 64 * 3, dtype=np.uint16) + offset) % 256
    image = Image.fromarray(values.astype(np.uint8).reshape(64, 64, 3), mode="RGB")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_generator_normalization_preserves_exact_identity_boundaries() -> None:
    assert canonical_generator_id(" StabilityAI/Stable_Diffusion 2 ") == (
        "stabilityai/stable-diffusion-2"
    )
    assert canonical_generator_id("author/model-a") != canonical_generator_id(
        "author/model-b"
    )


def test_real_source_normalization() -> None:
    assert canonical_real_source("landscapesHQ") == "Landscapes HQ"
    assert canonical_real_source("coco") == "COCO"
    assert canonical_real_source("RAISE") == "RAISE"
    assert canonical_real_source("unknown") == ""


def test_proportional_allocation_is_exact_and_bounded() -> None:
    allocation = _allocate_proportional({"a": 7, "b": 2, "c": 1}, 6)
    assert sum(allocation.values()) == 6
    assert all(allocation[key] <= limit for key, limit in {"a": 7, "b": 2, "c": 1}.items())


def test_external_generator_cohorts_require_exact_and_family_boundaries() -> None:
    seen_family, family_unseen = classify_external_eval_generators(
        {"train-exact", "small-val-exact", "same-family-new", "new-family-new"},
        {
            "train-exact": "LatDiff",
            "small-val-exact": "LatDiff",
            "same-family-new": "LatDiff",
            "new-family-new": "Commercial",
        },
        {"train-exact"},
        {"small-val-exact"},
        {"LatDiff"},
    )
    assert seen_family == {"same-family-new"}
    assert family_unseen == {"new-family-new"}


def test_validation_v2_protocol_constants_are_exact_and_balanced() -> None:
    assert canonical_generator_id("CompVis/stable-diffusion-v1-4") == (
        EXACT_SEEN_GENERATOR_ID
    )
    assert HARD_GENERATORS == ("hourglass", "dfgan", "galip")
    assert TARGETS["exact_real"] == TARGETS["exact_sd14"] == 1_000
    assert all(TARGETS[f"hard_{generator}"] == 250 for generator in HARD_GENERATORS)


def test_validation_v2_selection_reuses_aigibench_row_groups() -> None:
    common = {
        "source_key": "aigibench",
        "source_file": "data/validation-00000.parquet",
    }
    real = {**common, "selection_group": "exact_real", "row_group": 2, "row_index": 1}
    fake = {**common, "selection_group": "exact_sd14", "row_group": 2, "row_index": 8}
    other = {**common, "selection_group": "exact_sd14", "row_group": 3, "row_index": 1}
    assert _candidate_sort_key(real)[0] == _candidate_sort_key(fake)[0]
    assert _candidate_sort_key(fake)[0] != _candidate_sort_key(other)[0]


def test_perceptual_hash_is_deterministic() -> None:
    first = perceptual_hash(_image())
    second = perceptual_hash(_image())
    changed = perceptual_hash(_image(17))
    assert len(first) == 16
    assert first == second
    assert 0 <= phash_distance(first, changed) <= 64
