from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.select_student_v31_ablation import _group_metrics


def test_group_metrics_use_a_shared_real_pool_and_architecture_macro(
    tmp_path: Path,
) -> None:
    groups = {
        "real-1": {"label": "0", "architecture": "Real", "generator": "real"},
        "real-2": {"label": "0", "architecture": "Real", "generator": "real"},
        "lat": {"label": "1", "architecture": "LatDiff", "generator": "lat-g"},
        "gan": {"label": "1", "architecture": "GAN", "generator": "gan-g"},
        "pix": {"label": "1", "architecture": "PixDiff", "generator": "pix-g"},
    }
    path = tmp_path / "predictions.jsonl"
    rows = []
    for transform in ("clean", "jpeg50"):
        scores = {"real-1": 0.1, "real-2": 0.2, "lat": 0.9, "gan": 0.8, "pix": 0.7}
        for sample_id, score in scores.items():
            rows.append(
                {
                    "sample_id": sample_id,
                    "label": int(groups[sample_id]["label"]),
                    "pred": score,
                    "transform": transform,
                }
            )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    metrics = _group_metrics(path, groups)

    assert metrics["clean_architecture_macro_auroc"] == pytest.approx(1.0)
    assert metrics["clean_architecture_worst_auroc"] == pytest.approx(1.0)
    assert metrics["robust_architecture_macro_auroc"] == pytest.approx(1.0)
    assert metrics["robust_worst_architecture_condition_auroc"] == pytest.approx(1.0)
    assert metrics["robust_hierarchical_generator_macro_auroc"] == pytest.approx(1.0)
