from __future__ import annotations

import pytest

from repostguard.infer import validate_prediction_records


def test_prediction_schema_accepts_sorted_probabilities() -> None:
    validate_prediction_records(
        [
            {"image_path": "a.jpg", "pred": 0.1},
            {"image_path": "nested/b.png", "pred": 0.9},
        ]
    )


def test_prediction_schema_rejects_extra_fields_and_invalid_probability() -> None:
    with pytest.raises(ValueError):
        validate_prediction_records([{"image_path": "a.jpg", "pred": 1.1}])
    with pytest.raises(ValueError):
        validate_prediction_records([{"image_path": "a.jpg", "pred": 0.2, "label": 0}])

