"""Frozen zero-variance detector used by direct-rubric training."""

from __future__ import annotations

import copy
from typing import Any


VARIANCE_DETECTOR: dict[str, Any] = {
    "schema_version": "handbook_zero_variance_detector.v2",
    "feature": "weighted_rubric_range_mean",
    "fusion": {"rule": "explicit_abstain_or_numeric_abstain"},
    "score_mapping": [1, 2, 4, 6, 8],
    "threshold": 0.18,
    "threshold_decimal_places": 2,
    "variance_direction": "ge",
}


def load_variance_detector() -> dict[str, Any]:
    """Return an isolated copy of the frozen training detector contract."""

    return copy.deepcopy(VARIANCE_DETECTOR)
