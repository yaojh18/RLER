"""Length-penalized boxed-letter scorer for HLE MC.

Base reward
-----------
1.0 if the last ``\\boxed{LETTER}`` in the response matches the ground-truth
label (case-insensitive), else 0.0. Truncated samples without an extractable
boxed answer get 0.0.

Length penalty (multiplier on base reward)
------------------------------------------
- ``response_length <= 32768``: 1.0 (no penalty)
- ``response_length >= 40960``: 0.0 (full penalty -> reward is 0)
- in between:  ``(40960 - response_length) / (40960 - 32768)`` (linear)

Final reward = ``base_reward * multiplier``. We multiply rather than subtract
so a wrong answer never sneaks above 0 just because it was short.

Wire via ``--custom-rm-path length_penalty_boxed_rm.custom_rm``.
"""

from __future__ import annotations

import re

from slime.utils.types import Sample

_BOXED_LETTER_RE = re.compile(r"\\boxed\{\s*([A-Za-z])\s*\}")

LEN_FULL = 32768  # reward unaffected at or below this token count
LEN_ZERO = 40960  # reward zeroed at or above this token count


async def custom_rm(args, sample_or_samples, **kwargs):
    if isinstance(sample_or_samples, list):
        return [_score_one(s) for s in sample_or_samples]
    return _score_one(sample_or_samples)


def _score_one(sample: Sample) -> float:
    base = _base_reward(sample)
    if base == 0.0:
        return 0.0
    return base * _length_mult(sample.response_length)


def _base_reward(sample: Sample) -> float:
    label = sample.label
    if not isinstance(label, str) or len(label) != 1:
        return 0.0
    label_letter = label.strip().upper()
    response = sample.response or ""
    matches = _BOXED_LETTER_RE.findall(response)
    if not matches:
        return 0.0
    return 1.0 if matches[-1].upper() == label_letter else 0.0


def _length_mult(rlen: int) -> float:
    if rlen <= LEN_FULL:
        return 1.0
    if rlen >= LEN_ZERO:
        return 0.0
    return (LEN_ZERO - rlen) / (LEN_ZERO - LEN_FULL)
