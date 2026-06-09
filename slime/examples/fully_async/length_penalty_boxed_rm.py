"""DAPO-style Soft Overlong Punishment for HLE MC.

Reference: DAPO paper, Eq. (13) -- Soft Overlong Punishment.

Base reward
-----------
1.0 if the last ``\\boxed{LETTER}`` in the response matches the ground-truth
label (case-insensitive), else 0.0.

Length penalty (added to base, NOT multiplied)
----------------------------------------------
Let ``Lmax = 40960`` and ``Lcache = 8192`` (so ``Lmax - Lcache = 32768``).
``R_length(|y|) =``

  - 0,                               if ``|y| <= Lmax - Lcache``  (= 32768)
  - ``(Lmax - Lcache - |y|) / Lcache``,  if ``Lmax - Lcache < |y| <= Lmax``
  - -1,                              if ``Lmax < |y|``

Final reward = ``base + R_length(|y|)``.

Semantics (with Lmax=40960, Lcache=8192):
  short correct  (|y|<=32K)  -> base=1 + 0    =  1.00
  short wrong               -> base=0 + 0    =  0.00
  edge correct (|y|=40K)    -> base=1 + (-1) =  0.00
  edge wrong   (|y|=40K)    -> base=0 + (-1) = -1.00
  >40K correct/wrong         -> base + (-1)   = (negative or 0)

Wrong answers can therefore go negative; the linear region punishes both
correctness classes but never pushes a correct answer below 0 (since
R_length >= -1 and base = 1 at the edge).

Wire via ``--custom-rm-path length_penalty_boxed_rm.custom_rm``.
"""

from __future__ import annotations

import re

from slime.utils.types import Sample

_BOXED_LETTER_RE = re.compile(r"\\boxed\{\s*([A-Za-z])\s*\}")

LMAX = 40960   # response length at/above which R_length is clamped to -1
LCACHE = 8192  # width of the soft-penalty interval


async def custom_rm(args, sample_or_samples, **kwargs):
    if isinstance(sample_or_samples, list):
        return [_score_one(s) for s in sample_or_samples]
    return _score_one(sample_or_samples)


def _score_one(sample: Sample) -> float:
    return _base_reward(sample) + _length_penalty(sample.response_length)


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


def _length_penalty(rlen: int) -> float:
    threshold = LMAX - LCACHE  # = 32768
    if rlen <= threshold:
        return 0.0
    if rlen > LMAX:
        return -1.0
    # Linear ramp from 0 (at threshold) to -1 (at LMAX).
    return (threshold - rlen) / LCACHE
