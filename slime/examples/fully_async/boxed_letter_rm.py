"""Strict-format multi-choice scorer.

Reward = 1.0 iff the response contains exactly one `\\boxed{X}` whose letter
matches the ground-truth label (case-insensitive). Everything else — no box,
malformed box, wrong letter, or truncated sample — gets 0.0.

Wire via `--custom-rm-path boxed_letter_rm.custom_rm`.

This replaces `truncation_penalty_rm.custom_rm` because the stock gpqa scorer
had two reward-hack paths (`<|im_end|>` normalizing to "im end" which contains
"d/e/i/m/n"; single-letter label being substring-matched against any word).
After the truncation penalty was added the policy collapsed to 2–3 token
"stub" responses that exploited the substring fallback. The strict boxed
regex closes both paths.
"""

from __future__ import annotations

import re

from slime.utils.types import Sample

# Last `\boxed{X}` in the response. We take the last match because the model
# may reason about candidate answers in earlier `\boxed{...}` blocks before
# committing.
_BOXED_LETTER_RE = re.compile(r"\\boxed\{\s*([A-Za-z])\s*\}")


async def custom_rm(args, sample_or_samples, **kwargs):
    if isinstance(sample_or_samples, list):
        return [_score_one(s) for s in sample_or_samples]
    return _score_one(sample_or_samples)


def _score_one(sample: Sample) -> float:
    # Truncated samples (sglang finish_reason="length") get 0 reward — they
    # didn't reach a `\boxed{}` and we don't want to reward incidental letter
    # matches inside their cut-off reasoning.
    if sample.status == Sample.Status.TRUNCATED:
        return 0.0

    label = sample.label
    if not isinstance(label, str) or len(label) != 1:
        return 0.0
    label_letter = label.strip().upper()

    response = sample.response or ""
    matches = _BOXED_LETTER_RE.findall(response)
    if not matches:
        return 0.0
    extracted = matches[-1].upper()
    return 1.0 if extracted == label_letter else 0.0
