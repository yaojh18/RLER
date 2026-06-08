"""Custom RM that zeroes the reward for any TRUNCATED sample.

Wire via `--custom-rm-path examples.fully_async.truncation_penalty_rm.custom_rm`.

Slime's stock `async_rm` and `batched_async_rm` both dispatch to the
function pointed to by --custom-rm-path; they call it with EITHER a
single Sample (single-sample path) OR a list of Samples (batched path),
so this function handles both.
"""

from __future__ import annotations

from slime.utils.types import Sample
from slime.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward
from slime.rollout.rm_hub.gpqa import compute_gpqa_reward
from slime.rollout.rm_hub.math_dapo_utils import compute_score as compute_score_dapo
from slime.rollout.rm_hub.math_utils import grade_answer_verl


async def custom_rm(args, sample_or_samples, **kwargs):
    if isinstance(sample_or_samples, list):
        return [_score_one(args, s) for s in sample_or_samples]
    return _score_one(args, sample_or_samples)


def _score_one(args, sample: Sample) -> float:
    # Hard zero for truncated samples — they almost always represent
    # generation loops where the model degenerated rather than reasoning
    # to a real answer, and rule-based scorers happily extract a stray
    # letter from inside the loop and give credit.
    if sample.status == Sample.Status.TRUNCATED:
        return 0.0

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    rm_type = (metadata.get("rm_type") or args.rm_type or "").strip()
    response = sample.response or ""
    label = sample.label

    if rm_type == "gpqa":
        return float(compute_gpqa_reward(response, label, metadata=metadata))
    if rm_type == "math":
        return 1.0 if grade_answer_verl(response, label) else 0.0
    if rm_type == "dapo":
        return float(compute_score_dapo(response, label))
    if rm_type == "deepscaler":
        return float(get_deepscaler_rule_based_reward(response, label))

    raise NotImplementedError(
        f"truncation_penalty_rm: rm_type={rm_type!r} not wired. "
        "Add a branch above if you need another scorer."
    )
