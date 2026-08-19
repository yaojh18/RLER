from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def has_exact_rollout_tokens(message: dict[str, Any]) -> bool:
    prompt_token_ids = message.get("prompt_token_ids")
    token_ids = message.get("token_ids")
    logprobs = message.get("logprobs")
    if not all(isinstance(values, list) and values for values in (prompt_token_ids, token_ids, logprobs)):
        return False
    if len(logprobs) != len(token_ids):
        return False
    try:
        return all(isinstance(token, int) for token in prompt_token_ids + token_ids) and all(
            math.isfinite(float(logprob)) for logprob in logprobs
        )
    except (TypeError, ValueError):
        return False


@dataclass
class ExportSample:
    sample_id: str
    group_id: str
    prompt: list[dict[str, Any]]
    turns: list[dict[str, Any]]
    reward: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # ---- exact rollout token fields (required for training) ----
    # Slime consumes these directly; text re-tokenization is not supported.
    #   token_ids       = full sequence (parent's shared prefix + this branch's
    #                     new messages, encoded with the same chat template
    #                     used by sglang)
    #   loss_mask       = same length as token_ids; 1 for tokens that should
    #                     contribute to the policy gradient, 0 otherwise.
    #                     Crucially this is 0 over the parent's shared prefix
    #                     so M=8 siblings don't repeatedly train on the same
    #                     parent tokens, AND 0 over this branch's user/tool
    #                     tokens (only assistant generations are trainable).
    #   response_length = number of branch's NEW tokens (= len(token_ids) -
    #                     len(parent_prefix)). slime takes
    #                     loss_mask[-response_length:] as the per-token
    #                     advantage mask.
    #   rollout_logprobs= dense response-aligned logprobs from sglang; masked
    #                     non-assistant positions contain zero.
    token_ids: list[int] | None = None
    loss_mask: list[int] | None = None
    response_length: int | None = None
    rollout_logprobs: list[float] | None = None


@dataclass
class ExportGroup:
    group_id: str
    samples: list[ExportSample]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GRPOExportBundle:
    instance_id: str
    run_dir: str
    policy_groups: list[ExportGroup]
    rubric_groups: list[ExportGroup]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            "GRPOExportBundle("
            f"instance_id={self.instance_id!r}, "
            f"policy_groups={len(self.policy_groups)}, "
            f"rubric_groups={len(self.rubric_groups)}, "
            f"run_dir={self.run_dir!r})"
        )
