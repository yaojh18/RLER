"""Convert v1 lane-based InstanceRecord -> GRPOExportBundle.

One ExportGroup per ForkGroup (= per Lane A mid_cp). Each group has M=8
ExportSamples, one per Lane B branch. Lane A's tokens are never included.

The token-level prefix invariant + per-turn loss-mask placement is COPIED
verbatim from pds_to_grpo_bundle.py — that logic is the core fix for the
v0 7.6B-loss bug and is independent of the search topology. We re-implement
locally (rather than import) so the v1 path has zero runtime dependency on
v0's bundle module, but the semantics are byte-equal.

Rubric-training bundles (rubric_groups) are NOT generated. True rubric-
model training is a separate design — see docs/rubric_rl.md. The lanes
bundle path here is policy-only; launching with --target=rubric will
yield no training data (empty buffer), fail-loud by design.
"""

from __future__ import annotations

from typing import Any

from swe_agent.contracts import ExportGroup, ExportSample, GRPOExportBundle, has_exact_rollout_tokens
from swe_agent.trajectory_search_parallel import (
    ForkGroup,
    InstanceRecord,
    LaneBBranch,
)
from swe_agent.trajectory_search import _build_step_cards


def _branch_overall_rubric_score(
    branch_index: int, judge_response: dict[str, Any]
) -> float | None:
    """Mean normalized judge score across all rubrics for a branch.
    Returns None when no rubric judged this branch successfully."""
    branch_key = f"branch_{branch_index:02d}"
    scores: list[float] = []
    for _rubric_id, branch_map in (judge_response or {}).items():
        record = (branch_map or {}).get(branch_key)
        if record is None or "error" in record:
            continue
        s = record.get("score_normalized")
        if s is None:
            continue
        scores.append(float(s))
    if not scores:
        return None
    return float(sum(scores) / len(scores))


def _build_branch_sample(
    *,
    instance_id: str,
    group: ForkGroup,
    branch: LaneBBranch,
    steps_per_round: int | None = None,
    gt_only_reward: bool = False,
) -> ExportSample | None:
    """Build one ExportSample from a single Lane B branch.

    Prompt = Lane A messages up to mid_cp (the SHARED parent state across
    all m=8 branches in the group). Turns = the branch's own messages
    (continuation past mid_cp, up to its termination).

    Token-level path: when each asst turn in branch.messages carries the
    sglang-stored prompt_token_ids + token_ids + logprobs, we use the LAST
    asst's prompt + token_ids as the full sequence and place per-turn
    1-mask intervals at the position each a_i's tokens occupy inside
    last.prompt_token_ids + last.token_ids. The HARD ASSERT requires
    a_i.prompt to be a true prefix of last.prompt — guaranteed by the
    custom chat template in swe_agent.tokenization (pure-splice renderer).

    steps_per_round (task 3): if set, truncate the trainable window to the
    first `steps_per_round` ASSISTANT TURNS past the shared parent. Tokens
    emitted after that cutoff are dropped entirely (token_ids ends at the
    last in-window assistant turn's output). Rationale: the rubric judges
    only the first steps_per_round of each branch (Lane C input is the
    round1 truncated trajectory), so training on tokens beyond that point
    has no judged signal — and the GRPO advantage applied to those tokens
    is just noise from the group baseline.
    """
    # Reward = pure rubric (task 4). Earlier we blended alpha*gt +
    # (1-alpha)*rubric; that biased the policy toward whichever signal had
    # higher variance in a given group. With rubric-only, every group's
    # advantage is driven by the judge — which is the signal we actually
    # want the policy to optimize. A missing rubric or invalid branch makes
    # the complete sibling group unusable.
    #
    # gt_only_reward (paired with ParallelSearchConfig.disable_rubric): use
    # the centrally computed branch.gt_score as the reward. Same scale as
    # the naive baseline; each ForkGroup's M branches
    # are GRPO-normalized against each other conditioned on the shared
    # MidCp state.
    import math as _math
    if branch.error is not None or branch.gt_score is None:
        return None
    elif gt_only_reward:
        gt = branch.gt_score
        if gt is None or not _math.isfinite(float(gt)):
            return None
        reward = float(gt)
    else:
        rubric = _branch_overall_rubric_score(
            branch.branch_index, group.judge_response
        )
        if rubric is None or not _math.isfinite(rubric):
            return None
        reward = float(rubric)

    # Parent messages = Lane A's spine messages up to mid_cp.
    # Pulled from the MidCp snapshot we stored at emit time.
    mid_cp = group.mid_cp
    parent_messages = list(
        (mid_cp.snapshot.get("agent", {}).get("state", {}) or {}).get("messages", [])
    )
    branch_messages = list(branch.messages or [])
    if not branch_messages:
        return None

    token_ids: list[int] | None = None
    loss_mask: list[int] | None = None
    response_length: int | None = None
    rollout_logprobs: list[float] | None = None

    branch_assistant_messages = [
        m for m in branch_messages if m.get("role") == "assistant"
    ]
    if not branch_assistant_messages or not all(
        map(has_exact_rollout_tokens, branch_assistant_messages)
    ):
        return None
    branch_assistants_with_tokens = branch_assistant_messages
    # Cap trainable assistant turns to first steps_per_round (task 3).
    # Anything past that cap is dropped from token_ids entirely — we slice
    # `last` to the last in-window assistant turn, so full_token_ids =
    # last_prompt + last_out naturally truncates the sequence.
    if steps_per_round is not None and steps_per_round > 0:
        branch_assistants_with_tokens = branch_assistants_with_tokens[:steps_per_round]
    if branch_assistants_with_tokens:
        last = branch_assistants_with_tokens[-1]
        last_prompt = list(last["prompt_token_ids"])
        last_out = list(last["token_ids"])
        full_token_ids = last_prompt + last_out
        parent_prefix_len = len(
            branch_assistants_with_tokens[0]["prompt_token_ids"]
        )
        response_length = max(0, len(full_token_ids) - parent_prefix_len)
        if response_length == 0:
            return None
        mask = [0] * len(full_token_ids)
        # Dense per-response-token logprob vector aligned with
        # loss_mask[-response_length:]. Non-assistant positions stay 0.0;
        # this is required by build_rollout_samples's
        # `len(lp) == response_length` guard for TIS + rollout_logprob_mean
        # logging. (Previously sparse, which dropped the field silently.)
        dense_lp: list[float] = [0.0] * response_length
        for asst in branch_assistants_with_tokens:
            a_prompt = list(asst["prompt_token_ids"])
            # HARD ASSERT — see pds_to_grpo_bundle for the long form. Any
            # mismatch here means the custom chat template is no longer
            # producing prefix-stable renderings and PG ratio will explode.
            if len(a_prompt) > len(last_prompt):
                raise AssertionError(
                    f"v1 sample {branch.node_id}: a_i.prompt len "
                    f"({len(a_prompt)}) exceeds last.prompt len "
                    f"({len(last_prompt)}) — invariant violated."
                )
            if last_prompt[: len(a_prompt)] != a_prompt:
                div = next(
                    (i for i in range(len(a_prompt))
                     if last_prompt[i] != a_prompt[i]),
                    -1,
                )
                raise AssertionError(
                    f"v1 sample {branch.node_id}: a_i.prompt is NOT a prefix "
                    f"of last.prompt. first_divergence_idx={div}, "
                    f"len(a_i)={len(a_prompt)}, len(last)={len(last_prompt)}. "
                    f"Custom chat template must be broken — fix tokenization.py."
                )
            start = len(a_prompt)
            out_tok = asst["token_ids"]
            end = min(start + len(out_tok), len(mask))
            for k in range(start, end):
                mask[k] = 1
            lp = asst["logprobs"]
            for j in range(end - start):
                local_pos = (start - parent_prefix_len) + j
                if 0 <= local_pos < response_length:
                    dense_lp[local_pos] = float(lp[j])
        token_ids = full_token_ids
        loss_mask = mask
        rollout_logprobs = dense_lp

    # Reflect the steps_per_round cap in the emitted `turns` payload for
    # consistency with the token-level path. The cap is applied as a
    # bound on assistant turns; user/tool messages after the final kept
    # assistant turn are dropped.
    capped_branch_messages_for_turns = branch_messages
    n_trainable_asst_turns = sum(
        1 for m in branch_messages if m.get("role") == "assistant"
    )
    if steps_per_round is not None and steps_per_round > 0:
        asst_count = 0
        cutoff_idx = len(branch_messages)
        for idx, m in enumerate(branch_messages):
            if m.get("role") == "assistant":
                asst_count += 1
                if asst_count >= steps_per_round:
                    cutoff_idx = idx + 1
                    break
        capped_branch_messages_for_turns = branch_messages[:cutoff_idx]
        n_trainable_asst_turns = min(steps_per_round, n_trainable_asst_turns)

    branch_step_cards = _build_step_cards(branch.events, branch.parent_asst_step)
    return ExportSample(
        sample_id=branch.node_id,
        group_id=f"policy-{instance_id}-g{group.group_index:03d}",
        prompt=[
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in parent_messages
        ],
        turns=[
            {"role": m.get("role", "assistant"), "content": m.get("content", "")}
            for m in capped_branch_messages_for_turns
        ],
        reward=float(reward),
        metadata={
            "branch_index": branch.branch_index,
            "group_index": group.group_index,
            "mid_cp_image_tag": mid_cp.image_tag,
            "mid_cp_asst_step": mid_cp.asst_step,
            "terminated_early": branch.terminated_early,
            "raw_gt_score": branch.gt_score,
            "raw_rubric_score": _branch_overall_rubric_score(
                branch.branch_index, group.judge_response
            ),
            "total_tokens": branch.total_tokens,
            "parent_token_count": (len(token_ids) - response_length)
            if token_ids is not None and response_length is not None else None,
            "token_source": "sglang_stored",
            # Per-sample step counts — surfaced for wandb aggregation in
            # collect_lanes_rollout_async.generate_rollout's metrics dict.
            "n_continuation_steps": len(branch_step_cards),
            "n_trainable_asst_turns": n_trainable_asst_turns,
            "steps_per_round_cap": steps_per_round,
            "n_parent_steps": mid_cp.asst_step,
            "n_full_trace_steps": mid_cp.asst_step + len(branch_step_cards),
            # Patch field used by rollout metrics.
            "terminal_patch_len": len(branch.terminal_patch or ""),
        },
        token_ids=token_ids,
        loss_mask=loss_mask,
        response_length=response_length,
        rollout_logprobs=rollout_logprobs,
    )


def fork_group_to_export_group(
    *,
    instance_id: str,
    group: ForkGroup,
    steps_per_round: int | None = None,
    gt_only_reward: bool = False,
) -> ExportGroup | None:
    """Convert one fork-group into an ExportGroup of m=8 samples.

    Used by the streaming on_group_done callback in the slime rollout fn
    — fires per-group as each group completes, without waiting for the
    instance to finish.

    An invalid branch invalidates the complete sibling group so GRPO never
    computes an advantage from a partial group.

    steps_per_round (task 3): cap each branch's trainable window to its
    first steps_per_round assistant turns past the shared parent. Tokens
    after the cap are dropped from token_ids entirely.
    """
    samples: list[ExportSample] = []
    for branch in group.branches:
        s = _build_branch_sample(
            instance_id=instance_id, group=group, branch=branch,
            steps_per_round=steps_per_round,
            gt_only_reward=gt_only_reward,
        )
        if s is None:
            return None
        samples.append(s)
    if not samples:
        return None
    return ExportGroup(
        group_id=samples[0].group_id,
        samples=samples,
        metadata={
            "group_index": group.group_index,
            "mid_cp_idx": group.mid_cp.idx,
            "mid_cp_image_tag": group.mid_cp.image_tag,
            "mid_cp_asst_step": group.mid_cp.asst_step,
            "n_branches": len(group.branches),
            "n_real": len(samples),
        },
    )


def instance_record_to_bundle(
    record: InstanceRecord,
    *,
    steps_per_round: int | None = None,
    gt_only_reward: bool = False,
) -> GRPOExportBundle:
    """Convert one v1 InstanceRecord into a GRPOExportBundle.

    policy_groups: one ExportGroup per ForkGroup (M=8 Lane B siblings
    forked from the same Lane A MidCp). Lane A's trajectory is never
    emitted as a sample. Reward is pure rubric (task 4) and each branch's
    trainable token window is capped at steps_per_round assistant turns
    past the shared parent (task 3, defaults to record.config['steps_per_round']).

    rubric_groups: always empty — true rubric-model training is a
    separate piece of work (see docs/rubric_rl.md)."""
    cap = steps_per_round
    if cap is None:
        try:
            cap = int(record.config.get("steps_per_round")) if record.config else None
        except (TypeError, ValueError):
            cap = None
    policy_groups: list[ExportGroup] = []
    for group in record.groups:
        eg = fork_group_to_export_group(
            instance_id=record.instance_id, group=group, steps_per_round=cap,
            gt_only_reward=gt_only_reward,
        )
        if eg is not None:
            policy_groups.append(eg)
    return GRPOExportBundle(
        instance_id=record.instance_id,
        run_dir=record.run_dir,
        policy_groups=policy_groups,
        rubric_groups=[],
        metadata={
            "config": record.config,
            "reward_recipe": "gt_only" if gt_only_reward else "rubric_only",
            "steps_per_round_cap": cap,
            "completed": record.completed,
            "error": record.error,
            "seconds": record.seconds,
            "num_groups": len(record.groups),
            "num_mid_cps": len(record.lane_a.mid_cps),
            "scheme": "lane_v1",
        },
    )
