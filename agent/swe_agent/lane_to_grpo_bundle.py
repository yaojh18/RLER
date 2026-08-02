"""Convert lane-based training records to atomic GRPO groups.

Both root and beam groups contain exactly ``m`` samples.  In a beam sample the
common prompt is still the root system/user prefix; the corresponding Lane-A
parent and Lane-B child are one trainable response.

The token-level prefix invariant and per-turn loss-mask placement match the
established PDS exporter and are independent of search topology.

Rubric-training bundles (rubric_groups) are NOT generated. True rubric-
model training is a separate design — see docs/rubric_rl.md. The lanes
bundle path here is policy-only; launching with --target=rubric will
yield no training data (empty buffer), fail-loud by design.
"""

from __future__ import annotations

import math
from typing import Any

from swe_agent.contracts import (
    ExportGroup,
    ExportSample,
    GRPOExportBundle,
    has_exact_rollout_tokens,
)
from swe_agent.trajectory_search_parallel import (
    ForkGroup,
    InstanceRecord,
    LaneBBranch,
)
from swe_agent.trajectory_search import _build_step_cards


def _branch_overall_rubric_score(
    node_id: str, judge_score_by_node: dict[str, float]
) -> float | None:
    """Return the final oracle-equivalent score for the real runtime node."""
    value = (judge_score_by_node or {}).get(node_id)
    return None if value is None else float(value)


def _build_branch_sample(
    *,
    instance_id: str,
    group: ForkGroup,
    branch: LaneBBranch,
    steps_per_round: int | None = None,
    gt_only_reward: bool = False,
) -> ExportSample | None:
    """Build one ExportSample from a single Lane B branch.

    Prompt is always the shared root system/user state.  Root-group turns are
    the Lane-B continuation. Beam-group turns are the corresponding Lane-A
    parent followed by its Lane-B child.

    Token-level path: when each asst turn in branch.messages carries the
    sglang-stored prompt_token_ids + token_ids + logprobs, we use the LAST
    asst's prompt + token_ids as the full sequence and place per-turn
    1-mask intervals at the position each a_i's tokens occupy inside
    last.prompt_token_ids + last.token_ids. The HARD ASSERT requires
    a_i.prompt to be a true prefix of last.prompt — guaranteed by the
    custom chat template in swe_agent.tokenization (pure-splice renderer).

    ``steps_per_round`` caps root samples at k assistant turns and beam samples
    at 2k assistant turns, so both the judged parent and judged child stay in
    the trainable sequence.
    """
    # Formal submissions use binary SWE-bench reward. Unfinished trajectories
    # use the final oracle-equivalent hosted-judge score.
    if branch.error is not None:
        return None
    if gt_only_reward or branch.terminated_early:
        gt = branch.gt_score
        if gt is None or not math.isfinite(float(gt)):
            return None
        reward = float(gt)
        reward_source = (
            "gt_only"
            if gt_only_reward
            else "terminal_swebench_binary"
        )
    else:
        rubric = _branch_overall_rubric_score(
            branch.node_id, group.judge_score_by_node
        )
        if rubric is None or not math.isfinite(rubric):
            return None
        # The oracle joint aggregation is signed when a negative-direction
        # rubric fires, while this training recipe requires the judge signal
        # to stay on the same non-negative scale as terminal binary reward.
        # Preserve the signed value in raw_rubric_score for auditability, but
        # clip only the reward consumed by GRPO.
        reward = max(0.0, float(rubric))
        reward_source = "hosted_judge"

    # Every training group shares the root system/user prompt.  For a beam
    # sample, prepend the corresponding Lane-A parent continuation to the
    # Lane-B child so both assistant segments receive loss mask 1.
    mid_cp = group.mid_cp
    parent_messages = list(
        (mid_cp.snapshot.get("agent", {}).get("state", {}) or {}).get("messages", [])
    )
    branch_messages = list(branch.messages or [])
    parent_continuation_messages: list[dict[str, Any]] = []
    if group.group_kind == "beam":
        if (
            branch.beam_parent_index is None
            or branch.beam_parent_index < 0
            or branch.beam_parent_index >= len(group.parent_mid_cps)
        ):
            return None
        beam_parent = group.parent_mid_cps[branch.beam_parent_index]
        beam_parent_messages = list(
            (
                beam_parent.snapshot.get("agent", {}).get("state", {}) or {}
            ).get("messages", [])
        )
        if beam_parent_messages[: len(parent_messages)] != parent_messages:
            return None
        parent_continuation_messages = beam_parent_messages[len(parent_messages):]
        branch_messages = parent_continuation_messages + branch_messages
    if not branch_messages:
        return None

    token_ids: list[int] | None = None
    loss_mask: list[int] | None = None
    response_length: int | None = None
    rollout_logprobs: list[float] | None = None
    parent_assistant_spans: list[list[int]] = []
    child_assistant_spans: list[list[int]] = []

    branch_assistant_messages = [
        m for m in branch_messages if m.get("role") == "assistant"
    ]
    parent_assistant_count = sum(
        1
        for message in parent_continuation_messages
        if message.get("role") == "assistant"
    )
    if not branch_assistant_messages or not all(
        map(has_exact_rollout_tokens, branch_assistant_messages)
    ):
        return None
    branch_assistants_with_tokens = branch_assistant_messages
    # Cap trainable assistant turns to the judged window.
    # Anything past that cap is dropped from token_ids entirely — we slice
    # `last` to the last in-window assistant turn, so full_token_ids =
    # last_prompt + last_out naturally truncates the sequence.
    assistant_turn_cap = steps_per_round
    if (
        assistant_turn_cap is not None
        and assistant_turn_cap > 0
        and group.group_kind == "beam"
    ):
        assistant_turn_cap *= 2
    if assistant_turn_cap is not None and assistant_turn_cap > 0:
        branch_assistants_with_tokens = branch_assistants_with_tokens[:assistant_turn_cap]
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
        for assistant_index, asst in enumerate(branch_assistants_with_tokens):
            a_prompt = list(asst["prompt_token_ids"])
            # HARD ASSERT — see pds_to_grpo_bundle for the long form. Any
            # mismatch here means the custom chat template is no longer
            # producing prefix-stable renderings and PG ratio will explode.
            if len(a_prompt) > len(last_prompt):
                raise AssertionError(
                    f"lane sample {branch.node_id}: a_i.prompt len "
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
                    f"lane sample {branch.node_id}: a_i.prompt is NOT a prefix "
                    f"of last.prompt. first_divergence_idx={div}, "
                    f"len(a_i)={len(a_prompt)}, len(last)={len(last_prompt)}. "
                    f"Custom chat template must be broken — fix tokenization.py."
                )
            start = len(a_prompt)
            out_tok = list(asst["token_ids"])
            end = start + len(out_tok)
            if (
                start < parent_prefix_len
                or end <= start
                or end > len(full_token_ids)
            ):
                raise AssertionError(
                    f"lane sample {branch.node_id}: invalid assistant response "
                    f"span start={start} end={end} "
                    f"parent_prefix_len={parent_prefix_len} "
                    f"full_token_count={len(full_token_ids)}."
                )
            if full_token_ids[start:end] != out_tok:
                raise AssertionError(
                    f"lane sample {branch.node_id}: assistant output tokens are "
                    f"misaligned at span [{start}, {end})."
                )
            lp = list(asst["logprobs"])
            if len(lp) != len(out_tok):
                raise AssertionError(
                    f"lane sample {branch.node_id}: assistant logprobs len "
                    f"({len(lp)}) != output tokens len ({len(out_tok)})."
                )
            response_span = [
                start - parent_prefix_len,
                end - parent_prefix_len,
            ]
            if assistant_index < parent_assistant_count:
                parent_assistant_spans.append(response_span)
            else:
                child_assistant_spans.append(response_span)
            for k in range(start, end):
                mask[k] = 1
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
    if assistant_turn_cap is not None and assistant_turn_cap > 0:
        asst_count = 0
        cutoff_idx = len(branch_messages)
        for idx, m in enumerate(branch_messages):
            if m.get("role") == "assistant":
                asst_count += 1
                if asst_count >= assistant_turn_cap:
                    cutoff_idx = idx + 1
                    break
        capped_branch_messages_for_turns = branch_messages[:cutoff_idx]
        n_trainable_asst_turns = min(
            assistant_turn_cap, n_trainable_asst_turns
        )

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
                branch.node_id, group.judge_score_by_node
            ),
            "reward_source": reward_source,
            "policy_overlength_reason": branch.overlength_reason,
            "group_kind": group.group_kind,
            "beam_parent_index": branch.beam_parent_index,
            "beam_parent_node_id": branch.beam_parent_node_id,
            "total_tokens": branch.total_tokens,
            "parent_token_count": (len(token_ids) - response_length)
            if token_ids is not None and response_length is not None else None,
            # Exact response-relative token intervals. These make the
            # parent-training contract independently verifiable from the
            # accepted Slime rollout dump: root samples have no parent
            # intervals; beam samples have one interval per Lane-A assistant
            # turn followed by one per Lane-B child assistant turn.
            "parent_assistant_spans": parent_assistant_spans,
            "child_assistant_spans": child_assistant_spans,
            "token_source": "sglang_stored",
            # Per-sample step counts — surfaced for wandb aggregation in
            # collect_lanes_rollout_async.generate_rollout's metrics dict.
            "n_continuation_steps": len(branch_step_cards),
            "n_trainable_asst_turns": n_trainable_asst_turns,
            "steps_per_round_cap": steps_per_round,
            "n_parent_steps": (
                group.parent_mid_cps[branch.beam_parent_index].asst_step
                if group.group_kind == "beam"
                and branch.beam_parent_index is not None
                else mid_cp.asst_step
            ),
            "n_full_trace_steps": (
                (
                    group.parent_mid_cps[branch.beam_parent_index].asst_step
                    if group.group_kind == "beam"
                    and branch.beam_parent_index is not None
                    else mid_cp.asst_step
                )
                + len(branch_step_cards)
            ),
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
    expected_group_size: int | None = None,
) -> ExportGroup | None:
    """Convert one fork-group into an ExportGroup of m=8 samples.

    Used by the streaming on_group_done callback in the slime rollout fn
    — fires per-group as each group completes, without waiting for the
    instance to finish.

    An invalid branch invalidates the complete sibling group so GRPO never
    computes an advantage from a partial group.

    ``steps_per_round`` caps each judged segment; tokens after the cap are
    omitted from the training sequence.
    """
    if expected_group_size is not None and len(group.branches) != expected_group_size:
        return None
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
            "group_kind": group.group_kind,
            "n_beam_parents": len(group.parent_mid_cps),
        },
    )


def instance_record_to_bundle(
    record: InstanceRecord,
    *,
    steps_per_round: int | None = None,
    gt_only_reward: bool = False,
) -> GRPOExportBundle:
    """Convert one lane InstanceRecord into a GRPOExportBundle.

    ``depth1`` exports the root M=8 group. ``depth2`` exports that same group
    plus a second M=8 beam group when at least one sampled parent remains
    unfinished. Lane-A parent tokens are included in each corresponding beam
    response. Unfinished traces use the hosted judge score; formal submissions
    use their binary SWE-bench terminal result.

    rubric_groups: always empty — true rubric-model training is a
    separate piece of work (see docs/rubric_rl.md)."""
    cap = steps_per_round
    if cap is None:
        try:
            cap = int(record.config.get("steps_per_round")) if record.config else None
        except (TypeError, ValueError):
            cap = None
    policy_groups: list[ExportGroup] = []
    expected_group_size = None
    try:
        expected_group_size = int(record.config.get("m"))
    except (TypeError, ValueError, AttributeError):
        pass
    for group in record.groups:
        eg = fork_group_to_export_group(
            instance_id=record.instance_id, group=group, steps_per_round=cap,
            gt_only_reward=gt_only_reward,
            expected_group_size=expected_group_size,
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
            "reward_recipe": (
                "gt_only"
                if gt_only_reward
                else "terminal_swebench_binary_else_nonnegative_hosted_judge"
            ),
            "steps_per_round_cap": cap,
            "completed": record.completed,
            "error": record.error,
            "seconds": record.seconds,
            "num_groups": len(record.groups),
            "skipped_group_reasons": {
                str(index): reason
                for index, reason in record.skipped_group_reasons.items()
            },
            "num_mid_cps": len(record.lane_a.mid_cps),
            "scheme": "lane_v1",
        },
    )
