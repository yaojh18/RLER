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

from swe_agent.contracts import ExportGroup, ExportSample, GRPOExportBundle
from swe_agent.trajectory_search_parallel import (
    ForkGroup,
    InstanceRecord,
    LaneBBranch,
)


# Pure-rubric reward for policy update. The earlier blend (alpha*gt +
# (1-alpha)*rubric) is gone — see commit message and docs/rubric_rl.md for
# the rationale (GT-only and blended rewards both biased the policy toward
# patch-format hacking; rubric-only forces the policy to satisfy the judge's
# rubric criteria, which is the actual training signal we want to optimize).
EMPTY_RUBRIC_REWARD = 0.0


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


def _normalize_message(msg: dict) -> dict:
    return {
        "role": msg.get("role", "assistant") or "assistant",
        "content": msg.get("content", "") or "",
    }


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

    Fallback path: re-tokenize via the chat template when stored token
    fields are absent (eval / non-rollout-traffic paths).

    steps_per_round (task 3): if set, truncate the trainable window to the
    first `steps_per_round` ASSISTANT TURNS past the shared parent. Tokens
    emitted after that cutoff are dropped entirely (token_ids ends at the
    last in-window assistant turn's output). Rationale: the rubric judges
    only the first steps_per_round of each branch (Lane C input is the
    round1 truncated trajectory), so training on tokens beyond that point
    has no judged signal — and the GRPO advantage applied to those tokens
    is just noise from the group baseline.
    """
    from swe_agent.tokenization import (
        compute_loss_mask_for_messages,
        tokenize_messages_with_template,
    )

    # Reward = pure rubric (task 4). Earlier we blended alpha*gt +
    # (1-alpha)*rubric; that biased the policy toward whichever signal had
    # higher variance in a given group. With rubric-only, every group's
    # advantage is driven by the judge — which is the signal we actually
    # want the policy to optimize. NaN-safe: if no rubric judged this
    # branch (errored branch / Lane C failed / overflow), reward is 0
    # so it sits at the baseline rather than poisoning the group with NaN.
    #
    # gt_only_reward (paired with ParallelSearchConfig.disable_rubric): use
    # branch.gt_score (already has fallback_patch_penalty baked in) as the
    # reward. Same scale as the naive baseline; each ForkGroup's M branches
    # are GRPO-normalized against each other conditioned on the shared
    # MidCp state.
    import math as _math
    if branch.error is not None:
        reward = 0.0
    elif gt_only_reward:
        gt = branch.gt_score
        if gt is None or _math.isnan(float(gt)):
            reward = 0.0
        else:
            reward = float(gt)
    else:
        rubric = _branch_overall_rubric_score(
            branch.branch_index, group.judge_response
        )
        if rubric is None or _math.isnan(rubric):
            reward = 0.0
        else:
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

    branch_assistants_with_tokens = [
        m for m in branch_messages
        if (m.get("role") == "assistant")
        and m.get("prompt_token_ids")
        and m.get("token_ids")
    ]
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
            lp = asst.get("logprobs") or []
            n_lp = min(end - start, len(lp))
            for j in range(n_lp):
                local_pos = (start - parent_prefix_len) + j
                if 0 <= local_pos < response_length:
                    dense_lp[local_pos] = float(lp[j])
        token_ids = full_token_ids
        loss_mask = mask
        rollout_logprobs = dense_lp
    else:
        # Fallback (eval / non-rollout): re-tokenize via chat template.
        # Apply the same steps_per_round cap by trimming branch_messages
        # to end after the steps_per_round-th assistant turn.
        capped_branch_messages = branch_messages
        if steps_per_round is not None and steps_per_round > 0:
            asst_count = 0
            cutoff_idx = len(branch_messages)
            for idx, m in enumerate(branch_messages):
                if m.get("role") == "assistant":
                    asst_count += 1
                    if asst_count >= steps_per_round:
                        cutoff_idx = idx + 1
                        break
            capped_branch_messages = branch_messages[:cutoff_idx]
        parent_norm = [_normalize_message(m) for m in parent_messages]
        branch_norm = [_normalize_message(m) for m in capped_branch_messages]
        try:
            if parent_norm:
                parent_ids = tokenize_messages_with_template(
                    parent_norm, add_generation_prompt=False,
                )
            else:
                parent_ids = []
            full_norm = parent_norm + branch_norm
            full_ids, full_mask = compute_loss_mask_for_messages(full_norm)
            zero_through = min(len(parent_ids), len(full_mask))
            for k in range(zero_through):
                full_mask[k] = 0
            token_ids = full_ids
            loss_mask = full_mask
            response_length = max(0, len(full_ids) - len(parent_ids))
            if response_length == 0:
                return None
            # Fallback path has no sglang-stored logprobs; emit a zero
            # vector so build_rollout_samples passes its length guard and
            # TIS / rollout_logprob_mean stay shape-compatible. loss_mask
            # zeros at these positions ensure no PG / IS contribution.
            rollout_logprobs = [0.0] * response_length
        except Exception:
            token_ids = None
            loss_mask = None
            response_length = None

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

    gt_payload = branch.gt_payload or {}
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
            "token_source": "sglang_stored"
            if branch_assistants_with_tokens else "rechattemplate",
            # Per-sample step counts — surfaced for wandb aggregation in
            # collect_lanes_rollout_async.generate_rollout's metrics dict.
            "n_continuation_steps": len(branch.step_cards),
            "n_trainable_asst_turns": n_trainable_asst_turns,
            "steps_per_round_cap": steps_per_round,
            "n_parent_steps": mid_cp.asst_step,
            "n_full_trace_steps": mid_cp.asst_step + len(branch.step_cards),
            # Patch + eval payload fields surfaced for per-batch WandB ratios
            # (ratio_zero_patch, ratio_full_pass, ratio_regression, ...).
            "terminal_patch_len": len(branch.terminal_patch or ""),
            "terminal_patch_from_fallback": bool(branch.terminal_patch_from_fallback),
            "eval_status": gt_payload.get("status"),
            "eval_note": gt_payload.get("note"),
            "f2p_passed_count": gt_payload.get("f2p_passed_count"),
            "f2p_total": gt_payload.get("f2p_total"),
            "p2p_passed_count": gt_payload.get("p2p_passed_count"),
            "p2p_total": gt_payload.get("p2p_total"),
        },
        token_ids=token_ids,
        loss_mask=loss_mask,
        response_length=response_length,
        rollout_logprobs=rollout_logprobs,
    )


def _dummy_sample_for_branch(
    *, instance_id: str, group: ForkGroup, branch: LaneBBranch
) -> ExportSample:
    """Placeholder sample for a branch that produced no asst tokens (e.g.,
    sglang context-length overflow on the FIRST call from a late mid_cp —
    97% of empty branches in 51963 rollout_0 died this way).

    The dummy carries reward=0.0 with a 2-token sequence whose loss_mask
    is all-zero, so it contributes nothing to the policy gradient. But it
    DOES participate in the GRPO baseline (group mean/std), preventing
    the singleton-group NaN we saw in 51963 step 0 when 7 of 8 siblings
    failed and only 1 real sample was left.

    Mirrors v0's approach in pds_to_grpo_bundle (set reward=0.0 for
    errored branches) but extends it to the response_length==0 case which
    v0 also drops.
    """
    return ExportSample(
        sample_id=f"dummy-{branch.node_id}",
        group_id=f"policy-{instance_id}-g{group.group_index:03d}",
        prompt=[],
        turns=[],
        reward=0.0,
        metadata={
            "branch_index": branch.branch_index,
            "group_index": group.group_index,
            "mid_cp_image_tag": group.mid_cp.image_tag,
            "mid_cp_asst_step": group.mid_cp.asst_step,
            "terminated_early": False,
            "raw_gt_score": 0.0,
            "raw_rubric_score": None,
            "total_tokens": {"prompt": 0, "completion": 0},
            "parent_token_count": 1,
            "token_source": "dummy_empty_branch",
            "n_continuation_steps": 0,
            "n_parent_steps": group.mid_cp.asst_step,
            "n_full_trace_steps": group.mid_cp.asst_step,
            "is_dummy": True,
            "branch_error": branch.error or "no_asst_token_ids",
            "terminal_patch_len": 0,
            "terminal_patch_from_fallback": False,
            "eval_status": None,
            "eval_note": None,
            "f2p_passed_count": None,
            "f2p_total": None,
            "p2p_passed_count": None,
            "p2p_total": None,
        },
        # 2 tokens (parent + response); loss_mask all-zero → no PG signal.
        # Using small pad ids (0) keeps slime's _post_process happy.
        token_ids=[0, 0],
        loss_mask=[0, 0],
        response_length=1,
        rollout_logprobs=[0.0],
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

    Always emits one sample per branch, even when a branch errored or
    produced no asst tokens — empty branches get a dummy 0-reward
    placeholder (see _dummy_sample_for_branch). This keeps groups at
    full M=M=8, matches v0's GRPO baseline shape, and prevents the
    singleton-NaN bug (51963 step 0). If ALL branches are empty, the
    whole group has zero signal — caller drops it.

    steps_per_round (task 3): cap each branch's trainable window to its
    first steps_per_round assistant turns past the shared parent. Tokens
    after the cap are dropped from token_ids entirely.
    """
    samples: list[ExportSample] = []
    n_real = 0
    for branch in group.branches:
        s = _build_branch_sample(
            instance_id=instance_id, group=group, branch=branch,
            steps_per_round=steps_per_round,
            gt_only_reward=gt_only_reward,
        )
        if s is None:
            s = _dummy_sample_for_branch(
                instance_id=instance_id, group=group, branch=branch,
            )
        else:
            n_real += 1
        samples.append(s)
    if n_real == 0:
        # Pure-dummy group → no learning signal at all. Drop.
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
            "n_real": n_real,
            "n_dummy": len(samples) - n_real,
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
