"""Convert v1 lane-based InstanceRecord -> GRPOExportBundle.

One ExportGroup per ForkGroup (= per Lane A mid_cp). Each group has M=8
ExportSamples, one per Lane B branch. Lane A's tokens are never included.

The token-level prefix invariant + per-turn loss-mask placement is COPIED
verbatim from pds_to_grpo_bundle.py — that logic is the core fix for the
v0 7.6B-loss bug and is independent of the search topology. We re-implement
locally (rather than import) so the v1 path has zero runtime dependency on
v0's bundle module, but the semantics are byte-equal.

Rubric-training bundles (rubric_groups) are NOT generated in v1 first pass
— the GRPO --target=policy launcher only consumes policy_groups. Adding
rubric_groups would require a cross-group baseline that's nontrivial in
the lane scheme; defer until needed.
"""

from __future__ import annotations

from typing import Any

from swe_agent.contracts import ExportGroup, ExportSample, GRPOExportBundle
from swe_agent.trajectory_search_parallel import (
    ForkGroup,
    InstanceRecord,
    LaneBBranch,
)


DEFAULT_POLICY_REWARD_ALPHA = 1.0  # 1.0 = pure GT, 0.0 = pure rubric
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
    alpha: float,
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
    """
    from swe_agent.tokenization import (
        compute_loss_mask_for_messages,
        tokenize_messages_with_template,
    )

    # Reward folding: same recipe as v0 (alpha*gt + (1-alpha)*rubric_mean).
    # NaN-safe: skip the rubric term when alpha==1.0 (pure GT) OR when the
    # rubric score itself is NaN, because IEEE 0.0 * NaN = NaN poisons the
    # reward even when we meant to weight rubric to zero. This was the
    # root cause of the v1 grad-norm NaN at step 0/1/5 across 51783/51840/
    # 51898/51963 — a single NaN-tainted rubric in a group made
    # std-normalized advantages NaN and Megatron's rerun_state_machine
    # caught the gradient before backward propagation.
    import math as _math
    if branch.error is not None:
        reward = 0.0
    else:
        gt = float(branch.gt_score) if branch.gt_score is not None else 0.0
        rubric = _branch_overall_rubric_score(
            branch.branch_index, group.judge_response
        )
        if alpha == 1.0 or rubric is None or _math.isnan(rubric):
            reward = gt
        else:
            reward = alpha * gt + (1.0 - alpha) * rubric
        if _math.isnan(reward):
            reward = 0.0  # defense in depth — never let NaN reach slime

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
        lp_concat: list[float] = []
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
            lp_concat.extend(lp[: max(0, end - start)])
        token_ids = full_token_ids
        loss_mask = mask
        rollout_logprobs = lp_concat
    else:
        # Fallback (eval / non-rollout): re-tokenize via chat template.
        parent_norm = [_normalize_message(m) for m in parent_messages]
        branch_norm = [_normalize_message(m) for m in branch_messages]
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
        except Exception:
            token_ids = None
            loss_mask = None
            response_length = None

    return ExportSample(
        sample_id=branch.node_id,
        group_id=f"policy-{instance_id}-g{group.group_index:03d}",
        prompt=[
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in parent_messages
        ],
        turns=[
            {"role": m.get("role", "assistant"), "content": m.get("content", "")}
            for m in branch_messages
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
            "n_parent_steps": mid_cp.asst_step,
            "n_full_trace_steps": mid_cp.asst_step + len(branch.step_cards),
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
    alpha: float = DEFAULT_POLICY_REWARD_ALPHA,
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
    """
    samples: list[ExportSample] = []
    n_real = 0
    for branch in group.branches:
        s = _build_branch_sample(
            instance_id=instance_id, group=group, branch=branch, alpha=alpha,
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
    policy_reward_alpha: float = DEFAULT_POLICY_REWARD_ALPHA,
) -> GRPOExportBundle:
    """Convert one v1 InstanceRecord into a GRPOExportBundle.

    policy_groups: one ExportGroup per ForkGroup (M=8 Lane B siblings
    forked from the same Lane A MidCp). Lane A's trajectory is never
    emitted as a sample.

    rubric_groups: empty in v1 first pass (no rubric-target training in
    the lane scheme yet)."""
    policy_groups: list[ExportGroup] = []
    for group in record.groups:
        eg = fork_group_to_export_group(
            instance_id=record.instance_id, group=group, alpha=policy_reward_alpha,
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
            "policy_reward_alpha": policy_reward_alpha,
            "completed": record.completed,
            "error": record.error,
            "seconds": record.seconds,
            "num_groups": len(record.groups),
            "num_mid_cps": len(record.lane_a.mid_cps),
            "scheme": "lane_v1",
        },
    )
