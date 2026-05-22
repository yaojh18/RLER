"""Lane-based parallel trajectory search.

Flat Lane A / Lane B / Lane C scheme, three concurrent lanes per instance:

* Lane A (spine, NOT trained)
    A single agent runs the problem linearly to terminal/submit. Never blocks.
    Every `k` assistant turns: `docker commit` the container,
    emit a MidCp = {idx, image_tag, snapshot, asst_step}. Purely a fork-point
    provider — its trajectory is NOT included in any training group.

* Lane B (M forks per mid_cp, sampling temperature = lane_b_temperature)
    For each mid_cp Lane A emits, fork M agents from that docker snapshot.
    Each Lane B branch runs independently to its own termination. The
    branches forked from mid_cp_i form fork-group i.

* Lane C (async rubric + judge per fork-group)
    Trigger: when all Lane B branches in group i have run to terminal/limit.
    Lane C then dispatches summary/rubric_gen/judge on the shared parent state
    + each branch's first-`k` continuation steps. Lane C is serialized only for
    rubric-bank state; it does not block Lane A or other groups' Lane B rollout.

GRPO bundle per fork-group: M trajectories sharing prompt = Lane A history
up to mid_cp_i, each with its own continuation to termination. Per-branch
reward = gt_score * fallback_penalty + rubric_judge_score; advantage = reward
- mean across the group (standard GRPO baseline).
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import logging
import os
import subprocess
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import pvariance
from typing import Any

from agent_rl import RolloutSessionSpec, RolloutSnapshot
from swe_agent.backend import SWEAgentRolloutBackend
from swe_agent.contracts import ExportGroup, ExportSample, GRPOExportBundle

# Reuse shared artifact/prompt helpers plus trajectory_search's rubric,
# judge, and persistent-state implementations so the parallel path stays
# aligned with the non-parallel search semantics.
from swe_agent.parallel_utils import (
    _build_rubric_prompt,
    _ensure_litellm_prefix,
    _stamp_steps,
    _atomic_write_json,
    TurnTokenInfo,
)
from swe_agent.prompt import EMPTY_PERSISTENT_STATE, EMPTY_WORKSPACE_META
from swe_agent.run.run_swe_agent import build_messages, evaluate_swebench_instance_patches
from swe_agent.rubric_bank import ExperienceRubricBank, ScoreRubricBank
from swe_agent.trajectory_search import (
    JUDGE_ERROR_REWARD,
    _generate_round_rubrics,
    _redundancy_reward,
    _score_round,
    _update_persistent_state,
    _build_step_cards,
    _collect_workspace_meta,
    _docker_commit,
)


logger = logging.getLogger("swe_agent.trajectory_search_parallel")


_HEAVY_TOKEN_KEYS = {
    "input_token_ids",
    "output_logprobs",
    "output_token_ids",
    "prompt_token_ids",
    "rollout_logprobs",
    "token_ids",
    "logprobs",
}


def _clone_without_heavy_token_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _clone_without_heavy_token_fields(item)
            for key, item in value.items()
            if key not in _HEAVY_TOKEN_KEYS
        }
    if isinstance(value, list):
        return [_clone_without_heavy_token_fields(item) for item in value]
    return copy.deepcopy(value)


def _build_parent_messages_payload(messages: list[dict[str, Any]], *, model_name: str) -> dict[str, Any]:
    parent_stamped, _ = _stamp_steps(messages, start_step=0)
    return build_messages(
        parent_stamped,
        model_name=model_name,
        preserve_token_fields=True,
    )


_STEP_CARD_EVENT_KINDS = {
    "model_response",
    "environment_action",
    "environment_result",
    "agent_interrupt",
}


def _step_card_event_dicts(events: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event in events:
        kind = getattr(event, "kind", None)
        payload = getattr(event, "payload", None)
        if kind is None and isinstance(event, dict):
            kind = event.get("kind")
            payload = event.get("payload")
        if kind not in _STEP_CARD_EVENT_KINDS:
            continue
        out.append(
            {
                "kind": kind,
                "payload": _clone_without_heavy_token_fields(payload or {}),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Configuration & data model
# ---------------------------------------------------------------------------


@dataclass
class ParallelSearchConfig:
    """Parallel search config. The public sampling knobs mirror SearchConfig:
    m sibling branches, n rubric lists, k rollout steps per fork interval,
    p active parents (currently p=1 for the single Lane A spine), and
    max_rounds total fork-groups, counting the root fork as round 1."""

    m: int = 8  # forks per mid_cp (NOT M-1)
    n: int = 1  # rubric-list samples per group
    k: int = 20  # assistant turns between Lane A checkpoints
    p: int = 1  # parallel currently follows one Lane A spine
    max_rounds: int = 5  # cap on fork-groups per instance; root counts
    step_limit: int = 100  # hard cap for the whole trajectory
    max_active_rubrics: int = 6

    # Lane A sampling — runs deterministic-ish spine; lower temp is fine.
    policy_temperature: float = 1.0
    policy_top_p: float = 0.95
    # Lane B sampling — higher for intra-group diversity (we WANT different
    # solution attempts from the same parent state).
    lane_b_temperature: float = 1.0
    lane_b_top_p: float = 0.95

    # Lane C (rubric/judge) reuses v0 defaults.
    rubric_temperature: float = 1.0
    rubric_top_p: float = 0.95
    rubric_max_tokens: int = 4096
    judge_temperature: float = 0.1
    judge_top_p: float = 0.95
    judge_max_tokens: int = 1024
    psu_max_tokens: int = 4096

    gt_eval_workers: int = 8
    lane_a_pool_size: int = 1  # Lane A is a single agent
    lane_b_pool_size: int | None = None  # default = m * max_rounds

    keep_images: bool = False
    return_logprobs: bool = True
    # Same recipe as v0: multiplicative penalty when the agent never invoked
    # the formal submit command and we fell back to `git diff` of the
    # working copy.
    fallback_patch_penalty: float = 0.5

    def __post_init__(self) -> None:
        if self.p != 1:
            raise ValueError("trajectory_search_parallel currently supports p=1 only.")


@dataclass
class MidCp:
    """Snapshot Lane A emits every `k` assistant turns.

    Used as the fork point for `m` Lane B branches. The docker image_tag
    captures the container filesystem state; the snapshot dict is runtime-only
    resume state for Lane B and intentionally omits parent events/model_turns.
    """

    idx: int  # 0-indexed mid_cp position along Lane A
    asst_step: int  # cumulative Lane A asst step at emission
    image_tag: str  # docker image committed from Lane A's container
    snapshot: dict[str, Any] = field(default_factory=dict)
    parent_messages_payload: dict[str, Any] = field(default_factory=dict)
    parent_step_cards: list[dict[str, Any]] = field(default_factory=list)
    # Token-level position markers — for verifying prefix invariance later.
    parent_message_count: int = 0  # len(snapshot.messages) at emission
    parent_event_count: int = 0
    parent_turn_count: int = 0
    emitted_at: float = 0.0  # perf_counter timestamp


@dataclass
class LaneBBranch:
    """A single Lane B fork. Runs to termination after being forked from a
    MidCp. Records its own messages, token info, terminal patch, GT score.

    Flattened vs v0's BranchState — no separate short/terminal phase since
    Lane B doesn't have round boundaries. We just record everything that
    happened between fork and termination.
    """

    group_index: int  # which fork-group (= which MidCp idx)
    branch_index: int  # 0..m-1 within the group
    node_id: str
    parent_image_tag: str
    session_id: str = ""
    policy_model_name: str = ""
    parent_asst_step: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    workspace_meta: dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(EMPTY_WORKSPACE_META)
    )
    turns: list[TurnTokenInfo] = field(default_factory=list)
    total_tokens: dict[str, int] = field(default_factory=dict)
    status: str = ""  # session status string at exit
    terminated_early: bool = False  # True iff agent invoked formal submit
    terminal_patch: str = ""
    terminal_patch_from_fallback: bool = False
    error: str | None = None
    gt_score: float | None = None
    gt_payload: dict[str, Any] | None = None
    started_at: float = 0.0
    finished_at: float = 0.0


@dataclass
class ForkGroup:
    """Lane B branches forked from the same MidCp + per-group Lane C output.

    Ready-for-bundle when: all `m` branches have terminated (or hit the
    remaining global step budget) AND Lane C has returned. Lane A's status is
    irrelevant for readiness.
    """

    group_index: int  # = mid_cp.idx
    mid_cp: MidCp
    branches: list[LaneBBranch] = field(default_factory=list)
    rubric_prompt: str = ""
    rubric_model_response: dict[str, Any] = field(default_factory=dict)
    rubric_samples: list[dict[str, Any]] = field(default_factory=list)
    judge_response: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    experience_bank_update: dict[str, Any] = field(default_factory=dict)
    experience_bank_message: list[dict[str, Any]] = field(default_factory=list)
    persisted_parent_state: dict[str, Any] = field(default_factory=dict)
    lane_c_started_at: float = 0.0
    lane_c_done_at: float = 0.0
    bundled_at: float = 0.0


@dataclass
class LaneAState:
    """Lane A's overall record — for inspection / debug. NOT trained on."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    mid_cps: list[MidCp] = field(default_factory=list)
    terminal_patch: str = ""
    terminal_patch_from_fallback: bool = False
    terminated_early: bool = False
    status: str = ""
    error: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    total_tokens: dict[str, int] = field(default_factory=dict)


@dataclass
class InstanceRecord:
    """Top-level record persisted to disk per instance. Conceptually replaces
    v0's InstanceRecord; the slime data-buffer hookup reads `groups`
    (and converts each ForkGroup to a GRPOExportBundle)."""

    instance_id: str
    run_dir: str
    task_id: str
    config: dict[str, Any]
    lane_a: LaneAState = field(default_factory=LaneAState)
    groups: list[ForkGroup] = field(default_factory=list)
    completed: bool = False
    error: str | None = None
    seconds: float = 0.0


class LaneGRPOCollector:
    DEFAULT_POLICY_REWARD_ALPHA = 1.0

    def __init__(self, *, policy_reward_alpha: float = DEFAULT_POLICY_REWARD_ALPHA) -> None:
        self.policy_reward_alpha = float(policy_reward_alpha)

    @staticmethod
    def _branch_overall_rubric_score(branch: LaneBBranch, judge_response: dict[str, Any]) -> float | None:
        scores: list[float] = []
        for branch_map in (judge_response or {}).values():
            record = (branch_map or {}).get(branch.node_id)
            if record is None or "error" in record:
                continue
            score = record.get("score_normalized")
            if score is not None:
                scores.append(float(score))
        if not scores:
            return None
        return float(sum(scores) / len(scores))

    @staticmethod
    def _export_message(message: dict[str, Any]) -> dict[str, Any]:
        item = {
            "role": message.get("role", "assistant") or "assistant",
            "content": message.get("content", message.get("message", "")) or "",
        }
        if "content_no_thinking" in message:
            item["content_no_thinking"] = message["content_no_thinking"]
        return item

    def _branch_reward(self, *, branch: LaneBBranch, group: ForkGroup) -> float:
        import math

        if branch.error is not None:
            return 0.0
        gt = float(branch.gt_score) if branch.gt_score is not None else 0.0
        rubric = self._branch_overall_rubric_score(branch, group.judge_response)
        if self.policy_reward_alpha == 1.0 or rubric is None or math.isnan(rubric):
            return gt
        reward = self.policy_reward_alpha * gt + (1.0 - self.policy_reward_alpha) * rubric
        if math.isnan(reward):
            return 0.0
        return float(reward)

    def _build_branch_sample(self, *, instance_id: str, group: ForkGroup, branch: LaneBBranch) -> ExportSample | None:
        from swe_agent.tokenization import compute_loss_mask_for_messages, tokenize_messages_with_template

        branch_step_cards = _build_step_cards(branch.events, branch.parent_asst_step)
        parent_messages = list(
            (group.mid_cp.snapshot.get("agent", {}).get("state", {}) or {}).get("messages", [])
        )
        branch_messages = list(branch.messages or [])
        if not branch_messages:
            return None

        branch_assistants_with_tokens = [
            message for message in branch_messages
            if message.get("role") == "assistant"
            and message.get("prompt_token_ids")
            and message.get("token_ids")
        ]
        token_ids: list[int]
        loss_mask: list[int]
        response_length: int
        rollout_logprobs: list[float] | None
        token_source: str

        if branch_assistants_with_tokens:
            last = branch_assistants_with_tokens[-1]
            last_prompt = list(last["prompt_token_ids"])
            last_out = list(last["token_ids"])
            full_token_ids = last_prompt + last_out
            parent_prefix_len = len(branch_assistants_with_tokens[0]["prompt_token_ids"])
            response_length = max(0, len(full_token_ids) - parent_prefix_len)
            if response_length == 0:
                return None
            mask = [0] * len(full_token_ids)
            logprobs: list[float] = []
            for assistant in branch_assistants_with_tokens:
                assistant_prompt = list(assistant["prompt_token_ids"])
                if len(assistant_prompt) > len(last_prompt):
                    raise AssertionError(
                        f"lane sample {branch.node_id}: assistant prompt len {len(assistant_prompt)} exceeds final prompt len {len(last_prompt)}"
                    )
                if last_prompt[: len(assistant_prompt)] != assistant_prompt:
                    divergence = next(
                        index for index in range(len(assistant_prompt))
                        if last_prompt[index] != assistant_prompt[index]
                    )
                    raise AssertionError(
                        f"lane sample {branch.node_id}: assistant prompt is not a prefix of final prompt; first_divergence_idx={divergence}"
                    )
                start = len(assistant_prompt)
                end = min(start + len(assistant["token_ids"]), len(mask))
                for index in range(start, end):
                    mask[index] = 1
                logprobs.extend(list(assistant.get("logprobs") or [])[: max(0, end - start)])
            token_ids = full_token_ids
            loss_mask = mask
            rollout_logprobs = logprobs
            token_source = "sglang_stored"
        else:
            parent_norm = [self._export_message(message) for message in parent_messages]
            branch_norm = [self._export_message(message) for message in branch_messages]
            parent_ids = (
                tokenize_messages_with_template(
                    parent_norm,
                    add_generation_prompt=False,
                    model_path=branch.policy_model_name,
                )
                if parent_norm else []
            )
            full_ids, full_mask = compute_loss_mask_for_messages(
                parent_norm + branch_norm,
                model_path=branch.policy_model_name,
            )
            for index in range(min(len(parent_ids), len(full_mask))):
                full_mask[index] = 0
            response_length = max(0, len(full_ids) - len(parent_ids))
            if response_length == 0:
                return None
            token_ids = full_ids
            loss_mask = full_mask
            rollout_logprobs = None
            token_source = "rechattemplate"

        return ExportSample(
            sample_id=branch.node_id,
            group_id=f"policy-{instance_id}-g{group.group_index:03d}",
            prompt=[self._export_message(message) for message in parent_messages],
            turns=[self._export_message(message) for message in branch_messages],
            reward=self._branch_reward(branch=branch, group=group),
            metadata={
                "branch_index": branch.branch_index,
                "group_index": group.group_index,
                "mid_cp_image_tag": group.mid_cp.image_tag,
                "mid_cp_asst_step": group.mid_cp.asst_step,
                "terminated_early": branch.terminated_early,
                "raw_gt_score": branch.gt_score,
                "raw_rubric_score": self._branch_overall_rubric_score(branch, group.judge_response),
                "total_tokens": branch.total_tokens,
                "parent_token_count": len(token_ids) - response_length,
                "token_source": token_source,
                "n_continuation_steps": len(branch_step_cards),
                "n_parent_steps": group.mid_cp.asst_step,
                "n_full_trace_steps": group.mid_cp.asst_step + len(branch_step_cards),
            },
            token_ids=token_ids,
            loss_mask=loss_mask,
            response_length=response_length,
            rollout_logprobs=rollout_logprobs,
        )

    @staticmethod
    def _dummy_sample_for_branch(*, instance_id: str, group: ForkGroup, branch: LaneBBranch) -> ExportSample:
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
            token_ids=[0, 0],
            loss_mask=[0, 0],
            response_length=1,
            rollout_logprobs=[0.0],
        )

    def fork_group_to_export_group(self, *, instance_id: str, group: ForkGroup) -> ExportGroup | None:
        samples: list[ExportSample] = []
        n_real = 0
        for branch in group.branches:
            sample = self._build_branch_sample(instance_id=instance_id, group=group, branch=branch)
            if sample is None:
                sample = self._dummy_sample_for_branch(instance_id=instance_id, group=group, branch=branch)
            else:
                n_real += 1
            samples.append(sample)
        if n_real == 0:
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

    def instance_record_to_bundle(self, record: InstanceRecord) -> GRPOExportBundle:
        policy_groups: list[ExportGroup] = []
        for group in record.groups:
            export_group = self.fork_group_to_export_group(instance_id=record.instance_id, group=group)
            if export_group is not None:
                policy_groups.append(export_group)
        return GRPOExportBundle(
            instance_id=record.instance_id,
            run_dir=record.run_dir,
            policy_groups=policy_groups,
            rubric_groups=[],
            metadata={
                "config": record.config,
                "policy_reward_alpha": self.policy_reward_alpha,
                "completed": record.completed,
                "error": record.error,
                "seconds": record.seconds,
                "num_groups": len(record.groups),
                "num_mid_cps": len(record.lane_a.mid_cps),
                "scheme": "lane_v1",
            },
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class TrajectorySearchParallelRunner:
    """Lane-based parallel trajectory search per instance.

    Public API mirrors v0's ParallelDataSearchRunner:
        runner = TrajectorySearchParallelRunner(...)
        record = await runner.run(on_group_done=callback)

    `on_group_done(fork_group) -> None` fires once per ForkGroup as soon as
    it is ready-for-bundle (all m branches done + Lane C complete). The
    slime rollout fn converts the ForkGroup into a GRPOExportBundle inside
    the callback for streaming to training.
    """

    def __init__(
        self,
        *,
        instance: dict[str, Any],
        backend: SWEAgentRolloutBackend,
        run_dir: Path,
        policy_model_name: str,
        rubric_model_name: str | None,
        judge_model_name: str | None,
        config: ParallelSearchConfig,
        harness_namespace: str | None,
        policy_base_url: str,
        rubric_base_url: str,
        api_key: str = "EMPTY",
    ) -> None:
        self.instance = copy.deepcopy(instance)
        self.backend = backend
        self.run_dir = Path(run_dir)
        self.policy_model_name = policy_model_name
        self.rubric_model_name = rubric_model_name or policy_model_name
        self.judge_model_name = judge_model_name or self.rubric_model_name
        self.config = config
        self.harness_namespace = harness_namespace
        self.policy_base_url = policy_base_url.rstrip("/")
        self.rubric_base_url = rubric_base_url.rstrip("/")
        self.api_key = api_key

        self.task = instance["problem_statement"]
        self.task_id = instance["instance_id"]
        self.run_id = f"{self.task_id}-{time.strftime('%Y%m%d-%H%M%S')}"
        self.base_image = str(self.backend.environment_config.get("image", ""))
        self.docker_executable = str(
            self.backend.environment_config.get("executable", "docker")
        )
        # Image-tag namespace per instance, same convention as v0.
        self.image_repository = f"rler-pds/{self.task_id.replace('__', '-').lower()}"

        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "groups").mkdir(parents=True, exist_ok=True)
        self._created_image_tags: list[str] = []
        self._live_sessions: list[Any] = []
        shared_extra_body = (
            {"chat_template_kwargs": {"enable_thinking": True}}
            if "qwen" in self.rubric_model_name.lower() or "qwen" in self.judge_model_name.lower()
            else {}
        )
        rubric_api_base = (
            self.rubric_base_url + "/v1"
            if not self.rubric_base_url.endswith("/v1")
            else self.rubric_base_url
        )
        self.rubric_model_kwargs: dict[str, Any] = {
            "api_base": rubric_api_base,
            "api_key": self.api_key,
            **({"extra_body": shared_extra_body} if shared_extra_body else {}),
        }
        self.judge_model_kwargs: dict[str, Any] = copy.deepcopy(self.rubric_model_kwargs)
        self.rubric_bank = ScoreRubricBank(max_active_rubrics=config.max_active_rubrics)
        self.rubric_bank.initialize(self.task)
        self.experience_bank = ExperienceRubricBank(write_artifacts=True)
        self._summary_persistent_state = copy.deepcopy(EMPTY_PERSISTENT_STATE)
        self._summary_recent_segments: list[dict[str, Any]] = []
        self._summary_processed_segments = 0
        self._lane_c_condition: asyncio.Condition | None = None
        self._next_lane_c_group_index = 0

        # Captured by Lane A startup and reused by Lane C prompt construction.
        self.system_prompt = ""
        self.user_prompt = ""

    # -- session/docker plumbing ---------------------------------------------
    #
    # Copied verbatim from v0's ParallelDataSearchRunner. We don't import
    # them as bound methods because they reference v0-specific attributes;
    # but the logic is stable and proven so we keep the implementations
    # identical to avoid drift.

    def _make_initial_session(self) -> Any:
        spec = RolloutSessionSpec(
            session_id=f"lane-a-root-{uuid.uuid4().hex}",
            task=self.task,
            task_id=self.task_id,
            sample_index=0,
            policy_ref=self.policy_model_name,
            policy_version=self.policy_model_name,
            dataset_name="swebench",
            ground_truth=self.instance.get("patch"),
            raw_user_query=self.task,
            limits={
                "step_limit": (
                    self.backend.agent_config.get("step_limit", 0)
                    if hasattr(self.backend, "agent_config")
                    else 0
                )
            },
            metadata={"template_vars": copy.deepcopy(self.instance)},
        )
        session = self.backend.create_session(spec)
        # Pin model calls to the policy URL with the Lane A sampling params.
        try:
            mk = session.agent.model.config.model_kwargs
            mk["api_base"] = (
                self.policy_base_url + "/v1"
                if not self.policy_base_url.endswith("/v1")
                else self.policy_base_url
            )
            mk.setdefault("api_key", self.api_key)
            mk["temperature"] = float(self.config.policy_temperature)
            mk["top_p"] = float(self.config.policy_top_p)
            session.agent.model.config.model_name = _ensure_litellm_prefix(
                session.agent.model.config.model_name
            )
        except Exception:
            logger.warning("Could not pin api_base / model_name on Lane A session")
        self._live_sessions.append(session)
        return session

    def _step_session(self, session: Any, max_steps: int) -> dict[str, Any]:
        executed_steps = 0
        while True:
            if session._mark_paused_if_needed(
                max_steps=max_steps, executed_steps=executed_steps
            ):
                break
            session.step()
            executed_steps += 1
        final_message = session.agent.messages[-1] if session.agent.messages else {}
        final_extra = final_message.get("extra", {}) or {}
        return {
            "session_id": session.spec.session_id,
            "status": session.status,
            "exit_status": final_extra.get("exit_status", ""),
            "submission": final_extra.get("submission", ""),
            "executed_steps": executed_steps,
            "metadata": {
                "n_calls": session.agent.n_calls,
                "cost": session.agent.cost,
            },
        }

    def _make_resume_snapshot(self, session: Any) -> dict[str, Any]:
        """Build the runtime-only parent state needed to resume Lane B.

        We keep agent messages with token fields intact so route_textbased_model
        can splice prior assistant token IDs exactly. Parent events/model_turns
        are not needed for Lane B execution and are intentionally omitted.
        """
        status = "finished" if session.is_finished() else "paused"
        return {
            "session_id": session.spec.session_id,
            "status": status,
            "spec": session.spec.model_dump(mode="json"),
            "agent": session._component_state(
                session.agent, key="agent"
            ).model_dump(mode="json"),
            "model": session._component_state(
                session.agent.model, key="model"
            ).model_dump(mode="json"),
            "environment": session._component_state(
                session.agent.env, key="environment"
            ).model_dump(mode="json"),
            "last_step_index": session.last_step_index,
            "last_event_id": None,
            "metadata": {"events": [], "model_turns": []},
        }

    def _branch_initial_budget(self, parent_asst_step: int) -> int:
        return min(self.config.k, max(self.config.step_limit - parent_asst_step, 0))

    def _branch_terminal_budget(self, branch_step_end: int) -> int:
        # Match trajectory_search.py terminal patch continuation accounting.
        completed_steps = max(branch_step_end + 1, 0)
        return max(self.config.step_limit - completed_steps, 0)

    def _branch_total_budget_from_parent(self, parent_asst_step: int) -> int:
        initial_budget = self._branch_initial_budget(parent_asst_step)
        if initial_budget <= 0:
            return 0
        branch_step_end = parent_asst_step + initial_budget
        return initial_budget + self._branch_terminal_budget(branch_step_end)

    def _extract_turn_token_info_from_turns(
        self, turns: list[Any], starting_turn_index: int
    ) -> list[TurnTokenInfo]:
        result: list[TurnTokenInfo] = []
        for offset, turn in enumerate(turns, start=0):
            response_msg = getattr(turn, "response_message", None)
            response_meta = getattr(response_msg, "metadata", {}) or {}
            turn_meta = getattr(turn, "metadata", {}) or {}
            usage = response_meta.get("usage") or turn_meta.get("usage") or {}
            output_token_ids = (
                response_meta.get("output_token_ids")
                or response_meta.get("token_ids")
                or turn_meta.get("output_token_ids")
                or turn_meta.get("token_ids")
                or []
            )
            output_logprobs = (
                response_meta.get("output_logprobs")
                or response_meta.get("logprobs")
                or turn_meta.get("output_logprobs")
                or turn_meta.get("logprobs")
                or []
            )
            role = getattr(response_msg, "role", None) or "assistant"
            result.append(
                TurnTokenInfo(
                    turn_index=starting_turn_index + offset,
                    role=role,
                    prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                    output_token_ids=list(output_token_ids),
                    output_logprobs=list(output_logprobs),
                )
            )
        return result

    def _extract_terminal_patch(
        self, result: dict[str, Any], session: Any
    ) -> tuple[str, bool]:
        """(patch_text, is_fallback). Same recipe as v0."""

        def _normalize(p: str) -> str:
            p = p.rstrip()
            return p + "\n" if p else ""

        patch = _normalize(result.get("submission") or "")
        if patch:
            return patch, False
        try:
            diff = session.agent.env.execute(
                {
                    "command": (
                        'repo=$(git -C /testbed rev-parse --show-toplevel 2>/dev/null '
                        '|| git rev-parse --show-toplevel 2>/dev/null || pwd); '
                        'cd "$repo" && git add -N . >/dev/null 2>&1; git diff'
                    )
                },
                timeout=30,
            )
            return _normalize(diff.get("output") or ""), True
        except Exception:
            return "", True

    def _would_fork_overflow(self, session: Any) -> bool:
        """Predict whether a Lane B fork from current Lane A state would
        fail with sglang context-length 400 on its first /generate call.

        Lane B's first call input = tokenize(messages, add_gen_prompt=True)
        (same pure-splice path the agent uses). Compare against the
        served sglang context length minus the per-call max_new_tokens
        budget; if it doesn't fit, skip emitting the mid_cp.

        Returns True if fork would overflow. Returns False on tokenizer
        errors (fail-open: don't block emission if we can't measure)."""
        try:
            from swe_agent.tokenization import tokenize_messages_with_template
            msgs = getattr(session.agent, "messages", [])
            if not msgs:
                return False
            # Lane B's first call includes ALL of Lane A's messages so far
            # (which Lane B sees as the conversation prefix) + the new
            # `<|im_start|>assistant\n` generation prompt.
            input_ids = tokenize_messages_with_template(
                msgs,
                add_generation_prompt=True,
                model_path=self.policy_model_name,
            )
            # sglang context length: read from session's model_kwargs.
            # max_new_tokens: how much output room Lane B will request.
            try:
                mk = session.agent.model.config.model_kwargs
            except Exception:
                mk = {}
            max_new_tokens = int(mk.get("max_tokens") or mk.get("max_completion_tokens") or 4096)
            # 80960 = current --sglang-context-length in grpo.sh
            sglang_ctx = int(os.environ.get("SWE_AGENT_LANES_SGLANG_CTX", "80960"))
            need = len(input_ids) + max_new_tokens
            return need > sglang_ctx
        except Exception as exc:
            logger.warning(
                "[%s] _would_fork_overflow check FAILED, defaulting to no-skip: %s",
                self.task_id, exc,
            )
            return False

    def _commit_container(self, session: Any, tag_kind: str) -> str | None:
        """`docker commit` the session's container to a fresh image tag.
        Returns the new image tag, or None if the container is gone."""
        container_id = getattr(session.agent.env, "container_id", None) or getattr(
            session.agent.env, "_container_id", None
        )
        if not container_id:
            return None
        tag = self._new_image_tag(tag_kind)
        try:
            _docker_commit(
                self.docker_executable,
                container_id,
                tag,
                inspect_image=False,
            )
            return tag
        except Exception as exc:
            logger.warning(
                "[%s] docker commit failed for tag=%s: %s", self.task_id, tag, exc
            )
            return None

    def _new_image_tag(self, tag_kind: str) -> str:
        suffix = uuid.uuid4().hex[:6]
        tag = f"{self.image_repository}:{tag_kind}-{suffix}"
        self._created_image_tags.append(tag)
        return tag

    def _delete_image(self, image_tag: str | None) -> None:
        if not image_tag or self.config.keep_images:
            return
        if image_tag == self.base_image:
            return
        try:
            subprocess.run(
                [self.docker_executable, "image", "rm", "-f", image_tag],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except Exception as exc:
            logger.warning("Failed to delete image %s: %s", image_tag, exc)

    def _cleanup_all(self) -> None:
        for session in self._live_sessions:
            try:
                session.close()
            except Exception:
                pass
            try:
                env = getattr(getattr(session, "agent", None), "env", None)
                if env is not None and hasattr(env, "cleanup"):
                    env.cleanup()
            except Exception:
                pass
        self._live_sessions = []
        for tag in list(self._created_image_tags):
            self._delete_image(tag)
        self._created_image_tags = []

    # -- Lane B: per-mid_cp forks --------------------------------------------

    def _override_model_kwargs(
        self,
        snapshot_dict: dict[str, Any],
        base_url: str,
        temperature: float,
        top_p: float,
    ) -> None:
        """Pin a resumed snapshot's model calls to a specific sglang base_url
        and override sampling params (typically temp=1.0 for Lane B)."""
        model_section = snapshot_dict.setdefault("model", {})
        config_section = model_section.setdefault("config", {})
        kwargs = config_section.setdefault("model_kwargs", {})
        kwargs["api_base"] = (
            base_url + "/v1" if not base_url.endswith("/v1") else base_url
        )
        kwargs.setdefault("api_key", self.api_key)
        kwargs["temperature"] = float(temperature)
        kwargs["top_p"] = float(top_p)
        if "model_name" in config_section:
            config_section["model_name"] = _ensure_litellm_prefix(
                config_section["model_name"]
            )

    def _fork_lane_b(self, *, mid_cp: MidCp, branch_index: int, node_id: str | None = None) -> Any:
        """Fork a single Lane B branch from a MidCp. Returns the resumed
        session pinned to a fresh docker container (instantiated from
        mid_cp.image_tag) with Lane B sampling temperature.

        Mirrors v0's _fork_branch container-isolation contract:
        reuse_container_id=None + container_id=None + owns_container=False
        force DockerEnvironment.__init__ down the _start_container path
        so each Lane B gets its own container. Without this, all siblings
        would share Lane A's container and the first cleanup()
        would kill the rest."""
        node_id = node_id or (
            f"lane-b-g{mid_cp.idx:03d}-b{branch_index:02d}-{uuid.uuid4().hex[:6]}"
        )
        session_id = f"{node_id}-session"
        resumed = {
            "session_id": session_id,
            "status": mid_cp.snapshot["status"],
            "spec": copy.deepcopy(mid_cp.snapshot["spec"]),
            "agent": mid_cp.snapshot["agent"],
            "model": copy.deepcopy(mid_cp.snapshot["model"]),
            "environment": copy.deepcopy(mid_cp.snapshot["environment"]),
            "last_step_index": -1,
            "last_event_id": None,
            "metadata": {"events": [], "model_turns": []},
        }
        if mid_cp.snapshot.get("memory") is not None:
            resumed["memory"] = mid_cp.snapshot["memory"]
        resumed["spec"]["session_id"] = session_id
        self._override_model_kwargs(
            resumed,
            self.policy_base_url,
            temperature=self.config.lane_b_temperature,
            top_p=self.config.lane_b_top_p,
        )
        env_section = resumed.setdefault("environment", {})
        env_config = env_section.setdefault("config", {})
        env_state = env_section.setdefault("state", {})
        env_config["image"] = mid_cp.image_tag
        env_config["reuse_container_id"] = None
        env_state["container_id"] = None
        env_state["owns_container"] = False
        session = self.backend.resume_session(RolloutSnapshot(**resumed))
        self._live_sessions.append(session)
        return session

    def _run_lane_b_branch(
        self,
        *,
        mid_cp: MidCp,
        branch_index: int,
        budget_steps: int,
        total_step_limit: int,
    ) -> LaneBBranch:
        """Run a single Lane B branch end-to-end (blocking, called from
        the executor): fork, run the first budget with Lane B sampling,
        switch to terminal sampling if the branch is still live, then capture
        the final trace and terminal patch. Lane C derives first-k views from
        the final trace instead of this function materializing round state.

        GT eval is dispatched separately by `_run_fork_group` so that the
        gt_pool can be shared across all groups."""
        node_id = (
            f"lane-b-g{mid_cp.idx:03d}-b{branch_index:02d}-{uuid.uuid4().hex[:6]}"
        )
        branch = LaneBBranch(
            group_index=mid_cp.idx,
            branch_index=branch_index,
            node_id=node_id,
            parent_image_tag=mid_cp.image_tag,
            policy_model_name=self.policy_model_name,
            parent_asst_step=mid_cp.asst_step,
            started_at=time.perf_counter(),
        )
        session = None
        try:
            if budget_steps <= 0:
                raise RuntimeError(
                    f"no remaining branch budget at parent_asst_step={mid_cp.asst_step}"
                )
            session = self._fork_lane_b(
                mid_cp=mid_cp, branch_index=branch_index, node_id=node_id
            )
            branch.session_id = session.spec.session_id
            session.agent.config.step_limit = int(total_step_limit)
            before_event_count = len(session.events)
            before_message_count = mid_cp.parent_message_count
            before_turn_count = len(session.model_turns)

            result = self._step_session(session, max_steps=budget_steps)
            first_phase_steps = int(result.get("executed_steps", budget_steps) or 0)
            logger.info(
                "[%s] lane_b g=%d b=%d first_phase_done dt=%.1fs steps=%d status=%s submitted=%s",
                self.task_id,
                mid_cp.idx,
                branch_index,
                time.perf_counter() - branch.started_at,
                first_phase_steps,
                result.get("status", ""),
                result.get("exit_status") == "Submitted",
            )

            terminal_result = result
            if result.get("status") != "finished":
                branch_step_end = mid_cp.asst_step + first_phase_steps
                remaining_steps = self._branch_terminal_budget(branch_step_end)
                branch_model_kwargs = session.agent.model.config.model_kwargs
                branch_model_kwargs["temperature"] = self.config.judge_temperature
                branch_model_kwargs["top_p"] = self.config.judge_top_p
                if remaining_steps > 0:
                    terminal_result = self._step_session(session, max_steps=remaining_steps)

            result = terminal_result
            workspace_meta = _collect_workspace_meta(session.agent.env)
            messages = copy.deepcopy(
                session.agent.messages[before_message_count:]
            )
            branch.messages = messages
            branch.events = _step_card_event_dicts(session.events[before_event_count:])
            branch.workspace_meta = workspace_meta
            branch.turns = self._extract_turn_token_info_from_turns(
                session.model_turns[before_turn_count:],
                starting_turn_index=before_turn_count,
            )
            branch.total_tokens = {
                "prompt": sum(t.prompt_tokens for t in branch.turns),
                "completion": sum(t.completion_tokens for t in branch.turns),
            }
            branch.status = result.get("status", "")
            branch.terminated_early = result.get("exit_status") == "Submitted"
            branch.terminal_patch, branch.terminal_patch_from_fallback = (
                self._extract_terminal_patch(result, session)
            )
            logger.info(
                "[%s] lane_b g=%d b=%d done dt=%.1fs events=%d submitted=%s "
                "status=%s patch_len=%d tokens_p=%d tokens_c=%d",
                self.task_id, mid_cp.idx, branch_index,
                time.perf_counter() - branch.started_at,
                len(branch.events), branch.terminated_early, branch.status,
                len(branch.terminal_patch),
                branch.total_tokens.get("prompt", 0),
                branch.total_tokens.get("completion", 0),
            )
        except Exception as exc:
            branch.error = f"{type(exc).__name__}: {exc}"
            branch.status = "error"
            logger.warning(
                "[%s] lane_b g=%d b=%d FAILED dt=%.1fs %s",
                self.task_id, mid_cp.idx, branch_index,
                time.perf_counter() - branch.started_at, branch.error,
            )
        finally:
            branch.finished_at = time.perf_counter()
            if session is not None:
                try:
                    session.close()
                except Exception:
                    pass
                try:
                    env = getattr(getattr(session, "agent", None), "env", None)
                    if env is not None and hasattr(env, "cleanup"):
                        env.cleanup()
                except Exception:
                    pass
                try:
                    if session in self._live_sessions:
                        self._live_sessions.remove(session)
                except Exception:
                    pass
        return branch

    def _evaluate_gt(self, branch: LaneBBranch) -> None:
        """Score a Lane B branch's terminal patch via swebench harness.
        Same recipe as v0's _evaluate_gt (with fallback-patch penalty)."""
        t_gt = time.perf_counter()
        raw = (branch.terminal_patch or "").rstrip()
        patch = (raw + "\n") if raw else ""
        if not patch:
            branch.gt_score = 0.0
            branch.gt_payload = {"reward": 0.0, "note": "empty_patch"}
            logger.info(
                "[%s] gt_done node=%s dt=0.0s reward=0.0 note=empty_patch",
                self.task_id, branch.node_id,
            )
            return
        try:
            payload = evaluate_swebench_instance_patches(
                instance=self.instance,
                patches_by_key={branch.node_id: patch},
                model_name=self.policy_model_name,
                max_workers=1,
                namespace=self.harness_namespace,
                work_dir=self.run_dir / "groups",
            )
            branch_payload = (
                payload.get(branch.node_id, {}) if isinstance(payload, dict) else {}
            )
            raw_reward = float(branch_payload.get("reward", 0.0))
            penalty = float(self.config.fallback_patch_penalty)
            if branch.terminal_patch_from_fallback and penalty != 1.0:
                branch.gt_score = raw_reward * penalty
                branch.gt_payload = {
                    **branch_payload,
                    "raw_reward": raw_reward,
                    "fallback_penalty": penalty,
                    "note": "patch_from_git_diff_fallback",
                }
            else:
                branch.gt_payload = branch_payload
                branch.gt_score = raw_reward
            logger.info(
                "[%s] gt_done node=%s dt=%.1fs reward=%.3f",
                self.task_id, branch.node_id, time.perf_counter() - t_gt,
                branch.gt_score if branch.gt_score is not None else -1.0,
            )
        except Exception as exc:
            branch.gt_payload = {"error": f"{type(exc).__name__}: {exc}"}
            branch.gt_score = None
            logger.warning(
                "[%s] gt_FAILED node=%s dt=%.1fs %s",
                self.task_id, branch.node_id, time.perf_counter() - t_gt, exc,
            )

    # -- Lane C: rubric + judge per fork-group -------------------------------

    def _build_continuation_view_lane_b(self, branch: LaneBBranch) -> dict[str, Any]:
        """Lane C input shape per branch.

        Lane B only records its final trace. Lane C slices the first-k step
        cards here so execution and judging data derivation stay separated.
        """
        step_cards = _build_step_cards(branch.events, branch.parent_asst_step)[: self.config.k]
        workspace = branch.workspace_meta or {}
        step_start = branch.parent_asst_step
        step_end = step_start + len(step_cards)
        return {
            "node_id": branch.node_id,
            "summary": {
                "step_count": len(step_cards),
                "changed_files": list(workspace.get("changed_files", []))[:8],
                "untracked_files": list(workspace.get("untracked_files", []))[:8],
                "diff_stat": workspace.get("diff_stat", ""),
                "current_patch_chars": int(workspace.get("current_patch_chars", 0) or 0),
                "result_status": branch.status,
                "exit_status": "Submitted" if branch.terminated_early else "",
            },
            "trajectory_continuation": {
                "step_cards": step_cards,
                "segment_step_range": [step_start, step_end],
            },
        }

    async def _run_lane_c(
        self,
        *,
        group: ForkGroup,
    ) -> None:
        """Run Lane C for one fork-group in bank order."""
        cfg = self.config
        t_c = time.perf_counter()
        all_shared_step_cards = list(group.mid_cp.parent_step_cards)

        segments: list[dict[str, Any]] = []
        spr = max(1, int(cfg.k))
        for start in range(0, len(all_shared_step_cards), spr):
            end = min(start + spr, len(all_shared_step_cards))
            segments.append(
                {
                    "step_cards": copy.deepcopy(all_shared_step_cards[start:end]),
                    "segment_step_range": [start, end],
                }
            )
        summary_update_messages: list[dict[str, Any]] = []
        while self._summary_processed_segments < len(segments):
            segment = copy.deepcopy(segments[self._summary_processed_segments])
            self._summary_recent_segments.append(segment)
            self._summary_processed_segments += 1
            if len(self._summary_recent_segments) <= 1:
                continue
            evicted = self._summary_recent_segments.pop(0)
            update_payload = await _update_persistent_state(
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
                previous_state=self._summary_persistent_state,
                evicted_step_cards=evicted.get("step_cards", []),
                workspace_meta=copy.deepcopy(EMPTY_WORKSPACE_META),
                model_name=self.judge_model_name,
                temperature=cfg.judge_temperature,
                top_p=cfg.judge_top_p,
                max_tokens=cfg.psu_max_tokens,
                model_kwargs=self.judge_model_kwargs,
            )
            self._summary_persistent_state = copy.deepcopy(update_payload["state"])
            summary_update_messages.extend(copy.deepcopy(update_payload["messages"]))

        previous_state = copy.deepcopy(self._summary_persistent_state)
        recent_segments = copy.deepcopy(self._summary_recent_segments)
        latest_shared_segment = copy.deepcopy(recent_segments[-1]) if recent_segments else None
        group.persisted_parent_state = previous_state
        group.summary = {
            "persistent_state": previous_state,
            "recent_segments": recent_segments,
            "persistent_state_update_messages": summary_update_messages,
        }

        continuations = [
            self._build_continuation_view_lane_b(b) for b in group.branches
        ]
        group.lane_c_started_at = t_c
        logger.info(
            "[%s] lane_c g=%d rubric_start branches=%d",
            self.task_id, group.group_index, len(group.branches),
        )

        question = {
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "instance_id": self.task_id,
        }
        shared_context = {
            "previous_persistent_state": previous_state,
            "latest_agent_trajectory": latest_shared_segment,
        }
        existing_rubrics = []
        extra_prompt_sections: list[str] = []
        score_context = self.rubric_bank.build_generation_context()
        existing_rubrics = score_context.existing_rubrics
        extra_prompt_sections.extend(score_context.extra_prompt_sections)
        retrieved_experiences = []
        retrieve_messages: list[dict[str, Any]] = []
        if self.experience_bank.experiences:
            experience_context = await self.experience_bank.build_generation_context(
                question=question,
                previous_state=previous_state,
                latest_shared_segment=latest_shared_segment,
                continuations=continuations,
                model_name=self.rubric_model_name,
                temperature=cfg.rubric_temperature,
                top_p=cfg.rubric_top_p,
                max_tokens=cfg.rubric_max_tokens,
                model_kwargs=self.rubric_model_kwargs,
            )
            extra_prompt_sections.extend(experience_context.extra_prompt_sections)
            retrieved_experiences = copy.deepcopy(experience_context.retrieved)
            retrieve_messages = copy.deepcopy(experience_context.retrieve_messages)

        group.rubric_prompt = _build_rubric_prompt(
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
            previous_state=previous_state,
            latest_shared_segment=latest_shared_segment,
            continuations=continuations,
        )
        try:
            generated_samples = await asyncio.gather(
                *[
                    _generate_round_rubrics(
                        question=question,
                        previous_state=previous_state,
                        latest_shared_segment=latest_shared_segment,
                        continuations=continuations,
                        model_name=self.rubric_model_name,
                        temperature=cfg.rubric_temperature,
                        top_p=cfg.rubric_top_p,
                        max_tokens=cfg.rubric_max_tokens,
                        round_index=group.group_index + 1,
                        sample_index=sample_index,
                        model_kwargs=self.rubric_model_kwargs,
                        extra_prompt_sections=extra_prompt_sections,
                    )
                    for sample_index in range(max(1, cfg.n))
                ]
            )
            group.rubric_model_response = {
                "num_samples": len(generated_samples),
                "generated_rubrics": [
                    asdict(rubric)
                    for sample in generated_samples
                    for rubric in sample.generated
                ],
                "format_errors": [
                    error
                    for sample in generated_samples
                    for error in (sample.format_errors or [])
                ],
                "terminal_errors": [
                    sample.terminal_error
                    for sample in generated_samples
                    if sample.terminal_error
                ],
            }
            logger.info(
                "[%s] lane_c g=%d rubric_done dt=%.1fs rubrics=%d term_err=%s",
                self.task_id, group.group_index, time.perf_counter() - t_c,
                len(group.rubric_model_response.get("generated_rubrics", [])),
                group.rubric_model_response.get("terminal_errors"),
            )
        except Exception as exc:
            group.rubric_model_response = {"error": f"{type(exc).__name__}: {exc}"}
            logger.warning(
                "[%s] lane_c g=%d rubric_FAILED %s",
                self.task_id, group.group_index, exc,
            )
            group.lane_c_done_at = time.perf_counter()
            return

        t_judge = time.perf_counter()
        aggregate_judge_response: dict[str, dict[str, Any]] = {}
        active_ids = {rubric.rubric_id for rubric in existing_rubrics}
        group_reward_by_rubric: dict[str, float] = {}
        bank_active_before = copy.deepcopy(self.rubric_bank.active_bank)
        bank_inactive_before = copy.deepcopy(self.rubric_bank.inactive_bank)
        sample_generated_rubrics: list[list[Any]] = []
        for generated_sample in generated_samples:
            seen_rubric_ids: set[str] = set()
            generated_rubrics = []
            for rubric in generated_sample.generated:
                if rubric.rubric_id in active_ids or rubric.rubric_id in seen_rubric_ids:
                    continue
                seen_rubric_ids.add(rubric.rubric_id)
                generated_rubrics.append(rubric)
            sample_generated_rubrics.append(generated_rubrics)
        group_generated_rubrics = [
            rubric
            for generated_rubrics in sample_generated_rubrics
            for rubric in generated_rubrics
        ]
        active_score_task = asyncio.create_task(
            _score_round(
                question=question,
                shared_context=shared_context,
                continuations=continuations,
                rubrics=existing_rubrics,
                model_name=self.judge_model_name,
                temperature=cfg.judge_temperature,
                top_p=cfg.judge_top_p,
                max_tokens=cfg.judge_max_tokens,
                model_kwargs=self.judge_model_kwargs,
            )
        )
        generated_score_tasks: list[tuple[int, asyncio.Task]] = [
            (
                sample_index,
                asyncio.create_task(
                    _score_round(
                        question=question,
                        shared_context=shared_context,
                        continuations=continuations,
                        rubrics=generated_rubrics,
                        model_name=self.judge_model_name,
                        temperature=cfg.judge_temperature,
                        top_p=cfg.judge_top_p,
                        max_tokens=cfg.judge_max_tokens,
                        model_kwargs=self.judge_model_kwargs,
                    )
                ),
            )
            for sample_index, generated_rubrics in enumerate(sample_generated_rubrics)
            if generated_rubrics
        ]
        await asyncio.gather(
            active_score_task,
            *[task for _, task in generated_score_tasks],
        )
        active_scores, active_errors = active_score_task.result()
        generated_score_results: dict[int, tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]] = {
            sample_index: ([[] for _ in continuations], [])
            for sample_index in range(len(generated_samples))
        }
        for sample_index, task in generated_score_tasks:
            generated_score_results[sample_index] = task.result()
        group.rubric_samples = []
        for sample_index, generated_sample in enumerate(generated_samples):
            generated_rubrics = sample_generated_rubrics[sample_index]
            generated_scores, generated_errors = generated_score_results[sample_index]
            scored_continuations = [
                copy.deepcopy(active_records) + generated_records
                for active_records, generated_records in zip(active_scores, generated_scores)
            ]
            scoring_rubrics = existing_rubrics + generated_rubrics
            judge_errors_payload = copy.deepcopy(active_errors + generated_errors)
            child_score_lookup_by_node = {branch.node_id: {} for branch in group.branches}
            judge_response_for_sample: dict[str, dict[str, Any]] = {}
            for branch, score_records in zip(group.branches, scored_continuations):
                for record in score_records:
                    rubric_id = record["rubric_id"]
                    child_score_lookup_by_node[branch.node_id][rubric_id] = float(record["score_normalized"])
                    entry = {
                        "rubric_id": rubric_id,
                        "branch_index": branch.branch_index,
                        "node_id": branch.node_id,
                        "score_raw": record.get("score_raw"),
                        "score_normalized": record.get("score_normalized"),
                        "weighted_score": record.get("weighted_score"),
                        "judge_response": record.get("judge_response"),
                        "judge_message": record.get("judge_message"),
                    }
                    judge_response_for_sample.setdefault(rubric_id, {})[branch.node_id] = entry
                    aggregate_judge_response.setdefault(rubric_id, {})[branch.node_id] = entry
            for error in judge_errors_payload:
                rubric_id = error.get("rubric_id")
                node_id = error.get("node_id")
                if rubric_id and node_id:
                    judge_response_for_sample.setdefault(rubric_id, {}).setdefault(node_id, {})["error"] = error.get("error")
                    aggregate_judge_response.setdefault(rubric_id, {}).setdefault(node_id, {})["error"] = error.get("error")

            child_score_by_rubric: dict[str, dict[str, float]] = {}
            variance_by_rubric: dict[str, float] = {}
            redundency_by_rubric: dict[str, float] = {}
            judge_error_by_rubric: dict[str, float] = {}
            reward_by_rubric: dict[str, float] = {}
            previous_score_vectors: list[list[float]] = []
            for error in judge_errors_payload:
                rubric_id = error.get("rubric_id")
                if rubric_id:
                    judge_error_by_rubric[rubric_id] = judge_error_by_rubric.get(rubric_id, 0.0) + JUDGE_ERROR_REWARD
            for rubric in scoring_rubrics:
                vector = [
                    child_score_lookup_by_node[branch.node_id].get(rubric.rubric_id, 0.0)
                    for branch in group.branches
                ]
                child_score_by_rubric[rubric.rubric_id] = {
                    branch.node_id: child_score_lookup_by_node[branch.node_id].get(rubric.rubric_id, 0.0)
                    for branch in group.branches
                }
                variance_by_rubric[rubric.rubric_id] = 0.0 if len(vector) <= 1 else float(pvariance(vector))
                redundancy_reward = _redundancy_reward(vector, previous_score_vectors)
                redundency_by_rubric[rubric.rubric_id] = redundancy_reward
                reward_by_rubric[rubric.rubric_id] = (
                    variance_by_rubric[rubric.rubric_id]
                    + redundancy_reward
                    + judge_error_by_rubric.get(rubric.rubric_id, 0.0)
                )
                group_reward_by_rubric[rubric.rubric_id] = reward_by_rubric[rubric.rubric_id]
                previous_score_vectors.append(vector)

            bank_scoring_rubrics = scoring_rubrics
            child_rewards: dict[str, float] = {}
            for branch in group.branches:
                score_lookup = child_score_lookup_by_node[branch.node_id]
                child_rewards[branch.node_id] = (
                    sum(score_lookup.get(rubric.rubric_id, 0.0) * rubric.weight for rubric in bank_scoring_rubrics)
                    / len(bank_scoring_rubrics)
                    if bank_scoring_rubrics
                    else 0.0
                )
            generated_ids = {rubric.rubric_id for rubric in generated_sample.generated}
            sample_payload = {
                "sample_index": generated_sample.sample_index,
                "rubric_list_id": generated_sample.rubric_list_id,
                "generated": [asdict(rubric) for rubric in generated_sample.generated],
                "messages": copy.deepcopy(generated_sample.messages),
                "format_errors": copy.deepcopy(generated_sample.format_errors or []),
                "terminal_error": generated_sample.terminal_error,
                "generated_titles": [rubric.title for rubric in generated_sample.generated],
                "scoring_rubrics": [asdict(rubric) for rubric in scoring_rubrics],
                "child_score_by_rubric": {
                    rubric_id: scores
                    for rubric_id, scores in child_score_by_rubric.items()
                    if rubric_id in generated_ids or rubric_id in active_ids
                },
                "parent_score_by_rubric": {},
                "child_rewards": child_rewards,
                "parent_reward": None,
                "variance_by_rubric": {
                    rubric_id: value
                    for rubric_id, value in variance_by_rubric.items()
                    if rubric_id in generated_ids or rubric_id in active_ids
                },
                "redundency_by_rubric": {
                    rubric_id: value
                    for rubric_id, value in redundency_by_rubric.items()
                    if rubric_id in generated_ids or rubric_id in active_ids
                },
                "judge_error_by_rubric": {
                    rubric_id: judge_error_by_rubric.get(rubric_id, 0.0)
                    for rubric_id in (generated_ids | active_ids)
                },
                "reward_by_rubric": {
                    rubric_id: value
                    for rubric_id, value in reward_by_rubric.items()
                    if rubric_id in generated_ids or rubric_id in active_ids
                },
                "judge_errors": judge_errors_payload,
                "judge_response": judge_response_for_sample,
                "gt_by_rubric": {
                    rubric.rubric_id: {
                        "parent_score": None,
                        "child_scores": child_score_by_rubric.get(rubric.rubric_id, {}),
                        "ground_truth_by_node": {
                            branch.node_id: float(branch.gt_score or 0.0)
                            for branch in group.branches
                        },
                    }
                    for rubric in scoring_rubrics
                },
                "retrieved": [asdict(experience) for experience in retrieved_experiences],
                "retrieve_messages": copy.deepcopy(retrieve_messages),
                "selected": False,
            }
            group.rubric_samples.append(sample_payload)
        combined_bank_rewards = {
            rubric.rubric_id: float(rubric.reward or 0.0)
            for rubric in (bank_active_before + bank_inactive_before)
        }
        combined_bank_rewards.update(group_reward_by_rubric)
        bank_update = self.rubric_bank.update_after_round(
            generated=group_generated_rubrics,
            rewards=combined_bank_rewards,
        )
        self.rubric_bank.set_state(
            active_bank=copy.deepcopy(bank_update.active_after),
            inactive_bank=copy.deepcopy(bank_update.inactive_after),
        )
        group.judge_response = aggregate_judge_response
        group.lane_c_done_at = time.perf_counter()
        n_err = sum(
            1
            for branch_map in aggregate_judge_response.values()
            for record in branch_map.values()
            if isinstance(record, dict) and record.get("error")
        )
        logger.info(
            "[%s] lane_c g=%d judge_done dt=%.1fs calls=%d err=%d (rubrics=%d branches=%d)",
            self.task_id, group.group_index, time.perf_counter() - t_judge,
            sum(len(branch_map) for branch_map in aggregate_judge_response.values()),
            n_err,
            len(aggregate_judge_response),
            len(continuations),
        )

    async def _run_lane_c_in_order(self, group: ForkGroup) -> None:
        """Run Lane C as soon as this group is terminal, preserving rubric-bank order."""
        condition = self._lane_c_condition
        if condition is None:
            await self._run_lane_c(group=group)
            return

        async with condition:
            await condition.wait_for(
                lambda: self._next_lane_c_group_index == group.group_index
            )
        try:
            await self._run_lane_c(group=group)
        finally:
            async with condition:
                self._next_lane_c_group_index += 1
                condition.notify_all()

    async def _run_fork_group(
        self,
        *,
        mid_cp: MidCp,
        loop: asyncio.AbstractEventLoop,
        lane_b_pool: ThreadPoolExecutor,
        gt_pool: ThreadPoolExecutor,
    ) -> ForkGroup:
        """Fork m Lane B branches from mid_cp, run them all in parallel,
        evaluate GT scores concurrently. Returns the populated ForkGroup
        ready for Lane C dispatch in step 4."""
        cfg = self.config
        t_group = time.perf_counter()
        group = ForkGroup(group_index=mid_cp.idx, mid_cp=mid_cp)
        first_budget = self._branch_initial_budget(mid_cp.asst_step)
        total_budget = self._branch_total_budget_from_parent(mid_cp.asst_step)
        if first_budget <= 0 or total_budget <= 0:
            raise RuntimeError(
                f"fork_group g={mid_cp.idx} has no remaining step budget "
                f"at parent_asst_step={mid_cp.asst_step}"
            )
        logger.info(
            "[%s] fork_group g=%d start m=%d parent_image=%s first_budget=%d total_budget=%d",
            self.task_id, mid_cp.idx, cfg.m, mid_cp.image_tag, first_budget, total_budget,
        )
        branch_tasks: list[asyncio.Task] = []
        for bi in range(cfg.m):
            ctx = contextvars.copy_context()

            def _wrapped(mc=mid_cp, b=bi, c=ctx):
                return c.run(
                    self._run_lane_b_branch,
                    mid_cp=mc,
                    branch_index=b,
                    budget_steps=first_budget,
                    total_step_limit=cfg.step_limit,
                )

            future = loop.run_in_executor(lane_b_pool, _wrapped)

            async def _await_branch(branch_index=bi, fut=future):
                try:
                    return branch_index, await fut
                except BaseException as exc:
                    return branch_index, exc

            branch_tasks.append(asyncio.create_task(_await_branch()))

        branches_by_index: dict[int, LaneBBranch] = {}
        gt_tasks: list[asyncio.Task] = []

        async def _await_gt(branch: LaneBBranch, future: asyncio.Future):
            try:
                await future
            except BaseException as exc:
                branch.gt_payload = {"error": f"{type(exc).__name__}: {exc}"}
                branch.gt_score = None
                logger.warning(
                    "[%s] gt_task_FAILED node=%s %s",
                    self.task_id, branch.node_id, exc,
                )
            return branch

        for done in asyncio.as_completed(branch_tasks):
            bi, item = await done
            if isinstance(item, BaseException):
                err_branch = LaneBBranch(
                    group_index=mid_cp.idx,
                    branch_index=bi,
                    node_id=f"lane-b-g{mid_cp.idx:03d}-b{bi:02d}-err",
                    parent_image_tag=mid_cp.image_tag,
                    policy_model_name=self.policy_model_name,
                    parent_asst_step=mid_cp.asst_step,
                    error=f"executor: {type(item).__name__}: {item}",
                    status="error",
                )
                err_branch.gt_score = 0.0
                err_branch.gt_payload = {"reward": 0.0, "note": "branch_error"}
                branch = err_branch
            else:
                branch = item
            branches_by_index[bi] = branch
            group.branches = [
                branches_by_index[index] for index in sorted(branches_by_index)
            ]
            self._dump_group_scaffold(group)
            self._dump_branch(group, branch)
            if branch.error is None:
                ctx = contextvars.copy_context()
                gt_future = loop.run_in_executor(
                    gt_pool, lambda b=branch, c=ctx: c.run(self._evaluate_gt, b)
                )
                gt_tasks.append(asyncio.create_task(_await_gt(branch, gt_future)))
            else:
                self._dump_branch(group, branch)

        branches = [branches_by_index[index] for index in range(cfg.m)]
        group.branches = branches

        # GT eval starts as each branch completes. Lane C still waits for the
        # whole group and then runs in group_index order, preserving deterministic
        # score-rubric-bank state transitions while making artifacts visible early.
        for done in asyncio.as_completed(gt_tasks):
            branch = await done
            self._dump_branch(group, branch)

        # Free the mid_cp image — all branches forked from it have terminated,
        # been GT-scored. Lane C only reads the saved snapshot.
        self._delete_image(mid_cp.image_tag)
        logger.info(
            "[%s] fork_group g=%d done dt=%.1fs gt_scores=%s lane_c_dt=%.1fs",
            self.task_id, mid_cp.idx, time.perf_counter() - t_group,
            [
                round(b.gt_score, 3) if b.gt_score is not None else None
                for b in branches
            ],
            (group.lane_c_done_at - group.lane_c_started_at)
            if group.lane_c_started_at else 0.0,
        )
        return group

    # -- Lane A: linear spine ------------------------------------------------

    def _start_lane_a(self, lane_a: LaneAState) -> Any:
        """Spin up Lane A's root session, capture system+user prompts.
        Returns the session. Records started_at on `lane_a`."""
        lane_a.started_at = time.perf_counter()
        session = self._make_initial_session()
        msgs = session.agent.messages
        if msgs:
            self.system_prompt = msgs[0].get("content", "") if len(msgs) >= 1 else ""
            self.user_prompt = msgs[1].get("content", "") if len(msgs) >= 2 else ""
        return session

    def _run_lane_a_chunk(
        self,
        session: Any,
        *,
        max_steps: int,
        prev_message_count: int,
        prev_turn_count: int,
        prev_event_count: int,
    ) -> dict[str, Any]:
        """Run Lane A for up to `max_steps` agent steps. Returns a dict with:
        result            (compact run result)
        new_messages      (messages added in this chunk)
        new_step_cards    (step cards from this chunk's events)
        new_turns         (TurnTokenInfo extracted for this chunk)
        workspace_meta    (current workspace state)
        """
        result = self._step_session(session, max_steps=max_steps)
        all_msgs = session.agent.messages
        new_messages = copy.deepcopy(all_msgs[prev_message_count:])
        segment_events = _step_card_event_dicts(
            session.events[prev_event_count:]
        )
        new_step_cards = _build_step_cards(segment_events, 0)
        new_turns = self._extract_turn_token_info_from_turns(
            session.model_turns[prev_turn_count:],
            starting_turn_index=prev_turn_count,
        )
        workspace_meta = _collect_workspace_meta(session.agent.env)
        return {
            "result": result,
            "new_messages": new_messages,
            "new_step_cards": new_step_cards,
            "new_turns": new_turns,
            "workspace_meta": workspace_meta,
            "message_count": len(session.agent.messages),
            "turn_count": len(session.model_turns),
            "event_count": len(session.events),
        }

    async def _lane_a_loop(
        self,
        *,
        lane_a: LaneAState,
        loop: asyncio.AbstractEventLoop,
        lane_a_pool: ThreadPoolExecutor,
        on_mid_cp=None,
    ) -> None:
        """Drive Lane A: run in chunks of k, commit + emit
        a MidCp at each boundary, until the agent terminates or step_limit
        is reached. Lane A runs in a background thread (its agent calls
        block on sglang) so this coroutine just orchestrates."""
        cfg = self.config
        t_start = time.perf_counter()
        try:
            ctx = contextvars.copy_context()
            session = await loop.run_in_executor(
                lane_a_pool, lambda c=ctx: c.run(self._start_lane_a, lane_a)
            )
        except Exception as exc:
            lane_a.error = f"lane_a_start: {type(exc).__name__}: {exc}"
            lane_a.status = "error"
            lane_a.finished_at = time.perf_counter()
            logger.warning("[%s] lane_a_start FAILED: %s", self.task_id, lane_a.error)
            return

        # The initial state already contains the system+user prompts but no
        # assistant turns yet. Count baseline without materializing a snapshot.
        all_msgs_count = len(session.agent.messages)
        all_turns_count = len(session.model_turns)
        all_events_count = len(session.events)

        chunks_emitted = 0
        groups_emitted = 0
        cumulative_steps = 0
        last_result: dict[str, Any] | None = None

        # Root fork is round 1, matching trajectory_search's first round
        # from the root node. Later Lane A checkpoints consume the remaining
        # max_rounds budget.
        try:
            root_snap = await loop.run_in_executor(
                lane_a_pool,
                lambda s=session, c=contextvars.copy_context(): c.run(
                    self._make_resume_snapshot, s
                ),
            )
            ctx_root_commit = contextvars.copy_context()
            root_image_tag = await loop.run_in_executor(
                lane_a_pool,
                lambda s=session, c=ctx_root_commit: c.run(
                    self._commit_container, s, "mid-root"
                ),
            )
        except Exception as exc:
            logger.warning(
                "[%s] root fork emit FAILED (skipping root branch): %s",
                self.task_id, exc,
            )
            root_image_tag = None
        if (
            root_image_tag is not None
            and groups_emitted < cfg.max_rounds
            and self._branch_total_budget_from_parent(0) > 0
        ):
            root_mid_cp = MidCp(
                idx=0,
                asst_step=0,
                image_tag=root_image_tag,
                snapshot=root_snap,
                parent_messages_payload=_build_parent_messages_payload(
                    session.agent.messages,
                    model_name=self.policy_model_name,
                ),
                parent_step_cards=[],
                parent_message_count=all_msgs_count,
                parent_event_count=all_events_count,
                parent_turn_count=all_turns_count,
                emitted_at=time.perf_counter(),
            )
            lane_a.mid_cps.append(root_mid_cp)
            logger.info(
                "[%s] lane_a mid_cp emitted idx=0 image=%s asst_step=0 (ROOT)",
                self.task_id, root_image_tag,
            )
            if on_mid_cp is not None:
                try:
                    on_mid_cp(root_mid_cp)
                except Exception as exc:
                    logger.warning(
                        "[%s] on_mid_cp(root) raised: %s", self.task_id, exc,
                    )
            groups_emitted = 1

        try:
            while cumulative_steps < cfg.step_limit:
                remaining = cfg.step_limit - cumulative_steps
                chunk_size = min(cfg.k, remaining)
                t_chunk = time.perf_counter()
                ctx_c = contextvars.copy_context()
                try:
                    chunk = await loop.run_in_executor(
                        lane_a_pool,
                        lambda s=session, ms=chunk_size, pmc=all_msgs_count, ptc=all_turns_count, pec=all_events_count, c=ctx_c: c.run(
                            self._run_lane_a_chunk,
                            s,
                            max_steps=ms,
                            prev_message_count=pmc,
                            prev_turn_count=ptc,
                            prev_event_count=pec,
                        ),
                    )
                except Exception as exc:
                    lane_a.error = f"lane_a_chunk: {type(exc).__name__}: {exc}"
                    lane_a.status = "error"
                    break
                result = chunk["result"]
                new_msgs = chunk["new_messages"]
                new_step_cards = chunk["new_step_cards"]
                new_turns = chunk["new_turns"]
                last_result = result

                # Append to Lane A record.
                lane_a.messages.extend(new_msgs)
                tp = lane_a.total_tokens.get("prompt", 0)
                tc = lane_a.total_tokens.get("completion", 0)
                lane_a.total_tokens = {
                    "prompt": tp + sum(t.prompt_tokens for t in new_turns),
                    "completion": tc + sum(t.completion_tokens for t in new_turns),
                }

                # Update cumulative position trackers.
                all_msgs_count = int(chunk["message_count"])
                all_turns_count = int(chunk["turn_count"])
                all_events_count = int(chunk["event_count"])
                cumulative_steps += len(new_step_cards)
                lane_a.status = result.get("status", "")
                terminated = result.get("exit_status") == "Submitted"
                logger.info(
                    "[%s] lane_a chunk=%d dt=%.1fs new_steps=%d cumulative=%d status=%s submitted=%s",
                    self.task_id,
                    chunks_emitted,
                    time.perf_counter() - t_chunk,
                    len(new_step_cards),
                    cumulative_steps,
                    lane_a.status,
                    terminated,
                )

                if terminated:
                    lane_a.terminated_early = True
                    try:
                        lane_a.terminal_patch, lane_a.terminal_patch_from_fallback = (
                            self._extract_terminal_patch(result, session)
                        )
                    except Exception:
                        pass
                    break

                # Emit a MidCp at this boundary IFF we still have room and
                # the chunk produced at least one new step. Empty chunks
                # mean the agent stalled inside its loop — no point forking
                # from a no-progress state.
                #
                # NEW: also skip if Lane B fork from this state would
                # immediately fail with sglang context-overflow. We
                # tokenize the current Lane A messages with the agent's
                # gen-prompt suffix and check if input + max_new_tokens
                # exceeds sglang's served context length. 51963 had 113
                # Lane B branches (28% of all branches) die DOA from
                # this overflow because Lane A's prefix at late mid_cps
                # was already too close to 80960. Skipping the emit
                # saves ~docker_run + ~retry-loop wasted cost per branch.
                branch_budget = self._branch_total_budget_from_parent(cumulative_steps)
                if (
                    groups_emitted < cfg.max_rounds
                    and len(new_step_cards) > 0
                    and branch_budget > 0
                ):
                    ctx_o = contextvars.copy_context()
                    would_overflow = await loop.run_in_executor(
                        lane_a_pool,
                        lambda s=session, c=ctx_o: c.run(
                            self._would_fork_overflow, s
                        ),
                    )
                    mid_cp_idx = groups_emitted
                    if would_overflow:
                        logger.info(
                            "[%s] lane_a mid_cp emit SKIPPED idx=%d: "
                            "Lane B fork would overflow sglang context "
                            "(asst_step=%d)",
                            self.task_id, mid_cp_idx, cumulative_steps,
                        )
                        groups_emitted = cfg.max_rounds
                        chunks_emitted += 1
                        continue
                    image_tag = await loop.run_in_executor(
                        lane_a_pool,
                        lambda s=session, tk=f"mid-{mid_cp_idx:03d}", c=contextvars.copy_context(): c.run(
                            self._commit_container, s, tk
                        ),
                    )
                    if image_tag is None:
                        logger.warning(
                            "[%s] lane_a mid_cp idx=%d: docker commit returned None, "
                            "skipping fork-group emit",
                            self.task_id,
                            mid_cp_idx,
                        )
                    else:
                        resume_snapshot = await loop.run_in_executor(
                            lane_a_pool,
                            lambda s=session, c=contextvars.copy_context(): c.run(
                                self._make_resume_snapshot, s
                            ),
                        )
                        parent_step_cards = await loop.run_in_executor(
                            lane_a_pool,
                            lambda s=session, c=contextvars.copy_context(): c.run(
                                lambda: _build_step_cards(
                                    _step_card_event_dicts(s.events), 0
                                )
                            ),
                        )
                        mid_cp = MidCp(
                            idx=mid_cp_idx,
                            asst_step=cumulative_steps,
                            image_tag=image_tag,
                            snapshot=resume_snapshot,
                            parent_messages_payload=_build_parent_messages_payload(
                                session.agent.messages,
                                model_name=self.policy_model_name,
                            ),
                            parent_step_cards=parent_step_cards,
                            parent_message_count=all_msgs_count,
                            parent_event_count=all_events_count,
                            parent_turn_count=all_turns_count,
                            emitted_at=time.perf_counter(),
                        )
                        lane_a.mid_cps.append(mid_cp)
                        logger.info(
                            "[%s] lane_a mid_cp emitted idx=%d image=%s asst_step=%d",
                            self.task_id,
                            mid_cp.idx,
                            mid_cp.image_tag,
                            mid_cp.asst_step,
                        )
                        if on_mid_cp is not None:
                            try:
                                on_mid_cp(mid_cp)
                            except Exception as exc:
                                logger.warning(
                                    "[%s] on_mid_cp callback raised: %s",
                                    self.task_id,
                                    exc,
                                )
                        groups_emitted += 1

                # Soft cap: if we've already emitted max_rounds, finishing
                # the run is fine but no more forks will be scheduled.
                chunks_emitted += 1

            # Loop exit: if not terminated_early, capture best-effort terminal patch.
            if not lane_a.terminated_early:
                try:
                    lane_a.terminal_patch, lane_a.terminal_patch_from_fallback = (
                        self._extract_terminal_patch(last_result or {}, session)
                    )
                except Exception:
                    pass
        finally:
            try:
                session.close()
            except Exception:
                pass
            try:
                env = getattr(getattr(session, "agent", None), "env", None)
                if env is not None and hasattr(env, "cleanup"):
                    env.cleanup()
            except Exception:
                pass
            try:
                if session in self._live_sessions:
                    self._live_sessions.remove(session)
            except Exception:
                pass
            lane_a.finished_at = time.perf_counter()
            logger.info(
                "[%s] lane_a_done dt=%.1fs status=%s mid_cps=%d cum_steps=%d patch_len=%d",
                self.task_id,
                time.perf_counter() - t_start,
                lane_a.status,
                len(lane_a.mid_cps),
                cumulative_steps,
                len(lane_a.terminal_patch),
            )

    # -- main loop -----------------------------------------------------------

    async def run(self, *, on_group_done=None) -> InstanceRecord:
        """Run the full lane-based pipeline.

        on_group_done: optional sync callback invoked once per ForkGroup
            AFTER (a) all m Lane B branches terminated or hit the remaining
            global step budget,
            and (b) Lane C judge/rubric returned. Callback signature:
            (fork_group: ForkGroup) -> None. Used by slime's rollout fn to
            stream completed groups to the data buffer without waiting for
            the whole instance to finish.
        """
        instance_record = InstanceRecord(
            instance_id=self.task_id,
            run_dir=str(self.run_dir),
            task_id=self.task_id,
            config=asdict(self.config),
        )
        _atomic_write_json(self.run_dir / "config.json", asdict(self.config))
        started = time.perf_counter()
        logger.info(
            "[%s] run.start m=%d n=%d k=%d p=%d max_rounds=%d step_limit=%d policy_url=%s rubric_url=%s",
            self.task_id,
            self.config.m,
            self.config.n,
            self.config.k,
            self.config.p,
            self.config.max_rounds,
            self.config.step_limit,
            self.policy_base_url,
            self.rubric_base_url,
        )

        loop = asyncio.get_running_loop()
        cfg = self.config
        lane_a_pool = ThreadPoolExecutor(
            max_workers=cfg.lane_a_pool_size,
            thread_name_prefix=f"lane-a-{self.task_id[:12]}",
        )
        lane_b_pool = ThreadPoolExecutor(
            max_workers=cfg.lane_b_pool_size or (cfg.m * cfg.max_rounds),
            thread_name_prefix=f"lane-b-{self.task_id[:12]}",
        )
        gt_pool = ThreadPoolExecutor(
            max_workers=cfg.gt_eval_workers,
            thread_name_prefix=f"lane-gt-{self.task_id[:12]}",
        )
        self._lane_c_condition = asyncio.Condition()
        self._next_lane_c_group_index = 0
        # Per-group on_group_done fires AS SOON AS that group is ready
        # (all m Lane B branches terminal + GT + Lane C done). Slime rollout
        # reads these to stream completed groups without waiting for the whole
        # instance. Lane C is serialized by group index only for rubric-bank
        # state; it does not block Lane A or other groups' Lane B rollout.

        group_tasks: list[asyncio.Task] = []

        async def _await_group_then_callback(group_index: int, coro):
            try:
                grp = await coro
            except Exception as exc:
                logger.warning("[%s] fork-group raised: %s", self.task_id, exc)
                condition = self._lane_c_condition
                if condition is not None:
                    async with condition:
                        if self._next_lane_c_group_index == group_index:
                            self._next_lane_c_group_index += 1
                            condition.notify_all()
                return None
            self._dump_group(grp)
            await self._run_lane_c_in_order(grp)
            self._dump_group(grp)
            if on_group_done is not None:
                try:
                    on_group_done(grp)
                except Exception as cb_exc:
                    logger.warning(
                        "[%s] on_group_done raised: %s", self.task_id, cb_exc
                    )
            return grp

        def _on_mid_cp(mid_cp: MidCp) -> None:
            # Fire-and-track: each mid_cp emission spawns an async task that
            # forks m Lane B branches, runs them to termination, then
            # GT + Lane C concurrently. We do NOT await here — Lane A
            # keeps producing more mid_cps.
            coro = self._run_fork_group(
                mid_cp=mid_cp,
                loop=loop,
                lane_b_pool=lane_b_pool,
                gt_pool=gt_pool,
            )
            group_tasks.append(
                asyncio.create_task(_await_group_then_callback(mid_cp.idx, coro))
            )

        try:
            await self._lane_a_loop(
                lane_a=instance_record.lane_a,
                loop=loop,
                lane_a_pool=lane_a_pool,
                on_mid_cp=_on_mid_cp,
            )
            # Lane A done. Wait for all fork-groups it spawned to finish.
            if group_tasks:
                logger.info(
                    "[%s] lane_a finished, awaiting %d fork-groups",
                    self.task_id, len(group_tasks),
                )
                t_drain = time.perf_counter()
                groups_or_excs = await asyncio.gather(
                    *group_tasks, return_exceptions=True
                )
                for item in groups_or_excs:
                    if isinstance(item, BaseException):
                        logger.warning(
                            "[%s] fork-group raised: %s", self.task_id, item
                        )
                        continue
                    if item is None:
                        # _await_group_then_callback swallowed the exception
                        # and logged it; on_group_done still got skipped.
                        continue
                    instance_record.groups.append(item)
                # Sort by group_index for deterministic disk layout.
                instance_record.groups.sort(key=lambda g: g.group_index)
                logger.info(
                    "[%s] fork-groups drained dt=%.1fs n_groups=%d",
                    self.task_id,
                    time.perf_counter() - t_drain,
                    len(instance_record.groups),
                )
            rubric_payloads = [
                {
                    **copy.deepcopy(sample),
                    "group_index": group.group_index,
                    "messages": copy.deepcopy(sample.get("messages", [])),
                }
                for group in instance_record.groups
                for sample in group.rubric_samples
            ]
            experience_update_payload = await self.experience_bank.update_after_instance(
                run_dir=self.run_dir,
                instance=self.instance,
                rubric_payloads=rubric_payloads,
                model_name=self.rubric_model_name,
                temperature=cfg.rubric_temperature,
                top_p=cfg.rubric_top_p,
                max_tokens=cfg.rubric_max_tokens,
                model_kwargs=self.rubric_model_kwargs,
            )
            experience_updates_by_group = {
                int(update["group_index"]): update
                for update in experience_update_payload.get("groups", [])
            }
            for group in instance_record.groups:
                update = experience_updates_by_group.get(group.group_index)
                if update is None:
                    continue
                group.experience_bank_update = {
                    "before": copy.deepcopy(update["before"]),
                    "actions": copy.deepcopy(update["actions"]),
                    "after": copy.deepcopy(update["after"]),
                }
                group.experience_bank_message = copy.deepcopy(update["messages"])
            instance_record.completed = (
                instance_record.lane_a.error is None
                and all(
                    all(b.error is None for b in g.branches)
                    for g in instance_record.groups
                )
            )
            if not instance_record.completed and instance_record.error is None:
                instance_record.error = (
                    instance_record.lane_a.error
                    or "one or more lane_b branches errored"
                )
        except Exception as exc:
            instance_record.error = f"runner exception: {exc}\n{traceback.format_exc()}"
            instance_record.completed = False
        finally:
            instance_record.seconds = time.perf_counter() - started
            # Lane A is the spine; not trained, not dumped as its own dir.
            # Each group writes its own shared_parent_message.json (sliced from the
            # MidCp snapshot). Lane A's terminal patch + per-mid_cp markers
            # are summarized in top-level config.json.
            for grp in instance_record.groups:
                self._dump_group(grp)
            self._dump_instance_record(instance_record)
            for pool in (lane_a_pool, lane_b_pool, gt_pool):
                try:
                    pool.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
            self._cleanup_all()
        return instance_record

    # -- persistence ---------------------------------------------------------

    def _group_dir(self, group: ForkGroup) -> Path:
        return self.run_dir / "groups" / f"group_{group.group_index:03d}"

    def _dump_group_scaffold(self, group: ForkGroup) -> Path:
        gdir = self._group_dir(group)
        gdir.mkdir(parents=True, exist_ok=True)
        parent_payload = group.mid_cp.parent_messages_payload
        if not parent_payload:
            parent_messages = list(
                (group.mid_cp.snapshot.get("agent", {}).get("state", {}) or {})
                .get("messages", [])
            )
            parent_payload = _build_parent_messages_payload(
                parent_messages,
                model_name=self.policy_model_name,
            )
        _atomic_write_json(
            gdir / "shared_parent_message.json",
            parent_payload,
        )
        _atomic_write_json(gdir / "summary.json", group.summary)
        if group.experience_bank_update:
            _atomic_write_json(gdir / "rubric_bank.json", group.experience_bank_update)
            _atomic_write_json(gdir / "rubric_bank_message.json", group.experience_bank_message)
        return gdir

    def _dump_branch(self, group: ForkGroup, branch: LaneBBranch) -> None:
        gdir = self._group_dir(group)
        bdir = gdir / "branches" / f"branch_{branch.branch_index:02d}"
        bdir.mkdir(parents=True, exist_ok=True)
        branch_step_cards = _build_step_cards(branch.events, branch.parent_asst_step)
        stamped, _ = _stamp_steps(branch.messages, start_step=0)
        _atomic_write_json(
            bdir / "message.json",
            build_messages(
                stamped,
                model_name=self.policy_model_name,
                preserve_token_fields=True,
            ),
        )
        _atomic_write_json(
            bdir / "terminal_patch.json",
            {
                self.task_id: {
                    "model_name_or_path": self.policy_model_name,
                    "instance_id": self.task_id,
                    "model_patch": branch.terminal_patch or "",
                    "terminal_error": branch.error,
                    "patch_from_fallback": branch.terminal_patch_from_fallback,
                }
            },
        )
        _atomic_write_json(
            bdir / "terminal_evaluation.json",
            {
                "gt_score": branch.gt_score,
                "gt_payload": branch.gt_payload,
            },
        )
        _atomic_write_json(
            bdir / "node.json",
            {
                "node_id": branch.node_id,
                "parent_id": f"group_{group.group_index:03d}",
                "round_index": group.group_index + 1,
                "depth": 1,
                "session_id": branch.session_id or f"{branch.node_id}-session",
                "status": branch.status or ("error" if branch.error else "finished"),
                "step_start": branch.parent_asst_step,
                "step_end": branch.parent_asst_step + len(branch_step_cards),
                "submission": (
                    branch.terminal_patch
                    if branch.terminated_early and not branch.terminal_patch_from_fallback
                    else ""
                ),
                "exit_status": "Submitted" if branch.terminated_early else "",
                "policy_source": "student",
                "policy_model_name": self.policy_model_name,
                "branch_index": branch.branch_index,
                "group_index": group.group_index,
                "parent_image_tag": branch.parent_image_tag,
                "workspace_meta": branch.workspace_meta,
                "terminated_early": branch.terminated_early,
                "terminal_patch_from_fallback": branch.terminal_patch_from_fallback,
                "terminal_patch_chars": len(branch.terminal_patch or ""),
                "error": branch.error,
                "n_messages": len(branch.messages),
                "n_step_cards": len(branch_step_cards),
                "total_tokens": branch.total_tokens,
                "started_at": branch.started_at,
                "finished_at": branch.finished_at,
            },
        )

    def _dump_group(self, group: ForkGroup) -> None:
        """Persist one fork-group under run_dir/groups/group_NNN/."""
        gdir = self._dump_group_scaffold(group)
        rubrics_dir = gdir / "rubrics"
        rubrics_dir.mkdir(parents=True, exist_ok=True)
        for sample in group.rubric_samples:
            rubric_dir = rubrics_dir / str(sample.get("rubric_list_id", f"sample-{sample.get('sample_index', 0):02d}"))
            rubric_dir.mkdir(parents=True, exist_ok=True)
            rubric_payload = {
                k: copy.deepcopy(v)
                for k, v in sample.items()
                if k not in {"messages", "judge_response", "retrieve_messages"}
            }
            _atomic_write_json(rubric_dir / "rubric.json", rubric_payload)
            _atomic_write_json(rubric_dir / "rubric_message.json", sample.get("messages", []))
            _atomic_write_json(rubric_dir / "rubric_retrieve_message.json", sample.get("retrieve_messages", []))
            _atomic_write_json(
                rubric_dir / "judge.json",
                {
                    "rubric_list_id": sample.get("rubric_list_id"),
                    "judge_response": sample.get("judge_response", {}),
                    "judge_errors": sample.get("judge_errors", []),
                    "child_score_by_rubric": sample.get("child_score_by_rubric", {}),
                    "judge_error_by_rubric": sample.get("judge_error_by_rubric", {}),
                },
            )
        for branch in group.branches:
            self._dump_branch(group, branch)

    def _dump_instance_record(self, instance_record: InstanceRecord) -> None:
        path = self.run_dir / "config.json"
        gt_scores_per_group: list[list[float | None]] = []
        for g in instance_record.groups:
            gt_scores_per_group.append([b.gt_score for b in g.branches])
        _atomic_write_json(
            path,
            {
                "instance_id": instance_record.instance_id,
                "task_id": instance_record.task_id,
                "run_dir": instance_record.run_dir,
                "completed": instance_record.completed,
                "error": instance_record.error,
                "seconds": instance_record.seconds,
                "num_groups": len(instance_record.groups),
                "num_mid_cps": len(instance_record.lane_a.mid_cps),
                "config": instance_record.config,
                "lane_a": {
                    "status": instance_record.lane_a.status,
                    "error": instance_record.lane_a.error,
                    "terminated_early": instance_record.lane_a.terminated_early,
                    "terminal_patch_chars": len(instance_record.lane_a.terminal_patch or ""),
                    "terminal_patch_from_fallback": instance_record.lane_a.terminal_patch_from_fallback,
                    "total_tokens": instance_record.lane_a.total_tokens,
                    "num_messages": len(instance_record.lane_a.messages),
                    "num_mid_cps": len(instance_record.lane_a.mid_cps),
                    "started_at": instance_record.lane_a.started_at,
                    "finished_at": instance_record.lane_a.finished_at,
                },
                "groups": [
                    {
                        "group_index": g.group_index,
                        "mid_cp": {
                            "idx": g.mid_cp.idx,
                            "image_tag": g.mid_cp.image_tag,
                            "asst_step": g.mid_cp.asst_step,
                            "emitted_at": g.mid_cp.emitted_at,
                        },
                        "num_branches": len(g.branches),
                        "num_rubric_samples": len(g.rubric_samples),
                        "branches": [
                            {
                                "branch_index": b.branch_index,
                                "node_id": b.node_id,
                                "status": b.status,
                                "gt_score": b.gt_score,
                                "n_step_cards": len(_build_step_cards(b.events, b.parent_asst_step)),
                                "terminated_early": b.terminated_early,
                                "error": b.error,
                            }
                            for b in g.branches
                        ],
                    }
                    for g in instance_record.groups
                ],
                "gt_scores_per_group": gt_scores_per_group,
            },
        )
