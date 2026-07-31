"""Training-oriented Lane A / Lane B / Lane C trajectory generation.

The topology is deliberately much smaller than TTS search:

* ``depth1`` emits one GRPO group.  Lane B independently samples ``m`` strict
  ``k``-step continuations from the root and Lane C judges the complete group.
* ``depth2`` emits the same root group plus one beam group.  Lane A samples
  ``p`` independent strict ``k``-step parents from the root.  Lane B samples
  ``m / p`` strict ``k``-step children from each parent, and Lane C judges all
  ``m`` parent+child trajectories as one group.

There is no best-branch selection and no parent-child (PC) reward.  A parent
is a randomly sampled policy continuation and is part of every corresponding
child's trainable response.  Terminal continuation and SWE-bench evaluation
are controlled by one validation-only flag and are disabled for training by
default.
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent_rl import RolloutSessionSpec, RolloutSnapshot
from swe_agent.backend import SWEAgentRolloutBackend

# Reuse shared artifact/prompt helpers plus trajectory_search's rubric,
# judge, and persistent-state implementations so the parallel path stays
# aligned with the non-parallel search semantics.
from swe_agent.parallel_utils import (
    TurnTokenInfo,
    _ensure_litellm_prefix,
    _stamp_steps,
    _atomic_write_json,
    extract_terminal_patch_from_session,
)
from swe_agent.prompt import (
    EMPTY_PERSISTENT_STATE,
    EMPTY_WORKSPACE_META,
    SWE_TRAJECTORY_RUBRIC_GENERATION_PROMPT,
    SWE_TRAJECTORY_RUBRIC_JUDGE_PROMPT,
)
from swe_agent.run.run_swe_agent import (
    EvaluationRewardConfig,
    _litellm_model_kwargs,
    build_messages,
    evaluate_swebench_instance_patches,
    make_evaluation_payload,
)
from swe_agent.rubric_bank import ExperienceRubricBank, ScoreRubricBank
from swe_agent.usage import usage_context
from swe_agent.trajectory_search import (
    _avg_scores_from_rubrics,
    _generate_and_score_rubric_batch,
    _run_score_tie_break,
    _rubric_metrics_payload,
    _rubric_sample_evaluation,
    _update_persistent_state,
    _build_step_cards,
    _collect_workspace_meta,
    _container_environment_kind,
    _docker_commit,
)


logger = logging.getLogger("swe_agent.trajectory_search_parallel")

_POLICY_OVERLENGTH_EXIT_STATUSES = frozenset(
    {"ContextWindowExceeded", "CompletionLengthExceeded"}
)


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


def _score_map_errors(
    scores: Any,
    *,
    node_ids: list[str],
    label: str,
) -> list[str]:
    """Validate exact, finite score coverage for the real runtime nodes."""
    errors: list[str] = []
    expected = set(node_ids)
    if len(expected) != len(node_ids):
        errors.append(f"{label}:duplicate_expected_node_ids")
    if not isinstance(scores, dict):
        errors.append(f"{label}:not_a_mapping")
        return errors
    actual = set(scores)
    if actual != expected:
        missing = sorted(str(node_id) for node_id in expected - actual)
        extra = sorted(str(node_id) for node_id in actual - expected)
        errors.append(f"{label}:coverage:missing={missing}:extra={extra}")
    for node_id in expected & actual:
        if isinstance(scores[node_id], bool) or not isinstance(
            scores[node_id], (int, float)
        ):
            errors.append(f"{label}:non_numeric:{node_id}")
            continue
        try:
            score = float(scores[node_id])
        except (TypeError, ValueError, OverflowError):
            errors.append(f"{label}:non_numeric:{node_id}")
            continue
        if not math.isfinite(score):
            errors.append(f"{label}:non_finite:{node_id}:{score}")
    return errors


def _tie_break_errors(
    payload: dict[str, Any] | None,
    *,
    node_ids: list[str],
) -> list[str]:
    """Return fatal tie-break errors while retaining recovery diagnostics."""
    if payload is None:
        return []
    errors: list[str] = []
    status = str(payload.get("status") or "")
    if status not in {"success", "fallback"}:
        errors.append(f"tie_break_status:{status or 'missing'}")
    # The oracle records corrected format turns and generation-cap terminals
    # even when a later rubric is usable. They remain in the artifact payload
    # but are not failures by themselves.
    for error in payload.get("judge_errors") or []:
        errors.append(f"tie_break_judge:{error}")
    for message in payload.get("judge_messages") or []:
        if isinstance(message, dict) and message.get("error"):
            errors.append(f"tie_break_judge:{message['error']}")
    if status == "success":
        errors.extend(
            _score_map_errors(
                payload.get("adjusted_scores"),
                node_ids=node_ids,
                label="tie_break_adjusted_scores",
            )
        )
    return errors


def _unexpected_rollout_exit(
    result: dict[str, Any] | None, *, lane: str
) -> str | None:
    """Classify clean backend returns that nevertheless ended abnormally.

    SWE-agent represents preflight context overflow and limits exhaustion as
    structured exit messages, so ``session.step`` returns normally instead of
    raising.  Empty exit status means the strict-k budget paused normally;
    ``Submitted`` is the only successful terminal status.
    """
    exit_status = str((result or {}).get("exit_status") or "").strip()
    if (
        not exit_status
        or exit_status == "Submitted"
        or exit_status in _POLICY_OVERLENGTH_EXIT_STATUSES
    ):
        return None
    return f"{lane}_exit_status:{exit_status}"


def _policy_overlength_exit_status(
    result: dict[str, Any] | None,
) -> str | None:
    exit_status = str((result or {}).get("exit_status") or "").strip()
    return (
        exit_status
        if exit_status in _POLICY_OVERLENGTH_EXIT_STATUSES
        else None
    )


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
    """Parallel training rollout configuration.

    ``m`` is always the GRPO group cardinality.  In ``depth2``, ``p`` is the
    number of independently sampled Lane-A parents and must divide ``m``.
    """

    m: int = 8  # forks per mid_cp (NOT M-1)
    n: int = 1  # rubric-list samples per group
    k: int = 20  # assistant turns between Lane A checkpoints
    p: int = 2  # Lane-A beam parents in depth2
    topology: str = "depth1"
    max_rounds: int = 1  # retained for CLI compatibility; topology is explicit
    step_limit: int = 100  # hard cap for the whole trajectory
    max_active_rubrics: int = 6

    # Lane A sampling — runs deterministic-ish spine; lower temp is fine.
    policy_temperature: float = 1.0
    policy_top_p: float = 0.95
    # Lane B sampling — higher for intra-group diversity (we WANT different
    # solution attempts from the same parent state).
    lane_b_temperature: float = 1.0
    lane_b_top_p: float = 0.95

    # Hosted GLM can spend more than the completed TTS launch's 8096-token
    # budget on reasoning alone.  Keep enough room for its final structured
    # answer; lower values remain available as explicit caller overrides.
    rubric_temperature: float = 1.0
    rubric_top_p: float = 0.95
    rubric_max_tokens: int = 20480
    judge_temperature: float = 0.01
    judge_top_p: float = 0.95
    judge_max_tokens: int = 20480
    psu_max_tokens: int = 20480

    gt_eval_workers: int = 8
    lane_a_pool_size: int = 1  # Lane A is a single agent
    lane_b_pool_size: int | None = None  # default = m * max_rounds

    keep_images: bool = False
    return_logprobs: bool = True
    terminal_rollout: bool = False
    reward_kind: str = "joint"
    joint_alpha: float = 1.0
    all_pass_reward: float = 1.0
    disable_rubric: bool = False
    # Same recipe as v0: multiplicative penalty when the agent never invoked
    # the formal submit command and we fell back to `git diff` of the
    # working copy.
    fallback_patch_penalty: float = 0.5
    no_action_patch_penalty: float = -0.1
    score_tie_break: bool = True

    def __post_init__(self) -> None:
        if self.p <= 0:
            raise ValueError("trajectory_search_parallel requires p > 0.")
        if self.m <= 0:
            raise ValueError("trajectory_search_parallel requires m > 0.")
        if self.n <= 0:
            raise ValueError("trajectory_search_parallel requires n > 0.")
        if self.k <= 0:
            raise ValueError("trajectory_search_parallel requires k > 0.")
        if self.max_rounds <= 0:
            raise ValueError("trajectory_search_parallel requires max_rounds > 0.")
        if self.step_limit <= 0:
            raise ValueError("trajectory_search_parallel requires step_limit > 0.")
        if self.gt_eval_workers <= 0:
            raise ValueError("trajectory_search_parallel requires gt_eval_workers > 0.")
        if self.lane_a_pool_size <= 0:
            raise ValueError("trajectory_search_parallel requires lane_a_pool_size > 0.")
        if self.lane_b_pool_size is not None and self.lane_b_pool_size <= 0:
            raise ValueError("trajectory_search_parallel requires lane_b_pool_size > 0 when set.")
        if self.reward_kind not in {"hard", "soft", "joint", "f2p_only"}:
            raise ValueError(f"unsupported trajectory_search_parallel reward_kind={self.reward_kind!r}")
        if self.topology not in {"depth1", "depth2"}:
            raise ValueError(
                "trajectory_search_parallel topology must be 'depth1' or 'depth2'"
            )
        if self.topology == "depth2" and self.m % self.p:
            raise ValueError(
                "trajectory_search_parallel depth2 requires m divisible by p "
                f"(got m={self.m}, p={self.p})"
            )


@dataclass
class MidCp:
    """Snapshot Lane A emits every `k` assistant turns.

    Used as the fork point for `m` Lane B branches. The image_tag field stores
    either a Docker image tag or a Singularity sandbox path that captures the
    container filesystem state; the snapshot dict is runtime-only
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
    node_id: str = ""


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
    # Multiplicative reward adjustment for a policy overlength terminal.  GT
    # rewards already apply this inside EvaluationRewardConfig; hosted-judge
    # rewards apply it in lane_to_grpo_bundle.
    reward_penalty: float = 1.0
    overlength_reason: str | None = None
    error: str | None = None
    gt_score: float | None = None
    gt_payload: dict[str, Any] | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    # ``None`` for the root group; 0..p-1 for a depth2 child.
    beam_parent_index: int | None = None
    beam_parent_node_id: str | None = None


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
    group_kind: str = "root"
    # Empty for depth1/root.  For a beam group these are the independently
    # sampled Lane-A parent checkpoints, in parent-index order.
    parent_mid_cps: list[MidCp] = field(default_factory=list)
    rubric_model_response: dict[str, Any] = field(default_factory=dict)
    rubric_samples: list[dict[str, Any]] = field(default_factory=list)
    judge_response: dict[str, Any] = field(default_factory=dict)
    # Final oracle-equivalent score, keyed by the real runtime node id.
    judge_score_by_node: dict[str, float] = field(default_factory=dict)
    lane_c_errors: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    summary_messages: list[dict[str, Any]] = field(default_factory=list)
    experience_bank_update: dict[str, Any] = field(default_factory=dict)
    experience_bank_message: list[dict[str, Any]] = field(default_factory=list)
    persisted_parent_state: dict[str, Any] = field(default_factory=dict)
    lane_c_started_at: float = 0.0
    lane_c_done_at: float = 0.0
    bundled_at: float = 0.0


@dataclass
class LaneAState:
    """Lane-A parent-generation record for inspection and usage accounting."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    mid_cps: list[MidCp] = field(default_factory=list)
    terminal_patch: str = ""
    terminal_patch_from_fallback: bool = False
    terminated_early: bool = False
    status: str = ""
    error: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    total_tokens: dict[str, int] = field(default_factory=lambda: {"prompt": 0, "completion": 0})


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
        policy_version: str | None = None,
        enforce_policy_version: bool = False,
        rubric_model_name: str | None,
        judge_model_name: str | None,
        config: ParallelSearchConfig,
        harness_namespace: str | None,
        policy_base_url: str,
        rubric_base_url: str,
        api_key: str = "EMPTY",
        policy_api_key: str | None = None,
        rubric_api_key: str | None = None,
        usage_group_prefix: str | None = None,
        policy_base_urls: list[str] | None = None,
        score_banks: dict[str, ScoreRubricBank] | None = None,
        experience_banks: dict[str, ExperienceRubricBank] | None = None,
    ) -> None:
        self.instance = copy.deepcopy(instance)
        self.backend = backend
        self.run_dir = Path(run_dir)
        self.policy_model_name = policy_model_name
        self.policy_version = policy_version or policy_model_name
        self.enforce_policy_version = bool(enforce_policy_version)
        self.rubric_model_name = rubric_model_name or policy_model_name
        self.judge_model_name = judge_model_name or self.rubric_model_name
        self.config = config
        self.harness_namespace = harness_namespace
        self.policy_base_url = policy_base_url.rstrip("/")
        self.policy_base_urls = [
            url.rstrip("/") for url in (policy_base_urls or [self.policy_base_url])
        ]
        self.rubric_base_url = rubric_base_url.rstrip("/")
        # Keep the legacy ``api_key`` argument as a fallback, but never route
        # the remote judge credential into local policy requests.
        self.policy_api_key = policy_api_key or api_key
        self.rubric_api_key = rubric_api_key or api_key
        self.skip_lane_c = bool(config.disable_rubric)

        self.task = instance["problem_statement"]
        self.task_id = instance["instance_id"]
        requested_usage_prefix = (
            str(usage_group_prefix).strip()
            if usage_group_prefix is not None
            else ""
        )
        self.usage_group_prefix = requested_usage_prefix or self.task_id
        self.run_id = f"{self.task_id}-{time.strftime('%Y%m%d-%H%M%S')}"
        self.base_image = str(self.backend.environment_config["image"])
        self.environment_class = _container_environment_kind(
            str(self.backend.environment_config.get("environment_class", "docker"))
        )
        self.docker_executable = str(
            self.backend.environment_config.get("executable", "docker")
        )
        # Image-tag namespace per instance, same convention as v0.
        self.image_repository = f"rler-pds/{self.task_id.replace('__', '-').lower()}"

        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "groups").mkdir(parents=True, exist_ok=True)
        self._created_image_tags: list[str] = []
        self._created_sandbox_dirs: list[Path] = []
        self._live_sessions: list[Any] = []
        provided_score_banks = score_banks if score_banks is not None else {}
        provided_experience_banks = experience_banks if experience_banks is not None else {}
        rubric_api_base = (
            self.rubric_base_url + "/v1"
            if not self.rubric_base_url.endswith("/v1")
            else self.rubric_base_url
        )
        self.rubric_model_kwargs: dict[str, Any] = {
            **_litellm_model_kwargs(self.rubric_model_name),
            "api_base": rubric_api_base,
            "api_key": self.rubric_api_key,
            "completion_backend": "litellm",
        }
        self.judge_model_kwargs: dict[str, Any] = {
            **_litellm_model_kwargs(self.judge_model_name),
            "api_base": rubric_api_base,
            "api_key": self.rubric_api_key,
            "completion_backend": "litellm",
        }
        # Training uses one sibling-judging scope only.  PC was a TTS search
        # signal and is intentionally absent from the parallel trainer.
        self.rubric_scopes = ("siblings",)
        self.score_banks: dict[str, ScoreRubricBank] = {}
        self.experience_banks: dict[str, ExperienceRubricBank] = {}
        for scope in self.rubric_scopes:
            score_bank = provided_score_banks.get(scope)
            if score_bank is None:
                score_bank = ScoreRubricBank(max_active_rubrics=config.max_active_rubrics, scope=scope)
            self.score_banks[scope] = score_bank
            experience_bank = provided_experience_banks.get(scope)
            if not self.skip_lane_c:
                if not isinstance(experience_bank, ExperienceRubricBank):
                    raise ValueError(
                        "Parallel rubric training requires a frozen siblings bank"
                    )
                self.experience_banks[scope] = experience_bank
        self.rubric_bank = self.score_banks["siblings"]
        self.experience_bank = self.experience_banks.get("siblings")
        self._summary_persistent_state = copy.deepcopy(EMPTY_PERSISTENT_STATE)
        self._summary_recent_segments: list[dict[str, Any]] = []
        self._summary_processed_segments = 0
        self._lane_c_condition: asyncio.Condition | None = None
        self._next_lane_c_group_index = 0

        # Captured by Lane A startup and reused by Lane C prompt construction.
        self.system_prompt = ""
        self.user_prompt = ""

    def _usage_group_id(self, group_index: int) -> str:
        return f"{self.usage_group_prefix}:g{group_index}"

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
            policy_version=(
                self.policy_version if self.enforce_policy_version else None
            ),
            dataset_name="swebench",
            ground_truth=self.instance.get("patch"),
            raw_user_query=self.task,
            limits={
                "step_limit": self.backend.agent_config.get("step_limit", 0)
            },
            metadata={"template_vars": copy.deepcopy(self.instance)},
        )
        session = self.backend.create_session(spec)
        # Pin model calls to the policy URL with the Lane A sampling params.
        mk = session.agent.model.config.model_kwargs
        mk["api_base"] = (
            self.policy_base_url + "/v1"
            if not self.policy_base_url.endswith("/v1")
            else self.policy_base_url
        )
        mk["api_key"] = self.policy_api_key
        mk["temperature"] = float(self.config.policy_temperature)
        mk["top_p"] = float(self.config.policy_top_p)
        session.agent.model.config.model_name = _ensure_litellm_prefix(
            session.agent.model.config.model_name
        )
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
        return extract_terminal_patch_from_session(result, session)

    @staticmethod
    def _is_context_window_error(exc: Exception) -> bool:
        message = str(exc)
        return (
            "ContextWindowExceeded" in message
            or "maximum context length" in message
            or "context length" in message
            or "Requested token count exceeds" in message
        )

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
            max_new_tokens = int(
                mk.get("max_tokens")
                or mk.get("max_completion_tokens")
                or 20480
            )
            sglang_ctx = int(
                os.environ.get("SWE_AGENT_LANES_SGLANG_CTX", "128000")
            )
            need = len(input_ids) + max_new_tokens
            return need > sglang_ctx
        except Exception as exc:
            logger.warning(
                "[%s] _would_fork_overflow check FAILED, defaulting to no-skip: %s",
                self.task_id, exc,
            )
            return False

    def _copy_singularity_sandbox(self, source: Path, tag_kind: str) -> Path:
        if not source.exists():
            raise FileNotFoundError(f"Singularity sandbox does not exist: {source}")
        safe_task_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in self.task_id)[:80]
        dst = Path(tempfile.gettempdir()) / f"swe-agent-{safe_task_id}-{tag_kind}-{uuid.uuid4().hex[:6]}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.mkdir(parents=True, exist_ok=False)
        try:
            subprocess.run(
                ["cp", "-a", "--reflink=auto", f"{source}/.", str(dst)],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            shutil.rmtree(dst, ignore_errors=True)
            raise
        self._created_sandbox_dirs.append(dst)
        return dst

    def _checkpoint_environment(self, session: Any, tag_kind: str) -> str | None:
        """Checkpoint the session filesystem for future Lane B forks."""
        env = session.agent.env
        if self.environment_class == "singularity":
            sandbox_dir = getattr(env, "sandbox_dir", None)
            if sandbox_dir is None:
                raise RuntimeError("Singularity trajectory-search session has no sandbox_dir to checkpoint")
            return str(self._copy_singularity_sandbox(Path(sandbox_dir), tag_kind))

        container_id = getattr(env, "container_id", None) or getattr(
            env, "_container_id", None
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

    def _delete_image(self, image_tag: str) -> None:
        if self.environment_class != "docker":
            raise RuntimeError("Docker image cleanup was requested for a Singularity trajectory-search runner")
        if self.config.keep_images:
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

    def _delete_checkpoint(self, checkpoint_ref: str) -> None:
        if self.environment_class == "singularity":
            path = Path(checkpoint_ref)
            if not path.is_absolute():
                raise RuntimeError(f"Singularity checkpoint is not an absolute sandbox path: {checkpoint_ref}")
            if not self.config.keep_images:
                shutil.rmtree(path, ignore_errors=True)
            return
        self._delete_image(checkpoint_ref)

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
        if self.environment_class == "docker":
            for tag in list(self._created_image_tags):
                self._delete_image(tag)
            self._created_image_tags = []
        else:
            for sandbox_dir in list(self._created_sandbox_dirs):
                if not self.config.keep_images:
                    shutil.rmtree(sandbox_dir, ignore_errors=True)
            self._created_sandbox_dirs = []

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
        model_section = snapshot_dict["model"]
        config_section = model_section["config"]
        kwargs = config_section["model_kwargs"]
        kwargs["api_base"] = (
            base_url + "/v1" if not base_url.endswith("/v1") else base_url
        )
        kwargs["api_key"] = self.policy_api_key
        kwargs["temperature"] = float(temperature)
        kwargs["top_p"] = float(top_p)
        if "model_name" in config_section:
            config_section["model_name"] = _ensure_litellm_prefix(
                config_section["model_name"]
            )

    def _fork_lane_b(
        self,
        *,
        mid_cp: MidCp,
        branch_index: int,
        node_id: str,
        temperature: float,
        top_p: float,
    ) -> Any:
        """Fork a single Lane B branch from a MidCp."""
        session_id = f"{node_id}-session"
        resumed = {
            "session_id": session_id,
            "status": mid_cp.snapshot["status"],
            "spec": copy.deepcopy(mid_cp.snapshot["spec"]),
            "agent": copy.deepcopy(mid_cp.snapshot["agent"]),
            "model": copy.deepcopy(mid_cp.snapshot["model"]),
            "environment": copy.deepcopy(mid_cp.snapshot["environment"]),
            "last_step_index": -1,
            "last_event_id": None,
            "metadata": {"events": [], "model_turns": []},
        }
        if mid_cp.snapshot.get("memory") is not None:
            resumed["memory"] = copy.deepcopy(mid_cp.snapshot["memory"])
        resumed["spec"]["session_id"] = session_id
        lane_b_base_url = self.policy_base_urls[mid_cp.idx % len(self.policy_base_urls)]
        self._override_model_kwargs(
            resumed,
            lane_b_base_url,
            temperature=temperature,
            top_p=top_p,
        )
        env_section = resumed["environment"]
        env_config = env_section["config"]
        env_state = env_section["state"]
        snapshot_environment_class = _container_environment_kind(str(env_section.get("type_path", "")))
        if snapshot_environment_class != self.environment_class:
            raise RuntimeError(
                f"Snapshot environment {snapshot_environment_class!r} does not match runner "
                f"environment {self.environment_class!r}"
            )
        if self.environment_class == "singularity":
            branch_sandbox = self._copy_singularity_sandbox(
                Path(mid_cp.image_tag),
                f"lane-b-g{mid_cp.idx:03d}-{branch_index:02d}",
            )
            env_config["image"] = self.base_image
            env_config["reuse_sandbox_dir"] = str(branch_sandbox)
            env_state["sandbox_dir"] = str(branch_sandbox)
            env_state["owns_sandbox"] = True
        else:
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
        initial_temperature: float,
        initial_top_p: float,
        group_index: int | None = None,
        beam_parent_index: int | None = None,
        beam_parent_node_id: str | None = None,
    ) -> LaneBBranch:
        """Sample one Lane-B continuation.

        Training stops after ``budget_steps``.  Only validation explicitly
        enables ``terminal_rollout``, which also enables terminal patch
        extraction and GT evaluation in the group orchestrator.
        """
        effective_group_index = mid_cp.idx if group_index is None else group_index
        node_id = (
            f"lane-b-g{effective_group_index:03d}-b{branch_index:02d}-"
            f"{uuid.uuid4().hex[:6]}"
        )
        branch = LaneBBranch(
            group_index=effective_group_index,
            branch_index=branch_index,
            node_id=node_id,
            parent_image_tag=mid_cp.image_tag,
            policy_model_name=self.policy_model_name,
            parent_asst_step=mid_cp.asst_step,
            started_at=time.perf_counter(),
            beam_parent_index=beam_parent_index,
            beam_parent_node_id=beam_parent_node_id,
        )
        session = None
        result: dict[str, Any] = {"status": "error", "exit_status": "not_started"}
        workspace_meta = None
        before_event_count = 0
        before_message_count = 0
        before_turn_count = 0
        try:
            session = self._fork_lane_b(
                mid_cp=mid_cp,
                branch_index=branch_index,
                node_id=node_id,
                temperature=initial_temperature,
                top_p=initial_top_p,
            )
            branch.session_id = session.spec.session_id
            session.agent.config.step_limit = int(total_step_limit)
            before_event_count = len(session.events)
            before_message_count = mid_cp.parent_message_count
            before_turn_count = len(session.model_turns)

            result = self._step_session(session, max_steps=budget_steps)
            workspace_meta = _collect_workspace_meta(session.agent.env)
            first_phase_steps = int(result["executed_steps"])
            logger.info(
                "[%s] lane_b g=%d b=%d first_phase_done dt=%.1fs steps=%d status=%s submitted=%s",
                self.task_id,
                effective_group_index,
                branch_index,
                time.perf_counter() - branch.started_at,
                first_phase_steps,
                result["status"],
                result["exit_status"] == "Submitted",
            )

            first_phase_exit_error = _unexpected_rollout_exit(
                result, lane="lane_b"
            )
            if (
                self.config.terminal_rollout
                and first_phase_exit_error is None
                and not str(result.get("exit_status") or "").strip()
                and result["status"] != "finished"
            ):
                branch_step_end = mid_cp.asst_step + first_phase_steps
                remaining_steps = self._branch_terminal_budget(branch_step_end)
                if remaining_steps > 0:
                    result = self._step_session(session, max_steps=remaining_steps)

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
            branch.status = result["status"]
            branch.terminated_early = result["exit_status"] == "Submitted"
            overlength_status = _policy_overlength_exit_status(result)
            exit_error = _unexpected_rollout_exit(result, lane="lane_b")
            if exit_error is not None:
                branch.error = exit_error
            if overlength_status is not None:
                branch.status = "policy_overlength"
                branch.overlength_reason = overlength_status
                branch.reward_penalty = float(
                    self.config.fallback_patch_penalty
                )
                branch.terminal_patch, branch.terminal_patch_from_fallback = (
                    self._extract_terminal_patch(result, session)
                )
            elif self.config.terminal_rollout:
                branch.terminal_patch, branch.terminal_patch_from_fallback = (
                    self._extract_terminal_patch(result, session)
                )
            logger.info(
                "[%s] lane_b g=%d b=%d done dt=%.1fs events=%d submitted=%s "
                "status=%s patch_len=%d tokens_p=%d tokens_c=%d",
                self.task_id, effective_group_index, branch_index,
                time.perf_counter() - branch.started_at,
                len(branch.events), branch.terminated_early, branch.status,
                len(branch.terminal_patch),
                branch.total_tokens["prompt"],
                branch.total_tokens["completion"],
            )
        except Exception as exc:
            branch.error = f"{type(exc).__name__}: {exc}"
            branch.status = "error"
            logger.warning(
                "[%s] lane_b g=%d b=%d FAILED dt=%.1fs %s",
                self.task_id, effective_group_index, branch_index,
                time.perf_counter() - branch.started_at, branch.error,
            )
            if self._is_context_window_error(exc) and session is not None:
                # Provider-side overflow is semantically identical to the
                # structured pre-flight exit: keep the partial trajectory,
                # collect the current diff, and let Lane C score it with the
                # configured multiplicative penalty.
                branch.error = None
                branch.status = "policy_overlength"
                branch.overlength_reason = "ContextWindowExceeded"
                branch.reward_penalty = float(
                    self.config.fallback_patch_penalty
                )
                branch.terminated_early = False
                branch.messages = copy.deepcopy(session.agent.messages[before_message_count:])
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
                branch.terminal_patch, branch.terminal_patch_from_fallback = (
                    self._extract_terminal_patch(result, session)
                )
                logger.warning(
                    "[%s] lane_b g=%d b=%d stopped by context window dt=%.1fs "
                    "patch_len=%d tokens_p=%d tokens_c=%d",
                    self.task_id, effective_group_index, branch_index,
                    time.perf_counter() - branch.started_at,
                    len(branch.terminal_patch),
                    branch.total_tokens["prompt"],
                    branch.total_tokens["completion"],
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
        """Score a Lane B branch with the shared evaluator reward definition."""
        t_gt = time.perf_counter()
        raw = (branch.terminal_patch or "").rstrip()
        patch = (raw + "\n") if raw else ""
        n_action_steps = sum(
            1
            for card in _build_step_cards(branch.events, branch.parent_asst_step)
            if card.get("commands")
        )
        reward_config = EvaluationRewardConfig(
            kind=self.config.reward_kind,
            joint_alpha=self.config.joint_alpha,
            all_pass_reward=self.config.all_pass_reward,
            fallback_patch_penalty=(
                self.config.fallback_patch_penalty
                if branch.terminal_patch_from_fallback
                else 1.0
            ),
            no_action_patch_penalty=(
                self.config.no_action_patch_penalty if n_action_steps == 0 else 0.0
            ),
        )
        if not patch:
            branch.gt_payload = make_evaluation_payload(
                "empty", reward_config=reward_config
            )
            branch.gt_score = float(branch.gt_payload["reward"])
            logger.info(
                "[%s] gt_done node=%s dt=0.0s reward=%.3f note=empty_patch",
                self.task_id, branch.node_id, branch.gt_score,
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
                reward_config=reward_config,
            )
            if not isinstance(payload, dict) or branch.node_id not in payload:
                raise RuntimeError(f"missing evaluation payload for {branch.node_id}")
            branch_payload = payload[branch.node_id]
            branch.gt_payload = copy.deepcopy(branch_payload)
            if branch_payload.get("metainfo", {}).get("infrastructure_error"):
                branch.gt_score = None
            else:
                branch.gt_score = float(branch_payload["reward"])
            logger.info(
                "[%s] gt_done node=%s dt=%.1fs reward=%.3f",
                self.task_id, branch.node_id, time.perf_counter() - t_gt,
                branch.gt_score if branch.gt_score is not None else -1.0,
            )
        except Exception as exc:
            branch.gt_payload = make_evaluation_payload(
                "error", error=exc, reward_config=reward_config, infrastructure_error=True
            )
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
        view = {
            "node_id": branch.node_id,
            "summary": {
                "step_count": len(step_cards),
                "changed_files": list(workspace.get("changed_files", []))[:8],
                "untracked_files": list(workspace.get("untracked_files", []))[:8],
                "git_diff": workspace.get("git_diff", ""),
            },
            "raw_continuation": {
                "step_cards": step_cards,
                "segment_step_range": [step_start, step_end],
            },
        }
        if branch.beam_parent_index is not None:
            view["parent_index"] = branch.beam_parent_index + 1
        return view

    async def _run_lane_c_scope(
        self,
        *,
        group: ForkGroup,
        scope: str,
        question: dict[str, Any],
        shared_context: dict[str, Any],
        previous_state: dict[str, Any],
        latest_shared_segment: dict[str, Any] | None,
        continuations: list[dict[str, Any]],
        score_bank: ScoreRubricBank,
        experience_bank: ExperienceRubricBank,
        generation_prompt: str,
        rubric_list_prefix: str,
        judge_prompt: str,
    ) -> dict[str, Any]:
        cfg = self.config
        experience_context = await experience_bank.build_generation_context(
            question=question,
            previous_state=previous_state,
            latest_shared_segment=latest_shared_segment,
            continuations=continuations,
            model_name=self.rubric_model_name,
            top_p=cfg.rubric_top_p,
            model_kwargs=self.rubric_model_kwargs,
            summary_max_tokens=cfg.psu_max_tokens,
            instance_id=self.task_id,
            round_index=group.group_index + 1,
        )
        score_context = score_bank.build_generation_context()
        extra_prompt_sections = list(score_context.extra_prompt_sections)
        extra_prompt_sections.extend(experience_context.extra_prompt_sections)
        retrieved_experiences = copy.deepcopy(experience_context.retrieved)
        retrieve_messages = copy.deepcopy(experience_context.retrieve_messages)
        retrieval_errors: list[str] = []
        if (
            retrieve_messages
            and retrieve_messages[-1].get("role") == "user"
            and "valid final JSON object" in str(
                retrieve_messages[-1].get("content") or ""
            )
        ):
            retrieval_errors.append("experience_retrieval_format_exhausted")
        generation_context = {
            "question": copy.deepcopy(question),
            "previous_state": copy.deepcopy(previous_state),
            "latest_shared_segment": copy.deepcopy(latest_shared_segment),
            "continuations": copy.deepcopy(continuations),
            "retrieved_rubric_experiences": [asdict(experience) for experience in retrieved_experiences],
            "previous_generated_rubrics": [asdict(rubric) for rubric in score_context.existing_rubrics],
        }

        generation_kwargs = {
            "model_name": self.rubric_model_name,
            "temperature": cfg.rubric_temperature,
            "top_p": cfg.rubric_top_p,
            "max_tokens": cfg.rubric_max_tokens,
            "model_kwargs": self.rubric_model_kwargs,
        }
        judge_kwargs = {
            "model_name": self.judge_model_name,
            "temperature": cfg.judge_temperature,
            "top_p": cfg.judge_top_p,
            "max_tokens": cfg.judge_max_tokens,
            "model_kwargs": self.judge_model_kwargs,
        }
        score_batch = await _generate_and_score_rubric_batch(
            sample_count=cfg.n,
            round_index=group.group_index + 1,
            generation_kwargs=generation_kwargs,
            generation_prompt=generation_prompt,
            rubric_list_prefix=rubric_list_prefix,
            question=question,
            shared_context=shared_context,
            continuations=continuations,
            extra_prompt_sections=extra_prompt_sections,
            judge_kwargs=judge_kwargs,
            judge_prompt=judge_prompt,
        )
        model_response = {
            "scope": scope,
            "num_samples": len(score_batch["generated_samples"]),
            "generated_rubrics": [
                asdict(rubric)
                for sample in score_batch["generated_samples"]
                for rubric in sample.generated
            ],
            "format_errors": [
                error
                for sample in score_batch["generated_samples"]
                for error in (sample.format_errors or [])
            ],
            "terminal_errors": [
                sample.terminal_error
                for sample in score_batch["generated_samples"]
                if sample.terminal_error
            ],
        }

        group_generated_rubrics = [
            rubric
            for generated_rubrics in score_batch["sample_generated_rubrics"]
            for rubric in generated_rubrics
        ]

        aggregate_judge_response: dict[str, dict[str, Any]] = {}
        rubric_samples: list[dict[str, Any]] = []
        valid_score_samples: list[dict[str, float]] = []
        # Keep generation diagnostics in model_response/sample artifacts.
        # Usability follows the oracle: a recovered sample is valid when it
        # produced rubrics and complete judge scores.
        scope_errors: list[str] = list(retrieval_errors)
        if len(score_batch["generated_samples"]) != cfg.n:
            scope_errors.append(
                "rubric_sample_count:"
                f"{len(score_batch['generated_samples'])}/{cfg.n}"
            )
        node_ids = [branch.node_id for branch in group.branches]
        for sample_index, generated_sample in enumerate(score_batch["generated_samples"]):
            generated_rubrics = score_batch["sample_generated_rubrics"][sample_index]
            scoring_rubrics = generated_rubrics
            if not scoring_rubrics:
                scope_errors.append(
                    f"rubric_sample_{sample_index}:no_valid_rubrics"
                )
            evaluation = _rubric_sample_evaluation(
                score_batch=score_batch,
                sample_index=sample_index,
                node_ids=node_ids,
                scoring_rubrics=scoring_rubrics,
                include_variance_reward=True,
            )
            metrics = evaluation["metrics"]
            judge_response_for_sample: dict[str, dict[str, Any]] = {}
            for branch, score_records in zip(group.branches, evaluation["scored_continuations"]):
                for record in score_records:
                    rubric_id = record["rubric_id"]
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
            for error in evaluation["judge_errors"]:
                rubric_id = error.get("rubric_id")
                node_id = error.get("node_id")
                if rubric_id and node_id:
                    judge_response_for_sample.setdefault(rubric_id, {}).setdefault(node_id, {})["error"] = error.get("error")
                    aggregate_judge_response.setdefault(rubric_id, {}).setdefault(node_id, {})["error"] = error.get("error")

            avg_scores = _avg_scores_from_rubrics(
                node_ids=node_ids,
                score_lookup_by_node=evaluation["score_lookup_by_node"],
                rubrics=scoring_rubrics,
            )
            tie_break = None
            tie_break_messages: list[dict[str, Any]] = []
            tie_break_judge_messages: list[dict[str, Any]] = []
            sample_errors = [
                f"judge:{error}" for error in evaluation["judge_errors"]
            ]
            base_score_errors = _score_map_errors(
                avg_scores,
                node_ids=node_ids,
                label=f"rubric_sample_{sample_index}_scores",
            )
            sample_errors.extend(base_score_errors)
            scope_errors.extend(base_score_errors)
            if (
                cfg.score_tie_break
                and scoring_rubrics
                and not sample_errors
            ):
                tie_break_result = await _run_score_tie_break(
                    scope=scope,
                    round_index=group.group_index + 1,
                    generation_prompt=generation_prompt,
                    rubric_list_prefix=rubric_list_prefix,
                    judge_prompt=judge_prompt,
                    question=question,
                    shared_context=shared_context,
                    continuations=continuations,
                    extra_prompt_sections=extra_prompt_sections,
                    generation_kwargs=generation_kwargs,
                    judge_kwargs=judge_kwargs,
                    initial_rubrics=scoring_rubrics,
                    initial_score_by_rubric=metrics["score_by_rubric"],
                    initial_scores=avg_scores,
                )
                if tie_break_result is not None:
                    tie_errors = _tie_break_errors(
                        tie_break_result,
                        node_ids=node_ids,
                    )
                    sample_errors.extend(tie_errors)
                    scope_errors.extend(tie_errors)
                    tie_break = copy.deepcopy(tie_break_result)
                    tie_break_messages = tie_break.pop("messages", [])
                    tie_break_judge_messages = tie_break.pop("judge_messages", [])
                    if not tie_errors and tie_break.get("status") == "success":
                        avg_scores = copy.deepcopy(tie_break["adjusted_scores"])
            # Match trajectory_search: a node's final reward is the average
            # of the oracle-weighted score over valid rubric-list samples.
            # Training is fail-closed, so a judge error in any call prevents
            # this rubric-list sample from contributing and ultimately leaves
            # the group without a score map.
            if (
                scoring_rubrics
                and not sample_errors
            ):
                valid_score_samples.append(copy.deepcopy(avg_scores))
            scope_errors.extend(
                f"judge:{error}" for error in evaluation["judge_errors"]
            )
            generated_ids = {rubric.rubric_id for rubric in generated_sample.generated}
            sample_payload = {
                "scope": scope,
                "sample_index": generated_sample.sample_index,
                "rubric_list_id": generated_sample.rubric_list_id,
                "generated": [asdict(rubric) for rubric in generated_sample.generated],
                "messages": copy.deepcopy(generated_sample.messages),
                "generation_context": copy.deepcopy(generation_context),
                "format_errors": copy.deepcopy(generated_sample.format_errors or []),
                "terminal_error": generated_sample.terminal_error,
                "generated_titles": [rubric.title for rubric in generated_sample.generated],
                "scoring_rubrics": [asdict(rubric) for rubric in scoring_rubrics],
                **_rubric_metrics_payload(metrics, generated_ids),
                "average_rubric_judged_scores": avg_scores,
                "judge_errors": evaluation["judge_errors"],
                "judge_response": judge_response_for_sample,
                "tie_break": tie_break,
                "tie_break_messages": tie_break_messages,
                "tie_break_judge_messages": tie_break_judge_messages,
                "reward": 0.0,
                "retrieved": [asdict(experience) for experience in retrieved_experiences],
                "retrieve_messages": copy.deepcopy(retrieve_messages),
                "selected": False,
            }
            rubric_samples.append(sample_payload)

        bank_update = score_bank.update_from_model(generated=group_generated_rubrics)
        score_bank.set_state(
            active_bank=copy.deepcopy(bank_update.active_after),
            inactive_bank=copy.deepcopy(bank_update.inactive_after),
        )
        judge_score_by_node: dict[str, float] = {}
        if (
            not scope_errors
            and len(valid_score_samples) == len(score_batch["generated_samples"])
            and valid_score_samples
        ):
            aggregated_scores = {
                node_id: sum(scores[node_id] for scores in valid_score_samples)
                / len(valid_score_samples)
                for node_id in node_ids
            }
            aggregate_errors = _score_map_errors(
                aggregated_scores,
                node_ids=node_ids,
                label="judge_score_by_node",
            )
            if aggregate_errors:
                scope_errors.extend(aggregate_errors)
            else:
                judge_score_by_node = aggregated_scores
        return {
            "scope": scope,
            "samples": rubric_samples,
            "judge_response": aggregate_judge_response,
            "model_response": model_response,
            "judge_score_by_node": judge_score_by_node,
            "errors": scope_errors,
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
        spr = cfg.k
        for start in range(0, len(all_shared_step_cards), spr):
            end = min(start + spr, len(all_shared_step_cards))
            segments.append(
                {
                    "step_cards": copy.deepcopy(all_shared_step_cards[start:end]),
                    "segment_step_range": [start, end],
                }
            )
        summary_update_messages: list[dict[str, Any]] = []
        summary_update_errors: list[str] = []
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
                evicted_step_cards=evicted["step_cards"],
                workspace_meta=copy.deepcopy(EMPTY_WORKSPACE_META),
                model_name=self.judge_model_name,
                temperature=cfg.judge_temperature,
                top_p=cfg.judge_top_p,
                max_tokens=cfg.psu_max_tokens,
                model_kwargs=self.judge_model_kwargs,
            )
            self._summary_persistent_state = copy.deepcopy(update_payload["state"])
            summary_update_messages.extend(copy.deepcopy(update_payload["messages"]))
            if update_payload.get("error"):
                summary_update_errors.append(str(update_payload["error"]))

        previous_state = copy.deepcopy(self._summary_persistent_state)
        recent_segments = copy.deepcopy(self._summary_recent_segments)
        latest_shared_segment = copy.deepcopy(recent_segments[-1]) if recent_segments else None
        if group.parent_mid_cps:
            # Same multi-parent context shape used by trajectory_search:
            # parent 1 owns continuations 1..m/p, parent 2 the next range.
            per_parent = len(group.branches) // len(group.parent_mid_cps)
            latest_shared_segment = {
                "parent_views": [
                    {
                        "parent_index": parent_index + 1,
                        "continuations": (
                            f"{parent_index * per_parent + 1}-"
                            f"{(parent_index + 1) * per_parent}"
                        ),
                        "trajectory": {
                            "step_cards": copy.deepcopy(parent_mid_cp.parent_step_cards),
                            "segment_step_range": [0, parent_mid_cp.asst_step],
                        },
                    }
                    for parent_index, parent_mid_cp in enumerate(group.parent_mid_cps)
                ]
            }
            previous_state = {
                "parent_states": [
                    {
                        "parent_index": parent_index + 1,
                        "continuations": (
                            f"{parent_index * per_parent + 1}-"
                            f"{(parent_index + 1) * per_parent}"
                        ),
                        "state": copy.deepcopy(EMPTY_PERSISTENT_STATE),
                    }
                    for parent_index in range(len(group.parent_mid_cps))
                ]
            }
        group.persisted_parent_state = previous_state
        group.summary = {
            "persistent_state": previous_state,
            "recent_segments": recent_segments,
            "persistent_state_errors": summary_update_errors,
        }
        group.summary_messages = summary_update_messages

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
        scope_specs = [{
            "scope": "siblings",
            "score_bank": self.rubric_bank,
            "experience_bank": self.experience_bank,
            "shared_context": shared_context,
            "latest_shared_segment": latest_shared_segment,
            "generation_prompt": SWE_TRAJECTORY_RUBRIC_GENERATION_PROMPT,
            "rubric_list_prefix": "rubric",
            "judge_prompt": SWE_TRAJECTORY_RUBRIC_JUDGE_PROMPT,
            "samples_attr": "rubric_samples",
            "judge_attr": "judge_response",
            "model_response_attr": "rubric_model_response",
        }]
        try:
            scope_results = await asyncio.gather(
                *[
                    self._run_lane_c_scope(
                        group=group,
                        scope=spec["scope"],
                        question=question,
                        shared_context=spec["shared_context"],
                        previous_state=previous_state,
                        latest_shared_segment=spec["latest_shared_segment"],
                        continuations=continuations,
                        score_bank=spec["score_bank"],
                        experience_bank=spec["experience_bank"],
                        generation_prompt=spec["generation_prompt"],
                        rubric_list_prefix=spec["rubric_list_prefix"],
                        judge_prompt=spec["judge_prompt"],
                    )
                    for spec in scope_specs
                ]
            )
        except Exception as exc:
            group.rubric_model_response = {"error": f"{type(exc).__name__}: {exc}"}
            group.lane_c_errors = [
                f"lane_c_exception:{type(exc).__name__}:{exc}"
            ]
            logger.warning("[%s] lane_c g=%d rubric_FAILED %s", self.task_id, group.group_index, exc)
            group.lane_c_done_at = time.perf_counter()
            return

        results_by_scope = {result["scope"]: result for result in scope_results}
        for spec in scope_specs:
            result = results_by_scope[spec["scope"]]
            setattr(group, spec["samples_attr"], result["samples"])
            setattr(group, spec["judge_attr"], result["judge_response"])
            setattr(group, spec["model_response_attr"], result["model_response"])
            group.judge_score_by_node = copy.deepcopy(
                result["judge_score_by_node"]
            )
            group.lane_c_errors = copy.deepcopy(result["errors"])
        group.lane_c_done_at = time.perf_counter()
        n_err = sum(
            1
            for response in (group.judge_response,)
            for branch_map in response.values()
            for record in branch_map.values()
            if isinstance(record, dict) and record.get("error")
        )
        logger.info(
            "[%s] lane_c g=%d judge_done dt=%.1fs err=%d siblings_rubrics=%d branches=%d",
            self.task_id, group.group_index, time.perf_counter() - t_c,
            n_err,
            len(group.judge_response),
            len(continuations),
        )

    async def _run_lane_c_in_order(self, group: ForkGroup) -> None:
        """Run Lane C as soon as this group is terminal, preserving rubric-bank order."""
        condition = self._lane_c_condition
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
        group_index: int | None = None,
        group_kind: str = "root",
        parent_mid_cps: list[MidCp] | None = None,
    ) -> ForkGroup:
        """Sample one complete ``m``-branch training group.

        A root group forks all branches from ``mid_cp``.  A beam group forks
        ``m / p`` branches from each Lane-A parent checkpoint while retaining
        ``mid_cp`` as the common training prompt.
        """
        cfg = self.config
        t_group = time.perf_counter()
        effective_group_index = mid_cp.idx if group_index is None else group_index
        parent_mid_cps = list(parent_mid_cps or [])
        group = ForkGroup(
            group_index=effective_group_index,
            mid_cp=mid_cp,
            group_kind=group_kind,
            parent_mid_cps=parent_mid_cps,
        )
        if parent_mid_cps:
            per_parent = cfg.m // len(parent_mid_cps)
            fork_plan = [
                (parent_mid_cp, parent_index, parent_mid_cp.node_id)
                for parent_index, parent_mid_cp in enumerate(parent_mid_cps)
                for _ in range(per_parent)
            ]
        else:
            fork_plan = [(mid_cp, None, None) for _ in range(cfg.m)]
        if len(fork_plan) != cfg.m:
            raise RuntimeError(
                f"fork_group g={effective_group_index} expected {cfg.m} branches, "
                f"planned {len(fork_plan)}"
            )
        logger.info(
            "[%s] fork_group g=%d kind=%s start m=%d parents=%d",
            self.task_id,
            effective_group_index,
            group_kind,
            cfg.m,
            len(parent_mid_cps),
        )
        branch_tasks: list[asyncio.Task] = []
        for bi, (branch_mid_cp, parent_index, parent_node_id) in enumerate(fork_plan):
            first_budget = self._branch_initial_budget(branch_mid_cp.asst_step)
            if first_budget <= 0:
                raise RuntimeError(
                    f"fork_group g={effective_group_index} has no remaining "
                    f"budget at parent step {branch_mid_cp.asst_step}"
                )
            ctx = contextvars.copy_context()

            def _wrapped(
                mc=branch_mid_cp,
                b=bi,
                c=ctx,
                parent_idx=parent_index,
                parent_id=parent_node_id,
                budget=first_budget,
            ):
                return c.run(
                    self._run_lane_b_branch,
                    mid_cp=mc,
                    branch_index=b,
                    budget_steps=budget,
                    total_step_limit=cfg.step_limit,
                    initial_temperature=cfg.lane_b_temperature,
                    initial_top_p=cfg.lane_b_top_p,
                    group_index=effective_group_index,
                    beam_parent_index=parent_idx,
                    beam_parent_node_id=parent_id,
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
                branch.gt_payload = make_evaluation_payload(
                    "error", error=exc, infrastructure_error=True
                )
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
                    group_index=effective_group_index,
                    branch_index=bi,
                    node_id=f"lane-b-g{effective_group_index:03d}-b{bi:02d}-err",
                    parent_image_tag=fork_plan[bi][0].image_tag,
                    policy_model_name=self.policy_model_name,
                    parent_asst_step=fork_plan[bi][0].asst_step,
                    error=f"executor: {type(item).__name__}: {item}",
                    status="error",
                    beam_parent_index=fork_plan[bi][1],
                    beam_parent_node_id=fork_plan[bi][2],
                )
                err_branch.gt_score = None
                err_branch.gt_payload = make_evaluation_payload(
                    "error", error=err_branch.error
                )
                branch = err_branch
            else:
                branch = item
            branches_by_index[bi] = branch
            group.branches = [
                branches_by_index[index] for index in sorted(branches_by_index)
            ]
            self._dump_group_scaffold(group)
            self._dump_branch(group, branch)
            if self.config.terminal_rollout and branch.error is None:
                ctx = contextvars.copy_context()
                gt_future = loop.run_in_executor(
                    gt_pool, lambda b=branch, c=ctx: c.run(self._evaluate_gt, b)
                )
                gt_tasks.append(asyncio.create_task(_await_gt(branch, gt_future)))
            else:
                self._dump_branch(group, branch)

        branches = [branches_by_index[index] for index in range(cfg.m)]
        group.branches = branches

        # GT is validation-only.  Training branches stop at k and go directly
        # to Lane C without patch extraction or evaluator work.
        for done in asyncio.as_completed(gt_tasks):
            branch = await done
            self._dump_branch(group, branch)

        for checkpoint in parent_mid_cps:
            self._delete_checkpoint(checkpoint.image_tag)
        logger.info(
            "[%s] fork_group g=%d kind=%s done dt=%.1fs gt_scores=%s lane_c_dt=%.1fs",
            self.task_id,
            effective_group_index,
            group_kind,
            time.perf_counter() - t_group,
            [round(b.gt_score, 3) if b.gt_score is not None else None for b in branches],
            (group.lane_c_done_at - group.lane_c_started_at)
            if group.lane_c_started_at else 0.0,
        )
        return group

    # -- Lane A: root + independent beam parents -----------------------------

    def _start_lane_a(self, lane_a: LaneAState) -> Any:
        """Spin up Lane A's root session, capture system+user prompts.
        Returns the session. Records started_at on `lane_a`."""
        lane_a.started_at = time.perf_counter()
        session = self._make_initial_session()
        msgs = session.agent.messages
        self.system_prompt = msgs[0]["content"]
        self.user_prompt = msgs[1]["content"]
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

    def _create_root_mid_cp(self, lane_a: LaneAState) -> MidCp:
        """Create the common root checkpoint without sampling policy tokens."""
        session = self._start_lane_a(lane_a)
        try:
            image_tag = self._checkpoint_environment(session, "mid-root")
            if image_tag is None:
                raise RuntimeError("root environment checkpoint returned None")
            snapshot = self._make_resume_snapshot(session)
            mid_cp = MidCp(
                idx=0,
                asst_step=0,
                image_tag=image_tag,
                snapshot=snapshot,
                parent_messages_payload=_build_parent_messages_payload(
                    session.agent.messages,
                    model_name=self.policy_model_name,
                ),
                parent_step_cards=[],
                parent_message_count=len(session.agent.messages),
                parent_event_count=len(session.events),
                parent_turn_count=len(session.model_turns),
                emitted_at=time.perf_counter(),
                node_id="root",
            )
            lane_a.mid_cps.append(mid_cp)
            lane_a.status = "root_ready"
            return mid_cp
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
            if session in self._live_sessions:
                self._live_sessions.remove(session)

    def _run_lane_a_parent(self, root_mid_cp: MidCp, parent_index: int) -> MidCp:
        """Sample and checkpoint one independent strict-k Lane-A parent."""
        node_id = (
            f"lane-a-parent-{parent_index:02d}-{uuid.uuid4().hex[:6]}"
        )
        session = self._fork_lane_b(
            mid_cp=root_mid_cp,
            branch_index=parent_index,
            node_id=node_id,
            temperature=self.config.policy_temperature,
            top_p=self.config.policy_top_p,
        )
        before_message_count = root_mid_cp.parent_message_count
        before_event_count = len(session.events)
        before_turn_count = len(session.model_turns)
        try:
            result = self._step_session(
                session,
                max_steps=self._branch_initial_budget(0),
            )
            overlength_status = _policy_overlength_exit_status(result)
            exit_error = _unexpected_rollout_exit(
                result, lane=f"lane_a_parent_{parent_index}"
            )
            if exit_error is not None:
                raise RuntimeError(exit_error)
            if overlength_status is not None:
                patch, is_fallback = self._extract_terminal_patch(
                    result, session
                )
                overflow_dir = self.run_dir / "groups" / "group_001"
                _atomic_write_json(
                    overflow_dir
                    / f"beam_parent_{parent_index:02d}_overlength.json",
                    {
                        "node_id": node_id,
                        "exit_status": overlength_status,
                        "terminal_patch": patch,
                        "terminal_patch_from_fallback": is_fallback,
                    },
                )
                raise RuntimeError(
                    f"Lane-A parent {parent_index} stopped by "
                    f"{overlength_status}; beam group cannot form children"
                )
            messages = copy.deepcopy(
                session.agent.messages[before_message_count:]
            )
            assistant_messages = [
                message for message in messages
                if message.get("role") == "assistant"
            ]
            if not assistant_messages:
                raise RuntimeError(
                    f"Lane-A parent {parent_index} produced no assistant turns"
                )
            snapshot = self._make_resume_snapshot(session)
            image_tag = self._checkpoint_environment(
                session, f"beam-parent-{parent_index:02d}"
            )
            if image_tag is None:
                raise RuntimeError(
                    f"Lane-A parent {parent_index} checkpoint returned None"
                )
            events = _step_card_event_dicts(
                session.events[before_event_count:]
            )
            step_cards = _build_step_cards(events, 0)
            asst_step = len(step_cards)
            if asst_step <= 0:
                raise RuntimeError(
                    f"Lane-A parent {parent_index} produced no step cards"
                )
            turns = self._extract_turn_token_info_from_turns(
                session.model_turns[before_turn_count:],
                starting_turn_index=before_turn_count,
            )
            logger.info(
                "[%s] lane_a parent=%d node=%s steps=%d status=%s",
                self.task_id,
                parent_index,
                node_id,
                asst_step,
                result["status"],
            )
            return MidCp(
                # Parent index is only for endpoint spreading/artifacts.  The
                # beam training group receives its own group_index=1.
                idx=parent_index + 1,
                asst_step=asst_step,
                image_tag=image_tag,
                snapshot=snapshot,
                parent_messages_payload=_build_parent_messages_payload(
                    session.agent.messages,
                    model_name=self.policy_model_name,
                ),
                parent_step_cards=step_cards,
                parent_message_count=len(session.agent.messages),
                parent_event_count=len(session.events),
                parent_turn_count=len(session.model_turns),
                emitted_at=time.perf_counter(),
                node_id=node_id,
            )
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
            if session in self._live_sessions:
                self._live_sessions.remove(session)

    async def _lane_a_loop(
        self,
        *,
        lane_a: LaneAState,
        loop: asyncio.AbstractEventLoop,
        lane_a_pool: ThreadPoolExecutor,
        on_mid_cp=None,
    ) -> None:
        """Legacy pre-training driver retained for old artifact readers only.

        The training entrypoint does not call this method; it uses
        ``_create_root_mid_cp`` plus ``_run_lane_a_parent``.  Drive Lane A:
        run in chunks of k, commit + emit
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
                    self._checkpoint_environment, s, "mid-root"
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
                tp = lane_a.total_tokens["prompt"]
                tc = lane_a.total_tokens["completion"]
                lane_a.total_tokens = {
                    "prompt": tp + sum(t.prompt_tokens for t in new_turns),
                    "completion": tc + sum(t.completion_tokens for t in new_turns),
                }

                # Update cumulative position trackers.
                all_msgs_count = int(chunk["message_count"])
                all_turns_count = int(chunk["turn_count"])
                all_events_count = int(chunk["event_count"])
                cumulative_steps += len(new_step_cards)
                lane_a.status = result["status"]
                terminated = result["exit_status"] == "Submitted"
                exit_error = _unexpected_rollout_exit(
                    result, lane="lane_a"
                )
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

                if exit_error is not None:
                    lane_a.error = exit_error
                    lane_a.status = "error"
                    break
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
                            self._checkpoint_environment, s, tk
                        ),
                    )
                    if image_tag is None:
                        logger.warning(
                            "[%s] lane_a mid_cp idx=%d: checkpoint returned None, "
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
            max_workers=max(
                cfg.lane_a_pool_size,
                cfg.p if cfg.topology == "depth2" else 1,
            ),
            thread_name_prefix=f"lane-a-{self.task_id[:12]}",
        )
        lane_b_pool = ThreadPoolExecutor(
            max_workers=(
                cfg.lane_b_pool_size
                if cfg.lane_b_pool_size is not None
                else cfg.m * cfg.max_rounds
            ),
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
                async with condition:
                    if self._next_lane_c_group_index == group_index:
                        self._next_lane_c_group_index += 1
                        condition.notify_all()
                return None
            self._dump_group(grp)
            if getattr(self, "skip_lane_c", False):
                now = time.perf_counter()
                grp.lane_c_started_at = grp.lane_c_started_at or now
                grp.lane_c_done_at = grp.lane_c_done_at or now
                logger.info(
                    "[%s] lane_c g=%d skipped by trajectory-only mode",
                    self.task_id,
                    grp.group_index,
                )
            elif len(grp.branches) != cfg.m or any(
                branch.error is not None for branch in grp.branches
            ):
                now = time.perf_counter()
                grp.lane_c_started_at = grp.lane_c_started_at or now
                grp.lane_c_done_at = grp.lane_c_done_at or now
                logger.warning(
                    "[%s] lane_c g=%d skipped: incomplete/errored policy group",
                    self.task_id,
                    grp.group_index,
                )
                # Preserve serialized group ordering even when fail-closed
                # validation prevents a Lane-C request.
                condition = self._lane_c_condition
                async with condition:
                    await condition.wait_for(
                        lambda: self._next_lane_c_group_index == grp.group_index
                    )
                    self._next_lane_c_group_index += 1
                    condition.notify_all()
            else:
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

        try:
            with usage_context(
                phase="train", group_id=self._usage_group_id(0)
            ):
                root_ctx = contextvars.copy_context()
                root_mid_cp = await loop.run_in_executor(
                    lane_a_pool,
                    lambda c=root_ctx: c.run(
                        self._create_root_mid_cp, instance_record.lane_a
                    ),
                )
                root_coro = self._run_fork_group(
                    mid_cp=root_mid_cp,
                    loop=loop,
                    lane_b_pool=lane_b_pool,
                    gt_pool=gt_pool,
                    group_index=0,
                    group_kind="root",
                )
                group_tasks.append(
                    asyncio.create_task(
                        _await_group_then_callback(0, root_coro)
                    )
                )

            if cfg.topology == "depth2":
                with usage_context(
                    phase="train", group_id=self._usage_group_id(1)
                ):
                    parent_futures = []
                    for parent_index in range(cfg.p):
                        parent_ctx = contextvars.copy_context()
                        parent_futures.append(
                            loop.run_in_executor(
                                lane_a_pool,
                                lambda i=parent_index, c=parent_ctx: c.run(
                                    self._run_lane_a_parent, root_mid_cp, i
                                ),
                            )
                        )
                    parent_results = await asyncio.gather(
                        *parent_futures, return_exceptions=True
                    )
                parent_mid_cps: list[MidCp] = []
                parent_errors: list[str] = []
                for parent_index, item in enumerate(parent_results):
                    if isinstance(item, BaseException):
                        parent_errors.append(
                            f"parent {parent_index}: {type(item).__name__}: {item}"
                        )
                    else:
                        parent_mid_cps.append(item)
                instance_record.lane_a.mid_cps.extend(parent_mid_cps)
                if parent_errors:
                    instance_record.lane_a.error = "; ".join(parent_errors)
                    instance_record.lane_a.status = "beam_parent_error"
                    logger.warning(
                        "[%s] beam group skipped: %s",
                        self.task_id,
                        instance_record.lane_a.error,
                    )
                else:
                    with usage_context(
                        phase="train", group_id=self._usage_group_id(1)
                    ):
                        beam_coro = self._run_fork_group(
                            mid_cp=root_mid_cp,
                            loop=loop,
                            lane_b_pool=lane_b_pool,
                            gt_pool=gt_pool,
                            group_index=1,
                            group_kind="beam",
                            parent_mid_cps=parent_mid_cps,
                        )
                        group_tasks.append(
                            asyncio.create_task(
                                _await_group_then_callback(1, beam_coro)
                            )
                        )
                    instance_record.lane_a.status = "beam_parents_ready"

            if group_tasks:
                logger.info(
                    "[%s] awaiting %d topology groups",
                    self.task_id,
                    len(group_tasks),
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
            expected_groups = 2 if cfg.topology == "depth2" else 1
            instance_record.completed = (
                instance_record.lane_a.error is None
                and len(instance_record.groups) == expected_groups
                and all(
                    len(g.branches) == cfg.m
                    and all(b.error is None for b in g.branches)
                    and (
                        self.skip_lane_c
                        or set(g.judge_score_by_node)
                        == {b.node_id for b in g.branches}
                    )
                    for g in instance_record.groups
                )
            )
            if not instance_record.completed and instance_record.error is None:
                instance_record.error = (
                    instance_record.lane_a.error
                    or "one or more groups were incomplete, errored, or lacked judge scores"
                )
        except Exception as exc:
            instance_record.error = f"runner exception: {exc}\n{traceback.format_exc()}"
            instance_record.completed = False
        finally:
            instance_record.seconds = time.perf_counter() - started
            # Lane-A parents are not exported as a separate GRPO group. In
            # depth2 their assistant turns are instead prepended to each
            # corresponding child response and are trainable there. Each group
            # writes its own shared_parent_message.json from the root MidCp.
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
        _atomic_write_json(
            gdir / "shared_parent_message.json",
            parent_payload,
        )
        _atomic_write_json(gdir / "summary.json", group.summary)
        if group.summary_messages:
            _atomic_write_json(gdir / "summary_message.json", group.summary_messages)
        sibling_rubrics_dir = gdir / "sibling_rubrics"
        sibling_rubrics_dir.mkdir(parents=True, exist_ok=True)
        if group.experience_bank_update:
            _atomic_write_json(sibling_rubrics_dir / "rubric_bank.json", group.experience_bank_update)
            _atomic_write_json(sibling_rubrics_dir / "rubric_bank_message.json", group.experience_bank_message)
        _atomic_write_json(
            gdir / "judge_score_by_node.json", group.judge_score_by_node
        )
        _atomic_write_json(gdir / "lane_c_errors.json", group.lane_c_errors)
        return gdir

    def _dump_branch(self, group: ForkGroup, branch: LaneBBranch) -> None:
        gdir = self._group_dir(group)
        bdir = gdir / "branches" / f"branch_{branch.branch_index:02d}"
        bdir.mkdir(parents=True, exist_ok=True)
        branch_model_name = branch.policy_model_name or self.policy_model_name
        branch_policy_source = getattr(
            branch,
            "policy_source",
            "student",
        )
        branch_step_cards = _build_step_cards(branch.events, branch.parent_asst_step)
        stamped, _ = _stamp_steps(branch.messages, start_step=0)
        _atomic_write_json(
            bdir / "message.json",
            build_messages(
                stamped,
                model_name=branch_model_name,
                preserve_token_fields=True,
            ),
        )
        _atomic_write_json(
            bdir / "terminal_patch.json",
            {
                self.task_id: {
                    "model_name_or_path": branch_model_name,
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
                "parent_id": (
                    branch.beam_parent_node_id
                    or f"group_{group.group_index:03d}"
                ),
                "round_index": group.group_index + 1,
                "depth": 2 if group.group_kind == "beam" else 1,
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
                "policy_source": branch_policy_source,
                "policy_model_name": branch_model_name,
                "branch_index": branch.branch_index,
                "group_index": group.group_index,
                "group_kind": group.group_kind,
                "beam_parent_index": branch.beam_parent_index,
                "beam_parent_node_id": branch.beam_parent_node_id,
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
        self._dump_rubric_samples(gdir / "sibling_rubrics", group.rubric_samples)
        for branch in group.branches:
            self._dump_branch(group, branch)

    def _dump_rubric_samples(self, rubrics_dir: Path, samples: list[dict[str, Any]]) -> None:
        rubrics_dir.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            rubric_dir = rubrics_dir / str(sample["rubric_list_id"])
            rubric_dir.mkdir(parents=True, exist_ok=True)
            rubric_payload = {
                k: copy.deepcopy(v)
                for k, v in sample.items()
                if k
                not in {
                    "messages",
                    "judge_response",
                    "retrieve_messages",
                    "generation_context",
                    "tie_break_messages",
                    "tie_break_judge_messages",
                }
            }
            _atomic_write_json(rubric_dir / "rubric.json", rubric_payload)
            _atomic_write_json(rubric_dir / "rubric_message.json", sample["messages"])
            _atomic_write_json(rubric_dir / "rubric_retrieve_message.json", sample["retrieve_messages"])
            if sample.get("tie_break_messages"):
                _atomic_write_json(
                    rubric_dir / "tie_break_message.json",
                    sample["tie_break_messages"],
                )
            judge_messages = [
                {
                    "rubric_id": rubric_id,
                    "node_id": node_id,
                    "messages": copy.deepcopy(entry.get("judge_message") or []),
                    "error": entry.get("error"),
                }
                for rubric_id, by_node in sample["judge_response"].items()
                for node_id, entry in by_node.items()
            ]
            judge_messages.extend(copy.deepcopy(sample.get("tie_break_judge_messages") or []))
            _atomic_write_json(rubric_dir / "judge_message.json", judge_messages)
            _atomic_write_json(
                rubric_dir / "judge.json",
                {
                    "rubric_list_id": sample["rubric_list_id"],
                    "judge_errors": sample["judge_errors"],
                    "score_by_rubric": sample["score_by_rubric"],
                    "judge_error_by_rubric": sample["judge_error_by_rubric"],
                },
            )

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
                "lane_c_skipped": bool(getattr(self, "skip_lane_c", False)),
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
                        "group_kind": g.group_kind,
                        "mid_cp": {
                            "idx": g.mid_cp.idx,
                            "image_tag": g.mid_cp.image_tag,
                            "asst_step": g.mid_cp.asst_step,
                            "emitted_at": g.mid_cp.emitted_at,
                        },
                        "num_branches": len(g.branches),
                        "num_sibling_rubric_samples": len(g.rubric_samples),
                        "judge_score_by_node": copy.deepcopy(
                            g.judge_score_by_node
                        ),
                        "lane_c_errors": copy.deepcopy(g.lane_c_errors),
                        "beam_parent_node_ids": [
                            parent.node_id for parent in g.parent_mid_cps
                        ],
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
