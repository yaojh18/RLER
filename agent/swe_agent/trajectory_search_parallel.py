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
child's trainable response.  Terminal continuation is used by rollout-40 and
validation diagnostics; only the configured visible prefix is trainable.
SWE-bench evaluation remains diagnostic unless the explicit GT-on-submit
ablation is enabled.
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import logging
import math
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

# Reuse shared rollout/artifact helpers.  Lane C itself is training-only and
# lives in direct_rubric_judge; the TTS trajectory-search judge is unchanged.
from swe_agent.parallel_utils import (
    TurnTokenInfo,
    _ensure_litellm_prefix,
    _stamp_steps,
    _atomic_write_json,
    exact_rollout_token_error,
    extract_terminal_patch_from_session,
    normalize_terminal_patch_text,
)
from swe_agent.prompt import EMPTY_WORKSPACE_META
from swe_agent.run.run_swe_agent import (
    EvaluationRewardConfig,
    _litellm_model_kwargs,
    build_messages,
    evaluate_swebench_instance_patches,
    make_evaluation_payload,
)
from swe_agent.collapse_detector import trajectory_collapse_reason
from swe_agent.direct_rubric_judge import (
    DEFAULT_RUBRIC_BANK,
    DirectRubricBank,
    judge_direct_rubric_group,
)
from swe_agent.usage import usage_context
from swe_agent.variance_detector import load_variance_detector
from swe_agent.trajectory_search import (
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
    k: int = 20  # assistant turns between Lane A checkpoints
    p: int = 2  # Lane-A beam parents in depth2
    topology: str = "depth1"
    # Empty means every group implied by topology. The collector uses this
    # internal subset only for same-instance zero-variance resampling.
    requested_group_kinds: tuple[str, ...] = ()
    step_limit: int = 100  # hard cap for the whole trajectory

    # Lane A sampling — runs deterministic-ish spine; lower temp is fine.
    policy_temperature: float = 1.0
    policy_top_p: float = 0.95
    # Lane B sampling — higher for intra-group diversity (we WANT different
    # solution attempts from the same parent state).
    lane_b_temperature: float = 1.0
    lane_b_top_p: float = 0.95

    # Direct Lane C copies 1..6 frozen golden rubrics and judges all rubric x
    # branch calls concurrently.  It never retrieves experience or generates
    # rubrics.
    judge_temperature: float = 0.02
    judge_top_p: float = 1.0
    judge_max_tokens: int = 20480
    judge_context_length: int = 256000
    judge_format_correction_rounds: int = 8
    direct_rubric_bank_path: str = str(DEFAULT_RUBRIC_BANK)
    enable_variance_detector: bool = True
    collapse_reward_margin: float | None = None

    gt_eval_workers: int = 8
    lane_a_pool_size: int = 1  # Lane A is a single agent
    lane_b_pool_size: int | None = None

    keep_images: bool = False
    # rollout40 continues to terminal for GT diagnostics while Lane C and
    # training remain prefix-bounded by k. Legacy depth2 leaves this false.
    terminal_rollout: bool = False
    reward_kind: str = "joint"
    joint_alpha: float = 1.0
    all_pass_reward: float = 1.0
    disable_rubric: bool = False
    gt_on_submit: bool = False
    # Retained for explicit terminal continuation (rollout40/validation).
    # Strict-k depth2 does not extract fallback patches for unfinished or
    # overlength rollouts; Lane C scores those trajectories directly.
    fallback_patch_penalty: float = 0.5
    no_action_patch_penalty: float = -0.1

    def __post_init__(self) -> None:
        if self.p <= 0:
            raise ValueError("trajectory_search_parallel requires p > 0.")
        if self.m <= 0:
            raise ValueError("trajectory_search_parallel requires m > 0.")
        if self.k <= 0:
            raise ValueError("trajectory_search_parallel requires k > 0.")
        if self.step_limit <= 0:
            raise ValueError("trajectory_search_parallel requires step_limit > 0.")
        if self.gt_eval_workers <= 0:
            raise ValueError("trajectory_search_parallel requires gt_eval_workers > 0.")
        if self.lane_a_pool_size <= 0:
            raise ValueError("trajectory_search_parallel requires lane_a_pool_size > 0.")
        if self.lane_b_pool_size is not None and self.lane_b_pool_size <= 0:
            raise ValueError("trajectory_search_parallel requires lane_b_pool_size > 0 when set.")
        if self.judge_context_length <= 0:
            raise ValueError("trajectory_search_parallel requires judge_context_length > 0.")
        if self.judge_max_tokens <= 0:
            raise ValueError("trajectory_search_parallel requires judge_max_tokens > 0.")
        if self.collapse_reward_margin is not None and (
            not math.isfinite(float(self.collapse_reward_margin))
            or float(self.collapse_reward_margin) <= 0.0
        ):
            raise ValueError("collapse_reward_margin must be finite and positive")
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
        requested = tuple(self.requested_group_kinds)
        if len(set(requested)) != len(requested) or any(
            kind not in {"root", "beam"} for kind in requested
        ):
            raise ValueError("requested_group_kinds must be unique root/beam values")
        if self.topology == "depth1" and "beam" in requested:
            raise ValueError("depth1 cannot request a beam group")


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
    # Lane-A beam parents that formally submit are terminal and must never be
    # resumed by Lane B.  They remain in the instance record for audit only.
    terminated_early: bool = False
    # A policy-overlength parent is not an infra failure, but it also has no
    # resumable endpoint from which Lane B can generate a child.
    continuation_unavailable_reason: str = ""

    @property
    def can_continue(self) -> bool:
        return (
            not self.terminated_early
            and not self.continuation_unavailable_reason
        )


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
    direct_judge: dict[str, Any] = field(default_factory=dict)
    # Final oracle-equivalent score, keyed by the real runtime node id.
    judge_score_by_node: dict[str, float] = field(default_factory=dict)
    training_reward_by_node: dict[str, float] = field(default_factory=dict)
    reward_source_by_node: dict[str, str] = field(default_factory=dict)
    collapse_reason_by_node: dict[str, str] = field(default_factory=dict)
    predicted_zero_variance: bool = False
    lane_c_errors: list[str] = field(default_factory=list)
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
    # Topology-driven omissions are valid outcomes, not failed GRPO groups.
    # Currently only depth2 group 1 can be skipped when both Lane-A parents
    # formally submit before any child continuation is needed.
    skipped_group_reasons: dict[int, str] = field(default_factory=dict)
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
        judge_model_name: str | None,
        config: ParallelSearchConfig,
        harness_namespace: str | None,
        policy_base_url: str,
        rubric_base_url: str,
        policy_api_key: str,
        rubric_api_key: str,
        usage_group_prefix: str | None = None,
        policy_base_urls: list[str] | None = None,
    ) -> None:
        self.instance = copy.deepcopy(instance)
        self.backend = backend
        self.run_dir = Path(run_dir)
        self.policy_model_name = policy_model_name
        self.policy_version = policy_version or policy_model_name
        self.enforce_policy_version = bool(enforce_policy_version)
        self.judge_model_name = judge_model_name or policy_model_name
        self.config = config
        self.harness_namespace = harness_namespace
        self.policy_base_url = policy_base_url.rstrip("/")
        self.policy_base_urls = [
            url.rstrip("/") for url in (policy_base_urls or [self.policy_base_url])
        ]
        self.rubric_base_url = rubric_base_url.rstrip("/")
        self.policy_api_key = policy_api_key
        self.rubric_api_key = rubric_api_key
        self.skip_lane_c = bool(config.disable_rubric)

        self.task = instance["problem_statement"]
        self.task_id = instance["instance_id"]
        requested_usage_prefix = (
            str(usage_group_prefix).strip()
            if usage_group_prefix is not None
            else ""
        )
        self.usage_group_prefix = requested_usage_prefix or self.task_id
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
        rubric_api_base = (
            self.rubric_base_url + "/v1"
            if not self.rubric_base_url.endswith("/v1")
            else self.rubric_base_url
        )
        self.judge_model_kwargs: dict[str, Any] = {
            **_litellm_model_kwargs(self.judge_model_name),
            "api_base": rubric_api_base,
            "api_key": self.rubric_api_key,
            "completion_backend": "litellm",
        }
        self.direct_rubric_bank: DirectRubricBank | None = None
        self.variance_detector: dict[str, Any] | None = None
        if not self.skip_lane_c:
            self.direct_rubric_bank = DirectRubricBank(
                config.direct_rubric_bank_path or DEFAULT_RUBRIC_BANK
            )
            if not self.direct_rubric_bank.is_eligible(self.task_id):
                raise ValueError(
                    f"instance is not eligible for direct rubric training: {self.task_id}"
                )
            if config.enable_variance_detector:
                self.variance_detector = load_variance_detector()
                detector_mapping = tuple(
                    float(value)
                    for value in self.variance_detector.get(
                        "score_mapping", ()
                    )
                )
                if detector_mapping != self.direct_rubric_bank.score_mapping:
                    raise ValueError(
                        "variance detector score mapping does not match the "
                        "direct rubric bank"
                    )


    def _usage_group_id(self, group_index: int) -> str:
        return f"{self.usage_group_prefix}:g{group_index}"

    # -- session/container plumbing ------------------------------------------

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
        tag = f"{self.image_repository}:{tag_kind}-{uuid.uuid4().hex[:6]}"
        self._created_image_tags.append(tag)
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
        model_config = resumed["model"]["config"]
        model_kwargs = model_config["model_kwargs"]
        model_kwargs.update(
            {
                "api_base": (
                    lane_b_base_url
                    if lane_b_base_url.endswith("/v1")
                    else f"{lane_b_base_url}/v1"
                ),
                "api_key": self.policy_api_key,
                "temperature": float(temperature),
                "top_p": float(top_p),
            }
        )
        if "model_name" in model_config:
            model_config["model_name"] = _ensure_litellm_prefix(
                model_config["model_name"]
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

        The trainable response stops after ``budget_steps``. Rollout40 and
        validation can continue an unfinished branch via ``terminal_rollout``
        for terminal GT diagnostics. A formal submission is already terminal,
        so its patch is always extracted and evaluated as a diagnostic (or for
        the explicit GT-on-submit ablation).
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
                completed_steps = max(branch_step_end + 1, 0)
                remaining_steps = max(
                    self.config.step_limit - completed_steps,
                    0,
                )
                if remaining_steps > 0:
                    result = self._step_session(session, max_steps=remaining_steps)

            messages = copy.deepcopy(
                session.agent.messages[before_message_count:]
            )
            branch.messages = messages
            branch.error = exact_rollout_token_error(messages)
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
            if branch.error is not None:
                branch.status = "error"
            branch.terminated_early = result["exit_status"] == "Submitted"
            overlength_status = _policy_overlength_exit_status(result)
            exit_error = _unexpected_rollout_exit(result, lane="lane_b")
            if exit_error is not None:
                branch.error = exit_error
            if overlength_status is not None:
                branch.status = "policy_overlength"
                branch.overlength_reason = overlength_status
                # A depth-2 child can overflow before producing its first
                # new assistant turn because its sampled Lane-A parent is
                # already close to the policy context limit.  The parent
                # assistant turns are still the trainable response for this
                # sample, so treat this as an unfinished policy outcome for
                # Lane C instead of an invalid completion.  A root branch has
                # no such trainable parent and must remain fail-closed.
                if (
                    branch.beam_parent_index is not None
                    and branch.error
                    == "invalid_policy_completion:no_assistant_turns"
                ):
                    branch.error = None
            elif branch.terminated_early or self.config.terminal_rollout:
                # A formal submission is a terminal policy decision even in
                # the normal strict-k training path. Preserve exactly the
                # submitted/fallback patch for GT diagnostics.
                branch.terminal_patch, branch.terminal_patch_from_fallback = (
                    extract_terminal_patch_from_session(result, session)
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
            context_error = str(exc)
            if session is not None and any(
                marker in context_error
                for marker in (
                    "ContextWindowExceeded",
                    "maximum context length",
                    "context length",
                    "Requested token count exceeds",
                )
            ):
                # Provider-side overflow is semantically identical to the
                # structured pre-flight exit: keep the partial trajectory and
                # let Lane C score it as an unfinished rollout.  It is not a
                # formal submission, so do not extract or evaluate a patch.
                branch.error = None
                branch.status = "policy_overlength"
                branch.overlength_reason = "ContextWindowExceeded"
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
                logger.warning(
                    "[%s] lane_b g=%d b=%d stopped by context window dt=%.1fs "
                    "tokens_p=%d tokens_c=%d",
                    self.task_id, effective_group_index, branch_index,
                    time.perf_counter() - branch.started_at,
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
        """Score a Lane B branch with the shared SWE-bench evaluator.

        Formal submissions use a strict 0/1 resolved diagnostic. The direct
        judge remains the training reward unless GT-on-submit is explicitly
        enabled; non-submitted terminal continuations retain the configured
        evaluator recipe for diagnostics.
        """
        t_gt = time.perf_counter()
        patch = normalize_terminal_patch_text(branch.terminal_patch)
        n_action_steps = sum(
            1
            for card in _build_step_cards(branch.events, branch.parent_asst_step)
            if card.get("commands")
        )
        binary_terminal_reward = bool(branch.terminated_early)
        reward_config = (
            EvaluationRewardConfig(
                kind="hard",
                all_pass_reward=1.0,
                fallback_patch_penalty=1.0,
                no_action_patch_penalty=0.0,
            )
            if binary_terminal_reward
            else EvaluationRewardConfig(
                kind=self.config.reward_kind,
                joint_alpha=self.config.joint_alpha,
                all_pass_reward=self.config.all_pass_reward,
                fallback_patch_penalty=(
                    self.config.fallback_patch_penalty
                    if branch.terminal_patch_from_fallback
                    else 1.0
                ),
                no_action_patch_penalty=(
                    self.config.no_action_patch_penalty
                    if n_action_steps == 0
                    else 0.0
                ),
            )
        )
        if not patch:
            # The shared evaluator's legacy empty-patch sentinel is -0.2.
            # Terminal training rewards are explicitly binary, so an empty
            # formal submission is unresolved=0 rather than that sentinel.
            empty_status = "unresolved" if binary_terminal_reward else "empty"
            branch.gt_payload = make_evaluation_payload(
                empty_status,
                output="empty_patch" if binary_terminal_reward else "",
                reward_config=reward_config,
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

    def _branch_visible_step_cards(
        self,
        group: ForkGroup,
        branch: LaneBBranch,
    ) -> list[dict[str, Any]]:
        child_cards = _build_step_cards(
            branch.events, branch.parent_asst_step
        )
        if group.group_kind != "beam":
            return child_cards[: self.config.k]
        if (
            branch.beam_parent_index is None
            or branch.beam_parent_index < 0
            or branch.beam_parent_index >= len(group.parent_mid_cps)
        ):
            raise ValueError(
                f"invalid beam parent for direct judge node={branch.node_id}"
            )
        parent = group.parent_mid_cps[branch.beam_parent_index]
        return (
            copy.deepcopy(parent.parent_step_cards[: self.config.k])
            + child_cards[: self.config.k]
        )

    def _branch_visible_messages(
        self,
        group: ForkGroup,
        branch: LaneBBranch,
    ) -> list[dict[str, Any]]:
        if group.group_kind != "beam":
            return list(branch.messages or [])
        if branch.beam_parent_index is None:
            raise ValueError(
                f"missing beam parent for collapse detector node={branch.node_id}"
            )
        root_messages = list(
            (
                group.mid_cp.snapshot.get("agent", {}).get("state", {}) or {}
            ).get("messages", [])
        )
        parent = group.parent_mid_cps[branch.beam_parent_index]
        parent_messages = list(
            (
                parent.snapshot.get("agent", {}).get("state", {}) or {}
            ).get("messages", [])
        )
        if parent_messages[: len(root_messages)] != root_messages:
            raise ValueError(
                f"beam parent prefix mismatch for node={branch.node_id}"
            )
        return parent_messages[len(root_messages) :] + list(
            branch.messages or []
        )

    def _build_direct_prefix(self, group: ForkGroup) -> dict[str, Any]:
        branches: list[dict[str, Any]] = []
        for branch in group.branches:
            step_cards = self._branch_visible_step_cards(group, branch)
            branches.append(
                {
                    "node_id": branch.node_id,
                    "visible_assistant_steps": len(step_cards),
                    "ended_by_prefix": bool(branch.terminated_early),
                    "step_cards": step_cards,
                    "workspace_meta": copy.deepcopy(
                        branch.workspace_meta or EMPTY_WORKSPACE_META
                    ),
                }
            )
        return {
            "instance_id": self.task_id,
            "branches": branches,
        }

    def _apply_collapse_rewards(
        self,
        group: ForkGroup,
        visible_steps_by_node: dict[str, int],
    ) -> None:
        group.training_reward_by_node = copy.deepcopy(
            group.judge_score_by_node
        )
        group.reward_source_by_node = {
            branch.node_id: "direct_golden_rubric_judge"
            for branch in group.branches
        }
        if self.config.gt_on_submit:
            for branch in group.branches:
                if not branch.terminated_early:
                    continue
                if branch.gt_score is None or not math.isfinite(
                    float(branch.gt_score)
                ):
                    raise ValueError(
                        "GT-on-submit requires a finite terminal evaluator "
                        f"score for node={branch.node_id}"
                    )
                group.training_reward_by_node[branch.node_id] = float(
                    branch.gt_score
                )
                group.reward_source_by_node[branch.node_id] = (
                    "terminal_swebench_binary"
                )
        margin = self.config.collapse_reward_margin
        if margin is None:
            return
        collapse_reward = (
            min(group.training_reward_by_node.values()) - float(margin)
        )
        for branch in group.branches:
            reason = trajectory_collapse_reason(
                self._branch_visible_messages(group, branch),
                assistant_step_limit=visible_steps_by_node[branch.node_id],
            )
            if reason is None:
                continue
            group.training_reward_by_node[branch.node_id] = collapse_reward
            group.reward_source_by_node[branch.node_id] = (
                "code_mode_collapse"
            )
            group.collapse_reason_by_node[branch.node_id] = reason
            logger.warning(
                "[%s] collapse_reward g=%d node=%s reward=%.6f reason=%s",
                self.task_id,
                group.group_index,
                branch.node_id,
                collapse_reward,
                reason,
            )

    async def _run_lane_c(self, *, group: ForkGroup) -> None:
        """Judge one complete policy group with frozen golden rubrics."""
        started = time.perf_counter()
        group.lane_c_started_at = started
        if self.direct_rubric_bank is None:
            raise RuntimeError("direct rubric bank is not initialized")
        try:
            prefix = self._build_direct_prefix(group)
            result = await judge_direct_rubric_group(
                instance_id=self.task_id,
                problem_statement=self.task,
                prefix=prefix,
                bank=self.direct_rubric_bank,
                judge_model_name=self.judge_model_name,
                judge_model_kwargs=self.judge_model_kwargs,
                detector=self.variance_detector,
                context_length=self.config.judge_context_length,
                max_tokens=self.config.judge_max_tokens,
                judge_temperature=self.config.judge_temperature,
                judge_top_p=self.config.judge_top_p,
                format_correction_rounds=(
                    self.config.judge_format_correction_rounds
                ),
            )
            scores = {
                str(node_id): float(score)
                for node_id, score in result["judge_scores"].items()
            }
            expected = {branch.node_id for branch in group.branches}
            if set(scores) != expected or any(
                not math.isfinite(score) for score in scores.values()
            ):
                raise ValueError(
                    "direct judge returned incomplete or non-finite scores"
                )
            group.direct_judge = result
            group.judge_score_by_node = scores
            group.predicted_zero_variance = bool(
                result.get("predicted_zero_variance", False)
            )
            self._apply_collapse_rewards(
                group,
                {
                    str(branch["node_id"]): int(branch["visible_assistant_steps"])
                    for branch in prefix["branches"]
                },
            )
        except Exception as exc:
            group.direct_judge = {
                "schema_version": "direct_golden_rubric_group_judge.v1",
                "instance_id": self.task_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
            group.lane_c_errors = [
                f"lane_c_exception:{type(exc).__name__}:{exc}"
            ]
            logger.warning(
                "[%s] lane_c g=%d direct_judge_FAILED %s",
                self.task_id,
                group.group_index,
                exc,
            )
        finally:
            group.lane_c_done_at = time.perf_counter()
        logger.info(
            "[%s] lane_c g=%d direct_judge_done dt=%.1fs "
            "rubrics=%d branches=%d collapse=%d predicted_zero=%s",
            self.task_id,
            group.group_index,
            group.lane_c_done_at - started,
            len(group.direct_judge.get("rubrics") or []),
            len(group.branches),
            len(group.collapse_reason_by_node),
            group.predicted_zero_variance,
        )

    async def _run_fork_group(
        self,
        *,
        mid_cp: MidCp,
        loop: asyncio.AbstractEventLoop,
        lane_b_pool: ThreadPoolExecutor,
        gt_pool: ThreadPoolExecutor,
        policy_done_event: asyncio.Event,
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
            if (
                (self.config.terminal_rollout or branch.terminated_early)
                and branch.error is None
            ):
                ctx = contextvars.copy_context()
                gt_future = loop.run_in_executor(
                    gt_pool, lambda b=branch, c=ctx: c.run(self._evaluate_gt, b)
                )
                gt_tasks.append(asyncio.create_task(_await_gt(branch, gt_future)))
            else:
                self._dump_branch(group, branch)

        branches = [branches_by_index[index] for index in range(cfg.m)]
        group.branches = branches

        # Policy capacity belongs to Lane A/B only. Release it before the
        # independent evaluator and hosted Lane C drain so the next instance
        # can immediately start rollout work on the inference GPUs.
        policy_done_event.set()

        lane_c_task: asyncio.Task | None = None
        valid_policy_group = (
            len(branches) == cfg.m
            and all(branch.error is None for branch in branches)
        )
        if self.skip_lane_c:
            now = time.perf_counter()
            group.lane_c_started_at = now
            group.lane_c_done_at = now
        elif not valid_policy_group:
            now = time.perf_counter()
            group.lane_c_started_at = now
            group.lane_c_done_at = now
        elif not self.config.gt_on_submit:
            lane_c_task = asyncio.create_task(self._run_lane_c(group=group))

        # Rollout40/validation evaluate every terminal continuation for GT
        # diagnostics. Strict-k depth2 evaluates formal submissions only.
        for done in asyncio.as_completed(gt_tasks):
            branch = await done
            self._dump_branch(group, branch)

        if valid_policy_group and not self.skip_lane_c:
            if self.config.gt_on_submit:
                await self._run_lane_c(group=group)
            elif lane_c_task is not None:
                await lane_c_task

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

    def _create_root_mid_cp(self, lane_a: LaneAState) -> MidCp:
        """Create the common root checkpoint without sampling policy tokens."""
        lane_a.started_at = time.perf_counter()
        session = self._make_initial_session()
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
                overflow_dir = self.run_dir / "groups" / "group_001"
                _atomic_write_json(
                    overflow_dir
                    / f"beam_parent_{parent_index:02d}_overlength.json",
                    {
                        "node_id": node_id,
                        "exit_status": overlength_status,
                    },
                )
                logger.info(
                    "[%s] lane_a parent=%d node=%s unavailable=%s",
                    self.task_id,
                    parent_index,
                    node_id,
                    overlength_status,
                )
                return MidCp(
                    idx=parent_index + 1,
                    asst_step=0,
                    image_tag="",
                    emitted_at=time.perf_counter(),
                    node_id=node_id,
                    continuation_unavailable_reason=(
                        f"policy_overlength:{overlength_status}"
                    ),
                )
            messages = copy.deepcopy(
                session.agent.messages[before_message_count:]
            )
            token_error = exact_rollout_token_error(messages)
            if token_error is not None:
                raise RuntimeError(
                    f"Lane-A parent {parent_index} {token_error}"
                )
            terminated_early = result["exit_status"] == "Submitted"
            snapshot = self._make_resume_snapshot(session)
            image_tag = ""
            if not terminated_early:
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
                "[%s] lane_a parent=%d node=%s steps=%d status=%s submitted=%s",
                self.task_id,
                parent_index,
                node_id,
                asst_step,
                result["status"],
                terminated_early,
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
                terminated_early=terminated_early,
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

    # -- main loop -----------------------------------------------------------

    async def run(
        self,
        *,
        on_group_done=None,
        on_policy_done=None,
    ) -> InstanceRecord:
        """Run the full lane-based pipeline.

        on_group_done: optional sync callback invoked once per ForkGroup
            AFTER (a) all m Lane B branches terminated or hit the remaining
            global step budget,
            and (b) Lane C judge/rubric returned. Callback signature:
            (fork_group: ForkGroup) -> None. Used by slime's rollout fn to
            stream completed groups to the data buffer without waiting for
            the whole instance to finish.
        on_policy_done: optional sync callback invoked once after every
            topology group's Lane B rollout has completed, without waiting
            for SWE-bench evaluation or hosted Lane C. The training collector
            uses this boundary to admit the next policy instance.
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
            "[%s] run.start m=%d k=%d p=%d topology=%s step_limit=%d policy_url=%s judge_url=%s",
            self.task_id,
            self.config.m,
            self.config.k,
            self.config.p,
            self.config.topology,
            self.config.step_limit,
            self.policy_base_url,
            self.rubric_base_url,
        )

        loop = asyncio.get_running_loop()
        cfg = self.config
        requested_group_kinds = set(
            cfg.requested_group_kinds
            or (("root", "beam") if cfg.topology == "depth2" else ("root",))
        )
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
                else cfg.m * (2 if cfg.topology == "depth2" else 1)
            ),
            thread_name_prefix=f"lane-b-{self.task_id[:12]}",
        )
        gt_pool = ThreadPoolExecutor(
            max_workers=cfg.gt_eval_workers,
            thread_name_prefix=f"lane-gt-{self.task_id[:12]}",
        )
        # Per-group on_group_done fires AS SOON AS that group is ready
        # (all m Lane B branches terminal + GT + Lane C done). Slime rollout
        # reads these to stream completed groups without waiting for the whole
        # instance. Direct Lane C has no shared bank state, so root and beam
        # judging can run concurrently while another instance uses Lane A/B.

        group_tasks: list[asyncio.Task] = []
        policy_done_events: list[asyncio.Event] = []

        async def _await_group_then_callback(
            coro,
            policy_done_event: asyncio.Event,
        ):
            try:
                grp = await coro
            except Exception as exc:
                policy_done_event.set()
                logger.warning("[%s] fork-group raised: %s", self.task_id, exc)
                return None
            self._dump_group(grp)
            if getattr(self, "skip_lane_c", False):
                logger.info(
                    "[%s] lane_c g=%d skipped by trajectory-only mode",
                    self.task_id,
                    grp.group_index,
                )
            elif len(grp.branches) != cfg.m or any(
                branch.error is not None for branch in grp.branches
            ):
                logger.warning(
                    "[%s] lane_c g=%d skipped: incomplete/errored policy group",
                    self.task_id,
                    grp.group_index,
                )
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
                if "root" in requested_group_kinds:
                    root_policy_done = asyncio.Event()
                    root_coro = self._run_fork_group(
                        mid_cp=root_mid_cp,
                        loop=loop,
                        lane_b_pool=lane_b_pool,
                        gt_pool=gt_pool,
                        policy_done_event=root_policy_done,
                        group_index=0,
                        group_kind="root",
                    )
                    policy_done_events.append(root_policy_done)
                    group_tasks.append(
                        asyncio.create_task(
                            _await_group_then_callback(
                                root_coro,
                                root_policy_done,
                            )
                        )
                    )

            if cfg.topology == "depth2" and "beam" in requested_group_kinds:
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
                sampled_parent_mid_cps: list[MidCp] = []
                parent_errors: list[str] = []
                for parent_index, item in enumerate(parent_results):
                    if isinstance(item, BaseException):
                        parent_errors.append(
                            f"parent {parent_index}: {type(item).__name__}: {item}"
                        )
                    else:
                        sampled_parent_mid_cps.append(item)
                instance_record.lane_a.mid_cps.extend(sampled_parent_mid_cps)
                if parent_errors:
                    instance_record.lane_a.error = "; ".join(parent_errors)
                    instance_record.lane_a.status = "beam_parent_error"
                    logger.warning(
                        "[%s] beam group skipped: %s",
                        self.task_id,
                        instance_record.lane_a.error,
                    )
                else:
                    parent_mid_cps = [
                        parent
                        for parent in sampled_parent_mid_cps
                        if parent.can_continue
                    ]
                    if not parent_mid_cps:
                        all_submitted = all(
                            parent.terminated_early
                            for parent in sampled_parent_mid_cps
                        )
                        skip_reason = (
                            "all_lane_a_parents_submitted"
                            if all_submitted
                            else "no_continuable_lane_a_parents"
                        )
                        instance_record.skipped_group_reasons[1] = skip_reason
                        instance_record.lane_a.status = f"beam_skipped_{skip_reason}"
                        logger.info(
                            "[%s] beam group skipped: %s (%d sampled parents)",
                            self.task_id,
                            skip_reason,
                            len(sampled_parent_mid_cps),
                        )
                    else:
                        with usage_context(
                            phase="train", group_id=self._usage_group_id(1)
                        ):
                            beam_policy_done = asyncio.Event()
                            beam_coro = self._run_fork_group(
                                mid_cp=root_mid_cp,
                                loop=loop,
                                lane_b_pool=lane_b_pool,
                                gt_pool=gt_pool,
                                policy_done_event=beam_policy_done,
                                group_index=1,
                                group_kind="beam",
                                parent_mid_cps=parent_mid_cps,
                            )
                            policy_done_events.append(beam_policy_done)
                            group_tasks.append(
                                asyncio.create_task(
                                    _await_group_then_callback(
                                        beam_coro,
                                        beam_policy_done,
                                    )
                                )
                            )
                        instance_record.lane_a.status = (
                            f"beam_parents_ready_{len(parent_mid_cps)}_of_"
                            f"{len(sampled_parent_mid_cps)}"
                        )

            if policy_done_events:
                await asyncio.gather(
                    *(event.wait() for event in policy_done_events)
                )
                if on_policy_done is not None:
                    try:
                        on_policy_done()
                    except Exception as cb_exc:
                        logger.warning(
                            "[%s] on_policy_done raised: %s",
                            self.task_id,
                            cb_exc,
                        )

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
            expected_groups = len(requested_group_kinds) - sum(
                1
                for group_index in instance_record.skipped_group_reasons
                if ("root" if group_index == 0 else "beam")
                in requested_group_kinds
            )
            instance_record.completed = (
                instance_record.lane_a.error is None
                and len(instance_record.groups) == expected_groups
                and all(
                    len(g.branches) == cfg.m
                    and all(b.error is None for b in g.branches)
                    and all(
                        not b.terminated_early or b.gt_score is not None
                        for b in g.branches
                    )
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
        _atomic_write_json(gdir / "direct_judge.json", group.direct_judge)
        _atomic_write_json(
            gdir / "judge_score_by_node.json", group.judge_score_by_node
        )
        _atomic_write_json(
            gdir / "training_reward_by_node.json",
            group.training_reward_by_node,
        )
        _atomic_write_json(
            gdir / "collapse_reason_by_node.json",
            group.collapse_reason_by_node,
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
                "judge_score": group.judge_score_by_node.get(branch.node_id),
                "training_reward": group.training_reward_by_node.get(
                    branch.node_id
                ),
                "reward_source": group.reward_source_by_node.get(
                    branch.node_id
                ),
                "collapse_reason": group.collapse_reason_by_node.get(
                    branch.node_id
                ),
                "n_messages": len(branch.messages),
                "n_step_cards": len(branch_step_cards),
                "total_tokens": branch.total_tokens,
                "started_at": branch.started_at,
                "finished_at": branch.finished_at,
            },
        )

    def _dump_group(self, group: ForkGroup) -> None:
        """Persist one fork-group under run_dir/groups/group_NNN/."""
        self._dump_group_scaffold(group)
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
                "lane_c_skipped": bool(getattr(self, "skip_lane_c", False)),
                "num_groups": len(instance_record.groups),
                "skipped_group_reasons": {
                    str(index): reason
                    for index, reason in instance_record.skipped_group_reasons.items()
                },
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
                        "direct_rubric_count": len(
                            g.direct_judge.get("rubrics") or []
                        ),
                        "judge_score_by_node": copy.deepcopy(
                            g.judge_score_by_node
                        ),
                        "training_reward_by_node": copy.deepcopy(
                            g.training_reward_by_node
                        ),
                        "collapse_reason_by_node": copy.deepcopy(
                            g.collapse_reason_by_node
                        ),
                        "predicted_zero_variance": (
                            g.predicted_zero_variance
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
