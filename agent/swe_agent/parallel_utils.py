from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import copy
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import openai
from agent_rl.run_utils import extract_json_from_response
from swe_agent.contracts import ExportGroup, ExportSample, GRPOExportBundle
from swe_agent.prompt import SWE_TRAJECTORY_RUBRIC_GENERATION_PROMPT
from swe_agent.rubric_bank import build_terminal_update_evidence

INVALID_SAMPLE_REWARD = -1.0
RUBRIC_FORMAT_ERROR_REWARD = -1.0
RUBRIC_TERMINAL_ERROR_REWARD = -0.2

MAX_RUBRICS = 6
MAX_RUBRIC_GENERATION_ROUNDS = 10


@dataclass
class TurnTokenInfo:
    turn_index: int
    role: str
    prompt_tokens: int
    completion_tokens: int
    output_token_ids: list[int]
    output_logprobs: list[float]


_RAW_ONLY_KEYS = ("prompt_token_ids", "token_ids", "logprobs", "extra", "usage", "content_no_thinking")


def _strip_token_fields(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in message.items() if key not in _RAW_ONLY_KEYS} for message in messages]


def _stamp_steps(
    messages: list[dict[str, Any]], *, start_step: int
) -> tuple[list[dict[str, Any]], int]:
    stamped: list[dict[str, Any]] = []
    step = start_step
    for message in messages:
        item = dict(message)
        if item.get("role") == "assistant":
            step += 1
        item["step"] = step
        stamped.append(item)
    return stamped, step


def _ensure_litellm_prefix(model_name: str) -> str:
    for prefix in ("openai/", "azure/", "anthropic/", "huggingface/", "hosted_vllm/"):
        if model_name.startswith(prefix):
            return model_name
    return "openai/" + model_name


def _evaluation_error_payload(error: Any) -> dict[str, Any]:
    return {
        "status": "error",
        "resolved": False,
        "reward": 0.0,
        "passed_tests": [],
        "failed_tests": [],
        "error": f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error),
    }


# NOTE: I am not sure if current reward design is approciate, we can iterate on this later.
def progress_reward(
    labels: list[float],
    predictions: list[float],
) -> float:
    label_values = [float(value) for value in labels]
    prediction_values = [float(value) for value in predictions]
    if len(label_values) != len(prediction_values):
        raise ValueError("labels and predictions must have the same length")
    if not label_values:
        return 0.0
    rewards = []
    for label, prediction in zip(label_values, prediction_values):
        if label > 0.5:
            rewards.append(prediction - 0.5)
        elif label < 0.5:
            rewards.append(0.5 - prediction)        
    if not rewards:
        return 0.0
    return float(sum(rewards) / len(rewards))


_COMPACT_JSON_ARRAY_KEYS = {
    "prompt_token_ids",
    "token_ids",
    "logprobs",
    "output_token_ids",
    "output_logprobs",
    "input_token_ids",
    "loss_mask",
    "rollout_logprobs",
}


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _dumps_readable_json(data: Any, *, indent: int = 2) -> str:
    def render(value: Any, level: int, key: str | None = None) -> str:
        if isinstance(value, dict):
            if not value:
                return "{}"
            lines = ["{"]
            items = list(value.items())
            for index, (item_key, item_value) in enumerate(items):
                comma = "," if index < len(items) - 1 else ""
                rendered = render(item_value, level + 1, item_key)
                lines.append(
                    " " * (indent * (level + 1))
                    + json.dumps(str(item_key), ensure_ascii=False)
                    + ": "
                    + rendered
                    + comma
                )
            lines.append(" " * (indent * level) + "}")
            return "\n".join(lines)
        if isinstance(value, list):
            if key in _COMPACT_JSON_ARRAY_KEYS and all(not isinstance(item, (dict, list)) for item in value):
                return json.dumps([_json_safe(item) for item in value], ensure_ascii=False, separators=(",", ":"))
            if not value:
                return "[]"
            lines = ["["]
            for index, item in enumerate(value):
                comma = "," if index < len(value) - 1 else ""
                rendered = render(item, level + 1)
                lines.append(" " * (indent * (level + 1)) + rendered + comma)
            lines.append(" " * (indent * level) + "]")
            return "\n".join(lines)
        return json.dumps(_json_safe(value), ensure_ascii=False)

    return render(data, 0) + "\n"


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(suffix=".tmp", prefix=f"{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(_dumps_readable_json(data))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def extract_terminal_patch_from_session(result: dict[str, Any], session: Any) -> tuple[str, bool]:
    def normalize_patch_text(patch: str) -> str:
        patch = (patch or "").rstrip()
        return patch + "\n" if patch else ""

    patch = normalize_patch_text(str(result.get("submission") or ""))
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
        return normalize_patch_text(str(diff.get("output") or "")), True
    except Exception:
        return "", True


def compact_workspace_meta(workspace_meta: dict[str, Any], *, file_limit: int = 8) -> dict[str, Any]:
    return {
        "head_commit": workspace_meta.get("head_commit", ""),
        "changed_files": list(workspace_meta.get("changed_files", []))[:file_limit],
        "untracked_files": list(workspace_meta.get("untracked_files", []))[:file_limit],
        "diff_stat": workspace_meta.get("diff_stat", ""),
        "current_patch_chars": int(workspace_meta.get("current_patch_chars", 0) or 0),
    }


def rubric_score_record(rubric: Any, score_raw: int, judge_message: Any) -> dict[str, Any]:
    normalized = max(0.0, min(1.0, (float(score_raw) - 1.0) / 4.0))
    importance = abs(float(rubric.weight))
    directional_score = -normalized if rubric.direction == "negative" else normalized
    return {
        "rubric_id": rubric.rubric_id,
        "rubric": {
            "rubric_id": rubric.rubric_id,
            "title": rubric.title,
            "direction": rubric.direction,
            "description": rubric.description,
            "metadata": copy.deepcopy(getattr(rubric, "metadata", {}) or {}),
            "scale": copy.deepcopy(rubric.scale),
            "weight": importance,
            "source_round": rubric.source_round,
        },
        "score_raw": int(score_raw),
        "score_normalized": normalized,
        "weighted_score": importance * directional_score,
        "judge_message": judge_message,
    }


def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    item = {
        "role": message["role"],
        "content": message.get("content", message.get("message", "")) or "",
    }
    if "content_no_thinking" in message:
        item["content_no_thinking"] = message["content_no_thinking"]
    return item

def _normalize_messages(payload: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_messages = payload if isinstance(payload, list) else payload.get("messages")
    messages = [_normalize_message(message) for message in source_messages]
    if len(messages) > 1 and messages[-1]["role"] == "user":
        return messages[:-1]
    return messages

def _normalize_prompt(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [_normalize_message(message) for message in payload.get("messages")]


def _rubric_turn_rewards(
    *,
    payload: dict[str, Any],
    conversation: list[dict[str, Any]],
    alpha: float,
    gamma: float,
    theta: float,
) -> list[float]:
    generated = list(payload.get("generated") or [])
    variance_by_rubric = payload.get("variance_by_rubric") or {}
    redundency_by_rubric = payload.get("redundency_by_rubric") or {}
    judge_error_by_rubric = payload.get("judge_error_by_rubric") or {}
    denominator = alpha + gamma + theta
    if denominator == 0:
        denominator = 1.0
    generated_rewards = [
        (
            alpha * float(variance_by_rubric.get(rubric["rubric_id"], 0.0))
            + gamma * float(redundency_by_rubric.get(rubric["rubric_id"], 0.0))
            + theta * float(judge_error_by_rubric.get(rubric["rubric_id"], 0.0))
        ) / denominator
        for rubric in generated
    ]
    format_error_turns = set()
    for error in payload.get("format_errors") or []:
        format_error_turns.add(int(error.get("turn_index")))
    turn_rewards: list[float] = []
    assistant_turn_index = 0
    generated_index = 0
    for message in conversation:
        if message.get("role") != "assistant":
            continue
        assistant_turn_index += 1
        if assistant_turn_index in format_error_turns:
            turn_rewards.append(RUBRIC_FORMAT_ERROR_REWARD)
        elif generated_index < len(generated_rewards):
            turn_rewards.append(float(generated_rewards[generated_index]))
            generated_index += 1
        else:
            turn_rewards.append(0.0)
    return turn_rewards


@dataclass
class NodeArtifactBundle:
    node_id: str
    node_dir: Path
    node_payload: dict[str, Any]
    messages_payload: dict[str, Any]
    judge_payload: dict[str, Any]
    prompt_payload: dict[str, Any]
    snapshot_payload: dict[str, Any] | None = None
    terminal_messages_payload: dict[str, Any] | None = None
    terminal_patch_payload: dict[str, Any] | None = None


@dataclass
class RubricArtifactBundle:
    rubric_dir: Path
    rubric_payload: dict[str, Any]
    messages_payload: dict[str, Any]
    retrieve_messages_payload: list[dict[str, Any]] | None = None
    round_summary_path: Path | None = None
    selected_for_round_summary: bool = False
    scope: str = "siblings"


class GRPOCollector:
    def __init__(
        self,
        *,
        instance_id: str,
        run_dir: Path,
        gamma: float = 1.0,
        theta: float = 1.0,
    ) -> None:
        self.instance_id = instance_id
        self.run_dir = run_dir
        self.gamma = gamma
        self.theta = theta
        self.bundle = GRPOExportBundle(
            instance_id=instance_id,
            run_dir=str(run_dir),
            policy_groups=[],
            rubric_groups=[],
        )

    def collect(
        self,
        bundles: list[NodeArtifactBundle],
        rubric_bundles: list[RubricArtifactBundle],
        group_id: str,
    ) -> None:
        policy_samples = self._build_policy_samples(bundles, group_id)
        rubric_samples = self._build_rubric_samples(rubric_bundles, group_id)
        if policy_samples:
            self.bundle.policy_groups.append(
                ExportGroup(
                    group_id=group_id,
                    samples=policy_samples,
                )
            )
        if rubric_samples:
            self.bundle.rubric_groups.append(
                ExportGroup(
                    group_id=group_id,
                    samples=rubric_samples,
                )
            )

    def _build_policy_samples(
        self,
        bundles: list[NodeArtifactBundle],
        group_id: str,
    ) -> list[ExportSample]:
        samples: list[ExportSample] = []
        for bundle in bundles:
            turns = _normalize_messages(bundle.messages_payload)
            reward = (
                INVALID_SAMPLE_REWARD
                if bundle.judge_payload.get("is_valid") is False
                else bundle.judge_payload.get("overall_reward", bundle.judge_payload.get("overall_score"))
            )
            if not turns or reward is None:
                continue
            samples.append(
                ExportSample(
                    sample_id=bundle.node_id,
                    group_id=group_id,
                    prompt=_normalize_prompt(bundle.prompt_payload),
                    turns=turns,
                    reward=float(reward),
                )
            )
        return samples

    def _build_rubric_samples(
        self,
        rubric_bundles: list[RubricArtifactBundle],
        group_id: str,
    ) -> list[ExportSample]:
        samples: list[ExportSample] = []
        for bundle in rubric_bundles:
            payload = bundle.rubric_payload
            rubric_list_id = str(
                payload.get("sample_id")
                or payload.get("rubric_list_id")
                or bundle.rubric_dir.name
            )
            conversation = _normalize_messages(bundle.messages_payload)
            if not rubric_list_id or not conversation:
                continue
            if bundle.scope == "siblings":
                reward_key = "gt_reward_siblings"
                turn_reward_alpha = 1.0
            elif bundle.scope == "pc":
                reward_key = "gt_reward_pc"
                turn_reward_alpha = 0.0
            else:
                raise ValueError(f"Unsupported rubric scope: {bundle.scope}")
            turn_rewards = _rubric_turn_rewards(
                payload=payload,
                conversation=conversation,
                alpha=turn_reward_alpha,
                gamma=self.gamma,
                theta=self.theta,
            )
            scalar_reward = float(payload.get(reward_key, 0.0))
            if payload.get("terminal_error"):
                scalar_reward += RUBRIC_TERMINAL_ERROR_REWARD
            samples.append(
                ExportSample(
                    sample_id=rubric_list_id,
                    group_id=group_id,
                    prompt=conversation[:1],
                    turns=conversation[1:],
                    reward=scalar_reward,
                    metadata={
                        "turn_rewards": turn_rewards,
                    },
                )
            )
        return samples


def _write_base_artifacts(
    bundles: list[NodeArtifactBundle] | None = None,
    rubric_bundles: list[RubricArtifactBundle] | None = None,
    extra_json_writes: list[tuple[Path, Any]] | None = None,
) -> None:
    for bundle in bundles or []:
        bundle.node_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(bundle.node_dir / "messages.json", bundle.messages_payload)
        if bundle.snapshot_payload is not None:
            _atomic_write_json(bundle.node_dir / "snapshot.json", bundle.snapshot_payload)
        _atomic_write_json(bundle.node_dir / "node.json", bundle.node_payload)
        if bundle.terminal_messages_payload is not None:
            _atomic_write_json(bundle.node_dir / "terminal_messages.json", bundle.terminal_messages_payload)
        if bundle.terminal_patch_payload is not None:
            _atomic_write_json(bundle.node_dir / "terminal_patch.json", bundle.terminal_patch_payload)
    for bundle in rubric_bundles or []:
        bundle.rubric_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(bundle.rubric_dir / "messages.json", bundle.messages_payload)
        if bundle.retrieve_messages_payload is not None:
            _atomic_write_json(bundle.rubric_dir / "rubric_retrieve_message.json", bundle.retrieve_messages_payload)
    for path, payload in extra_json_writes or []:
        _atomic_write_json(path, payload)


def _write_gt_artifacts(
    bundles: list[NodeArtifactBundle] | None = None,
    rubric_bundles: list[RubricArtifactBundle] | None = None,
    extra_json_writes: list[tuple[Path, Any]] | None = None,
) -> None:
    for bundle in bundles or []:
        bundle.node_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(bundle.node_dir / "judge.json", bundle.judge_payload)
    for bundle in rubric_bundles or []:
        bundle.rubric_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(bundle.rubric_dir / "rubric.json", bundle.rubric_payload)
    for path, payload in extra_json_writes or []:
        _atomic_write_json(path, payload)


def _pairwise_diff(x):
    x = np.asarray(x, dtype=float)
    i, j = np.triu_indices(len(x), 1)
    return x[i] - x[j]


# NOTE: I am not sure if current reward design is approciate, we can iterate on this later.
def gap_corr(a, b):
    da, db = _pairwise_diff(a), _pairwise_diff(b)
    if len(da) == 0:
        return 0.0
    return float(1.0 - np.mean(np.abs(da - db)))


def gap_redundancy(a, b):
    da, db = _pairwise_diff(a), _pairwise_diff(b)
    if len(da) == 0:
        return 0.0
    return float(1.0 - min(np.mean(np.abs(da - db)), np.mean(np.abs(da + db))))


class ArtifactWriter:
    def __init__(self, max_workers: int = 1, write_artifacts: bool = True) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="search-artifacts")
        self._futures: list[Future] = []
        self.write_artifacts = write_artifacts

    def write_round(
        self,
        bundles: list[NodeArtifactBundle],
        rubric_bundles: list[RubricArtifactBundle] | None = None,
        extra_json_writes: list[tuple[Path, Any]] | None = None,
    ) -> None:
        if self.write_artifacts:
            _write_base_artifacts(bundles, rubric_bundles, extra_json_writes)
            _write_gt_artifacts(bundles, rubric_bundles, None)

    def submit_round(
        self,
        bundles: list[NodeArtifactBundle] | None = None,
        rubric_bundles: list[RubricArtifactBundle] | None = None,
        extra_json_writes: list[tuple[Path, Any]] | None = None,
        *,
        write_gt_files: bool = True,
    ) -> Future:
        if write_gt_files:
            future = self._executor.submit(self.write_round, bundles, rubric_bundles, extra_json_writes)
        else:
            future = self._executor.submit(_write_base_artifacts, bundles, rubric_bundles, extra_json_writes)
        self._futures.append(future)
        return future

    def wait(self) -> None:
        while self._futures:
            self._futures.pop(0).result()

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)


class PatchEvalManager:
    def __init__(
        self,
        *,
        instance: dict[str, Any],
        task_id: str,
        model_name: str,
        namespace: str | None,
        work_dir: Path,
        evaluate_patches_fn: Callable[..., dict[str, dict[str, Any]]],
        collector: GRPOCollector | None = None,
        write_artifacts: bool = True,
        max_workers: int = 1,
    ) -> None:
        self.instance = instance
        self.task_id = task_id
        self.model_name = model_name
        self.namespace = namespace
        self.work_dir = work_dir
        self.evaluate_patches_fn = evaluate_patches_fn
        self.collector = collector
        self.write_artifacts = write_artifacts
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="search-gt-eval")
        self._futures: list[Future] = []

    def _evaluate_round(
        self,
        bundles: list[NodeArtifactBundle],
        rubric_bundles: list[RubricArtifactBundle] | None,
        extra_json_writes: list[tuple[Path, Any]] | None,
    ) -> dict[str, Any]:
        terminal_patch_by_node_id: dict[str, str] = {}
        terminal_evaluation_by_node_id: dict[str, dict[str, Any]] = {}

        patches_by_node_id: dict[str, str] = {}
        empty_node_ids: list[str] = []
        for bundle in bundles:
            if bundle.terminal_patch_payload is None:
                continue
            patch = bundle.terminal_patch_payload.get(self.task_id, {}).get("model_patch") or ""
            if not patch.strip():
                # Skip docker grading entirely for empty patches — they always score 0
                # but waste 30 s – 5 min of pytest per node. Record reward=0.0 directly.
                empty_node_ids.append(bundle.node_id)
                continue
            patches_by_node_id[bundle.node_id] = patch

        evaluations: dict[str, dict[str, Any]] = {}
        if patches_by_node_id:
            try:
                evaluations = self.evaluate_patches_fn(
                    instance=self.instance,
                    patches_by_key=patches_by_node_id,
                    model_name=self.model_name,
                    max_workers=1,
                    namespace=self.namespace,
                    work_dir=self.work_dir,
                )
            except Exception as exc:
                evaluations = {node_id: _evaluation_error_payload(exc) for node_id in patches_by_node_id}
            for node_id in patches_by_node_id:
                if node_id not in evaluations:
                    evaluations[node_id] = _evaluation_error_payload("Missing node evaluation")
        for node_id in empty_node_ids:
            evaluations[node_id] = {"reward": 0.0}
        for bundle in bundles:
            if bundle.node_id in evaluations:
                bundle.judge_payload["ground_truth_reward"] = float(evaluations[bundle.node_id]["reward"])

        eval_extra_writes = [
            (bundle.node_dir / "terminal_evalution.json", evaluations[bundle.node_id])
            for bundle in bundles
            if bundle.node_id in evaluations
        ]
        for bundle in bundles:
            node_id = bundle.node_id
            terminal_patch_payload = bundle.terminal_patch_payload or {}
            task_payload = terminal_patch_payload.get(self.task_id, {}) if isinstance(terminal_patch_payload, dict) else {}
            if isinstance(task_payload, dict):
                terminal_patch_by_node_id[node_id] = str(task_payload.get("model_patch") or "")
            if node_id in evaluations:
                terminal_evaluation_by_node_id[node_id] = evaluations[node_id]
        rubric_update_payloads: list[dict[str, Any]] = []
        gt_by_node_id = {node_id: float(payload["reward"]) for node_id, payload in evaluations.items()}
        for bundle in rubric_bundles or []:
            payload = bundle.rubric_payload
            avg_scores = payload["average_rubric_judged_scores"]
            score_by_rubric = payload["score_by_rubric"]
            ordered_node_ids = sorted(node_id for node_id in avg_scores if node_id in gt_by_node_id)
            gt_by_ordered_node_id = {node_id: gt_by_node_id[node_id] for node_id in ordered_node_ids}
            parent_node_id = str(payload.get("parent_node_id") or "root")
            terminal_update_evidence = build_terminal_update_evidence(
                parent_patch=terminal_patch_by_node_id.get(parent_node_id, ""),
                parent_evaluation=terminal_evaluation_by_node_id.get(parent_node_id),
                continuations=[
                    {
                        "patch": terminal_patch_by_node_id.get(node_id, ""),
                        "evaluation": terminal_evaluation_by_node_id.get(node_id),
                    }
                    for node_id in ordered_node_ids
                ],
            )
            if bundle.scope == "siblings":
                ground_truth_by_node = gt_by_ordered_node_id
            elif bundle.scope == "pc":
                parent_node_id = str(payload["parent_node_id"])
                parent_gt = 0.0
                if parent_node_id != "root":
                    parent_judge_path = self.work_dir / "nodes" / parent_node_id / "judge.json"
                    parent_gt_value = json.loads(parent_judge_path.read_text(encoding="utf-8")).get("ground_truth_reward")
                    if parent_gt_value is None:
                        raise RuntimeError(f"parent node {parent_node_id} has no ground_truth_reward")
                    parent_gt = float(parent_gt_value)
                ground_truth_by_node = {
                    node_id: 1.0 if child_gt > parent_gt else 0.0 if child_gt < parent_gt else 0.5
                    for node_id, child_gt in gt_by_ordered_node_id.items()
                }
            else:
                raise ValueError(f"Unsupported rubric scope: {bundle.scope}")
            payload["gt_by_rubric"] = {
                rubric_id: {
                    "child_scores": {
                        node_id: float(scores.get(node_id, 0.0))
                        for node_id in ordered_node_ids
                    },
                    "ground_truth_by_node": ground_truth_by_node,
                }
                for rubric_id, scores in sorted(score_by_rubric.items())
            }
            predicted_scores = [float(avg_scores.get(node_id, 0.0)) for node_id in ordered_node_ids]
            gt_scores = [ground_truth_by_node.get(node_id, 0.0) for node_id in ordered_node_ids]
            if bundle.scope == "siblings":
                payload["reward"] = gap_corr(predicted_scores, gt_scores)
            else:
                payload["reward"] = progress_reward(gt_scores, predicted_scores)
            rubric_update_payloads.append(
                {
                    "scope": bundle.scope,
                    "rubric_list_id": payload.get("rubric_list_id") or payload.get("sample_id") or bundle.rubric_dir.name,
                    **terminal_update_evidence,
                }
            )


        if self.collector is not None:
            parent_id = bundles[0].node_payload.get("parent_id")
            self.collector.collect(bundles, rubric_bundles or [], parent_id)
        if self.write_artifacts:
            _write_gt_artifacts(bundles, rubric_bundles, (extra_json_writes or []) + eval_extra_writes)
        return {
            "evaluations": evaluations,
            "rubric_update_payloads": rubric_update_payloads,
        }

    def submit_round(
        self,
        bundles: list[NodeArtifactBundle],
        rubric_bundles: list[RubricArtifactBundle] | Future | None = None,
        extra_json_writes: list[tuple[Path, Any]] | None = None,
    ) -> Future:
        future = self._executor.submit(self._evaluate_round, bundles, rubric_bundles, extra_json_writes)
        self._futures.append(future)
        return future

    def wait(self) -> list[dict[str, Any]]:
        results = []
        while self._futures:
            results.append(self._futures.pop(0).result())
        return results

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
