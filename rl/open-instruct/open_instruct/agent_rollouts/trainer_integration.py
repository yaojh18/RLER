from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from agent_rl import RolloutResult, RolloutSessionSpec, TrainingProjection
from transformers import PreTrainedTokenizerBase

from .projection import pack_rollout_results


@dataclass
class SessionSpecBatch:
    specs: list[RolloutSessionSpec]
    ground_truths: list[Any]
    datasets: list[Any]
    raw_user_queries: list[str]


@dataclass
class ExternalInferenceBatch:
    queries: list[list[int]]
    responses: list[list[int]]
    masks: list[list[int]]
    finish_reasons: list[str]
    infos: tuple[list[int], list[bool], list[str], list[str], list[float], list[bool]]
    results: list[RolloutResult]
    projections: list[TrainingProjection]


def build_rollout_session_batch(
    raw_user_queries: Sequence[str],
    ground_truths: Sequence[Any],
    datasets: Sequence[Any],
    *,
    training_step: int,
    num_samples_per_prompt_rollout: int,
    policy_ref: str = "policy",
    policy_version: str | None = None,
    task_id_prefix: str = "train",
    limits: dict[str, Any] | None = None,
    pause_points: Sequence[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> SessionSpecBatch:
    specs: list[RolloutSessionSpec] = []
    expanded_ground_truths: list[Any] = []
    expanded_datasets: list[Any] = []
    expanded_raw_user_queries: list[str] = []

    for prompt_index, raw_user_query in enumerate(raw_user_queries):
        group_id = f"{task_id_prefix}:{training_step}:{prompt_index}"
        for sample_index in range(num_samples_per_prompt_rollout):
            session_id = f"{group_id}:{sample_index}"
            specs.append(
                RolloutSessionSpec(
                    session_id=session_id,
                    task=raw_user_query,
                    task_id=f"{task_id_prefix}:{training_step}:{prompt_index}",
                    group_id=group_id,
                    sample_index=sample_index,
                    policy_ref=policy_ref,
                    policy_version=policy_version,
                    dataset_name=copy.deepcopy(datasets[prompt_index]),
                    ground_truth=copy.deepcopy(ground_truths[prompt_index]),
                    raw_user_query=raw_user_query,
                    limits=copy.deepcopy(limits or {}),
                    pause_points=list(pause_points or []),
                    metadata=copy.deepcopy(extra_metadata or {}),
                )
            )
            expanded_ground_truths.append(copy.deepcopy(ground_truths[prompt_index]))
            expanded_datasets.append(copy.deepcopy(datasets[prompt_index]))
            expanded_raw_user_queries.append(raw_user_query)

    return SessionSpecBatch(
        specs=specs,
        ground_truths=expanded_ground_truths,
        datasets=expanded_datasets,
        raw_user_queries=expanded_raw_user_queries,
    )


def _collect_tool_outputs(result: RolloutResult) -> str:
    outputs: list[str] = []
    for event in result.events:
        if event.kind not in {"environment_result", "agent_interrupt"}:
            continue
        for message in event.payload.get("messages", []):
            role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
            content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
            if role == "exit" or content in (None, ""):
                continue
            outputs.append(str(content))
    return "\n".join(outputs)


def _collect_tool_errors(result: RolloutResult) -> str:
    if result.status != "failed":
        return ""
    return result.exit_status or "rollout_failed"


def _count_tool_calls(result: RolloutResult) -> int:
    count = 0
    for event in result.events:
        if event.kind == "environment_action":
            count += len(event.payload.get("actions", []))
    return count


def build_external_inference_batch(
    results: Iterable[RolloutResult],
    tokenizer: PreTrainedTokenizerBase,
    *,
    pad_token_id: int,
    pack_length: int,
) -> ExternalInferenceBatch:
    result_list = list(results)
    packed_batch = pack_rollout_results(
        result_list,
        tokenizer,
        pad_token_id=pad_token_id,
        pack_length=pack_length,
    )
    projections = packed_batch.projections
    num_calls = [_count_tool_calls(result) for result in result_list]
    finish_reasons = [projection.finish_reason for projection in projections]
    timeouts = [False] * len(result_list)
    tool_errors = [_collect_tool_errors(result) for result in result_list]
    tool_outputs = [_collect_tool_outputs(result) for result in result_list]
    tool_runtimes = [0.0] * len(result_list)
    tool_calleds = [num_call > 0 for num_call in num_calls]
    return ExternalInferenceBatch(
        queries=[projection.prompt_token_ids for projection in projections],
        responses=[projection.continuation_token_ids for projection in projections],
        masks=[projection.trainable_mask for projection in projections],
        finish_reasons=finish_reasons,
        infos=(num_calls, timeouts, tool_errors, tool_outputs, tool_runtimes, tool_calleds),
        results=result_list,
        projections=projections,
    )
