from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, List

from agent_rl import ModelTurn, RolloutResult, TrainingProjection
from transformers import PreTrainedTokenizerBase

from open_instruct.rl_utils2 import PackedSequences, pack_sequences


@dataclass
class TokenizedModelTurn:
    session_id: str
    step_index: int
    query_token_ids: List[int]
    response_token_ids: List[int]
    trainable_mask: List[int]
    reward: float | None
    metadata: dict[str, Any]


@dataclass
class PackedRolloutBatch:
    packed_sequences: PackedSequences
    tokenized_turns: List[TokenizedModelTurn]


@dataclass
class PackedEpisodeBatch:
    packed_sequences: PackedSequences
    projections: List[TrainingProjection]


def _sanitize_message(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        message = message.model_dump(mode="python")
    message = copy.deepcopy(message)
    sanitized = {k: v for k, v in message.items() if k not in {"metadata", "trainable", "source", "extra"}}
    if "role" not in sanitized or "content" not in sanitized:
        raise ValueError(f"Invalid chat message: {message}")
    return sanitized


def tokenize_model_turn(turn: ModelTurn, tokenizer: PreTrainedTokenizerBase) -> TokenizedModelTurn:
    query_messages = [_sanitize_message(message) for message in turn.query_messages]
    response_message = _sanitize_message(turn.response_message)
    if response_message.get("content") in (None, ""):
        raise ValueError(
            "Model turns with empty response content need a backend-specific text normalizer before tokenization."
        )

    query_token_ids = tokenizer.apply_chat_template(query_messages, add_generation_prompt=True)
    full_token_ids = tokenizer.apply_chat_template(query_messages + [response_message])
    response_token_ids = full_token_ids[len(query_token_ids) :]
    if not response_token_ids:
        raise ValueError(f"Could not derive response tokens for model turn {turn.session_id}:{turn.step_index}")

    return TokenizedModelTurn(
        session_id=turn.session_id,
        step_index=turn.step_index,
        query_token_ids=query_token_ids,
        response_token_ids=response_token_ids,
        trainable_mask=[1] * len(response_token_ids),
        reward=turn.reward,
        metadata=copy.deepcopy(turn.metadata),
    )


def pack_model_turns(
    turns: List[ModelTurn],
    tokenizer: PreTrainedTokenizerBase,
    *,
    pad_token_id: int,
    pack_length: int,
) -> PackedRolloutBatch:
    tokenized_turns = [tokenize_model_turn(turn, tokenizer) for turn in turns]
    packed_sequences = pack_sequences(
        queries=[turn.query_token_ids for turn in tokenized_turns],
        responses=[turn.response_token_ids for turn in tokenized_turns],
        masks=[turn.trainable_mask for turn in tokenized_turns],
        pack_length=pack_length,
        pad_token_id=pad_token_id,
    )
    return PackedRolloutBatch(
        packed_sequences=packed_sequences,
        tokenized_turns=tokenized_turns,
    )


def _sanitize_role(role: str) -> str:
    if role == "exit":
        return "assistant"
    return role


def _tokenize_full_messages(
    tokenizer: PreTrainedTokenizerBase,
    messages: List[dict[str, Any]],
    *,
    add_generation_prompt: bool = False,
) -> List[int]:
    return tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)


def tokenize_rollout_result(result: RolloutResult, tokenizer: PreTrainedTokenizerBase) -> TrainingProjection:
    if not result.model_turns:
        raise ValueError(f"Rollout result {result.session_id} does not contain any model turns.")

    prompt_messages = [_sanitize_message(message) for message in result.model_turns[0].query_messages]
    prompt_token_ids = _tokenize_full_messages(tokenizer, prompt_messages, add_generation_prompt=True)
    final_messages = [
        copy.deepcopy(message.model_dump(mode="python") if hasattr(message, "model_dump") else message)
        for message in result.final_messages
    ]
    continuation_messages = final_messages[len(prompt_messages) :]

    prefixes = copy.deepcopy(prompt_messages)
    previous_token_ids = list(prompt_token_ids)
    continuation_token_ids: List[int] = []
    trainable_mask: List[int] = []
    assistant_message_count = 0
    observation_segments = 0

    for message in continuation_messages:
        role = message.get("role", "")
        if role == "exit":
            continue
        sanitized_message = _sanitize_message({**message, "role": _sanitize_role(role)})
        cumulative_token_ids = _tokenize_full_messages(tokenizer, prefixes + [sanitized_message])
        delta = cumulative_token_ids[len(previous_token_ids) :]
        if not delta:
            raise ValueError(
                f"Could not derive continuation tokens for rollout result {result.session_id} "
                f"at message index {len(prefixes)}."
            )
        is_trainable = sanitized_message["role"] == "assistant"
        continuation_token_ids.extend(delta)
        trainable_mask.extend([1 if is_trainable else 0] * len(delta))
        assistant_message_count += int(is_trainable)
        observation_segments += int(not is_trainable)
        prefixes.append(sanitized_message)
        previous_token_ids = cumulative_token_ids

    if not continuation_token_ids:
        raise ValueError(f"Rollout result {result.session_id} does not contain a tokenizable continuation.")

    finish_reason = "stop" if result.status == "finished" else "length"
    spec = result.spec
    metadata = copy.deepcopy(result.metadata)
    metadata.update(
        {
            "assistant_message_count": assistant_message_count,
            "observation_segment_count": observation_segments,
            "exit_status": result.exit_status,
            "submission": result.submission,
        }
    )
    return TrainingProjection(
        session_id=result.session_id,
        group_id=spec.group_id,
        policy_ref=spec.policy_ref,
        policy_version=spec.policy_version,
        prompt_token_ids=prompt_token_ids,
        continuation_token_ids=continuation_token_ids,
        trainable_mask=trainable_mask,
        finish_reason=finish_reason,
        dataset_name=spec.dataset_name,
        ground_truth=spec.ground_truth,
        raw_user_query=spec.raw_user_query,
        metadata=metadata,
    )


def pack_rollout_results(
    results: List[RolloutResult],
    tokenizer: PreTrainedTokenizerBase,
    *,
    pad_token_id: int,
    pack_length: int,
) -> PackedEpisodeBatch:
    projections = [tokenize_rollout_result(result, tokenizer) for result in results]
    packed_sequences = pack_sequences(
        queries=[projection.prompt_token_ids for projection in projections],
        responses=[projection.continuation_token_ids for projection in projections],
        masks=[projection.trainable_mask for projection in projections],
        pack_length=pack_length,
        pad_token_id=pad_token_id,
    )
    return PackedEpisodeBatch(
        packed_sequences=packed_sequences,
        projections=projections,
    )
