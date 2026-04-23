from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

try:
    from slime.utils.mask_utils import MultiTurnLossMaskGenerator
    from slime.utils.processing_utils import load_tokenizer
    from slime.utils.types import Sample
except ModuleNotFoundError:
    from slime.slime.utils.mask_utils import MultiTurnLossMaskGenerator
    from slime.slime.utils.processing_utils import load_tokenizer
    from slime.slime.utils.types import Sample

try:
    from slime.swe_agent.contracts import ExportSample
except ModuleNotFoundError:
    from swe_agent.contracts import ExportSample


def build_training_messages(prompt: list[dict[str, Any]], turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for message in prompt:
        messages.append(
            {
                "role": str(message.get("role") or ""),
                "content": str(message.get("content") or ""),
                "step_loss_mask": 0,
            }
        )
    for message in turns:
        messages.append(
            {
                "role": str(message.get("role") or ""),
                "content": str(message.get("content") or ""),
                "step_loss_mask": 1 if str(message.get("role") or "") == "assistant" else 0,
            }
        )
    return messages


def build_sft_jsonl_records(samples: list[ExportSample]) -> list[dict[str, Any]]:
    return [{"messages": build_training_messages(sample.prompt, sample.turns)} for sample in samples]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_debug_sample(
    *,
    export_sample: ExportSample,
    tokenizer,
    loss_mask_type: str,
    sample_index: int,
    group_index: int,
    include_turn_rewards: bool,
) -> Sample:
    messages = build_training_messages(export_sample.prompt, export_sample.turns)
    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=loss_mask_type)
    token_ids, full_loss_mask = mask_generator.get_loss_mask(messages)
    response_length = mask_generator.get_response_lengths([full_loss_mask])[0]
    if response_length <= 0:
        raise ValueError(f"Sample {export_sample.sample_id} does not contain any trainable assistant tokens.")

    sample = Sample(
        group_index=group_index,
        index=sample_index,
        prompt=messages,
        tokens=token_ids,
        response_length=response_length,
        reward=float(export_sample.reward or 0.0),
        loss_mask=full_loss_mask[-response_length:],
        status=Sample.Status.COMPLETED,
        metadata={**export_sample.metadata, "group_id": export_sample.group_id, "raw_reward": export_sample.reward},
    )

    if not include_turn_rewards:
        return sample

    assistant_indices = [index for index, message in enumerate(messages) if message["role"] == "assistant" and message["step_loss_mask"] == 1]
    turn_masks: list[list[int]] = []
    for assistant_index in assistant_indices:
        isolated_messages = []
        for index, message in enumerate(messages):
            isolated_messages.append(
                {
                    "role": message["role"],
                    "content": message["content"],
                    "step_loss_mask": 1 if index == assistant_index else 0,
                }
            )
        isolated_token_ids, isolated_mask = mask_generator.get_loss_mask(isolated_messages)
        if isolated_token_ids != token_ids:
            raise ValueError(f"Turn mask tokenization drift for sample {export_sample.sample_id}")
        turn_masks.append(isolated_mask[-response_length:])

    sample.train_metadata = {
        "turn_rewards": list(export_sample.metadata.get("turn_rewards", [])),
        "turn_loss_masks": turn_masks,
        "group_id": export_sample.group_id,
        "raw_reward": export_sample.reward,
    }
    return sample


def write_debug_rollout_data(
    *,
    path: Path,
    samples: list[Sample],
    rollout_id: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rollout_id": rollout_id, "samples": [sample.to_dict() for sample in samples]}, path)


def load_tokenizer_and_type(hf_checkpoint: str | Path, loss_mask_type: str = "qwen3_5"):
    return load_tokenizer(str(hf_checkpoint), trust_remote_code=True), loss_mask_type
