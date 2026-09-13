from __future__ import annotations

from typing import Any

from slime.utils.types import Sample
from swe_agent.contracts import ExportGroup


def build_rollout_samples(
    *,
    groups: list[ExportGroup],
    group_index_offset: int = 0,
    max_sample_tokens: int | None = None,
) -> tuple[list[Sample], int]:
    """Convert exact SWE rollout exports into trainable Slime samples."""
    samples: list[Sample] = []
    truncated = 0
    for group_index, group in enumerate(groups):
        for export_sample in group.samples:
            if export_sample.group_id != group.group_id:
                raise ValueError(
                    f"Sample {export_sample.sample_id} declares group "
                    f"{export_sample.group_id!r}, expected {group.group_id!r}."
                )
            if (
                export_sample.token_ids is None
                or export_sample.loss_mask is None
                or export_sample.response_length is None
                or export_sample.rollout_logprobs is None
                or export_sample.reward is None
            ):
                raise ValueError(
                    f"Sample {export_sample.sample_id} is missing exact "
                    "rollout training fields."
                )
            token_ids = list(export_sample.token_ids)
            full_loss_mask = list(export_sample.loss_mask)
            response_length = int(export_sample.response_length)
            rollout_logprobs = list(export_sample.rollout_logprobs)
            if len(token_ids) != len(full_loss_mask):
                raise ValueError(
                    f"Sample {export_sample.sample_id}: token_ids len "
                    f"({len(token_ids)}) != loss_mask len "
                    f"({len(full_loss_mask)})"
                )
            if response_length <= 0 or response_length > len(token_ids):
                raise ValueError(
                    f"Sample {export_sample.sample_id} has invalid "
                    f"response_length={response_length}."
                )
            if len(rollout_logprobs) != response_length:
                raise ValueError(
                    f"Sample {export_sample.sample_id}: rollout_logprobs len "
                    f"({len(rollout_logprobs)}) != response_length "
                    f"({response_length})"
                )

            messages: list[dict[str, Any]] = []
            for message in export_sample.prompt + export_sample.turns:
                role = str(message.get("role"))
                if role not in {"system", "user", "assistant", "tool"}:
                    continue
                item = {
                    "role": role,
                    "content": str(message.get("content")),
                    "step_loss_mask": 1 if role == "assistant" else 0,
                }
                if "content_no_thinking" in message:
                    item["content_no_thinking"] = str(
                        message.get("content_no_thinking")
                    )
                messages.append(item)

            was_truncated = (
                max_sample_tokens is not None
                and len(token_ids) > max_sample_tokens
            )
            if was_truncated:
                if max_sample_tokens <= 0:
                    raise ValueError(
                        "max_sample_tokens must be positive, got "
                        f"{max_sample_tokens}"
                    )
                response_start = len(token_ids) - response_length
                retained_response_length = max(
                    0,
                    max_sample_tokens - response_start,
                )
                token_ids = token_ids[:max_sample_tokens]
                full_loss_mask = full_loss_mask[:max_sample_tokens]
                rollout_logprobs = rollout_logprobs[:retained_response_length]
                response_length = retained_response_length
                if response_length <= 0:
                    raise ValueError(
                        f"Sample {export_sample.sample_id} has no response "
                        "tokens after truncation."
                    )
                truncated += 1
                export_sample.metadata = {
                    **(export_sample.metadata or {}),
                    "right_truncated_tokens": (
                        len(export_sample.token_ids) - max_sample_tokens
                    ),
                }

            sample = Sample(
                group_index=group_index_offset + group_index,
                prompt=messages,
                tokens=token_ids,
                response_length=response_length,
                reward=float(export_sample.reward),
                loss_mask=full_loss_mask[-response_length:],
                status=(
                    Sample.Status.TRUNCATED
                    if was_truncated
                    else Sample.Status.COMPLETED
                ),
            )
            sample.metadata = {
                **(export_sample.metadata or {}),
                "export_sample_id": str(export_sample.sample_id),
                "export_group_id": str(group.group_id),
            }
            sample.rollout_log_probs = rollout_logprobs
            samples.append(sample)
    return samples, truncated
