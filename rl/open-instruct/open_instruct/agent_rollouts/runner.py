from __future__ import annotations

from typing import Any, Iterable, Tuple

from agent_rl import RolloutResult, RolloutSessionSpec
from transformers import PreTrainedTokenizerBase

from .projection import PackedRolloutBatch, pack_model_turns
from .registry import get_rollout_backend_class


class ExternalRolloutRunner:
    def __init__(
        self,
        backend: str | type,
        *,
        backend_kwargs: dict[str, Any] | None = None,
        tokenizer: PreTrainedTokenizerBase,
        pad_token_id: int,
        pack_length: int,
    ) -> None:
        backend_cls = get_rollout_backend_class(backend) if isinstance(backend, str) else backend
        self.backend = backend_cls(**(backend_kwargs or {}))
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.pack_length = pack_length

    def run(self, specs: Iterable[RolloutSessionSpec], *, max_steps: int | None = None) -> Tuple[list[RolloutResult], PackedRolloutBatch]:
        results: list[RolloutResult] = []
        turns = []
        for spec in specs:
            session = self.backend.create_session(spec)
            result = session.run_until_pause(max_steps=max_steps)
            results.append(result)
            turns.extend(result.model_turns)
        packed_batch = pack_model_turns(
            turns,
            self.tokenizer,
            pad_token_id=self.pad_token_id,
            pack_length=self.pack_length,
        )
        return results, packed_batch
