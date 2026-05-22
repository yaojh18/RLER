"""Shared tokenizer + chat-template helpers for token-level training pipeline.

The PDS pipeline (parallel_data_search.py + pds_to_grpo_bundle.py) keeps
token IDs alongside text content for every conversation message, so that
training samples can be assembled WITHOUT a separate tokenization pass on
the slime side. This avoids text round-tripping mismatches between sglang
(rollout) and Megatron (training) — both must share the same tokenizer
+ chat template, and we centralize that here.

Conventions:
  - Single global tokenizer instance, lazily initialized from the model
    path the agent's config knows about. Loaded with trust_remote_code=True
    to pick up Qwen3.5's chat template.
  - `tokenize_messages_with_template(messages, ..., add_generation_prompt)`
    reproduces what sglang would do for a chat completion request: applies
    the model's chat template (including <|im_start|> role markers and
    <|im_end|> separators), returning a flat token id list.
  - `compute_loss_mask_for_messages(messages)` returns a per-token list of
    {0,1} aligned with the tokenized output — 1 for tokens that fall inside
    an assistant response (= what we want to train on), 0 otherwise.
"""
from __future__ import annotations

import os
import threading
from typing import Any

from transformers import AutoTokenizer


_TOKENIZER = None
_TOKENIZER_PATH = None
_TOKENIZER_LOCK = threading.Lock()


def _resolve_tokenizer_path(model_path: str | None = None) -> str:
    path = model_path or os.environ.get("SWE_AGENT_TOKENIZER_PATH") or os.environ.get("SWE_AGENT_GRPO_MODEL_NAME")
    if not path:
        raise RuntimeError("Tokenizer path is required; pass model_path or set SWE_AGENT_TOKENIZER_PATH/SWE_AGENT_GRPO_MODEL_NAME.")
    for prefix in ("openai/", "azure/", "anthropic/"):
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


def get_tokenizer(model_path: str | None = None):
    """Lazy-load the shared tokenizer. Reused across all PDS workers."""
    global _TOKENIZER, _TOKENIZER_PATH, _ASST_GEN_PROMPT_IDS, _NEWLINE_IDS, _IM_END_ID
    path = _resolve_tokenizer_path(model_path)
    with _TOKENIZER_LOCK:
        if _TOKENIZER is None or _TOKENIZER_PATH != path:
            _TOKENIZER = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            _TOKENIZER_PATH = path
            _ASST_GEN_PROMPT_IDS = None
            _NEWLINE_IDS = None
            _IM_END_ID = None
    return _TOKENIZER


# Cached token-id sequences for the role markers / boundaries used by the
# splice-based renderer below. Populated lazily by _ensure_special_tokens().
_ASST_GEN_PROMPT_IDS: list[int] | None = None  # `<|im_start|>assistant\n`
_NEWLINE_IDS: list[int] | None = None           # `\n` (boundary between messages)
_IM_END_ID: int | None = None                   # `<|im_end|>`


def _ensure_special_tokens(model_path: str | None = None) -> None:
    """Populate the cached id sequences once (idempotent)."""
    global _ASST_GEN_PROMPT_IDS, _NEWLINE_IDS, _IM_END_ID
    if _IM_END_ID is not None:
        return
    tok = get_tokenizer(model_path)
    _ASST_GEN_PROMPT_IDS = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
    _NEWLINE_IDS = tok.encode("\n", add_special_tokens=False)
    _IM_END_ID = int(tok.convert_tokens_to_ids("<|im_end|>"))


def _encode_text_piece(text: str, *, model_path: str | None = None) -> list[int]:
    """Tokenize a literal text piece with no special-token augmentation."""
    return list(get_tokenizer(model_path).encode(text, add_special_tokens=False))


def tokenize_messages_with_template(
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool = False,
    enable_thinking: bool = True,
    model_path: str | None = None,
) -> list[int]:
    """Render a chat-style message list to token IDs via PURE SPLICING.

    Why splice instead of using a chat template:
      sglang's /generate path stores each assistant call's exact
      `token_ids` (what the model produced) on its message dict. Splicing
      those tokens VERBATIM into subsequent prompts guarantees byte-exact
      preservation of what sglang saw — there is no tokenize/detokenize
      round-trip that could drop a `\\n` or split `</think>` differently.
      That, in turn, makes a_i.prompt_token_ids a TRUE PREFIX of
      a_j.prompt_token_ids for i < j, which is required for the
      multi-turn loss-mask placement in pds_to_grpo_bundle.

    Rendering rules (one pass over messages):
      - system  -> `<|im_start|>system\\n{content}<|im_end|>\\n`
      - user    -> `<|im_start|>user\\n{content}<|im_end|>\\n`
      - tool    -> `<|im_start|>user\\n<tool_response>\\n{content}\\n</tool_response><|im_end|>\\n`
      - asst:
          - if message has `token_ids` (PDS rollout path): splice those
            tokens, sandwiched by `<|im_start|>assistant\\n` and
            `\\n`. If `token_ids` doesn't already end with `<|im_end|>`
            (sglang truncated), append one before the trailing `\\n` so
            the turn is properly closed.
          - else (eval / non-PDS): tokenize the text content wrapped as
            `<|im_start|>assistant\\n{content}<|im_end|>\\n`.
      - end:   if `add_generation_prompt`, append
            `<|im_start|>assistant\\n` (the gen prompt).

    `enable_thinking` is ignored here: if a thinking model is used, the
    generated assistant content itself must include `<think>...</think>`.
    """
    del enable_thinking
    _ensure_special_tokens(model_path)
    result: list[int] = []
    for i, msg in enumerate(messages):
        role = msg.get("role") or "assistant"
        content = msg.get("content") or ""
        if role == "system":
            if i != 0:
                raise ValueError("System message must be at the beginning.")
            result.extend(_encode_text_piece(f"<|im_start|>system\n{content}<|im_end|>\n", model_path=model_path))
        elif role == "user":
            result.extend(_encode_text_piece(f"<|im_start|>user\n{content}<|im_end|>\n", model_path=model_path))
        elif role == "tool":
            result.extend(_encode_text_piece(
                f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n",
                model_path=model_path,
            ))
        elif role == "assistant":
            stored = msg.get("token_ids")
            if stored:
                # SPLICE PATH (PDS rollout / training): use sglang's exact
                # output tokens for byte-perfect alignment across turns.
                result.extend(_ASST_GEN_PROMPT_IDS)  # type: ignore[arg-type]
                stored_list = list(stored)
                result.extend(stored_list)
                # If the model hit max_new_tokens, sglang returns output
                # without the stop token; close the turn so the next role's
                # `<|im_start|>` is preceded by `<|im_end|>\n` as expected.
                if not stored_list or stored_list[-1] != _IM_END_ID:
                    result.append(int(_IM_END_ID))  # type: ignore[arg-type]
                result.extend(_NEWLINE_IDS)  # type: ignore[arg-type]
            else:
                # FALLBACK PATH (eval / no stored tokens): match what the
                # splice path would have produced if we'd run the content
                # text through the model — wrap with the same prefix/suffix.
                result.extend(_encode_text_piece(
                    f"<|im_start|>assistant\n{content}<|im_end|>\n",
                    model_path=model_path,
                ))
        else:
            raise ValueError(f"Unexpected message role: {role!r}")
    if add_generation_prompt:
        result.extend(_ASST_GEN_PROMPT_IDS)  # type: ignore[arg-type]
    return result


def get_stop_token_ids(model_path: str | None = None) -> list[int]:
    """Token ids that should terminate /generate calls for this model.

    Chat completions endpoint auto-derives these from the chat template; the
    raw /generate path does NOT — without this, the model emits a real
    response then keeps generating past <|im_end|> until max_new_tokens
    (we observed an infinite <think></think> loop tail).

    Includes both <|im_end|> (chat-template turn terminator) and the
    tokenizer's configured eos_token_id (generation_config default).
    """
    tok = get_tokenizer(model_path)
    ids: list[int] = []
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        ids.append(int(im_end))
    eos = getattr(tok, "eos_token_id", None)
    if isinstance(eos, int) and eos >= 0 and eos not in ids:
        ids.append(int(eos))
    return ids


def compute_loss_mask_for_messages(
    messages: list[dict[str, Any]],
    *,
    enable_thinking: bool = True,
    model_path: str | None = None,
) -> tuple[list[int], list[int]]:
    """Tokenize `messages` and return (token_ids, loss_mask) where loss_mask
    is 1 for tokens that fall inside an assistant turn's response payload
    and 0 elsewhere (system prompt, user query, tool observations,
    chat-template scaffolding, etc.).

    Implementation: tokenize the cumulative prefix up to (but not including)
    each assistant message, then the cumulative prefix INCLUDING that
    message; the difference is the assistant's contribution. Mark those
    positions as trainable.
    """
    full_ids = tokenize_messages_with_template(
        messages,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
        model_path=model_path,
    )
    loss_mask = [0] * len(full_ids)
    # Walk message by message and mark assistant ranges.
    prev_len = 0
    for i in range(len(messages)):
        prefix_ids = tokenize_messages_with_template(
            messages[: i + 1],
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
            model_path=model_path,
        )
        cur_len = len(prefix_ids)
        if messages[i].get("role") == "assistant":
            # Mark [prev_len, cur_len) as trainable. We could refine to skip
            # the <|im_start|>assistant header but Megatron's mask treats
            # that header similarly so keeping it simple matches behavior.
            for k in range(prev_len, min(cur_len, len(loss_mask))):
                loss_mask[k] = 1
        prev_len = cur_len
    return full_ids, loss_mask
