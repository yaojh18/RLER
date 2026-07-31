import asyncio
import ast
import copy
import json
import logging
import os
import re
import time
import weakref
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Literal

import jsonlines
import litellm
from agent_rl import ChatCompletion, ChatSamplingParams, call_model_service_async, get_model_service

# Configure LiteLLM to drop unsupported parameters instead of raising errors
litellm.drop_params = True
litellm.turn_off_message_logging = True
litellm.suppress_debug_info = True
litellm.callbacks = []
litellm.success_callback = []
litellm.failure_callback = []
if hasattr(litellm, "_async_success_callback"):
    litellm._async_success_callback = []
if hasattr(litellm, "_async_failure_callback"):
    litellm._async_failure_callback = []
if hasattr(litellm, "_logging") and hasattr(litellm._logging, "_disable_debugging"):
    litellm._logging._disable_debugging()

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelRouteConfig:
    backend: Literal["litellm", "service"] = "litellm"
    service_name: Optional[str] = None
    model_name: Optional[str] = None


_MODEL_ROUTE_CONFIGS: Dict[str, ModelRouteConfig] = {}


# Per-event-loop concurrency control for LiteLLM async calls.
_LITELLM_SEMAPHORES = weakref.WeakKeyDictionary()


def _get_litellm_semaphore() -> asyncio.Semaphore:
    """Return a per-event-loop semaphore limiting concurrent LiteLLM async requests.

    Limit can be configured with env var `LITELLM_MAX_CONCURRENT_CALLS` (default 256).
    """
    loop = asyncio.get_running_loop()
    sem = _LITELLM_SEMAPHORES.get(loop)
    if sem is None:
        raw_max_concurrent = os.environ.get("LITELLM_MAX_CONCURRENT_CALLS", "256")
        try:
            max_concurrent = int(raw_max_concurrent)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "LITELLM_MAX_CONCURRENT_CALLS must be a positive integer, "
                f"got {raw_max_concurrent!r}"
            ) from exc
        if max_concurrent <= 0:
            raise ValueError(
                "LITELLM_MAX_CONCURRENT_CALLS must be a positive integer, "
                f"got {raw_max_concurrent!r}"
            )
        sem = asyncio.Semaphore(max_concurrent)
        _LITELLM_SEMAPHORES[loop] = sem
    return sem


def _default_litellm_timeout_seconds() -> float:
    return float(os.environ.get("LITELLM_DEFAULT_TIMEOUT", os.environ.get("MSWEA_LITELLM_TIMEOUT", "600")))


def _litellm_outer_timeout_seconds(timeout: Any) -> float:
    try:
        timeout_seconds = float(timeout)
    except (TypeError, ValueError):
        return 0.0
    if timeout_seconds <= 0:
        return 0.0
    grace = float(os.environ.get("LITELLM_OUTER_TIMEOUT_GRACE", "5"))
    return timeout_seconds + max(0.0, grace)


def configure_model_route(name: str, config: ModelRouteConfig) -> None:
    _MODEL_ROUTE_CONFIGS[name] = config


def clear_model_routes() -> None:
    _MODEL_ROUTE_CONFIGS.clear()


def get_model_route(name: str) -> Optional[ModelRouteConfig]:
    return _MODEL_ROUTE_CONFIGS.get(name)


def _build_messages(
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    if messages is not None:
        return messages
    return (
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if system_prompt is not None
        else [{"role": "user", "content": user_prompt}]
    )


def _build_service_sampling(chat_kwargs: Dict[str, Any]) -> ChatSamplingParams:
    kwargs = dict(chat_kwargs)
    response_format = kwargs.pop("response_format", None)
    json_mode = response_format == {"type": "json_object"}
    if response_format is not None and not json_mode:
        kwargs["response_format"] = response_format
    return ChatSamplingParams(
        temperature=kwargs.pop("temperature", 0),
        top_p=kwargs.pop("top_p", 1.0),
        max_tokens=kwargs.pop("max_tokens", kwargs.pop("max_completion_tokens", 16384)),
        stop=kwargs.pop("stop", None),
        json_mode=json_mode,
        extra=kwargs,
    )


def _split_inline_thinking_content(content: str) -> tuple[str, str]:
    if not isinstance(content, str):
        return content, content
    if not content.startswith("<think>"):
        return content, content
    closing_tag = "</think>"
    closing_index = content.find(closing_tag)
    if closing_index < 0:
        return content, content
    content_no_thinking = content[closing_index + len(closing_tag) :].lstrip("\r\n")
    return content, content_no_thinking


def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    json_end = response.rfind("}") + 1
    if json_end == 0:
        return None
    
    # Try to find valid JSON by testing different starting positions
    json_start = response.find("{")
    while json_start != -1 and json_start < json_end:
        json_str = response[json_start:json_end]
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                # Clean the JSON string of potential invisible characters and extra whitespace
                cleaned_json = json_str.strip().encode('utf-8').decode('utf-8-sig')
                return json.loads(cleaned_json)
            except json.JSONDecodeError:
                try:
                    # Fix doubled braces (e.g., '{{' -> '{', '}}' -> '}')
                    fixed_braces = json_str.replace('{{', '{').replace('}}', '}')
                    return json.loads(fixed_braces)
                except json.JSONDecodeError:
                    # Try next { position
                    json_start = response.find("{", json_start + 1)
                    continue
        break

    LOGGER.warning(f"Could not decode JSON from response: {repr(response)}")
    return None


def _repair_json_object_text(candidate: str) -> str:
    """Repair common line-oriented JSON mistakes without guessing structure."""
    lines = candidate.strip().splitlines()
    repaired: List[str] = []
    member_re = re.compile(r'^(?P<indent>\s*)"(?P<key>[^"\\]+)"\s*:\s*(?P<value>.*?)\s*$')
    json_scalar_re = re.compile(r'-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null')

    for line in lines:
        match = member_re.match(line)
        if match is None:
            repaired.append(line)
            continue
        raw_value = match.group("value").rstrip()
        comma = "," if raw_value.endswith(",") else ""
        value = raw_value[:-1].rstrip() if comma else raw_value
        if value and not (
            value.startswith(('"', "{", "["))
            or json_scalar_re.fullmatch(value)
        ):
            value = json.dumps(value, ensure_ascii=False)
        repaired.append(f'{match.group("indent")}"{match.group("key")}": {value}{comma}')

    # Models often omit the comma after a bare string or a nested object. Only
    # insert one when the following line is unambiguously another object member.
    for index in range(len(repaired) - 1):
        current = repaired[index].rstrip()
        following = repaired[index + 1].lstrip()
        if not re.match(r'"[^"\\]+"\s*:', following):
            continue
        if current.endswith((",", "{", "[")):
            continue
        if member_re.match(current) is not None or current in {"}", "]"}:
            repaired[index] = current + ","

    return re.sub(r",\s*([}\]])", r"\1", "\n".join(repaired))


def _load_relaxed_json_object(candidate: str) -> Optional[Dict[str, Any]]:
    cleaned = candidate.strip().encode("utf-8").decode("utf-8-sig")
    for value in (cleaned, _repair_json_object_text(cleaned)):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, UnicodeDecodeError):
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                continue
        if isinstance(parsed, dict):
            return parsed
    return None


def extract_last_json_object(response: str) -> Optional[Dict[str, Any]]:
    if not isinstance(response, str):
        return None
    decoder = json.JSONDecoder()
    candidates: List[tuple[int, int, Dict[str, Any]]] = []
    for match in re.finditer(r"\{", response):
        try:
            parsed, length = decoder.raw_decode(response[match.start() :])
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(parsed, dict):
            candidates.append((match.start() + length, match.start(), parsed))
    # If the final object is malformed, a strict-only scan can silently return
    # an earlier draft object from the thought text. Only repair suffixes ending
    # at the final closing brace; normal valid responses stay on the fast path.
    final_end = response.rfind("}") + 1
    if candidates and max(candidates, key=lambda item: (item[0], -item[1]))[0] == final_end:
        return max(candidates, key=lambda item: (item[0], -item[1]))[2]
    opening_positions = [match.start() for match in re.finditer(r"\{", response[:final_end])][-64:]
    for start in opening_positions:
        parsed = _load_relaxed_json_object(response[start:final_end])
        if parsed is not None:
            candidates.append((final_end, start, parsed))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], -item[1]))[2]


def freeform_thought_model_kwargs(model_kwargs: Dict[str, Any] | None) -> Dict[str, Any]:
    kwargs = copy.deepcopy(model_kwargs or {})
    extra_body = kwargs.setdefault("extra_body", {})
    if not isinstance(extra_body, dict):
        raise TypeError("model_kwargs.extra_body must be a dict")
    chat_template_kwargs = extra_body.setdefault("chat_template_kwargs", {})
    if not isinstance(chat_template_kwargs, dict):
        raise TypeError("model_kwargs.extra_body.chat_template_kwargs must be a dict")
    chat_template_kwargs["enable_thinking"] = True
    return kwargs


def load_jsonlines(file):
    with jsonlines.open(file, "r") as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode="w") as writer:
        writer.write_all(data)


async def run_litellm_completion_async(
    model_name: str,
    user_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    usage_model_role: Optional[str] = None,
    **chat_kwargs,
) -> ChatCompletion:
    """Run one hosted logical call and account only its final success.

    Retry attempts and failed/fallback requests are deliberately omitted from
    usage accounting.  With no fallback configured, LiteLLM retains ownership
    of its established retry behavior.  The early-prediction launcher may
    configure a fallback route, in which case this wrapper surfaces retries so
    a capacity error can switch providers immediately.
    """
    from swe_agent.usage import (
        current_usage_context,
        infer_model_family,
        new_logical_call_id,
        record_model_usage,
    )

    msgs = _build_messages(system_prompt=system_prompt, user_prompt=user_prompt, messages=messages)
    raw_completion_cap = os.environ.get(
        "RLER_HOSTED_MAX_COMPLETION_TOKENS",
        os.environ.get("RLER_MAX_COMPLETION_TOKENS"),
    )
    completion_cap = (
        int(raw_completion_cap)
        if raw_completion_cap is not None
        else None
    )
    raw_requested_max_tokens = chat_kwargs.get(
        "max_tokens",
        chat_kwargs.get("max_completion_tokens", completion_cap),
    )
    requested_max_tokens = (
        int(raw_requested_max_tokens)
        if raw_requested_max_tokens is not None
        else None
    )
    if completion_cap is not None and requested_max_tokens is not None:
        requested_max_tokens = min(requested_max_tokens, completion_cap)
    if requested_max_tokens is not None:
        if "max_completion_tokens" in chat_kwargs and "max_tokens" not in chat_kwargs:
            chat_kwargs["max_completion_tokens"] = requested_max_tokens
        else:
            chat_kwargs["max_tokens"] = requested_max_tokens
    raw_context_cap = os.environ.get("RLER_HOSTED_MODEL_CONTEXT_LENGTH")
    context_cap = int(raw_context_cap) if raw_context_cap is not None else 0
    known_prompt_tokens: int | None = None
    if context_cap > 0:
        try:
            known_prompt_tokens = int(
                litellm.token_counter(model=model_name, messages=msgs)
            )
        except Exception as exc:
            LOGGER.warning(
                "Could not preflight hosted-model context for %s: %s",
                model_name,
                exc,
            )
        else:
            if known_prompt_tokens >= context_cap:
                raise litellm.exceptions.ContextWindowExceededError(
                    message=(
                        "ContextWindowExceeded (pre-flight: "
                        f"prompt_tokens={known_prompt_tokens} >= "
                        f"context={context_cap})"
                    ),
                    model=model_name,
                    llm_provider="litellm",
                )
            available_completion_tokens = context_cap - known_prompt_tokens
            if (
                requested_max_tokens is not None
                and requested_max_tokens > available_completion_tokens
            ):
                LOGGER.warning(
                    "Clipping hosted-model max completion for %s from %d to "
                    "%d so prompt_tokens=%d stays within context=%d",
                    model_name,
                    requested_max_tokens,
                    available_completion_tokens,
                    known_prompt_tokens,
                    context_cap,
                )
                requested_max_tokens = available_completion_tokens
                if (
                    "max_completion_tokens" in chat_kwargs
                    and "max_tokens" not in chat_kwargs
                ):
                    chat_kwargs["max_completion_tokens"] = requested_max_tokens
                else:
                    chat_kwargs["max_tokens"] = requested_max_tokens
    rate_limit_fallback_model = str(
        os.environ.get("RLER_LITELLM_RATE_LIMIT_FALLBACK_MODEL") or ""
    ).strip()
    if rate_limit_fallback_model == model_name:
        rate_limit_fallback_model = ""
    if rate_limit_fallback_model:
        configured_retries = max(
            0,
            int(
                chat_kwargs.pop(
                    "num_retries",
                    os.environ.get("RLER_LITELLM_EXPLICIT_RETRIES", "4"),
                )
            ),
        )
        # Only the explicit early-prediction fallback path owns physical
        # attempts.  This makes a GLM 429/529 switch to Ultra immediately.
        chat_kwargs["num_retries"] = 0
    else:
        # Preserve the original generic helper contract: LiteLLM owns retries
        # and receives the caller's value (or the historical default of four).
        configured_retries = 0
        chat_kwargs["num_retries"] = chat_kwargs.get("num_retries", 4)
    active_model_name = model_name
    fallback_used = False
    chat_kwargs["timeout"] = chat_kwargs.get("timeout", _default_litellm_timeout_seconds())
    context = current_usage_context()
    logical_call_id = context.logical_call_id or new_logical_call_id()
    base_attempt_index = int(context.attempt_index or 0)
    response = None
    last_exception: Exception | None = None
    response_latency_ms = 0.0
    response_request_started_at: float | None = None

    max_physical_attempts = (
        configured_retries + 1 + int(bool(rate_limit_fallback_model))
    )
    for retry_index in range(max_physical_attempts):
        started: float | None = None
        request_started_at: float | None = None
        try:
            async with _get_litellm_semaphore():
                started = time.monotonic()
                request_started_at = time.time()
                completion_call = asyncio.to_thread(
                    litellm.completion,
                    messages=msgs,
                    model=active_model_name,
                    **chat_kwargs,
                )
                outer_timeout = _litellm_outer_timeout_seconds(
                    chat_kwargs.get("timeout")
                )
                if outer_timeout > 0:
                    response = await asyncio.wait_for(
                        completion_call,
                        timeout=outer_timeout,
                    )
                else:
                    response = await completion_call
        except Exception as exc:
            if started is None:
                raise
            last_exception = exc

            rate_limit_type = getattr(litellm, "RateLimitError", None)
            is_rate_limit = (
                isinstance(rate_limit_type, type)
                and isinstance(exc, rate_limit_type)
            )
            status_code = getattr(exc, "status_code", None)
            is_capacity_error = is_rate_limit or status_code in {429, 529}
            if (
                is_capacity_error
                and rate_limit_fallback_model
                and active_model_name == model_name
            ):
                active_model_name = rate_limit_fallback_model
                fallback_used = True
                LOGGER.warning(
                    "Hosted primary model %s rate-limited; immediately "
                    "falling back to %s",
                    model_name,
                    active_model_name,
                )
                continue
            if is_capacity_error:
                # Do not stall the whole rollout pipeline when both hosted
                # routes reject this call, or when no fallback was configured.
                # The enclosing judge group fails closed and the collector
                # moves on to another source attempt.
                break

            # Preserve the established correction flow. It is a retry result,
            # so it is intentionally absent from approximate usage accounting.
            if isinstance(exc, litellm.JSONSchemaValidationError):
                raw_content = exc.raw_response if isinstance(exc.raw_response, str) else str(exc.raw_response)
                return ChatCompletion(
                    content=raw_content,
                    finish_reason="stop",
                    model_name=active_model_name,
                    cost=0.0,
                    metadata={
                        "timestamp": time.time(),
                        "validation_error": str(exc),
                    },
                )

            _exc_msg = str(exc)
            _is_overflow = (
                isinstance(exc, getattr(litellm, "ContextWindowExceededError", ()))
                or "ContextWindowExceededError" in type(exc).__name__
                or "Requested token count exceeds" in _exc_msg
                or "longer than the model's context length" in _exc_msg
                or "maximum context length" in _exc_msg
            )
            non_retryable_types = tuple(
                error_type
                for error_type in (
                    asyncio.TimeoutError,
                    getattr(litellm, "AuthenticationError", None),
                    getattr(litellm, "PermissionDeniedError", None),
                    getattr(litellm, "BadRequestError", None),
                    getattr(litellm, "NotFoundError", None),
                    getattr(litellm, "UnsupportedParamsError", None),
                )
                if isinstance(error_type, type)
            )
            if _is_overflow:
                print(f"Error in run_litellm_completion_async (FATAL, raising): {exc}")
                raise
            if (
                active_model_name == model_name
                and retry_index < configured_retries
                and not isinstance(exc, non_retryable_types)
            ):
                continue
            break
        else:
            response_latency_ms = (time.monotonic() - started) * 1000
            response_request_started_at = request_started_at
            break

    if response is None:
        exc = last_exception or RuntimeError("LiteLLM returned no response")
        print(f"Error in run_litellm_completion_async: {exc}")
        return ChatCompletion(
            content="",
            model_name=active_model_name,
            metadata={
                "timestamp": time.time(),
                "error": f"{type(exc).__name__}: {exc}",
                "timeout": chat_kwargs.get("timeout") if isinstance(exc, asyncio.TimeoutError) else None,
                "primary_model": model_name,
                "rate_limit_fallback_model": (
                    rate_limit_fallback_model if fallback_used else None
                ),
            },
        )

    choice = response.choices[0]
    message = choice.message
    inline_content = message.content or ""
    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content is None:
        reasoning_content = getattr(message, "reasoning", None)
    if reasoning_content is None:
        provider_specific_fields = getattr(message, "provider_specific_fields", None)
        if isinstance(provider_specific_fields, dict):
            reasoning_content = provider_specific_fields.get("reasoning_content")
    if reasoning_content:
        content_no_thinking = inline_content
    else:
        _, content_no_thinking = _split_inline_thinking_content(inline_content)
    content = inline_content
    if reasoning_content:
        content = (
            f"<think>{reasoning_content}</think>\n{content_no_thinking}"
            if content_no_thinking
            else f"<think>{reasoning_content}</think>"
        )
    usage = {}
    if getattr(response, "usage", None) is not None:
        usage = response.usage.model_dump() if hasattr(response.usage, "model_dump") else dict(response.usage)

    event = record_model_usage(
        usage=usage,
        known_input_tokens=known_prompt_tokens,
        status="success",
        model_family=infer_model_family(active_model_name),
        model_role=usage_model_role or "litellm",
        logical_call_id=logical_call_id,
        attempt_index=base_attempt_index,
        latency_ms=response_latency_ms,
        request_started_at=response_request_started_at,
    )
    return ChatCompletion(
        content=content,
        finish_reason=choice.finish_reason,
        model_name=active_model_name,
        cost=0.0,
        usage=usage,
        metadata={
            "timestamp": time.time(),
            "content_no_thinking": content_no_thinking,
            "usage_event_ids": [event["event_id"]] if event is not None else [],
            "primary_model": model_name,
            "rate_limit_fallback_model": (
                rate_limit_fallback_model if fallback_used else None
            ),
        },
    )


async def run_generate_with_route_async(
    *,
    route_name: str,
    input_ids: List[int],
    api_base: str,
    api_key: str = "EMPTY",
    sampling_params: Optional[Dict[str, Any]] = None,
    return_logprobs: bool = True,
    require_reasoning: bool = False,
    request_timeout: float = 600.0,
) -> ChatCompletion:
    """Token-in / token-out call against sglang's `/generate` endpoint.

    Pre-conditions:
      - `input_ids` has been chat-template-encoded by the caller using the
        same tokenizer sglang loaded (see swe_agent.tokenization).
      - `api_base` ends in `/v1` (chat-completions endpoint convention) or
        is the bare hostname; this function strips `/v1` and posts to
        `{base}/generate`.

    Returns ChatCompletion with:
      - content              = decoded text of generated tokens
      - output_token_ids     = sglang's exact output_ids
      - output_logprobs      = per-token chosen logprobs in token order
      - input_token_ids      = echo of the input_ids we sent
      - usage                = {prompt_tokens, completion_tokens}
      - finish_reason        = inferred from sglang meta_info
    """
    import aiohttp  # local import — only token-IO path needs aiohttp
    if sampling_params is None:
        sampling_params = {}
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    url = f"{base}/generate"
    payload = {
        "input_ids": list(input_ids),
        "sampling_params": sampling_params,
        "return_logprob": bool(return_logprobs),
        # -1 = output tokens only. 0 would request logprobs over the FULL
        # prompt too, which forces sglang to allocate a (prompt_len, vocab)
        # buffer per request. With long chat-template prompts (16k+ tokens)
        # this triggered GPU OOM in the scheduler.
        "logprob_start_len": -1,
        "stream": False,
    }
    if require_reasoning:
        payload["require_reasoning"] = True
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"
    timeout = aiohttp.ClientTimeout(total=request_timeout)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(
                        f"sglang /generate {resp.status}: {body[:500]}"
                    )
                data = await resp.json()
    except Exception as exc:
        # Mirror the litellm path: swallow non-fatal errors so the agent
        # retry loop in models/utils/retry.py can re-attempt. ContextWindow
        # / overflow errors are NOT swallowed (sglang returns 4xx and we
        # raise above).
        msg = str(exc)
        if "longer than" in msg or "context length" in msg or "Requested token count exceeds" in msg:
            raise
        print(f"Error in run_generate_with_route_async: {exc}")
        return ChatCompletion(
            content="",
            model_name=route_name,
            metadata={
                "timestamp": time.time(),
                "error": msg,
                "usage_event_ids": [],
            },
        )

    # sglang /generate response shape:
    #   {
    #     "text": "<decoded text>",
    #     "output_ids": [int, ...],
    #     "meta_info": {
    #         "prompt_tokens": N,
    #         "completion_tokens": M,
    #         "finish_reason": {"type": "stop"|"length"|...},
    #         "output_token_logprobs": [(logprob, token_id, decoded), ...],  # if return_logprob
    #         "input_token_logprobs": [...],                                  # if requested
    #     }
    #   }
    text = data.get("text", "") or ""
    content, content_no_thinking = _split_inline_thinking_content(text)
    output_ids = list(data.get("output_ids") or [])
    meta = data.get("meta_info") or {}
    finish_info = meta.get("finish_reason") or {}
    if isinstance(finish_info, dict):
        finish_reason = finish_info.get("type", "stop") or "stop"
    else:
        finish_reason = str(finish_info) or "stop"
    prompt_tokens = meta.get("prompt_tokens")
    completion_tokens = meta.get("completion_tokens")
    usage = {
        # Raw token ids are authoritative when older SGLang versions omit
        # usage fields.
        "prompt_tokens": len(input_ids)
        if prompt_tokens is None
        else int(prompt_tokens),
        "completion_tokens": len(output_ids)
        if completion_tokens is None
        else int(completion_tokens),
    }
    for key in ("cached_tokens", "cached_input_tokens", "prompt_tokens_details"):
        if key in meta:
            usage[key] = meta[key]
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    output_logprobs: List[float] = []
    raw_lp = meta.get("output_token_logprobs") or []
    for entry in raw_lp:
        if isinstance(entry, (list, tuple)) and len(entry) >= 1:
            try:
                output_logprobs.append(float(entry[0]))
            except (TypeError, ValueError):
                pass
        elif isinstance(entry, dict) and "logprob" in entry:
            try:
                output_logprobs.append(float(entry["logprob"]))
            except (TypeError, ValueError):
                pass
    return ChatCompletion(
        content=content,
        finish_reason=finish_reason,
        model_name=route_name,
        cost=0.0,
        usage=usage,
        metadata={
            "timestamp": time.time(),
            "content_no_thinking": content_no_thinking,
            "usage_event_ids": [],
        },
        output_token_ids=output_ids,
        output_logprobs=output_logprobs,
        input_token_ids=list(input_ids),
    )


def _response_format_json_schema(response_format: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(response_format, dict):
        return None
    if response_format.get("type") == "json_schema":
        schema = response_format.get("json_schema", {}).get("schema")
        if not isinstance(schema, dict):
            raise ValueError("response_format json_schema has no dict schema")
        return schema
    if response_format.get("type") == "json_object":
        return {"type": "object"}
    return None


def _strip_chat_only_message_fields(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stripped: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError(f"message must be dict, got {type(message).__name__}")
        item = {
            "role": message.get("role", "assistant") or "assistant",
            "content": message.get("content", "") or "",
        }
        if "tool_calls" in message:
            item["tool_calls"] = message["tool_calls"]
        stripped.append(item)
    return stripped


def _normalize_messages_for_generate(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError(f"message must be dict, got {type(message).__name__}")
        role = message.get("role", "assistant") or "assistant"
        item: Dict[str, Any] = {"role": role, "content": message.get("content", "") or ""}
        if role == "assistant" and message.get("token_ids"):
            item["token_ids"] = list(message["token_ids"])
        normalized.append(item)
    return normalized


def _completion_to_assistant_message(
    completion: Any,
) -> Dict[str, Any]:
    content = completion.content or ""
    assistant_message: Dict[str, Any] = {
        "role": "assistant",
        "content": content,
        "content_no_thinking": completion.metadata.get("content_no_thinking", content),
        "usage": dict(completion.usage) if completion.usage else {},
    }
    if completion.input_token_ids:
        assistant_message["prompt_token_ids"] = list(completion.input_token_ids)
    if completion.output_token_ids:
        assistant_message["token_ids"] = list(completion.output_token_ids)
    if completion.output_logprobs:
        assistant_message["logprobs"] = list(completion.output_logprobs)
    return assistant_message


def _route_service_base(
    route_name: str,
    model_name: str,
    api_key: str,
) -> tuple[str | None, str, str]:
    route = get_model_route(route_name)
    effective_model = route.model_name if route is not None and route.model_name else model_name
    if route is None or route.backend != "service" or not route.service_name:
        return None, api_key, effective_model
    service = get_model_service(route.service_name)
    api_base = getattr(service, "_base_url", None)
    if not api_base:
        raise RuntimeError(f"service route {route_name!r} has no base url")
    return (
        api_base,
        getattr(service, "_api_key", api_key),
        route.model_name or getattr(service, "default_model_name", None) or model_name,
    )


async def route_completion_message(
    *,
    route_name: str,
    model_name: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    response_format: Dict[str, Any] | None = None,
    model_kwargs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    kwargs = copy.deepcopy(model_kwargs or {})
    api_base = kwargs.pop("api_base", None)
    api_key = kwargs.pop("api_key", "EMPTY")
    completion_backend = str(
        kwargs.pop("completion_backend", "auto") or "auto"
    ).strip().lower()
    if completion_backend not in {"auto", "sglang_generate", "litellm"}:
        raise ValueError(
            "model_kwargs.completion_backend must be one of "
            "'auto', 'sglang_generate', or 'litellm'"
        )
    enable_json_schema_validation = bool(kwargs.pop("enable_json_schema_validation", True))
    extra_body = kwargs.get("extra_body", {}) if isinstance(kwargs.get("extra_body"), dict) else {}
    if api_base is None:
        api_base, api_key, effective_model = _route_service_base(route_name, model_name, api_key)
    else:
        effective_model = model_name

    # An explicit hosted-chat transport prevents OpenAI-compatible provider
    # endpoints from being mistaken for a local SGLang server.  Keep a narrow
    # NVIDIA-host fallback so the canonical experiment endpoint is safe even
    # when an older caller omits the new marker.
    hosted_nvidia_chat = False
    if api_base:
        from urllib.parse import urlparse

        hostname = (urlparse(str(api_base)).hostname or "").lower()
        hosted_nvidia_chat = hostname in {
            "inference-api.nvidia.com",
            "integrate.api.nvidia.com",
        }
    use_litellm = completion_backend == "litellm" or (
        completion_backend == "auto" and hosted_nvidia_chat
    )
    if use_litellm:
        # The NVIDIA gateway's Azure GLM-5.2 deployment rejects the
        # OpenAI-compatible ``chat_template_kwargs`` extension with HTTP 400.
        # That extension is only a tokenizer hint; the deployment controls its
        # own reasoning template.  Strip it narrowly for this exact transport
        # while leaving the canonical NVIDIA route and every other extra-body
        # field unchanged.
        if effective_model.lower().startswith("openai/azure/"):
            hosted_extra_body = kwargs.get("extra_body")
            if isinstance(hosted_extra_body, dict):
                hosted_extra_body = copy.deepcopy(hosted_extra_body)
                hosted_extra_body.pop("chat_template_kwargs", None)
                if hosted_extra_body:
                    kwargs["extra_body"] = hosted_extra_body
                else:
                    kwargs.pop("extra_body", None)
        completion = await run_litellm_completion_async(
            model_name=effective_model,
            messages=_strip_chat_only_message_fields(messages),
            usage_model_role=route_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            response_format=response_format,
            api_base=api_base,
            api_key=api_key,
            **kwargs,
        )
        error = completion.metadata.get("error")
        if error is not None:
            raise RuntimeError(f"{route_name} hosted chat completion failed: {error}")
        return _completion_to_assistant_message(completion)

    if api_base:
        from swe_agent.tokenization import get_stop_token_ids, tokenize_messages_with_template

        normalized = _normalize_messages_for_generate(messages)
        enable_thinking = bool(
            (extra_body.get("chat_template_kwargs") or {}).get("enable_thinking", True)
        )
        input_ids = tokenize_messages_with_template(
            normalized,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            model_path=effective_model,
        )
        sampling_params: Dict[str, Any] = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_new_tokens": int(max_tokens),
            "stop_token_ids": get_stop_token_ids(model_path=effective_model),
        }
        schema = _response_format_json_schema(response_format)
        if schema is not None and enable_json_schema_validation:
            sampling_params["json_schema"] = json.dumps(schema, ensure_ascii=False)
        for key in ("top_k", "min_p", "frequency_penalty", "presence_penalty", "repetition_penalty"):
            if key in kwargs:
                sampling_params[key] = kwargs[key]
        completion = await run_generate_with_route_async(
            route_name=route_name,
            input_ids=input_ids,
            api_base=api_base,
            api_key=api_key,
            sampling_params=sampling_params,
            return_logprobs=True,
            require_reasoning=bool(enable_thinking and schema is not None),
        )
        error = completion.metadata.get("error")
        if error is not None:
            raise RuntimeError(f"{route_name} token-in/out generation failed: {error}")
        return _completion_to_assistant_message(
            completion,
        )

    completion = await run_chat_with_route_completion_async(
        route_name,
        model_name=model_name,
        messages=_strip_chat_only_message_fields(messages),
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        response_format=response_format,
        **kwargs,
    )
    return _completion_to_assistant_message(
        completion,
    )


async def run_chat_with_route_completion_async(
    route_name: str,
    model_name: str,
    user_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    **chat_kwargs,
) -> ChatCompletion:
    route = get_model_route(route_name)
    routed_kwargs = dict(chat_kwargs)
    policy_version = routed_kwargs.pop("policy_version", None)
    # LOGGER.debug(f"Running route backend: {route.backend}")
    if route is None or route.backend == "litellm":
        routed_model_name = route.model_name if route and route.model_name else model_name
        return await run_litellm_completion_async(
            model_name=routed_model_name,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            messages=messages,
            usage_model_role=route_name,
            **routed_kwargs,
        )
    try:
        if route.service_name is None:
            raise RuntimeError(f"Route {route_name} is configured for service backend without a service_name")

        completion = await call_model_service_async(
            route.service_name,
            messages=_build_messages(system_prompt=system_prompt, user_prompt=user_prompt, messages=messages),
            model_name=route.model_name or model_name,
            sampling=_build_service_sampling(routed_kwargs),
            policy_version=policy_version,
        )
    except Exception as e:
        print(f"Error in run_chat_with_route_async({route_name}): {e}")
        return ChatCompletion(content="")
    return completion

async def run_chat_with_route_async(
    route_name: str,
    model_name: str,
    user_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    **chat_kwargs,
) -> str:
    completion = await run_chat_with_route_completion_async(
        route_name,
        model_name,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        messages=messages,
        **chat_kwargs,
    )
    return completion.content


if __name__ == "__main__":
    pass
