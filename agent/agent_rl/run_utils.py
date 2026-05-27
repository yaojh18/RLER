import json
import asyncio
import weakref
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Literal

import jsonlines
import litellm
from agent_rl import ChatCompletion, ChatSamplingParams, call_model_service_async

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


# Per-event-loop concurrency control for LiteLLM async calls to avoid event loop binding issues
_LITELLM_SEMAPHORES = weakref.WeakKeyDictionary()


def _get_litellm_semaphore() -> asyncio.Semaphore:
    """Return a per-event-loop semaphore limiting concurrent LiteLLM async requests.

    Limit can be configured with env var `LITELLM_MAX_CONCURRENT_CALLS` (default 256).
    """
    loop = asyncio.get_running_loop()
    sem = _LITELLM_SEMAPHORES.get(loop)
    if sem is None:
        max_concurrent = int(os.environ.get("LITELLM_MAX_CONCURRENT_CALLS", "256"))
        sem = asyncio.Semaphore(max_concurrent)
        _LITELLM_SEMAPHORES[loop] = sem
    return sem


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
    if not isinstance(content, str) or not content.startswith("<think>"):
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
    **chat_kwargs,
) -> ChatCompletion:
    msgs = _build_messages(system_prompt=system_prompt, user_prompt=user_prompt, messages=messages)
    chat_kwargs["num_retries"] = chat_kwargs.get("num_retries", 4)
    chat_kwargs["timeout"] = chat_kwargs.get(
        "timeout", float(os.environ.get("LITELLM_DEFAULT_TIMEOUT", "600"))
    )
    try:
        async with _get_litellm_semaphore():
            response = await asyncio.to_thread(litellm.completion, messages=msgs, model=model_name, **chat_kwargs)
    except Exception as exc:
        if isinstance(exc, litellm.JSONSchemaValidationError):
            raw_content = exc.raw_response if isinstance(exc.raw_response, str) else str(exc.raw_response)
            return ChatCompletion(
                content=raw_content,
                finish_reason="stop",
                model_name=model_name,
                cost=0.0,
                raw_response={"validation_error": str(exc)},
                metadata={"timestamp": time.time(), "content_no_thinking": raw_content},
            )
        # ContextWindow / token-count overflow must NOT be silently swallowed: doing so
        # causes the agent loop to inject a format_error template and retry with a
        # still-longer prompt, growing the conversation past the cap (we observed
        # 250k -> 263k+ via +152-tok-per-iter loops).
        # sglang/litellm reports overflow under TWO exception classes depending on
        # version/path:
        #   1) litellm.ContextWindowExceededError
        #   2) litellm.BadRequestError with message "Requested token count exceeds ..."
        #      or "longer than the model's context length"
        # Re-raise both so the calling rollout terminates this branch cleanly.
        _exc_msg = str(exc)
        _is_overflow = (
            isinstance(exc, getattr(litellm, "ContextWindowExceededError", ()))
            or "ContextWindowExceededError" in type(exc).__name__
            or "Requested token count exceeds" in _exc_msg
            or "longer than the model's context length" in _exc_msg
        )
        if _is_overflow:
            print(f"Error in run_litellm_completion_async (FATAL, raising): {exc}")
            raise
        print(f"Error in run_litellm_completion_async: {exc}")
        return ChatCompletion(content="", model_name=model_name, metadata={"timestamp": time.time()})
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
        content = f"<think>{reasoning_content}</think>\n{content_no_thinking}" if content_no_thinking else f"<think>{reasoning_content}</think>"
    usage = {}
    if getattr(response, "usage", None) is not None:
        usage = response.usage.model_dump() if hasattr(response.usage, "model_dump") else dict(response.usage)
    return ChatCompletion(
        content=content,
        finish_reason=choice.finish_reason or "stop",
        model_name=model_name,
        cost=0.0,
        usage=usage,
        raw_response=response.model_dump() if hasattr(response, "model_dump") else {},
        metadata={"timestamp": time.time(), "content_no_thinking": content_no_thinking},
    )


async def run_generate_with_route_async(
    *,
    route_name: str,
    input_ids: List[int],
    api_base: str,
    api_key: str = "EMPTY",
    sampling_params: Optional[Dict[str, Any]] = None,
    return_logprobs: bool = True,
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
      - raw_response         = the full sglang JSON for debugging
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
        return ChatCompletion(content="", model_name=route_name, metadata={"timestamp": time.time(), "error": msg})

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
    output_ids = list(data.get("output_ids") or [])
    meta = data.get("meta_info") or {}
    finish_info = meta.get("finish_reason") or {}
    if isinstance(finish_info, dict):
        finish_reason = finish_info.get("type", "stop") or "stop"
    else:
        finish_reason = str(finish_info) or "stop"
    usage = {
        "prompt_tokens": int(meta.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(meta.get("completion_tokens", 0) or 0),
    }
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
        content=text,
        finish_reason=finish_reason,
        model_name=route_name,
        cost=0.0,
        usage=usage,
        raw_response=data,
        metadata={
            "timestamp": time.time(),
            # No <think>/</think> stripping here — sglang already returns the
            # decoded text including thinking content; downstream agents do
            # their own parsing on `content`.
            "content_no_thinking": text,
            "endpoint": "generate",
        },
        output_token_ids=output_ids,
        output_logprobs=output_logprobs,
        input_token_ids=list(input_ids),
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
