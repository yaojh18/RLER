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
    json_mode = kwargs.pop("response_format", None) == {"type": "json_object"}
    return ChatSamplingParams(
        temperature=kwargs.pop("temperature", 0),
        top_p=kwargs.pop("top_p", 1.0),
        max_tokens=kwargs.pop("max_tokens", kwargs.pop("max_completion_tokens", 16384)),
        stop=kwargs.pop("stop", None),
        json_mode=json_mode,
        extra=kwargs,
    )


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
        print(f"Error in run_litellm_completion_async: {exc}")
        return ChatCompletion(content="", model_name=model_name, metadata={"timestamp": time.time()})
    choice = response.choices[0]
    usage = {}
    if getattr(response, "usage", None) is not None:
        usage = response.usage.model_dump() if hasattr(response.usage, "model_dump") else dict(response.usage)
    return ChatCompletion(
        content=choice.message.content or "",
        finish_reason=choice.finish_reason or "stop",
        model_name=model_name,
        cost=0.0,
        usage=usage,
        raw_response=response.model_dump() if hasattr(response, "model_dump") else {},
        metadata={"timestamp": time.time()},
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
