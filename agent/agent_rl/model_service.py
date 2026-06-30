from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class ChatSamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    stop: Optional[List[str]] = None
    json_mode: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatCompletion:
    content: str
    finish_reason: str = "stop"
    model_name: Optional[str] = None
    cost: float = 0.0
    usage: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Token-level fields populated by sglang /generate path. When set,
    # downstream consumers (PDS, training-side sample builder) prefer
    # output_token_ids over re-tokenizing `content`. logprobs are only
    # the chosen tokens' logprobs, in token order.
    output_token_ids: List[int] = field(default_factory=list)
    output_logprobs: List[float] = field(default_factory=list)
    input_token_ids: List[int] = field(default_factory=list)


@runtime_checkable
class ChatModelService(Protocol):
    def generate_text(
        self,
        *,
        messages: List[Dict[str, Any]],
        model_name: Optional[str],
        sampling: ChatSamplingParams,
        policy_version: Optional[str] = None,
    ) -> ChatCompletion:
        ...

    async def agenerate_text(
        self,
        *,
        messages: List[Dict[str, Any]],
        model_name: Optional[str],
        sampling: ChatSamplingParams,
        policy_version: Optional[str] = None,
    ) -> ChatCompletion:
        ...


_MODEL_SERVICE_REGISTRY: Dict[str, ChatModelService] = {}


def register_model_service(name: str, service: ChatModelService) -> None:
    _MODEL_SERVICE_REGISTRY[name] = service


def unregister_model_service(name: str) -> None:
    _MODEL_SERVICE_REGISTRY.pop(name, None)


def clear_model_services() -> None:
    _MODEL_SERVICE_REGISTRY.clear()


def get_model_service(name: str) -> ChatModelService:
    if name not in _MODEL_SERVICE_REGISTRY:
        raise KeyError(f"Unknown model service: {name}")
    return _MODEL_SERVICE_REGISTRY[name]


async def call_model_service_async(
    service_name: str,
    *,
    messages: List[Dict[str, Any]],
    model_name: Optional[str],
    sampling: ChatSamplingParams,
    policy_version: Optional[str] = None,
) -> ChatCompletion:
    return await get_model_service(service_name).agenerate_text(
        messages=messages,
        model_name=model_name,
        sampling=sampling,
        policy_version=policy_version,
    )


def call_model_service(
    service_name: str,
    *,
    messages: List[Dict[str, Any]],
    model_name: Optional[str],
    sampling: ChatSamplingParams,
    policy_version: Optional[str] = None,
) -> ChatCompletion:
    service = get_model_service(service_name)
    return service.generate_text(
        messages=messages,
        model_name=model_name,
        sampling=sampling,
        policy_version=policy_version,
    )


def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError("run_async cannot be used inside an already-running event loop.")
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)
