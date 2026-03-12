from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, List, Optional

import litellm
import ray

from agent_rl import ChatCompletion, ChatSamplingParams


class LiteLLMChatService:
    def __init__(self, *, default_model_name: Optional[str] = None):
        self.default_model_name = default_model_name

    def _normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{k: v for k, v in message.items() if k not in {"metadata", "source", "trainable", "extra"}} for message in messages]

    def _completion_kwargs(self, sampling: ChatSamplingParams) -> Dict[str, Any]:
        kwargs = dict(sampling.extra)
        kwargs.setdefault("temperature", sampling.temperature)
        kwargs.setdefault("top_p", sampling.top_p)
        kwargs.setdefault("max_tokens", sampling.max_tokens)
        kwargs.setdefault("num_retries", 5)
        kwargs.setdefault("fallbacks", [])
        if sampling.stop:
            kwargs.setdefault("stop", sampling.stop)
        if sampling.json_mode:
            kwargs.setdefault("response_format", {"type": "json_object"})
        return kwargs

    def generate_text(
        self,
        *,
        messages: List[Dict[str, Any]],
        model_name: Optional[str],
        sampling: ChatSamplingParams,
        policy_version: Optional[str] = None,
    ) -> ChatCompletion:
        del policy_version
        response = litellm.completion(
            model=model_name or self.default_model_name,
            messages=self._normalize_messages(messages),
            **self._completion_kwargs(sampling),
        )
        choice = response.choices[0]
        return ChatCompletion(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason or "stop",
            model_name=model_name or self.default_model_name,
            raw_response=response.model_dump(),
        )

    async def agenerate_text(
        self,
        *,
        messages: List[Dict[str, Any]],
        model_name: Optional[str],
        sampling: ChatSamplingParams,
        policy_version: Optional[str] = None,
    ) -> ChatCompletion:
        del policy_version
        response = await litellm.acompletion(
            model=model_name or self.default_model_name,
            messages=self._normalize_messages(messages),
            **self._completion_kwargs(sampling),
        )
        choice = response.choices[0]
        return ChatCompletion(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason or "stop",
            model_name=model_name or self.default_model_name,
            raw_response=response.model_dump(),
        )


class VLLMChatService:
    def __init__(
        self,
        *,
        vllm_engines,
        tokenizer,
        default_model_name: Optional[str] = None,
        include_stop_str_in_output: bool = True,
        skip_special_tokens: bool = False,
    ):
        self.vllm_engines = list(vllm_engines)
        self.tokenizer = tokenizer
        self.default_model_name = default_model_name
        self.include_stop_str_in_output = include_stop_str_in_output
        self.skip_special_tokens = skip_special_tokens
        self._lock = threading.Lock()
        self._engine_index = 0
        self.current_policy_version: Optional[str] = None

    def set_policy_version(self, policy_version: Optional[str]) -> None:
        self.current_policy_version = policy_version

    def _next_engine(self):
        if not self.vllm_engines:
            raise RuntimeError("VLLMChatService was created without any engines.")
        with self._lock:
            engine = self.vllm_engines[self._engine_index % len(self.vllm_engines)]
            self._engine_index += 1
        return engine

    def _build_sampling_params(self, sampling: ChatSamplingParams):
        from vllm import SamplingParams

        allowed_extra_keys = {
            "best_of",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "top_k",
            "min_p",
            "seed",
            "use_beam_search",
            "length_penalty",
            "early_stopping",
            "stop_token_ids",
            "ignore_eos",
            "min_tokens",
            "logprobs",
            "prompt_logprobs",
            "detokenize",
            "skip_special_tokens",
            "spaces_between_special_tokens",
            "truncate_prompt_tokens",
            "include_stop_str_in_output",
        }
        filtered_extra = {key: value for key, value in sampling.extra.items() if key in allowed_extra_keys}
        return SamplingParams(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_tokens,
            include_stop_str_in_output=self.include_stop_str_in_output,
            skip_special_tokens=self.skip_special_tokens,
            stop=sampling.stop,
            **filtered_extra,
        )

    def _prompt_token_ids(self, messages: List[Dict[str, Any]]) -> List[int]:
        normalized_messages = [
            {k: v for k, v in message.items() if k not in {"metadata", "source", "trainable", "extra"}}
            for message in messages
        ]
        return self.tokenizer.apply_chat_template(normalized_messages, add_generation_prompt=True)

    def generate_text(
        self,
        *,
        messages: List[Dict[str, Any]],
        model_name: Optional[str],
        sampling: ChatSamplingParams,
        policy_version: Optional[str] = None,
    ) -> ChatCompletion:
        if (
            policy_version is not None
            and self.current_policy_version is not None
            and policy_version != self.current_policy_version
        ):
            raise RuntimeError(
                f"Requested policy_version={policy_version}, but VLLMChatService currently serves "
                f"policy_version={self.current_policy_version}."
            )
        prompt_token_ids = self._prompt_token_ids(messages)
        engine = self._next_engine()
        outputs = ray.get(
            engine.generate.remote(
                sampling_params=self._build_sampling_params(sampling),
                prompt_token_ids=[prompt_token_ids],
                use_tqdm=False,
            )
        )
        output = outputs[0].outputs[0]
        content = getattr(output, "text", None)
        if content is None:
            content = self.tokenizer.decode(output.token_ids, skip_special_tokens=self.skip_special_tokens)
        return ChatCompletion(
            content=content,
            finish_reason=getattr(output, "finish_reason", "stop") or "stop",
            model_name=model_name or self.default_model_name,
            usage={
                "prompt_tokens": len(prompt_token_ids),
                "completion_tokens": len(output.token_ids),
            },
            raw_response={
                "prompt_token_ids": prompt_token_ids,
                "token_ids": list(output.token_ids),
            },
            metadata={
                "requested_policy_version": policy_version,
                "service_policy_version": self.current_policy_version,
            },
        )

    async def agenerate_text(
        self,
        *,
        messages: List[Dict[str, Any]],
        model_name: Optional[str],
        sampling: ChatSamplingParams,
        policy_version: Optional[str] = None,
    ) -> ChatCompletion:
        return await asyncio.to_thread(
            self.generate_text,
            messages=messages,
            model_name=model_name,
            sampling=sampling,
            policy_version=policy_version,
        )
