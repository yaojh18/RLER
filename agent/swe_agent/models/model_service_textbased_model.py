import time
from typing import Any

from agent_rl import ChatSamplingParams, call_model_service

from swe_agent.models import GLOBAL_MODEL_STATS
from swe_agent.models.litellm_model import LitellmModel
from swe_agent.models.litellm_textbased_model import LitellmTextbasedModel, LitellmTextbasedModelConfig
from swe_agent.models.utils.actions_text import parse_regex_actions
from swe_agent.models.utils.retry import retry
from swe_agent.models.litellm_model import logger


class ModelServiceTextbasedModelConfig(LitellmTextbasedModelConfig):
    service_name: str = "policy"
    service_model_name: str | None = None


class ModelServiceTextbasedModel(LitellmTextbasedModel):
    def __init__(self, **kwargs):
        LitellmModel.__init__(self, config_class=ModelServiceTextbasedModelConfig, **kwargs)
        self.policy_version: str | None = None

    def set_policy_version(self, policy_version: str | None) -> None:
        self.policy_version = policy_version

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        sampling = ChatSamplingParams(
            temperature=kwargs.get("temperature", self.config.model_kwargs.get("temperature", 0.0)),
            top_p=kwargs.get("top_p", self.config.model_kwargs.get("top_p", 1.0)),
            max_tokens=kwargs.get("max_tokens", self.config.model_kwargs.get("max_tokens", 1024)),
            stop=kwargs.get("stop"),
            extra={k: v for k, v in (self.config.model_kwargs | kwargs).items() if k not in {"temperature", "top_p", "max_tokens", "stop"}},
        )
        prepared_messages = self._prepare_messages_for_api(messages)
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                completion = call_model_service(
                    self.config.service_name,
                    messages=prepared_messages,
                    model_name=self.config.service_model_name or self.config.model_name,
                    sampling=sampling,
                    policy_version=self.policy_version,
                )

        content = completion.content or ""
        actions = parse_regex_actions(
            content,
            action_regex=self.config.action_regex,
            format_error_template=self.config.format_error_template,
        )
        GLOBAL_MODEL_STATS.add(completion.cost)
        return {
            "role": "assistant",
            "content": content,
            "extra": {
                "actions": actions,
                "response": completion.raw_response,
                "cost": completion.cost,
                "timestamp": time.time(),
                "finish_reason": completion.finish_reason,
                **completion.metadata,
            },
        }

    def get_state(self) -> dict[str, Any]:
        return {"policy_version": self.policy_version}

    def set_state(self, state: dict[str, Any]) -> None:
        self.policy_version = state.get("policy_version")
