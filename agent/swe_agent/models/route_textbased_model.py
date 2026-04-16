import time
from typing import Any

from agent_rl.model_service import run_async
from agent_rl.run_utils import run_chat_with_route_completion_async

from swe_agent.models import GLOBAL_MODEL_STATS
from swe_agent.models.litellm_model import LitellmModel, logger
from swe_agent.models.litellm_textbased_model import LitellmTextbasedModel, LitellmTextbasedModelConfig
from swe_agent.models.utils.actions_text import parse_regex_actions
from swe_agent.models.utils.retry import retry


class RouteTextbasedModelConfig(LitellmTextbasedModelConfig):
    route_name: str = "policy"


class RouteTextbasedModel(LitellmTextbasedModel):
    def __init__(self, **kwargs):
        LitellmModel.__init__(self, config_class=RouteTextbasedModelConfig, **kwargs)
        self.policy_version: str | None = None

    def set_policy_version(self, policy_version: str | None) -> None:
        self.policy_version = policy_version

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        prepared_messages = self._prepare_messages_for_api(messages)
        query_kwargs = self.config.model_kwargs | kwargs
        temperature = query_kwargs.pop("temperature", 0.0)
        top_p = query_kwargs.pop("top_p", 1.0)
        max_tokens = query_kwargs.pop("max_tokens", query_kwargs.pop("max_completion_tokens", 1024))
        stop = query_kwargs.pop("stop", None)

        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                completion = run_async(
                    run_chat_with_route_completion_async(
                        self.config.route_name,
                        model_name=self.config.model_name,
                        messages=prepared_messages,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        stop=stop,
                        policy_version=self.policy_version,
                        **query_kwargs,
                    )
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
                "timestamp": completion.metadata.get("timestamp", time.time()),
                "finish_reason": completion.finish_reason,
                "route_name": self.config.route_name,
                "policy_version": self.policy_version,
            },
        }

    def get_state(self) -> dict[str, Any]:
        return {"policy_version": self.policy_version}

    def set_state(self, state: dict[str, Any]) -> None:
        self.policy_version = state.get("policy_version")
