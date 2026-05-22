"""RouteTextbasedModel — token-in / token-out path against sglang.

We talk to sglang via /generate (raw input_ids -> output_ids + logprobs)
instead of /v1/chat/completions, so:
  - The SAME tokenizer (loaded by swe_agent.tokenization) renders the chat
    template locally; nothing is re-tokenized server-side. This eliminates
    encode/decode round-trip mismatches between rollout and training.
  - We get back per-token logprobs essentially for free; downstream
    consumers (PDS, training-side TIS) can use them as ground truth.
  - Token IDs are the source of truth: assistant_message["token_ids"]
    holds sglang's exact output_ids; assistant_message["logprobs"] holds
    the chosen-token logprobs in token order.
"""
import time
from typing import Any

from agent_rl.model_service import run_async
from agent_rl.run_utils import compact_completion_response, run_generate_with_route_async

from swe_agent.exceptions import FormatError
from swe_agent.models import GLOBAL_MODEL_STATS
from swe_agent.models.litellm_model import LitellmModel, logger
from swe_agent.models.litellm_textbased_model import LitellmTextbasedModel, LitellmTextbasedModelConfig
from swe_agent.models.utils.actions_text import parse_regex_actions
from swe_agent.models.utils.retry import retry
from swe_agent.tokenization import get_stop_token_ids, tokenize_messages_with_template


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
        api_base = query_kwargs.pop("api_base", None)
        api_key = query_kwargs.pop("api_key", "EMPTY")
        # extra_body may carry legacy chat_template_kwargs. The local
        # token-in/out renderer does not prefill thinking tags; thinking
        # models must generate `<think>...</think>` in output_ids.
        extra_body = query_kwargs.pop("extra_body", {}) or {}
        chat_template_kwargs = extra_body.get("chat_template_kwargs", {}) if isinstance(extra_body, dict) else {}
        enable_thinking = bool(chat_template_kwargs.get("enable_thinking", True))

        if not api_base:
            raise RuntimeError(
                f"RouteTextbasedModel.query: api_base required for token-in/token-out "
                f"path against sglang /generate (route={self.config.route_name})."
            )

        # Apply tokenization LOCALLY → input_ids that sglang will run as-is.
        # Strip extraneous fields, BUT KEEP `token_ids` on assistant messages
        # so the splicing path in tokenize_messages_with_template can use
        # sglang's exact prior-turn output (byte-perfect, no template round
        # trip). Without keeping token_ids, the splice would silently fall
        # back to the text-encoded form and the prefix invariant would break
        # (asserts catch this).
        normalized: list[dict[str, Any]] = []
        for m in prepared_messages:
            role = m.get("role", "assistant") or "assistant"
            item: dict[str, Any] = {"role": role, "content": m.get("content", "") or ""}
            if role == "assistant" and m.get("token_ids"):
                item["token_ids"] = m["token_ids"]
            normalized.append(item)
        input_ids = tokenize_messages_with_template(
            normalized,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            model_path=self.config.model_name,
        )
        # === PREFIX INVARIANT CHECK ===
        # The custom chat template in swe_agent.tokenization is designed so that
        # every successive call's input_ids EXTENDS the previous call's
        # input_ids + the previous call's output. Concretely: if prior_asst
        # in `prepared_messages` carries the stored sglang token-level fields
        # (prompt_token_ids + token_ids), then the new input_ids[:N] must equal
        # prior_asst.prompt_token_ids + prior_asst.token_ids (where N is the
        # sum of those lengths). If this ever fails the loss_mask placement
        # in pds_to_grpo_bundle would be misaligned and PG ratio would
        # explode (see commit a6d1789's commit message; this was the 51615
        # 7.6B loss bug). Assert hard at the source so we catch any future
        # template/tokenizer regression immediately, not 30 minutes later
        # when training crashes.
        prior_asst = None
        for m in reversed(prepared_messages[:-1] if prepared_messages else []):
            if m.get("role") == "assistant" and m.get("prompt_token_ids") and m.get("token_ids"):
                prior_asst = m
                break
        if prior_asst is not None:
            prior_prompt = list(prior_asst["prompt_token_ids"])
            prior_out = list(prior_asst["token_ids"])
            expected_prefix_len = len(prior_prompt) + len(prior_out)
            if expected_prefix_len > len(input_ids):
                raise AssertionError(
                    f"Prefix invariant violated: new input_ids len ({len(input_ids)}) "
                    f"is shorter than prior asst.prompt_token_ids+token_ids "
                    f"({expected_prefix_len}). Chat template may not be rendering "
                    f"consistently across turns."
                )
            actual_prefix = list(input_ids[:expected_prefix_len])
            expected_prefix = prior_prompt + prior_out
            if actual_prefix != expected_prefix:
                # Locate the first divergence to make the error actionable.
                divergence = next(
                    (i for i in range(expected_prefix_len)
                     if actual_prefix[i] != expected_prefix[i]),
                    -1,
                )
                raise AssertionError(
                    f"Prefix invariant violated: new input_ids does NOT start with "
                    f"prior asst.prompt+token_ids. first_divergence_idx={divergence}, "
                    f"len(prior_prompt)={len(prior_prompt)}, len(prior_out)={len(prior_out)}, "
                    f"len(new_input)={len(input_ids)}. Chat template / tokenization "
                    f"is no longer round-trip consistent — fix tokenization.py."
                )

        sampling_params: dict[str, Any] = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_new_tokens": int(max_tokens),
            # /generate does NOT auto-derive stop tokens from the chat template
            # like /v1/chat/completions does. Without this, the model emits its
            # real response, hits <|im_end|>, and then keeps generating until
            # max_new_tokens — we observed an infinite <think></think> tail.
            "stop_token_ids": get_stop_token_ids(model_path=self.config.model_name),
        }
        if stop:
            sampling_params["stop"] = stop

        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions, model_name=self.config.model_name):
            with attempt:
                try:
                    completion = run_async(
                        run_generate_with_route_async(
                            route_name=self.config.route_name,
                            input_ids=input_ids,
                            api_base=api_base,
                            api_key=api_key,
                            sampling_params=sampling_params,
                            return_logprobs=True,
                        )
                    )
                except RuntimeError as exc:
                    # sglang /generate returns 400 with a context-overflow
                    # message when input_ids+max_new_tokens exceeds the
                    # served model's context length. Retrying with the SAME
                    # input gives the SAME error — it's deterministic, not
                    # transient. Convert to a litellm abort exception so
                    # tenacity (configured with retry_if_not_exception_type)
                    # stops retrying immediately. Saves ~4 wasted retries
                    # per dead-on-arrival Lane B branch (51963 had 113
                    # such branches per rollout = ~450 wasted calls).
                    msg = str(exc)
                    if (
                        "maximum context length" in msg
                        or "is longer than the model's context length" in msg
                        or "Requested token count exceeds" in msg
                    ):
                        import litellm
                        raise litellm.exceptions.ContextWindowExceededError(
                            message=msg,
                            model=self.config.model_name,
                            llm_provider="sglang",
                        ) from exc
                    raise
        content = completion.content or ""
        assistant_message = {
            "role": "assistant",
            "content": completion.content,
            "content_no_thinking": completion.metadata.get("content_no_thinking", content),
            "usage": dict(completion.usage) if completion.usage else {},
            # ---- Token-level fields (the source of truth for training) ----
            # The trio (prompt_token_ids, token_ids, logprobs) records EXACTLY
            # what flowed across the sglang boundary for this turn:
            #   prompt_token_ids = the input_ids we sent (chat-template
            #                      encoded full conversation prefix)
            #   token_ids        = sglang's exact output_ids (no re-tok)
            #   logprobs         = per-output-token chosen logprob, one entry
            #                      per output token, in token order
            # PDS sample build then uses LAST assistant's prompt_token_ids +
            # token_ids as the full sequence for training, with no need to
            # re-apply the chat template (avoids any encode/decode mismatch).
            "prompt_token_ids": list(completion.input_token_ids),
            "token_ids": list(completion.output_token_ids),
            "logprobs": list(completion.output_logprobs),
            "extra": {
                "actions": [],
                "response": compact_completion_response(completion),
                "cost": completion.cost,
                "timestamp": completion.metadata.get("timestamp", time.time()),
                "finish_reason": completion.finish_reason,
                "route_name": self.config.route_name,
                "policy_version": self.policy_version,
                "input_token_count": len(completion.input_token_ids),
            },
        }
        GLOBAL_MODEL_STATS.add(completion.cost)
        try:
            assistant_message["extra"]["actions"] = parse_regex_actions(
                content,
                action_regex=self.config.action_regex,
                format_error_template=self.config.format_error_template,
            )
        except FormatError as exc:
            assistant_message["extra"]["format_error"] = True
            setattr(exc, "assistant_message", assistant_message)
            raise
        return assistant_message

    def get_state(self) -> dict[str, Any]:
        return {"policy_version": self.policy_version}

    def set_state(self, state: dict[str, Any]) -> None:
        self.policy_version = state.get("policy_version")
