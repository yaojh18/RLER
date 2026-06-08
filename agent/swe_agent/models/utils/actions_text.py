"""Parse actions & format observations without toolcalls.
This was the method used for mini-swe-agent v1.0 and the original SWE-agent.
As of mini-swe-agent v2.0, we strongly recommend to use toolcalls instead.
"""

import re
import time

from jinja2 import StrictUndefined, Template

from swe_agent.exceptions import FormatError
from swe_agent.models.utils.openai_multimodal import expand_multimodal_content


def _invalid_content_error(content) -> str:
    content_type = type(content).__name__
    if content is None:
        return (
            "Expected assistant content to be a text string containing exactly one action, "
            "but the provider returned content=None. Your previous response had no textual "
            "message body, so the command parser could not inspect it. Please respond again "
            "with normal text and exactly one fenced bash command."
        )
    preview = repr(content)
    if len(preview) > 500:
        preview = preview[:500] + "...<truncated>"
    return (
        "Expected assistant content to be a text string containing exactly one action, "
        f"but got content of type {content_type}: {preview}. Please respond again with "
        "normal text and exactly one fenced bash command."
    )


def parse_regex_actions(content: str, *, action_regex: str, format_error_template: str) -> list[dict]:
    """Parse actions from text content using regex. Raises FormatError if not exactly one action."""
    content_error = None
    model_response = content
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    elif not isinstance(content, str):
        content_error = _invalid_content_error(content)
        content = ""
    actions = [a.strip() for a in re.findall(action_regex, content, re.DOTALL)]
    if len(actions) != 1:
        error_msg = content_error or f"Expected exactly 1 action, found {len(actions)}."
        raise FormatError(
            {
                "role": "user",
                "content": Template(format_error_template, undefined=StrictUndefined).render(
                    actions=actions, error=error_msg
                ),
                "extra": {
                    "interrupt_type": "FormatError",
                    "n_actions": len(actions),
                    "model_response": model_response,
                    **({"content_type_error": True} if content_error else {}),
                },
            }
        )
    return [{"command": action} for action in actions]


def format_observation_messages(
    outputs: list[dict],
    *,
    observation_template: str,
    template_vars: dict | None = None,
    multimodal_regex: str = "",
) -> list[dict]:
    """Format execution outputs into user observation messages."""
    results = []
    for output in outputs:
        content = Template(observation_template, undefined=StrictUndefined).render(
            output=output, **(template_vars or {})
        )
        msg: dict = {
            "role": "user",
            "content": content,
            "extra": {
                "raw_output": output.get("output", ""),
                "returncode": output.get("returncode"),
                "timestamp": time.time(),
                "exception_info": output.get("exception_info"),
                **output.get("extra", {}),
            },
        }
        if multimodal_regex:
            msg = expand_multimodal_content(msg, pattern=multimodal_regex)
        results.append(msg)
    return results
