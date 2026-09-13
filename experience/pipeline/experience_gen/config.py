from __future__ import annotations

from dataclasses import dataclass


MODEL_COMPLETION_TOKENS = 20_480


@dataclass(frozen=True)
class ModelConfig:
    """The accepted model schedule from the final experiment."""

    generator: str = "nvidia/zai-org/glm-5.2"
    refiner: str = "openai/openai/gpt-5.5"
    judge: str = "nvidia/nvidia/nemotron-3-ultra"
    temperature: float = 0.1
    judge_temperature: float = 0.01
    top_p: float = 0.95
    max_tokens: int = MODEL_COMPLETION_TOKENS
    retry_max_tokens: int = MODEL_COMPLETION_TOKENS
