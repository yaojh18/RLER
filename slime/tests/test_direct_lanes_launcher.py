from __future__ import annotations

import os
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from train_agent.run.grpo_async_lanes import _export_lanes_env, _parse_lanes_args


def _load_direct_filter_without_torch():
    """Load the tiny filter in host-only test environments without PyTorch."""

    class _Tensor:
        def __init__(self, values):
            self.values = [float(value) for value in values]

        def std(self):
            if len(self.values) < 2:
                return 0.0
            mean = sum(self.values) / len(self.values)
            return (
                sum((value - mean) ** 2 for value in self.values)
                / (len(self.values) - 1)
            ) ** 0.5

    fake_torch = types.ModuleType("torch")
    fake_torch.float64 = object()
    fake_torch.tensor = lambda values, dtype=None: _Tensor(values)
    fake_base = types.ModuleType("slime.rollout.filter_hub.base_types")

    class _Output:
        def __init__(self, keep, reason=None):
            self.keep = bool(keep)
            self.reason = reason

    fake_base.DynamicFilterOutput = _Output
    fake_types = types.ModuleType("slime.utils.types")
    fake_types.Sample = object
    saved = {
        name: sys.modules.get(name)
        for name in (
            "torch",
            "slime.rollout.filter_hub.base_types",
            "slime.utils.types",
        )
    }
    sys.modules["torch"] = fake_torch
    sys.modules["slime.rollout.filter_hub.base_types"] = fake_base
    sys.modules["slime.utils.types"] = fake_types
    try:
        path = (
            Path(__file__).resolve().parents[1]
            / "slime/rollout/filter_hub/dynamic_sampling_filters.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_direct_sampling_filters_test", path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.check_direct_judge_variance
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


check_direct_judge_variance = _load_direct_filter_without_torch()


@pytest.mark.parametrize(
    ("mode", "topology", "steps", "terminal"),
    [
        ("rollout40", "depth1", "40", "1"),
        ("depth2", "depth2", "20", "0"),
        ("custom", "depth2", "17", "0"),
    ],
)
def test_direct_rollout_presets_preserve_topology_contract(
    monkeypatch, mode, topology, steps, terminal
):
    for key in list(os.environ):
        if key.startswith("SWE_AGENT_LANES_"):
            monkeypatch.delenv(key, raising=False)
    args = [
        "--lanes-rollout-mode",
        mode,
        "--lanes-topology",
        "depth2",
        "--lanes-steps-per-round",
        "17",
    ]
    parsed, forwarded = _parse_lanes_args(args)
    assert forwarded == []

    _export_lanes_env(parsed)

    assert os.environ["SWE_AGENT_LANES_TOPOLOGY"] == topology
    assert os.environ["SWE_AGENT_LANES_STEPS_PER_ROUND"] == steps
    assert os.environ["SWE_AGENT_LANES_TERMINAL_ROLLOUT"] == terminal
    assert "SWE_AGENT_LANES_RETRIEVAL_MODEL" not in os.environ
    assert "SWE_AGENT_LANES_RUBRIC_MODEL" not in os.environ
    assert "SWE_AGENT_LANES_EXPERIENCE_BANK" not in os.environ


def test_direct_optional_detectors_can_be_disabled(monkeypatch):
    monkeypatch.setenv("SWE_AGENT_LANES_ENABLE_VARIANCE_DETECTOR", "1")
    monkeypatch.setenv("SWE_AGENT_LANES_COLLAPSE_REWARD_MARGIN", "0.5")
    parsed, _ = _parse_lanes_args(
        ["--lanes-disable-variance-detector", "--lanes-collapse-reward-margin", "0"]
    )

    _export_lanes_env(parsed)

    assert os.environ["SWE_AGENT_LANES_ENABLE_VARIANCE_DETECTOR"] == "0"
    assert "SWE_AGENT_LANES_COLLAPSE_REWARD_MARGIN" not in os.environ


def test_training_and_hosted_contexts_remain_separate(monkeypatch):
    monkeypatch.setenv("MODEL_CONTEXT_LENGTH", "65536")
    monkeypatch.delenv("RLER_HOSTED_MODEL_CONTEXT_LENGTH", raising=False)
    parsed, _ = _parse_lanes_args([])

    _export_lanes_env(parsed)

    assert os.environ["SWE_AGENT_MODEL_CONTEXT_LENGTH"] == "65536"
    assert os.environ["RLER_HOSTED_MODEL_CONTEXT_LENGTH"] == "256000"


class _FilterSample:
    def __init__(self, reward: float, predicted_zero: bool):
        self.reward = reward
        self.metadata = {"predicted_zero_variance": predicted_zero}

    def get_reward_value(self, _args):
        return self.reward


def test_direct_filter_combines_detector_with_reward_variance():
    zero = [_FilterSample(float(index), True) for index in range(8)]
    keep = [_FilterSample(float(index), False) for index in range(8)]
    tied = [_FilterSample(0.5, False) for _ in range(8)]

    assert not bool(check_direct_judge_variance(None, zero).keep)
    assert bool(check_direct_judge_variance(None, keep).keep)
    assert not bool(check_direct_judge_variance(None, tied).keep)
