from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
import types
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _load_file(monkeypatch, module_name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _load_collector(monkeypatch, filename: str):
    class _Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        FAILED = "failed"

    class _Sample:
        Status = _Status

        def __init__(self, **kwargs):
            self.group_index = kwargs.pop("group_index", None)
            self.index = kwargs.pop("index", None)
            self.prompt = kwargs.pop("prompt", "")
            self.tokens = kwargs.pop("tokens", [])
            self.response = kwargs.pop("response", "")
            self.response_length = kwargs.pop("response_length", 0)
            self.reward = kwargs.pop("reward", None)
            self.status = kwargs.pop("status", _Status.PENDING)
            self.metadata = kwargs.pop("metadata", {})
            for key, value in kwargs.items():
                setattr(self, key, value)

        def get_reward_value(self, args):
            reward_key = getattr(args, "reward_key", None)
            return self.reward if not reward_key else self.reward[reward_key]

    class _EvalOutput:
        def __init__(self, data, metrics=None):
            self.data = data
            self.metrics = metrics

    class _TrainOutput:
        def __init__(self, samples, metrics=None):
            self.samples = samples
            self.metrics = metrics

    class _Remote:
        def __call__(self, *args, **kwargs):
            def decorate(value):
                return value

            return decorate

    class _TrainingInstanceBudgetExhausted(RuntimeError):
        def __init__(self, *, attempted_instances, budget):
            self.attempted_instances = attempted_instances
            self.budget = budget
            super().__init__(
                f"budget exhausted: attempted={attempted_instances}, "
                f"budget={budget}"
            )

    class _TrainingValidationBoundaryReached(RuntimeError):
        def __init__(self, *, attempted_instances, boundary):
            self.attempted_instances = attempted_instances
            self.boundary = boundary
            super().__init__(
                f"validation boundary: attempted={attempted_instances}, "
                f"boundary={boundary}"
            )

    class _PolicyVersionMismatch(RuntimeError):
        pass

    def _checkpoint_policy_stale_lag(
        policy_version, *, consumer_rollout_id
    ):
        if policy_version == "checkpoint-base":
            checkpoint_id = -1
        elif (
            isinstance(policy_version, str)
            and policy_version.startswith("checkpoint-")
            and policy_version.removeprefix("checkpoint-").isdigit()
        ):
            checkpoint_id = int(
                policy_version.removeprefix("checkpoint-")
            )
        else:
            return None
        return int(consumer_rollout_id) - 1 - checkpoint_id

    usage_state = {"path": None, "events": [], "tracker_calls": []}

    class _UsageMetricsTracker:
        def __init__(self, path):
            self.path = path

        def peek(self):
            usage_state["tracker_calls"].append(
                ("peek", list(usage_state["events"]))
            )
            return {"usage/event_step": len(usage_state["events"])}

        def snapshot(self):
            return self.peek()

        def commit_update(self):
            usage_state["tracker_calls"].append(
                ("commit_update", list(usage_state["events"]))
            )
            return {
                "usage/event_step": len(usage_state["events"]),
                "usage/qwen_input_tokens_delta": 0,
            }

        def wandb_metrics(self):
            return self.commit_update()

    def _configure_usage_ledger(path):
        usage_state["path"] = path
        return path

    def _configure_usage_resume_window(offset):
        usage_state["resume_offset"] = offset

    def _record_group_disposition(
        group_id, *, disposition, reason="", phase=None, **_kwargs
    ):
        if usage_state["path"] is None:
            return None
        event = {
            "group_id": group_id,
            "disposition": disposition,
            "reason": reason,
            "phase": phase,
        }
        usage_state["events"].append(event)
        return event

    @contextlib.contextmanager
    def _usage_context(**_kwargs):
        yield None

    usage_module = _module(
        "swe_agent.usage",
        UsageMetricsTracker=_UsageMetricsTracker,
        build_usage_group_prefix=lambda **kwargs: (
            f"{kwargs['phase']}/attempt-test/r{kwargs['rollout_id']:04d}"
            f"{('/' + kwargs['dataset_name']) if kwargs.get('dataset_name') else ''}"
            f"/{kwargs['instance_id']}/t{kwargs['task_index']:06d}"
        ),
        configure_usage_ledger=_configure_usage_ledger,
        configure_usage_resume_window=_configure_usage_resume_window,
        infer_model_family=lambda name: (
            "qwen" if "qwen" in (name or "").lower() else "other"
        ),
        record_group_disposition=_record_group_disposition,
        usage_ledger_offset=lambda path=None: 123,
        usage_context=_usage_context,
        _test_state=usage_state,
    )

    ray = _module(
        "ray",
        remote=_Remote(),
        nodes=lambda: [],
        get=lambda value, **kwargs: value,
        wait=lambda *args, **kwargs: ([], []),
        kill=lambda *args, **kwargs: None,
    )
    stubs = {
        "ray": ray,
        "ray.util": _module("ray.util"),
        "ray.util.scheduling_strategies": _module(
            "ray.util.scheduling_strategies",
            NodeAffinitySchedulingStrategy=lambda **kwargs: kwargs,
        ),
        "slime": _module("slime"),
        "slime.rollout": _module("slime.rollout"),
        "slime.rollout.base_types": _module(
            "slime.rollout.base_types",
            RolloutFnEvalOutput=_EvalOutput,
            RolloutFnTrainOutput=_TrainOutput,
        ),
        "slime.rollout.data_source": _module(
            "slime.rollout.data_source",
            TrainingInstanceBudgetExhausted=_TrainingInstanceBudgetExhausted,
            TrainingValidationBoundaryReached=(
                _TrainingValidationBoundaryReached
            ),
        ),
        "slime.rollout.filter_hub": _module("slime.rollout.filter_hub"),
        "slime.rollout.filter_hub.base_types": _module(
            "slime.rollout.filter_hub.base_types",
            call_dynamic_filter=lambda *args, **kwargs: None,
        ),
        "slime.utils": _module("slime.utils"),
        "slime.utils.misc": _module(
            "slime.utils.misc", load_function=lambda path: None
        ),
        "slime.utils.types": _module("slime.utils.types", Sample=_Sample),
        "swe_agent": _module("swe_agent"),
        "swe_agent.exceptions": _module(
            "swe_agent.exceptions",
            PolicyVersionMismatch=_PolicyVersionMismatch,
        ),
        "swe_agent.policy_version": _module(
            "swe_agent.policy_version",
            checkpoint_policy_stale_lag=_checkpoint_policy_stale_lag,
            committed_policy_version=lambda **_kwargs: None,
        ),
        "swe_agent.usage": usage_module,
        "train_agent": _module("train_agent"),
        "train_agent.collect_grpo_rollout": _module(
            "train_agent.collect_grpo_rollout",
            build_rollout_samples=lambda **kwargs: ([], 0),
        ),
        "train_agent.serving": _module("train_agent.serving"),
        "train_agent.serving.sglang_chat_service": _module(
            "train_agent.serving.sglang_chat_service",
            start_slime_policy_route_warmup=lambda **kwargs: None,
        ),
    }
    for name, stub in stubs.items():
        monkeypatch.setitem(sys.modules, name, stub)
    _load_file(
        monkeypatch,
        "train_agent.collector_checkpoint",
        "slime/train_agent/collector_checkpoint.py",
    )
    if filename == "collect_lanes_rollout_async.py":
        _load_file(
            monkeypatch,
            "train_agent.collect_naive_rollout_async",
            "slime/train_agent/collect_naive_rollout_async.py",
        )
    return _load_file(
        monkeypatch,
        f"_test_{filename.removesuffix('.py')}",
        f"slime/train_agent/{filename}",
    )


def test_naive_wrapper_defaults_to_fair_joint_reward(monkeypatch):
    module = _load_file(
        monkeypatch,
        "_test_grpo_async_naive",
        "slime/train_agent/run/grpo_async_naive.py",
    )
    args, forwarded = module._parse_naive_args([])
    assert forwarded == []
    assert args.naive_reward_kind == "joint"
    assert args.naive_all_pass_reward == 1.0
    assert args.validation_temperature == 0.2
    assert args.validation_top_p == 0.95
    assert args.validation_process_workers == 8
    assert args.naive_gt_eval_timeout == 600
    assert args.validation_gt_eval_timeout == 1800
    module._export_naive_env(args)
    assert module.os.environ["SWE_AGENT_NAIVE_REWARD_KIND"] == "joint"
    assert module.os.environ["SWE_AGENT_NAIVE_ALL_PASS_REWARD"] == "1.0"
    assert module.os.environ["SWE_AGENT_VALIDATION_TEMPERATURE"] == "0.2"
    assert module.os.environ["SWE_AGENT_VALIDATION_TOP_P"] == "0.95"
    assert module.os.environ["SWE_AGENT_VALIDATION_PROCESS_WORKERS"] == "8"
    assert module.os.environ["SWE_AGENT_NAIVE_GT_EVAL_TIMEOUT"] == "600"
    assert (
        module.os.environ["SWE_AGENT_VALIDATION_GT_EVAL_TIMEOUT"]
        == "1800"
    )


def test_lanes_wrapper_defaults_to_depth1_hosted_glm(monkeypatch):
    module = _load_file(
        monkeypatch,
        "_test_grpo_async_lanes",
        "slime/train_agent/run/grpo_async_lanes.py",
    )
    args, forwarded = module._parse_lanes_args([])
    assert forwarded == []
    assert args.lanes_topology == "depth1"
    assert args.lanes_beam_parents == 2
    assert args.lanes_terminal_rollout is False
    assert args.lanes_rubric_model == "nvidia/zai-org/glm-5.2"
    assert args.lanes_judge_model == "nvidia/zai-org/glm-5.2"
    assert args.lanes_rubric_max_tokens == 20480
    assert args.lanes_judge_max_tokens == 20480
    assert args.validation_temperature == 0.2
    assert args.validation_top_p == 0.95
    assert args.validation_process_workers == 8
    assert args.validation_gt_eval_timeout == 1800
    module._export_lanes_env(args)
    assert module.os.environ["SWE_AGENT_LANES_TERMINAL_ROLLOUT"] == "0"
    assert module.os.environ["SWE_AGENT_LANES_RUBRIC_MAX_TOKENS"] == "20480"
    assert module.os.environ["SWE_AGENT_LANES_JUDGE_MAX_TOKENS"] == "20480"
    assert module.os.environ["SWE_AGENT_VALIDATION_TEMPERATURE"] == "0.2"
    assert module.os.environ["SWE_AGENT_VALIDATION_TOP_P"] == "0.95"
    assert module.os.environ["SWE_AGENT_VALIDATION_PROCESS_WORKERS"] == "8"
    assert (
        module.os.environ["SWE_AGENT_VALIDATION_GT_EVAL_TIMEOUT"]
        == "1800"
    )
    assert module.os.environ["SWE_AGENT_LANES_RUBRIC_API_BASE"].startswith(
        "https://inference-api.nvidia.com/"
    )


def test_lanes_wrapper_exports_explicit_lane_c_token_limits(monkeypatch):
    module = _load_file(
        monkeypatch,
        "_test_grpo_async_lanes_token_override",
        "slime/train_agent/run/grpo_async_lanes.py",
    )
    args, forwarded = module._parse_lanes_args(
        [
            "--lanes-rubric-max-tokens",
            "32768",
            "--lanes-judge-max-tokens",
            "32768",
        ]
    )
    assert forwarded == []
    module._export_lanes_env(args)
    assert module.os.environ["SWE_AGENT_LANES_RUBRIC_MAX_TOKENS"] == "32768"
    assert module.os.environ["SWE_AGENT_LANES_JUDGE_MAX_TOKENS"] == "32768"


@pytest.mark.parametrize(
    ("filename", "parse_name", "export_name"),
    [
        ("grpo_async_naive.py", "_parse_naive_args", "_export_naive_env"),
        ("grpo_async_lanes.py", "_parse_lanes_args", "_export_lanes_env"),
    ],
)
def test_wrappers_export_explicit_validation_sampling(
    monkeypatch, filename, parse_name, export_name
):
    module = _load_file(
        monkeypatch,
        f"_test_{filename.removesuffix('.py')}_validation_sampling",
        f"slime/train_agent/run/{filename}",
    )
    args, forwarded = getattr(module, parse_name)(
        [
            "--validation-temperature",
            "0.1",
            "--validation-top-p",
            "0.9",
            "--validation-gt-eval-timeout",
            "2400",
        ]
    )
    assert forwarded == []
    getattr(module, export_name)(args)
    assert module.os.environ["SWE_AGENT_VALIDATION_TEMPERATURE"] == "0.1"
    assert module.os.environ["SWE_AGENT_VALIDATION_TOP_P"] == "0.9"
    assert (
        module.os.environ["SWE_AGENT_VALIDATION_GT_EVAL_TIMEOUT"]
        == "2400"
    )


def test_validation_manifest_preserves_fold_fields_and_physical_split(
    monkeypatch, tmp_path
):
    module = _load_collector(monkeypatch, "collect_naive_rollout_async.py")
    manifest = tmp_path / "val.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "input": [{"role": "user", "content": "ignored"}],
                        "metadata": {
                            "instance_id": "django__django-1",
                            "subset": "verified",
                            "split": "test",
                            "fold_id": 0,
                            "fold_partition": "validation",
                        },
                    }
                ),
                json.dumps(
                    {
                        "instance_id": "pytest-dev__pytest-2",
                        "subset": "verified",
                        "hf_split": "test",
                        "fold_id": 0,
                        "partition": "validation",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        eval_datasets=[
            SimpleNamespace(
                name="fold0_val",
                path=str(manifest),
                n_samples_per_eval_prompt=1,
                metadata_key="metadata",
                metadata_overrides={},
                min_eval_samples=2,
            )
        ]
    )
    rows = module._validation_rows_from_args(args)
    assert [row[1]["instance_id"] for row in rows] == [
        "django__django-1",
        "pytest-dev__pytest-2",
    ]
    assert all(row[1]["split"] == "test" for row in rows)
    assert all(row[1]["subset"] == "verified" for row in rows)
    assert all(row[1]["fold_partition"] == "validation" for row in rows)
    assert all(row[1]["fold_id"] == 0 for row in rows)


def test_validation_manifest_rejects_duplicate_instances(monkeypatch, tmp_path):
    module = _load_collector(monkeypatch, "collect_naive_rollout_async.py")
    manifest = tmp_path / "val.json"
    manifest.write_text(
        json.dumps(
            [
                {"instance_id": "django__django-1"},
                {"instance_id": "django__django-1"},
            ]
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        eval_datasets=[
            SimpleNamespace(
                name="val",
                path=str(manifest),
                n_samples_per_eval_prompt=1,
                metadata_key="metadata",
                metadata_overrides={},
                min_eval_samples=None,
            )
        ]
    )
    with pytest.raises(ValueError, match="duplicate validation instance_id"):
        module._validation_rows_from_args(args)


def test_validation_retry_only_regenerates_missing_instances(
    monkeypatch, tmp_path
):
    module = _load_collector(
        monkeypatch, "collect_naive_rollout_async.py"
    )
    monkeypatch.setenv(
        "RLER_USAGE_LEDGER_PATH", str(tmp_path / "usage.jsonl")
    )
    rows = [
        (
            "swebench_validation",
            {
                "instance_id": "django__django-1",
                "subset": "verified",
                "split": "test",
            },
        ),
        (
            "swebench_validation",
            {
                "instance_id": "pytest-dev__pytest-2",
                "subset": "verified",
                "split": "test",
            },
        ),
    ]
    monkeypatch.setattr(
        module, "_validation_rows_from_args", lambda _args: rows
    )
    monkeypatch.setattr(
        module,
        "_policy_endpoints_for_model",
        lambda *_args, **_kwargs: [("127.0.0.1", 30000)],
    )
    ensure_calls = []
    monkeypatch.setattr(
        module,
        "_ensure_validation_node_workers",
        lambda **kwargs: ensure_calls.append(kwargs),
    )

    submissions = []

    class SubmitPolicyBatch:
        def remote(self, tasks):
            submissions.extend(tasks)
            return SimpleNamespace(kind="policy", tasks=tasks)

    class SubmitGt:
        def remote(self, task):
            return SimpleNamespace(kind="gt", task=task)

    module._NODE_WORKERS.append(
        SimpleNamespace(
            submit_validation_policy_batch=SubmitPolicyBatch(),
            submit_validation_gt=SubmitGt(),
        )
    )
    result_attempts = {}

    def ray_get(ref, **_kwargs):
        if ref.kind == "policy":
            results = []
            for task in ref.tasks:
                instance_id = task["instance_id"]
                result_attempts[instance_id] = (
                    result_attempts.get(instance_id, 0) + 1
                )
                if (
                    instance_id == "pytest-dev__pytest-2"
                    and result_attempts[instance_id] == 1
                ):
                    results.append(
                        {
                            "bundle": None,
                            "validation_policy": None,
                            "error": "validation actor died",
                            "usage_group_id": task["usage_group_id"],
                        }
                    )
                else:
                    results.append(
                        {
                            "bundle": None,
                            "validation_policy": {
                                "rollout_dir": str(tmp_path / instance_id),
                                "terminal_patch": "diff",
                                "rollout_error": "",
                            },
                            "error": "",
                            "usage_group_id": task["usage_group_id"],
                        }
                    )
            return results
        task = ref.task
        return {
            "bundle": None,
            "validation": {
                "status": "unresolved",
                "infrastructure_error": False,
                "rollout_error": "",
            },
            "usage_group_id": task["usage_group_id"],
        }

    monkeypatch.setattr(module.ray, "get", ray_get)
    args = SimpleNamespace(eval_instance_attempt=100)

    first = module.generate_validation_rollout(
        args,
        rollout_id=7,
        output_root=tmp_path,
        model_name="Qwen",
    )
    assert first.metrics["eval/attempted"] == 2
    assert first.metrics["eval/completed"] == 1
    assert first.metrics["eval/incomplete"] == 1
    assert module._EVAL_PARTIAL_STATE is not None

    second = module.generate_validation_rollout(
        args,
        rollout_id=7,
        output_root=tmp_path,
        model_name="Qwen",
    )
    assert second.metrics["eval/attempted"] == 2
    assert second.metrics["eval/completed"] == 2
    assert second.metrics["eval/incomplete"] == 0
    assert module._EVAL_PARTIAL_STATE is None
    assert result_attempts == {
        "django__django-1": 1,
        "pytest-dev__pytest-2": 2,
    }
    assert [call["verify_existing"] for call in ensure_calls] == [
        False,
        True,
    ]
    assert len(submissions) == 3
    dispositions = [
        event["disposition"]
        for event in sys.modules["swe_agent.usage"]._test_state["events"]
    ]
    assert dispositions.count("accepted") == 2
    assert dispositions.count("invalid") == 1


def test_validation_restart_restores_complete_artifacts_only(
    monkeypatch, tmp_path
):
    module = _load_collector(
        monkeypatch, "collect_naive_rollout_async.py"
    )
    monkeypatch.setenv(
        "RLER_USAGE_LEDGER_PATH", str(tmp_path / "usage.jsonl")
    )
    rows = [
        (
            "swebench_validation",
            {
                "instance_id": "django__django-1",
                "subset": "verified",
                "split": "test",
            },
        ),
        (
            "swebench_validation",
            {
                "instance_id": "pytest-dev__pytest-2",
                "subset": "verified",
                "split": "test",
            },
        ),
    ]
    monkeypatch.setattr(
        module, "_validation_rows_from_args", lambda _args: rows
    )
    artifact_root = (
        tmp_path
        / "validation"
        / "rollout_0007"
        / "swebench_validation"
        / "django__django-1"
        / "20260101-000000"
        / "task-000000"
        / "django__django-1"
        / "20260101-000000-r00"
    )
    artifact_root.mkdir(parents=True)
    (artifact_root / "evaluation.json").write_text(
        json.dumps(
            {
                "status": "unresolved",
                "reward": 0,
                "metainfo": {"infrastructure_error": False},
            }
        ),
        encoding="utf-8",
    )
    (artifact_root / "messages.json").write_text(
        json.dumps({"messages": []}),
        encoding="utf-8",
    )
    (artifact_root / "model_patch.json").write_text(
        json.dumps({"django__django-1": {"model_patch": ""}}),
        encoding="utf-8",
    )
    (artifact_root.parent.parent / "validation_policy.json").write_text(
        json.dumps({"policy_version": "legacy-rollout-7"}),
        encoding="utf-8",
    )

    ensure_calls = []
    monkeypatch.setattr(
        module,
        "_ensure_validation_node_workers",
        lambda **kwargs: ensure_calls.append(kwargs),
    )
    monkeypatch.setattr(
        module,
        "_policy_endpoints_for_model",
        lambda *_args, **_kwargs: [("127.0.0.1", 30000)],
    )
    submissions = []

    class SubmitPolicyBatch:
        def remote(self, tasks):
            submissions.extend(tasks)
            return SimpleNamespace(kind="policy", tasks=tasks)

    class SubmitGt:
        def remote(self, task):
            return SimpleNamespace(kind="gt", task=task)

    module._NODE_WORKERS.append(
        SimpleNamespace(
            submit_validation_policy_batch=SubmitPolicyBatch(),
            submit_validation_gt=SubmitGt(),
        )
    )

    def ray_get(ref, **_kwargs):
        if ref.kind == "policy":
            return [
                {
                    "bundle": None,
                    "validation_policy": {
                        "rollout_dir": str(
                            tmp_path / task["instance_id"]
                        ),
                        "terminal_patch": "diff",
                        "rollout_error": "",
                    },
                    "error": "",
                    "usage_group_id": task["usage_group_id"],
                }
                for task in ref.tasks
            ]
        return {
            "bundle": None,
            "validation": {
                "status": "resolved",
                "infrastructure_error": False,
                "rollout_error": "",
            },
            "usage_group_id": ref.task["usage_group_id"],
        }

    monkeypatch.setattr(module.ray, "get", ray_get)
    result = module.generate_validation_rollout(
        SimpleNamespace(eval_instance_attempt=156),
        rollout_id=7,
        output_root=tmp_path,
        model_name="Qwen",
    )

    assert result.metrics["eval/attempted"] == 2
    assert result.metrics["eval/completed"] == 2
    assert result.metrics["eval/resolved"] == 1
    assert result.metrics["eval/incomplete"] == 0
    assert [task["instance_id"] for task in submissions] == [
        "pytest-dev__pytest-2"
    ]
    assert ensure_calls == [
        {"instance_workers": 8, "verify_existing": False}
    ]
    restored = result.data["swebench_validation"]["samples"][0]
    assert restored.metadata["validation_persisted"] is True
    assert module._EVAL_PARTIAL_STATE is None


def test_validation_restart_skips_worker_pool_when_all_artifacts_exist(
    monkeypatch, tmp_path
):
    module = _load_collector(
        monkeypatch, "collect_naive_rollout_async.py"
    )
    rows = [
        (
            "swebench_validation",
            {
                "instance_id": "django__django-1",
                "subset": "verified",
                "split": "test",
            },
        )
    ]
    monkeypatch.setattr(
        module, "_validation_rows_from_args", lambda _args: rows
    )
    artifact_root = (
        tmp_path
        / "validation"
        / "rollout_0005"
        / "swebench_validation"
        / "django__django-1"
        / "20260101-000000"
        / "task-000000"
        / "django__django-1"
        / "20260101-000000-r00"
    )
    artifact_root.mkdir(parents=True)
    for name, payload in (
        (
            "evaluation.json",
            {
                "status": "resolved",
                "metainfo": {"infrastructure_error": False},
            },
        ),
        ("messages.json", {"messages": []}),
        (
            "model_patch.json",
            {"django__django-1": {"model_patch": "diff"}},
        ),
    ):
        (artifact_root / name).write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
    (artifact_root.parent.parent / "validation_policy.json").write_text(
        json.dumps({"policy_version": "legacy-rollout-5"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "_ensure_validation_node_workers",
        lambda **_kwargs: pytest.fail("worker pool should not start"),
    )
    monkeypatch.setattr(
        module,
        "_policy_endpoints_for_model",
        lambda *_args, **_kwargs: pytest.fail("endpoints should not resolve"),
    )

    result = module.generate_validation_rollout(
        SimpleNamespace(eval_instance_attempt=100),
        rollout_id=5,
        output_root=tmp_path,
        model_name="Qwen",
    )

    assert result.metrics["eval/completed"] == 1
    assert result.metrics["eval/resolved"] == 1
    assert module._EVAL_PARTIAL_STATE is None


def test_stale_validation_before_dispatch_does_not_start_worker_pool(
    monkeypatch,
    tmp_path,
):
    module = _load_collector(
        monkeypatch,
        "collect_naive_rollout_async.py",
    )
    monkeypatch.setattr(
        module,
        "_validation_rows_from_args",
        lambda _args: [
            (
                "swebench_validation",
                {
                    "instance_id": "django__django-1",
                    "subset": "verified",
                    "split": "test",
                },
            )
        ],
    )
    monkeypatch.setattr(
        module,
        "_ensure_validation_node_workers",
        lambda **_kwargs: pytest.fail(
            "stale validation must not start a worker pool"
        ),
    )
    monkeypatch.setattr(
        module,
        "_policy_endpoints_for_model",
        lambda *_args, **_kwargs: pytest.fail(
            "stale validation must not reserve a policy endpoint"
        ),
    )

    result = module.generate_validation_rollout(
        SimpleNamespace(
            eval_instance_attempt=128,
            eval_policy_version="checkpoint-0000007",
            eval_policy_stale_before_dispatch=True,
        ),
        rollout_id=8,
        output_root=tmp_path,
        model_name="Qwen",
    )

    assert result.data == {}
    assert result.metrics["eval/attempted"] == 1
    assert result.metrics["eval/completed"] == 0
    assert result.metrics["eval/stale_policy_aborted"] == 1
    assert result.metrics["eval/incomplete"] == 1


def test_lane_c_route_is_hosted_glm_and_not_local_policy(monkeypatch):
    module = _load_collector(monkeypatch, "collect_lanes_rollout_async.py")
    monkeypatch.delenv("SWE_AGENT_LANES_RUBRIC_API_KEY", raising=False)
    monkeypatch.setenv("LITELLM_API_KEY", "secret-for-test")
    monkeypatch.setenv(
        "SWE_AGENT_LANES_RUBRIC_API_BASE",
        "https://inference-api.nvidia.com/v1/chat/completions",
    )
    rubric, judge, api_base, api_key = module._hosted_lane_c_settings()
    assert rubric == judge == "nvidia/zai-org/glm-5.2"
    assert api_base == "https://inference-api.nvidia.com/v1"
    assert api_key == "secret-for-test"
    monkeypatch.setenv(
        "SWE_AGENT_LANES_RUBRIC_MODEL",
        "openai/azure/zai-org/glm-5.2",
    )
    monkeypatch.setenv(
        "SWE_AGENT_LANES_JUDGE_MODEL",
        "openai/azure/zai-org/glm-5.2",
    )
    rubric, judge, api_base, api_key = module._hosted_lane_c_settings()
    assert rubric == judge == "openai/azure/zai-org/glm-5.2"
    assert api_base == "https://inference-api.nvidia.com/v1"
    assert api_key == "secret-for-test"
    monkeypatch.setenv(
        "SWE_AGENT_LANES_JUDGE_MODEL",
        "nvidia/zai-org/glm-5.2",
    )
    with pytest.raises(
        ValueError,
        match="approved NVIDIA-gateway deployment",
    ):
        module._hosted_lane_c_settings()
    monkeypatch.setenv(
        "SWE_AGENT_LANES_JUDGE_MODEL",
        "openai/azure/zai-org/glm-5.2",
    )
    monkeypatch.setenv("SWE_AGENT_LANES_RUBRIC_API_BASE", "http://127.0.0.1:30000")
    with pytest.raises(ValueError, match="hosted NVIDIA HTTPS endpoint"):
        module._hosted_lane_c_settings()


def test_depth2_buffer_pop_preserves_fifo_without_reweighting(monkeypatch):
    module = _load_collector(monkeypatch, "collect_lanes_rollout_async.py")
    module._BUFFER[:] = [
        module._BufferedGroup(
            samples=[], group_kind="root", usage_group_id="attempt-a:g0"
        ),
        module._BufferedGroup(
            samples=[], group_kind="root", usage_group_id="attempt-b:g0"
        ),
        module._BufferedGroup(
            samples=[], group_kind="root", usage_group_id="attempt-c:g0"
        ),
        module._BufferedGroup(
            samples=[], group_kind="beam", usage_group_id="attempt-b:g1"
        ),
        module._BufferedGroup(
            samples=[], group_kind="beam", usage_group_id="attempt-a:g1"
        ),
    ]
    selected = module._pop_groups(4)
    assert [group.usage_group_id for group in selected] == [
        "attempt-a:g0",
        "attempt-b:g0",
        "attempt-c:g0",
        "attempt-b:g1",
    ]
    assert module._buffer_kind_counts() == {"beam": 1}


def test_lanes_usage_ids_are_unique_and_exact_for_depth2(monkeypatch):
    module = _load_collector(monkeypatch, "collect_lanes_rollout_async.py")
    task = {
        "usage_group_prefix": "train/r0003/django__django-1/t000012",
        "topology": "depth2",
    }
    assert module._expected_usage_group_ids(task) == [
        "train/r0003/django__django-1/t000012:g0",
        "train/r0003/django__django-1/t000012:g1",
    ]
    assert module._export_group_usage_id(
        task,
        SimpleNamespace(metadata={"group_index": 1, "group_kind": "beam"}),
    ).endswith(":g1")


@pytest.mark.parametrize(
    "filename",
    ["collect_naive_rollout_async.py", "collect_lanes_rollout_async.py"],
)
def test_stale_one_buffer_contract_keeps_current_and_previous_only(
    monkeypatch, tmp_path, filename
):
    module = _load_collector(monkeypatch, filename)
    monkeypatch.setenv("RLER_USAGE_LEDGER_PATH", str(tmp_path / "usage.jsonl"))
    if filename == "collect_naive_rollout_async.py":
        make_group = lambda rollout_id: module._BufferedGroup(
            samples=[],
            rollout_id=rollout_id,
            usage_group_id=f"train/r{rollout_id:04d}/instance:g0",
        )
    else:
        make_group = lambda rollout_id: module._BufferedGroup(
            samples=[],
            rollout_id=rollout_id,
            group_kind="root",
            usage_group_id=f"train/r{rollout_id:04d}/instance:g0",
        )
    module._BUFFER[:] = [
        make_group(2),
        make_group(3),
        make_group(4),
    ]

    module._drop_stale_buffer(current_rollout_id=4)

    assert [group.rollout_id for group in module._BUFFER] == [3, 4]
    assert module._STALE_DROPPED_GROUPS == 1
    if filename == "collect_lanes_rollout_async.py":
        assert module._STALE_DROPPED_BY_KIND == {"root": 1}
    events = sys.modules["swe_agent.usage"]._test_state["events"]
    assert [
        (event["group_id"], event["disposition"], event["reason"])
        for event in events
    ] == [("train/r0002/instance:g0", "dropped", "stale")]


@pytest.mark.parametrize(
    "filename",
    ["collect_naive_rollout_async.py", "collect_lanes_rollout_async.py"],
)
def test_stale_buffer_contract_uses_true_policy_checkpoint(
    monkeypatch, tmp_path, filename
):
    module = _load_collector(monkeypatch, filename)
    monkeypatch.setenv(
        "RLER_USAGE_LEDGER_PATH",
        str(tmp_path / "usage.jsonl"),
    )

    def make_group(policy_version, suffix):
        kwargs = {
            "samples": [
                module.Sample(
                    metadata={"policy_version": policy_version}
                )
            ],
            "rollout_id": 1,
            "usage_group_id": f"train/r0001/instance:{suffix}",
        }
        if filename == "collect_lanes_rollout_async.py":
            kwargs["group_kind"] = "root"
        return module._BufferedGroup(**kwargs)

    module._BUFFER[:] = [
        make_group("checkpoint-base", "g0"),
        make_group("checkpoint-0000000", "g1"),
    ]

    module._drop_stale_buffer(current_rollout_id=2)

    assert [
        group.samples[0].metadata["policy_version"]
        for group in module._BUFFER
    ] == ["checkpoint-0000000"]
    assert module._STALE_DROPPED_GROUPS == 1


def test_naive_pending_stale_group_is_dropped_before_attempt_denominator(
    monkeypatch, tmp_path
):
    module = _load_collector(monkeypatch, "collect_naive_rollout_async.py")
    monkeypatch.setenv("RLER_USAGE_LEDGER_PATH", str(tmp_path / "usage.jsonl"))
    ref = object()
    module._PENDING[ref] = {
        "rollout_id": 2,
        "instance_id": "django__django-1",
        "usage_group_id": "train/r0002/instance:g0",
        "_endpoints_picked": [],
    }
    monkeypatch.setattr(
        module.ray,
        "wait",
        lambda *_args, **_kwargs: ([ref], []),
    )
    monkeypatch.setattr(
        module.ray,
        "get",
        lambda _ref: {"bundle": None, "error": "aborted by weight update"},
    )

    assert module._harvest_ready(
        args=SimpleNamespace(
            dynamic_sampling_filter_path=None,
            max_tokens_per_gpu=0,
            context_parallel_size=1,
        ),
        target="policy",
        current_rollout_id=4,
        block=True,
    ) == 0
    assert module._TOTAL_GROUPS_ATTEMPTED == 0
    assert module._INFRA_DROPPED_GROUPS == 0
    assert module._STALE_DROPPED_GROUPS == 1
    assert module._PENDING == {}
    events = sys.modules["swe_agent.usage"]._test_state["events"]
    assert [
        (event["group_id"], event["disposition"], event["reason"])
        for event in events
    ] == [("train/r0002/instance:g0", "dropped", "stale")]


def test_depth2_pending_stale_task_drops_root_and_beam_atomically(
    monkeypatch, tmp_path
):
    module = _load_collector(monkeypatch, "collect_lanes_rollout_async.py")
    monkeypatch.setenv("RLER_USAGE_LEDGER_PATH", str(tmp_path / "usage.jsonl"))
    ref = object()
    module._PENDING[ref] = {
        "rollout_id": 2,
        "instance_id": "django__django-1",
        "usage_group_prefix": "train/r0002/instance",
        "topology": "depth2",
        "m": 8,
        "_pinned_endpoints": [],
    }
    monkeypatch.setattr(
        module.ray,
        "wait",
        lambda *_args, **_kwargs: ([ref], []),
    )
    monkeypatch.setattr(
        module.ray,
        "get",
        lambda _ref: {"bundle": None, "error": "aborted by weight update"},
    )

    assert module._harvest_ready(
        args=SimpleNamespace(
            dynamic_sampling_filter_path=None,
            max_tokens_per_gpu=0,
            context_parallel_size=1,
        ),
        target="policy",
        current_rollout_id=4,
        block=True,
    ) == 0
    assert module._TOTAL_GROUPS_ATTEMPTED == 0
    assert module._INVALID_DROPPED_GROUPS == 0
    assert module._STALE_DROPPED_GROUPS == 2
    assert module._STALE_DROPPED_BY_KIND == {"root": 1, "beam": 1}
    assert module._PENDING == {}
    events = sys.modules["swe_agent.usage"]._test_state["events"]
    assert {
        (event["group_id"], event["disposition"], event["reason"])
        for event in events
    } == {
        ("train/r0002/instance:g0", "dropped", "stale"),
        ("train/r0002/instance:g1", "dropped", "stale"),
    }


@pytest.mark.parametrize(
    "filename",
    ["collect_naive_rollout_async.py", "collect_lanes_rollout_async.py"],
)
@pytest.mark.parametrize(
    "error_name,message",
    [
        ("OutOfMemoryError", "node running low on memory"),
        ("ActorDiedError", "actor died unexpectedly"),
        ("WorkerCrashedError", "worker died unexpectedly"),
        (
            "RayTaskError",
            "1 worker(s) were killed due to the node running low on memory",
        ),
    ],
)
def test_collectors_fail_fast_on_systemic_ray_failure(
    monkeypatch, filename, error_name, message
):
    module = _load_collector(monkeypatch, filename)
    fatal_type = type(error_name, (RuntimeError,), {})
    ref = object()
    task = {
        "rollout_id": 1,
        "instance_id": "django__django-1",
        "policy_version": "checkpoint-0000000",
        "m": 8,
    }
    if filename == "collect_naive_rollout_async.py":
        task.update(
            usage_group_id="train/r0001/instance:g0",
            _endpoints_picked=[],
        )
    else:
        task.update(
            usage_group_prefix="train/r0001/instance",
            topology="depth2",
            _pinned_endpoints=[],
        )
    module._PENDING[ref] = task
    monkeypatch.setattr(
        module.ray,
        "wait",
        lambda *_args, **_kwargs: ([ref], []),
    )

    def fail_get(_ref):
        raise fatal_type(message)

    monkeypatch.setattr(module.ray, "get", fail_get)
    with pytest.raises(
        module.CollectorInfrastructureError,
        match="aborting before drawing replacement source instances",
    ):
        module._harvest_ready(
            args=SimpleNamespace(
                dynamic_sampling_filter_path=None,
                max_tokens_per_gpu=0,
                context_parallel_size=1,
            ),
            target="policy",
            current_rollout_id=1,
            block=True,
        )

    # A fatal event is not a model/instance outcome and is not committed as an
    # attempted or invalid GRPO group. The process will resume from the last
    # durable model/data checkpoint.
    assert module._TOTAL_GROUPS_ATTEMPTED == 0
    if filename == "collect_naive_rollout_async.py":
        assert module._INFRA_DROPPED_GROUPS == 0
    else:
        assert module._INVALID_DROPPED_GROUPS == 0


@pytest.mark.parametrize(
    "filename",
    ["collect_naive_rollout_async.py", "collect_lanes_rollout_async.py"],
)
def test_collectors_keep_ordinary_ray_task_error_as_invalid(
    monkeypatch, filename
):
    module = _load_collector(monkeypatch, filename)
    assert not module._is_fatal_ray_infrastructure_error(
        RuntimeError("ordinary per-instance runner failure")
    )


def test_naive_stale_one_samples_retain_source_policy_version(
    monkeypatch
):
    module = _load_collector(monkeypatch, "collect_naive_rollout_async.py")
    ref = object()
    module._PENDING[ref] = {
        "rollout_id": 3,
        "instance_id": "django__django-1",
        "usage_group_id": "train/r0003/instance:g0",
        "policy_version": "checkpoint-0000003",
        "m": 8,
        "_endpoints_picked": [],
    }
    bundle = SimpleNamespace(
        policy_groups=[SimpleNamespace(metadata={})],
        rubric_groups=[],
        metadata={},
    )
    monkeypatch.setattr(
        module.ray,
        "wait",
        lambda *_args, **_kwargs: ([ref], []),
    )
    monkeypatch.setattr(
        module.ray,
        "get",
        lambda _ref: {
            "bundle": bundle,
            "error": "",
            "policy_version": "checkpoint-0000003",
        },
    )
    monkeypatch.setattr(
        module,
        "build_rollout_samples",
        lambda **_kwargs: (
            [
                module.Sample(metadata={}, reward=float(index % 2))
                for index in range(8)
            ],
            0,
        ),
    )

    assert module._harvest_ready(
        args=SimpleNamespace(
            dynamic_sampling_filter_path=None,
            max_tokens_per_gpu=0,
            context_parallel_size=1,
            reward_key=None,
        ),
        target="policy",
        current_rollout_id=4,
        block=True,
    ) == 1
    samples = module._BUFFER[0].samples
    assert module._BUFFER[0].rollout_id == 3
    assert {tuple(sample.weight_versions) for sample in samples} == {
        ("checkpoint-0000003",)
    }
    assert {
        (
            sample.metadata["policy_version"],
            sample.metadata["source_rollout_id"],
        )
        for sample in samples
    } == {("checkpoint-0000003", 3)}


@pytest.mark.parametrize(
    "filename",
    ["collect_naive_rollout_async.py", "collect_lanes_rollout_async.py"],
)
def test_train_usage_id_uses_checkpointed_source_cursor(
    monkeypatch, tmp_path, filename
):
    module = _load_collector(monkeypatch, filename)
    monkeypatch.setenv("RLER_USAGE_LEDGER_PATH", str(tmp_path / "usage.jsonl"))
    module._TASK_INDEX = 7
    captured_tasks = []

    class SubmitTask:
        def remote(self, task):
            captured_tasks.append(task)
            return object()

    module._NODE_WORKERS.append(
        SimpleNamespace(submit_task=SubmitTask())
    )

    sample_cls = module.Sample
    prompt_group = [
        sample_cls(
            group_index=432,
            metadata={
                "instance_id": "django__django-1",
                "subset": "princeton-nlp/SWE-Bench_Verified",
                "split": "test",
            },
        )
        for _ in range(8)
    ]

    class DataBuffer:
        def __init__(self):
            self.returned = False

        def get_samples(self, _count):
            if self.returned:
                return []
            self.returned = True
            return [prompt_group]

    if filename == "collect_naive_rollout_async.py":
        monkeypatch.setattr(
            module,
            "_policy_endpoints_for_model",
            lambda *_args, **_kwargs: [("127.0.0.1", 30000)],
        )
        submitted = module._submit_until_full(
            args=SimpleNamespace(),
            data_buffer=DataBuffer(),
            rollout_id=3,
            output_root=tmp_path,
            model_name="Qwen",
            max_pending=1,
            target_groups=1,
        )
    else:
        monkeypatch.setattr(
            module,
            "_policy_urls_for_model",
            lambda *_args, **_kwargs: ["http://127.0.0.1:30000"],
        )
        monkeypatch.setattr(
            module,
            "_hosted_lane_c_settings",
            lambda: (
                "nvidia/zai-org/glm-5.2",
                "nvidia/zai-org/glm-5.2",
                "https://inference-api.nvidia.com/v1",
                "secret",
            ),
        )
        submitted = module._submit_until_full(
            args=SimpleNamespace(),
            data_buffer=DataBuffer(),
            rollout_id=3,
            output_root=tmp_path,
            model_name="Qwen",
            max_pending=1,
            over_sampling_groups=1,
            est_groups_per_instance=1.0,
        )

    assert submitted == 1
    assert len(captured_tasks) == 1
    assert captured_tasks[0]["index"] == 7
    assert captured_tasks[0]["usage_group_prefix"].endswith(
        "/t000432"
    )


def test_lanes_discards_excess_groups_and_updates_usage_disposition(
    monkeypatch, tmp_path
):
    module = _load_collector(monkeypatch, "collect_lanes_rollout_async.py")
    monkeypatch.setenv("RLER_USAGE_LEDGER_PATH", str(tmp_path / "usage.jsonl"))
    module._BUFFER[:] = [
        module._BufferedGroup(
            samples=[],
            group_kind="root",
            usage_group_id="train/r0001/instance/t000001:g0",
        ),
        module._BufferedGroup(
            samples=[],
            group_kind="beam",
            usage_group_id="train/r0001/instance/t000001:g1",
        ),
    ]
    assert module._discard_excess_buffer() == 2
    assert module._BUFFER == []
    assert module._EXCESS_DROPPED_GROUPS == 2
    assert module._EXCESS_DROPPED_BY_KIND == {"root": 1, "beam": 1}
    events = sys.modules["swe_agent.usage"]._test_state["events"]
    assert [
        (event["group_id"], event["disposition"], event["reason"])
        for event in events
    ] == [
        (
            "train/r0001/instance/t000001:g0",
            "dropped",
            "excess_after_quota",
        ),
        (
            "train/r0001/instance/t000001:g1",
            "dropped",
            "excess_after_quota",
        ),
    ]


def test_lanes_harvest_records_accepted_and_missing_depth2_groups(
    monkeypatch, tmp_path
):
    module = _load_collector(monkeypatch, "collect_lanes_rollout_async.py")
    monkeypatch.setenv("RLER_USAGE_LEDGER_PATH", str(tmp_path / "usage.jsonl"))
    task = {
        "instance_id": "django__django-1",
        "usage_group_prefix": "train/r0001/django__django-1/t000001",
        "topology": "depth2",
        "policy_version": "checkpoint-0000001",
        "m": 8,
        "_pinned_endpoints": [],
    }
    root = SimpleNamespace(
        metadata={"group_index": 0, "group_kind": "root"}
    )
    bundle = SimpleNamespace(policy_groups=[root], rubric_groups=[])
    ref = object()
    module._PENDING[ref] = task
    monkeypatch.setattr(
        module.ray,
        "wait",
        lambda *_args, **_kwargs: ([ref], []),
    )
    monkeypatch.setattr(
        module.ray,
        "get",
        lambda _ref: {
            "bundle": bundle,
            "error": "",
            "policy_version": "checkpoint-0000001",
        },
    )
    monkeypatch.setattr(
        module,
        "build_rollout_samples",
        lambda **_kwargs: (
            [
                module.Sample(metadata={}, reward=float(index % 2))
                for index in range(8)
            ],
            0,
        ),
    )

    harvested = module._harvest_ready(
        args=SimpleNamespace(
            dynamic_sampling_filter_path=None,
            max_tokens_per_gpu=0,
            context_parallel_size=1,
        ),
        target="policy",
        block=True,
    )

    assert harvested == 1
    assert len(module._BUFFER) == 1
    assert module._TOTAL_GROUPS_ATTEMPTED == 2
    assert module._GROUPS_ATTEMPTED_BY_KIND == {"root": 1, "beam": 1}
    assert module._INVALID_DROPPED_GROUPS == 1
    assert module._INVALID_DROPPED_BY_KIND == {"beam": 1}
    assert module._REWARD_GROUPS_OBSERVED_BY_KIND == {"root": 1}
    assert module._ZERO_VARIANCE_GROUPS_BY_KIND == {}
    events = sys.modules["swe_agent.usage"]._test_state["events"]
    assert {
        (event["group_id"], event["disposition"], event["reason"])
        for event in events
    } == {
        (
            "train/r0001/django__django-1/t000001:g1",
            "invalid",
            "missing_group",
        ),
    }


def test_lanes_harvest_normalizes_zero_variance_disposition(
    monkeypatch, tmp_path
):
    module = _load_collector(monkeypatch, "collect_lanes_rollout_async.py")
    monkeypatch.setenv("RLER_USAGE_LEDGER_PATH", str(tmp_path / "usage.jsonl"))
    task = {
        "instance_id": "django__django-1",
        "usage_group_prefix": "train/r0002/django__django-1/t000002",
        "topology": "depth1",
        "policy_version": "checkpoint-0000002",
        "m": 8,
        "_pinned_endpoints": [],
    }
    root = SimpleNamespace(
        metadata={"group_index": 0, "group_kind": "root"}
    )
    bundle = SimpleNamespace(policy_groups=[root], rubric_groups=[])
    ref = object()
    module._PENDING[ref] = task
    monkeypatch.setattr(
        module.ray,
        "wait",
        lambda *_args, **_kwargs: ([ref], []),
    )
    monkeypatch.setattr(
        module.ray,
        "get",
        lambda _ref: {
            "bundle": bundle,
            "error": "",
            "policy_version": "checkpoint-0000002",
        },
    )
    monkeypatch.setattr(
        module,
        "build_rollout_samples",
        lambda **_kwargs: (
            [module.Sample(metadata={}, reward=1.0) for _ in range(8)],
            0,
        ),
    )
    monkeypatch.setattr(module, "load_function", lambda _path: object())
    monkeypatch.setattr(
        module,
        "call_dynamic_filter",
        lambda *_args, **_kwargs: SimpleNamespace(
            keep=False,
            reason="zero_std_1.0",
        ),
    )

    harvested = module._harvest_ready(
        args=SimpleNamespace(
            dynamic_sampling_filter_path="dynamic.filter",
            max_tokens_per_gpu=0,
            context_parallel_size=1,
        ),
        target="policy",
        block=True,
    )

    assert harvested == 0
    assert module._BUFFER == []
    assert module._TOTAL_GROUPS_ATTEMPTED == 1
    assert module._INVALID_DROPPED_GROUPS == 0
    assert module._FILTER_DROPPED_GROUPS == 1
    assert module._FILTER_DROPPED_BY_KIND == {"root": 1}
    # Reward statistics are deliberately observed before the zero-variance
    # filter, so a filtered group still appears in the efficiency dashboard.
    assert module._REWARD_GROUPS_OBSERVED_BY_KIND == {"root": 1}
    assert module._ZERO_VARIANCE_GROUPS_BY_KIND == {"root": 1}
    events = sys.modules["swe_agent.usage"]._test_state["events"]
    assert [
        (event["group_id"], event["disposition"], event["reason"])
        for event in events
    ] == [
        (
            "train/r0002/django__django-1/t000002:g0",
            "filtered",
            "zero_variance",
        )
    ]


@pytest.mark.parametrize("failure_kind", ["ray", "runner", "missing_bundle"])
def test_lanes_task_failures_count_each_expected_group_once(
    monkeypatch, tmp_path, failure_kind
):
    module = _load_collector(monkeypatch, "collect_lanes_rollout_async.py")
    monkeypatch.setenv("RLER_USAGE_LEDGER_PATH", str(tmp_path / "usage.jsonl"))
    task = {
        "instance_id": "django__django-1",
        "usage_group_prefix": "train/r0003/django__django-1/t000003",
        "topology": "depth2",
        "m": 8,
        "_pinned_endpoints": [],
    }
    ref = object()
    module._PENDING[ref] = task
    monkeypatch.setattr(
        module.ray,
        "wait",
        lambda *_args, **_kwargs: ([ref], []),
    )
    if failure_kind == "ray":
        def fail_get(_ref):
            raise RuntimeError("ray failed")

        monkeypatch.setattr(module.ray, "get", fail_get)
    elif failure_kind == "runner":
        monkeypatch.setattr(
            module.ray,
            "get",
            lambda _ref: {"bundle": None, "error": "runner failed"},
        )
    else:
        monkeypatch.setattr(
            module.ray,
            "get",
            lambda _ref: {"bundle": None, "error": ""},
        )

    assert module._harvest_ready(
        args=SimpleNamespace(
            dynamic_sampling_filter_path=None,
            max_tokens_per_gpu=0,
            context_parallel_size=1,
        ),
        target="policy",
        block=True,
    ) == 0

    assert module._TOTAL_GROUPS_ATTEMPTED == 2
    assert module._GROUPS_ATTEMPTED_BY_KIND == {"root": 1, "beam": 1}
    assert module._INVALID_DROPPED_GROUPS == 2
    assert module._INVALID_DROPPED_BY_KIND == {"root": 1, "beam": 1}
    events = sys.modules["swe_agent.usage"]._test_state["events"]
    assert len(events) == 2
    assert {event["group_id"].rsplit(":", 1)[-1] for event in events} == {
        "g0",
        "g1",
    }
    assert {event["disposition"] for event in events} == {"invalid"}


def test_group_outcome_metrics_use_attempted_denominator_and_conserve(
    monkeypatch
):
    module = _load_collector(monkeypatch, "collect_naive_rollout_async.py")
    metrics = module._group_outcome_metrics(
        attempted=8,
        accepted=3,
        invalid=2,
        dynamic_filtered=1,
        excess=2,
    )
    assert metrics["swe_agent/groups_attempted"] == 8
    assert metrics["swe_agent/groups_accepted_rate"] == pytest.approx(3 / 8)
    assert metrics["swe_agent/groups_invalid_rate"] == pytest.approx(2 / 8)
    assert metrics["swe_agent/groups_dynamic_filtered_rate"] == pytest.approx(1 / 8)
    assert metrics["swe_agent/groups_excess_rate"] == pytest.approx(2 / 8)
    assert sum(
        metrics[key]
        for key in (
            "swe_agent/groups_accepted_rate",
            "swe_agent/groups_invalid_rate",
            "swe_agent/groups_dynamic_filtered_rate",
            "swe_agent/groups_excess_rate",
        )
    ) == pytest.approx(1.0)
    with pytest.raises(AssertionError, match="conservation failed"):
        module._group_outcome_metrics(
            attempted=8,
            accepted=3,
            invalid=2,
            dynamic_filtered=1,
            excess=1,
        )


def test_naive_harvest_counts_prefilter_zero_variance_group(
    monkeypatch, tmp_path
):
    module = _load_collector(monkeypatch, "collect_naive_rollout_async.py")
    monkeypatch.setenv("RLER_USAGE_LEDGER_PATH", str(tmp_path / "usage.jsonl"))
    usage_group_id = "train/r0004/django__django-1/t000004:g0"
    task = {
        "instance_id": "django__django-1",
        "usage_group_id": usage_group_id,
        "policy_version": "checkpoint-0000004",
        "m": 8,
        "_endpoints_picked": [],
    }
    group = SimpleNamespace(metadata={})
    bundle = SimpleNamespace(
        policy_groups=[group],
        rubric_groups=[],
        metadata={},
    )
    ref = object()
    module._PENDING[ref] = task
    monkeypatch.setattr(
        module.ray,
        "wait",
        lambda *_args, **_kwargs: ([ref], []),
    )
    monkeypatch.setattr(
        module.ray,
        "get",
        lambda _ref: {
            "bundle": bundle,
            "error": "",
            "policy_version": "checkpoint-0000004",
        },
    )
    monkeypatch.setattr(
        module,
        "build_rollout_samples",
        lambda **_kwargs: (
            [module.Sample(metadata={}, reward=0.25) for _ in range(8)],
            0,
        ),
    )
    monkeypatch.setattr(module, "load_function", lambda _path: object())
    monkeypatch.setattr(
        module,
        "call_dynamic_filter",
        lambda *_args, **_kwargs: SimpleNamespace(
            keep=False,
            reason="zero_std_0.2",
        ),
    )

    assert module._harvest_ready(
        args=SimpleNamespace(
            dynamic_sampling_filter_path="dynamic.filter",
            max_tokens_per_gpu=0,
            context_parallel_size=1,
            reward_key=None,
        ),
        target="policy",
        block=True,
    ) == 0
    assert module._TOTAL_GROUPS_ATTEMPTED == 1
    assert module._INFRA_DROPPED_GROUPS == 0
    assert module._FILTER_DROPPED_GROUPS == 1
    assert module._REWARD_GROUPS_OBSERVED == 1
    assert module._ZERO_VARIANCE_GROUPS == 1
    events = sys.modules["swe_agent.usage"]._test_state["events"]
    assert [
        (event["group_id"], event["disposition"], event["reason"])
        for event in events
    ] == [(usage_group_id, "filtered", "zero_variance")]


def test_naive_runner_failure_is_one_attempt_and_one_invalid(
    monkeypatch, tmp_path
):
    module = _load_collector(monkeypatch, "collect_naive_rollout_async.py")
    monkeypatch.setenv("RLER_USAGE_LEDGER_PATH", str(tmp_path / "usage.jsonl"))
    usage_group_id = "train/r0005/django__django-1/t000005:g0"
    task = {
        "instance_id": "django__django-1",
        "usage_group_id": usage_group_id,
        "_endpoints_picked": [],
    }
    ref = object()
    module._PENDING[ref] = task
    monkeypatch.setattr(
        module.ray,
        "wait",
        lambda *_args, **_kwargs: ([ref], []),
    )
    monkeypatch.setattr(
        module.ray,
        "get",
        lambda _ref: {"bundle": None, "error": "runner failed"},
    )

    assert module._harvest_ready(
        args=SimpleNamespace(
            dynamic_sampling_filter_path=None,
            max_tokens_per_gpu=0,
            context_parallel_size=1,
        ),
        target="policy",
        block=True,
    ) == 0
    assert module._TOTAL_GROUPS_ATTEMPTED == 1
    assert module._INFRA_DROPPED_GROUPS == 1
    events = sys.modules["swe_agent.usage"]._test_state["events"]
    assert [
        (event["group_id"], event["disposition"], event["reason"])
        for event in events
    ] == [(usage_group_id, "invalid", "runner_error")]


@pytest.mark.parametrize(
    "filename",
    ["collect_naive_rollout_async.py", "collect_lanes_rollout_async.py"],
)
def test_collectors_stop_source_dispatch_on_validation_boundary(
    monkeypatch, tmp_path, filename
):
    module = _load_collector(monkeypatch, filename)
    monkeypatch.setenv("RLER_USAGE_LEDGER_PATH", str(tmp_path / "usage.jsonl"))

    class BoundaryDataBuffer:
        last_validation_attempt = 0

        def get_samples(self, _count):
            raise module.TrainingValidationBoundaryReached(
                attempted_instances=100,
                boundary=100,
            )

    if filename == "collect_naive_rollout_async.py":
        monkeypatch.setattr(
            module,
            "_policy_endpoints_for_model",
            lambda *_args, **_kwargs: [("127.0.0.1", 30000)],
        )
        submitted = module._submit_until_full(
            args=SimpleNamespace(),
            data_buffer=BoundaryDataBuffer(),
            rollout_id=1,
            output_root=tmp_path,
            model_name="Qwen",
            max_pending=1,
            target_groups=1,
        )
    else:
        monkeypatch.setattr(
            module,
            "_policy_urls_for_model",
            lambda *_args, **_kwargs: ["http://127.0.0.1:30000"],
        )
        monkeypatch.setattr(
            module,
            "_hosted_lane_c_settings",
            lambda: (
                "nvidia/zai-org/glm-5.2",
                "nvidia/zai-org/glm-5.2",
                "https://inference-api.nvidia.com/v1",
                "secret",
            ),
        )
        submitted = module._submit_until_full(
            args=SimpleNamespace(),
            data_buffer=BoundaryDataBuffer(),
            rollout_id=1,
            output_root=tmp_path,
            model_name="Qwen",
            max_pending=1,
            over_sampling_groups=1,
            est_groups_per_instance=1.0,
        )

    assert submitted == 0
    assert module._SOURCE_VALIDATION_ERROR.attempted_instances == 100
    assert module._SOURCE_VALIDATION_ERROR.boundary == 100


@pytest.mark.parametrize(
    "filename",
    ["collect_naive_rollout_async.py", "collect_lanes_rollout_async.py"],
)
def test_collectors_preserve_partial_through_validation_ack_and_refill(
    monkeypatch, tmp_path, filename
):
    module = _load_collector(monkeypatch, filename)
    monkeypatch.setenv("RLER_USAGE_LEDGER_PATH", str(tmp_path / "usage.jsonl"))
    module._WARMUP_DONE = True
    module._NODE_WORKERS.append(object())
    usage_group_id = "train/r0100/django__django-1/t000100:g0"
    boundary_error = module.TrainingValidationBoundaryReached(
        attempted_instances=100,
        boundary=100,
    )

    class DataBuffer:
        last_validation_attempt = 0

    data_buffer = DataBuffer()
    calls = 0

    def submit_partial_then_stop(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            if filename == "collect_naive_rollout_async.py":
                group = module._BufferedGroup(
                    samples=[],
                    rollout_id=1,
                    usage_group_id=usage_group_id,
                )
                module._TOTAL_GROUPS_ATTEMPTED += 3
                module._INFRA_DROPPED_GROUPS += 1
                module._FILTER_DROPPED_GROUPS += 1
            else:
                group = module._BufferedGroup(
                    samples=[],
                    rollout_id=1,
                    group_kind="root",
                    usage_group_id=usage_group_id,
                )
                module._TOTAL_GROUPS_ATTEMPTED += 3
                module._GROUPS_ATTEMPTED_BY_KIND["root"] = 3
                module._INVALID_DROPPED_GROUPS += 1
                module._INVALID_DROPPED_BY_KIND["root"] = 1
                module._FILTER_DROPPED_GROUPS += 1
                module._FILTER_DROPPED_BY_KIND["root"] = 1
            module._BUFFER.append(group)
            module._SOURCE_VALIDATION_ERROR = boundary_error
        elif calls == 2 and data_buffer.last_validation_attempt >= 100:
            module._SOURCE_VALIDATION_ERROR = None
            if filename == "collect_naive_rollout_async.py":
                group = module._BufferedGroup(
                    samples=[],
                    rollout_id=1,
                    usage_group_id=f"{usage_group_id}-refill",
                )
            else:
                group = module._BufferedGroup(
                    samples=[],
                    rollout_id=1,
                    group_kind="root",
                    usage_group_id=f"{usage_group_id}-refill",
                )
                module._GROUPS_ATTEMPTED_BY_KIND["root"] += 1
            module._TOTAL_GROUPS_ATTEMPTED += 1
            module._BUFFER.append(group)
        return 0

    monkeypatch.setattr(module, "_submit_until_full", submit_partial_then_stop)
    monkeypatch.setattr(
        module,
        "_harvest_ready",
        lambda **_kwargs: 0,
    )
    args = SimpleNamespace(
        rollout_batch_size=2,
        over_sampling_batch_size=2,
    )

    with pytest.raises(module.TrainingValidationBoundaryReached) as exc_info:
        module.generate_rollout(
            args,
            rollout_id=1,
            data_buffer=data_buffer,
        )

    assert exc_info.value is boundary_error
    assert exc_info.value.preserved_partial is True
    assert exc_info.value.preserved_group_count == 1
    assert len(module._BUFFER) == 1
    events = sys.modules["swe_agent.usage"]._test_state["events"]
    assert not any(
        event["disposition"] == "accepted" for event in events
    )
    with pytest.raises(RuntimeError, match="before validation is scheduled"):
        module.generate_rollout(
            args,
            rollout_id=1,
            data_buffer=data_buffer,
        )

    data_buffer.last_validation_attempt = 100
    output = module.generate_rollout(
        args,
        rollout_id=1,
        data_buffer=data_buffer,
    )
    assert output.metrics["swe_agent/groups"] == 2
    assert output.metrics["swe_agent/groups_attempted"] == 4
    assert output.metrics["swe_agent/groups_accepted"] == 2
    assert output.metrics["swe_agent/groups_invalid"] == 1
    assert output.metrics["swe_agent/groups_dynamic_filtered"] == 1
    assert module._BUFFER == []
    assert module._VALIDATION_PARTIAL_STATE is None
    assert calls == 3
    assert [
        event["disposition"]
        for event in events
        if event["group_id"].startswith(usage_group_id)
    ] == ["accepted", "accepted"]
    assert not any(
        event["reason"] == "validation_boundary_partial_update"
        for event in events
    )
    commits = [
        call
        for call in sys.modules["swe_agent.usage"]._test_state[
            "tracker_calls"
        ]
        if call[0] == "commit_update"
    ]
    assert len(commits) == 1
    assert [
        event["disposition"] for event in commits[0][1]
    ] == ["accepted", "accepted"]


def test_depth2_partial_refill_preserves_fifo_groups(
    monkeypatch, tmp_path
):
    module = _load_collector(
        monkeypatch, "collect_lanes_rollout_async.py"
    )
    monkeypatch.setenv("SWE_AGENT_LANES_TOPOLOGY", "depth2")
    module._WARMUP_DONE = True
    module._NODE_WORKERS.append(object())
    boundary_error = module.TrainingValidationBoundaryReached(
        attempted_instances=100,
        boundary=100,
    )

    class DataBuffer:
        last_validation_attempt = 0

    data_buffer = DataBuffer()
    refilled = False

    def add_group(kind, suffix):
        module._BUFFER.append(
            module._BufferedGroup(
                samples=[],
                rollout_id=7,
                group_kind=kind,
                usage_group_id=f"train/r0001/instance/{suffix}",
            )
        )
        module._TOTAL_GROUPS_ATTEMPTED += 1
        module._GROUPS_ATTEMPTED_BY_KIND[kind] = (
            module._GROUPS_ATTEMPTED_BY_KIND.get(kind, 0) + 1
        )

    def submit_partial_then_refill(**_kwargs):
        nonlocal refilled
        if not module._BUFFER:
            if refilled:
                return 0
            add_group("root", "root-before")
            add_group("beam", "beam-before")
            module._SOURCE_VALIDATION_ERROR = boundary_error
        elif data_buffer.last_validation_attempt >= 100 and not refilled:
            refilled = True
            module._SOURCE_VALIDATION_ERROR = None
            add_group("root", "root-after")
            add_group("beam", "beam-after")
        return 0

    monkeypatch.setattr(
        module, "_submit_until_full", submit_partial_then_refill
    )
    monkeypatch.setattr(module, "_harvest_ready", lambda **_kwargs: 0)
    args = SimpleNamespace(
        rollout_batch_size=4,
        over_sampling_batch_size=4,
    )

    with pytest.raises(module.TrainingValidationBoundaryReached):
        module.generate_rollout(args, 7, data_buffer)
    assert boundary_error.preserved_group_kinds == {
        "root": 1,
        "beam": 1,
    }

    data_buffer.last_validation_attempt = 100
    output = module.generate_rollout(args, 7, data_buffer)
    assert output.metrics["swe_agent/root_groups"] == 2
    assert output.metrics["swe_agent/beam_groups"] == 2
    assert output.metrics["swe_agent/root_groups_attempted"] == 2
    assert output.metrics["swe_agent/beam_groups_attempted"] == 2
    assert output.metrics["swe_agent/groups_excess"] == 0


def test_naive_forbids_hidden_buffer_outside_validation_resume(monkeypatch):
    module = _load_collector(
        monkeypatch,
        "collect_naive_rollout_async.py",
    )
    carried = module._BufferedGroup(samples=[], rollout_id=2)
    module._BUFFER.append(carried)
    assert (
        module._validation_partial_resume_state(
            data_buffer=SimpleNamespace(last_validation_attempt=100),
            rollout_id=3,
        )
        is None
    )
    assert module._BUFFER == [carried]


def test_lanes_reuses_normal_excess_buffer_across_rollout_cycles(monkeypatch):
    module = _load_collector(
        monkeypatch,
        "collect_lanes_rollout_async.py",
    )
    carried = module._BufferedGroup(samples=[], group_kind="root")
    module._BUFFER.append(carried)

    assert (
        module._validation_partial_resume_state(
            data_buffer=SimpleNamespace(last_validation_attempt=100),
            rollout_id=3,
        )
        is None
    )
    assert module._pop_groups(1) == [carried]
    assert module._BUFFER == []


def test_lanes_validates_before_using_quota_ready_carry_buffer(
    monkeypatch, tmp_path
):
    module = _load_collector(
        monkeypatch,
        "collect_lanes_rollout_async.py",
    )
    monkeypatch.setenv("RLER_USAGE_LEDGER_PATH", str(tmp_path / "usage.jsonl"))
    module._WARMUP_DONE = True
    module._NODE_WORKERS.append(object())
    carried = module._BufferedGroup(
        samples=[],
        rollout_id=11,
        group_kind="root",
        usage_group_id="train/r0100/instance/t000100:g0",
    )
    module._BUFFER.append(carried)

    class DataBuffer:
        last_validation_attempt = 0

        def training_progress(self):
            return {
                "attempted_instances": 100,
                "last_validation_attempt": self.last_validation_attempt,
                "eval_instance_interval": 100,
                "instance_budget": 1250,
                "epoch": 0.4,
            }

    data_buffer = DataBuffer()
    dispatch_calls = 0

    def record_post_pop_prefetch(**_kwargs):
        nonlocal dispatch_calls
        dispatch_calls += 1
        return 0

    monkeypatch.setattr(module, "_submit_until_full", record_post_pop_prefetch)
    args = SimpleNamespace(
        rollout_batch_size=1,
        over_sampling_batch_size=1,
    )

    with pytest.raises(module.TrainingValidationBoundaryReached) as exc_info:
        module.generate_rollout(args, 11, data_buffer)
    assert exc_info.value.boundary == 100
    assert exc_info.value.preserved_partial is True
    assert exc_info.value.preserved_group_count == 1
    assert exc_info.value.preserved_group_kinds == {"root": 1}
    assert module._BUFFER == [carried]
    assert dispatch_calls == 0

    data_buffer.last_validation_attempt = 100
    output = module.generate_rollout(args, 11, data_buffer)
    assert output.metrics["swe_agent/groups"] == 1
    assert output.metrics["swe_agent/train_instances_attempted"] == 100
    assert output.metrics["swe_agent/last_validation_attempt"] == 100
    assert module._BUFFER == []
    assert module._VALIDATION_PARTIAL_STATE is None
    assert module._SOURCE_VALIDATION_ERROR is None
    # Once validation has been scheduled, attempt 101 is allowed to prefetch
    # immediately after the carried group is returned for training.
    assert dispatch_calls == 1


@pytest.mark.parametrize(
    "filename",
    ["collect_naive_rollout_async.py", "collect_lanes_rollout_async.py"],
)
def test_collectors_validate_when_attempt_100_exactly_fills_batch(
    monkeypatch, tmp_path, filename
):
    module = _load_collector(monkeypatch, filename)
    monkeypatch.setenv("RLER_USAGE_LEDGER_PATH", str(tmp_path / "usage.jsonl"))
    module._WARMUP_DONE = True
    module._NODE_WORKERS.append(object())

    class DataBuffer:
        attempted_instances = 99
        last_validation_attempt = 0

        def training_progress(self):
            return {
                "attempted_instances": self.attempted_instances,
                "last_validation_attempt": self.last_validation_attempt,
                "eval_instance_interval": 100,
                "instance_budget": 1250,
                "epoch": self.attempted_instances / 250,
            }

    data_buffer = DataBuffer()
    dispatch_calls = 0
    dispatched = False
    usage_group_id = "train/r0100/instance/t000100:g0"

    def fill_exact_target(**_kwargs):
        nonlocal dispatch_calls, dispatched
        if dispatched:
            return 0
        dispatched = True
        dispatch_calls += 1
        if filename == "collect_naive_rollout_async.py":
            group = module._BufferedGroup(
                samples=[],
                rollout_id=13,
                usage_group_id=usage_group_id,
            )
            module._TOTAL_GROUPS_ATTEMPTED += 1
        else:
            group = module._BufferedGroup(
                samples=[],
                rollout_id=13,
                group_kind="root",
                usage_group_id=usage_group_id,
            )
            module._TOTAL_GROUPS_ATTEMPTED += 1
            module._GROUPS_ATTEMPTED_BY_KIND["root"] = 1
        module._BUFFER.append(group)
        data_buffer.attempted_instances = 100
        return 1

    monkeypatch.setattr(module, "_submit_until_full", fill_exact_target)
    monkeypatch.setattr(module, "_harvest_ready", lambda **_kwargs: 0)
    args = SimpleNamespace(
        rollout_batch_size=1,
        over_sampling_batch_size=1,
    )

    with pytest.raises(module.TrainingValidationBoundaryReached) as exc_info:
        module.generate_rollout(args, 13, data_buffer)
    assert exc_info.value.attempted_instances == 100
    assert exc_info.value.boundary == 100
    assert exc_info.value.preserved_partial is True
    assert exc_info.value.preserved_group_count == 1
    assert len(module._BUFFER) == 1
    assert dispatch_calls == 1

    data_buffer.last_validation_attempt = 100
    output = module.generate_rollout(args, 13, data_buffer)
    assert output.metrics["swe_agent/groups"] == 1
    assert output.metrics["swe_agent/train_instances_attempted"] == 100
    assert output.metrics["swe_agent/last_validation_attempt"] == 100
    assert module._BUFFER == []
    assert module._VALIDATION_PARTIAL_STATE is None
    assert module._SOURCE_VALIDATION_ERROR is None
    assert dispatch_calls == 1


@pytest.mark.parametrize(
    "filename",
    ["collect_naive_rollout_async.py", "collect_lanes_rollout_async.py"],
)
def test_collectors_stop_source_dispatch_on_instance_budget(
    monkeypatch, tmp_path, filename
):
    module = _load_collector(monkeypatch, filename)
    monkeypatch.setenv("RLER_USAGE_LEDGER_PATH", str(tmp_path / "usage.jsonl"))

    class ExhaustedDataBuffer:
        def get_samples(self, _count):
            raise module.TrainingInstanceBudgetExhausted(
                attempted_instances=1250,
                budget=1250,
            )

    if filename == "collect_naive_rollout_async.py":
        monkeypatch.setattr(
            module,
            "_policy_endpoints_for_model",
            lambda *_args, **_kwargs: [("127.0.0.1", 30000)],
        )
        submitted = module._submit_until_full(
            args=SimpleNamespace(),
            data_buffer=ExhaustedDataBuffer(),
            rollout_id=1,
            output_root=tmp_path,
            model_name="Qwen",
            max_pending=1,
            target_groups=1,
        )
    else:
        monkeypatch.setattr(
            module,
            "_policy_urls_for_model",
            lambda *_args, **_kwargs: ["http://127.0.0.1:30000"],
        )
        monkeypatch.setattr(
            module,
            "_hosted_lane_c_settings",
            lambda: (
                "nvidia/zai-org/glm-5.2",
                "nvidia/zai-org/glm-5.2",
                "https://inference-api.nvidia.com/v1",
                "secret",
            ),
        )
        submitted = module._submit_until_full(
            args=SimpleNamespace(),
            data_buffer=ExhaustedDataBuffer(),
            rollout_id=1,
            output_root=tmp_path,
            model_name="Qwen",
            max_pending=1,
            over_sampling_groups=1,
            est_groups_per_instance=1.0,
        )

    assert submitted == 0
    assert module._SOURCE_BUDGET_EXHAUSTED is True
    assert module._SOURCE_BUDGET_ERROR.attempted_instances == 1250
    assert module._SOURCE_BUDGET_ERROR.budget == 1250
    usage_group_id = "train/r1250/django__django-1/t001250:g0"
    if filename == "collect_naive_rollout_async.py":
        module._BUFFER.append(
            module._BufferedGroup(samples=[], usage_group_id=usage_group_id)
        )
        module._drop_incomplete_source_budget_buffer()
    else:
        module._BUFFER.append(
            module._BufferedGroup(
                samples=[],
                group_kind="root",
                usage_group_id=usage_group_id,
            )
        )
        module._discard_excess_buffer(reason="source_budget_partial_update")
    events = sys.modules["swe_agent.usage"]._test_state["events"]
    assert [
        (event["group_id"], event["disposition"], event["reason"])
        for event in events
    ] == [
        (
            usage_group_id,
            "dropped",
            "source_budget_partial_update",
        )
    ]


def test_heartbeat_event_step_resumes_from_durable_last_pulse(
    monkeypatch, tmp_path
):
    module = _load_collector(monkeypatch, "collect_naive_rollout_async.py")
    output_root = tmp_path / "rollouts"
    output_root.mkdir()
    heartbeat_path = output_root / "heartbeat.jsonl"
    heartbeat_path.write_text(
        "\n".join(
            [
                json.dumps({"event_step": 40}),
                json.dumps({"event_step": 41}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(sys.modules, "wandb", _module("wandb", run=None))

    module._emit_heartbeat(
        source="naive",
        rollout_id=9,
        elapsed_seconds=12.0,
        pending=2,
        buffered_groups=1,
        submitted=3,
        output_root=output_root,
    )

    records = [
        json.loads(line)
        for line in heartbeat_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[-1]["event_step"] == 42
    assert module._HEARTBEAT_EVENT_STEP == 42


def test_heartbeat_peeks_usage_without_committing_update_delta(
    monkeypatch, tmp_path
):
    module = _load_collector(monkeypatch, "collect_naive_rollout_async.py")
    output_root = tmp_path / "rollouts"
    logged = []
    monkeypatch.setenv(
        "RLER_USAGE_LEDGER_PATH",
        str(tmp_path / "usage.jsonl"),
    )
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        _module(
            "wandb",
            run=object(),
            log=lambda metrics: logged.append(dict(metrics)),
        ),
    )

    module._emit_heartbeat(
        source="naive",
        rollout_id=1,
        elapsed_seconds=2.0,
        pending=1,
        buffered_groups=0,
        submitted=1,
        output_root=output_root,
    )

    calls = sys.modules["swe_agent.usage"]._test_state["tracker_calls"]
    assert [call[0] for call in calls] == ["peek"]
    assert logged[-1]["usage/event_step"] == 0
    assert not any(key.endswith("_delta") for key in logged[-1])


@pytest.mark.parametrize(
    "filename",
    [
        "collect_naive_rollout_async.py",
        "collect_lanes_rollout_async.py",
    ],
)
def test_collector_checkpoint_round_trips_buffer_pending_and_counters(
    monkeypatch,
    filename,
):
    module = _load_collector(monkeypatch, filename)
    saved_random_state = (
        module.random.getstate() if hasattr(module, "random") else None
    )
    try:
        sample = module.Sample(
            group_index=3,
            index=9,
            tokens=[1, 2, 3],
            response="answer",
            response_length=3,
            reward=0.75,
            status=module.Sample.Status.COMPLETED,
            metadata={"instance_id": "repo__task-1"},
            weight_versions=["checkpoint-0000003"],
        )
        group_kwargs = {
            "samples": [sample],
            "rollout_id": 3,
            "usage_group_id": "train/group-1",
        }
        if filename.startswith("collect_lanes"):
            group_kwargs["group_kind"] = "beam"
        module._BUFFER.append(module._BufferedGroup(**group_kwargs))
        module._PENDING[object()] = {
            "index": 17,
            "rollout_id": 3,
            "instance_id": "repo__task-2",
            "subset": "verified",
            "split": "test",
            "model_name": "Qwen/Qwen3.5-9B",
            "m": 8,
            "max_mid_cps": 2,
            "usage_group_prefix": "train/pending",
            "usage_group_id": "train/pending:g0",
            "policy_version": "checkpoint-0000002",
            "policy_api_key": "must-not-be-checkpointed",
            "rubric_api_key": "must-not-be-checkpointed",
            "api_key": "must-not-be-checkpointed",
            "policy_base_urls": ["http://stale-route"],
            "_pinned_endpoints": ["http://stale-route"],
        }
        module._TASK_INDEX = 18
        module._DISPATCH_COUNTER = 11
        module._FAILED_INSTANCES = 4
        module._FILTER_DROPPED_GROUPS = 5
        if filename.startswith("collect_lanes"):
            module._OBSERVED_GROUPS_TOTAL = 13
            module._OBSERVED_INSTANCES = 7
        else:
            module._TOTAL_GROUPS_ATTEMPTED = 12
            module._REWARD_GROUPS_OBSERVED = 10

        state = module.checkpoint_state_dict(3)
        assert state["usage_ledger_offset"] == 123
        (pending_state,) = state["pending_tasks"]
        assert "api_key" not in pending_state
        assert "policy_api_key" not in pending_state
        assert "rubric_api_key" not in pending_state
        assert "policy_base_urls" not in pending_state
        assert "_pinned_endpoints" not in pending_state

        module._BUFFER.clear()
        module._PENDING.clear()
        module._REPLAY_PENDING_TASKS.clear()
        module._FAILED_INSTANCES = 0
        module._FILTER_DROPPED_GROUPS = 0
        module.load_checkpoint_state_dict(state, 3)
        assert (
            sys.modules["swe_agent.usage"]._test_state["resume_offset"]
            == 123
        )

        assert module._TASK_INDEX == 18
        assert module._DISPATCH_COUNTER == 11
        assert module._FAILED_INSTANCES == 4
        assert module._FILTER_DROPPED_GROUPS == 5
        assert len(module._BUFFER) == 1
        assert module._BUFFER[0].rollout_id == 3
        assert module._BUFFER[0].samples[0].tokens == [1, 2, 3]
        assert (
            module._BUFFER[0].samples[0].status
            == module.Sample.Status.COMPLETED
        )
        assert len(module._REPLAY_PENDING_TASKS) == 1
        assert module._PENDING == {}
        if filename.startswith("collect_lanes"):
            assert module._BUFFER[0].group_kind == "beam"
            assert module._OBSERVED_GROUPS_TOTAL == 13
            assert module._OBSERVED_INSTANCES == 7
        else:
            assert module._TOTAL_GROUPS_ATTEMPTED == 12
            assert module._REWARD_GROUPS_OBSERVED == 10
    finally:
        if saved_random_state is not None:
            module.random.setstate(saved_random_state)


@pytest.mark.parametrize(
    "filename",
    [
        "collect_naive_rollout_async.py",
        "collect_lanes_rollout_async.py",
    ],
)
def test_checkpoint_pending_replay_uses_current_routes_without_source_draw(
    monkeypatch,
    filename,
    tmp_path,
):
    module = _load_collector(monkeypatch, filename)
    captured = []
    replay_dispositions = []

    class Submit:
        def remote(self, task):
            captured.append(task)
            return object()

    module._NODE_WORKERS = [
        SimpleNamespace(submit_task=Submit())
    ]
    module._REPLAY_PENDING_TASKS = [
        {
            "index": 21,
            "rollout_id": 3,
            "instance_id": "repo__task-3",
            "subset": "verified",
            "split": "test",
            "model_name": "Qwen/Qwen3.5-9B",
            "m": 8,
            "max_mid_cps": 2,
            "topology": "depth2",
            "source_group_index": 123,
            "usage_group_prefix": (
                "train/attempt-old/r0003/repo__task-3/t000123"
            ),
            "usage_group_id": (
                "train/attempt-old/r0003/repo__task-3/t000123:g0"
            ),
            "policy_version": "checkpoint-0000002",
        }
    ]
    monkeypatch.setattr(
        module,
        "_dispatch_policy_version",
        lambda rollout_id: "checkpoint-0000003",
    )
    monkeypatch.setattr(
        module,
        "_ensure_usage_tracking",
        lambda: str(tmp_path / "usage.jsonl"),
    )
    monkeypatch.setattr(
        module,
        "_record_usage_disposition",
        lambda group_id, *, disposition, reason="", phase="train": (
            replay_dispositions.append(
                {
                    "group_id": group_id,
                    "disposition": disposition,
                    "reason": reason,
                    "phase": phase,
                }
            )
        ),
    )
    if filename.startswith("collect_lanes"):
        monkeypatch.setattr(
            module,
            "_policy_urls_for_model",
            lambda args, model_name: [
                "http://new-policy-0",
                "http://new-policy-1",
            ],
        )
        monkeypatch.setattr(
            module,
            "_hosted_lane_c_settings",
            lambda: (
                "nvidia/zai-org/glm-5.2",
                "nvidia/zai-org/glm-5.2",
                "https://inference-api.nvidia.com/v1",
                "new-hosted-key",
            ),
        )
    else:
        monkeypatch.setattr(
            module,
            "_policy_endpoints_for_model",
            lambda args, model_name: [
                ("new-policy-0", 8000),
                ("new-policy-1", 8001),
            ],
        )

    count = module._replay_checkpoint_pending_tasks(
        args=SimpleNamespace(),
        rollout_id=4,
        output_root=tmp_path / "search",
        model_name="Qwen/Qwen3.5-9B",
    )

    assert count == 1
    assert len(captured) == 1
    assert len(module._PENDING) == 1
    assert module._REPLAY_PENDING_TASKS == []
    replayed = captured[0]
    assert replayed["rollout_id"] == 4
    assert replayed["policy_version"] == "checkpoint-0000003"
    assert replayed["source_group_index"] == 123
    assert replayed["suppress_usage_accounting"] is True
    assert replayed["usage_group_prefix"] == (
        "train/attempt-test/r0004/repo__task-3/t000123"
    )
    assert replayed["policy_base_urls"]
    assert all(
        "new-policy" in url for url in replayed["policy_base_urls"]
    )
    if filename.startswith("collect_lanes"):
        assert "usage_group_id" not in replayed
        expected_old_ids = {
            "train/attempt-old/r0003/repo__task-3/t000123:g0",
            "train/attempt-old/r0003/repo__task-3/t000123:g1",
        }
    else:
        assert replayed["usage_group_id"] == (
            "train/attempt-test/r0004/repo__task-3/t000123:g0"
        )
        expected_old_ids = {
            "train/attempt-old/r0003/repo__task-3/t000123:g0"
        }
    replay_events = [
        event
        for event in replay_dispositions
        if event["reason"] == "checkpoint_replay"
    ]
    assert {event["group_id"] for event in replay_events} == expected_old_ids
    assert {
        event["disposition"] for event in replay_events
    } == {"dropped"}
