import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

from agent_rl import ChatCompletion
from agent_rl import run_utils
import swe_agent.trajectory_search as trajectory_search
import swe_agent.trajectory_search_parallel as trajectory_search_parallel
from swe_agent.exceptions import FormatError
from swe_agent.models.litellm_textbased_model import LitellmTextbasedModel
from swe_agent.models.utils.actions_text import parse_regex_actions
from swe_agent.parallel_utils import NodeArtifactBundle
from swe_agent.parallel_utils import PatchEvalManager
from swe_agent.parallel_utils import RubricArtifactBundle
from swe_agent.parallel_utils import _write_base_artifacts
from swe_agent.parallel_utils import progress_reward
from swe_agent.prompt import PC_RUBRIC_EXPERIENCE_UPDATE_PROMPT
from swe_agent.rubric_bank import ExperienceRubricBank
from swe_agent.rubric_bank import RubricRecord
from swe_agent.rubric_bank import ScoreRubricBank
from swe_agent.rubric_bank import _convert_rubric_item
from swe_agent.rubric_bank import render_compact_markdown


def test_rubric_converter_accepts_unambiguous_field_aliases_and_scale_list():
    rubric = _convert_rubric_item(
        "task",
        {
            "Direction": "positive",
            "Importance": "1.5",
            "Name": "Contract coverage",
            "Criterion": "The continuation verifies the exposed contract.",
            "Scale": ["none", "weak", "partial", "strong", "complete"],
            "Metadata": {"stage": "validation"},
        },
        1,
    )
    assert rubric is not None
    assert rubric.weight == 1.5
    assert rubric.scale["5"] == "complete"
from swe_agent.run.search_swe_agent import _flush_experience_bank_summaries
from swe_agent.run.search_swe_agent import build_arg_parser
from swe_agent.trajectory_search import (
    SearchConfig,
    SearchNode,
    TrajectorySearchRunner,
    _avg_scores_from_rubrics,
    _combine_score_results,
    _first_round_gt_gate,
    _generate_and_score_rubric_batch,
    _normalize_terminal_patch_text,
    _pc_avg_scores_from_rubrics,
    _resume_snapshot_payload,
    _select_beam_branches,
)
from swe_agent.trajectory_search_parallel import TrajectorySearchParallelRunner


def test_first_round_gt_gate_only_continues_for_mixed_resolved_branches():
    assert _first_round_gt_gate(
        {"node-1": {"reward": 2.0}, "node-2": {"reward": 0.5}},
        branch_count=2,
    )["continue_search"] is True
    assert _first_round_gt_gate(
        {"node-1": {"reward": 2.0}, "node-2": {"reward": 2.0}},
        branch_count=2,
    )["continue_search"] is False
    assert _first_round_gt_gate(
        {"node-1": {"reward": 0.5}, "node-2": {"reward": 0.0}},
        branch_count=2,
    )["continue_search"] is False


def test_singularity_defaults_hide_host_bind_paths(monkeypatch):
    from swe_agent.environments.singularity import (
        SingularityEnvironmentConfig,
        _runtime_environment,
    )

    config = SingularityEnvironmentConfig(image="image.sif")
    assert config.exec_args[-2:] == [
        "--no-mount",
        "home,cwd,tmp,hostfs,bind-paths",
    ]
    monkeypatch.setenv("APPTAINER_BIND", "/tmp:/tmp")
    monkeypatch.setenv("SINGULARITY_BINDPATH", "/workspace:/workspace")
    environment = _runtime_environment()
    assert "APPTAINER_BIND" not in environment
    assert "SINGULARITY_BINDPATH" not in environment


def test_singularity_runtime_guard_is_opt_in(monkeypatch):
    from swe_agent.environments.singularity import (
        _runtime_guard_enabled,
        _runtime_memory_limit_kib,
    )

    monkeypatch.delenv("RLER_SINGULARITY_RUNTIME_GUARD", raising=False)
    monkeypatch.delenv("RLER_SINGULARITY_MEMORY_LIMIT_GB", raising=False)
    assert _runtime_guard_enabled() is False
    assert _runtime_memory_limit_kib() is None

    monkeypatch.setenv("RLER_SINGULARITY_RUNTIME_GUARD", "1")
    monkeypatch.setenv("RLER_SINGULARITY_MEMORY_LIMIT_GB", "192")
    assert _runtime_guard_enabled() is True
    assert _runtime_memory_limit_kib() == 192 * 1024 * 1024


def test_singularity_runtime_guard_wraps_action(monkeypatch, tmp_path):
    import subprocess

    from swe_agent.environments.singularity import SingularityEnvironment

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured.update(kwargs)
        return subprocess.CompletedProcess(cmd, 0, "ok")

    environment = SingularityEnvironment.__new__(SingularityEnvironment)
    environment.config = SimpleNamespace(
        executable="apptainer",
        global_args=[],
        exec_args=[],
        cwd="/testbed",
        timeout=60,
        forward_env=[],
        env={},
    )
    environment.sandbox_dir = tmp_path
    environment._owns_sandbox = False
    monkeypatch.setenv("RLER_SINGULARITY_RUNTIME_GUARD", "1")
    monkeypatch.setenv("RLER_SINGULARITY_MEMORY_LIMIT_GB", "192")
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert environment.execute({"command": "echo ok"})["output"] == "ok"
    assert captured["cmd"][:4] == [
        "timeout",
        "--signal=TERM",
        "--kill-after=5s",
        "60s",
    ]
    assert captured["cmd"][-1].startswith("ulimit -v 201326592\n")
    assert captured["timeout"] == 70


def test_partially_initialized_singularity_environment_can_cleanup():
    from swe_agent.environments.singularity import SingularityEnvironment

    environment = SingularityEnvironment.__new__(SingularityEnvironment)
    environment._owns_sandbox = True

    environment.cleanup()


def test_singularity_evaluator_keeps_only_explicit_bind_paths(monkeypatch, tmp_path):
    from swe_agent.run.benchmarks import container_runtime

    captured = {}

    class FakeEnvironment:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.sandbox_dir = tmp_path / "sandbox"
            self.sandbox_dir.mkdir()

    monkeypatch.setattr(container_runtime, "select_container_backend", lambda: "singularity")
    monkeypatch.setattr(container_runtime, "resolve_singularity_image", lambda *_: tmp_path / "image.sif")
    monkeypatch.setattr(container_runtime, "SingularityEnvironment", FakeEnvironment)
    source = tmp_path / "evaluation"
    source.mkdir()
    container_runtime.make_bound_environment(
        image="image",
        instance={},
        binds=[(source, "/eval")],
        cwd="/testbed",
        timeout=60,
    )

    assert captured["exec_args"][:4] == [
        "--cleanenv",
        "--no-home",
        "--no-mount",
        "home,cwd,tmp,hostfs",
    ]
    assert captured["exec_args"][4:] == [
        "--bind",
        f"{source.resolve()}:/eval",
    ]


def test_new_continuation_view_renders_raw_trajectory_and_truncated_diff():
    assert trajectory_search.MAX_WORKSPACE_DIFF_CHARS == 4000
    long_diff = "diff --git a/a.py b/a.py\n@@ -1 +1 @@\n" + ("+changed\n" * 500)
    summary = trajectory_search._truncate_middle(long_diff, 300)
    assert len(summary) <= 300
    assert "diff --git a/a.py b/a.py" in summary
    assert "+changed" in summary

    rendered = trajectory_search._render_continuation_view(
        {
            "node_id": "node-1",
            "summary": {"git_diff": summary},
            "raw_continuation": {
                "segment_step_range": [0, 1],
                "step_cards": [
                    {
                        "step_index": 1,
                        "assistant_message": "Run `pwd`.",
                        "commands": ["pwd"],
                        "observation": "/testbed",
                    }
                ],
            },
        }
    )
    assert "Trajectory:\nSegment step range: 0-1" in rendered
    assert "pwd" in rendered


def test_route_completion_message_requires_reasoning_for_structured_sglang(monkeypatch):
    import swe_agent.tokenization as tokenization

    captured = {}

    monkeypatch.setattr(tokenization, "tokenize_messages_with_template", lambda *args, **kwargs: [1, 2, 3])
    monkeypatch.setattr(tokenization, "get_stop_token_ids", lambda *args, **kwargs: [4])

    async def fake_generate(**kwargs):
        captured.update(kwargs)
        return ChatCompletion(
            content="<think>reasoning</think>\n{\"score\": 1}",
            model_name=kwargs["route_name"],
            usage={"prompt_tokens": 3, "completion_tokens": 8, "total_tokens": 11},
            metadata={"content_no_thinking": "{\"score\": 1}"},
            input_token_ids=[1, 2, 3],
            output_token_ids=[5, 6],
            output_logprobs=[-0.1, -0.2],
        )

    monkeypatch.setattr(run_utils, "run_generate_with_route_async", fake_generate)

    message = asyncio.run(
        run_utils.route_completion_message(
            route_name="rubric_judge",
            model_name="Qwen/Qwen3.5-9B",
            messages=[{"role": "user", "content": "return a score"}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=64,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "generic_test_schema",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"score": {"type": "integer"}},
                        "required": ["score"],
                    },
                },
            },
            model_kwargs={
                "api_base": "http://127.0.0.1:8021/v1",
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
            },
        )
    )

    assert captured["require_reasoning"] is True
    assert "json_schema" in captured["sampling_params"]
    assert message["content"].startswith("<think>reasoning</think>")
    assert message["content_no_thinking"] == "{\"score\": 1}"
    assert message["prompt_token_ids"] == [1, 2, 3]
    assert message["token_ids"] == [5, 6]


def test_route_completion_message_uses_litellm_for_hosted_nvidia(monkeypatch):
    captured = {}

    async def fail_generate(**kwargs):
        raise AssertionError("hosted NVIDIA must not use SGLang /generate")

    async def fake_litellm(**kwargs):
        captured.update(kwargs)
        return ChatCompletion(
            content='{"score": 0.75}',
            model_name=kwargs["model_name"],
            usage={"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
        )

    monkeypatch.setattr(run_utils, "run_generate_with_route_async", fail_generate)
    monkeypatch.setattr(run_utils, "run_litellm_completion_async", fake_litellm)

    message = asyncio.run(
        run_utils.route_completion_message(
            route_name="rubric_judge",
            model_name="nvidia/zai-org/glm-5.2",
            messages=[{"role": "user", "content": "judge"}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
            model_kwargs={
                "api_base": "https://inference-api.nvidia.com/v1",
                "api_key": "secret",
                "custom_llm_provider": "openai",
            },
        )
    )

    assert captured["api_base"] == "https://inference-api.nvidia.com/v1"
    assert captured["api_key"] == "secret"
    assert captured["custom_llm_provider"] == "openai"
    assert captured["usage_model_role"] == "rubric_judge"
    assert captured["messages"] == [{"role": "user", "content": "judge"}]
    assert message["content"] == '{"score": 0.75}'


def test_route_completion_message_strips_unsupported_azure_chat_template_kwargs(
    monkeypatch,
):
    captured = {}

    async def fake_litellm(**kwargs):
        captured.update(kwargs)
        return ChatCompletion(
            content='{"score": 0.75}',
            model_name=kwargs["model_name"],
        )

    monkeypatch.setattr(run_utils, "run_litellm_completion_async", fake_litellm)

    asyncio.run(
        run_utils.route_completion_message(
            route_name="rubric_judge",
            model_name="openai/azure/zai-org/glm-5.2",
            messages=[{"role": "user", "content": "judge"}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=20480,
            model_kwargs={
                "api_base": "https://inference-api.nvidia.com/v1",
                "api_key": "secret",
                "completion_backend": "litellm",
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": True},
                    "provider_supported_field": "keep-me",
                },
            },
        )
    )

    assert captured["model_name"] == "openai/azure/zai-org/glm-5.2"
    assert captured["extra_body"] == {"provider_supported_field": "keep-me"}


def test_text_action_parser_turns_non_string_content_into_format_error():
    try:
        parse_regex_actions(
            None,
            action_regex=r"```mswea_bash_command\s*\n(.*?)\n```",
            format_error_template="{{ error }}",
        )
    except FormatError as exc:
        assert exc.messages[0]["role"] == "user"
        assert "content=None" in exc.messages[0]["content"]
        assert exc.messages[0]["extra"]["content_type_error"] is True
        assert exc.messages[0]["extra"]["model_response"] is None
    else:
        raise AssertionError("Expected FormatError")


def test_text_action_parser_reports_non_string_content_type():
    try:
        parse_regex_actions(
            {"text": "hello"},
            action_regex=r"```mswea_bash_command\s*\n(.*?)\n```",
            format_error_template="{{ error }}",
        )
    except FormatError as exc:
        assert "content of type dict" in exc.messages[0]["content"]
        assert "{'text': 'hello'}" in exc.messages[0]["content"]
        assert exc.messages[0]["extra"]["content_type_error"] is True
        assert exc.messages[0]["extra"]["model_response"] == {"text": "hello"}
    else:
        raise AssertionError("Expected FormatError")


def test_litellm_textbased_gemini_uses_timeout_non_streaming_client():
    model = LitellmTextbasedModel(model_name="gemini/gemini-3.1-pro-preview", model_kwargs={})

    kwargs = model._completion_kwargs({})

    assert kwargs["timeout"] > 0
    assert kwargs["stream"] is False
    assert kwargs["client"].__class__.__name__ == "HTTPHandler"


def test_route_litellm_completion_preserves_outer_mswea_timeout(
    monkeypatch,
):
    captured = {}

    def slow_completion(**kwargs):
        captured["timeout"] = kwargs.get("timeout")
        time.sleep(0.2)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="late success"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                model_dump=lambda: {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                }
            ),
        )

    monkeypatch.delenv("LITELLM_DEFAULT_TIMEOUT", raising=False)
    monkeypatch.setenv("MSWEA_LITELLM_TIMEOUT", "0.05")
    monkeypatch.setenv("LITELLM_OUTER_TIMEOUT_GRACE", "0")
    monkeypatch.setattr(run_utils.litellm, "completion", slow_completion)

    completion = asyncio.run(
        run_utils.run_litellm_completion_async(
            model_name="gemini/gemini-3.1-pro-preview",
            messages=[{"role": "user", "content": "score"}],
        )
    )

    assert captured["timeout"] == 0.05
    # Preserve the pre-training helper's fail-fast outer watchdog. A timeout is
    # not retried here, and its unknown provider cost is intentionally omitted
    # from approximate training usage.
    assert completion.content == ""
    assert "TimeoutError" in completion.metadata["error"]
    assert completion.metadata["timeout"] == 0.05


def test_progress_reward_returns_zero_for_tie_only_labels():
    assert progress_reward([0.5, 0.5], [0.1, 0.9]) == 0.0


def test_terminal_patch_normalization_preserves_patch_newline():
    patch = "diff --git a/a b/a\n@@ -1 +1 @@\n-a\n+b"

    normalized = _normalize_terminal_patch_text(patch)

    assert normalized.endswith("\n")
    assert normalized[:-1] == patch
    assert _normalize_terminal_patch_text(" \n\t") == ""


def test_finalize_outputs_reuses_selected_node_terminal_artifacts(tmp_path):
    task_id = "repo__pkg-1"
    node_id = "node-r005-s00-final"
    node_dir = tmp_path / "nodes" / node_id
    node_dir.mkdir(parents=True)
    terminal_messages = {
        "messages": [
            {"role": "user", "message": "task"},
            {"role": "assistant", "message": "terminal answer"},
        ],
        "model_name": "terminal-model",
    }
    terminal_patch = "diff --git a/pkg.py b/pkg.py\n--- a/pkg.py\n+++ b/pkg.py\n@@ -1 +1 @@\n-a\n+b\n"
    (node_dir / "terminal_messages.json").write_text(json.dumps(terminal_messages), encoding="utf-8")
    (node_dir / "terminal_patch.json").write_text(
        json.dumps(
            {
                task_id: {
                    "model_name_or_path": "terminal-model",
                    "instance_id": task_id,
                    "model_patch": terminal_patch,
                }
            }
        ),
        encoding="utf-8",
    )

    runner = TrajectorySearchRunner.__new__(TrajectorySearchRunner)
    runner.run_dir = tmp_path
    runner.nodes_dir = tmp_path / "nodes"
    runner.task_id = task_id
    runner.policy_model_name = "policy-model"
    runner.search_config = SearchConfig(
        calculate_gt_reward=True,
        evaluate_final_patch=False,
        write_artifacts=True,
    )
    runner.nodes = {
        node_id: SearchNode(
            node_id=node_id,
            parent_id="root",
            round_index=5,
            depth=5,
            session_id="session",
            status="frontier",
            submission="intermediate patch",
            policy_model_name="policy-model",
        )
    }
    runner.finished_node_ids = []
    runner.frontier_ids = [node_id]
    runner.best_node_id = node_id
    runner._node_judge_cache = {node_id: {"overall_score": 1.0}}
    runner._node_snapshot_cache = {
        node_id: {
            "agent": {
                "state": {
                    "messages": [
                        {"role": "user", "content": "task"},
                        {"role": "assistant", "content": "intermediate answer"},
                    ]
                }
            }
        }
    }
    runner._save_manifest = lambda: None

    runner._finalize_outputs()

    assert json.loads((tmp_path / "messages.json").read_text()) == terminal_messages
    model_patch = json.loads((tmp_path / "model_patch.json").read_text())
    assert model_patch[task_id]["model_name_or_path"] == "terminal-model"
    assert model_patch[task_id]["model_patch"] == terminal_patch
    assert runner.nodes[node_id].submission == terminal_patch


def test_checkpoint_environment_copies_singularity_sandbox(tmp_path, monkeypatch):
    source = tmp_path / "source-sandbox"
    source.mkdir()
    (source / "state.txt").write_text("checkpointed", encoding="utf-8")
    monkeypatch.setattr(trajectory_search.tempfile, "gettempdir", lambda: str(tmp_path))
    runner = TrajectorySearchRunner.__new__(TrajectorySearchRunner)
    runner.task_id = "repo__pkg-1"
    runner.environment_class = "singularity"
    session = SimpleNamespace(
        agent=SimpleNamespace(env=SimpleNamespace(sandbox_dir=source, container_id="must-not-be-used"))
    )

    checkpoint, image_id = runner._checkpoint_environment(session, "unused-docker-tag")

    assert image_id == ""
    assert (Path(checkpoint) / "state.txt").read_text(encoding="utf-8") == "checkpointed"


def test_singularity_checkpoint_sweep_does_not_call_docker(tmp_path, monkeypatch):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    runner = TrajectorySearchRunner.__new__(TrajectorySearchRunner)
    runner.frontier_ids = []
    runner.nodes = {}
    runner.base_image = "/images/base.sif"
    runner.environment_class = "singularity"
    runner.docker_executable = "definitely-missing-docker-executable"
    runner.image_repository = "rler-search/test"
    runner._node_snapshot_cache = {
        "node-a": {"metadata": {"checkpoint_image_tag": str(sandbox)}}
    }

    monkeypatch.setattr(
        trajectory_search.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Docker must not be called")),
    )

    runner._sweep_checkpoint_images(remove_all=True)

    assert not sandbox.exists()


def test_resume_snapshot_uses_environment_specific_ownership():
    base_snapshot = {
        "session_id": "session",
        "status": "paused",
        "spec": {},
        "agent": {},
        "model": {},
        "environment": {
            "type_path": "swe_agent.environments.singularity.SingularityEnvironment",
            "config": {"image": "/images/base.sif", "reuse_sandbox_dir": None},
            "state": {"sandbox_dir": "/tmp/old-sandbox", "owns_sandbox": True},
        },
    }

    singularity_payload = _resume_snapshot_payload(
        base_snapshot,
        image_tag="/tmp/checkpoint-sandbox",
        image_id="",
    )
    assert singularity_payload["environment"]["state"] == {"owns_sandbox": True}

    docker_snapshot = json.loads(json.dumps(base_snapshot))
    docker_snapshot["environment"]["type_path"] = "swe_agent.environments.docker.DockerEnvironment"
    docker_payload = _resume_snapshot_payload(
        docker_snapshot,
        image_tag="rler-search/test:checkpoint",
        image_id="sha256:123",
    )
    assert docker_payload["environment"]["state"] == {"owns_container": True}


def test_load_manifest_restores_round_rubric_cache(tmp_path):
    node = SearchNode(
        node_id="node-r001-s00-test",
        parent_id="root",
        round_index=1,
        depth=1,
        session_id="session",
        status="frontier",
    )
    node_dir = tmp_path / "nodes" / node.node_id
    node_dir.mkdir(parents=True)
    (tmp_path / "rubrics").mkdir()
    (node_dir / "node.json").write_text(json.dumps(node.__dict__))
    round_payload = {
        "round_index": 1,
        "active_bank_after": [{"rubric_id": "rubric-r001-s00"}],
    }
    (tmp_path / "rubrics" / "round_001.json").write_text(json.dumps(round_payload))
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "resume-test",
                "frontier_ids": [node.node_id],
                "best_node_id": node.node_id,
                "finished_node_ids": [],
                "current_round": 2,
                "node_ids": [node.node_id],
            }
        )
    )

    runner = TrajectorySearchRunner.__new__(TrajectorySearchRunner)
    runner.run_id = "initial"
    runner.manifest_path = tmp_path / "run_manifest.json"
    runner.nodes_dir = tmp_path / "nodes"
    runner.rubrics_dir = tmp_path / "rubrics"
    runner.rubric_bank = None
    runner.pc_rubric_bank = None

    runner._load_manifest()

    assert runner.current_round == 1
    assert runner._node_rubric_round_cache[node.node_id] == round_payload


def test_docker_checkpoint_and_sweep_do_not_use_singularity_paths(monkeypatch):
    committed = []
    docker_calls = []

    def fake_commit(executable, container_id, image_tag):
        committed.append((executable, container_id, image_tag))
        return image_tag, "sha256:checkpoint"

    def fake_run(command, **kwargs):
        docker_calls.append(command)
        if command[1:3] == ["image", "ls"]:
            return SimpleNamespace(returncode=0, stdout="rler-search/test:old\n")
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(trajectory_search, "_docker_commit", fake_commit)
    monkeypatch.setattr(trajectory_search.subprocess, "run", fake_run)
    runner = TrajectorySearchRunner.__new__(TrajectorySearchRunner)
    runner.environment_class = "docker"
    runner.docker_executable = "docker"
    runner.image_repository = "rler-search/test"
    runner.base_image = "base:latest"
    runner.frontier_ids = []
    runner.nodes = {}
    runner._node_snapshot_cache = {
        "node-a": {"metadata": {"checkpoint_image_tag": "rler-search/test:old"}}
    }
    session = SimpleNamespace(
        agent=SimpleNamespace(env=SimpleNamespace(container_id="container-123", sandbox_dir="/must/not/use"))
    )

    checkpoint = runner._checkpoint_environment(session, "rler-search/test:new")
    runner._sweep_checkpoint_images(remove_all=True)

    assert checkpoint == ("rler-search/test:new", "sha256:checkpoint")
    assert committed == [("docker", "container-123", "rler-search/test:new")]
    assert docker_calls[-1] == ["docker", "image", "rm", "-f", "rler-search/test:old"]


def test_session_disposal_dispatches_by_environment(monkeypatch):
    subprocess_calls = []
    monkeypatch.setattr(
        trajectory_search.subprocess,
        "run",
        lambda command, **kwargs: subprocess_calls.append(command),
    )
    singularity_cleanups = []
    singularity_runner = TrajectorySearchRunner.__new__(TrajectorySearchRunner)
    singularity_runner.environment_class = "singularity"
    singularity_runner.docker_executable = "apptainer"
    singularity_env = SimpleNamespace(
        sandbox_dir="/tmp/sandbox",
        container_id="must-not-be-used",
        cleanup=lambda: singularity_cleanups.append(True),
    )
    singularity_runner._dispose_session(SimpleNamespace(agent=SimpleNamespace(env=singularity_env)))

    docker_runner = TrajectorySearchRunner.__new__(TrajectorySearchRunner)
    docker_runner.environment_class = "docker"
    docker_runner.docker_executable = "docker"
    docker_env = SimpleNamespace(container_id="container-123", sandbox_dir="/must/not/use")
    docker_runner._dispose_session(SimpleNamespace(agent=SimpleNamespace(env=docker_env)))

    assert singularity_cleanups == [True]
    assert subprocess_calls == [["docker", "rm", "-f", "container-123"]]
    assert docker_env.container_id is None


def test_parallel_checkpoint_dispatches_by_environment(tmp_path, monkeypatch):
    singularity_runner = TrajectorySearchParallelRunner.__new__(TrajectorySearchParallelRunner)
    singularity_runner.environment_class = "singularity"
    singularity_runner._copy_singularity_sandbox = lambda source, tag: tmp_path / "checkpoint"
    singularity_session = SimpleNamespace(
        agent=SimpleNamespace(env=SimpleNamespace(sandbox_dir=tmp_path, container_id="must-not-be-used"))
    )
    assert singularity_runner._checkpoint_environment(singularity_session, "mid") == str(tmp_path / "checkpoint")

    commits = []
    monkeypatch.setattr(
        trajectory_search_parallel,
        "_docker_commit",
        lambda executable, container_id, image_tag, inspect_image: commits.append(
            (executable, container_id, image_tag, inspect_image)
        ),
    )
    docker_runner = TrajectorySearchParallelRunner.__new__(TrajectorySearchParallelRunner)
    docker_runner.environment_class = "docker"
    docker_runner.docker_executable = "docker"
    docker_runner.image_repository = "rler-pds/test"
    docker_runner._created_image_tags = []
    docker_session = SimpleNamespace(
        agent=SimpleNamespace(env=SimpleNamespace(container_id="container-456", sandbox_dir=tmp_path))
    )
    checkpoint = docker_runner._checkpoint_environment(docker_session, "mid")

    assert checkpoint.startswith("rler-pds/test:mid-")
    assert commits == [("docker", "container-456", checkpoint, False)]


def test_score_result_combines_active_bank_and_generated_scores():
    active_score = {
        "rubric_id": "active-rubric",
        "score_normalized": 0.25,
    }
    generated_score = {
        "rubric_id": "generated-rubric",
        "score_normalized": 0.75,
    }

    combined_scores, combined_errors = _combine_score_results(
        ([[active_score]], [{"rubric_id": "active-rubric", "error": "retry"}]),
        ([[generated_score]], []),
        continuation_count=1,
    )

    assert [score["rubric_id"] for score in combined_scores[0]] == ["active-rubric", "generated-rubric"]
    assert combined_errors == [{"rubric_id": "active-rubric", "error": "retry"}]


def test_score_rubric_bank_uses_model_output_as_next_active_bank():
    bank = ScoreRubricBank(max_active_rubrics=2)
    active = RubricRecord(
        rubric_id="active-rubric",
        title="Active",
        direction="positive",
        description="Existing active rubric.",
        scale={str(i): str(i) for i in range(1, 6)},
        weight=1,
        source_round=0,
        reward=None,
    )
    generated = RubricRecord(
        rubric_id="generated-rubric",
        title="Generated",
        direction="positive",
        description="Model-selected rubric.",
        scale={str(i): str(i) for i in range(1, 6)},
        weight=2,
        source_round=1,
        reward=None,
    )
    bank.set_state(
        active_bank=[active],
        inactive_bank=[],
    )

    update = bank.update_from_model(generated=[generated])

    assert [rubric.rubric_id for rubric in update.active_after] == ["generated-rubric"]
    assert [rubric.rubric_id for rubric in update.inactive_after] == ["active-rubric"]


def test_patch_eval_manager_writes_error_payload_when_evaluation_raises(tmp_path):
    task_id = "repo__pkg-1"
    node_dir = tmp_path / "nodes" / "node-a"
    rubric_dir = tmp_path / "rubrics" / "siblings" / "rubric-r001-s00"

    bundle = NodeArtifactBundle(
        node_id="node-a",
        node_dir=node_dir,
        node_payload={"node_id": "node-a", "parent_id": "root"},
        messages_payload=[{"role": "assistant", "content": "done"}],
        judge_payload={"overall_score": 0.75, "ground_truth_reward": None},
        prompt_payload={"messages": [{"role": "user", "content": "task"}]},
        terminal_patch_payload={
            task_id: {
                "model_patch": "diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-a\n+b\n"
            }
        },
    )
    rubric_bundle = RubricArtifactBundle(
        rubric_dir=rubric_dir,
        rubric_payload={
            "rubric_list_id": "rubric-r001-s00",
            "parent_node_id": "root",
            "generated": [],
            "format_errors": [],
            "terminal_error": None,
            "average_rubric_judged_scores": {"node-a": 0.75},
            "score_by_rubric": {"rubric-1": {"node-a": 0.75}},
        },
        messages_payload=[{"role": "assistant", "content": "{}"}],
    )

    def raising_eval(**kwargs):
        raise RuntimeError("harness crashed")

    manager = PatchEvalManager(
        instance={"instance_id": task_id},
        task_id=task_id,
        model_name="model",
        namespace=None,
        work_dir=tmp_path,
        evaluate_patches_fn=raising_eval,
        write_artifacts=True,
    )
    try:
        manager.submit_round([bundle], [rubric_bundle])
        [result] = manager.wait()
    finally:
        manager.close()

    evaluation = json.loads((node_dir / "terminal_evalution.json").read_text())
    judge = json.loads((node_dir / "judge.json").read_text())
    rubric = json.loads((rubric_dir / "rubric.json").read_text())
    assert evaluation["status"] == "error"
    assert evaluation["reward"] == 0.0
    assert "harness crashed" in evaluation["error"]
    assert judge["ground_truth_reward"] == 0.0
    assert rubric["gt_by_rubric"]["rubric-1"]["ground_truth_by_node"] == {"node-a": 0.0}
    assert result["rubric_update_payloads"][0]["scope"] == "siblings"


def test_rubric_base_artifacts_use_dedicated_message_files(tmp_path):
    rubric_dir = tmp_path / "rubrics" / "siblings" / "rubric-r001-s00"
    messages = [{"role": "assistant", "content": "{}"}]
    retrieve_messages = [{"role": "assistant", "content": '{"titles": []}'}]
    judge_messages = [
        {
            "node_id": "node-a",
            "rubric_id": "rubric-1",
            "messages": [{"role": "assistant", "content": '{"score": 4}'}],
            "error": None,
        }
    ]
    summary_messages = [{"role": "assistant", "content": '{"state": "ready"}'}]
    tie_break_messages = [{"role": "assistant", "content": '{"rubrics": []}'}]
    bundle = RubricArtifactBundle(
        rubric_dir=rubric_dir,
        rubric_payload={"rubric_list_id": "rubric-r001-s00"},
        messages_payload=messages,
        retrieve_messages_payload=retrieve_messages,
        judge_messages_payload=judge_messages,
        tie_break_messages_payload=tie_break_messages,
        summary_messages_payload=summary_messages,
    )

    _write_base_artifacts(rubric_bundles=[bundle])

    assert json.loads((rubric_dir / "rubric_message.json").read_text()) == messages
    assert json.loads((rubric_dir / "rubric_retrieve_message.json").read_text()) == retrieve_messages
    assert json.loads((rubric_dir / "judge_message.json").read_text()) == judge_messages
    assert json.loads((rubric_dir / "tie_break_message.json").read_text()) == tie_break_messages
    assert json.loads((rubric_dir / "summary_message.json").read_text()) == summary_messages
    assert not (rubric_dir / "messages.json").exists()


def test_persistent_state_format_failure_reuses_parent_and_keeps_assistant_suffix(monkeypatch):
    parent_state = {"current_state": "keep this"}

    async def fake_route_completion_message(**kwargs):
        return {
            "role": "assistant",
            "content": "no parseable object",
            "content_no_thinking": "no parseable object",
        }

    monkeypatch.setattr(trajectory_search, "route_completion_message", fake_route_completion_message)
    result = asyncio.run(
        trajectory_search._update_persistent_state(
            system_prompt="system",
            user_prompt="task",
            previous_state=parent_state,
            evicted_step_cards=[{"step_index": 0}],
            workspace_meta={},
            model_name="dummy",
            temperature=0.0,
            top_p=1.0,
            max_tokens=16,
        )
    )

    assert result["state"] == parent_state
    assert result["error"] == "InvalidPersistentStateResponse"
    assert result["messages"][0]["role"] == "assistant"
    assert all(message["content"] != "task" for message in result["messages"])


def test_rubric_scope_only_judges_model_selected_rubrics(monkeypatch):
    active = RubricRecord(
        rubric_id="active-rubric",
        title="Active",
        direction="positive",
        description="Existing active rubric.",
        scale={str(i): str(i) for i in range(1, 6)},
        weight=1,
        source_round=0,
    )
    generated = RubricRecord(
        rubric_id="generated-rubric",
        title="Generated",
        direction="positive",
        description="New rubric.",
        scale={str(i): str(i) for i in range(1, 6)},
        weight=1,
        source_round=1,
    )
    score_bank = ScoreRubricBank(max_active_rubrics=2)
    score_bank.set_state(active_bank=[active], inactive_bank=[])
    calls = []

    async def fake_score_round(**kwargs):
        raise AssertionError("Existing rubrics must be re-output by the model before judging")

    async def fake_generate_and_score_batch(**kwargs):
        calls.append(("generate", None))
        return {
            "generated_samples": [
                SimpleNamespace(
                    sample_index=0,
                    rubric_list_id="rubric-r001-s00",
                    generated=[generated],
                    messages=[],
                    format_errors=[],
                    terminal_error=None,
                )
            ],
            "sample_generated_rubrics": [[generated]],
            "generated_score_results": {
                0: (
                    [
                        [{"rubric_id": "generated-rubric", "score_normalized": 0.0}],
                        [{"rubric_id": "generated-rubric", "score_normalized": 0.0}],
                    ],
                    [],
                )
            },
        }

    monkeypatch.setattr(trajectory_search, "_score_round", fake_score_round)
    monkeypatch.setattr(trajectory_search, "_generate_and_score_rubric_batch", fake_generate_and_score_batch)
    runner = TrajectorySearchRunner.__new__(TrajectorySearchRunner)
    runner.search_config = SearchConfig(
        n=1,
        rubric_temperature=0.0,
        rubric_top_p=1.0,
        rubric_max_tokens=16,
        score_tie_break=False,
    )
    runner.rubric_model_name = "dummy"
    runner.rubric_model_kwargs = {}

    result = asyncio.run(
        runner._run_rubric_scope_judging(
            scope_spec={
                "scope": "siblings",
                "score_bank": score_bank,
                "experience_bank": None,
                "generation_prompt": "generate",
                "judge_prompt": "judge",
                "rubric_list_prefix": "rubric",
                "include_variance_reward": True,
            },
            question={"system_prompt": "system", "user_prompt": "task"},
            shared_context={"previous_persistent_state": {}, "latest_agent_trajectory": None},
            previous_state={},
            latest_shared_segment=None,
            continuations=[{"node_id": "node-a"}, {"node_id": "node-b"}],
            node_ids=["node-a", "node-b"],
            round_index=1,
            generation_kwargs={},
            judge_kwargs={},
        )
    )

    assert calls == [("generate", None)]
    sample = result["samples"][0]
    assert sample["average_rubric_judged_scores"] == {"node-a": 0.0, "node-b": 0.0}
    assert sample["active_after"][0].rubric_id == "generated-rubric"


def test_experience_bank_evidence_uses_average_judged_scores_and_does_not_write(tmp_path):
    bank_path = tmp_path / "siblings_rubric_bank.json"
    bank = ExperienceRubricBank(bank_path=bank_path, scope="siblings")
    assert (
        ExperienceRubricBank(scope="pc").update_prompt
        == PC_RUBRIC_EXPERIENCE_UPDATE_PROMPT
    )
    payload = {
        "messages": [{"role": "user", "content": "prompt"}],
        "average_rubric_judged_scores": {"node-a": 0.2, "node-b": 0.8},
        "generated": [
            {
                "rubric_id": "rubric-1",
                "direction": "positive",
                "weight": 1.0,
                "title": "Semantic Fix",
                "description": "Rewards a real fix.",
                "metadata": {},
                "scale": {str(i): str(i) for i in range(1, 6)},
            }
        ],
        "score_by_rubric": {"rubric-1": {"node-a": 0.0, "node-b": 1.0}},
        "gt_by_rubric": {
            "rubric-1": {
                "ground_truth_by_node": {"node-a": 0.0, "node-b": 1.0},
            }
        },
    }

    evidence = bank._build_instance_evidence(
        instance={"instance_id": "demo", "patch": "diff --git a/a b/a\n"},
        rubric_payloads=[payload],
    )

    assert evidence["rubric_attempts"][0]["average_rubric_judged_scores"] == [0.2, 0.8]
    assert "avg_scores" not in evidence["rubric_attempts"][0]

    result = asyncio.run(
        bank.update_after_instance(
            instance={"instance_id": "demo", "patch": ""},
            rubric_payloads=[],
            model_name="dummy",
            temperature=0.0,
            top_p=1.0,
            max_tokens=16,
        )
    )

    assert "before" in result and "after" in result
    assert not bank_path.exists()


def test_trajectory_search_experience_updates_are_written_by_runner(tmp_path):
    class FakeBank:
        def __init__(self, scope):
            self.scope = scope
            self.seen_payloads = None

        async def update_after_instance(self, **kwargs):
            self.seen_payloads = kwargs["rubric_payloads"]
            round_index = kwargs["rubric_payloads"][0]["round_index"]
            return {
                "before": [{"title": f"{self.scope}-before"}],
                "after": [{"title": f"{self.scope}-after"}],
                "groups": [
                    {
                        "round_index": round_index,
                        "before": [{"title": f"{self.scope}-before"}],
                        "actions": [{"action": "add"}],
                        "after": [{"title": f"{self.scope}-after"}],
                        "messages": [{"role": "user", "content": f"{self.scope}-update"}],
                    }
                ],
            }

    runner = TrajectorySearchRunner.__new__(TrajectorySearchRunner)
    runner.run_dir = tmp_path
    runner.rubrics_dir = tmp_path / "rubrics"
    runner.rubric_scopes = ("siblings", "pc")
    runner.experience_banks = {"siblings": FakeBank("siblings"), "pc": FakeBank("pc")}
    runner.instance = {"instance_id": "demo", "patch": ""}
    runner.rubric_model_name = "dummy"
    runner.rubric_model_kwargs = {}
    runner.search_config = SearchConfig(write_artifacts=True)

    runner._update_experience_banks(
        [
            {
                "scope": "siblings",
                "round_index": 1,
                "rubric_payload": {"rubric_list_id": "rubric-r001-s00"},
                "messages": [{"role": "user", "content": "siblings"}],
            },
            {
                "scope": "pc",
                "round_index": 2,
                "rubric_payload": {"rubric_list_id": "pc-rubric-r002-s00"},
                "messages": [{"role": "user", "content": "pc"}],
            },
        ]
    )

    assert (tmp_path / "siblings_rubric_bank.json").exists()
    assert (tmp_path / "pc_rubric_bank.json").exists()
    assert not (tmp_path / "rubric_bank.json").exists()
    assert json.loads((tmp_path / "siblings_rubric_bank.json").read_text())["after"] == [
        {"title": "siblings-after"}
    ]
    sibling_bank = tmp_path / "rubrics" / "siblings" / "rubric-r001-s00" / "rubric_bank.json"
    sibling_message = tmp_path / "rubrics" / "siblings" / "rubric-r001-s00" / "rubric_bank_message.json"
    pc_bank = tmp_path / "rubrics" / "pc" / "pc-rubric-r002-s00" / "rubric_bank.json"
    assert sibling_bank.exists()
    assert sibling_message.exists()
    assert pc_bank.exists()
    assert not (tmp_path / "rubrics" / "siblings" / "round_001" / "rubric_bank.json").exists()
    assert sorted(json.loads(sibling_bank.read_text())) == ["actions", "after", "before"]
    assert "messages" not in runner.experience_banks["siblings"].seen_payloads[0]
    assert runner.experience_banks["siblings"].seen_payloads[0]["generation_context"] == {}


def test_prompt_context_uses_compact_markdown():
    rendered = render_compact_markdown(
        {
            "question": "line one\nline two",
            "rubrics": [{"title": "Focused check", "weight": 1.0}],
        }
    )

    assert "- **question**:" in rendered
    assert "- **title**: Focused check" in rendered
    assert '"question"' not in rendered
    assert "\\n" not in rendered


def test_search_defaults_to_search_outputs_and_tmp_logs():
    parser = build_arg_parser()
    args = parser.parse_args([])

    assert args.output_root.name == "search_outputs"
    assert args.output_root.parent.name == "agent"
    assert args.beam_size == 2
    assert args.experience_bank.name == "nemotron_ultra"
    assert args.judge_temperature == 0.01
    assert args.score_tie_break is True
    assert args.stop_on_first_round_no_variance is False
    assert SearchConfig().judge_temperature == 0.01
    assert SearchConfig().score_tie_break is True
    assert SearchConfig().stop_on_first_round_no_variance is False
    assert parser.parse_args(
        ["--stop-on-first-round-no-variance"]
    ).stop_on_first_round_no_variance is True


def test_round_parent_selection_uses_same_depth_and_beam_size():
    runner = TrajectorySearchRunner.__new__(TrajectorySearchRunner)
    runner.search_config = SearchConfig(m=8, beam_size=2)
    runner.frontier_ids = ["a", "b", "older"]
    runner.nodes = {
        node_id: SearchNode(
            node_id=node_id,
            parent_id="root",
            round_index=1,
            depth=depth,
            session_id=node_id,
            status="frontier",
        )
        for node_id, depth in (("a", 2), ("b", 2), ("older", 1))
    }

    assert runner._round_parent_ids() == ["a", "b"]
    runner.search_config.beam_size = 1
    assert runner._round_parent_ids() == ["a"]


def test_beam_selection_generalizes_baseline_and_ablation_three_rules():
    def branch(node_id, score, regressed):
        return {"node_id": node_id, "score": score, "regressed_vs_parent": regressed}

    branches = [
        branch("valid-a", 0.8, False),
        branch("valid-b", 0.7, False),
        branch("valid-c", 0.6, False),
        branch("regressed-a", 0.9, True),
        branch("regressed-b", 0.5, True),
    ]
    baseline = _select_beam_branches(
        branches,
        beam_width=1,
        retention_limit=2,
        strategy="best",
    )
    assert [item["node_id"] for item in baseline] == ["valid-a", "valid-b"]

    paired = _select_beam_branches(
        branches,
        beam_width=2,
        retention_limit=4,
        strategy="best",
    )
    assert [item["node_id"] for item in paired] == ["valid-a", "valid-b"]

    four_valid = _select_beam_branches(
        [
            branch("valid-a", 0.8, False),
            branch("valid-b", 0.7, False),
            branch("valid-c", 0.6, False),
            branch("valid-d", 0.5, False),
            branch("valid-e", 0.4, False),
        ],
        beam_width=2,
        retention_limit=4,
        strategy="best",
    )
    assert [item["node_id"] for item in four_valid] == [
        "valid-a",
        "valid-b",
        "valid-c",
        "valid-d",
    ]

    one_valid = _select_beam_branches(
        [branches[0], branches[3], branches[4]],
        beam_width=2,
        retention_limit=4,
        strategy="best",
    )
    assert [item["node_id"] for item in one_valid] == ["valid-a", "regressed-a"]
    assert _select_beam_branches(
        [branches[3], branches[4]],
        beam_width=2,
        retention_limit=4,
        strategy="best",
    ) == []


def test_multi_parent_judging_context_maps_each_continuation_to_its_parent():
    runner = TrajectorySearchRunner.__new__(TrajectorySearchRunner)
    runner.search_config = SearchConfig(n=1)
    runner.system_prompt = "system"
    runner.task = "task"
    runner.rubric_model_name = "rubric-model"
    runner.judge_model_name = "judge-model"
    runner.rubric_model_kwargs = {}
    runner.judge_model_kwargs = {}
    runner.score_banks = {"siblings": None, "pc": None}
    runner.experience_banks = {"siblings": None, "pc": None}
    runner._node_judge_cache = {}
    captured = []

    async def fake_scope_judging(**kwargs):
        captured.append(kwargs)
        sample = {
            "sample_index": 0,
            "selected": False,
            "average_rubric_judged_scores": {
                branch["node_id"]: 0.5 for branch in kwargs["continuations"]
            },
        }
        return {"scope": kwargs["scope_spec"]["scope"], "samples": [sample], "valid_samples": [sample]}

    runner._run_rubric_scope_judging = fake_scope_judging
    parent_contexts = []
    branches = []
    for parent_index in (1, 2):
        parent_id = f"parent-{parent_index}"
        parent_contexts.append(
            {
                "node": SearchNode(
                    node_id=parent_id,
                    parent_id="root",
                    round_index=1,
                    depth=1,
                    session_id=parent_id,
                    status="frontier",
                ),
                "judge": {
                    "persistent_state": {"current_state": f"state-{parent_index}"},
                    "recent_segments": [
                        {"step_cards": [{"step_index": parent_index}], "segment_step_range": [0, 1]}
                    ],
                },
            }
        )
        for child_index in (1, 2):
            branches.append(
                {
                    "node_id": f"child-{parent_index}-{child_index}",
                    "parent_id": parent_id,
                    "recent_segments": [
                        {"step_cards": [{"step_index": child_index}], "segment_step_range": [1, 2]}
                    ],
                    "workspace_meta": {},
                    "result": {},
                }
            )

    asyncio.run(
        runner._prepare_round_judging(
            parent_contexts=parent_contexts,
            branch_records=branches,
            round_index=2,
            compare_parent=True,
        )
    )

    context = captured[0]["shared_context"]
    assert [item["continuations"] for item in context["previous_persistent_state"]["parent_states"]] == [
        "1-2",
        "3-4",
    ]
    assert [item["parent_index"] for item in captured[0]["continuations"]] == [1, 1, 2, 2]
    assert all(item.get("node_id") for item in captured[0]["continuations"])
    assert all(item.get("raw_continuation") for item in captured[0]["continuations"])
    rendered_state = trajectory_search._render_persistent_state(context["previous_persistent_state"])
    assert "### Persistent State of Parent 1 (Continuations 1-2)" in rendered_state
    assert "### Persistent State of Parent 2 (Continuations 3-4)" in rendered_state
    assert "parent_states" not in rendered_state
    rendered = trajectory_search._render_trajectory_segment(context["latest_agent_trajectory"])
    assert "Parent 1 (Continuations 1-2)" in rendered
    assert "Parent 2 (Continuations 3-4)" in rendered


def test_duplicate_rubrics_are_judged_and_weighted_once(monkeypatch):
    rubric = RubricRecord(
        rubric_id="same-rubric",
        title="Same",
        direction="positive",
        description="Same criterion.",
        scale={str(index): str(index) for index in range(1, 6)},
        weight=1.0,
        source_round=1,
    )
    other = RubricRecord(
        rubric_id="other-rubric",
        title="Other",
        direction="positive",
        description="Other criterion.",
        scale={str(index): str(index) for index in range(1, 6)},
        weight=1.0,
        source_round=1,
    )
    judged_rubrics = []

    async def fake_generate(**kwargs):
        return trajectory_search.RubricGenerationSample(
            sample_index=0,
            rubric_list_id="rubric-r001-s00",
            generated=[rubric, rubric, other],
            messages=[],
        )

    async def fake_score(**kwargs):
        judged_rubrics.extend(kwargs["rubrics"])
        return ([[]], [])

    monkeypatch.setattr(trajectory_search, "_generate_round_rubrics", fake_generate)
    monkeypatch.setattr(trajectory_search, "_score_round", fake_score)
    result = asyncio.run(
        _generate_and_score_rubric_batch(
            sample_count=1,
            round_index=1,
            generation_kwargs={},
            generation_prompt="generate",
            rubric_list_prefix="rubric",
            question={},
            shared_context={"previous_persistent_state": {}, "latest_agent_trajectory": None},
            continuations=[{"node_id": "node"}],
            extra_prompt_sections=[],
            judge_kwargs={},
        )
    )

    assert [item.rubric_id for item in judged_rubrics] == ["same-rubric", "other-rubric"]
    assert [item.rubric_id for item in result["sample_generated_rubrics"][0]] == [
        "same-rubric",
        "other-rubric",
    ]
    scores = {"node": {"same-rubric": 1.0, "other-rubric": 0.0}}
    assert _avg_scores_from_rubrics(
        node_ids=["node"],
        score_lookup_by_node=scores,
        rubrics=[rubric, rubric, other],
    ) == {"node": 0.5}
    assert _pc_avg_scores_from_rubrics(
        node_ids=["node"],
        score_lookup_by_node=scores,
        rubrics=[rubric, rubric, other],
    ) == {"node": 0.5}


def test_score_tie_break_only_rescores_tied_nodes(monkeypatch):
    tie_rubric = RubricRecord(
        rubric_id="tie-rubric",
        title="Tie breaker",
        direction="positive",
        description="Distinguishes the visible tied branches.",
        scale={str(index): str(index) for index in range(1, 6)},
        weight=1.0,
        source_round=1,
    )
    captured = {}

    async def fake_batch(**kwargs):
        captured.update(kwargs)
        return {
            "generated_samples": [
                SimpleNamespace(
                    sample_index=0,
                    rubric_list_id="rubric-r001-s00-tie",
                    generated=[tie_rubric],
                    messages=[{"role": "assistant", "content": "rubric"}],
                    format_errors=[],
                    terminal_error=None,
                )
            ],
            "sample_generated_rubrics": [[tie_rubric]],
            "generated_score_results": {
                0: (
                    [
                        [
                            {
                                "rubric_id": "tie-rubric",
                                "score_normalized": 0.2,
                                "judge_message": [],
                            }
                        ],
                        [
                            {
                                "rubric_id": "tie-rubric",
                                "score_normalized": 0.8,
                                "judge_message": [],
                            }
                        ],
                    ],
                    [],
                )
            },
        }

    monkeypatch.setattr(trajectory_search, "_generate_and_score_rubric_batch", fake_batch)
    initial_rubric = RubricRecord(
        rubric_id="initial",
        title="Initial",
        direction="positive",
        description="Initial score.",
        scale={str(index): str(index) for index in range(1, 6)},
        weight=1.0,
        source_round=1,
    )
    result = asyncio.run(
        trajectory_search._run_score_tie_break(
            scope="siblings",
            round_index=1,
            generation_prompt=(
                "## Task\nOriginal task.\n"
                "This is a multi-turn rubric generation setting.\n"
            ),
            rubric_list_prefix="rubric-r001-s00",
            judge_prompt="judge",
            question={},
            shared_context={},
            continuations=[
                {"node_id": "node-a"},
                {"node_id": "node-b"},
                {"node_id": "node-c"},
            ],
            extra_prompt_sections=[
                "active-rubrics",
                "## Retrieved Rubric Experiences:\nexperience-a",
            ],
            generation_kwargs={},
            judge_kwargs={},
            initial_rubrics=[initial_rubric],
            initial_score_by_rubric={
                "initial": {"node-a": 0.5, "node-b": 0.5, "node-c": 0.2}
            },
            initial_scores={"node-a": 0.5, "node-b": 0.5, "node-c": 0.2},
        )
    )

    assert [item["node_id"] for item in captured["continuations"]] == ["node-a", "node-b"]
    assert captured["extra_prompt_sections"][0] == "active-rubrics"
    assert captured["extra_prompt_sections"][1].startswith(
        "## Existing Tie-Producing Rubrics"
    )
    assert captured["extra_prompt_sections"][2].startswith(
        "## Retrieved Rubric Experiences:"
    )
    assert "currently highest-scoring branches" in captured["generation_prompt"]
    assert result["adjusted_scores"]["node-b"] > result["adjusted_scores"]["node-a"]
    assert result["adjusted_scores"]["node-c"] == 0.2


def test_rubric_artifact_payload_excludes_all_message_fields():
    runner = TrajectorySearchRunner.__new__(TrajectorySearchRunner)
    sample = {
        "rubric_list_id": "rubric-r001-s00",
        "round_index": 1,
        "generated": [],
        "format_errors": [],
        "terminal_error": None,
        "variance_by_rubric": {},
        "redundency_by_rubric": {},
        "judge_error_by_rubric": {},
        "reward_by_rubric": {},
        "judge_errors": [],
        "retrieved": [],
        "retrieve_messages": [{"role": "user", "content": "retrieve"}],
        "messages": [{"role": "user", "content": "generate"}],
        "score_by_rubric": {},
        "average_rubric_judged_scores": {},
        "generated_titles": [],
        "selected": False,
    }

    payload = runner._rubric_artifact_payload(sample, "root")

    assert not any("message" in key for key in payload)


def test_search_runner_flushes_final_root_experience_bank_summary(tmp_path):
    class FakeRootBank:
        def __init__(self, values):
            self.values = values

        def to_list(self):
            return list(self.values)

    banks = {
        "siblings": FakeRootBank([{"title": "siblings-initial"}]),
        "pc": FakeRootBank([{"title": "pc-initial"}]),
    }
    initial = {scope: bank.to_list() for scope, bank in banks.items()}

    banks["siblings"].values.append({"title": "siblings-final"})
    banks["pc"].values.append({"title": "pc-final"})

    _flush_experience_bank_summaries(
        run_root=tmp_path,
        experience_banks=banks,
        initial_snapshots=initial,
        write_artifacts=True,
    )

    siblings = json.loads((tmp_path / "siblings_rubric_bank.json").read_text())
    pc = json.loads((tmp_path / "pc_rubric_bank.json").read_text())

    assert siblings == {
        "before": [{"title": "siblings-initial"}],
        "after": [{"title": "siblings-initial"}, {"title": "siblings-final"}],
    }
    assert pc == {
        "before": [{"title": "pc-initial"}],
        "after": [{"title": "pc-initial"}, {"title": "pc-final"}],
    }
