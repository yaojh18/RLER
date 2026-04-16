import json
import io
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    import docker  # noqa: F401
except ModuleNotFoundError:
    docker_module = types.ModuleType("docker")
    docker_errors_module = types.ModuleType("docker.errors")

    class _DockerNotFound(Exception):
        pass

    class _DockerAPIError(Exception):
        pass

    docker_errors_module.NotFound = _DockerNotFound
    docker_errors_module.APIError = _DockerAPIError
    docker_module.errors = docker_errors_module
    docker_module.from_env = lambda: None
    sys.modules.setdefault("docker", docker_module)
    sys.modules.setdefault("docker.errors", docker_errors_module)

import dr_agent.utils as dr_utils
import litellm
import swe_agent.run.benchmarks.swebench as swebench_run
from dr_agent.utils import _resolve_vllm_base_command
from swe_agent.environments.docker import DockerEnvironment
from swe_agent.exceptions import Submitted
from swe_agent.models.litellm_model import LitellmModel
from swe_agent.run.run_swe_agent import (
    AGENT_ROOT,
    BackendResult,
    TeeStream,
    build_arg_parser,
    build_slim_trajectory,
    choose_gpus,
    evaluate_swebench_instance_patches,
    find_repo_root,
    parse_nvidia_smi_csv,
    run_harness_evaluation,
    run_swe_agent_backend,
)


def test_parse_nvidia_smi_csv():
    text = "0, NVIDIA H200, 143771, 10, 143761, 0\n1, NVIDIA H200, 143771, 20, 143751, 0\n"
    records = parse_nvidia_smi_csv(text)
    assert records[0]["index"] == 0
    assert records[1]["memory_free"] == 143751


def test_build_arg_parser_parses_instance_ids_into_flat_list():
    args = build_arg_parser().parse_args(["--instance-id", "a", "b,c"])
    assert args.instance_id == ["a", "b", "c"]


def test_choose_gpus_supports_auto_count_and_explicit_lists(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "swe_agent.run.run_swe_agent.query_gpu_inventory",
        lambda: [
            {"index": 0, "memory_free": 10},
            {"index": 1, "memory_free": 20},
            {"index": 2, "memory_free": 15},
        ],
    )
    assert choose_gpus("auto:2") == [1, 2]
    assert choose_gpus("0,2") == [0, 2]
    assert choose_gpus("none") == []


def test_litellm_bad_request_errors_abort_without_retry(monkeypatch: pytest.MonkeyPatch):
    model = LitellmModel(model_name="openai/test-model")
    calls = {"count": 0}

    def fail_once(*args, **kwargs):
        calls["count"] += 1
        raise litellm.exceptions.BadRequestError(
            message="context too long",
            model="openai/test-model",
            llm_provider="openai",
            response=None,
        )

    monkeypatch.setattr(model, "_query", fail_once)

    with pytest.raises(litellm.exceptions.BadRequestError):
        model.query([{"role": "user", "content": "hello"}])

    assert calls["count"] == 1


def test_tee_stream_strips_control_sequences_from_log_copy():
    console = io.StringIO()
    log = io.StringIO()
    tee = TeeStream(console, log)

    tee.write("\x1b[32mhello\x1b[0m\rworld\n")

    assert console.getvalue() == "\x1b[32mhello\x1b[0m\rworld\n"
    assert log.getvalue() == "helloworld\n"


def test_find_repo_root_points_at_current_repo():
    expected_root = Path(__file__).resolve().parents[2]
    assert find_repo_root(Path(__file__).resolve()) == expected_root
    assert AGENT_ROOT == expected_root / "agent"


def test_load_rebench_v2_alias(monkeypatch: pytest.MonkeyPatch):
    calls = {}

    def fake_load_dataset(path, split):
        calls["path"] = path
        calls["split"] = split
        return [{"instance_id": "demo__demo"}]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))

    records = swebench_run.load_swebench_instances("rebench_v2", "train")

    assert calls == {"path": "nebius/SWE-rebench-V2", "split": "train"}
    assert records == [{"instance_id": "demo__demo"}]


def test_run_harness_evaluation_uses_rebench_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    patch_path = run_dir / "model_patch.json"
    patch_path.write_text(
        json.dumps(
            {
                "demo__demo-1": {
                    "model_name_or_path": "Qwen/Qwen3.5-9B",
                    "instance_id": "demo__demo-1",
                    "model_patch": "diff --git a/a b/a\n",
                }
            }
        )
    )
    log_path = tmp_path / "driver.log"
    log_path.write_text("")
    result = BackendResult(
        benchmark_name="rebench_v2",
        split="train",
        backend="vllm",
        model_name="Qwen/Qwen3.5-9B",
        instance_id="demo__demo-1",
        run_dir=str(run_dir),
        raw_trajectory_path=None,
        slim_trajectory_path=None,
        patch_path=str(patch_path),
        log_path=str(log_path),
        evaluation_result_path=None,
        exit_status="submitted",
        submission_chars=12,
        prediction_chars=18,
        evaluation_completed=False,
        resolved=None,
        run_id=None,
        error=None,
        harness_namespace=None,
        swebench_command=None,
        evaluation_command=None,
        vllm_command=None,
        vllm_log_path=None,
        gpu_id=None,
    )

    monkeypatch.setattr(
        "swe_agent.run.run_swe_agent.evaluate_rebench_prediction",
        lambda **kwargs: {
            "instance_id": kwargs["instance"]["instance_id"],
            "resolved": True,
            "passed_actual": ["tests::ok"],
            "failed_actual": [],
            "passed_expected": ["tests::ok"],
            "log_path": str(run_dir / "rebench_eval" / "demo__demo-1.log"),
        },
    )

    updated = run_harness_evaluation(
        results=[result],
        dataset_name="nebius/SWE-rebench-V2",
        timeout=60,
        max_workers=1,
        instances_by_id={
            "demo__demo-1": {
                "instance_id": "demo__demo-1",
                "repo": "demo/demo",
                "image_name": "docker.io/swerebenchv2/demo-demo:1",
                "install_config": {"test_cmd": ["pytest"], "log_parser": "parse_log_pytest"},
                "PASS_TO_PASS": [],
                "FAIL_TO_PASS": [],
                "test_patch": "diff --git a/tests b/tests\n",
            }
        },
    )

    assert updated[0].resolved is True
    assert updated[0].evaluation_completed is True
    assert Path(updated[0].evaluation_result_path).exists()
    assert (run_dir / "rebench_evaluation.json").exists()


def test_evaluate_swebench_instance_patches_dispatches_rebench_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured = {}

    def fake_eval_backend(**kwargs):
        captured.update(kwargs)
        return {"node-1": 1.0}

    monkeypatch.setattr("swe_agent.run.run_swe_agent.evaluate_rebench_instance_patches_backend", fake_eval_backend)

    rewards = evaluate_swebench_instance_patches(
        instance={
            "instance_id": "demo__demo-1",
            "image_name": "docker.io/swerebenchv2/demo-demo:1",
            "install_config": {"test_cmd": ["pytest"], "log_parser": "parse_log_pytest"},
        },
        patches_by_key={"node-1": "diff --git a/a b/a\n"},
        model_name="Qwen/Qwen3.5-9B",
        max_workers=2,
        namespace=None,
        work_dir=tmp_path,
    )

    assert rewards == {"node-1": 1.0}
    assert captured["max_workers"] == 2
    assert captured["timeout"] == 900


def test_resolve_vllm_serve_command_prefers_interpreter_adjacent_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    real_python_dir = tmp_path / "real"
    real_python_dir.mkdir()
    real_python_path = real_python_dir / "python3.10"
    real_python_path.write_text("")

    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    python_path = bin_dir / "python"
    python_path.symlink_to(real_python_path)
    vllm_path = bin_dir / "vllm"
    vllm_path.write_text("")
    monkeypatch.setattr(dr_utils.sys, "executable", str(python_path))
    monkeypatch.setattr(dr_utils.shutil, "which", lambda _: None)

    command = _resolve_vllm_base_command()

    assert command[0] == str(vllm_path.resolve())
    assert command[1] == "serve"


def test_resolve_vllm_serve_command_falls_back_to_project_venv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    project_root = tmp_path / "agent"
    venv_bin = project_root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    project_vllm = venv_bin / "vllm"
    project_vllm.write_text("")
    module_file = project_root / "dr_agent" / "utils.py"
    module_file.parent.mkdir(parents=True)
    module_file.write_text("")

    monkeypatch.setattr(dr_utils, "__file__", str(module_file))
    monkeypatch.setattr(dr_utils.sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(dr_utils.shutil, "which", lambda _: None)

    command = _resolve_vllm_base_command()

    assert command[0] == str(project_vllm.resolve())
    assert command[1] == "serve"


def test_launch_vllm_server_handle_uses_vllm_generation_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    commands: list[list[str]] = []
    envs: list[dict[str, str]] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"pong"}}]}'

    class FakeProcess:
        pid = 123

        @staticmethod
        def poll():
            return None

    vllm_bin = tmp_path / "bin"
    vllm_bin.mkdir(parents=True)
    vllm_path = vllm_bin / "vllm"
    vllm_path.write_text("")

    monkeypatch.setattr(dr_utils, "_resolve_vllm_base_command", lambda: [str(vllm_path), "serve"])
    monkeypatch.setattr(dr_utils, "check_port", lambda port: True)
    monkeypatch.setattr(dr_utils, "time", types.SimpleNamespace(time=lambda: 0, sleep=lambda _: None))
    monkeypatch.setattr(dr_utils.urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse())

    def fake_popen(cmd, **kwargs):
        commands.append(cmd)
        envs.append(kwargs["env"])
        return FakeProcess()

    monkeypatch.setattr(dr_utils.subprocess, "Popen", fake_popen)

    handle = dr_utils.launch_vllm_server_handle("Qwen/Qwen3-8B", 8011, gpu_id=0)

    assert handle.command == commands[0]
    assert "--generation-config" in commands[0]
    assert "vllm" in commands[0]
    assert "--gpu-memory-utilization" in commands[0]
    assert "0.9" in commands[0]
    assert str(vllm_bin) in envs[0]["PATH"].split(":")


def test_launch_vllm_server_handle_supports_tensor_parallel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    commands: list[list[str]] = []
    envs: list[dict[str, str]] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"pong"}}]}'

    class FakeProcess:
        pid = 456

        @staticmethod
        def poll():
            return None

    vllm_bin = tmp_path / "bin"
    vllm_bin.mkdir(parents=True)
    vllm_path = vllm_bin / "vllm"
    vllm_path.write_text("")

    monkeypatch.setattr(dr_utils, "_resolve_vllm_base_command", lambda: [str(vllm_path), "serve"])
    monkeypatch.setattr(dr_utils, "check_port", lambda port: True)
    monkeypatch.setattr(dr_utils, "time", types.SimpleNamespace(time=lambda: 0, sleep=lambda _: None))
    monkeypatch.setattr(dr_utils.urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse())

    def fake_popen(cmd, **kwargs):
        commands.append(cmd)
        envs.append(kwargs["env"])
        return FakeProcess()

    monkeypatch.setattr(dr_utils.subprocess, "Popen", fake_popen)

    handle = dr_utils.launch_vllm_server_handle(
        "Qwen/Qwen3.5-9B",
        8012,
        gpu_id=0,
        gpu_ids=[0, 1],
        max_model_len=81920,
        gpu_memory_utilization=0.45,
    )

    assert handle.gpu_ids == [0, 1]
    assert "--tensor-parallel-size" in commands[0]
    assert "2" in commands[0]
    assert "--gpu-memory-utilization" in commands[0]
    assert "0.45" in commands[0]
    assert envs[0]["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_launch_vllm_server_handle_supports_served_model_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    commands: list[list[str]] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"pong"}}]}'

    class FakeProcess:
        pid = 789

        @staticmethod
        def poll():
            return None

    vllm_bin = tmp_path / "bin"
    vllm_bin.mkdir(parents=True)
    vllm_path = vllm_bin / "vllm"
    vllm_path.write_text("")

    monkeypatch.setattr(dr_utils, "_resolve_vllm_base_command", lambda: [str(vllm_path), "serve"])
    monkeypatch.setattr(dr_utils, "check_port", lambda port: True)
    monkeypatch.setattr(dr_utils, "time", types.SimpleNamespace(time=lambda: 0, sleep=lambda _: None))
    monkeypatch.setattr(dr_utils.urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse())

    def fake_popen(cmd, **kwargs):
        commands.append(cmd)
        return FakeProcess()

    monkeypatch.setattr(dr_utils.subprocess, "Popen", fake_popen)

    handle = dr_utils.launch_vllm_server_handle(
        "/models/Qwen3.5-9B",
        8013,
        gpu_id=0,
        served_model_name="Qwen/Qwen3.5-9B",
    )

    assert "--served-model-name" in commands[0]
    assert "Qwen/Qwen3.5-9B" in commands[0]
    assert handle.model_name == "Qwen/Qwen3.5-9B"


def test_resolve_swebench_image_prefers_official_registry(monkeypatch: pytest.MonkeyPatch):
    swebench_run._IMAGE_RESOLUTION_CACHE.clear()
    swebench_run._PREPARED_IMAGES.clear()
    build_calls: list[str] = []

    class FakeSpec:
        def __init__(self, namespace: str | None):
            self.namespace = namespace
            self.instance_image_tag = "latest"
            self.env_image_tag = "latest"
            if namespace == swebench_run.OFFICIAL_IMAGE_NAMESPACE:
                self.instance_image_key = "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"
            else:
                self.instance_image_key = "sweb.eval.x86_64.astropy__astropy-12907:latest"

    monkeypatch.setattr(
        swebench_run,
        "make_test_spec",
        lambda instance, namespace=None: FakeSpec(namespace),
    )
    monkeypatch.setattr(swebench_run, "_registry_image_exists", lambda image_name: image_name.startswith("swebench/"))
    monkeypatch.setattr(swebench_run, "build_env_images", lambda *args, **kwargs: build_calls.append("env"))
    monkeypatch.setattr(swebench_run, "build_instance_image", lambda *args, **kwargs: build_calls.append("instance"))

    image_name, namespace = swebench_run.resolve_swebench_image({"instance_id": "astropy__astropy-12907"})

    assert image_name == "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"
    assert namespace == swebench_run.OFFICIAL_IMAGE_NAMESPACE
    assert build_calls == []


def test_resolve_swebench_image_builds_local_when_registry_missing(monkeypatch: pytest.MonkeyPatch):
    swebench_run._IMAGE_RESOLUTION_CACHE.clear()
    swebench_run._PREPARED_IMAGES.clear()
    build_calls: list[str] = []

    class FakeSpec:
        def __init__(self, namespace: str | None):
            self.namespace = namespace
            self.instance_image_tag = "latest"
            self.env_image_tag = "latest"
            if namespace == swebench_run.OFFICIAL_IMAGE_NAMESPACE:
                self.instance_image_key = "swebench/sweb.eval.x86_64.missing_1776_instance:latest"
            else:
                self.instance_image_key = "sweb.eval.x86_64.missing__instance:latest"

    class FakeDockerClient:
        def close(self):
            return None

    monkeypatch.setattr(
        swebench_run,
        "make_test_spec",
        lambda instance, namespace=None: FakeSpec(namespace),
    )
    monkeypatch.setattr(swebench_run, "_registry_image_exists", lambda image_name: False)
    monkeypatch.setattr(swebench_run.docker, "from_env", lambda: FakeDockerClient())
    monkeypatch.setattr(swebench_run, "build_env_images", lambda *args, **kwargs: build_calls.append("env"))
    monkeypatch.setattr(swebench_run, "build_instance_image", lambda *args, **kwargs: build_calls.append("instance"))

    image_name, namespace = swebench_run.resolve_swebench_image({"instance_id": "missing__instance"})

    assert image_name == "sweb.eval.x86_64.missing__instance:latest"
    assert namespace is None
    assert build_calls == ["env", "instance"]


def test_resolve_swebench_image_with_dataset_image_and_missing_version_skips_namespace_inference(
    monkeypatch: pytest.MonkeyPatch,
):
    swebench_run._IMAGE_RESOLUTION_CACHE.clear()
    instance = {
        "instance_id": "axios__axios-5831",
        "repo": "axios/axios",
        "version": None,
        "image_name": "docker.io/swerebenchv2/axios-axios:5831-3f53eb6",
    }

    def fake_make_test_spec(*args, **kwargs):
        raise KeyError(None)

    monkeypatch.setattr(swebench_run, "make_test_spec", fake_make_test_spec)

    image_name, namespace = swebench_run.resolve_swebench_image(instance)

    assert image_name == "docker.io/swerebenchv2/axios-axios:5831-3f53eb6"
    assert namespace is None


def test_resolve_swebench_image_builds_rebench_image_when_registry_missing(monkeypatch: pytest.MonkeyPatch):
    swebench_run._IMAGE_RESOLUTION_CACHE.clear()
    swebench_run._PREPARED_IMAGES.clear()
    build_calls: list[str] = []
    instance = {
        "instance_id": "pvlib__pvlib-python-278",
        "repo": "pvlib/pvlib-python",
        "base_commit": "abc123",
        "image_name": "docker.io/swerebenchv2/pvlib-pvlib-python:278-6a41299",
        "install_config": {
            "base_image_name": "python_base_37",
            "install": ["pip install -e ."],
        },
    }

    monkeypatch.setattr(swebench_run, "_local_image_exists", lambda image_name: False)
    monkeypatch.setattr(swebench_run, "_registry_image_exists", lambda image_name: False)
    monkeypatch.setattr(swebench_run, "_build_rebench_base_image", lambda image_name: build_calls.append(f"base:{image_name}"))
    monkeypatch.setattr(swebench_run.subprocess, "run", lambda command, check=True: build_calls.append(f"instance:{command[-2]}"))

    image_name, namespace = swebench_run.resolve_swebench_image(instance)

    assert image_name == instance["image_name"]
    assert namespace is None
    assert build_calls[0] == "base:python_base_37"
    assert build_calls[1].startswith("instance:docker.io/swerebenchv2/pvlib-pvlib-python:278-6a41299")


def test_resolve_rebench_base_dockerfile_maps_python_alias():
    dockerfile = swebench_run._resolve_rebench_base_dockerfile("python_base_37")

    assert dockerfile.name == "Dockerfile_python_3.7"


def test_process_instance_cleans_up_environment_after_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cleaned = {"value": False}

    class FakeEnv:
        def cleanup(self):
            cleaned["value"] = True

    class FakeModel:
        config = SimpleNamespace(model_name="openai/test-model")

    class FakeAgent:
        def __init__(self, model, env, **kwargs):
            self.env = env

        def run(self, task):
            raise RuntimeError(f"boom: {task}")

        def save(self, path, *extra_dicts):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"extra": extra_dicts}))

    progress = SimpleNamespace(
        on_instance_start=lambda *args, **kwargs: None,
        update_instance_status=lambda *args, **kwargs: None,
        on_instance_end=lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(swebench_run, "get_model", lambda config=None: FakeModel())
    monkeypatch.setattr(swebench_run, "get_sb_environment", lambda config, instance: FakeEnv())
    monkeypatch.setattr(swebench_run, "ProgressTrackingAgent", FakeAgent)
    monkeypatch.setattr(swebench_run, "update_preds_file", lambda *args, **kwargs: None)

    swebench_run.process_instance(
        {"instance_id": "sympy__sympy-20590", "problem_statement": "task"},
        tmp_path,
        {"model": {}, "agent": {}},
        progress,
    )

    assert cleaned["value"] is True


def test_build_slim_trajectory_for_mini_swe_agent_messages():
    raw_traj = {
        "trajectory_format": "mini-swe-agent-1.1",
        "messages": [
            {"role": "system", "content": "system prompt"},
            {
                "role": "assistant",
                "content": "THOUGHT\n```mswea_bash_command\npwd\n```",
                "extra": {"actions": [{"command": "pwd"}]},
            },
            {
                "role": "user",
                "content": "<output>/testbed</output>",
                "extra": {"raw_output": "/testbed\n"},
            },
        ],
    }
    slim = build_slim_trajectory(raw_traj, model_name="gemini/gemini-3-pro-preview")
    assert slim["parser"] == "parse_mini_swe_message"
    assert slim["messages"] == [
        {
            "index": 1,
            "role": "assistant",
            "message": "THOUGHT\n```mswea_bash_command\npwd\n```",
            "tool_calls": [{"command": "pwd"}],
            "parser_model": "gemini/gemini-3-pro-preview",
        },
        {
            "index": 2,
            "role": "user",
            "message": "/testbed\n",
            "tool_calls": [],
            "parser_model": "gemini/gemini-3-pro-preview",
        },
    ]


def test_run_harness_evaluations_batches_instances_by_namespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import swebench.harness.reporting as swebench_reporting
    import swebench.harness.run_evaluation as swebench_run_evaluation

    run_root = tmp_path / "verified_test_model"
    log_path = tmp_path / "logs" / "20260317-070000.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("")
    results: list[BackendResult] = []
    for instance_id in ["astropy__astropy-12907", "django__django-10097"]:
        run_dir = run_root / instance_id / "20260317-070000"
        run_dir.mkdir(parents=True)
        results.append(
            BackendResult(
                benchmark_name="verified",
                split="test",
                backend="vllm",
                model_name="openai/Qwen/Qwen3-8B",
                instance_id=instance_id,
                run_dir=str(run_dir),
                raw_trajectory_path=None,
                slim_trajectory_path=None,
                patch_path=None,
                log_path=str(log_path),
                evaluation_result_path=None,
                exit_status="Submitted",
                submission_chars=10,
                prediction_chars=10,
                evaluation_completed=False,
                resolved=None,
                run_id=None,
                error=None,
                harness_namespace="swebench",
                swebench_command=None,
                evaluation_command=None,
                vllm_command=None,
                vllm_log_path=None,
                gpu_id=None,
            )
        )

    predictions_path = tmp_path / "preds.json"
    predictions_path.write_text(
        json.dumps(
            {
                result.instance_id: {
                    "model_name_or_path": result.model_name,
                    "instance_id": result.instance_id,
                    "model_patch": "diff --git a/foo b/foo\n",
                }
                for result in results
            }
        )
    )

    calls: list[tuple[list[str], int, str | None]] = []
    summary_report = tmp_path / "raw-summary.json"

    def fake_main(
        *,
        dataset_name,
        split,
        instance_ids,
        predictions_path,
        max_workers,
        force_rebuild,
        cache_level,
        clean,
        open_file_limit,
        run_id,
        timeout,
        namespace,
        rewrite_reports,
        modal,
        report_dir,
        ):
        calls.append((list(instance_ids), max_workers, namespace))
        for instance_id in instance_ids:
            report_path = (
                swebench_run_evaluation.RUN_EVALUATION_LOG_DIR
                / run_id
                / "openai__Qwen__Qwen3-8B"
                / instance_id
                / "report.json"
            )
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps({instance_id: {"resolved": instance_id.startswith("astropy")}}))
        summary_report.write_text(json.dumps({"run_id": run_id}))
        return str(summary_report)

    monkeypatch.setattr(swebench_run_evaluation, "main", fake_main)
    monkeypatch.setattr(swebench_run_evaluation, "RUN_EVALUATION_LOG_DIR", tmp_path / "unused_eval_root")
    monkeypatch.setattr(swebench_reporting, "RUN_EVALUATION_LOG_DIR", tmp_path / "unused_report_root")

    updated = run_harness_evaluation(
        results=results,
        dataset_name="princeton-nlp/SWE-Bench_Verified",
        timeout=900,
        max_workers=2,
    )

    assert calls == [(["astropy__astropy-12907", "django__django-10097"], 2, "swebench")]
    assert not summary_report.exists()
    evaluations = {
        result.instance_id: json.loads((Path(result.run_dir) / "evaluation.json").read_text())
        for result in updated
    }
    assert evaluations["astropy__astropy-12907"]["resolved_ids"] == ["astropy__astropy-12907"]
    assert evaluations["django__django-10097"]["unresolved_ids"] == ["django__django-10097"]


def test_run_harness_evaluations_writes_error_report_when_harness_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import swebench.harness.reporting as swebench_reporting
    import swebench.harness.run_evaluation as swebench_run_evaluation

    run_dir = tmp_path / "verified_test_model" / "astropy__astropy-12907" / "20260318-000000"
    run_dir.mkdir(parents=True)
    log_path = tmp_path / "logs" / "20260318-000000.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("")
    patch_path = run_dir / "model_patch.json"
    patch_path.write_text(
        json.dumps(
            {
                "astropy__astropy-12907": {
                    "model_name_or_path": "openai/Qwen/Qwen3-8B",
                    "instance_id": "astropy__astropy-12907",
                    "model_patch": "diff --git a/foo b/foo\n",
                }
            }
        )
    )
    monkeypatch.setattr(swebench_run_evaluation, "RUN_EVALUATION_LOG_DIR", tmp_path / "eval_root")
    monkeypatch.setattr(swebench_reporting, "RUN_EVALUATION_LOG_DIR", tmp_path / "report_root")
    monkeypatch.setattr(swebench_run_evaluation, "main", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    updated = run_harness_evaluation(
        results=[
            BackendResult(
                benchmark_name="verified",
                split="test",
                backend="vllm",
                model_name="openai/Qwen/Qwen3-8B",
                instance_id="astropy__astropy-12907",
                run_dir=str(run_dir),
                raw_trajectory_path=None,
                slim_trajectory_path=None,
                patch_path=str(patch_path),
                log_path=str(log_path),
                evaluation_result_path=None,
                exit_status="Submitted",
                submission_chars=10,
                prediction_chars=10,
                evaluation_completed=False,
                resolved=None,
                run_id=None,
                error=None,
                harness_namespace="swebench",
                swebench_command=None,
                evaluation_command=None,
                vllm_command=None,
                vllm_log_path=None,
                gpu_id=None,
            )
        ],
        dataset_name="princeton-nlp/SWE-Bench_Verified",
        timeout=900,
        max_workers=1,
    )

    evaluation = json.loads((run_dir / "evaluation.json").read_text())
    assert evaluation["error_ids"] == ["astropy__astropy-12907"]
    assert updated[0].error == "boom"


def test_evaluate_swebench_instance_patches_deduplicates_patches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls = {"build_env_images": 0, "build_instance_image": 0, "run_instance": []}

    monkeypatch.setattr("swe_agent.run.run_swe_agent.swebench_run_evaluation.docker.from_env", lambda: object())
    monkeypatch.setattr(
        "swe_agent.run.run_swe_agent.swebench_run_evaluation.make_test_spec",
        lambda instance, namespace=None: SimpleNamespace(instance_id=instance["instance_id"]),
    )
    monkeypatch.setattr(
        "swe_agent.run.run_swe_agent.swebench_run_evaluation.build_env_images",
        lambda client, instances, force_rebuild, max_workers: calls.__setitem__("build_env_images", calls["build_env_images"] + 1),
    )
    monkeypatch.setattr(
        "swe_agent.run.run_swe_agent.build_instance_image",
        lambda test_spec, client, logger=None, nocache=False: calls.__setitem__("build_instance_image", calls["build_instance_image"] + 1),
    )

    def fake_run_instance(test_spec, pred, rm_image, force_rebuild, client, run_id, timeout, rewrite_reports):
        calls["run_instance"].append(pred["model_patch"])
        report_path = (
            Path(run_module.swebench_run_evaluation.RUN_EVALUATION_LOG_DIR)
            / run_id
            / pred["model_name_or_path"].replace("/", "__")
            / test_spec.instance_id
            / "report.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({test_spec.instance_id: {"resolved": pred["model_patch"] == "patch-a"}}),
            encoding="utf-8",
        )

    monkeypatch.setattr("swe_agent.run.run_swe_agent.swebench_run_evaluation.run_instance", fake_run_instance)
    monkeypatch.setattr(
        "swe_agent.run.run_swe_agent.swebench_run_evaluation.run_threadpool",
        lambda func, payloads, max_workers: [func(*payload) for payload in payloads],
    )

    from swe_agent.run import run_swe_agent as run_module

    rewards = evaluate_swebench_instance_patches(
        instance={"instance_id": "demo__demo"},
        patches_by_key={
            "node-a": "patch-a",
            "node-b": "patch-a",
            "node-c": "patch-c",
            "node-empty": "",
        },
        model_name="openai/fake",
        max_workers=4,
        namespace=None,
        work_dir=tmp_path,
    )

    assert rewards == {
        "node-a": 1.0,
        "node-b": 1.0,
        "node-c": 0.0,
        "node-empty": 0.0,
    }
    assert calls["build_env_images"] == 1
    assert calls["build_instance_image"] == 1
    assert calls["run_instance"] == ["patch-a", "patch-c"]


def test_docker_environment_rejects_empty_submission():
    env = DockerEnvironment.__new__(DockerEnvironment)
    env._owns_container = False
    env.container_id = None
    output = {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n", "returncode": 0}

    env._check_finished(output)

    assert output["returncode"] == 1
    assert "Submission rejected" in output["output"]


def test_docker_environment_accepts_non_empty_submission():
    env = DockerEnvironment.__new__(DockerEnvironment)
    env._owns_container = False
    env.container_id = None

    with pytest.raises(Submitted):
        env._check_finished({"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndiff --git a/foo b/foo\n", "returncode": 0})


def test_run_swe_agent_backend_evaluation_only_uses_existing_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from swe_agent.run import run_swe_agent as run_module

    instance_id = "astropy__astropy-12907"
    timestamp = "20260318-123456"
    run_dir = tmp_path / "verified_test_gemini__gemini-3.1-pro-preview" / instance_id / timestamp
    run_dir.mkdir(parents=True)
    (run_dir / "raw_traj.json").write_text(
        json.dumps({"info": {"exit_status": "Submitted", "submission": "diff --git a/foo b/foo\n"}, "messages": []})
    )
    (run_dir / "model_patch.json").write_text(
        json.dumps(
            {
                instance_id: {
                    "model_name_or_path": "gemini/gemini-3.1-pro-preview",
                    "instance_id": instance_id,
                    "model_patch": "diff --git a/foo b/foo\n",
                }
            }
        )
    )

    captured: dict[str, object] = {}

    monkeypatch.setattr(run_module, "load_swebench_instances", lambda subset, split: [{"instance_id": instance_id}])
    monkeypatch.setattr(run_module, "get_swebench_harness_namespace", lambda instance: "swebench")

    def fake_run_harness_evaluation(*, results, dataset_name, timeout, max_workers, instances_by_id=None):
        captured["results"] = results
        captured["dataset_name"] = dataset_name
        captured["timeout"] = timeout
        captured["max_workers"] = max_workers
        captured["instances_by_id"] = instances_by_id
        return results

    monkeypatch.setattr(run_module, "run_harness_evaluation", fake_run_harness_evaluation)

    args = SimpleNamespace(
        subset="verified",
        split="test",
        output_root=tmp_path,
        workers=2,
        eval_timeout=900,
        openai_model="gemini/gemini-3.1-pro-preview",
        vllm_model="Qwen/Qwen3-8B",
        _evaluation_only=timestamp,
        _run_log_path=tmp_path / "logs" / f"{timestamp}.log",
    )

    results = run_swe_agent_backend(args, "openai", None)

    assert len(results) == 1
    assert (run_dir / "messages.json").exists()
    assert captured["dataset_name"] == "princeton-nlp/SWE-Bench_Verified"
    assert captured["timeout"] == 900
    assert captured["max_workers"] == 2
    assert captured["instances_by_id"] == {instance_id: {"instance_id": instance_id}}
    assert captured["results"][0].instance_id == instance_id
