#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import contextlib
import hashlib
import json
import logging
import os
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Sequence, TextIO
import yaml
from swebench.harness import run_evaluation as swebench_run_evaluation
from swebench.harness.constants import LOG_REPORT, LOG_TEST_OUTPUT
from swebench.harness.docker_build import build_instance_image

os.environ.setdefault("LITELLM_LOG", "ERROR")

from swe_agent.run.benchmarks.container_runtime import select_container_backend
from swe_agent.run.benchmarks.deepswe_eval import (
    evaluate_deepswe_instances,
    is_deepswe_dataset_name,
    is_deepswe_instance,
)
from swe_agent.run.benchmarks.rebench_eval import (
    evaluate_rebench_instance as evaluate_rebench_prediction,
    is_rebench_dataset_name,
    is_rebench_instance,
)
from swe_agent.run.benchmarks.r2egym_eval import (
    evaluate_r2egym_instances,
    is_r2egym_dataset_name,
    is_r2egym_instance,
)
from swe_agent.run.benchmarks.swebench import (
    DATASET_MAPPING,
    build_swebench_config,
    load_swebench_instances_by_id,
    load_swebench_instances,
    load_swebench_instances_slice,
    run_swebench_instances,
)
from swe_agent.run.benchmarks.swebench_pro_eval import (
    evaluate_swebench_pro_instances,
    is_swebench_pro_dataset_name,
    is_swebench_pro_instance,
)
from swe_agent.run.benchmarks.swebench_singularity_eval import evaluate_swebench_instances_singularity
from swe_agent.models.utils.actions_text import format_observation_messages


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "agent").is_dir() and ((candidate / "slime").is_dir() or (candidate / "rl").is_dir()):
            return candidate
    raise RuntimeError(f"Could not find repository root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
AGENT_ROOT = REPO_ROOT / "agent"
DEFAULT_SUBSET = "verified"
DEFAULT_SPLIT = "test"
DEFAULT_OUTPUT_ROOT = AGENT_ROOT / "outputs"
DEFAULT_LOG_ROOT = Path(os.environ.get("RUN_SWE_AGENT_LOG_ROOT", AGENT_ROOT / "logs"))
DEFAULT_SERVE_MODEL = "Qwen/Qwen3-8B"
DEFAULT_OPENAI_MODEL = "gemini/gemini-3-pro-preview"
DEFAULT_MODEL_CLASS = "route_textbased"
DEFAULT_MAX_MODEL_LEN = 128000
DEFAULT_STEP_LIMIT = 250
DEFAULT_COMPLETION_MAX_TOKENS = 8096
EMPTY_REWARD = -0.2
ERROR_REWARD = -0.5
SUPPORTED_REWARD_KINDS = {"hard", "soft", "joint", "f2p_only"}
DEFAULT_VLLM_PORT = 30000
DEFAULT_ENV_TIMEOUT = 300
DEFAULT_PULL_TIMEOUT = 600
DEFAULT_EVAL_TIMEOUT = 900
SWE_AGENT_TEXTBASED_CONFIG = AGENT_ROOT / "swe_agent" / "config" / "benchmarks" / "swebench_backticks.yaml"
SLIME_SERVICE_NAME = "slime"
VLLM_SERVICE_NAME = "vllm"
SLIME_API_BASE = os.environ.get("SEARCH_SWE_SLIME_API_BASE", "http://127.0.0.1:8021")
SLIME_API_KEY = os.environ.get("SEARCH_SWE_SLIME_API_KEY", "EMPTY")
DEFAULT_SGLANG_IMAGE = "slimerl/slime:qwen35-route-fixed-20260416-v1"
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


@lru_cache(maxsize=1)
def _swe_agent_observation_template() -> str:
    config = yaml.safe_load(SWE_AGENT_TEXTBASED_CONFIG.read_text(encoding="utf-8"))
    return str(config["model"]["observation_template"])


def format_swe_agent_observation(
    output: str,
    *,
    returncode: int | None = None,
    exception_info: str | None = None,
) -> str:
    return format_observation_messages(
        [
            {
                "output": output,
                "returncode": returncode,
                "exception_info": exception_info,
            }
        ],
        observation_template=_swe_agent_observation_template(),
    )[0]["content"]


@dataclass(frozen=True)
class EvaluationRewardConfig:
    kind: str = "joint"
    joint_alpha: float = 1.0
    all_pass_reward: float = 2.0
    fallback_patch_penalty: float = 1.0
    no_action_patch_penalty: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in SUPPORTED_REWARD_KINDS:
            raise ValueError(
                f"Unsupported reward kind {self.kind!r}; choose from {sorted(SUPPORTED_REWARD_KINDS)}"
            )


def default_evaluation_reward_config() -> EvaluationRewardConfig:
    kind = os.environ.get("RLER_REWARD_KIND", "joint")
    return EvaluationRewardConfig(
        kind=kind.strip().lower() or "joint",
        joint_alpha=float(os.environ.get("RLER_JOINT_ALPHA", "1.0")),
        all_pass_reward=float(os.environ.get("RLER_ALL_PASS_REWARD", "2.0")),
    )


class TeeStream:
    def __init__(self, console_stream: TextIO, log_stream: TextIO):
        self._console_stream = console_stream
        self._log_stream = log_stream

    def write(self, data: str) -> int:
        self._console_stream.write(data)
        self._console_stream.flush()
        self._log_stream.write(re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", data).replace("\r", ""))
        self._log_stream.flush()
        return len(data)

    def flush(self) -> None:
        self._console_stream.flush()
        self._log_stream.flush()

class ParseInstanceIds(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parsed = list(getattr(namespace, self.dest) or [])
        for value in values:
            parsed.extend(part for part in value.split(",") if part)
        setattr(namespace, self.dest, parsed)


def _resolve_model_name(args: argparse.Namespace) -> str:
    if args.backend == "vllm":
        return args.vllm_model
    if args.backend == "sglang":
        return args.sglang_model
    return args.openai_model


def select_instances(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.instance_id:
        return load_swebench_instances_by_id(args.subset, args.split, list(args.instance_id))
    instances = load_swebench_instances_slice(args.subset, args.split, args.offset, args.limit)
    if not instances:
        raise RuntimeError(f"No instances selected from {args.subset}/{args.split} at offset={args.offset}")
    return instances


def query_gpu_inventory() -> list[dict[str, int | str]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    records: list[dict[str, int | str]] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            raise ValueError(f"Unexpected nvidia-smi line: {raw_line!r}")
        records.append({
            "index": int(parts[0]),
            "name": parts[1],
            "memory_total": int(parts[2]),
            "memory_used": int(parts[3]) if len(parts) > 3 else 0,
            "memory_free": int(parts[4]) if len(parts) > 4 else int(parts[2]),
            "utilization_gpu": int(parts[5]) if len(parts) > 5 else 0,
        })
    return records


def choose_gpus(gpu_spec: str) -> list[int]:
    spec = gpu_spec.strip().lower()
    if spec == "none":
        return []
    if spec.startswith("auto"):
        count = 1
        if ":" in spec:
            count = int(spec.split(":", 1)[1])
        records = query_gpu_inventory()
        if not records:
            raise RuntimeError("No GPUs found in nvidia-smi output")
        ranked = sorted(records, key=lambda record: (-int(record["memory_free"]), int(record["utilization_gpu"]), int(record["index"])))
        ranked = [record for record in ranked if int(record["memory_free"]) >= int(record["memory_total"]) * 0.9]
        if len(ranked) < count:
            inventory = ", ".join(
                f"{record['index']}:free={record['memory_free']}MiB/total={record['memory_total']}MiB"
                for record in records
            )
            raise RuntimeError(f"Requested {count} auto GPUs but only {len(ranked)} have at least 90% free memory: {inventory}")
        return [int(record["index"]) for record in ranked[:count]]
    return [int(part.strip()) for part in gpu_spec.split(",") if part.strip()]


def infer_litellm_api_env(model_name: str) -> str | None:
    prefix = model_name.split("/", 1)[0].lower()
    if prefix == "openai":
        return "OPENAI_API_KEY"
    if prefix == "gemini":
        return "GEMINI_API_KEY"
    if prefix == "anthropic":
        return "ANTHROPIC_API_KEY"
    return None


def find_free_port(start_port: int) -> int:
    port = start_port
    while port < start_port + 100:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    raise RuntimeError(f"Could not find a free TCP port near {start_port}")


def _openai_models_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"


def _litellm_api_base(base_url: str) -> str:
    base = base_url.rstrip("/")
    suffix = "/chat/completions"
    if base.endswith(suffix):
        base = base[: -len(suffix)]
    return base


def wait_for_openai_server(base_url: str, api_key: str, *, timeout: int = 60) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        request = urllib.request.Request(
            _openai_models_url(base_url),
            headers={"Authorization": f"Bearer {api_key}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        time.sleep(5)
    raise RuntimeError(f"OpenAI-compatible server at {base_url} is not healthy: {last_error}")


def _openai_server_ready(base_url: str, api_key: str) -> bool:
    request = urllib.request.Request(
        _openai_models_url(base_url),
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _url_port(base_url: str) -> int:
    match = re.search(r":(\d+)(?:/|$)", base_url)
    if not match:
        raise ValueError(f"Cannot infer port from --sglang-api-base={base_url!r}")
    return int(match.group(1))


def _start_sglang_server(args: argparse.Namespace, log_path: Path) -> tuple[str, subprocess.Popen[str]]:
    selected_gpus = choose_gpus(args.gpu_id)
    if not selected_gpus:
        raise RuntimeError("sglang backend requires at least one GPU; got --gpu-id=none")

    name = f"run-swe-agent-sglang-{int(time.time())}-{os.getpid()}"
    gpu_list = ",".join(str(gpu) for gpu in selected_gpus)
    docker_cmd = [
        "docker", "run", "--rm", "--name", name, "--runtime", "nvidia", "--net=host", "--shm-size=64g",
        "-e", f"NVIDIA_VISIBLE_DEVICES={gpu_list}",
        "-e", "FLASHINFER_USE_CUDA_NORM=1",
        "-v", f"{REPO_ROOT.resolve()}:/workspace/rler",
        "-w", "/workspace/rler/slime",
        args.sglang_image,
        "bash", "--noprofile", "--norc", "-lc",
        " ".join(
            [
                "exec python3 -m sglang.launch_server",
                "--model-path", shlex.quote(args.sglang_model),
                "--host 127.0.0.1",
                "--port", str(_url_port(args.sglang_api_base)),
                "--tensor-parallel-size", str(len(selected_gpus)),
                "--context-length", str(args.max_model_len),
                "--served-model-name", shlex.quote(args.sglang_model),
                "--reasoning-parser qwen3",
                "--watchdog-timeout 3600",
            ]
        ),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(docker_cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    log_file.close()
    deadline = time.time() + args.server_timeout
    while time.time() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
            raise RuntimeError(f"sglang server exited before ready; see {log_path}\n{tail[-8000:]}")
        if _openai_server_ready(args.sglang_api_base, args.sglang_api_key):
            return name, process
        time.sleep(5)
    _stop_sglang_server((name, process))
    tail = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    raise RuntimeError(f"sglang server did not become ready; see {log_path}\n{tail[-8000:]}")


def _stop_sglang_server(server: tuple[str, subprocess.Popen[str]] | None) -> None:
    if server is None:
        return
    name, process = server
    terminate_process(process)
    subprocess.run(["docker", "rm", "-f", name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def terminate_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=20)


@contextlib.contextmanager
def temporary_env(updates: dict[str, str | None]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def tee_console(log_path: Path) -> Iterator[None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        stdout = TeeStream(sys.stdout, log_file)
        stderr = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            yield


def parse_trajectory_message(
    message: dict[str, Any],
    *,
    model_name: str,
    preserve_token_fields: bool = False,
) -> dict[str, Any] | None:
    role = message.get("role")
    if role not in {"assistant", "user", "exit"}:
        return None
    extra = message.get("extra") or {}
    if role == "assistant":
        text = message.get("content", message.get("message", "")) or ""
        tool_calls = extra.get("actions") or message.get("tool_calls") or []
    elif role == "exit":
        role = "assistant"
        text = extra.get("exception_str") or message.get("content", message.get("message", "")) or ""
        tool_calls = []
    else:
        text = extra.get("raw_output")
        if text is None:
            text = message.get("content", message.get("message", "")) or ""
        tool_calls = []
    parsed = {
        "role": role,
        "message": text,
        "tool_calls": tool_calls,
        "parser_model": model_name,
    }
    if preserve_token_fields and role == "assistant":
        for key in (
            "content",
            "prompt_token_ids",
            "token_ids",
            "tokens",
            "logprobs",
            "usage",
        ):
            if key in message:
                parsed[key] = message[key]
        finish_reason = extra.get("finish_reason") or message.get("finish_reason")
        if finish_reason:
            parsed["finish_reason"] = finish_reason
        if "content_no_thinking" in message:
            parsed["content_no_thinking"] = message["content_no_thinking"]
        elif "content" in parsed:
            content = str(parsed["content"])
            closing_tag = "</think>"
            closing_index = content.rfind(closing_tag)
            if closing_index >= 0:
                parsed["content_no_thinking"] = content[closing_index + len(closing_tag):].lstrip("\r\n")
    return parsed


def build_messages(
    messages_payload: list[dict[str, Any]],
    *,
    model_name: str,
    trajectory_format: str | None = "mini-swe-agent-1.1",
    preserve_token_fields: bool = False,
) -> dict[str, Any]:
    messages = []
    for index, message in enumerate(messages_payload):
        parsed = parse_trajectory_message(
            message,
            model_name=model_name,
            preserve_token_fields=preserve_token_fields,
        )
        if parsed is None:
            continue
        parsed["index"] = index
        messages.append(parsed)
    return {
        "trajectory_format": trajectory_format,
        "parser": parse_trajectory_message.__name__,
        "model_name": model_name,
        "messages": messages,
    }


def make_run_root(
    *,
    output_root: Path,
    benchmark_name: str,
    split: str,
    model_name: str,
) -> Path:
    return output_root / (
        f"{re.sub(r'[^A-Za-z0-9._-]+', '_', benchmark_name.replace('/', '__'))}_"
        f"{re.sub(r'[^A-Za-z0-9._-]+', '_', split.replace('/', '__'))}_"
        f"{re.sub(r'[^A-Za-z0-9._-]+', '_', model_name.replace('/', '__'))}"
    )


def _json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _append_error_to_log(log_path: Path | None, error: Exception | str) -> None:
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n\nERROR: {error}\n")


def make_evaluation_payload(
    status: str,
    passed_tests: Sequence[str] | None = None,
    failed_tests: Sequence[str] | None = None,
    output: str = "",
    error: Exception | str | None = None,
    pass_to_pass_expected: Sequence[str] | None = None,
    fail_to_pass_expected: Sequence[str] | None = None,
    reward_config: EvaluationRewardConfig | None = None,
    infrastructure_error: bool = False,
) -> dict[str, Any]:
    if status not in {"resolved", "unresolved", "empty", "error"}:
        raise ValueError(f"Unknown evaluation status: {status}")

    passed = sorted({str(test) for test in (passed_tests or []) if str(test)}) if status in {"resolved", "unresolved"} else []
    failed = sorted({str(test) for test in (failed_tests or []) if str(test)}) if status in {"resolved", "unresolved"} else []
    if error is not None:
        error_output = (
            "".join(traceback.format_exception(type(error), error, error.__traceback__))
            if isinstance(error, Exception)
            else str(error)
        )
        output = "\n".join(part for part in [output, error_output] if part)

    passed_set = set(passed)
    f2p = {str(t) for t in _test_sequence(fail_to_pass_expected) if str(t)}
    p2p = {str(t) for t in _test_sequence(pass_to_pass_expected) if str(t)}
    reward_config = reward_config or default_evaluation_reward_config()
    f2p_passed = len(passed_set & f2p)
    p2p_passed = len(passed_set & p2p)
    total_expected = len(f2p) + len(p2p)
    visible_total = len(passed_set | set(failed))
    all_pass = status == "resolved"

    if status == "empty":
        reward = EMPTY_REWARD
    elif status == "error":
        # A model-produced patch can make the benchmark command fail or time
        # out.  That is an unresolved policy outcome, not missing training
        # data.  Only failures explicitly classified as evaluator
        # infrastructure errors are discarded by callers.
        reward = ERROR_REWARD if infrastructure_error else 0.0
    elif reward_config.kind == "hard":
        reward = 1.0 if all_pass else 0.0
    elif reward_config.kind == "soft":
        reward = len(passed_set) / visible_total if visible_total else float(all_pass)
    elif total_expected == 0:
        reward = float(all_pass)
    else:
        f2p_pass_rate = f2p_passed / len(f2p) if f2p else 1.0
        p2p_pass_rate = p2p_passed / len(p2p) if p2p else 1.0
        reward = (
            f2p_pass_rate - reward_config.joint_alpha * (1.0 - p2p_pass_rate)
            if reward_config.kind == "joint"
            else f2p_pass_rate
        )

    if status in {"resolved", "unresolved"}:
        if all_pass:
            reward *= reward_config.all_pass_reward
        reward *= reward_config.fallback_patch_penalty
        reward += reward_config.no_action_patch_penalty

    return {
        "status": status,
        "passed_tests": passed,
        "failed_tests": failed,
        "reward": float(reward),
        "metainfo": {
            **({"output": str(output)} if output else {}),
            "infrastructure_error": bool(infrastructure_error),
        },
    }


def _evaluator_exception_payload(
    error: Exception,
    reward_config: EvaluationRewardConfig | None = None,
    *,
    output: str = "",
) -> dict[str, Any]:
    current: BaseException | None = error
    seen: set[int] = set()
    timed_out = False
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if (
            isinstance(current, (subprocess.TimeoutExpired, TimeoutError))
            or "timed out after" in str(current).lower()
        ):
            timed_out = True
            break
        current = current.__cause__ or current.__context__
    return make_evaluation_payload(
        "error",
        output=output,
        error=error,
        reward_config=reward_config,
        infrastructure_error=not timed_out,
    )


def _test_sequence(value: Sequence[str] | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(stripped)
                if isinstance(parsed, (list, tuple, set)):
                    return [str(item) for item in parsed if str(item)]
            except Exception:
                pass
        return [stripped]
    return [str(item) for item in value if str(item)]


def write_failure_artifacts(*, instance_id: str, run_dir: Path, error: Exception | str, log_path: Path | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _append_error_to_log(log_path, error)
    _json_dump(run_dir / "evaluation.json", make_evaluation_payload("error", error=error))


def materialize_backend_run(*, model_name: str, instance_id: str, run_dir: Path, temp_output_dir: Path) -> None:
    traj_path = temp_output_dir / instance_id / f"{instance_id}.traj.json"
    traj = json.loads(traj_path.read_text(encoding="utf-8")) if traj_path.exists() else {}
    _json_dump(
        run_dir / "messages.json",
        build_messages(
            traj.get("messages", []),
            model_name=model_name,
            trajectory_format=traj.get("trajectory_format"),
            preserve_token_fields=True,
        ),
    )

    prediction = {}
    preds_path = temp_output_dir / "preds.json"
    if preds_path.exists():
        predictions = json.loads(preds_path.read_text(encoding="utf-8"))
        if instance_id not in predictions:
            raise RuntimeError(f"Prediction for instance '{instance_id}' not found in {preds_path}")
        prediction = predictions[instance_id]
    _json_dump(
        run_dir / "model_patch.json",
        {
            instance_id: {
                "model_name_or_path": prediction.get("model_name_or_path", model_name),
                "instance_id": prediction.get("instance_id", instance_id),
                "model_patch": prediction.get("model_patch", "") or "",
            }
        },
    )


def run_harness_evaluation(
    *,
    run_dirs: list[Path],
    split: str,
    model_name: str,
    dataset_name: str,
    log_path: Path | None,
    timeout: int,
    max_workers: int,
    instances_by_id: dict[str, dict[str, Any]] | None = None,
    tee_output: bool = True,
) -> None:
    if not run_dirs:
        return
    by_id = instances_by_id
    if by_id is None:
        subset = next((key for key, value in DATASET_MAPPING.items() if value == dataset_name), dataset_name)
        instance_ids = [run_dir.parent.name for run_dir in run_dirs]
        by_id = {
            instance["instance_id"]: instance
            for instance in load_swebench_instances_by_id(subset, split, instance_ids)
        }
    rebench = is_rebench_dataset_name(dataset_name)
    r2egym = is_r2egym_dataset_name(dataset_name)
    swebench_pro = is_swebench_pro_dataset_name(dataset_name)
    deepswe = is_deepswe_dataset_name(dataset_name)

    def evaluate(run_dir: Path) -> tuple[Path, dict[str, Any]]:
        instance_id = run_dir.parent.name
        patch_path = run_dir / "model_patch.json"
        patch_record = json.loads(patch_path.read_text(encoding="utf-8")).get(instance_id, {}) if patch_path.exists() else {}
        patch = str(patch_record.get("model_patch") or "")
        if instance_id not in by_id:
            return run_dir, make_evaluation_payload("error", error=f"Instance not found: {instance_id}")
        instance = by_id[instance_id]
        if r2egym or is_r2egym_instance(instance):
            result = evaluate_r2egym_instances(
                instance=instance,
                patches_by_key={str(run_dir): patch},
                max_workers=1,
                timeout=timeout,
                work_dir=run_dir.resolve(),
            )[str(run_dir)]
            return run_dir, _r2egym_result_payload(result)
        if swebench_pro or is_swebench_pro_instance(instance):
            result = evaluate_swebench_pro_instances(
                instance=instance,
                patches_by_key={str(run_dir): patch},
                max_workers=1,
                timeout=timeout,
                work_dir=run_dir.resolve(),
            )[str(run_dir)]
            return run_dir, _benchmark_result_payload(result)
        if deepswe or is_deepswe_instance(instance):
            result = evaluate_deepswe_instances(
                instance=instance,
                patches_by_key={str(run_dir): patch},
                max_workers=1,
                timeout=timeout,
                work_dir=run_dir.resolve(),
            )[str(run_dir)]
            return run_dir, _benchmark_result_payload(result)
        if not patch:
            return run_dir, make_evaluation_payload("empty")
        if rebench:
            result = evaluate_rebench_prediction(instance=instance, patch_text=patch, timeout=timeout, work_dir=run_dir.resolve())
            return run_dir, _rebench_result_payload(result)
        evaluations = evaluate_swebench_instance_patches(
            instance=instance,
            patches_by_key={str(run_dir): patch},
            model_name=model_name,
            max_workers=1,
            timeout=timeout,
            namespace=None,
            work_dir=run_dir,
        )
        return run_dir, evaluations[str(run_dir)]

    with (tee_console(log_path) if log_path and tee_output else contextlib.nullcontext()):
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(run_dirs)))) as executor:
            future_map = {executor.submit(evaluate, run_dir): run_dir for run_dir in run_dirs}
            for future in concurrent.futures.as_completed(future_map):
                run_dir = future_map[future]
                try:
                    result_dir, payload = future.result()
                except Exception as exc:
                    result_dir, payload = run_dir, _evaluator_exception_payload(exc)
                    _append_error_to_log(log_path, exc)
                _json_dump(result_dir / "evaluation.json", payload)


def _rebench_result_payload(
    result: dict[str, Any], reward_config: EvaluationRewardConfig | None = None
) -> dict[str, Any]:
    if result.get("error"):
        return make_evaluation_payload(
            "error",
            output=result.get("evaluation_output") or "",
            error=str(result["error"]),
            reward_config=reward_config,
        )
    expected = {str(test) for test in result.get("passed_expected", []) if str(test)}
    passed_actual = {str(test) for test in result.get("passed_actual", []) if str(test)}
    failed_actual = {str(test) for test in result.get("failed_actual", []) if str(test)}
    passed_tests = sorted(passed_actual)
    failed_tests = sorted((expected - passed_actual) | failed_actual)
    resolved = bool(result.get("resolved"))
    return make_evaluation_payload(
        status="resolved" if resolved else "unresolved",
        passed_tests=passed_tests,
        failed_tests=failed_tests,
        output=result.get("evaluation_output") or "",
        pass_to_pass_expected=result.get("pass_to_pass_expected", []),
        fail_to_pass_expected=result.get("fail_to_pass_expected", []),
        reward_config=reward_config,
    )


def _r2egym_result_payload(
    result: dict[str, Any], reward_config: EvaluationRewardConfig | None = None
) -> dict[str, Any]:
    if result.get("error"):
        return make_evaluation_payload(
            "error",
            output=result.get("evaluation_output") or "",
            error=str(result["error"]),
            reward_config=reward_config,
        )
    resolved = bool(result.get("resolved"))
    return make_evaluation_payload(
        status="resolved" if resolved else "unresolved",
        passed_tests=result.get("passed_actual", []),
        failed_tests=result.get("failed_actual", []),
        output=result.get("evaluation_output") or "",
        pass_to_pass_expected=result.get("pass_to_pass_expected", []),
        fail_to_pass_expected=result.get("fail_to_pass_expected", []),
        reward_config=reward_config,
    )


def _benchmark_result_payload(
    result: dict[str, Any], reward_config: EvaluationRewardConfig | None = None
) -> dict[str, Any]:
    if result.get("error"):
        return make_evaluation_payload(
            "error",
            output=result.get("evaluation_output") or "",
            error=str(result["error"]),
            reward_config=reward_config,
        )
    resolved = bool(result.get("resolved"))
    return make_evaluation_payload(
        status="resolved" if resolved else "unresolved",
        passed_tests=result.get("passed_actual", []),
        failed_tests=result.get("failed_actual", []),
        output=result.get("evaluation_output") or "",
        pass_to_pass_expected=result.get("pass_to_pass_expected", []),
        fail_to_pass_expected=result.get("fail_to_pass_expected", []),
        reward_config=reward_config,
    )


def _swebench_report_payload(
    report: dict[str, Any],
    instance_id: str,
    output: str = "",
    *,
    fail_to_pass_expected: Sequence[str] | None = None,
    pass_to_pass_expected: Sequence[str] | None = None,
    reward_config: EvaluationRewardConfig | None = None,
) -> dict[str, Any]:
    instance_report = report.get(instance_id, {}) if isinstance(report, dict) else {}
    tests_status = instance_report.get("tests_status", {}) if isinstance(instance_report, dict) else {}
    passed: list[str] = []
    failed: list[str] = []
    for group in tests_status.values():
        if not isinstance(group, dict):
            continue
        passed.extend(str(test) for test in group.get("success", []) if str(test))
        failed.extend(str(test) for test in group.get("failure", []) if str(test))
    resolved = bool(instance_report.get("resolved"))
    return make_evaluation_payload(
        status="resolved" if resolved else "unresolved",
        passed_tests=passed,
        failed_tests=failed,
        output=output,
        fail_to_pass_expected=fail_to_pass_expected,
        pass_to_pass_expected=pass_to_pass_expected,
        reward_config=reward_config,
    )


def evaluate_swebench_instance_patches(
    *,
    instance: dict[str, Any],
    patches_by_key: dict[str, str],
    model_name: str,
    max_workers: int,
    timeout: int = 600,
    namespace: str | None = None,
    work_dir: Path | None = None,
    reward_config: EvaluationRewardConfig | None = None,
) -> dict[str, dict[str, Any]]:
    eval_work_dir = Path(work_dir or Path.cwd()).resolve()
    if is_rebench_instance(instance):
        evaluations: dict[str, dict[str, Any]] = {}
        unique_patches: dict[str, dict[str, Any]] = {}
        for key, patch in patches_by_key.items():
            patch_text = patch or ""
            if not patch_text.strip():
                evaluations[key] = make_evaluation_payload("empty", reward_config=reward_config)
                continue
            unique_patches.setdefault(patch_text, {"patch": patch_text, "keys": []})["keys"].append(key)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(unique_patches) or 1))) as executor:
            future_map = {
                executor.submit(
                    evaluate_rebench_prediction,
                    instance=instance,
                    patch_text=entry["patch"],
                    timeout=timeout,
                    work_dir=eval_work_dir,
                ): entry["keys"]
                for entry in unique_patches.values()
            }
            for future, keys in future_map.items():
                try:
                    payload = _rebench_result_payload(future.result(), reward_config)
                except Exception as exc:
                    payload = _evaluator_exception_payload(exc, reward_config)
                for key in keys:
                    evaluations[key] = payload
        return evaluations

    if is_r2egym_instance(instance):
        evaluations: dict[str, dict[str, Any]] = {}
        try:
            results = evaluate_r2egym_instances(
                instance=instance,
                patches_by_key=patches_by_key,
                max_workers=max_workers,
                timeout=timeout,
                work_dir=eval_work_dir,
            )
        except Exception as exc:
            return {
                key: _evaluator_exception_payload(exc, reward_config)
                for key in patches_by_key
            }
        for key, result in results.items():
            try:
                evaluations[key] = _r2egym_result_payload(result, reward_config)
            except Exception as exc:
                evaluations[key] = _evaluator_exception_payload(exc, reward_config)
        return evaluations

    if is_swebench_pro_instance(instance):
        try:
            results = evaluate_swebench_pro_instances(
                instance=instance,
                patches_by_key=patches_by_key,
                max_workers=max_workers,
                timeout=timeout,
                work_dir=eval_work_dir,
            )
            return {key: _benchmark_result_payload(result, reward_config) for key, result in results.items()}
        except Exception as exc:
            return {
                key: _evaluator_exception_payload(exc, reward_config)
                for key in patches_by_key
            }

    if is_deepswe_instance(instance):
        try:
            results = evaluate_deepswe_instances(
                instance=instance,
                patches_by_key=patches_by_key,
                max_workers=max_workers,
                timeout=timeout,
                work_dir=eval_work_dir,
            )
            return {key: _benchmark_result_payload(result, reward_config) for key, result in results.items()}
        except Exception as exc:
            return {
                key: _evaluator_exception_payload(exc, reward_config)
                for key in patches_by_key
            }

    unique_patches: dict[str, dict[str, Any]] = {}
    evaluations = {
        key: make_evaluation_payload("empty", reward_config=reward_config)
        for key, patch in patches_by_key.items()
        if not (patch or "").strip()
    }
    for key, patch in patches_by_key.items():
        patch_text = patch or ""
        if patch_text.strip():
            patch_hash = hashlib.sha256(patch_text.encode("utf-8")).hexdigest()
            unique_patches.setdefault(patch_hash, {"patch": patch_text, "keys": []})["keys"].append(key)
    if not unique_patches:
        return evaluations

    if select_container_backend() == "singularity":
        try:
            raw_results = evaluate_swebench_instances_singularity(
                instance=instance,
                patches_by_key={
                    key: patch
                    for key, patch in patches_by_key.items()
                    if (patch or "").strip()
                },
                model_name=model_name,
                max_workers=max_workers,
                namespace=namespace,
                work_dir=eval_work_dir,
                timeout=timeout,
            )
        except Exception as exc:
            for key, patch in patches_by_key.items():
                if (patch or "").strip():
                    evaluations[key] = _evaluator_exception_payload(exc, reward_config)
            return evaluations
        for key, result in raw_results.items():
            try:
                evaluations[key] = _swebench_report_payload(
                    result["report"],
                    instance["instance_id"],
                    result.get("output", ""),
                    fail_to_pass_expected=instance.get("FAIL_TO_PASS", []) or [],
                    pass_to_pass_expected=instance.get("PASS_TO_PASS", []) or [],
                    reward_config=reward_config,
                )
            except Exception as exc:
                evaluations[key] = _evaluator_exception_payload(
                    exc,
                    reward_config,
                    output=result.get("output", ""),
                )
        return evaluations

    previous_eval_root = swebench_run_evaluation.RUN_EVALUATION_LOG_DIR
    with tempfile.TemporaryDirectory(prefix=".node-eval-", dir=eval_work_dir) as tmp_dir:
        temp_root = Path(tmp_dir)
        eval_log_root = temp_root / "logs" / "run_evaluation"
        swebench_run_evaluation.RUN_EVALUATION_LOG_DIR = eval_log_root
        try:
            client = swebench_run_evaluation.docker.from_env()
            test_spec = swebench_run_evaluation.make_test_spec(instance, namespace=namespace)
            if namespace is None:
                swebench_run_evaluation.build_env_images(client, [instance], False, 1)
                build_instance_image(test_spec, client, logger=None, nocache=False)

            runs = []
            for index, entry in enumerate(unique_patches.values()):
                run_id = f"node-eval-{int(time.time())}-{index}-{uuid.uuid4().hex[:6]}"
                prediction = {
                    "model_name_or_path": model_name,
                    "instance_id": instance["instance_id"],
                    "model_patch": entry["patch"],
                }
                runs.append((entry["keys"], run_id, prediction, (test_spec, prediction, False, False, client, run_id, None, False)))

            swebench_run_evaluation.run_threadpool(
                swebench_run_evaluation.run_instance,
                [payload for *_, payload in runs],
                max(1, min(max_workers, len(runs))),
            )

            for keys, run_id, prediction, _ in runs:
                log_dir = eval_log_root / run_id / prediction["model_name_or_path"].replace("/", "__") / test_spec.instance_id
                report_path = log_dir / LOG_REPORT
                output_path = log_dir / LOG_TEST_OUTPUT
                output = output_path.read_text(encoding="utf-8", errors="replace") if output_path.exists() else ""
                if report_path.exists():
                    try:
                        report = json.loads(report_path.read_text())
                        payload = _swebench_report_payload(
                            report,
                            test_spec.instance_id,
                            output,
                            fail_to_pass_expected=instance.get("FAIL_TO_PASS", []) or [],
                            pass_to_pass_expected=instance.get("PASS_TO_PASS", []) or [],
                            reward_config=reward_config,
                        )
                    except (json.JSONDecodeError, KeyError, TypeError) as exc:
                        payload = make_evaluation_payload(
                            "error", output=output, error=exc, reward_config=reward_config
                        )
                else:
                    payload = make_evaluation_payload(
                        "error",
                        output=output,
                        error="Missing SWE-bench evaluation report",
                        reward_config=reward_config,
                    )
                evaluations.update({key: payload for key in keys})
        finally:
            swebench_run_evaluation.RUN_EVALUATION_LOG_DIR = previous_eval_root
    return evaluations


@contextlib.contextmanager
def _scoped_swe_agent_file_handlers() -> Iterator[None]:
    logger = logging.getLogger("swe_agent")
    previous_handlers = set(logger.handlers)
    try:
        yield
    finally:
        for handler in list(logger.handlers):
            if handler not in previous_handlers and isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
                handler.close()


def run_swe_instance_multi(
    benchmark_name: str,
    split: str,
    instance_ids: Sequence[str] | None = None,
    *,
    output_root: Path,
    config: dict[str, Any],
    model_name: str,
    eval_timeout: int,
    workers: int = 1,
    pass_n: int = 1,
    redo_existing: bool = True,
    timestamp: str | None = None,
    run_log_path: Path | None = None,
) -> None:
    if instance_ids is not None:
        instances = load_swebench_instances_by_id(benchmark_name, split, list(instance_ids))
    else:
        instances = load_swebench_instances(benchmark_name, split)
    timestamp = timestamp or time.strftime("%Y%m%d-%H%M%S")
    output_root.mkdir(parents=True, exist_ok=True)
    effective_run_log_path = run_log_path or (DEFAULT_LOG_ROOT / f"{timestamp}.log")
    effective_run_log_path.parent.mkdir(parents=True, exist_ok=True)
    run_root = make_run_root(
        output_root=output_root,
        benchmark_name=benchmark_name,
        split=split,
        model_name=model_name,
    )
    dataset_name = DATASET_MAPPING.get(benchmark_name, benchmark_name)
    instances_by_id = {instance["instance_id"]: instance for instance in instances}
    evaluation_errors: list[Exception] = []

    def evaluate_pass(run_dirs: list[Path]) -> None:
        try:
            run_harness_evaluation(
                run_dirs=run_dirs,
                split=split,
                model_name=model_name,
                dataset_name=dataset_name,
                log_path=effective_run_log_path,
                timeout=eval_timeout,
                max_workers=workers,
                instances_by_id=instances_by_id,
                tee_output=False,
            )
        except Exception as exc:
            for run_dir in run_dirs:
                write_failure_artifacts(
                    instance_id=run_dir.parent.name,
                    run_dir=run_dir,
                    error=exc,
                    log_path=effective_run_log_path,
                )
            raise

    with (
        tempfile.TemporaryDirectory(prefix=".run-", dir=output_root) as tmp_dir,
        concurrent.futures.ThreadPoolExecutor(max_workers=1) as evaluation_executor,
    ):
        temp_root = Path(tmp_dir)
        evaluation_futures: list[tuple[int, concurrent.futures.Future[None]]] = []
        run_timestamps: set[str] = set()
        try:
            for pass_index in range(pass_n):
                run_timestamp = timestamp if pass_index == 0 else time.strftime("%Y%m%d-%H%M%S")
                while run_timestamp in run_timestamps:
                    time.sleep(1)
                    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
                run_timestamps.add(run_timestamp)
                temp_output_dir = temp_root / f"batch_outputs_pass_{pass_index + 1:03d}"
                temp_output_dir.mkdir(parents=True, exist_ok=True)
                with tee_console(effective_run_log_path), _scoped_swe_agent_file_handlers():
                    run_swebench_instances(
                        instances=instances,
                        output_path=temp_output_dir,
                        config=config,
                        workers=workers,
                        redo_existing=redo_existing,
                        show_live_progress=False,
                    )
                pass_run_dirs: list[Path] = []
                for instance in instances:
                    run_dir = run_root / instance["instance_id"] / run_timestamp
                    materialize_backend_run(
                        model_name=model_name,
                        instance_id=instance["instance_id"],
                        run_dir=run_dir,
                        temp_output_dir=temp_output_dir,
                    )
                    pass_run_dirs.append(run_dir)
                evaluation_futures.append(
                    (pass_index + 1, evaluation_executor.submit(evaluate_pass, pass_run_dirs))
                )
        finally:
            for pass_number, future in evaluation_futures:
                try:
                    future.result()
                except Exception as exc:
                    _append_error_to_log(
                        effective_run_log_path,
                        f"Evaluation for pass {pass_number} failed: {exc}",
                    )
                    evaluation_errors.append(exc)
    if evaluation_errors:
        raise RuntimeError(
            f"{len(evaluation_errors)} pass evaluation(s) failed; see {effective_run_log_path}"
        ) from evaluation_errors[0]


def _litellm_model_kwargs(model_name: str) -> dict[str, Any]:
    model_lower = model_name.lower()
    kwargs = {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}} if "qwen" in model_lower else {}
    if model_lower.startswith("nvidia/"):
        kwargs["custom_llm_provider"] = "openai"
    return kwargs


def configure_policy_route(
    *,
    backend_name: str,
    model_name: str,
    base_url: str | None = None,
    api_key: str = "EMPTY",
) -> None:
    from agent_rl import register_model_service
    from agent_rl.run_utils import ModelRouteConfig, configure_model_route
    from swe_agent.serving import SGLangChatService

    if backend_name == "openai":
        configure_model_route("policy", ModelRouteConfig(backend="litellm", model_name=model_name))
        return
    if base_url is None:
        raise RuntimeError(f"{backend_name} backend requires a server base URL")
    service_name = VLLM_SERVICE_NAME if backend_name == "vllm" else SLIME_SERVICE_NAME
    register_model_service(
        service_name,
        SGLangChatService(
            base_url=base_url,
            api_key=api_key,
            default_model_name=model_name,
        ),
    )
    configure_model_route(
        "policy",
        ModelRouteConfig(backend="service", service_name=service_name, model_name=model_name),
    )


def clear_policy_route() -> None:
    from agent_rl import clear_model_services
    from agent_rl.run_utils import clear_model_routes

    clear_model_services()
    clear_model_routes()


def run_swe_agent_backend(args: argparse.Namespace, instance_ids: Sequence[str] | None) -> None:
    route_configured = False
    model_name = _resolve_model_name(args)
    if args._evaluation_only:
        selected_instances = getattr(args, "_selected_instances", None)
        if selected_instances is None:
            selected_instances = (
                load_swebench_instances_by_id(args.subset, args.split, list(instance_ids))
                if instance_ids is not None
                else load_swebench_instances(args.subset, args.split)
            )
        by_id = {instance["instance_id"]: instance for instance in selected_instances}
        run_root = make_run_root(
            output_root=args.output_root,
            benchmark_name=args.subset,
            split=args.split,
            model_name=model_name,
        )
        selected_instance_ids = list(instance_ids) if instance_ids is not None else []
        if not selected_instance_ids:
            selected_instance_ids = sorted(
                path.parent.name for path in run_root.glob(f"*/{args._evaluation_only}") if path.is_dir()
            )
        if not selected_instance_ids:
            raise RuntimeError(f"No existing runs found under {run_root} for timestamp {args._evaluation_only}")

        run_dirs: list[Path] = []
        for instance_id in selected_instance_ids:
            if instance_id not in by_id:
                raise RuntimeError(f"Instance not found in {args.subset}/{args.split}: {instance_id}")
            run_dir = run_root / instance_id / args._evaluation_only
            if not (run_dir / "messages.json").exists() or not (run_dir / "model_patch.json").exists():
                raise RuntimeError(f"Missing messages.json or model_patch.json in {run_dir}")
            run_dirs.append(run_dir)
        run_harness_evaluation(
            run_dirs=run_dirs,
            split=args.split,
            model_name=model_name,
            dataset_name=DATASET_MAPPING.get(args.subset, args.subset),
            log_path=args._run_log_path,
            timeout=args.eval_timeout,
            max_workers=args.workers,
            instances_by_id=by_id,
        )
        return

    gpu_ids: list[int] = []
    vllm_handle = None
    sglang_server: tuple[str, subprocess.Popen[str]] | None = None
    if args.backend == "vllm":
        from dr_agent.utils import launch_vllm_server_handle

        gpu_ids = choose_gpus(args.gpu_id)
        if not gpu_ids:
            raise RuntimeError("vLLM backend requires a GPU; got gpu_id=none")
        vllm_handle = launch_vllm_server_handle(
            model_name=args.vllm_model,
            port=find_free_port(args.vllm_port),
            gpu_id=gpu_ids[0],
            gpu_ids=gpu_ids,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            allow_long_max_model_len=args.allow_long_max_model_len,
        )
        configure_policy_route(backend_name=args.backend, model_name=model_name, base_url=vllm_handle.base_url)
        route_configured = True
    elif args.backend == "sglang":
        if args.use_existing_sglang_server:
            wait_for_openai_server(args.sglang_api_base, args.sglang_api_key, timeout=args.server_timeout)
        elif not _openai_server_ready(args.sglang_api_base, args.sglang_api_key):
            sglang_server = _start_sglang_server(args, DEFAULT_LOG_ROOT / f"{args._run_timestamp}-sglang.log")
        configure_policy_route(
            backend_name=args.backend,
            model_name=model_name,
            base_url=args.sglang_api_base,
            api_key=args.sglang_api_key,
        )
        route_configured = True
    else:
        required_env = infer_litellm_api_env(args.openai_model)
        if required_env and not os.getenv(required_env):
            raise RuntimeError(f"{required_env} is not set for model {args.openai_model}")
        configure_policy_route(backend_name=args.backend, model_name=model_name)
        route_configured = True

    try:
        litellm_model_kwargs: dict[str, Any] = {}
        agent_model_class = DEFAULT_MODEL_CLASS
        if args.backend == "openai":
            agent_model_class = "litellm_textbased"
            if args.litellm_api_base:
                litellm_model_kwargs["api_base"] = _litellm_api_base(args.litellm_api_base)
            if args.litellm_api_key:
                litellm_model_kwargs["api_key"] = args.litellm_api_key
        config = build_swebench_config(
            config_spec=[str(SWE_AGENT_TEXTBASED_CONFIG)],
            model=model_name,
            model_class=agent_model_class,
            extra_overrides={
                "agent": {
                    "step_limit": args.step_limit,
                    "cost_limit": 0,
                    "wall_clock_limit_seconds": args.wall_clock_limit_seconds,
                },
                "environment": {
                    "timeout": 120,
                    "pull_timeout": 600,
                },
                "model": {
                    "model_kwargs": {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_tokens": DEFAULT_COMPLETION_MAX_TOKENS,
                        **litellm_model_kwargs,
                        **_litellm_model_kwargs(model_name),
                    },
                    "cost_tracking": "ignore_errors",
                },
            },
        )
        with temporary_env(
            {
                "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": "2",
                "LITELLM_LOG": "ERROR",
            }
        ):
            run_swe_instance_multi(
                args.subset,
                args.split,
                instance_ids,
                output_root=args.output_root,
                config=config,
                model_name=model_name,
                eval_timeout=args.eval_timeout,
                workers=args.workers,
                pass_n=args.pass_n,
                redo_existing=True,
                timestamp=args._run_timestamp,
                run_log_path=args._run_log_path,
            )
    finally:
        if route_configured:
            clear_policy_route()
        terminate_process(vllm_handle.process if vllm_handle else None)
        _stop_sglang_server(sglang_server)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SWE-agent on one or more SWE-bench instances.")
    parser.add_argument("--backend", choices=["vllm", "openai", "sglang"], default="openai")
    parser.add_argument("--instance-id", action=ParseInstanceIds, nargs="+", default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--pass-n", type=int, default=1)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--evaluation-only", default=None)
    parser.add_argument("--timestamp", default=None, help="Override the run timestamp used in output/log directories.")
    parser.add_argument("--run-log-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--eval-timeout", type=int, default=600)
    parser.add_argument("--step-limit", type=int, default=DEFAULT_STEP_LIMIT)
    parser.add_argument(
        "--wall-clock-limit-seconds",
        type=int,
        default=6900,
        help="Hard wall-clock cap on each agent trajectory. Default 6900 (= 1h55m) "
        "sits below mini-swe-agent's hardcoded `sleep 2h` container fuse so the "
        "agent exits cleanly with LimitsExceeded before the container dies and "
        "pollutes the trace with 'No such container' errors. Set 0 to disable.",
    )
    parser.add_argument("--gpu-id", default="auto:2")
    parser.add_argument("--vllm-port", type=int, default=8011)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--allow-long-max-model-len", action="store_true")
    parser.add_argument("--vllm-model", default=DEFAULT_SERVE_MODEL)
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument(
        "--litellm-api-base",
        default=os.environ.get("LITELLM_API_BASE", os.environ.get("OPENAI_API_BASE", os.environ.get("NVIDIA_API_BASE", ""))),
    )
    parser.add_argument(
        "--litellm-api-key",
        default=os.environ.get(
            "LITELLM_API_KEY",
            os.environ.get("NVIDIA_API_KEY", os.environ.get("NVIDIA_NIM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))),
        ),
    )
    parser.add_argument("--sglang-model", default=DEFAULT_SERVE_MODEL)
    parser.add_argument("--sglang-api-base", default=SLIME_API_BASE)
    parser.add_argument("--sglang-api-key", default=SLIME_API_KEY)
    parser.add_argument("--sglang-image", default=os.environ.get("SEARCH_SWE_SLIME_IMAGE", DEFAULT_SGLANG_IMAGE))
    parser.add_argument("--use-existing-sglang-server", action="store_true")
    parser.add_argument("--server-timeout", type=int, default=600)
    parser.add_argument("--temperature", "--tempeterature", dest="temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    args._evaluation_only = args.evaluation_only
    args._run_timestamp = args.evaluation_only or args.timestamp or time.strftime("%Y%m%d-%H%M%S")
    args._run_log_path = args.run_log_path or (DEFAULT_LOG_ROOT / f"{args._run_timestamp}.log")
    if args.pass_n < 1:
        raise ValueError("--pass-n must be positive")
    args._selected_instances = select_instances(args)
    instance_ids = [instance["instance_id"] for instance in args._selected_instances]

    run_root = make_run_root(
        output_root=args.output_root,
        benchmark_name=args.subset,
        split=args.split,
        model_name=_resolve_model_name(args),
    )
    try:
        run_swe_agent_backend(args, instance_ids)
    except Exception as exc:
        for instance_id in instance_ids:
            run_dir = run_root / instance_id / args._run_timestamp
            if not (run_dir / "messages.json").exists() and not (run_dir / "model_patch.json").exists():
                continue
            if (run_dir / "evaluation.json").exists():
                continue
            try:
                write_failure_artifacts(
                    instance_id=instance_id,
                    run_dir=run_dir,
                    error=exc,
                    log_path=args._run_log_path,
                )
            except Exception as artifact_exc:
                _append_error_to_log(args._run_log_path, f"Failed to write failure artifacts for {instance_id}: {artifact_exc}")
        raise RuntimeError(f"{args.backend}: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
