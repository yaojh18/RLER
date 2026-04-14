#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence, TextIO
from swebench.harness import reporting as swebench_reporting
from swebench.harness import run_evaluation as swebench_run_evaluation
from swebench.harness.constants import LOG_REPORT
from swebench.harness.docker_build import build_instance_image

os.environ.setdefault("LITELLM_LOG", "ERROR")

from dr_agent.utils import launch_vllm_server_handle
from swe_agent.run.benchmarks.swebench import (
    DATASET_MAPPING,
    build_swebench_config,
    get_swebench_harness_namespace,
    load_swebench_instances,
    run_swebench_instances,
)


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "agent").is_dir() and (candidate / "rl").is_dir():
            return candidate
    raise RuntimeError(f"Could not find repository root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
AGENT_ROOT = REPO_ROOT / "agent"
DEFAULT_INSTANCE_ID = "astropy__astropy-12907"
DEFAULT_SUBSET = "verified"
DEFAULT_SPLIT = "test"
DEFAULT_OUTPUT_ROOT = AGENT_ROOT / "outputs"
DEFAULT_LOG_ROOT = AGENT_ROOT / "logs"
DEFAULT_VLLM_SERVE_MODEL = "Qwen/Qwen3-8B"
DEFAULT_VLLM_CLIENT_MODEL = "openai/Qwen/Qwen3-8B"
DEFAULT_OPENAI_MODEL = "gemini/gemini-3-pro-preview"
DEFAULT_MODEL_CLASS = "litellm_textbased"
DEFAULT_VLLM_PORT = 8011
DEFAULT_MAX_MODEL_LEN = 80960
DEFAULT_STEP_LIMIT = 100
DEFAULT_ENV_TIMEOUT = 120
DEFAULT_PULL_TIMEOUT = 600
DEFAULT_EVAL_TIMEOUT = 900
DEFAULT_COMPLETION_MAX_TOKENS = 4096

SWE_AGENT_TEXTBASED_CONFIG = AGENT_ROOT / "swe_agent" / "config" / "benchmarks" / "swebench_backticks.yaml"

logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
try:
    import litellm

    litellm.turn_off_message_logging = True
    if hasattr(litellm, "suppress_debug_info"):
        litellm.suppress_debug_info = True
    if hasattr(litellm, "_logging") and hasattr(litellm._logging, "_disable_debugging"):
        litellm._logging._disable_debugging()
except Exception:
    pass


@dataclass
class BackendResult:
    benchmark_name: str
    split: str
    backend: str
    model_name: str
    instance_id: str
    run_dir: str
    raw_trajectory_path: str | None
    slim_trajectory_path: str | None
    patch_path: str | None
    log_path: str | None
    evaluation_result_path: str | None
    exit_status: str
    submission_chars: int
    prediction_chars: int
    evaluation_completed: bool
    resolved: bool | None
    run_id: str | None
    error: str | None
    harness_namespace: str | None
    swebench_command: list[str] | None
    evaluation_command: list[str] | None
    vllm_command: list[str] | None
    vllm_log_path: str | None
    gpu_id: int | None


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

    def isatty(self) -> bool:
        return False


class ParseInstanceIds(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parsed = list(getattr(namespace, self.dest) or [])
        for value in values:
            parsed.extend(part for part in value.split(",") if part)
        setattr(namespace, self.dest, parsed)


def parse_nvidia_smi_csv(text: str) -> list[dict[str, int | str]]:
    records: list[dict[str, int | str]] = []
    for raw_line in text.splitlines():
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


def query_gpu_inventory() -> list[dict[str, int | str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    return parse_nvidia_smi_csv(completed.stdout)


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
        ranked = sorted(records, key=lambda record: int(record["memory_free"]), reverse=True)
        if len(ranked) < count:
            raise RuntimeError(f"Requested {count} GPUs but only found {len(ranked)}")
        return [int(record["index"]) for record in ranked[:count]]
    return [int(part.strip()) for part in gpu_spec.split(",") if part.strip()]


def choose_gpu(gpu_spec: str) -> int | None:
    selected = choose_gpus(gpu_spec)
    return selected[0] if selected else None


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
def pushd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextlib.contextmanager
def tee_console(log_path: Path) -> Iterator[None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        stdout = TeeStream(sys.stdout, log_file)
        stderr = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            yield


def parse_mini_swe_message(message: dict[str, Any], *, model_name: str) -> dict[str, Any] | None:
    role = message.get("role")
    if role not in {"assistant", "user"}:
        return None
    extra = message.get("extra") or {}
    if role == "assistant":
        text = message.get("content", "") or ""
        tool_calls = extra.get("actions") or message.get("tool_calls") or []
    else:
        text = extra.get("raw_output")
        if text is None:
            text = message.get("content", "") or ""
        tool_calls = []
    return {
        "role": role,
        "message": text,
        "tool_calls": tool_calls,
        "parser_model": model_name,
    }


def parse_generic_message(message: dict[str, Any], *, model_name: str) -> dict[str, Any] | None:
    role = message.get("role")
    if role not in {"assistant", "user"}:
        return None
    return {
        "role": role,
        "message": message.get("content", "") or "",
        "tool_calls": (message.get("extra") or {}).get("actions") or message.get("tool_calls") or [],
        "parser_model": model_name,
    }


SLIM_TRAJECTORY_PARSERS = {
    "mini-swe-agent-1.1": parse_mini_swe_message,
}


def build_slim_trajectory(raw_traj: dict[str, Any], *, model_name: str) -> dict[str, Any]:
    parser = SLIM_TRAJECTORY_PARSERS.get(raw_traj.get("trajectory_format"), parse_generic_message)
    messages = []
    for index, message in enumerate(raw_traj.get("messages", [])):
        parsed = parser(message, model_name=model_name)
        if parsed is None:
            continue
        parsed["index"] = index
        messages.append(parsed)
    return {
        "trajectory_format": raw_traj.get("trajectory_format"),
        "parser": parser.__name__,
        "model_name": model_name,
        "messages": messages,
    }


def extract_backend_result(
    *,
    benchmark_name: str,
    split: str,
    backend: str,
    model_name: str,
    instance_id: str,
    run_dir: Path,
    temp_output_dir: Path,
    run_log_path: Path,
    swebench_command: list[str] | None,
    harness_namespace: str | None = None,
    vllm_command: list[str] | None = None,
    vllm_log_path: Path | None = None,
    gpu_id: int | None = None,
) -> BackendResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_traj_temp = temp_output_dir / instance_id / f"{instance_id}.traj.json"
    raw_traj_path = run_dir / "raw_traj.json"
    raw_traj: dict[str, Any] = {}
    if raw_traj_temp.exists():
        shutil.copy2(raw_traj_temp, raw_traj_path)
    else:
        raw_traj_path.write_text(json.dumps({}, indent=2))
    if raw_traj_path.exists():
        raw_traj = json.loads(raw_traj_path.read_text())

    slim_traj_path = run_dir / "messages.json"
    slim_traj_path.write_text(json.dumps(build_slim_trajectory(raw_traj, model_name=model_name), indent=2))

    preds_path = temp_output_dir / "preds.json"
    prediction = {"model_patch": ""}
    if preds_path.exists():
        predictions = json.loads(preds_path.read_text())
        if instance_id not in predictions:
            raise RuntimeError(f"Prediction for instance '{instance_id}' not found in {preds_path}")
        prediction = predictions[instance_id]
    patch_record = {
        "model_name_or_path": prediction.get("model_name_or_path", model_name),
        "instance_id": prediction.get("instance_id", instance_id),
        "model_patch": prediction.get("model_patch", "") or "",
    }
    patch_path = run_dir / "model_patch.json"
    patch_path.write_text(json.dumps({instance_id: patch_record}, indent=2))

    info = raw_traj.get("info", {})
    submission = info.get("submission", "") or ""
    return BackendResult(
        benchmark_name=benchmark_name,
        split=split,
        backend=backend,
        model_name=model_name,
        instance_id=instance_id,
        run_dir=str(run_dir),
        raw_trajectory_path=str(raw_traj_path),
        slim_trajectory_path=str(slim_traj_path),
        patch_path=str(patch_path),
        log_path=str(run_log_path),
        evaluation_result_path=None,
        exit_status=info.get("exit_status", ""),
        submission_chars=len(submission),
        prediction_chars=len(patch_record["model_patch"]),
        evaluation_completed=False,
        resolved=None,
        run_id=None,
        error=None,
        harness_namespace=harness_namespace,
        swebench_command=swebench_command,
        evaluation_command=None,
        vllm_command=vllm_command,
        vllm_log_path=str(vllm_log_path) if vllm_log_path else None,
        gpu_id=gpu_id,
    )


def run_harness_evaluation(
    *,
    results: list[BackendResult],
    dataset_name: str,
    timeout: int,
    max_workers: int,
) -> list[BackendResult]:
    if not results:
        return []

    updated_results = {result.instance_id: result for result in results}
    grouped_results: dict[str | None, list[BackendResult]] = {}
    for result in results:
        grouped_results.setdefault(result.harness_namespace, []).append(result)

    temp_parent = Path(results[0].run_dir).resolve().parents[2]
    log_path = Path(results[0].log_path) if results[0].log_path else None
    with tempfile.TemporaryDirectory(prefix=".eval-", dir=temp_parent) as tmp_dir:
        temp_root = Path(tmp_dir)
        eval_log_root = temp_root / "logs" / "run_evaluation"
        predictions_path = (temp_root / "preds.json").resolve()
        predictions_path.write_text(
            json.dumps(
                {
                    result.instance_id: json.loads(Path(result.patch_path).resolve().read_text())[result.instance_id]
                    for result in results
                    if result.patch_path
                },
                indent=2,
            )
        )
        previous_eval_root = swebench_run_evaluation.RUN_EVALUATION_LOG_DIR
        previous_report_root = swebench_reporting.RUN_EVALUATION_LOG_DIR
        swebench_run_evaluation.RUN_EVALUATION_LOG_DIR = eval_log_root
        swebench_reporting.RUN_EVALUATION_LOG_DIR = eval_log_root
        try:
            evaluation_log_context = tee_console(log_path) if log_path else contextlib.nullcontext()
            with evaluation_log_context:
                for group_index, (namespace, group) in enumerate(grouped_results.items()):
                    run_id = f"verify-{group[0].backend}-{int(time.time())}-{group_index}"
                    evaluation_command = [
                        "python_api",
                        "swebench.harness.run_evaluation.main",
                        dataset_name,
                        group[0].split,
                        ",".join(result.instance_id for result in group),
                        str(predictions_path),
                        namespace or "none",
                        run_id,
                    ]
                    try:
                        with pushd(temp_root):
                            summary_report = swebench_run_evaluation.main(
                                dataset_name=dataset_name,
                                split=group[0].split,
                                instance_ids=[result.instance_id for result in group],
                                predictions_path=str(predictions_path),
                                max_workers=max(1, min(max_workers, len(group))),
                                force_rebuild=False,
                                cache_level="env",
                                clean=False,
                                open_file_limit=4096,
                                run_id=run_id,
                                timeout=timeout,
                                namespace=namespace,
                                rewrite_reports=False,
                                modal=False,
                                report_dir=str(temp_root),
                            )
                            model_key = group[0].model_name.replace("/", "__")
                            for candidate in [
                                Path(summary_report) if summary_report else None,
                                temp_root / f"{model_key}.{run_id}.json",
                                AGENT_ROOT / f"{model_key}.{run_id}.json",
                                AGENT_ROOT / "preds.json",
                            ]:
                                if candidate and candidate.exists():
                                    candidate.unlink()
                    except Exception as exc:
                        for result in group:
                            evaluation_result_path = Path(result.run_dir).resolve() / "evaluation.json"
                            evaluation_result_path.write_text(
                                json.dumps(
                                    {
                                        "completed_ids": [],
                                        "incomplete_ids": [],
                                        "empty_patch_ids": [result.instance_id] if result.prediction_chars == 0 else [],
                                        "submitted_ids": [result.instance_id],
                                        "resolved_ids": [],
                                        "unresolved_ids": [],
                                        "error_ids": [result.instance_id],
                                        "schema_version": 2,
                                    },
                                    indent=2,
                                )
                            )
                            updated_results[result.instance_id] = BackendResult(
                                **{
                                    **asdict(result),
                                    "evaluation_result_path": str(evaluation_result_path),
                                    "error": str(exc),
                                    "run_id": run_id,
                                    "evaluation_command": evaluation_command,
                                }
                            )
                        continue

                    for result in group:
                        model_key = result.model_name.replace("/", "__")
                        report_path = eval_log_root / run_id / model_key / result.instance_id / LOG_REPORT
                        empty_patch = result.prediction_chars == 0
                        completed = False
                        resolved = False
                        error = False
                        if not empty_patch and report_path.exists():
                            try:
                                report = json.loads(report_path.read_text())
                                completed = True
                                resolved = bool(report[result.instance_id]["resolved"])
                            except (json.JSONDecodeError, KeyError):
                                error = True
                        elif not empty_patch:
                            error = True

                        evaluation = {
                            "completed_ids": [result.instance_id] if completed else [],
                            "incomplete_ids": [],
                            "empty_patch_ids": [result.instance_id] if empty_patch else [],
                            "submitted_ids": [result.instance_id],
                            "resolved_ids": [result.instance_id] if resolved else [],
                            "unresolved_ids": [result.instance_id] if completed and not resolved else [],
                            "error_ids": [result.instance_id] if error else [],
                            "schema_version": 2,
                        }
                        evaluation_result_path = Path(result.run_dir).resolve() / "evaluation.json"
                        evaluation_result_path.write_text(json.dumps(evaluation, indent=2))
                        updated_results[result.instance_id] = BackendResult(
                            **{
                                **asdict(result),
                                "evaluation_result_path": str(evaluation_result_path),
                                "evaluation_completed": True,
                                "resolved": resolved,
                                "run_id": run_id,
                                "evaluation_command": evaluation_command,
                            }
                        )
        finally:
            swebench_run_evaluation.RUN_EVALUATION_LOG_DIR = previous_eval_root
            swebench_reporting.RUN_EVALUATION_LOG_DIR = previous_report_root
    return [updated_results[result.instance_id] for result in results]


def evaluate_swebench_instance_patches(
    *,
    instance: dict[str, Any],
    patches_by_key: dict[str, str],
    model_name: str,
    max_workers: int,
    namespace: str | None,
    work_dir: Path,
) -> dict[str, float]:
    rewards: dict[str, float] = {}
    unique_patches: dict[str, dict[str, Any]] = {}
    for key, patch in patches_by_key.items():
        patch_text = patch or ""
        if not patch_text.strip():
            rewards[key] = 0.0
            continue
        patch_hash = hashlib.sha256(patch_text.encode("utf-8")).hexdigest()
        entry = unique_patches.setdefault(patch_hash, {"patch": patch_text, "keys": []})
        entry["keys"].append(key)
    if not unique_patches:
        return rewards

    temp_parent = Path(work_dir).resolve()
    previous_eval_root = swebench_run_evaluation.RUN_EVALUATION_LOG_DIR
    with tempfile.TemporaryDirectory(prefix=".node-eval-", dir=temp_parent) as tmp_dir:
        temp_root = Path(tmp_dir)
        eval_log_root = temp_root / "logs" / "run_evaluation"
        swebench_run_evaluation.RUN_EVALUATION_LOG_DIR = eval_log_root
        try:
            client = swebench_run_evaluation.docker.from_env()
            test_spec = swebench_run_evaluation.make_test_spec(instance, namespace=namespace)
            if namespace is None:
                swebench_run_evaluation.build_env_images(client, [instance], False, 1)
                build_instance_image(test_spec, client, logger=None, nocache=False)

            payloads = []
            run_entries = []
            for index, entry in enumerate(unique_patches.values()):
                run_id = f"node-eval-{int(time.time())}-{index}-{uuid.uuid4().hex[:6]}"
                prediction = {
                    "model_name_or_path": model_name,
                    "instance_id": instance["instance_id"],
                    "model_patch": entry["patch"],
                }
                payloads.append((test_spec, prediction, False, False, client, run_id, None, False))
                run_entries.append((entry["keys"], run_id, prediction["model_name_or_path"]))

            swebench_run_evaluation.run_threadpool(
                swebench_run_evaluation.run_instance,
                payloads,
                max(1, min(max_workers, len(payloads))),
            )

            for keys, run_id, prediction_model_name in run_entries:
                report_path = (
                    eval_log_root
                    / run_id
                    / prediction_model_name.replace("/", "__")
                    / test_spec.instance_id
                    / LOG_REPORT
                )
                resolved = False
                if report_path.exists():
                    try:
                        report = json.loads(report_path.read_text())
                        resolved = bool(report[test_spec.instance_id]["resolved"])
                    except (json.JSONDecodeError, KeyError, TypeError):
                        resolved = False
                score = 1.0 if resolved else 0.0
                for key in keys:
                    rewards[key] = score
        finally:
            swebench_run_evaluation.RUN_EVALUATION_LOG_DIR = previous_eval_root
    return rewards


def build_failed_result(
    *,
    benchmark_name: str,
    split: str,
    backend: str,
    model_name: str,
    instance_id: str,
    run_dir: Path,
    error: Exception,
    log_path: Path | None = None,
    swebench_command: list[str] | None = None,
    evaluation_command: list[str] | None = None,
    vllm_command: list[str] | None = None,
    vllm_log_path: Path | None = None,
    gpu_id: int | None = None,
) -> BackendResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    target_log_path = log_path or (run_dir / "run.log")
    target_log_path.parent.mkdir(parents=True, exist_ok=True)
    if not target_log_path.exists():
        target_log_path.write_text("")
    with target_log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n\nERROR: {error}\n")
    return BackendResult(
        benchmark_name=benchmark_name,
        split=split,
        backend=backend,
        model_name=model_name,
        instance_id=instance_id,
        run_dir=str(run_dir),
        raw_trajectory_path=None,
        slim_trajectory_path=None,
        patch_path=None,
        log_path=str(target_log_path),
        evaluation_result_path=None,
        exit_status="error",
        submission_chars=0,
        prediction_chars=0,
        evaluation_completed=False,
        resolved=None,
        run_id=None,
        error=str(error),
        harness_namespace=None,
        swebench_command=swebench_command,
        evaluation_command=evaluation_command,
        vllm_command=vllm_command,
        vllm_log_path=str(vllm_log_path) if vllm_log_path else None,
        gpu_id=gpu_id,
    )


def run_swe_instance_multi(
    benchmark_name: str,
    split: str,
    instance_ids: Sequence[str] | None = None,
    *,
    output_root: Path,
    config: dict[str, Any],
    backend: str,
    model_name: str,
    eval_timeout: int,
    workers: int = 1,
    redo_existing: bool = True,
    gpu_id: int | None = None,
    vllm_command: list[str] | None = None,
    vllm_log_path: Path | None = None,
    timestamp: str | None = None,
    run_log_path: Path | None = None,
) -> list[BackendResult]:
    available_instances = load_swebench_instances(benchmark_name, split)
    if instance_ids is None:
        instances = available_instances
    else:
        by_id = {instance["instance_id"]: instance for instance in available_instances}
        missing = [instance_id for instance_id in instance_ids if instance_id not in by_id]
        if missing:
            raise RuntimeError(f"Instances not found in {benchmark_name}/{split}: {', '.join(missing)}")
        instances = [by_id[instance_id] for instance_id in instance_ids]
    timestamp = timestamp or time.strftime("%Y%m%d-%H%M%S")
    output_root.mkdir(parents=True, exist_ok=True)
    effective_run_log_path = run_log_path or (DEFAULT_LOG_ROOT / f"{timestamp}.log")
    effective_run_log_path.parent.mkdir(parents=True, exist_ok=True)
    run_root = output_root / (
        f"{re.sub(r'[^A-Za-z0-9._-]+', '_', benchmark_name.replace('/', '__'))}_"
        f"{re.sub(r'[^A-Za-z0-9._-]+', '_', split.replace('/', '__'))}_"
        f"{re.sub(r'[^A-Za-z0-9._-]+', '_', model_name.replace('/', '__'))}"
    )
    invocation = [
        "python_api",
        "run_swe_instance_multi",
        benchmark_name,
        split,
        ",".join(instance["instance_id"] for instance in instances),
    ]

    with tempfile.TemporaryDirectory(prefix=".run-", dir=output_root) as tmp_dir:
        temp_root = Path(tmp_dir)
        temp_output_dir = temp_root / "batch_outputs"
        temp_output_dir.mkdir(parents=True, exist_ok=True)

        with tee_console(effective_run_log_path):
            run_swebench_instances(
                instances=instances,
                output_path=temp_output_dir,
                config=config,
                workers=workers,
                redo_existing=redo_existing,
                show_live_progress=False,
            )

        results: list[BackendResult] = []
        for instance in instances:
            run_dir = run_root / instance["instance_id"] / timestamp
            result = extract_backend_result(
                benchmark_name=benchmark_name,
                split=split,
                backend=backend,
                model_name=model_name,
                instance_id=instance["instance_id"],
                run_dir=run_dir,
                temp_output_dir=temp_output_dir,
                run_log_path=effective_run_log_path,
                swebench_command=invocation,
                harness_namespace=get_swebench_harness_namespace(instance),
                vllm_command=vllm_command,
                vllm_log_path=vllm_log_path,
                gpu_id=gpu_id,
            )
            results.append(result)
        try:
            return run_harness_evaluation(
                results=results,
                dataset_name=DATASET_MAPPING.get(benchmark_name, benchmark_name),
                timeout=eval_timeout,
                max_workers=workers,
            )
        except Exception as exc:
            if effective_run_log_path:
                with effective_run_log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(f"\n\nERROR: {exc}\n")
            failed_results: list[BackendResult] = []
            for result in results:
                evaluation_result_path = Path(result.run_dir) / "evaluation.json"
                evaluation_result_path.write_text(
                    json.dumps(
                        {
                            "completed_ids": [],
                            "incomplete_ids": [],
                            "empty_patch_ids": [result.instance_id] if result.prediction_chars == 0 else [],
                            "submitted_ids": [result.instance_id],
                            "resolved_ids": [],
                            "unresolved_ids": [],
                            "error_ids": [result.instance_id],
                            "schema_version": 2,
                        },
                        indent=2,
                    )
                )
                failed_results.append(
                    BackendResult(
                        **{
                            **asdict(result),
                            "evaluation_result_path": str(evaluation_result_path),
                            "error": str(exc),
                        }
                    )
                )
            return failed_results


def run_swe_agent_backend(
    args: argparse.Namespace,
    backend_name: str,
    instance_ids: Sequence[str] | None,
) -> list[BackendResult]:
    model_name = args.vllm_client_model if backend_name == "vllm" else args.openai_model
    if args._evaluation_only:
        available_instances = load_swebench_instances(args.subset, args.split)
        by_id = {instance["instance_id"]: instance for instance in available_instances}
        run_root = args.output_root / (
            f"{re.sub(r'[^A-Za-z0-9._-]+', '_', args.subset.replace('/', '__'))}_"
            f"{re.sub(r'[^A-Za-z0-9._-]+', '_', args.split.replace('/', '__'))}_"
            f"{re.sub(r'[^A-Za-z0-9._-]+', '_', model_name.replace('/', '__'))}"
        )
        selected_instance_ids = list(instance_ids) if instance_ids is not None else []
        if not selected_instance_ids:
            selected_instance_ids = sorted(
                path.parent.name for path in run_root.glob(f"*/{args._evaluation_only}") if path.is_dir()
            )
        if not selected_instance_ids:
            raise RuntimeError(f"No existing runs found under {run_root} for timestamp {args._evaluation_only}")

        results: list[BackendResult] = []
        for instance_id in selected_instance_ids:
            if instance_id not in by_id:
                raise RuntimeError(f"Instance not found in {args.subset}/{args.split}: {instance_id}")
            run_dir = run_root / instance_id / args._evaluation_only
            raw_trajectory_path = run_dir / "raw_traj.json"
            patch_path = run_dir / "model_patch.json"
            if not raw_trajectory_path.exists() or not patch_path.exists():
                raise RuntimeError(f"Missing raw_traj.json or model_patch.json in {run_dir}")
            raw_traj = json.loads(raw_trajectory_path.read_text())
            slim_trajectory_path = run_dir / "messages.json"
            if not slim_trajectory_path.exists():
                slim_trajectory_path.write_text(
                    json.dumps(build_slim_trajectory(raw_traj, model_name=model_name), indent=2)
                )
            patch_record = json.loads(patch_path.read_text())[instance_id]
            submission = raw_traj.get("info", {}).get("submission", "") or ""
            results.append(
                BackendResult(
                    benchmark_name=args.subset,
                    split=args.split,
                    backend=backend_name,
                    model_name=model_name,
                    instance_id=instance_id,
                    run_dir=str(run_dir),
                    raw_trajectory_path=str(raw_trajectory_path),
                    slim_trajectory_path=str(slim_trajectory_path),
                    patch_path=str(patch_path),
                    log_path=str(args._run_log_path),
                    evaluation_result_path=str(run_dir / "evaluation.json") if (run_dir / "evaluation.json").exists() else None,
                    exit_status=raw_traj.get("info", {}).get("exit_status", ""),
                    submission_chars=len(submission),
                    prediction_chars=len(patch_record.get("model_patch", "") or ""),
                    evaluation_completed=False,
                    resolved=None,
                    run_id=None,
                    error=None,
                    harness_namespace=get_swebench_harness_namespace(by_id[instance_id]),
                    swebench_command=None,
                    evaluation_command=None,
                    vllm_command=None,
                    vllm_log_path=None,
                    gpu_id=None,
                )
            )
        return run_harness_evaluation(
            results=results,
            dataset_name=DATASET_MAPPING.get(args.subset, args.subset),
            timeout=args.eval_timeout,
            max_workers=args.workers,
        )

    gpu_id: int | None = None
    gpu_ids: list[int] = []
    vllm_handle = None
    if backend_name == "vllm":
        gpu_ids = choose_gpus(args.gpu_id)
        gpu_id = gpu_ids[0] if gpu_ids else None
        if gpu_id is None:
            raise RuntimeError("vLLM backend requires a GPU; got gpu_id=none")
        vllm_handle = launch_vllm_server_handle(
            model_name=args.vllm_serve_model,
            port=find_free_port(args.vllm_port),
            gpu_id=gpu_id,
            gpu_ids=gpu_ids,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            allow_long_max_model_len=args.allow_long_max_model_len,
        )
        model_kwargs = {"api_base": vllm_handle.base_url, "api_key": "EMPTY"}
    else:
        required_env = infer_litellm_api_env(args.openai_model)
        if required_env and not os.getenv(required_env):
            raise RuntimeError(f"{required_env} is not set for model {args.openai_model}")
        model_kwargs = {}

    try:
        config = build_swebench_config(
            config_spec=[str(SWE_AGENT_TEXTBASED_CONFIG)],
            model=model_name,
            model_class=DEFAULT_MODEL_CLASS,
            extra_overrides={
                "agent": {
                    "step_limit": args.step_limit,
                    "cost_limit": 0,
                },
                "environment": {
                    "timeout": args.environment_timeout,
                    "pull_timeout": args.pull_timeout,
                },
                "model": {
                    "model_kwargs": {
                        "temperature": 1.0 if model_name.lower().startswith("gemini/gemini-3") else 0.0,
                        "max_tokens": args.completion_max_tokens,
                        **model_kwargs,
                    },
                    "cost_tracking": "ignore_errors",
                },
            },
        )
        with temporary_env(
            {
                "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": str(args.model_retry_attempts),
                "LITELLM_LOG": "ERROR",
            }
        ):
            return run_swe_instance_multi(
                args.subset,
                args.split,
                instance_ids,
                output_root=args.output_root,
                config=config,
                backend=backend_name,
                model_name=model_name,
                eval_timeout=args.eval_timeout,
                workers=args.workers,
                redo_existing=True,
                gpu_id=gpu_id,
                vllm_command=vllm_handle.command if vllm_handle else None,
                vllm_log_path=vllm_handle.log_file if vllm_handle else None,
                timestamp=args._run_timestamp,
                run_log_path=args._run_log_path,
            )
    finally:
        terminate_process(vllm_handle.process if vllm_handle else None)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SWE-agent on one or more SWE-bench instances.")
    parser.add_argument("--backend", choices=["vllm", "openai", "both"], default="openai")
    parser.add_argument("--instance-id", action=ParseInstanceIds, nargs="+", default=None)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--evaluation-only", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--step-limit", type=int, default=DEFAULT_STEP_LIMIT)
    parser.add_argument("--environment-timeout", type=int, default=DEFAULT_ENV_TIMEOUT)
    parser.add_argument("--pull-timeout", type=int, default=DEFAULT_PULL_TIMEOUT)
    parser.add_argument("--eval-timeout", type=int, default=DEFAULT_EVAL_TIMEOUT)
    parser.add_argument("--gpu-id", default="auto:2", help="GPU id(s) for local vLLM, e.g. 'auto:2', '0,1', 'auto', or 'none'.")
    parser.add_argument("--vllm-port", type=int, default=DEFAULT_VLLM_PORT)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--allow-long-max-model-len", action="store_true")
    parser.add_argument("--vllm-serve-model", default=DEFAULT_VLLM_SERVE_MODEL)
    parser.add_argument("--vllm-client-model", default=DEFAULT_VLLM_CLIENT_MODEL)
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--model-retry-attempts", type=int, default=2)
    parser.add_argument("--completion-max-tokens", type=int, default=DEFAULT_COMPLETION_MAX_TOKENS)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_arg_parser()
    args = parser.parse_args(raw_argv)
    args._evaluation_only = args.evaluation_only
    args._run_timestamp = args.evaluation_only or time.strftime("%Y%m%d-%H%M%S")
    args._run_log_path = DEFAULT_LOG_ROOT / f"{args._run_timestamp}.log"
    instance_ids = args.instance_id

    backend_names = ["vllm", "openai"] if args.backend == "both" else [args.backend]
    results: list[BackendResult] = []
    for backend_name in backend_names:
        prepared_model_name = args.vllm_client_model if backend_name == "vllm" else args.openai_model
        target_label = "__all__" if instance_ids is None else "__".join(instance_ids)
        fallback_run_dir = (
            args.output_root
            / (
                f"{re.sub(r'[^A-Za-z0-9._-]+', '_', args.subset.replace('/', '__'))}_"
                f"{re.sub(r'[^A-Za-z0-9._-]+', '_', args.split.replace('/', '__'))}_"
                f"{re.sub(r'[^A-Za-z0-9._-]+', '_', prepared_model_name.replace('/', '__'))}"
            )
            / re.sub(r"[^A-Za-z0-9._-]+", "_", target_label)
            / args._run_timestamp
        )
        try:
            results.extend(run_swe_agent_backend(args, backend_name, instance_ids))
        except Exception as exc:
            results.append(
                build_failed_result(
                    benchmark_name=args.subset,
                    split=args.split,
                    backend=backend_name,
                    model_name=prepared_model_name,
                    instance_id=target_label,
                    run_dir=fallback_run_dir,
                    error=exc,
                    log_path=args._run_log_path,
                )
            )

    print(json.dumps([asdict(result) for result in results], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
