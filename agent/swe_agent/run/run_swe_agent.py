#!/usr/bin/env python3

from __future__ import annotations

import argparse
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
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Iterator, Sequence, TextIO
from swebench.harness import run_evaluation as swebench_run_evaluation
from swebench.harness.constants import LOG_REPORT
from swebench.harness.docker_build import build_instance_image

os.environ.setdefault("LITELLM_LOG", "ERROR")

from swe_agent.run.benchmarks.rebench_eval import (
    evaluate_rebench_instances as evaluate_rebench_instance_patches_backend,
    evaluate_rebench_instance as evaluate_rebench_prediction,
    is_rebench_dataset_name,
    is_rebench_instance,
)
from swe_agent.run.benchmarks.swebench import (
    DATASET_MAPPING,
    build_swebench_config,
    load_swebench_instances,
    run_swebench_instances,
)


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
DEFAULT_LOG_ROOT = AGENT_ROOT / "logs"
DEFAULT_SERVE_MODEL = "Qwen/Qwen3-8B"
DEFAULT_OPENAI_MODEL = "gemini/gemini-3-pro-preview"
DEFAULT_MODEL_CLASS = "route_textbased"
DEFAULT_MAX_MODEL_LEN = 128000
DEFAULT_STEP_LIMIT = 160
DEFAULT_COMPLETION_MAX_TOKENS = 4096
SWE_AGENT_TEXTBASED_CONFIG = AGENT_ROOT / "swe_agent" / "config" / "benchmarks" / "swebench_backticks.yaml"
SLIME_SERVICE_NAME = "slime"
VLLM_SERVICE_NAME = "vllm"
SLIME_API_BASE = os.environ.get("SEARCH_SWE_SLIME_API_BASE", "http://127.0.0.1:8021")
SLIME_API_KEY = os.environ.get("SEARCH_SWE_SLIME_API_KEY", "EMPTY")
DEFAULT_SGLANG_IMAGE = "slimerl/slime:qwen35-route-fixed-20260416-v1"
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


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
    from datasets import load_dataset

    dataset_path = DATASET_MAPPING.get(args.subset, args.subset)
    if args.instance_id:
        wanted = set(args.instance_id)
        found: dict[str, dict[str, Any]] = {}
        try:
            dataset = load_dataset(dataset_path, split=args.split, streaming=True)
            for instance in dataset:
                instance_id = instance["instance_id"]
                if instance_id in wanted:
                    found[instance_id] = dict(instance)
                    if len(found) == len(wanted):
                        break
        except Exception:
            instances = load_swebench_instances(args.subset, args.split)
            found = {instance["instance_id"]: instance for instance in instances if instance["instance_id"] in wanted}
        missing = [instance_id for instance_id in args.instance_id if instance_id not in found]
        if missing:
            raise RuntimeError(f"Instances not found in {args.subset}/{args.split}: {', '.join(missing)}")
        return [found[instance_id] for instance_id in args.instance_id]
    if args.offset < 0 or args.limit < 1:
        raise ValueError("--offset must be non-negative and --limit must be positive")
    instances = list(load_dataset(dataset_path, split=f"{args.split}[{args.offset}:{args.offset + args.limit}]"))
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
                "--disable-radix-cache",
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


def parse_trajectory_message(message: dict[str, Any], *, model_name: str) -> dict[str, Any] | None:
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
    return {
        "role": role,
        "message": text,
        "tool_calls": tool_calls,
        "parser_model": model_name,
    }


def build_messages(
    messages_payload: list[dict[str, Any]],
    *,
    model_name: str,
    trajectory_format: str | None = "mini-swe-agent-1.1",
) -> dict[str, Any]:
    messages = []
    for index, message in enumerate(messages_payload):
        parsed = parse_trajectory_message(message, model_name=model_name)
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


def _load_patch_record(run_dir: Path, instance_id: str) -> dict[str, Any]:
    empty = {"model_name_or_path": "", "instance_id": instance_id, "model_patch": ""}
    patch_path = run_dir / "model_patch.json"
    return json.loads(patch_path.read_text(encoding="utf-8")).get(instance_id, empty) if patch_path.exists() else empty


def _patch_text(run_dir: Path, instance_id: str) -> str:
    return str(_load_patch_record(run_dir, instance_id).get("model_patch") or "")


def _evaluation_payload(instance_id: str, *, completed: bool = False, resolved: bool = False, empty_patch: bool = False, error: bool = False) -> dict[str, Any]:
    return {
        "completed_ids": [instance_id] if completed else [],
        "incomplete_ids": [],
        "empty_patch_ids": [instance_id] if empty_patch else [],
        "submitted_ids": [instance_id],
        "resolved_ids": [instance_id] if resolved else [],
        "unresolved_ids": [instance_id] if completed and not resolved else [],
        "error_ids": [instance_id] if error else [],
        "schema_version": 2,
    }


def write_failure_artifacts(*, instance_id: str, run_dir: Path, error: Exception | str, log_path: Path | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _append_error_to_log(log_path, error)
    _json_dump(run_dir / "evaluation.json", _evaluation_payload(instance_id, empty_patch=True, error=True))


def materialize_backend_run(*, model_name: str, instance_id: str, run_dir: Path, temp_output_dir: Path) -> None:
    traj_path = temp_output_dir / instance_id / f"{instance_id}.traj.json"
    traj = json.loads(traj_path.read_text(encoding="utf-8")) if traj_path.exists() else {}
    _json_dump(
        run_dir / "messages.json",
        build_messages(traj.get("messages", []), model_name=model_name, trajectory_format=traj.get("trajectory_format")),
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


def _instances_by_id(dataset_name: str, split: str, instances_by_id: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    if instances_by_id is not None:
        return instances_by_id
    subset = next((key for key, value in DATASET_MAPPING.items() if value == dataset_name), dataset_name)
    return {instance["instance_id"]: instance for instance in load_swebench_instances(subset, split)}


def _evaluate_rebench(run_dirs: list[Path], *, split: str, model_name: str, dataset_name: str, log_path: Path | None, timeout: int, max_workers: int, instances_by_id: dict[str, dict[str, Any]] | None) -> None:
    del model_name
    by_id = _instances_by_id(dataset_name, split, instances_by_id)

    def evaluate(run_dir: Path) -> tuple[Path, dict[str, Any]]:
        instance_id = run_dir.parent.name
        patch = _patch_text(run_dir, instance_id)
        if not patch:
            return run_dir, _evaluation_payload(instance_id, empty_patch=True)
        if instance_id not in by_id:
            return run_dir, _evaluation_payload(instance_id, error=True)
        result = evaluate_rebench_prediction(instance=by_id[instance_id], patch_text=patch, timeout=timeout, work_dir=run_dir.resolve())
        return run_dir, _evaluation_payload(instance_id, completed=True, resolved=bool(result["resolved"]))

    with (tee_console(log_path) if log_path else contextlib.nullcontext()):
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(run_dirs)))) as executor:
            future_map = {executor.submit(evaluate, run_dir): run_dir for run_dir in run_dirs}
            for future in concurrent.futures.as_completed(future_map):
                run_dir = future_map[future]
                try:
                    result_dir, payload = future.result()
                except Exception as exc:
                    instance_id = run_dir.parent.name
                    result_dir, payload = run_dir, _evaluation_payload(instance_id, error=True)
                    _append_error_to_log(log_path, exc)
                _json_dump(result_dir / "evaluation.json", payload)


def _evaluate_swebench(run_dirs: list[Path], *, split: str, model_name: str, dataset_name: str, log_path: Path | None, timeout: int, max_workers: int, instances_by_id: dict[str, dict[str, Any]] | None) -> None:
    by_id = _instances_by_id(dataset_name, split, instances_by_id)
    del timeout

    def evaluate(run_dir: Path) -> tuple[Path, dict[str, Any]]:
        instance_id = run_dir.parent.name
        patch = _patch_text(run_dir, instance_id)
        if not patch:
            return run_dir, _evaluation_payload(instance_id, empty_patch=True)
        if instance_id not in by_id:
            return run_dir, _evaluation_payload(instance_id, error=True)
        rewards = evaluate_swebench_instance_patches(
            instance=by_id[instance_id],
            patches_by_key={str(run_dir): patch},
            model_name=model_name,
            max_workers=1,
            namespace=None,
            work_dir=run_dir,
        )
        return run_dir, _evaluation_payload(instance_id, completed=True, resolved=bool(rewards.get(str(run_dir))))

    with (tee_console(log_path) if log_path else contextlib.nullcontext()):
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(run_dirs)))) as executor:
            future_map = {executor.submit(evaluate, run_dir): run_dir for run_dir in run_dirs}
            for future in concurrent.futures.as_completed(future_map):
                run_dir = future_map[future]
                try:
                    result_dir, payload = future.result()
                except Exception as exc:
                    instance_id = run_dir.parent.name
                    result_dir, payload = run_dir, _evaluation_payload(instance_id, error=True)
                    _append_error_to_log(log_path, exc)
                _json_dump(result_dir / "evaluation.json", payload)


def run_harness_evaluation(*, run_dirs: list[Path], split: str, model_name: str, dataset_name: str, log_path: Path | None, timeout: int, max_workers: int, instances_by_id: dict[str, dict[str, Any]] | None = None) -> None:
    if not run_dirs:
        return
    evaluator = _evaluate_rebench if is_rebench_dataset_name(dataset_name) else _evaluate_swebench
    evaluator(run_dirs, split=split, model_name=model_name, dataset_name=dataset_name, log_path=log_path, timeout=timeout, max_workers=max_workers, instances_by_id=instances_by_id)


def evaluate_swebench_instance_patches(
    *,
    instance: dict[str, Any],
    patches_by_key: dict[str, str],
    model_name: str,
    max_workers: int,
    namespace: str | None,
    work_dir: Path,
) -> dict[str, float]:
    if is_rebench_instance(instance):
        return evaluate_rebench_instance_patches_backend(
            instance=instance,
            patches_by_key=patches_by_key,
            max_workers=max_workers,
            timeout=900,
            work_dir=work_dir,
        )

    unique_patches: dict[str, dict[str, Any]] = {}
    rewards = {key: 0.0 for key, patch in patches_by_key.items() if not (patch or "").strip()}
    for key, patch in patches_by_key.items():
        patch_text = patch or ""
        if patch_text.strip():
            patch_hash = hashlib.sha256(patch_text.encode("utf-8")).hexdigest()
            unique_patches.setdefault(patch_hash, {"patch": patch_text, "keys": []})["keys"].append(key)
    if not unique_patches:
        return rewards

    previous_eval_root = swebench_run_evaluation.RUN_EVALUATION_LOG_DIR
    with tempfile.TemporaryDirectory(prefix=".node-eval-", dir=Path(work_dir).resolve()) as tmp_dir:
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
                report_path = eval_log_root / run_id / prediction["model_name_or_path"].replace("/", "__") / test_spec.instance_id / LOG_REPORT
                if report_path.exists():
                    try:
                        report = json.loads(report_path.read_text())
                        resolved = bool(report[test_spec.instance_id]["resolved"])
                    except (json.JSONDecodeError, KeyError, TypeError):
                        resolved = False
                else:
                    resolved = False
                rewards.update({key: 1.0 if resolved else 0.0 for key in keys})
        finally:
            swebench_run_evaluation.RUN_EVALUATION_LOG_DIR = previous_eval_root
    return rewards


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
    instances = load_swebench_instances(benchmark_name, split)
    if instance_ids is not None:
        by_id = {instance["instance_id"]: instance for instance in instances}
        missing = [instance_id for instance_id in instance_ids if instance_id not in by_id]
        if missing:
            raise RuntimeError(f"Instances not found in {benchmark_name}/{split}: {', '.join(missing)}")
        instances = [by_id[instance_id] for instance_id in instance_ids]
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
    with tempfile.TemporaryDirectory(prefix=".run-", dir=output_root) as tmp_dir:
        temp_root = Path(tmp_dir)
        run_dirs: list[Path] = []
        run_timestamps: set[str] = set()
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
            for instance in instances:
                run_dir = run_root / instance["instance_id"] / run_timestamp
                materialize_backend_run(
                    model_name=model_name,
                    instance_id=instance["instance_id"],
                    run_dir=run_dir,
                    temp_output_dir=temp_output_dir,
                )
                run_dirs.append(run_dir)
        try:
            run_harness_evaluation(
                run_dirs=run_dirs,
                split=split,
                model_name=model_name,
                dataset_name=DATASET_MAPPING.get(benchmark_name, benchmark_name),
                log_path=effective_run_log_path,
                timeout=eval_timeout,
                max_workers=workers,
                instances_by_id={instance["instance_id"]: instance for instance in instances},
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


def _qwen_model_kwargs(model_name: str) -> dict[str, Any]:
    return {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}} if "qwen" in model_name.lower() else {}


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
            selected_instances = load_swebench_instances(args.subset, args.split)
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
            timeout=900,
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
                    "timeout": 120,
                    "pull_timeout": 600,
                },
                "model": {
                    "model_kwargs": {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_tokens": DEFAULT_COMPLETION_MAX_TOKENS,
                        **_qwen_model_kwargs(model_name),
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
                eval_timeout=900,
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
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--step-limit", type=int, default=DEFAULT_STEP_LIMIT)
    parser.add_argument("--gpu-id", default="auto:2")
    parser.add_argument("--vllm-port", type=int, default=8011)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--allow-long-max-model-len", action="store_true")
    parser.add_argument("--vllm-model", default=DEFAULT_SERVE_MODEL)
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--sglang-model", default=DEFAULT_SERVE_MODEL)
    parser.add_argument("--sglang-api-base", default=SLIME_API_BASE)
    parser.add_argument("--sglang-api-key", default=SLIME_API_KEY)
    parser.add_argument("--sglang-image", default=os.environ.get("SEARCH_SWE_SLIME_IMAGE", DEFAULT_SGLANG_IMAGE))
    parser.add_argument("--use-existing-sglang-server", action="store_true")
    parser.add_argument("--server-timeout", type=int, default=900)
    parser.add_argument("--temperature", "--tempeterature", dest="temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    args._evaluation_only = args.evaluation_only
    args._run_timestamp = args.evaluation_only or time.strftime("%Y%m%d-%H%M%S")
    args._run_log_path = DEFAULT_LOG_ROOT / f"{args._run_timestamp}.log"
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
