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
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Sequence, TextIO
from swebench.harness import reporting as swebench_reporting
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
    get_swebench_harness_namespace,
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
DEFAULT_INSTANCE_ID = "astropy__astropy-12907"
DEFAULT_SUBSET = "verified"
DEFAULT_SPLIT = "test"
DEFAULT_OUTPUT_ROOT = AGENT_ROOT / "outputs"
DEFAULT_LOG_ROOT = AGENT_ROOT / "logs"
DEFAULT_SERVE_MODEL = "Qwen/Qwen3-8B"
DEFAULT_OPENAI_MODEL = "gemini/gemini-3-pro-preview"
DEFAULT_MODEL_CLASS = "route_textbased"
DEFAULT_VLLM_PORT = 8011
DEFAULT_MAX_MODEL_LEN = 80960
DEFAULT_STEP_LIMIT = 100
DEFAULT_ENV_TIMEOUT = 120
DEFAULT_PULL_TIMEOUT = 600
DEFAULT_EVAL_TIMEOUT = 900
DEFAULT_COMPLETION_MAX_TOKENS = 4096
SWE_AGENT_TEXTBASED_CONFIG = AGENT_ROOT / "swe_agent" / "config" / "benchmarks" / "swebench_backticks.yaml"
SLIME_SERVICE_NAME = "slime"
VLLM_SERVICE_NAME = "vllm"
SLIME_API_BASE = os.environ.get("SEARCH_SWE_SLIME_API_BASE", "http://127.0.0.1:8021")
SLIME_API_KEY = os.environ.get("SEARCH_SWE_SLIME_API_KEY", "EMPTY")
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

    def isatty(self) -> bool:
        return False


class ParseInstanceIds(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parsed = list(getattr(namespace, self.dest) or [])
        for value in values:
            parsed.extend(part for part in value.split(",") if part)
        setattr(namespace, self.dest, parsed)


def _resolve_model_name(args: argparse.Namespace) -> str:
    if args.backend == "vllm":
        return args.vllm_model
    if args.backend == "slime":
        return args.slime_model
    return args.openai_model


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


def _append_error_to_log(log_path: Path | None, error: Exception | str) -> None:
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text("", encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n\nERROR: {error}\n")

def _load_patch_record(run_dir: Path, instance_id: str) -> dict[str, Any]:
    patch_path = run_dir / "model_patch.json"
    if not patch_path.exists():
        return {
            "model_name_or_path": "",
            "instance_id": instance_id,
            "model_patch": "",
        }
    return json.loads(patch_path.read_text(encoding="utf-8")).get(
        instance_id,
        {
            "model_name_or_path": "",
            "instance_id": instance_id,
            "model_patch": "",
        },
    )


def _prediction_chars(run_dir: Path, instance_id: str) -> int:
    return len(str(_load_patch_record(run_dir, instance_id).get("model_patch") or ""))


def materialize_backend_run(
    *,
    model_name: str,
    instance_id: str,
    run_dir: Path,
    temp_output_dir: Path,
    run_log_path: Path | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_traj_temp = temp_output_dir / instance_id / f"{instance_id}.traj.json"
    raw_traj_path = run_dir / "raw_traj.json"
    raw_traj: dict[str, Any] = {}
    if raw_traj_temp.exists():
        shutil.copy2(raw_traj_temp, raw_traj_path)
    else:
        raw_traj_path.write_text(json.dumps({}, indent=2), encoding="utf-8")
    if raw_traj_path.exists():
        raw_traj = json.loads(raw_traj_path.read_text(encoding="utf-8"))

    slim_traj_path = run_dir / "messages.json"
    slim_traj_path.write_text(
        json.dumps(build_slim_trajectory(raw_traj, model_name=model_name), indent=2),
        encoding="utf-8",
    )

    preds_path = temp_output_dir / "preds.json"
    prediction = {"model_patch": ""}
    if preds_path.exists():
        predictions = json.loads(preds_path.read_text(encoding="utf-8"))
        if instance_id not in predictions:
            raise RuntimeError(f"Prediction for instance '{instance_id}' not found in {preds_path}")
        prediction = predictions[instance_id]
    patch_record = {
        "model_name_or_path": prediction.get("model_name_or_path", model_name),
        "instance_id": prediction.get("instance_id", instance_id),
        "model_patch": prediction.get("model_patch", "") or "",
    }
    patch_path = run_dir / "model_patch.json"
    patch_path.write_text(json.dumps({instance_id: patch_record}, indent=2), encoding="utf-8")
    if run_log_path is not None and not run_log_path.exists():
        run_log_path.parent.mkdir(parents=True, exist_ok=True)
        run_log_path.write_text("", encoding="utf-8")


def write_failure_artifacts(
    *,
    instance_id: str,
    run_dir: Path,
    error: Exception | str,
    log_path: Path | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _append_error_to_log(log_path, error)
    (run_dir / "evaluation.json").write_text(
        json.dumps(
            _build_evaluation_payload(
                instance_id=instance_id,
                completed=False,
                resolved=False,
                empty_patch=True,
                error=True,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


def _build_evaluation_payload(
    *,
    instance_id: str,
    completed: bool,
    resolved: bool,
    empty_patch: bool,
    error: bool,
) -> dict[str, Any]:
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


def _run_rebench_harness_evaluation(
    *,
    run_dirs: list[Path],
    split: str,
    dataset_name: str,
    log_path: Path | None,
    timeout: int,
    max_workers: int,
    instances_by_id: dict[str, dict[str, Any]] | None,
) -> None:
    if instances_by_id is None:
        subset = next(key for key, value in DATASET_MAPPING.items() if value == dataset_name)
        instances_by_id = {
            instance["instance_id"]: instance
            for instance in load_swebench_instances(subset, split)
        }
    evaluation_log_context = tee_console(log_path) if log_path else contextlib.nullcontext()
    with evaluation_log_context:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(run_dirs)))) as executor:
            futures: dict[concurrent.futures.Future, tuple[str, Path, bool]] = {}
            for run_dir in run_dirs:
                instance_id = run_dir.parent.name
                evaluation_result_path = run_dir.resolve() / "evaluation.json"
                prediction_chars = _prediction_chars(run_dir, instance_id)
                if prediction_chars == 0:
                    evaluation_result_path.write_text(
                        json.dumps(
                            _build_evaluation_payload(
                                instance_id=instance_id,
                                completed=False,
                                resolved=False,
                                empty_patch=True,
                                error=False,
                            ),
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    continue
                if instance_id not in instances_by_id:
                    evaluation_result_path.write_text(
                        json.dumps(
                            _build_evaluation_payload(
                                instance_id=instance_id,
                                completed=False,
                                resolved=False,
                                empty_patch=False,
                                error=True,
                            ),
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    continue
                futures[
                    executor.submit(
                        evaluate_rebench_prediction,
                        instance=instances_by_id[instance_id],
                        patch_text=str(_load_patch_record(run_dir, instance_id).get("model_patch") or ""),
                        timeout=timeout,
                        work_dir=run_dir.resolve(),
                    )
                ] = (instance_id, run_dir, False)

            for future in concurrent.futures.as_completed(futures):
                instance_id, run_dir, empty_patch = futures[future]
                evaluation_result_path = run_dir.resolve() / "evaluation.json"
                try:
                    payload = future.result()
                    resolved = bool(payload["resolved"])
                    completed = True
                    error = False
                except Exception as exc:
                    resolved = False
                    completed = False
                    error = True
                    _append_error_to_log(log_path, exc)
                evaluation_result_path.write_text(
                    json.dumps(
                        _build_evaluation_payload(
                            instance_id=instance_id,
                            completed=completed,
                            resolved=resolved,
                            empty_patch=empty_patch,
                            error=error,
                        ),
                        indent=2,
                    ),
                    encoding="utf-8",
                )


def _run_swebench_harness_evaluation(
    *,
    run_dirs: list[Path],
    split: str,
    model_name: str,
    dataset_name: str,
    log_path: Path | None,
    timeout: int,
    max_workers: int,
    instances_by_id: dict[str, dict[str, Any]] | None,
) -> None:
    if instances_by_id is None:
        subset = next(key for key, value in DATASET_MAPPING.items() if value == dataset_name)
        instances_by_id = {
            instance["instance_id"]: instance
            for instance in load_swebench_instances(subset, split)
        }
    grouped_run_dirs: dict[str | None, list[Path]] = {}
    for run_dir in run_dirs:
        instance_id = run_dir.parent.name
        instance = instances_by_id.get(instance_id)
        grouped_run_dirs.setdefault(get_swebench_harness_namespace(instance) if instance else None, []).append(run_dir)

    temp_parent = run_dirs[0].resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix=".eval-", dir=temp_parent) as tmp_dir:
        temp_root = Path(tmp_dir)
        eval_log_root = temp_root / "logs" / "run_evaluation"
        predictions_path = (temp_root / "preds.json").resolve()
        predictions_path.write_text(
            json.dumps(
                {
                    run_dir.parent.name: _load_patch_record(run_dir, run_dir.parent.name)
                    for run_dir in run_dirs
                    if (run_dir / "model_patch.json").exists()
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        previous_eval_root = swebench_run_evaluation.RUN_EVALUATION_LOG_DIR
        previous_report_root = swebench_reporting.RUN_EVALUATION_LOG_DIR
        swebench_run_evaluation.RUN_EVALUATION_LOG_DIR = eval_log_root
        swebench_reporting.RUN_EVALUATION_LOG_DIR = eval_log_root
        try:
            evaluation_log_context = tee_console(log_path) if log_path else contextlib.nullcontext()
            with evaluation_log_context:
                for group_index, (namespace, group) in enumerate(grouped_run_dirs.items()):
                    run_id = f"verify-{int(time.time())}-{group_index}"
                    try:
                        with pushd(temp_root):
                            summary_report = swebench_run_evaluation.main(
                                dataset_name=dataset_name,
                                split=split,
                                instance_ids=[run_dir.parent.name for run_dir in group],
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
                            model_key = model_name.replace("/", "__")
                            for candidate in [
                                Path(summary_report) if summary_report else None,
                                temp_root / f"{model_key}.{run_id}.json",
                                AGENT_ROOT / f"{model_key}.{run_id}.json",
                                AGENT_ROOT / "preds.json",
                            ]:
                                if candidate and candidate.exists():
                                    candidate.unlink()
                    except Exception as exc:
                        for run_dir in group:
                            instance_id = run_dir.parent.name
                            evaluation_result_path = run_dir.resolve() / "evaluation.json"
                            evaluation_result_path.write_text(
                                json.dumps(
                                    _build_evaluation_payload(
                                        instance_id=instance_id,
                                        completed=False,
                                        resolved=False,
                                        empty_patch=_prediction_chars(run_dir, instance_id) == 0,
                                        error=True,
                                    ),
                                    indent=2,
                                ),
                                encoding="utf-8",
                            )
                            _append_error_to_log(log_path, exc)
                        continue

                    for run_dir in group:
                        instance_id = run_dir.parent.name
                        model_key = model_name.replace("/", "__")
                        report_path = eval_log_root / run_id / model_key / instance_id / LOG_REPORT
                        empty_patch = _prediction_chars(run_dir, instance_id) == 0
                        completed = False
                        resolved = False
                        error = False
                        if not empty_patch and report_path.exists():
                            try:
                                report = json.loads(report_path.read_text())
                                completed = True
                                resolved = bool(report[instance_id]["resolved"])
                            except (json.JSONDecodeError, KeyError):
                                error = True
                        elif not empty_patch:
                            error = True

                        evaluation_result_path = run_dir.resolve() / "evaluation.json"
                        evaluation_result_path.write_text(
                            json.dumps(
                                _build_evaluation_payload(
                                    instance_id=instance_id,
                                    completed=completed,
                                    resolved=resolved,
                                    empty_patch=empty_patch,
                                    error=error,
                                ),
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
        finally:
            swebench_run_evaluation.RUN_EVALUATION_LOG_DIR = previous_eval_root
            swebench_reporting.RUN_EVALUATION_LOG_DIR = previous_report_root


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
) -> None:
    if not run_dirs:
        return
    if is_rebench_dataset_name(dataset_name):
        _run_rebench_harness_evaluation(
            run_dirs=run_dirs,
            split=split,
            dataset_name=dataset_name,
            log_path=log_path,
            timeout=timeout,
            max_workers=max_workers,
            instances_by_id=instances_by_id,
        )
        return
    _run_swebench_harness_evaluation(
        run_dirs=run_dirs,
        split=split,
        model_name=model_name,
        dataset_name=dataset_name,
        log_path=log_path,
        timeout=timeout,
        max_workers=max_workers,
        instances_by_id=instances_by_id,
    )


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
            timeout=DEFAULT_EVAL_TIMEOUT,
            work_dir=work_dir,
        )

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
    timestamp: str | None = None,
    run_log_path: Path | None = None,
) -> None:
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
    run_root = make_run_root(
        output_root=output_root,
        benchmark_name=benchmark_name,
        split=split,
        model_name=model_name,
    )
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

        run_dirs: list[Path] = []
        for instance in instances:
            run_dir = run_root / instance["instance_id"] / timestamp
            materialize_backend_run(
                model_name=model_name,
                instance_id=instance["instance_id"],
                run_dir=run_dir,
                temp_output_dir=temp_output_dir,
                run_log_path=effective_run_log_path,
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


def run_swe_agent_backend(
    args: argparse.Namespace,
    backend_name: str,
    instance_ids: Sequence[str] | None,
) -> None:
    model_name = args.vllm_model if backend_name == "vllm" else args.openai_model
    if args._evaluation_only:
        available_instances = load_swebench_instances(args.subset, args.split)
        by_id = {instance["instance_id"]: instance for instance in available_instances}
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
            raw_trajectory_path = run_dir / "raw_traj.json"
            patch_path = run_dir / "model_patch.json"
            if not raw_trajectory_path.exists() or not patch_path.exists():
                raise RuntimeError(f"Missing raw_traj.json or model_patch.json in {run_dir}")
            raw_traj = json.loads(raw_trajectory_path.read_text())
            slim_trajectory_path = run_dir / "messages.json"
            if not slim_trajectory_path.exists():
                slim_trajectory_path.write_text(
                    json.dumps(build_slim_trajectory(raw_traj, model_name=model_name), indent=2),
                    encoding="utf-8",
                )
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

    gpu_id: int | None = None
    gpu_ids: list[int] = []
    vllm_handle = None
    if backend_name == "vllm":
        from dr_agent.utils import launch_vllm_server_handle

        gpu_ids = choose_gpus(args.gpu_id)
        gpu_id = gpu_ids[0] if gpu_ids else None
        if gpu_id is None:
            raise RuntimeError("vLLM backend requires a GPU; got gpu_id=none")
        vllm_handle = launch_vllm_server_handle(
            model_name=args.vllm_model,
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
            run_swe_instance_multi(
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
    parser.add_argument("--vllm-model", default=DEFAULT_SERVE_MODEL)
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
    errors: list[str] = []
    for backend_name in backend_names:
        prepared_model_name = args.vllm_model if backend_name == "vllm" else args.openai_model
        target_label = "__all__" if instance_ids is None else "__".join(instance_ids)
        fallback_run_dir = make_run_root(
            output_root=args.output_root,
            benchmark_name=args.subset,
            split=args.split,
            model_name=prepared_model_name,
        ) / re.sub(r"[^A-Za-z0-9._-]+", "_", target_label) / args._run_timestamp
        try:
            run_swe_agent_backend(args, backend_name, instance_ids)
        except Exception as exc:
            errors.append(f"{backend_name}: {exc}")
            write_failure_artifacts(
                instance_id=target_label,
                run_dir=fallback_run_dir,
                error=exc,
                log_path=args._run_log_path,
            )
    if errors:
        raise RuntimeError("; ".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
