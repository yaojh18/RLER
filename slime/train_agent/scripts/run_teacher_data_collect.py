#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (REPO_ROOT / "agent", REPO_ROOT / "slime"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from swe_agent.run.benchmarks.swebench import load_swebench_instances
from swe_agent.run.benchmarks.swebench import get_swebench_docker_image_name
from swe_agent.run.run_swe_agent import infer_litellm_api_env
from train_agent.collect_sft_rollout import DEFAULT_TEACHER_MODEL, collect_teacher_student_export
from train_agent.contracts import ExportSample
from train_agent.data_export import _RunArtifacts
from train_agent.serving.sglang_chat_service import (
    SWE_AGENT_POLICY_SYSTEM_PROMPT,
    start_slime_policy_route_warmup,
    start_slime_server,
    stop_slime_server,
)


DEFAULT_SLIME_IMAGE = "slimerl/slime:qwen35-route-fixed-20260416-v1"
DEFAULT_STUDENT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_MODEL_DIR = Path("/m-coriander/coriander/zhichen/models/Qwen3.5-9B")


def select_rebench_instances(
    *,
    subset: str,
    split: str,
    instance_ids: list[str] | None,
    offset: int,
    limit: int,
) -> list[dict[str, Any]]:
    instances = load_swebench_instances(subset, split)
    if instance_ids:
        instances_by_id = {str(instance["instance_id"]): instance for instance in instances}
        missing = [instance_id for instance_id in instance_ids if instance_id not in instances_by_id]
        if missing:
            raise RuntimeError(f"Could not find instance ids in {subset}/{split}: {missing}")
        return [instances_by_id[instance_id] for instance_id in instance_ids]
    selected = instances[offset : offset + limit]
    if len(selected) < limit:
        raise RuntimeError(f"Only found {len(selected)} instances for {subset}/{split} at offset={offset}, limit={limit}.")
    return selected


def docker_image_exists(image_name: str) -> bool:
    return subprocess.run(
        ["docker", "image", "inspect", image_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def prepare_one_instance_image(instance: dict[str, Any]) -> dict[str, str]:
    image_hint = instance.get("image_name") or instance.get("docker_image")
    was_present = isinstance(image_hint, str) and docker_image_exists(image_hint)
    image_name = get_swebench_docker_image_name(instance)
    if docker_image_exists(image_name):
        action = "present" if was_present else "built"
    else:
        action = "pull"
        subprocess.run(["docker", "pull", image_name], check=True)
    return {"instance_id": str(instance["instance_id"]), "image": image_name, "action": action}


def prepare_instance_images(instances: list[dict[str, Any]], *, log_path: Path, workers: int) -> list[dict[str, str]]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    with log_path.open("w", encoding="utf-8") as log_file:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [executor.submit(prepare_one_instance_image, instance) for instance in instances]
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                record = future.result()
                records.append(record)
                print(json.dumps({"index": index, "total": len(instances), **record}, ensure_ascii=False), file=log_file, flush=True)
    return records


def render_messages(messages: list[dict[str, Any]]) -> str:
    return "\n".join(f"{message.get('role', '')}: {message.get('content', '')}" for message in messages)


def messages_length_record(
    *,
    target: str,
    instance_id: str,
    sample_id: str,
    group_id: str,
    prompt_messages: list[dict[str, Any]],
    response_messages: list[dict[str, Any]],
    source: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt = render_messages(prompt_messages)
    response = render_messages(response_messages)
    record = {
        "source": source,
        "target": target,
        "instance_id": instance_id,
        "sample_id": sample_id,
        "group_id": group_id,
        "prompt_chars": len(prompt),
        "response_chars": len(response),
        "total_chars": len(prompt) + len(response),
        "prompt_messages": len(prompt_messages),
        "response_messages": len(response_messages),
    }
    if metadata:
        record.update(metadata)
    return record


def sample_length_record(target: str, sample: ExportSample, instance_id: str) -> dict[str, Any]:
    return messages_length_record(
        target=target,
        instance_id=instance_id,
        sample_id=sample.sample_id,
        group_id=sample.group_id,
        prompt_messages=sample.prompt,
        response_messages=sample.turns,
        source="accepted_sft",
    )


def split_run_dirs(run_dir_field: str) -> list[Path]:
    return [Path(part) for part in str(run_dir_field).split(",") if part]


def collect_raw_teacher_length_records(run_dirs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for run_dir in run_dirs:
        try:
            artifacts = _RunArtifacts(run_dir)
        except Exception as exc:
            errors.append({"run_dir": str(run_dir), "error": f"{type(exc).__name__}: {exc}"})
            continue

        for node_id, node in sorted(artifacts.nodes.items()):
            if node_id == "root" or str(node.get("policy_source")) != "teacher":
                continue
            turns = artifacts.node_messages.get(node_id, [])
            if not turns:
                continue
            parent_id = node.get("parent_id")
            records.append(
                messages_length_record(
                    target="policy",
                    instance_id=artifacts.instance_id,
                    sample_id=node_id,
                    group_id=str(parent_id or ""),
                    prompt_messages=artifacts.build_prefix_messages(parent_id),
                    response_messages=turns,
                    source="raw_teacher",
                    metadata={
                        "run_dir": str(run_dir),
                        "parent_node_id": parent_id,
                        "policy_source": node.get("policy_source"),
                    },
                )
            )

        for rubric_id, messages in sorted(artifacts.rubric_messages.items()):
            if not messages:
                continue
            records.append(
                messages_length_record(
                    target="rubric",
                    instance_id=artifacts.instance_id,
                    sample_id=rubric_id,
                    group_id=rubric_id,
                    prompt_messages=messages[:1],
                    response_messages=messages[1:],
                    source="raw_teacher",
                    metadata={"run_dir": str(run_dir)},
                )
            )
    return records, errors


def summarize_numeric(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    sorted_values = sorted(values)
    return {
        "count": len(values),
        "min": sorted_values[0],
        "median": statistics.median(sorted_values),
        "mean": statistics.fmean(sorted_values),
        "p90": sorted_values[min(len(sorted_values) - 1, max(0, math.ceil(0.9 * len(sorted_values)) - 1))],
        "max": sorted_values[-1],
    }


def summarize_lengths(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for target in ("policy", "rubric"):
        target_records = [record for record in records if record["target"] == target]
        summary[target] = {
            "prompt_chars": summarize_numeric([int(record["prompt_chars"]) for record in target_records]),
            "response_chars": summarize_numeric([int(record["response_chars"]) for record in target_records]),
            "prompt_plus_response_chars": summarize_numeric([int(record["total_chars"]) for record in target_records]),
        }
    return summary


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def update_summary_progress(summary: dict[str, Any], *, started: float, successful_instances: int) -> None:
    summary["completed"] = len(summary["results"])
    summary["successful_instances"] = successful_instances
    summary["failed_instances"] = len(summary["results"]) - successful_instances
    summary["seconds"] = time.perf_counter() - started
    summary["length_summary"] = summarize_lengths(summary["length_records"])


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log_file:
        print(json.dumps({"timestamp": time.time(), **record}, ensure_ascii=False), file=log_file, flush=True)


def safe_log_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "instance"


def path_arg(path: Path) -> str:
    return str(path)


def slime_api_base(args: argparse.Namespace) -> str:
    return os.environ.get("SEARCH_SWE_SLIME_API_BASE") or args.slime_api_base


def slime_api_key(args: argparse.Namespace) -> str:
    return os.environ.get("SEARCH_SWE_SLIME_API_KEY") or args.slime_api_key


def slime_server_healthy(base_url: str, api_key: str, timeout: int = 5) -> bool:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/models",
        headers=headers,
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def post_slime_keepalive(base_url: str, api_key: str, model_name: str, timeout: int = 300) -> None:
    payload = json.dumps(
        {
            "model": model_name,
            "messages": [
                {"role": "system", "content": SWE_AGENT_POLICY_SYSTEM_PROMPT},
                {"role": "user", "content": " "},
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 1,
            "chat_template_kwargs": {"enable_thinking": True},
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()


class SlimeKeepalive:
    def __init__(self, *, base_url: str, api_key: str, model_name: str, interval_seconds: int, log_path: Path) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.interval_seconds = interval_seconds
        self.log_path = log_path
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "SlimeKeepalive":
        if self.interval_seconds <= 0:
            return self
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            record: dict[str, Any] = {"timestamp": time.time(), "base_url": self.base_url}
            try:
                post_slime_keepalive(self.base_url, self.api_key, self.model_name)
                record["status"] = "ok"
            except Exception as exc:
                record["status"] = "error"
                record["error"] = f"{type(exc).__name__}: {exc}"
            with self.log_path.open("a", encoding="utf-8") as log_file:
                print(json.dumps(record, ensure_ascii=False), file=log_file, flush=True)


def ensure_instance_slime_server(
    *,
    args: argparse.Namespace,
    logs_dir: Path,
    instance_id: str,
    index: int,
    total: int,
    server: Any,
) -> Any:
    event_log = logs_dir / "slime_server_events.jsonl"
    if args.use_existing_slime_server:
        os.environ["SEARCH_SWE_SLIME_API_BASE"] = args.slime_api_base
        os.environ["SEARCH_SWE_SLIME_API_KEY"] = args.slime_api_key
        if not slime_server_healthy(args.slime_api_base, args.slime_api_key):
            append_jsonl(
                event_log,
                {"event": "existing_unhealthy", "instance_id": instance_id, "index": index, "total": total, "base_url": args.slime_api_base},
            )
            raise RuntimeError(f"existing SGLang server is not healthy before {instance_id}: {args.slime_api_base}")
        append_jsonl(
            event_log,
            {"event": "existing_healthy", "instance_id": instance_id, "index": index, "total": total, "base_url": args.slime_api_base},
        )
        return None

    base_url = slime_api_base(args)
    api_key = slime_api_key(args)
    process_status = None if server is None else server.process.poll()
    healthy = server is not None and process_status is None and slime_server_healthy(base_url, api_key)
    if healthy:
        append_jsonl(
            event_log,
            {"event": "healthy_reuse", "instance_id": instance_id, "index": index, "total": total, "base_url": base_url},
        )
        return server

    append_jsonl(
        event_log,
        {
            "event": "restart",
            "instance_id": instance_id,
            "index": index,
            "total": total,
            "base_url": base_url,
            "previous_process_status": process_status,
        },
    )
    stop_slime_server(server)
    log_path = logs_dir / f"slime_server_{index:03d}_of_{total:03d}_{safe_log_stem(instance_id)}.log"
    server = start_slime_server(args, log_path)
    base_url = slime_api_base(args)
    api_key = slime_api_key(args)
    if not slime_server_healthy(base_url, api_key):
        stop_slime_server(server)
        append_jsonl(
            event_log,
            {"event": "started_unhealthy", "instance_id": instance_id, "index": index, "total": total, "base_url": base_url},
        )
        raise RuntimeError(f"SGLang server is not healthy after startup before {instance_id}: {base_url}")
    append_jsonl(
        event_log,
        {"event": "started_healthy", "instance_id": instance_id, "index": index, "total": total, "base_url": base_url, "log_path": str(log_path)},
    )
    start_slime_policy_route_warmup(
        base_url=base_url,
        api_key=api_key,
        model_name=args.student_model,
        requests=max(1, args.rollout_gpus),
    )
    return server


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description="Collect Gemini-teacher/Qwen3.5-student search outputs on ReBench v2.")
    parser.add_argument("--subset", default="rebench_v2")
    parser.add_argument("--split", default="train")
    parser.add_argument("--instance-id", nargs="+", default=None, help="Override the first-N selection with explicit ids.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--teacher-backend", default="openai")
    parser.add_argument("--teacher-api-key", default=os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--student-model", default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--slime-image", default=DEFAULT_SLIME_IMAGE)
    parser.add_argument("--search-gpus", default="auto:2")
    parser.add_argument("--slime-port", type=int, default=8032)
    parser.add_argument("--slime-api-base", default="http://127.0.0.1:8032")
    parser.add_argument("--slime-api-key", default="EMPTY")
    parser.add_argument("--use-existing-slime-server", action="store_true")
    parser.add_argument("--serving-timeout", type=int, default=1200)
    parser.add_argument(
        "--slime-keepalive-interval-seconds",
        type=int,
        default=120,
        help="Send a tiny SGLang request during long teacher/eval phases. Use 0 to disable.",
    )
    parser.add_argument("--rollout-gpus", type=int, default=2)
    parser.add_argument(
        "--instance-workers",
        type=int,
        default=2,
        help="Run this many ReBench instances concurrently. Values >1 use subprocess workers sharing one SGLang server.",
    )
    parser.add_argument(
        "--instance-start-delay-seconds",
        type=float,
        default=300.0,
        help="Delay between launching consecutive instance workers in concurrent mode.",
    )
    parser.add_argument("--skip-prepare-instance-images", action="store_true")
    parser.add_argument("--image-prepare-workers", type=int, default=4)
    parser.add_argument("--search-m", type=int, default=8)
    parser.add_argument("--search-n", type=int, default=2)
    parser.add_argument("--search-p", type=int, default=2)
    parser.add_argument("--search-k", type=int, default=20)
    parser.add_argument("--search-max-rounds", type=int, default=5)
    parser.add_argument("--search-step-limit", type=int, default=100)
    parser.add_argument(
        "--search-output-root",
        "--output-root",
        dest="search_output_root",
        type=Path,
        default=REPO_ROOT / "agent/outputs/search_outputs" / f"rebench_v2_teacher_student_{timestamp}",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPO_ROOT / "slime/train_agent/artifacts" / f"rebench_v2_teacher_student_{timestamp}",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
    )
    return parser.parse_args(argv)


def build_instance_worker_command(
    *,
    args: argparse.Namespace,
    instance_id: str,
    worker_summary_path: Path,
    worker_artifact_root: Path,
    base_url: str,
    api_key: str,
) -> tuple[list[str], dict[str, str]]:
    env = os.environ.copy()
    env["SEARCH_SWE_SLIME_API_BASE"] = base_url
    env["SEARCH_SWE_SLIME_API_KEY"] = api_key
    required_env = infer_litellm_api_env(args.teacher_model) if args.teacher_backend == "openai" else None
    if required_env and args.teacher_api_key:
        env[required_env] = args.teacher_api_key

    cmd = [
        sys.executable,
        path_arg(Path(__file__).resolve()),
        "--subset",
        args.subset,
        "--split",
        args.split,
        "--instance-id",
        instance_id,
        "--limit",
        "1",
        "--teacher-model",
        args.teacher_model,
        "--teacher-backend",
        args.teacher_backend,
        "--student-model",
        args.student_model,
        "--model-dir",
        path_arg(args.model_dir),
        "--slime-image",
        args.slime_image,
        "--search-gpus",
        args.search_gpus,
        "--slime-port",
        str(args.slime_port),
        "--slime-api-base",
        base_url,
        "--slime-api-key",
        api_key,
        "--use-existing-slime-server",
        "--serving-timeout",
        str(args.serving_timeout),
        "--slime-keepalive-interval-seconds",
        "0",
        "--rollout-gpus",
        str(args.rollout_gpus),
        "--instance-workers",
        "1",
        "--skip-prepare-instance-images",
        "--search-m",
        str(args.search_m),
        "--search-n",
        str(args.search_n),
        "--search-p",
        str(args.search_p),
        "--search-k",
        str(args.search_k),
        "--search-max-rounds",
        str(args.search_max_rounds),
        "--search-step-limit",
        str(args.search_step_limit),
        "--search-output-root",
        path_arg(args.search_output_root),
        "--artifact-root",
        path_arg(worker_artifact_root),
        "--summary-path",
        path_arg(worker_summary_path),
    ]
    return cmd, env


def merge_worker_summary(
    *,
    summary: dict[str, Any],
    worker_summary_path: Path,
    worker_log_path: Path,
    returncode: int,
    instance_id: str,
    index: int,
    total: int,
    seconds: float,
) -> int:
    if not worker_summary_path.exists():
        summary["results"].append(
            {
                "status": "error",
                "instance_id": instance_id,
                "index": index,
                "total": total,
                "seconds": seconds,
                "error": f"worker exited with returncode={returncode} before writing summary",
                "worker_returncode": returncode,
                "worker_log_path": str(worker_log_path),
            }
        )
        return 0

    try:
        worker_summary = json.loads(worker_summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        summary["results"].append(
            {
                "status": "error",
                "instance_id": instance_id,
                "index": index,
                "total": total,
                "seconds": seconds,
                "error": f"could not read worker summary: {type(exc).__name__}: {exc}",
                "worker_returncode": returncode,
                "worker_log_path": str(worker_log_path),
            }
        )
        return 0

    worker_results = worker_summary.get("results") or []
    if not worker_results:
        summary["results"].append(
            {
                "status": "error",
                "instance_id": instance_id,
                "index": index,
                "total": total,
                "seconds": seconds,
                "error": f"worker wrote no result, returncode={returncode}",
                "worker_returncode": returncode,
                "worker_log_path": str(worker_log_path),
            }
        )
        return 0

    ok_count = 0
    for result in worker_results:
        if isinstance(result, dict):
            merged_result = dict(result)
        else:
            merged_result = {"status": "error", "instance_id": instance_id, "error": f"invalid worker result: {result!r}"}
        merged_result.setdefault("instance_id", instance_id)
        merged_result.setdefault("index", index)
        merged_result.setdefault("total", total)
        merged_result["worker_returncode"] = returncode
        merged_result["worker_summary_path"] = str(worker_summary_path)
        merged_result["worker_log_path"] = str(worker_log_path)
        summary["results"].append(merged_result)
        if merged_result.get("status") == "ok":
            ok_count += 1

    summary["accepted_sft_length_records"].extend(worker_summary.get("accepted_sft_length_records") or [])
    summary["length_records"].extend(worker_summary.get("length_records") or [])
    summary["length_record_errors"].extend(worker_summary.get("length_record_errors") or [])
    return ok_count


def run_concurrent_instances(
    *,
    args: argparse.Namespace,
    logs_dir: Path,
    instance_ids: list[str],
    summary: dict[str, Any],
    started: float,
) -> int:
    worker_count = max(1, args.instance_workers)
    delay_seconds = max(0.0, args.instance_start_delay_seconds)
    worker_logs_dir = logs_dir / "instance_workers"
    worker_summaries_dir = logs_dir / "instance_worker_summaries"
    worker_artifacts_dir = args.artifact_root / "instance_workers"
    worker_logs_dir.mkdir(parents=True, exist_ok=True)
    worker_summaries_dir.mkdir(parents=True, exist_ok=True)
    worker_artifacts_dir.mkdir(parents=True, exist_ok=True)

    successful_instances = 0
    server = None
    active: list[dict[str, Any]] = []
    pending = list(enumerate(instance_ids, start=1))
    next_launch_time = time.monotonic()
    event_log = logs_dir / "instance_worker_events.jsonl"

    try:
        server = ensure_instance_slime_server(
            args=args,
            logs_dir=logs_dir,
            instance_id=instance_ids[0] if instance_ids else "none",
            index=1,
            total=len(instance_ids),
            server=server,
        )
        with SlimeKeepalive(
            base_url=slime_api_base(args),
            api_key=slime_api_key(args),
            model_name=args.student_model,
            interval_seconds=args.slime_keepalive_interval_seconds,
            log_path=logs_dir / "slime_keepalive.log",
        ):
            while pending or active:
                now = time.monotonic()
                launched = False
                while pending and len(active) < worker_count and now >= next_launch_time:
                    index, instance_id = pending.pop(0)
                    server = ensure_instance_slime_server(
                        args=args,
                        logs_dir=logs_dir,
                        instance_id=instance_id,
                        index=index,
                        total=len(instance_ids),
                        server=server,
                    )
                    stem = f"{index:03d}_of_{len(instance_ids):03d}_{safe_log_stem(instance_id)}"
                    worker_summary_path = worker_summaries_dir / f"{stem}.json"
                    worker_log_path = worker_logs_dir / f"{stem}.log"
                    worker_artifact_root = worker_artifacts_dir / stem
                    cmd, env = build_instance_worker_command(
                        args=args,
                        instance_id=instance_id,
                        worker_summary_path=worker_summary_path,
                        worker_artifact_root=worker_artifact_root,
                        base_url=slime_api_base(args),
                        api_key=slime_api_key(args),
                    )
                    log_file = worker_log_path.open("w", encoding="utf-8")
                    process = subprocess.Popen(
                        cmd,
                        cwd=REPO_ROOT,
                        env=env,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    active.append(
                        {
                            "process": process,
                            "log_file": log_file,
                            "instance_id": instance_id,
                            "index": index,
                            "summary_path": worker_summary_path,
                            "log_path": worker_log_path,
                            "started": time.perf_counter(),
                        }
                    )
                    append_jsonl(
                        event_log,
                        {
                            "event": "worker_started",
                            "instance_id": instance_id,
                            "index": index,
                            "total": len(instance_ids),
                            "pid": process.pid,
                            "log_path": str(worker_log_path),
                            "summary_path": str(worker_summary_path),
                        },
                    )
                    next_launch_time = time.monotonic() + delay_seconds
                    now = time.monotonic()
                    launched = True

                finished_any = False
                for worker in list(active):
                    process = worker["process"]
                    returncode = process.poll()
                    if returncode is None:
                        continue
                    worker["log_file"].close()
                    active.remove(worker)
                    worker_seconds = time.perf_counter() - worker["started"]
                    append_jsonl(
                        event_log,
                        {
                            "event": "worker_finished",
                            "instance_id": worker["instance_id"],
                            "index": worker["index"],
                            "total": len(instance_ids),
                            "pid": process.pid,
                            "returncode": returncode,
                            "seconds": worker_seconds,
                        },
                    )
                    successful_instances += merge_worker_summary(
                        summary=summary,
                        worker_summary_path=worker["summary_path"],
                        worker_log_path=worker["log_path"],
                        returncode=returncode,
                        instance_id=worker["instance_id"],
                        index=worker["index"],
                        total=len(instance_ids),
                        seconds=worker_seconds,
                    )
                    update_summary_progress(summary, started=started, successful_instances=successful_instances)
                    write_summary(args.summary_path, summary)
                    finished_any = True

                if not launched and not finished_any:
                    sleep_for = 2.0
                    if pending and len(active) < worker_count:
                        sleep_for = min(sleep_for, max(0.1, next_launch_time - time.monotonic()))
                    time.sleep(sleep_for)
    finally:
        for worker in active:
            process = worker["process"]
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
            worker["log_file"].close()
        stop_slime_server(server)

    return successful_instances


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.summary_path is None:
        args.summary_path = args.artifact_root / "summary.json"
    args.search_output_root.mkdir(parents=True, exist_ok=True)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)

    required_env = infer_litellm_api_env(args.teacher_model) if args.teacher_backend == "openai" else None
    if required_env and not args.teacher_api_key:
        raise RuntimeError(f"{required_env} is required for teacher model {args.teacher_model}.")

    instances = select_rebench_instances(
        subset=args.subset,
        split=args.split,
        instance_ids=list(args.instance_id) if args.instance_id else None,
        offset=args.offset,
        limit=args.limit,
    )
    instance_ids = [str(instance["instance_id"]) for instance in instances]
    summary: dict[str, Any] = {
        "settings": {
            "subset": args.subset,
            "split": args.split,
            "offset": args.offset,
            "limit": args.limit,
            "teacher_model": args.teacher_model,
            "teacher_backend": args.teacher_backend,
            "student_model": args.student_model,
            "search": {
                "m": args.search_m,
                "n": args.search_n,
                "p": args.search_p,
                "k": args.search_k,
                "max_rounds": args.search_max_rounds,
                "step_limit": args.search_step_limit,
                "calculate_ground_truth": True,
            },
            "search_gpus": args.search_gpus,
            "prepare_instance_images": not args.skip_prepare_instance_images,
            "image_prepare_workers": args.image_prepare_workers,
            "slime_lifecycle": "check_before_each_instance_restart_on_unhealthy",
            "slime_keepalive_interval_seconds": args.slime_keepalive_interval_seconds,
            "instance_workers": args.instance_workers,
            "instance_start_delay_seconds": args.instance_start_delay_seconds,
            "search_output_root": str(args.search_output_root),
            "artifact_root": str(args.artifact_root),
        },
        "instances": instance_ids,
        "results": [],
        "accepted_sft_length_records": [],
        "length_records": [],
        "length_record_errors": [],
        "length_summary": {},
    }
    write_summary(args.summary_path, summary)

    logs_dir = args.summary_path.parent / "logs"
    if not args.skip_prepare_instance_images:
        summary["image_prepare"] = prepare_instance_images(
            instances,
            log_path=logs_dir / "image_prepare.log",
            workers=args.image_prepare_workers,
        )
        write_summary(args.summary_path, summary)

    started = time.perf_counter()
    if args.instance_workers > 1 and len(instance_ids) > 1:
        successful_instances = run_concurrent_instances(
            args=args,
            logs_dir=logs_dir,
            instance_ids=instance_ids,
            summary=summary,
            started=started,
        )
        update_summary_progress(summary, started=started, successful_instances=successful_instances)
        write_summary(args.summary_path, summary)
        if successful_instances == 0:
            raise RuntimeError("all instances failed; see summary results for per-instance errors.")
        print(json.dumps({"summary_path": str(args.summary_path), "length_summary": summary["length_summary"]}, indent=2, ensure_ascii=False))
        return 1 if summary.get("failed_instances") else 0

    successful_instances = 0
    server = None
    try:
        for index, instance_id in enumerate(instance_ids, start=1):
            instance_started = time.perf_counter()
            try:
                server = ensure_instance_slime_server(
                    args=args,
                    logs_dir=logs_dir,
                    instance_id=instance_id,
                    index=index,
                    total=len(instance_ids),
                    server=server,
                )
                with SlimeKeepalive(
                    base_url=slime_api_base(args),
                    api_key=slime_api_key(args),
                    model_name=args.student_model,
                    interval_seconds=args.slime_keepalive_interval_seconds,
                    log_path=logs_dir / "slime_keepalive.log",
                ):
                    sft_bundle = collect_teacher_student_export(
                        instance_id=instance_id,
                        output_root=args.search_output_root / "teacher_student",
                        teacher_api_key=args.teacher_api_key,
                        student_model_name=args.student_model,
                        teacher_model_name=args.teacher_model,
                        teacher_backend=args.teacher_backend,
                        subset=args.subset,
                        split=args.split,
                        m=args.search_m,
                        n=args.search_n,
                        k=args.search_k,
                        p=args.search_p,
                        max_rounds=args.search_max_rounds,
                        step_limit=args.search_step_limit,
                    )
                successful_instances += 1
                accepted_sft_length_records = [
                    *(sample_length_record("policy", sample, sft_bundle.instance_id) for sample in sft_bundle.policy_samples),
                    *(sample_length_record("rubric", sample, sft_bundle.instance_id) for sample in sft_bundle.rubric_samples),
                ]
                length_records, length_record_errors = collect_raw_teacher_length_records(split_run_dirs(sft_bundle.run_dir))
                summary["accepted_sft_length_records"].extend(accepted_sft_length_records)
                summary["length_records"].extend(length_records)
                summary["length_record_errors"].extend(length_record_errors)
                summary["results"].append(
                    {
                        "status": "ok",
                        "instance_id": instance_id,
                        "index": index,
                        "total": len(instance_ids),
                        "run_dir": sft_bundle.run_dir,
                        "accepted_group_ids": list(sft_bundle.accepted_group_ids),
                        "policy_samples": len(sft_bundle.policy_samples),
                        "rubric_samples": len(sft_bundle.rubric_samples),
                        "seconds": time.perf_counter() - instance_started,
                        "metadata": dict(sft_bundle.metadata),
                    }
                )
            except Exception as exc:
                summary["results"].append(
                    {
                        "status": "error",
                        "instance_id": instance_id,
                        "index": index,
                        "total": len(instance_ids),
                        "seconds": time.perf_counter() - instance_started,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            finally:
                update_summary_progress(summary, started=started, successful_instances=successful_instances)
                write_summary(args.summary_path, summary)
    finally:
        stop_slime_server(server)

    if successful_instances == 0:
        raise RuntimeError("all instances failed; see summary results for per-instance errors.")
    write_summary(args.summary_path, summary)
    print(json.dumps({"summary_path": str(args.summary_path), "length_summary": summary["length_summary"]}, indent=2, ensure_ascii=False))
    return 1 if summary.get("failed_instances") else 0


if __name__ == "__main__":
    raise SystemExit(main())
