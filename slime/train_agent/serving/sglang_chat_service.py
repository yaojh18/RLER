from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from swe_agent.run.run_swe_agent import choose_gpus, query_gpu_inventory


REPO_ROOT = Path(__file__).resolve().parents[3]
AGENT_ROOT = REPO_ROOT / "agent"
SWE_AGENT_POLICY_SYSTEM_PROMPT = """You are a helpful assistant that can interact multiple times with a computer shell to solve programming tasks.
Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).

Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <format_example>.

<format_example>
THOUGHT: Your reasoning and analysis here

```mswea_bash_command
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected."""


@dataclass
class ManagedServer:
    name: str
    process: subprocess.Popen[str]
    log_file: object


def to_container_path(path: Path) -> str:
    resolved = path.resolve()
    if resolved == REPO_ROOT.resolve() or REPO_ROOT.resolve() in resolved.parents:
        return str(Path("/workspace/rler") / resolved.relative_to(REPO_ROOT.resolve()))
    return str(resolved)


def extra_mount_args(paths: list[Path]) -> list[str]:
    mounts: list[str] = []
    for path in paths:
        resolved = path.resolve()
        if resolved == REPO_ROOT.resolve() or REPO_ROOT.resolve() in resolved.parents:
            continue
        mounts.extend(["-v", f"{resolved}:{resolved}:ro"])
    return mounts


def _auto_gpu_count(gpu_spec: str) -> int | None:
    spec = gpu_spec.strip().lower()
    if not spec.startswith("auto"):
        return None
    if ":" not in spec:
        return 1
    return int(spec.split(":", 1)[1])


def _gpu_numa_affinity() -> dict[int, str]:
    completed = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, check=True)
    rows = [
        [cell.strip() for cell in re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", line).split("\t")]
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    if not rows:
        return {}
    header = rows[0]
    try:
        numa_index = header.index("NUMA Affinity")
    except ValueError:
        return {}
    affinity: dict[int, str] = {}
    for row in rows[1:]:
        if not row or not row[0].startswith("GPU") or len(row) <= numa_index:
            continue
        try:
            gpu_index = int(row[0][3:])
        except ValueError:
            continue
        numa = row[numa_index]
        if numa and numa != "N/A":
            affinity[gpu_index] = numa
    return affinity


def choose_search_gpus(gpu_spec: str) -> list[int]:
    selected_gpus = choose_gpus(gpu_spec)
    count = _auto_gpu_count(gpu_spec)
    if count is None or count <= 1:
        return selected_gpus
    try:
        free_memory = {int(record["index"]): int(record["memory_free"]) for record in query_gpu_inventory()}
        numa_affinity = _gpu_numa_affinity()
    except (OSError, subprocess.CalledProcessError, ValueError):
        return selected_gpus
    groups: dict[str, list[int]] = {}
    for gpu_index, numa in numa_affinity.items():
        if gpu_index in free_memory:
            groups.setdefault(numa, []).append(gpu_index)
    candidates: list[list[int]] = []
    for group in groups.values():
        if len(group) < count:
            continue
        ranked = sorted(group, key=lambda gpu_index: free_memory[gpu_index], reverse=True)
        candidates.append(ranked[:count])
    if not candidates:
        return selected_gpus
    best = max(candidates, key=lambda group: (min(free_memory[gpu] for gpu in group), sum(free_memory[gpu] for gpu in group)))
    return sorted(best)


def _post_warmup_chat_completion(target_url: str, api_key: str, model_name: str) -> None:
    payload = json.dumps(
        {
            "model": model_name,
            "messages": [
                {"role": "system", "content": SWE_AGENT_POLICY_SYSTEM_PROMPT},
                {"role": "user", "content": " "},
            ],
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": True},
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{target_url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            response.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return


def start_slime_policy_route_warmup(*, base_url: str, api_key: str, model_name: str, requests: int = 1) -> None:
    if requests <= 0:
        return

    def run_warmup() -> None:
        target_urls = [base_url.rstrip("/")] * requests
        try:
            with urllib.request.urlopen(f"{base_url.rstrip('/')}/workers", timeout=5) as response:
                worker_payload = json.loads(response.read().decode("utf-8"))
            worker_urls = [
                str(worker["url"]).rstrip("/")
                for worker in worker_payload.get("workers", [])
                if isinstance(worker, dict) and worker.get("url")
            ]
            if worker_urls:
                target_urls = [worker_urls[index % len(worker_urls)] for index in range(requests)]
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            pass

        for target_url in target_urls:
            threading.Thread(
                target=_post_warmup_chat_completion,
                kwargs={"target_url": target_url, "api_key": api_key, "model_name": model_name},
                daemon=True,
            ).start()

    threading.Thread(target=run_warmup, daemon=True).start()


def start_slime_server(args: argparse.Namespace, log_path: Path) -> ManagedServer | None:
    if args.use_existing_slime_server:
        os.environ["SEARCH_SWE_SLIME_API_BASE"] = args.slime_api_base
        os.environ["SEARCH_SWE_SLIME_API_KEY"] = args.slime_api_key
        return None

    name = f"slime-search-{int(time.time())}-{os.getpid()}"
    selected_gpus = choose_search_gpus(args.search_gpus)
    docker_cmd = [
        "docker", "run", "--rm", "--name", name, "--runtime", "nvidia", "--net=host", "--shm-size=64g",
        "-e", f"NVIDIA_VISIBLE_DEVICES={','.join(str(gpu) for gpu in selected_gpus)}",
        "-e", "FLASHINFER_USE_CUDA_NORM=1",
        *extra_mount_args([args.model_dir]),
        "-v", f"{REPO_ROOT.resolve()}:/workspace/rler",
        "-w", "/workspace/rler/slime",
        args.slime_image,
        "bash", "--noprofile", "--norc", "-lc",
        " ".join(
            [
                "exec python3 -m sglang.launch_server",
                "--model-path", shlex.quote(to_container_path(args.model_dir)),
                "--host 127.0.0.1",
                "--port", str(args.slime_port),
                "--tensor-parallel-size", str(len(selected_gpus)),
                "--context-length 80960",
                "--served-model-name", shlex.quote(args.student_model),
                "--reasoning-parser qwen3",
                "--disable-radix-cache",
                "--watchdog-timeout 3600",
            ]
        ),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(docker_cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + args.serving_timeout
    while time.time() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(f"slime server exited before ready\n{tail[-8000:]}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{args.slime_port}/v1/models", timeout=5) as response:
                if response.status == 200:
                    os.environ["SEARCH_SWE_SLIME_API_BASE"] = f"http://127.0.0.1:{args.slime_port}"
                    os.environ["SEARCH_SWE_SLIME_API_KEY"] = "EMPTY"
                    return ManagedServer(name=name, process=process, log_file=log_file)
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(5)
    process.terminate()
    process.wait(timeout=30)
    log_file.close()
    raise RuntimeError("slime server did not become ready")


def stop_slime_server(server: ManagedServer | None) -> None:
    if server is None:
        return
    if server.process.poll() is None:
        server.process.terminate()
        try:
            server.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.process.kill()
            server.process.wait(timeout=30)
    server.log_file.close()
    subprocess.run(["docker", "rm", "-f", server.name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
