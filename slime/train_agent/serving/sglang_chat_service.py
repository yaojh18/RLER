from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from agent_rl import register_model_service, unregister_model_service
from agent_rl.model_service import run_async
from agent_rl.run_utils import ModelRouteConfig, configure_model_route, run_chat_with_route_completion_async
from swe_agent.run.run_swe_agent import choose_gpus
from swe_agent.serving import SGLangChatService


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


def wait_for_slime_generation_ready(base_url: str, timeout: int = 900) -> None:
    deadline = time.time() + timeout
    normalized = base_url.rstrip("/")
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{normalized}/health_generate", timeout=10) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(5)
    raise RuntimeError(f"slime server did not pass /health_generate within {timeout}s")


def warmup_slime_policy_route(
    *,
    base_url: str,
    api_key: str,
    model_name: str,
    requests: int = 1,
    wait_for_health: bool = False,
    health_timeout: int = 900,
) -> list[float]:
    if requests <= 0:
        return []
    if wait_for_health:
        wait_for_slime_generation_ready(base_url, timeout=health_timeout)

    normalized_base_url = base_url.rstrip("/")
    target_urls = [normalized_base_url] * requests
    try:
        with urllib.request.urlopen(f"{normalized_base_url}/workers", timeout=5) as response:
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

    service_name = "_swe_agent_policy_warmup_service"
    route_name = "_swe_agent_policy_warmup"
    elapsed_times: list[float] = []
    try:
        for target_url in target_urls:
            register_model_service(
                service_name,
                SGLangChatService(base_url=target_url, api_key=api_key, default_model_name=model_name),
            )
            configure_model_route(route_name, ModelRouteConfig(backend="service", service_name=service_name, model_name=model_name))
            started = time.perf_counter()
            run_async(
                run_chat_with_route_completion_async(
                    route_name,
                    model_name=model_name,
                    messages=[
                        {"role": "system", "content": SWE_AGENT_POLICY_SYSTEM_PROMPT},
                        {"role": "user", "content": " "},
                    ],
                    temperature=0.5,
                    top_p=0.9,
                    max_tokens=64,
                    stop=None,
                    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                    drop_params=True,
                )
            )
            elapsed_times.append(time.perf_counter() - started)
    finally:
        unregister_model_service(service_name)
    return elapsed_times


def start_slime_server(args: argparse.Namespace, log_path: Path) -> ManagedServer | None:
    if args.use_existing_slime_server:
        os.environ["SEARCH_SWE_SLIME_API_BASE"] = args.slime_api_base
        os.environ["SEARCH_SWE_SLIME_API_KEY"] = args.slime_api_key
        return None

    name = f"slime-search-{int(time.time())}-{os.getpid()}"
    selected_gpus = choose_gpus(args.search_gpus)
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
