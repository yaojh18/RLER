from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from swe_agent.run.run_swe_agent import choose_gpus
from swe_agent.serving import SGLangChatService


REPO_ROOT = Path(__file__).resolve().parents[3]
AGENT_ROOT = REPO_ROOT / "agent"

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


def start_slime_server(args: argparse.Namespace, log_path: Path) -> ManagedServer | None:
    if args.use_existing_slime_server:
        os.environ["SEARCH_SWE_SLIME_API_BASE"] = args.slime_api_base
        os.environ["SEARCH_SWE_SLIME_API_KEY"] = args.slime_api_key
        return None

    name = f"slime-search-{int(time.time())}-{os.getpid()}"
    docker_cmd = [
        "docker", "run", "--rm", "--name", name, "--runtime", "nvidia", "--net=host", "--shm-size=64g",
        "-e", f"NVIDIA_VISIBLE_DEVICES={','.join(str(gpu) for gpu in choose_gpus(args.search_gpus))}",
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
                "--tensor-parallel-size 2",
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
