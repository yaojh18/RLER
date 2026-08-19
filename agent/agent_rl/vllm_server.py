"""Launch a local vLLM OpenAI-compatible server for agent rollouts."""

import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console


DEFAULT_VLLM_MAX_MODEL_LEN = 81920


@dataclass
class VLLMServerHandle:
    process: subprocess.Popen
    base_url: str


def _check_port(port: int, timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex(("localhost", port)) == 0


def _tail_text(path: Path, *, max_chars: int = 2000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    return text if len(text) <= max_chars else text[-max_chars:]


def _emit(message: str, *, logger: Optional[logging.Logger] = None) -> None:
    if logger:
        logger.info(message)
    else:
        Console().print(message)


def _warn(message: str, *, logger: Optional[logging.Logger] = None) -> None:
    if logger:
        logger.warning(message)
    else:
        Console().print(f"[yellow]⚠[/yellow] {message}")


def _error(message: str, *, logger: Optional[logging.Logger] = None) -> None:
    if logger:
        logger.error(message)
    else:
        Console().print(f"[red]❌[/red] {message}")


def _resolve_vllm_base_command() -> list[str] | None:
    sibling_cli = Path(sys.executable).with_name("vllm")
    if sibling_cli.exists():
        return [str(sibling_cli), "serve"]

    project_venv_cli = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "vllm"
    if project_venv_cli.exists():
        return [str(project_venv_cli), "serve"]

    discovered_cli = shutil.which("vllm")
    if discovered_cli:
        return [discovered_cli, "serve"]

    is_uv = (
        "uv" in sys.executable.lower()
        or os.environ.get("UV_PROJECT_ENVIRONMENT")
        or os.environ.get("VIRTUAL_ENV", "").endswith(".venv")
    )
    uv_binary = shutil.which("uv")
    if is_uv and uv_binary:
        return [uv_binary, "run", "--extra", "vllm", "vllm", "serve"]
    if sys.executable:
        return [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    return None


def launch_vllm_server_handle(
    model_name: str,
    port: int,
    gpu_id: int = 0,
    logger: Optional[logging.Logger] = None,
    *,
    served_model_name: Optional[str] = None,
    gpu_ids: Optional[list[int]] = None,
    max_model_len: int = DEFAULT_VLLM_MAX_MODEL_LEN,
    gpu_memory_utilization: float = 0.9,
    allow_long_max_model_len: bool = False,
    startup_timeout_seconds: int = 300,
    env_overrides: Optional[dict[str, str]] = None,
) -> VLLMServerHandle:
    """Launch vLLM and return its process and OpenAI endpoint metadata."""
    _emit(
        f"🚀 Launching vLLM server for model {model_name} on port {port}...",
        logger=logger,
    )
    vllm_base_cmd = _resolve_vllm_base_command()
    if not vllm_base_cmd:
        message = "vllm command not found; install the agent package with its vllm extra"
        _error(message, logger=logger)
        raise RuntimeError(message)

    selected_gpu_ids = (
        list(gpu_ids) if gpu_ids is not None else ([] if gpu_id is None else [gpu_id])
    )
    cmd = vllm_base_cmd + [
        model_name,
        "--port",
        str(port),
        "--dtype",
        "auto",
        "--generation-config",
        "vllm",
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    warmup_model_name = served_model_name or model_name
    if "qwen3" in warmup_model_name.lower():
        cmd.extend(["--reasoning-parser", "qwen3"])
    if served_model_name:
        cmd.extend(["--served-model-name", served_model_name])
    if len(selected_gpu_ids) > 1:
        cmd.extend(["--tensor-parallel-size", str(len(selected_gpu_ids))])

    env = os.environ.copy()
    path_entries = [entry for entry in env.get("PATH", "").split(os.pathsep) if entry]
    command_path = Path(vllm_base_cmd[0]).expanduser()
    if command_path.is_absolute() or "/" in vllm_base_cmd[0]:
        command_dir = str(
            (command_path.resolve() if command_path.exists() else command_path).parent
        )
        if command_dir not in path_entries:
            path_entries.insert(0, command_dir)
    project_venv_bin = Path(__file__).resolve().parents[1] / ".venv" / "bin"
    if project_venv_bin.exists() and str(project_venv_bin) not in path_entries:
        path_entries.insert(0, str(project_venv_bin))
    if path_entries:
        env["PATH"] = os.pathsep.join(path_entries)
    if selected_gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in selected_gpu_ids)
        _emit(
            f"Using CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} with "
            f"gpu_memory_utilization={gpu_memory_utilization}",
            logger=logger,
        )
    if allow_long_max_model_len:
        env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    if env_overrides:
        env.update(env_overrides)

    log_file = Path(f"/tmp/vllm_server_{port}.log")
    _emit(f"vLLM output for {model_name} will be logged to {log_file}", logger=logger)
    _emit("Waiting for vLLM server to become ready...", logger=logger)
    with log_file.open("w") as output:
        output.write(f"$ {' '.join(cmd)}\n\n")
        output.flush()
        process = subprocess.Popen(
            cmd,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )

    start_time = time.time()
    while time.time() - start_time < startup_timeout_seconds:
        if _check_port(port):
            base_url = f"http://127.0.0.1:{port}/v1"
            request = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=json.dumps(
                    {
                        "model": warmup_model_name,
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 1,
                        "temperature": 0.0,
                    }
                ).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer EMPTY",
                },
                method="POST",
            )
            warmup_error: Exception | None = None
            for _ in range(3):
                try:
                    with urllib.request.urlopen(
                        request, timeout=min(120, startup_timeout_seconds)
                    ) as response:
                        response.read()
                    _emit(f"✓ vLLM server started (PID: {process.pid})", logger=logger)
                    return VLLMServerHandle(
                        process=process,
                        base_url=base_url,
                    )
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                    warmup_error = exc
                    if process.poll() is not None:
                        break
                    time.sleep(2)
            log_tail = _tail_text(log_file)
            details = f"\n--- vLLM log tail ---\n{log_tail}" if log_tail else ""
            message = (
                f"vLLM opened port {port} but failed its warmup request for "
                f"{model_name}: {warmup_error}{details}"
            )
            _error(message, logger=logger)
            raise RuntimeError(message)

        if process.poll() is not None:
            log_tail = _tail_text(log_file)
            details = f"\n--- vLLM log tail ---\n{log_tail}" if log_tail else ""
            message = (
                f"vLLM failed to start (exit code: {process.returncode}); "
                f"check {log_file}{details}"
            )
            _error(message, logger=logger)
            raise RuntimeError(message)

        time.sleep(2)
        elapsed = int(time.time() - start_time)
        if elapsed > 0 and elapsed % 30 == 0:
            _emit(f"Still waiting for vLLM server ({elapsed}s)...", logger=logger)

    if process.poll() is None:
        _warn(
            "vLLM opened no port before the startup timeout; it may still be initializing",
            logger=logger,
        )
        return VLLMServerHandle(
            process=process,
            base_url=f"http://127.0.0.1:{port}/v1",
        )

    message = f"vLLM failed to start (exit code: {process.returncode})"
    _error(message, logger=logger)
    raise RuntimeError(message)
