"""
Utility functions for checking and launching services required by workflows.

This module provides functions to:
- Check if services are running on specific ports
- Launch MCP servers and vLLM servers in the background
- Extract port numbers from URLs
"""

import logging
import os
import shutil
import socket
import subprocess
import sys
import time
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console


DEFAULT_VLLM_MAX_MODEL_LEN = 81920
_MODEL_LENGTH_FIELDS = (
    "max_position_embeddings",
    "model_max_length",
    "n_positions",
    "max_seq_len",
    "seq_length",
    "max_sequence_length",
)


@dataclass
class VLLMServerHandle:
    process: subprocess.Popen
    command: list[str]
    log_file: Path
    port: int
    base_url: str
    model_name: str
    max_model_len: int
    gpu_id: int | None
    gpu_ids: list[int] | None = None


def check_port(port: int, timeout: float = 1.0) -> bool:
    """Check if a port is listening."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    result = sock.connect_ex(("localhost", port))
    sock.close()
    return result == 0


def extract_port_from_url(url_str: str) -> Optional[int]:
    """Extract port number from URL string."""
    # Remove the scheme (http:// or https://) to avoid false positives
    if "://" in url_str:
        url_str = url_str.split("://", 1)[1]

    url = url_str.rstrip("/")

    if ":" in url:
        port_str = url.split(":")[-1].split("/")[0]
        return int(port_str)
    return None


def _tail_text(path: Path, *, max_chars: int = 2000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _emit(message: str, *, logger: Optional[logging.Logger] = None, style: str | None = None) -> None:
    if logger:
        logger.info(message)
        return
    console = Console()
    console.print(message if style is None else f"[{style}]{message}[/{style}]")


def _warn(message: str, *, logger: Optional[logging.Logger] = None) -> None:
    if logger:
        logger.warning(message)
        return
    Console().print(f"[yellow]⚠[/yellow] {message}")


def _error(message: str, *, logger: Optional[logging.Logger] = None) -> None:
    if logger:
        logger.error(message)
        return
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


def launch_mcp_server(
    port: int = 8000, logger: Optional[logging.Logger] = None
) -> Optional[subprocess.Popen]:
    """Launch MCP server in background."""
    console = Console()

    if logger:
        logger.info(f"Launching MCP server on port {port}...")
    else:
        console.print(
            f"[cyan]🚀[/cyan] Launching MCP server on port [bold]{port}[/bold]..."
        )

    env = os.environ.copy()
    env["MCP_CACHE_DIR"] = (
        f".cache-{os.uname().nodename if hasattr(os, 'uname') else 'localhost'}"
    )

    log_file = Path(f"/tmp/mcp_server_{port}.log")
    if logger:
        logger.info(f"MCP server output will be logged to {log_file}")
    else:
        console.print(f"[dim]📋 MCP server output will be logged to {log_file}[/dim]")

    with open(log_file, "w") as f:
        process = subprocess.Popen(
            [sys.executable, "-m", "dr_agent.mcp_backend.main", "--port", str(port)],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )

    # Wait for server to start
    if logger:
        logger.info("Waiting for MCP server to start...")
    else:
        console.print("[yellow]⏳[/yellow] Waiting for MCP server to start...")

    for _ in range(20):
        time.sleep(0.5)
        if check_port(port):
            if logger:
                logger.info(f"MCP server started (PID: {process.pid})")
            else:
                console.print(
                    f"[green]✓[/green] MCP server started [dim](PID: {process.pid})[/dim]"
                )
            return process

    if process.poll() is None:
        if logger:
            logger.warning(
                "MCP server process started but port check failed. Continuing anyway..."
            )
        else:
            console.print(
                "[yellow]⚠[/yellow] MCP server process started but port check failed. Continuing anyway..."
            )
        return process
    else:
        if logger:
            logger.error(
                f"MCP server failed to start (exit code: {process.returncode}). Check logs: {log_file}"
            )
        else:
            console.print(
                f"[red]❌[/red] MCP server failed to start [dim](exit code: {process.returncode})[/dim]"
            )
            console.print(f"[dim]Check logs: {log_file}[/dim]")
        return None


def launch_vllm_server(
    model_name: str,
    port: int,
    gpu_id: int = 0,
    logger: Optional[logging.Logger] = None,
    *,
    max_model_len: int = DEFAULT_VLLM_MAX_MODEL_LEN,
    gpu_memory_utilization: float = 0.9,
    allow_long_max_model_len: bool = False,
    startup_timeout_seconds: int = 300,
    env_overrides: Optional[dict[str, str]] = None,
) -> Optional[subprocess.Popen]:
    """Launch vLLM server in background and return the process handle."""
    try:
        handle = launch_vllm_server_handle(
            model_name=model_name,
            port=port,
            gpu_id=gpu_id,
            logger=logger,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            allow_long_max_model_len=allow_long_max_model_len,
            startup_timeout_seconds=startup_timeout_seconds,
            env_overrides=env_overrides,
        )
    except RuntimeError:
        return None
    return handle.process


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
    """Launch vLLM server in background and return structured launch metadata."""
    _emit(
        f"🚀 Launching vLLM server for model {model_name} on port {port}...",
        logger=logger,
    )

    vllm_base_cmd = _resolve_vllm_base_command()
    if not vllm_base_cmd:
        message = "vllm command not found. Install vllm with: uv pip install -e '.[vllm]' or uv pip install 'dr_agent[vllm]'"
        _error(message, logger=logger)
        raise RuntimeError(message)
    
    selected_gpu_ids = list(gpu_ids) if gpu_ids is not None else ([] if gpu_id is None else [gpu_id])
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
    normalized_qwen_name = (served_model_name or model_name).lower()
    if "qwen3" in normalized_qwen_name:
        cmd.extend(["--reasoning-parser", "qwen3"])
    if served_model_name:
        cmd.extend(["--served-model-name", served_model_name])
    if len(selected_gpu_ids) > 1:
        cmd.extend(["--tensor-parallel-size", str(len(selected_gpu_ids))])

    env = os.environ.copy()
    path_entries = [entry for entry in env.get("PATH", "").split(os.pathsep) if entry]
    command_path = Path(vllm_base_cmd[0]).expanduser()
    if command_path.is_absolute() or "/" in vllm_base_cmd[0]:
        command_bin_dir = str((command_path.resolve() if command_path.exists() else command_path).parent)
        if command_bin_dir not in path_entries:
            path_entries.insert(0, command_bin_dir)
    project_venv_bin = Path(__file__).resolve().parents[1] / ".venv" / "bin"
    if project_venv_bin.exists() and str(project_venv_bin) not in path_entries:
        path_entries.insert(0, str(project_venv_bin))
    if path_entries:
        env["PATH"] = os.pathsep.join(path_entries)
    if selected_gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in selected_gpu_ids)
        _emit(
            f"Using CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} with gpu_memory_utilization={gpu_memory_utilization}",
            logger=logger,
        )
    if allow_long_max_model_len:
        env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    if env_overrides:
        env.update(env_overrides)

    log_file = Path(f"/tmp/vllm_server_{port}.log")
    _emit(f"vLLM output for {model_name} will be logged to {log_file}", logger=logger)
    _emit("Waiting for vLLM server to become ready (this may take a few minutes)...", logger=logger)

    with open(log_file, "w") as f:
        f.write(f"$ {' '.join(cmd)}\n\n")
        f.flush()
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )

    start_time = time.time()
    while time.time() - start_time < startup_timeout_seconds:
        if check_port(port):
            base_url = f"http://127.0.0.1:{port}/v1"
            warmup_model_name = served_model_name or model_name
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
                headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
                method="POST",
            )
            warmup_error: Exception | None = None
            for _ in range(3):
                try:
                    with urllib.request.urlopen(request, timeout=min(120, startup_timeout_seconds)) as response:
                        response.read()
                    _emit(f"✓ vLLM server started (PID: {process.pid})", logger=logger)
                    return VLLMServerHandle(
                        process=process,
                        command=cmd,
                        log_file=log_file,
                        port=port,
                        base_url=base_url,
                        model_name=warmup_model_name,
                        max_model_len=max_model_len,
                        gpu_id=selected_gpu_ids[0] if selected_gpu_ids else None,
                        gpu_ids=selected_gpu_ids,
                    )
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                    warmup_error = exc
                    if process.poll() is not None:
                        break
                    time.sleep(2)
            log_tail = _tail_text(log_file)
            details = f"\n--- vLLM log tail ---\n{log_tail}" if log_tail else ""
            message = (
                f"vLLM server opened port {port} but failed a warmup chat request for model {model_name}: "
                f"{warmup_error}{details}"
            )
            _error(message, logger=logger)
            raise RuntimeError(message)

        if process.poll() is not None:
            log_tail = _tail_text(log_file)
            details = f"\n--- vLLM log tail ---\n{log_tail}" if log_tail else ""
            message = f"vLLM server failed to start (exit code: {process.returncode}). Check logs: {log_file}{details}"
            _error(message, logger=logger)
            raise RuntimeError(message)

        time.sleep(2)
        elapsed = int(time.time() - start_time)
        if elapsed > 0 and elapsed % 30 == 0:
            _emit(f"Still waiting for vLLM server ({elapsed}s)...", logger=logger)

    if process.poll() is None:
        _warn(
            "vLLM server process started but port check timed out. It may still be initializing...",
            logger=logger,
        )
        return VLLMServerHandle(
            process=process,
            command=cmd,
            log_file=log_file,
            port=port,
            base_url=f"http://127.0.0.1:{port}/v1",
            model_name=model_name,
            max_model_len=max_model_len,
            gpu_id=selected_gpu_ids[0] if selected_gpu_ids else None,
            gpu_ids=selected_gpu_ids,
        )

    message = f"vLLM server failed to start (exit code: {process.returncode})"
    _error(message, logger=logger)
    raise RuntimeError(message)
