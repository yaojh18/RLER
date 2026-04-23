#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
agent_root = REPO_ROOT / "agent"
if str(agent_root) not in sys.path:
    sys.path.insert(0, str(agent_root))

from slime.swe_agent.data_export import GRPODataExporter
from slime.swe_agent.teacher_student import collect_teacher_student_export
from swe_agent.run.search_swe_agent import build_arg_parser, run_search
from swe_agent.run.run_swe_agent import choose_gpus

DEFAULT_SLIME_IMAGE = "slimerl/slime:qwen35-route-fixed-20260416-v1"
DEFAULT_SLIME_PORT = 8021
DEFAULT_CONTEXT_LENGTH = 80960
DEFAULT_WATCHDOG_TIMEOUT = 3600
DEFAULT_MEM_FRACTION_STATIC = 0.85


@dataclass
class ManagedSlimeServer:
    container_name: str
    process: subprocess.Popen[str]
    log_path: Path
    log_file: object


def _build_rl_args(*, instance_id: str, output_root: Path) -> argparse.Namespace:
    args = build_arg_parser().parse_args([])
    args.backend = "slime"
    args.instance_id = [instance_id]
    args.subset = "verified"
    args.split = "test"
    args.output_root = output_root
    args.resume_run_dir = None
    args.workers = 1
    args.slime_model = "Qwen/Qwen3.5-9B"
    args.student_model = None
    args.m = 2
    args.k = 20
    args.p = 1
    args.max_rounds = 5
    args.calculate_gt_reward = True
    return args


def _start_managed_slime_server(args: argparse.Namespace, repo_root: Path) -> ManagedSlimeServer:
    logs_dir = args.output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "sglang_server.log"
    model_dir = args.model_dir.resolve()
    repo_root = repo_root.resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"slime model dir does not exist: {model_dir}")
    if repo_root not in model_dir.parents and model_dir != repo_root:
        raise RuntimeError(f"model dir must live under repo root for bind mount reuse: {model_dir}")

    subprocess.run(["docker", "image", "inspect", args.slime_image], check=True, stdout=subprocess.DEVNULL)
    container_name = f"slime-step4-debug-{int(time.time())}-{os.getpid()}"
    container_model_dir = Path("/workspace/rler") / model_dir.relative_to(repo_root)
    selected_gpus = choose_gpus(args.gpus)
    if not selected_gpus:
        raise RuntimeError("managed slime server requires at least one GPU")
    visible_devices = ",".join(str(gpu_id) for gpu_id in selected_gpus)
    launch_cmd = " ".join(
        [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            shlex.quote(str(container_model_dir)),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.slime_port),
            "--tensor-parallel-size",
            "2",
            "--context-length",
            str(args.context_length),
            "--mem-fraction-static",
            str(DEFAULT_MEM_FRACTION_STATIC),
            "--served-model-name",
            shlex.quote(args.model_name),
            "--reasoning-parser",
            "qwen3",
            "--disable-radix-cache",
            "--watchdog-timeout",
            str(args.watchdog_timeout),
        ]
    )
    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--runtime",
        "nvidia",
        "--net=host",
        "--shm-size=64g",
        "-e",
        f"NVIDIA_VISIBLE_DEVICES={visible_devices}",
        "-e",
        "FLASHINFER_USE_CUDA_NORM=1",
        "-v",
        f"{repo_root}:/workspace/rler",
        "-v",
        f"{repo_root / 'slime'}:/workspace/slime",
        "-v",
        f"{args.output_root}:/verify_outputs",
        "-w",
        "/workspace/slime",
        args.slime_image,
        "bash",
        "--noprofile",
        "--norc",
        "-lc",
        f"exec {launch_cmd}",
    ]
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        docker_cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.time() + 20 * 60
    models_url = f"http://127.0.0.1:{args.slime_port}/v1/models"
    while time.time() < deadline:
        if process.poll() is not None:
            log_tail = ""
            if log_path.exists():
                log_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                log_tail = "\n".join(log_lines[-40:])
            raise RuntimeError(
                "managed slime server exited before becoming ready"
                + (f"\nLast server log lines:\n{log_tail}" if log_tail else "")
            )
        try:
            with urllib.request.urlopen(models_url, timeout=5) as response:
                payload = json.loads(response.read().decode())
            (logs_dir / "sglang_models.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            os.environ["SEARCH_SWE_SLIME_API_BASE"] = f"http://127.0.0.1:{args.slime_port}"
            os.environ["SEARCH_SWE_SLIME_API_KEY"] = "EMPTY"
            return ManagedSlimeServer(container_name=container_name, process=process, log_path=log_path, log_file=log_file)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            time.sleep(5)
    if process.poll() is None:
        process.terminate()
        process.wait(timeout=30)
    log_file.close()
    raise RuntimeError(f"managed slime server did not become ready on port {args.slime_port}")


def _stop_managed_slime_server(server: ManagedSlimeServer | None) -> None:
    if not server:
        return
    if server.process.poll() is None:
        server.process.terminate()
        try:
            server.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.process.kill()
            server.process.wait(timeout=30)
    try:
        server.log_file.close()
    except Exception:
        pass
    subprocess.run(["docker", "rm", "-f", server.container_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _run_sft_verification(args: argparse.Namespace) -> None:
    teacher_bundle = collect_teacher_student_export(
        instance_id=args.instance_id,
        output_root=args.output_root / "sft_mixed",
        teacher_api_key=os.environ.get("GEMINI_API_KEY", ""),
        student_model_name=args.model_name,
        teacher_model_name="gemini/gemini-3.1-pro-preview",
        subset="verified",
        split="test",
        m=2,
        k=20,
        p=1,
        max_rounds=5,
        completion_max_tokens=4096,
    )
    sft_summary = {
        "run_dir": teacher_bundle.run_dir,
        "accepted_group_ids": teacher_bundle.accepted_group_ids,
        "policy_sft_record_count": len(teacher_bundle.policy_samples),
        "rubric_sft_record_count": len(teacher_bundle.rubric_samples),
    }
    if sft_summary["policy_sft_record_count"] == 0 or sft_summary["rubric_sft_record_count"] == 0:
        raise RuntimeError("offline SFT exporter returned empty records")
    (args.output_root / "sft_export_summary.json").write_text(
        json.dumps(sft_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _run_rl_verification(args: argparse.Namespace) -> None:
    rl_args = _build_rl_args(instance_id=args.instance_id, output_root=args.output_root / "rl_runtime")
    rl_results = run_search(rl_args, [args.instance_id])
    if not rl_results or rl_results[0].error:
        raise RuntimeError(f"rl runtime capture failed: {rl_results[0].error if rl_results else 'no result'}")
    export_bundle = GRPODataExporter(run_dir=Path(rl_results[0].run_dir)).export_bundle()
    rl_summary = {
        "run_dir": rl_results[0].run_dir,
        "policy_group_count": len(export_bundle.policy_groups),
        "rubric_group_count": len(export_bundle.rubric_groups),
        "policy_sample_count": sum(len(group.samples) for group in export_bundle.policy_groups),
        "rubric_sample_count": sum(len(group.samples) for group in export_bundle.rubric_groups),
    }
    if rl_summary["policy_group_count"] == 0 or rl_summary["rubric_group_count"] == 0:
        raise RuntimeError("rl runtime capture returned an incomplete bundle")
    (args.output_root / "rl_runtime_summary.json").write_text(
        json.dumps(rl_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify step4 SFT/RL data pipeline on a single instance.")
    parser.add_argument("--instance-id", default="psf__requests-1142")
    parser.add_argument("--mode", choices=["sft", "rl", "both"], default="both")
    parser.add_argument("--manage-slime-server", action="store_true")
    parser.add_argument("--slime-image", default=DEFAULT_SLIME_IMAGE)
    parser.add_argument("--slime-port", type=int, default=DEFAULT_SLIME_PORT)
    parser.add_argument("--gpus", default="auto:2")
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--watchdog-timeout", type=int, default=DEFAULT_WATCHDOG_TIMEOUT)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("slime/swe_agent/artifacts/step1_smoke_verify/qwen35_fullverify_20260415/models/Qwen3.5-9B"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("slime/swe_agent/artifacts/step4_data_pipeline_verify") / time.strftime("%Y%m%d-%H%M%S"),
    )
    os.environ["GEMINI_API_KEY"] = "AQ.Ab8RN6Ij-4PRghnr2J1tu3KSG3Rdzqvdg6B329fvsBt-fJWAHw"
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    server: ManagedSlimeServer | None = None
    try:
        if args.manage_slime_server:
            server = _start_managed_slime_server(args, REPO_ROOT)
        elif "SEARCH_SWE_SLIME_API_BASE" not in os.environ:
            raise RuntimeError("SEARCH_SWE_SLIME_API_BASE must be set when --manage-slime-server is not used")

        if args.mode in {"sft", "both"}:
            _run_sft_verification(args)

        if args.mode in {"rl", "both"}:
            if args.manage_slime_server and args.mode == "both":
                _stop_managed_slime_server(server)
                server = _start_managed_slime_server(args, REPO_ROOT)
            _run_rl_verification(args)
    finally:
        _stop_managed_slime_server(server)
    print(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
