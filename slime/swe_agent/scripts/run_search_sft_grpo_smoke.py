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
AGENT_ROOT = REPO_ROOT / "agent"
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from slime.swe_agent.teacher_student import collect_teacher_student_export
from swe_agent.run.run_swe_agent import choose_gpus
from swe_agent.run.search_swe_agent import build_arg_parser, run_search
import swe_agent.run.search_swe_agent as search_module


DEFAULT_INSTANCE_ID = "pallets__click-2380"
DEFAULT_SERVING_IMAGE = "slimerl/slime:qwen35-route-fixed-20260416-v1"
DEFAULT_TRAINING_IMAGE = "slimerl/slime:qwen35-grpo-fixed-20260415-v4"
DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-9B"
DEFAULT_MODEL_DIR = REPO_ROOT / "slime/swe_agent/artifacts/step1_smoke_verify/qwen35_fullverify_20260415/models/Qwen3.5-9B"
DEFAULT_MODEL_TORCH_DIST_DIR = REPO_ROOT / "slime/swe_agent/artifacts/step1_smoke_verify/qwen35_fullverify_20260415/models/Qwen3.5-9B_torch_dist"


@dataclass
class ManagedServer:
    container_name: str
    process: subprocess.Popen[str]
    log_file: object
    log_path: Path


def _start_slime_server(*, image: str, repo_root: Path, model_dir: Path, model_name: str, port: int, gpu_spec: str, log_path: Path) -> ManagedServer:
    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        f"slime-search-{int(time.time())}-{os.getpid()}",
        "--runtime",
        "nvidia",
        "--net=host",
        "--shm-size=64g",
        "-e",
        f"NVIDIA_VISIBLE_DEVICES={','.join(str(gpu) for gpu in choose_gpus(gpu_spec))}",
        "-e",
        "FLASHINFER_USE_CUDA_NORM=1",
        "-v",
        f"{repo_root}:/workspace/rler",
        "-w",
        "/workspace/rler/slime",
        image,
        "bash",
        "--noprofile",
        "--norc",
        "-lc",
        " ".join(
            [
                "exec",
                "python3",
                "-m",
                "sglang.launch_server",
                "--model-path",
                shlex.quote(str(Path("/workspace/rler") / model_dir.resolve().relative_to(repo_root))),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--tensor-parallel-size",
                "2",
                "--context-length",
                "80960",
                "--served-model-name",
                shlex.quote(model_name),
                "--reasoning-parser",
                "qwen3",
                "--disable-radix-cache",
                "--watchdog-timeout",
                "3600",
            ]
        ),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(docker_cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 1200
    while time.time() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(f"slime server exited before ready\n{tail[-8000:]}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
                if response.status == 200:
                    os.environ["SEARCH_SWE_SLIME_API_BASE"] = f"http://127.0.0.1:{port}"
                    os.environ["SEARCH_SWE_SLIME_API_KEY"] = "EMPTY"
                    return ManagedServer(
                        container_name=docker_cmd[4],
                        process=process,
                        log_file=log_file,
                        log_path=log_path,
                    )
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            time.sleep(5)
    process.terminate()
    process.wait(timeout=30)
    log_file.close()
    raise RuntimeError("slime server did not become ready")


def _stop_slime_server(server: ManagedServer | None) -> None:
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
    subprocess.run(["docker", "rm", "-f", server.container_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _build_student_search_args(
    *,
    instance_id: str,
    subset: str,
    split: str,
    output_root: Path,
    model_name: str,
    m: int,
    k: int,
    p: int,
    max_rounds: int,
) -> argparse.Namespace:
    args = build_arg_parser().parse_args([])
    args.backend = "slime"
    args.instance_id = [instance_id]
    args.subset = subset
    args.split = split
    args.output_root = output_root
    args.resume_run_dir = None
    args.workers = 1
    args.slime_model = model_name
    args.student_model = None
    args.m = m
    args.k = k
    args.p = p
    args.max_rounds = max_rounds
    return args


def _run_command(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    command_env = os.environ.copy()
    if env is not None:
        command_env.update(env)
    pythonpath_parts = [str(REPO_ROOT / "slime"), str(REPO_ROOT), str(AGENT_ROOT)]
    existing_pythonpath = command_env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    command_env["PYTHONPATH"] = ":".join(pythonpath_parts)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=command_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{tail[-8000:]}")


def _run_container_script(*, image: str, repo_root: Path, gpu_spec: str, script_path: str, log_path: Path, env_vars: dict[str, str]) -> None:
    gpu_list = ",".join(str(gpu) for gpu in choose_gpus(gpu_spec))
    env_args = []
    merged_env = {
        "RAY_USE_MULTIPROCESSING_CPU_COUNT": "1",
        "RAY_DISABLE_DOCKER_CPU_WARNING": "1",
        **env_vars,
    }
    for key, value in merged_env.items():
        env_args.extend(["-e", f"{key}={value}"])
    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--runtime",
        "nvidia",
        "--ipc=host",
        "--shm-size=64g",
        "--ulimit",
        "nofile=1048576:1048576",
        "-e",
        f"CUDA_VISIBLE_DEVICES={gpu_list}",
        *env_args,
        "-v",
        f"{repo_root}:/workspace/rler",
        "-w",
        "/workspace/rler/slime",
        image,
        "bash",
        "--noprofile",
        "--norc",
        "-lc",
        f"bash {script_path}",
    ]
    _run_command(docker_cmd, log_path)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize_run(run_dir: Path) -> dict[str, object]:
    evaluation_path = run_dir / "evaluation.json"
    model_patch_path = run_dir / "model_patch.json"
    round_files = sorted((run_dir / "rubrics").glob("round_*.json"))
    patch_chars = 0
    if model_patch_path.exists():
        payload = _load_json(model_patch_path)
        patch_chars = len(next(iter(payload.values())).get("model_patch", "")) if payload else 0
    rubric_count = 0
    for round_file in round_files:
        round_payload = _load_json(round_file)
        rubric_count += len(round_payload.get("generated", []))
    summary = {
        "run_dir": str(run_dir),
        "evaluation_exists": evaluation_path.exists(),
        "patch_chars": patch_chars,
        "rubric_count": rubric_count,
        "round_count": len(round_files),
    }
    if evaluation_path.exists():
        summary["evaluation"] = _load_json(evaluation_path)
    return summary


def _to_container_path(path: Path) -> str:
    return str(Path("/workspace/rler") / path.resolve().relative_to(REPO_ROOT))


def _assert_checkpoint(save_dir: Path, log_path: Path) -> dict[str, object]:
    latest_path = save_dir / "latest_checkpointed_iteration.txt"
    if not latest_path.exists():
        raise RuntimeError(f"missing checkpoint marker under {save_dir}")
    iteration = int(latest_path.read_text(encoding="utf-8").strip())
    iter_dir = save_dir / f"iter_{iteration:07d}"
    if not (iter_dir / "common.pt").exists():
        raise RuntimeError(f"missing common.pt under {iter_dir}")
    if "successfully saved checkpoint from iteration" not in log_path.read_text(encoding="utf-8", errors="replace"):
        raise RuntimeError(f"training log {log_path} does not contain checkpoint save confirmation")
    return {"save_dir": str(save_dir), "iteration": iteration, "log_path": str(log_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run search -> export -> SFT/GRPO smoke on one SWE-rebench-V2 instance.")
    parser.add_argument("--instance-id", default=DEFAULT_INSTANCE_ID)
    parser.add_argument("--subset", default="rebench_v2")
    parser.add_argument("--split", default="train")
    parser.add_argument("--teacher-model", default="openai/gpt-5.4")
    parser.add_argument("--student-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--teacher-api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--teacher-run-dir", type=Path, default=None)
    parser.add_argument("--student-run-dir", type=Path, default=None)
    parser.add_argument("--serving-image", default=DEFAULT_SERVING_IMAGE)
    parser.add_argument("--training-image", default=DEFAULT_TRAINING_IMAGE)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-torch-dist-dir", type=Path, default=DEFAULT_MODEL_TORCH_DIST_DIR)
    parser.add_argument("--search-gpus", default="2,3")
    parser.add_argument("--sft-train-gpus", default="2,3,4,5")
    parser.add_argument("--grpo-train-gpus", default="2,3,4,5")
    parser.add_argument("--slime-port", type=int, default=8032)
    parser.add_argument("--search-m", type=int, default=2)
    parser.add_argument("--search-k", type=int, default=20)
    parser.add_argument("--search-p", type=int, default=1)
    parser.add_argument("--search-max-rounds", type=int, default=1)
    parser.add_argument("--policy-sft-limit", type=int, default=8)
    parser.add_argument("--rubric-sft-limit", type=int, default=1)
    parser.add_argument("--policy-rollout-limit", type=int, default=6)
    parser.add_argument("--rubric-rollout-limit", type=int, default=2)
    parser.add_argument(
        "--search-output-root",
        type=Path,
        default=REPO_ROOT / "agent/outputs/search_outputs" / f"stage5_search_training_{time.strftime('%Y%m%d-%H%M%S')}",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPO_ROOT / "slime/swe_agent/artifacts" / f"stage5_search_training_{time.strftime('%Y%m%d-%H%M%S')}",
    )
    args = parser.parse_args()

    args.search_output_root.mkdir(parents=True, exist_ok=True)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    logs_dir = args.artifact_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    server = None
    try:
        if args.teacher_run_dir is None or args.student_run_dir is None:
            server = _start_slime_server(
                image=args.serving_image,
                repo_root=REPO_ROOT,
                model_dir=args.model_dir,
                model_name=args.student_model,
                port=args.slime_port,
                gpu_spec=args.search_gpus,
                log_path=logs_dir / "slime_server.log",
            )
            search_module.SLIME_API_BASE = os.environ["SEARCH_SWE_SLIME_API_BASE"]
            search_module.SLIME_API_KEY = os.environ["SEARCH_SWE_SLIME_API_KEY"]

        if args.teacher_run_dir is not None:
            teacher_run_dir = args.teacher_run_dir
        else:
            teacher_bundle = collect_teacher_student_export(
                instance_id=args.instance_id,
                output_root=args.search_output_root / "teacher_student",
                teacher_api_key=args.teacher_api_key,
                student_model_name=args.student_model,
            teacher_model_name=args.teacher_model,
            subset=args.subset,
            split=args.split,
            m=args.search_m,
            k=args.search_k,
            p=args.search_p,
            max_rounds=args.search_max_rounds,
        )
            teacher_run_dir = Path(teacher_bundle.run_dir)
        teacher_summary = _summarize_run(teacher_run_dir)
        if not teacher_summary["evaluation_exists"] or teacher_summary["rubric_count"] <= 0:
            raise RuntimeError(f"teacher/student search did not produce usable artifacts: {teacher_summary}")

        _run_command(
            [
                sys.executable,
                str(REPO_ROOT / "slime/swe_agent/run/prepare_search_sft_data.py"),
                "--run-dir",
                str(teacher_run_dir),
                "--output-dir",
                str(args.artifact_root / "datasets"),
                "--pad-to-multiple",
                "4",
            ],
            logs_dir / "prepare_sft_data.log",
            env=os.environ.copy(),
        )

        if args.student_run_dir is not None:
            student_run_dir = args.student_run_dir
        else:
            student_args = _build_student_search_args(
                instance_id=args.instance_id,
                subset=args.subset,
                split=args.split,
                output_root=args.search_output_root / "student_only",
                model_name=args.student_model,
                m=args.search_m,
                k=args.search_k,
                p=args.search_p,
                max_rounds=args.search_max_rounds,
            )
            student_results = run_search(student_args, [args.instance_id])
            if not student_results or student_results[0].error:
                raise RuntimeError(f"student-only search failed: {student_results[0].error if student_results else 'no result'}")
            student_run_dir = Path(student_results[0].run_dir)
        student_summary = _summarize_run(student_run_dir)
        if not student_summary["evaluation_exists"] or student_summary["rubric_count"] <= 0:
            raise RuntimeError(f"student-only search did not produce usable artifacts: {student_summary}")

        _run_command(
            [
                sys.executable,
                str(REPO_ROOT / "slime/swe_agent/run/prepare_search_grpo_rollout.py"),
                "--run-dir",
                str(student_run_dir),
                "--hf-checkpoint",
                str(args.model_dir),
                "--output-path",
                str(args.artifact_root / "grpo_policy_rollout.pt"),
                "--target",
                "policy",
                "--pad-to-multiple",
                "2",
            ],
            logs_dir / "prepare_policy_grpo.log",
            env=os.environ.copy(),
        )
        _run_command(
            [
                sys.executable,
                str(REPO_ROOT / "slime/swe_agent/run/prepare_search_grpo_rollout.py"),
                "--run-dir",
                str(student_run_dir),
                "--hf-checkpoint",
                str(args.model_dir),
                "--output-path",
                str(args.artifact_root / "grpo_rubric_rollout.pt"),
                "--target",
                "rubric",
                "--pad-to-multiple",
                "2",
            ],
            logs_dir / "prepare_rubric_grpo.log",
            env=os.environ.copy(),
        )
        _run_command(
            [
                sys.executable,
                str(REPO_ROOT / "slime/swe_agent/run/prepare_smoke_subsets.py"),
                "--model-dir",
                str(args.model_dir),
                "--policy-sft-path",
                str(args.artifact_root / "datasets" / "policy_sft.jsonl"),
                "--rubric-sft-path",
                str(args.artifact_root / "datasets" / "rubric_sft.jsonl"),
                "--policy-rollout-path",
                str(args.artifact_root / "grpo_policy_rollout.pt"),
                "--rubric-rollout-path",
                str(args.artifact_root / "grpo_rubric_rollout.pt"),
                "--output-dir",
                str(args.artifact_root / "smoke_subsets"),
                "--policy-sft-limit",
                str(args.policy_sft_limit),
                "--rubric-sft-limit",
                str(args.rubric_sft_limit),
                "--policy-rollout-limit",
                str(args.policy_rollout_limit),
                "--rubric-rollout-limit",
                str(args.rubric_rollout_limit),
            ],
            logs_dir / "prepare_smoke_subsets.log",
            env=os.environ.copy(),
        )
    finally:
        _stop_slime_server(server)

    sft_summary = _load_json(args.artifact_root / "datasets" / "sft_export_summary.json")
    smoke_subset_summary = _load_json(args.artifact_root / "smoke_subsets" / "smoke_subset_summary.json")
    policy_dataset_size = int(smoke_subset_summary["policy_sft"]["selected_count"])
    rubric_dataset_size = int(smoke_subset_summary["rubric_sft"]["selected_count"])
    if policy_dataset_size <= 0 or rubric_dataset_size <= 0:
        raise RuntimeError(f"SFT export is empty: {sft_summary}")

    _run_container_script(
        image=args.training_image,
        repo_root=REPO_ROOT,
        gpu_spec=args.sft_train_gpus,
        script_path="/workspace/rler/slime/swe_agent/run/run_policy_sft_smoke.sh",
        log_path=logs_dir / "policy_sft_train.log",
        env_vars={
            "PROMPT_DATA": _to_container_path(args.artifact_root / "smoke_subsets" / "policy_sft_smoke.jsonl"),
            "DATASET_SIZE": str(policy_dataset_size),
            "GLOBAL_BATCH_SIZE": str(policy_dataset_size),
            "NUM_GPUS": "4",
            "MODEL_DIR": _to_container_path(args.model_dir),
            "MODEL_TORCH_DIST_DIR": _to_container_path(args.model_torch_dist_dir),
            "SAVE_DIR": _to_container_path(args.artifact_root / "policy_sft"),
            "WANDB_DIR": _to_container_path(args.artifact_root / "policy_sft" / "wandb"),
            "TENSOR_MODEL_PARALLEL_SIZE": "2",
            "MAX_TOKENS_PER_GPU": "8192",
        },
    )
    _run_container_script(
        image=args.training_image,
        repo_root=REPO_ROOT,
        gpu_spec=args.sft_train_gpus,
        script_path="/workspace/rler/slime/swe_agent/run/run_rubric_sft_smoke.sh",
        log_path=logs_dir / "rubric_sft_train.log",
        env_vars={
            "PROMPT_DATA": _to_container_path(args.artifact_root / "smoke_subsets" / "rubric_sft_smoke.jsonl"),
            "DATASET_SIZE": str(rubric_dataset_size),
            "GLOBAL_BATCH_SIZE": "1",
            "NUM_GPUS": "4",
            "MODEL_DIR": _to_container_path(args.model_dir),
            "MODEL_TORCH_DIST_DIR": _to_container_path(args.model_torch_dist_dir),
            "SAVE_DIR": _to_container_path(args.artifact_root / "rubric_sft"),
            "WANDB_DIR": _to_container_path(args.artifact_root / "rubric_sft" / "wandb"),
            "TENSOR_MODEL_PARALLEL_SIZE": "1",
            "PIPELINE_MODEL_PARALLEL_SIZE": "2",
            "CONTEXT_PARALLEL_SIZE": "2",
            "MAX_TOKENS_PER_GPU": "16384",
        },
    )

    policy_rollout_summary = _load_json(args.artifact_root / "grpo_policy_rollout.json")
    rubric_rollout_summary = _load_json(args.artifact_root / "grpo_rubric_rollout.json")
    policy_rollout_size = int(smoke_subset_summary["policy_rollout"]["selected_count"])
    rubric_rollout_size = int(smoke_subset_summary["rubric_rollout"]["selected_count"])
    if policy_rollout_size <= 0 or rubric_rollout_size <= 0:
        raise RuntimeError(
            f"GRPO rollout export is empty: policy={policy_rollout_summary} rubric={rubric_rollout_summary}"
        )

    _run_container_script(
        image=args.training_image,
        repo_root=REPO_ROOT,
        gpu_spec=args.grpo_train_gpus,
        script_path="/workspace/rler/slime/swe_agent/run/run_policy_grpo_smoke.sh",
        log_path=logs_dir / "policy_grpo_train.log",
        env_vars={
            "DEBUG_ROLLOUT_PATH": _to_container_path(args.artifact_root / "smoke_subsets" / "policy_rollout_smoke.pt"),
            "ROLLOUT_BATCH_SIZE": str(policy_rollout_size),
            "GLOBAL_BATCH_SIZE": str(policy_rollout_size),
            "NUM_GPUS": "4",
            "HF_CHECKPOINT": _to_container_path(args.model_dir),
            "LOAD_DIR": _to_container_path(args.artifact_root / "policy_sft"),
            "REF_LOAD_DIR": _to_container_path(args.artifact_root / "policy_sft"),
            "SAVE_DIR": _to_container_path(args.artifact_root / "policy_grpo"),
            "WANDB_DIR": _to_container_path(args.artifact_root / "policy_grpo" / "wandb"),
            "TENSOR_MODEL_PARALLEL_SIZE": "2",
            "MAX_TOKENS_PER_GPU": "8192",
        },
    )
    _run_container_script(
        image=args.training_image,
        repo_root=REPO_ROOT,
        gpu_spec=args.grpo_train_gpus,
        script_path="/workspace/rler/slime/swe_agent/run/run_rubric_grpo_smoke.sh",
        log_path=logs_dir / "rubric_grpo_train.log",
        env_vars={
            "DEBUG_ROLLOUT_PATH": _to_container_path(args.artifact_root / "smoke_subsets" / "rubric_rollout_smoke.pt"),
            "ROLLOUT_BATCH_SIZE": str(rubric_rollout_size),
            "GLOBAL_BATCH_SIZE": str(rubric_rollout_size),
            "NUM_GPUS": "4",
            "HF_CHECKPOINT": _to_container_path(args.model_dir),
            "LOAD_DIR": _to_container_path(args.artifact_root / "rubric_sft"),
            "REF_LOAD_DIR": _to_container_path(args.artifact_root / "rubric_sft"),
            "SAVE_DIR": _to_container_path(args.artifact_root / "rubric_grpo"),
            "WANDB_DIR": _to_container_path(args.artifact_root / "rubric_grpo" / "wandb"),
            "TENSOR_MODEL_PARALLEL_SIZE": "1",
            "PIPELINE_MODEL_PARALLEL_SIZE": "2",
            "CONTEXT_PARALLEL_SIZE": "2",
            "MAX_TOKENS_PER_GPU": "16384",
        },
    )

    summary = {
        "instance_id": args.instance_id,
        "teacher_run": teacher_summary,
        "student_run": student_summary,
        "sft_export": sft_summary,
        "smoke_subsets": smoke_subset_summary,
        "policy_grpo_export": policy_rollout_summary,
        "rubric_grpo_export": rubric_rollout_summary,
        "policy_sft_checkpoint": _assert_checkpoint(args.artifact_root / "policy_sft", logs_dir / "policy_sft_train.log"),
        "rubric_sft_checkpoint": _assert_checkpoint(args.artifact_root / "rubric_sft", logs_dir / "rubric_sft_train.log"),
        "policy_grpo_checkpoint": _assert_checkpoint(args.artifact_root / "policy_grpo", logs_dir / "policy_grpo_train.log"),
        "rubric_grpo_checkpoint": _assert_checkpoint(args.artifact_root / "rubric_grpo", logs_dir / "rubric_grpo_train.log"),
    }
    (args.artifact_root / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
