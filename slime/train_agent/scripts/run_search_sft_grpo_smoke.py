#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from train_agent.collect_sft_rollout import DEFAULT_TEACHER_MODEL, build_sft_training_rows, collect_teacher_student_exports
from train_agent.collect_grpo_rollout import build_grpo_prompt_rows
from train_agent.data_export import SFTDataExporter
from swe_agent.run.run_swe_agent import choose_gpus
from train_agent.serving.sglang_chat_service import (
    AGENT_ROOT,
    REPO_ROOT,
    extra_mount_args,
    start_slime_server,
    stop_slime_server,
    to_container_path,
    warmup_slime_policy_route,
)


DEFAULT_INSTANCE_IDS = ["elastic__synthetics-316", "wtforms__wtforms-614"]
DEFAULT_SLIME_IMAGE = "slimerl/slime:qwen35-route-agentdeps-20260428-v1"
DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-9B"
DEFAULT_MODEL_DIR = Path("/m-coriander/coriander/zhichen/models/Qwen3.5-9B")
DEFAULT_TORCH_DIST_DIR = REPO_ROOT / "slime/train_agent/artifacts/stage7_qwen35_compatible_torch_dist"
DEFAULT_WANDB_API_KEY = "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"


@dataclass(frozen=True)
class TrainingGpuLayout:
    selected_gpus: list[int]
    actor_gpus: int
    rollout_gpus: int

    @property
    def total_gpus(self) -> int:
        return len(self.selected_gpus)

    @property
    def docker_gpu_spec(self) -> str:
        return ",".join(str(gpu) for gpu in self.selected_gpus)

    @property
    def actor_gpu_spec(self) -> str:
        return ",".join(str(gpu) for gpu in self.selected_gpus[: self.actor_gpus])


def plan_training_gpu_layout(
    train_gpus: str,
    rollout_count: int | str,
) -> TrainingGpuLayout:
    rollout_count = int(rollout_count)
    selected_gpus = choose_gpus(train_gpus)
    if rollout_count < 1:
        raise ValueError("--rollout-gpus must be at least 1.")
    if len(selected_gpus) <= rollout_count:
        raise ValueError(
            f"--train-gpus selected {len(selected_gpus)} GPU(s), but online GRPO needs more GPUs than "
            f"the {rollout_count} rollout GPU(s)."
        )

    actor_gpus = len(selected_gpus) - rollout_count
    return TrainingGpuLayout(
        selected_gpus=selected_gpus,
        actor_gpus=actor_gpus,
        rollout_gpus=rollout_count,
    )


def run_training_module(
    *,
    image: str,
    gpus: str,
    module: str,
    module_args: list[str],
    log_path: Path,
    mounts: list[Path],
    env_vars: dict[str, str],
) -> None:
    selected_gpus = choose_gpus(gpus)
    gpu_list = ",".join(str(gpu) for gpu in selected_gpus)
    container_gpu_list = ",".join(str(index) for index in range(len(selected_gpus)))
    docker_env = [item for key, value in env_vars.items() for item in ("-e", f"{key}={value}") if value]
    docker_mounts = [
        "-v", f"{REPO_ROOT.resolve()}:/workspace/rler",
        "-v", f"{REPO_ROOT.resolve()}:{REPO_ROOT.resolve()}",
    ]
    if Path("/var/run/docker.sock").exists():
        docker_mounts.extend(["-v", "/var/run/docker.sock:/var/run/docker.sock"])
    if Path("/usr/bin/docker").exists():
        docker_mounts.extend(["-v", "/usr/bin/docker:/usr/bin/docker:ro"])
    agent_python = (AGENT_ROOT / ".venv/bin/python").resolve()
    if agent_python.exists():
        docker_mounts.extend(["-v", f"{agent_python.parents[1]}:{agent_python.parents[1]}:ro"])
    docker_cmd = [
        "docker", "run", "--rm", "--runtime", "nvidia", "--ipc=host", "--shm-size=64g",
        "--ulimit", "nofile=1048576:1048576",
        "-e", f"CUDA_VISIBLE_DEVICES={container_gpu_list}",
        "-e", f"NVIDIA_VISIBLE_DEVICES={gpu_list}",
        "-e", "RAY_USE_MULTIPROCESSING_CPU_COUNT=1",
        "-e", "RAY_DISABLE_DOCKER_CPU_WARNING=1",
        "-e", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        *docker_env,
        *extra_mount_args(mounts),
        *docker_mounts,
        "-w", "/workspace/rler/slime",
        image,
        "bash", "--noprofile", "--norc", "-lc",
        shlex.join(["python3", "-m", module, *module_args]),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(docker_cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(f"Training command failed; see {log_path}\n{tail[-8000:]}")


def materialize_ephemeral_jsonl(temp_root: Path, rows_by_name: dict[str, list[dict[str, object]]]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, rows in rows_by_name.items():
        path = temp_root / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                print(json.dumps(row, ensure_ascii=False), file=handle)
        paths[name] = path
    return paths


def training_env(args: argparse.Namespace) -> dict[str, str]:
    env = {"WANDB_MODE": args.wandb_mode}
    if args.wandb_key:
        env["WANDB_API_KEY"] = args.wandb_key
    if args.wandb_team:
        env["WANDB_ENTITY"] = args.wandb_team
    return env


def search_override_args(args: argparse.Namespace) -> list[str]:
    overrides: list[str] = []
    for cli_name, value in (
        ("--search-m", args.search_m),
        ("--search-n", args.search_n),
        ("--search-k", args.search_k),
        ("--search-p", args.search_p),
        ("--search-max-rounds", args.search_max_rounds),
        ("--search-step-limit", args.search_step_limit),
    ):
        if value is not None:
            overrides.extend([cli_name, str(value)])
    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Two-instance SWE-agent search SFT/online-GRPO smoke pipeline.")
    parser.add_argument("--instance-id", nargs="+", default=DEFAULT_INSTANCE_IDS)
    parser.add_argument("--subset", default="rebench_v2")
    parser.add_argument("--split", default="train")
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--teacher-backend", default="openai")
    parser.add_argument("--teacher-api-key", default=os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--student-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-torch-dist-dir", type=Path, default=DEFAULT_TORCH_DIST_DIR)
    parser.add_argument("--sft-config-path", type=Path, default=Path("/workspace/rler/slime/train_agent/configs/sft.sh"))
    parser.add_argument("--grpo-config-path", type=Path, default=Path("/workspace/rler/slime/train_agent/configs/grpo.sh"))
    parser.add_argument("--slime-image", default=DEFAULT_SLIME_IMAGE)
    parser.add_argument("--train-gpus", default="auto:4")
    parser.add_argument("--rollout-gpus", type=int, default=2)
    parser.add_argument("--search-gpus", default="auto:1")
    parser.add_argument("--slime-port", type=int, default=8032)
    parser.add_argument("--slime-api-base", default="http://127.0.0.1:8032")
    parser.add_argument("--slime-api-key", default="EMPTY")
    parser.add_argument("--use-existing-slime-server", action="store_true")
    parser.add_argument("--serving-timeout", type=int, default=1200)
    parser.add_argument("--search-m", type=int, default=2)
    parser.add_argument("--search-n", type=int, default=2)
    parser.add_argument("--search-k", type=int, default=20)
    parser.add_argument("--search-p", type=int, default=1)
    parser.add_argument("--search-max-rounds", type=int, default=2)
    parser.add_argument("--search-step-limit", type=int, default=40)
    parser.add_argument("--search-output-root", type=Path, default=REPO_ROOT / "agent/outputs/search_outputs" / f"search_training_{time.strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--artifact-root", type=Path, default=REPO_ROOT / "slime/train_agent/artifacts" / f"search_training_{time.strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--sft-run-dir", type=Path)
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE") or ("online" if os.environ.get("WANDB_API_KEY") else "offline"))
    parser.add_argument("--wandb-key", default=os.environ.get("SWE_WANDB_API_KEY") or DEFAULT_WANDB_API_KEY)
    parser.add_argument("--wandb-team", default=os.environ.get("WANDB_ENTITY"))
    args = parser.parse_args(argv)

    args.artifact_root.mkdir(parents=True, exist_ok=True)
    logs_dir = args.artifact_root / "logs"
    gpu_layout = plan_training_gpu_layout(args.train_gpus, args.rollout_gpus)
    server = None
    if args.sft_run_dir is not None:
        sft_bundle = SFTDataExporter(run_dir=args.sft_run_dir).export_bundle()
    else:
        try:
            server = start_slime_server(args, logs_dir / "slime_server.log")
            warmup_times = warmup_slime_policy_route(
                base_url=os.environ["SEARCH_SWE_SLIME_API_BASE"],
                api_key=os.environ.get("SEARCH_SWE_SLIME_API_KEY", "EMPTY"),
                model_name=args.student_model,
                wait_for_health=True,
                health_timeout=args.serving_timeout,
            )
            print(json.dumps({"sft_rollout_warmup_seconds": warmup_times}, ensure_ascii=False), flush=True)
            sft_bundle = collect_teacher_student_exports(
                instance_ids=args.instance_id,
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
        finally:
            stop_slime_server(server)
    sft_rows = build_sft_training_rows(
        bundle=sft_bundle,
        pad_to_multiple=gpu_layout.actor_gpus,
    )

    grpo_rows = build_grpo_prompt_rows(args.instance_id, args.subset, args.split)
    sft_counts = {target: len(rows) for target, rows in sft_rows.items()}
    if min(sft_counts["policy"], sft_counts["rubric"], len(grpo_rows)) <= 0:
        raise RuntimeError(f"empty training data: sft={sft_counts} grpo_count={len(grpo_rows)}")

    common_mounts = [args.model_dir, args.model_torch_dist_dir]
    common_env = training_env(args)
    rollout_instance_workers = min(len(grpo_rows), gpu_layout.rollout_gpus)

    with tempfile.TemporaryDirectory(prefix="swe-training-data-") as temp_dir:
        temp_path = Path(temp_dir)
        ephemeral_paths = materialize_ephemeral_jsonl(
            temp_path,
            {
                "policy_sft": sft_rows["policy"],
                "rubric_sft": sft_rows["rubric"],
                "grpo_instances": grpo_rows,
            },
        )
        for target in ("rubric", "policy"):
            run_training_module(
                image=args.slime_image,
                gpus=gpu_layout.actor_gpu_spec,
                module="train_agent.run.sft",
                log_path=logs_dir / f"{target}_sft_train.log",
                mounts=common_mounts + [temp_path],
                env_vars=common_env,
                module_args=[
                    "--target", target,
                    "--prompt-data", str(ephemeral_paths[f"{target}_sft"]),
                    "--dataset-size", str(sft_counts[target]),
                    "--global-batch-size", str(sft_counts[target]),
                    "--model-dir", to_container_path(args.model_dir),
                    "--model-torch-dist-dir", to_container_path(args.model_torch_dist_dir),
                    "--save-dir", to_container_path(args.artifact_root / f"{target}_sft"),
                    "--config-path", str(args.sft_config_path),
                    "--num-gpus", str(gpu_layout.actor_gpus),
                    "--wandb-mode", args.wandb_mode,
                    "--wandb-project", "swe-agent-sft",
                    "--wandb-group", f"{args.artifact_root.name}-{target}-sft",
                ],
            )
            target_search_args = search_override_args(args)
            if target == "policy":
                if "--search-n" in target_search_args:
                    target_search_args[target_search_args.index("--search-n") + 1] = "1"
                else:
                    target_search_args.extend(["--search-n", "1"])
            run_training_module(
                image=args.slime_image,
                gpus=gpu_layout.docker_gpu_spec,
                module="train_agent.run.grpo",
                log_path=logs_dir / f"{target}_grpo_train.log",
                mounts=common_mounts + [temp_path],
                env_vars=common_env,
                module_args=[
                    "--target", target,
                    "--prompt-data", str(ephemeral_paths["grpo_instances"]),
                    "--hf-checkpoint", to_container_path(args.model_dir),
                    "--load-dir", to_container_path(args.artifact_root / f"{target}_sft"),
                    "--save-dir", to_container_path(args.artifact_root / f"{target}_grpo"),
                    "--config-path", str(args.grpo_config_path),
                    "--student-model", args.student_model,
                    "--search-output-root", str((args.search_output_root / f"online_grpo_{target}").resolve()),
                    *target_search_args,
                    "--rollout-batch-size", str(len(grpo_rows)),
                    "--actor-num-gpus", str(gpu_layout.actor_gpus),
                    "--rollout-num-gpus", str(gpu_layout.rollout_gpus),
                    "--rollout-instance-workers", str(rollout_instance_workers),
                    "--wandb-mode", args.wandb_mode,
                    "--wandb-project", "swe-agent-grpo",
                    "--wandb-group", f"{args.artifact_root.name}-{target}-grpo",
                ],
            )

    print(json.dumps({"instances": args.instance_id, "sft_bundle": repr(sft_bundle), "sft_counts": sft_counts, "grpo_count": len(grpo_rows)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
