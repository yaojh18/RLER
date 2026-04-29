from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        rows = sum(1 for line in handle if line.strip())
    if rows <= 0:
        raise ValueError(f"SFT dataset is empty: {path}")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run slime SFT for SWE-agent policy or rubric data.")
    parser.add_argument("--target", choices=["policy", "rubric"], required=True)
    parser.add_argument("--prompt-data", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-torch-dist-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=Path("/workspace/rler/slime/train_agent/configs/sft.sh"))
    parser.add_argument("--dataset-size", type=int)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--ray-num-cpus", type=int, default=32)
    parser.add_argument("--wandb-dir", type=Path)
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE") or ("online" if os.environ.get("WANDB_API_KEY") else "offline"))
    parser.add_argument("--wandb-key", default=os.environ.get("WANDB_API_KEY"))
    parser.add_argument("--wandb-team", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-project", default="swe-agent-sft")
    parser.add_argument("--wandb-group")
    args = parser.parse_args(argv)

    dataset_size = args.dataset_size or count_jsonl_rows(args.prompt_data)
    wandb_dir = args.wandb_dir or (args.save_dir / "wandb")
    global_batch_size = args.global_batch_size or dataset_size
    wandb_args = ""
    if args.wandb_mode != "disabled":
        pieces = [
            "--use-wandb",
            "--wandb-mode", args.wandb_mode,
            "--wandb-project", args.wandb_project,
            "--wandb-group", args.wandb_group or f"{args.target}-sft",
            "--wandb-dir", str(wandb_dir),
            "--disable-wandb-random-suffix",
        ]
        if args.wandb_team:
            pieces.extend(["--wandb-team", args.wandb_team])
        wandb_args = shlex.join(pieces)

    command = f"""
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH="/workspace/rler/slime:/workspace/rler:/workspace/rler/agent:/root/Megatron-LM"
export CUDA_DEVICE_MAX_CONNECTIONS=1
trap 'ray stop --force >/dev/null 2>&1 || true' EXIT
pkill -9 sglang >/dev/null 2>&1 || true
ray stop --force >/dev/null 2>&1 || true
cd /workspace/rler/slime
source /workspace/rler/slime/train_agent/configs/qwen3.5-9B.sh
source {shlex.quote(str(args.config_path))}
if [ {shlex.quote(args.target)} = "rubric" ]; then
  SFT_TARGET_ARGS=("${{SFT_RUBRIC_ARGS[@]}}")
else
  SFT_TARGET_ARGS=("${{SFT_POLICY_ARGS[@]}}")
fi
RAY_TMPDIR="/tmp/ray-swe-{args.target}-sft-$$"
mkdir -p "${{RAY_TMPDIR}}"
ray start --head --node-ip-address 127.0.0.1 --num-gpus {args.num_gpus} --num-cpus {args.ray_num_cpus} --temp-dir "${{RAY_TMPDIR}}" --disable-usage-stats >/dev/null
python3 train_async.py \\
  --actor-num-nodes 1 \\
  --actor-num-gpus-per-node {args.num_gpus} \\
  "${{MODEL_ARGS[@]}}" \\
  --hf-checkpoint {shlex.quote(str(args.model_dir))} \\
  --ref-load {shlex.quote(str(args.model_torch_dist_dir))} \\
  --save {shlex.quote(str(args.save_dir))} \\
  --prompt-data {shlex.quote(str(args.prompt_data))} \\
  --rollout-batch-size {dataset_size} \\
  --global-batch-size {global_batch_size} \\
  "${{SFT_COMMON_ARGS[@]}}" \\
  "${{SFT_TARGET_ARGS[@]}}" \\
  "${{SFT_PARALLEL_ARGS[@]}}" \\
  "${{SFT_RECOMPUTE_ARGS[@]}}" \\
  "${{SFT_OPTIMIZER_ARGS[@]}}" \\
  "${{SFT_MISC_ARGS[@]}}" \\
  {wandb_args}
"""
    subprocess.run(["bash", "-lc", command], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
