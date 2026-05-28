#!/bin/bash
# Submit 3 verification jobs (iter9/19/59 of 55433) using the wall_clock_limit
# patch on commit 5157a4d. One job per checkpoint, 5 train shards each.
set -eu

SLURM=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench/RLER-zsh-trim/slime/train_agent/scripts/eval/eval_run_swe_agent_5shards.slurm
LOG=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench/RLER-zsh-trim/slime/train_agent/scripts/eval/_launch_logs/wallclock-$(date +%Y%m%d-%H%M%S).tsv
mkdir -p "$(dirname "$LOG")"

# Use HF dirs converted by the prior 56244 pass@4 sweep — already exist on Lustre.
declare -A CKPTS=(
  [iter9]=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/56233-evp4v2-9b-55433-iter9/hf
  [iter19]=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/56234-evp4v2-9b-55433-iter19/hf
  [iter59]=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/56244-evp4v2-9b-55433-iter59-sticky/hf
)

echo -e "iter\tjid\ttag\tckpt_hf" > "$LOG"
for ITER in iter9 iter19 iter59; do
  CKPT_HF="${CKPTS[$ITER]}"
  TAG="55433-${ITER}-wallclock"
  JID=$(CKPT_HF="$CKPT_HF" TAG="$TAG" sbatch --parsable --export=ALL "$SLURM")
  echo "submitted $ITER  jid=$JID  ckpt=$CKPT_HF"
  echo -e "${ITER}\t${JID}\t${TAG}\t${CKPT_HF}" >> "$LOG"
done
echo "launch log: $LOG"
