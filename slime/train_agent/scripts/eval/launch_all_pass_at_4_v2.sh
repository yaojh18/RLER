#!/bin/bash
# Launcher v2: submit 6 9B + 1 DSv4 sbatches using the Option-A (single-queue) v2 scripts.
#
# Defaults to WORKERS_PER_ATTEMPT=8 → TOTAL_WORKERS = ATTEMPTS(4) * 8 = 32 per job.
# Override WORKERS_PER_ATTEMPT to scale up/down.

set -euo pipefail

DSV4_BASE_URL=${DSV4_BASE_URL:?must set DSV4_BASE_URL=http://<dsv4-node>:8888}
WORKERS_PER_ATTEMPT=${WORKERS_PER_ATTEMPT:-8}
ATTEMPTS=${ATTEMPTS:-4}
RLER=${RLER:-/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench/RLER-eval}

BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench
SCRIPTS=$RLER/slime/train_agent/scripts/eval
SUBMIT_LOG=$BASE/../rler-runs/_slurm-logs/evp4v2-submit-$(date +%Y%m%d-%H%M%S).log

declare -a CKPTS=(
  "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/53533-grpo-naive-cp4/policy_ckpt/iter_0000019|53533-iter19|/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/54431-evp4-9b-53533-iter19/hf"
  "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/53911-grpo-naive-cp4/policy_ckpt/iter_0000019|53911-iter19|/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/54432-evp4-9b-53911-iter19/hf"
  "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/53911-grpo-naive-cp4/policy_ckpt/iter_0000029|53911-iter29|/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/54433-evp4-9b-53911-iter29/hf"
  "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/53911-grpo-naive-cp4/policy_ckpt/iter_0000039|53911-iter39|/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/54434-evp4-9b-53911-iter39/hf"
  "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/53911-grpo-naive-cp4/policy_ckpt/iter_0000049|53911-iter49|/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/54435-evp4-9b-53911-iter49/hf"
)
# Entry format: <ckpt_path>|<tag>|<optional HF_OVERRIDE>
#   HF_OVERRIDE = directory of an already-converted HF (must have config.json).
#                 When set, the v2 SLURM skips torch_dist->HF conversion.
#                 Leave blank ("ckpt|tag|") to force re-conversion.

echo "Submitting v2 jobs (workers_per_attempt=$WORKERS_PER_ATTEMPT, total=$((ATTEMPTS * WORKERS_PER_ATTEMPT)), rler=$RLER)"
echo "Log: $SUBMIT_LOG" | tee $SUBMIT_LOG

# 9B baseline (no CKPT, BASELINE=1)
JID=$(sbatch --parsable --export=ALL,BASELINE=1,TAG=qwen3.5-9b-baseline,WORKERS_PER_ATTEMPT=$WORKERS_PER_ATTEMPT,ATTEMPTS=$ATTEMPTS,RLER=$RLER \
    $SCRIPTS/eval_pass_at_4_9b_v2.slurm)
echo "  $JID  9B baseline" | tee -a $SUBMIT_LOG

# 9B trained checkpoints
for entry in "${CKPTS[@]}"; do
  IFS='|' read -r CKPT TAG HF_OVERRIDE <<< "$entry"
  if [ ! -d "$CKPT" ]; then
    echo "  SKIP missing checkpoint: $CKPT" | tee -a $SUBMIT_LOG
    continue
  fi
  EXPORT="ALL,CKPT=$CKPT,TAG=$TAG,WORKERS_PER_ATTEMPT=$WORKERS_PER_ATTEMPT,ATTEMPTS=$ATTEMPTS,RLER=$RLER"
  HF_NOTE="(convert)"
  if [ -n "$HF_OVERRIDE" ] && [ -f "$HF_OVERRIDE/config.json" ]; then
    EXPORT="$EXPORT,HF_OVERRIDE=$HF_OVERRIDE"
    HF_NOTE="(reuse HF: $HF_OVERRIDE)"
  fi
  JID=$(sbatch --parsable --export=$EXPORT $SCRIPTS/eval_pass_at_4_9b_v2.slurm)
  echo "  $JID  9B $TAG $HF_NOTE" | tee -a $SUBMIT_LOG
done

# DSv4
JID=$(sbatch --parsable --export=ALL,DSV4_BASE_URL=$DSV4_BASE_URL,TAG=dsv4-pro,WORKERS_PER_ATTEMPT=$WORKERS_PER_ATTEMPT,ATTEMPTS=$ATTEMPTS,RLER=$RLER \
    $SCRIPTS/eval_pass_at_4_dsv4_v2.slurm)
echo "  $JID  DSv4-pro" | tee -a $SUBMIT_LOG

echo
echo "Watch with:  squeue -u \$USER -o '%.10i %.18j %.2t %.10M %.6D %.20R'"
echo "Aggregate when done:"
echo "  python3 $SCRIPTS/aggregate_pass_at_k.py /mnt/lustre/.../<jid>-evp4v2-*"
