#!/bin/bash
# Launcher: submit 6 9B sbatches + 1 DSv4 sbatch for pass@4 on v2_python1k_split_eval.
#
# Each sbatch is a single 8-GPU node running ATTEMPTS=4 attempts in parallel.
# Pass DSV4_BASE_URL=http://<dsv4-node>:8888 (the existing DSv4 server's URL).

set -euo pipefail

DSV4_BASE_URL=${DSV4_BASE_URL:?must set DSV4_BASE_URL=http://<dsv4-node>:8888}

BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench
RLER=$BASE/RLER-naive
SCRIPTS=$RLER/slime/train_agent/scripts/eval
SUBMIT_LOG=$BASE/../rler-runs/_slurm-logs/evp4-submit-$(date +%Y%m%d-%H%M%S).log

declare -a CKPTS=(
  "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/53533-grpo-naive-cp4/policy_ckpt/iter_0000019|53533-iter19"
  "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/53911-grpo-naive-cp4/policy_ckpt/iter_0000019|53911-iter19"
  "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/53911-grpo-naive-cp4/policy_ckpt/iter_0000029|53911-iter29"
  "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/53911-grpo-naive-cp4/policy_ckpt/iter_0000039|53911-iter39"
  "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/53911-grpo-naive-cp4/policy_ckpt/iter_0000049|53911-iter49"
)

echo "Submitting jobs (log: $SUBMIT_LOG)" | tee $SUBMIT_LOG

# 9B baseline (no CKPT, BASELINE=1)
JID=$(sbatch --parsable --export=ALL,BASELINE=1,TAG=qwen3.5-9b-baseline $SCRIPTS/eval_pass_at_4_9b.slurm)
echo "  $JID  9B baseline" | tee -a $SUBMIT_LOG

# 9B trained checkpoints
for entry in "${CKPTS[@]}"; do
  CKPT="${entry%%|*}"
  TAG="${entry##*|}"
  if [ ! -d "$CKPT" ]; then
    echo "  SKIP missing checkpoint: $CKPT" | tee -a $SUBMIT_LOG
    continue
  fi
  JID=$(sbatch --parsable --export=ALL,CKPT=$CKPT,TAG=$TAG $SCRIPTS/eval_pass_at_4_9b.slurm)
  echo "  $JID  9B $TAG" | tee -a $SUBMIT_LOG
done

# DSv4
JID=$(sbatch --parsable --export=ALL,DSV4_BASE_URL=$DSV4_BASE_URL,TAG=dsv4-pro $SCRIPTS/eval_pass_at_4_dsv4.slurm)
echo "  $JID  DSv4-pro" | tee -a $SUBMIT_LOG

echo
echo "All submitted. Watch with:"
echo "  squeue -u \$USER -o '%.10i %.18j %.2t %.10M %.6D %.20R'"
echo
echo "After they finish, aggregate with:"
echo "  python3 $SCRIPTS/aggregate_pass_at_k.py /mnt/lustre/.../<jid>-evp4-*"
