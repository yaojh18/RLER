#!/bin/bash
# Submit one rescore_v2.slurm per v2 eval run dir. Each rescorer is independent
# (different rundir, different output) so they can run in parallel.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

declare -a RDS=(
  "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/54560-evp4v2-9b-qwen3.5-9b-baseline"
  "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/54566-evp4v2-dsv4-dsv4-pro"
  "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/54567-evp4v2-9b-53533-iter19"
  "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/54568-evp4v2-9b-53911-iter19"
  "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/54569-evp4v2-9b-53911-iter29"
  "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/54570-evp4v2-9b-53911-iter39"
  "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/54571-evp4v2-9b-53911-iter49"
)

SUBMIT_LOG=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/rescore-v2-submit-$(date +%Y%m%d-%H%M%S).log
echo "Submitting rescore jobs (log: $SUBMIT_LOG)" | tee $SUBMIT_LOG

for RD in "${RDS[@]}"; do
  TAG=$(basename $RD)
  if [ ! -d "$RD" ]; then
    echo "  SKIP missing $RD" | tee -a $SUBMIT_LOG
    continue
  fi
  JID=$(sbatch --parsable --export=ALL,RD=$RD $SCRIPT_DIR/rescore_v2.slurm)
  echo "  $JID  $TAG" | tee -a $SUBMIT_LOG
done

echo
echo "Watch with:  squeue -u \$USER -o '%.10i %.18j %.2t %.10M %.6D'"
echo "Aggregate after all done:"
echo "  python3 $SCRIPT_DIR/aggregate_pass_at_k.py /mnt/lustre/.../<jid>-evp4v2-*"
