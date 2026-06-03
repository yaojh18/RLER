#!/bin/bash
# Generic batch launcher for `eval_pass_at_4_9b_v2.slurm` over an arbitrary
# set of (checkpoint_dir, tag) pairs.
#
# Reads pairs from stdin, one per line, whitespace-separated:
#
#   <abs ckpt path>  <tag>
#
# Or take pairs from a file:
#
#   bash launch_pass_at_4_batch.sh < pairs.tsv
#
# Defaults to WORKERS_PER_ATTEMPT=16 (sweep-recipe value). The 50-sample
# eval doc's sampled sglang queue depth across our B200 nodes is uniformly
# 0 with running-req=1-2/rank at WORKERS=8 -- huge unused capacity --
# so bumping to 16 reclaims ~30% of wall without server-side queueing.
# Override with `WORKERS_PER_ATTEMPT=N bash launch_pass_at_4_batch.sh`.
#
# Other env knobs (with defaults):
#   ATTEMPTS=4
#   TIME_CAP=10:00:00
#   STICKY_DP_SIZE=8  (sticky-DP per-instance pinning; 0 disables)
#   RLER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench/RLER
#
# Example:
#
#   cat <<EOF | bash $0
#   /mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler_ckpts/59305-grpo-delta-wvunified-stalediag/policy_ckpt/iter_0000079  59305-iter79
#   /mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler_ckpts/59305-grpo-delta-wvunified-stalediag/policy_ckpt/iter_0000089  59305-iter89
#   EOF
#
# Submits one sbatch per row and prints `JID  TAG  CKPT` to stdout + a
# timestamped log under rler-runs/_slurm-logs/.

set -euo pipefail
RLER=${RLER:-/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench/RLER}
WORKERS_PER_ATTEMPT=${WORKERS_PER_ATTEMPT:-16}
ATTEMPTS=${ATTEMPTS:-4}
TIME_CAP=${TIME_CAP:-10:00:00}
STICKY_DP_SIZE=${STICKY_DP_SIZE:-8}
SLURM=$RLER/slime/train_agent/scripts/eval/eval_pass_at_4_9b_v2.slurm

[ ! -f "$SLURM" ] && { echo "ERROR: SLURM=$SLURM not found"; exit 1; }

LOG=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/evp4v2-batch-$(date +%Y%m%d-%H%M%S).log
mkdir -p "$(dirname $LOG)"

echo "RLER=$RLER" | tee $LOG
echo "WORKERS_PER_ATTEMPT=$WORKERS_PER_ATTEMPT  ATTEMPTS=$ATTEMPTS  TIME_CAP=$TIME_CAP  STICKY_DP_SIZE=$STICKY_DP_SIZE" | tee -a $LOG
echo "SLURM=$SLURM" | tee -a $LOG
echo "" | tee -a $LOG

COUNT=0
SKIP=0
while read -r CKPT TAG _rest; do
  # Skip empty lines, comments, malformed rows
  [ -z "${CKPT:-}" ] && continue
  case "$CKPT" in '#'*) continue ;; esac
  if [ -z "${TAG:-}" ]; then
    echo "  SKIP (no TAG): $CKPT" | tee -a $LOG
    SKIP=$((SKIP+1)); continue
  fi
  if [ ! -d "$CKPT" ]; then
    echo "  SKIP (missing ckpt): $TAG  $CKPT" | tee -a $LOG
    SKIP=$((SKIP+1)); continue
  fi
  JID=$(sbatch --parsable --time=$TIME_CAP \
    --export=ALL,CKPT=$CKPT,TAG=$TAG,WORKERS_PER_ATTEMPT=$WORKERS_PER_ATTEMPT,ATTEMPTS=$ATTEMPTS,STICKY_DP_SIZE=$STICKY_DP_SIZE,RLER=$RLER \
    $SLURM)
  echo "  $JID  $TAG  ckpt=$CKPT" | tee -a $LOG
  COUNT=$((COUNT+1))
done

echo "" | tee -a $LOG
echo "Submitted $COUNT jobs (skipped $SKIP). Log: $LOG" | tee -a $LOG
