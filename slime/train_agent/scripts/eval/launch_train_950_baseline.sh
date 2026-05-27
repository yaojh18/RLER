#!/bin/bash
# Submit pass@4 baseline eval (Qwen3.5-9B HF init) on all 19 shards of
# v2_python1k_split_train (950 instances total). One sbatch per shard.
#
# Optional env:
#   START_SHARD     default 0   (inclusive)
#   END_SHARD       default 18  (inclusive)
#   WORKERS_PER_ATTEMPT   passthrough to slurm (default in slurm = 16)
#   STICKY_DP_SIZE        passthrough (default in slurm = 8)
#   DRY_RUN=1       print the sbatch commands instead of running them

set -u
BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench
SHARDS_ROOT=$BASE/datasets/v2_python1k_split_train_shards
SLURM=$BASE/RLER-eval/slime/train_agent/scripts/eval/eval_pass_at_4_9b_v2_shard.slurm
LOG_DIR=$BASE/RLER-eval/slime/train_agent/scripts/eval/_launch_logs
mkdir -p $LOG_DIR

START_SHARD=${START_SHARD:-0}
END_SHARD=${END_SHARD:-18}
DRY_RUN=${DRY_RUN:-0}

EXPORTS="ALL,BASELINE=1"
[ -n "${WORKERS_PER_ATTEMPT:-}" ] && EXPORTS="$EXPORTS,WORKERS_PER_ATTEMPT=$WORKERS_PER_ATTEMPT"
[ -n "${STICKY_DP_SIZE:-}" ]      && EXPORTS="$EXPORTS,STICKY_DP_SIZE=$STICKY_DP_SIZE"

STAMP=$(date +%Y%m%d-%H%M%S)
SUMMARY=$LOG_DIR/launch-${STAMP}.tsv
echo -e "shard_id\teval_data\tjob_id" > $SUMMARY
echo "submitting shards ${START_SHARD}..${END_SHARD} (dry_run=$DRY_RUN)"

for i in $(seq -f "%02g" $START_SHARD $END_SHARD); do
  SHARD_DIR=$SHARDS_ROOT/shard_$i
  if [ ! -d $SHARD_DIR ]; then
    echo "MISSING shard $SHARD_DIR; skipping"
    continue
  fi
  TAG="9b-baseline-train-sh${i}"
  CMD="sbatch --parsable --export=${EXPORTS},SHARD_ID=$i,EVAL_DATA=$SHARD_DIR,TAG=$TAG $SLURM"
  echo "[shard $i] $CMD"
  if [ "$DRY_RUN" = "1" ]; then
    echo -e "$i\t$SHARD_DIR\t<dry>" >> $SUMMARY
    continue
  fi
  JID=$(eval $CMD)
  echo "  -> job $JID"
  echo -e "$i\t$SHARD_DIR\t$JID" >> $SUMMARY
done

echo "summary: $SUMMARY"
column -t $SUMMARY || cat $SUMMARY
