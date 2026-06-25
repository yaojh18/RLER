#!/bin/bash
# Submit pass@4 sweep (Qwen3.5-9B HF init) over all v2 python rows
# excluding the 1k v2_python1k subset.
# 16 sbatch jobs (15 x 400 + 1 x 274 = 6274 instances).
#
# Optional env:
#   START_SHARD     default 0   (inclusive)
#   END_SHARD       default 15  (inclusive)
#   DRY_RUN=1       print the sbatch commands instead of running them

set -u
BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench
SHARDS_ROOT=$BASE/datasets/v2_python_sweep_shards
SLURM=$BASE/RLER-eval/slime/train_agent/scripts/eval/eval_pass_at_4_9b_v2_sweep.slurm
LOG_DIR=$BASE/RLER-eval/slime/train_agent/scripts/eval/_launch_logs
mkdir -p $LOG_DIR

START_SHARD=${START_SHARD:-0}
END_SHARD=${END_SHARD:-15}
DRY_RUN=${DRY_RUN:-0}

EXPORTS="ALL,BASELINE=1"

STAMP=$(date +%Y%m%d-%H%M%S)
SUMMARY=$LOG_DIR/sweep-launch-${STAMP}.tsv
echo -e "shard_id\teval_data\tjob_id" > $SUMMARY
echo "submitting sweep shards ${START_SHARD}..${END_SHARD} (dry_run=$DRY_RUN)"

for i in $(seq -f "%02g" $START_SHARD $END_SHARD); do
  SHARD_DIR=$SHARDS_ROOT/shard_$i
  if [ ! -d $SHARD_DIR ]; then
    echo "MISSING shard $SHARD_DIR; skipping"
    continue
  fi
  TAG="9b-baseline-pysweep-sh${i}"
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
