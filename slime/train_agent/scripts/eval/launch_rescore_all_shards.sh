#!/bin/bash
# Submit rescore for shards 01..18 in parallel (shard 00 already started as canary).
# Reads existing launch-*.tsv to find original (shard, run_dir) pairs.
set -u
BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench
SLURM=$BASE/RLER-eval/slime/train_agent/scripts/eval/rescore_one_shard.slurm
LOG_DIR=$BASE/RLER-eval/slime/train_agent/scripts/eval/_launch_logs
TSV=$LOG_DIR/launch-20260523-115136.tsv
SUMMARY=$LOG_DIR/rescore-$(date +%Y%m%d-%H%M%S).tsv
echo -e "shard_id\torig_jobid\trescore_jobid\trun_dir" > $SUMMARY

START=${START_SHARD:-1}
END=${END_SHARD:-18}

# also include shard 00 if user asks (default skips since canary already running)
INCLUDE_SH00=${INCLUDE_SH00:-0}

if [ "$INCLUDE_SH00" = "1" ]; then
  SHARDS_TO_RUN=(00 $(seq -f '%02g' $START $END))
else
  SHARDS_TO_RUN=($(seq -f '%02g' $START $END))
fi

# Read shard_id -> orig_jobid from the original launch tsv
declare -A JOBMAP
while IFS=$'\t' read shard_id eval_data orig_jobid; do
  [ "$shard_id" = "shard_id" ] && continue
  JOBMAP[$shard_id]=$orig_jobid
done < $TSV
# also seed shard 00 from earlier launch tsv (its jobid is 55436 per the user)
JOBMAP[00]=55436

for sh in "${SHARDS_TO_RUN[@]}"; do
  JID=${JOBMAP[$sh]:-}
  if [ -z "$JID" ]; then
    echo "[$sh] no orig jobid known; skip"; continue
  fi
  TAG=9b-baseline-train-sh$sh
  RUN_DIR=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/$JID-evp4v2-9b-$TAG
  EVAL_DATA=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench/datasets/v2_python1k_split_train_shards/shard_$sh
  if [ ! -d $RUN_DIR ]; then
    echo "[$sh] missing run_dir $RUN_DIR"; continue
  fi
  CMD="sbatch --parsable --export=ALL,RUN_DIR=$RUN_DIR,EVAL_DATA=$EVAL_DATA $SLURM"
  echo "[$sh] $CMD"
  NEW_JID=$(eval $CMD)
  echo "  -> rescore job $NEW_JID"
  echo -e "$sh\t$JID\t$NEW_JID\t$RUN_DIR" >> $SUMMARY
done

echo summary: $SUMMARY
column -t $SUMMARY || cat $SUMMARY
