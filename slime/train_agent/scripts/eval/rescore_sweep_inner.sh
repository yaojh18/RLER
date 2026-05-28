#!/bin/bash
# Re-score selected attempts of one sweep shard, sequentially.
# Critical TMPDIR fix included so docker -v mounts work from the enroot container.
set -eu

RUN_DIR=${RUN_DIR:?need RUN_DIR}
EVAL_DATA=${EVAL_DATA:?need EVAL_DATA}
ATTEMPTS_LIST=${ATTEMPTS_LIST:?need ATTEMPTS_LIST=e.g. "2,3,4"}
DOCKER_WORKERS=${DOCKER_WORKERS:-8}
BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench
CACHE_BASE=${CACHE_BASE:-/mnt/localssd/cache/rescore-${SLURM_JOB_ID:-noslurm}}

mkdir -p $CACHE_BASE/tmp
export TMPDIR=$CACHE_BASE/tmp
export TMP=$TMPDIR
export TEMP=$TMPDIR
export HOME=$CACHE_BASE
export XDG_CACHE_HOME=$CACHE_BASE

# Lustre docker shim
export PATH=$BASE/bin:$PATH
export DOCKER_LUSTRE_TARDIR=$BASE/docker_tarballs
export DOCKER_LUSTRE_REAL=/usr/bin/docker

mkdir -p $RUN_DIR/_logs
IFS=',' read -ra ATTEMPTS <<< "$ATTEMPTS_LIST"
echo "[$(date)] rescoring shard $RUN_DIR : attempts=${ATTEMPTS[@]}, docker_workers=$DOCKER_WORKERS"

for AID in "${ATTEMPTS[@]}"; do
  OUTDIR=$RUN_DIR/attempt-$AID
  if [ ! -f $OUTDIR/preds.json ]; then
    echo "[$(date)] attempt $AID: no preds.json, skip"
    continue
  fi
  rm -rf $OUTDIR/nebius-eval
  mkdir -p $OUTDIR/nebius-eval
  echo "[$(date)] rescoring attempt $AID"
  python $BASE/dump_specs_and_score.py \
    $OUTDIR/preds.json \
    $OUTDIR/nebius-eval/ \
    $EVAL_DATA/train \
    $DOCKER_WORKERS 2>&1 | tee $RUN_DIR/_logs/rescore-attempt-$AID.log
  echo "[$(date)] attempt $AID done"
done

echo "[$(date)] ALL DONE"
