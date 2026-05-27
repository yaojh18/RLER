#!/bin/bash
# Rescore one shard run by re-invoking dump_specs_and_score.py per attempt.
# Critical: TMPDIR is set to a host-visible bind-mounted path so docker -v works.
set -eu

RUN_DIR=${RUN_DIR:?need RUN_DIR}
EVAL_DATA=${EVAL_DATA:?need EVAL_DATA}
ATTEMPTS=${ATTEMPTS:-4}
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

echo "[$(date)] rescore inner: RUN_DIR=$RUN_DIR"
echo "[$(date)] TMPDIR=$TMPDIR"
mount | grep -E '/tmp |/mnt/local|/mnt/lustre' | head -5 || true

for AID in $(seq 1 $ATTEMPTS); do
  OUTDIR=$RUN_DIR/attempt-$AID
  if [ ! -f $OUTDIR/preds.json ]; then
    echo "[$(date)] attempt $AID: no preds.json; skip"
    continue
  fi
  rm -rf $OUTDIR/nebius-eval
  mkdir -p $OUTDIR/nebius-eval
  echo "[$(date)] scoring attempt $AID"
  python $BASE/dump_specs_and_score.py \
    $OUTDIR/preds.json \
    $OUTDIR/nebius-eval/ \
    $EVAL_DATA/train \
    $DOCKER_WORKERS 2>&1 | tee $RUN_DIR/_logs/rescore-attempt-$AID.log
done

python - <<'PY' || true
import json, os, glob
RD = os.environ['RUN_DIR']
attempts = []
for d in sorted(glob.glob(os.path.join(RD, 'attempt-*'))):
    rep = os.path.join(d, 'nebius-eval', 'eval_report.json')
    if not os.path.exists(rep):
        print(f'[summary] missing {rep}'); continue
    r = json.load(open(rep))
    by = {it['instance_id']: bool(it.get('passed_match')) for it in r.get('items', [])}
    attempts.append(by)
    n = len(by); p = sum(1 for v in by.values() if v)
    print(f'[summary] {os.path.basename(d)}: pass = {p}/{n} = {100.0*p/n if n else 0:.1f}%')
if attempts:
    iids = sorted(set().union(*[a.keys() for a in attempts]))
    for k in (1,2,4):
        if k > len(attempts): continue
        passed = sum(1 for iid in iids if any(a.get(iid, False) for a in attempts[:k]))
        print(f'[summary] pass@{k} = {passed}/{len(iids)} = {100.0*passed/len(iids):.1f}%')
PY
echo "[$(date)] inner DONE"
