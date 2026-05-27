#!/bin/bash
#SBATCH --job-name=rescore-v2
#SBATCH --account=ads_cpt
#SBATCH --qos=ads_cpt
#SBATCH --partition=a4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --gpus-per-node=0
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/rescore-v2-%j.log

# CPU-only re-score of a v2 eval run dir.
#
# Works around the documented inline-scoring bug (eval inside the slime sqsh
# container loses host-docker /tmp mount visibility -> all eval items end with
# exit_code=128, scoring 0%). This script runs the scorer in the host shell with
# the host docker daemon, so harness containers mount /tmp correctly.
#
# Required env:
#   RD               path to the run dir (e.g. /mnt/lustre/.../54560-evp4v2-9b-baseline)
# Optional:
#   ATTEMPTS         number of attempts to score (default 4)
#   EVAL_DATA        eval dataset dir (default v2_python1k_split_eval)
#   MAX_WORKERS      parallel docker containers in the scorer (default 8)
#   BASE             swe-rebench Lustre root (default standard)

set -u
: "${RD:?must set RD=<run dir>}"
ATTEMPTS=${ATTEMPTS:-4}
MAX_WORKERS=${MAX_WORKERS:-8}
BASE=${BASE:-/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench}
EVAL_DATA=${EVAL_DATA:-$BASE/datasets/v2_python1k_split_eval}

echo "[$(date)] rescore RD=$RD attempts=$ATTEMPTS max_workers=$MAX_WORKERS"
[ ! -d "$RD" ] && { echo "ERROR: RD=$RD not a directory"; exit 1; }

# Host conda env that has datasets + the scorer's deps.
source /home/sihanzeng_meta_com/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate rankagi 2>/dev/null || echo "[warn] conda env rankagi not activated; relying on system python"

# Lustre docker shim for the harness's `docker run` calls.
export PATH=$BASE/bin:$PATH
export DOCKER_LUSTRE_TARDIR=$BASE/docker_tarballs
export DOCKER_LUSTRE_REAL=/usr/bin/docker

# 1) If attempt-K/preds.json files don't exist (TIMEOUT'd jobs only have
#    single_run/preds.json), demux from single_run by __aK suffix.
SINGLE=$RD/single_run/preds.json
NEED_DEMUX=0
for A in $(seq 1 $ATTEMPTS); do
  [ ! -f "$RD/attempt-$A/preds.json" ] && NEED_DEMUX=1
done

if [ $NEED_DEMUX -eq 1 ]; then
  if [ ! -f $SINGLE ]; then
    echo "ERROR: no $SINGLE to demux from"; exit 1
  fi
  echo "[$(date)] demux $SINGLE -> attempt-{1..$ATTEMPTS}/preds.json"
  python3 - <<PYEOF
import json, os
ATTEMPTS = $ATTEMPTS
RD = "$RD"
preds = json.load(open("$SINGLE"))
per = {k: {} for k in range(1, ATTEMPTS + 1)}
unmatched = 0
for synth, v in preds.items():
    hit = False
    for k in range(1, ATTEMPTS + 1):
        suf = f"__a{k}"
        if synth.endswith(suf):
            orig = synth[: -len(suf)]
            nv = dict(v) if isinstance(v, dict) else {"patch": v}
            nv["instance_id"] = orig
            per[k][orig] = nv
            hit = True
            break
    if not hit:
        unmatched += 1
for k in range(1, ATTEMPTS + 1):
    d = os.path.join(RD, f"attempt-{k}")
    os.makedirs(d, exist_ok=True)
    json.dump(per[k], open(os.path.join(d, "preds.json"), "w"))
    print(f"  attempt-{k}: {len(per[k])} preds")
if unmatched:
    print(f"  WARN: {unmatched} preds without __aK suffix; dropped")
PYEOF
fi

# 2) Re-score each attempt sequentially with the host scorer.
for A in $(seq 1 $ATTEMPTS); do
  OUT=$RD/attempt-$A
  P=$OUT/preds.json
  if [ ! -f $P ]; then
    echo "[$(date)] attempt-$A: no preds.json, skipping"
    continue
  fi
  N=$(python3 -c "import json; print(len(json.load(open('$P'))))" 2>/dev/null)
  echo "[$(date)] === scoring attempt-$A ($N preds) ==="
  rm -rf $OUT/nebius-eval
  mkdir -p $OUT/nebius-eval
  python3 $BASE/dump_specs_and_score.py \
      $P $OUT/nebius-eval/ $EVAL_DATA/train $MAX_WORKERS \
      2>&1 | tee $OUT/nebius-eval/score.log
done

# 3) Pass@k summary across this rundir.
echo
echo "=== pass@k for $RD ==="
python3 - <<PYEOF
import json, os, glob
attempts = []
for d in sorted(glob.glob(os.path.join("$RD", "attempt-*"))):
    rep = os.path.join(d, "nebius-eval", "eval_report.json")
    if not os.path.exists(rep):
        print(f"  missing {rep}"); continue
    r = json.load(open(rep))
    by_iid = {it["instance_id"]: bool(it.get("passed_match")) for it in r.get("items", [])}
    attempts.append(by_iid)
    n = len(by_iid); p = sum(1 for v in by_iid.values() if v)
    print(f"  {os.path.basename(d)}: pass@1 = {p}/{n} = {100.0*p/n if n else 0:.1f}%")
if attempts:
    iids = sorted(set().union(*[a.keys() for a in attempts]))
    for k in (1, 2, 4):
        if k > len(attempts): continue
        passed = sum(1 for iid in iids if any(a.get(iid, False) for a in attempts[:k]))
        print(f"  pass@{k} = {passed}/{len(iids)} = {100.0*passed/len(iids):.1f}%")
PYEOF

echo "[$(date)] DONE"
