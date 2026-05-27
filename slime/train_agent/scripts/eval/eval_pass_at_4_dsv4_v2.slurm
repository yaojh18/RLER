#!/bin/bash
#SBATCH --job-name=evp4v2-dsv4
#SBATCH --account=ads_cpt
#SBATCH --qos=ads_cpt
#SBATCH --partition=a4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=07:00:00
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/evp4v2-dsv4-%j.log

# pass@4 eval of DeepSeek-V4-Pro on v2_python1k_split_eval (50 instances) —
# OPTION-A variant. Reuses an existing DSv4 sglang server (e.g. 54210),
# duplicates the eval split into 50 × ATTEMPTS synthetic rows, runs mini ONCE
# with --workers = ATTEMPTS × WORKERS_PER_ATTEMPT against the shared queue,
# then demuxes preds back into per-attempt dirs for the unchanged scorer.
#
# Note: this still allocates a whole B200 node even though we use 0 GPUs locally
# (the agent only makes HTTP calls). Justified by docker daemon load — 32 parallel
# swebench containers want CPU/RAM headroom — and to keep the layout symmetric with
# the 9B v2 sbatch.
#
# Required env:
#   DSV4_BASE_URL    e.g. http://metavmds1-a4-69:8888
#   TAG              short tag for the run dir name
# Optional:
#   EVAL_DATA            eval dataset dir (default v2_python1k_split_eval)
#   ATTEMPTS             attempts per instance (default 4)
#   WORKERS_PER_ATTEMPT  default 2 → 8 total workers; pass 8 for 32 total
#   BASE_SEED            attempt-N uses BASE_SEED+(N-1) (default 13)
#   STEP_LIMIT           per-agent step cap (default 120)
#   DOCKER_CWD           agent's cwd inside the swebench container (default /)
#   RLER                 RLER worktree on Lustre (default $BASE/RLER-naive)

set -u
: "${DSV4_BASE_URL:?must set DSV4_BASE_URL}"
TAG="${TAG:?must set TAG=<short-name>}"
ATTEMPTS=${ATTEMPTS:-4}
WORKERS_PER_ATTEMPT=${WORKERS_PER_ATTEMPT:-2}
TOTAL_WORKERS=$((ATTEMPTS * WORKERS_PER_ATTEMPT))
BASE_SEED=${BASE_SEED:-13}
STEP_LIMIT=${STEP_LIMIT:-120}
DOCKER_CWD=${DOCKER_CWD:-/}

BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench
RLER=${RLER:-$BASE/RLER-naive}
EVAL_DATA=${EVAL_DATA:-$BASE/datasets/v2_python1k_split_eval}

RUN_DIR=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/${SLURM_JOB_ID}-evp4v2-dsv4-${TAG}
CACHE_BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/.cache/evp4v2dsv4-${SLURM_JOB_ID}
SLURM_LOGS=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs
mkdir -p $RUN_DIR/_logs $RUN_DIR/_snapshot $SLURM_LOGS $CACHE_BASE

( cd $RLER && git rev-parse HEAD ) > $RUN_DIR/_snapshot/RLER_HEAD.txt 2>/dev/null || true
cp $0 $RUN_DIR/_snapshot/ 2>/dev/null || true

CONTAINER_IMAGE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/images/slime_v0_2_4_baked.sqsh

echo "[$(date)] evp4v2-dsv4 job=$SLURM_JOB_ID  tag=$TAG  dsv4=$DSV4_BASE_URL"
echo "[$(date)] attempts=$ATTEMPTS  workers_per_attempt=$WORKERS_PER_ATTEMPT  total_workers=$TOTAL_WORKERS"

INNER=$RUN_DIR/_snapshot/inner.sh
cat > $INNER <<EOF
#!/bin/bash
set -eu

TAG="$TAG"
DSV4_BASE_URL="$DSV4_BASE_URL"
ATTEMPTS=$ATTEMPTS
TOTAL_WORKERS=$TOTAL_WORKERS
BASE_SEED=$BASE_SEED
STEP_LIMIT=$STEP_LIMIT
DOCKER_CWD="$DOCKER_CWD"
EVAL_DATA="$EVAL_DATA"
RUN_DIR="$RUN_DIR"
RLER="$RLER"
BASE="$BASE"
CACHE_BASE="$CACHE_BASE"

export HOME=\$CACHE_BASE
export XDG_CACHE_HOME=\$CACHE_BASE
mkdir -p \$CACHE_BASE/{hf,torch}
export HF_HOME=\$CACHE_BASE/hf
export TORCH_HOME=\$CACHE_BASE/torch
export PYTHONPATH=\$BASE/SWE-rebench-V2:\$RLER:\$RLER/agent:\$RLER/slime:\${PYTHONPATH:-}

export PATH=\$BASE/bin:\$PATH
export MSWEA_DOCKER_EXECUTABLE=\$BASE/bin/docker
export DOCKER_LUSTRE_TARDIR=\$BASE/docker_tarballs
export DOCKER_LUSTRE_REAL=/usr/bin/docker
export MSWEA_COST_TRACKING=ignore_errors

# === step 1: probe DSv4 endpoint ===
echo "[\$(date)] step 1: probe \$DSV4_BASE_URL"
for i in \$(seq 1 30); do
  curl -sf --max-time 5 \$DSV4_BASE_URL/health 2>/dev/null && { echo "DSv4 healthy"; break; }
  if [ \$i -eq 30 ]; then
    echo "ERROR: DSv4 endpoint \$DSV4_BASE_URL unreachable after 5 min"
    exit 1
  fi
  sleep 10
done

# === step 2 (NEW): duplicate eval split into ATTEMPTS copies, suffixed instance_id ===
EVAL_PARQUET_DIR=\$RUN_DIR/eval_data
mkdir -p \$EVAL_PARQUET_DIR
python - <<PYEOF
from datasets import load_from_disk, Dataset
ds = load_from_disk("\$EVAL_DATA")
if hasattr(ds, "keys") and "train" in ds:
    ds = ds["train"]
print(f"[duplicate] orig rows: {len(ds)} x attempts: \$ATTEMPTS = {len(ds) * \$ATTEMPTS} synthetic rows")
rows = []
for r in ds:
    for k in range(1, \$ATTEMPTS + 1):
        new = dict(r)
        new["instance_id"] = f"{r['instance_id']}__a{k}"
        rows.append(new)
Dataset.from_list(rows).to_parquet("\$EVAL_PARQUET_DIR/train-00000.parquet")
print(f"[duplicate] wrote {len(rows)} rows to \$EVAL_PARQUET_DIR/train-00000.parquet")
PYEOF

# === step 3 (CHANGED): single mini run over the shared queue ===
export OPENAI_API_BASE=\$DSV4_BASE_URL/v1
export OPENAI_API_KEY=EMPTY
SINGLE_OUT=\$RUN_DIR/single_run
mkdir -p \$SINGLE_OUT
echo "[\$(date)] step 3: single mini run, --workers \$TOTAL_WORKERS over 50 x \$ATTEMPTS = \$((50 * \$ATTEMPTS)) tasks"
python -m swe_agent.run.benchmarks.swebench \\
  --subset \$EVAL_PARQUET_DIR \\
  --split train \\
  --workers \$TOTAL_WORKERS \\
  --model-class swe_agent.models.litellm_textbased_model.LitellmTextbasedModel \\
  --model openai/deepseek-v4-pro \\
  -c \$RLER/agent/swe_agent/config/benchmarks/swebench_backticks.yaml \\
  -c agent.step_limit=\$STEP_LIMIT \\
  -c environment.cwd=\$DOCKER_CWD \\
  -c model.model_kwargs.temperature=1.0 \\
  -c model.model_kwargs.top_p=0.95 \\
  -c model.model_kwargs.max_tokens=4096 \\
  -c model.model_kwargs.drop_params=true \\
  -o \$SINGLE_OUT \\
  2>&1 | tee \$RUN_DIR/_logs/eval-single.log

# === step 4 (NEW): demux preds.json into ATTEMPTS per-attempt files ===
python - <<PYEOF
import json, os
single = "\$SINGLE_OUT/preds.json"
if not os.path.exists(single):
    raise SystemExit(f"missing {single}; mini did not produce preds")
preds = json.load(open(single))
print(f"[demux] read {len(preds)} synthetic preds from {single}")
per_attempt = {k: {} for k in range(1, \$ATTEMPTS + 1)}
for synth_iid, v in preds.items():
    matched = False
    for k in range(1, \$ATTEMPTS + 1):
        suf = f"__a{k}"
        if synth_iid.endswith(suf):
            orig = synth_iid[: -len(suf)]
            new_v = dict(v) if isinstance(v, dict) else {"patch": v}
            new_v["instance_id"] = orig
            per_attempt[k][orig] = new_v
            matched = True
            break
    if not matched:
        print(f"[demux] WARN: pred id {synth_iid} has no __aK suffix; dropping")
for k in range(1, \$ATTEMPTS + 1):
    d = f"\$RUN_DIR/attempt-{k}"
    os.makedirs(d, exist_ok=True)
    json.dump(per_attempt[k], open(os.path.join(d, "preds.json"), "w"))
    print(f"[demux] attempt-{k}: wrote {len(per_attempt[k])} preds")
PYEOF

# === step 5: score each attempt SEQUENTIALLY ===
for AID in \$(seq 1 \$ATTEMPTS); do
  OUTDIR=\$RUN_DIR/attempt-\$AID
  if [ ! -f \$OUTDIR/preds.json ]; then
    echo "[\$(date)] attempt \$AID: no preds.json"; continue
  fi
  mkdir -p \$OUTDIR/nebius-eval
  echo "[\$(date)] scoring attempt \$AID"
  python \$BASE/dump_specs_and_score.py \\
    \$OUTDIR/preds.json \\
    \$OUTDIR/nebius-eval/ \\
    \$EVAL_DATA/train \\
    8 2>&1 | tee \$RUN_DIR/_logs/score-attempt-\$AID.log
done

# === step 6: summary ===
python - <<PYEOF || true
import json, os, glob
attempts = []
for d in sorted(glob.glob(os.path.join("\$RUN_DIR", "attempt-*"))):
    rep = os.path.join(d, "nebius-eval", "eval_report.json")
    if not os.path.exists(rep):
        print(f"[summary] missing {rep}")
        continue
    r = json.load(open(rep))
    by_iid = {it["instance_id"]: bool(it.get("passed_match")) for it in r.get("items", [])}
    attempts.append(by_iid)
    n = len(by_iid); p = sum(1 for v in by_iid.values() if v)
    print(f"[summary] {os.path.basename(d)}: pass@1 = {p}/{n} = {100.0*p/n if n else 0:.1f}%")

if attempts:
    iids = sorted(set().union(*[a.keys() for a in attempts]))
    for k in (1, 2, 4):
        if k > len(attempts): continue
        passed = sum(1 for iid in iids if any(a.get(iid, False) for a in attempts[:k]))
        print(f"[summary] pass@{k} = {passed}/{len(iids)} = {100.0*passed/len(iids):.1f}%")
PYEOF

echo "[\$(date)] DONE"
EOF
chmod +x $INNER

echo "[$(date)] === srun into container ==="
srun --ntasks=1 --cpus-per-task=128 \
  --container-image=$CONTAINER_IMAGE \
  --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre,/var/run/docker.sock:/var/run/docker.sock,/usr/bin/docker:/usr/bin/docker \
  --container-workdir=$RLER/slime \
  bash $INNER

echo "[$(date)] outer DONE"
