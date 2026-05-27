#!/bin/bash
#SBATCH --job-name=evp4-dsv4
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
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/evp4-dsv4-%j.log

# pass@4 eval of DeepSeek-V4-Pro on v2_python1k_split_eval (50 instances).
#
# Reuses an existing DSv4 sglang server (e.g. job 54210). Launches 4 mini-swe-agent
# attempts in parallel on one node (workers=2 each, different seeds), then scores
# each sequentially with the swe-rebench-v2 docker harness.
#
# Required env:
#   DSV4_BASE_URL    e.g. http://metavmds1-a4-69:8888
#   TAG              short tag for the run dir name
# Optional:
#   EVAL_DATA        eval dataset dir (default v2_python1k_split_eval)
#   ATTEMPTS         number of parallel attempts (default 4)
#   BASE_SEED        first attempt's seed (default 13)
#   STEP_LIMIT       per-agent step cap (default 120)
#   DOCKER_CWD       agent's cwd inside the swebench container (default /)

set -u
: "${DSV4_BASE_URL:?must set DSV4_BASE_URL}"
TAG="${TAG:?must set TAG=<short-name>}"
ATTEMPTS=${ATTEMPTS:-4}
BASE_SEED=${BASE_SEED:-13}
STEP_LIMIT=${STEP_LIMIT:-120}
DOCKER_CWD=${DOCKER_CWD:-/}

BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench
RLER=$BASE/RLER-naive
EVAL_DATA=${EVAL_DATA:-$BASE/datasets/v2_python1k_split_eval}

RUN_DIR=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/${SLURM_JOB_ID}-evp4-dsv4-${TAG}
CACHE_BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/.cache/evp4dsv4-${SLURM_JOB_ID}
SLURM_LOGS=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs
mkdir -p $RUN_DIR/_logs $RUN_DIR/_snapshot $SLURM_LOGS $CACHE_BASE

( cd $RLER && git rev-parse HEAD ) > $RUN_DIR/_snapshot/RLER_HEAD.txt 2>/dev/null || true
cp $0 $RUN_DIR/_snapshot/ 2>/dev/null || true

CONTAINER_IMAGE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/images/slime_v0_2_4_baked.sqsh

echo "[$(date)] evp4-dsv4 job=$SLURM_JOB_ID  tag=$TAG  dsv4=$DSV4_BASE_URL"
echo "[$(date)] attempts=$ATTEMPTS  base_seed=$BASE_SEED  step_limit=$STEP_LIMIT  eval_data=$EVAL_DATA"

INNER=$RUN_DIR/_snapshot/inner.sh
cat > $INNER <<EOF
#!/bin/bash
set -eu

TAG="$TAG"
DSV4_BASE_URL="$DSV4_BASE_URL"
ATTEMPTS=$ATTEMPTS
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

# === step 1.5: emit eval split as parquet ===
EVAL_PARQUET_DIR=\$RUN_DIR/eval_data
mkdir -p \$EVAL_PARQUET_DIR
python -c "
from datasets import load_from_disk
ds = load_from_disk('\$EVAL_DATA')
if hasattr(ds, 'keys') and 'train' in ds:
    ds = ds['train']
print('eval rows:', len(ds))
ds.to_parquet('\$EVAL_PARQUET_DIR/train-00000.parquet')
"

# === step 2: launch ATTEMPTS mini-swe-agent processes in parallel ===
export OPENAI_API_BASE=\$DSV4_BASE_URL/v1
export OPENAI_API_KEY=EMPTY
declare -a APIDS=()
for AID in \$(seq 1 \$ATTEMPTS); do
  SEED=\$(( BASE_SEED + AID - 1 ))
  OUTDIR=\$RUN_DIR/attempt-\$AID
  mkdir -p \$OUTDIR
  echo "[\$(date)] attempt \$AID: seed=\$SEED  out=\$OUTDIR"
  (
    python -m swe_agent.run.benchmarks.swebench \\
      --subset \$EVAL_PARQUET_DIR \\
      --split train \\
      --workers 2 \\
      --model-class swe_agent.models.litellm_textbased_model.LitellmTextbasedModel \\
      --model openai/deepseek-v4-pro \\
      -c \$RLER/agent/swe_agent/config/benchmarks/swebench_backticks.yaml \\
      -c agent.step_limit=\$STEP_LIMIT \\
      -c environment.cwd=\$DOCKER_CWD \\
      -c model.model_kwargs.temperature=1.0 \\
      -c model.model_kwargs.top_p=0.95 \\
      -c model.model_kwargs.seed=\$SEED \\
      -c model.model_kwargs.max_tokens=4096 \\
      -c model.model_kwargs.drop_params=true \\
      -o \$OUTDIR \\
      2>&1 | tee \$RUN_DIR/_logs/eval-attempt-\$AID.log
  ) &
  APIDS+=(\$!)
done
echo "[\$(date)] waiting for \$ATTEMPTS attempts (pids=\${APIDS[@]})"
for PID in "\${APIDS[@]}"; do
  wait \$PID || echo "[\$(date)] attempt pid=\$PID exited non-zero (continuing)"
done
echo "[\$(date)] all attempts done"

# === step 3: score each attempt SEQUENTIALLY ===
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

# === step 4: summary ===
python - <<PYEOF || true
import json, os, glob
attempts = []
for d in sorted(glob.glob(os.path.join("$RUN_DIR", "attempt-*"))):
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
