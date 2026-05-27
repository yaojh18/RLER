#!/bin/bash
#SBATCH --job-name=evp4-9b
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
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/evp4-9b-%j.log

# pass@4 eval of a single Qwen3.5-9B checkpoint on v2_python1k_split_eval (50 instances).
#
# ONE sbatch = ONE node = ONE sglang DP=8 server + 4 parallel mini-swe-agent attempts
# (2 workers each, different seeds). Each attempt writes to attempt-{1..4}/, scored
# sequentially after agents finish (parallel scoring fights docker harness per docs).
#
# Required env (sbatch --export=ALL,...):
#   CKPT             torch_dist ckpt dir, e.g. /mnt/lustre/.../53911-grpo-naive-cp4/policy_ckpt/iter_0000049
#                    OR set BASELINE=1 to skip conversion and use Qwen3.5-9B HF dir directly.
#   TAG              short tag for the run dir name (e.g. "53911-iter49", "baseline")
# Optional:
#   EVAL_DATA        eval dataset dir (default v2_python1k_split_eval)
#   ATTEMPTS         number of parallel attempts (default 4)
#   BASE_SEED        first attempt's seed; attempt-N uses BASE_SEED+(N-1) (default 13)
#   STEP_LIMIT       per-agent step cap (default 120)

set -u
TAG="${TAG:?must set TAG=<short-name>}"
BASELINE="${BASELINE:-0}"
if [ "$BASELINE" = "1" ]; then
  CKPT="${CKPT:-baseline}"
else
  CKPT="${CKPT:?must set CKPT=<torch_dist ckpt path>  OR  BASELINE=1}"
fi

ATTEMPTS=${ATTEMPTS:-4}
BASE_SEED=${BASE_SEED:-13}
STEP_LIMIT=${STEP_LIMIT:-120}
BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench
RLER=$BASE/RLER-naive
EVAL_DATA=${EVAL_DATA:-$BASE/datasets/v2_python1k_split_eval}
HF_BASE=/mnt/lustre/metavmds0lstre/teams/mrs_ranking_core_modeling/daili1/deepeyes/models/Qwen3.5-9B

RUN_DIR=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/${SLURM_JOB_ID}-evp4-9b-${TAG}
HF_OUT=$RUN_DIR/hf
CACHE_BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/.cache/evp4-${SLURM_JOB_ID}
SLURM_LOGS=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs
mkdir -p $RUN_DIR/_logs $RUN_DIR/_snapshot $HF_OUT $SLURM_LOGS $CACHE_BASE

( cd $RLER && git rev-parse HEAD ) > $RUN_DIR/_snapshot/RLER_HEAD.txt 2>/dev/null || true
cp $0 $RUN_DIR/_snapshot/ 2>/dev/null || true

CONTAINER_IMAGE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/images/slime_v0_2_4_baked.sqsh

echo "[$(date)] evp4-9b job=$SLURM_JOB_ID  tag=$TAG  ckpt=$CKPT  baseline=$BASELINE"
echo "[$(date)] attempts=$ATTEMPTS  base_seed=$BASE_SEED  step_limit=$STEP_LIMIT  eval_data=$EVAL_DATA"
echo "[$(date)] container=$CONTAINER_IMAGE  run_dir=$RUN_DIR"

INNER=$RUN_DIR/_snapshot/inner.sh
cat > $INNER <<EOF
#!/bin/bash
set -eu

CKPT="$CKPT"
TAG="$TAG"
BASELINE="$BASELINE"
ATTEMPTS=$ATTEMPTS
BASE_SEED=$BASE_SEED
STEP_LIMIT=$STEP_LIMIT
EVAL_DATA="$EVAL_DATA"
RUN_DIR="$RUN_DIR"
HF_BASE="$HF_BASE"
HF_OUT="$HF_OUT"
RLER="$RLER"
BASE="$BASE"
CACHE_BASE="$CACHE_BASE"

# writable cache (HOME inside container is RO without this)
export HOME=\$CACHE_BASE
export XDG_CACHE_HOME=\$CACHE_BASE
mkdir -p \$CACHE_BASE/{flashinfer,triton,hf,torch}
export FLASHINFER_WORKSPACE_DIR=\$CACHE_BASE/flashinfer
export TRITON_CACHE_DIR=\$CACHE_BASE/triton
export HF_HOME=\$CACHE_BASE/hf
export TORCH_HOME=\$CACHE_BASE/torch

export PYTHONPATH=\$BASE/SWE-rebench-V2:\$RLER:\$RLER/agent:\$RLER/slime:/root/Megatron-LM:\${PYTHONPATH:-}

# Lustre docker shim
export PATH=\$BASE/bin:\$PATH
export MSWEA_DOCKER_EXECUTABLE=\$BASE/bin/docker
export DOCKER_LUSTRE_TARDIR=\$BASE/docker_tarballs
export DOCKER_LUSTRE_REAL=/usr/bin/docker
export MSWEA_COST_TRACKING=ignore_errors
export SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# === step 1: pick the HF model dir ===
if [ "\$BASELINE" = "1" ]; then
  MODEL_DIR=\$HF_BASE
  echo "[\$(date)] step 1: baseline mode, using HF_BASE=\$MODEL_DIR"
else
  echo "[\$(date)] step 1: convert torch_dist -> HF"
  cd \$RLER/slime
  python tools/convert_torch_dist_to_hf.py \\
    --input-dir \$CKPT \\
    --output-dir \$HF_OUT \\
    --origin-hf-dir \$HF_BASE \\
    --force 2>&1 | tee \$RUN_DIR/_logs/convert.log
  MODEL_DIR=\$HF_OUT
fi

# === step 2: launch sglang DP=8 ===
echo "[\$(date)] step 2: launch sglang DP=8"
python -m sglang.launch_server \\
  --model \$MODEL_DIR \\
  --tp 1 --dp 8 \\
  --port 20000 --host 0.0.0.0 \\
  --context-length 131072 \\
  --chunked-prefill-size 8192 \\
  --max-prefill-tokens 32768 \\
  --mem-fraction-static 0.85 \\
  --trust-remote-code \\
  --random-seed \$BASE_SEED \\
  > \$RUN_DIR/_logs/sglang.log 2>&1 &
SPID=\$!
trap "kill \$SPID 2>/dev/null || true" EXIT

# Wait up to 30 min for warmup
for i in \$(seq 1 180); do
  curl -sf http://localhost:20000/health 2>/dev/null && { echo "[\$(date)] sglang ready @ \${i}0s"; break; }
  if ! kill -0 \$SPID 2>/dev/null; then
    echo "sglang died before ready"; tail -120 \$RUN_DIR/_logs/sglang.log; exit 1
  fi
  sleep 10
done
curl -sf http://localhost:20000/health || { tail -120 \$RUN_DIR/_logs/sglang.log; exit 1; }

# === step 2.5: emit eval split as parquet ===
EVAL_PARQUET_DIR=\$RUN_DIR/eval_data
mkdir -p \$EVAL_PARQUET_DIR
python -c "
from datasets import load_from_disk
ds = load_from_disk('\$EVAL_DATA')
if hasattr(ds, 'keys') and 'train' in ds:
    ds = ds['train']
print('eval rows:', len(ds), 'cols:', ds.column_names[:5], '...')
ds.to_parquet('\$EVAL_PARQUET_DIR/train-00000.parquet')
print('wrote', '\$EVAL_PARQUET_DIR/train-00000.parquet')
"

# === step 3: launch ATTEMPTS mini-swe-agent processes in parallel ===
# Each attempt: workers=2, distinct seed. 4 attempts x 2 workers = 8 concurrent agents,
# matching sglang DP=8.
export OPENAI_API_BASE=http://localhost:20000/v1
export OPENAI_API_KEY=dummy
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
      --model openai/\$MODEL_DIR \\
      -c \$RLER/agent/swe_agent/config/benchmarks/swebench_backticks.yaml \\
      -c agent.step_limit=\$STEP_LIMIT \\
      -c model.model_kwargs.temperature=1.0 \\
      -c model.model_kwargs.top_p=0.95 \\
      -c model.model_kwargs.seed=\$SEED \\
      -o \$OUTDIR \\
      2>&1 | tee \$RUN_DIR/_logs/eval-attempt-\$AID.log
  ) &
  APIDS+=(\$!)
done
echo "[\$(date)] waiting for \$ATTEMPTS attempts to finish (pids=\${APIDS[@]})"
for PID in "\${APIDS[@]}"; do
  wait \$PID || echo "[\$(date)] attempt pid=\$PID exited non-zero (continuing)"
done
echo "[\$(date)] all \$ATTEMPTS attempts done; killing sglang"
kill \$SPID 2>/dev/null || true
sleep 5

# === step 4: score each attempt SEQUENTIALLY (parallel scoring conflicts with docker) ===
for AID in \$(seq 1 \$ATTEMPTS); do
  OUTDIR=\$RUN_DIR/attempt-\$AID
  if [ ! -f \$OUTDIR/preds.json ]; then
    echo "[\$(date)] attempt \$AID: no preds.json, skipping score"
    continue
  fi
  mkdir -p \$OUTDIR/nebius-eval
  echo "[\$(date)] scoring attempt \$AID"
  python \$BASE/dump_specs_and_score.py \\
    \$OUTDIR/preds.json \\
    \$OUTDIR/nebius-eval/ \\
    \$EVAL_DATA/train \\
    8 2>&1 | tee \$RUN_DIR/_logs/score-attempt-\$AID.log
done

# === step 5: print pass@k summary ===
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

if not attempts:
    print("[summary] no attempts scored")
else:
    iids = sorted(set().union(*[a.keys() for a in attempts]))
    for k in (1, 2, 4):
        if k > len(attempts): continue
        passed = 0
        for iid in iids:
            results = [a.get(iid, False) for a in attempts[:k]]
            if any(results):
                passed += 1
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
