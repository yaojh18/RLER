#!/bin/bash
#SBATCH --job-name=evp4v2-9b
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
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/evp4v2-9b-%j.log

# pass@4 eval of a single Qwen3.5-9B checkpoint on v2_python1k_split_eval (50 instances).
#
# OPTION-A "single queue" variant of eval_pass_at_4_9b.slurm. Instead of spawning
# ATTEMPTS parallel mini-swe-agent processes (each with its own --workers pool), this
# script:
#   1. duplicates the eval set into N_INSTANCES * ATTEMPTS rows, suffixing instance_id
#      with __a{1..ATTEMPTS}.
#   2. runs mini ONCE with --workers = ATTEMPTS * WORKERS_PER_ATTEMPT against the
#      flattened 200-task queue. Threads pull dynamically; no per-attempt long-tail
#      blocks the others.
#   3. demuxes the single preds.json back into attempt-{1..ATTEMPTS}/preds.json with
#      the original instance_ids restored, so the existing scorer + aggregator work
#      without changes.
#
# Required env (sbatch --export=ALL,...):
#   CKPT             torch_dist ckpt dir; or set BASELINE=1 to use Qwen3.5-9B HF init.
#   TAG              short tag for the run dir name (e.g. "53911-iter49-v2")
# Optional:
#   EVAL_DATA            eval dataset dir (default v2_python1k_split_eval)
#   ATTEMPTS             number of attempts per instance (default 4)
#   WORKERS_PER_ATTEMPT  worker slots per attempt's nominal allocation (default 2).
#                        Total --workers in the single mini call = ATTEMPTS * this.
#   BASE_SEED            attempt-N uses BASE_SEED + (N-1)  (default 13)
#   STEP_LIMIT           per-agent step cap (default 120)
#   RLER                 RLER worktree on Lustre (default $BASE/RLER-naive)
#   HF_OVERRIDE          existing HF dir to reuse (must contain config.json).
#                        Skips the torch_dist->HF conversion (saves ~30-45 min).
#                        Ignored when BASELINE=1.

set -u
TAG="${TAG:?must set TAG=<short-name>}"
BASELINE="${BASELINE:-0}"
if [ "$BASELINE" = "1" ]; then
  CKPT="${CKPT:-baseline}"
else
  CKPT="${CKPT:?must set CKPT=<torch_dist ckpt path>  OR  BASELINE=1}"
fi

ATTEMPTS=${ATTEMPTS:-4}
WORKERS_PER_ATTEMPT=${WORKERS_PER_ATTEMPT:-2}
TOTAL_WORKERS=$((ATTEMPTS * WORKERS_PER_ATTEMPT))
BASE_SEED=${BASE_SEED:-13}
STEP_LIMIT=${STEP_LIMIT:-120}
BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench
RLER=${RLER:-$BASE/RLER-naive}
EVAL_DATA=${EVAL_DATA:-$BASE/datasets/v2_python1k_split_eval}
HF_BASE=/mnt/lustre/metavmds0lstre/teams/mrs_ranking_core_modeling/daili1/deepeyes/models/Qwen3.5-9B

RUN_DIR=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/${SLURM_JOB_ID}-evp4v2-9b-${TAG}
HF_OUT=$RUN_DIR/hf
CACHE_BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/.cache/evp4v2-${SLURM_JOB_ID}
SLURM_LOGS=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs
mkdir -p $RUN_DIR/_logs $RUN_DIR/_snapshot $HF_OUT $SLURM_LOGS $CACHE_BASE

( cd $RLER && git rev-parse HEAD ) > $RUN_DIR/_snapshot/RLER_HEAD.txt 2>/dev/null || true
cp $0 $RUN_DIR/_snapshot/ 2>/dev/null || true

CONTAINER_IMAGE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/images/slime_v0_2_4_baked.sqsh

echo "[$(date)] evp4v2-9b job=$SLURM_JOB_ID  tag=$TAG  ckpt=$CKPT  baseline=$BASELINE"
echo "[$(date)] attempts=$ATTEMPTS  workers_per_attempt=$WORKERS_PER_ATTEMPT  total_workers=$TOTAL_WORKERS"
echo "[$(date)] base_seed=$BASE_SEED  step_limit=$STEP_LIMIT  eval_data=$EVAL_DATA"
echo "[$(date)] container=$CONTAINER_IMAGE  run_dir=$RUN_DIR  rler=$RLER"

INNER=$RUN_DIR/_snapshot/inner.sh
cat > $INNER <<EOF
#!/bin/bash
set -eu

CKPT="$CKPT"
TAG="$TAG"
BASELINE="$BASELINE"
ATTEMPTS=$ATTEMPTS
TOTAL_WORKERS=$TOTAL_WORKERS
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
# Priority: BASELINE=1 -> HF_BASE, else HF_OVERRIDE (if set and has config.json),
# else convert torch_dist -> HF freshly. HF_OVERRIDE lets us reuse a previously
# converted HF dir from a prior run (saves ~30-45 min per checkpoint).
HF_OVERRIDE="${HF_OVERRIDE:-}"
if [ "\$BASELINE" = "1" ]; then
  MODEL_DIR=\$HF_BASE
  echo "[\$(date)] step 1: baseline mode, using HF_BASE=\$MODEL_DIR"
elif [ -n "\$HF_OVERRIDE" ] && [ -f "\$HF_OVERRIDE/config.json" ]; then
  MODEL_DIR=\$HF_OVERRIDE
  echo "[\$(date)] step 1: reusing HF_OVERRIDE=\$MODEL_DIR (skip conversion)"
elif [ -n "\$HF_OVERRIDE" ]; then
  echo "[\$(date)] step 1: HF_OVERRIDE=\$HF_OVERRIDE set but missing config.json; falling back to conversion"
  echo "[\$(date)] step 1: convert torch_dist -> HF"
  cd \$RLER/slime
  python tools/convert_torch_dist_to_hf.py \\
    --input-dir \$CKPT \\
    --output-dir \$HF_OUT \\
    --origin-hf-dir \$HF_BASE \\
    --force 2>&1 | tee \$RUN_DIR/_logs/convert.log
  MODEL_DIR=\$HF_OUT
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

for i in \$(seq 1 180); do
  curl -sf http://localhost:20000/health 2>/dev/null && { echo "[\$(date)] sglang ready @ \${i}0s"; break; }
  if ! kill -0 \$SPID 2>/dev/null; then
    echo "sglang died before ready"; tail -120 \$RUN_DIR/_logs/sglang.log; exit 1
  fi
  sleep 10
done
curl -sf http://localhost:20000/health || { tail -120 \$RUN_DIR/_logs/sglang.log; exit 1; }

# === step 3 (NEW): duplicate eval split into ATTEMPTS copies, suffixed instance_id ===
# This is the option-A trick: by exposing 50 x ATTEMPTS = 200 distinct "instances" to
# mini's runner, the single ThreadPoolExecutor pulls them dynamically across a shared
# queue. There's no per-attempt boundary, so a long-tail run in attempt-1 never blocks
# attempt-3 from chewing through new work. Suffix is stripped in step 5 so the
# downstream scorer sees the original instance_id.
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

# === step 4 (CHANGED): run mini ONCE with TOTAL_WORKERS over the 200-task queue ===
export OPENAI_API_BASE=http://localhost:20000/v1
export OPENAI_API_KEY=dummy
SINGLE_OUT=\$RUN_DIR/single_run
mkdir -p \$SINGLE_OUT
echo "[\$(date)] step 4: single mini run, --workers \$TOTAL_WORKERS over 50 x \$ATTEMPTS = \$((50 * \$ATTEMPTS)) tasks"
# Per-task seed: each row's instance_id ends with __a{1..ATTEMPTS}; we can't easily
# pass per-row seed via mini's CLI (it sets one model.model_kwargs.seed). The
# diversity across attempts on the same instance comes from T=1.0 + sglang's
# internal RNG, which is already independent across calls. (sglang_random_seed
# above only sets the engine's startup RNG, not per-request determinism.)
python -m swe_agent.run.benchmarks.swebench \\
  --subset \$EVAL_PARQUET_DIR \\
  --split train \\
  --workers \$TOTAL_WORKERS \\
  --model-class swe_agent.models.litellm_textbased_model.LitellmTextbasedModel \\
  --model openai/\$MODEL_DIR \\
  -c \$RLER/agent/swe_agent/config/benchmarks/swebench_backticks.yaml \\
  -c agent.step_limit=\$STEP_LIMIT \\
  -c model.model_kwargs.temperature=1.0 \\
  -c model.model_kwargs.top_p=0.95 \\
  -o \$SINGLE_OUT \\
  2>&1 | tee \$RUN_DIR/_logs/eval-single.log

echo "[\$(date)] mini run done; killing sglang"
kill \$SPID 2>/dev/null || true
sleep 5

# === step 5 (NEW): demux single preds.json into ATTEMPTS per-attempt preds files ===
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

# === step 6: score each attempt SEQUENTIALLY ===
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

# === step 7: pass@k summary ===
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
