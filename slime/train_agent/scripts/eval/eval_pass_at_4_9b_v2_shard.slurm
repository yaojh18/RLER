#!/bin/bash
#SBATCH --job-name=evp4v2-9b-sh
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
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/evp4v2-9b-sh-%j.log

# pass@4 eval of one Qwen3.5-9B checkpoint on a SINGLE shard of the train set.
#
# Differences vs eval_pass_at_4_9b_v2.slurm:
#   * uses UPSTREAM mini-swe-agent (minisweagent) at $MINI_SWE_AGENT_SRC,
#     not the RLER fork at $RLER/agent/swe_agent.
#   * injects sticky DP routing via a thin StickyDPLitellmTextbasedModel that
#     adds `data_parallel_rank` (round-robin across the 8 sglang DP ranks,
#     incremented once per per-instance model object) to extra_body of every
#     LLM call -- so the 4 attempts of one instance pin to 4 different ranks
#     and each rank's radix cache amortises the system+tool prompt across the
#     ~25 instances it owns.
#   * WORKERS_PER_ATTEMPT default 16  (32 total -> 64 total).
#   * EVAL_DATA defaults to a SHARD path; SHARD_ID is just a label on the run dir.
#
# Required env (sbatch --export=ALL,...):
#   SHARD_ID         e.g. "00" .. "18" (only used to name the run dir)
#   EVAL_DATA        full path to the shard DatasetDict
#                    e.g. .../datasets/v2_python1k_split_train_shards/shard_00
# Optional:
#   TAG              short tag (default "9b-baseline-train-sh${SHARD_ID}")
#   CKPT             torch_dist ckpt dir; or BASELINE=1 for Qwen3.5-9B HF init
#                    (default BASELINE=1 because that's the current target)
#   ATTEMPTS             default 4
#   WORKERS_PER_ATTEMPT  default 16
#   BASE_SEED            default 13
#   STEP_LIMIT           default 120
#   STICKY_DP_SIZE       default 8 (set 0 to disable sticky routing)
#   MINI_SWE_AGENT_SRC   default $BASE/mini-swe-agent/src
#   STICKY_MODEL_DIR     default $BASE/RLER-eval/slime/train_agent/scripts/eval
#                        (must contain sticky_dp_model.py)

set -u
SHARD_ID="${SHARD_ID:?must set SHARD_ID=00..18}"
EVAL_DATA="${EVAL_DATA:?must set EVAL_DATA=<path to shard DatasetDict>}"
BASELINE="${BASELINE:-1}"
TAG="${TAG:-9b-baseline-train-sh${SHARD_ID}}"

if [ "$BASELINE" = "1" ]; then
  CKPT="${CKPT:-baseline}"
else
  CKPT="${CKPT:?must set CKPT=<torch_dist ckpt path>  OR  BASELINE=1}"
fi

ATTEMPTS=${ATTEMPTS:-4}
WORKERS_PER_ATTEMPT=${WORKERS_PER_ATTEMPT:-16}
TOTAL_WORKERS=$((ATTEMPTS * WORKERS_PER_ATTEMPT))
BASE_SEED=${BASE_SEED:-13}
STEP_LIMIT=${STEP_LIMIT:-120}
STICKY_DP_SIZE=${STICKY_DP_SIZE:-8}

BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench
MINI_SWE_AGENT_SRC=${MINI_SWE_AGENT_SRC:-$BASE/mini-swe-agent/src}
STICKY_MODEL_DIR=${STICKY_MODEL_DIR:-$BASE/RLER-eval/slime/train_agent/scripts/eval}
HF_BASE=/mnt/lustre/metavmds0lstre/teams/mrs_ranking_core_modeling/daili1/deepeyes/models/Qwen3.5-9B

RUN_DIR=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/${SLURM_JOB_ID}-evp4v2-9b-${TAG}
HF_OUT=$RUN_DIR/hf
# Per durable rule feedback_write_transient_caches_to_localssd.md: caches go on /mnt/localssd, not Lustre.
CACHE_BASE=/mnt/localssd/cache/evp4v2-${SLURM_JOB_ID}
SLURM_LOGS=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs
mkdir -p $RUN_DIR/_logs $RUN_DIR/_snapshot $HF_OUT $SLURM_LOGS

cp $0 $RUN_DIR/_snapshot/ 2>/dev/null || true
cp $STICKY_MODEL_DIR/sticky_dp_model.py $RUN_DIR/_snapshot/ 2>/dev/null || true

CONTAINER_IMAGE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/images/slime_v0_2_4_baked.sqsh

echo "[$(date)] evp4v2-9b-shard job=$SLURM_JOB_ID  shard=$SHARD_ID  tag=$TAG  baseline=$BASELINE"
echo "[$(date)] attempts=$ATTEMPTS  workers_per_attempt=$WORKERS_PER_ATTEMPT  total_workers=$TOTAL_WORKERS"
echo "[$(date)] step_limit=$STEP_LIMIT  sticky_dp_size=$STICKY_DP_SIZE  eval_data=$EVAL_DATA"
echo "[$(date)] container=$CONTAINER_IMAGE  run_dir=$RUN_DIR"
echo "[$(date)] mini-swe-agent-src=$MINI_SWE_AGENT_SRC  sticky_model_dir=$STICKY_MODEL_DIR"

INNER=$RUN_DIR/_snapshot/inner.sh
cat > $INNER <<EOF
#!/bin/bash
set -eu

CKPT="$CKPT"
TAG="$TAG"
BASELINE="$BASELINE"
SHARD_ID="$SHARD_ID"
ATTEMPTS=$ATTEMPTS
TOTAL_WORKERS=$TOTAL_WORKERS
BASE_SEED=$BASE_SEED
STEP_LIMIT=$STEP_LIMIT
STICKY_DP_SIZE=$STICKY_DP_SIZE
EVAL_DATA="$EVAL_DATA"
RUN_DIR="$RUN_DIR"
HF_BASE="$HF_BASE"
HF_OUT="$HF_OUT"
BASE="$BASE"
CACHE_BASE="$CACHE_BASE"
MINI_SWE_AGENT_SRC="$MINI_SWE_AGENT_SRC"
STICKY_MODEL_DIR="$STICKY_MODEL_DIR"

# transient caches on local SSD, not Lustre
mkdir -p \$CACHE_BASE/{flashinfer,triton,hf,torch,tmp}
export HOME=\$CACHE_BASE
export XDG_CACHE_HOME=\$CACHE_BASE
# Critical: TMPDIR must be on a host-visible bind-mount so eval.py's
# tempfile.TemporaryDirectory is mountable by the host docker daemon (the
# container's default /tmp is an isolated tmpfs and would silently mount empty).
export TMPDIR=\$CACHE_BASE/tmp
export TMP=\$TMPDIR
export TEMP=\$TMPDIR
export FLASHINFER_WORKSPACE_DIR=\$CACHE_BASE/flashinfer
export TRITON_CACHE_DIR=\$CACHE_BASE/triton
export HF_HOME=\$CACHE_BASE/hf
export TORCH_HOME=\$CACHE_BASE/torch

# upstream mini-swe-agent + sticky-DP shim, NO RLER fork on PYTHONPATH
export PYTHONPATH=\$MINI_SWE_AGENT_SRC:\$STICKY_MODEL_DIR:\${PYTHONPATH:-}
export STICKY_DP_SIZE=\$STICKY_DP_SIZE

# Lustre docker shim
export PATH=\$BASE/bin:\$PATH
export MSWEA_DOCKER_EXECUTABLE=\$BASE/bin/docker
export DOCKER_LUSTRE_TARDIR=\$BASE/docker_tarballs
export DOCKER_LUSTRE_REAL=/usr/bin/docker
export MSWEA_COST_TRACKING=ignore_errors
export SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# === step 1: pick HF model dir ===
if [ "\$BASELINE" = "1" ]; then
  MODEL_DIR=\$HF_BASE
  echo "[\$(date)] step 1: baseline mode, using HF_BASE=\$MODEL_DIR"
else
  echo "[\$(date)] step 1: convert torch_dist -> HF"
  cd \$BASE/RLER-eval/slime
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

# === step 3: duplicate shard rows into ATTEMPTS copies ===
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

# === step 4: single mini run with TOTAL_WORKERS over the queue ===
export OPENAI_API_BASE=http://localhost:20000/v1
export OPENAI_API_KEY=dummy
SINGLE_OUT=\$RUN_DIR/single_run
mkdir -p \$SINGLE_OUT
echo "[\$(date)] step 4: single mini run, --workers \$TOTAL_WORKERS, sticky_dp_size=\$STICKY_DP_SIZE"
python -m minisweagent.run.benchmarks.swebench \\
  --subset \$EVAL_PARQUET_DIR \\
  --split train \\
  --workers \$TOTAL_WORKERS \\
  --model-class sticky_dp_model.StickyDPLitellmTextbasedModel \\
  --model openai/\$MODEL_DIR \\
  -c \$BASE/RLER-eval/agent/swe_agent/config/benchmarks/swebench_backticks.yaml \\
  -c agent.step_limit=\$STEP_LIMIT \\
  -c model.model_kwargs.temperature=1.0 \\
  -c model.model_kwargs.top_p=0.95 \\
  -o \$SINGLE_OUT \\
  2>&1 | tee \$RUN_DIR/_logs/eval-single.log

echo "[\$(date)] mini run done; killing sglang"
kill \$SPID 2>/dev/null || true
sleep 5

# === step 5: demux into per-attempt preds ===
python - <<PYEOF
import json, os
single = "\$SINGLE_OUT/preds.json"
if not os.path.exists(single):
    raise SystemExit(f"missing {single}; mini did not produce preds")
preds = json.load(open(single))
print(f"[demux] read {len(preds)} synthetic preds")
per_attempt = {k: {} for k in range(1, \$ATTEMPTS + 1)}
for synth_iid, v in preds.items():
    for k in range(1, \$ATTEMPTS + 1):
        suf = f"__a{k}"
        if synth_iid.endswith(suf):
            orig = synth_iid[: -len(suf)]
            new_v = dict(v) if isinstance(v, dict) else {"patch": v}
            new_v["instance_id"] = orig
            per_attempt[k][orig] = new_v
            break
    else:
        print(f"[demux] WARN: {synth_iid} has no __aK suffix; dropping")
for k in range(1, \$ATTEMPTS + 1):
    d = f"\$RUN_DIR/attempt-{k}"
    os.makedirs(d, exist_ok=True)
    json.dump(per_attempt[k], open(os.path.join(d, "preds.json"), "w"))
    print(f"[demux] attempt-{k}: {len(per_attempt[k])} preds")
PYEOF

# === step 6: score each attempt (sequential, 8 docker workers each) ===
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
  --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre,/mnt/localssd:/mnt/localssd,/var/run/docker.sock:/var/run/docker.sock,/usr/bin/docker:/usr/bin/docker \
  --container-workdir=$BASE \
  bash $INNER

echo "[$(date)] outer DONE"
