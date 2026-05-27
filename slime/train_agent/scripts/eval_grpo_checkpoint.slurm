#!/bin/bash
#SBATCH --job-name=grpo-eval
#SBATCH --account=ads_cpt
#SBATCH --qos=ads_cpt
#SBATCH --partition=a4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=07:00:00
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/grpo-eval-%j.log

# Eval a GRPO policy checkpoint on a swe-rebench eval split.
# Reproduces the 50338 flow with throughput fixes — see docs/speed_comparison.md.
#
# Required env (pass via `sbatch --export=ALL,CKPT=...,EVAL_DATA=...`):
#   CKPT       — torch_dist checkpoint dir, e.g.
#                /mnt/lustre/.../rler-runs/<jid>-grpo-pds-policy-1n/policy_ckpt/iter_NNNNNNN
#   EVAL_DATA  — datasets dir (load_from_disk format), e.g.
#                /mnt/lustre/.../swe-rebench/datasets/v2_python1k_split_eval

set -u
CKPT="${CKPT:?must set CKPT=<torch_dist ckpt path>}"
EVAL_DATA="${EVAL_DATA:?must set EVAL_DATA=<eval dataset dir>}"

BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench
RLER=$BASE/RLER
RUN_DIR=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/${SLURM_JOB_ID}-grpo-eval
SLURM_LOGS=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs
HF_BASE=/mnt/lustre/metavmds0lstre/teams/mrs_ranking_core_modeling/daili1/deepeyes/models/Qwen3.5-9B
HF_OUT=$RUN_DIR/hf
CACHE_BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/.cache/eval-${SLURM_JOB_ID}
mkdir -p $RUN_DIR/_logs $RUN_DIR/_snapshot $HF_OUT $SLURM_LOGS $CACHE_BASE

# Snapshot for reproducibility
( cd $RLER && git rev-parse HEAD ) > $RUN_DIR/_snapshot/RLER_HEAD.txt 2>/dev/null || true
cp $0 $RUN_DIR/_snapshot/ 2>/dev/null || true

CONTAINER_IMAGE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/images/slime_v0_2_4_baked.sqsh

echo "[$(date)] eval job=$SLURM_JOB_ID  ckpt=$CKPT  eval_data=$EVAL_DATA  host=$(hostname)"
echo "[$(date)] container=$CONTAINER_IMAGE  workdir=$RLER/slime"

# Inner script — convert ckpt, launch sglang DP=8, run agent eval with 8 workers.
# Runs inside the slime_v0_2_4_baked container which already has Megatron-LM,
# torch, sglang, and the agent deps installed.
INNER=$RUN_DIR/_snapshot/inner.sh
cat > $INNER <<EOF
#!/bin/bash
set -eu

CKPT="$CKPT"
EVAL_DATA="$EVAL_DATA"
RUN_DIR="$RUN_DIR"
HF_BASE="$HF_BASE"
HF_OUT="$HF_OUT"
RLER="$RLER"
BASE="$BASE"
CACHE_BASE="$CACHE_BASE"

# Per-node writable cache (HOME inside container is RO without this)
export HOME=\$CACHE_BASE
export XDG_CACHE_HOME=\$CACHE_BASE
mkdir -p \$CACHE_BASE/{flashinfer,triton,hf,torch}
export FLASHINFER_WORKSPACE_DIR=\$CACHE_BASE/flashinfer
export TRITON_CACHE_DIR=\$CACHE_BASE/triton
export HF_HOME=\$CACHE_BASE/hf
export TORCH_HOME=\$CACHE_BASE/torch

export PYTHONPATH=\$BASE/SWE-rebench-V2:\$RLER:\$RLER/agent:\$RLER/slime:/root/Megatron-LM:\${PYTHONPATH:-}

# Lustre docker shim — load images from local tarballs, not Docker Hub.
export PATH=\$BASE/bin:\$PATH
export MSWEA_DOCKER_EXECUTABLE=\$BASE/bin/docker
export DOCKER_LUSTRE_TARDIR=\$BASE/docker_tarballs
export DOCKER_LUSTRE_REAL=/usr/bin/docker
export MSWEA_COST_TRACKING=ignore_errors
export SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# === step 1: convert torch_dist -> HF ===
echo "[\$(date)] === step 1: convert torch_dist -> HF ==="
cd \$RLER/slime
python tools/convert_torch_dist_to_hf.py \\
  --input-dir \$CKPT \\
  --output-dir \$HF_OUT \\
  --origin-hf-dir \$HF_BASE \\
  --force 2>&1 | tee \$RUN_DIR/_logs/convert.log

# === step 2: launch sglang DP=8 ===
# DP=8 (TP=1): 8 independent replicas behind one port. The DP scheduler
# routes each request to a free replica — implicit one-agent-to-one-GPU
# binding, matching what the 47774 leaderboard run used.
echo "[\$(date)] === step 2: launch sglang (TP=1 DP=8) ==="
python -m sglang.launch_server \\
  --model \$HF_OUT \\
  --tp 1 --dp 8 \\
  --port 20000 --host 0.0.0.0 \\
  --context-length 250000 \\
  --chunked-prefill-size 8192 \\
  --max-prefill-tokens 32768 \\
  --mem-fraction-static 0.88 \\
  --trust-remote-code \\
  > \$RUN_DIR/_logs/sglang.log 2>&1 &
SPID=\$!
trap "kill \$SPID 2>/dev/null || true" EXIT

# Allow up to 30 min for model load + DP warmup
for i in \$(seq 1 180); do
  curl -sf http://localhost:20000/health 2>/dev/null && { echo "[\$(date)] sglang ready @ \${i}0s"; break; }
  if ! kill -0 \$SPID 2>/dev/null; then
    echo "sglang died before ready"; tail -120 \$RUN_DIR/_logs/sglang.log; exit 1
  fi
  sleep 10
done
curl -sf http://localhost:20000/health \\
  || { tail -120 \$RUN_DIR/_logs/sglang.log; exit 1; }

# === step 2.5: convert load_from_disk eval dataset -> parquet ===
# swe_agent.run.benchmarks.swebench calls load_dataset(path, split=split),
# which doesn't understand save_to_disk dirs (it tries to load the
# dataset_dict.json / dataset_info.json files as data rows -> KeyError on
# instance_id). Re-emit the train split as parquet so load_dataset finds
# it via the standard HF data-files pattern.
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

# === step 3: run agent eval ===
# Speed fixes vs the 50338 invocation (see docs/speed_comparison.md):
#   --workers 8                      one agent per DP replica (was implicit 1)
#   model.model_kwargs.temperature=1.0 + top_p=0.95   (was T=0)
#   agent.step_limit=120             cap deterministic-loop hangs
#
# Note: scoring is sequential (step 4 below) NOT overlapped with rollout.
# The earlier overlapping-watcher experiment (50676) regressed pass@1 from
# 18% -> 0% due to docker daemon contention between the agent's persistent
# containers and the scoring harness's transient ones — almost every
# scoring container was SIGKILLed (exit_code=128). See docs/speed_comparison.md.
echo "[\$(date)] === step 3: run agent eval (workers=8, T=1.0, step_limit=120) ==="
export OPENAI_API_BASE=http://localhost:20000/v1
export OPENAI_API_KEY=dummy

OUTDIR=\$RUN_DIR/agent_runs
mkdir -p \$OUTDIR

python -m swe_agent.run.benchmarks.swebench \\
  --subset \$EVAL_PARQUET_DIR \\
  --split train \\
  --workers 8 \\
  --model-class swe_agent.models.litellm_textbased_model.LitellmTextbasedModel \\
  --model openai/\$HF_OUT \\
  -c \$RLER/agent/swe_agent/config/benchmarks/swebench_backticks.yaml \\
  -c agent.step_limit=120 \\
  -c model.model_kwargs.temperature=1.0 \\
  -c model.model_kwargs.top_p=0.95 \\
  -o \$OUTDIR \\
  2>&1 | tee \$RUN_DIR/_logs/eval.log

# Free GPUs before scoring (docker-only).
kill \$SPID 2>/dev/null || true
sleep 5

# === step 4: score (run swe-rebench-v2 docker harness over preds.json) ===
echo "[\$(date)] === step 4: score ==="
mkdir -p \$RUN_DIR/nebius-eval
python \$BASE/dump_specs_and_score.py \\
  \$OUTDIR/preds.json \\
  \$RUN_DIR/nebius-eval/ \\
  \$EVAL_DATA/train \\
  8 2>&1 | tee \$RUN_DIR/_logs/score.log

# Print headline pass@1.
python - <<PYEOF || true
import json
r = json.load(open("\$RUN_DIR/nebius-eval/eval_report.json"))
items = r.get("items", [])
n = len(items)
passed = sum(1 for i in items if i.get("passed_match"))
print(f"PASS_AT_1: {passed}/{n} = {100.0*passed/n:.2f}%" if n else "PASS_AT_1: no items")
PYEOF

echo "[\$(date)] DONE"
EOF
chmod +x $INNER

echo "[$(date)] === srun into container ==="
srun --ntasks=1 --cpus-per-task=64 \
  --container-image=$CONTAINER_IMAGE \
  --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre,/var/run/docker.sock:/var/run/docker.sock,/usr/bin/docker:/usr/bin/docker \
  --container-workdir=$RLER/slime \
  bash $INNER

echo "[$(date)] outer DONE"
