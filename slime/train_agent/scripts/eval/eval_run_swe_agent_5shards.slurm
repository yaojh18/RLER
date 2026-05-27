#!/bin/bash
#SBATCH --job-name=evrsa-9b-55433
#SBATCH --account=ads_cpt
#SBATCH --qos=ads_cpt
#SBATCH --partition=a4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/evrsa-9b-55433-%j.log

# Verification eval for the wall_clock_limit_seconds patch
# (commit 5157a4d on zsh-trim_v2).
#
# Runs RLER's run_swe_agent.py (NOT upstream mini-swe-agent) against one
# 55433 GRPO checkpoint over the first 5 train shards (250 instances),
# 1 attempt per instance, with --wall-clock-limit-seconds=6900. Expected
# outcome: "No such container" cascades drop from ~14% (iter59 baseline)
# to ~0%, and every truncated trajectory ends with exit_status=LimitsExceeded.
#
# Required env (sbatch --export=ALL,...):
#   CKPT_HF   absolute path to a HF model dir (already converted)
#   TAG       short tag, e.g. "55433-iter9-wallclock"
# Optional:
#   N_SHARDS              default 5
#   STEP_LIMIT            default 120
#   WALL_CLOCK_SECONDS    default 6900
#   WORKERS               default 32

set -u
CKPT_HF="${CKPT_HF:?must set CKPT_HF=<path to HF dir>}"
TAG="${TAG:?must set TAG=<short label>}"
N_SHARDS=${N_SHARDS:-5}
STEP_LIMIT=${STEP_LIMIT:-120}
WALL_CLOCK_SECONDS=${WALL_CLOCK_SECONDS:-6900}
WORKERS=${WORKERS:-32}

BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench
RLER=$BASE/RLER-zsh-trim
SHARDS_ROOT=$BASE/datasets/v2_python1k_split_train_shards

RUN_DIR=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/${SLURM_JOB_ID}-evrsa-9b-${TAG}
CACHE_BASE=/mnt/localssd/cache/evrsa-${SLURM_JOB_ID}
SLURM_LOGS=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs
mkdir -p $RUN_DIR/_logs $RUN_DIR/_snapshot $SLURM_LOGS

cp $0 $RUN_DIR/_snapshot/ 2>/dev/null || true

CONTAINER_IMAGE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/images/slime_v0_2_4_baked.sqsh

echo "[$(date)] evrsa job=$SLURM_JOB_ID  tag=$TAG"
echo "[$(date)] ckpt_hf=$CKPT_HF"
echo "[$(date)] n_shards=$N_SHARDS  step_limit=$STEP_LIMIT  wall_clock=${WALL_CLOCK_SECONDS}s  workers=$WORKERS"
echo "[$(date)] run_dir=$RUN_DIR"
echo "[$(date)] container=$CONTAINER_IMAGE"

INNER=$RUN_DIR/_snapshot/inner.sh
cat > $INNER <<EOF
#!/bin/bash
set -eu

CKPT_HF="$CKPT_HF"
TAG="$TAG"
N_SHARDS=$N_SHARDS
STEP_LIMIT=$STEP_LIMIT
WALL_CLOCK_SECONDS=$WALL_CLOCK_SECONDS
WORKERS=$WORKERS
RUN_DIR="$RUN_DIR"
BASE="$BASE"
RLER="$RLER"
SHARDS_ROOT="$SHARDS_ROOT"
CACHE_BASE="$CACHE_BASE"

mkdir -p \$CACHE_BASE/{flashinfer,triton,hf,torch,tmp}
export HOME=\$CACHE_BASE
export XDG_CACHE_HOME=\$CACHE_BASE
export TMPDIR=\$CACHE_BASE/tmp
export TMP=\$TMPDIR
export TEMP=\$TMPDIR
export FLASHINFER_WORKSPACE_DIR=\$CACHE_BASE/flashinfer
export TRITON_CACHE_DIR=\$CACHE_BASE/triton
export HF_HOME=\$CACHE_BASE/hf
export TORCH_HOME=\$CACHE_BASE/torch

# RLER fork on PYTHONPATH (NOT upstream mini).
# \$BASE/SWE-rebench-V2 must come first so the vendored `from lib.agent.swe_constants`
# absolute import in agent/swe_agent/run/benchmarks/swe_rebench_v2/lib/agent/log_parsers.py
# resolves — without it, run_swe_agent.py fails at import time.
export PYTHONPATH=\$BASE/SWE-rebench-V2:\$RLER:\$RLER/agent:\${PYTHONPATH:-}

# Lustre docker shim so per-instance docker pulls hit cached tarballs
export PATH=\$BASE/bin:\$PATH
export MSWEA_DOCKER_EXECUTABLE=\$BASE/bin/docker
export DOCKER_LUSTRE_TARDIR=\$BASE/docker_tarballs
export DOCKER_LUSTRE_REAL=/usr/bin/docker
export MSWEA_COST_TRACKING=ignore_errors
export SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# === step 1: launch sglang DP=8 on the checkpoint ===
echo "[\$(date)] step 1: launch sglang DP=8 on \$CKPT_HF"
python -m sglang.launch_server \\
  --model \$CKPT_HF \\
  --tp 1 --dp 8 \\
  --port 20000 --host 0.0.0.0 \\
  --context-length 131072 \\
  --chunked-prefill-size 8192 \\
  --max-prefill-tokens 32768 \\
  --mem-fraction-static 0.85 \\
  --trust-remote-code \\
  --random-seed 13 \\
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

# === step 2: build combined parquet from first N_SHARDS train shards ===
COMBINED=\$RUN_DIR/combined_5shards
mkdir -p \$COMBINED
python - <<PYEOF
import os
from datasets import load_from_disk, Dataset, concatenate_datasets
shards = []
for i in range(\$N_SHARDS):
    p = os.path.join("\$SHARDS_ROOT", f"shard_{i:02d}")
    ds = load_from_disk(p)
    if hasattr(ds, "keys") and "train" in ds:
        ds = ds["train"]
    print(f"[combine] {p}: {len(ds)} rows")
    shards.append(ds)
combined = concatenate_datasets(shards)
print(f"[combine] total: {len(combined)} rows -> \$COMBINED/train-00000.parquet")
combined.to_parquet(os.path.join("\$COMBINED", "train-00000.parquet"))
PYEOF
N_INST=\$(python -c "import pyarrow.parquet as pq; print(pq.read_table('\$COMBINED/train-00000.parquet').num_rows)")
echo "[\$(date)] combined dataset has \$N_INST instances"

# === step 3: run run_swe_agent.py over the combined set ===
mkdir -p \$RUN_DIR/single_run
cd \$RLER/agent
echo "[\$(date)] step 3: run_swe_agent.py  workers=\$WORKERS  step_limit=\$STEP_LIMIT  wall_clock=\${WALL_CLOCK_SECONDS}s"
python -m swe_agent.run.run_swe_agent \\
  --backend sglang \\
  --use-existing-sglang-server \\
  --sglang-api-base http://localhost:20000/v1 \\
  --sglang-api-key dummy \\
  --sglang-model \$CKPT_HF \\
  --subset \$COMBINED \\
  --split train \\
  --offset 0 \\
  --limit \$N_INST \\
  --workers \$WORKERS \\
  --step-limit \$STEP_LIMIT \\
  --wall-clock-limit-seconds \$WALL_CLOCK_SECONDS \\
  --pass-n 1 \\
  --output-root \$RUN_DIR/single_run \\
  2>&1 | tee \$RUN_DIR/_logs/run_swe_agent.log

echo "[\$(date)] run_swe_agent done; killing sglang"
kill \$SPID 2>/dev/null || true
sleep 5

# === step 4: NSC + exit_status post-mortem ===
python - <<PYEOF || true
import json, os, glob, collections
root = "\$RUN_DIR/single_run"
traj_files = sorted(glob.glob(os.path.join(root, "**", "*.traj.json"), recursive=True))
print(f"[postmortem] traj files: {len(traj_files)}")
exit_counts = collections.Counter()
nsc_traj = 0
nsc_total = 0
empty_bash = 0
n_msgs_total = 0
for f in traj_files:
    try:
        d = json.loads(open(f).read())
    except Exception as e:
        print(f"[postmortem] skip {f}: {e}"); continue
    info = d.get("info", {})
    exit_counts[info.get("exit_status", "?")] += 1
    msgs = d.get("messages", [])
    n_msgs_total += len(msgs)
    nsc_in_this = sum(1 for m in msgs if "No such container" in str(m.get("content", "")))
    if nsc_in_this:
        nsc_traj += 1
        nsc_total += nsc_in_this
    empty_bash += sum(1 for m in msgs if m.get("role") == "user" and "<returncode>0</returncode><output></output>" in str(m.get("content", "")))
print(f"[postmortem] exit_status counts:")
for k, v in exit_counts.most_common():
    print(f"  {k}: {v}")
print(f"[postmortem] trajectories with >=1 'No such container': {nsc_traj}/{len(traj_files)} ({100*nsc_traj/max(len(traj_files),1):.1f}%)")
print(f"[postmortem] total 'No such container' messages: {nsc_total}")
print(f"[postmortem] empty-bash-output messages: {empty_bash}")
print(f"[postmortem] mean trajectory length: {n_msgs_total/max(len(traj_files),1):.1f} messages")
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
