#!/bin/bash
#SBATCH --job-name=q35-earlypred-eval
#SBATCH --account=nemotron_rl_rm
#SBATCH --qos=normal
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=/lustre/fsw/portfolios/nemotron/projects/nemotron_rl_rm/users/jiyao/tmp/rler_earlypred_final/logs/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/nemotron/projects/nemotron_rl_rm/users/jiyao/tmp/rler_earlypred_final/logs/%x-%j.err

set -euo pipefail

HOST_ROOT="${RLER_HOST_ROOT:-/lustre/fsw/portfolios/nemotron/projects/nemotron_rl_rm/users/jiyao}"
WORKSPACE=/workspace
RLER_SOURCE_REL="${RLER_SOURCE_REL:-RLER}"
TASKS_REL="${EVAL_TASKS_CONFIG:?set EVAL_TASKS_CONFIG to a path relative to RLER_HOST_ROOT}"
IMAGE="${RUNTIME_IMAGE_HOST:-$HOST_ROOT/enroot_images/slime/rler_slime_nemotron3_nano_v1.sqsh}"
mkdir -p "$HOST_ROOT/tmp/rler_earlypred_final/logs"

for fold in 0 1 2; do
  python3 "$HOST_ROOT/$RLER_SOURCE_REL/slime/train_agent/scripts/prepare_fold_data.py" \
    --workspace-root "$HOST_ROOT" --fold "$fold" \
    --output-root "$HOST_ROOT/tmp/rler_earlypred_final/inputs"
done

export EVAL_TASKS_CONFIG="$WORKSPACE/$TASKS_REL"
export RLER_SOURCE_REL
export EVALUATION_OUTPUT_ROOT="${EVALUATION_OUTPUT_ROOT:-$WORKSPACE/tmp/rler_earlypred_final/evaluations}"
export ENROOT_CACHE_PATH="$HOST_ROOT/tmp/rler_earlypred_final/cache/enroot/cache"
export ENROOT_DATA_PATH="$HOST_ROOT/tmp/rler_earlypred_final/cache/enroot/data"
export ENROOT_RUNTIME_PATH="/tmp/rler-q35-eval-enroot-$SLURM_JOB_ID"
export ENROOT_TEMP_PATH="/tmp/rler-q35-eval-enroot-tmp-$SLURM_JOB_ID"
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH"

requeue_requested=0
srun_pid=
requeue_for_walltime() {
  requeue_requested=1
  if [ -n "${srun_pid:-}" ]; then
    kill -TERM "$srun_pid" 2>/dev/null || true
    wait "$srun_pid" 2>/dev/null || true
  fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_for_walltime USR1
set +e
srun --exclusive --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  --gpus-per-task=8 --kill-on-bad-exit=1 \
  --container-image="$IMAGE" --no-container-mount-home \
  --container-mounts="$HOST_ROOT:$WORKSPACE" --container-workdir="$WORKSPACE" \
  bash "$WORKSPACE/$RLER_SOURCE_REL/slime/train_agent/scripts/qwen35_earlypred_eval_worker.sh" &
srun_pid=$!
wait "$srun_pid"
worker_rc=$?
set -e
trap - USR1
[ "$requeue_requested" -eq 0 ] || exit 0
exit "$worker_rc"
