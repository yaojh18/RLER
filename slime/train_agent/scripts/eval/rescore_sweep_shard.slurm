#!/bin/bash
#SBATCH --job-name=rescore-sw
#SBATCH --account=ads_cpt
#SBATCH --qos=ads_cpt
#SBATCH --partition=a4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/rescore-sw-%j.log

set -u
RUN_DIR="${RUN_DIR:?must set RUN_DIR}"
EVAL_DATA="${EVAL_DATA:?must set EVAL_DATA}"
ATTEMPTS_LIST="${ATTEMPTS_LIST:?must set ATTEMPTS_LIST e.g. 2,3,4}"
DOCKER_WORKERS="${DOCKER_WORKERS:-8}"

BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench
INNER=$BASE/RLER-eval/slime/train_agent/scripts/eval/rescore_sweep_inner.sh
CONTAINER_IMAGE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/images/slime_v0_2_4_baked.sqsh

echo "[$(date)] rescore-sweep job=$SLURM_JOB_ID  run_dir=$RUN_DIR  attempts=$ATTEMPTS_LIST"
echo "[$(date)] === srun into container ==="
srun --ntasks=1 --cpus-per-task=64 \
  --container-image=$CONTAINER_IMAGE \
  --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre,/mnt/localssd:/mnt/localssd,/var/run/docker.sock:/var/run/docker.sock,/usr/bin/docker:/usr/bin/docker \
  --container-workdir=$BASE \
  --export=ALL,RUN_DIR=$RUN_DIR,EVAL_DATA=$EVAL_DATA,ATTEMPTS_LIST=$ATTEMPTS_LIST,DOCKER_WORKERS=$DOCKER_WORKERS \
  bash $INNER
echo "[$(date)] outer DONE"
