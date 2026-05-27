#!/bin/bash
#SBATCH --job-name=rescore-v2
#SBATCH --account=ads_cpt
#SBATCH --qos=ads_cpt
#SBATCH --partition=a4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/rescore-v2-%j.log

set -u
RUN_DIR=${RUN_DIR:?must set RUN_DIR}
EVAL_DATA=${EVAL_DATA:?must set EVAL_DATA}
ATTEMPTS=${ATTEMPTS:-4}
DOCKER_WORKERS=${DOCKER_WORKERS:-8}

BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench
IMG=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/images/slime_v0_2_4_baked.sqsh
INNER=$BASE/RLER-eval/slime/train_agent/scripts/eval/rescore_inner.sh

mkdir -p $RUN_DIR/_logs
echo "[$(date)] rescore job=$SLURM_JOB_ID  run_dir=$RUN_DIR  eval_data=$EVAL_DATA  attempts=$ATTEMPTS  workers=$DOCKER_WORKERS"

srun --ntasks=1 --cpus-per-task=64 \
  --container-image=$IMG \
  --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre,/mnt/localssd:/mnt/localssd,/var/run/docker.sock:/var/run/docker.sock,/usr/bin/docker:/usr/bin/docker \
  --container-workdir=$BASE \
  --export=ALL,RUN_DIR=$RUN_DIR,EVAL_DATA=$EVAL_DATA,ATTEMPTS=$ATTEMPTS,DOCKER_WORKERS=$DOCKER_WORKERS \
  bash $INNER

echo "[$(date)] outer DONE"
