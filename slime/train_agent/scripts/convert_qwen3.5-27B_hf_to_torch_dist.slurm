#!/bin/bash
#SBATCH --job-name=convert-qwen35-27b
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
#SBATCH --output=/home/sihanzeng_meta_com/rler_runs/_slurm-logs/convert-qwen35-27b-%j.log

# One-shot HF -> Megatron torch_dist conversion for Qwen3.5-27B.
# convert_hf_to_torch_dist.py auto-picks pipeline_model_parallel_size so that
# num_layers (64) divides cleanly across world_size (8) -> PP=8, 8 layers/stage.
# Runs once; output ckpt is shared by all downstream 27B GRPO/SFT launchers.

set -u

RLER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench/RLER
BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench

# === paths to fill ===
# Where the HF Qwen3.5-27B snapshot lives on Lustre. The shared HF cache
# under /mnt/lustre/metavmds0lstre/share/hf/hub/ resolves the snapshot via
# Qwen--Qwen3.5-27B/snapshots/<rev>; pin the current rev so the converter
# sees the same blobs as eval/serving paths.
HF_CHECKPOINT=${HF_CHECKPOINT:-/mnt/lustre/metavmds0lstre/share/hf/hub/models--Qwen--Qwen3.5-27B/snapshots/b7ca741b86de18df552fd2cc952861e04621a4bd}
# Where to write the torch_dist conversion (re-used by the GRPO launcher).
SAVE_DIR=${SAVE_DIR:-/mnt/lustre/metavmds0lstre/share/sihanzeng_meta_com/slime_qwen35/ckpt/Qwen3.5-27B_torch_dist_latest}

mkdir -p $(dirname "$SAVE_DIR")
mkdir -p /home/sihanzeng_meta_com/rler_runs/_slurm-logs

[ ! -d "$HF_CHECKPOINT" ] && { echo "ERROR: HF_CHECKPOINT=$HF_CHECKPOINT does not exist"; exit 1; }

NODE=$(hostname)
echo "[$(date)] jobid=$SLURM_JOB_ID host=$NODE hf=$HF_CHECKPOINT save=$SAVE_DIR"

# Per-node scratch for caches (Lustre is routinely 85-95% full — never write
# transient torch/triton/hf caches there).
NODE_CACHE=/mnt/localssd/convert-${SLURM_JOB_ID}/$NODE
mkdir -p $NODE_CACHE/{flashinfer,triton,hf,torch}

PER_NODE=$(mktemp -p $NODE_CACHE per_node_XXXX.sh)
cat > $PER_NODE <<EOF
#!/bin/bash
set -eu
export PYTHONUNBUFFERED=1
export FLASHINFER_WORKSPACE_DIR=$NODE_CACHE/flashinfer
export TRITON_CACHE_DIR=$NODE_CACHE/triton
export HF_HOME=$NODE_CACHE/hf
export TORCH_HOME=$NODE_CACHE/torch
export XDG_CACHE_HOME=$NODE_CACHE
export HOME=$NODE_CACHE

export PYTHONPATH=$RLER/slime:$RLER:/root/Megatron-LM:\${PYTHONPATH:-}

cd $RLER/slime
source scripts/models/qwen3.5-27B.sh

torchrun --nproc_per_node 8 --nnodes 1 \\
    --rdzv_id $SLURM_JOB_ID --rdzv_backend c10d \\
    --rdzv_endpoint localhost:0 \\
    tools/convert_hf_to_torch_dist.py \\
    "\${MODEL_ARGS[@]}" \\
    --hf-checkpoint $HF_CHECKPOINT \\
    --save $SAVE_DIR
EOF
chmod +x $PER_NODE

echo "[$(date)] === srun convert ==="
srun --ntasks=1 --ntasks-per-node=1 \
    --container-image=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/images/slime_v0_2_4_baked.sqsh \
    --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre,/mnt/localssd:/mnt/localssd,/home/sihanzeng_meta_com:/home/sihanzeng_meta_com \
    --container-workdir=$RLER/slime \
    bash $PER_NODE

echo "[$(date)] DONE jobid=$SLURM_JOB_ID"
ls -la $SAVE_DIR | head -20
