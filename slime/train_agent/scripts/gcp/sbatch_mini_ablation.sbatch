#!/bin/bash
#SBATCH --job-name=mini_ablation
#SBATCH --account=ads_cpt
#SBATCH --qos=ads_cpt
#SBATCH --partition=a4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/mini_ablation-%j.log

# Ablation: run the *untouched* mini-swe-agent against DSv4 on the same
# 30 easy instances we ran our search pipeline on. Comparing pass@1 of the
# vanilla agent vs our search pipeline isolates whether the bottleneck is
# the search/rubric/judge code or the model itself.
#
# Required env (passed via --export):
#   DSV4_BASE_URL    full base URL of DSv4 sglang endpoint, e.g. http://metavmds1-a4-70:8888
#   OUTPUT_DIR       output directory on Lustre
#   INSTANCE_REGEX   pipe-separated instance ID regex for --filter
# Optional env:
#   THINKING_MODE    "on" -> sends chat_template_kwargs.thinking=true; "off" -> doesn't (default off)
#   WORKERS          parallel workers (default 4)
#   MAX_STEPS        per-agent step cap (default 120)

set -euo pipefail
: "${DSV4_BASE_URL:?DSV4_BASE_URL is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${INSTANCE_REGEX:?INSTANCE_REGEX is required (pipe-separated)}"

THINKING_MODE=${THINKING_MODE:-off}
WORKERS=${WORKERS:-4}
MAX_STEPS=${MAX_STEPS:-120}
# Optional: path to a custom agent config yaml (overrides mini's swebench_backticks.yaml).
# Also: working dir to set in docker exec (defaults to / so we don't reference /testbed).
CONFIG_OVERRIDE=${CONFIG_OVERRIDE:-}
DOCKER_CWD=${DOCKER_CWD:-/}

RLER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench/RLER-zsh-trim
BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench
MINI=$BASE/mini-swe-agent
DATASET=${DATASET_OVERRIDE:-$BASE/datasets/v2_python1k_split_train}
IMAGE_SQSH=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/images/slime_v0_2_4_baked.sqsh

mkdir -p "$OUTPUT_DIR"
echo "[$(date)] jobid=$SLURM_JOB_ID thinking=$THINKING_MODE workers=$WORKERS"
echo "  dsv4=$DSV4_BASE_URL"
echo "  output=$OUTPUT_DIR"
echo "  instance regex (truncated): ${INSTANCE_REGEX:0:200}..."

CACHE_BASE=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/.cache/mini-${SLURM_JOB_ID}
mkdir -p "$CACHE_BASE"

# Build the chat_template_kwargs override. Mini's recursive_merge picks up
# nested dict via dot-separated YAML; pass via -c key=value.
THINKING_ARGS=""
if [ "$THINKING_MODE" = "on" ]; then
    # mini-swe-agent uses litellm; extra_body propagates to the openai client.
    THINKING_ARGS='-c model.model_kwargs.extra_body={"chat_template_kwargs":{"thinking":true}}'
fi

exec srun --container-image="$IMAGE_SQSH" \
     --container-mounts=/mnt/lustre/metavmds0lstre:/mnt/lustre/metavmds0lstre,/var/run/docker.sock:/var/run/docker.sock,/usr/bin/docker:/usr/bin/docker \
     --container-workdir=$MINI \
     --container-writable \
     --no-container-mount-home \
     --export=ALL,RLER=$RLER,BASE=$BASE,MINI=$MINI,DATASET=$DATASET,CACHE_BASE=$CACHE_BASE,\
DSV4_BASE_URL=$DSV4_BASE_URL,OUTPUT_DIR=$OUTPUT_DIR,INSTANCE_REGEX="$INSTANCE_REGEX",\
THINKING_MODE=$THINKING_MODE,WORKERS=$WORKERS,MAX_STEPS=$MAX_STEPS,THINKING_ARGS="$THINKING_ARGS",\
CONFIG_OVERRIDE=$CONFIG_OVERRIDE,DOCKER_CWD=$DOCKER_CWD \
     bash -lc '
set -euo pipefail
export PYTHONPATH=$MINI/src:${PYTHONPATH:-}
export HOME=$CACHE_BASE
export XDG_CACHE_HOME=$CACHE_BASE
export HF_HOME=$CACHE_BASE/hf
mkdir -p "$HF_HOME"

# Lustre docker shim
export MSWEA_DOCKER_EXECUTABLE=$BASE/bin/docker
export DOCKER_LUSTRE_TARDIR=$BASE/docker_tarballs
export DOCKER_LUSTRE_REAL=/usr/bin/docker
# DSv4 not in litellm price registry -> cost calc crashes every request. Ignore.
export MSWEA_COST_TRACKING=ignore_errors

# Point litellm openai client at DSv4
export OPENAI_API_BASE=$DSV4_BASE_URL/v1
export OPENAI_API_KEY=EMPTY

pip install --quiet --no-deps -e $MINI 2>&1 | tail -5 || true

cd $MINI
# Use the backticks system prompt (asks model to emit ```mswea_bash_command code blocks)
# + LitellmTextbasedModel which parses actions out of `content` via regex instead of
# requiring OpenAI tool_calls. Avoids the DSv4-DSML <-> OpenAI-tool_calls translation
# gap entirely; works against any sglang DSv4 server regardless of --tool-call-parser.
CFG_PATH=${CONFIG_OVERRIDE:-$MINI/src/minisweagent/config/benchmarks/swebench_backticks.yaml}
echo "[mini-ablation] config: $CFG_PATH  docker_cwd: $DOCKER_CWD"
python -m minisweagent.run.benchmarks.swebench \
    --subset "$DATASET" \
    --split train \
    --filter "$INSTANCE_REGEX" \
    --model openai/deepseek-v4-pro \
    --model-class minisweagent.models.litellm_textbased_model.LitellmTextbasedModel \
    --output "$OUTPUT_DIR" \
    --workers "$WORKERS" \
    -c $CFG_PATH \
    -c agent.cost_limit=0 \
    -c agent.step_limit=$MAX_STEPS \
    -c environment.cwd=$DOCKER_CWD \
    -c model.model_kwargs.api_base=$OPENAI_API_BASE \
    -c model.model_kwargs.api_key=EMPTY \
    -c model.model_kwargs.temperature=1.0 \
    -c model.model_kwargs.top_p=0.95 \
    -c model.model_kwargs.max_tokens=4096 \
    -c model.model_kwargs.drop_params=true \
    $THINKING_ARGS
'
