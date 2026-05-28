# GCP parallel-search SFT capture

Cold-start SFT data collection using `trajectory_search_parallel` +
DeepSeek-V4-Pro teacher on the `mrs-research-colen` GCP cluster.
Sequential pipeline (Qwen3.5-9B student + multi-server orchestrator)
was removed 2026-05-21 — superseded by the parallel path which uses
the teacher for everything and exports via `ParallelSFTDataExporter`.

**Operator runbook**: `RUNBOOK_PARALLEL.md` — copy-paste launch
commands, monitoring, verification, common failure modes.

## Launchers

| Launcher | Topology | When to use |
|---|---|---|
| `launch_sft_parallel.sh` | 1 LL DSv4 + 1 full-node collector | One-off debug / minimal recipe sanity check |
| `launch_sft_parallel_ht.sh` | 1 HT DSv4 + 1 full-node collector | Single-server HT recipe (~38% faster than LL) |
| `launch_sft_parallel_sharded.sh` | K × HT DSv4 + K × CPU-only collector (round-robin instance shards) | **Scale-out (recommended for 30+ instances)** |

`launch_sft_parallel_sharded.sh` supports `REUSE_EXISTING_TEACHERS=csv`
to skip teacher launch and reuse already-running endpoints.

## sbatches

| File | Role | Resource |
|---|---|---|
| `sbatch_dsv4_teacher.sbatch` | DSv4 sglang server, **low-latency recipe** (TP=8, no DP-attn, no megamoe, radix off) | 1 × B200 (8 GPU exclusive) |
| `sbatch_dsv4_teacher_high_throughput.sbatch` | DSv4 sglang server, **HT recipe** (DP=8 + DP-attn + mega-MoE + radix ON) | 1 × B200 (8 GPU exclusive) |
| `sbatch_collector_sft_parallel.sbatch` | Full-node collector; polls DSv4 discovery; reads run_teacher_data_collect | 1 × full B200 node (CPU only, but `--exclusive`) |
| `sbatch_collector_sft_parallel_cpuonly.sbatch` | **Shareable-node collector** (no `--exclusive`, gpus=0, cpus=64, mem=512G) — co-locates with other GPU jobs | shared a4 node |
| `sbatch_mini_ablation.sbatch` | Run untouched mini-swe-agent against DSv4 — A/B baseline | 1 × B200 (8 GPU exclusive) |
| `sbatch_qwen35_student.sbatch` | Qwen3.5-9B sglang server (legacy, retained for later student-side experiments) | 1 × B200 (8 GPU exclusive) |

## Helpers

| File | Role |
|---|---|
| `pick_instances_by_patch_length.py` | Select N instances from `v2_python1k_split_train` sorted by gold-patch length |
| `verify_sft_run.py` | Per-instance + aggregate health checker; surfaces decoupled `groups_with_policy/rubric/any` counts; falls back to `summary.json` on login node |
| `sft_full_breakdown.py` | Per-instance breakdown analyzer (mid_cps, Lane A steps, Lane B steps, group acceptance, rubric draft yield) — runs inside the baked slime container |

## Topology (Plan C, sharded)

```
shard 0  ┌──────────────────────┐  writes  ┌────────────────────────────┐
         │ dsv4_teacher_ht      │ ───────► │ shard_00/discovery/dsv4.json│
         │ (DP=8 + mega-MoE)    │          └────────────┬────────────────┘
         └──────────────────────┘                       │ polled by
                                                        ▼
         ┌──────────────────────────────────────────────────────────────┐
         │ collector_cpu_00 (shareable node, no GPU ask)                │
         │ INSTANCE_WORKERS in-flight, each hitting shard 0's DSv4      │
         │ Lane A spine + m=8 Lane B + N=2 rubric drafts + GT eval      │
         └──────────────────────────────────────────────────────────────┘

... × K shards (1 per DSv4 server) in parallel ...
```

Each instance is pinned for its full lifetime to one server (policy
AND rubric go to the same endpoint — radix cache stays warm). Round-
robin sharding by patch-length-sorted order keeps difficulty balanced
across shards.

## Quick start (sharded)

```bash
ssh -p 2224 -i ~/.ssh/google_compute_engine sihanzeng_meta_com@localhost
RLER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench/RLER-zsh-trim
cd $RLER && git fetch origin && git checkout -B zsh-trim_v2 origin/zsh-trim_v2

RUN_TAG=sft_train_$(date +%Y%m%d_%H%M%S)
INSTANCE_IDS=$(python3 $RLER/slime/train_agent/scripts/gcp/pick_instances_by_patch_length.py --offset 0 --limit 950)
ARTIFACT_ROOT=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/$RUN_TAG \
INSTANCE_IDS="$INSTANCE_IDS" N_SHARDS=4 INSTANCE_WORKERS_PER_SHARD=12 \
SEARCH_M=8 SEARCH_N=2 LANES_MAX_MID_CPS=6 LANES_STEPS_PER_ROUND=20 \
SEARCH_STEP_LIMIT=120 LANES_GT_EVAL_WORKERS=8 RUBRIC_BANK_STRATEGY=score \
bash $RLER/slime/train_agent/scripts/gcp/launch_sft_parallel_sharded.sh
```

## What this does NOT do

- **No slime SFT phase.** The pipeline stops after writing per-instance
  artifacts + bundle. Run slime SFT separately on the collected data.
- **No socat / tunnel relay.** DSv4 endpoints stay private to the GCP
  cluster — collectors talk directly over the A4 intra-cluster network.
- **No docker-image warmup.** SWE-rebench instance images are loaded
  on demand by the lustre docker shim (`MSWEA_DOCKER_EXECUTABLE` +
  `DOCKER_LUSTRE_TARDIR`). Missing images cause first-step `docker pull`
  failure.

## Sample knobs

| Knob | Default | Where |
|---|---|---|
| `N_SHARDS` | (required) | `launch_sft_parallel_sharded.sh` |
| `INSTANCE_WORKERS_PER_SHARD` | 8 | `launch_sft_parallel_sharded.sh` |
| `SEARCH_M` (Lane B forks) | 8 | `run_teacher_data_collect --search-m` |
| `SEARCH_N` (rubric drafts per group) | 2 | `--search-n` (bump to 4–8 for more rubric SFT) |
| `LANES_MAX_MID_CPS` | 6 | `--lanes-max-mid-cps` |
| `LANES_STEPS_PER_ROUND` | 20 | `--lanes-steps-per-round` |
| `SEARCH_STEP_LIMIT` | 120 | `--search-step-limit` |
| `LANES_GT_EVAL_WORKERS` | 8 | `--lanes-gt-eval-workers` |
| `RUBRIC_BANK_STRATEGY` | score | `--rubric-bank-strategy` (score / experience / both) |
