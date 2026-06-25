# Parallel SFT Pipeline — Operator Runbook

End-to-end recipe for collecting SFT data via DSv4 teacher + parallel
trajectory search (v1-lanes machinery) + full-parity rubric pipeline.

Updated: 2026-05-21. Branch: `zsh-trim_v2` on `yaojh18/RLER`.

---

## TL;DR — copy-paste launch

On the GCP login node:

```bash
# 1. Pull the new branch
RLER=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench/RLER-zsh-trim
cd $RLER
git fetch origin && git checkout -B zsh-trim_v2 origin/zsh-trim_v2

# 2. Output dir
RUN_TAG=sft_parallel_30easy_$(date +%Y%m%d_%H%M%S)
ARTIFACT_ROOT=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/$RUN_TAG

# 3. The 30 easy SWE-rebench-v2 instances (gold patch 397–585 chars)
INSTANCE_IDS="amoffat__sh-744 aristanetworks__anta-1125 aristanetworks__anta-978 aspp__pelita-875 aws-cloudformation__cfn-lint-4023 celery__celery-9614 copier-org__copier-1998 getmoto__moto-8780 getmoto__moto-8796 getmoto__moto-8828 holoviz__holoviews-6534 jax-ml__jax-28612 lektor__lektor-1224 matthewwithanm__python-markdownify-202 meltano__sdk-3019 modin-project__modin-7491 more-itertools__more-itertools-1028 mpmath__mpmath-904 narwhals-dev__narwhals-1934 patrick-kidger__equinox-993 pybamm-team__pybamm-4753 pymc-devs__pymc-7858 python-attrs__attrs-1417 quantecon__quantecon.py-769 rhayes777__pyautofit-1133 scrapy__scrapy-6606 scrapy__scrapy-6867 sdv-dev__rdt-970 tatuylonen__wiktextract-971 tornadoweb__tornado-3488"

# 4. Submit (DSv4 teacher + collector with --dependency=after)
ARTIFACT_ROOT=$ARTIFACT_ROOT \
INSTANCE_IDS="$INSTANCE_IDS" \
INSTANCE_WORKERS=16 \
SEARCH_M=8 SEARCH_N=2 LANES_MAX_MID_CPS=6 LANES_STEPS_PER_ROUND=20 \
SEARCH_STEP_LIMIT=120 LANES_GT_EVAL_WORKERS=8 \
RUBRIC_BANK_STRATEGY=score \
bash $RLER/slime/train_agent/scripts/gcp/launch_sft_parallel.sh
```

Output: two SLURM jobids — `dsv4 jobid=N1`, `collector jobid=N2`.

---

## Topology

```
                ┌──────────────────────────────────────────┐
                │  sbatch_dsv4_teacher.sbatch              │
                │  1 × B200 (8 GPU)                        │
                │  sglang serve --tp 8 (low-latency recipe)│
                │  pinned image sglang_dsv4_b200.sqsh      │
                │  writes discovery/dsv4.json              │
                └────────────────┬─────────────────────────┘
                                 │ polled by
                                 ▼
                ┌──────────────────────────────────────────┐
                │  sbatch_collector_sft_parallel.sbatch    │
                │  1 × cpu-only (--gpus-per-node=0)        │
                │  baked slime image                       │
                │  run_teacher_data_collect.py             │
                │    --use-parallel-search                 │
                │    --instance-workers 16                 │
                │                                          │
                │  Per instance (16 concurrent):           │
                │    TrajectorySearchParallelRunner.run()  │
                │      Lane A: spine, emit MidCps          │
                │      Lane B: 8 forks per MidCp           │
                │      Lane C: full-parity rubric pipeline │
                │              (bank carry-forward chain   │
                │               by group_index, idempotent │
                │               global route registration) │
                │                                          │
                │  Per group, ParallelSFTDataExporter:     │
                │    pick passing branch (gt>=1.0)         │
                │    pick rubric sample (gap_corr>0.8)     │
                │    emit (policy_sample, rubric_sample)   │
                └──────────────────────────────────────────┘
```

Approx concurrent DSv4 requests: 16 workers × ~20 calls per instance peak
= ~320 concurrent. Above the ~CONC=64 sweet spot, well within the
scheduler's capacity.

---

## What changed since 2026-05-21

`zsh-trim_v2` branch reorganizes `zsh-trim` into ≤10 logical commits and adds:

- **Full-parity rubric machinery** in `trajectory_search_parallel.py`
  (task 2): `ScoreRubricBank` + `ExperienceRubricBank` carry forward
  along Lane A's MidCp sequence, chained via ordered `asyncio.Event`s on
  group_index. Per-group N-sample rubric fan-out + two-phase scoring +
  per-rubric reward aggregation (variance + redundancy + judge_error)
  + per-instance `ExperienceRubricBank.update_after_instance` epilogue.

- **Trainable-token cap** in `lane_to_grpo_bundle._build_branch_sample`
  (task 3): only the first `steps_per_round` assistant turns past the
  shared parent contribute to loss; later tokens are dropped entirely
  from `token_ids` (saves training compute, matches the window Lane C
  judges).

- **Pure-rubric reward** (task 4): blended `alpha*gt + (1-alpha)*rubric`
  is gone; policy reward = mean rubric judge score for the branch.

- **Parallel SFT collector hookup** (task 5):
  - `collect_teacher_student_export_parallel` drives the parallel runner
    against an external sglang teacher.
  - `ParallelSFTDataExporter` reads parallel runner artifacts and
    applies the same two-stage SFT filter (passing GT branch + rubric
    correlation > 0.8) as the sequential `SFTDataExporter`.
  - `run_teacher_data_collect.py --use-parallel-search` + `--teacher-base-url`
    + `--lanes-*` flags + new `run_parallel_instances` ThreadPoolExecutor
    cross-instance concurrency path.
  - `sbatch_collector_sft_parallel.sbatch` + `launch_sft_parallel.sh`:
    no-GPU collector + DSv4 teacher launcher with `--dependency=after`.

- **Audit fixes** (post-launch hardening):
  - `configure_model_route` / `ModelRouteConfig` imports from
    `agent_rl.run_utils` (not `agent_rl`).
  - `SGLangChatService` imported from `swe_agent.serving` (not
    `train_agent.serving.sglang_chat_service`).
  - Global route registration is now idempotent per `(base_url,
    model_name)` so 16 concurrent runners don't race.
  - `ParallelSFTDataExporter._rubric_judge_corr_gt` joins `child_rewards`
    on `node_id` (from `summary.json`) instead of fragile ordinal pairing.

---

## Monitoring

```bash
# 1. Wait for DSv4 to be ready (~13 min cold start)
tail -F /mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs/_slurm-logs/dsv4_teacher-${DSV4_JID}.log

# 2. Once collector starts: watch summary.json
watch -n 30 "python3 -c \"import json; s=json.load(open('$ARTIFACT_ROOT/summary.json')); print(f\\\"completed={s.get('completed',0)} ok={s.get('successful_instances',0)} fail={s.get('failed_instances',0)} elapsed={s.get('seconds',0):.0f}s\\\")\""

# 3. DSv4 GPU utilization (run on the dsv4 node)
srun --jobid=$DSV4_JID --pty bash -c "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"

# 4. Per-instance progress (read from per-instance instance_record.json)
ls -la $ARTIFACT_ROOT/search_outputs/teacher_student_parallel/ | head -40
```

---

## Verification

Once `summary.json` shows N completed:

```bash
python3 $RLER/slime/train_agent/scripts/gcp/verify_sft_run.py $ARTIFACT_ROOT
```

Sample output:

```
=== SFT verification: /mnt/lustre/.../rler-runs/sft_parallel_30easy_... ===
  instances:                 30
  completed:                 28
  with at least 1 group:     30
  total groups (forks):      178
  accepted groups:           42
  acceptance rate:           23.6%
  policy SFT samples:        42
  rubric SFT samples:        42
  malformed policy:          0
  malformed rubric:          0
  branches GT-scored:        1424
  branches GT-passed (>=1):  389
  GT pass rate:              27.3%
  bank carry-forward OK:     147
  bank carry-forward died:   1
  ...
OVERALL HEALTH: OK
```

What "OK" means:

- **acceptance rate > 0%** — SFT is producing some data.
- **malformed policy / rubric = 0** — all samples have well-formed
  `prompt`+`turns` and parseable rubric JSON.
- **bank carry-forward died = 0** — every group with a non-empty prior
  bank had a non-empty bank after; no collapse.
- **GT pass rate > 20%** — reasonable for DSv4 on the 30 easy instances.

To inspect one sample concretely:

```bash
python3 $RLER/slime/train_agent/scripts/gcp/verify_sft_run.py \
    $ARTIFACT_ROOT --show-sample
```

To inspect one specific instance:

```bash
python3 $RLER/slime/train_agent/scripts/gcp/verify_sft_run.py \
    $ARTIFACT_ROOT --instance amoffat__sh-744 --json | jq .
```

Manual rubric-bank carry-forward spot-check:

```bash
INST_DIR=$ARTIFACT_ROOT/search_outputs/teacher_student_parallel/<instance-uid>
for g in $INST_DIR/groups/group_*/rubric_bank_after.json; do
    gid=$(basename $(dirname $g))
    n=$(python3 -c "import json; d=json.load(open('$g')); print(len(d.get('active_bank_after',[])))")
    echo "$gid: active_after=$n"
done
```

Expected: counts should be stable around 3–6 (matches
`max_active_rubrics=6`), NEVER 0 after the first non-empty group.

---

## Common failure modes

| Symptom | Diagnosis | Fix |
|---|---|---|
| `discovery/dsv4.json` never appears | DSv4 teacher job pending/failed | `squeue -j $DSV4_JID`, check `dsv4_teacher-*.log` |
| Discovery file exists but `ready=false` for >15 min | DSv4 sglang startup failed | Check the dsv4 log for cuda errors. Likely cause: wrong IMAGE_SQSH. Must be the **pinned** `sglang_dsv4_b200.sqsh`, NOT `:latest` (which has the fp8_einsum bug from pre-PR #25733). |
| Collector dies with `ContextWindowExceededError` | One instance hit DSv4's 80960 ctx | Expected — the runner converts to litellm abort and skips. Check the log for `lane_a mid_cp emit SKIPPED idx=N` markers. |
| Collector dies with `_ParallelRunArtifacts: no instance_record.json` | The runner crashed mid-instance | Check stack trace in collector log; the instance subdir was created but `run()` exited before the `finally`-block dump. |
| `acceptance rate = 0%` | Filter too strict | First check `GT pass rate` — if low, the model isn't solving any. If GT pass rate > 20% but acceptance still 0, check rubric correlation (look at `groups/group_NNN/rubric_samples.json` `child_rewards` vs branches' `gt.json`). |
| `bank carry-forward died > 0` | `ScoreRubricBank.update_after_round` returned empty | All rubrics had reward <= 0 (variance + redundancy + judge_error). Investigate one group's `rubric_samples.json` reward_by_rubric values. |
| DSv4 GPU util < 30% | Not enough concurrent requests | Bump `INSTANCE_WORKERS` (try 24 or 30). Each instance is bounded by Lane B (m=8) + Lane C async judging, so cross-instance concurrency is the main lever. |

---

## File map

```
slime/train_agent/scripts/gcp/
├── launch_sft_parallel.sh             ← launcher (1 line invocation)
├── sbatch_dsv4_teacher.sbatch         ← DSv4 sglang teacher (low-latency recipe)
├── sbatch_collector_sft_parallel.sbatch ← cpu-only collector
├── verify_sft_run.py                  ← health check
└── README.md                          ← legacy README, sequential pipeline
slime/train_agent/scripts/
└── run_teacher_data_collect.py        ← CLI entry point, --use-parallel-search
slime/train_agent/
├── collect_sft_rollout.py             ← collect_teacher_student_export_parallel
├── data_export.py                     ← ParallelSFTDataExporter
└── contracts.py                       ← SFTExportBundle / ExportSample
agent/swe_agent/
├── trajectory_search_parallel.py      ← TrajectorySearchParallelRunner + Lane C full-parity
├── lane_to_grpo_bundle.py             ← steps_per_round cap, pure-rubric reward
├── parallel_utils.py                  ← _do_rubric_gen, _do_judge_one, gap_corr
└── rubric_bank.py                     ← ScoreRubricBank, ExperienceRubricBank
```
