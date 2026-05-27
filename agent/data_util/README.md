# `agent/data_util/` — SWE-rebench-v2 training-set builders

Utilities that turn the upstream `nebius/SWE-rebench-v2` HF dataset into a
GRPO-ready training set whose difficulty is concentrated in the "middle band"
(instances where a baseline solver passes some but not all attempts).

The final artifact this pipeline produces is, e.g.:

```
/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench/datasets/v2_python_train_plus_sweep_middle
```

a `DatasetDict({"train": ...})` saved with `datasets.save_to_disk`. The
training-set composition is **950 base train rows + 400 sweep middle-band
rows**, repo-disjoint from the 50-instance eval split, shuffled with
`seed=17`.

---

## Pipeline at a glance

```
nebius/SWE-rebench-v2  (HF hub)
        │
        │  build_v2_python_sweep.py          [step 2]
        ▼
v2_python_sweep_shards/shard_{00..15}        (16 × 400 python rows
                                              excluding v2_python1k ids,
                                              non-empty image_name,
                                              newest first)
        │
        │  pass@4 eval w/ Qwen3.5-9B baseline [step 3, run via SLURM]
        ▼
rler-runs/559*-evp4v2-9b-9b-baseline-pysweep-sh*/
   attempt-{1..4}/nebius-eval/eval_report.json
        │
        │  list_middle_band.py               [step 4]
        ▼
v2_python_pass1to3of4.tsv                    (instance_id, pass_count,
                                              source_shard)
        │
        │  build_train_plus_sweep.py         [step 5]
        ▼
v2_python_train_plus_sweep_middle/           (final training set)
```

Inputs that exist before step 2 (`v2_python1k_split_train`,
`v2_python1k_split_eval`, `v2_python1k/instance_ids.txt`) come from an earlier
1k-instance python subset that is **out of scope of this module** — assume
they are already on disk.

---

## Step 2 — `build_v2_python_sweep.py`

Load `nebius/SWE-rebench-v2`, narrow to one language, exclude an existing id
list, drop rows with empty `image_name` (unscorable), sort by `created_at`
descending for reproducible shard order, then write fixed-size shards as
`DatasetDict({"train": ...})` directories.

```bash
python -m agent.data_util.build_v2_python_sweep \
    --base /mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench \
    --language python \
    --shard-size 400
```

Defaults reproduce the original run:
* exclude-ids → `<base>/datasets/v2_python1k/instance_ids.txt`
* out-root → `<base>/datasets/v2_python_sweep_shards`
* `HF_HOME` → `<base>/hf-cache`

Output: `shard_00 … shard_15` (16 × 400 rows ≈ 6,243 python instances after
filtering and exclusion).

---

## Step 3 — pass@4 sweep eval (external)

For each sweep shard, run the Option-A pass@4 harness with Qwen3.5-9B
baseline (sglang DP=8, 4 attempts × `WORKERS_PER_ATTEMPT` workers against a
flattened 200-task queue). The reference SLURM scripts live under
`/mnt/lustre/.../swe-rebench/RLER-eval/.../scripts/eval/` (e.g.
`eval_pass_at_4_9b_v2.slurm`, `launch_all_pass_at_4_v2.sh`).

This step is **not** part of this module — it produces the inputs that
`list_middle_band.py` consumes. Each run dir is expected to look like:

```
<runs-root>/{prefix}-sh{NN}/
    attempt-{1,2,3,4}/nebius-eval/eval_report.json
```

with `eval_report.json` containing `items: [{instance_id, passed_match, ...}]`.

---

## Step 4 — `list_middle_band.py`

Walk every `eval_report.json`, keep instances with **all four** attempts
reported, tally the per-instance pass count, and emit a TSV of rows in the
middle band (`pass_count ∈ {1, 2, 3}`):

```
instance_id\tpass_count\tsource_shard
```

`source_shard` is a tag derived from the directory glob, e.g. `pysweep-sh07`,
so downstream steps can distinguish where each id came from.

```bash
python -m agent.data_util.list_middle_band \
    --runs-root /mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler-runs \
    --out-tsv  /mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench/datasets/v2_python_pass1to3of4.tsv
```

`--source` is repeatable as `tag=globpat`; defaults reproduce the original
950-train + sweep run:

```
--source 'train=55*-evp4v2-9b-9b-baseline-train-sh*'
--source 'pysweep=559*-evp4v2-9b-9b-baseline-pysweep-sh*'
```

The printed pass-count distribution doubles as a sanity check.

---

## Step 5 — `build_train_plus_sweep.py`

Combine the base train split with the sweep middle-band picks, drop any row
whose repo is in the eval split, shuffle, and save.

```bash
python -m agent.data_util.build_train_plus_sweep \
    --base /mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench \
    --num-sweep-shards 16 \
    --source-prefix pysweep \
    --seed 17
```

Procedure:

1. `eval_repos = set(load_from_disk(EVAL)["train"]["repo"])` — repos to exclude.
2. Read `<tsv>`, keep rows where `source_shard.startswith(args.source_prefix)`
   → `sweep_ids` (400 in the canonical run).
3. Concat all sweep shards, `.filter(id ∈ sweep_ids)`, dedup by `instance_id`.
4. Concat with the base train split.
5. Drop rows whose `repo ∈ eval_repos`.
6. `shuffle(seed=17)`; save as `DatasetDict({"train": ...})` to `--out`.

Default output: `<base>/datasets/v2_python_train_plus_sweep_middle`.

---

## Reproducing the canonical artifact

The dataset that already lives at
`<base>/datasets/v2_python_train_plus_sweep_middle` was produced by running
steps 2 → 3 → 4 → 5 with all default flags above. To rebuild it from
scratch, run them in order on a node that can reach Lustre and HF.
