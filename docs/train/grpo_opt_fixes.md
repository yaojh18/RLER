# GRPO naive-async optimizations & bug fixes

Captured from the 58541 → 58673 investigation (2026-05-29 → 2026-05-30). Each
section says **what was wrong / suspected**, **how we measured it**, and **the
exact change that landed**.

Validating diagnostic scripts live in
`slime/train_agent/scripts/reconstruct_weight_ages.py`,
`slime/train_agent/scripts/analyze_weight_versions.py`,
`slime/train_agent/scripts/trace_latency_breakdown.py`.

Run dirs referenced (on metavmds1):
* `/home/sihanzeng_meta_com/rler_runs/58541-grpo-soft-wvunified-stalediag/` (pre-patch)
* `/home/sihanzeng_meta_com/rler_runs/58673-grpo-soft-wvunified-stalediag/` (post-patch)

---

## 1. O(N²) `copy.deepcopy` in `backend.py:step()` (the big one)

**Symptom.** On 120-step naive trials in 58541, the wall-clock gap between
the last assistant turn's `extra.timestamp` and the `naive r=N done` log line
("phase C") was 459–684 s for yt-dlp / openmdao instances. No single
`docker_exec` exceeded 60 s, so it wasn't a hung test.

**Root cause.** `agent/swe_agent/backend.py:step()` did
`query_messages = copy.deepcopy(self.agent.messages)` *every step* and stashed
it on **both** `model_request` event's `payload.messages` *and*
`ModelTurn.query_messages`. By step N each event held a snapshot of N-1 prior
messages, so the in-memory events table grew O(N²) in total tokens. The
session-end `session.snapshot().model_dump(mode="json")` and the two
`copy.deepcopy` calls at `agent/swe_agent/naive_search.py:549` and `:552` then
walked that O(N²) blob.

Standalone repro at `slime/train_agent/scripts/repro_post_llm_oN2.py`
(loop deepcopy + json.dumps simulating the same shape) confirms clean
quadratic scaling on a devserver:

```
N_STEPS    phase1_loop   phase2_dump   phase3_post    TOTAL    RSS
   30          3.3s         1.0s          3.5s         7.8s    0.5 GB
   60         14.0s         4.0s         14.2s        32.2s    1.7 GB   (4.1×)
  120         56.3s        15.7s         59.3s       131.3s    6.6 GB   (16.8×)
```

**Why we could safely remove it.** Grep across `agent/` and `slime/`:

```
grep -r 'query_messages\|kind.*model_request' agent/ slime/
  agent/agent_rl/protocol.py:21   query_messages: List[ConversationMessage]     ← only DEFINED
  agent/swe_agent/backend.py:152  query_messages = copy.deepcopy(...)            ← only WRITTEN
  agent/swe_agent/backend.py:156  payload={"messages": query_messages}           ← only WRITTEN
  agent/swe_agent/backend.py:175  query_messages=_messages_to_protocol(...)      ← only WRITTEN
```

Zero readers anywhere — `_build_step_cards`
(`agent/swe_agent/trajectory_search.py:185+`) explicitly skips
`model_request` events, and no code reads `ModelTurn.query_messages`. The
trace dump (`messages_raw.json`) reads `r.messages`, which comes from
`self.agent.messages` (live, append-only list) via
`session.snapshot().agent.state.messages` — NOT from event payloads. The
next-step prompt composition (`agents/default.py:130`,
`message = self.model.query(self.messages)`) also reads the live list, never
the deepcopy.

**Fix** (commit `9b7c2bf`):

* `agent/swe_agent/backend.py:step()` — replace the per-step
  `copy.deepcopy(self.agent.messages)` with a `request_msg_count` back-pointer
  on the `model_request` event payload. Reconstruction (if ever needed) is
  `self.agent.messages[:request_msg_count]` since `agent.add_messages` is
  append-only (`agents/default.py:67`).
* `agent/swe_agent/backend.py:step()` — pass empty list for
  `ModelTurn.query_messages`.
* `agent/agent_rl/protocol.py` — `ModelTurn.query_messages` gets a
  `default_factory=list` so older `.pt` snapshots still deserialize.

**Validation** (58541 vs 58673 same instance):

| run | trial | A pre_llm | B gen | C post_llm | naive_done_dt |
|---|---|---|---|---|---|
| 58541 pre-patch | yt-dlp__yt-dlp-12667 r=0 | 3.6 s | 696 s | **684 s** | 1387 s |
| 58541 pre-patch | yt-dlp__yt-dlp-12667 r=2 | 1.7 s | 778 s | 585 s | 1365 s |
| 58541 pre-patch | yt-dlp__yt-dlp-12667 r=5 | 2.7 s | 881 s | 459 s | 1344 s |
| **58673 post-patch** | yt-dlp__yt-dlp-12667 r=7 | 4.2 s | 443 s | **0.1 s** | 452 s |
| 58673 post-patch | stfc__psyclone-3037 r=3 | 19.9 s | 793 s | 5.3 s | 819 s |

Phase C dropped 459-684 s → 0.1-5 s. Phase B also smaller (696 → 443 s on
yt-dlp r=7) thanks to lower GC/allocator pressure across the rest of the
trial.

---

## 2. `OVER_SAMPLING_BATCH_SIZE = 32 → 24` for the wvunified-stalediag launcher

**Why.** With the per-step deepcopy gone, the per-worker working set during
post-LLM is no longer multi-GB, so a smaller in-flight queue gives roughly
the same throughput at lower memory pressure (less GC, fewer dropped-stale
groups burning compute).

**Where.** `slime/train_agent/scripts/grpo_soft_wvunified_stalediag.slurm:70`
only. Sibling v6/v7 launchers left at 32 by request.

**Effect on 58673.** Step cadence (interval between `Timer update_weights end`
events) dropped from 15–35 min on 58541 to 6–15 min on 58673 — a 2-3×
speedup, larger than the per-trial speedup alone would predict because the
reduced concurrent working set unblocks faster broadcast-to-broadcast.

---

## 3. Weight-version recording is correct (false-alarm investigation)

**Concern.** "Most traces have the same `extra.weight_version` across all
turns, and that value ≈ `rollout_id`. Is the wv field being recorded wrong?"

**Investigation** via
`slime/train_agent/scripts/analyze_weight_versions.py` cross-referencing the
per-turn `extra.weight_version` in
`naive_out/.../rollouts/rollout_NN/messages_raw.json` against the SGLang
broadcast timeline parsed from `Timer update_weights end` events in
training.log:

```
58541:  3762 traces parsed
  json first_wv == timeline first_wv:    3738 / 3762  (99.4%)
  off by +1 (within broadcast propagation window):  24 / 3762
  73.9% of traces have single wv across all turns
```

Knife-edge spot checks confirmed the recording is correct even within
seconds of broadcast boundaries (e.g. K=2 getmoto__moto-8504 r05 first_turn
at 16:39:54 → wv=2; broadcast wv=3 fired 4 s later at 16:39:59 → next-turn
serves at wv=3). **The wv field is faithful**; the "wv ≈ rollout_id" pattern
is a real off-by-one between SGLang's 1-indexed counter (initial broadcast =
wv=1) and slime's 0-indexed `rollout_id`, plus normal queueing where a task
submitted in batch K's `_submit_until_full` may have its first turn served
after the next broadcast.

---

## 4. `_drop_stale_buffer` semantics — eldest-age is bounded ≤ 1

**Was confused.** An initial log-only reconstruction script claimed max
eldest age ramped 0 → 10 across 26 batches on 58063 (FIFO partition over all
completed groups). This was wrong.

**Reality** at `agent/swe_agent/../slime/train_agent/collect_naive_rollout_async.py:705`:

```python
def _drop_stale_buffer(current_rollout_id: int) -> None:
    kept = [g for g in _BUFFER if g.rollout_id >= current_rollout_id - 1]
    _STALE_DROPPED_GROUPS += len(_BUFFER) - len(kept)
    _BUFFER[:] = kept
```

Each `generate_rollout(K)` discards any buffered group whose **submitter**
`rollout_id < K-1`. So consumer batch K only ever sees groups submitted in
rollout K-1 or K — runtime cap of 1 step of submitter-staleness.

`_pop_groups` (line 712) then does `random.shuffle(_BUFFER); _BUFFER[:16]` —
**uniform random over the kept buffer, NOT newest-first**. The shuffle is
explicit (comment in code: "easy instances finish first and would dominate
early batches otherwise"). Override with `SWE_AGENT_NAIVE_NO_SHUFFLE=1`.

**Fix to the reconstruction script** (commit `845025d`,
`slime/train_agent/scripts/reconstruct_weight_ages.py`):

* Bucket every trace into a *submitter* window using
  `generate_rollout START rollout_id=K` log markers.
* Cap consumer-batch eligibility to groups with `submitter ∈ {K-1, K}` that
  completed before `generate_rollout(K)` starts.
* Report per-`submitter_K` drop counts (groups whose completion lands after
  `generate_rollout(submitter_K + 2)` starts — they get evicted by the next
  `_drop_stale_buffer` and never train).
* Docstring documents the log-only over-estimate vs the JSON-based ground
  truth, since `naive_done_dt − dt` approximates worker-start time, not
  first-turn time (docker setup pushes the real first turn ~30-60 s later,
  often across a broadcast boundary, which the JSON-based analyzer captures
  exactly).

---

## 5. Docker pull path is healthy (no change needed)

**Two-layer lustre cache:**

1. PATH shim at `/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench/bin/docker`.
   On `docker run`, if `docker image inspect $IMG` fails, runs
   `docker load < $DOCKER_LUSTRE_TARDIR/<sanitized_name>.tar` then exec's the
   real docker. Pass-through for all other subcommands.
2. Python-side fallback at
   `agent/swe_agent/run/benchmarks/swebench.py:188+`
   (`_try_load_from_lustre_tarball`) for code paths that bypass the CLI
   (e.g. `docker-py` SDK). Capped by
   `Semaphore(DOCKER_LUSTRE_LOAD_PARALLEL=8)` so the 8-GPU node doesn't
   saturate Lustre I/O.

Tarball cache:
`/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/swe-rebench/docker_tarballs/`
holds 1196 tarballs totalling 3.1 TB (~2.5 GB avg per image).

**Phase A measurement** on 58541 (run_dir mkdir → first asst turn):
mean 5.1 s, p50 2.7 s, p95 10.1 s, p99 58 s, max 120 s. The 691 s outlier
that initially raised suspicion was phase C (above), not phase A. No action
needed on docker pull.

---

## 6. Diagnostic scripts (all under `slime/train_agent/scripts/`)

* `reconstruct_weight_ages.py <training.log>` — log-only per-batch eldest-age
  trend with `_drop_stale_buffer` modelling + per-submitter drop counts.
* `analyze_weight_versions.py <RUN_DIR> [OUT_FILE]` — JSON ground-truth
  cross-checker. Walks `naive_out/.../messages_raw.json`, validates
  `extra.weight_version` against the `Timer update_weights end` timeline,
  emits (K, first_wv) cross-tab + per-batch eldest-age trend + match stats.
* `trace_latency_breakdown.py <RUN_DIR> [OUT_FILE]` — per-trial e2e split
  into setup / gen / eval, percentiles + per-batch summary.
* `repro_post_llm_oN2.py` — standalone offline repro of the deepcopy + dump
  shape; useful for sanity-checking future regressions or comparing Python
  versions.

---

## 7. Open follow-ups (not blocking)

* The `_collect_workspace_meta` Python heredoc at
  `agent/swe_agent/trajectory_search.py:219+` does `git diff --no-ext-diff`
  plus `sha256` of every changed/untracked file inside a single
  `env.execute()` (30 s docker timeout). On test-suite-heavy repos this
  fingerprint sweep can hit the cap; consider a size threshold on per-file
  hashing. Bounded so not urgent.
* The harvest phase (gt score returned → `_BUFFER.append`) is not logged
  per-trace. `_harvest_ready`
  (`slime/train_agent/collect_naive_rollout_async.py:585+`) batches additions
  every ~1 s during the WAITING loop. Drop a `logger.info("[INST] buffered")`
  there if per-trial harvest latency ever becomes a question.

---

## 8. `docker.py` reused `container_name` across retries → 100% of L3 trips
(commit `6d90e6e`)

**Symptom.** 58673 tripped the L3 circuit-breaker at step 33
(`infra_drop_rate=69 %`, 56/81). Siblings 58717 (rolloutlp) and 58718 (tis)
ran with the same buggy module pre-loaded and sustained 30-45 % infra-drop
the whole way; 58718 eventually tripped at its own step ~33.

**Root cause.** `agent/swe_agent/environments/docker.py:_start_container()`
generated `container_name = "swe_agent-<uuid_hex[:8]>"` **once** outside the
retry loop, then reused it for every `start_container_retries` attempt.
When attempt N actually started the container but `subprocess.run` thought it
failed (timeout from the lustre wrapper's `docker load`, flock contention
from the new per-image-flock wrapper, etc.), attempts N+1 and N+2 hit:

```
docker: Error response from daemon: Conflict.
The container name "/swe_agent-XYZ" is already in use by container "..."
```

100 % (1644/1644) of trial-level RuntimeError failures in 58717's
training.log had this exact signature.

**Fix.** Move `container_name` generation INSIDE the retry loop and rebuild
`cmd` per attempt. The half-started container from the first attempt exits
cleanly via `--rm` when its `sleep` timer expires; no need for active
`docker rm -f`.

**Validation.** 58775 (post-fix, before pull_timeout fix below): **0 name
conflicts in 492 trial failures**. The name-conflict bug never re-appeared
in 58923's full 50-step run either.

---

## 9. `pull_timeout` 120 → 600 s
(commit `d69c178`)

**Symptom (uncovered by 58775).** After the container-name fix held, a new
failure class surfaced: trials failing with
`TimeoutExpired: ... timed out after 120 seconds | docker_stderr=''`.
Cold lustre cache + 2.5 GB image tarballs routinely take >120 s to
`docker load`. 58775's `infra_drop_rate` climbed 0 → 0.26 → 0.32 → 0.39 over
3 batches; cancelled preemptively to prevent the (then-default) 0.5 L3 trip.

**Fix.** `pull_timeout: int = 120` → `pull_timeout: int = 600` in
`agent/swe_agent/environments/docker.py:37`. Same retry contract (3
attempts), each now has 10 min to complete `docker load` + container start.

**Caveat.** Some specific images still time out at 600 s under lustre
contention (observed in 58923: `meltano__sdk-2144`, `aio-libs__aiohttp-9318`,
`pymodbus-dev__pymodbus-2678`, `banesullivan__localtileserver-236`,
`getmoto__moto-8796`, `pythainlp__pythainlp-1062` — each lost 8/8 trials).
Single bad-image instances do not trip L3; cumulative ~1000 trial failures
across 50 steps in 58923 stayed under the cap.

---

## 10. `NAIVE_INFRA_DROP_RATE_LIMIT` 0.5 → 0.7
(commit `d69c178`, set in launcher)

**Why.** Defense in depth alongside §9. Even with pull_timeout=600 the
infra-drop trajectory can spike to ~0.3-0.5 transiently when many workers
hit the same cold image simultaneously. Default 0.5 trips on those spikes;
0.7 absorbs them while still tripping on cluster-wide outages.

**Where.** Exported in
`slime/train_agent/scripts/grpo_soft_wvunified_stalediag.slurm:248`
(close to `SWE_AGENT_NAIVE_MAX_PENDING`). Read by
`collect_naive_rollout_async.py:1090` via `os.getenv`.

**Validation.** 58923 saw a single-batch spike to **0.534** at step 24 (the
veth-bridge incident in §12) — would have tripped the default 0.5, was
absorbed by 0.7. Run continued and finished 50 steps.

---

## 11. NCCL FlightRecorder + lustre SAVE_DIR
(commit `72e31b8`)

**Symptom.** 58778 (post §8 + §9 fixes) ran 13.5 h, trained 29 steps, then
died on a `MegatronTrainRayActor` NCCL ALLGATHER timeout on
`TENSOR_MODEL_PARALLEL_GROUP`. The slurm log said *"Stack trace of the
failed collective not found, potentially because FlightRecorder is
disabled"* — nothing actionable. Memory was rock-stable across the run
(allocated_GB 65.22 → 65.28 over 13.5 h; reserved_GB 129.85 throughout),
ruling out OOM/leak. Most likely transient IB/RDMA blip on
`TENSOR_MODEL_PARALLEL_GROUP` rank 0.

**Fix part A — NCCL diagnostics.** Added to the launcher (after
`NAIVE_INFRA_DROP_RATE_LIMIT`):

```bash
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=$RUN_DIR/_logs/nccl_trace
export NCCL_DEBUG=WARN
```

Per-rank ring buffer of last 2000 NCCL ops (~2 min at the observed
~17 TP-ops/sec rate). Next hang will write per-rank stack-traced dumps
under `$RUN_DIR/_logs/nccl_trace_*`.

**Fix part B — Lustre SAVE_DIR.** 58778's `SAVE_DIR=/mnt/localssd/...` was
unrecoverable post-crash. The compute nodes' `/mnt/localssd` *isn't* wiped
by slurm (verified — old job dirs from 57661/57673/57827 still present on
a4-2's localssd), BUT 58778's `iter_*` directories were gone anyway from
all 4 of its assigned nodes (a4-138/120/65/36). The slurm log explicitly
reported `successfully saved checkpoint from iteration 9` and `... 19` to
localssd, but follow-up `find` on every job node showed only `wandb/` and
`rollout/` subdirs — no `iter_*`, no `latest_checkpointed_iteration.txt`.
Most likely an out-of-band cleanup hits localssd asynchronously (another
tenant's prolog, kernel pressure, or a node reboot between jobs).

The fix moves SAVE_DIR to lustre:
```
SAVE_DIR=/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/rler_ckpts/${SLURM_JOB_ID}-...
```
With `save-interval=10` and Qwen3.5-9B bf16 (~18 GB/ckpt), a 60-rollout run
uses ~108 GB on lustre — acceptable given lustre is at 85-95 % per tenant.

**Validation.** 58923 produced **5 resumable lustre checkpoints**:
`iter_9`, `iter_19`, `iter_29`, `iter_39`, `iter_49`. All persist after job
end, all readable for resume.

---

## 12. Docker veth bridge exhaustion (NOT FIXED — documented)

**Symptom (surfaced in 58923 step 24).** A single batch hit
`infra_drop_rate=0.534` with **0** new `TimeoutExpired` errors. Inspecting
the actual error stack:

```
docker: Error response from daemon: failed to set up container networking:
failed to create endpoint swe_agent-XXX on network bridge:
adding interface vethXXXX to bridge docker0 failed: exchange full
```

Linux kernel limits the number of veth interfaces a single bridge can
hold (`docker0` default ≤ 1024). With ~1000+ short-lived containers churning
during long runs and `--rm` cleanup running asynchronously, veth pairs
accumulate faster than they can be torn down.

**Current behaviour.** Self-recovers as containers cycle out — 58923's
0.534 spike at step 24 was followed by 0.0 for the next several batches.
Bounded enough that the 0.7 L3 limit absorbs it.

**Fix candidates (none landed yet).**
1. Periodic `docker network prune -f` on each compute node — needs a slurm
   prolog or background cron alongside the worker processes.
2. Raise the kernel bridge limit
   (`sysctl net.bridge.bridge-nf-call-iptables=0` won't do it directly;
   need to bump `MAX_BRIDGE_PORTS` which is compile-time).
3. Use `--network=none` for trials that don't need network egress (most
   swe-rebench instances don't talk to anything external — the sglang HTTP
   endpoints are reached via the agent process, not from inside the
   per-trial container).
4. Spin up containers in a per-node user-namespace network rather than
   sharing `docker0`.

For now: tolerate the occasional spike. If sustained >0.3 across many
batches becomes the norm, prioritise option (1).

---

## 13. Trainer NCCL hang post-mortem (one-off; recovery validated)

**58778's actual failure mode** (detail backing §11). The crashing op:

```
[Rank 0] Watchdog caught collective operation timeout:
  WorkNCCL(SeqNum=809868, OpType=ALLGATHER,
           NumelIn=100663296, NumelOut=100663296,
           Timeout(ms)=600000) ran for 600018 ms
  [PG ID 5 PG GUID 22(TENSOR_MODEL_PARALLEL_GROUP) Rank 0]
  last enqueued work: 809948, last completed work: 809867
```

Rank 0 had **enqueued 80 ops AHEAD** of the one that hung — its CUDA queue
was healthy. The hang was on collective communication with a peer rank.
The 100,663,296 = 96 × 1024² elements ≈ 200 MB bf16 → normal Megatron
column-parallel ALLGATHER, not an anomalous tensor size. Work seq 809 868
on TP group → ~17 ops/sec over 13.5 h, normal Megatron op rate.

Other rank logs showed `Last enqueued NCCL work: 58, last completed: 58`
for `default_pg` — red herring. Those are different communicators; the
"58" count just means those ranks had been quiescent on default_pg,
blocked downstream of the actual culprit (the TP-group ALLGATHER).

**Verdict.** Transient IB/RDMA blip on actor node `a4-138` during
58778's step 29 training. Memory profile across the entire run was flat;
broadcast 2 s before the hang was healthy; backend.py / docker.py changes
don't touch NCCL paths.

58923 — same code, same hyperparameters, different actor node — ran 50
steps over 24 h without any NCCL timeout. Confirms one-off. The
FlightRecorder hook (§11) is in place for the next occurrence.

---

## 14. Resume launcher pattern
(commit `c3db812`, validated by 59135)

**Why.** 58923 hit the 24 h slurm time-limit at step 50/60. The
`grpo_resume_from50.sh` + `..._resume_from58923_iter49.slurm` pair finishes
the last 10 rollouts from iter_49 ckpt **without touching the canonical
launcher**. Pattern is reusable for any future time-truncated run.

**Two-file diff:**
- `configs/grpo_resume_from50.sh` — copy of `grpo.sh` with
  `--start-rollout-id 0` → `--start-rollout-id 50`. `--finetune` kept
  (weights-only resume — the path validated in 58514 per
  `project_slime_grpo_resume_gotcha` memory; optimizer state restarts fresh).
- `scripts/grpo_soft_wvunified_stalediag_resume_from58923_iter49.slurm` —
  copy of the canonical launcher with `LOAD_DIR` → 58923's saved ckpt,
  job name + SAVE_DIR retagged, `--config-path` pointing at the resume
  config.

**Slime's resume contract** (read off `slime/utils/arguments.py:1560-1584`):
- If `args.load` points at a dir with `latest_checkpointed_iteration.txt`
  → keep `args.start_rollout_id` as set; Megatron loads the iter dir.
- If `args.load` is missing or HF-only → force `start_rollout_id=0` and
  `args.finetune=True, no_load_optim=True, no_load_rng=True`.

So the resume just needs both:
(a) `--load` pointing at a real Megatron ckpt dir with `latest_checkpointed_iteration.txt`,
(b) `--start-rollout-id N` set to the next rollout index.

---

## 15. Combined fix tally as of `c3db812`

| commit  | file                                                  | what                                                          |
|---------|-------------------------------------------------------|---------------------------------------------------------------|
| 9b7c2bf | `agent/swe_agent/backend.py` + `agent_rl/protocol.py` | drop O(N²) `query_messages` deepcopy in `step()` (§1)         |
| 9b7c2bf | `grpo_soft_wvunified_stalediag.slurm`                 | OVER_SAMPLING_BATCH_SIZE 32 → 24 (§2)                         |
| 6d90e6e | `agent/swe_agent/environments/docker.py`              | fresh container_name per retry attempt (§8)                   |
| d69c178 | `agent/swe_agent/environments/docker.py`              | pull_timeout 120 → 600 s (§9)                                 |
| d69c178 | `grpo_soft_wvunified_stalediag.slurm`                 | NAIVE_INFRA_DROP_RATE_LIMIT 0.5 → 0.7 (§10)                   |
| 72e31b8 | `grpo_soft_wvunified_stalediag.slurm`                 | NCCL FlightRecorder env + lustre SAVE_DIR (§11)               |
| c3db812 | `configs/grpo_resume_from50.sh` (new)                 | resume config: --start-rollout-id 50 (§14)                    |
| c3db812 | `scripts/..._resume_from58923_iter49.slurm` (new)     | resume launcher: LOAD_DIR → 58923's iter_49 ckpt (§14)        |

Run progression validating the stack:

| run    | died/finished at | failure / outcome                                                                  |
|--------|-------------------|------------------------------------------------------------------------------------|
| 58541  | step 33 / 60      | L3 trip 69 % (container_name retry bug — §8)                                       |
| 58673  | step 33 / 60      | same bug (the original; spawned this investigation)                                |
| 58717  | step ~33 / 60     | same bug with old module pre-loaded (per-sample-mean + rolloutlp variant)          |
| 58718  | step ~33 / 60     | same bug with old module pre-loaded (per-sample-mean + tis variant)                |
| 58775  | step 9 / 60       | cancelled — pull_timeout (§9) climbing toward L3                                   |
| 58778  | step 29 / 60      | trainer NCCL hang on TP group (§13) — transient                                    |
| 58920  | ~21 min / 60      | cancelled — superseded by 58923 to pick up NCCL trace + lustre SAVE_DIR            |
| **58923** | **step 50 / 60** | **finished 24 h slurm time-limit cleanly**; 5 lustre ckpts saved                   |
| 59135  | TBD               | resume from 58923's iter_49, target steps 50-59                                    |
