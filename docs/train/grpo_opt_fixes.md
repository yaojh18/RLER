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
