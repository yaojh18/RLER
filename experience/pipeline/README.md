# Automatic TTS experience pipeline

This is the tested automatic pipeline used to build the final Qwen and
Nemotron TTS experience banks. It starts directly from trajectory-search
artifact directories produced with terminal GT calculation enabled; the
`prepare-contexts` stage freezes their model-visible branch views and terminal
GT into normalized contexts. It does not train an RL policy, invoke human
review, or fit retrieval hyperparameters.

The frozen method is:

1. Retain every landed group containing both reward `2` and non-`2` branches.
2. For each group in that cohort whose historically selected branch has final
   reward other than `2`, generate a scope-specific card with
   `nvidia/zai-org/glm-5.2`; if its
   Nemotron-Ultra replay does not strictly improve tie-aware top selection or
   pairwise accuracy, refine once with `openai/openai/gpt-5.5` and replay.
3. Extract two keyword sources for every locally accepted card: model-generated
   summary phrases and deterministic phrases that occur in both the visible
   query and the card's activation contract. Their safe union is used only for
   the provisional replay.
4. Replay the provisional bank over the frozen full population. Keep a card
   only when it is retrieved at least once and its mean replay pairwise
   accuracy is at least `0.5`. Select final keywords from the extracted union
   using the same joint-judge replay results. A keyword is retained when the
   mean pairwise accuracy of the replay events attributable to that phrase is
   at least `0.5`; when literal attribution is unavailable, the replayed union
   is preserved instead of making an unsupported deletion.

Generated cards and keywords are schema-checked before filtering. No retrieval-config fitting or retrieval
hyperparameter search is included. Rejected model attempts remain in artifacts
with status `filtered` but never enter the active bank; this package contains no
human selection or refinement stage.

Run the local pipeline tests from the repository root:

```bash
PYTHONPATH=experience/pipeline:agent \
  agent/.venv/bin/python -m pytest -q experience/pipeline/tests
```

Start with either a canonical calculate-GT plan or a search-artifact root:

```bash
python -m experience_gen.cli prepare-contexts \
  --plan /path/to/plan.json --workspace-root /path/to/workspace \
  --output /path/to/contexts.json
```

The CLI intentionally exposes only the source adapter and the four production
stages: automatic generation/refinement, keyword extraction, joint replay, and
final filtering/keyword selection. Use `python -m experience_gen.cli --help`
for their arguments.
