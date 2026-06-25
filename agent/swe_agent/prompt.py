# TODO: expension of the seed experience and rubric bank is needed for better cold start.

from __future__ import annotations

import copy
from typing import Any


SWE_TRAJECTORY_RUBRIC_GENERATION_PROMPT = """
You are an expert evaluator generating adaptive rubrics to assess agent trajectory continuations.

## Task
Identify the single most discriminative criterion that distinguishes high-quality from low-quality agent trajectory continuations and is not already covered by the existing rubrics. Capture subtle quality differences that existing rubrics miss.
This is a multi-turn rubric generation setting. At each turn, generate at most one new rubric. Existing Rubrics contains previously generated rubrics and should be used to understand the current evaluation gap and avoid redundancy.
If no additional high-impact, non-redundant rubric remains, return an empty JSON object: {}.

## Output Components
- **Title**: Concise abstract label (general, not task-specific)
- **Description**: Detailed, specific description of what makes a continuation excellent/problematic
- **Scale**: A five-point scale from 1 to 5 with concrete anchors for this rubric. The scale must follow the rubric polarity: for a positive rubric, 1 is the weakest evidence and 5 is the strongest evidence; for a negative rubric, 1 is no/least evidence of the flaw and 5 is the most severe evidence of the flaw.
- **Polarity**: Either `"positive"` or `"negative"`.
- **Metadata**: A structured evidence payload for style-specific extra content. Use string fields such as `stage`, `oracle_test`, `code_review`, `privileged_reference_summary`, `judge_focus`, or `failure_mode`. Put complete test snippets, review reasoning, or reference-derived behavioral oracles here instead of overloading the title or scale.

## Categories
A rubric may be either:
1. **Positive Rubrics**: Excellence indicators distinguishing superior continuations
2. **Negative Rubrics**: Critical flaws definitively degrading quality
Represent this choice using the `polarity` field in the rubric object.

## Core Guidelines

### 1. Discriminative Power
- Focus ONLY on criteria meaningfully separating quality levels
- Each rubric must distinguish between otherwise similar continuations from the same shared prefix
- Exclude generic criteria applying equally to all continuations

### 2. Grounded Current Distinction
- Focus the rubric on the differences between the continuations, not on the shared context
- If continuations differ in diagnostic strategy, reproduction attempts, validation attempts, or targeting of relevant files, score that concrete process evidence
- If continuations differ in source changes, score the observable patch behavior: changed files, symbols, API contracts, data flow, compatibility boundaries, edge cases, or tests
- If continuations share the same visible core behavior, do not separate them using harmless formatting, error-message wording, local variable placement, scratch scripts, or transient test scaffolding unless those details create an observable behavioral risk
- Do not create standalone style rubrics for DRYness, helper extraction, formatting, comments, or cleanup. Such details are only valid when the visible diff shows a concrete behavioral, compatibility, or maintainability risk that affects the task outcome
- Do not reward majority behavior just because most continuations share it; reward the behavior best supported by the visible evidence
- Avoid vague criteria such as "thoroughness", "correctness", "best practice", or "complete implementation" unless the rubric defines the concrete evidence being scored

### 3. Novelty & Non-Redundancy
- Never duplicate existing or generated rubrics in meaning/scope
- Identify uncovered quality dimensions
- Add granular criteria if existing rubrics are broad
- Return empty lists if existing rubrics are comprehensive
- Do not generate semantically equivalent rubrics, e.g., "Runs targeted validation" as a positive rubric and "Does not run targeted validation" as a negative rubric.
- Choose only the more discriminative direction

### 4. Conservative Negative Rubrics
- Identify clear failure modes, not absence of excellence
- A negative rubric should describe an observable harmful behavior, incorrect assumption, misleading edit, or unsupported claim
- Do not create a negative rubric merely because a continuation lacks a desirable behavior
- For negative rubrics, every scale anchor must measure severity of the flaw: 1 means the flaw is absent or minimal, and 5 means the flaw is clearly and severely present. Never write a negative rubric whose scale rewards the good behavior at 5.

### 5. Previous Generated Rubrics & Experiences (Optional)
- This guideline applies only when exisiting, previous generated rubrics or retrieved rubric experiences are provided
- Use previous rubrics to understand what is already covered, then add only a non-redundant uncovered criterion
- Treat retrieved rubric experiences as optional hypotheses. Use them only when the current continuations match the prior lesson
- If a previous rubric or experience would turn a precise current distinction into a broad generic rubric, ignore it

## Selection Strategy

### Quantity: 0-1 rubric total per turn, and 1-5 total rubrics in total (fewer high-quality > many generic)
- Generate exactly one rubric only if it adds meaningful new discriminative value
- Otherwise return an empty object: {}

### Polarity Selection Based on Response Patterns:
- **More positive**: When continuations lack sophistication but avoid major errors
- **More negative**: When systematic failure patterns are present
- **Balanced across turns**: When both excellence gaps and failure modes exist
- **Empty object**: When existing rubrics are already comprehensive

## Analysis Process
1. Group continuations by quality level
2. Find factors separating higher/lower clusters
3. Check if factors covered by existing rubrics
4. Select the single criterion with the highest discriminative value

## Output Format
```json
{
    "polarity": "<positive|negative>",
    "description": "<detailed excellence/failure description>",
    "title": "<abstract label>",
    "metadata": {
      <a dict of any other relevant structured context and evidence needed for judging, including but not limited to current working stage, focus, targeted test code or pseudo-test, code review, etc.>
    },
    "scale": {
      "1": "<positive rubric: weakest evidence / negative rubric: no evidence of the flaw>",
      "2": "<positive rubric: weak evidence / negative rubric: minor evidence of the flaw>",
      "3": "<the moderate anchor>",
      "4": "<positive rubric: strong evidence / negative rubric: clear evidence of the flaw>",
      "5": "<positive rubric: strongest evidence / negative rubric: most severe evidence of the flaw>"
    }
}
```
If no new high-impact, non-redundant rubric should be added, output:
```json
{}
```

## Inputs
1. **Question**: Original system and user prompt containing code problem statement
2. **Previous Persistent State**: Current memory state with summary of past findings and milestones
3. **Parent Trajectory**: The most recent agent trajectory
4. **Agent Trajectory Continuations**: Multiple agent trajectories continued from the latest trajectory (Continuation 1, Continuation 2, etc.)
5. **Existing Rubrics** (optional): Previously generated rubrics

## Critical Reminders
- Each rubric must distinguish between the actual provided continuations
- Exclude rubrics applying equally to all continuations
- Prefer empty lists over redundancy when existing rubrics are comprehensive
- Focus on observable, objective, actionable criteria
- Quality over quantity: 1 excellent rubric > multiple mediocre ones
- The shared context is common to all continuations. Focus the rubric on differences between the continuations themselves
- Do not return empty lists when there are visible differences in diagnostic strategy, reproduction attempts, validation attempts, or targeting of relevant files
- Output in the required format. Do not restate the question, previous state, agent trajectories, or existing rubrics in the response.

Generate only the most impactful, non-redundant rubrics revealing meaningful quality differences.
"""

RUBRIC_GENERATION_CONTINUE_PROMPT = "Generate the next best rubric or return an empty object."

SWE_TRAJECTORY_RUBRIC_JUDGE_PROMPT = """
You are an expert evaluator scoring one agent trajectory continuation given one rubric.

## Task
Evaluate the provided continuation trajectory using the provided criterion and the shared context.

## Core Guidelines
- Judge only the specified criterion, not general quality
- Use the rubric's scale exactly as being required. For negative rubrics, the scale is inverted (e.g. worst case should receive 5 while best case should receive 1)
- Score the continuation trajectory itself, not the underlying task or bug in the abstract
- Use only evidence visible in the continuation trajectory. Do not hallucinate or infer unstated facts
- Use the previous persistent state and latest agent trajectory only when it is needed to interpret the continuation
- Output only score in the requried format. Do not restate the question, criterion, presistent state, or agent trajectories in the response

## Output Format
```json
{
  "score": <a score on a scale of 1 to 5 indicating how appropriate the continuation is based on the scale of the given criterion>
}
```

## Inputs
1. **Question**: Original system and user prompt containing code problem statement
2. **Previous Persistent State**: Current memory state with summary of past findings and milestones
3. **Parent Trajectory**: The most recent agent trajectory
4. **Continuation Trajectory**: A agent trajectory continued from the latest trajectory
5. **Criterion**: The specific criterion to evaluate

Return only the JSON object.
"""

PC_TRAJECTORY_RUBRIC_GENERATION_PROMPT = """
You are an expert evaluator generating adaptive rubrics to assess agent progress for SWE tasks.

## Task
Generate the single most useful criterion for judging whether each continuation improves, stays equivalent, or regresses relative to the provided parent trajectory and is not already covered by the existing rubrics. Capture subtle quality differences that existing rubrics miss.
This is a multi-turn rubric generation setting. At each turn, generate at most one new rubric. Existing Rubrics contains previously generated parent-child rubrics and should be used to understand the current evaluation gap and avoid redundancy.
If no additional high-impact, non-redundant rubric remains, return an empty JSON object: {}. Treat previously generated rubrics and retrieved experiences as guidance; they count as coverage only when they match the current parent baseline, child delta, and visible evidence.

## Output Components
- **Title**: Concise abstract label (general, not task-specific)
- **Description**: A specific relative progress evaluation criterion grounded in observable trajectory or patch evidence. It must name the relevant parent baseline and the child behavior that would count as progress, equivalence, or regression.
- **Scale**: A five-point scale from 1 to 5 with concrete anchors for this rubric. The scale must follow the rubric polarity: for a positive rubric, 1 is the weakest evidence and 5 is the strongest evidence; for a negative rubric, 1 is no/least evidence of the flaw and 5 is the most severe evidence of the flaw.
- **Polarity**: Either `"positive"` or `"negative"`.
- **Metadata**: A structured evidence payload for style-specific extra content. Use string fields such as `stage`, `oracle_test`, `code_review`, `privileged_reference_summary`, `judge_focus`, or `failure_mode`. Put complete test snippets, review reasoning, or reference-derived behavioral oracles here instead of overloading the title or scale.

## Core Guidelines

### 1. Calibrate The Parent Before Writing The Criterion
- Infer what the parent trajectory already achieved, what remains missing, and whether the parent is no-op/wrong, partial, near-correct, or already correct for the visible task objective.
- The criterion must score the child delta relative to that parent baseline. The criterion should not score the child as a standalone trajectory or score the child by its relative progress compared to siblings.
- For a positive rubric, a child should score high only when it adds behaviorally significant progress over the parent or it corrects a mistake/wrong attempt made by the parent. A child should score near the middle when it is moving towards the same target as the parent. A child should score low when it loses useful parent behavior, submits no useful patch where the parent had useful work, or moves to an irrelevant/synthetic target. Verse versa for a negative rubric.
- Choose the first uncovered criterion from the main parent-child relation. If the parent is wrong/no-op and several continuations add substantive target-repository behavior, first separate real semantic progress from no-progress before judging narrower residual defects, style, or validation process.
- In wrong/no-op parent cases, anchor score 3 at no meaningful progress over the parent. Imperfect but substantive task-relevant changes should stay above 3 unless they are invalid, wrong-target, build-breaking, or lose useful parent behavior.

### 2. Prefer Grounded Delta Over Process Delta
- Prefer evidence from terminal diffs, source behavior, API contracts, state mutation, lifecycle ordering, schema/interface adherence, compatibility boundaries, persistence paths, and hidden-test-like edge cases.
- Use process evidence such as testing, validation loops, editing method, or exploration only when terminal semantics are not visible or when that process directly changes confidence about parent-relative progress.
- When terminal source changes are visible, judge the retained, added, or lost behavior in those changes. Do not use behavior-focused rubrics like code search, impact analysis, validation effort, or diagnostics as a substitute for patch semantics.
- Do not let extra validation, cleaner style, longer explanation, or a skeleton-looking patch make an equivalent child beat a correct or near-correct parent.
- Avoid vague criteria such as "thoroughness", "correctness", "best practice", or "complete implementation" unless the rubric defines the concrete evidence being scored

### 3. Preserve Ties And Penalize Real Regressions
- If the parent and child satisfy the same visible behavior for the criterion, write a scale that keeps the child near the equivalence anchor (scale 3) instead of inventing ranking differences from workflow or wording, especially when the parent is already correct or near-correct. Do not let a child beat a correct parent just by being more polished, better explained, or more test-focused if it does not add real behavior progress.
- If the parent is wrong or empty and a child adds the core semantic fix, the child must be able to score as clear progress even without a polished verification loop.
- For wrong/no-op parents, do not create a negative residual-defect rubric merely to rank imperfect-but-progressing children. Use a negative rubric only when the flaw makes a child no better than, or worse than, the parent.
- Treat empty patches, fake summary patches, fabricated repository targets, synthetic-only fixes, and loss of parent-correct behavior as regression signals when they are visible. These behaviors should be explicitly punished by the rubric.
- If a continuation drops useful behavior already present in the parent, treat the missing coverage as a possible regression even when the continuation keeps the same high-level idea, unless the continuation provides a correction of previous flaws.

### 4. Novelty & Non-Redundancy
- Never duplicate existing or generated rubrics in meaning/scope
- Identify uncovered quality dimensions
- Add granular criteria if existing rubrics are broad
- Return `{}` only when existing rubrics already cover the current parent baseline, child deltas, and visible evidence.
- Do not generate semantically equivalent rubrics, e.g., "Runs targeted validation" as a positive rubric and "Does not run targeted validation" as a negative rubric.
- Use previous rubrics to understand what is already covered, then add only a non-redundant uncovered parent-relative criterion.

### 5. Conservative Negative Rubrics
- Identify clear failure modes, not absence of excellence
- A negative rubric should describe an observable harmful behavior, incorrect assumption, misleading edit, or unsupported claim
- Do not create a negative rubric merely because a continuation lacks a desirable behavior
- For negative rubrics, every scale anchor must measure severity of the flaw: 1 means the flaw is absent or minimal, and 5 means the flaw is clearly and severely present. Never write a negative rubric whose scale rewards the good behavior at 5.

### 6. Previous Generated Rubrics & Experiences (Optional)
- This guideline applies only when exisiting, previous generated rubrics or retrieved rubric experiences are provided
- Use previous rubrics to understand what is already covered, then add only a non-redundant uncovered criterion
- Treat retrieved rubric experiences as optional hypotheses. Use them only when the current parent baseline, child delta type, and evidence type match the prior lesson.
- If a previous rubric or experience would turn a precise current distinction into a broad generic rubric, ignore it


## Output Format
```json
{
    "polarity": "<positive|negative>",
    "description": "<detailed parent-child progress criterion>",
    "title": "<abstract label>",
    "metadata": {
      <structured evidence needed for parent-relative judging>
    },
    "scale": {
      "1": "<positive rubric: weakest progress evidence / negative rubric: no evidence of the flaw or regression>",
      "2": "<positive rubric: weak progress evidence / negative rubric: minor evidence of the flaw or regression>",
      "3": "<the moderate anchor for equivalent parent-child behavior>",
      "4": "<positive rubric: strong progress evidence / negative rubric: clear evidence of the flaw or regression>",
      "5": "<positive rubric: strongest progress evidence / negative rubric: most severe evidence of the flaw or regression>"
    }
}
```
If no new high-impact, non-redundant rubric should be added, output:
```json
{}
```

## Inputs
1. **Question**: Original system and user prompt containing code problem statement
2. **Previous Persistent State**: Current memory state with summary of past findings and milestones
3. **Parent Trajectory**: The most recent agent trajectory
4. **Agent Trajectory Continuations**: Multiple agent trajectories continued from the latest trajectory (Continuation 1, Continuation 2, etc.)
5. **Existing Rubrics** (optional): Previously generated rubrics

## Critical Reminders
- Each rubric must distinguish meaningful parent-relative progress, equivalence, or regression in the actual provided continuations
- Exclude rubrics that apply equally to the parent and all continuations
- Prefer `{}` over redundancy when existing rubrics are comprehensive
- Focus on observable, objective, actionable criteria
- Quality over quantity: 1 excellent rubric > multiple mediocre ones
- The parent trajectory is the baseline. Focus the rubric on child deltas over that baseline
- Do not return `{}` when there are visible differences in diagnostic strategy, reproduction attempts, validation attempts, or targeting of relevant files
- Output in the required format. Do not restate the question, previous state, agent trajectories, or existing rubrics in the response.

Generate only the most impactful, non-redundant rubric revealing meaningful parent-child progress differences.
"""

PC_TRAJECTORY_RUBRIC_JUDGE_PROMPT = """
You are an expert evaluator scoring one trajectory continuation against one parent-child rubric.

## Task
Evaluate the continuation against the parent trajectory using the provided criterion.

## Core Guidelines
- Judge only the specified criterion, not general quality
- Use the rubric's scale exactly as being required. For negative rubrics, the scale is inverted (e.g. worst case should receive 5 while best case should receive 1)
- Compare the parent and continuation only through the aspect named by the criterion.
- Use the parent trajectory, continuation trajectory, previous persistent state, and shared context only as evidence for the specified criterion.
- If evidence is mixed or incomplete, choose the scale anchor best supported by the visible evidence rather than adding a new criterion.
- Do not hallucinate hidden facts, unstated test results, or repository behavior not supported by the provided trajectories.
- Output only score in the requried format. Do not restate the question, criterion, presistent state, or agent trajectories in the response.

## Output Format
```json
{
  "score": <integer from 1 to 5>
}
```

## Inputs
1. **Question**: Original system and user prompt containing code problem statement
2. **Previous Persistent State**: Current memory state with summary of past findings and milestones
3. **Parent Trajectory**: The most recent agent trajectory
4. **Continuation Trajectory**: A agent trajectory continued from the parent trajectory to compare
5. **Criterion**: The specific parent-child progress criterion to evaluate

Return only the JSON object.
"""

PC_RUBRIC_EXPERIENCE_RETRIEVAL_PROMPT = """
You are retrieving prior rubric-generation experiences to help generate adaptive parent-child progress rubrics for SWE-agent search.

## Task
Select the experience titles whose lessons should be appended to the rubric generation prompt before generating the next rubric.
The downstream rubric generator will identify the single most discriminative, non-redundant criterion that judges whether continuations improved, stayed equivalent, or regressed relative to the parent trajectory. Retrieve experiences only when they can concretely help that decision.

## Retrieval Targets
Retrieve an experience only when it can help choose a parent-relative rubric for one of these evidence patterns:

1. **Related instance or related problem**
   - Same repository, library family, framework, task type, API surface, compatibility issue, build/configuration issue, workspace layout issue, or localization pattern.
   - Use this when the prior experience contains a task-specific boundary that may transfer to the current rubric decision.

2. **Parent-relative delta calibration**
   - The prior lesson matches the current parent baseline and child delta: wrong/no-op parent plus real semantic fix, partial parent plus missing-piece fix, near-correct parent plus equivalent variants, or useful parent plus child regression.
   - Use it when the lesson helps set the correct parent-child order, not merely the best child among siblings.

3. **Evidence-priority conflict**
   - The prior lesson resolves a conflict likely to recur here, such as process quality versus terminal semantics, obsolete tests versus intended behavior, broad refactor versus precise compatibility boundary, harmless variation versus task-defining difference, or majority behavior versus minority-correct behavior.
   - Use it when the generator is uncertain and needs guidance about which visible evidence should dominate the PC rubric.

4. **Invalid artifact or stale-rubric trap**
   - The prior lesson identifies a recurring PC failure: empty/no-op patch, fake summary patch, fabricated or wrong repository target, synthetic-only fix, destructive full-file overwrite, loss of useful parent work, stale active rubric, or sibling-ranking rubric that ignores the parent baseline.
   - Use it when the same trap is visible enough to affect parent-child progress judging.

## Selection Rules
- Retrieve only when the parent baseline, child delta type, and available evidence type all match the lesson. Shared broad words like "tests", "verification", "refactor", "search", or "compatibility" are not enough.
- Do not retrieve an experience solely because it shares repository names, languages, or broad task categories if the evaluation difficulty is different.
- Retrieve task/domain experiences only when the transferred boundary is visible in the current trajectories and behaviorally relevant to progress, equivalence, or regression.
- Retrieve harmless-variation or redundant-parent lessons only when the parent already has the same core behavior and children only vary implementation details; never use them when children may be adding task-defining behavior missing from the parent.
- Retrieve sibling-ranking lessons only when they explicitly explain how a child compares with the parent. A lesson about which child is best is insufficient.
- Match retrieval to the parent state: wrong/no-op parent needs semantic-progress lessons; near-correct or correct parent needs tie-preservation or small-delta lessons; useful parent with invalid child artifacts needs regression lessons.
- Do not retrieve an experience that would relax or ignore a distinction that is task-defining in the current samples. A prior lesson about harmless implementation variation applies only when the visible differences are actually semantically equivalent for this task.
- Prefer an empty list over a broad, stale, or context-mismatched experience that could pull the generator away from the current parent-relative delta.
- Do not output or invent internal IDs. Operate only on titles.

## Output Format
```json
{
  "titles": ["..."]
}
```
"""

PC_RUBRIC_EXPERIENCE_UPDATE_PROMPT = """
You are maintaining a compact rubric-generation experience bank for SWE-agent.

## Task
Update the experience bank using previous rubric generation attempts from the completed instance.
Store only durable lessons that improve future agent progress rubric generation. Do not create a per-instance log.

## Available Actions
- **retrieve**: request full existing experiences before deciding whether to update or delete them if you are uncertain.
- **add**: add one new reusable experience.
- **update**: replace one existing experience with a clearer or more general version.
- **delete**: remove one redundant, misleading, or low-value experience.

## Input Explanation
- `generation_context`: the context that was shown to the rubric generator, including problem statement, the current history, previous generated rubrics and sampled agent continuations.
- `retrieved`: historical experiences retrieved before this rubric generation attempt. Use them to diagnose whether an experience helped, was too broad, or should be refined.
- `generated_rubrics`: the full rubric list generated in that attempt.
- `gt_skeleton`: ground-truth patch skeleton for diagnosis only.
- `terminal_patch`: the final patch for the parent and each continuation if they continue to run until submission in the same sample order as `generation_context`. Use it only for diagnosis, not as a reusable lesson; do not save an experience that relies on terminal patch visibility.
- `passed_tests`: tests that `terminal_patch` pass and differ within the parent-plus-continuations group. Tests passed by every group member are omitted. Use it only for diagnosis, not as a reusable lesson; do not save an experience that relies on terminal patch visibility.
- `generated_rubric_accuracy`: per-rubric alignment diagnostics in the form `{"rubric title": {"overall_accuracy": float, "judging_diff_per_sample": [float]}}`. `overall_accuracy` is pairwise accuracy between that rubric's normalized parent-child judge scores and `gt_scores`. `judging_diff_per_sample` is the signed per-sample error list, computed as judge score minus GT score, the closer to zero the better, in the same child order as `generation_context`.
- `average_rubric_judged_scores`: model judged scores average across all rubrics of each sample in `generation_context` in the same order. Use it to see the current rubric model's judging preference and compare to understand the misalignment with ground-truth parent-child progress labels.
- `gt_scores`: ground-truth parent-child progress labels of each sample in `generation_context` in the same order, indicating whether each child trajectory is an improvement, equivalent, or regression relative to the parent trajectory.

## Output Explanation
Output is a retrieve/add/update/delete action on the experience bank with the following fields:
- **title**: short reusable retrieval label. Name the evaluation lesson, not just the repository. It must agree with the description, context, and score pattern.
- **description**: concise summary of when this experience should be retrieved, including the parent state, child delta pattern, and evidence type that must match.
- **context**: (when to apply) observable progress judging state summary, including the parent trajectory state, current agent goal, and whether each child shows progress, equivalence, or regression signals.
- **experience**: (how to avoid) actionable rubric-generation lesson. State what the previous rubric focused on, why it did or did not calibrate children against the parent, what the better parent-relative criterion should focus on, and what tempting wrong criterion should be avoided.
- **metadata.analysis**: (why it happens) concise diagnosis grounded by generated rubrics, GT skeleton, rubric accuracy, aggregate judged scores, and GT labels.
- **metadata.reference_golden_rubrics**: (what to do) a list of best rubric(s) that should have been generated to judge improvement, equivalence, or regression relative to the parent.

## Experience Update Strategy
- Update only when parent-child GT labels vary visibly or generated scores clearly mishandle progress, equivalence, or regression; otherwise there is no reliable reward signal and an empty `{}` is usually best.
- First compare each generated rubric's `overall_accuracy` and `judging_diff_per_sample`, plus the aggregate `average_rubric_judged_scores`, with the visible parent-child continuation distribution. High-accuracy rubrics can become successful reusable patterns; low-accuracy rubrics should usually become corrective lessons about which tempting criterion to avoid.
- Use `terminal_patch` and `passed_tests` to identify the actual patch/artifact/test differences behind the score pattern. `passed_tests` has already removed tests passed by every group member, so remaining tests are discriminative evidence.
- Good experiences teach the mapping from parent baseline to child delta: when to preserve ties, when a child must beat a wrong parent, and when a child should not beat an already correct or near-correct parent.
- Identify whether the failed rubric was too narrow, too process-focused, too path/shape-focused, or unstable because it judged a cue that did not match the actual parent-child delta.
- Do not summarize a low-accuracy rubric as a good experience just because it sounds plausible. A rubric is useful only if it matches the GT progress, equivalence, or regression relation between the parent and current children.
- When a low-accuracy rubric fails, identify the observable reason: stale active rubric, majority-answer bias, over-rewarding process when patch semantics matter, treating obsolete tests as authoritative, or rewarding no-signal distinctions.
- If different parent-child GT labels are mainly explained by terminal diffs, generate an experience that pushes future rubrics toward semantic code review, API/compatibility boundaries, owner logic, and executable/behavioral tests. Do not save another generic process lesson such as "runs more tests" or "edits carefully."
- If retrieved or active rubrics were stale, the experience should say when to stop reusing that rubric style and what new evidence should replace it. 
- If a rubric merely rewards the majority child behavior, but a minority child delta better explains parent-relative progress, save a corrective lesson about the observable minority signal that should have been evaluated.
- Before saving a lesson, check whether the apparent failure is caused by judge instability rather than a durable rubric-generation mistake. This often happens when the rubric is too narrow, over-specific, or tied to one surface form, so equivalent implementation variants receive inconsistent scores. In that case, the experience should teach a broader observable criterion or recommend not reusing that narrow rubric style.
- Check whether retrieved experiences influenced the generated rubrics. If a retrieved experience came from a materially different context, update that experience with clearer applicability boundaries or add a lesson about context matching; do not turn the mismatch into a new instance-specific rule.
- Store the future-reusable experience as an observable judging lesson, not as a hidden-GT fact. The retrievable `title`, `description`, `context`, and `experience` must describe non-privileged warning signs visible to a future rubric generator. These fields must not use words such as `GT`, `ground truth`, `reward`, `score`, `accuracy`, `high-scoring`, or `low-scoring`; describe observable behavior clusters instead, such as "samples that located the local repository" or "samples that edited only a scratch test." Put GT-based justification and exact accuracy evidence only in `metadata.analysis` for diagnosis. You should also delete or update previous experiences if they use GT information in `title`, `description`, `context`, or `experience`.
- Avoid lessons that merely say to follow GT, exact files, exact patch shape, or a hidden reference.
- `metadata.reference_golden_rubrics` should contain the rubric(s) that would have matched the score distribution: concrete, grounded, and judgeable, not a generic instruction to follow the reference patch.


## Guidelines
- Add or update only high-impact experiences likely to improve future rubric generation.
- Use delete only for experiences that are redundant, misleading, or unsafe to retrieve.
- If you need full context for existing experiences before updating or deleting them, output a retrieve action first using their titles.
- Add or update at least one grounded experience before returning `{}` unless the evidence is internally contradictory.
- Prefer quality over quantity: one reusable experience is better than several narrow instance notes.
- `title`, `description`, `context`, and `experience` will be retrieved in future non-privileged rubric generation and should not include any ground-truth information, including `gt_skeleton`, `gt_scores`, exact accuracy values, or phrases such as "ground truth evaluates", "high-scoring sample", or "low-scoring sample." Put GT-based justification only in `metadata.analysis`.
- Before returning an add/update action, rewrite the retrievable fields as behavior clusters, not score clusters. Bad: "High-scoring samples implement X while low-scoring samples do Y." Good: "One cluster implements X in the source files; another cluster leaves no source patch or only edits a scratch script." If you cannot state the lesson without score or GT language in retrievable fields, return `{}` instead of adding/updating an experience.
- Do not save lessons whose operational instruction is merely "follow the ground truth" or "prefer the GT patch."
- Do not preserve sibling-only ranking lessons unless they also explain how a child should compare to the parent.
- Existing experience titles are unique handles. For add, the new `experience.title` must not match any current bank title. For update, `target_title` must select the existing experience; the replacement `experience.title` may keep that title or use a new title that does not match any other current bank title.
- Base `metadata.analysis`, `context`, and `experience` only on the current update input and any retrieved existing experiences. Do not cite external reports, source files, or prior analyses unless they are explicitly present in the input.
- In each response, return either one retrieve/add/update/delete action, or an empty dict representing the end of session. Do not return multiple actions.

## Output Format
Return only JSON in the following format:
```json
{
    "action": "retrieve",
    "titles": ["existing experience titles"],
}
```

For add:
```json
{
    "action": "add",
    "title": "a new unique experience title",
    "description": "...",
    "context": "...",
    "experience": "...",
    "metadata": {
        "analysis": "...",
        "reference_golden_rubrics": []
    }
}
```

For update:
```json
{
    "action": "update",
    "target_title": "an existing experience title",
    "title": "the existing experience title or a new unique title",
    "description": "...",
    "context": "...",
    "experience": "...",
    "metadata": {
        "analysis": "...",
        "reference_golden_rubrics": []
    }
}
```

For delete:
```json
{
    "action": "delete",
    "title": "an existing experience title",
}
```

## Reference Golden Rubrics Output Format
```json
[
    {
        "polarity": "<positive|negative>",
        "description": "<detailed excellence/failure description>",
        "title": "<abstract label>",
        "metadata": {
        <a dict of any other relevant structured context and evidence needed for judging, including but not limited to current working stage, focus, targeted test code or pseudo-test, code review, etc.>
        },
        "scale": {
            "1": "<positive rubric: weakest progress evidence / negative rubric: no evidence of the flaw or regression>",
            "2": "<positive rubric: weak progress evidence / negative rubric: minor evidence of the flaw or regression>",
            "3": "<the moderate anchor for equivalent parent-child behavior>",
            "4": "<positive rubric: strong progress evidence / negative rubric: clear evidence of the flaw or regression>",
            "5": "<positive rubric: strongest progress evidence / negative rubric: most severe evidence of the flaw or regression>"
        }
    },
    {
    ...
    }
]
```
"""

PERSISTENT_STATE_UPDATE_PROMPT = """
You are maintaining a compact, durable working memory for a long-running software-debugging trajectory.

## Goal
Update the persistent state after older trajectory segments are evicted. Preserve the most important actionable context needed for later judging, and continuation of the work.

## Required Sections
Return exactly these 8 top-level string fields:
- **current_state**: What is actively being worked on right now, pending tasks, and immediate next steps. Always refresh this section so it reflects the latest work.
- **task_specification**: What the user asked for, important constraints, acceptance criteria, design decisions, and explanatory context.
- **files_and_functions**: Important files, functions, classes, modules, and why they matter. Include concrete file paths and identifiers.
- **errors_and_corrections**: Errors encountered, failed attempts, rejected hypotheses, and how they were corrected. Record approaches that should not be retried.
- **codebase_and_system_documentation**: Important components, interfaces, workflows, or architectural relationships and how they fit together.
- **learnings**: Actionable lessons about what worked well, what did not, and what to avoid. Do not duplicate material already captured in other sections.
- **key_results**: Exact or near-exact outputs that should be preserved, such as a patch idea, a concrete answer, a command result, or another critical artifact.
- **worklog**: Very terse step-by-step record of what was attempted or completed.

## Writing Guidelines
- Keep only information supported by the previous state, the evicted trajectory, or the workspace metadata.
- Be detailed and information-dense. Include concrete file paths, function names, commands, test names, error messages, patch details, and technical observations when useful.
- Focus on actionable, specific context that would help someone understand, judge, or recreate the work.
- It is OK to leave a section unchanged or blank if there are no substantial new insights. Do not add filler such as "No info yet".
- Keep each section under 400 words. If a section gets too long, remove lower-value details while preserving the most decision-relevant information.
- Preserve older facts that still matter.
- Merge redundant details instead of repeating them.
- If an earlier belief was revised, record that correction explicitly in the appropriate section.
- Do not hallucinate. Prefer omission to speculation.

## Output Format
```json
{
  "current_state": "",
  "task_specification": "",
  "files_and_functions": "",
  "errors_and_corrections": "",
  "codebase_and_system_documentation": "",
  "learnings": "",
  "key_results": "",
  "worklog": ""
}
```

## Inputs
1. **Question**: Original system and user prompt containing the coding task
2. **Previous Persistent State**: Previous memory state
3. **Evicted Older Trajectory**: Older trajectory segments that must now be compressed
4. **Workspace Metadata**: Compact git-based metadata at the current step

Return only the updated JSON object.
"""

EMPTY_PERSISTENT_STATE = {
    "current_state": "",
    "task_specification": "",
    "files_and_functions": "",
    "errors_and_corrections": "",
    "codebase_and_system_documentation": "",
    "learnings": "",
    "key_results": "",
    "worklog": "",
}

EMPTY_WORKSPACE_META = {
    "cwd": "/testbed",
    "git_repo": False,
    "head_commit": "",
    "changed_files": [],
    "untracked_files": [],
    "status": [],
    "diff_stat": "",
    "current_patch_chars": 0,
    "workspace_fingerprint": None,
}

RUBRIC_SCALE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {str(score): {"type": "string"} for score in range(1, 6)},
    "required": [str(score) for score in range(1, 6)],
}

RUBRIC_ITEM_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "polarity": {"type": "string", "enum": ["positive", "negative"]},
        "title": {"type": "string"},
        "description": {"type": "string"},
        "metadata": {"type": "object", "additionalProperties": {"type": "string"}},
        "scale": RUBRIC_SCALE_JSON_SCHEMA,
    },
    "required": ["polarity", "title", "description", "metadata", "scale"],
}

PERSISTENT_STATE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "persistent_state_update",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {key: {"type": "string"} for key in EMPTY_PERSISTENT_STATE},
            "required": list(EMPTY_PERSISTENT_STATE),
        },
    },
}

RUBRIC_GENERATION_RESPONSE_SCHEMA = copy.deepcopy(RUBRIC_ITEM_JSON_SCHEMA)
RUBRIC_GENERATION_RESPONSE_SCHEMA.pop("required", None)

REQUIRED_RUBRIC_GENERATION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "adaptive_rubric_generation_required",
        "strict": True,
        "schema": RUBRIC_ITEM_JSON_SCHEMA,
    },
}

RUBRIC_GENERATION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "adaptive_rubric_generation",
        "strict": True,
        "schema": RUBRIC_GENERATION_RESPONSE_SCHEMA,
    },
}

JUDGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "rubric_judge",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "score": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["score"],
        },
    },
}

RUBRIC_EXPERIENCE_RETRIEVAL_PROMPT = """
You are retrieving prior rubric-generation experiences to help generate adaptive rubrics for SWE-agent trajectory continuations.

## Task
Select the experience titles whose lessons should be appended to the rubric generation prompt before generating the next rubric.
The downstream rubric generator will identify the single most discriminative, non-redundant criterion separating the current continuation samples. Retrieve experiences only when they can concretely help that decision.

## Retrieval Targets
Retrieve an experience including but not limited to the following types of lessons:

1. **Related instance or related problem**
   - Same repository, library family, framework, task type, API surface, compatibility issue, build/configuration issue, workspace layout issue, or localization pattern.
   - Use this when the prior experience contains a task-specific boundary that may transfer to the current rubric decision.

2. **Related evaluation difficulty**
   - The current continuations are hard to rank for a reason seen before: visible process quality conflicts with semantic correctness, tests may encode obsolete behavior, a workaround may pass the real oracle, or a broad refactor may hide the actual compatibility boundary.
   - Use this when the prior experience helps decide what evidence the rubric should privilege under uncertainty.

3. **Historical counterexample to a likely rubric-model mistake**
   - Retrieve prior cases where the rubric model made a high-frequency error that is likely to recur now, such as rewarding generic test running over semantic coverage, punishing a valid compatibility-preserving revert, treating any reproduction project as a solution, over-penalizing messy but oracle-correct code, or generating a process/editing rubric when the samples differ by functional behavior.
   - Use this to warn the rubric generator away from a tempting but wrong criterion.

## Inputs
1. **Experience Index**: Existing experience titles with descriptions. Titles are the only retrieval handles.
2. **Current Round Context**: The current problem, previous persistent state, latest shared trajectory segment, and candidate continuations.

## Selection Rules
- Do not retrieve an experience solely because it shares broad words like "tests", "verification", "refactor", "search", or "compatibility"; the current continuation behavior must match the prior lesson.
- Do not retrieve an experience solely because it shares repository names, languages, or broad task categories if the evaluation difficulty is different.
- Retrieve an experience only if its stated applicability condition matches the current continuation distribution.
- Do not retrieve an experience that would relax or ignore a distinction that is task-defining in the current samples. A prior lesson about harmless implementation variation applies only when the visible differences are actually semantically equivalent for this task.
- Do not retrieve an early-stage process lesson when the current samples already contain enough terminal code or patch evidence to judge the implementation directly, unless the same process failure is still visibly causing the bad implementation. Vice versa.
- Return an empty list when the index contains no experience with a concrete target match.
- Do not output or invent internal IDs. Operate only on titles.

## Output Format
```json
{
  "titles": ["..."]
}
```
"""

RUBRIC_EXPERIENCE_UPDATE_PROMPT = """
You are maintaining a compact rubric-generation experience bank for SWE-agent.

## Task
Update the experience bank using previous rubric generation attempts from the completed instance.
Store only durable lessons that improve future rubric generation. Do not create a per-instance log.

## Available Actions
- **retrieve**: request full existing experiences before deciding whether to update or delete them if you are uncertain.
- **add**: add one new reusable experience.
- **update**: replace one existing experience with a clearer or more general version.
- **delete**: remove one redundant, misleading, or low-value experience.

## Input Explanation
- `generation_context`: the context that was shown to the rubric generator, including problem statement, the current history, previous generated rubrics and sampled agent continuations.
- `retrieved_experience`: historical experiences retrieved before this rubric generation attempt. Use them to diagnose whether an experience helped, was too broad, or should be refined.
- `generated_rubrics`: the full rubric list generated in that attempt.
- `gt_skeleton`: ground-truth patch skeleton for diagnosis only.
- `terminal_patch`: the final patch for the parent and each continuation if they continue to run until submission in the same sample order as `generation_context`. Use it only for diagnosis, not as a reusable lesson; do not save an experience that relies on terminal patch visibility.
- `passed_tests`: tests that `terminal_patch` pass and differ within the parent-plus-continuations group. Tests passed by every group member are omitted. Use it only for diagnosis, not as a reusable lesson; do not save an experience that relies on terminal patch visibility.
- `generated_rubric_accuracy`: per-rubric alignment diagnostics in the form `{"rubric title": {"overall_accuracy": float, "judging_diff_per_sample": [float]}}`. `overall_accuracy` is pairwise accuracy between that rubric's judge scores and GT scores. `judging_diff_per_sample` is the signed per-sample error list, computed as judge score minus GT score, the closer to zero the better, in the same sample order as `generation_context`.
- `average_rubric_judged_scores`: model judged scores average across all rubrics of each sample in `generation_context` in the same order. Use it to see the current rubric model's judging preference and compare to understand the misalignment with ground-truths.
- `gt_scores`: ground-truth scores of each sample in `generation_context` in the same order.


## Output Explanation
Output is a retrieve/add/update/delete action on the experience bank with the following fields:
- **title**: short reusable retrieval label. Name the evaluation lesson, not just the repository. It must agree with the description, context, and score pattern.
- **description**: concise summary of when this experience should be retrieved.
- **context**: (when to apply) current judging state summary, including history state, current agent goal and focus and the behavior differences and distribution across samples.
- **experience**: (how to avoid) actionable rubric-generation lesson. State what the previous rubrics generated, why they are correct or wrong, what the better rubric should focus on, and what tempting wrong criterion should be avoided.
- **metadata.analysis**: (why it happens) concise evidence analysis explaining the reason for the lesson, grounded by evidence from generated rubrics, GT skeleton, rubric accuracy, aggregate judged scores, and GT labels.
- **metadata.reference_golden_rubrics**: (what to do) a list of best rubric(s) that should have been generated.

## Experience Update Strategy
- Update the bank only from attempts where `gt_scores` vary visibly across samples; otherwise there is no reliable reward signal and an empty `{}` is usually best.
- First compare each generated rubric's `overall_accuracy` and `judging_diff_per_sample`, plus the aggregate `average_rubric_judged_scores`, with the visible continuation distribution. High-accuracy rubrics can become positive reusable patterns; low-accuracy rubrics should usually become corrective lessons about which tempting criterion to avoid.
- When the scores vary and a generated rubric or rubric list is clearly misaligned, prefer adding or updating one corrective experience over returning `{}`. Return `{}` only when the failure cannot be explained as a reusable visible rubric-generation lesson without leaking hidden ground truth.
- Use `terminal_patch` and `passed_tests` to identify the actual patch/artifact/test differences behind the score pattern. `passed_tests` has already removed tests passed by every group member, so remaining tests are discriminative evidence.
- Do not summarize a low-accuracy rubric as a good experience just because it sounds plausible. A rubric is useful only if its score ordering matches the GT ordering for the current samples.
- When a low-accuracy rubric fails, identify the observable reason: stale active rubric, majority-answer bias, over-rewarding process when patch semantics matter, treating obsolete tests as authoritative, or rewarding no-signal distinctions.
- If high-GT and low-GT samples differ mainly in terminal diffs, generate an experience that pushes future rubrics toward semantic code review, API/compatibility boundaries, owner logic, and executable/behavioral tests. Do not save another generic process lesson such as "runs more tests" or "edits carefully."
- If retrieved or active rubrics were stale, the experience should say when to stop reusing that rubric style and what new evidence should replace it. 
- If a rubric merely rewards the majority behavior, but the minority samples have better GT scores, save a corrective lesson about the observable minority signal that should have been evaluated.
- Before saving a lesson, check whether the apparent failure is caused by judge instability rather than a durable rubric-generation mistake. This often happens when the rubric is too narrow, over-specific, or tied to one surface form, so equivalent implementation variants receive inconsistent scores. In that case, the experience should teach a broader observable criterion or recommend not reusing that narrow rubric style.
- Check whether retrieved experiences influenced the generated rubrics. If a retrieved experience came from a materially different context, update that experience with clearer applicability boundaries or add a lesson about context matching; do not turn the mismatch into a new instance-specific rule.
- Store the future-reusable experience as an observable judging lesson, not as a hidden-GT fact. The retrievable `title`, `description`, `context`, and `experience` must describe non-privileged warning signs visible to a future rubric generator. These fields must not use words such as `GT`, `ground truth`, `reward`, `score`, `accuracy`, `high-scoring`, or `low-scoring`; describe observable behavior clusters instead, such as "samples that located the local repository" or "samples that edited only a scratch test." Put GT-based justification and exact accuracy evidence only in `metadata.analysis` for diagnosis. You should also delete or update previous experiences if they use GT information in `title`, `description`, `context`, or `experience`.
- Avoid lessons that merely say to follow GT, exact files, exact patch shape, or a hidden reference.
- `metadata.reference_golden_rubrics` should contain the rubric(s) that would have matched the score distribution: concrete, grounded, and judgeable, not a generic instruction to follow the reference patch.

## Guidelines
- Add or update only high-impact experiences likely to improve future rubric generation.
- Use delete only for experiences that are redundant, misleading, or unsafe to retrieve.
- If you need full context for existing experiences before updating or deleting them, output a retrieve action first using their titles.
- Add or update at least one grounded experience before returning `{}` unless the evidence is internally contradictory.
- Prefer quality over quantity: one reusable experience is better than several narrow instance notes.
- `title`, `description`, `context`, and `experience` will be retrieved in future non-privileged rubric generation and should not include any ground-truth information, including `gt_skeleton`, `gt_scores`, exact accuracy values, or phrases such as "ground truth evaluates", "high-scoring sample", or "low-scoring sample." Put GT-based justification only in `metadata.analysis`.
- Before returning an add/update action, rewrite the retrievable fields as behavior clusters, not score clusters. Bad: "High-scoring samples implement X while low-scoring samples do Y." Good: "One cluster implements X in the source files; another cluster leaves no source patch or only edits a scratch script." If you cannot state the lesson without score or GT language in retrievable fields, return `{}` instead of adding/updating an experience.
- Do not save lessons whose operational instruction is merely "follow the ground truth" or "prefer the GT patch."
- Existing experience titles are unique handles. For add, the new `experience.title` must not match any current bank title. For update, `target_title` must select the existing experience; the replacement `experience.title` may keep that title or use a new title that does not match any other current bank title.
- Base `metadata.analysis`, `context`, and `experience` only on the current update input and any retrieved existing experiences. Do not cite external reports, source files, or prior analyses unless they are explicitly present in the input.
- In each response, return either one retrieve/add/update/delete action, or an empty dict representing the end of session. Do not return multiple actions.

## Output Format
Return only JSON in the following format:
```json
{
    "action": "retrieve",
    "titles": ["existing experience titles"],
}
```

For add:
```json
{
    "action": "add",
    "title": "a new unique experience title",
    "description": "...",
    "context": "...",
    "experience": "...",
    "metadata": {
        "analysis": "...",
        "reference_golden_rubrics": []
    }
}
```

For update:
```json
{
    "action": "update",
    "target_title": "an existing experience title",
    "title": "the existing experience title or a new unique title",
    "description": "...",
    "context": "...",
    "experience": "...",
    "metadata": {
        "analysis": "...",
        "reference_golden_rubrics": []
    }
}
```

For delete:
```json
{
    "action": "delete",
    "title": "an existing experience title",
}
```

## Reference Golden Rubrics Output Format
```json
[
    {
        "polarity": "<positive|negative>",
        "description": "<detailed excellence/failure description>",
        "title": "<abstract label>",
        "metadata": {
        <a dict of any other relevant structured context and evidence needed for judging, including but not limited to current working stage, focus, targeted test code or pseudo-test, code review, etc.>
        },
        "scale": {
            "1": "<positive rubric: weakest evidence / negative rubric: flaw absent or minimal>",
            "2": "<positive rubric: weak evidence / negative rubric: minor evidence of the flaw>",
            "3": "<the moderate anchor>",
            "4": "<positive rubric: strong evidence / negative rubric: clear evidence of the flaw>",
            "5": "<positive rubric: strongest evidence / negative rubric: most severe evidence of the flaw>"
        }
    },
    {
    ...
    }
]
```
"""

RUBRIC_EXPERIENCE_RETRIEVAL_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "rubric_experience_retrieval",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "titles": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["titles"],
        },
    },
}

RUBRIC_EXPERIENCE_UPDATE_RESPONSE_FORMAT = {"type": "json_object"}


def _seed_experiences(scope: str = "siblings") -> list[dict[str, Any]]:
    if scope == "pc":
        parent_relative_semantic_progress = {
            "polarity": "positive",
            "title": "Parent-Relative Semantic Progress",
            "description": (
                "Scores whether the continuation changes the real owner code to add task-relevant behavior that the parent lacked, rather than merely "
                "looking cleaner, testing more, or matching a preferred implementation shape."
            ),
            "metadata": {
                "source_groups": (
                    "wtforms__wtforms-614::20260505-235109::round_002; elastic__synthetics-316::20260505-234609::round_003; "
                    "keras-team__keras-19955::20260506-031156::round_002"
                ),
                "visible_boundary": (
                    "real owner-code changes such as WTForms HTML5 field/widget migration, Elastic runner hook-error propagation, or KerasVariable "
                    "magic-method behavior versus partial default tweaks, empty patches, or non-owner work"
                ),
                "positive_evidence": (
                    "terminal diff in the task owner path that changes API contract, state mutation, lifecycle event emission, schema handling, compatibility "
                    "boundary, or backend behavior missing from the parent"
                ),
                "non_evidence": (
                    "extra validation, longer reasoning, cleaner edits, synthetic reproduction files, or residual imperfections that do not make the child "
                    "worse than a wrong/no-op parent"
                ),
            },
            "scale": {
                "1": "The child loses useful parent behavior, targets the wrong workspace, or replaces the task with synthetic/no-op work.",
                "2": "The child edits real files but misses the owner behavior or introduces likely harmful semantics compared with the parent.",
                "3": "The child is behaviorally equivalent to the parent for the visible task objective, including no meaningful patch over a no-op parent.",
                "4": "The child adds a partial but real owner-code behavior missing from the parent, even if validation or edge-case handling is imperfect.",
                "5": "The child clearly implements the parent-missing behavior in the real owner code while preserving relevant parent constraints.",
            },
        }
        parent_equivalence_preservation = {
            "polarity": "positive",
            "title": "Sitemap Index Patch Equivalence Preservation",
            "description": (
                "Scores whether the continuation preserves the parent's sitemap-index parser behavior: flushing a buffered <loc> when a closing loc or "
                "sitemap boundary is reached, ignoring blank loc text, and keeping the XMLIndexHandler/DelegatorHandler edits in the real patch. "
                "Different validation logs, final stdout, or harmless structure should tie unless they change this parser behavior."
            ),
            "metadata": {
                "source_case": "crawler-commons__crawler-commons-227::20260506-000111::round_003",
                "stage": "near-correct parent parser patch",
                "judge_focus": (
                    "Compare the XMLIndexHandler.java loc-buffer handling, maybeAddSiteMap call sites, DelegatorHandler.isAllBlank helper, and the final "
                    "terminal patch artifact against the parent workspace."
                ),
                "evidence": (
                    "Useful child variants keep the sitemap parser edits in src/main/java/crawlercommons/sitemaps/sax/XMLIndexHandler.java and "
                    "DelegatorHandler.java; empty final diffs or losing those edits are regressions."
                ),
            },
            "scale": {
                "1": "The child loses the parent sitemap parser edits or submits an empty/wrong-target patch.",
                "2": "The child keeps part of the parser change but drops an important loc-buffer or blank-loc condition.",
                "3": "The child preserves the same sitemap parsing behavior as the parent despite different logs, command syntax, or patch shape.",
                "4": "The child preserves the parent behavior and fixes a visible remaining parser edge case.",
                "5": "The child clearly improves the sitemap-index parser while preserving the parent loc-buffer and blank-loc behavior.",
            },
        }
        invalid_artifact_or_parent_work_loss = {
            "polarity": "negative",
            "title": "WTForms HTML5 Patch Artifact Integrity",
            "description": (
                "Penalizes continuations that lose or fake the WTForms HTML5 migration patch when the parent already has useful source edits. The key "
                "artifact is the real repository diff that moves HTML5 fields/widgets into core WTForms modules and preserves the core source behavior; "
                "missing docs or noisy submission stdout alone should not be treated as complete parent-work loss."
            ),
            "metadata": {
                "source_case": "wtforms__wtforms-614::20260505-235109::round_004",
                "stage": "useful parent source patch with fragile final artifact",
                "judge_focus": (
                    "Inspect the terminal patch and workspace diff for src/wtforms/fields/core.py, src/wtforms/widgets/core.py, and related import moves. "
                    "Separate core source-loss from docs-only omissions or malformed final command output."
                ),
                "failure_mode": (
                    "Empty patch, fake natural-language patch, wrong target, or losing the core field/widget edits is a regression; a real partial patch "
                    "that keeps the core HTML5 migration should tie the parent."
                ),
            },
            "scale": {
                "1": "No artifact flaw: the child submits a real WTForms patch preserving the core HTML5 field/widget source edits.",
                "2": "Minor submission or peripheral-file issue, but the core WTForms source migration remains present.",
                "3": "Ambiguous artifact completeness: some real core edits remain, but an important source file may be missing.",
                "4": "Clear artifact regression: the child loses substantial parent WTForms source work or targets the wrong files.",
                "5": "Severe artifact failure: empty/no-op patch, fake summary patch, synthetic-only edit, or complete loss of the parent source changes.",
            },
        }
        implementation_of_core_error_propagation = {
            "polarity": "positive",
            "title": "Implementation of Core Error Propagation",
            "description": (
                "Scores whether the continuation successfully catches errors from the hooks and propagates them to the test reporter or result state, "
                "resolving the issue missing in the parent."
            ),
            "metadata": {
                "judge_focus": (
                    "Check if try/catch blocks are added for beforeAll/afterAll hooks, and if the error is recorded, emitted, or assigned to the journey "
                    "results. Accept various valid approaches like throwing the caught error after handling, or emitting it via synthetic or registered journeys."
                ),
                "evidence": "Terminal patch modifications in the runner file.",
            },
            "scale": {
                "1": "The continuation introduces broken logic or syntax errors that make the code unrunnable, regressing from the parent.",
                "2": "The continuation attempts to catch errors but fails to propagate them to any results or reporters.",
                "3": "The continuation leaves the code functionally identical to the empty parent.",
                "4": "The continuation partially implements error propagation but misses one of the hooks or has minor logical gaps.",
                "5": (
                    "The continuation successfully catches errors from both hooks and propagates them to the result state or reporters, clearly improving over "
                    "the empty parent regardless of specific stylistic choices."
                ),
            },
        }
        return [
            {
                "title": "Reward Core Semantic Progress Over Implementation Style",
                "description": (
                    "Retrieve when the parent lacks the core fix and continuations solve it using diverse structural approaches, while empty continuations show "
                    "no progress."
                ),
                "context": (
                    "The parent trajectory is a no-op or lacks the required feature. Children either implement the core task requirement using different valid "
                    "control-flow or object-shape choices, or fail to produce a terminal patch. The judging risk is penalizing functionally successful children "
                    "for arbitrary style choices relative to the no-op parent."
                ),
                "experience": (
                    "Focus the rubric on whether the continuation achieves the primary semantic objective compared to the empty parent. Do not mandate specific "
                    "control flow, exact object structures, clean editing strategy, or polished validation unless the task explicitly requires them. Reward any "
                    "robust implementation that resolves the core issue as progress."
                ),
                "metadata": {
                    "generated_rubrics": [
                        {
                            "polarity": "positive",
                            "title": "Graceful Hook Error Event Emission",
                            "description": (
                                "Scores whether the continuation implements the core semantic fix (missing from the empty parent) by catching hook errors "
                                "and emitting 'journey:end' events using valid registered journeys, without throwing errors that disrupt graceful return."
                            ),
                            "metadata": {
                                "judge_focus": (
                                    "Check if try/catch blocks are added for both hooks. Verify that 'journey:end' is emitted using 'this.journeys' "
                                    "(not synthetic journeys) and that the runner gracefully returns 'result' instead of throwing the error."
                                ),
                                "evidence": "Terminal patch files (src/core/runner.ts) - look for catch blocks, 'this.emit', and 'throw'.",
                            },
                            "scale": {
                                "1": "The continuation introduces severe syntax or build errors, regressing below the empty parent.",
                                "2": "The continuation adds a patch that fails to emit 'journey:end' events or otherwise prevents reporter consumption.",
                                "3": "The continuation provides no meaningful terminal patch, remaining equivalent to the empty parent trajectory.",
                                "4": "The continuation successfully catches hook errors and emits 'journey:end' events, but disrupts graceful execution.",
                                "5": "The continuation catches hook errors and gracefully emits 'journey:end' events using valid registered journeys.",
                            },
                        },
                        {
                            "polarity": "negative",
                            "title": "Journey Execution Despite beforeAll Failure",
                            "description": (
                                "Penalizes continuations that catch the `beforeAll` hook error but fail to prevent the subsequent execution of journeys."
                            ),
                            "metadata": {
                                "judge_focus": (
                                    "Check if the code immediately following the `runBeforeAllHook` catch block allows the journey execution loop to run "
                                    "when a `beforeAllError` is present."
                                ),
                                "evidence": "Terminal patch file (`src/core/runner.ts`)",
                            },
                            "scale": {
                                "1": "No evidence of the flaw; the continuation prevents journey execution when a beforeAll hook fails.",
                                "2": "Minor evidence; the continuation skips journey execution but with convoluted or edge-case-prone logic.",
                                "3": "Moderate evidence; the continuation attempts to prevent execution but may allow partial execution.",
                                "4": "Clear evidence; the continuation catches the beforeAll error but allows normal journey execution.",
                                "5": "Most severe evidence; journey results overwrite or hide the beforeAll failure.",
                            },
                        }
                    ],
                    "gt_skeleton": (
                        "Source record: agent/search_outputs/analysis/results/pc_experience_bank_active_context_runs/"
                        "pc_experience_bank_instance_updates.jsonl, instance elastic__synthetics-316, action "
                        "`Reward Core Semantic Progress Over Implementation Style`.\n"
                        "diff --git a/src/core/runner.ts b/src/core/runner.ts\n"
                        "src/core/runner.ts: @@ -131,6 +131,7 @@ export default class Runner extends EventEmitter {\n"
                        "+ hookError: Error | undefined;\n"
                        "src/core/runner.ts: @@ -313,6 +313,33 @@ export default class Runner extends EventEmitter {\n"
                        "+ async runFakeJourney(journey: Journey, options: RunOptions) {\n"
                        "+ this.emit('journey:start', { journey, timestamp: getTimestamp(), params: options.params });\n"
                        "+ this.emit('journey:end', { journey, start, options, end: monotonicTimeInSeconds(), ...result });\n"
                        "+ if (options.reporter === 'json') { await once(this, 'journey:end:reported'); }\n"
                        "+ return result;\n"
                        "+ }\n"
                        "src/core/runner.ts: @@ -389,7 +417,9 @@ export default class Runner extends EventEmitter {\n"
                        "+ const journeyResult: JourneyResult = this.hookError\n"
                        "+ ? await this.runFakeJourney(journey, options)\n"
                        "+ : await this.runJourney(journey, options);"
                    ),
                    "generated_rubric_accuracy": {
                        "Graceful Hook Error Event Emission": {
                            "overall_accuracy": 0.571,
                            "judging_diff_per_sample": [-0.25, -0.25, 0.0, 0.0, -0.75, -0.75, 0.0, 0.0],
                        },
                        "Journey Execution Despite beforeAll Failure": {
                            "overall_accuracy": 0.607,
                            "judging_diff_per_sample": [-0.5, -0.5, -0.5, -0.875, -0.875, -0.875, -0.375, -0.375],
                        },
                    },
                    "gt_scores": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
                    "analysis": (
                        "Source: pc_experience_bank_instance_updates.jsonl, instance elastic__synthetics-316, action "
                        "`Reward Core Semantic Progress Over Implementation Style`. The parent had an empty terminal patch (GT=0.5). Children 1-6 all caught "
                        "hook errors and propagated them (GT=1.0), while children 7-8 produced no patch (GT=0.5). The generated rubric `Graceful Hook Error "
                        "Event Emission` penalized valid implementation variants such as throwing after emitting, synthetic journey use, or result-map mutation."
                    ),
                    "reference_golden_rubrics": [copy.deepcopy(implementation_of_core_error_propagation)],
                },
            },
            {
                "title": "Preserve Ties Around Near-Correct Parents",
                "description": (
                    "Retrieve when the parent is already useful or near-correct and most continuations are behaviorally equivalent variants, with at most one "
                    "small visible improvement or one real artifact regression."
                ),
                "context": (
                    "Crawler r3 is the canonical source case: the parent and most children preserve the sitemap parsing fix; a malformed final submission command "
                    "should tie when the workspace diff remains intact, while an actually empty terminal patch should regress. The judging risk is manufacturing "
                    "progress or regression from submission stdout, validation logs, or non-behavioral refactoring."
                ),
                "experience": (
                    "Generate a tie-preserving parent-child rubric. Score 3 should be the default when parent and child keep the same observable behavior, even "
                    "if the child has different logs, validation scripts, submission stdout, code layout, or closer structural alignment with a reference. Score "
                    "above 3 only for a visible remaining defect fixed by the child; score below 3 only for actual loss of parent behavior, invalid artifact, "
                    "forbidden edit, or empty terminal patch."
                ),
                "metadata": {
                    "generated_rubrics": [
                        {
                            "polarity": "negative",
                            "title": "Invalid Patch Artifact Submission",
                            "description": (
                                "Penalizes continuations that fail to properly submit the final git patch, either by constructing a malformed submission "
                                "command that emits empty output, getting distracted by test compilation errors, or terminating without submitting the "
                                "parent's visible codebase progress."
                            ),
                            "metadata": {
                                "regression_signal": "Submission rejected due to empty output, malformed shell pipes, or failure to reach the submission step.",
                                "evidence": "The final COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT command, its system observation, and terminal patch content.",
                            },
                            "scale": {
                                "1": "No evidence of submission failure; the continuation emits a valid patch preserving the intended fixes.",
                                "2": "The continuation submits the patch but includes minor extraneous files alongside source modifications.",
                                "3": "The continuation maintains the parent baseline, submitting an equivalent valid patch without artifact corruption.",
                                "4": "The continuation gets caught in test errors or debugging loops and fails to submit before terminating.",
                                "5": "The continuation attempts to submit but a malformed command yields a rejected or empty patch where the parent had useful work.",
                            },
                        },
                        {
                            "polarity": "negative",
                            "title": "Submission Failure and Patch Loss",
                            "description": (
                                "Penalizes continuations that attempt to submit but fail to emit a valid patch artifact, such as executing malformed "
                                "submission commands that yield empty output, or abandoning the parent's valid codebase progress due to broken test loops."
                            ),
                            "metadata": {
                                "regression_signal": "Rejected submission command, 0-byte terminal patch, or loops that prevent successful finalization.",
                                "evidence": "Final submission command syntax, system rejection messages, terminal patch size, and unresolved test compilation errors.",
                            },
                            "scale": {
                                "1": "No evidence of submission failure; the continuation cleanly submits the parent baseline or refined source modifications.",
                                "2": "Minor workflow issues, but the agent preserves source modifications and avoids an invalid artifact.",
                                "3": "The continuation remains equivalent to the parent, neither successfully submitting nor explicitly destroying the patch.",
                                "4": "The continuation gets trapped in broken tests or compilation errors and fails to submit productive codebase work.",
                                "5": "The continuation explicitly submits with a malformed command or empty diff, entirely losing the parent's useful work.",
                            },
                        }
                    ],
                    "gt_skeleton": (
                        "Source record: agent/search_outputs/analysis/results/pc_experience_bank_active_context_runs/"
                        "pc_experience_bank_instance_updates.jsonl, instance crawler-commons__crawler-commons-227, action "
                        "`Preserve Ties Around Near-Correct Parents`.\n"
                        "diff --git a/CHANGES.txt b/CHANGES.txt\n"
                        "CHANGES.txt: @@ -1,6 +1,7 @@\n"
                        "+ - [Sitemaps] Sitemap index: stop URL at closing </loc> (sebastian-nagel, kkrugler) #213\n"
                        "diff --git a/src/main/java/crawlercommons/sitemaps/sax/DelegatorHandler.java b/src/main/java/crawlercommons/sitemaps/sax/DelegatorHandler.java\n"
                        "src/main/java/crawlercommons/sitemaps/sax/DelegatorHandler.java: @@ -208,4 +208,14 @@ public class DelegatorHandler extends DefaultHandler {\n"
                        "+ public static boolean isAllBlank(CharSequence charSeq) {\n"
                        "+ for (int i = 0; i < charSeq.length(); i++) {\n"
                        "+ if (!Character.isWhitespace(charSeq.charAt(i))) { return false; }\n"
                        "+ }\n"
                        "+ return true;\n"
                        "+ }\n"
                        "diff --git a/src/main/java/crawlercommons/sitemaps/sax/XMLIndexHandler.java b/src/main/java/crawlercommons/sitemaps/sax/XMLIndexHandler.java\n"
                        "src/main/java/crawlercommons/sitemaps/sax/XMLIndexHandler.java: @@ -67,6 +67,19 @@ class XMLIndexHandler extends DelegatorHandler {\n"
                        "+ // flush any unclosed or missing <sitemap> element\n"
                        "+ if (loc.length() > 0 && (\"loc\".equals(localName) || \"sitemap\".equals(localName))) {\n"
                        "+ if (!isAllBlank(loc)) { maybeAddSiteMap(); return; }\n"
                        "+ loc = new StringBuilder();"
                    ),
                    "generated_rubric_accuracy": {
                        "Invalid Patch Artifact Submission": {
                            "overall_accuracy": 0.607,
                            "judging_diff_per_sample": [0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0],
                        },
                        "Submission Failure and Patch Loss": {
                            "overall_accuracy": 0.589,
                            "judging_diff_per_sample": [0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.375, 0.0],
                        },
                    },
                    "gt_scores": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5],
                    "analysis": (
                        "Source: pc_experience_bank_instance_updates.jsonl, instance crawler-commons__crawler-commons-227, action "
                        "`Preserve Ties Around Near-Correct Parents`. The parent had an unconditional attribute reset, while some children refined it to "
                        "perfectly match a reference file's conditional reset. However, both passed all tests and had identical observable behavior, so GT "
                        "labeled them equivalent (0.5). A rubric mistakenly rewarded the perfect structural alignment, causing false progress signals."
                    ),
                    "reference_golden_rubrics": [copy.deepcopy(parent_equivalence_preservation)],
                },
            },
            {
                "title": "No-Op Artifacts Are Regressions But Partial Patches Tie",
                "description": (
                    "Retrieve when a parent has useful real-repository progress and continuations differ between complete artifact loss and valid partial patches "
                    "that still preserve the core parent work."
                ),
                "context": (
                    "WTForms r4 and Crawler r3 separate two artifact cases that should not be merged. A child with git/submission failure and an empty terminal "
                    "patch is a real regression from a useful parent. A child with a messy final command, rejected stdout, or omitted peripheral files can still "
                    "tie the parent if the workspace or submitted patch contains the core source edits. Elastic r3 also has no-patch children that should tie a "
                    "no-op parent, not beat real source-patch children."
                ),
                "experience": (
                    "Generate an artifact/work-continuity rubric that checks the terminal workspace and submitted patch separately from final-command stdout. "
                    "Penalize empty patches, fake summaries, wrong-target/synthetic patches, or complete loss of parent edits. Do not penalize a valid partial "
                    "patch merely because it omits peripheral files or has a noisy submission command; if the core parent behavior remains, it should tie."
                ),
                "metadata": {
                    "generated_rubrics": [
                        {
                            "polarity": "negative",
                            "title": "Invalid Artifact Or Parent Work Loss",
                            "description": (
                                "Penalizes continuations that fail to successfully submit the complete set of modifications prepared in the parent "
                                "trajectory. This includes failing to output the patch content during the final submission command, or omitting necessary "
                                "modified files in the generated diff."
                            ),
                            "metadata": {
                                "failure_mode": (
                                    "Failing to print the patch content during submission or excluding key files from the final diff, leading to a loss "
                                    "of useful parent work."
                                ),
                                "judge_focus": (
                                    "Check the terminal patch artifact for completeness. Look for missing modified files compared to the parent baseline, "
                                    "or an empty patch caused by an incorrect submission command."
                                ),
                            },
                            "scale": {
                                "1": "No evidence of the flaw. The continuation successfully generates and submits a complete patch containing all intended modifications.",
                                "2": "Minor evidence of the flaw. The continuation encounters slight issues generating the patch but recovers to submit the full intended changes.",
                                "3": "Equivalent to the parent baseline: successfully submits the full, complete patch artifact without losing parent work.",
                                "4": "Clear evidence of the flaw. The continuation successfully submits a patch but omits important files that were correctly modified in the parent trajectory.",
                                "5": "Most severe evidence. The continuation completely fails to submit the patch or submits an entirely empty or invalid artifact.",
                            },
                        }
                    ],
                    "gt_skeleton": (
                        "Source record: agent/search_outputs/analysis/results/pc_experience_bank_active_context_runs/"
                        "pc_experience_bank_instance_updates.jsonl, instance wtforms__wtforms-614, action "
                        "`No-Op Artifacts Are Regressions But Partial Patches Tie`.\n"
                        "diff --git a/CHANGES.rst b/CHANGES.rst\n"
                        "CHANGES.rst: @@ -34,7 +34,7 @@ Unreleased\n"
                        "+ - Flags can take non-boolean values. :issue:`406` :pr:`467`\n"
                        "diff --git a/docs/fields.rst b/docs/fields.rst\n"
                        "docs/fields.rst: @@ -182,10 +182,10 @@ The Field base class\n"
                        "- An object containing boolean flags set either by the field itself, or\n"
                        "+ An object containing flags set either by the field itself, or\n"
                        "- An unset flag will result in :const:`False`.\n"
                        "+ An unset flag will result in :const:`None`.\n"
                        "diff --git a/src/wtforms/widgets/core.py b/src/wtforms/widgets/core.py\n"
                        "- class SearchInput(Input):\n"
                        "+ class DateTimeLocalInput(Input):\n"
                        "+ input_type = \"datetime-local\"\n"
                        "+ class NumberInput(Input):\n"
                        "+ input_type = \"number\"\n"
                        "+ def __init__(self, step=None, min=None, max=None):"
                    ),
                    "generated_rubric_accuracy": {
                        "Invalid Artifact Or Parent Work Loss": {
                            "overall_accuracy": 0.625,
                            "judging_diff_per_sample": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.375],
                        }
                    },
                    "gt_scores": [0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5],
                    "analysis": (
                        "Source: pc_experience_bank_instance_updates.jsonl, instance wtforms__wtforms-614, action "
                        "`No-Op Artifacts Are Regressions But Partial Patches Tie`. Sample 6 failed to generate a patch due to git errors and received a GT "
                        "score of 0.0. Sample 8 omitted some peripheral files from its `git diff` command but successfully submitted the core edits, receiving "
                        "a GT score of 0.5. The previous rubric mistakenly penalized Sample 8 for omitting important files."
                    ),
                    "reference_golden_rubrics": [copy.deepcopy(invalid_artifact_or_parent_work_loss)],
                },
            },
        ]
    selective_compatibility_boundary = {
            "polarity": "positive",
            "title": "Selective Compatibility Boundary",
            "description": (
                "Scores whether the continuation identifies and preserves the task-specific boundary between behavior that should change "
                "and behavior that must remain backward-compatible, instead of treating every failing test or every desired feature as globally authoritative."
            ),
            "metadata": {
                "stage": "refinement",
                "focus": "terminal diff and tests around field defaults, validator flags, and compatibility cases",
                "test_code": (
                    "Python pseudo-test: define fields using Length and NumberRange validators; render them; assert expected HTML5 attributes where required; "
                    "also assert a compatibility-sensitive field keeps its legacy input type or unset flag behavior."
                ),
                "code_review": (
                    "Owner: WTForms field/widget and flag generation. Invariant: new HTML5 behavior must not erase public compatibility cases. "
                    "Failure mode: blanket conversion or blanket reversion hides the actual API boundary."
                ),
            },
            "scale": {
                "1": "Blindly changes or reverts behavior without identifying any compatibility boundary.",
                "2": "Mentions compatibility but applies it to the wrong API surface or inconsistently.",
                "3": "Handles the main behavior but misses one important compatibility exception.",
                "4": "Mostly preserves the correct boundary with minor omissions.",
                "5": "Clearly implements the intended behavior while preserving the compatibility-sensitive cases supported by evidence.",
            },
    }
    fabricated_workspace_patch = {
            "polarity": "negative",
            "title": "Fabricated Workspace Patch",
            "description": (
                "Penalizes continuations that create or modify a synthetic project or dummy files and present that work as the solution, rather than grounding "
                "the patch in the actual target repository or explicitly treating the synthetic project only as a disposable reproduction."
            ),
            "metadata": {
                "stage": "localization",
                "focus": "created files, terminal patch paths, and claims that synthetic files solve the task",
                "test_code": (
                    "Shell pseudo-check: inspect git diff --name-status and repository root; fail if the submitted patch consists only of newly created toy "
                    "project files that are not part of the target checkout."
                ),
                "code_review": (
                    "Owner: repository targeting. Invariant: final patches must modify code hidden tests import or execute. "
                    "Failure mode: synthetic reproduction files are submitted as the real fix."
                ),
            },
            "scale": {
                "1": "Targets the real repository and keeps any synthetic reproduction clearly separate.",
                "2": "Creates disposable repro artifacts but does not patch them as the solution.",
                "3": "Mixes real and synthetic targets, leaving ownership ambiguous.",
                "4": "Primarily patches generated dummy files while showing some awareness of the real task.",
                "5": "Submits a solution built on fabricated files or a synthetic project as if it were the target repository.",
            },
    }
    post_modification_regression_testing = {
            "polarity": "positive",
            "title": "Post-Modification Regression Testing",
            "description": (
                "The agent validates its codebase modifications by actively running the project's official, built-in test suite (e.g., `pytest`, `unittest`) "
                "to catch regressions. Excellence is shown by relying on comprehensive existing tests to verify correctness and guide subsequent fixes, rather "
                "than depending solely on basic syntax checks or writing limited custom test scripts that miss broader system impacts."
            ),
            "metadata": {},
            "scale": {
                "1": "Applies modifications without executing any subsequent tests, syntax checks, or validations, assuming the edits are correct.",
                "2": "Performs only superficial checks after modifying code, such as basic syntax verification or checking that a module imports without crashing.",
                "3": "Writes and executes a narrow custom script to verify the immediate change, but fails to leverage the project's existing test suite to catch broader, unintended regressions.",
                "4": "Actively invokes the project's official test suite to validate changes, using the test output to identify broken functionality or regressions.",
                "5": "Methodically and iteratively runs the project's official test suite after making changes, successfully using test failures to pinpoint regressions and driving the subsequent fix process entirely through test feedback.",
            },
    }
    destructive_test_appeasement = {
            "polarity": "negative",
            "title": "Destructive Test Appeasement",
            "description": (
                "Evaluates whether the agent erroneously reverts valid, requested source code modifications to appease outdated tests. A poor continuation treats all "
                "test failures as source code regressions, failing to recognize when a test fails precisely because it strictly asserts the old behavior that the "
                "prompt explicitly asked to change. This results in the agent reverting or deleting its correct implementation just to make an obsolete test pass, "
                "thereby failing the task objective."
            ),
            "metadata": {},
            "scale": {
                "1": "The agent correctly recognizes that the test failure is due to the test asserting outdated behavior. It preserves the valid source code changes, choosing to either update the obsolete test or safely ignore the expected failure.",
                "2": "The agent briefly investigates the test failure as a potential regression before correctly concluding that the test is outdated, leaving the correct source code modifications intact.",
                "3": "The agent is confused by the test failure and wastes effort trying to debug the source code, but ultimately stops short of reverting the valid feature changes.",
                "4": "The agent partially degrades the requested feature, adding hacky workarounds or reverting minor parts of the correct implementation in an attempt to pacify the outdated test.",
                "5": "The agent explicitly and completely reverts its valid, required feature modifications in the source code solely to make the obsolete test pass, fundamentally failing the task objective.",
            },
    }
    targeting_fabricated_code = {
            "polarity": "negative",
            "title": "Targeting Fabricated Code",
            "description": (
                "The agent applies proposed fixes to a dummy project or synthetic files it created from scratch (often derived from the issue description's "
                "reproduction steps) instead of locating and modifying the actual target repository. This reflects a severe failure in environment navigation "
                "and task comprehension."
            ),
            "metadata": {},
            "scale": {
                "1": "The agent correctly locates the real codebase and exclusively targets it for any proposed fixes.",
                "2": "The agent creates synthetic files for reproduction or testing, but clearly distinguishes them from the real codebase and applies fixes correctly.",
                "3": "The agent creates a synthetic project and mistakenly attempts to apply some fixes to it, but shows some awareness that it is not the actual repository.",
                "4": "The agent focuses heavily on fixing a fabricated project and drafts patches for the synthetic code, indicating deep confusion about the target repository.",
                "5": "The agent completely fails to locate or ignores the real repository, builds a synthetic project from scratch, applies a fix to the fabricated files, and generates a patch against its own creation.",
            },
    }
    wtforms_gt_skeleton = """diff --git a/CHANGES.rst b/CHANGES.rst
CHANGES.rst: @@ -34,7 +34,7 @@ Unreleased
+ - Flags can take non-boolean values. :issue:`406` :pr:`467`
diff --git a/docs/fields.rst b/docs/fields.rst
docs/fields.rst: @@ -182,10 +182,10 @@ The Field base class
- An object containing boolean flags set either by the field itself, or
+ An object containing flags set either by the field itself, or
- An unset flag will result in :const:`False`.
+ An unset flag will result in :const:`None`.
docs/fields.rst: @@ -220,10 +220,14 @@ refer to a single input from the form.
- For better date/time fields, see the :mod:`dateutil extension <wtforms.ext.dateutil.fields>`
+ .. autoclass:: DateTimeLocalField(default field arguments, format='%Y-%m-%d %H:%M:%S')
+ .. autoclass:: DecimalRangeField(default field arguments)
+ .. autoclass:: EmailField(default field arguments)
docs/fields.rst: @@ -252,6 +256,8 @@ refer to a single input from the form.
+ .. autoclass:: IntegerRangeField(default field arguments)
docs/fields.rst: @@ -323,6 +329,8 @@ refer to a single input from the form.
+ .. autoclass:: SearchField(default field arguments)
docs/fields.rst: @@ -338,6 +346,13 @@ refer to a single input from the form.
+ .. autoclass:: TelField(default field arguments)
+ .. autoclass:: TimeField(default field arguments, format='%H:%M')
+ .. autoclass:: URLField(default field arguments)
docs/fields.rst: @@ -559,40 +574,3 @@ Additional Helper Classes
- HTML5 Fields
- In addition to basic HTML fields, WTForms also supplies fields for the HTML5
- standard. These fields can be accessed under the :mod:`wtforms.fields.html5` namespace.
- In reality, these fields are just convenience fields that extend basic fields
- and implement HTML5 specific widgets. These widgets are located in the :mod:`wtforms.widgets.html5`
- namespace and can be overridden or modified just like any other widget.
- .. module:: wtforms.fields.html5
- .. autoclass:: SearchField(default field arguments)
- .. autoclass:: TelField(default field arguments)
- .. autoclass:: URLField(default fie
...[truncated]...
- input_type = "number"
- def __init__(self, step=None, min=None, max=None):
- self.step = step
- self.min = min
- self.max = max
- def __call__(self, field, **kwargs):
- if self.step is not None:
- kwargs.setdefault("step", self.step)
- if self.min is not None:
- kwargs.setdefault("min", self.min)
- if self.max is not None:
- kwargs.setdefault("max", self.max)
- return super().__call__(field, **kwargs)
- class RangeInput(Input):
- \"\"\"
- Renders an input with type "range".
- \"\"\"
- input_type = "range"
- def __init__(self, step=None):
- self.step = step
- def __call__(self, field, **kwargs):
- if self.step is not None:
- kwargs.setdefault("step", self.step)
- return super().__call__(field, **kwargs)
- class ColorInput(Input):
- \"\"\"
- Renders an input with type "color".
- \"\"\"
- input_type = "color\""""
    gradle_gt_skeleton = """diff --git a/build.gradle.kts b/build.gradle.kts
build.gradle.kts: @@ -36,8 +36,8 @@ kotlin {
- @Suppress("DEPRECATION") // TODO: bump apiVersion to 2.0 to match Gradle 9.0
- apiVersion = KotlinVersion.KOTLIN_1_8
+ apiVersion = KotlinVersion.KOTLIN_2_0
+ languageVersion = apiVersion
diff --git a/docs/changes/README.md b/docs/changes/README.md
docs/changes/README.md: @@ -2,6 +2,11 @@
+ **Fixed**
+ - Pin the plugin's Kotlin language level on 2.0. ([#1448](https://github.com/GradleUp/shadow/pull/1448))
+ The language level used in `9.0.0-beta14` is 2.2, which may cause compatibility issues for the plugins depending on
+ Shadow.
diff --git a/src/main/kotlin/com/github/jengelman/gradle/plugins/shadow/transformers/PropertiesFileTransformer.kt b/src/main/kotlin/com/github/jengelman/gradle/plugins/shadow/transformers/PropertiesFileTransformer.kt
src/main/kotlin/com/github/jengelman/gradle/plugins/shadow/transformers/PropertiesFileTransformer.kt: @@ -238,7 +238,6 @@ public open class PropertiesFileTransformer @Inject constructor(
- @OptIn(ExperimentalStdlibApi::class)"""
    wtforms_r2_metadata = {
        "generated_rubrics": [copy.deepcopy(post_modification_regression_testing)],
        "gt_skeleton": wtforms_gt_skeleton,
        "generated_rubric_accuracy": {
            "Post-Modification Regression Testing": {
                "overall_accuracy": 0.589,
                "judging_diff_per_sample": [0.023, -0.727, -0.981, 0.019, 0.500, 0.000, 0.500, 0.000],
            }
        },
        "gt_scores": [0.977, 0.977, 0.981, 0.981, 0.000, 0.000, 0.000, 0.000],
        "analysis": (
            "Source: persisted large_scale_experiment_results.jsonl, gemini_disk vanilla_generated, "
            "wtforms__wtforms-614::20260505-235109::round_002, rubric_id f3fd402b88bf. "
            "The generated process rubric had overall_accuracy 0.589 and large negative signed errors on high-GT children, while GT rewards "
            "[0.977, 0.977, 0.981, 0.981, 0.000, 0.000, 0.000, 0.000] depended on semantic HTML5 namespace/widget/flag behavior. "
            "The better reflection is to generate a compatibility-boundary rubric."
        ),
        "reference_golden_rubrics": [copy.deepcopy(selective_compatibility_boundary)],
    }
    wtforms_r3_metadata = {
        "generated_rubrics": [copy.deepcopy(destructive_test_appeasement)],
        "gt_skeleton": wtforms_gt_skeleton,
        "generated_rubric_accuracy": {
            "Destructive Test Appeasement": {
                "overall_accuracy": 0.375,
                "judging_diff_per_sample": [-0.981, 0.023, 0.023, 0.023, 0.023, 0.023, 0.023, 0.023],
            }
        },
        "gt_scores": [0.981, 0.977, 0.977, 0.977, 0.977, 0.977, 0.977, 0.977],
        "analysis": (
            "Source: persisted large_scale_experiment_results.jsonl, gemini_disk vanilla_generated, "
            "wtforms__wtforms-614::20260505-235109::round_003, rubric_id 4a10209ceeba. "
            "Destructive Test Appeasement had overall_accuracy 0.375 and a large negative signed error on the first child, "
            "while that child had the best GT reward in [0.981, 0.977, 0.977, 0.977, 0.977, 0.977, 0.977, 0.977]. "
            "The mistake was failing to distinguish obsolete tests from intentional compatibility constraints."
        ),
        "reference_golden_rubrics": [copy.deepcopy(selective_compatibility_boundary)],
    }
    gradle_r1_metadata = {
        "generated_rubrics": [copy.deepcopy(targeting_fabricated_code)],
        "gt_skeleton": gradle_gt_skeleton,
        "generated_rubric_accuracy": {
            "Targeting Fabricated Code": {
                "overall_accuracy": 0.482,
                "judging_diff_per_sample": [-0.202, 1.000, 1.000, 1.000, 1.000, 0.000, 0.048, 0.048],
            }
        },
        "gt_scores": [0.952, 0.000, 0.000, 0.000, 0.000, 0.000, 0.952, 0.952],
        "analysis": (
            "Source: persisted large_scale_experiment_results.jsonl, gemini_disk vanilla_generated, "
            "gradleup__shadow-1448::20260506-011019::round_001, rubric_id abbf836c4765. "
            "The generated rubric had overall_accuracy 0.482 and wrongly aligned fabricated-workspace evidence with the sixth child, "
            "whose GT was 0.000, while the first, seventh, and eighth children had GT 0.952 because they patched the real repository. The golden rewrite keeps "
            "the same evidence but phrases it as a grounded repository-targeting criterion."
        ),
        "reference_golden_rubrics": [copy.deepcopy(fabricated_workspace_patch)],
    }
    return [
        {
            "title": "WTForms Process Rubric Mismatch",
            "description": "Do not use a process-only testing rubric when WTForms samples are separated by HTML5 namespace/widget/flag compatibility behavior.",
            "context": (
                "WTForms round 2 continuations share a history where HTML5 field/widget migration and validator-derived flags are already visible. Scores are "
                "ordered by the eight child samples. The first sample broadly moves HTML5 fields/widgets into core and manually fixes a missing range field; "
                "it gets generated testing score 1.000 and GT 0.977. The second sample performs a similar broad refactor with import/deprecation cleanup; score "
                "0.250, GT 0.977. The third sample does a large scripted migration of HTML5 widgets and fields; score 0.000, GT 0.981. The fourth sample keeps "
                "the broad migration but removes a warning that breaks local tests; score 1.000, GT 0.981. The fifth, sixth, seventh, and eighth samples only "
                "make partial widget/default changes such as changing FloatField, appending large chunks of core fields, fixing StringField after a bad replace, "
                "or adding Date/Time widget overrides; their GT scores are all 0.000 even when the testing rubric gives 0.500 to some of them."
            ),
            "experience": (
                "The prior generated choice was `Post-Modification Regression Testing`, and it was the wrong lesson for this context. The best replacement is "
                "`Selective Compatibility Boundary`: the group is separated by whether the continuation identifies which WTForms behavior should change and "
                "which public compatibility cases must remain, not by whether it runs broader tests. A wrong approach is to reward generic regression testing, "
                "reward blanket HTML5 conversion, or punish every rollback before checking whether it preserves an intentional API boundary."
            ),
            "metadata": copy.deepcopy(wtforms_r2_metadata),
        },
        {
            "title": "Fabricated Target Repository",
            "description": "Use an active negative rubric when samples in a confusing or empty workspace fabricate a project and submit created files as the solution.",
            "context": (
                "Gradle Shadow round 1 continuations start from a confusing workspace. Scores are ordered by the eight child samples. The first child creates a "
                "temporary reproduction but still patches the real repository and reaches GT 0.952; the fabricated-code rubric gives it 0.250 because the repro "
                "is visible. The second, third, and fourth children patch real build.gradle.kts variants that do not match the GT patch and receive GT 0.000. "
                "The fifth child mostly investigates real Kotlin source and does not land a useful patch, also GT 0.000. The sixth child creates files under a "
                "synthetic /testbed workspace and submits that fabricated patch; the rubric correctly scores it 1.000 and GT is 0.000. The seventh and eighth "
                "children patch the real build script in ways close to the golden version and receive GT 0.952 with fabricated-code scores 0.000."
            ),
            "experience": (
                "The prior generated choice `Targeting Fabricated Code` captured the useful failure mode: the bad sample actively patches created dummy files "
                "instead of code hidden tests will import. The best golden rewrite is `Fabricated Workspace Patch`, which keeps the criterion grounded in created "
                "files, terminal patch paths, and claims that the synthetic project is the solution. A wrong approach is to reward having any reproduction project, "
                "score generic localization thoroughness, or ignore whether the final patch lands in the real repository."
            ),
            "metadata": copy.deepcopy(gradle_r1_metadata),
        },
        {
            "title": "Avoid Obsolete-Test Overcorrection",
            "description": "Do not generate a destructive-test-appeasement rubric when the real missing distinction is WTForms compatibility boundary reasoning.",
            "context": (
                "WTForms round 3 continuations have already moved much of the HTML5 field/widget code toward the main namespace. Scores are ordered by the eight "
                "child samples. The first sample reverts `FloatField` after old tests fail while keeping the broader migration; the destructive-test rubric gives "
                "it 1.000 badness, but its GT is the best at 0.981. The second through sixth samples submit similar broad HTML5 refactors without that specific "
                "compatibility revert and all score 0.000 on the destructive-test rubric with GT 0.977. The seventh sample runs broad tests, observes the legacy "
                "FloatField expectation, and treats it as outdated rather than preserving it; score 0.000, GT 0.977. The eighth sample verifies HTML5 defaults "
                "with a custom script and submits the broad refactor; score 0.000, GT 0.977."
            ),
            "experience": (
                "The prior bad choice was the negative rubric `Destructive Test Appeasement`; it was plausible but wrong for this context. The better choice is to "
                "generate a rubric like `Selective Compatibility Boundary`, because the judge needs to distinguish obsolete tests from intentional compatibility "
                "constraints before penalizing a revert. If generating again, prefer a rubric that asks whether the continuation preserves the correct API boundary. "
                "The wrong approach is to assume every legacy-test-driven rollback is destructive, which can punish the best continuation when it keeps the migration "
                "while preserving a compatibility-sensitive exception."
            ),
            "metadata": copy.deepcopy(wtforms_r3_metadata),
        },
    ]


def _seed_rubrics(scope: str = "siblings") -> list[dict[str, Any]]:
    if scope == "pc":
        return [
            {
                    "polarity": "positive",
                    "title": "Parent-Relative Semantic Progress",
                    "description": (
                        "Scores whether the continuation adds task-relevant behavior that the parent trajectory lacked in the real target workspace. "
                        "Progress should be judged by observable semantic change over the parent, not by longer reasoning, cleaner style, or extra process."
                    ),
                    "scale": {
                        "1": "The child loses useful parent behavior, targets the wrong workspace, or moves farther from the task objective.",
                        "2": "The child edits real files but misses the parent-missing behavior or introduces likely harmful semantics.",
                        "3": "The child is behaviorally equivalent to the parent for the visible task objective.",
                        "4": "The child adds partial but meaningful task-relevant behavior over the parent.",
                        "5": "The child clearly fixes a parent-missing behavior or closes the main visible parent defect while preserving relevant constraints.",
                    },
                    "metadata": {
                        "judge_focus": "parent baseline, terminal semantic delta, and whether the child changes the behavior hidden tests are likely to exercise",
                        "evidence": "terminal diff, changed owner path, API/state/control-flow/data-flow behavior, and validation output only as supporting evidence",
                    },
            },
            {
                    "polarity": "positive",
                    "title": "Owner-Surface Integration Progress",
                    "description": (
                        "Scores whether the continuation places the fix on the code path or owner surface that the real system uses, and wires the related "
                        "interfaces consistently. This avoids rewarding isolated local edits, partial adapters, or changes that look plausible but are not "
                        "connected to the behavior under test."
                    ),
                    "scale": {
                        "1": "The child moves work away from the owner surface or breaks an interface the parent preserved.",
                        "2": "The child makes isolated edits that are unlikely to affect the real execution path.",
                        "3": "The child is equivalent to the parent in owner-surface coverage.",
                        "4": "The child integrates the fix into the main owner path but misses a related interface or edge path.",
                        "5": "The child clearly connects the fix through the relevant owner surface and related interfaces.",
                    },
                    "metadata": {
                        "judge_focus": "whether the patched files are the real owner surface and whether adjacent interfaces, adapters, or call sites remain consistent",
                        "evidence": "diff paths, import/export or call-chain changes, public API boundary, and behavior reached by normal execution",
                    },
            },
            {
                    "polarity": "positive",
                    "title": "Correct Target Workspace Progress",
                    "description": (
                        "Scores whether the continuation improves over a confused or synthetic parent by finding the actual task workspace and making "
                        "substantive progress there, rather than continuing in an empty directory, scratch reproduction, external checkout, or wrong target."
                    ),
                    "scale": {
                        "1": "The child stays in the wrong workspace or loses useful parent work in the real target.",
                        "2": "The child finds hints of the target workspace but still edits mostly synthetic or irrelevant files.",
                        "3": "The child is equivalent to the parent in repository targeting and substantive patch progress.",
                        "4": "The child locates the real workspace and starts a plausible task-relevant fix.",
                        "5": "The child clearly escapes the wrong target and applies a substantive fix in real task-relevant files.",
                    },
                    "metadata": {
                        "judge_focus": "workspace targeting, final diff paths, created scratch files, and whether patched files belong to the actual task",
                        "evidence": "repository discovery commands, current working directory, final patch paths, and distinction between reproduction artifacts and solution files",
                    },
            },
            {
                    "polarity": "negative",
                    "title": "Invalid Artifact Or Parent Work Loss",
                    "description": (
                        "Penalizes continuations that regress from useful parent work by submitting an empty, fake, summary-only, wrong-target, or incomplete "
                        "artifact, while preserving ties for partial patches that keep the core parent behavior in real source files."
                    ),
                    "scale": {
                        "1": "No artifact flaw: the child preserves the useful parent work in a real patch.",
                        "2": "Minor submission issue, but the core parent behavior remains in the workspace or patch.",
                        "3": "Ambiguous artifact completeness with some real parent work still present.",
                        "4": "Clear loss of substantial parent work, wrong-target edits, or missing key source files.",
                        "5": "Severe regression: empty/no-op patch, fake summary patch, synthetic-only fix, or complete parent-work loss.",
                    },
                    "metadata": {
                        "judge_focus": "whether useful parent work remains in the workspace or final artifact",
                        "evidence": "terminal patch artifact, diff paths, workspace edits, final submission output, and distinction between malformed stdout and true work loss",
                    },
            },
            {
                    "polarity": "negative",
                    "title": "Unsafe Broad Edit Regression",
                    "description": (
                        "Penalizes continuations that use broad or poorly controlled edits which delete unrelated existing behavior, corrupt source structure, "
                        "or regress functionality the parent preserved."
                    ),
                    "scale": {
                        "1": "No unsafe broad edit; changes are localized and preserve surrounding behavior.",
                        "2": "Minor risky rewrite with no visible loss of required existing behavior.",
                        "3": "Ambiguous broad edit where unrelated behavior may have been disturbed.",
                        "4": "Clear broad rewrite or replacement that drops important surrounding behavior.",
                        "5": "Severe destructive edit that removes large unrelated sections, corrupts syntax, or breaks preserved behavior.",
                    },
                    "metadata": {
                        "judge_focus": "whether the child regresses preserved behavior through uncontrolled editing rather than task semantics",
                        "evidence": "large unrelated deletions, syntax corruption, lost imports/exports/configuration, or broad replacement commands",
                    },
            },
        ]
    return [
        {
                "polarity": "negative",
                "title": "Target Workspace Bypass",
                "description": (
                    "Penalizes continuations that bypass the provided task workspace and treat an external checkout, newly initialized repository, or synthetic "
                    "project as the solution target."
                ),
                "scale": {
                    "1": "Uses the provided task workspace and keeps any reproductions clearly separate.",
                    "2": "Briefly creates scratch files but returns to the task workspace for the final patch.",
                    "3": "Mixes local and synthetic targets, leaving the intended patch target ambiguous.",
                    "4": "Mostly works in an external, newly initialized, or synthetic target.",
                    "5": "Submits a patch against a fabricated or externally fetched target instead of the provided workspace.",
                },
                "metadata": {
                    "judge_focus": "workspace root, target ownership, final diff paths, and whether reproduction artifacts are separated from solution files",
                },
        },
        {
                "polarity": "negative",
                "title": "Unsupported Environment Assumption",
                "description": (
                    "Penalizes continuations that choose tooling, file searches, tests, or patch targets from an unsupported assumption about the project "
                    "environment instead of first grounding that choice in visible repository evidence."
                ),
                "scale": {
                    "1": "Inspects the environment neutrally before choosing specific tools or targets.",
                    "2": "Makes a brief unsupported assumption but quickly corrects it from repository evidence.",
                    "3": "Spends noticeable effort on an unsupported assumption before recovering.",
                    "4": "Persists with unsupported tooling or target choices despite contradictory evidence.",
                    "5": "Builds the investigation or patch around a fabricated environment model.",
                },
                "metadata": {
                    "judge_focus": "whether search/tool/test choices are grounded in repository evidence before they drive the trajectory",
                },
        },
        {
                "polarity": "positive",
                "title": "Targeted Behavior Validation",
                "description": (
                    "Scores whether the continuation validates the decisive behavior through a focused check that reaches the relevant code path, instead of "
                    "relying only on broad tests, guessed APIs, syntax checks, or scripts detached from the task behavior."
                ),
                "scale": {
                    "1": "No meaningful validation or validation is detached from the relevant code path.",
                    "2": "Attempts validation but uses guessed interfaces or the wrong execution path.",
                    "3": "Uses a narrow check that touches the right area but misses the decisive behavior.",
                    "4": "Runs a targeted check against the relevant behavior with minor gaps.",
                    "5": "Executes a focused validation that directly exercises the bug path and can distinguish the correct fix from plausible wrong fixes.",
                },
                "metadata": {
                    "judge_focus": "validation target, exercised behavior, asserted outcome, and whether the check distinguishes plausible fixes",
                },
        },
        {
                "polarity": "positive",
                "title": "Compatibility Boundary Preservation",
                "description": (
                    "Scores whether the continuation identifies what behavior should change while preserving surrounding compatibility boundaries, instead "
                    "of applying a blanket conversion, blanket rollback, or test-driven overcorrection."
                ),
                "scale": {
                    "1": "Blindly changes or reverts behavior without identifying the compatibility boundary.",
                    "2": "Mentions compatibility but applies it to the wrong boundary.",
                    "3": "Handles the main behavior but misses one important compatibility exception.",
                    "4": "Mostly preserves the correct boundary with minor omissions.",
                    "5": "Clearly implements the intended behavior while preserving compatibility-sensitive cases supported by evidence.",
                },
                "metadata": {
                    "judge_focus": "changed behavior, preserved public contract, ordering/lifecycle/schema constraints, and evidence for compatibility-sensitive edge cases",
                },
        },
        {
                "polarity": "negative",
                "title": "Unsafe Source Modification",
                "description": (
                    "Penalizes continuations whose editing method or patch shape corrupts source structure, deletes unrelated behavior, or makes correctness "
                    "depend on fragile incidental file layout rather than a controlled semantic change."
                ),
                "scale": {
                    "1": "Edits are localized, reviewable, and preserve surrounding source structure.",
                    "2": "Minor brittle editing risk but no visible source corruption.",
                    "3": "Some broad replacement or manual reconstruction with ambiguous source integrity.",
                    "4": "Clear unsafe edit that corrupts syntax or drops unrelated behavior.",
                    "5": "Severe source corruption from broad rewrite, fragile deletion, or malformed patch application.",
                },
                "metadata": {
                    "judge_focus": "edit locality, diff hunks, syntax integrity, unrelated deletions, and preservation of existing behavior",
                },
        },
    ]


AGGREGATE_TRAJECTORY_SUMMARY_PROMPT = """
You are an expert evaluator compressing one complete SWE-agent trajectory into a compact, comparable summary.

## Task
Summarize the full trajectory from the initial task prompt through the final submitted patch or step-limit stop. The summary will be compared against other complete trajectories for the same coding task, so preserve evidence that helps decide which full attempt produced the best final patch.

## Required Sections
Return exactly these 8 top-level string fields:
- **current_state**: Where this trajectory ended, whether it submitted or stopped at the step limit, and what remained unresolved.
- **task_specification**: The original coding problem, constraints, acceptance criteria, and any task-specific behavior the trajectory identified.
- **files_and_functions**: Important files, functions, classes, modules, and why they matter. Include concrete file paths and identifiers.
- **errors_and_corrections**: Failed commands, incorrect hypotheses, format/tool errors, dead ends, and later corrections.
- **codebase_and_system_documentation**: Relevant repository architecture, interfaces, workflows, invariants, or external contracts discovered by the trajectory.
- **learnings**: Actionable lessons about what worked, what did not, and what should be avoided when judging this attempt.
- **key_results**: Final patch behavior, edited files, validation results, exact test/command outcomes, and other decisive artifacts.
- **worklog**: Terse chronological record of the main investigation, edit, validation, and submission steps.

## Writing Guidelines
- Use only evidence visible in the provided trajectory and workspace metadata.
- Be specific and information-dense. Include concrete commands, file paths, function names, tests, errors, patch details, and validation outcomes when useful.
- Preserve the distinction between confirmed facts, attempted fixes, failed checks, and unsupported claims.
- Focus on evidence that helps compare complete trajectories: final patch semantics, target-file relevance, bug understanding, validation quality, regressions, empty/no-op patches, or step-limit soft patches.
- It is OK to leave a section blank if there is no substantial evidence for it. Do not add filler such as "No info yet".
- Keep each section under 400 words. Prefer compression over copying long logs verbatim.
- Do not hallucinate hidden repository behavior, test results, or patch effects. Prefer omission to speculation.

## Output Format
```json
{
  "current_state": "",
  "task_specification": "",
  "files_and_functions": "",
  "errors_and_corrections": "",
  "codebase_and_system_documentation": "",
  "learnings": "",
  "key_results": "",
  "worklog": ""
}
```

## Inputs
1. **Question**: Original system and user prompt containing the coding task
2. **Trajectory Metadata**: Node id, stop status, patch length, fallback-patch flag, and model statistics
3. **Full Agent Trajectory**: Step cards for the complete sampled trajectory

Return only the updated JSON object.
"""

AGGREGATE_RUBRIC_GENERATION_PROMPT = """
You are an expert evaluator generating rubrics to compare complete SWE-agent trajectories for the same coding task.

## Task
Generate the single most useful non-redundant criterion for ranking the provided complete trajectory summaries. The criterion should help select the trajectory whose final patch should be submitted. This is a multi-turn rubric generation setting: each turn generates at most one new rubric, and rubrics already generated in this aggregate run count as existing coverage. On the first turn, if trajectory summaries are provided, you must generate one concrete rubric that can compare them. Only return an empty JSON object `{}` on later turns when the already generated rubrics cover all high-impact distinctions.

## Output Components
- **Title**: Concise abstract label that is reusable across tasks.
- **Description**: A concrete criterion grounded in observable trajectory, patch, validation, or repository evidence.
- **Scale**: A five-point scale from 1 to 5 with concrete anchors. For a positive rubric, 5 is strongest evidence of quality. For a negative rubric, 5 is most severe evidence of the flaw.
- **Polarity**: Either `"positive"` or `"negative"`.
- **Metadata**: Structured judging evidence such as `judge_focus`, `failure_mode`, `patch_semantics`, `validation_signal`, `target_files`, `oracle_test`, or `code_review`.

## Core Guidelines

### 1. Compare Complete Attempts
- Focus on differences among complete trajectories.
- Prefer criteria that predict final patch quality: correct target behavior, relevant source edits, preserved compatibility, edge-case coverage, regression risk, validation strength, and whether the patch is empty/no-op/synthetic.
- Process evidence such as exploration or testing is useful only when it changes confidence about the final patch or exposes a concrete failure mode.

### 2. Ground The Criterion In Visible Evidence
- If summaries differ in edited files, APIs, data flow, lifecycle behavior, schema/interface handling, or tests, write the rubric around that observable difference.
- If summaries differ mainly in validation, score the relevance and outcome of those checks, not generic "thoroughness".
- Do not reward majority behavior merely because many trajectories share it. Reward the behavior best supported by task-relevant evidence.
- Avoid vague criteria such as "correctness", "best practice", or "complete implementation" unless the scale defines concrete evidence to score.

### 3. Novelty And Non-Redundancy
- Do not duplicate a rubric already generated in this aggregate run.
- Generate a new rubric only when it adds a distinct ranking signal.
- Prefer `{}` over a weak, generic, or redundant rubric.
- Do not generate paired duplicates such as "runs validation" and "does not run validation"; choose the direction that best separates these trajectories.

### 4. Conservative Negative Rubrics
- Negative rubrics should describe clear harmful behavior: wrong target, empty patch, fabricated evidence, source corruption, build-breaking edit, lost behavior, irrelevant scratch-only work, or unsupported submission.
- Do not create a negative rubric merely because a trajectory lacks an optional excellence signal.
- For negative rubrics, scale anchors must measure severity of the flaw: 1 means absent/minimal flaw and 5 means severe flaw.

## Selection Strategy
- Generate 0-1 rubric per turn, with 1-5 total rubrics across the run.
- The first turn should generate exactly one useful rubric whenever at least one trajectory summary is present.
- Prefer a small set of strong, independent rubrics over many broad rubrics.
- If a generated rubric would apply equally to every trajectory, return `{}`.

## Output Format
```json
{
  "polarity": "<positive|negative>",
  "description": "<detailed aggregate trajectory ranking criterion>",
  "title": "<abstract label>",
  "metadata": {
    <structured evidence needed for judging>
  },
  "scale": {
    "1": "<positive rubric: weakest evidence / negative rubric: no evidence of the flaw>",
    "2": "<positive rubric: weak evidence / negative rubric: minor evidence of the flaw>",
    "3": "<moderate or mixed evidence>",
    "4": "<positive rubric: strong evidence / negative rubric: clear evidence of the flaw>",
    "5": "<positive rubric: strongest evidence / negative rubric: most severe evidence of the flaw>"
  }
}
```
If no new high-impact, non-redundant rubric should be added, output:
```json
{}
```

## Inputs
1. **Question**: Original system and user prompt containing the coding task
2. **Trajectory Summaries**: Multiple compressed summaries of complete sampled trajectories

Return only the JSON object.
"""

AGGREGATE_RUBRIC_JUDGE_PROMPT = """
You are an expert evaluator scoring one complete SWE-agent trajectory summary against one evaluation rubric.

## Task
Evaluate the provided complete trajectory summary using only the provided criterion and task context.

## Core Guidelines
- Judge only the specified criterion, not general quality.
- Use the rubric scale exactly. For negative rubrics, 5 means the harmful behavior is severe and 1 means it is absent or minimal.
- Score the complete trajectory as a standalone attempt for the task.
- Use only evidence visible in the summary and task context. Do not infer hidden test results, unstated repository behavior, or patch effects.
- If evidence is mixed or incomplete, choose the scale anchor best supported by visible evidence.
- Output only the score JSON. Do not restate the question, criterion, or trajectory.

## Output Format
```json
{
  "score": <integer from 1 to 5>
}
```

## Inputs
1. **Question**: Original system and user prompt containing the coding task
2. **Complete Trajectory Summary**: The compressed view of one sampled trajectory
3. **Criterion**: The evaluation rubric to apply

Return only the JSON object.
"""
