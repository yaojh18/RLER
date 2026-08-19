# TODO: expension of the seed experience and rubric bank is needed for better cold start.

from __future__ import annotations

from typing import Any


SWE_TRAJECTORY_RUBRIC_GENERATION_PROMPT = """
You are an expert evaluator generating adaptive rubrics to assess agent trajectory continuations.

## Task
Identify the single most discriminative rubric to output next for judging the current trajectory continuations. Capture subtle quality differences that existing rubrics miss.
This is a multi-turn rubric generation setting. At each turn, output exactly one rubric object or an empty JSON object `{}`. The rubric object may be either a newly generated rubric or a reused/adapted existing rubric. Existing Rubrics contains previously generated rubrics that may be reused/adapted and should be used to understand the current evaluation gap and avoid redundancy.
Every generation sample must output at least one non-empty rubric before it may return `{}`. Existing rubrics are not reused automatically; reuse requires outputting the full rubric object again.
If no further rubric should be output in this multi-turn generation process, return an empty JSON object: {}.

## Output Components
- **Title**: Concise abstract label (general, not task-specific)
- **Description**: Detailed, specific description of what makes a continuation excellent/problematic
- **Scale**: A five-point scale from 1 to 5 with concrete anchors for this rubric. The scale must follow the rubric polarity: for a positive rubric, 1 is the weakest evidence and 5 is the strongest evidence; for a negative rubric, 1 is no/least evidence of the flaw and 5 is the most severe evidence of the flaw.
- **Polarity**: Either `"positive"` or `"negative"`.
- **Weight**: A positive number expressing this rubric's relative importance among all the rubrics generated. Use `1.0` as a neutral default and assign higher weights to impactful rubrics and lower weights to less impactful ones. Weight is always positive even for negative rubrics.
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
- If continuations differ in diagnostic strategy, reproduction attempts, validation attempts, or targeting of relevant files, define a rubric around that concrete process evidence
- If continuations differ in source changes, define a rubric around the observable patch behavior: changed files, symbols, API contracts, data flow, compatibility boundaries, edge cases, or tests
- If continuations share the same visible core behavior, do not separate them using harmless formatting, error-message wording, local variable placement, scratch scripts, or transient test scaffolding unless those details create an observable behavioral risk
- Do not create standalone style rubrics for DRYness, helper extraction, formatting, comments, or cleanup. Such details are only valid when the visible diff shows a concrete behavioral, compatibility, or maintainability risk that affects the task outcome
- Do not reward majority behavior just because most continuations share it; reward the behavior best supported by the visible evidence
- Avoid vague criteria such as "thoroughness", "correctness", "best practice", or "complete implementation" unless the rubric defines the concrete evidence being scored

### 3. Novelty & Non-Redundancy
- Do not generate a new rubric that duplicates any existing or already generated rubric in meaning/scope. Re-outputing an existing rubric as a reused/adapted rubric is allowed and is not considered duplication.
- Identify uncovered quality dimensions
- Add granular criteria if existing rubrics are broad
- Return `{}` only when no remaining existing rubric should be reused/adapted and no new non-redundant rubric should be generated
- Do not generate semantically equivalent rubrics, e.g., "Runs targeted validation" as a positive rubric and "Does not run targeted validation" as a negative rubric.
- Choose only the more discriminative direction

### 4. Conservative Negative Rubrics
- Identify clear failure modes, not absence of excellence
- A negative rubric should describe an observable harmful behavior, incorrect assumption, misleading edit, or unsupported claim
- Do not create a negative rubric merely because a continuation lacks a desirable behavior
- For negative rubrics, every scale anchor must measure severity of the flaw: 1 means the flaw is absent or minimal, and 5 means the flaw is clearly and severely present. Never write a negative rubric whose scale rewards the good behavior at 5.

### 5. Previous Generated Rubrics & Experiences
- In each turn, output exactly one rubric object or `{}`. A non-empty output may either reuse/adapt a previously generated rubric if it is still applicable and discriminative for the current trajectory continuations, or generate one new rubric, optionally informed by retrieved experiences.
- If a concrete, important trajectory behavior gap is not captured by any existing rubric or experience, generate a new rubric for that missing evaluation criterion. This is especially needed when you identify task-specific mistakes at the specific agent stage.
- Rubric reuse is opt-in, not automatic. Previously generated rubrics are not used by the judge merely because they appear under Existing Rubrics. To reuse an existing rubric, output the full rubric object in one generation turn. Any existing rubric that is not output in the current multi-turn generation process is considered dropped and will not participate in judging.
- You may freely modify `weight` or other details of an existing rubric or reference golden rubric in the experience to better reflect the judging importance and focus at the current stage. A modified existing rubric still counts as reuse/adaptation as long as it is derived from an existing rubric.
- Use previous rubrics and experiences to understand what is already covered, then output only one useful next rubric: either a reused/adapted existing rubric or a non-redundant uncovered criterion.
- An experience is guidance about when a rubric is useful or misleading. Treat its `context` and `experience` fields as applicability conditions, not as facts about the current task.
- `metadata.reference_golden_rubrics` contains candidate criteria learned from earlier cases. You may adapt a candidate's polarity, scope, wording, metadata, scale, or weight only when the current continuations match the prior lesson.
- Do not copy a reference rubric merely because it was retrieved. A copied or adapted rubric must be independently relevant and judgeable from the current task and trajectory continuations.
- Retrieved experiences are not privileged oracles. Never infer the current final patch, hidden tests, or outcome from their presence.
- If a previous rubric or experience would turn a precise current distinction into a broad generic or misleading rubric, ignore it.

## Selection Strategy

### Quantity: in each turn, output exactly one rubric object or return `{}`. Across the full multi-turn generation process, output at least 1 and at most 6 non-empty rubrics in total.
- Output exactly one rubric object if the next rubric should participate in judging, whether it is newly generated or reused/adapted from existing rubrics.
- Return `{}` only when no remaining existing rubric should be reused/adapted and no new high-impact, non-redundant rubric should be generated.

### Polarity Selection Based on Response Patterns:
- **More positive**: When continuations lack sophistication but avoid major errors
- **More negative**: When systematic failure patterns are present
- **Balanced across turns**: When both excellence gaps and failure modes exist
- **Empty object**: When no further rubric should be output: all useful existing rubrics have already been reused/adapted in previous generation turns, and no new high-impact, non-redundant rubric remains

## Analysis Process
1. Group continuations by quality level
2. Find factors separating higher/lower clusters
3. Check if factors are covered by rubrics already output in the current multi-turn generation process
4. Select the single criterion with the highest discriminative value

## Output Format Example
<format_example>

THOUGHT: <your reasoning process>

```json
{
    "polarity": "<positive|negative>",
    "weight": <positive number>,
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

</format_example>

If no further rubric should be output in this multi-turn generation process, output:
<format_example>

THOUGHT: <your reasoning process>

```json
{}
```

</format_example>

## Inputs
1. **Question**: Original system and user prompt containing code problem statement
2. **Previous Persistent State**: Current memory state with summary of past findings and milestones
3. **Parent Trajectory**: The most recent agent trajectory
4. **Agent Trajectory Continuations**: Multiple agent trajectories continued from the latest trajectory (Continuation 1, Continuation 2, etc.)
5. **Existing Rubrics** (optional): Previously generated rubrics available for reuse/adaptation and redundancy checking. They are not automatically used for judging unless explicitly output as full rubric objects in the current multi-turn generation process.

## Critical Reminders
- Each rubric must distinguish between the actual provided continuations
- Exclude rubrics applying equally to all continuations
- Prefer `{}` over redundant rubric generation when no remaining existing rubric should be reused/adapted and no new non-redundant rubric remains
- Focus on observable, objective, actionable criteria
- Quality over quantity: 1 excellent rubric > multiple mediocre ones
- The shared context is common to all continuations. Focus the rubric on differences between the continuations themselves
- Do not return `{}` when there is still a useful existing rubric to reuse/adapt or a visible, important, non-redundant difference in diagnostic strategy, reproduction attempts, validation attempts, or targeting of relevant files
- Never output a list of rubrics. Each generation turn must output exactly one rubric object or `{}`
- Output in the required format. Do not restate the question, previous state, agent trajectories, or existing rubrics in the response.

Generate only the most impactful, non-redundant rubric revealing meaningful quality differences, or explicitly reuse/adapt one existing rubric that should participate in judging.
"""

RUBRIC_GENERATION_CONTINUE_PROMPT = (
    "Output the next rubric that should participate in judging, either newly generated or reused/adapted from existing rubrics, or return an empty object `{}` to stop. Output exactly one rubric object or `{}`; never output a list."
)

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
- Ground the score in exact visible trajectory evidence. In the final JSON object, put the evidence field before the score field.
- Keep the structured answer limited to the evidence and score fields. Do not restate the full question, criterion, persistent state, or trajectories inside the JSON block.

## Output Format Example
<format_example>

THOUGHT: <your reasoning process>

```json
{
  "evidence": "<exact visible evidence supporting the selected scale anchor>",
  "score": <a score on a scale of 1 to 5 indicating how appropriate the continuation is based on the scale of the given criterion>
}
```

</format_example>

## Inputs
1. **Question**: Original system and user prompt containing code problem statement
2. **Previous Persistent State**: Current memory state with summary of past findings and milestones
3. **Parent Trajectory**: The most recent agent trajectory
4. **Continuation Trajectory**: A agent trajectory continued from the latest trajectory
5. **Criterion**: The specific criterion to evaluate

"""

PC_TRAJECTORY_RUBRIC_GENERATION_PROMPT = """
You are an expert evaluator generating adaptive rubrics to assess agent progress for SWE tasks.

## Task
Identify the single most useful rubric to output next for judging whether each continuation improves, stays equivalent, or regresses relative to the provided parent trajectory. Capture subtle quality differences that existing rubrics miss.
This is a multi-turn rubric generation setting. At each turn, output exactly one rubric object or an empty JSON object `{}`. The rubric object may be either a newly generated rubric or a reused/adapted existing parent-child rubric. Existing Rubrics contains previously generated rubrics that may be reused/adapted and should be used to understand the current evaluation gap and avoid redundancy.
Every generation sample must output at least one non-empty rubric before it may return `{}`. Existing rubrics are not reused automatically; reuse requires outputting the full rubric object again.
If no further rubric should be output in this multi-turn generation process, return an empty JSON object: {}.

## Output Components
- **Title**: Concise abstract label (general, not task-specific)
- **Description**: A specific relative progress evaluation criterion grounded in observable trajectory or patch evidence. It must name the relevant parent baseline and the child behavior that would count as progress, equivalence, or regression.
- **Scale**: A five-point scale from 1 to 5 with concrete anchors for this rubric. The scale must follow the rubric polarity: for a positive rubric, 1 is the weakest evidence and 5 is the strongest evidence; for a negative rubric, 1 is no/least evidence of the flaw and 5 is the most severe evidence of the flaw.
- **Polarity**: Either `"positive"` or `"negative"`.
- **Weight**: A positive number expressing this rubric's relative importance among all the rubrics generated. Use `1.0` as a neutral default and assign higher weights to impactful rubrics and lower weights to less impactful ones. Weight is always positive even for negative rubrics.
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
- Do not generate a new rubric that duplicates any existing or already generated rubric in meaning/scope. Re-outputing an existing rubric as a reused/adapted rubric is allowed and is not considered duplication.
- Identify uncovered quality dimensions
- Add granular criteria if existing rubrics are broad
- Return `{}` only when no remaining existing rubric should be reused/adapted and no new non-redundant parent-relative rubric should be generated.
- Do not generate semantically equivalent rubrics, e.g., "Runs targeted validation" as a positive rubric and "Does not run targeted validation" as a negative rubric.
- Use previous rubrics to understand what is already covered, then output only one useful next rubric: either a reused/adapted existing rubric or a non-redundant uncovered parent-relative criterion.

### 5. Conservative Negative Rubrics
- Identify clear failure modes, not absence of excellence
- A negative rubric should describe an observable harmful behavior, incorrect assumption, misleading edit, or unsupported claim
- Do not create a negative rubric merely because a continuation lacks a desirable behavior
- For negative rubrics, every scale anchor must measure severity of the flaw: 1 means the flaw is absent or minimal, and 5 means the flaw is clearly and severely present. Never write a negative rubric whose scale rewards the good behavior at 5.

### 6. Previous Generated Rubrics & Experiences
- In each turn, output exactly one rubric object or `{}`. A non-empty output may either reuse/adapt a previously generated rubric if it is still applicable and discriminative for the current parent-child comparison, or generate one new rubric, optionally informed by retrieved experiences.
- Rubric reuse is opt-in, not automatic. Previously generated rubrics are not used by the judge merely because they appear under Existing Rubrics. To reuse an existing rubric, output the full rubric object in one generation turn. Any existing rubric that is not output in the current multi-turn generation process is considered dropped and will not participate in judging.
- You may freely modify `weight` or other details of an existing rubric or reference golden rubric to better reflect the current parent baseline, child delta, and judging importance. A modified existing rubric still counts as reuse/adaptation as long as it is derived from an existing rubric.
- Use previous rubrics and experiences to understand what is already covered, then output only one useful next rubric: either a reused/adapted existing rubric or a non-redundant uncovered criterion.
- An experience is guidance about when a rubric is useful or misleading. Treat its `context` and `experience` fields as applicability conditions, not as facts about the current task.
- `metadata.reference_golden_rubrics` contains candidate criteria learned from earlier cases. You may adapt a candidate's polarity, scope, wording, metadata, scale, or weight only when the current parent-child evidence matches the prior lesson.
- Do not copy a reference rubric merely because it was retrieved. A copied or adapted rubric must be independently relevant and judgeable from the current parent and continuations.
- Retrieved experiences are not privileged oracles. Never infer the current final patch, hidden tests, or outcome from their presence.
- If a previous rubric or experience would turn a precise current distinction into a broad generic rubric, ignore it

### 7. Quantity
- In each turn, output exactly one rubric object or return `{}`. Across the full multi-turn generation process, output at least 1 and at most 6 non-empty rubrics in total.
- Return `{}` only when all useful existing rubrics have already been reused/adapted in previous turns and no new high-impact, non-redundant parent-relative rubric remains.


## Output Format Example
<format_example>

THOUGHT: <your reasoning process>

```json
{
    "polarity": "<positive|negative>",
    "weight": <positive number>,
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

</format_example>

If no further rubric should be output in this multi-turn generation process, output:
<format_example>

THOUGHT: <your reasoning process>

```json
{}
```

</format_example>

## Inputs
1. **Question**: Original system and user prompt containing code problem statement
2. **Previous Persistent State**: Current memory state with summary of past findings and milestones
3. **Parent Trajectory**: The most recent agent trajectory
4. **Agent Trajectory Continuations**: Multiple agent trajectories continued from the latest trajectory (Continuation 1, Continuation 2, etc.)
5. **Existing Rubrics** (optional): Previously generated rubrics available for reuse/adaptation and redundancy checking. They are not automatically used for judging unless explicitly output as full rubric objects in the current multi-turn generation process.

## Critical Reminders
- Each rubric must distinguish meaningful parent-relative progress, equivalence, or regression in the actual provided continuations
- Exclude rubrics that apply equally to the parent and all continuations
- Prefer `{}` over redundant rubric generation when no remaining existing rubric should be reused/adapted and no new non-redundant rubric remains
- Focus on observable, objective, actionable criteria
- Quality over quantity: 1 excellent rubric > multiple mediocre ones
- The parent trajectory is the baseline. Focus the rubric on child deltas over that baseline
- Do not return `{}` when there is still a useful existing rubric to reuse/adapt or a visible, important, non-redundant parent-relative difference in diagnostic strategy, reproduction attempts, validation attempts, or targeting of relevant files
- Never output a list of rubrics. Each generation turn must output exactly one rubric object or `{}`
- Output in the required format. Do not restate the question, previous state, agent trajectories, or existing rubrics in the response.

Generate only the most impactful, non-redundant rubric revealing meaningful parent-child progress differences, or explicitly reuse/adapt one existing rubric that should participate in judging.
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
- Ground the score in exact visible parent/continuation evidence. In the final JSON object, put the evidence field before the score field.
- Keep the structured answer limited to the evidence and score fields. Do not restate the full question, criterion, persistent state, or trajectories inside the JSON block.

## Output Format Example
<format_example>

THOUGHT: <your reasoning process>

```json
{
  "evidence": "<exact visible evidence supporting the selected scale anchor>",
  "score": <integer from 1 to 5>
}
```

</format_example>

## Inputs
1. **Question**: Original system and user prompt containing code problem statement
2. **Previous Persistent State**: Current memory state with summary of past findings and milestones
3. **Parent Trajectory**: The most recent agent trajectory
4. **Continuation Trajectory**: A agent trajectory continued from the parent trajectory to compare
5. **Criterion**: The specific parent-child progress criterion to evaluate

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

## Output Format Example
<format_example>

THOUGHT: <your reasoning process>

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

</format_example>

## Inputs
1. **Question**: Original system and user prompt containing the coding task
2. **Previous Persistent State**: Previous memory state
3. **Evicted Older Trajectory**: Older trajectory segments that must now be compressed
4. **Workspace Metadata**: Compact git-based metadata at the current step

"""

STRUCTURED_SUMMARY_FORMAT_CORRECTION_PROMPT = """
The previous response could not be parsed into the required summary fields. Return a corrected response following the output format example.

## Output Format Example
<format_example>

THOUGHT: <your reasoning process>

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

</format_example>
"""

RUBRIC_JUDGE_FORMAT_CORRECTION_PROMPT = """
The previous response could not be parsed into valid evidence and a score. Keep any substantive reasoning, then append a corrected final JSON object following the output format example.

## Output Format Example
<format_example>

THOUGHT: <your reasoning process>

```json
{
  "evidence": "<exact visible evidence supporting the selected scale anchor>",
  "score": <integer from 1 to 5>
}
```

</format_example>
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
    "git_diff": "",
    "current_patch_chars": 0,
    "workspace_fingerprint": None,
}


def _seed_rubrics(scope: str = "siblings") -> list[dict[str, Any]]:
    if scope == "pc":
        return [
            {
                    "polarity": "positive",
                    "weight": 1.0,
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
                    "weight": 1.0,
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
                    "weight": 1.0,
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
                    "weight": 1.0,
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
                    "weight": 1.0,
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
                "weight": 1.0,
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
                "weight": 1.0,
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
                "weight": 1.0,
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
                "weight": 1.0,
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
                "weight": 1.0,
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

## Output Format Example
<format_example>

THOUGHT: <your reasoning process>

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

</format_example>

## Inputs
1. **Question**: Original system and user prompt containing the coding task
2. **Trajectory Metadata**: Node id, stop status, patch length, fallback-patch flag, and model statistics
3. **Full Agent Trajectory**: Step cards for the complete sampled trajectory

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
- **Weight**: A positive number expressing this rubric's relative importance among all the rubrics generated. Use `1.0` as a neutral default and assign higher weights to impactful rubrics and lower weights to less impactful ones. Weight is always positive even for negative rubrics.
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

## Output Format Example
<format_example>

THOUGHT: <your reasoning process>

```json
{
  "polarity": "<positive|negative>",
  "weight": <positive number>,
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

</format_example>

If no new high-impact, non-redundant rubric should be added, output:
<format_example>

THOUGHT: <your reasoning process>

```json
{}
```

</format_example>

## Inputs
1. **Question**: Original system and user prompt containing the coding task
2. **Trajectory Summaries**: Multiple compressed summaries of complete sampled trajectories

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
- Ground the score in exact visible summary evidence. In the final JSON object, put the evidence field before the score field.
- Keep the structured answer limited to the evidence and score fields. Do not restate the full question, criterion, or trajectory inside the JSON block.

## Output Format Example
<format_example>

THOUGHT: <your reasoning process>

```json
{
  "evidence": "<exact visible evidence supporting the selected scale anchor>",
  "score": <integer from 1 to 5>
}
```

</format_example>

## Inputs
1. **Question**: Original system and user prompt containing the coding task
2. **Complete Trajectory Summary**: The compressed view of one sampled trajectory
3. **Criterion**: The evaluation rubric to apply

"""
