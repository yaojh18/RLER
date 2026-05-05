SWE_TRAJECTORY_RUBRIC_GENERATION_PROMPT = """
You are an expert evaluator generating adaptive rubrics to assess agent trajectory continuations.

## Task
Identify the single most discriminative criterion that distinguishes high-quality from low-quality agent trajectory continuations and is not already covered by the existing rubrics. Capture subtle quality differences that existing rubrics miss.
This is a multi-turn rubric generation setting. At each turn, generate at most one new rubric. The user may ask to continue in later turns. Existing Rubrics contains previously generated rubrics and should be used to understand the current evaluation gap and avoid redundancy.
If no additional high-impact, non-redundant rubric remains, return an empty JSON object: {}.

## Output Components
- **Description**: Detailed, specific description of what makes a continuation excellent/problematic
- **Title**: Concise abstract label (general, not task-specific)
- **Scale**: A five-point scale from 1 to 5 with concrete anchors for this rubric
- **Polarity**: Either `"positive"` or `"negative"`

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

### 2. Novelty & Non-Redundancy
With existing rubrics:
- Never duplicate overlapping rubrics in meaning/scope
- Identify uncovered quality dimensions
- Add granular criteria if existing rubrics are broad
- Return empty lists if existing rubrics are comprehensive

### 3. Avoid Mirror Rubrics
Never create positive/negative versions of same criterion:
- ❌ "Runs targeted validation" + "Does not run targeted validation"
- ✅ Choose only the more discriminative direction

### 4. Conservative Negative Rubrics
- Identify clear failure modes, not absence of excellence
- Response penalized if it exhibits ANY negative rubric behavior
- Focus on active mistakes vs missing features

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
  "rubric": {
    "polarity": "<positive|negative>",
    "description": "<detailed excellence/failure description>",
    "title": "<abstract label>",
    "scale": {
      "1": "<worst case anchor>",
      "2": "<weak/minor issue anchor>",
      "3": "<partial/moderate anchor>",
      "4": "<strong/serious issue anchor>",
      "5": "<best case/severe issue anchor>"
    }
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
3. **Latest Agent Trajectory**: The most recent agent trajectory
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
- Output in the requried format. Do not restate the question, previous state, agent tracjectories, or existing rubrics in the response.

Generate only the most impactful, non-redundant rubrics revealing meaningful quality differences.
"""

RUBRIC_GENERATION_CONTINUE_PROMPT = "Generate the next best rubric or return an empty object."

SWE_TRAJECTORY_RUBRIC_JUDGE_PROMPT = """
You are an expert evaluator scoring one agent trajectory continuation against one rubric.

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
3. **Latest Agent Trajectory**: The most recent agent trajectory
4. **Continuation Trajectory**: A agent trajectory continued from the latest trajectory
5. **Criterion**: The specific aspect to evaluate

Return only the JSON object.
"""

SWE_TRAJECTORY_RUBRIC_JUDGE_PARENT_PROMPT = """
You are an expert evaluator scoring one agent trajectory against one rubric.

## Task
Evaluate the provided agent trajectory using the provided criterion and the shared context.

## Core Guidelines
- Judge only the specified criterion, not general quality
- Use the rubric's scale exactly as written. For negative rubrics, do not invert the scale
- Score the agent trajectory itself, not the underlying task or bug in the abstract
- Use only evidence visible in the trajectory. Do not hallucinate or infer unstated facts
- Use the previous persistent state only when it is needed to interpret the tracjectory
Output only score in the requried format. Do not restate the question, criterion, presistent state, or agent trajectories in the response


## Output Format
```json
{
  "score": <a score on a scale of 1 to 5 indicating how appropriate the continuation is based on the scale of the given criterion>
}
```

## Inputs
1. **Question**: Original system and user prompt containing code problem statement
2. **Previous Persistent State**: Current memory state with summary of past findings and milestones
3. **Agent Trajectory**: The most recent agent trajectory
5. **Criterion**: The specific aspect to evaluate

Return only the JSON object.
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
        "title": {"type": "string"},
        "description": {"type": "string"},
        "scale": RUBRIC_SCALE_JSON_SCHEMA,
    },
    "required": ["title", "description", "scale"],
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

RUBRIC_GENERATION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "adaptive_rubric_generation",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "rubric": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "polarity": {"type": "string", "enum": ["positive", "negative"]},
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "scale": RUBRIC_SCALE_JSON_SCHEMA,
                    },
                    "required": ["polarity", "title", "description", "scale"],
                },
            },
        },
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