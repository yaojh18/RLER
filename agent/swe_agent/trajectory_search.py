from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import pvariance
from typing import Any, Literal

repo_root = Path(__file__).resolve().parents[2]
candidate = repo_root / "rl" / "open-instruct"
if str(candidate) not in sys.path and candidate.exists():
    sys.path.append(str(candidate))
from open_instruct.search_rewards.utils.run_utils import extract_json_from_response, run_chat_with_route_async

from swe_agent import __version__
from agent_rl import RolloutSessionSpec, RolloutSnapshot
from swe_agent.rl_backend import SWEAgentRolloutBackend
from swe_agent.run.run_swe_agent import build_slim_trajectory, evaluate_swebench_instance_patches


SWE_TRAJECTORY_RUBRIC_GENERATION_PROMPT = """
You are an expert evaluator generating adaptive rubrics to assess model responses.

## Task
Identify the most discriminative criteria that distinguish high-quality from low-quality agent trajectory continuations. Capture subtle quality differences that existing rubrics miss.

## Output Components
- **Description**: Detailed, specific description of what makes a continuation excellent/problematic
- **Title**: Concise abstract label (general, not task-specific)
- **Scale**: A five-point scale from 1 to 5 with concrete anchors for this rubric

## Categories
1. **Positive Rubrics**: Excellence indicators distinguishing superior continuations
2. **Negative Rubrics**: Critical flaws definitively degrading quality

## Core Guidelines

### 1. Discriminative Power
- Focus ONLY on criteria meaningfully separating quality levels
- Each rubric must distinguish between otherwise similar continuations from the same shared prefix
- Exclude generic criteria applying equally to all continuations
- If the continuations are still in the exploration stage, prefer rubrics about exploration quality, reproduction quality, repository grounding, file targeting, and follow-through rather than returning empty lists

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

### Quantity: 1-5 total rubrics (fewer high-quality > many generic)

### Distribution Based on Response Patterns:
- **More positive**: Continuations lack sophistication but avoid major errors
- **More negative**: Systematic failure patterns present
- **Balanced**: Both excellence gaps and failure modes exist
- **Empty lists**: Existing rubrics already comprehensive

## Analysis Process
1. Group continuations by quality level
2. Find factors separating higher/lower clusters
3. Check if factors covered by existing rubrics
4. Select criteria with highest discriminative value
5. Brief reasoning is allowed and should stay concrete

## Output Format
```json
{
  "reasoning": "<brief grounded analysis>",
  "positive_rubrics": [
    {
      "description": "<detailed excellence description>",
      "title": "<abstract label>",
      "scale": {
        "1": "<worst case for this positive rubric>",
        "2": "<weak>",
        "3": "<partial>",
        "4": "<strong>",
        "5": "<best case>"
      }
    }
  ],
  "negative_rubrics": [
    {
      "description": "<detailed failure description>",
      "title": "<abstract label>",
      "scale": {
        "1": "<no issue>",
        "2": "<minor issue>",
        "3": "<moderate issue>",
        "4": "<serious issue>",
        "5": "<severe issue>"
      }
    }
  ]
}
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
- Quality over quantity: 2 excellent rubrics > 5 mediocre ones
- The shared context is common to all continuations. Focus the rubric on differences between the continuations themselves
- Do not return empty lists when there are visible differences in diagnostic strategy, reproduction attempts, validation attempts, or targeting of relevant files
- Output in the requried format. Do not restate the question, previous state, agent tracjectories, or existing rubrics in the response.

Generate only the most impactful, non-redundant rubrics revealing meaningful quality differences.
"""

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
- Brief reasoning is allowed and should stay concrete
- Output in the requried format. Do not restate the question, criterion, presistent state, or agent trajectories in the response

## Output Format
```json
{
  "reasoning": "<brief grounded explanation>",
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
- Brief reasoning is allowed and should stay concrete
- Output in the requried format. Do not restate the question, criterion, presistent state, or agent trajectories in the response


## Output Format
```json
{
  "reasoning": "<brief grounded explanation>",
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

RETURN_CODE_RE = re.compile(r"<returncode>(.*?)</returncode>", re.DOTALL)
EXCEPTION_RE = re.compile(r"<exception>(.*?)</exception>", re.DOTALL)
OUTPUT_RE = re.compile(r"<output>\s*(.*?)</output>", re.DOTALL)
PR_DESCRIPTION_RE = re.compile(r"<pr_description>\s*(.*?)\s*</pr_description>", re.DOTALL)
OBSERVATION_TRUNCATION_MARKER = "\n[... Observation truncated due to length ...]\n"
MAX_OBSERVATION_CHARS = 2048
MIN_OBSERVATION_SECTION_CHARS = 256
EVALUATOR_MAX_RETRIES = 4

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


@dataclass
class SearchConfig:
    m: int = 8
    k: int = 10
    p: int = 2
    max_rounds: int = 25
    max_active_rubrics: int = 6
    policy_temperature: float = 1.0
    policy_top_p: float = 1.0
    rubric_temperature: float = 0.0
    rubric_top_p: float = 1.0
    rubric_max_tokens: int = 1024
    judge_temperature: float = 0.0
    judge_top_p: float = 1.0
    judge_max_tokens: int = 1024
    regression_margin: float = 0.0
    calculate_gt_reward: bool = False
    gt_reward_workers: int = 1


@dataclass
class RubricRecord:
    rubric_id: str
    title: str
    direction: Literal["positive", "negative"]
    description: str
    scale: dict[str, str]
    weight: int
    source_round: int
    variance: float | None = None


@dataclass
class SearchNode:
    node_id: str
    parent_id: str | None
    round_index: int
    depth: int
    session_id: str
    status: str
    checkpoint_image_tag: str | None
    checkpoint_image_id: str | None
    workspace_fingerprint: str | None
    raw_traj_path: str
    messages_path: str
    judge_path: str
    snapshot_path: str | None
    rubric_ref: str | None
    score: float | None = None
    step_start: int = 0
    step_end: int = -1
    submission: str = ""
    exit_status: str = ""
    regressed_vs_parent: bool = False


@dataclass
class TrajectorySearchResult:
    run_dir: str
    best_node_id: str | None
    raw_trajectory_path: str | None
    slim_trajectory_path: str | None
    patch_path: str | None
    exit_status: str
    submission_chars: int
    finished: bool


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(suffix=".tmp", prefix=f"{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def _make_raw_trajectory(
    *,
    snapshot: dict[str, Any],
    result: dict[str, Any] | None = None,
    info_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    agent_component = snapshot["agent"]
    model_component = snapshot["model"]
    env_component = snapshot["environment"]
    result_meta = (result or {}).get("metadata", {})
    payload = {
        "info": {
            "model_stats": {
                "instance_cost": result_meta.get("cost", agent_component.get("state", {}).get("cost", 0.0)),
                "api_calls": result_meta.get("n_calls", agent_component.get("state", {}).get("n_calls", 0)),
            },
            "config": {
                "agent": copy.deepcopy(agent_component["config"]),
                "agent_type": agent_component["type_path"],
                "model": copy.deepcopy(model_component["config"]),
                "model_type": model_component["type_path"],
                "environment": copy.deepcopy(env_component["config"]),
                "environment_type": env_component["type_path"],
            },
            "mini_version": __version__,
            "exit_status": (result or {}).get("exit_status", ""),
            "submission": (result or {}).get("submission", ""),
        },
        "messages": copy.deepcopy(agent_component.get("state", {}).get("messages", [])),
        "trajectory_format": "mini-swe-agent-1.1",
    }
    if info_extra:
        payload["info"].update(copy.deepcopy(info_extra))
    return payload


def _truncate_middle(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= len(OBSERVATION_TRUNCATION_MARKER):
        return text[:limit]
    remaining = limit - len(OBSERVATION_TRUNCATION_MARKER)
    head = max(1, remaining // 2)
    tail = max(1, remaining - head)
    if head + tail >= len(text):
        return text
    return text[:head] + OBSERVATION_TRUNCATION_MARKER + text[-tail:]


def _render_structured_observation(
    *,
    exception: str | None,
    returncode: str | None,
    output: str | None,
) -> str:
    parts: list[str] = []
    if exception is not None:
        parts.append(f"<exception>{exception}</exception>")
    if returncode is not None:
        parts.append(f"<returncode>{returncode}</returncode>")
    if output is not None:
        parts.append(f"<output>\n{output}</output>")
    return "\n".join(parts)


def _truncate_structured_observation(observation: str) -> str:
    if len(observation) <= MAX_OBSERVATION_CHARS:
        return observation
    exception_match = EXCEPTION_RE.search(observation or "")
    returncode_match = RETURN_CODE_RE.search(observation or "")
    output_match = OUTPUT_RE.search(observation or "")
    if returncode_match is None and output_match is None and exception_match is None:
        return _truncate_middle(observation, MAX_OBSERVATION_CHARS)

    exception = exception_match.group(1) if exception_match is not None else None
    returncode = returncode_match.group(1).strip() if returncode_match is not None else None
    output = output_match.group(1) if output_match is not None else None
    sections = {}
    if exception is not None:
        sections["exception"] = exception
    if output is not None:
        sections["output"] = output

    base_length = len(
        _render_structured_observation(
            exception="" if exception is not None else None,
            returncode=returncode,
            output="" if output is not None else None,
        )
    )

    fixed_sections = {
        name: content for name, content in sections.items() if len(content) <= MIN_OBSERVATION_SECTION_CHARS
    }
    truncatable_sections = {
        name: content for name, content in sections.items() if len(content) > MIN_OBSERVATION_SECTION_CHARS
    }
    fixed_budget = sum(len(content) for content in fixed_sections.values())
    content_budget = MAX_OBSERVATION_CHARS - base_length - fixed_budget

    section_budgets = {name: len(content) for name, content in fixed_sections.items()}
    truncatable_lengths = {name: len(content) for name, content in truncatable_sections.items()}
    if truncatable_lengths and content_budget > 0:
        total_truncatable_length = sum(truncatable_lengths.values())
        shrink_ratio = content_budget / total_truncatable_length
        for name, length in truncatable_lengths.items():
            section_budgets[name] = min(length, max(int(length * shrink_ratio), MIN_OBSERVATION_SECTION_CHARS))

    if exception is not None:
        exception = _truncate_middle(exception, section_budgets.get("exception", len(exception)))
    if output is not None:
        output = _truncate_middle(output, section_budgets.get("output", len(output)))

    truncated = _render_structured_observation(exception=exception, returncode=returncode, output=output)
    return truncated


def _build_step_cards(segment_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards: dict[int, dict[str, Any]] = {}
    for event in segment_events:
        step_index = int(event.get("step_index", -1))
        if step_index < 0:
            continue
        card = cards.setdefault(
            step_index,
            {
                "step_index": step_index,
                "assistant_message": "",
                "commands": [],
                "observation": "",
            },
        )
        if event.get("kind") == "model_response":
            assistant_message = event.get("payload", {}).get("message", {}).get("content", "")
            if assistant_message:
                card["assistant_message"] = assistant_message
            continue
        if event.get("kind") == "environment_action":
            commands = [action.get("command", "") for action in (event.get("payload") or {}).get("actions", [])]
            card["commands"] = commands
            continue
        if event.get("kind") not in {"environment_result", "agent_interrupt"}:
            continue
        messages = (event.get("payload") or {}).get("messages", [])
        if not messages:
            continue
        observation = messages[0].get("content", "")
        card["observation"] = _truncate_structured_observation(observation or "")
    return [cards[index] for index in sorted(cards)]


def _collect_workspace_meta(environment: Any) -> dict[str, Any]:
    cwd = getattr(getattr(environment, "config", None), "cwd", "") or "/testbed"
    command = f"""python - <<'PY'
import hashlib
import json
import os
import pathlib
import subprocess

cwd = {json.dumps(cwd)}
os.chdir(cwd)

def run(cmd):
    completed = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()

payload = {{
    "cwd": cwd,
    "git_repo": False,
    "head_commit": "",
    "changed_files": [],
    "untracked_files": [],
    "status": [],
    "diff_stat": "",
    "current_patch_chars": 0,
    "workspace_fingerprint": None,
}}
git_ok, _, _ = run("git rev-parse --is-inside-work-tree")
payload["git_repo"] = git_ok == 0

if payload["git_repo"]:
    _, head_commit, _ = run("git rev-parse HEAD")
    _, changed_files, _ = run("git diff --name-only --diff-filter=ACMRTUXB")
    _, untracked_files, _ = run("git ls-files --others --exclude-standard")
    _, status, _ = run("git status --porcelain=v1")
    _, diff_stat, _ = run("git diff --stat --compact-summary")
    _, full_diff, _ = run("git diff --no-ext-diff")
    changed = [line for line in changed_files.splitlines() if line.strip()]
    untracked = [line for line in untracked_files.splitlines() if line.strip()]
    status_lines = [line for line in status.splitlines() if line.strip()]
    payload["head_commit"] = head_commit
    payload["changed_files"] = changed
    payload["untracked_files"] = untracked
    payload["status"] = status_lines
    payload["diff_stat"] = diff_stat
    payload["current_patch_chars"] = len(full_diff)
    fingerprints = {{}}
    for rel_path in changed + untracked:
        path = pathlib.Path(rel_path)
        if path.is_file():
            try:
                fingerprints[rel_path] = hashlib.sha256(path.read_bytes()).hexdigest()
            except Exception:
                pass
    payload["workspace_fingerprint"] = hashlib.sha256(
        json.dumps(
            {{
                "head_commit": head_commit,
                "changed_files": changed,
                "untracked_files": untracked,
                "fingerprints": fingerprints,
            }},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

print(json.dumps(payload))
PY"""
    result = environment.execute({"command": command})
    if result.get("returncode") != 0 or not result.get("output", "").strip():
        return {**EMPTY_WORKSPACE_META, "cwd": cwd}
    parsed = extract_json_from_response(result["output"])
    if not isinstance(parsed, dict):
        return {**EMPTY_WORKSPACE_META, "cwd": cwd}
    workspace_meta = {**EMPTY_WORKSPACE_META, **parsed}
    workspace_meta["cwd"] = cwd
    workspace_meta["changed_files"] = list(workspace_meta.get("changed_files", []))
    workspace_meta["untracked_files"] = list(workspace_meta.get("untracked_files", []))
    workspace_meta["status"] = list(workspace_meta.get("status", []))
    workspace_meta["current_patch_chars"] = int(workspace_meta.get("current_patch_chars", 0) or 0)
    return workspace_meta


async def _update_persistent_state(
    *,
    system_prompt: str,
    user_prompt: str,
    previous_state: dict[str, Any],
    evicted_step_cards: list[dict[str, Any]] | None,
    workspace_meta: dict[str, Any],
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if evicted_step_cards is None:
        return copy.deepcopy(previous_state)
    prompt = "\n\n".join(
        [
            PERSISTENT_STATE_UPDATE_PROMPT.strip(),
            f"\n\n## Question:\n System Prompt:\n{system_prompt}\n User Prompt:\n{user_prompt}",
            f"## Previous Persistent State:\n{json.dumps(previous_state, indent=2, ensure_ascii=False)}",
            f"## Evicted Older Trajectory:\n{json.dumps(evicted_step_cards, indent=2, ensure_ascii=False)}",
            f"## Workspace Metadata:\n{json.dumps(workspace_meta, indent=2, ensure_ascii=False)}",
        ]
    )
    for _ in range(EVALUATOR_MAX_RETRIES):
        response = await run_chat_with_route_async(
            "rubric_generation",
            model_name=model_name,
            user_prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            **(model_kwargs or {}),
        )
        parsed = extract_json_from_response(response)
        if isinstance(parsed, dict):
            state = copy.deepcopy(previous_state)
            for key in state:
                if isinstance(parsed.get(key), str):
                    state[key] = parsed[key]
            return state
    return copy.deepcopy(previous_state)


def _convert_generated_rubrics(task: str, payload: dict[str, Any], round_index: int) -> list[RubricRecord]:
    rubrics: list[RubricRecord] = []
    for direction, weight, key in [("positive", 1, "positive_rubrics"), ("negative", -1, "negative_rubrics")]:
        for item in payload.get(key, []) or []:
            title = str(item.get("title", "")).strip()
            description = str(item.get("description", "")).strip()
            scale = {str(score): str(text) for score, text in (item.get("scale") or {}).items()}
            if not title or not description or set(scale) != {"1", "2", "3", "4", "5"}:
                continue
            rubric_id = hashlib.md5(
                json.dumps(
                    {
                        "task": task,
                        "direction": direction,
                        "title": title,
                        "description": description,
                        "scale": scale,
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()[:12]
            rubrics.append(
                RubricRecord(
                    rubric_id=rubric_id,
                    title=title,
                    direction=direction,
                    description=description,
                    scale=scale,
                    weight=weight,
                    source_round=round_index,
                )
            )
    return list({rubric.rubric_id: rubric for rubric in rubrics}.values())


def _build_initial_rubric_bank(task: str) -> list[RubricRecord]:
    payload = {
        "positive_rubrics": [
            {
                "title": "Validation Follow-Through",
                "description": "Runs focused reproduction or validation commands and uses the results to refine the next step instead of treating validation as a box-checking exercise.",
                "scale": {
                    "1": "No meaningful validation or reproduction attempt",
                    "2": "Weak or poorly targeted validation",
                    "3": "Relevant validation with limited follow-through",
                    "4": "Focused validation that materially informs the next step",
                    "5": "Focused validation with strong interpretation and follow-through",
                },
            },
            {
                "title": "Change Precision",
                "description": "Keeps edits narrowly scoped to the implicated logic and avoids unnecessary churn, speculative rewrites, or unrelated modifications.",
                "scale": {
                    "1": "No clear edit strategy or broadly unfocused changes",
                    "2": "Mostly unfocused or weakly scoped changes",
                    "3": "Partially focused but with some unnecessary churn",
                    "4": "Mostly precise, relevant changes",
                    "5": "Highly precise and well-targeted changes only where needed",
                },
            }
        ],
        "negative_rubrics": [
            {
                "title": "Premature Resolution",
                "description": "Acts as if the issue is solved, or submits a patch, without enough evidence, validation, or a coherent causal link from the observed problem to the proposed fix.",
                "scale": {
                    "1": "No premature resolution behavior",
                    "2": "Minor overclaiming",
                    "3": "Noticeable overclaiming or weakly justified completion",
                    "4": "Serious premature resolution behavior",
                    "5": "Severe premature resolution dominating the continuation",
                },
            },
            {
                "title": "Invalid Patch Format",
                "description": "Produces a final patch that is not a valid, directly applicable code patch, such as emitting prose, raw source code, or malformed diff content instead of a legitimate patch.",
                "scale": {
                    "1": "Final patch is a valid, directly applicable patch",
                    "2": "Minor patch-format issues but still mostly usable",
                    "3": "Noticeable patch-format problems creating ambiguity or manual cleanup",
                    "4": "Patch is largely malformed or not directly applicable",
                    "5": "Patch is not a legitimate patch at all",
                },
            }
        ],
    }
    return _convert_generated_rubrics(task, payload, 0)


def _parse_judge_score(response: str) -> int | None:
    parsed = extract_json_from_response(response)
    if not isinstance(parsed, dict):
        return None
    try:
        score = int(parsed.get("score"))
    except (TypeError, ValueError):
        return None
    return score if 1 <= score <= 5 else None


async def _generate_round_rubrics(
    *,
    question: dict[str, str],
    previous_state: dict[str, Any],
    latest_shared_segment: dict[str, Any] | None,
    continuations: list[dict[str, Any]],
    active_bank: list[RubricRecord],
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    round_index: int,
    model_kwargs: dict[str, Any] | None = None,
) -> list[RubricRecord]:
    latest_shared_segment_text = json.dumps(latest_shared_segment, indent=2, ensure_ascii=False) if latest_shared_segment else "None"
    prompt_parts = [
        SWE_TRAJECTORY_RUBRIC_GENERATION_PROMPT.strip(),
        "\n\n## Question:",
        f"System Prompt:\n{question.get('system_prompt', '')}",
        "",
        f"User Prompt:\n{question.get('user_prompt', '')}",
        "",
        "## Previous Persistent State:",
        json.dumps(previous_state, ensure_ascii=False, indent=2),
        "",
        "## Latest Agent Trajectory:",
        latest_shared_segment_text,
        "",
        "## Agent Trajectory Continuations:"
    ]
    for index, continuation in enumerate(continuations, start=1):
        prompt_parts.extend(
            [
                f"## Continuation {index}:",
                json.dumps(
                    {
                        "summary": continuation.get("summary", {}),
                        "trajectory_continuation": continuation.get("trajectory_continuation"),
                        **({"note": continuation["note"]} if continuation.get("note") else {}),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                "",
            ]
        )
    if active_bank:
        prompt_parts.extend(
            [
                "## Existing Rubrics:",
                json.dumps(
                    {
                        "positive_rubrics": [
                            {"title": rubric.title, "description": rubric.description, "scale": rubric.scale}
                            for rubric in active_bank
                            if rubric.direction == "positive"
                        ],
                        "negative_rubrics": [
                            {"title": rubric.title, "description": rubric.description, "scale": rubric.scale}
                            for rubric in active_bank
                            if rubric.direction == "negative"
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            ]
        )
    prompt = "\n".join(prompt_parts)
    task_text = "\n\n".join(part for part in [question.get("system_prompt", ""), question.get("user_prompt", "")] if part)
    for _ in range(EVALUATOR_MAX_RETRIES):
        response = await run_chat_with_route_async(
            "rubric_generation",
            model_name=model_name,
            user_prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            **(model_kwargs or {}),
        )
        parsed = extract_json_from_response(response)
        if not isinstance(parsed, dict):
            continue
        rubrics = _convert_generated_rubrics(task_text, parsed, round_index)
        if rubrics:
            return rubrics
    return []


async def _score_round(
    *,
    question: dict[str, str],
    shared_context: dict[str, Any],
    continuations: list[dict[str, Any]],
    rubrics: list[RubricRecord],
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    model_kwargs: dict[str, Any] | None = None,
) -> tuple[list[list[dict[str, Any]]], dict[str, float], dict[int, str]]:
    if not continuations or not rubrics:
        return [[] for _ in continuations], {}, {}
    calls = []
    mapping: list[tuple[int, RubricRecord]] = []
    question_text = f"System Prompt:\n{question.get('system_prompt', '')}\n\nUser Prompt:\n{question.get('user_prompt', '')}"
    for view_index, continuation in enumerate(continuations):
        response_text = json.dumps(
            {
                "summary": continuation.get("summary", {}),
                "trajectory_continuation": continuation.get("trajectory_continuation"),
                **({"note": continuation["note"]} if continuation.get("note") else {}),
            },
            indent=2,
            ensure_ascii=False,
        )
        for rubric in rubrics:
            criterion = "\n".join(
                [
                    f"Title: {rubric.title}",
                    f"Type: {rubric.direction}",
                    f"Description: {rubric.description}",
                    "Scale:",
                    *[f"{score}: {rubric.scale[str(score)]}" for score in range(1, 6)],
                ]
            )

            async def _judge_single(
                *,
                response_text: str = response_text,
                criterion: str = criterion,
            ) -> tuple[str, int]:
                for _ in range(EVALUATOR_MAX_RETRIES):
                    response = await run_chat_with_route_async(
                        "rubric_judge",
                        model_name=model_name,
                        user_prompt=
                            SWE_TRAJECTORY_RUBRIC_JUDGE_PROMPT.strip() + (
                            f"\n\n## Question:\n{question_text}\n"
                            f"## Previous Persistent State:\n{shared_context.get('previous_persistent_state')}\n"
                            f"## Previous Trajectory:\n{shared_context.get('latest_agent_trajectory')}\n"
                            f"## Continuation Trajectory:\n{response_text}\n"
                            f"## Criterion:\n{criterion}"
                        ),
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"},
                        **(model_kwargs or {}),
                    )
                    score_raw = _parse_judge_score(response)
                    if score_raw is not None:
                        return response, score_raw
                return (
                    json.dumps(
                        {
                            "reasoning": "Fallback to minimum score after repeated judge failures.",
                            "score": 1,
                        },
                        ensure_ascii=False,
                    ),
                    1,
                )

            calls.append(
                _judge_single()
            )
            mapping.append((view_index, rubric))
    responses = await asyncio.gather(*calls, return_exceptions=True)
    per_view_scores: list[list[dict[str, Any]]] = [[] for _ in continuations]
    rubric_values: dict[str, list[float]] = {}
    view_errors: dict[int, str] = {}
    for (view_index, rubric), response in zip(mapping, responses):
        if isinstance(response, Exception):
            view_errors.setdefault(view_index, f"{type(response).__name__}: {response}")
            continue
        judge_response, score_raw = response
        normalized = max(0.0, min(1.0, (score_raw - 1.0) / 4.0))
        record = {
            "rubric_id": rubric.rubric_id,
            "rubric": {
                "rubric_id": rubric.rubric_id,
                "title": rubric.title,
                "direction": rubric.direction,
                "description": rubric.description,
                "scale": copy.deepcopy(rubric.scale),
                "weight": rubric.weight,
                "source_round": rubric.source_round,
            },
            "score_raw": score_raw,
            "score_normalized": normalized,
            "weighted_score": float(rubric.weight) * normalized,
            "judge_response": judge_response,
        }
        per_view_scores[view_index].append(record)
        rubric_values.setdefault(rubric.rubric_id, []).append(normalized)
    variances = {rubric_id: (0.0 if len(values) <= 1 else float(pvariance(values))) for rubric_id, values in rubric_values.items()}
    return per_view_scores, variances, view_errors


async def _score_parent_round(
    *,
    question: dict[str, str],
    shared_context: dict[str, Any],
    rubrics: list[RubricRecord],
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    model_kwargs: dict[str, Any] | None = None,
) -> tuple[list[list[dict[str, Any]]], dict[str, float], dict[int, str]]:
    if not rubrics:
        return [[]], {}, {}
    calls = []
    mapping: list[RubricRecord] = []
    question_text = f"System Prompt:\n{question.get('system_prompt', '')}\n\nUser Prompt:\n{question.get('user_prompt', '')}"
    for rubric in rubrics:
        criterion = "\n".join(
            [
                f"Title: {rubric.title}",
                f"Type: {rubric.direction}",
                f"Description: {rubric.description}",
                "Scale:",
                *[f"{score}: {rubric.scale[str(score)]}" for score in range(1, 6)],
            ]
        )

        async def _judge_single(
            *,
            criterion: str = criterion,
        ) -> tuple[str, int]:
            for _ in range(EVALUATOR_MAX_RETRIES):
                response = await run_chat_with_route_async(
                    "rubric_judge",
                    model_name=model_name,
                    user_prompt=
                        SWE_TRAJECTORY_RUBRIC_JUDGE_PARENT_PROMPT.strip() + (
                        f"\n\n## Question:\n{question_text}\n"
                        f"## Previous Persistent State:\n{shared_context.get('previous_persistent_state')}\n"
                        f"## Agent Trajectory:\n{shared_context.get('latest_agent_trajectory')}\n"
                        f"## Criterion:\n{criterion}"
                    ),
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    **(model_kwargs or {}),
                )
                score_raw = _parse_judge_score(response)
                if score_raw is not None:
                    return response, score_raw
            return (
                json.dumps(
                    {
                        "reasoning": "Fallback to minimum score after repeated judge failures.",
                        "score": 1,
                    },
                    ensure_ascii=False,
                ),
                1,
            )

        calls.append(
            _judge_single()
        )
        mapping.append(rubric)
    responses = await asyncio.gather(*calls, return_exceptions=True)
    per_view_scores: list[list[dict[str, Any]]] = [[]]
    rubric_values: dict[str, list[float]] = {}
    view_errors: dict[int, str] = {}
    for rubric, response in zip(mapping, responses):
        if isinstance(response, Exception):
            view_errors.setdefault(0, f"{type(response).__name__}: {response}")
            continue
        judge_response, score_raw = response
        normalized = max(0.0, min(1.0, (score_raw - 1.0) / 4.0))
        record = {
            "rubric_id": rubric.rubric_id,
            "rubric": {
                "rubric_id": rubric.rubric_id,
                "title": rubric.title,
                "direction": rubric.direction,
                "description": rubric.description,
                "scale": copy.deepcopy(rubric.scale),
                "weight": rubric.weight,
                "source_round": rubric.source_round,
            },
            "score_raw": score_raw,
            "score_normalized": normalized,
            "weighted_score": float(rubric.weight) * normalized,
            "judge_response": judge_response,
        }
        per_view_scores[0].append(record)
        rubric_values.setdefault(rubric.rubric_id, []).append(normalized)
    variances = {rubric_id: (0.0 if len(values) <= 1 else float(pvariance(values))) for rubric_id, values in rubric_values.items()}
    return per_view_scores, variances, view_errors


def _compute_weighted_reward(score_records: list[dict[str, Any]]) -> float:
    numerator = 0.0
    positive_weight = 0.0
    for record in score_records:
        weight = float(record["rubric"]["weight"])
        numerator += weight * float(record["score_normalized"])
        if weight > 0:
            positive_weight += weight
    return numerator / max(positive_weight, 1.0)


def _update_rubric_bank(
    *,
    active_bank: list[RubricRecord],
    inactive_bank: list[RubricRecord],
    generated: list[RubricRecord],
    variances: dict[str, float],
    max_active_rubrics: int,
) -> tuple[list[RubricRecord], list[RubricRecord], list[RubricRecord]]:
    deduped_by_title: dict[str, RubricRecord] = {}
    for rubric in active_bank + generated:
        candidate = RubricRecord(**{**asdict(rubric), "variance": variances.get(rubric.rubric_id, 0.0)})
        title_key = candidate.title.strip().casefold()
        existing = deduped_by_title.get(title_key)
        if existing is None or (candidate.variance or 0.0) > (existing.variance or 0.0) or (
            (candidate.variance or 0.0) == (existing.variance or 0.0) and candidate.source_round > existing.source_round
        ):
            deduped_by_title[title_key] = candidate
    ranked = list(deduped_by_title.values())
    ranked.sort(key=lambda rubric: (rubric.variance or 0.0), reverse=True)
    active = [rubric for rubric in ranked][:max_active_rubrics]
    active_ids = {rubric.rubric_id for rubric in active}
    inactive = [rubric for rubric in ranked + inactive_bank if rubric.rubric_id not in active_ids]
    return active, inactive, ranked


def _docker_commit(executable: str, container_id: str, image_tag: str) -> tuple[str, str]:
    subprocess.run([executable, "commit", container_id, image_tag], check=True, capture_output=True, text=True)
    image_id = subprocess.run(
        [executable, "image", "inspect", image_tag, "--format", "{{.Id}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return image_tag, image_id


class TrajectorySearchRunner:
    def __init__(
        self,
        *,
        instance: dict[str, Any],
        backend: SWEAgentRolloutBackend,
        run_dir: Path,
        policy_model_name: str,
        search_config: SearchConfig | None = None,
        rubric_model_name: str | None = None,
        judge_model_name: str | None = None,
        rubric_model_kwargs: dict[str, Any] | None = None,
        judge_model_kwargs: dict[str, Any] | None = None,
        harness_namespace: str | None = None,
        resume: bool = False,
    ) -> None:
        self.instance = copy.deepcopy(instance)
        self.backend = backend
        self.run_dir = Path(run_dir)
        self.nodes_dir = self.run_dir / "nodes"
        self.rubrics_dir = self.run_dir / "rubrics"
        self.policy_model_name = policy_model_name
        self.rubric_model_name = rubric_model_name or policy_model_name
        self.judge_model_name = judge_model_name or policy_model_name
        self.search_config = search_config or SearchConfig()
        self.rubric_model_kwargs = copy.deepcopy(rubric_model_kwargs or {})
        self.judge_model_kwargs = copy.deepcopy(judge_model_kwargs or {})
        self.harness_namespace = harness_namespace
        self.resume = resume
        self.task = instance["problem_statement"]
        self.task_id = instance["instance_id"]
        self.run_id = f"{self.task_id}-{time.strftime('%Y%m%d-%H%M%S')}"
        self.base_image = str(self.backend.environment_config.get("image", ""))
        self.docker_executable = str(self.backend.environment_config.get("executable", "docker"))
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.node_index_path = self.run_dir / "node_index.jsonl"
        base_image_lookup = subprocess.run(
            [self.docker_executable, "image", "inspect", self.base_image, "--format", "{{.Id}}"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.base_image_id = base_image_lookup.stdout.strip() or None if base_image_lookup.returncode == 0 else None
        self.nodes: dict[str, SearchNode] = {}
        self.frontier_ids: list[str] = []
        self.active_bank: list[RubricRecord] = []
        self.inactive_bank: list[RubricRecord] = []
        self.current_round = 0
        self.system_prompt = ""
        self.user_prompt = ""
        self.best_node_id: str | None = None
        self.finished_node_ids: list[str] = []
        self.image_repository = f"rler-search/{self.task_id.replace('__', '-').lower()}"

    def run(self) -> TrajectorySearchResult:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        try:
            if self.resume and self.manifest_path.exists():
                self._load_manifest()
            else:
                self._initialize_root()            
            while self.current_round < self.search_config.max_rounds:
                if self.frontier_ids and self.nodes[self.frontier_ids[0]].status == "finished":
                    break
                if len(self.frontier_ids) == 0:
                    break
                self.current_round += 1
                self._run_round(self.frontier_ids[0], self.current_round)
                self._save_manifest()
            if self.search_config.calculate_gt_reward:
                self._evaluate_ground_truth_rewards()
            result = self._finalize_outputs()
            self._sweep_checkpoint_images()
            return result
        finally:
            self._sweep_checkpoint_images()
            if self.nodes:
                self._save_manifest()

    def _save_manifest(self) -> None:
        _atomic_write_json(
            self.manifest_path,
            {
                "run_id": self.run_id,
                "instance_id": self.task_id,
                "search_config": asdict(self.search_config),
                "base_image": self.base_image,
                "base_image_id": self.base_image_id,
                "policy_model_name": self.policy_model_name,
                "rubric_model_name": self.rubric_model_name,
                "judge_model_name": self.judge_model_name,
                "frontier_ids": self.frontier_ids,
                "best_node_id": self.best_node_id,
                "finished_node_ids": self.finished_node_ids,
                "current_round": self.current_round,
                "system_prompt": self.system_prompt,
                "user_prompt": self.user_prompt,
                "active_bank": [asdict(rubric) for rubric in self.active_bank],
                "inactive_bank": [asdict(rubric) for rubric in self.inactive_bank],
                "node_ids": sorted(self.nodes),
            },
        )
        lines = [json.dumps(asdict(self.nodes[node_id]), ensure_ascii=False, default=str) for node_id in sorted(self.nodes)]
        self.node_index_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _load_manifest(self) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.run_id = manifest.get("run_id", self.run_id)
        self.frontier_ids = list(manifest.get("frontier_ids", []))
        self.best_node_id = manifest.get("best_node_id")
        self.finished_node_ids = list(manifest.get("finished_node_ids", []))
        self.current_round = int(manifest.get("current_round", 0))
        self.system_prompt = manifest.get("system_prompt", "")
        self.user_prompt = manifest.get("user_prompt", "")
        self.active_bank = [RubricRecord(**rubric) for rubric in manifest.get("active_bank", [])]
        self.inactive_bank = [RubricRecord(**rubric) for rubric in manifest.get("inactive_bank", [])]
        self.nodes = {}
        for node_id in manifest.get("node_ids", []):
            node_path = self.nodes_dir / node_id / "node.json"
            if node_path.exists():
                self.nodes[node_id] = SearchNode(**json.loads(node_path.read_text(encoding="utf-8")))

    def _write_node(
        self,
        node: SearchNode,
        *,
        raw_traj: dict[str, Any],
        messages: dict[str, Any],
        judge: dict[str, Any],
        snapshot: dict[str, Any] | None,
    ) -> None:
        node_dir = self.nodes_dir / node.node_id
        node_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(node_dir / "raw_traj.json", raw_traj)
        _atomic_write_json(node_dir / "messages.json", messages)
        _atomic_write_json(node_dir / "judge.json", judge)
        if snapshot is not None:
            _atomic_write_json(node_dir / "snapshot.json", snapshot)
            node.snapshot_path = str(node_dir / "snapshot.json")
        else:
            node.snapshot_path = None
        _atomic_write_json(node_dir / "node.json", asdict(node))
        self.nodes[node.node_id] = node

    def _dispose_session(self, session: Any) -> None:
        env = getattr(session.agent, "env", None)
        container_id = getattr(env, "container_id", None)
        if container_id:
            subprocess.run([self.docker_executable, "rm", "-f", container_id], check=False, capture_output=True, text=True)
        if env is not None:
            setattr(env, "container_id", None)
            setattr(env, "_owns_container", False)

    def _sweep_checkpoint_images(self) -> None:
        keep_tags = {
            node.checkpoint_image_tag
            for node in self.nodes.values()
            if node.checkpoint_image_tag
            and node.checkpoint_image_tag != self.base_image
            and (node.node_id == self.best_node_id or node.node_id in self.frontier_ids)
        }
        listed = subprocess.run(
            [self.docker_executable, "image", "ls", self.image_repository, "--format", "{{.Repository}}:{{.Tag}}"],
            check=False,
            capture_output=True,
            text=True,
        )
        if listed.returncode != 0:
            return
        for image_tag in {line.strip() for line in listed.stdout.splitlines() if line.strip()} - keep_tags:
            subprocess.run([self.docker_executable, "image", "rm", "-f", image_tag], check=False, capture_output=True, text=True)
            for node in self.nodes.values():
                if node.checkpoint_image_tag != image_tag:
                    continue
                node.checkpoint_image_tag = None
                node.checkpoint_image_id = None
                if node.snapshot_path:
                    snapshot_path = Path(node.snapshot_path)
                    if snapshot_path.exists():
                        snapshot_path.unlink()
                    node.snapshot_path = None
                    _atomic_write_json(snapshot_path.parent / "node.json", asdict(node))

    def _initialize_root(self) -> None:
        spec = RolloutSessionSpec(
            session_id=f"root-{uuid.uuid4().hex}",
            task=self.task,
            task_id=self.task_id,
            sample_index=0,
            policy_ref=self.policy_model_name,
            policy_version=self.policy_model_name,
            dataset_name="swebench",
            ground_truth=self.instance.get("patch"),
            raw_user_query=self.task,
            limits={"step_limit": self.backend.agent_config.get("step_limit", 0)},
            metadata={"template_vars": copy.deepcopy(self.instance)},
        )
        session = self.backend.create_session(spec)
        snapshot = session.snapshot().model_dump(mode="json")
        workspace_meta = _collect_workspace_meta(session.agent.env)
        messages = snapshot["agent"]["state"].get("messages", [])
        self.system_prompt = messages[0].get("content", "") if messages else ""
        self.user_prompt = messages[1].get("content", "") if len(messages) > 1 else self.task
        initial_task_text = "\n\n".join(part for part in [self.system_prompt, self.user_prompt] if part)
        self.active_bank = _build_initial_rubric_bank(initial_task_text or self.task)
        self.inactive_bank = []
        root_node = SearchNode(
            node_id="root",
            parent_id=None,
            round_index=0,
            depth=0,
            session_id=spec.session_id,
            status="frontier",
            checkpoint_image_tag=self.base_image,
            checkpoint_image_id=self.base_image_id,
            workspace_fingerprint=workspace_meta.get("workspace_fingerprint"),
            raw_traj_path=str(self.nodes_dir / "root" / "raw_traj.json"),
            messages_path=str(self.nodes_dir / "root" / "messages.json"),
            judge_path=str(self.nodes_dir / "root" / "judge.json"),
            snapshot_path=str(self.nodes_dir / "root" / "snapshot.json"),
            rubric_ref=None,
            score=0.0,
            step_start=0,
            step_end=-1,
        )
        raw_traj = _make_raw_trajectory(snapshot=snapshot, info_extra={"segment_step_range": [-1, -1]})
        raw_traj["messages"] = []
        self._write_node(
            root_node,
            raw_traj=raw_traj,
            messages=build_slim_trajectory(raw_traj, model_name=self.policy_model_name),
            judge={
                "persistent_state": copy.deepcopy(EMPTY_PERSISTENT_STATE),
                "recent_segments": [],
                "workspace_meta": workspace_meta,
                "rubric_round": {
                    "generated": [],
                    "active_bank_before": [],
                    "active_bank_after": [asdict(rubric) for rubric in self.active_bank],
                    "inactive_bank_after": [],
                    "variance_by_rubric": {},
                },
                "scores": [],
                "overall_reward": 0.0,
                "baseline_parent_score": None,
                "regressed_vs_parent": False,
            },
            snapshot={
                **copy.deepcopy(snapshot),
                "environment": {
                    **copy.deepcopy(snapshot["environment"]),
                    "config": {
                        **copy.deepcopy(snapshot["environment"]["config"]),
                        "image": self.base_image,
                    },
                    "state": {"owns_container": True},
                },
                "metadata": {
                    **copy.deepcopy(snapshot.get("metadata", {})),
                    "checkpoint_image_id": self.base_image_id,
                    "checkpoint_image_tag": self.base_image,
                },
            },
        )
        self.frontier_ids = ["root"]
        self.best_node_id = "root"
        self.finished_node_ids = []
        self._save_manifest()
        self._dispose_session(session)

    async def _prepare_round_judging(
        self,
        *,
        parent_node: SearchNode,
        parent_judge: dict[str, Any],
        branch_records: list[dict[str, Any]],
        round_index: int,
        compare_parent: bool,
    ) -> dict[str, Any]:
        parent_state = parent_judge["persistent_state"]
        parent_recent_segments = copy.deepcopy(parent_judge.get("recent_segments", []))
        latest_shared_segment = copy.deepcopy(parent_recent_segments[-1]) if parent_recent_segments else None
        updated_parent_state = copy.deepcopy(parent_state)
        evicted_step_cards = None
        if len(parent_recent_segments) >= 2:
            evicted_step_cards = copy.deepcopy(parent_recent_segments[0].get("step_cards", []))
        if evicted_step_cards:
            evicted_workspace_meta = copy.deepcopy(EMPTY_WORKSPACE_META)
            if parent_node.parent_id:
                grandparent_node = self.nodes.get(parent_node.parent_id)
                if grandparent_node and Path(grandparent_node.judge_path).exists():
                    grandparent_judge = json.loads(Path(grandparent_node.judge_path).read_text(encoding="utf-8"))
                    evicted_workspace_meta = copy.deepcopy(grandparent_judge.get("workspace_meta", evicted_workspace_meta))
            try:
                updated_parent_state = await _update_persistent_state(
                    system_prompt=self.system_prompt,
                    user_prompt=self.task,
                    previous_state=parent_state,
                    evicted_step_cards=evicted_step_cards,
                    workspace_meta=evicted_workspace_meta,
                    model_name=self.rubric_model_name,
                    temperature=self.search_config.rubric_temperature,
                    top_p=self.search_config.rubric_top_p,
                    max_tokens=self.search_config.rubric_max_tokens,
                    model_kwargs=self.rubric_model_kwargs,
                )
            except Exception:
                updated_parent_state = copy.deepcopy(parent_state)
        question = {"system_prompt": self.system_prompt, "user_prompt": self.task}
        shared_context = {
            "previous_persistent_state": updated_parent_state,
            "latest_agent_trajectory": latest_shared_segment,
        }
        continuations = []
        branch_indices: list[int] = []
        judge_errors: dict[int, str] = {}
        for index, branch in enumerate(branch_records):
            branch["persistent_state"] = copy.deepcopy(updated_parent_state)
            step_cards = branch["recent_segments"][-1].get("step_cards", []) if branch["recent_segments"] else []
            workspace_meta = branch.get("workspace_meta", {})
            result = branch.get("result", {})
            branch["continuation_view"] = {
                "summary": {
                    "step_count": len(step_cards),
                    "changed_files": list(workspace_meta.get("changed_files", []))[:8],
                    "untracked_files": list(workspace_meta.get("untracked_files", []))[:8],
                    "diff_stat": workspace_meta.get("diff_stat", ""),
                    "current_patch_chars": int(workspace_meta.get("current_patch_chars", 0) or 0),
                    "result_status": result.get("status", ""),
                    "exit_status": result.get("exit_status", ""),
                    "submission_chars": len(result.get("submission", "") or ""),
                },
                "trajectory_continuation": copy.deepcopy(branch["recent_segments"][-1]) if branch["recent_segments"] else None,
            }
            continuations.append(branch["continuation_view"])
            branch_indices.append(index)
        if not continuations:
            return {
                "generated": [],
                "child_scores": [[] for _ in branch_records],
                "variances": {},
                "parent_score_records": None,
                "active_after": self.active_bank,
                "inactive_after": self.inactive_bank,
                "round_bank": list(self.active_bank) + list(self.inactive_bank),
                "judge_errors": judge_errors,
            }
        try:
            generated = await _generate_round_rubrics(
                question=question,
                previous_state=updated_parent_state,
                latest_shared_segment=latest_shared_segment,
                continuations=continuations,
                active_bank=self.active_bank + self.inactive_bank,
                model_name=self.rubric_model_name,
                temperature=self.search_config.rubric_temperature,
                top_p=self.search_config.rubric_top_p,
                max_tokens=self.search_config.rubric_max_tokens,
                round_index=round_index,
                model_kwargs=self.rubric_model_kwargs,
            )
        except Exception as exc:
            generated = []
        scoring_rubrics = list({rubric.rubric_id: rubric for rubric in self.active_bank + generated}.values())
        if scoring_rubrics:
            scored_continuations, variances, continuation_errors = await _score_round(
                question=question,
                shared_context=shared_context,
                continuations=continuations,
                rubrics=scoring_rubrics,
                model_name=self.judge_model_name,
                temperature=self.search_config.judge_temperature,
                top_p=self.search_config.judge_top_p,
                max_tokens=self.search_config.judge_max_tokens,
                model_kwargs=self.judge_model_kwargs,
            )
        else:
            scored_continuations = [[] for _ in continuations]
            variances = {}
            continuation_errors = {}
        child_scores = [[] for _ in branch_records]
        for local_index, branch_index in enumerate(branch_indices):
            if local_index in continuation_errors:
                judge_errors.setdefault(branch_index, continuation_errors[local_index])
                continue
            child_scores[branch_index] = scored_continuations[local_index]
        parent_score_records = None
        if compare_parent and scoring_rubrics:
            baseline_scores, _, baseline_errors = await _score_parent_round(
                question=question,
                shared_context=shared_context,
                rubrics=scoring_rubrics,
                model_name=self.judge_model_name,
                temperature=self.search_config.judge_temperature,
                top_p=self.search_config.judge_top_p,
                max_tokens=self.search_config.judge_max_tokens,
                model_kwargs=self.judge_model_kwargs,
            )
            parent_score_records = None if baseline_errors else baseline_scores[0]
        active_after, inactive_after, round_bank = _update_rubric_bank(
            active_bank=self.active_bank,
            inactive_bank=self.inactive_bank,
            generated=generated,
            variances=variances,
            max_active_rubrics=self.search_config.max_active_rubrics,
        )
        active_ids = {rubric.rubric_id for rubric in active_after}
        child_scores = [
            [record for record in score_records if record["rubric_id"] in active_ids]
            for score_records in child_scores
        ]
        if parent_score_records is not None:
            parent_score_records = [record for record in parent_score_records if record["rubric_id"] in active_ids]
        return {
            "generated": generated,
            "child_scores": child_scores,
            "variances": variances,
            "parent_score_records": parent_score_records,
            "active_after": active_after,
            "inactive_after": inactive_after,
            "round_bank": round_bank,
            "judge_errors": judge_errors,
        }

    def _run_round(self, parent_id: str, round_index: int) -> None:
        parent_node = self.nodes[parent_id]
        if not parent_node.snapshot_path:
            raise RuntimeError(f"Node {parent_id} does not have a restorable snapshot")
        parent_snapshot = json.loads(Path(parent_node.snapshot_path).read_text(encoding="utf-8"))
        parent_judge = json.loads(Path(parent_node.judge_path).read_text(encoding="utf-8"))
        previous_frontier = list(self.frontier_ids)
        branch_records: list[dict[str, Any]] = []

        for sample_index in range(self.search_config.m + 2 if parent_id == "root" else self.search_config.m):
            node_id = f"node-r{round_index:03d}-s{sample_index:02d}-{uuid.uuid4().hex[:6]}"
            session = None
            try:
                resumed_snapshot = copy.deepcopy(parent_snapshot)
                resumed_snapshot["session_id"] = f"{node_id}-session"
                resumed_snapshot["spec"]["session_id"] = resumed_snapshot["session_id"]
                for index, event in enumerate(resumed_snapshot.get("metadata", {}).get("events", [])):
                    event["session_id"] = resumed_snapshot["session_id"]
                    event["event_id"] = f"{resumed_snapshot['session_id']}:{index}"
                for turn in resumed_snapshot.get("metadata", {}).get("model_turns", []):
                    turn["session_id"] = resumed_snapshot["session_id"]
                session = self.backend.resume_session(
                    RolloutSnapshot(**resumed_snapshot)
                )
                result = session.run_until_pause(max_steps=self.search_config.k).model_dump(mode="json")
                snapshot_after = session.snapshot().model_dump(mode="json")
                workspace_meta = _collect_workspace_meta(session.agent.env)
                before_event_count = len(parent_snapshot.get("metadata", {}).get("events", []))
                before_message_count = len(parent_snapshot.get("agent", {}).get("state", {}).get("messages", []))
                segment_events = copy.deepcopy(snapshot_after.get("metadata", {}).get("events", [])[before_event_count:])
                segment_messages = copy.deepcopy(
                    snapshot_after.get("agent", {}).get("state", {}).get("messages", [])[before_message_count:]
                )
                step_cards = _build_step_cards(segment_events)
                step_start = step_cards[0]["step_index"] if step_cards else parent_node.step_end + 1
                step_end = step_cards[-1]["step_index"] if step_cards else parent_node.step_end
                segment_raw = _make_raw_trajectory(
                    snapshot=snapshot_after,
                    result=result,
                    info_extra={"segment_step_range": [step_start, step_end]},
                )
                segment_raw["messages"] = segment_messages
                recent_segments = copy.deepcopy(parent_judge["recent_segments"])
                recent_segments.append({"step_cards": copy.deepcopy(step_cards), "segment_step_range": [step_start, step_end]})
                if len(recent_segments) > 2:
                    recent_segments = recent_segments[-2:]
                branch_records.append(
                    {
                        "node_id": node_id,
                        "session": session,
                        "result": result,
                        "snapshot_after": snapshot_after,
                        "workspace_meta": workspace_meta,
                        "segment_raw": segment_raw,
                        "segment_messages": build_slim_trajectory(segment_raw, model_name=self.policy_model_name),
                        "step_start": step_start,
                        "step_end": step_end,
                        "recent_segments": recent_segments,
                    }
                )
            except Exception as exc:
                if session is not None:
                    self._dispose_session(session)
                node = SearchNode(
                    node_id=node_id,
                    parent_id=parent_id,
                    round_index=round_index,
                    depth=parent_node.depth + 1,
                    session_id=f"{node_id}-session",
                    status="dropped",
                    checkpoint_image_tag=None,
                    checkpoint_image_id=None,
                    workspace_fingerprint=None,
                    raw_traj_path=str(self.nodes_dir / node_id / "raw_traj.json"),
                    messages_path=str(self.nodes_dir / node_id / "messages.json"),
                    judge_path=str(self.nodes_dir / node_id / "judge.json"),
                    snapshot_path=None,
                    rubric_ref=None,
                    score=float("-inf"),
                    step_start=parent_node.step_end + 1,
                    step_end=parent_node.step_end,
                    exit_status=type(exc).__name__,
                )
                self._write_node(
                    node,
                    raw_traj={"error": str(exc), "trajectory_format": "mini-swe-agent-1.1", "messages": []},
                    messages={"messages": [], "error": str(exc)},
                    judge={
                        "persistent_state": copy.deepcopy(parent_judge.get("persistent_state", EMPTY_PERSISTENT_STATE)),
                        "recent_segments": copy.deepcopy(parent_judge.get("recent_segments", [])),
                        "workspace_meta": {**EMPTY_WORKSPACE_META},
                        "rubric_round": {
                            "generated": [],
                            "active_bank_before": [asdict(rubric) for rubric in self.active_bank],
                            "active_bank_after": [asdict(rubric) for rubric in self.active_bank],
                            "inactive_bank_after": [asdict(rubric) for rubric in self.inactive_bank],
                            "variance_by_rubric": {},
                        },
                        "scores": [],
                        "overall_reward": float("-inf"),
                        "baseline_parent_score": None,
                        "regressed_vs_parent": False,
                        "error": str(exc),
                    }
                    | ({"ground_truth_reward": None} if self.search_config.calculate_gt_reward else {}),
                    snapshot=None,
                )

        if not branch_records:
            return

        try:
            judged = asyncio.run(
                self._prepare_round_judging(
                    parent_node=parent_node,
                    parent_judge=parent_judge,
                    branch_records=branch_records,
                    round_index=round_index,
                    compare_parent=parent_id != "root",
                )
            )
        except Exception as exc:
            for branch in branch_records:
                node = SearchNode(
                    node_id=branch["node_id"],
                    parent_id=parent_id,
                    round_index=round_index,
                    depth=parent_node.depth + 1,
                    session_id=branch["snapshot_after"]["session_id"],
                    status="dropped",
                    checkpoint_image_tag=None,
                    checkpoint_image_id=None,
                    workspace_fingerprint=branch["workspace_meta"].get("workspace_fingerprint"),
                    raw_traj_path=str(self.nodes_dir / branch["node_id"] / "raw_traj.json"),
                    messages_path=str(self.nodes_dir / branch["node_id"] / "messages.json"),
                    judge_path=str(self.nodes_dir / branch["node_id"] / "judge.json"),
                    snapshot_path=None,
                    rubric_ref=None,
                    score=float("-inf"),
                    step_start=branch["step_start"],
                    step_end=branch["step_end"],
                    exit_status=type(exc).__name__,
                )
                self._write_node(
                    node,
                    raw_traj=branch["segment_raw"],
                    messages=branch["segment_messages"],
                    judge={
                        "persistent_state": copy.deepcopy(parent_judge.get("persistent_state", EMPTY_PERSISTENT_STATE)),
                        "recent_segments": branch["recent_segments"],
                        "workspace_meta": branch["workspace_meta"],
                        "rubric_round": {
                            "generated": [],
                            "active_bank_before": [asdict(rubric) for rubric in self.active_bank],
                            "active_bank_after": [asdict(rubric) for rubric in self.active_bank],
                            "inactive_bank_after": [asdict(rubric) for rubric in self.inactive_bank],
                            "variance_by_rubric": {},
                        },
                        "scores": [],
                        "overall_reward": float("-inf"),
                        "baseline_parent_score": None,
                        "regressed_vs_parent": False,
                        "error": str(exc),
                    }
                    | ({"ground_truth_reward": None} if self.search_config.calculate_gt_reward else {}),
                    snapshot=None,
                )
                self._dispose_session(branch["session"])
            return
        parent_baseline_score = (
            _compute_weighted_reward(judged["parent_score_records"])
            if judged["parent_score_records"] is not None
            else 0.0
        )
        node_round_records = []
        valid_branches: list[dict[str, Any]] = []
        for index, (branch, score_records) in enumerate(zip(branch_records, judged["child_scores"])):
            judge_error = judged["judge_errors"].get(index)
            branch["judge_error"] = judge_error
            for record in score_records:
                record["variance"] = judged["variances"].get(record["rubric_id"], 0.0)
            branch["score_records"] = [] if judge_error else score_records
            branch["reward"] = float("-inf") if judge_error else _compute_weighted_reward(score_records)
            if not judge_error:
                valid_branches.append(branch)
            node_round_records.append(
                {
                    "node_id": branch["node_id"],
                    "score": branch["reward"],
                    "finished": branch["result"]["status"] == "finished",
                    "judge_error": judge_error,
                }
            )
        valid_branches.sort(key=lambda branch: (branch["reward"], branch["result"]["status"] == "finished"), reverse=True)
        frontier_branches = (
            valid_branches
            if parent_id == "root"
            else [branch for branch in valid_branches if branch["reward"] >= parent_baseline_score + self.search_config.regression_margin]
        )
        frontier_branches.sort(key=lambda branch: (branch["reward"], branch["result"]["status"] == "finished"), reverse=True)
        regressed = parent_id != "root" and not frontier_branches

        top_child_ids = [branch["node_id"] for branch in frontier_branches[: self.search_config.p]]
        candidate_frontier_ids = list(top_child_ids)
        candidate_frontier_ids.extend(
            node_id for node_id in previous_frontier if node_id != parent_id and node_id not in candidate_frontier_ids
        )
        kept_child_ids = set(top_child_ids)

        round_rubric_path = self.rubrics_dir / f"round_{round_index:03d}.json"
        round_payload = {
            "round_index": round_index,
            "parent_id": parent_id,
            "generated": [asdict(rubric) for rubric in judged["generated"]],
            "active_bank_before": [asdict(rubric) for rubric in self.active_bank],
            "active_bank_after": [asdict(rubric) for rubric in judged["active_after"]],
            "inactive_bank_after": [asdict(rubric) for rubric in judged["inactive_after"]],
            "variance_by_rubric": judged["variances"],
            "parent_baseline_score": parent_baseline_score,
            "regressed": regressed,
            "node_scores": node_round_records,
            "judge_errors": {branch["node_id"]: branch["judge_error"] for branch in branch_records if branch["judge_error"]},
        }
        _atomic_write_json(round_rubric_path, round_payload)
        self.active_bank = judged["active_after"]
        self.inactive_bank = judged["inactive_after"]

        for branch in branch_records:
            keep_snapshot = branch["node_id"] in kept_child_ids
            image_tag = None
            image_id = None
            if keep_snapshot:
                image_tag = f"{self.image_repository}:round-{round_index:03d}-{branch['node_id'][-6:]}"
                image_tag, image_id = _docker_commit(
                    self.docker_executable,
                    getattr(branch["session"].agent.env, "container_id", None),
                    image_tag,
                )
            node = SearchNode(
                node_id=branch["node_id"],
                parent_id=parent_id,
                round_index=round_index,
                depth=parent_node.depth + 1,
                session_id=branch["snapshot_after"]["session_id"],
                status="finished" if branch["result"]["status"] == "finished" and keep_snapshot else ("frontier" if keep_snapshot else "dropped"),
                checkpoint_image_tag=image_tag,
                checkpoint_image_id=image_id,
                workspace_fingerprint=branch["workspace_meta"].get("workspace_fingerprint"),
                raw_traj_path=str(self.nodes_dir / branch["node_id"] / "raw_traj.json"),
                messages_path=str(self.nodes_dir / branch["node_id"] / "messages.json"),
                judge_path=str(self.nodes_dir / branch["node_id"] / "judge.json"),
                snapshot_path=str(self.nodes_dir / branch["node_id"] / "snapshot.json") if keep_snapshot else None,
                rubric_ref=str(round_rubric_path),
                score=branch["reward"],
                step_start=branch["step_start"],
                step_end=branch["step_end"],
                submission=branch["result"].get("submission", ""),
                exit_status=branch["result"].get("exit_status", ""),
                regressed_vs_parent=regressed and bool(valid_branches) and branch["node_id"] == valid_branches[0]["node_id"],
            )
            self._write_node(
                node,
                raw_traj=branch["segment_raw"],
                messages=branch["segment_messages"],
                judge={
                    "persistent_state": branch.get("persistent_state", copy.deepcopy(parent_judge.get("persistent_state", EMPTY_PERSISTENT_STATE))),
                    "recent_segments": branch["recent_segments"],
                    "workspace_meta": branch["workspace_meta"],
                    "rubric_round": {
                        "generated": [asdict(rubric) for rubric in judged["generated"]],
                        "active_bank_before": round_payload["active_bank_before"],
                        "active_bank_after": [asdict(rubric) for rubric in judged["active_after"]],
                        "inactive_bank_after": [asdict(rubric) for rubric in judged["inactive_after"]],
                        "variance_by_rubric": judged["variances"],
                    },
                    "scores": branch["score_records"],
                    "overall_reward": branch["reward"],
                    "baseline_parent_score": parent_baseline_score,
                    "regressed_vs_parent": node.regressed_vs_parent,
                    "error": branch["judge_error"],
                }
                | ({"ground_truth_reward": None} if self.search_config.calculate_gt_reward else {}),
                snapshot=(
                    {
                        **copy.deepcopy(branch["snapshot_after"]),
                        "environment": {
                            **copy.deepcopy(branch["snapshot_after"]["environment"]),
                            "config": {
                                **copy.deepcopy(branch["snapshot_after"]["environment"]["config"]),
                                "image": image_tag,
                            },
                            "state": {"owns_container": True},
                        },
                        "metadata": {
                            **copy.deepcopy(branch["snapshot_after"].get("metadata", {})),
                            "checkpoint_image_id": image_id,
                            "checkpoint_image_tag": image_tag,
                        },
                    }
                    if keep_snapshot
                    else None
                ),
            )
            if self.search_config.calculate_gt_reward:
                node_dir = Path(node.raw_traj_path).parent
                terminal_result = branch["result"]
                terminal_snapshot = branch["snapshot_after"]
                if branch["result"]["status"] != "finished":
                    model_config = getattr(getattr(branch["session"].agent, "model", None), "config", None)
                    model_kwargs = getattr(model_config, "model_kwargs", None)
                    if isinstance(model_kwargs, dict):
                        model_kwargs["temperature"] = 0.0
                        model_kwargs["top_p"] = 1.0
                    terminal_result = branch["session"].run_until_pause(max_steps=None).model_dump(mode="json")
                    terminal_snapshot = branch["session"].snapshot().model_dump(mode="json")
                terminal_raw = _make_raw_trajectory(
                    snapshot=terminal_snapshot,
                    result=terminal_result,
                    info_extra={"terminal_rollout_from_node_id": node.node_id},
                )
                _atomic_write_json(node_dir / "terminal_raw_traj.json", terminal_raw)
                _atomic_write_json(
                    node_dir / "terminal_messages.json",
                    build_slim_trajectory(terminal_raw, model_name=self.policy_model_name),
                )
                _atomic_write_json(
                    node_dir / "terminal_patch.json",
                    {
                        self.task_id: {
                            "model_name_or_path": self.policy_model_name,
                            "instance_id": self.task_id,
                            "model_patch": terminal_result.get("submission", "") or "",
                        }
                    },
                )
            if node.status == "finished":
                self.finished_node_ids.append(node.node_id)
            self._dispose_session(branch["session"])

        self.finished_node_ids = sorted(set(self.finished_node_ids))
        self.frontier_ids = [
            node_id
            for node_id in candidate_frontier_ids
            if node_id in self.nodes and self.nodes[node_id].status in {"frontier", "finished"}
        ]

        for node_id in previous_frontier:
            if node_id in self.nodes and self.nodes[node_id].status == "frontier" and node_id not in self.frontier_ids:
                self.nodes[node_id].status = "archived"
                _atomic_write_json(Path(self.nodes[node_id].raw_traj_path).parent / "node.json", asdict(self.nodes[node_id]))

        if self.frontier_ids:
            self.best_node_id = self.frontier_ids[0]
        self._sweep_checkpoint_images()

    def _evaluate_ground_truth_rewards(self) -> None:
        patches_by_node_id: dict[str, str] = {}
        judge_payloads: dict[str, dict[str, Any]] = {}
        judge_paths: dict[str, Path] = {}
        for node_id, node in self.nodes.items():
            if node_id == "root":
                continue
            judge_path = Path(node.judge_path)
            if not judge_path.exists():
                continue
            judge_payload = json.loads(judge_path.read_text(encoding="utf-8"))
            judge_payload["ground_truth_reward"] = None
            judge_payloads[node_id] = judge_payload
            judge_paths[node_id] = judge_path
            terminal_patch_path = Path(node.raw_traj_path).parent / "terminal_patch.json"
            if not terminal_patch_path.exists():
                continue
            try:
                patch_payload = json.loads(terminal_patch_path.read_text(encoding="utf-8"))
                patches_by_node_id[node_id] = patch_payload[self.task_id]["model_patch"] or ""
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

        rewards_by_node_id = evaluate_swebench_instance_patches(
            instance=self.instance,
            patches_by_key=patches_by_node_id,
            model_name=self.policy_model_name,
            max_workers=self.search_config.gt_reward_workers,
            namespace=self.harness_namespace,
            work_dir=self.run_dir,
        )
        for node_id, judge_payload in judge_payloads.items():
            if node_id in rewards_by_node_id:
                judge_payload["ground_truth_reward"] = rewards_by_node_id[node_id]
            _atomic_write_json(judge_paths[node_id], judge_payload)

    def _finalize_outputs(self) -> TrajectorySearchResult:
        if self.finished_node_ids:
            final_node_id = max(
                (self.nodes[node_id] for node_id in self.finished_node_ids if node_id in self.nodes),
                key=lambda node: float(node.score or 0.0),
            ).node_id
        elif self.best_node_id is not None:
            final_node_id = self.best_node_id
        elif self.frontier_ids:
            final_node_id = self.frontier_ids[0]
        else:
            return TrajectorySearchResult(
                run_dir=str(self.run_dir),
                best_node_id=None,
                raw_trajectory_path=None,
                slim_trajectory_path=None,
                patch_path=None,
                exit_status="",
                submission_chars=0,
                finished=False,
            )

        final_node = self.nodes[final_node_id]
        if not final_node.snapshot_path:
            raise RuntimeError(f"Node {final_node_id} does not have a restorable snapshot")
        snapshot = json.loads(Path(final_node.snapshot_path).read_text(encoding="utf-8"))
        raw_traj = _make_raw_trajectory(
            snapshot=snapshot,
            result={"exit_status": final_node.exit_status, "submission": final_node.submission, "metadata": {}},
            info_extra={"search": {"best_node_id": final_node_id, "current_round": self.current_round, "frontier_ids": self.frontier_ids}},
        )
        raw_traj_path = self.run_dir / "raw_traj.json"
        messages_path = self.run_dir / "messages.json"
        patch_path = self.run_dir / "model_patch.json"
        _atomic_write_json(raw_traj_path, raw_traj)
        _atomic_write_json(messages_path, build_slim_trajectory(raw_traj, model_name=self.policy_model_name))
        _atomic_write_json(
            patch_path,
            {
                self.task_id: {
                    "model_name_or_path": self.policy_model_name,
                    "instance_id": self.task_id,
                    "model_patch": final_node.submission,
                }
            },
        )
        self.best_node_id = final_node_id
        if final_node.status == "finished":
            self.frontier_ids = [final_node_id]
        self._save_manifest()
        return TrajectorySearchResult(
            run_dir=str(self.run_dir),
            best_node_id=final_node_id,
            raw_trajectory_path=str(raw_traj_path),
            slim_trajectory_path=str(messages_path),
            patch_path=str(patch_path),
            exit_status=final_node.exit_status,
            submission_chars=len(final_node.submission or ""),
            finished=final_node.status == "finished",
        )
