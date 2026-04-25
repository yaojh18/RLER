#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
import shutil
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from statistics import pvariance
from typing import Any, Sequence

from agent_rl import RolloutSessionSpec
from dr_agent.utils import launch_vllm_server_handle
from agent_rl.run_utils import (
    ModelRouteConfig,
    clear_model_routes,
    configure_model_route,
    extract_json_from_response,
    run_chat_with_route_async,
)

from swe_agent.rl_backend import SWEAgentRolloutBackend
from swe_agent.run.benchmarks.swebench import (
    DATASET_MAPPING,
    build_swebench_config,
    get_swebench_docker_image_name,
    load_swebench_instances,
)
from swe_agent.run.run_swe_agent import (
    DEFAULT_COMPLETION_MAX_TOKENS,
    DEFAULT_ENV_TIMEOUT,
    DEFAULT_EVAL_TIMEOUT,
    DEFAULT_LOG_ROOT,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_MODEL_CLASS,
    DEFAULT_PULL_TIMEOUT,
    DEFAULT_SERVE_MODEL,
    DEFAULT_SPLIT,
    DEFAULT_STEP_LIMIT,
    DEFAULT_SUBSET,
    DEFAULT_VLLM_PORT,
    ParseInstanceIds,
    build_slim_trajectory,
    choose_gpus,
    find_free_port,
    infer_litellm_api_env,
    run_harness_evaluation,
    tee_console,
    temporary_env,
    terminate_process,
    write_failure_artifacts,
)
from swe_agent.run.run_swe_agent import SWE_AGENT_TEXTBASED_CONFIG
from swe_agent.trajectory_search import (
    EMPTY_PERSISTENT_STATE,
    EVALUATOR_MAX_RETRIES,
    JUDGE_RESPONSE_FORMAT,
    PERSISTENT_STATE_RESPONSE_FORMAT,
    RubricRecord,
    RUBRIC_GENERATION_RESPONSE_FORMAT,
    SearchConfig,
    _build_initial_rubric_bank,
    _build_step_cards,
    _collect_workspace_meta,
    _compute_weighted_reward,
    _make_raw_trajectory,
    _parse_judge_score,
    _convert_generated_rubric,
    _update_rubric_bank,
)


DEFAULT_AGGREGATE_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "aggregate_outputs"

AGGREGATE_TRAJECTORY_SUMMARY_PROMPT = """
You are an expert evaluator compressing a complete software-debugging trajectory into a compact view for later comparison.

## Task
Summarize the full agent trajectory from start to finish so another evaluator can compare complete trajectories without reading the raw transcript.

## Required Sections
Return exactly these 8 top-level string fields:
- **current_state**: Where the trajectory currently ended, what remained pending, and the immediate next step implied by the trajectory. Always keep this section up to date with the latest point reached.
- **task_specification**: What the user asked the agent to build or fix, plus important constraints and design context.
- **files_and_functions**: Important files, functions, classes, modules, and why they matter. Include concrete paths and identifiers.
- **errors_and_corrections**: Errors encountered, failed attempts, rejected hypotheses, and how they were corrected.
- **codebase_and_system_documentation**: Important system components, architecture, interfaces, workflows, and how they fit together.
- **learnings**: Actionable lessons about what worked, what did not, and what to avoid. Do not repeat material already captured elsewhere.
- **key_results**: Exact or near-exact outputs worth preserving, such as the final patch direction, test results, or the final answer.
- **worklog**: Very terse step-by-step record of what the trajectory attempted and completed.

## Core Guidelines
- Use only evidence visible in the provided trajectory and metadata.
- Be specific and information-dense. Include concrete commands, file paths, function names, test names, errors, validation outcomes, and technical conclusions when useful.
- Focus on actionable context that would help someone understand or compare the trajectory.
- It is OK to leave a section blank if there is no substantial information for it. Do not add filler such as "No info yet".
- Keep each section under 400 words. If a section becomes too long, remove less important details while preserving the most critical information.
- Prefer compression over copying long logs verbatim.
- Preserve whether the agent validated the fix, changed code, submitted a patch, or stopped early.
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
1. **Question**: Original system and user prompt
2. **Trajectory Metadata**: Compact metadata about the full run
3. **Full Agent Trajectory**: The full trajectory represented as step cards

Return only the JSON object.
"""

AGGREGATE_RUBRIC_GENERATION_PROMPT = """
You are an expert evaluator generating evaluation rubrics to compare summarized software-debugging trajectories.

## Task
Identify the most discriminative criteria that distinguish high-quality from low-quality summarized trajectories.

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
- Each rubric must distinguish between otherwise similar trajectories
- Exclude generic criteria applying equally to all trajectories

### 2. Novelty & Non-Redundancy
With existing rubrics:
- Never duplicate overlapping rubrics in meaning/scope
- Identify uncovered quality dimensions

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

## Analysis Process
1. Group continuations by quality level
2. Find factors separating higher/lower clusters
3. Select criteria with highest discriminative value

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
1. **Question**: Original system and user prompt
2. **Trajectory Summaries**: Multiple compressed views of full trajectories

## Critical Reminders
- Each rubric must distinguish between the actual provided trajectories
- Exclude rubrics applying equally to all trajectories
- Prefer empty lists over redundancy when existing rubrics are comprehensive
- Focus on observable, objective, actionable criteria
- Quality over quantity: 2 excellent rubrics > 5 mediocre ones
- Do not return empty lists when there are visible differences in diagnostic strategy, reproduction attempts, validation attempts, or targeting of relevant files
- Do not copy the full question or trajectory text into the output JSON. Return rubric objects only

Generate only the most impactful, non-redundant rubrics revealing meaningful quality differences.
"""

AGGREGATE_RUBRIC_JUDGE_PROMPT = """
You are an expert evaluator scoring one full trajectory summary against one rubric.

## Task
Evaluate the provided trajectory summary using only the provided criterion and question context.

## Core Guidelines
- Judge only the specified criterion
- Score the trajectory summary itself, not the bug in the abstract
- Use the rubric's scale exactly as being required. For negative rubrics, the scale is inverted (e.g. worst case should receive 5 while best case should receive 1)
- Use only visible evidence from the question and trajectory summary
- Brief grounded reasoning is allowed

## Output Format
```json
{
  "reasoning": "<brief grounded explanation>",
  "score": <integer from 1 to 5>
}
```

## Inputs
1. **Question**: Original system and user prompt
2. **Trajectory Summary**: One compressed view of a full trajectory
3. **Criterion**: The specific rubric to evaluate

Return only the JSON object.
"""


async def _summarize_aggregate_trajectory(
    *,
    question: dict[str, str],
    step_cards: list[dict[str, Any]],
    trajectory_metadata: dict[str, Any],
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    user_prompt = "\n".join(
        [
            "## Question:",
            f"System Prompt:\n{question.get('system_prompt', '')}",
            "",
            f"User Prompt:\n{question.get('user_prompt', '')}",
            "",
            "## Trajectory Metadata:",
            json.dumps(trajectory_metadata, indent=2, ensure_ascii=False),
            "",
            "## Full Agent Trajectory:",
            json.dumps(step_cards, indent=2, ensure_ascii=False),
        ]
    )
    for _ in range(EVALUATOR_MAX_RETRIES):
        response = await run_chat_with_route_async(
            "rubric_generation",
            model_name=model_name,
            system_prompt=AGGREGATE_TRAJECTORY_SUMMARY_PROMPT.strip(),
            user_prompt=user_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            response_format=copy.deepcopy(PERSISTENT_STATE_RESPONSE_FORMAT),
            enable_json_schema_validation=True,
            **(model_kwargs or {}),
        )
        parsed = extract_json_from_response(response)
        if isinstance(parsed, dict):
            summary = copy.deepcopy(EMPTY_PERSISTENT_STATE)
            for key in summary:
                if isinstance(parsed.get(key), str):
                    summary[key] = parsed[key]
            return summary
    return copy.deepcopy(EMPTY_PERSISTENT_STATE)


async def _generate_aggregate_rubrics(
    *,
    question: dict[str, str],
    trajectories_summaries: list[dict[str, Any]],
    active_bank: list[RubricRecord],
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    round_index: int,
    model_kwargs: dict[str, Any] | None = None,
) -> list[RubricRecord]:
    prompt_parts = [
        "## Question:",
        f"System Prompt:\n{question.get('system_prompt', '')}",
        "",
        f"User Prompt:\n{question.get('user_prompt', '')}",
        "",
        "## Trajectory Summaries:",
    ]
    for index, trajectory_summary in enumerate(trajectories_summaries, start=1):
        prompt_parts.extend(
            [
                f"## Trajectory {index}:",
                json.dumps(trajectory_summary, indent=2, ensure_ascii=False),
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
                    indent=2,
                    ensure_ascii=False,
                ),
            ]
        )
    user_prompt = "\n".join(prompt_parts)
    task_text = "\n\n".join(part for part in [question.get("system_prompt", ""), question.get("user_prompt", "")] if part)
    for _ in range(EVALUATOR_MAX_RETRIES):
        response = await run_chat_with_route_async(
            "rubric_generation",
            model_name=model_name,
            system_prompt=AGGREGATE_RUBRIC_GENERATION_PROMPT.strip(),
            user_prompt=user_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            response_format=copy.deepcopy(RUBRIC_GENERATION_RESPONSE_FORMAT),
            enable_json_schema_validation=True,
            **(model_kwargs or {}),
        )
        if os.getenv("AGGREGATE_DEBUG_RUBRICS"):
            print("\n===== AGGREGATE RUBRIC RESPONSE =====\n")
            print(response)
        parsed = extract_json_from_response(response)
        if not isinstance(parsed, dict):
            continue
        rubric = _convert_generated_rubric(task_text, parsed, round_index)
        if rubric is not None:
            return [rubric]
    return []


async def _score_aggregate_summaries(
    *,
    question: dict[str, str],
    trajectories_summaries: list[dict[str, Any]],
    rubrics: list[RubricRecord],
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    model_kwargs: dict[str, Any] | None = None,
) -> tuple[list[list[dict[str, Any]]], dict[str, float], dict[int, str]]:
    if not trajectories_summaries or not rubrics:
        return [[] for _ in trajectories_summaries], {}, {}
    calls = []
    mapping: list[tuple[int, RubricRecord]] = []
    question_text = f"System Prompt:\n{question.get('system_prompt', '')}\n\nUser Prompt:\n{question.get('user_prompt', '')}"
    for summary_index, trajectory_summary in enumerate(trajectories_summaries):
        summary_text = json.dumps(trajectory_summary, indent=2, ensure_ascii=False)
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
                summary_text: str = summary_text,
                criterion: str = criterion,
            ) -> tuple[str, int]:
                user_prompt = "\n".join(
                    [
                        "## Question:",
                        question_text,
                        "",
                        "## Trajectory Summary:",
                        summary_text,
                        "",
                        "## Criterion:",
                        criterion,
                    ]
                )
                for _ in range(EVALUATOR_MAX_RETRIES):
                    response = await run_chat_with_route_async(
                        "rubric_judge",
                        model_name=model_name,
                        system_prompt=AGGREGATE_RUBRIC_JUDGE_PROMPT.strip(),
                        user_prompt=user_prompt,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        response_format=copy.deepcopy(JUDGE_RESPONSE_FORMAT),
                        enable_json_schema_validation=True,
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

            calls.append(_judge_single())
            mapping.append((summary_index, rubric))
    responses = await asyncio.gather(*calls, return_exceptions=True)
    per_view_scores: list[list[dict[str, Any]]] = [[] for _ in trajectories_summaries]
    rubric_values: dict[str, list[float]] = {}
    view_errors: dict[int, str] = {}
    for (summary_index, rubric), response in zip(mapping, responses):
        if isinstance(response, Exception):
            view_errors.setdefault(summary_index, f"{type(response).__name__}: {response}")
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
        per_view_scores[summary_index].append(record)
        rubric_values.setdefault(rubric.rubric_id, []).append(normalized)
    variances = {
        rubric_id: (0.0 if len(values) <= 1 else float(pvariance(values)))
        for rubric_id, values in rubric_values.items()
    }
    return per_view_scores, variances, view_errors


class AggregateTrajectoryRunner:
    def __init__(
        self,
        *,
        instance: dict[str, Any],
        backend: SWEAgentRolloutBackend,
        run_dir: Path,
        policy_model_name: str,
        rubric_model_name: str,
        judge_model_name: str,
        search_config: SearchConfig,
        num_trajectories: int,
        compression_chunk_size: int,
        rubric_model_kwargs: dict[str, Any] | None = None,
        judge_model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.instance = copy.deepcopy(instance)
        self.backend = backend
        self.run_dir = Path(run_dir)
        self.policy_model_name = policy_model_name
        self.rubric_model_name = rubric_model_name or policy_model_name
        self.judge_model_name = judge_model_name or policy_model_name
        self.search_config = search_config
        self.num_trajectories = num_trajectories
        self.compression_chunk_size = max(1, compression_chunk_size)
        self.rubric_model_kwargs = copy.deepcopy(rubric_model_kwargs or {})
        self.judge_model_kwargs = copy.deepcopy(judge_model_kwargs or {})
        self.task = instance["problem_statement"]
        self.task_id = instance["instance_id"]
        self.system_prompt = ""
        self.user_prompt = self.task

    def _run_candidate(self, candidate_index: int) -> dict[str, Any]:
        candidate_id = f"candidate-{candidate_index:02d}"
        candidate_dir = self.run_dir / "candidates" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        spec = RolloutSessionSpec(
            session_id=f"{candidate_id}-{uuid.uuid4().hex}",
            task=self.task,
            task_id=self.task_id,
            sample_index=candidate_index,
            policy_ref=self.policy_model_name,
            policy_version=self.policy_model_name,
            dataset_name="swebench",
            ground_truth=self.instance.get("patch"),
            raw_user_query=self.task,
            limits={"step_limit": self.backend.agent_config.get("step_limit", 0)},
            metadata={"template_vars": copy.deepcopy(self.instance)},
        )
        session = self.backend.create_session(spec)
        initial_messages = session.snapshot().model_dump(mode="json")["agent"]["state"].get("messages", [])
        if initial_messages:
            self.system_prompt = initial_messages[0].get("content", "") or self.system_prompt
        if len(initial_messages) > 1:
            self.user_prompt = initial_messages[1].get("content", "") or self.user_prompt
        result = session.run_until_pause(max_steps=self.backend.agent_config.get("step_limit"))
        result_payload = result.model_dump(mode="json")
        snapshot_payload = session.snapshot().model_dump(mode="json")
        workspace_meta = _collect_workspace_meta(session.agent.env)
        raw_traj = _make_raw_trajectory(
            snapshot=snapshot_payload,
            result=result_payload,
            info_extra={"aggregate_candidate_index": candidate_index},
        )
        slim_traj = build_slim_trajectory(raw_traj, model_name=self.policy_model_name)
        patch_record = {
            "model_name_or_path": self.policy_model_name,
            "instance_id": self.task_id,
            "model_patch": result_payload.get("submission", "") or "",
        }
        raw_traj_path = candidate_dir / "raw_traj.json"
        slim_traj_path = candidate_dir / "messages.json"
        patch_path = candidate_dir / "model_patch.json"
        raw_traj_path.write_text(json.dumps(raw_traj, indent=2, ensure_ascii=False), encoding="utf-8")
        slim_traj_path.write_text(json.dumps(slim_traj, indent=2, ensure_ascii=False), encoding="utf-8")
        patch_path.write_text(json.dumps({self.task_id: patch_record}, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "candidate_id": candidate_id,
            "candidate_index": candidate_index,
            "candidate_dir": candidate_dir,
            "raw_traj": raw_traj,
            "raw_traj_path": str(raw_traj_path),
            "slim_traj_path": str(slim_traj_path),
            "patch_path": str(patch_path),
            "snapshot": snapshot_payload,
            "result": result_payload,
            "workspace_meta": workspace_meta,
            "step_cards": _build_step_cards(snapshot_payload.get("metadata", {}).get("events", [])),
        }

    async def _compress_and_score(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        question = {
            "system_prompt": self.system_prompt,
            "user_prompt": self.task,
        }
        for candidate in candidates:
            state = await _summarize_aggregate_trajectory(
                question=question,
                step_cards=candidate["step_cards"],
                trajectory_metadata={
                    "step_count": len(candidate["step_cards"]),
                    "result_status": candidate["result"].get("status", ""),
                    "exit_status": candidate["result"].get("exit_status", ""),
                    "model_stats": candidate["raw_traj"].get("info", {}).get("model_stats", {}),
                },
                model_name=self.rubric_model_name,
                temperature=self.search_config.rubric_temperature,
                top_p=self.search_config.rubric_top_p,
                max_tokens=self.search_config.rubric_max_tokens,
                model_kwargs=self.rubric_model_kwargs,
            )
            candidate["compressed_view"] = {
                "meta_info": {
                    "step_count": int(candidate["result"].get("metadata", {}).get("n_calls", 0) or 0),
                    "result_status": candidate["result"].get("status", ""),
                    "exit_status": candidate["result"].get("exit_status", ""),
                },
                "trajectory": {
                    "compressed_trajectory": state,
                    "workspace_meta": {
                        "head_commit": candidate["workspace_meta"].get("head_commit", ""),
                        "changed_files": list(candidate["workspace_meta"].get("changed_files", []))[:8],
                        "untracked_files": list(candidate["workspace_meta"].get("untracked_files", []))[:8],
                        "diff_stat": candidate["workspace_meta"].get("diff_stat", ""),
                        "current_patch_chars": int(candidate["workspace_meta"].get("current_patch_chars", 0) or 0),
                    },
                },
            }
            (candidate["candidate_dir"] / "view.json").write_text(
                json.dumps(candidate["compressed_view"], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        generated = await _generate_aggregate_rubrics(
            question=question,
            trajectories_summaries=[candidate["compressed_view"] for candidate in candidates],
            active_bank=[],
            model_name=self.rubric_model_name,
            temperature=self.search_config.rubric_temperature,
            top_p=self.search_config.rubric_top_p,
            max_tokens=self.search_config.rubric_max_tokens,
            round_index=1,
            model_kwargs=self.rubric_model_kwargs,
        )
        scoring_rubrics = generated or _build_initial_rubric_bank("\n\n".join(part for part in [self.system_prompt, self.user_prompt] if part))
        score_records, variances, judge_errors = await _score_aggregate_summaries(
            question=question,
            trajectories_summaries=[candidate["compressed_view"] for candidate in candidates],
            rubrics=scoring_rubrics,
            model_name=self.judge_model_name,
            temperature=self.search_config.judge_temperature,
            top_p=self.search_config.judge_top_p,
            max_tokens=self.search_config.judge_max_tokens,
            model_kwargs=self.judge_model_kwargs,
        )
        active_after, inactive_after, round_bank = _update_rubric_bank(
            active_bank=[],
            inactive_bank=[],
            generated=scoring_rubrics,
            rewards=variances,
            max_active_rubrics=self.search_config.max_active_rubrics,
        )
        active_ids = {rubric.rubric_id for rubric in active_after}
        for index, candidate in enumerate(candidates):
            candidate["score_records"] = [record for record in score_records[index] if record["rubric_id"] in active_ids]
            candidate["judge_error"] = judge_errors.get(index)
            candidate["reward"] = (
                float("-inf")
                if candidate["judge_error"] or not candidate["score_records"]
                else _compute_weighted_reward(candidate["score_records"])
            )
            (candidate["candidate_dir"] / "judge.json").write_text(
                json.dumps(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "scores": candidate["score_records"],
                        "overall_reward": candidate["reward"],
                        "judge_error": candidate["judge_error"],
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        return {
            "generated": generated,
            "scoring_rubrics": scoring_rubrics,
            "variances": variances,
            "active_after": active_after,
            "inactive_after": inactive_after,
            "round_bank": round_bank,
            "judge_errors": judge_errors,
        }

    def run(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        candidates: list[dict[str, Any]] = []
        for candidate_index in range(self.num_trajectories):
            try:
                candidates.append(self._run_candidate(candidate_index))
            except Exception as exc:
                candidate_id = f"candidate-{candidate_index:02d}"
                candidate_dir = self.run_dir / "candidates" / candidate_id
                candidate_dir.mkdir(parents=True, exist_ok=True)
                (candidate_dir / "error.json").write_text(
                    json.dumps({"candidate_id": candidate_id, "error": str(exc)}, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        if not candidates:
            raise RuntimeError("No aggregate candidate trajectories completed successfully.")

        judged = asyncio.run(self._compress_and_score(candidates))
        best_candidate = max(
            candidates,
            key=lambda candidate: (candidate["reward"], candidate["result"].get("status") == "finished"),
        )

        rubrics_path = self.run_dir / "rubrics.json"
        rubrics_path.write_text(
            json.dumps(
                {
                    "generated": [asdict(rubric) for rubric in judged["generated"]],
                    "scoring_rubrics": [asdict(rubric) for rubric in judged["scoring_rubrics"]],
                    "active_after": [asdict(rubric) for rubric in judged["active_after"]],
                    "inactive_after": [asdict(rubric) for rubric in judged["inactive_after"]],
                    "variances": judged["variances"],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        scores_path = self.run_dir / "scores.json"
        scores_path.write_text(
            json.dumps(
                {
                    "candidates": [
                        {
                            "candidate_id": candidate["candidate_id"],
                            "reward": candidate["reward"],
                            "judge_error": candidate["judge_error"],
                            "status": candidate["result"].get("status", ""),
                            "exit_status": candidate["result"].get("exit_status", ""),
                            "scores": candidate["score_records"],
                        }
                        for candidate in candidates
                    ]
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        raw_traj_path = self.run_dir / "raw_traj.json"
        slim_traj_path = self.run_dir / "messages.json"
        patch_path = self.run_dir / "model_patch.json"
        shutil.copy2(best_candidate["raw_traj_path"], raw_traj_path)
        shutil.copy2(best_candidate["slim_traj_path"], slim_traj_path)
        shutil.copy2(best_candidate["patch_path"], patch_path)

        manifest_path = self.run_dir / "run_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "instance_id": self.task_id,
                    "policy_model_name": self.policy_model_name,
                    "rubric_model_name": self.rubric_model_name,
                    "judge_model_name": self.judge_model_name,
                    "num_trajectories": self.num_trajectories,
                    "compression_chunk_size": self.compression_chunk_size,
                    "selected_candidate_id": best_candidate["candidate_id"],
                    "candidates": [
                        {
                            "candidate_id": candidate["candidate_id"],
                            "reward": candidate["reward"],
                            "status": candidate["result"].get("status", ""),
                            "exit_status": candidate["result"].get("exit_status", ""),
                            "raw_trajectory_path": candidate["raw_traj_path"],
                            "slim_trajectory_path": candidate["slim_traj_path"],
                            "patch_path": candidate["patch_path"],
                            "view_path": str(candidate["candidate_dir"] / "view.json"),
                            "judge_path": str(candidate["candidate_dir"] / "judge.json"),
                        }
                        for candidate in candidates
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

def run_aggregate(
    args: argparse.Namespace,
    instance_ids: Sequence[str] | None,
) -> None:
    model_name = args.vllm_client_model if args.backend == "vllm" else args.openai_model
    available_instances = load_swebench_instances(args.subset, args.split)
    by_id = {instance["instance_id"]: instance for instance in available_instances}
    selected_ids = list(instance_ids or by_id)
    missing = [instance_id for instance_id in selected_ids if instance_id not in by_id]
    if missing:
        raise RuntimeError(f"Instances not found in {args.subset}/{args.split}: {', '.join(missing)}")
    instances = [by_id[instance_id] for instance_id in selected_ids]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_root = args.output_root / (
        f"{re.sub(r'[^A-Za-z0-9._-]+', '_', args.subset.replace('/', '__'))}_"
        f"{re.sub(r'[^A-Za-z0-9._-]+', '_', args.split.replace('/', '__'))}_"
        f"{re.sub(r'[^A-Za-z0-9._-]+', '_', model_name.replace('/', '__'))}"
    )
    run_log_path = DEFAULT_LOG_ROOT / f"aggregate-{timestamp}.log"
    run_log_path.parent.mkdir(parents=True, exist_ok=True)

    gpu_id: int | None = None
    gpu_ids: list[int] = []
    vllm_handle = None
    if args.backend == "vllm":
        gpu_ids = choose_gpus(args.gpu_id)
        gpu_id = gpu_ids[0] if gpu_ids else None
        if gpu_id is None:
            raise RuntimeError("vLLM backend requires a GPU; got gpu_id=none")
        vllm_handle = launch_vllm_server_handle(
            model_name=args.vllm_serve_model,
            port=find_free_port(args.vllm_port),
            gpu_id=gpu_id,
            gpu_ids=gpu_ids,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            allow_long_max_model_len=args.allow_long_max_model_len,
        )
        shared_model_kwargs = {"api_base": vllm_handle.base_url, "api_key": "EMPTY"}
    else:
        required_env = infer_litellm_api_env(args.openai_model)
        if required_env and not os.getenv(required_env):
            raise RuntimeError(f"{required_env} is not set for model {args.openai_model}")
        shared_model_kwargs = {}

    configure_model_route("rubric_generation", ModelRouteConfig(backend="litellm"))
    configure_model_route("rubric_judge", ModelRouteConfig(backend="litellm"))
    run_dirs: list[Path] = []
    errors: list[str] = []
    try:
        config = build_swebench_config(
            config_spec=[str(SWE_AGENT_TEXTBASED_CONFIG)],
            model=model_name,
            model_class=DEFAULT_MODEL_CLASS,
            extra_overrides={
                "agent": {
                    "step_limit": args.step_limit,
                    "cost_limit": 0,
                },
                "environment": {
                    "timeout": args.environment_timeout,
                    "pull_timeout": args.pull_timeout,
                },
                "model": {
                    "model_kwargs": {
                        "temperature": args.policy_temperature,
                        "top_p": args.policy_top_p,
                        "max_tokens": args.completion_max_tokens,
                        **shared_model_kwargs,
                    },
                    "cost_tracking": "ignore_errors",
                },
            },
        )
        search_config = SearchConfig(
            max_active_rubrics=args.max_active_rubrics,
            policy_temperature=args.policy_temperature,
            policy_top_p=args.policy_top_p,
            rubric_temperature=args.rubric_temperature,
            rubric_top_p=args.rubric_top_p,
            rubric_max_tokens=args.rubric_max_tokens,
            judge_temperature=args.judge_temperature,
            judge_top_p=args.judge_top_p,
            judge_max_tokens=args.judge_max_tokens,
        )
        rubric_model_name = args.rubric_model or model_name
        judge_model_name = args.judge_model or model_name
        rubric_model_kwargs = dict(shared_model_kwargs)
        judge_model_kwargs = dict(shared_model_kwargs)
        with temporary_env({"MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": str(args.model_retry_attempts), "LITELLM_LOG": "ERROR"}):
            with tee_console(run_log_path):
                for instance in instances:
                    run_dir = run_root / instance["instance_id"] / timestamp
                    run_dir.mkdir(parents=True, exist_ok=True)
                    instance_config = copy.deepcopy(config)
                    environment_config = instance_config.setdefault("environment", {})
                    if environment_config.get("environment_class", "docker") == "docker":
                        environment_config["image"] = get_swebench_docker_image_name(instance)
                    try:
                        runner = AggregateTrajectoryRunner(
                            instance=instance,
                            backend=SWEAgentRolloutBackend(
                                model=instance_config.get("model", {}),
                                environment=instance_config.get("environment", {}),
                                agent=instance_config.get("agent", {}),
                                default_agent_type="default",
                                default_environment_type=instance_config.get("environment", {}).get("environment_class", "docker"),
                            ),
                            run_dir=run_dir,
                            policy_model_name=model_name,
                            rubric_model_name=rubric_model_name,
                            judge_model_name=judge_model_name,
                            search_config=search_config,
                            num_trajectories=args.num_trajectories,
                            compression_chunk_size=args.compression_chunk_size,
                            rubric_model_kwargs=rubric_model_kwargs,
                            judge_model_kwargs=judge_model_kwargs,
                        )
                        runner.run()
                        run_dirs.append(run_dir)
                    except Exception as exc:
                        errors.append(f"{instance['instance_id']}: {exc}")
                        write_failure_artifacts(
                            instance_id=instance["instance_id"],
                            run_dir=run_dir,
                            error=exc,
                            log_path=run_log_path,
                        )
        if run_dirs:
            run_harness_evaluation(
                run_dirs=run_dirs,
                split=args.split,
                model_name=model_name,
                dataset_name=DATASET_MAPPING.get(args.subset, args.subset),
                log_path=run_log_path,
                timeout=args.eval_timeout,
                max_workers=max(1, min(args.workers, len(run_dirs))),
                instances_by_id=by_id,
            )
        if errors:
            raise RuntimeError("; ".join(errors))
    finally:
        clear_model_routes()
        terminate_process(vllm_handle.process if vllm_handle else None)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run aggregate rubric-selection SWE-agent on one or more SWE-bench instances.")
    parser.add_argument("--backend", choices=["vllm", "openai"], default="vllm")
    parser.add_argument("--instance-id", action=ParseInstanceIds, nargs="+", default=None)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_AGGREGATE_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--step-limit", type=int, default=DEFAULT_STEP_LIMIT)
    parser.add_argument("--environment-timeout", type=int, default=DEFAULT_ENV_TIMEOUT)
    parser.add_argument("--pull-timeout", type=int, default=DEFAULT_PULL_TIMEOUT)
    parser.add_argument("--eval-timeout", type=int, default=DEFAULT_EVAL_TIMEOUT)
    parser.add_argument("--gpu-id", default="auto:2")
    parser.add_argument("--vllm-port", type=int, default=DEFAULT_VLLM_PORT)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--allow-long-max-model-len", action="store_true")
    parser.add_argument("--vllm-serve-model", default=DEFAULT_SERVE_MODEL)
    parser.add_argument("--vllm-client-model", default=DEFAULT_SERVE_MODEL)
    parser.add_argument("--openai-model", default=DEFAULT_SERVE_MODEL)
    parser.add_argument("--model-retry-attempts", type=int, default=2)
    parser.add_argument("--completion-max-tokens", type=int, default=DEFAULT_COMPLETION_MAX_TOKENS)
    parser.add_argument("--num-trajectories", type=int, default=4)
    parser.add_argument("--compression-chunk-size", type=int, default=12)
    parser.add_argument("--max-active-rubrics", type=int, default=6)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--policy-top-p", type=float, default=0.95)
    parser.add_argument("--rubric-temperature", type=float, default=0.0)
    parser.add_argument("--rubric-top-p", type=float, default=1.0)
    parser.add_argument("--rubric-max-tokens", type=int, default=1024)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-top-p", type=float, default=1.0)
    parser.add_argument("--judge-max-tokens", type=int, default=1024)
    parser.add_argument("--rubric-model", default=None)
    parser.add_argument("--judge-model", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_aggregate(args, args.instance_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
