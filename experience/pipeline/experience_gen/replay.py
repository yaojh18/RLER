from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any

from swe_agent.prompt import (
    PC_TRAJECTORY_RUBRIC_GENERATION_PROMPT,
    PC_TRAJECTORY_RUBRIC_JUDGE_PROMPT,
    SWE_TRAJECTORY_RUBRIC_GENERATION_PROMPT,
    SWE_TRAJECTORY_RUBRIC_JUDGE_PROMPT,
)
from swe_agent.rubric_bank import RubricRecord, render_compact_markdown

from .config import ModelConfig
from .metrics import ranking_metrics


def _view_parts(context: dict[str, Any]) -> tuple[
    dict[str, Any], dict[str, Any], list[dict[str, Any]]
]:
    view = context["view"]
    question = copy.deepcopy(view["question"])
    shared = {
        "previous_persistent_state": copy.deepcopy(view.get("previous_state") or {}),
        "latest_agent_trajectory": copy.deepcopy(view.get("parent_trajectory")),
    }
    continuations = copy.deepcopy(view["continuations"])
    return question, shared, continuations


def _experience_section(cards: list[dict[str, Any]]) -> str:
    return "## Retrieved Rubric Experiences:\n" + render_compact_markdown(
        [
            {
                "title": card["title"],
                "description": card["description"],
                "context": card["context"],
                "experience": card["experience"],
                "reference_golden_rubrics": (
                    card.get("metadata") or {}
                ).get("reference_golden_rubrics", []),
            }
            for card in cards
        ]
    )


def _scope_source(context: dict[str, Any], scope: str) -> dict[str, Any]:
    if scope == "siblings":
        return {
            "generated": context.get("generated_rubrics") or [],
            "scores": context.get("judge_scores") or {},
        }
    for event in context.get("retrieval_events") or []:
        if event.get("scope") == scope:
            return {
                "generated": event.get("generated_rubrics") or [],
                "scores": event.get("judge_scores") or {},
            }
    return {"generated": [], "scores": {}}


def _baseline_metrics(context: dict[str, Any], scope: str) -> dict[str, float]:
    metric = (context.get("scope_metrics") or {}).get(scope)
    if metric:
        return {
            "tie_aware_success": float(metric["tie_aware_success"]),
            "pairwise_accuracy": float(metric["pairwise_accuracy"]),
        }
    scores = _scope_source(context, scope)["scores"]
    if not scores:
        raise ValueError(f"{context['context_id']}: no baseline metrics for {scope}")
    return ranking_metrics(scores, context["gt_rewards"], context["node_ids"])


class TrajectoryReplayEvaluator:
    """Model-backed round replay using the copied trajectory judge implementation."""

    def __init__(
        self,
        *,
        models: ModelConfig | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.models = models or ModelConfig()
        self.model_kwargs = copy.deepcopy(
            model_kwargs or {"custom_llm_provider": "openai"}
        )

    async def evaluate(
        self,
        *,
        context: dict[str, Any],
        scope: str,
        track: str,
        card: dict[str, Any] | None = None,
        cards: list[dict[str, Any]] | None = None,
        tie_break: bool = False,
    ) -> dict[str, Any]:
        selected_cards = copy.deepcopy(cards or ([card] if card is not None else []))
        if not selected_cards:
            raise ValueError("At least one experience card is required")
        if scope not in {"siblings", "pc"}:
            raise ValueError("scope must be siblings or pc")
        if track == "rubric":
            return await self._evaluate_rubric(
                context=context,
                scope=scope,
                cards=selected_cards,
                tie_break=tie_break,
            )
        if track == "judge":
            return await self._evaluate_judge(
                context=context, scope=scope, cards=selected_cards
            )
        raise ValueError("track must be rubric or judge")

    async def _evaluate_rubric(
        self,
        *,
        context: dict[str, Any],
        scope: str,
        cards: list[dict[str, Any]],
        tie_break: bool,
    ) -> dict[str, Any]:
        from swe_agent import trajectory_search as ts

        question, shared, continuations = _view_parts(context)
        generation_prompt = (
            PC_TRAJECTORY_RUBRIC_GENERATION_PROMPT
            if scope == "pc"
            else SWE_TRAJECTORY_RUBRIC_GENERATION_PROMPT
        )
        judge_prompt = (
            PC_TRAJECTORY_RUBRIC_JUDGE_PROMPT
            if scope == "pc"
            else SWE_TRAJECTORY_RUBRIC_JUDGE_PROMPT
        )
        experience_section = _experience_section(cards)
        generation_kwargs = {
            "model_name": self.models.judge,
            "temperature": self.models.temperature,
            "top_p": self.models.top_p,
            "max_tokens": self.models.max_tokens,
            "model_kwargs": self.model_kwargs,
        }
        judge_kwargs = {
            "model_name": self.models.judge,
            "temperature": self.models.judge_temperature,
            "top_p": self.models.top_p,
            "max_tokens": self.models.max_tokens,
            "model_kwargs": self.model_kwargs,
        }
        batch = await ts._generate_and_score_rubric_batch(
            sample_count=1,
            round_index=int(context["round_index"]),
            generation_kwargs=generation_kwargs,
            generation_prompt=generation_prompt,
            rubric_list_prefix=f"experience-{scope}",
            question=question,
            shared_context=shared,
            continuations=continuations,
            extra_prompt_sections=[experience_section],
            judge_kwargs=judge_kwargs,
            judge_prompt=judge_prompt,
        )
        rubrics = batch["sample_generated_rubrics"][0]
        evaluation = ts._rubric_sample_evaluation(
            score_batch=batch,
            sample_index=0,
            node_ids=context["node_ids"],
            scoring_rubrics=rubrics,
            include_variance_reward=scope == "siblings",
        )
        score_lookup = evaluation["score_lookup_by_node"]
        scores = (
            ts._pc_avg_scores_from_rubrics(
                node_ids=context["node_ids"],
                score_lookup_by_node=score_lookup,
                rubrics=rubrics,
            )
            if scope == "pc"
            else ts._avg_scores_from_rubrics(
                node_ids=context["node_ids"],
                score_lookup_by_node=score_lookup,
                rubrics=rubrics,
            )
        )
        generated = batch["generated_samples"][0]
        initial_scores = copy.deepcopy(scores)
        initial_metrics = ranking_metrics(
            initial_scores, context["gt_rewards"], context["node_ids"]
        )
        tie_result = None
        tie_messages: list[dict[str, Any]] = []
        if tie_break and rubrics:
            tie_result = await ts._run_score_tie_break(
                scope=scope,
                round_index=int(context["round_index"]),
                generation_prompt=generation_prompt,
                rubric_list_prefix=f"experience-{scope}",
                judge_prompt=judge_prompt,
                question=question,
                shared_context=shared,
                continuations=continuations,
                extra_prompt_sections=[experience_section],
                generation_kwargs=generation_kwargs,
                judge_kwargs=judge_kwargs,
                initial_rubrics=rubrics,
                initial_score_by_rubric=evaluation["metrics"]["score_by_rubric"],
                initial_scores=initial_scores,
            )
            if tie_result is not None:
                tie_messages = copy.deepcopy(tie_result.pop("messages", []))
                if tie_result.get("status") == "success":
                    scores = copy.deepcopy(tie_result["adjusted_scores"])
        all_errors = [
            *evaluation["judge_errors"],
            *((tie_result or {}).get("judge_errors") or []),
        ]
        return {
            "status": "success" if not all_errors else "judge_error",
            "track": "rubric",
            "scope": scope,
            "baseline_metrics": _baseline_metrics(context, scope),
            "candidate_metrics": ranking_metrics(
                scores, context["gt_rewards"], context["node_ids"]
            ),
            "initial_candidate_metrics": initial_metrics,
            "initial_average_rubric_judged_scores": initial_scores,
            "generated": [asdict(rubric) for rubric in rubrics],
            "average_rubric_judged_scores": scores,
            "score_by_rubric": evaluation["metrics"]["score_by_rubric"],
            "judge_errors": all_errors,
            "generation_messages": generated.messages,
            "tie_break": tie_result,
            "tie_break_messages": tie_messages,
        }

    async def _evaluate_judge(
        self,
        *,
        context: dict[str, Any],
        scope: str,
        cards: list[dict[str, Any]],
    ) -> dict[str, Any]:
        from swe_agent import trajectory_search as ts

        question, shared, continuations = _view_parts(context)
        raw_rubrics = _scope_source(context, scope)["generated"]
        rubrics = []
        for item in raw_rubrics:
            try:
                rubrics.append(RubricRecord(**item))
            except TypeError:
                converted = ts._convert_rubric_item(
                    "\n".join(
                        [
                            str(question.get("system_prompt") or ""),
                            str(question.get("user_prompt") or ""),
                        ]
                    ),
                    item,
                    int(context["round_index"]),
                )
                if converted is not None:
                    rubrics.append(converted)
        if not rubrics:
            raise ValueError(f"{context['context_id']}: no landed {scope} rubrics")
        base_prompt = (
            PC_TRAJECTORY_RUBRIC_JUDGE_PROMPT
            if scope == "pc"
            else SWE_TRAJECTORY_RUBRIC_JUDGE_PROMPT
        )
        judge_prompt = base_prompt.rstrip() + "\n\n" + _experience_section(cards)
        per_view, errors = await ts._score_round(
            question=question,
            shared_context=shared,
            continuations=continuations,
            rubrics=rubrics,
            model_name=self.models.judge,
            temperature=self.models.judge_temperature,
            top_p=self.models.top_p,
            max_tokens=self.models.max_tokens,
            model_kwargs=self.model_kwargs,
            judge_prompt=judge_prompt,
        )
        score_batch = {"generated_score_results": {0: (per_view, errors)}}
        evaluation = ts._rubric_sample_evaluation(
            score_batch=score_batch,
            sample_index=0,
            node_ids=context["node_ids"],
            scoring_rubrics=rubrics,
            include_variance_reward=scope == "siblings",
        )
        scores = (
            ts._pc_avg_scores_from_rubrics(
                node_ids=context["node_ids"],
                score_lookup_by_node=evaluation["score_lookup_by_node"],
                rubrics=rubrics,
            )
            if scope == "pc"
            else ts._avg_scores_from_rubrics(
                node_ids=context["node_ids"],
                score_lookup_by_node=evaluation["score_lookup_by_node"],
                rubrics=rubrics,
            )
        )
        return {
            "status": "success" if not errors else "judge_error",
            "track": "judge",
            "scope": scope,
            "baseline_metrics": _baseline_metrics(context, scope),
            "candidate_metrics": ranking_metrics(
                scores, context["gt_rewards"], context["node_ids"]
            ),
            "generated": [asdict(rubric) for rubric in rubrics],
            "average_rubric_judged_scores": scores,
            "score_by_rubric": evaluation["metrics"]["score_by_rubric"],
            "judge_errors": errors,
        }
