from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import ModelConfig
from .contexts import strict_reward_variance
from .filters import evaluate_candidate, normalize_card
from .io import atomic_json, load_json, stable_id
from .models import InvalidJsonResponse, JsonModelClient
from .prompts import (
    GENERATION_SYSTEM_PROMPT,
    REFINE_SYSTEM_PROMPT,
    generation_prompt,
)


def _experience_id(
    context: dict[str, Any], scope: str, track: str
) -> str:
    key = {
        "context_id": context["context_id"],
        "scope": scope,
        "track": track,
    }
    return "EXP_" + stable_id(key)[:16].upper()


def _diagnostic(context: dict[str, Any], scope: str) -> dict[str, Any]:
    event = next(
        (
            row
            for row in context.get("retrieval_events") or []
            if row.get("scope") == scope
        ),
        {},
    )
    result = {
        "primary_cluster": "measured_branch_ranking_failure",
        "analysis": context.get("analysis") or "",
        "baseline_metrics": (context.get("scope_metrics") or {}).get(scope),
        "rubrics": (
            context.get("generated_rubrics")
            if scope == "siblings"
            else event.get("generated_rubrics")
        )
        or [],
        "score_by_rubric": event.get("score_by_rubric"),
        "key_visible_evidence": context.get("key_visible_evidence") or [],
    }
    return result


class ExperienceGenerator:
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
        self.client = JsonModelClient(
            top_p=self.models.top_p,
            temperature=self.models.temperature,
        )

    async def generate_and_evaluate(
        self,
        *,
        context: dict[str, Any],
        scope: str,
        track: str,
        output_dir: str | Path,
        existing_cards: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        from .replay import TrajectoryReplayEvaluator

        evaluator = TrajectoryReplayEvaluator(
            models=self.models,
            model_kwargs=self.model_kwargs,
        )
        if not strict_reward_variance(context):
            raise ValueError(
                f"{context['context_id']}: reward variance gate rejected this context"
            )
        root = Path(output_dir)
        attempts = []
        previous = None
        for attempt in range(1, 3):
            model = self.models.generator if attempt == 1 else self.models.refiner
            result_path = root / f"attempt_{attempt:02d}/result.json"
            if result_path.is_file():
                result = load_json(result_path)
                attempts.append(result)
                previous = result
                if result.get("accepted"):
                    break
                continue
            experience_id = _experience_id(context, scope, track)
            try:
                parsed, messages = await self.client.call(
                    model=model,
                    system=(
                        GENERATION_SYSTEM_PROMPT
                        if attempt == 1
                        else REFINE_SYSTEM_PROMPT
                    ),
                    user=generation_prompt(
                        context=context,
                        scope=scope,
                        track=track,
                        diagnostic=_diagnostic(context, scope),
                        attempt=attempt,
                        previous=previous,
                    ),
                    max_tokens=self.models.max_tokens,
                    retry_max_tokens=self.models.retry_max_tokens,
                    validator=lambda value: normalize_card(
                        value, experience_id=experience_id
                    )
                    is not None,
                    model_kwargs=self.model_kwargs,
                )
            except InvalidJsonResponse as exc:
                atomic_json(
                    root / f"attempt_{attempt:02d}/invalid_messages.json",
                    exc.messages,
                )
                raise
            card = normalize_card(
                parsed,
                experience_id=experience_id,
            )
            if card is None:
                raise ValueError("Model returned an invalid experience card")
            evaluation = await evaluator.evaluate(
                context=context,
                scope=scope,
                track=track,
                card=card,
            )
            decision = evaluate_candidate(
                card=card,
                baseline_metrics=evaluation["baseline_metrics"],
                candidate_metrics=evaluation["candidate_metrics"],
                errors=evaluation.get("judge_errors"),
                existing_cards=existing_cards or [],
            )
            result = {
                "schema_version": 1,
                "context_id": context["context_id"],
                "scope": scope,
                "track": track,
                "attempt": attempt,
                "model": model,
                "card": card,
                "evaluation": evaluation,
                "decision": asdict(decision),
                "accepted": decision.accepted,
            }
            atomic_json(root / f"attempt_{attempt:02d}/generation_messages.json", messages)
            atomic_json(result_path, result)
            attempts.append(result)
            previous = result
            if result["accepted"]:
                break
        accepted = next((row for row in attempts if row.get("accepted")), None)
        payload = {
            "schema_version": 1,
            "context_id": context["context_id"],
            "scope": scope,
            "track": track,
            "variance_gate": True,
            "status": "accepted" if accepted else "filtered",
            "accepted_attempt": accepted,
            "attempts": attempts,
        }
        atomic_json(root / "result.json", payload)
        return payload
