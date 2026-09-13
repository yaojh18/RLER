from __future__ import annotations

import asyncio
import copy
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .materialize import update_counts
from .config import ModelConfig
from .contexts import (
    context_scopes,
    full_variance_population,
    historical_selection_error,
)
from .io import atomic_json, load_json
from .keywords import phrase_match


SCOPES = ("siblings", "pc")
FINAL_PAIRWISE_THRESHOLD = 0.5


def case_slug(context_id: str, scope: str) -> str:
    safe = (
        f"{scope}__{context_id}"
        .replace("/", "_")
        .replace(":", "__")
        .replace(" ", "_")
    )
    return safe


def _events(
    contexts: Iterable[dict[str, Any]],
    scopes: Iterable[str],
) -> list[dict[str, Any]]:
    allowed = set(scopes)
    return [
        {
            "case_id": f"{scope}::{context['context_id']}",
            "context_id": str(context["context_id"]),
            "instance_id": str(context["instance_id"]),
            "round_index": int(context["round_index"]),
            "scope": scope,
        }
        for context in contexts
        for scope in context_scopes(context)
        if scope in allowed
    ]


async def generate_overfits(
    *,
    contexts: list[dict[str, Any]],
    base_checkpoint: str | Path,
    output: str | Path,
    scopes: tuple[str, ...] = SCOPES,
    track: str = "rubric",
    models: ModelConfig | None = None,
    concurrency: int = 8,
) -> dict[str, Any]:
    """Generate locally gated experiences only for historical selection errors."""

    from .generation import ExperienceGenerator

    root = Path(output)
    population = full_variance_population(contexts)
    error_population = [
        context for context in population if historical_selection_error(context)
    ]
    by_id = {str(row["context_id"]): row for row in error_population}
    events = _events(error_population, scopes)
    if not events:
        raise ValueError(
            "The input contains no failed historical strict-variance scope events"
        )
    checkpoint = Path(base_checkpoint)
    existing = {
        scope: load_json(
            checkpoint / f"bank/{scope}/experience_bank.json"
        )["experiences"]
        for scope in scopes
    }
    generator = ExperienceGenerator(models=models)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def one(event: dict[str, Any]) -> dict[str, Any]:
        context = by_id[event["context_id"]]
        result_root = root / "overfit" / case_slug(
            event["context_id"], event["scope"]
        )
        async with semaphore:
            result = await generator.generate_and_evaluate(
                context=context,
                scope=event["scope"],
                track=track,
                output_dir=result_root,
                existing_cards=existing[event["scope"]],
            )
        return {
            **event,
            "status": result["status"],
            "accepted": result.get("accepted_attempt") is not None,
            "accepted_attempt": (
                (result.get("accepted_attempt") or {}).get("attempt")
            ),
            "accepted_model": (
                (result.get("accepted_attempt") or {}).get("model")
            ),
            "result_path": str(result_root / "result.json"),
        }

    rows = await asyncio.gather(*(one(event) for event in events))
    model_config = models or ModelConfig()
    population_payload = {
        "schema_version": 1,
        "dataset_role": "optimization",
        "definition": (
            "Every landed group containing reward==2 and reward!=2; generate "
            "only when the historically selected branch has reward != 2."
        ),
        "context_count": len(population),
        "instance_count": len({row["instance_id"] for row in population}),
        "error_context_count": len(error_population),
        "error_instance_count": len(
            {row["instance_id"] for row in error_population}
        ),
        "event_count": len(events),
        "scope_counts": dict(Counter(row["scope"] for row in events)),
        "instance_ids": sorted(
            {str(row["instance_id"]) for row in error_population}
        ),
        "context_ids": [row["context_id"] for row in error_population],
        "events": events,
    }
    atomic_json(root / "population.json", population_payload)
    payload = {
        "schema_version": 1,
        "pipeline": "per_variance_round_overfit",
        "dataset_role": "optimization",
        "model_schedule": {
            "attempt_1": model_config.generator,
            "attempt_2": model_config.refiner,
            "judge": model_config.judge,
        },
        "acceptance": (
            "keep the first GLM/GPT attempt whose replay has no judge error and "
            "strictly improves tie-aware success or pairwise accuracy"
        ),
        "targets": len(rows),
        "accepted": sum(row["accepted"] for row in rows),
        "filtered": sum(not row["accepted"] for row in rows),
        "attempt_1_accepted": sum(row["accepted_attempt"] == 1 for row in rows),
        "attempt_2_accepted": sum(row["accepted_attempt"] == 2 for row in rows),
        "rows": rows,
    }
    atomic_json(root / "overfit_summary.json", payload)
    return payload


async def build_provisional_checkpoint(
    *,
    contexts: list[dict[str, Any]],
    base_checkpoint: str | Path,
    overfit_summary: str | Path,
    output: str | Path,
    keyword_output: str | Path,
    models: ModelConfig | None = None,
    concurrency: int = 8,
) -> dict[str, Any]:
    """Add only locally accepted overfits and their filtered keywords."""

    from .keywords import KeywordBuilder

    summary = load_json(overfit_summary)
    accepted_rows = [row for row in summary.get("rows") or [] if row["accepted"]]
    if not accepted_rows:
        raise ValueError("No locally accepted overfit result is available")
    context_by_id = {str(row["context_id"]): row for row in contexts}
    builder = KeywordBuilder(models=models)
    root = Path(keyword_output)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def one(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        result = load_json(row["result_path"])
        accepted = result.get("accepted_attempt") or {}
        card = accepted.get("card")
        if not isinstance(card, dict):
            raise ValueError(f"{row['case_id']}: accepted result has no card")
        context = context_by_id[row["context_id"]]
        keyword_path = root / f"{case_slug(row['context_id'], row['scope'])}.json"
        if keyword_path.is_file():
            record = load_json(keyword_path)
        else:
            async with semaphore:
                record = await builder.generate(
                    card=card,
                    positive_views=[context["view"]],
                    output_path=keyword_path,
                )
        return row["scope"], record

    keyword_pairs = await asyncio.gather(*(one(row) for row in accepted_rows))
    keyword_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scope, record in keyword_pairs:
        keyword_records[scope].append(record)
    from .materialize import materialize_checkpoint

    materialized = materialize_checkpoint(
        base=base_checkpoint,
        output=output,
        accepted_results=[row["result_path"] for row in accepted_rows],
        keyword_records=dict(keyword_records),
    )
    payload = {
        "schema_version": 1,
        "pipeline": "locally_gated_provisional_bank",
        "dataset_role": "optimization",
        "accepted_overfits": len(accepted_rows),
        "keyword_records": {
            scope: len(records)
            for scope, records in sorted(keyword_records.items())
        },
        "checkpoint": str(output),
        "materialized": materialized,
    }
    return payload


async def run_full_replay(
    *,
    contexts: list[dict[str, Any]],
    checkpoint: str | Path,
    output: str | Path,
    scopes: tuple[str, ...] = SCOPES,
    track: str = "rubric",
    models: ModelConfig | None = None,
    concurrency: int = 8,
) -> dict[str, Any]:
    """Replay the provisional bank on the complete strict-variance population."""

    from swe_agent.experience_retrieval import (
        WeightedKeywordExperienceRetriever,
        _stage_query,
    )
    from swe_agent.rubric_bank import (
        _retrieval_summary_context,
        render_compact_markdown,
    )

    from .replay import TrajectoryReplayEvaluator

    root = Path(output)
    checkpoint_path = Path(checkpoint)
    model_config = models or ModelConfig()
    population = full_variance_population(contexts)
    by_id = {str(row["context_id"]): row for row in population}
    events = _events(population, scopes)
    if not events:
        raise ValueError("The full replay population has no eligible events")
    retrievers = {
        scope: WeightedKeywordExperienceRetriever(checkpoint_path, scope=scope)
        for scope in scopes
    }
    evaluator = TrajectoryReplayEvaluator(models=model_config)
    summary_retriever = retrievers["siblings"] if "siblings" in retrievers else next(
        iter(retrievers.values())
    )
    semaphore = asyncio.Semaphore(max(1, concurrency))
    summaries: dict[str, dict[str, Any]] = {}

    async def summarize_one(context: dict[str, Any]) -> None:
        summary_root = root / "query_summaries" / case_slug(
            str(context["context_id"]), "summary"
        )
        result_path = summary_root / "result.json"
        if result_path.is_file():
            cached = load_json(result_path)
            if isinstance(cached.get("summary"), dict):
                summaries[str(context["context_id"])] = cached["summary"]
                return
        async with semaphore:
            summary, messages = await summary_retriever.summarize(
                context_markdown=render_compact_markdown(
                    _retrieval_summary_context(
                        context["view"],
                        instance_id=str(context["instance_id"]),
                        round_index=int(context["round_index"]),
                    )
                ),
                stage=_stage_query(context["view"]),
                model_name=model_config.judge,
                top_p=model_config.top_p,
                model_kwargs={"custom_llm_provider": "openai"},
                max_tokens=model_config.max_tokens,
            )
        atomic_json(summary_root / "messages.json", messages)
        atomic_json(
            result_path,
            {
                "schema_version": 1,
                "context_id": context["context_id"],
                "summary": summary,
            },
        )
        summaries[str(context["context_id"])] = summary

    await asyncio.gather(*(summarize_one(context) for context in population))

    async def replay_one(event: dict[str, Any]) -> dict[str, Any]:
        context = by_id[event["context_id"]]
        retriever = retrievers[event["scope"]]
        selected_ids = retriever.rank(
            context=context["view"],
            summary=summaries[event["context_id"]],
            instance_id=context["instance_id"],
        )
        cards = [retriever.by_id[experience_id] for experience_id in selected_ids]
        result_root = root / "runs" / case_slug(
            event["context_id"], event["scope"]
        )
        result_path = result_root / "result.json"
        if result_path.is_file():
            cached = load_json(result_path)
            if cached.get("status") in {
                "success",
                "judge_error",
            }:
                return {
                    **event,
                    "status": "reused",
                    "result_path": str(result_path),
                }
        async with semaphore:
            evaluation = await evaluator.evaluate(
                context=context,
                scope=event["scope"],
                track=track,
                cards=cards,
                tie_break=True,
            )
        generation_messages = evaluation.pop("generation_messages", [])
        tie_break_messages = evaluation.pop("tie_break_messages", [])
        result = {
            "schema_version": 1,
            **event,
            "retrieved_experience_ids": selected_ids,
            **evaluation,
        }
        atomic_json(result_root / "generation_messages.json", generation_messages)
        atomic_json(result_root / "tie_break_messages.json", tie_break_messages)
        atomic_json(result_path, result)
        return {
            **event,
            "status": result["status"],
            "result_path": str(result_path),
        }

    rows = await asyncio.gather(*(replay_one(event) for event in events))
    population_payload = {
        "schema_version": 1,
        "dataset_role": "validation",
        "definition": (
            "Every landed strict reward==2/non-2 variance group."
        ),
        "checkpoint": str(checkpoint_path),
        "context_count": len(population),
        "instance_count": len({row["instance_id"] for row in population}),
        "event_count": len(events),
        "scope_counts": dict(Counter(row["scope"] for row in events)),
        "instance_ids": sorted({str(row["instance_id"]) for row in population}),
        "context_ids": [row["context_id"] for row in population],
        "contexts": population,
        "events": events,
    }
    atomic_json(root / "population.json", population_payload)
    statuses = Counter(row["status"] for row in rows)
    payload = {
        "schema_version": 1,
        "pipeline": "full_population_replay",
        "dataset_role": "validation",
        "targets": len(rows),
        "statuses": dict(sorted(statuses.items())),
        "rows": rows,
    }
    atomic_json(root / "run_summary.json", payload)
    return payload


def full_replay_decisions(
    *,
    checkpoint: str | Path,
    replay_root: str | Path,
    threshold: float = FINAL_PAIRWISE_THRESHOLD,
) -> dict[str, Any]:
    """Apply the final V108 whole-population card quality rule."""

    checkpoint_path = Path(checkpoint)
    root = Path(replay_root)
    population_path = (
        root / "population.json"
        if (root / "population.json").is_file()
        else root / "final_scope.json"
    )
    population = load_json(population_path)
    summary = load_json(root / "run_summary.json")
    events = population.get("events") or [
        {
            "case_id": f"{scope}::{context['context_id']}",
            "context_id": context["context_id"],
            "instance_id": context["instance_id"],
            "round_index": int(context["round_index"]),
            "scope": scope,
        }
        for context in population.get("contexts") or []
        for scope in context_scopes(context)
    ]
    if int(summary.get("targets") or 0) != len(events):
        raise ValueError("Full replay summary is incomplete for its frozen population")
    contexts = {
        str(row["context_id"]): row
        for row in population.get("contexts") or []
    }
    results = []
    for event in events:
        path = root / "runs" / case_slug(
            event["context_id"], event["scope"]
        ) / "result.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        result = load_json(path)
        if result.get("status") not in {"success", "judge_error"}:
            raise ValueError(f"Incomplete full replay result: {event['case_id']}")
        results.append(result)

    result_by_card: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    baseline_by_scope: dict[str, list[float]] = defaultdict(list)
    candidate_by_scope: dict[str, list[float]] = defaultdict(list)
    for result in results:
        scope = result["scope"]
        candidate_metrics = result.get("candidate_metrics") or result.get("metrics") or {}
        pairwise = float(candidate_metrics["pairwise_accuracy"])
        baseline_metrics = result.get("baseline_metrics")
        if baseline_metrics is None:
            context = contexts.get(str(result["context_id"])) or {}
            baseline_metrics = (context.get("scope_metrics") or {}).get(scope)
        if baseline_metrics is not None:
            baseline_by_scope[scope].append(
                float(baseline_metrics["pairwise_accuracy"])
            )
        candidate_by_scope[scope].append(pairwise)
        for experience_id in result.get("retrieved_experience_ids") or []:
            result_by_card[(scope, experience_id)].append(result)

    summaries = {}
    for context_id in {str(row["context_id"]) for row in results}:
        path = root / "query_summaries" / case_slug(
            context_id, "summary"
        ) / "result.json"
        if path.is_file():
            summaries[context_id] = load_json(path).get("summary") or {}

    keyword_bank = load_json(
        checkpoint_path / "retrieval/keyword_bank.json"
    )["bank"]

    def select_keywords(
        scope: str,
        experience_id: str,
        observed: list[dict[str, Any]],
    ) -> dict[str, Any]:
        record = keyword_bank[scope][experience_id]
        candidates = list(record.get("selected_keywords") or [])
        evidence = []
        for phrase in candidates:
            matched = []
            for result in observed:
                context_id = str(result["context_id"])
                context = contexts.get(context_id) or {}
                visible = {
                    "trajectory": context.get("view") or {},
                    "query_summary": summaries.get(context_id) or {},
                }
                if phrase_match(phrase, visible):
                    matched.append(result)
            accuracies = [
                float(
                    (row.get("candidate_metrics") or row.get("metrics"))[
                        "pairwise_accuracy"
                    ]
                )
                for row in matched
            ]
            mean_pairwise = mean(accuracies) if accuracies else None
            evidence.append(
                {
                    "keyword": phrase,
                    "joint_replay_hits": len(matched),
                    "mean_judge_pairwise_accuracy": mean_pairwise,
                    "selected": bool(
                        accuracies
                        and mean_pairwise is not None
                        and mean_pairwise >= threshold
                    ),
                }
            )
        selected = [row["keyword"] for row in evidence if row["selected"]]
        if selected:
            policy = "positive_joint_replay_evidence"
        else:
            # The full-document channel can retrieve a useful card even when no
            # literal candidate phrase occurs in the visible query. Preserve the
            # replayed union in that unidentifiable case instead of inventing an
            # unsupported keyword deletion.
            selected = candidates
            policy = "conservative_union_when_no_keyword_is_attributable"
        if not selected:
            raise ValueError(f"{scope}:{experience_id}: no keyword candidates")
        return {
            "selected_keywords": selected,
            "policy": policy,
            "candidates": evidence,
        }

    evaluated_scopes = {event["scope"] for event in events}
    decisions = []
    for scope in SCOPES:
        for card in load_json(
            checkpoint_path / f"bank/{scope}/experience_bank.json"
        )["experiences"]:
            experience_id = card["experience_id"]
            observed = result_by_card.get((scope, experience_id), [])
            values = [
                float(
                    (row.get("candidate_metrics") or row.get("metrics"))[
                        "pairwise_accuracy"
                    ]
                )
                for row in observed
            ]
            mean_pairwise = mean(values) if values else None
            if scope not in evaluated_scopes:
                keep = True
                reason = "scope_not_evaluated"
            else:
                keep = bool(
                    values
                    and mean_pairwise is not None
                    and mean_pairwise >= threshold
                )
                reason = (
                    "kept"
                    if keep
                    else "never_retrieved"
                    if not values
                    else f"mean_judge_pairwise_accuracy_below_{threshold:g}"
                )
            decisions.append(
                {
                    "scope": scope,
                    "experience_id": experience_id,
                    "title": card.get("title"),
                    "keep": keep,
                    "reason": reason,
                    "retrieval_count": len(values),
                    "mean_judge_pairwise_accuracy": mean_pairwise,
                    "keyword_selection": select_keywords(
                        scope, experience_id, observed
                    ),
                }
            )
    aggregate = {
        scope: {
            "events": len(candidate_by_scope[scope]),
            "baseline_mean_pairwise_accuracy": (
                mean(baseline_by_scope[scope])
                if baseline_by_scope[scope]
                else None
            ),
            "candidate_mean_pairwise_accuracy": (
                mean(candidate_by_scope[scope])
                if candidate_by_scope[scope]
                else None
            ),
        }
        for scope in SCOPES
    }
    kept_counts = Counter(row["scope"] for row in decisions if row["keep"])
    source_counts = {
        scope: len(
            load_json(
                checkpoint_path / f"bank/{scope}/experience_bank.json"
            )["experiences"]
        )
        for scope in SCOPES
    }
    return {
        "schema_version": 1,
        "pipeline": "full_population_quality_filter",
        "source_checkpoint": str(checkpoint_path),
        "population": {
            "dataset_role": population.get("dataset_role") or "validation",
            "definition": population.get("definition")
            or population.get("population_definition"),
            "context_count": population.get("context_count")
            or (population.get("counts") or {}).get("contexts")
            or len(contexts),
            "instance_count": population.get("instance_count")
            or (population.get("counts") or {}).get("instances"),
            "event_count": population.get("event_count")
            or (population.get("counts") or {}).get("events")
            or len(events),
            "scope_counts": population.get("scope_counts")
            or {
                scope: sum(row["scope"] == scope for row in events)
                for scope in SCOPES
            },
        },
        "policy": {
            "retrieval_count": "> 0",
            "mean_judge_pairwise_accuracy": f">= {threshold}",
            "source": "frozen full-population joint-judge replay",
        },
        "aggregate": aggregate,
        "source_counts": source_counts,
        "kept_counts": {
            scope: kept_counts.get(scope, 0)
            for scope in SCOPES
        },
        "removed_counts": {
            scope: source_counts[scope] - kept_counts.get(scope, 0)
            for scope in SCOPES
        },
        "decisions": decisions,
    }


def filter_full_replay(
    *,
    checkpoint: str | Path,
    replay_root: str | Path,
    output: str | Path,
    threshold: float = FINAL_PAIRWISE_THRESHOLD,
) -> dict[str, Any]:
    """Atomically materialize the whole-population-filtered final checkpoint."""

    source = Path(checkpoint)
    destination = Path(output)
    filter_result = full_replay_decisions(
        checkpoint=source,
        replay_root=replay_root,
        threshold=threshold,
    )
    temporary = destination.with_name(destination.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary)
    decisions = {
        (row["scope"], row["experience_id"]): row
        for row in filter_result["decisions"]
    }
    removed = []
    keyword = load_json(temporary / "retrieval/keyword_bank.json")
    for scope in SCOPES:
        bank_path = temporary / f"bank/{scope}/experience_bank.json"
        bank = load_json(bank_path)
        kept = []
        for card in bank["experiences"]:
            decision = decisions[(scope, card["experience_id"])]
            if decision["keep"]:
                kept.append(card)
                keyword["bank"][scope][card["experience_id"]][
                    "selected_keywords"
                ] = decision["keyword_selection"]["selected_keywords"]
                continue
            removed.append(copy.deepcopy(decision))
        if not kept:
            raise ValueError(f"Full-population filter removed the entire {scope} bank")
        active_ids = {row["experience_id"] for row in kept}
        bank["experiences"] = kept
        bank["experience_count"] = len(kept)
        keyword["bank"][scope] = {
            experience_id: value
            for experience_id, value in keyword["bank"][scope].items()
            if experience_id in active_ids
        }
        atomic_json(bank_path, bank)
    atomic_json(temporary / "retrieval/keyword_bank.json", keyword)
    counts = update_counts(temporary)
    if destination.exists():
        backup = destination.with_name(destination.name + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        destination.replace(backup)
    temporary.replace(destination)
    return {
        "schema_version": 1,
        "scope_counts": counts,
        "experience_count": sum(counts.values()),
        "filter_result": filter_result,
    }
