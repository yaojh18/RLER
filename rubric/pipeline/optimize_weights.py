#!/usr/bin/env python3
"""Fast offline heuristic for local rubric weights under a given score mapping.

The dataset is already frozen, so this implementation deliberately ignores rubric
text while searching. It converts every trajectory pair into numeric raw-score
histograms once, validates the supplied mapping against the legal primitive
domain, and uses successive weight-search stages. The result is explicitly
heuristic and must never be reported as an exact global optimum.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import itertools
import json
import math
import os
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from RLER.rubric.pipeline.common import (
    atomic_json,
    load_json,
    rubric_key,
)
SCHEMA = "heuristic_weight_optimization.v1"
ACTIVE_VALUES = np.asarray((0, *range(5, 51)), dtype=np.int16)
MAX_ACTIVE = 6


@dataclass(frozen=True)
class NumericProblem:
    instance_id: str
    status: str
    candidate_ids: tuple[str, ...]
    candidate_rubrics: tuple[dict[str, Any], ...]
    directions: np.ndarray
    pairs: tuple[dict[str, Any], ...]
    base_weights: tuple[int, ...]
    base_handbook: dict[str, Any]


@dataclass(frozen=True)
class PanelFeatures:
    raw_deltas: np.ndarray
    nominal_raw_deltas: np.ndarray
    pair_denominators: np.ndarray
    draws: int
    seed: int


@dataclass(frozen=True)
class Score:
    robust_correct: int
    robust_denominator: int
    cvar10_correct: int
    cvar10_denominator: int
    nominal_correct: int
    signed_margin: float
    negative_active: int
    minimum_to_maximum_ratio: float
    negative_l1: float
    per_draw_correct: tuple[int, ...]

    def key(self) -> tuple[int | float, ...]:
        return (
            self.robust_correct,
            self.cvar10_correct,
            self.nominal_correct,
            self.signed_margin,
            self.negative_active,
            self.minimum_to_maximum_ratio,
            self.negative_l1,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-manifest", type=Path, required=True)
    parser.add_argument("--base-handbook-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(64, os.cpu_count() or 1)))
    parser.add_argument("--screen-draws", type=int, default=10)
    parser.add_argument("--refine-draws", type=int, default=100)
    parser.add_argument("--final-draws", type=int, default=200)
    parser.add_argument("--validation-draws", type=int, default=2000)
    parser.add_argument("--screen-sweeps", type=int, default=1)
    parser.add_argument("--refine-sweeps", type=int, default=3)
    parser.add_argument("--final-sweeps", type=int, default=5)
    parser.add_argument("--random-restarts", type=int, default=12)
    parser.add_argument(
        "--coordinate-starts",
        type=int,
        default=12,
        help="Maximum number of ranked seeds receiving full coordinate search.",
    )
    parser.add_argument("--train-seed", type=int, default=20260814)
    parser.add_argument("--validation-seed", type=int, default=20260815)
    parser.add_argument(
        "--fixed-score-mapping",
        default="1,2,4,6,8",
        help="Given five-value score mapping; this pipeline optimizes weights only.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def handbook_references(payload: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        references = payload["handbook"]["rubric_and_weight_guidance"][
            "reference_golden_rubrics"
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError("handbook lacks reference_golden_rubrics") from exc
    if not isinstance(references, list) or not references:
        raise ValueError("handbook reference set must be non-empty")
    return references


def canonicalize_weights(values: Sequence[int | float], *, active_limit: int = MAX_ACTIVE) -> tuple[int, ...]:
    array = np.asarray(values, dtype=np.float64)
    array = np.where(array > 0, array, 0.0)
    if np.count_nonzero(array) > active_limit:
        keep = np.argsort(-array, kind="stable")[:active_limit]
        mask = np.zeros(len(array), dtype=bool)
        mask[keep] = True
        array = np.where(mask, array, 0.0)
    if not np.any(array > 0):
        array[0] = 1.0
    # Exact/panel warm starts already live on the legal integer-tenths grid.
    # Rescaling a vector whose maximum is below 50 changes its discrete ratio
    # after rounding and can flip pair predictions, so only scale true legacy
    # handbook weights that exceed the target 0--5 range.
    if float(array.max()) > 50.0:
        array *= 50.0 / float(array.max())
    output = np.rint(array).astype(np.int16)
    output[(output > 0) & (output < 5)] = 5
    output[output > 50] = 50
    if np.count_nonzero(output) > active_limit:
        raise AssertionError("canonicalization exceeded the active limit")
    return tuple(int(value) for value in output)


def load_numeric_problem(problem_path: Path, base_handbook_path: Path) -> NumericProblem:
    problem = load_json(problem_path)
    base = load_json(base_handbook_path)
    instance_id = problem.get("instance_id")
    if (
        problem.get("schema_version") != "robust_scale_problem.v1"
        or not isinstance(instance_id, str)
        or base.get("instance_id") != instance_id
    ):
        raise ValueError(f"problem/base mismatch: {problem_path}")
    candidate_ids = tuple(problem.get("candidate_ids") or ())
    rubrics = tuple(problem.get("candidate_rubrics") or ())
    directions = np.asarray(problem.get("directions") or (), dtype=np.int8)
    if (
        not candidate_ids
        or len(candidate_ids) != len(rubrics)
        or len(candidate_ids) != len(directions)
        or len(candidate_ids) != len(set(candidate_ids))
        or any(
            rubric_key(rubric) != candidate
            for rubric, candidate in zip(rubrics, candidate_ids)
        )
    ):
        raise ValueError(f"invalid candidates: {problem_path}")

    member_to_keep: dict[str, str] = {}
    for cluster in problem.get("semantic_clusters") or []:
        keep = cluster.get("keep_contract")
        for member in cluster.get("members") or []:
            if member in member_to_keep and member_to_keep[member] != keep:
                raise ValueError(f"overlapping semantic cluster: {instance_id}/{member}")
            member_to_keep[member] = keep
    index = {candidate: position for position, candidate in enumerate(candidate_ids)}
    numeric = np.zeros(len(candidate_ids), dtype=np.float64)
    for rubric in handbook_references(base):
        source = rubric_key(rubric)
        keep = member_to_keep.get(source, source)
        if keep not in index:
            raise ValueError(f"base rubric is absent from semantic candidates: {instance_id}/{source}")
        numeric[index[keep]] += float(rubric["weight"])
    if np.any(numeric > 0):
        numeric *= 50.0 / float(numeric.max())
    else:
        initial = [
            Fraction(int(value["numerator"]), int(value["denominator"]))
            for value in problem["initial_weight_tenths_exact"]
        ]
        numeric = np.asarray([float(value) for value in initial], dtype=np.float64)
    base_weights = canonicalize_weights(numeric)
    return NumericProblem(
        instance_id=instance_id,
        status=str(problem.get("status")),
        candidate_ids=candidate_ids,
        candidate_rubrics=rubrics,
        directions=directions,
        pairs=tuple(problem.get("pairs") or ()),
        base_weights=base_weights,
        base_handbook=base,
    )


def _balanced_offsets(draws: int, samples: int, material: str) -> np.ndarray:
    if draws < 10 or draws % 10:
        raise ValueError("draws must be a positive multiple of 10")
    count = draws // 10
    output = np.zeros((draws, samples), dtype=np.int8)
    for sample_index in range(samples):
        values = np.concatenate(
            (
                np.full(count, -1, dtype=np.int8),
                np.zeros(draws - 2 * count, dtype=np.int8),
                np.full(count, 1, dtype=np.int8),
            )
        )
        digest = hashlib.sha256(f"{material}|sample={sample_index}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        rng.shuffle(values)
        output[:, sample_index] = values
    return output


def build_panel_features(problem: NumericProblem, *, draws: int, seed: int) -> PanelFeatures:
    pair_count = len(problem.pairs)
    candidate_count = len(problem.candidate_ids)
    raw = np.zeros((draws, pair_count, candidate_count, 5), dtype=np.int64)
    nominal = np.zeros((pair_count, candidate_count, 5), dtype=np.int64)
    denominators = np.zeros(pair_count, dtype=np.int64)
    offsets_by_cell: dict[tuple[str, str, str], np.ndarray] = {}
    raw_by_cell: dict[tuple[str, str, str], tuple[int, ...]] = {}
    def mutated_histograms(
        *, group_id: str, node_id: str, candidate_id: str, samples: tuple[int, ...]
    ) -> np.ndarray:
        key = (group_id, node_id, candidate_id)
        offsets = offsets_by_cell.get(key)
        if offsets is None:
            material = (
                f"panel_exact_balanced_offsets.v1|seed={seed}|instance={problem.instance_id}|"
                f"group={group_id}|node={node_id}|rubric={candidate_id}|raw={samples}"
            )
            offsets = _balanced_offsets(draws, len(samples), material)
            offsets_by_cell[key] = offsets
            raw_by_cell[key] = samples
        elif raw_by_cell[key] != samples:
            raise ValueError(f"shared raw sample mismatch: {key}")
        values = np.clip(np.asarray(samples, dtype=np.int8)[None, :] + offsets, 1, 5)
        return np.stack([(values == score).sum(axis=1) for score in range(1, 6)], axis=1)

    for pair_index, pair in enumerate(problem.pairs):
        left_all = pair["left_raw_samples"]
        right_all = pair["right_raw_samples"]
        cell_denominators = [len(left) * len(right) for left, right in zip(left_all, right_all)]
        common = math.lcm(*cell_denominators)
        denominators[pair_index] = common
        for candidate_index, candidate_id in enumerate(problem.candidate_ids):
            left = tuple(int(value) for value in left_all[candidate_index])
            right = tuple(int(value) for value in right_all[candidate_index])
            sign = int(pair["gt_sign"]) * int(problem.directions[candidate_index])
            scale = common // (len(left) * len(right))
            left_hist = mutated_histograms(
                group_id=str(pair["group_id"]),
                node_id=str(pair["left_node_id"]),
                candidate_id=candidate_id,
                samples=left,
            )
            right_hist = mutated_histograms(
                group_id=str(pair["group_id"]),
                node_id=str(pair["right_node_id"]),
                candidate_id=candidate_id,
                samples=right,
            )
            raw[:, pair_index, candidate_index, :] = (
                sign * (len(right) * left_hist - len(left) * right_hist) * scale
            )
            left_nominal = np.bincount(np.asarray(left) - 1, minlength=5)
            right_nominal = np.bincount(np.asarray(right) - 1, minlength=5)
            nominal[pair_index, candidate_index, :] = (
                sign * (len(right) * left_nominal - len(left) * right_nominal) * scale
            )
    return PanelFeatures(raw, nominal, denominators, draws, seed)


def mapped_deltas(features: PanelFeatures, mapping: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    vector = np.asarray(mapping, dtype=np.int64)
    return (
        np.tensordot(features.raw_deltas, vector, axes=([3], [0])),
        np.tensordot(features.nominal_raw_deltas, vector, axes=([2], [0])),
    )


def _score_many(
    weights: np.ndarray,
    deltas: np.ndarray,
    nominal: np.ndarray,
    denominators: np.ndarray,
    *,
    score_range: int,
    base_weights: tuple[int, ...],
) -> list[Score]:
    if weights.ndim != 2:
        raise ValueError("weight matrix must be two-dimensional")
    margins = np.einsum("dpk,mk->mdp", deltas, weights, optimize=True)
    correct = margins > 0
    robust = correct.sum(axis=(1, 2))
    per_draw = correct.sum(axis=2)
    tail_count = max(1, math.ceil(deltas.shape[0] * 0.1))
    cvar = np.sort(per_draw, axis=1)[:, :tail_count].sum(axis=1)
    nominal_margins = np.einsum("pk,mk->mp", nominal, weights, optimize=True)
    nominal_correct = (nominal_margins > 0).sum(axis=1)
    totals = weights.sum(axis=1)
    normalized = nominal_margins / (
        totals[:, None] * score_range * denominators[None, :]
    )
    margin = normalized.sum(axis=1)
    base = np.asarray(base_weights, dtype=np.float64)
    base /= base.sum()
    output = []
    for row in range(len(weights)):
        active = weights[row] > 0
        current = weights[row].astype(np.float64) / float(totals[row])
        ratio = float(weights[row][active].min() / weights[row][active].max())
        output.append(
            Score(
                robust_correct=int(robust[row]),
                robust_denominator=int(deltas.shape[0] * deltas.shape[1]),
                cvar10_correct=int(cvar[row]),
                cvar10_denominator=int(tail_count * deltas.shape[1]),
                nominal_correct=int(nominal_correct[row]),
                signed_margin=float(margin[row]),
                negative_active=-int(active.sum()),
                minimum_to_maximum_ratio=ratio,
                negative_l1=-float(np.abs(current - base).sum()),
                per_draw_correct=tuple(int(value) for value in per_draw[row]),
            )
        )
    return output


def _best_index(scores: Sequence[Score], weights: Sequence[tuple[int, ...]]) -> int:
    return max(range(len(scores)), key=lambda index: (scores[index].key(), tuple(-x for x in weights[index])))


def _unique_weight_rows(rows: Iterable[Sequence[int]], width: int) -> list[tuple[int, ...]]:
    output: list[tuple[int, ...]] = []
    seen = set()
    for row in rows:
        canonical = canonicalize_weights(row)
        if len(canonical) != width or canonical in seen:
            continue
        seen.add(canonical)
        output.append(canonical)
    return output


def coordinate_search(
    deltas: np.ndarray,
    nominal: np.ndarray,
    denominators: np.ndarray,
    *,
    mapping: tuple[int, int, int, int, int],
    base_weights: tuple[int, ...],
    seeds: Sequence[Sequence[int]],
    sweeps: int,
    swap_rounds: int,
    coordinate_starts: int,
) -> tuple[tuple[int, ...], Score, int]:
    width = deltas.shape[2]
    seed_rows = _unique_weight_rows(seeds, width)
    if not seed_rows:
        seed_rows = [canonicalize_weights(base_weights)]
    seed_array = np.asarray(seed_rows, dtype=np.int16)
    seed_scores = _score_many(
        seed_array,
        deltas,
        nominal,
        denominators,
        score_range=mapping[-1] - mapping[0],
        base_weights=base_weights,
    )
    evaluations = len(seed_rows)
    order = sorted(range(len(seed_rows)), key=lambda i: seed_scores[i].key(), reverse=True)
    starts = [
        seed_rows[index]
        for index in order[: min(coordinate_starts, len(order))]
    ]
    best_weight = starts[0]
    best_score = seed_scores[seed_rows.index(best_weight)]

    for start in starts:
        current = np.asarray(start, dtype=np.int16)
        current_score = _score_many(
            current[None, :], deltas, nominal, denominators,
            score_range=mapping[-1] - mapping[0], base_weights=base_weights,
        )[0]
        evaluations += 1
        for _ in range(sweeps):
            changed = False
            for candidate in range(width):
                matrix = np.repeat(current[None, :], len(ACTIVE_VALUES), axis=0)
                matrix[:, candidate] = ACTIVE_VALUES
                active = (matrix > 0).sum(axis=1)
                matrix = matrix[(active >= 1) & (active <= MAX_ACTIVE)]
                rows = _unique_weight_rows(matrix, width)
                scores = _score_many(
                    np.asarray(rows, dtype=np.int16), deltas, nominal, denominators,
                    score_range=mapping[-1] - mapping[0], base_weights=base_weights,
                )
                evaluations += len(rows)
                chosen = _best_index(scores, rows)
                if scores[chosen].key() > current_score.key():
                    current = np.asarray(rows[chosen], dtype=np.int16)
                    current_score = scores[chosen]
                    changed = True
            if not changed:
                break

        for _ in range(swap_rounds):
            active_indices = np.flatnonzero(current > 0)
            inactive_indices = np.flatnonzero(current == 0)
            neighbors = []
            for old, new in itertools.product(active_indices, inactive_indices):
                for value in (5, 10, 20, 30, 40, 50, int(current[old])):
                    candidate = current.copy()
                    candidate[old] = 0
                    candidate[new] = value
                    neighbors.append(candidate)
            rows = _unique_weight_rows(neighbors, width)
            if not rows:
                break
            scores = _score_many(
                np.asarray(rows, dtype=np.int16), deltas, nominal, denominators,
                score_range=mapping[-1] - mapping[0], base_weights=base_weights,
            )
            evaluations += len(rows)
            chosen = _best_index(scores, rows)
            if scores[chosen].key() <= current_score.key():
                break
            current = np.asarray(rows[chosen], dtype=np.int16)
            current_score = scores[chosen]

        current_tuple = canonicalize_weights(current)
        if (current_score.key(), tuple(-x for x in current_tuple)) > (
            best_score.key(), tuple(-x for x in best_weight)
        ):
            best_weight, best_score = current_tuple, current_score
    return best_weight, best_score, evaluations


def _random_seeds(instance_id: str, mapping_index: int, width: int, count: int) -> list[tuple[int, ...]]:
    digest = hashlib.sha256(f"heuristic-seeds|{instance_id}|{mapping_index}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    output = []
    for _ in range(count):
        active = int(rng.integers(1, min(MAX_ACTIVE, width) + 1))
        indices = rng.choice(width, size=active, replace=False)
        weights = np.zeros(width, dtype=np.int16)
        weights[indices] = rng.integers(5, 51, size=active)
        output.append(canonicalize_weights(weights))
    return output


def seed_portfolio(
    problem: NumericProblem,
    deltas: np.ndarray,
    nominal: np.ndarray,
    denominators: np.ndarray,
    mapping: tuple[int, int, int, int, int],
    *,
    mapping_index: int,
    random_restarts: int,
) -> list[tuple[int, ...]]:
    width = len(problem.candidate_ids)
    singles = []
    for index in range(width):
        row = [0] * width
        row[index] = 50
        singles.append(tuple(row))
    single_scores = _score_many(
        np.asarray(singles, dtype=np.int16), deltas, nominal, denominators,
        score_range=mapping[-1] - mapping[0], base_weights=problem.base_weights,
    )
    ranking = sorted(range(width), key=lambda i: single_scores[i].key(), reverse=True)
    output: list[Sequence[int]] = [problem.base_weights]
    output.extend(singles[index] for index in ranking[: min(4, width)])
    for active in range(2, min(MAX_ACTIVE, width) + 1):
        row = [0] * width
        for index in ranking[:active]:
            row[index] = 50
        output.append(row)
    output.extend(_random_seeds(problem.instance_id, mapping_index, width, random_restarts))
    return _unique_weight_rows(output, width)


def _score_payload(score: Score) -> dict[str, Any]:
    return {
        "robust_correct": score.robust_correct,
        "robust_denominator": score.robust_denominator,
        "robust_accuracy": score.robust_correct / score.robust_denominator,
        "cvar10_correct": score.cvar10_correct,
        "cvar10_denominator": score.cvar10_denominator,
        "cvar10_accuracy": score.cvar10_correct / score.cvar10_denominator,
        "nominal_correct": score.nominal_correct,
        "signed_margin": score.signed_margin,
        "active_rubrics": -score.negative_active,
        "minimum_to_maximum_weight_ratio": score.minimum_to_maximum_ratio,
        "normalized_l1_from_base": -score.negative_l1,
        "per_draw_correct": list(score.per_draw_correct),
    }


def _worker_stage(task: dict[str, Any]) -> dict[str, Any]:
    problem = load_numeric_problem(Path(task["problem_path"]), Path(task["base_path"]))
    if problem.status != "OPTIMIZE":
        return {"instance_id": problem.instance_id, "status": "NO_TRAINING_SIGNAL", "results": {}}
    features = build_panel_features(problem, draws=task["draws"], seed=task["seed"])
    results = {}
    incoming = task.get("incoming") or {}
    for mapping_index in task["mapping_indices"]:
        mapping = tuple(task["mappings"][mapping_index])
        deltas, nominal = mapped_deltas(features, mapping)
        if task["mode"] == "validation":
            incoming_weight = incoming.get(str(mapping_index))
            if incoming_weight is None:
                raise ValueError(
                    f"validation lacks frozen final weights: {problem.instance_id}/{mapping_index}"
                )
            rows = _unique_weight_rows([incoming_weight], len(problem.candidate_ids))
            if len(rows) != 1:
                raise ValueError("validation weight did not resolve uniquely")
            score = _score_many(
                np.asarray(rows, dtype=np.int16),
                deltas,
                nominal,
                features.pair_denominators,
                score_range=mapping[-1] - mapping[0],
                base_weights=problem.base_weights,
            )[0]
            results[str(mapping_index)] = {
                "mapping": list(mapping),
                "weight_tenths": list(rows[0]),
                "score": _score_payload(score),
                "evaluations": 1,
                "random_restarts": 0,
                "fixed_weight_evaluation": True,
            }
            continue
        if task["mode"] == "screen":
            seeds = [problem.base_weights]
            random_restarts = 0
            swap_rounds = 0
        else:
            seeds = seed_portfolio(
                problem, deltas, nominal, features.pair_denominators, mapping,
                mapping_index=mapping_index, random_restarts=task["random_restarts"],
            )
            if str(mapping_index) in incoming:
                seeds.insert(0, incoming[str(mapping_index)])
            random_restarts = task["random_restarts"]
            swap_rounds = task["swap_rounds"]
        weights, score, evaluations = coordinate_search(
            deltas, nominal, features.pair_denominators,
            mapping=mapping, base_weights=problem.base_weights, seeds=seeds,
            sweeps=task["sweeps"], swap_rounds=swap_rounds,
            coordinate_starts=task["coordinate_starts"],
        )
        results[str(mapping_index)] = {
            "mapping": list(mapping),
            "weight_tenths": list(weights),
            "score": _score_payload(score),
            "evaluations": evaluations,
            "random_restarts": random_restarts,
            "coordinate_starts": task["coordinate_starts"],
        }
    return {
        "instance_id": problem.instance_id,
        "status": "complete",
        "draws": features.draws,
        "seed": features.seed,
        "results": results,
    }


def _run_stage(
    *,
    stage: str,
    instance_ids: Sequence[str],
    problem_dir: Path,
    base_dir: Path,
    mappings: tuple[tuple[int, int, int, int, int], ...],
    mapping_indices: Sequence[int],
    incoming: dict[str, dict[str, Any]] | None,
    draws: int,
    seed: int,
    sweeps: int,
    swap_rounds: int,
    random_restarts: int,
    coordinate_starts: int,
    workers: int,
    output_dir: Path,
    resume: bool,
) -> dict[str, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    collected: dict[str, dict[str, Any]] = {}
    tasks = []
    for instance_id in instance_ids:
        target = output_dir / f"{instance_id}.json"
        if resume and target.exists():
            payload = load_json(target)
            if payload.get("instance_id") == instance_id:
                collected[instance_id] = payload
                continue
        incoming_results = (incoming or {}).get(instance_id, {}).get("results") or {}
        tasks.append(
            {
                "problem_path": str(problem_dir / f"{instance_id}.json"),
                "base_path": str(base_dir / f"{instance_id}.json"),
                "mappings": [list(mapping) for mapping in mappings],
                "mapping_indices": list(mapping_indices),
                "incoming": {
                    key: value["weight_tenths"]
                    for key, value in incoming_results.items()
                    if int(key) in set(mapping_indices)
                },
                "mode": stage if stage in {"screen", "validation"} else "refine",
                "draws": draws,
                "seed": seed,
                "sweeps": sweeps,
                "swap_rounds": swap_rounds,
                "random_restarts": random_restarts,
                "coordinate_starts": coordinate_starts,
            }
        )
    if tasks:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for payload in executor.map(_worker_stage, tasks, chunksize=1):
                instance_id = payload["instance_id"]
                atomic_json(output_dir / f"{instance_id}.json", payload)
                collected[instance_id] = payload
    if set(collected) != set(instance_ids):
        raise ValueError(f"stage {stage} coverage mismatch")
    return collected


def aggregate_mappings(
    stage_results: dict[str, dict[str, Any]], mapping_indices: Sequence[int]
) -> list[dict[str, Any]]:
    output = []
    for mapping_index in mapping_indices:
        robust = nominal = cvar = active = evaluations = 0
        per_draw: np.ndarray | None = None
        denominator = cvar_denominator = 0
        for payload in stage_results.values():
            item = (payload.get("results") or {}).get(str(mapping_index))
            if item is None:
                continue
            score = item["score"]
            robust += int(score["robust_correct"])
            denominator += int(score["robust_denominator"])
            nominal += int(score["nominal_correct"])
            cvar += int(score["cvar10_correct"])
            cvar_denominator += int(score["cvar10_denominator"])
            active += int(score["active_rubrics"])
            evaluations += int(item["evaluations"])
            draw_values = np.asarray(score["per_draw_correct"], dtype=np.int64)
            per_draw = draw_values if per_draw is None else per_draw + draw_values
        if per_draw is None:
            raise ValueError(f"mapping {mapping_index} has no optimization records")
        tail_count = max(1, math.ceil(len(per_draw) * 0.1))
        global_cvar = int(np.sort(per_draw)[:tail_count].sum())
        output.append(
            {
                "mapping_index": mapping_index,
                "robust_correct": robust,
                "robust_denominator": denominator,
                "robust_accuracy": robust / denominator,
                "global_cvar10_correct": global_cvar,
                "global_cvar10_denominator": int(tail_count * (denominator // len(per_draw))),
                "nominal_correct": nominal,
                "summed_instance_cvar10_correct": cvar,
                "summed_instance_cvar10_denominator": cvar_denominator,
                "active_rubrics": active,
                "evaluations": evaluations,
            }
        )
    output.sort(
        key=lambda item: (
            item["robust_correct"], item["global_cvar10_correct"],
            item["nominal_correct"], -item["active_rubrics"], -item["mapping_index"],
        ),
        reverse=True,
    )
    return output


def _materialize(
    *,
    args: argparse.Namespace,
    instance_ids: Sequence[str],
    problem_dir: Path,
    base_dir: Path,
    selected_mapping_index: int,
    selected_mapping: tuple[int, int, int, int, int],
    final_results: dict[str, dict[str, Any]],
    validation_results: dict[str, dict[str, Any]],
    stage_summaries: dict[str, Any],
    started: float,
) -> Path:
    handbook_dir = args.output_root / "handbooks"
    handbook_dir.mkdir(parents=True, exist_ok=True)
    records = []
    total_validation_correct = total_validation_denominator = 0
    for instance_id in instance_ids:
        problem = load_numeric_problem(problem_dir / f"{instance_id}.json", base_dir / f"{instance_id}.json")
        if problem.status == "OPTIMIZE":
            train = final_results[instance_id]["results"][str(selected_mapping_index)]
            validation = validation_results[instance_id]["results"][str(selected_mapping_index)]
            weights = tuple(int(value) for value in train["weight_tenths"])
            validation_score = validation["score"]
            total_validation_correct += int(validation_score["robust_correct"])
            total_validation_denominator += int(validation_score["robust_denominator"])
        else:
            weights = problem.base_weights
            train = {"score": None, "evaluations": 0}
            validation_score = None
        active = [index for index, weight in enumerate(weights) if weight > 0]
        if not 1 <= len(active) <= MAX_ACTIVE:
            raise ValueError(f"invalid final support: {instance_id}")
        retained = []
        references = []
        for index in active:
            rubric = copy.deepcopy(problem.candidate_rubrics[index])
            rubric["weight"] = round(weights[index] / 10.0, 1)
            references.append(rubric)
            retained.append(
                {
                    "rubric_key": problem.candidate_ids[index],
                    "weight": rubric["weight"],
                }
            )
        removed = [
            {"rubric_key": contract}
            for index, contract in enumerate(problem.candidate_ids)
            if index not in active
        ]
        wrapper = copy.deepcopy(problem.base_handbook)
        wrapper["handbook"]["rubric_and_weight_guidance"]["reference_golden_rubrics"] = references
        wrapper["stage"] = "heuristic_numeric_weight_scale_optimization"
        handbook_path = handbook_dir / f"{instance_id}.json"
        atomic_json(handbook_path, wrapper)
        records.append(
            {
                "instance_id": instance_id,
                "handbook": str(handbook_path),
                "active_rubrics": len(active),
                "retained_rubrics": retained,
                "removed_rubrics": removed,
                "train": train,
                "validation": validation_score,
            }
        )
    result = {
        "schema_version": SCHEMA,
        "status": "complete",
        "search_kind": "successive_halving_coordinate_swap",
        "optimality_claim": "none",
        "score_mapping_mode": "given",
        "fixed_score_mapping": [
            int(value.strip()) for value in args.fixed_score_mapping.split(",")
        ],
        "selected_mapping_index": selected_mapping_index,
        "mapping": list(selected_mapping),
        "instances": len(instance_ids),
        "records": records,
        "validation": {
            "mutated_correct": total_validation_correct,
            "mutated_denominator": total_validation_denominator,
            "mutated_accuracy": (
                total_validation_correct / total_validation_denominator
                if total_validation_denominator else None
            ),
            "draws": args.validation_draws,
            "seed": args.validation_seed,
        },
        "search_parameters": {
            "workers": args.workers,
            "screen_draws": args.screen_draws,
            "refine_draws": args.refine_draws,
            "final_draws": args.final_draws,
            "validation_draws": args.validation_draws,
            "screen_sweeps": args.screen_sweeps,
            "refine_sweeps": args.refine_sweeps,
            "final_sweeps": args.final_sweeps,
            "random_restarts": args.random_restarts,
            "coordinate_starts": args.coordinate_starts,
            "active_weight_values": [0, *range(5, 51)],
            "max_active_rubrics": MAX_ACTIVE,
        },
        "stage_summaries": stage_summaries,
        "elapsed_seconds": time.monotonic() - started,
    }
    result_path = args.output_root / "optimization_result.json"
    atomic_json(result_path, result)
    return result_path


def main() -> None:
    args = parse_args()
    if args.coordinate_starts <= 0:
        raise ValueError("--coordinate-starts must be positive")
    started = time.monotonic()
    if args.output_root.exists() and not args.resume:
        raise ValueError(f"output root exists; use --resume intentionally: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    problem_manifest = load_json(args.problem_manifest)
    if problem_manifest.get("schema_version") != "robust_scale_problem_collection.v1":
        raise ValueError("unsupported problem collection")
    instance_ids = list(problem_manifest["instance_ids"])
    problem_dir = args.problem_manifest.parent / "instances"
    requested_mapping = tuple(
        int(value.strip()) for value in args.fixed_score_mapping.split(",")
    )
    if len(requested_mapping) != 5 or any(
        left >= right for left, right in zip(requested_mapping, requested_mapping[1:])
    ):
        raise ValueError("--fixed-score-mapping must contain five increasing integers")
    mappings = (requested_mapping,)
    mapping_indices = list(range(len(mappings)))
    common = dict(
        instance_ids=instance_ids,
        problem_dir=problem_dir,
        base_dir=args.base_handbook_dir,
        mappings=mappings,
        coordinate_starts=args.coordinate_starts,
        workers=args.workers,
        resume=args.resume,
    )
    screen = _run_stage(
        stage="screen", mapping_indices=mapping_indices, incoming=None,
        draws=args.screen_draws, seed=args.train_seed, sweeps=args.screen_sweeps,
        swap_rounds=0, random_restarts=0,
        output_dir=args.output_root / "screen", **common,
    )
    screen_summary = aggregate_mappings(screen, mapping_indices)
    shortlist = [item["mapping_index"] for item in screen_summary]
    refine = _run_stage(
        stage="refine", mapping_indices=shortlist, incoming=screen,
        draws=args.refine_draws, seed=args.train_seed, sweeps=args.refine_sweeps,
        swap_rounds=1, random_restarts=args.random_restarts,
        output_dir=args.output_root / "refine", **common,
    )
    refine_summary = aggregate_mappings(refine, shortlist)
    finalists = [item["mapping_index"] for item in refine_summary]
    final = _run_stage(
        stage="final", mapping_indices=finalists, incoming=refine,
        draws=args.final_draws, seed=args.train_seed, sweeps=args.final_sweeps,
        swap_rounds=2, random_restarts=args.random_restarts * 2,
        output_dir=args.output_root / "final", **common,
    )
    final_summary = aggregate_mappings(final, finalists)
    selected_mapping_index = int(final_summary[0]["mapping_index"])
    selected_mapping = mappings[selected_mapping_index]
    validation = _run_stage(
        stage="validation", mapping_indices=[selected_mapping_index], incoming=final,
        draws=args.validation_draws, seed=args.validation_seed, sweeps=0,
        swap_rounds=0, random_restarts=0,
        output_dir=args.output_root / "validation", **common,
    )
    stage_summaries = {
        "screen": screen_summary,
        "screen_shortlist": shortlist,
        "refine": refine_summary,
        "finalists": finalists,
        "final": final_summary,
    }
    atomic_json(args.output_root / "search_trace.json", stage_summaries)
    result_path = _materialize(
        args=args,
        instance_ids=instance_ids,
        problem_dir=problem_dir,
        base_dir=args.base_handbook_dir,
        selected_mapping_index=selected_mapping_index,
        selected_mapping=selected_mapping,
        final_results=final,
        validation_results=validation,
        stage_summaries=stage_summaries,
        started=started,
    )
    print(
        json.dumps(
            {
                "status": load_json(result_path)["status"],
                "result": str(result_path),
            }
        )
    )


if __name__ == "__main__":
    main()
