from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from .config import ModelConfig
from .contexts import load_contexts
from .io import atomic_json


def _add_models(parser: argparse.ArgumentParser) -> None:
    defaults = ModelConfig()
    parser.add_argument("--generator-model", default=defaults.generator)
    parser.add_argument("--refiner-model", default=defaults.refiner)
    parser.add_argument("--judge-model", default=defaults.judge)
    parser.add_argument("--temperature", type=float, default=defaults.temperature)
    parser.add_argument(
        "--judge-temperature", type=float, default=defaults.judge_temperature
    )
    parser.add_argument("--top-p", type=float, default=defaults.top_p)
    parser.add_argument("--max-tokens", type=int, default=defaults.max_tokens)
    parser.add_argument(
        "--retry-max-tokens", type=int, default=defaults.retry_max_tokens
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="experience-gen",
        description="Final trajectory-judge experience generation pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-contexts")
    source = prepare.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--plan",
        type=Path,
        help="Canonical calculate-GT plan with completed run records.",
    )
    source.add_argument(
        "--search-root",
        type=Path,
        help="Root containing calculate-GT search run directories.",
    )
    prepare.add_argument(
        "--workspace-root",
        type=Path,
        help="Map plan paths rooted at /workspace into this directory.",
    )
    prepare.add_argument("--output", required=True, type=Path)

    overfits = sub.add_parser("generate-overfits")
    overfits.add_argument(
        "--optimization-contexts",
        dest="contexts",
        nargs="+",
        required=True,
        type=Path,
        help="Optimization contexts used to generate and locally gate experiences.",
    )
    overfits.add_argument("--checkpoint", required=True, type=Path)
    overfits.add_argument("--output", required=True, type=Path)
    overfits.add_argument(
        "--scope", action="append", choices=("siblings", "pc")
    )
    overfits.add_argument("--track", choices=("rubric", "judge"), default="rubric")
    overfits.add_argument("--concurrency", type=int, default=8)
    _add_models(overfits)

    provisional = sub.add_parser("build-provisional")
    provisional.add_argument(
        "--optimization-contexts",
        dest="contexts",
        nargs="+",
        required=True,
        type=Path,
        help="The frozen optimization contexts used by generate-overfits.",
    )
    provisional.add_argument("--base", required=True, type=Path)
    provisional.add_argument("--overfit-summary", required=True, type=Path)
    provisional.add_argument("--output", required=True, type=Path)
    provisional.add_argument("--keyword-output", required=True, type=Path)
    provisional.add_argument("--concurrency", type=int, default=8)
    _add_models(provisional)

    full_replay = sub.add_parser("full-replay")
    full_replay.add_argument(
        "--validation-contexts",
        dest="contexts",
        nargs="+",
        required=True,
        type=Path,
        help="Held-out contexts used only for full-bank replay and global filtering.",
    )
    full_replay.add_argument("--checkpoint", required=True, type=Path)
    full_replay.add_argument("--output", required=True, type=Path)
    full_replay.add_argument(
        "--scope", action="append", choices=("siblings", "pc")
    )
    full_replay.add_argument("--track", choices=("rubric", "judge"), default="rubric")
    full_replay.add_argument("--concurrency", type=int, default=8)
    _add_models(full_replay)

    apply_filter = sub.add_parser("filter-full-replay")
    apply_filter.add_argument("--checkpoint", required=True, type=Path)
    apply_filter.add_argument("--replay-root", required=True, type=Path)
    apply_filter.add_argument("--output", required=True, type=Path)
    apply_filter.add_argument("--threshold", type=float, default=0.5)

    return parser


async def _dispatch_async(args: argparse.Namespace) -> dict[str, Any]:
    models = ModelConfig(
        generator=args.generator_model,
        refiner=args.refiner_model,
        judge=args.judge_model,
        temperature=args.temperature,
        judge_temperature=args.judge_temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        retry_max_tokens=args.retry_max_tokens,
    )
    if args.command == "generate-overfits":
        from .pipeline import generate_overfits

        return await generate_overfits(
            contexts=load_contexts(args.contexts, require_variance=False),
            base_checkpoint=args.checkpoint,
            output=args.output,
            scopes=tuple(args.scope or ("siblings", "pc")),
            track=args.track,
            models=models,
            concurrency=args.concurrency,
        )
    if args.command == "build-provisional":
        from .pipeline import build_provisional_checkpoint

        return await build_provisional_checkpoint(
            contexts=load_contexts(args.contexts, require_variance=False),
            base_checkpoint=args.base,
            overfit_summary=args.overfit_summary,
            output=args.output,
            keyword_output=args.keyword_output,
            models=models,
            concurrency=args.concurrency,
        )
    if args.command == "full-replay":
        from .pipeline import run_full_replay

        return await run_full_replay(
            contexts=load_contexts(args.contexts, require_variance=False),
            checkpoint=args.checkpoint,
            output=args.output,
            scopes=tuple(args.scope or ("siblings", "pc")),
            track=args.track,
            models=models,
            concurrency=args.concurrency,
        )
    raise AssertionError(args.command)


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-contexts":
        from .artifacts import (
            contexts_from_search_runs,
            search_run_records,
        )

        rows = contexts_from_search_runs(
            search_run_records(
                plan=args.plan,
                search_root=args.search_root,
                workspace_root=args.workspace_root,
            )
        )
        result = {
            "schema_version": 1,
            "contexts": rows,
        }
        atomic_json(args.output, result)
    elif args.command == "filter-full-replay":
        from .pipeline import filter_full_replay

        result = filter_full_replay(
            checkpoint=args.checkpoint,
            replay_root=args.replay_root,
            output=args.output,
            threshold=args.threshold,
        )
    else:
        result = asyncio.run(_dispatch_async(args))
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
