from __future__ import annotations

import argparse
import os
from pathlib import Path

from swe_agent.run.search_swe_agent import build_arg_parser, run_search

from .contracts import SFTExportBundle
from .data_export import SFTDataExporter


DEFAULT_TEACHER_MODEL = "gemini/gemini-3.1-pro-preview"


def build_search_args(
    *,
    backend: str,
    instance_id: str,
    subset: str,
    split: str,
    output_root: Path,
    model_name: str,
    student_model_name: str | None,
    m: int,
    k: int,
    p: int,
    max_rounds: int,
    completion_max_tokens: int = 4096,
) -> argparse.Namespace:
    args = build_arg_parser().parse_args([])
    args.backend = backend
    args.instance_id = [instance_id]
    args.subset = subset
    args.split = split
    args.output_root = output_root
    args.resume_run_dir = None
    args.workers = 1
    args.completion_max_tokens = completion_max_tokens
    args.m = m
    args.k = k
    args.p = p
    args.max_rounds = max_rounds
    args.calculate_gt_reward = True
    if backend == "slime":
        args.slime_model = model_name
    elif backend == "openai":
        args.openai_model = model_name
    else:
        args.vllm_model = model_name
    args.student_backend = "slime"
    args.student_model = student_model_name
    return args


def collect_teacher_student_export(
    *,
    instance_id: str,
    output_root: Path,
    teacher_api_key: str,
    student_model_name: str,
    teacher_model_name: str = DEFAULT_TEACHER_MODEL,
    subset: str = "rebench_v2",
    split: str = "train",
    m: int = 2,
    k: int = 20,
    p: int = 1,
    max_rounds: int = 5,
    completion_max_tokens: int = 4096,
) -> SFTExportBundle:
    if teacher_model_name.startswith("gemini/"):
        os.environ["GEMINI_API_KEY"] = teacher_api_key
    else:
        os.environ["OPENAI_API_KEY"] = teacher_api_key
    args = build_search_args(
        backend="openai",
        instance_id=instance_id,
        subset=subset,
        split=split,
        output_root=output_root / "mixed",
        model_name=teacher_model_name,
        student_model_name=student_model_name,
        m=m,
        k=k,
        p=p,
        max_rounds=max_rounds,
        completion_max_tokens=completion_max_tokens,
    )
    results = run_search(args, [instance_id])
    if not results:
        raise RuntimeError(f"Search run did not return a result for {instance_id}")
    if results[0].error:
        raise RuntimeError(f"Search run failed with error: {results[0].error}")
    return SFTDataExporter(run_dir=Path(results[0].run_dir)).export_bundle()
