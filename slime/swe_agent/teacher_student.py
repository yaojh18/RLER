from __future__ import annotations

import argparse
import os
from pathlib import Path

from swe_agent.run.search_swe_agent import build_arg_parser, get_search_run_dir, run_search

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
    student_backend: str = "slime",
    completion_max_tokens: int | None = None,
    m: int | None = None,
    k: int | None = None,
    p: int | None = None,
    max_rounds: int | None = None,
) -> argparse.Namespace:
    args = build_arg_parser().parse_args([])
    args.backend = backend
    args.instance_id = [instance_id]
    args.subset = subset
    args.split = split
    args.output_root = output_root
    args.resume_run_dir = None
    args.workers = 1
    if completion_max_tokens is not None:
        args.completion_max_tokens = completion_max_tokens
    if m is not None:
        args.m = m
    if k is not None:
        args.k = k
    if p is not None:
        args.p = p
    if max_rounds is not None:
        args.max_rounds = max_rounds
    if backend == "slime":
        args.slime_model = model_name
    elif backend == "openai":
        args.openai_model = model_name
    else:
        args.vllm_model = model_name
    args.student_backend = student_backend
    args.student_model = student_model_name
    return args


def collect_teacher_student_export(
    *,
    instance_id: str,
    output_root: Path,
    teacher_api_key: str,
    student_model_name: str,
    teacher_model_name: str = DEFAULT_TEACHER_MODEL,
    student_backend: str = "slime",
    teacher_backend: str = "openai",
    subset: str = "rebench_v2",
    split: str = "train",
    completion_max_tokens: int | None = None,
    m: int | None = None,
    k: int | None = None,
    p: int | None = None,
    max_rounds: int | None = None,
) -> SFTExportBundle:
    if teacher_model_name.startswith("gemini/"):
        os.environ["GEMINI_API_KEY"] = teacher_api_key
    else:
        raise ValueError(f"Unsupported teacher model: {teacher_model_name}")

    args = build_search_args(
        backend=teacher_backend,
        instance_id=instance_id,
        subset=subset,
        split=split,
        output_root=output_root,
        model_name=teacher_model_name,
        student_model_name=student_model_name,
        student_backend=student_backend,
        m=m,
        k=k,
        p=p,
        max_rounds=max_rounds,
        completion_max_tokens=completion_max_tokens,
    )
    run_dir = get_search_run_dir(args, instance_id)
    run_search(args, [instance_id])
    bundle = SFTDataExporter(run_dir=run_dir).export_bundle()
    bundle.metadata.update({
        "teacher_backend": teacher_backend,
        "teacher_model_name": teacher_model_name,
    })
    return bundle
