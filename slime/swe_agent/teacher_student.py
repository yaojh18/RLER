from __future__ import annotations

import argparse
import json
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
    run_dir: Path | None = None,
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
    if run_dir is None:
        run_dir = run_search(args, [instance_id])
    bundle = SFTDataExporter(run_dir=run_dir).export_bundle()
    bundle.metadata.update({
        "teacher_backend": teacher_backend,
        "teacher_model_name": teacher_model_name,
    })
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run teacher-student search and export SFT data.")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--student-model-name", required=True)
    parser.add_argument("--teacher-api-key", default=os.environ.get("GEMINI_API_KEY", "dummy"))
    parser.add_argument("--teacher-model-name", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--student-backend", default="slime")
    parser.add_argument("--teacher-backend", default="openai")
    parser.add_argument("--subset", default="rebench_v2")
    parser.add_argument("--split", default="train")
    parser.add_argument("--completion-max-tokens", type=int)
    parser.add_argument("--m", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--p", type=int)
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)

    bundle = collect_teacher_student_export(
        instance_id=args.instance_id,
        output_root=args.output_root,
        teacher_api_key=args.teacher_api_key,
        student_model_name=args.student_model_name,
        teacher_model_name=args.teacher_model_name,
        student_backend=args.student_backend,
        teacher_backend=args.teacher_backend,
        subset=args.subset,
        split=args.split,
        completion_max_tokens=args.completion_max_tokens,
        m=args.m,
        k=args.k,
        p=args.p,
        max_rounds=args.max_rounds,
        run_dir=args.run_dir,
    )

    print(json.dumps({
        "instance_id": bundle.instance_id,
        "run_dir": bundle.run_dir,
        "accepted_group_ids": bundle.accepted_group_ids,
        "policy_sample_count": len(bundle.policy_samples),
        "rubric_sample_count": len(bundle.rubric_samples),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
