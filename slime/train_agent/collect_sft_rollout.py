from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path
from typing import Any

from swe_agent.run.search_swe_agent import build_arg_parser, run_search

from .contracts import SFTExportBundle
from .data_export import ParallelSFTDataExporter, SFTDataExporter


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
    n: int | None = None,
    k: int | None = None,
    p: int | None = None,
    max_rounds: int | None = None,
    step_limit: int | None = None,
) -> argparse.Namespace:
    args = build_arg_parser().parse_args([])
    args.backend = backend
    args.instance_id = [instance_id]
    args.subset = subset
    args.split = split
    args.output_root = output_root
    args.resume_run_dir = None
    args.workers = 1
    args.student_backend = student_backend
    args.student_model = student_model_name
    args.export_grpo_bundles = False
    for name, value in {
        "completion_max_tokens": completion_max_tokens,
        "m": m,
        "n": n,
        "k": k,
        "p": p,
        "max_rounds": max_rounds,
        "step_limit": step_limit,
    }.items():
        if value is not None:
            setattr(args, name, value)
    if backend == "slime":
        args.slime_model = model_name
    elif backend == "openai":
        args.openai_model = model_name
    else:
        args.vllm_model = model_name
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
    n: int | None = None,
    k: int | None = None,
    p: int | None = None,
    max_rounds: int | None = None,
    step_limit: int | None = None,
    run_dir: Path | None = None,
) -> SFTExportBundle:
    if run_dir is None:
        if teacher_backend == "slime":
            pass
        elif teacher_model_name.startswith("gemini/"):
            os.environ["GEMINI_API_KEY"] = teacher_api_key
        elif teacher_model_name.startswith(("gpt-", "openai/")):
            os.environ["OPENAI_API_KEY"] = teacher_api_key
        else:
            raise ValueError(f"Unsupported teacher model: {teacher_model_name}")
        search_args = build_search_args(
            backend=teacher_backend,
            instance_id=instance_id,
            subset=subset,
            split=split,
            output_root=output_root,
            model_name=teacher_model_name,
            student_model_name=student_model_name,
            student_backend=student_backend,
            completion_max_tokens=completion_max_tokens,
            m=m,
            n=n,
            k=k,
            p=p,
            max_rounds=max_rounds,
            step_limit=step_limit,
        )
        result = run_search(search_args, [instance_id])
        run_dir = Path(result[0] if isinstance(result, tuple) else result)

    bundle = SFTDataExporter(run_dir=run_dir).export_bundle()
    bundle.metadata.update(
        {
            "teacher_backend": teacher_backend,
            "teacher_model_name": teacher_model_name,
        }
    )
    return bundle


def collect_teacher_student_export_parallel(
    *,
    instance_id: str,
    output_root: Path,
    teacher_model_name: str,
    teacher_base_url: str,
    teacher_api_key: str = "EMPTY",
    rubric_base_url: str | None = None,
    rubric_api_key: str | None = None,
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    student_model_name: str | None = None,
    subset: str = "rebench_v2",
    split: str = "train",
    m: int = 8,
    n: int = 1,
    max_mid_cps: int = 6,
    steps_per_round: int = 20,
    step_limit: int = 120,
    rubric_temperature: float = 1.0,
    rubric_top_p: float = 0.95,
    rubric_max_tokens: int = 4096,
    judge_temperature: float = 0.1,
    judge_top_p: float = 0.95,
    judge_max_tokens: int = 1024,
    rubric_bank_strategy: str = "score",
    max_active_rubrics: int = 6,
    gt_eval_workers: int = 8,
) -> SFTExportBundle:
    """Drive the v1-lanes parallel search runner against a single instance
    and export an SFT bundle from its on-disk artifacts.

    teacher_base_url: sglang URL of the policy server (Lane A + Lane B).
    rubric_base_url: defaults to teacher_base_url; can point at a separate
                     judge-quality model.

    The parallel runner produces:
      * Lane A spine + m Lane B forks per MidCp (parallel)
      * Full-parity rubric machinery per fork-group (task 2 — rubric_bank
        carry-forward, N-sample fan-out, two-phase scoring, PSU)
      * Per-group rubric_samples.json with the same payload shape as the
        sequential runner

    ParallelSFTDataExporter applies the same accept-filter as the sequential
    exporter: gt_reward >= 1.0 on at least one branch in the group AND
    gap_corr(rubric_judge_scores, gt_scores) > 0.8 across the group's
    branches. Groups failing either are dropped.
    """
    # Lazy imports keep this module light when the parallel path isn't used.
    from swe_agent.backend import SWEAgentRolloutBackend
    from swe_agent.run.benchmarks.swebench import (
        build_swebench_config,
        get_swebench_docker_image_name,
        get_swebench_harness_namespace,
        load_swebench_instances,
    )
    from swe_agent.run.run_swe_agent import (
        DEFAULT_COMPLETION_MAX_TOKENS,
        DEFAULT_ENV_TIMEOUT,
        DEFAULT_PULL_TIMEOUT,
        SWE_AGENT_TEXTBASED_CONFIG,
    )
    from swe_agent.trajectory_search_parallel import (
        ParallelSearchConfig,
        TrajectorySearchParallelRunner,
    )

    instances = load_swebench_instances(subset, split)
    matching = [inst for inst in instances if str(inst["instance_id"]) == instance_id]
    if not matching:
        raise RuntimeError(f"instance not found in {subset}/{split}: {instance_id}")
    instance = matching[0]

    run_dir = output_root / "teacher_student_parallel" / f"{instance_id}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build a backend pointing at the teacher endpoint via litellm_textbased
    # (matches search_swe_agent.py's openai backend path so token-vocab
    # mismatch is avoided when teacher is DSv4 and tokenizer is local Qwen).
    image_name = get_swebench_docker_image_name(instance)
    api_base = (
        teacher_base_url if teacher_base_url.endswith("/v1")
        else teacher_base_url.rstrip("/") + "/v1"
    )
    config = build_swebench_config(
        config_spec=[str(SWE_AGENT_TEXTBASED_CONFIG)],
        model=teacher_model_name,
        model_class="litellm_textbased",
        extra_overrides={
            "agent": {"step_limit": step_limit, "cost_limit": 0},
            "environment": {
                "image": image_name,
                "cwd": "",
                "timeout": DEFAULT_ENV_TIMEOUT,
                "pull_timeout": DEFAULT_PULL_TIMEOUT,
                "executable": os.environ.get("MSWEA_DOCKER_EXECUTABLE", "docker"),
            },
            "model": {
                "model_kwargs": {
                    "api_base": api_base,
                    "api_key": teacher_api_key,
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "max_tokens": DEFAULT_COMPLETION_MAX_TOKENS,
                },
                "cost_tracking": "ignore_errors",
            },
        },
    )
    backend = SWEAgentRolloutBackend(
        model=config.get("model", {}),
        environment=config.get("environment", {}),
        agent=config.get("agent", {}),
        default_agent_type="default",
        default_environment_type=config.get("environment", {}).get("environment_class", "docker"),
    )

    pcfg = ParallelSearchConfig(
        m=m,
        max_mid_cps=max_mid_cps,
        steps_per_round=steps_per_round,
        step_limit=step_limit,
        rubric_temperature=rubric_temperature,
        rubric_top_p=rubric_top_p,
        rubric_max_tokens=rubric_max_tokens,
        judge_temperature=judge_temperature,
        judge_top_p=judge_top_p,
        judge_max_tokens=judge_max_tokens,
        n=n,
        max_active_rubrics=max_active_rubrics,
        rubric_bank_strategy=rubric_bank_strategy,
        gt_eval_workers=gt_eval_workers,
        keep_images=False,
        return_logprobs=False,  # SFT export doesn't need rollout logprobs
    )

    runner = TrajectorySearchParallelRunner(
        instance=instance,
        backend=backend,
        run_dir=run_dir,
        policy_model_name=teacher_model_name,
        rubric_model_name=teacher_model_name,
        judge_model_name=teacher_model_name,
        config=pcfg,
        harness_namespace=get_swebench_harness_namespace(instance),
        policy_base_url=teacher_base_url,
        rubric_base_url=rubric_base_url or teacher_base_url,
        api_key=teacher_api_key,
    )
    asyncio.run(runner.run())
    bundle = ParallelSFTDataExporter(run_dir=run_dir).export_bundle()
    bundle.metadata.update({
        "teacher_model_name": teacher_model_name,
        "teacher_base_url": teacher_base_url,
        "rubric_bank_strategy": rubric_bank_strategy,
        "n": n,
        "m": m,
        "max_mid_cps": max_mid_cps,
        "steps_per_round": steps_per_round,
    })
    return bundle


def collect_teacher_student_exports(
    *,
    instance_ids: list[str],
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
    n: int | None = None,
    k: int | None = None,
    p: int | None = None,
    max_rounds: int | None = None,
    step_limit: int | None = None,
) -> SFTExportBundle:
    bundles: list[SFTExportBundle] = []
    failed_instances: list[dict[str, str]] = []
    for instance_id in instance_ids:
        try:
            bundles.append(
                collect_teacher_student_export(
                    instance_id=instance_id,
                    output_root=output_root,
                    teacher_api_key=teacher_api_key,
                    student_model_name=student_model_name,
                    teacher_model_name=teacher_model_name,
                    student_backend=student_backend,
                    teacher_backend=teacher_backend,
                    subset=subset,
                    split=split,
                    completion_max_tokens=completion_max_tokens,
                    m=m,
                    n=n,
                    k=k,
                    p=p,
                    max_rounds=max_rounds,
                    step_limit=step_limit,
                )
            )
        except Exception as exc:
            failed_instances.append({"instance_id": instance_id, "error": f"{type(exc).__name__}: {exc}"})
    if not bundles:
        raise RuntimeError(f"all SFT rollout instances failed: {failed_instances}")
    return SFTExportBundle(
        instance_id=",".join(bundle.instance_id for bundle in bundles),
        run_dir=",".join(bundle.run_dir for bundle in bundles),
        accepted_group_ids=[bundle.instance_id + '_' + group_id for bundle in bundles for group_id in bundle.accepted_group_ids],
        policy_samples=[sample for bundle in bundles for sample in bundle.policy_samples],
        rubric_samples=[sample for bundle in bundles for sample in bundle.rubric_samples],
        metadata={"failed_instances": failed_instances},
    )


_QWEN_CHAT_TEMPLATE_ROLES = {"system", "user", "assistant", "tool"}


def build_training_messages(prompt: list[dict[str, Any]], turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for message in prompt + turns:
        role = str(message.get("role"))
        # Drop synthetic control roles like mini-swe-agent's "exit" terminal
        # marker — qwen3.5's chat template raises 'Unexpected message role'
        # for anything outside system/user/assistant/tool.
        if role not in _QWEN_CHAT_TEMPLATE_ROLES:
            continue
        messages.append(
            {
                "role": role,
                "content": str(message.get("content")),
                "step_loss_mask": 1 if role == "assistant" else 0,
            }
        )
    return messages


def build_sft_training_rows(
    *,
    bundle: SFTExportBundle,
    pad_to_multiple: int = 1,
) -> dict[str, Any]:
    rows_by_target = {
        "policy": [{"messages": build_training_messages(sample.prompt, sample.turns)} for sample in bundle.policy_samples],
        "rubric": [{"messages": build_training_messages(sample.prompt, sample.turns)} for sample in bundle.rubric_samples],
    }
    for rows in rows_by_target.values():
        if pad_to_multiple > 1 and rows:
            missing = (-len(rows)) % pad_to_multiple
            if missing:
                rows.extend([rows[-1]] * missing)
    return rows_by_target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect teacher/student SWE-agent SFT data.")
    parser.add_argument("--instance-id", nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--student-model-name", required=True)
    parser.add_argument("--teacher-api-key", default=os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--teacher-model-name", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--student-backend", default="slime")
    parser.add_argument("--teacher-backend", default="openai")
    parser.add_argument("--subset", default="rebench_v2")
    parser.add_argument("--split", default="train")
    parser.add_argument("--completion-max-tokens", type=int)
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--p", type=int)
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument("--step-limit", type=int)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--pad-to-multiple", type=int, default=1)
    args = parser.parse_args(argv)

    if args.run_dir is not None:
        if len(args.instance_id) != 1:
            raise ValueError("--run-dir can only be used with one --instance-id")
        bundle = collect_teacher_student_export(
            instance_id=args.instance_id[0],
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
            n=args.n,
            k=args.k,
            p=args.p,
            max_rounds=args.max_rounds,
            step_limit=args.step_limit,
            run_dir=args.run_dir,
        )
    else:
        bundle = collect_teacher_student_exports(
            instance_ids=args.instance_id,
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
            n=args.n,
            k=args.k,
            p=args.p,
            max_rounds=args.max_rounds,
            step_limit=args.step_limit,
        )
    rows = build_sft_training_rows(
        bundle=bundle,
        pad_to_multiple=args.pad_to_multiple,
    )
    print(f"{bundle!r}; policy_rows={len(rows['policy'])}; rubric_rows={len(rows['rubric'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
