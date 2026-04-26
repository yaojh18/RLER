#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import os
import time
from pathlib import Path
from typing import Any, Sequence

from dr_agent.utils import launch_vllm_server_handle
from agent_rl import clear_model_services, register_model_service
from agent_rl.run_utils import ModelRouteConfig, clear_model_routes, configure_model_route
from swe_agent.rl_backend import SWEAgentRolloutBackend
from swe_agent.run.benchmarks.swebench import (
    build_swebench_config,
    get_swebench_docker_image_name,
    get_swebench_harness_namespace,
    load_swebench_instances,
)
from swe_agent.run.run_swe_agent import (
    DEFAULT_COMPLETION_MAX_TOKENS,
    DEFAULT_ENV_TIMEOUT,
    DEFAULT_EVAL_TIMEOUT,
    DEFAULT_LOG_ROOT,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_MODEL_CLASS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PULL_TIMEOUT,
    DEFAULT_STEP_LIMIT,
    DEFAULT_SUBSET,
    DEFAULT_SPLIT,
    DEFAULT_VLLM_PORT,
    DEFAULT_SERVE_MODEL,
    ParseInstanceIds,
    choose_gpus,
    find_free_port,
    infer_litellm_api_env,
    make_run_root,
    tee_console,
    temporary_env,
    terminate_process,
    write_failure_artifacts,
)
from swe_agent.run.run_swe_agent import SWE_AGENT_TEXTBASED_CONFIG
from swe_agent.trajectory_search import SearchConfig, TrajectorySearchRunner
from slime.swe_agent.serving import SGLangChatService

SLIME_SERVICE_NAME = "slime"
VLLM_SERVICE_NAME = "vllm"
SLIME_API_BASE = os.environ.get("SEARCH_SWE_SLIME_API_BASE", "http://127.0.0.1:8021")
SLIME_API_KEY = os.environ.get("SEARCH_SWE_SLIME_API_KEY", "EMPTY")


def _resolve_model_name(args: argparse.Namespace) -> str:
    if args.backend == "vllm":
        return args.vllm_model
    if args.backend == "slime":
        return args.slime_model
    return args.openai_model


def _run_single_instance(
    *,
    instance: dict[str, Any],
    run_dir: Path,
    config: dict[str, Any],
    policy_model_name: str,
    student_policy_model_name: str | None,
    rubric_model_name: str,
    judge_model_name: str,
    rubric_model_kwargs: dict[str, Any],
    judge_model_kwargs: dict[str, Any],
    search_config: SearchConfig,
    resume: bool,
) -> None:
    instance_config = copy.deepcopy(config)
    environment_config = instance_config.setdefault("environment", {})
    if environment_config.get("environment_class", "docker") == "docker":
        environment_config["image"] = get_swebench_docker_image_name(instance)
    runner = TrajectorySearchRunner(
        instance=instance,
        backend=SWEAgentRolloutBackend(
            model=instance_config.get("model", {}),
            environment=instance_config.get("environment", {}),
            agent=instance_config.get("agent", {}),
            default_agent_type="default",
            default_environment_type=instance_config.get("environment", {}).get("environment_class", "docker"),
        ),
        run_dir=run_dir,
        policy_model_name=policy_model_name,
        student_policy_model_name=student_policy_model_name,
        search_config=search_config,
        rubric_model_name=rubric_model_name,
        judge_model_name=judge_model_name,
        rubric_model_kwargs=rubric_model_kwargs,
        judge_model_kwargs=judge_model_kwargs,
        harness_namespace=get_swebench_harness_namespace(instance),
        resume=resume,
    )
    runner.run()


def run_search(
    args: argparse.Namespace,
    instance_ids: Sequence[str] | None,
) -> None:
    student_model_name = getattr(args, "student_model", None)
    model_name = _resolve_model_name(args)
    available_instances = load_swebench_instances(args.subset, args.split)
    by_id = {instance["instance_id"]: instance for instance in available_instances}
    selected_ids = list(instance_ids or by_id)
    missing = [instance_id for instance_id in selected_ids if instance_id not in by_id]
    if missing:
        raise RuntimeError(f"Instances not found in {args.subset}/{args.split}: {', '.join(missing)}")
    instances = [by_id[instance_id] for instance_id in selected_ids]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_root = make_run_root(
        output_root=args.output_root,
        benchmark_name=args.subset,
        split=args.split,
        model_name=model_name,
    )
    run_log_path = DEFAULT_LOG_ROOT / f"search-{timestamp}.log"
    run_log_path.parent.mkdir(parents=True, exist_ok=True)

    gpu_id: int | None = None
    gpu_ids: list[int] = []
    vllm_handle = None
    service_name: str | None = None
    shared_model_kwargs: dict[str, Any] = {}
    if "qwen" in model_name.lower():
        shared_model_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
    if args.backend == "vllm":
        gpu_ids = choose_gpus(args.gpu_id)
        gpu_id = gpu_ids[0] if gpu_ids else None
        if gpu_id is None:
            raise RuntimeError("vLLM backend requires a GPU; got gpu_id=none")
        vllm_handle = launch_vllm_server_handle(
            model_name=args.vllm_model,
            served_model_name=args.vllm_model,
            port=find_free_port(args.vllm_port),
            gpu_id=gpu_id,
            gpu_ids=gpu_ids,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            allow_long_max_model_len=args.allow_long_max_model_len,
        )
        service_name = VLLM_SERVICE_NAME
        register_model_service(
            service_name,
            SGLangChatService(
                base_url=vllm_handle.base_url,
                api_key="EMPTY",
                default_model_name=args.vllm_model,
            ),
        )
    elif args.backend == "openai":
        required_env = infer_litellm_api_env(args.openai_model)
        if required_env and not os.getenv(required_env):
            raise RuntimeError(f"{required_env} is not set for model {args.openai_model}")
    else:
        service_name = SLIME_SERVICE_NAME
        register_model_service(
            service_name,
            SGLangChatService(
                base_url=SLIME_API_BASE,
                api_key=SLIME_API_KEY,
                default_model_name=args.slime_model,
            ),
        )
    student_backend = getattr(args, "student_backend", "slime")
    if student_model_name and student_backend == "slime":
        if service_name != SLIME_SERVICE_NAME:
            register_model_service(
                SLIME_SERVICE_NAME,
                SGLangChatService(
                    base_url=SLIME_API_BASE,
                    api_key=SLIME_API_KEY,
                    default_model_name=student_model_name,
                ),
            )
        configure_model_route("policy_student", ModelRouteConfig(backend="service", service_name=SLIME_SERVICE_NAME, model_name=args.student_model))
    elif student_model_name:
        raise RuntimeError(f"student_backend={student_backend} is not supported for search mixed mode")

    if args.backend == "vllm":
        configure_model_route("policy", ModelRouteConfig(backend="service", service_name=service_name, model_name=args.vllm_model))
        configure_model_route("rubric_generation", ModelRouteConfig(backend="service", service_name=service_name, model_name=args.vllm_model))
        configure_model_route("rubric_judge", ModelRouteConfig(backend="service", service_name=service_name, model_name=args.vllm_model))
    elif args.backend == "openai":
        configure_model_route("policy", ModelRouteConfig(backend="litellm"))
        configure_model_route("rubric_generation", ModelRouteConfig(backend="litellm"))
        configure_model_route("rubric_judge", ModelRouteConfig(backend="litellm"))
    else:
        configure_model_route("policy", ModelRouteConfig(backend="service", service_name=service_name, model_name=args.slime_model))
        configure_model_route("rubric_generation", ModelRouteConfig(backend="service", service_name=service_name, model_name=args.slime_model))
        configure_model_route("rubric_judge", ModelRouteConfig(backend="service", service_name=service_name, model_name=args.slime_model))
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
            m=args.m,
            n=args.n,
            k=args.k,
            p=args.p,
            step_limit=args.step_limit,
            max_rounds=args.max_rounds,
            max_active_rubrics=args.max_active_rubrics,
            policy_temperature=args.policy_temperature,
            policy_top_p=args.policy_top_p,
            rubric_temperature=args.rubric_temperature,
            rubric_top_p=args.rubric_top_p,
            rubric_max_tokens=args.rubric_max_tokens,
            judge_temperature=args.judge_temperature,
            judge_top_p=args.judge_top_p,
            judge_max_tokens=args.judge_max_tokens,
            regression_margin=args.regression_margin,
            calculate_gt_reward=args.calculate_gt_reward,
            gt_reward_workers=args.workers,
            evaluate_final_patch=args.evaluate_final_patch,
            export_grpo_bundles=args.export_grpo_bundles,
            write_artifacts=args.write_artifacts,
            write_raw_traj=args.write_raw_traj,
            strategy=args.strategy,
        )
        rubric_model_name = args.rubric_model or model_name
        judge_model_name = args.judge_model or model_name
        rubric_model_kwargs = dict(shared_model_kwargs)
        judge_model_kwargs = dict(shared_model_kwargs)
        with temporary_env({"MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": str(args.model_retry_attempts), "LITELLM_LOG": "ERROR"}):
            with tee_console(run_log_path):
                for instance in instances:
                    run_dir = args.resume_run_dir or (run_root / instance["instance_id"] / timestamp)
                    run_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        _run_single_instance(
                            instance=instance,
                            run_dir=run_dir,
                            config=config,
                            policy_model_name=model_name,
                            student_policy_model_name=student_model_name,
                            rubric_model_name=rubric_model_name,
                            judge_model_name=judge_model_name,
                            rubric_model_kwargs=rubric_model_kwargs,
                            judge_model_kwargs=judge_model_kwargs,
                            search_config=search_config,
                            resume=bool(args.resume_run_dir),
                        )
                    except Exception as exc:
                        errors.append(f"{instance['instance_id']}: {exc}")
                        write_failure_artifacts(
                            instance_id=instance["instance_id"],
                            run_dir=run_dir,
                            error=exc,
                            log_path=run_log_path,
                        )
        if errors:
            raise RuntimeError("; ".join(errors))
        return run_dir
    finally:
        clear_model_services()
        clear_model_routes()
        terminate_process(vllm_handle.process if vllm_handle else None)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run segmented rubric-search SWE-agent on one or more SWE-bench instances.")
    parser.add_argument("--backend", choices=["vllm", "openai", "slime"], default="vllm")
    parser.add_argument("--instance-id", action=ParseInstanceIds, nargs="+", default=None)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
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
    parser.add_argument("--vllm-model", default=DEFAULT_SERVE_MODEL)
    parser.add_argument("--openai-model", default=DEFAULT_SERVE_MODEL)
    parser.add_argument("--slime-model", default=DEFAULT_SERVE_MODEL)
    parser.add_argument("--model-retry-attempts", type=int, default=2)
    parser.add_argument("--completion-max-tokens", type=int, default=DEFAULT_COMPLETION_MAX_TOKENS)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--max-active-rubrics", type=int, default=6)
    parser.add_argument("--policy-temperature", type=float, default=0.5)
    parser.add_argument("--policy-top-p", type=float, default=0.9)
    parser.add_argument("--rubric-temperature", type=float, default=0.5)
    parser.add_argument("--rubric-top-p", type=float, default=0.9)
    parser.add_argument("--rubric-max-tokens", type=int, default=4096)
    parser.add_argument("--judge-temperature", type=float, default=0.1)
    parser.add_argument("--judge-top-p", type=float, default=0.9)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--regression-margin", type=float, default=0.0)
    parser.add_argument("--rubric-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--calculate-gt-reward", action="store_true", default=True)
    parser.add_argument("--write-raw-traj", action="store_false", default=False)
    parser.add_argument("--strategy", choices=["best", "probability", "random"], default="best")
    parser.add_argument("--student-backend", choices=["vllm", "openai", "slime"], default="slime")
    parser.add_argument("--student-model", default=None)
    parser.add_argument("--evaluate-final-patch", action="store_true", default=True)
    parser.add_argument("--export-grpo-bundles", action="store_true", default=True)
    parser.add_argument("--write-artifacts", action="store_true", default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_search(args, args.instance_id)


if __name__ == "__main__":
    raise SystemExit(main())
