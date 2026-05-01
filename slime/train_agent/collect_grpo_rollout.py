#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import swe_agent.run.search_swe_agent as search_module
from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from train_agent.collect_sft_rollout import build_training_messages
from train_agent.contracts import ExportGroup, ExportSample, GRPOExportBundle

_TOKENIZER = None


def collect_grpo_bundle(
    *,
    instance_id: str,
    subset: str,
    split: str,
    output_root: Path,
    model_name: str,
    workers: int = 1,
    m: int | None = None,
    n: int | None = None,
    k: int | None = None,
    p: int | None = None,
    max_rounds: int | None = None,
    step_limit: int | None = None,
) -> GRPOExportBundle:
    output_root.mkdir(parents=True, exist_ok=True)
    args = search_module.build_arg_parser().parse_args([])
    args.backend = "slime"
    args.instance_id = [instance_id]
    args.subset = subset
    args.split = split
    args.output_root = output_root
    args.slime_model = model_name
    args.workers = workers
    args.calculate_gt_reward = True
    args.export_grpo_bundles = True
    args.write_artifacts = True
    for name, value in (("m", m), ("k", k), ("p", p), ("n", n), ("max_rounds", max_rounds), ("step_limit", step_limit)):
        if value is not None:
            setattr(args, name, value)

    run_dir = None
    try:
        run_dir, bundle = search_module.run_search(args, [instance_id])
        bundle.run_dir = str(run_dir)
        return bundle
    finally:
        chmod_root = run_dir or output_root
        if chmod_root.exists():
            for path in [chmod_root, *chmod_root.rglob("*")]:
                path.chmod(0o755 if path.is_dir() else 0o644)


def build_rollout_samples(
    *,
    groups: list[ExportGroup],
    tokenizer,
    loss_mask_type: str,
    include_turn_rewards: bool,
    group_index_offset: int = 0,
) -> list[Sample]:
    samples: list[Sample] = []
    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=loss_mask_type)
    for group_index, group in enumerate(groups):
        for export_sample in group.samples:
            messages = build_training_messages(export_sample.prompt, export_sample.turns)
            token_ids, full_loss_mask = mask_generator.get_loss_mask(messages)
            response_length = mask_generator.get_response_lengths([full_loss_mask])[0]
            if response_length <= 0:
                raise ValueError(f"Sample {export_sample.sample_id} does not contain trainable assistant tokens.")
            sample = Sample(
                group_index=group_index_offset + group_index,
                prompt=messages,
                tokens=token_ids,
                response_length=response_length,
                reward=float(export_sample.reward or 0.0),
                loss_mask=full_loss_mask[-response_length:],
                status=Sample.Status.COMPLETED,
            )
            if include_turn_rewards:
                sample.train_metadata = _build_turn_metadata(
                    full_loss_mask=full_loss_mask,
                    response_length=response_length,
                    export_sample=export_sample,
                )
            samples.append(sample)
    return samples


def _build_turn_metadata(
    *,
    full_loss_mask: list[int],
    response_length: int,
    export_sample: ExportSample,
) -> dict[str, Any]:
    turn_rewards = list(export_sample.metadata.get("turn_rewards") or [])
    response_mask = full_loss_mask[-response_length:]
    turn_masks: list[list[int]] = []
    start: int | None = None
    for index, value in enumerate(response_mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            turn_mask = [0] * response_length
            turn_mask[start:index] = [1] * (index - start)
            turn_masks.append(turn_mask)
            start = None
    if start is not None:
        turn_mask = [0] * response_length
        turn_mask[start:] = [1] * (response_length - start)
        turn_masks.append(turn_mask)
    if len(turn_rewards) > len(turn_masks):
        raise ValueError(
            f"Sample {export_sample.sample_id} has {len(turn_rewards)} turn rewards but only {len(turn_masks)} trainable turns."
        )
    if len(turn_rewards) < len(turn_masks):
        turn_rewards.extend([0.0] * (len(turn_masks) - len(turn_rewards)))
    return {"turn_rewards": turn_rewards, "turn_loss_masks": turn_masks}


def _rollout_tokenizer(args):
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    return _TOKENIZER


def _configure_slime_route(args) -> str:
    routers = getattr(args, "sglang_model_routers", None) or {}
    router_ip, router_port = routers.get("default", (getattr(args, "sglang_router_ip", None), getattr(args, "sglang_router_port", None)))
    if not router_ip or not router_port:
        raise RuntimeError("SGLang router address is unavailable for SWE-agent rollout.")
    base_url = f"http://{router_ip}:{router_port}"
    os.environ["SEARCH_SWE_SLIME_API_BASE"] = base_url
    os.environ["SEARCH_SWE_SLIME_API_KEY"] = "EMPTY"
    return base_url


def build_grpo_prompt_rows(instance_ids: list[str], subset: str, split: str) -> list[dict[str, object]]:
    return [
        {
            "input": [{"role": "user", "content": instance_id}],
            "metadata": {"instance_id": instance_id, "subset": subset, "split": split},
        }
        for instance_id in instance_ids
    ]

def generate_rollout(args, rollout_id: int, data_buffer, evaluation: bool = False):
    if evaluation:
        return RolloutFnEvalOutput(data={}, metrics={})

    started = time.perf_counter()
    _configure_slime_route(args)
    tokenizer = _rollout_tokenizer(args)
    target = os.environ.get("SWE_AGENT_GRPO_TARGET", "policy")
    output_root = Path(os.environ.get("SWE_AGENT_GRPO_OUTPUT_ROOT", "/workspace/rler/agent/outputs/search_outputs/train_async_grpo"))
    model_name = os.environ.get("SWE_AGENT_GRPO_MODEL_NAME") or "Qwen/Qwen3.5-9B"
    all_samples: list[Sample] = []
    group_index_offset = 0
    collected_instances: list[str] = []
    failed_instances: list[str] = []
    search_values: dict[str, int | None] = {}
    for env_name, arg_name in (
        ("SWE_AGENT_GRPO_M", "m"),
        ("SWE_AGENT_GRPO_N", "n"),
        ("SWE_AGENT_GRPO_K", "k"),
        ("SWE_AGENT_GRPO_P", "p"),
        ("SWE_AGENT_GRPO_MAX_ROUNDS", "max_rounds"),
        ("SWE_AGENT_GRPO_STEP_LIMIT", "step_limit"),
    ):
        value = os.environ[env_name]
        search_values[arg_name] = int(value) if value else None

    prompt_groups = data_buffer.get_samples(args.rollout_batch_size)
    for prompt_group in prompt_groups:
        instance_id = prompt_group[0].metadata["instance_id"]
        try:
            bundle = collect_grpo_bundle(
                instance_id=instance_id,
                subset=prompt_group[0].metadata["subset"],
                split=prompt_group[0].metadata["split"],
                output_root=output_root / f"rollout_{rollout_id:04d}" / f"attempt_{attempt_index:02d}",
                model_name=model_name,
                workers=int(os.environ["SWE_AGENT_GRPO_WORKERS"]),
                **search_values,
            )
        except Exception as exc:
            failed_instances.append(f"{instance_id}: {type(exc).__name__}: {exc}")
            continue
        groups = bundle.policy_groups if target == "policy" else bundle.rubric_groups
        samples = build_rollout_samples(
            groups=groups,
            tokenizer=tokenizer,
            loss_mask_type=getattr(args, "loss_mask_type", "qwen3_5"),
            include_turn_rewards=target == "rubric",
            group_index_offset=group_index_offset,
        )
        group_index_offset += len(groups)
        collected_instances.append(instance_id)
        all_samples.extend(samples)

    if not all_samples:
        raise RuntimeError(f"SWE-agent GRPO rollout produced no {target} samples; failed={failed_instances}")
    for index, sample in enumerate(all_samples):
        sample.index = index
    return RolloutFnTrainOutput(
        samples=all_samples,
        metrics={
            "swe_agent/target": target,
            "swe_agent/instances": len(collected_instances),
            "swe_agent/samples": len(all_samples),
            "swe_agent/groups": group_index_offset,
            "swe_agent/failed_instances": len(failed_instances),
            "swe_agent/seconds": time.perf_counter() - started,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect one SWE-agent GRPO bundle.")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--subset", default="rebench_v2")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--p", type=int)
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument("--step-limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    bundle = collect_grpo_bundle(
        instance_id=args.instance_id,
        subset=args.subset,
        split=args.split,
        output_root=args.output_root,
        model_name=args.model_name,
        workers=args.workers,
        m=args.m,
        n=args.n,
        k=args.k,
        p=args.p,
        max_rounds=args.max_rounds,
        step_limit=args.step_limit,
    )
    print(repr(bundle))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
