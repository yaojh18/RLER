#!/usr/bin/env python3
from __future__ import annotations

import argparse
import multiprocessing
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample
import swe_agent.run.search_swe_agent as search_module

from train_agent.collect_sft_rollout import build_training_messages
from train_agent.contracts import ExportGroup, ExportSample, GRPOExportBundle
from train_agent.serving.sglang_chat_service import start_slime_policy_route_warmup

_TOKENIZER = None
_ROLLOUT_WARMUP_DONE = False


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
    max_sample_tokens: int | None = None,
) -> tuple[list[Sample], int]:
    """Materialize Sample objects from PDS / search-output ExportGroups.

    Two paths:
      A. Precomputed (PDS): export_sample.token_ids + .loss_mask +
         .response_length are populated by _build_policy_samples_for_round.
         We use them directly — they encode the correct loss_mask (parent's
         shared prefix masked OUT, only THIS branch's new assistant tokens
         trainable). This fixes the bug where the legacy path applied loss
         to parent's shared trace x M=8 siblings per round.
      B. Re-tokenization fallback: text-only export_sample. Build messages,
         tokenize via slime's MultiTurnLossMaskGenerator. OK for non-PDS
         single-trajectory rollouts.

    Safety net: when max_sample_tokens is given, any sample with
    len(token_ids) > max_sample_tokens is dropped (logged + counted).
    slime's _get_capped_partitions asserts when a single sample exceeds the
    partition budget — dropping here keeps that from killing the whole
    training step. Returns (samples, dropped_count).
    """
    samples: list[Sample] = []
    dropped = 0
    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=loss_mask_type)
    for group_index, group in enumerate(groups):
        for export_sample in group.samples:
            has_precomputed = (
                export_sample.token_ids is not None
                and export_sample.loss_mask is not None
                and export_sample.response_length is not None
            )
            if has_precomputed:
                # Path A: use PDS-side precomputed token_ids + loss_mask.
                token_ids = list(export_sample.token_ids)
                full_loss_mask = list(export_sample.loss_mask)
                response_length = int(export_sample.response_length)
                if len(token_ids) != len(full_loss_mask):
                    raise ValueError(
                        f"Sample {export_sample.sample_id}: token_ids len "
                        f"({len(token_ids)}) != loss_mask len ({len(full_loss_mask)})"
                    )
                if response_length <= 0:
                    continue
                # Reconstruct text messages so Sample.prompt stays populated
                # for any downstream code that inspects it (logging/dump).
                messages = build_training_messages(export_sample.prompt, export_sample.turns)
            else:
                # Path B: text-only fallback.
                messages = build_training_messages(export_sample.prompt, export_sample.turns)
                token_ids, full_loss_mask = mask_generator.get_loss_mask(messages)
                response_length = mask_generator.get_response_lengths([full_loss_mask])[0]
                if response_length <= 0:
                    raise ValueError(f"Sample {export_sample.sample_id} does not contain trainable assistant tokens.")
            if max_sample_tokens is not None and len(token_ids) > max_sample_tokens:
                # sglang's context_length cap (80960 by default) bounds the
                # token-in/token-out path's full sequence; samples beyond
                # that would have come from a different / older code path
                # or some edge case. Drop them rather than asserting in
                # slime's _get_capped_partitions.
                print(
                    f"[build_rollout_samples] DROP sample {export_sample.sample_id}: "
                    f"len(token_ids)={len(token_ids)} > max_sample_tokens={max_sample_tokens}",
                    flush=True,
                )
                dropped += 1
                continue
            sample = Sample(
                group_index=group_index_offset + group_index,
                prompt=messages,
                tokens=token_ids,
                response_length=response_length,
                reward=float(export_sample.reward or 0.0),
                loss_mask=full_loss_mask[-response_length:],
                status=Sample.Status.COMPLETED,
            )
            # Propagate ExportSample.metadata onto Sample.metadata so
            # downstream metric aggregation in the rollout-fn metrics dict
            # (e.g. swe_agent/sample_cont_steps_mean) can read per-sample
            # fields that lane_to_grpo_bundle._build_branch_sample stashes
            # there: n_continuation_steps, n_parent_steps, n_full_trace_steps,
            # raw_gt_score, raw_rubric_score, terminated_early, is_dummy, etc.
            if export_sample.metadata:
                sample.metadata = dict(export_sample.metadata)
            # Propagate sglang-stored rollout-time logprobs onto the
            # slime Sample so TIS (off-policy IS correction) can compute
            # exp(actor_logprob - rollout_logprob). naive bundle emits a
            # dense per-response-token vector of length response_length,
            # aligned with loss_mask[-response_length:]. Lane bundle
            # still emits a sparse (asst-only) vector — wire TIS for
            # lanes when that's updated.
            if export_sample.rollout_logprobs is not None:
                lp = list(export_sample.rollout_logprobs)
                if len(lp) == response_length:
                    sample.rollout_log_probs = lp
            if include_turn_rewards:
                sample.train_metadata = _build_turn_metadata(
                    full_loss_mask=full_loss_mask,
                    response_length=response_length,
                    export_sample=export_sample,
                )
            samples.append(sample)
    return samples, dropped


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


def build_grpo_prompt_rows(instance_ids: list[str], subset: str, split: str) -> list[dict[str, object]]:
    return [
        {
            "input": [{"role": "user", "content": instance_id}],
            "metadata": {
                "instance_id": instance_id,
                "subset": subset,
                "split": split,
            },
        }
        for instance_id in instance_ids
    ]


def _collect_grpo_bundle_task(task: dict[str, Any]) -> dict[str, Any]:
    try:
        os.environ["SEARCH_SWE_SLIME_API_BASE"] = task["slime_api_base"]
        os.environ["SEARCH_SWE_SLIME_API_KEY"] = task["slime_api_key"]
        bundle = collect_grpo_bundle(
            instance_id=task["instance_id"],
            subset=task["subset"],
            split=task["split"],
            output_root=task["output_root"],
            model_name=task["model_name"],
            workers=task["workers"],
            **task["search_values"],
        )
        return {"index": task["index"], "instance_id": task["instance_id"], "bundle": bundle, "error": ""}
    except Exception as exc:
        return {
            "index": task["index"],
            "instance_id": task["instance_id"],
            "bundle": None,
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


def _collect_grpo_bundle_tasks(
    tasks: list[dict[str, Any]],
    max_workers: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    context_name = os.environ.get("SWE_AGENT_GRPO_MP_CONTEXT", "fork")
    context = multiprocessing.get_context(context_name)
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
        futures = {executor.submit(_collect_grpo_bundle_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(
                    {
                        "index": task["index"],
                        "instance_id": task["instance_id"],
                        "bundle": None,
                        "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                    }
                )
    return sorted(results, key=lambda item: item["index"])


def generate_rollout(args, rollout_id: int, data_buffer, evaluation: bool = False):
    global _ROLLOUT_WARMUP_DONE
    started = time.perf_counter()
    target = os.environ.get("SWE_AGENT_GRPO_TARGET", "policy")
    output_root = Path(os.environ.get("SWE_AGENT_GRPO_OUTPUT_ROOT", "/workspace/rler/agent/outputs/search_outputs/train_async_grpo"))
    model_name = os.environ.get("SWE_AGENT_GRPO_MODEL_NAME") or "Qwen/Qwen3.5-9B"
    router_ip, router_port = (getattr(args, "sglang_model_routers", None) or {}).get(
        model_name,
        (args.sglang_router_ip, args.sglang_router_port),
    )
    slime_api_base = f"http://{router_ip}:{router_port}"
    slime_api_key = os.environ.get("SEARCH_SWE_SLIME_API_KEY", "EMPTY")
    if not _ROLLOUT_WARMUP_DONE:
        start_slime_policy_route_warmup(
            base_url=slime_api_base,
            api_key=slime_api_key,
            model_name=model_name,
            requests=max(1, int(args.rollout_num_gpus or 1) // int(args.rollout_num_gpus_per_engine or 1)),
        )
        _ROLLOUT_WARMUP_DONE = True

    all_samples: list[Sample] = []
    group_index_offset = 0
    collected_instances: list[str] = []
    failed_instances: list[str] = []
    total_dropped_oversized = 0
    # Drop samples whose total token length exceeds slime's per-sample budget
    # (= max_tokens_per_gpu * cp_size). Without this, one oversized sample
    # trips slime's _get_capped_partitions assertion and kills training.
    max_sample_tokens: int | None = None
    try:
        mt = int(getattr(args, "max_tokens_per_gpu", 0) or 0)
        cp = int(getattr(args, "context_parallel_size", 1) or 1)
        if mt > 0:
            max_sample_tokens = mt * cp
    except Exception:
        max_sample_tokens = None
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
    if not prompt_groups:
        raise RuntimeError("SWE-agent GRPO rollout received an empty prompt batch.")

    tokenizer = _rollout_tokenizer(args)
    tasks: list[dict[str, Any]] = []
    for prompt_index, prompt_group in enumerate(prompt_groups):
        if len(prompt_group) != 1:
            raise RuntimeError(
                f"SWE-agent rollout expects one slime sample per prompt group; got {len(prompt_group)}. "
                "Keep --n-samples-per-prompt 1 because SWE search creates GRPO groups internally."
            )
        metadata = prompt_group[0].metadata
        tasks.append(
            {
                "index": prompt_index,
                "instance_id": metadata["instance_id"],
                "subset": metadata["subset"],
                "split": metadata["split"],
                "output_root": output_root / f"rollout_{rollout_id:04d}",
                "model_name": model_name,
                "workers": int(os.environ["SWE_AGENT_GRPO_WORKERS"]),
                "search_values": search_values,
                "slime_api_base": slime_api_base,
                "slime_api_key": slime_api_key,
            }
        )

    max_instance_workers = max(1, int(os.environ["SWE_AGENT_GRPO_INSTANCE_WORKERS"]))
    bundle_results = _collect_grpo_bundle_tasks(
        tasks,
        min(max_instance_workers, len(tasks)),
    )
    for result in bundle_results:
        instance_id = result["instance_id"]
        if result["error"]:
            failed_instances.append(f"{instance_id}: {result['error']}")
            continue
        bundle = result["bundle"]
        groups = bundle.policy_groups if target == "policy" else bundle.rubric_groups
        samples, dropped_here = build_rollout_samples(
            groups=groups,
            tokenizer=tokenizer,
            loss_mask_type=getattr(args, "loss_mask_type", "qwen3_5"),
            include_turn_rewards=target == "rubric",
            group_index_offset=group_index_offset,
            max_sample_tokens=max_sample_tokens,
        )
        total_dropped_oversized += dropped_here
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
            "swe_agent/dropped_oversized_samples": total_dropped_oversized,
            "swe_agent/max_sample_tokens": max_sample_tokens or 0,
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
