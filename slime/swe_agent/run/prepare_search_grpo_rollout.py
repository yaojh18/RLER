from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from slime.swe_agent.data_export import GRPODataExporter
    from slime.swe_agent.run.data_utils import build_debug_sample, load_tokenizer_and_type, write_debug_rollout_data
except ModuleNotFoundError:
    from swe_agent.data_export import GRPODataExporter
    from swe_agent.run.data_utils import build_debug_sample, load_tokenizer_and_type, write_debug_rollout_data


def _build_samples(groups, tokenizer, loss_mask_type: str, include_turn_rewards: bool):
    samples = []
    sample_index = 0
    for group_index, group in enumerate(groups):
        for export_sample in group.samples:
            samples.append(
                build_debug_sample(
                    export_sample=export_sample,
                    tokenizer=tokenizer,
                    loss_mask_type=loss_mask_type,
                    sample_index=sample_index,
                    group_index=group_index,
                    include_turn_rewards=include_turn_rewards,
                )
            )
            sample_index += 1
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare debug rollout data from a search run for GRPO training.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--hf-checkpoint", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--target", choices=["policy", "rubric"], required=True)
    parser.add_argument("--rubric-parent-weight", type=float, default=0.5)
    parser.add_argument("--loss-mask-type", default="qwen3_5")
    parser.add_argument("--pad-to-multiple", type=int, default=1)
    args = parser.parse_args()

    exporter = GRPODataExporter(run_dir=args.run_dir, rubric_parent_weight=args.rubric_parent_weight)
    bundle = exporter.export_bundle()
    tokenizer, loss_mask_type = load_tokenizer_and_type(args.hf_checkpoint, args.loss_mask_type)
    if args.target == "policy":
        samples = _build_samples(bundle.policy_groups, tokenizer, loss_mask_type, include_turn_rewards=False)
    else:
        samples = _build_samples(bundle.rubric_groups, tokenizer, loss_mask_type, include_turn_rewards=True)
    if args.pad_to_multiple > 1 and samples:
        while len(samples) % args.pad_to_multiple != 0:
            duplicate = samples[-1].to_dict()
            duplicate["index"] = len(samples)
            samples.append(type(samples[-1]).from_dict(duplicate))
    write_debug_rollout_data(path=args.output_path, samples=samples)
    summary = {
        "instance_id": bundle.instance_id,
        "run_dir": bundle.run_dir,
        "target": args.target,
        "sample_count": len(samples),
        "group_count": len(bundle.policy_groups) if args.target == "policy" else len(bundle.rubric_groups),
        "output_path": str(args.output_path),
    }
    args.output_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
