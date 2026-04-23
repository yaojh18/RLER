from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from slime.swe_agent.data_export import SFTDataExporter
    from slime.swe_agent.run.data_utils import build_sft_jsonl_records, write_jsonl
except ModuleNotFoundError:
    from swe_agent.data_export import SFTDataExporter
    from swe_agent.run.data_utils import build_sft_jsonl_records, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare policy and rubric SFT jsonl files from a search run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pad-to-multiple", type=int, default=1)
    args = parser.parse_args()

    bundle = SFTDataExporter(run_dir=args.run_dir).export_bundle()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = args.output_dir / "policy_sft.jsonl"
    rubric_path = args.output_dir / "rubric_sft.jsonl"
    policy_rows = build_sft_jsonl_records(bundle.policy_samples)
    rubric_rows = build_sft_jsonl_records(bundle.rubric_samples)
    if args.pad_to_multiple > 1:
        while policy_rows and len(policy_rows) % args.pad_to_multiple != 0:
            policy_rows.append(policy_rows[-1])
        while rubric_rows and len(rubric_rows) % args.pad_to_multiple != 0:
            rubric_rows.append(rubric_rows[-1])
    write_jsonl(policy_path, policy_rows)
    write_jsonl(rubric_path, rubric_rows)
    summary = {
        "instance_id": bundle.instance_id,
        "run_dir": bundle.run_dir,
        "accepted_group_ids": bundle.accepted_group_ids,
        "policy_sample_count": len(bundle.policy_samples),
        "rubric_sample_count": len(bundle.rubric_samples),
        "policy_written_count": len(policy_rows),
        "rubric_written_count": len(rubric_rows),
        "policy_path": str(policy_path),
        "rubric_path": str(rubric_path),
        "metadata": bundle.metadata,
    }
    (args.output_dir / "sft_export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
