from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _select_shortest_sft_rows(rows: list[dict], tokenizer, limit: int) -> tuple[list[dict], list[int]]:
    ranked_rows: list[tuple[int, int, dict]] = []
    for original_index, row in enumerate(rows):
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        ranked_rows.append((len(tokenizer.encode(text, add_special_tokens=False)), original_index, row))
    ranked_rows.sort(key=lambda item: (item[0], item[1]))
    selected = ranked_rows[: min(limit, len(ranked_rows))]
    return [row for _, _, row in selected], [length for length, _, _ in selected]


def _load_rollout(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _select_shortest_rollout_samples(payload: dict, limit: int) -> tuple[list[dict], list[int]]:
    ranked_samples = sorted(
        ((len(sample["tokens"]), index, sample) for index, sample in enumerate(payload["samples"])),
        key=lambda item: (item[0], item[1]),
    )
    selected = ranked_samples[: min(limit, len(ranked_samples))]
    subset: list[dict] = []
    for new_index, (_, _, sample) in enumerate(selected):
        sample_copy = dict(sample)
        sample_copy["index"] = new_index
        subset.append(sample_copy)
    return subset, [length for length, _, _ in selected]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build small training subsets from full SWE-agent exports.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--policy-sft-path", type=Path, required=True)
    parser.add_argument("--rubric-sft-path", type=Path, required=True)
    parser.add_argument("--policy-rollout-path", type=Path, required=True)
    parser.add_argument("--rubric-rollout-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-sft-limit", type=int, default=8)
    parser.add_argument("--rubric-sft-limit", type=int, default=4)
    parser.add_argument("--policy-rollout-limit", type=int, default=6)
    parser.add_argument("--rubric-rollout-limit", type=int, default=4)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    policy_rows = _load_jsonl(args.policy_sft_path)
    selected_policy_rows, policy_lengths = _select_shortest_sft_rows(policy_rows, tokenizer, args.policy_sft_limit)
    policy_output = args.output_dir / "policy_sft_smoke.jsonl"
    _write_jsonl(policy_output, selected_policy_rows)

    rubric_rows = _load_jsonl(args.rubric_sft_path)
    selected_rubric_rows, rubric_lengths = _select_shortest_sft_rows(rubric_rows, tokenizer, args.rubric_sft_limit)
    rubric_output = args.output_dir / "rubric_sft_smoke.jsonl"
    _write_jsonl(rubric_output, selected_rubric_rows)

    policy_rollout = _load_rollout(args.policy_rollout_path)
    selected_policy_rollout, policy_rollout_lengths = _select_shortest_rollout_samples(
        policy_rollout, args.policy_rollout_limit
    )
    policy_rollout_output = args.output_dir / "policy_rollout_smoke.pt"
    torch.save({"rollout_id": policy_rollout.get("rollout_id", 0), "samples": selected_policy_rollout}, policy_rollout_output)

    rubric_rollout = _load_rollout(args.rubric_rollout_path)
    selected_rubric_rollout, rubric_rollout_lengths = _select_shortest_rollout_samples(
        rubric_rollout, args.rubric_rollout_limit
    )
    rubric_rollout_output = args.output_dir / "rubric_rollout_smoke.pt"
    torch.save({"rollout_id": rubric_rollout.get("rollout_id", 0), "samples": selected_rubric_rollout}, rubric_rollout_output)

    summary = {
        "policy_sft": {
            "source_count": len(policy_rows),
            "selected_count": len(selected_policy_rows),
            "selected_lengths": policy_lengths,
            "path": str(policy_output),
        },
        "rubric_sft": {
            "source_count": len(rubric_rows),
            "selected_count": len(selected_rubric_rows),
            "selected_lengths": rubric_lengths,
            "path": str(rubric_output),
        },
        "policy_rollout": {
            "source_count": len(policy_rollout["samples"]),
            "selected_count": len(selected_policy_rollout),
            "selected_lengths": policy_rollout_lengths,
            "path": str(policy_rollout_output),
        },
        "rubric_rollout": {
            "source_count": len(rubric_rollout["samples"]),
            "selected_count": len(selected_rubric_rollout),
            "selected_lengths": rubric_rollout_lengths,
            "path": str(rubric_rollout_output),
        },
    }
    (args.output_dir / "smoke_subset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
