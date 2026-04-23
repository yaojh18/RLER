from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def _write_subset(split_name: str, size: int, output_dir: Path) -> Path:
    dataset = load_dataset("zhuzilin/gsm8k", split=split_name)
    subset = dataset.select(range(min(size, len(dataset))))
    subset = subset.map(
        lambda row: {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Solve the math problem and respond with only the final "
                        "numeric answer wrapped in \\boxed{}."
                    ),
                },
                {"role": "user", "content": row["question"]},
            ],
            "sft_messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Solve the math problem and respond with only the final "
                        "numeric answer wrapped in \\boxed{}."
                    ),
                },
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            ],
            "label": row["answer"],
        },
        remove_columns=subset.column_names,
    )
    path = output_dir / f"{split_name}_{len(subset)}.parquet"
    subset.to_parquet(str(path))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a tiny GSM8K subset for slime smoke runs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=8)
    parser.add_argument("--test-size", type=int, default=4)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = _write_subset("train", args.train_size, args.output_dir)
    test_path = _write_subset("test", args.test_size, args.output_dir)

    print(f"train_path={train_path}")
    print(f"test_path={test_path}")


if __name__ == "__main__":
    main()
