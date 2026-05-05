from __future__ import annotations

import sys

from train_agent.run.grpo import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "--rollout-function-path",
                "train_agent.collect_grpo_rollout_async.generate_rollout",
                *sys.argv[1:],
            ]
        )
    )
