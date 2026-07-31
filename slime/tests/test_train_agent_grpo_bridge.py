from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _load_grpo(monkeypatch):
    stubs = {
        "torch": _module("torch"),
        "slime": _module("slime"),
        "slime.rollout": _module("slime.rollout"),
        "slime.rollout.data_source": _module(
            "slime.rollout.data_source",
            ROLLOUT_CHECKPOINT_SCHEMA_VERSION=2,
            ROLLOUT_COLLECTOR_STATE_METADATA_KEY=(
                "__rler_rollout_collector_state_v1__"
            ),
        ),
        "slime.backends": _module("slime.backends"),
        "slime.backends.megatron_utils": _module("slime.backends.megatron_utils"),
        "slime.backends.megatron_utils.cp_utils": _module(
            "slime.backends.megatron_utils.cp_utils",
            slice_log_prob_with_cp=lambda *args, **kwargs: None,
        ),
        "slime.backends.megatron_utils.loss": _module(
            "slime.backends.megatron_utils.loss",
            policy_loss_function=lambda *args, **kwargs: (None, {}),
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = REPO_ROOT / "slime/train_agent/run/grpo.py"
    spec = importlib.util.spec_from_file_location("_test_grpo_bridge", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _required_args(tmp_path: Path) -> list[str]:
    return [
        "--target",
        "policy",
        "--prompt-data",
        str(tmp_path / "train.jsonl"),
        "--hf-checkpoint",
        str(tmp_path / "hf"),
        "--load-dir",
        str(tmp_path / "load"),
        "--save-dir",
        str(tmp_path / "save"),
        "--search-output-root",
        str(tmp_path / "search"),
        "--wandb-mode",
        "disabled",
    ]


def test_grpo_forwards_fixed_validation_usage_and_smoke_overrides(
    monkeypatch, tmp_path
):
    module = _load_grpo(monkeypatch)
    captured = {}

    def _capture(command, check):
        captured["command"] = command
        captured["check"] = check

    monkeypatch.setattr(module.subprocess, "run", _capture)
    rc = module.main(
        [
            *_required_args(tmp_path),
            "--eval-interval",
            "1",
            "--eval-instance-interval",
            "100",
            "--eval-prompt-data",
            "fold0_val",
            str(tmp_path / "val.jsonl"),
            "--n-samples-per-eval-prompt",
            "1",
            "--save-interval",
            "10",
            "--checkpoint-retain-latest",
            "2",
            "--train-instance-budget",
            "1250",
            "--require-train-instance-budget-exhaustion",
            "--stop-after-validation-attempt",
            "100",
            "--log-probs-chunk-size",
            "32",
            "--sglang-mem-fraction-static",
            "0.75",
            "--sglang-disable-custom-all-reduce",
        ]
    )

    assert rc == 0
    argv = captured["command"]
    assert argv[:2] == ["bash", "-lc"]
    script = argv[2]
    assert (
        f"export RLER_USAGE_LEDGER_PATH={tmp_path / 'save' / 'usage.jsonl'}"
        in script
    )
    assert "--eval-interval 1 --n-samples-per-eval-prompt 1" in script
    assert "--eval-instance-interval 100" in script
    assert f"--eval-prompt-data fold0_val {tmp_path / 'val.jsonl'}" in script
    assert script.count("--save-interval 10") == 1
    assert "--train-instance-budget 1250" in script
    assert script.count(
        "--require-train-instance-budget-exhaustion"
    ) == 1
    assert "--stop-after-validation-attempt 100" in script
    assert "export RLER_CHECKPOINT_RETAIN_LATEST=2" in script
    assert script.count("--log-probs-chunk-size 32") == 1
    assert script.count("--sglang-mem-fraction-static 0.75") == 1
    assert script.count("--sglang-disable-custom-all-reduce") == 1
    assert "GRPO_COMMON_ARGS_SAVE_OVERRIDE" in script
    assert "GRPO_COMMON_ARGS_LOG_PROBS_OVERRIDE" in script
    assert "GRPO_SGLANG_ARGS_MEM_OVERRIDE" in script


@pytest.mark.parametrize(
    "extra_args, error",
    [
        (
            [
                "--eval-interval",
                "1",
                "--eval-prompt-data",
                "val",
                "/tmp/val.jsonl",
                "--n-samples-per-eval-prompt",
                "2",
            ],
            "n-samples-per-eval-prompt 1",
        ),
        (["--save-interval", "0"], "--save-interval must be positive"),
        (
            ["--eval-instance-interval", "0"],
            "--eval-instance-interval must be positive",
        ),
        (
            ["--train-instance-budget", "0"],
            "--train-instance-budget must be positive",
        ),
        (
            ["--require-train-instance-budget-exhaustion"],
            "--require-train-instance-budget-exhaustion requires "
            "--train-instance-budget",
        ),
        (
            ["--stop-after-validation-attempt", "100"],
            "--stop-after-validation-attempt requires "
            "--eval-instance-interval",
        ),
        (
            ["--checkpoint-retain-latest", "-1"],
            "--checkpoint-retain-latest must be non-negative",
        ),
        (
            ["--sglang-mem-fraction-static", "1.1"],
            "--sglang-mem-fraction-static must be in",
        ),
    ],
)
def test_grpo_rejects_invalid_fixed_protocol_overrides(
    monkeypatch, tmp_path, capsys, extra_args, error
):
    module = _load_grpo(monkeypatch)
    with pytest.raises(SystemExit):
        module.main([*_required_args(tmp_path), *extra_args])
    assert error in capsys.readouterr().err


def _write_resume_checkpoint(tmp_path: Path, rollout_id: int = 7) -> Path:
    load_dir = tmp_path / "load"
    (load_dir / f"iter_{rollout_id:07d}").mkdir(parents=True)
    (load_dir / "latest_checkpointed_iteration.txt").write_text(
        f"{rollout_id}\n"
    )
    dataset_dir = load_dir / "rollout"
    dataset_dir.mkdir()
    (dataset_dir / f"global_dataset_state_dict_{rollout_id}.pt").touch()
    return load_dir


def test_grpo_explicit_resume_restores_training_state(monkeypatch, tmp_path):
    module = _load_grpo(monkeypatch)
    _write_resume_checkpoint(tmp_path)
    captured = {}

    def _capture(command, check):
        captured["command"] = command
        captured["check"] = check

    monkeypatch.setattr(module.subprocess, "run", _capture)
    rc = module.main([*_required_args(tmp_path), "--resume"])

    assert rc == 0
    script = captured["command"][2]
    assert "GRPO_COMMON_ARGS_RESUME" in script
    assert (
        "--no-load-optim|--no-load-optim=*|--no-load-rng|--no-load-rng=*"
        in script
    )
    assert "--start-rollout-id=*" in script
    assert "export RLER_CHECKPOINT_RETAIN_LATEST=0" in script
    assert "export RLER_USAGE_RESUME=1" in script
    assert "Explicit resume: preserve the checkpoint's recorded rollout id" in script
    # The seed-checkpoint compatibility rewrite must never turn resume
    # checkpoint 0 into checkpoint 1.
    assert "for CHECKPOINT_DIR in" not in script


def test_grpo_resume_requires_matching_dataset_state(
    monkeypatch, tmp_path, capsys
):
    module = _load_grpo(monkeypatch)
    load_dir = _write_resume_checkpoint(tmp_path)
    (load_dir / "rollout" / "global_dataset_state_dict_7.pt").unlink()

    with pytest.raises(SystemExit):
        module.main([*_required_args(tmp_path), "--resume"])

    assert "dataset state paired with checkpoint 7" in capsys.readouterr().err


def test_grpo_attempt_resume_requires_exact_collector_position(
    monkeypatch,
    tmp_path,
    capsys,
):
    module = _load_grpo(monkeypatch)
    _write_resume_checkpoint(tmp_path)
    monkeypatch.setattr(
        module,
        "_dataset_state_has_exact_collector_position",
        lambda path, checkpoint_id: False,
    )

    with pytest.raises(SystemExit):
        module.main(
            [
                *_required_args(tmp_path),
                "--resume",
                "--train-instance-budget",
                "1250",
                "--eval-instance-interval",
                "128",
                "--eval-prompt-data",
                "fold0_val",
                str(tmp_path / "val.jsonl"),
                "--n-samples-per-eval-prompt",
                "1",
            ]
        )

    assert (
        "exact collector buffer/pending/counter state"
        in capsys.readouterr().err
    )


def test_grpo_auto_resume_starts_fresh_without_saved_checkpoint(
    monkeypatch,
    tmp_path,
):
    module = _load_grpo(monkeypatch)
    captured = {}
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, check: captured.update(
            {"command": command, "check": check}
        ),
    )

    assert module.main([*_required_args(tmp_path), "--auto-resume"]) == 0

    script = captured["command"][2]
    assert "GRPO_COMMON_ARGS_RESUME" not in script
    assert "export RLER_USAGE_RESUME=0" in script
    assert "for CHECKPOINT_DIR in" in script
    assert f"--load {tmp_path / 'load'}" in script


def test_grpo_auto_resume_uses_latest_complete_model_cursor_pair(
    monkeypatch,
    tmp_path,
):
    module = _load_grpo(monkeypatch)
    save_dir = tmp_path / "save"
    (save_dir / "iter_0000005").mkdir(parents=True)
    (save_dir / "iter_0000007").mkdir()
    (save_dir / "rollout").mkdir()
    (
        save_dir
        / "rollout"
        / "global_dataset_state_dict_5.pt"
    ).touch()
    tracker = save_dir / "latest_checkpointed_iteration.txt"
    tracker.write_text("7\n")
    captured = {}
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, check: captured.update(
            {"command": command, "check": check}
        ),
    )

    assert module.main([*_required_args(tmp_path), "--auto-resume"]) == 0

    assert tracker.read_text() == "5\n"
    script = captured["command"][2]
    assert "GRPO_COMMON_ARGS_RESUME" in script
    assert "after rollout 5" in script
    assert f"--load {save_dir}" in script
    # The reference model remains the immutable base load directory.
    assert f"--ref-load {tmp_path / 'load'}" in script


def test_grpo_auto_resume_rejects_numeric_tracker_without_paired_cursor(
    monkeypatch,
    tmp_path,
    capsys,
):
    module = _load_grpo(monkeypatch)
    save_dir = tmp_path / "save"
    (save_dir / "iter_0000007").mkdir(parents=True)
    (save_dir / "latest_checkpointed_iteration.txt").write_text("7\n")

    with pytest.raises(SystemExit):
        module.main([*_required_args(tmp_path), "--auto-resume"])

    assert "no model/data checkpoint pair" in capsys.readouterr().err
