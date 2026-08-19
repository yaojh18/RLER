from types import SimpleNamespace

from slime.utils import wandb_utils


def test_wandb_config_only_contains_training_dynamic_allowlist(monkeypatch):
    for env_name in wandb_utils._WANDB_DYNAMIC_ENV_CONFIG:
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("SWE_AGENT_MODEL_CONTEXT_LENGTH", "128000")
    monkeypatch.setenv("SWE_AGENT_VALIDATION_TEMPERATURE", "0.2")
    monkeypatch.setenv("SWE_AGENT_VALIDATION_TOP_P", "0.95")
    monkeypatch.setenv("SWE_AGENT_VALIDATION_GT_EVAL_TIMEOUT", "1800")
    monkeypatch.setenv("SWE_AGENT_LANES_TOPOLOGY", "depth2")
    monkeypatch.setenv("SWE_AGENT_LANES_M", "8")
    monkeypatch.setenv("SWE_AGENT_LANES_BEAM_PARENTS", "2")
    monkeypatch.setenv("SWE_AGENT_LANES_TERMINAL_ROLLOUT", "0")
    monkeypatch.setenv("SWE_AGENT_LANES_POLICY_TEMPERATURE", "1.0")
    monkeypatch.setenv("SWE_AGENT_LANES_POLICY_TOP_P", "0.95")
    monkeypatch.setenv("SWE_AGENT_LANES_INSTANCE_WORKERS", "16")
    monkeypatch.setenv("SWE_AGENT_LANES_MAX_PENDING", "10")
    args = SimpleNamespace(
        num_epoch=5,
        global_batch_size=16,
        lr=3e-6,
        rollout_temperature=1.0,
        eval_temperature=None,
        eval_top_p=None,
        use_critic=False,
        # Static identity and sensitive/runtime fields must not be uploaded.
        method="judge_direct_depth2",
        fold=0,
        hf_checkpoint="/models/Qwen3.6-27B",
        prompt_data=["fold0", "/datasets/train.jsonl"],
        wandb_key="super-secret",
        http_proxy="http://credential-bearing-proxy",
        nvidia_api_key="also-secret",
    )

    config = wandb_utils._compute_config_for_logging(args)

    assert config == {
        "num_epoch": 5,
        "global_batch_size": 16,
        "lr": 3e-6,
        "model_context_length": 128000,
        "validation_temperature": 0.2,
        "validation_top_p": 0.95,
        "validation_gt_eval_timeout": 1800,
        "lanes_topology": "depth2",
        "lanes_m": 8,
        "lanes_beam_parents": 2,
        "lanes_terminal_rollout": False,
        "lanes_policy_temperature": 1.0,
        "lanes_policy_top_p": 0.95,
        "lanes_instance_workers": 16,
        "lanes_max_pending": 10,
    }
    assert wandb_utils._compute_secondary_config_for_logging(args) == config
    assert "env_vars" not in config
    assert "wandb_key" not in config
    assert "hf_checkpoint" not in config
    assert "prompt_data" not in config
    assert "eval_temperature" not in config
    assert "eval_top_p" not in config
    assert "rollout_temperature" not in config
    assert "rollout_top_p" not in config


def test_usage_metrics_have_an_independent_event_axis(monkeypatch):
    calls = []

    def define_metric(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(wandb_utils.wandb, "define_metric", define_metric)

    wandb_utils._init_wandb_common()

    assert (("usage/event_step",), {}) in calls
    assert (
        ("usage/*",),
        {"step_metric": "usage/event_step"},
    ) in calls
    assert (
        ("swe_agent/*",),
        {"step_metric": "rollout/step"},
    ) in calls
    assert (("heartbeat/event_step",), {}) in calls
    assert (
        ("heartbeat/*",),
        {"step_metric": "heartbeat/event_step"},
    ) in calls
    cumulative_names = {
        "total_tokens_cumulative",
        "train_tokens_cumulative",
        "validation_tokens_cumulative",
        "qwen_tokens_cumulative",
        "glm_tokens_cumulative",
    }
    for metric_name in cumulative_names:
        assert (
            (f"usage/{metric_name}",),
            {"step_metric": "usage/event_step", "summary": "max"},
        ) in calls
    assert not any(args == ("usage/*_cumulative",) for args, _ in calls)


def test_wandb_history_metric_allowlist_is_opt_in_and_keeps_step(monkeypatch):
    metrics = {
        "rollout/step": 3,
        "swe_agent/reward_group_mean": 0.4,
        "swe_agent/groups_attempted": 16,
    }

    monkeypatch.delenv(wandb_utils._WANDB_METRIC_ALLOWLIST_ENV, raising=False)
    assert (
        wandb_utils.filter_metrics_for_logging(
            metrics,
            step_key="rollout/step",
        )
        == metrics
    )

    monkeypatch.setenv(
        wandb_utils._WANDB_METRIC_ALLOWLIST_ENV,
        "swe_agent/reward_group_mean",
    )
    assert wandb_utils.filter_metrics_for_logging(
        metrics,
        step_key="rollout/step",
    ) == {
        "rollout/step": 3,
        "swe_agent/reward_group_mean": 0.4,
    }


def test_primary_wandb_run_resumes_external_run_id(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setenv("WANDB_RUN_ID", "fold0-baseline-stable")
    monkeypatch.setattr(
        wandb_utils.wandb,
        "init",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        wandb_utils.wandb,
        "run",
        SimpleNamespace(id="fold0-baseline-stable"),
    )
    monkeypatch.setattr(wandb_utils, "_init_wandb_common", lambda: None)
    args = SimpleNamespace(
        use_wandb=True,
        wandb_mode="offline",
        wandb_key=None,
        wandb_host=None,
        wandb_random_suffix=False,
        wandb_group="f0-baseline",
        wandb_team="team",
        wandb_project="project",
        wandb_dir=str(tmp_path),
        rank=0,
        use_critic=False,
        num_rollout=1250,
    )

    wandb_utils.init_wandb_primary(args)

    assert captured["id"] == "fold0-baseline-stable"
    assert captured["resume"] == "allow"
    assert args.wandb_run_id == "fold0-baseline-stable"
