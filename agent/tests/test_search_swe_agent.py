from pathlib import Path
from types import SimpleNamespace

import swe_agent.run.search_swe_agent as search_module


def test_build_arg_parser_preserves_search_defaults():
    args = search_module.build_arg_parser().parse_args([])

    assert args.m == 4
    assert args.k == 20
    assert args.p == 2
    assert args.calculate_gt_reward is True


def test_run_search_vllm_routes_use_service_backend(monkeypatch, tmp_path: Path):
    configured_routes = {}
    registered_services = {}

    def fake_launch_vllm_server_handle(*args, **kwargs):
        assert kwargs["served_model_name"] == "Qwen/Qwen3.5-9B"
        return SimpleNamespace(
            model_name="Qwen/Qwen3.5-9B",
            base_url="http://127.0.0.1:8011/v1",
            command=["vllm"],
            log_file=Path("/tmp/vllm.log"),
            process=None,
        )

    def fake_load_instances(*args, **kwargs):
        return [{"instance_id": "psf__requests-1766"}]

    def fake_build_config(*args, **kwargs):
        return {}

    def fake_run_single_instance(**kwargs):
        raise RuntimeError("stop after route configuration")

    def fake_configure_model_route(name, config):
        configured_routes[name] = config

    def fake_register_model_service(name, service):
        registered_services[name] = service

    monkeypatch.setattr(search_module, "launch_vllm_server_handle", fake_launch_vllm_server_handle)
    monkeypatch.setattr(search_module, "load_swebench_instances", fake_load_instances)
    monkeypatch.setattr(search_module, "build_swebench_config", fake_build_config)
    monkeypatch.setattr(search_module, "_run_single_instance", fake_run_single_instance)
    monkeypatch.setattr(search_module, "run_harness_evaluation", lambda **kwargs: kwargs["results"])
    monkeypatch.setattr(search_module, "configure_model_route", fake_configure_model_route)
    monkeypatch.setattr(search_module, "register_model_service", fake_register_model_service)
    monkeypatch.setattr(search_module, "clear_model_routes", lambda: None)
    monkeypatch.setattr(search_module, "clear_model_services", lambda: None)
    monkeypatch.setattr(search_module, "terminate_process", lambda proc: None)
    monkeypatch.setattr(search_module, "choose_gpus", lambda gpu_id: [0, 1])
    monkeypatch.setattr(search_module, "find_free_port", lambda port: port)
    monkeypatch.setattr(search_module, "tee_console", lambda path: search_module.temporary_env({}))

    args = SimpleNamespace(
        backend="vllm",
        vllm_model="Qwen/Qwen3.5-9B",
        slime_model="Qwen/Qwen3.5-9B",
        openai_model="openai/Qwen/Qwen3.5-9B",
        subset="verified",
        split="test",
        output_root=tmp_path / "outputs",
        resume_run_dir=None,
        gpu_id="0,1",
        vllm_port=8011,
        max_model_len=32768,
        gpu_memory_utilization=0.9,
        allow_long_max_model_len=False,
        step_limit=10,
        environment_timeout=60,
        pull_timeout=60,
        eval_timeout=60,
        workers=1,
        policy_temperature=0.1,
        policy_top_p=0.9,
        completion_max_tokens=512,
        m=2,
        k=4,
        p=1,
        max_rounds=1,
        max_active_rubrics=6,
        rubric_temperature=0.1,
        rubric_top_p=0.9,
        rubric_max_tokens=512,
        judge_temperature=0.1,
        judge_top_p=0.9,
        judge_max_tokens=512,
        regression_margin=0.0,
        rubric_model=None,
        judge_model=None,
        calculate_gt_reward=False,
        model_retry_attempts=1,
    )

    results = search_module.run_search(args, ["psf__requests-1766"])

    assert len(results) == 1
    assert search_module.VLLM_SERVICE_NAME in registered_services
    assert configured_routes["policy"].backend == "service"
    assert configured_routes["policy"].service_name == search_module.VLLM_SERVICE_NAME
    assert configured_routes["policy"].model_name == "Qwen/Qwen3.5-9B"
    assert configured_routes["rubric_generation"].service_name == search_module.VLLM_SERVICE_NAME
    assert configured_routes["rubric_judge"].service_name == search_module.VLLM_SERVICE_NAME


def test_run_search_slime_routes_bind_service_name(monkeypatch, tmp_path: Path):
    configured_routes = {}
    registered_services = {}

    def fake_load_instances(*args, **kwargs):
        return [{"instance_id": "psf__requests-1766"}]

    def fake_build_config(*args, **kwargs):
        return {}

    def fake_run_single_instance(**kwargs):
        raise RuntimeError("stop after route configuration")

    def fake_configure_model_route(name, config):
        configured_routes[name] = config

    def fake_register_model_service(name, service):
        registered_services[name] = service

    monkeypatch.setattr(search_module, "load_swebench_instances", fake_load_instances)
    monkeypatch.setattr(search_module, "build_swebench_config", fake_build_config)
    monkeypatch.setattr(search_module, "_run_single_instance", fake_run_single_instance)
    monkeypatch.setattr(search_module, "run_harness_evaluation", lambda **kwargs: kwargs["results"])
    monkeypatch.setattr(search_module, "configure_model_route", fake_configure_model_route)
    monkeypatch.setattr(search_module, "register_model_service", fake_register_model_service)
    monkeypatch.setattr(search_module, "SLIME_API_BASE", "http://127.0.0.1:8021")
    monkeypatch.setattr(search_module, "SLIME_API_KEY", "EMPTY")
    monkeypatch.setattr(search_module, "clear_model_routes", lambda: None)
    monkeypatch.setattr(search_module, "clear_model_services", lambda: None)
    monkeypatch.setattr(search_module, "terminate_process", lambda proc: None)
    monkeypatch.setattr(search_module, "tee_console", lambda path: search_module.temporary_env({}))

    args = SimpleNamespace(
        backend="slime",
        vllm_model="Qwen/Qwen3.5-9B",
        slime_model="Qwen/Qwen3.5-9B",
        openai_model="openai/Qwen/Qwen3.5-9B",
        subset="verified",
        split="test",
        output_root=tmp_path / "outputs",
        resume_run_dir=None,
        gpu_id="0,1",
        vllm_port=8011,
        max_model_len=32768,
        gpu_memory_utilization=0.9,
        allow_long_max_model_len=False,
        step_limit=10,
        environment_timeout=60,
        pull_timeout=60,
        eval_timeout=60,
        workers=1,
        policy_temperature=0.1,
        policy_top_p=0.9,
        completion_max_tokens=512,
        m=2,
        k=4,
        p=1,
        max_rounds=1,
        max_active_rubrics=6,
        rubric_temperature=0.1,
        rubric_top_p=0.9,
        rubric_max_tokens=512,
        judge_temperature=0.1,
        judge_top_p=0.9,
        judge_max_tokens=512,
        regression_margin=0.0,
        rubric_model=None,
        judge_model=None,
        calculate_gt_reward=False,
        model_retry_attempts=1,
    )

    results = search_module.run_search(args, ["psf__requests-1766"])

    assert len(results) == 1
    assert search_module.SLIME_SERVICE_NAME in registered_services
    assert configured_routes["policy"].backend == "service"
    assert configured_routes["policy"].service_name == search_module.SLIME_SERVICE_NAME
    assert configured_routes["rubric_generation"].service_name == search_module.SLIME_SERVICE_NAME
    assert configured_routes["rubric_judge"].service_name == search_module.SLIME_SERVICE_NAME


def test_run_search_enables_qwen_thinking(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_load_instances(*args, **kwargs):
        return [{"instance_id": "psf__requests-1766"}]

    def fake_build_config(*args, **kwargs):
        captured["config"] = kwargs["extra_overrides"]
        return {}

    def fake_run_single_instance(**kwargs):
        captured["rubric_model_kwargs"] = kwargs["rubric_model_kwargs"]
        captured["judge_model_kwargs"] = kwargs["judge_model_kwargs"]
        raise RuntimeError("stop after config capture")

    monkeypatch.setattr(search_module, "load_swebench_instances", fake_load_instances)
    monkeypatch.setattr(search_module, "build_swebench_config", fake_build_config)
    monkeypatch.setattr(search_module, "_run_single_instance", fake_run_single_instance)
    monkeypatch.setattr(search_module, "run_harness_evaluation", lambda **kwargs: kwargs["results"])
    monkeypatch.setattr(search_module, "configure_model_route", lambda *args, **kwargs: None)
    monkeypatch.setattr(search_module, "register_model_service", lambda *args, **kwargs: None)
    monkeypatch.setattr(search_module, "clear_model_routes", lambda: None)
    monkeypatch.setattr(search_module, "clear_model_services", lambda: None)
    monkeypatch.setattr(search_module, "terminate_process", lambda proc: None)
    monkeypatch.setattr(search_module, "choose_gpus", lambda gpu_id: [0, 1])
    monkeypatch.setattr(search_module, "find_free_port", lambda port: port)
    monkeypatch.setattr(search_module, "launch_vllm_server_handle", lambda *args, **kwargs: SimpleNamespace(base_url="http://127.0.0.1:8011/v1", command=["vllm"], log_file=Path("/tmp/vllm.log"), process=None))
    monkeypatch.setattr(search_module, "tee_console", lambda path: search_module.temporary_env({}))

    args = SimpleNamespace(
        backend="vllm",
        vllm_model="Qwen/Qwen3.5-9B",
        slime_model="Qwen/Qwen3.5-9B",
        openai_model="openai/Qwen/Qwen3.5-9B",
        subset="verified",
        split="test",
        output_root=tmp_path / "outputs",
        resume_run_dir=None,
        gpu_id="0,1",
        vllm_port=8011,
        max_model_len=32768,
        gpu_memory_utilization=0.9,
        allow_long_max_model_len=False,
        step_limit=10,
        environment_timeout=60,
        pull_timeout=60,
        eval_timeout=60,
        workers=1,
        policy_temperature=0.1,
        policy_top_p=0.9,
        completion_max_tokens=512,
        m=2,
        k=4,
        p=1,
        max_rounds=1,
        max_active_rubrics=6,
        rubric_temperature=0.1,
        rubric_top_p=0.9,
        rubric_max_tokens=512,
        judge_temperature=0.1,
        judge_top_p=0.9,
        judge_max_tokens=512,
        regression_margin=0.0,
        rubric_model=None,
        judge_model=None,
        calculate_gt_reward=False,
        model_retry_attempts=1,
    )

    results = search_module.run_search(args, ["psf__requests-1766"])

    assert len(results) == 1
    extra_body = {"chat_template_kwargs": {"enable_thinking": True}}
    assert captured["config"]["model"]["model_kwargs"]["extra_body"] == extra_body
    assert captured["rubric_model_kwargs"]["extra_body"] == extra_body
    assert captured["judge_model_kwargs"]["extra_body"] == extra_body
