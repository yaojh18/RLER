import asyncio
import json
import time
from types import SimpleNamespace

from agent_rl import ChatCompletion
from agent_rl import run_utils
import swe_agent.trajectory_search as trajectory_search
from swe_agent.exceptions import FormatError
from swe_agent.models.litellm_textbased_model import LitellmTextbasedModel
from swe_agent.models.utils.actions_text import parse_regex_actions
from swe_agent.parallel_utils import NodeArtifactBundle
from swe_agent.parallel_utils import PatchEvalManager
from swe_agent.parallel_utils import RubricArtifactBundle
from swe_agent.parallel_utils import progress_reward
from swe_agent.rubric_bank import ExperienceRubricBank
from swe_agent.rubric_bank import RubricRecord
from swe_agent.rubric_bank import ScoreRubricBank
from swe_agent.run.search_swe_agent import _flush_experience_bank_summaries
from swe_agent.trajectory_search import (
    SearchConfig,
    TrajectorySearchRunner,
    _combine_score_results,
    _normalize_terminal_patch_text,
)


def test_route_completion_message_requires_reasoning_for_structured_sglang(monkeypatch):
    import swe_agent.tokenization as tokenization

    captured = {}

    monkeypatch.setattr(tokenization, "tokenize_messages_with_template", lambda *args, **kwargs: [1, 2, 3])
    monkeypatch.setattr(tokenization, "get_stop_token_ids", lambda *args, **kwargs: [4])

    async def fake_generate(**kwargs):
        captured.update(kwargs)
        return ChatCompletion(
            content="<think>reasoning</think>\n{\"score\": 1}",
            model_name=kwargs["route_name"],
            usage={"prompt_tokens": 3, "completion_tokens": 8, "total_tokens": 11},
            metadata={"content_no_thinking": "{\"score\": 1}"},
            input_token_ids=[1, 2, 3],
            output_token_ids=[5, 6],
            output_logprobs=[-0.1, -0.2],
        )

    monkeypatch.setattr(run_utils, "run_generate_with_route_async", fake_generate)

    message = asyncio.run(
        run_utils.route_completion_message(
            route_name="rubric_judge",
            model_name="Qwen/Qwen3.5-9B",
            messages=[{"role": "user", "content": "return a score"}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=64,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "generic_test_schema",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"score": {"type": "integer"}},
                        "required": ["score"],
                    },
                },
            },
            model_kwargs={
                "api_base": "http://127.0.0.1:8021/v1",
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
            },
        )
    )

    assert captured["require_reasoning"] is True
    assert "json_schema" in captured["sampling_params"]
    assert message["content"].startswith("<think>reasoning</think>")
    assert message["content_no_thinking"] == "{\"score\": 1}"
    assert message["prompt_token_ids"] == [1, 2, 3]
    assert message["token_ids"] == [5, 6]


def test_text_action_parser_turns_non_string_content_into_format_error():
    try:
        parse_regex_actions(
            None,
            action_regex=r"```mswea_bash_command\s*\n(.*?)\n```",
            format_error_template="{{ error }}",
        )
    except FormatError as exc:
        assert exc.messages[0]["role"] == "user"
        assert "content=None" in exc.messages[0]["content"]
        assert exc.messages[0]["extra"]["content_type_error"] is True
        assert exc.messages[0]["extra"]["model_response"] is None
    else:
        raise AssertionError("Expected FormatError")


def test_text_action_parser_reports_non_string_content_type():
    try:
        parse_regex_actions(
            {"text": "hello"},
            action_regex=r"```mswea_bash_command\s*\n(.*?)\n```",
            format_error_template="{{ error }}",
        )
    except FormatError as exc:
        assert "content of type dict" in exc.messages[0]["content"]
        assert "{'text': 'hello'}" in exc.messages[0]["content"]
        assert exc.messages[0]["extra"]["content_type_error"] is True
        assert exc.messages[0]["extra"]["model_response"] == {"text": "hello"}
    else:
        raise AssertionError("Expected FormatError")


def test_litellm_textbased_gemini_uses_timeout_non_streaming_client():
    model = LitellmTextbasedModel(model_name="gemini/gemini-3.1-pro-preview", model_kwargs={})

    kwargs = model._completion_kwargs({})

    assert kwargs["timeout"] > 0
    assert kwargs["stream"] is False
    assert kwargs["client"].__class__.__name__ == "HTTPHandler"


def test_route_litellm_completion_uses_mswea_timeout(monkeypatch):
    captured = {}

    def slow_completion(**kwargs):
        captured["timeout"] = kwargs.get("timeout")
        time.sleep(0.2)
        return SimpleNamespace(choices=[])

    monkeypatch.delenv("LITELLM_DEFAULT_TIMEOUT", raising=False)
    monkeypatch.setenv("MSWEA_LITELLM_TIMEOUT", "0.05")
    monkeypatch.setenv("LITELLM_OUTER_TIMEOUT_GRACE", "0")
    monkeypatch.setattr(run_utils.litellm, "completion", slow_completion)

    completion = asyncio.run(
        run_utils.run_litellm_completion_async(
            model_name="gemini/gemini-3.1-pro-preview",
            messages=[{"role": "user", "content": "score"}],
        )
    )

    assert captured["timeout"] == 0.05
    assert completion.content == ""
    assert completion.metadata["timeout"] == 0.05
    assert "TimeoutError" in completion.metadata["error"]


def test_progress_reward_returns_zero_for_tie_only_labels():
    assert progress_reward([0.5, 0.5], [0.1, 0.9]) == 0.0


def test_terminal_patch_normalization_preserves_patch_newline():
    patch = "diff --git a/a b/a\n@@ -1 +1 @@\n-a\n+b"

    normalized = _normalize_terminal_patch_text(patch)

    assert normalized.endswith("\n")
    assert normalized[:-1] == patch
    assert _normalize_terminal_patch_text(" \n\t") == ""


def test_score_result_combines_active_bank_and_generated_scores():
    active_score = {
        "rubric_id": "active-rubric",
        "score_normalized": 0.25,
    }
    generated_score = {
        "rubric_id": "generated-rubric",
        "score_normalized": 0.75,
    }

    combined_scores, combined_errors = _combine_score_results(
        ([[active_score]], [{"rubric_id": "active-rubric", "error": "retry"}]),
        ([[generated_score]], []),
        continuation_count=1,
    )

    assert [score["rubric_id"] for score in combined_scores[0]] == ["active-rubric", "generated-rubric"]
    assert combined_errors == [{"rubric_id": "active-rubric", "error": "retry"}]


def test_score_rubric_bank_keeps_active_rubrics_with_current_reward():
    bank = ScoreRubricBank(max_active_rubrics=2)
    bank.set_state(
        active_bank=[
            RubricRecord(
                rubric_id="active-rubric",
                title="Active",
                direction="positive",
                description="Existing active rubric.",
                scale={str(i): str(i) for i in range(1, 6)},
                weight=1,
                source_round=0,
                reward=None,
            )
        ],
        inactive_bank=[],
    )

    update = bank.update_after_round(generated=[], rewards={"active-rubric": 0.5})

    assert update.active_after[0].rubric_id == "active-rubric"
    assert update.active_after[0].reward == 0.5


def test_patch_eval_manager_writes_error_payload_when_evaluation_raises(tmp_path):
    task_id = "repo__pkg-1"
    node_dir = tmp_path / "nodes" / "node-a"
    rubric_dir = tmp_path / "rubrics" / "siblings" / "rubric-r001-s00"

    bundle = NodeArtifactBundle(
        node_id="node-a",
        node_dir=node_dir,
        node_payload={"node_id": "node-a", "parent_id": "root"},
        messages_payload=[{"role": "assistant", "content": "done"}],
        judge_payload={"overall_score": 0.75, "ground_truth_reward": None},
        prompt_payload={"messages": [{"role": "user", "content": "task"}]},
        terminal_patch_payload={
            task_id: {
                "model_patch": "diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-a\n+b\n"
            }
        },
    )
    rubric_bundle = RubricArtifactBundle(
        rubric_dir=rubric_dir,
        rubric_payload={
            "rubric_list_id": "rubric-r001-s00",
            "parent_node_id": "root",
            "generated": [],
            "format_errors": [],
            "terminal_error": None,
            "average_rubric_judged_scores": {"node-a": 0.75},
            "score_by_rubric": {"rubric-1": {"node-a": 0.75}},
        },
        messages_payload=[{"role": "assistant", "content": "{}"}],
    )

    def raising_eval(**kwargs):
        raise RuntimeError("harness crashed")

    manager = PatchEvalManager(
        instance={"instance_id": task_id},
        task_id=task_id,
        model_name="model",
        namespace=None,
        work_dir=tmp_path,
        evaluate_patches_fn=raising_eval,
        write_artifacts=True,
    )
    try:
        manager.submit_round([bundle], [rubric_bundle])
        [result] = manager.wait()
    finally:
        manager.close()

    evaluation = json.loads((node_dir / "terminal_evalution.json").read_text())
    judge = json.loads((node_dir / "judge.json").read_text())
    rubric = json.loads((rubric_dir / "rubric.json").read_text())
    assert evaluation["status"] == "error"
    assert evaluation["reward"] == 0.0
    assert "harness crashed" in evaluation["error"]
    assert judge["ground_truth_reward"] == 0.0
    assert rubric["gt_by_rubric"]["rubric-1"]["ground_truth_by_node"] == {"node-a": 0.0}
    assert result["rubric_update_payloads"][0]["scope"] == "siblings"


def test_rubric_scope_judges_active_bank_once_before_generated(monkeypatch):
    active = RubricRecord(
        rubric_id="active-rubric",
        title="Active",
        direction="positive",
        description="Existing active rubric.",
        scale={str(i): str(i) for i in range(1, 6)},
        weight=1,
        source_round=0,
    )
    generated = RubricRecord(
        rubric_id="generated-rubric",
        title="Generated",
        direction="positive",
        description="New rubric.",
        scale={str(i): str(i) for i in range(1, 6)},
        weight=1,
        source_round=1,
    )
    score_bank = ScoreRubricBank(max_active_rubrics=2)
    score_bank.set_state(active_bank=[active], inactive_bank=[])
    calls = []

    async def fake_score_round(**kwargs):
        rubric_ids = [rubric.rubric_id for rubric in kwargs["rubrics"]]
        calls.append(("score", rubric_ids))
        if rubric_ids == ["active-rubric"]:
            return (
                [
                    [{"rubric_id": "active-rubric", "score_normalized": 1.0}],
                    [{"rubric_id": "active-rubric", "score_normalized": 0.0}],
                ],
                [],
            )
        raise AssertionError(f"Unexpected direct score call: {rubric_ids}")

    async def fake_generate_and_score_batch(**kwargs):
        calls.append(("generate", None))
        return {
            "generated_samples": [
                SimpleNamespace(
                    sample_index=0,
                    rubric_list_id="rubric-r001-s00",
                    generated=[generated],
                    messages=[],
                    format_errors=[],
                    terminal_error=None,
                )
            ],
            "sample_generated_rubrics": [[generated]],
            "generated_score_results": {
                0: (
                    [
                        [{"rubric_id": "generated-rubric", "score_normalized": 0.0}],
                        [{"rubric_id": "generated-rubric", "score_normalized": 0.0}],
                    ],
                    [],
                )
            },
        }

    monkeypatch.setattr(trajectory_search, "_score_round", fake_score_round)
    monkeypatch.setattr(trajectory_search, "_generate_and_score_rubric_batch", fake_generate_and_score_batch)
    runner = TrajectorySearchRunner.__new__(TrajectorySearchRunner)
    runner.search_config = SearchConfig(n=1, rubric_temperature=0.0, rubric_top_p=1.0, rubric_max_tokens=16)
    runner.rubric_model_name = "dummy"
    runner.rubric_model_kwargs = {}

    result = asyncio.run(
        runner._run_rubric_scope_judging(
            scope_spec={
                "scope": "siblings",
                "score_bank": score_bank,
                "experience_bank": None,
                "generation_prompt": "generate",
                "judge_prompt": "judge",
                "rubric_list_prefix": "rubric",
                "include_variance_reward": True,
            },
            question={"system_prompt": "system", "user_prompt": "task"},
            shared_context={"previous_persistent_state": {}, "latest_agent_trajectory": None},
            previous_state={},
            latest_shared_segment=None,
            continuations=[{"node_id": "node-a"}, {"node_id": "node-b"}],
            node_ids=["node-a", "node-b"],
            round_index=1,
            generation_kwargs={},
            judge_kwargs={},
        )
    )

    assert calls == [("score", ["active-rubric"]), ("generate", None)]
    sample = result["samples"][0]
    assert sample["average_rubric_judged_scores"] == {"node-a": 0.5, "node-b": 0.0}
    assert sample["active_after"][0].rubric_id == "active-rubric"


def test_experience_bank_evidence_uses_average_judged_scores_and_does_not_write(tmp_path):
    bank_path = tmp_path / "siblings_rubric_bank.json"
    bank = ExperienceRubricBank(bank_path=bank_path, scope="siblings")
    payload = {
        "messages": [{"role": "user", "content": "prompt"}],
        "average_rubric_judged_scores": {"node-a": 0.2, "node-b": 0.8},
        "generated": [
            {
                "rubric_id": "rubric-1",
                "direction": "positive",
                "title": "Semantic Fix",
                "description": "Rewards a real fix.",
                "metadata": {},
                "scale": {str(i): str(i) for i in range(1, 6)},
            }
        ],
        "score_by_rubric": {"rubric-1": {"node-a": 0.0, "node-b": 1.0}},
        "gt_by_rubric": {
            "rubric-1": {
                "ground_truth_by_node": {"node-a": 0.0, "node-b": 1.0},
            }
        },
    }

    evidence = bank._build_instance_evidence(
        instance={"instance_id": "demo", "patch": "diff --git a/a b/a\n"},
        rubric_payloads=[payload],
    )

    assert evidence["rubric_attempts"][0]["average_rubric_judged_scores"] == [0.2, 0.8]
    assert "avg_scores" not in evidence["rubric_attempts"][0]

    result = asyncio.run(
        bank.update_after_instance(
            instance={"instance_id": "demo", "patch": ""},
            rubric_payloads=[],
            model_name="dummy",
            temperature=0.0,
            top_p=1.0,
            max_tokens=16,
        )
    )

    assert "before" in result and "after" in result
    assert not bank_path.exists()


def test_trajectory_search_experience_updates_are_written_by_runner(tmp_path):
    class FakeBank:
        def __init__(self, scope):
            self.scope = scope
            self.seen_payloads = None

        async def update_after_instance(self, **kwargs):
            self.seen_payloads = kwargs["rubric_payloads"]
            round_index = kwargs["rubric_payloads"][0]["round_index"]
            return {
                "before": [{"title": f"{self.scope}-before"}],
                "after": [{"title": f"{self.scope}-after"}],
                "groups": [
                    {
                        "round_index": round_index,
                        "before": [{"title": f"{self.scope}-before"}],
                        "actions": [{"action": "add"}],
                        "after": [{"title": f"{self.scope}-after"}],
                        "messages": [{"role": "user", "content": f"{self.scope}-update"}],
                    }
                ],
            }

    runner = TrajectorySearchRunner.__new__(TrajectorySearchRunner)
    runner.run_dir = tmp_path
    runner.rubrics_dir = tmp_path / "rubrics"
    runner.rubric_scopes = ("siblings", "pc")
    runner.experience_banks = {"siblings": FakeBank("siblings"), "pc": FakeBank("pc")}
    runner.instance = {"instance_id": "demo", "patch": ""}
    runner.rubric_model_name = "dummy"
    runner.rubric_model_kwargs = {}
    runner.search_config = SearchConfig(write_artifacts=True)

    runner._update_experience_banks(
        [
            {
                "scope": "siblings",
                "round_index": 1,
                "rubric_payload": {"rubric_list_id": "rubric-r001-s00"},
                "messages": [{"role": "user", "content": "siblings"}],
            },
            {
                "scope": "pc",
                "round_index": 2,
                "rubric_payload": {"rubric_list_id": "pc-rubric-r002-s00"},
                "messages": [{"role": "user", "content": "pc"}],
            },
        ]
    )

    assert (tmp_path / "siblings_rubric_bank.json").exists()
    assert (tmp_path / "pc_rubric_bank.json").exists()
    assert not (tmp_path / "rubric_bank.json").exists()
    assert json.loads((tmp_path / "siblings_rubric_bank.json").read_text())["after"] == [
        {"title": "siblings-after"}
    ]
    sibling_bank = tmp_path / "rubrics" / "siblings" / "rubric-r001-s00" / "rubric_bank.json"
    sibling_message = tmp_path / "rubrics" / "siblings" / "rubric-r001-s00" / "rubric_bank_message.json"
    pc_bank = tmp_path / "rubrics" / "pc" / "pc-rubric-r002-s00" / "rubric_bank.json"
    assert sibling_bank.exists()
    assert sibling_message.exists()
    assert pc_bank.exists()
    assert not (tmp_path / "rubrics" / "siblings" / "round_001" / "rubric_bank.json").exists()
    assert sorted(json.loads(sibling_bank.read_text())) == ["after", "before", "generated"]
    assert runner.experience_banks["siblings"].seen_payloads[0]["messages"] == [
        {"role": "user", "content": "siblings"}
    ]


def test_search_runner_flushes_final_root_experience_bank_summary(tmp_path):
    class FakeRootBank:
        def __init__(self, values):
            self.values = values

        def to_list(self):
            return list(self.values)

    banks = {
        "siblings": FakeRootBank([{"title": "siblings-initial"}]),
        "pc": FakeRootBank([{"title": "pc-initial"}]),
    }
    initial = {scope: bank.to_list() for scope, bank in banks.items()}

    banks["siblings"].values.append({"title": "siblings-final"})
    banks["pc"].values.append({"title": "pc-final"})

    _flush_experience_bank_summaries(
        run_root=tmp_path,
        experience_banks=banks,
        initial_snapshots=initial,
        write_artifacts=True,
    )

    siblings = json.loads((tmp_path / "siblings_rubric_bank.json").read_text())
    pc = json.loads((tmp_path / "pc_rubric_bank.json").read_text())

    assert siblings == {
        "before": [{"title": "siblings-initial"}],
        "after": [{"title": "siblings-initial"}, {"title": "siblings-final"}],
    }
    assert pc == {
        "before": [{"title": "pc-initial"}],
        "after": [{"title": "pc-initial"}, {"title": "pc-final"}],
    }
