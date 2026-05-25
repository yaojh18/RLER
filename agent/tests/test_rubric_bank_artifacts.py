import asyncio
import json

from agent_rl import ChatCompletion
from agent_rl import run_utils
from swe_agent.prompt import JUDGE_RESPONSE_FORMAT
from swe_agent.rubric_bank import ExperienceRubricBank
from swe_agent.trajectory_search import SearchConfig, TrajectorySearchRunner


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
            response_format=JUDGE_RESPONSE_FORMAT,
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


def test_experience_bank_evidence_uses_average_judged_scores_and_does_not_write(tmp_path):
    bank_path = tmp_path / "siblings_rubric_bank.json"
    bank = ExperienceRubricBank(bank_path=bank_path, scope="siblings")
    payload = {
        "messages": [{"role": "user", "content": "prompt"}],
        "avg_scores": {"node-a": 0.2, "node-b": 0.8},
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
    assert (tmp_path / "rubrics" / "siblings" / "round_001" / "rubric_bank.json").exists()
    assert (tmp_path / "rubrics" / "siblings" / "round_001" / "rubric_bank_message.json").exists()
    assert (tmp_path / "rubrics" / "pc" / "round_002" / "rubric_bank.json").exists()
    assert runner.experience_banks["siblings"].seen_payloads[0]["messages"] == [
        {"role": "user", "content": "siblings"}
    ]
