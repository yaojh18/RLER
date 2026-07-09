import asyncio
import copy
import json

import swe_agent.run.aggregate_swe_agent as aggregate_swe_agent
import swe_agent.rubric_bank as rubric_bank
import swe_agent.trajectory_search as trajectory_search
from agent_rl.run_utils import extract_last_json_object
from swe_agent.trajectory_search import _normalized_weight_by_rubric, _pc_avg_scores_from_rubrics


def _rubric(title: str, *, weight: float = 1.0) -> dict:
    return {
        "polarity": "positive",
        "weight": weight,
        "title": title,
        "description": f"Use {title} to rank the complete trajectories.",
        "metadata": {"judge_focus": title},
        "scale": {str(score): f"Anchor {score}" for score in range(1, 6)},
    }


def _freeform_json(payload: dict, *, thought: str = "Analyze the evidence freely.") -> str:
    return f"THOUGHT: {thought}\n\n```json\n{json.dumps(payload)}\n```"


def _assert_freeform_call(kwargs: dict) -> None:
    assert "response_format" not in kwargs
    assert "enable_json_schema_validation" not in (kwargs.get("model_kwargs") or {})
    assert kwargs["model_kwargs"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def _run_generation(monkeypatch, responses: list[dict]) -> tuple[dict, list[list[dict]]]:
    captured_messages = []
    pending = iter(responses)

    async def fake_route_completion_message(**kwargs):
        _assert_freeform_call(kwargs)
        captured_messages.append(copy.deepcopy(kwargs["messages"]))
        content = _freeform_json(next(pending))
        return {
            "role": "assistant",
            "content": content,
            "content_no_thinking": content,
            "usage": {},
        }

    monkeypatch.setattr(aggregate_swe_agent, "route_completion_message", fake_route_completion_message)
    result = asyncio.run(
        aggregate_swe_agent._generate_aggregate_rubrics(
            question={"system_prompt": "", "user_prompt": "Fix the bug."},
            trajectory_summaries=[{"compressed_trajectory": {"key_results": "Changed source.py"}}],
            model_name="nvidia/nvidia/nemotron-3-ultra",
            temperature=0.0,
            top_p=1.0,
            max_tokens=8096,
            round_index=1,
        )
    )
    return result, captured_messages


def test_aggregate_rubrics_use_one_canonical_multiturn_conversation(monkeypatch):
    result, requests = _run_generation(monkeypatch, [_rubric("First"), _rubric("Second"), {}])

    assert [message["role"] for message in result["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert result["messages"][0]["content"].startswith(
        aggregate_swe_agent.AGGREGATE_RUBRIC_GENERATION_PROMPT.strip()
    )
    assert "**Weight**" in result["messages"][0]["content"]
    assert "THOUGHT:" in result["messages"][0]["content"]
    assert "```json" in result["messages"][0]["content"]
    assert result["messages"][2]["content"] == aggregate_swe_agent.AGGREGATE_RUBRIC_CONTINUE_PROMPT
    assert result["messages"][4]["content"] == aggregate_swe_agent.AGGREGATE_RUBRIC_CONTINUE_PROMPT
    assert [[message["role"] for message in request] for request in requests] == [
        ["user"],
        ["user", "assistant", "user"],
        ["user", "assistant", "user", "assistant", "user"],
    ]
    assert len(result["generated"]) == 2
    assert result["format_errors"] == []
    assert result["terminal_error"] is None
    assert result["stop_reason"] == "empty_object"


def test_aggregate_rubric_format_error_becomes_a_user_correction(monkeypatch):
    incomplete = {
        "polarity": "positive",
        "weight": 1.0,
        "description": "Missing title and metadata.",
        "scale": {str(score): f"Anchor {score}" for score in range(1, 6)},
    }
    result, requests = _run_generation(monkeypatch, [incomplete, _rubric("Recovered"), {}])

    roles = [message["role"] for message in result["messages"]]
    assert roles == ["user", "assistant", "user", "assistant", "user", "assistant"]
    assert "Expected a rubric object" in result["messages"][2]["content"]
    assert [[message["role"] for message in request] for request in requests] == [
        ["user"],
        ["user", "assistant", "user"],
        ["user", "assistant", "user", "assistant", "user"],
    ]
    assert len(result["generated"]) == 1
    assert len(result["format_errors"]) == 1
    assert result["terminal_error"] is None
    assert result["stop_reason"] == "empty_object"


def test_rubric_weights_are_positive_and_required_for_model_generation():
    item = _rubric("Weighted", weight=2.5)
    converted = aggregate_swe_agent._convert_rubric_item(
        "task",
        item,
        1,
    )

    assert converted is not None
    assert converted.weight == 2.5
    assert aggregate_swe_agent._convert_rubric_item(
        "task",
        {key: value for key, value in item.items() if key != "weight"},
        1,
    ) is None
    assert aggregate_swe_agent._convert_rubric_item(
        "task",
        {**item, "weight": 0},
        1,
    ) is None


def test_rubric_scores_use_generation_normalized_weighted_average():
    light = aggregate_swe_agent._convert_rubric_item(
        "task", _rubric("Light", weight=1), 1
    )
    heavy = aggregate_swe_agent._convert_rubric_item(
        "task", _rubric("Heavy", weight=3), 1
    )
    assert light is not None and heavy is not None
    rubrics = [light, heavy]
    scores = aggregate_swe_agent._avg_scores_from_rubrics(
        node_ids=["node-a", "node-b"],
        score_lookup_by_node={
            "node-a": {light.rubric_id: 1.0, heavy.rubric_id: 0.0},
            "node-b": {light.rubric_id: 0.0, heavy.rubric_id: 1.0},
        },
        rubrics=rubrics,
    )

    assert _normalized_weight_by_rubric(rubrics) == {
        light.rubric_id: 0.25,
        heavy.rubric_id: 0.75,
    }
    assert scores == {"node-a": 0.25, "node-b": 0.75}


def test_negative_rubric_uses_polarity_separately_from_importance():
    positive = aggregate_swe_agent._convert_rubric_item(
        "task", _rubric("Positive", weight=1), 1
    )
    negative_item = _rubric("Negative", weight=3)
    negative_item["polarity"] = "negative"
    negative = aggregate_swe_agent._convert_rubric_item("task", negative_item, 1)
    assert positive is not None and negative is not None
    scores = aggregate_swe_agent._avg_scores_from_rubrics(
        node_ids=["node"],
        score_lookup_by_node={"node": {positive.rubric_id: 0.8, negative.rubric_id: 0.2}},
        rubrics=[positive, negative],
    )

    assert abs(scores["node"] - 0.05) < 1e-12

    pc_scores = _pc_avg_scores_from_rubrics(
        node_ids=["node"],
        score_lookup_by_node={"node": {positive.rubric_id: 0.8, negative.rubric_id: 0.2}},
        rubrics=[positive, negative],
    )
    assert abs(pc_scores["node"] - 0.5) < 1e-12


def test_score_models_do_not_receive_rubric_weight(monkeypatch):
    rubric = aggregate_swe_agent._convert_rubric_item(
        "task", _rubric("Hidden Importance", weight=937.125), 1
    )
    assert rubric is not None
    captured_messages = []

    async def fake_route_completion_message(**kwargs):
        _assert_freeform_call(kwargs)
        captured_messages.append(copy.deepcopy(kwargs["messages"]))
        content = _freeform_json({"score": 4}, thought="Apply only the supplied criterion.")
        return {
            "role": "assistant",
            "content": content,
            "content_no_thinking": content,
            "usage": {},
        }

    monkeypatch.setattr(trajectory_search, "route_completion_message", fake_route_completion_message)
    asyncio.run(
        trajectory_search._score_round(
            question={"system_prompt": "System", "user_prompt": "Task"},
            shared_context={"previous_persistent_state": {}, "latest_agent_trajectory": []},
            continuations=[
                {"node_id": "node", "summary": {}, "trajectory_continuation": []}
            ],
            rubrics=[rubric],
            model_name="test-model",
            temperature=0.0,
            top_p=1.0,
            max_tokens=32,
        )
    )

    monkeypatch.setattr(aggregate_swe_agent, "route_completion_message", fake_route_completion_message)
    asyncio.run(
        aggregate_swe_agent._score_aggregate_summaries(
            question={"system_prompt": "System", "user_prompt": "Task"},
            trajectory_summaries=[{"key_results": "Result"}],
            rubrics=[rubric],
            model_name="test-model",
            temperature=0.0,
            top_p=1.0,
            max_tokens=32,
        )
    )

    assert len(captured_messages) == 2
    for messages in captured_messages:
        model_input = json.dumps(messages, ensure_ascii=False)
        assert "937.125" not in model_input
        assert "Weight:" not in model_input


def test_last_json_parser_ignores_all_text_before_and_after_the_final_object():
    response = _freeform_json(
        {"score": 4},
        thought='A draft object such as {"score": 1} is not the final answer.',
    )
    assert extract_last_json_object(response) == {"score": 4}
    assert extract_last_json_object('{"score": 4}') == {"score": 4}
    assert extract_last_json_object('```json\n{"score": 4}\n```') == {"score": 4}
    assert extract_last_json_object('<think>{"score": 1}</think>\n```json\n{"score": 4}\n```') == {"score": 4}
    assert extract_last_json_object('THOUGHT: draft</think>\n```json\n{"score": 4}\n```') == {"score": 4}
    assert extract_last_json_object(response + "\nextra prose") == {"score": 4}
    assert extract_last_json_object('{"metadata": {"draft": 1}, "score": 4}') == {
        "metadata": {"draft": 1},
        "score": 4,
    }
    assert trajectory_search._parse_persistent_state_response('prefix {"custom": 1} suffix') is None
    persistent = trajectory_search._parse_persistent_state_response(
        'prefix {"summary": {"Current State": "working", "worklog": ["one", "two"]}} suffix'
    )
    assert persistent is not None
    assert persistent["current_state"] == "working"
    assert "one" in persistent["worklog"]
    assert set(persistent) == {"current_state", "worklog"}
    assert trajectory_search._parse_judge_score('{"score": 4, "reason": "extra fields are allowed"}') == 4
    assert trajectory_search._parse_judge_score('{"score": "4"}') == 4
    assert trajectory_search._parse_judge_score('{"score": 4.0}') == 4
    assert trajectory_search._parse_judge_score("analysis\nFinal score: 4/5") == 4
    assert trajectory_search._parse_judge_score('{"rating": "3.0"}') == 3
    assert trajectory_search._parse_judge_score('{"score": 4.5}') is None


def test_relaxed_json_parser_recovers_unquoted_rubric_strings():
    response = '''Thought before the final answer.
```json
{
  "polarity": "negative",
  "weight": 1.0,
  "title": "Test expectations",
  "description": The fix changes the message but does not update exact assertions.
  "scale": {
    "1": All assertions are updated
    "2": Most assertions are updated,
    "3": "No assertions are updated",
    "4": "The change is submitted without verification",
    "5": "Known-broken assertions remain"
  },
  "metadata": {"stage": "tests"}
}
```
'''
    parsed = extract_last_json_object(response)
    assert parsed is not None
    assert parsed["description"].startswith("The fix changes")
    assert parsed["scale"]["1"] == "All assertions are updated"
    assert trajectory_search._convert_rubric_item("task", parsed, 2) is not None


def test_trajectory_judge_messages_start_at_first_assistant(monkeypatch):
    rubric = trajectory_search._convert_rubric_item("task", _rubric("Criterion"), 1)
    assert rubric is not None
    responses = iter(["not json", '{"score": "4"}'])

    async def fake_route_completion_message(**kwargs):
        content = next(responses)
        return {
            "role": "assistant",
            "content": content,
            "content_no_thinking": content,
        }

    monkeypatch.setattr(trajectory_search, "route_completion_message", fake_route_completion_message)
    scores, errors = asyncio.run(
        trajectory_search._score_round(
            question={"system_prompt": "System", "user_prompt": "Task"},
            shared_context={"previous_persistent_state": {}, "latest_agent_trajectory": []},
            continuations=[{"node_id": "node", "summary": {}, "trajectory_continuation": []}],
            rubrics=[rubric],
            model_name="test-model",
            temperature=0.0,
            top_p=1.0,
            max_tokens=32,
        )
    )

    assert errors == []
    assert scores[0][0]["score_raw"] == 4
    assert [message["role"] for message in scores[0][0]["judge_message"]] == [
        "assistant",
        "user",
        "assistant",
    ]


def test_trajectory_rubric_generation_uses_freeform_markdown_json(monkeypatch):
    calls = []
    responses = [
        _freeform_json(_rubric("Trajectory Criterion")),
        _freeform_json({}),
    ]

    async def fake_route_completion_message(**kwargs):
        _assert_freeform_call(kwargs)
        calls.append(copy.deepcopy(kwargs))
        content = responses.pop(0)
        return {
            "role": "assistant",
            "content": content,
            "content_no_thinking": content,
            "usage": {},
        }

    monkeypatch.setattr(trajectory_search, "route_completion_message", fake_route_completion_message)
    sample = asyncio.run(
        trajectory_search._generate_round_rubrics(
            question={"system_prompt": "System", "user_prompt": "Task"},
            previous_state=copy.deepcopy(trajectory_search.EMPTY_PERSISTENT_STATE),
            latest_shared_segment=None,
            continuations=[
                {"node_id": "node", "summary": {}, "trajectory_continuation": []}
            ],
            model_name="test-model",
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
            round_index=1,
            require_first_rubric=True,
        )
    )

    assert len(sample.generated) == 1
    assert "response_format" not in calls[0]
    assert "enable_json_schema_validation" not in (calls[0].get("model_kwargs") or {})
    assert "THOUGHT:" in calls[0]["messages"][0]["content"]
    assert [message["role"] for message in sample.messages] == ["user", "assistant", "user", "assistant"]
    assert "User Prompt:\nTask" in calls[0]["messages"][0]["content"]


def test_trajectory_rubric_generation_corrects_empty_first_response(monkeypatch):
    responses = [
        _freeform_json({}),
        _freeform_json(_rubric("Reused Criterion")),
        _freeform_json({}),
    ]

    async def fake_route_completion_message(**kwargs):
        content = responses.pop(0)
        return {
            "role": "assistant",
            "content": content,
            "content_no_thinking": content,
            "usage": {},
        }

    monkeypatch.setattr(trajectory_search, "route_completion_message", fake_route_completion_message)
    sample = asyncio.run(
        trajectory_search._generate_round_rubrics(
            question={"system_prompt": "System", "user_prompt": "Task"},
            previous_state=copy.deepcopy(trajectory_search.EMPTY_PERSISTENT_STATE),
            latest_shared_segment=None,
            continuations=[{"node_id": "node", "summary": {}, "trajectory_continuation": []}],
            model_name="test-model",
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
            round_index=2,
        )
    )

    assert [rubric.title for rubric in sample.generated] == ["Reused Criterion"]
    assert [message["role"] for message in sample.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert "Existing rubrics are not reused automatically" in sample.messages[2]["content"]


def test_aggregate_summary_uses_freeform_markdown_json(monkeypatch):
    calls = []
    summary = {key: f"value for {key}" for key in trajectory_search.EMPTY_PERSISTENT_STATE}

    async def fake_route_completion_message(**kwargs):
        _assert_freeform_call(kwargs)
        calls.append(copy.deepcopy(kwargs))
        content = _freeform_json(summary, thought="Compress the complete trajectory.")
        return {
            "role": "assistant",
            "content": content,
            "content_no_thinking": content,
            "usage": {},
        }

    monkeypatch.setattr(aggregate_swe_agent, "route_completion_message", fake_route_completion_message)
    result = asyncio.run(
        aggregate_swe_agent._summarize_aggregate_trajectory(
            question={"system_prompt": "System", "user_prompt": "Task"},
            step_cards=[
                {
                    "step_index": 0,
                    "assistant_message": "inspect",
                    "commands": ["run-tests"],
                    "observation": "formatted observation",
                }
            ],
            trajectory_metadata={"result_status": "finished"},
            model_name="test-model",
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
        )
    )

    assert result["state"] == summary
    assert result["format_error"] is None
    assert [message["role"] for message in result["messages"]] == ["system", "user", "assistant"]
    assert "response_format" not in calls[0]
    assert "enable_json_schema_validation" not in (calls[0].get("model_kwargs") or {})
    assert "THOUGHT:" in calls[0]["messages"][0]["content"]
    assert [message["role"] for message in calls[0]["messages"]] == ["system", "user"]
    summary_input = calls[0]["messages"][1]["content"]
    assert "## Question:" in summary_input
    assert "## Trajectory Metadata:" in summary_input
    assert "## Full Agent Trajectory:" in summary_input
    assert "### Step 0\nAssistant:\ninspect\nObservation:\nformatted observation" in summary_input
    assert "run-tests" not in summary_input


def test_aggregate_summary_retries_with_format_correction(monkeypatch):
    calls = []
    summary = {key: f"value for {key}" for key in trajectory_search.EMPTY_PERSISTENT_STATE}
    pending = iter(["no JSON object", "still no JSON object", _freeform_json(summary)])

    async def fake_route_completion_message(**kwargs):
        _assert_freeform_call(kwargs)
        calls.append(copy.deepcopy(kwargs))
        content = next(pending)
        return {
            "role": "assistant",
            "content": content,
            "content_no_thinking": content,
            "usage": {},
        }

    monkeypatch.setattr(aggregate_swe_agent, "route_completion_message", fake_route_completion_message)
    result = asyncio.run(
        aggregate_swe_agent._summarize_aggregate_trajectory(
            question={"system_prompt": "System", "user_prompt": "Task"},
            step_cards=[{"step": 1}],
            trajectory_metadata={"result_status": "finished"},
            model_name="test-model",
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
        )
    )

    assert result["state"] == summary
    assert result["format_error"] is None
    assert [message["role"] for message in calls[2]["messages"]][-4:] == [
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert "THOUGHT: <your reasoning process>" in calls[2]["messages"][-1]["content"]


def test_aggregate_summary_falls_back_to_full_content_for_json(monkeypatch):
    summary = {"current_state": "accepted"}

    async def fake_route_completion_message(**kwargs):
        _assert_freeform_call(kwargs)
        return {
            "role": "assistant",
            "content": f"<think>reasoning</think>\n{_freeform_json(summary)}",
            "content_no_thinking": "truncated before the final answer",
            "usage": {},
        }

    monkeypatch.setattr(aggregate_swe_agent, "route_completion_message", fake_route_completion_message)
    result = asyncio.run(
        aggregate_swe_agent._summarize_aggregate_trajectory(
            question={"system_prompt": "System", "user_prompt": "Task"},
            step_cards=[{"step": 1}],
            trajectory_metadata={"result_status": "finished"},
            model_name="test-model",
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
        )
    )

    assert result["state"]["current_state"] == "accepted"
    assert set(result["state"]) == set(trajectory_search.EMPTY_PERSISTENT_STATE)
    assert result["format_error"] is None


def test_aggregate_rubric_round_limit_is_a_normal_stop_after_a_valid_rubric(monkeypatch):
    monkeypatch.setattr(aggregate_swe_agent, "MAX_RUBRIC_GENERATION_ROUNDS", 2)
    result, _ = _run_generation(
        monkeypatch,
        [_rubric("Useful"), {"incomplete": "response"}],
    )

    assert len(result["generated"]) == 1
    assert result["stop_reason"] == "max_rounds"
    assert result["terminal_error"] is None


def test_source_step_cards_match_run_swe_agent_observation_truncation():
    output = "h" * 2_500 + "middle" + "t" * 2_500
    cards = aggregate_swe_agent._source_step_cards(
        [
            {"role": "user", "message": "task"},
            {
                "role": "assistant",
                "message": "<think>reasoning</think>\ninspect",
                "content": "<think>reasoning</think>\ninspect",
                "content_no_thinking": "inspect",
                "tool_calls": [{"command": "run-tests"}],
            },
            {"role": "user", "message": output},
        ]
    )

    assert len(cards) == 1
    assert cards[0]["assistant_message"] == "inspect"
    assert cards[0]["commands"] == ["run-tests"]
    observation = cards[0]["observation"]
    assert output[:1_500] in observation
    assert output[-1_500:] in observation
    assert "middle" not in observation
    assert "2006 characters elided" in observation


def test_aggregate_source_assistant_uses_its_own_truncation_budget():
    content = "h" * 2_500 + "middle" + "t" * 2_500
    cards = aggregate_swe_agent._source_step_cards(
        [
            {"role": "user", "message": "task"},
            {"role": "assistant", "content_no_thinking": content, "tool_calls": []},
            {"role": "user", "message": "done"},
        ]
    )

    assistant_message = cards[0]["assistant_message"]
    assert len(assistant_message) == aggregate_swe_agent.MAX_AGGREGATE_ASSISTANT_MESSAGE_CHARS
    assert assistant_message.startswith(content[:1_400])
    assert assistant_message.endswith(content[-1_400:])
    assert "Observation truncated due to length" in assistant_message
    assert "middle" not in assistant_message


def test_trajectory_search_plain_text_view_keeps_existing_card_content_without_commands():
    rendered = trajectory_search._render_step_cards(
        [
            {
                "step_index": 7,
                "assistant_message": "inspect",
                "commands": ["duplicated-command"],
                "observation": "line one\nline two",
            }
        ]
    )

    assert rendered == "### Step 7\nAssistant:\ninspect\nObservation:\nline one\nline two"
    assert "duplicated-command" not in rendered


def test_reset_run_dir_removes_stale_success_artifacts(tmp_path):
    run_dir = tmp_path / "instance" / "stable-run"
    (run_dir / "nodes" / "node_000").mkdir(parents=True)
    (run_dir / "rubric.json").write_text("{}", encoding="utf-8")
    (run_dir / "nodes" / "node_000" / "summary.json").write_text("{}", encoding="utf-8")

    aggregate_swe_agent._reset_run_dir(run_dir)

    assert run_dir.is_dir()
    assert list(run_dir.iterdir()) == []


def test_step_cards_bound_long_commands():
    cards = trajectory_search._build_step_cards(
        [
            {"kind": "model_response", "payload": {"message": {"content_no_thinking": "inspect"}}},
            {"kind": "environment_action", "payload": {"actions": [{"command": "x" * 4096}]}},
            {"kind": "environment_result", "payload": {"messages": [{"content": "done"}]}},
        ],
        step_start=0,
    )

    assert len(cards[0]["commands"][0]) <= trajectory_search.MAX_OBSERVATION_CHARS
    assert "truncated due to length" in cards[0]["commands"][0]


def test_experience_retrieval_uses_freeform_markdown_json(monkeypatch):
    bank = rubric_bank.ExperienceRubricBank()
    title = next(iter(bank.experiences))
    calls = []

    async def fake_route_completion_message(**kwargs):
        _assert_freeform_call(kwargs)
        calls.append(copy.deepcopy(kwargs))
        content = _freeform_json({"titles": [title]}, thought="Match prior experience applicability.")
        return {
            "role": "assistant",
            "content": content,
            "content_no_thinking": content,
            "usage": {},
        }

    monkeypatch.setattr(rubric_bank, "route_completion_message", fake_route_completion_message)
    retrieved, _ = asyncio.run(
        bank._retrieve(
            question={"system_prompt": "System", "user_prompt": "Task"},
            previous_state={},
            latest_shared_segment=None,
            continuations=[],
            model_name="test-model",
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
            model_kwargs=None,
        )
    )

    assert [item.title for item in retrieved] == [title]
    assert "response_format" not in calls[0]
    assert "THOUGHT:" in calls[0]["messages"][0]["content"]


def test_experience_update_uses_freeform_markdown_json(monkeypatch):
    bank = rubric_bank.ExperienceRubricBank()
    calls = []
    generated_rubric = _rubric("Generated Evidence")
    action = {
        "action": "add",
        "title": "Free Form Output Lesson",
        "description": "Use when structured decoding suppresses useful rubric analysis.",
        "context": "The evaluator needs room to compare trajectory evidence before committing to one action.",
        "experience": "Reason freely first, then emit one validated action in the final JSON block.",
        "metadata": {
            "analysis": "The visible evidence supports separating reasoning from the final action.",
            "reference_golden_rubrics": [_rubric("Reference Criterion")],
        },
    }
    pending = iter([action, {}])

    async def fake_route_completion_message(**kwargs):
        _assert_freeform_call(kwargs)
        calls.append(copy.deepcopy(kwargs))
        content = _freeform_json(next(pending), thought="Choose one durable bank action.")
        return {
            "role": "assistant",
            "content": content,
            "content_no_thinking": content,
            "usage": {},
        }

    monkeypatch.setattr(rubric_bank, "route_completion_message", fake_route_completion_message)
    applied, _ = asyncio.run(
        bank._update_bank_from_evidence(
            evidence={
                "instance_id": "demo",
                "rubric_attempts": [
                    {
                        "rubric_list_id": "rubric-demo",
                        "generated_rubrics": [generated_rubric],
                        "gt_skeleton": "",
                        "generated_rubric_accuracy": {},
                        "gt_scores": [0.0, 1.0],
                    }
                ],
            },
            model_name="test-model",
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
            model_kwargs=None,
        )
    )

    assert applied[0]["action"] == "add"
    assert all("response_format" not in call for call in calls)
    assert "THOUGHT:" in calls[0]["messages"][0]["content"]


def test_aggregate_rubric_duplicate_becomes_a_user_correction(monkeypatch):
    first = _rubric("First")
    same_title = _rubric("First")
    same_title["description"] = "Different wording for the same judging focus."
    result, requests = _run_generation(monkeypatch, [first, same_title, {}])

    assert [message["role"] for message in result["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert "duplicates one already generated" in result["messages"][4]["content"]
    assert [message["role"] for message in requests[-1]] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert len(result["generated"]) == 1
    assert len(result["format_errors"]) == 1
    assert result["terminal_error"] is None
    assert result["stop_reason"] == "empty_object"
