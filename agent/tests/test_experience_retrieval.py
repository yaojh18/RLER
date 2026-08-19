import asyncio
import json

import swe_agent.experience_retrieval as experience_retrieval
from swe_agent.experience_retrieval import (
    CANDIDATE_LIMIT,
    DOCUMENT_WEIGHTS,
    QUERY_WEIGHTS,
    SUMMARY_MAX_TOKENS,
    SUMMARY_TEMPERATURE,
    WeightedKeywordExperienceRetriever,
)
from swe_agent.rubric_bank import ExperienceRubricBank, _retrieval_summary_context


def _write_checkpoint(root):
    records = [
        {
            "experience_id": "exp-alpha",
            "title": "Shared contract",
            "description": "Judge the alpha API contract.",
            "context": "Apply when the alpha API is visible.",
            "experience": "Require concrete alpha behavior.",
            "instance_label": "repo__blocked-1",
            "metadata": {
                "reference_golden_rubrics": [
                    {
                        "title": "Alpha evidence",
                        "description": "Inspect alpha behavior.",
                        "polarity": "positive",
                        "weight": 1.0,
                    }
                ]
            },
        },
        {
            "experience_id": "exp-beta",
            "title": "Shared contract",
            "description": "Judge the beta API contract.",
            "context": "Apply when the beta API is visible.",
            "experience": "Require concrete beta behavior.",
            "instance_label": None,
            "metadata": {"reference_golden_rubrics": []},
        },
    ]
    bank_dir = root / "bank" / "siblings"
    retrieval_dir = root / "retrieval"
    bank_dir.mkdir(parents=True)
    retrieval_dir.mkdir(parents=True)
    (bank_dir / "experience_bank.json").write_text(
        json.dumps({"experiences": records}), encoding="utf-8"
    )
    (retrieval_dir / "keyword_bank.json").write_text(
        json.dumps(
            {
                "bank": {
                    "siblings": {
                        "exp-alpha": {
                            "generated_keywords": ["alpha", "contract"],
                            "selected_keywords": ["alpha contract"],
                        },
                        "exp-beta": {
                            "generated_keywords": ["beta", "contract"],
                            "selected_keywords": ["beta contract"],
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_weighted_retriever_uses_only_frozen_instance_excluding_method(tmp_path):
    _write_checkpoint(tmp_path)
    retriever = WeightedKeywordExperienceRetriever(tmp_path, scope="siblings")
    context = {
        "question": {"user_prompt": "Repair the alpha contract."},
        "previous_state": {},
        "parent_trajectory": None,
        "continuations": [],
    }

    assert retriever.rank(context=context, summary={}, instance_id="other") == [
        "exp-alpha",
        "exp-beta",
    ]
    assert retriever.rank(
        context=context,
        summary={},
        instance_id="repo__blocked-1",
    ) == ["exp-beta"]

    bank = ExperienceRubricBank(bank_path=tmp_path, scope="siblings")
    assert bank.retriever is not None
    assert bank.bank_path == tmp_path / "bank" / "siblings" / "experience_bank.json"
    assert {item.experience_id for item in bank.experiences.values()} == {
        "exp-alpha",
        "exp-beta",
    }
    async def fake_retrieve(**kwargs):
        return ["exp-beta", "exp-alpha"], []

    bank.retriever.retrieve = fake_retrieve
    generation_context = asyncio.run(
        bank.build_generation_context(
            question=context["question"],
            previous_state={},
            latest_shared_segment=None,
            continuations=[],
            model_name="model",
            top_p=0.95,
            model_kwargs={},
            instance_id="other",
            round_index=1,
        )
    )
    assert [item.experience_id for item in generation_context.retrieved] == [
        "exp-beta",
        "exp-alpha",
    ]


def test_weighted_retriever_has_only_the_selected_frozen_configuration(tmp_path):
    _write_checkpoint(tmp_path)
    retriever = WeightedKeywordExperienceRetriever(tmp_path, scope="siblings")

    assert set(retriever.bm25) == {"full", "keywords"}
    assert CANDIDATE_LIMIT == 6
    assert QUERY_WEIGHTS == {
        "prior": 0.6,
        "stage": 0.69,
        "llm_contract": 1.48,
        "llm_state": 4.01,
    }
    assert DOCUMENT_WEIGHTS == {"full": 0.19, "keywords": 1.39}


def test_summary_corrects_malformed_quoted_code_example(monkeypatch, tmp_path):
    _write_checkpoint(tmp_path)
    retriever = WeightedKeywordExperienceRetriever(tmp_path, scope="siblings")
    responses = [
        '{"retrieval_queries":["mf.clean([\\"\\", "\\"]) returns"]}',
        '{"retrieval_queries":["MultiValueField required child"]}',
    ]

    calls = []

    async def fake_completion_message(**kwargs):
        calls.append(kwargs)
        content = responses.pop(0)
        return {
            "role": "assistant",
            "content": content,
            "content_no_thinking": content,
        }

    monkeypatch.setattr(
        experience_retrieval,
        "route_completion_message",
        fake_completion_message,
    )
    summary, messages = asyncio.run(
        retriever.summarize(
            context_markdown="state",
            stage="pre-patch",
            model_name="glm",
            top_p=0.95,
            model_kwargs={},
            max_format_correction_rounds=8,
        )
    )

    assert summary["retrieval_queries"] == [
        "MultiValueField required child"
    ]
    assert len([message for message in messages if message["role"] == "assistant"]) == 2
    assert messages[-1]["role"] == "assistant"
    assert all(call["temperature"] == SUMMARY_TEMPERATURE for call in calls)
    assert all(call["max_tokens"] == SUMMARY_MAX_TOKENS for call in calls)


def test_retrieval_summary_context_accepts_current_raw_continuation_key():
    segment = {
        "segment_step_range": [0, 2],
        "visible_step_card_count": 2,
        "step_cards": [
            {"step_index": 0, "assistant_message": "inspect", "observation": "seen"},
            {"step_index": 1, "assistant_message": "edit", "observation": "changed"},
        ],
    }
    base = {
        "question": {"user_prompt": "task"},
        "previous_state": {},
        "parent_trajectory": None,
        "continuations": [{"node_id": "node-a", "summary": {}}],
    }
    current = json.loads(json.dumps(base))
    current["continuations"][0]["raw_continuation"] = segment
    historical = json.loads(json.dumps(base))
    historical["continuations"][0]["trajectory_continuation"] = segment

    assert _retrieval_summary_context(
        current,
        instance_id="repo__pkg-1",
        round_index=1,
    ) == _retrieval_summary_context(
        historical,
        instance_id="repo__pkg-1",
        round_index=1,
    )
