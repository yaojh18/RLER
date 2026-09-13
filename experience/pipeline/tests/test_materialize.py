import json
from pathlib import Path

from experience_gen.materialize import materialize_checkpoint


def _card(experience_id: str) -> dict:
    return {
        "experience_id": experience_id,
        "title": "Visible contract boundary",
        "description": "Retrieve for this visible implementation boundary.",
        "context": "Apply after the relevant source has been inspected.",
        "experience": "Prefer concrete causal evidence over a proposed change.",
        "metadata": {"reference_golden_rubrics": []},
    }


def test_materializer_writes_the_runtime_three_file_schema(tmp_path: Path):
    base = tmp_path / "base"
    for scope in ("siblings", "pc"):
        bank_path = base / f"bank/{scope}/experience_bank.json"
        bank_path.parent.mkdir(parents=True, exist_ok=True)
        bank_path.write_text(
            json.dumps({"experience_count": 1, "experiences": [_card(f"{scope}-base")]}),
            encoding="utf-8",
        )
    keyword_path = base / "retrieval/keyword_bank.json"
    keyword_path.parent.mkdir(parents=True, exist_ok=True)
    keyword_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "experience_count": 2,
                "scope_counts": {"siblings": 1, "pc": 1},
                "bank": {
                    scope: {
                        f"{scope}-base": {
                            "generated_keywords": [f"{scope} base"],
                            "selected_keywords": [f"{scope} base"],
                        }
                    }
                    for scope in ("siblings", "pc")
                },
            }
        ),
        encoding="utf-8",
    )
    result_path = tmp_path / "accepted.json"
    result_path.write_text(
        json.dumps(
            {
                "scope": "siblings",
                "accepted_attempt": {"card": _card("siblings-new")},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    result = materialize_checkpoint(
        base=base,
        output=output,
        accepted_results=[result_path],
        keyword_records={
            "siblings": [
                {
                    "experience_id": "siblings-new",
                    "base_keywords": ["visible boundary"],
                    "model_keywords": ["concrete source evidence"],
                    "query_keywords": ["source inspection"],
                    "keywords": ["visible boundary", "source inspection"],
                }
            ]
        },
    )
    assert result["scope_counts"] == {
        "siblings": 2,
        "pc": 1,
    }
    keyword = json.loads(
        (output / "retrieval/keyword_bank.json").read_text(encoding="utf-8")
    )
    assert keyword["bank"]["siblings"]["siblings-new"] == {
        "generated_keywords": ["visible boundary", "concrete source evidence"],
        "selected_keywords": ["visible boundary", "source inspection"],
    }
    assert not (output / "retrieval/card_features.json").exists()
    assert not (output / "retrieval/recall_config.json").exists()
    assert not (output / "bank/siblings/lineage.json").exists()
