import pytest

from agent_rl import ModelTurn, RolloutSessionSpec
from open_instruct.agent_rollouts import (
    ExternalRolloutRunner,
    build_external_inference_batch,
    build_rollout_session_batch,
    get_rollout_backend_class,
    pack_model_turns,
    pack_rollout_results,
    tokenize_model_turn,
    tokenize_rollout_result,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=False):
        rendered = []
        for message in messages:
            rendered.append(f"<{message['role']}>{message['content']}</{message['role']}>")
        if add_generation_prompt:
            rendered.append("<assistant>")
        return list("".join(rendered).encode("utf-8"))


def _build_backend_kwargs():
    return {
        "model": {
            "model_name": "deterministic",
            "model_class": "swe_agent.models.test_models.DeterministicModel",
            "outputs": [
                {
                    "role": "assistant",
                    "content": "```mswea_bash_command\necho hi\n```",
                    "extra": {"actions": [{"command": "printf 'hi\\n'"}], "cost": 1.0},
                },
                {
                    "role": "assistant",
                    "content": "```mswea_bash_command\nprintf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\ndone\\n'\n```",
                    "extra": {
                        "actions": [{"command": "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\ndone\\n'"}],
                        "cost": 1.0,
                    },
                },
            ],
        },
        "environment": {"environment_class": "local"},
        "agent": {
            "agent_class": "default",
            "system_template": "You are a coding agent.",
            "instance_template": "{{task}}",
            "step_limit": 4,
            "cost_limit": 0,
        },
    }


def test_tokenize_and_pack_model_turns():
    turn = ModelTurn(
        session_id="session-1",
        step_index=0,
        query_messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Solve the task."},
        ],
        response_message={"role": "assistant", "content": "```mswea_bash_command\necho hi\n```"},
        reward=1.5,
        metadata={"dataset_name": "dummy"},
    )

    tokenizer = FakeTokenizer()
    tokenized = tokenize_model_turn(turn, tokenizer)
    packed = pack_model_turns([turn], tokenizer, pad_token_id=0, pack_length=512)

    assert tokenized.query_token_ids
    assert tokenized.response_token_ids
    assert tokenized.trainable_mask == [1] * len(tokenized.response_token_ids)
    assert packed.packed_sequences.query_responses
    assert packed.tokenized_turns[0].reward == 1.5


def test_rollout_backend_registry_loads_swe_agent():
    backend_cls = get_rollout_backend_class("swe_agent")
    assert backend_cls.__name__ == "SWEAgentRolloutBackend"


def test_external_rollout_runner_uses_swe_agent_backend():
    runner = ExternalRolloutRunner(
        "swe_agent",
        backend_kwargs=_build_backend_kwargs(),
        tokenizer=FakeTokenizer(),
        pad_token_id=0,
        pack_length=512,
    )

    results, packed = runner.run([RolloutSessionSpec(session_id="runner-1", task="Say hi and submit.")])

    assert results[0].status == "finished"
    assert results[0].submission == "done\n"
    assert len(packed.tokenized_turns) == 2


def test_tokenize_model_turn_requires_response_content():
    turn = ModelTurn(
        session_id="session-2",
        step_index=0,
        query_messages=[{"role": "user", "content": "task"}],
        response_message={"role": "assistant", "content": ""},
    )
    with pytest.raises(ValueError, match="empty response content"):
        tokenize_model_turn(turn, FakeTokenizer())


def test_episode_projection_and_external_batch_follow_open_instruct_contract():
    backend_cls = get_rollout_backend_class("swe_agent")
    backend = backend_cls(**_build_backend_kwargs())
    batch = build_rollout_session_batch(
        ["Inspect and submit."],
        [{"expected": "done"}],
        ["dummy"],
        training_step=3,
        num_samples_per_prompt_rollout=2,
        policy_version="checkpoint-3",
    )

    assert len(batch.specs) == 2
    assert batch.specs[0].group_id == batch.specs[1].group_id

    results = []
    for spec in batch.specs:
        session = backend.create_session(spec)
        results.append(session.run_until_pause())

    tokenizer = FakeTokenizer()
    projection = tokenize_rollout_result(results[0], tokenizer)
    packed = pack_rollout_results(results, tokenizer, pad_token_id=0, pack_length=512)
    inference_batch = build_external_inference_batch(results, tokenizer, pad_token_id=0, pack_length=512)

    assert projection.prompt_token_ids
    assert projection.continuation_token_ids
    assert any(mask == 0 for mask in projection.trainable_mask)
    assert any(mask == 1 for mask in projection.trainable_mask)
    assert projection.finish_reason == "stop"
    assert projection.policy_version == "checkpoint-3"
    assert packed.packed_sequences.query_responses
    assert len(inference_batch.responses) == 2
    assert inference_batch.finish_reasons == ["stop", "stop"]
    assert inference_batch.infos[0] == [2, 2]
    assert inference_batch.infos[-1] == [True, True]
