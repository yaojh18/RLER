import uuid

from agent_rl import RolloutSessionSpec
from swe_agent.models.test_models import make_output
from swe_agent.rl_backend import SWEAgentRolloutBackend


def _build_backend() -> SWEAgentRolloutBackend:
    return SWEAgentRolloutBackend(
        model={
            "model_name": "deterministic",
            "model_class": "swe_agent.models.test_models.DeterministicModel",
            "outputs": [
                make_output("inspect workspace", [{"command": "printf 'step-one\\n'"}]),
                make_output(
                    "submit patch",
                    [{"command": "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\npatch complete\\n'"}],
                ),
            ],
        },
        environment={
            "environment_class": "local",
        },
        agent={
            "agent_class": "default",
            "system_template": "You are a coding agent.",
            "instance_template": "{{task}}",
            "step_limit": 4,
            "cost_limit": 0,
        },
    )


def test_swe_agent_session_pause_and_resume():
    backend = _build_backend()
    spec = RolloutSessionSpec(
        session_id=str(uuid.uuid4()),
        task="Print a line and then submit.",
        raw_user_query="Print a line and then submit.",
        policy_version="checkpoint-0001",
    )

    session = backend.create_session(spec)
    paused_result = session.run_until_pause(max_steps=1)

    assert paused_result.status == "paused"
    assert len(paused_result.model_turns) == 1
    assert paused_result.metadata["n_calls"] == 1
    assert paused_result.model_turns[0].response_message.content == "inspect workspace"

    snapshot = session.snapshot()
    resumed_session = backend.resume_session(snapshot)
    final_result = resumed_session.run_until_pause()

    assert final_result.status == "finished"
    assert final_result.exit_status == "Submitted"
    assert final_result.submission == "patch complete\n"
    assert len(final_result.model_turns) == 2
    assert len(final_result.model_turns[1].query_messages) > len(final_result.model_turns[0].query_messages)
    assert final_result.model_turns[1].metadata["policy_version"] == "checkpoint-0001"
    assert any(event.kind == "environment_action" for event in final_result.events)
