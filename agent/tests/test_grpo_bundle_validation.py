from swe_agent.lane_to_grpo_bundle import fork_group_to_export_group
from swe_agent.naive_search import NaiveRecord, NaiveRollout
from swe_agent.naive_to_grpo_bundle import naive_record_to_bundle
from swe_agent.trajectory_search_parallel import ForkGroup, LaneBBranch, MidCp


def _messages(*, valid: bool = True) -> list[dict]:
    assistant = {
        "role": "assistant",
        "content": "work",
        "prompt_token_ids": [1, 2],
        "token_ids": [3],
        "logprobs": [-0.1],
    }
    if not valid:
        assistant.pop("logprobs")
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "problem"},
        assistant,
    ]


def test_invalid_naive_rollout_drops_complete_group():
    record = NaiveRecord(
        instance_id="instance",
        run_dir="run",
        task_id="instance",
        config={},
        rollouts=[
            NaiveRollout(
                rollout_index=0,
                node_id="valid",
                messages=_messages(),
                gt_score=0.5,
            ),
            NaiveRollout(
                rollout_index=1,
                node_id="invalid",
                messages=_messages(valid=False),
                gt_score=0.0,
            ),
        ],
    )

    bundle = naive_record_to_bundle(record)

    assert bundle.policy_groups == []
    assert bundle.metadata["invalid_rollouts"] == ["invalid"]


def test_invalid_lane_branch_drops_complete_group():
    mid_cp = MidCp(
        idx=0,
        asst_step=1,
        image_tag="image",
        snapshot={"agent": {"state": {"messages": _messages()[:2]}}},
    )
    group = ForkGroup(
        group_index=0,
        mid_cp=mid_cp,
        branches=[
            LaneBBranch(
                group_index=0,
                branch_index=0,
                node_id="valid",
                parent_image_tag="image",
                messages=_messages()[2:],
                gt_score=0.5,
            ),
            LaneBBranch(
                group_index=0,
                branch_index=1,
                node_id="invalid",
                parent_image_tag="image",
                messages=_messages()[2:] + _messages(valid=False)[2:],
                gt_score=0.0,
            ),
        ],
    )

    assert fork_group_to_export_group(
        instance_id="instance",
        group=group,
        steps_per_round=1,
        gt_only_reward=True,
    ) is None
