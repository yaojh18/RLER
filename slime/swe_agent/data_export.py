from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import ExportGroup, ExportSample, GRPOExportBundle, SFTExportBundle


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    role = str(message.get("role") or "")
    content = message.get("content")
    if content is None:
        content = message.get("message", "")
    normalized = {
        "role": role,
        "content": str(content or ""),
    }
    if "content_no_thinking" in message:
        normalized["content_no_thinking"] = str(message.get("content_no_thinking") or "")
    return normalized


class _RunArtifacts:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.manifest = _load_json(run_dir / "run_manifest.json")
        self.instance_id = str(self.manifest["instance_id"])
        self.system_prompt = str(self.manifest.get("system_prompt") or "")
        self.user_prompt = str(self.manifest.get("user_prompt") or "")
        self.nodes: dict[str, dict[str, Any]] = {}
        self.node_messages: dict[str, list[dict[str, str]]] = {}
        self.node_judges: dict[str, dict[str, Any]] = {}
        self.round_payloads: dict[int, dict[str, Any]] = {}
        self.rubric_messages: dict[str, list[dict[str, str]]] = {}
        self.rubric_payloads: dict[str, dict[str, Any]] = {}

        node_index_path = run_dir / "node_index.jsonl"
        if node_index_path.exists():
            for raw_line in node_index_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                self.nodes[str(payload["node_id"])] = payload
        else:
            for node_path in sorted((run_dir / "nodes").glob("*/node.json")):
                payload = _load_json(node_path)
                self.nodes[str(payload["node_id"])] = payload

        for node_id in sorted(self.nodes):
            node_dir = run_dir / "nodes" / node_id
            messages_path = node_dir / "messages.json"
            judge_path = node_dir / "judge.json"
            if messages_path.exists():
                payload = _load_json(messages_path)
                self.node_messages[node_id] = [_normalize_message(message) for message in payload.get("messages", [])]
            else:
                self.node_messages[node_id] = []
            if judge_path.exists():
                self.node_judges[node_id] = _load_json(judge_path)
            else:
                self.node_judges[node_id] = {}

        for round_path in sorted((run_dir / "rubrics").glob("round_*.json")):
            payload = _load_json(round_path)
            self.round_payloads[int(payload["round_index"])] = payload

        for rubric_dir in sorted(path for path in (run_dir / "rubrics").iterdir() if path.is_dir()):
            rubric_id = rubric_dir.name
            messages_path = rubric_dir / "messages.json"
            rubric_path = rubric_dir / "rubric.json"
            if messages_path.exists():
                payload = _load_json(messages_path)
                self.rubric_messages[rubric_id] = [_normalize_message(message) for message in payload.get("messages", [])]
            if rubric_path.exists():
                self.rubric_payloads[rubric_id] = _load_json(rubric_path)

    def build_group_id(self, round_index: int, parent_id: str | None) -> str:
        return f"{self.instance_id}:round:{round_index}:parent:{parent_id or 'root'}"

    def build_prefix_messages(self, parent_node_id: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if self.user_prompt:
            messages.append({"role": "user", "content": self.user_prompt})

        lineage: list[str] = []
        current = parent_node_id
        while current and current != "root":
            lineage.append(current)
            current = self.nodes.get(current, {}).get("parent_id")
        for node_id in reversed(lineage):
            messages.extend(self.node_messages.get(node_id, []))
        return messages

    def build_rubric_sft_prompt_messages(self, conversation: list[dict[str, Any]], stop_index: int) -> list[dict[str, str]]:
        prompt: list[dict[str, str]] = []
        for message in conversation[:stop_index]:
            content = message.get("content")
            if message.get("role") == "assistant" and "content_no_thinking" in message:
                content = message.get("content_no_thinking")
            prompt.append(
                {
                    "role": str(message.get("role") or ""),
                    "content": str(content or ""),
                }
            )
        return prompt

    def build_policy_sft_samples(self, round_payload: dict[str, Any]) -> tuple[str, list[ExportSample]]:
        round_index = int(round_payload["round_index"])
        parent_id = round_payload.get("parent_id")
        group_id = self.build_group_id(round_index, parent_id)

        teacher_scores: list[float] = []
        student_scores: list[float] = []
        teacher_samples: list[ExportSample] = []
        for item in round_payload.get("node_scores", []):
            node_id = str(item.get("node_id") or "")
            node = self.nodes.get(node_id)
            judge = self.node_judges.get(node_id, {})
            if not node or not judge:
                continue
            overall_reward = judge.get("overall_reward")
            if overall_reward is None:
                continue
            if str(node.get("policy_source") or "") == "student":
                student_scores.append(float(overall_reward))
                continue
            if float(judge.get("ground_truth_reward") or 0.0) < 1.0:
                continue
            teacher_scores.append(float(overall_reward))
            messages = list(self.node_messages.get(node_id, []))
            if not messages:
                continue
            prefix_messages = self.build_prefix_messages(node.get("parent_id"))
            local_prefix: list[dict[str, str]] = []
            for turn_index, message in enumerate(messages):
                if message.get("role") != "assistant":
                    local_prefix.append(message)
                    continue
                teacher_samples.append(
                    ExportSample(
                        sample_id=f"{group_id}:policy:{node_id}:{turn_index}",
                        group_id=group_id,
                        prompt=prefix_messages + list(local_prefix),
                        turns=[message],
                        metadata={
                            "node_id": node_id,
                            "round_index": round_index,
                            "parent_id": parent_id,
                            "policy_source": "teacher",
                            "turn_index": turn_index,
                        },
                    )
                )
                local_prefix.append(message)

        if not teacher_samples or max(teacher_scores, default=float("-inf")) <= max(student_scores, default=float("-inf")):
            return group_id, []
        return group_id, teacher_samples

    def build_rubric_sft_samples(self, round_payload: dict[str, Any], group_id: str) -> list[ExportSample]:
        samples: list[ExportSample] = []
        seen_rubric_lists: set[str] = set()
        for sample_payload in round_payload.get("rubric_samples", []):
            rubric_list_id = str(sample_payload.get("rubric_list_id") or "")
            if not rubric_list_id or rubric_list_id in seen_rubric_lists:
                continue
            seen_rubric_lists.add(rubric_list_id)
            rubric_payload = self.rubric_payloads.get(rubric_list_id)
            conversation = self.rubric_messages.get(rubric_list_id, [])
            if not rubric_payload or not conversation:
                continue
            generated = list(rubric_payload.get("generated", []))
            assistant_indices = [index for index, message in enumerate(conversation) if message.get("role") == "assistant"]
            for turn_index, (assistant_index, rubric) in enumerate(zip(assistant_indices, generated, strict=False)):
                samples.append(
                    ExportSample(
                        sample_id=f"{group_id}:rubric:{rubric_list_id}:{turn_index}",
                        group_id=group_id,
                        prompt=self.build_rubric_sft_prompt_messages(conversation, assistant_index),
                        turns=[conversation[assistant_index]],
                        metadata={
                            "round_index": int(round_payload["round_index"]),
                            "parent_id": round_payload.get("parent_id"),
                            "rubric_list_id": rubric_list_id,
                            "rubric_id": rubric.get("rubric_id"),
                            "rubric_title": rubric.get("title"),
                            "turn_index": turn_index,
                        },
                    )
                )
        return samples


class SFTDataExporter:
    def __init__(self, *, run_dir: Path) -> None:
        self.artifacts = _RunArtifacts(run_dir)

    def export_bundle(self) -> SFTExportBundle:
        accepted_group_ids: list[str] = []
        policy_samples: list[ExportSample] = []
        rubric_samples: list[ExportSample] = []

        for round_index in sorted(self.artifacts.round_payloads):
            round_payload = self.artifacts.round_payloads[round_index]
            group_id, accepted_policy_samples = self.artifacts.build_policy_sft_samples(round_payload)
            if not accepted_policy_samples:
                continue
            accepted_group_ids.append(group_id)
            policy_samples.extend(accepted_policy_samples)
            rubric_samples.extend(self.artifacts.build_rubric_sft_samples(round_payload, group_id))

        return SFTExportBundle(
            instance_id=self.artifacts.instance_id,
            run_dir=str(self.artifacts.run_dir),
            accepted_group_ids=accepted_group_ids,
            policy_samples=policy_samples,
            rubric_samples=rubric_samples,
            metadata={
                "current_round": self.artifacts.manifest.get("current_round", 0),
                "teacher_model_name": self.artifacts.manifest.get("policy_model_name", ""),
                "student_model_name": self.artifacts.manifest.get("student_policy_model_name", ""),
            },
        )


class GRPODataExporter:
    def __init__(self, *, run_dir: Path, rubric_parent_weight: float = 0.5) -> None:
        self.artifacts = _RunArtifacts(run_dir)
        self.rubric_parent_weight = float(rubric_parent_weight)

    def export_bundle(self) -> GRPOExportBundle:
        policy_groups: list[ExportGroup] = []
        rubric_groups: list[ExportGroup] = []

        for round_index in sorted(self.artifacts.round_payloads):
            round_payload = self.artifacts.round_payloads[round_index]
            parent_id = round_payload.get("parent_id")
            group_id = self.artifacts.build_group_id(round_index, parent_id)

            policy_samples: list[ExportSample] = []
            for item in round_payload.get("node_scores", []):
                node_id = str(item.get("node_id") or "")
                node = self.artifacts.nodes.get(node_id)
                judge = self.artifacts.node_judges.get(node_id)
                turns = self.artifacts.node_messages.get(node_id, [])
                if not node or not judge or not turns or judge.get("overall_reward") is None:
                    continue
                policy_samples.append(
                    ExportSample(
                        sample_id=f"{group_id}:policy:{node_id}",
                        group_id=group_id,
                        prompt=self.artifacts.build_prefix_messages(node.get("parent_id")),
                        turns=turns,
                        reward=float(judge["overall_reward"]),
                        metadata={
                            "group_id": group_id,
                            "node_id": node_id,
                            "round_index": round_index,
                            "parent_id": parent_id,
                            "ground_truth_reward": float(judge.get("ground_truth_reward") or 0.0),
                            "policy_source": str(node.get("policy_source") or ""),
                        },
                    )
                )

            if policy_samples:
                policy_groups.append(
                    ExportGroup(
                        group_id=group_id,
                        samples=policy_samples,
                        metadata={
                            "round_index": round_index,
                            "parent_id": parent_id,
                        },
                    )
                )

            rubric_samples: list[ExportSample] = []
            for sample_payload in round_payload.get("rubric_samples", []):
                rubric_list_id = str(sample_payload.get("rubric_list_id") or "")
                rubric_payload = self.artifacts.rubric_payloads.get(rubric_list_id)
                conversation = self.artifacts.rubric_messages.get(rubric_list_id, [])
                if not rubric_payload or not conversation:
                    continue
                generated = list(rubric_payload.get("generated", []))
                variance_by_rubric = rubric_payload.get("variance_by_rubric", {})
                redundency_by_rubric = rubric_payload.get("redundency_by_rubric", {})
                turn_rewards = [
                    float(variance_by_rubric.get(rubric["rubric_id"], 0.0))
                    + float(redundency_by_rubric.get(rubric["rubric_id"], 0.0))
                    for rubric in generated
                ]
                scalar_reward = (
                    (1.0 - self.rubric_parent_weight) * float(rubric_payload.get("gt_reward_siblings") or 0.0)
                    + self.rubric_parent_weight * float(rubric_payload.get("gt_reward_parent") or 0.0)
                )
                rubric_samples.append(
                    ExportSample(
                        sample_id=f"{group_id}:rubric:{rubric_list_id}",
                        group_id=group_id,
                        prompt=conversation[:1],
                        turns=conversation[1:],
                        reward=scalar_reward,
                        metadata={
                            "group_id": group_id,
                            "round_index": round_index,
                            "parent_id": parent_id,
                            "rubric_list_id": rubric_list_id,
                            "selected": bool(rubric_payload.get("selected", False)),
                            "turn_rewards": turn_rewards,
                            "variance_by_rubric": {
                                rubric["rubric_id"]: float(variance_by_rubric.get(rubric["rubric_id"], 0.0))
                                for rubric in generated
                            },
                            "redundency_by_rubric": {
                                rubric["rubric_id"]: float(redundency_by_rubric.get(rubric["rubric_id"], 0.0))
                                for rubric in generated
                            },
                            "gt_reward_siblings": float(rubric_payload.get("gt_reward_siblings") or 0.0),
                            "gt_reward_parent": float(rubric_payload.get("gt_reward_parent") or 0.0),
                        },
                    )
                )

            if rubric_samples:
                rubric_groups.append(
                    ExportGroup(
                        group_id=group_id,
                        samples=rubric_samples,
                        metadata={
                            "round_index": round_index,
                            "parent_id": parent_id,
                            "rubric_parent_weight": self.rubric_parent_weight,
                        },
                    )
                )

        return GRPOExportBundle(
            instance_id=self.artifacts.instance_id,
            run_dir=str(self.artifacts.run_dir),
            policy_groups=policy_groups,
            rubric_groups=rubric_groups,
            metadata={"current_round": self.artifacts.manifest.get("current_round", 0)},
        )
