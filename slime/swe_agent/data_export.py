from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import ExportGroup, ExportSample, GRPOExportBundle, SFTExportBundle


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class _RunArtifacts:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.manifest = _load_json(self.run_dir / "run_manifest.json")
        self.instance_id = str(self.manifest["instance_id"])
        self.system_prompt = self.manifest.get("system_prompt", "")
        self.user_prompt = self.manifest.get("user_prompt", "")
        self.nodes: dict[str, dict[str, Any]] = {}
        self.node_messages: dict[str, dict[str, Any]] = {}
        self.node_judges: dict[str, dict[str, Any]] = {}
        self.round_payloads: dict[int, dict[str, Any]] = {}
        for node_path in sorted((self.run_dir / "nodes").glob("*/node.json")):
            node_payload = _load_json(node_path)
            node_id = str(node_payload["node_id"])
            self.nodes[node_id] = node_payload
            messages_path = node_path.parent / "messages.json"
            judge_path = node_path.parent / "judge.json"
            if messages_path.exists():
                self.node_messages[node_id] = _load_json(messages_path)
            if judge_path.exists():
                self.node_judges[node_id] = _load_json(judge_path)
        for round_path in sorted((self.run_dir / "rubrics").glob("round_*.json")):
            payload = _load_json(round_path)
            self.round_payloads[int(payload["round_index"])] = payload

    def build_prefix_messages(self, parent_node_id: str | None) -> list[dict[str, Any]]:
        prompt: list[dict[str, Any]] = []
        if self.system_prompt:
            prompt.append({"role": "system", "message": self.system_prompt, "tool_calls": [], "parser_model": ""})
        if self.user_prompt:
            prompt.append({"role": "user", "message": self.user_prompt, "tool_calls": [], "parser_model": ""})
        lineage: list[str] = []
        current = parent_node_id
        while current and current != "root":
            lineage.append(current)
            current = self.nodes[current].get("parent_id")
        for node_id in reversed(lineage):
            prompt.extend(self.node_messages.get(node_id, {}).get("messages", []))
        return prompt

    def build_group_id(self, *, round_index: int, parent_node_id: str | None) -> str:
        return f"{self.instance_id}:round:{round_index}:parent:{parent_node_id or 'root'}"

    def compute_list_reward(self, child_rewards: dict[str, float]) -> float:
        node_ids = [node_id for node_id in child_rewards if node_id in self.node_judges]
        if len(node_ids) < 2:
            return 0.0
        rewards = [float(child_rewards[node_id]) for node_id in node_ids]
        gt_values = [float(self.node_judges[node_id].get("ground_truth_reward") or 0.0) for node_id in node_ids]
        mean_reward = sum(rewards) / len(rewards)
        mean_gt = sum(gt_values) / len(gt_values)
        cov = sum((reward - mean_reward) * (gt - mean_gt) for reward, gt in zip(rewards, gt_values))
        var_reward = sum((reward - mean_reward) ** 2 for reward in rewards)
        var_gt = sum((gt - mean_gt) ** 2 for gt in gt_values)
        if var_reward <= 0 or var_gt <= 0:
            return 0.0
        return cov / (var_reward * var_gt) ** 0.5


class SFTDataExporter:
    def __init__(self, *, run_dir: Path) -> None:
        self.artifacts = _RunArtifacts(run_dir)

    def export_bundle(self) -> SFTExportBundle:
        accepted_group_ids: list[str] = []
        policy_samples: list[ExportSample] = []
        rubric_samples: list[ExportSample] = []
        for round_index in sorted(self.artifacts.round_payloads):
            round_payload = self.artifacts.round_payloads[round_index]
            parent_node_id = round_payload.get("parent_id")
            group_id = self.artifacts.build_group_id(round_index=round_index, parent_node_id=parent_node_id)
            child_ids = [str(item["node_id"]) for item in round_payload.get("node_scores", [])]
            teacher_nodes = [self.artifacts.nodes[node_id] for node_id in child_ids if self.artifacts.nodes.get(node_id, {}).get("policy_source") == "teacher"]
            student_nodes = [self.artifacts.nodes[node_id] for node_id in child_ids if self.artifacts.nodes.get(node_id, {}).get("policy_source") == "student"]
            teacher_candidates = []
            for node in teacher_nodes:
                judge = self.artifacts.node_judges.get(str(node["node_id"]), {})
                if float(judge.get("ground_truth_reward") or 0.0) >= 1.0:
                    teacher_candidates.append((float(judge.get("overall_reward") or float("-inf")), node, judge))
            best_student_score = max(
                (float(self.artifacts.node_judges.get(str(node["node_id"]), {}).get("overall_reward") or float("-inf")) for node in student_nodes),
                default=float("-inf"),
            )
            if not teacher_candidates:
                continue
            teacher_candidates.sort(key=lambda item: item[0], reverse=True)
            best_teacher_score, best_teacher_node, _ = teacher_candidates[0]
            if best_teacher_score <= best_student_score:
                continue
            accepted_group_ids.append(group_id)
            policy_messages = self.artifacts.node_messages.get(str(best_teacher_node["node_id"]), {}).get("messages", [])
            if policy_messages:
                policy_samples.append(
                    ExportSample(
                        sample_id=f"{group_id}:policy",
                        group_id=group_id,
                        prompt=self.artifacts.build_prefix_messages(best_teacher_node.get("parent_id")),
                        turns=policy_messages,
                        metadata={
                            "node_id": best_teacher_node["node_id"],
                            "overall_reward": best_teacher_score,
                            "ground_truth_reward": 1.0,
                        },
                    )
                )
            rubric_samples_payload = list(round_payload.get("rubric_samples", []))
            if not rubric_samples_payload:
                continue
            rubric_samples_payload.sort(
                key=lambda payload: float(payload.get("teacher_student_gap") or float("-inf")),
                reverse=True,
            )
            rubric_sample = rubric_samples_payload[0]
            rubric_dir = Path(rubric_sample["artifact_dir"])
            rubric_messages_path = rubric_dir / "messages.json"
            if not rubric_messages_path.exists():
                continue
            rubric_messages_payload = _load_json(rubric_messages_path)
            conversation = list(rubric_messages_payload.get("messages", []))
            prompt_messages = []
            response_turns = []
            for message in conversation:
                if message.get("role") == "assistant":
                    response_turns.append(message)
                else:
                    prompt_messages.append(message)
            if response_turns:
                rubric_samples.append(
                    ExportSample(
                        sample_id=f"{group_id}:rubric:{rubric_sample['sample_index']}",
                        group_id=group_id,
                        prompt=prompt_messages,
                        turns=response_turns,
                        metadata={
                            "sample_index": rubric_sample["sample_index"],
                            "teacher_student_gap": rubric_sample.get("teacher_student_gap"),
                            "list_reward": self.artifacts.compute_list_reward(rubric_sample.get("child_rewards", {})),
                        },
                    )
                )
        return SFTExportBundle(
            instance_id=self.artifacts.instance_id,
            run_dir=str(self.artifacts.run_dir),
            accepted_group_ids=accepted_group_ids,
            policy_samples=policy_samples,
            rubric_samples=rubric_samples,
            metadata={"current_round": self.artifacts.manifest.get("current_round", 0)},
        )


class GRPODataExporter:
    def __init__(self, *, run_dir: Path) -> None:
        self.artifacts = _RunArtifacts(run_dir)

    def export_bundle(self) -> GRPOExportBundle:
        policy_groups: list[ExportGroup] = []
        rubric_groups: list[ExportGroup] = []
        for round_index in sorted(self.artifacts.round_payloads):
            round_payload = self.artifacts.round_payloads[round_index]
            parent_node_id = round_payload.get("parent_id")
            group_id = self.artifacts.build_group_id(round_index=round_index, parent_node_id=parent_node_id)
            policy_samples: list[ExportSample] = []
            for item in round_payload.get("node_scores", []):
                node_id = str(item["node_id"])
                node = self.artifacts.nodes.get(node_id)
                judge = self.artifacts.node_judges.get(node_id, {})
                messages = self.artifacts.node_messages.get(node_id, {}).get("messages", [])
                if not node or not messages or judge.get("overall_reward") is None:
                    continue
                policy_samples.append(
                    ExportSample(
                        sample_id=f"{group_id}:policy:{node_id}",
                        group_id=group_id,
                        prompt=self.artifacts.build_prefix_messages(node.get("parent_id")),
                        turns=messages,
                        reward=float(judge.get("overall_reward")),
                        metadata={
                            "node_id": node_id,
                            "ground_truth_reward": judge.get("ground_truth_reward"),
                            "exit_status": node.get("exit_status", ""),
                        },
                    )
                )
            if policy_samples:
                policy_groups.append(
                    ExportGroup(
                        group_id=group_id,
                        samples=policy_samples,
                        metadata={"round_index": round_index, "parent_node_id": parent_node_id},
                    )
                )

            rubric_samples: list[ExportSample] = []
            for rubric_sample in round_payload.get("rubric_samples", []):
                rubric_dir = Path(rubric_sample["artifact_dir"])
                rubric_messages_path = rubric_dir / "messages.json"
                if not rubric_messages_path.exists():
                    continue
                conversation = list(_load_json(rubric_messages_path).get("messages", []))
                prompt_messages = []
                response_turns = []
                for turn_index, message in enumerate(conversation):
                    if message.get("role") == "assistant":
                        turn = dict(message)
                        turn_reward = rubric_sample.get("turn_rewards", [])
                        if turn_index - len(prompt_messages) < len(turn_reward):
                            turn["reward"] = turn_reward[turn_index - len(prompt_messages)]
                        response_turns.append(turn)
                    else:
                        prompt_messages.append(message)
                if not response_turns:
                    continue
                rubric_samples.append(
                    ExportSample(
                        sample_id=f"{group_id}:rubric:{rubric_sample['sample_index']}",
                        group_id=group_id,
                        prompt=prompt_messages,
                        turns=response_turns,
                        reward=self.artifacts.compute_list_reward(rubric_sample.get("child_rewards", {})),
                        metadata={
                            "sample_index": rubric_sample["sample_index"],
                            "teacher_student_gap": rubric_sample.get("teacher_student_gap"),
                            "selected": rubric_sample.get("selected", False),
                        },
                    )
                )
            if rubric_samples:
                rubric_groups.append(
                    ExportGroup(
                        group_id=group_id,
                        samples=rubric_samples,
                        metadata={"round_index": round_index, "parent_node_id": parent_node_id},
                    )
                )
        return GRPOExportBundle(
            instance_id=self.artifacts.instance_id,
            run_dir=str(self.artifacts.run_dir),
            policy_groups=policy_groups,
            rubric_groups=rubric_groups,
            metadata={"current_round": self.artifacts.manifest.get("current_round", 0)},
        )
