from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from swe_agent.parallel_utils import _normalize_messages, gap_corr

from .contracts import ExportSample, SFTExportBundle


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class _RunArtifacts:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.manifest = _load_json(run_dir / "run_manifest.json")
        self.instance_id = str(self.manifest["instance_id"])
        self.system_prompt = str(self.manifest.get("system_prompt"))
        self.user_prompt = str(self.manifest.get("user_prompt"))
        self.nodes: dict[str, dict[str, Any]] = {}
        self.node_messages: dict[str, list[dict[str, str]]] = {}
        self.rounds: dict[int, dict[str, Any]] = {}
        self.rubric_messages: dict[str, list[dict[str, str]]] = {}

        node_index_path = run_dir / "node_index.jsonl"
        for raw_line in node_index_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            payload = json.loads(line)
            self.nodes[str(payload["node_id"])] = payload

        for node_id in sorted(self.nodes):
            node_dir = run_dir / "nodes" / node_id
            messages_path = node_dir / "messages.json"
            payload = _load_json(messages_path)
            self.node_messages[node_id] = _normalize_messages(payload)
            judge_path = node_dir / "judge.json"
            payload = _load_json(judge_path)
            self.nodes[node_id]["ground_truth_reward"] = payload.get("ground_truth_reward", 0.0)

        rubrics_dir = run_dir / "rubrics"
        if rubrics_dir.exists():
            for round_path in sorted(rubrics_dir.glob("round_*.json")):
                payload = _load_json(round_path)
                self.rounds[int(payload["round_index"])] = {rubric["rubric_list_id"]: rubric for rubric in payload["rubric_samples"]}

            for rubric_dir in sorted(path for path in rubrics_dir.iterdir() if path.is_dir()):
                rubric_id = rubric_dir.name
                messages_path = rubric_dir / "messages.json"
                payload = _load_json(messages_path)
                self.rubric_messages[rubric_id] = _normalize_messages(payload)

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
            current = self.nodes[current].get("parent_id")
        for node_id in reversed(lineage):
            messages.extend(self.node_messages.get(node_id))
        return messages

    def _is_teacher_gt_node(self, node_id: str) -> bool:
        node = self.nodes.get(node_id)
        if not node or str(node.get("policy_source")) != "teacher":
            return False
        return float(node.get("ground_truth_reward")) >= 1.0

    def build_policy_sft_sample(self, round_index: int) -> tuple[str, list[str], ExportSample | None]:
        valid_rubrics = [rubric for rubric in self.rounds[round_index].values() if rubric.get("is_valid") is not False]
        if not valid_rubrics:
            return f"round_{round_index}", [], None
        first_rubric = valid_rubrics[0]
        group_id = first_rubric["parent_node_id"] if round_index in self.rounds else f"round_{round_index}"
        node_ids = list(first_rubric["child_rewards"].keys())
        teacher_nodes = [node_id for node_id in node_ids if self._is_teacher_gt_node(node_id)]
        if not teacher_nodes:
            return group_id, node_ids, None
        node_id = teacher_nodes[0]
        node = self.nodes[node_id]
        return group_id, node_ids, ExportSample(
            sample_id=node_id,
            group_id=group_id,
            prompt=self.build_prefix_messages(node.get("parent_id")),
            turns=self.node_messages[node_id],
        )

    def _rubic_judge_corr_gt(self, node_ids: list[str], rubric: dict[str, Any]) -> bool:
        rewards = [rubric.get("child_rewards").get(node_id) for node_id in node_ids]
        gts = [self.nodes[node_id].get("ground_truth_reward") for node_id in node_ids]
        return gap_corr(rewards, gts) > 0.8

    def build_rubric_sft_sample(self, round_index: int, group_id: str, node_ids: list[str]) -> ExportSample | None:
        teacher_nodes = [node_id for node_id in node_ids if self._is_teacher_gt_node(node_id)]
        if not teacher_nodes:
            return None

        for rubric in self.rounds[round_index].values():
            if rubric.get("is_valid") is False:
                continue
            if not self._rubic_judge_corr_gt(node_ids, rubric):
                continue
            rubric_list_id = str(rubric.get("rubric_list_id") or "")
            return ExportSample(
                sample_id=rubric_list_id,
                group_id=group_id,
                prompt=self.rubric_messages[rubric_list_id][:1],
                turns=self.rubric_messages[rubric_list_id][1:],
            )
        return None


class SFTDataExporter:
    def __init__(self, *, run_dir: Path) -> None:
        self.artifacts = _RunArtifacts(run_dir)

    def export_bundle(self) -> SFTExportBundle:
        accepted_group_ids: list[str] = []
        policy_samples: list[ExportSample] = []
        rubric_samples: list[ExportSample] = []

        for round_index in range(1, len(self.artifacts.rounds) + 1):
            group_id, node_ids, policy_sample = self.artifacts.build_policy_sft_sample(round_index)
            if policy_sample is None:
                continue
            rubric_sample = self.artifacts.build_rubric_sft_sample(round_index, group_id, node_ids)
            if rubric_sample is None:
                continue
            accepted_group_ids.append(group_id)
            rubric_samples.append(rubric_sample)   
            policy_samples.append(policy_sample)

        return SFTExportBundle(
            instance_id=self.artifacts.instance_id,
            run_dir=str(self.artifacts.run_dir),
            accepted_group_ids=accepted_group_ids,
            policy_samples=policy_samples,
            rubric_samples=rubric_samples,
        )
