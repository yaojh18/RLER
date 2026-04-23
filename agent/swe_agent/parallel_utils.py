from __future__ import annotations

import json
import os
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(suffix=".tmp", prefix=f"{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


@dataclass
class NodeArtifactBundle:
    node_id: str
    node_dir: Path
    node_payload: dict[str, Any]
    raw_traj_payload: dict[str, Any] | None
    messages_payload: dict[str, Any]
    judge_payload: dict[str, Any]
    snapshot_payload: dict[str, Any] | None = None
    terminal_raw_traj_payload: dict[str, Any] | None = None
    terminal_messages_payload: dict[str, Any] | None = None
    terminal_patch_payload: dict[str, Any] | None = None


@dataclass
class RubricArtifactBundle:
    rubric_dir: Path
    rubric_payload: dict[str, Any]
    messages_payload: dict[str, Any]
    raw_traj_payload: dict[str, Any] | None = None
    round_summary_path: Path | None = None
    selected_for_round_summary: bool = False


class ArtifactWriter:
    def __init__(self, max_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="search-artifacts")
        self._futures: list[Future] = []

    def write_round(self, bundles: list[NodeArtifactBundle], rubric_bundles: list[RubricArtifactBundle] | None = None) -> None:
        for bundle in bundles:
            bundle.node_dir.mkdir(parents=True, exist_ok=True)
            if bundle.raw_traj_payload is not None:
                _atomic_write_json(bundle.node_dir / "raw_traj.json", bundle.raw_traj_payload)
            _atomic_write_json(bundle.node_dir / "messages.json", bundle.messages_payload)
            _atomic_write_json(bundle.node_dir / "judge.json", bundle.judge_payload)
            if bundle.snapshot_payload is not None:
                _atomic_write_json(bundle.node_dir / "snapshot.json", bundle.snapshot_payload)
            if bundle.terminal_raw_traj_payload is not None:
                _atomic_write_json(bundle.node_dir / "terminal_raw_traj.json", bundle.terminal_raw_traj_payload)
            if bundle.terminal_messages_payload is not None:
                _atomic_write_json(bundle.node_dir / "terminal_messages.json", bundle.terminal_messages_payload)
            if bundle.terminal_patch_payload is not None:
                _atomic_write_json(bundle.node_dir / "terminal_patch.json", bundle.terminal_patch_payload)
            _atomic_write_json(bundle.node_dir / "node.json", bundle.node_payload)
        for bundle in rubric_bundles or []:
            bundle.rubric_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(bundle.rubric_dir / "rubric.json", bundle.rubric_payload)
            _atomic_write_json(bundle.rubric_dir / "messages.json", bundle.messages_payload)
            if bundle.raw_traj_payload is not None:
                _atomic_write_json(bundle.rubric_dir / "raw_traj.json", bundle.raw_traj_payload)

    def submit_round(self, bundles: list[NodeArtifactBundle], rubric_bundles: list[RubricArtifactBundle] | None = None) -> Future:
        future = self._executor.submit(self.write_round, bundles, rubric_bundles)
        self._futures.append(future)
        return future

    def wait(self) -> None:
        while self._futures:
            self._futures.pop(0).result()

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)


class PatchEvalManager:
    def __init__(
        self,
        *,
        instance: dict[str, Any],
        task_id: str,
        model_name: str,
        namespace: str | None,
        work_dir: Path,
        evaluate_patches_fn: Callable[..., dict[str, float]],
        max_workers: int = 1,
    ) -> None:
        self.instance = instance
        self.task_id = task_id
        self.model_name = model_name
        self.namespace = namespace
        self.work_dir = work_dir
        self.evaluate_patches_fn = evaluate_patches_fn
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="search-gt-eval")
        self._futures: list[Future] = []

    def _evaluate_round(
        self,
        bundles: list[NodeArtifactBundle],
        rubric_bundles: list[RubricArtifactBundle] | None,
        write_future: Future | None,
        previous_future: Future | None,
    ) -> dict[str, float]:
        if previous_future is not None:
            previous_future.result()
        if write_future is not None:
            write_future.result()

        patches_by_node_id: dict[str, str] = {}
        for bundle in bundles:
            if bundle.terminal_patch_payload is None:
                continue
            patch = bundle.terminal_patch_payload.get(self.task_id, {}).get("model_patch") or ""
            patches_by_node_id[bundle.node_id] = patch

        if not patches_by_node_id:
            return {}

        rewards = self.evaluate_patches_fn(
            instance=self.instance,
            patches_by_key=patches_by_node_id,
            model_name=self.model_name,
            max_workers=1,
            namespace=self.namespace,
            work_dir=self.work_dir,
        )
        for bundle in bundles:
            if bundle.node_id not in rewards:
                continue
            bundle.judge_payload["ground_truth_reward"] = rewards[bundle.node_id]
            _atomic_write_json(bundle.node_dir / "judge.json", bundle.judge_payload)
        gt_by_node_id = dict(rewards)
        for bundle in rubric_bundles or []:
            payload = bundle.rubric_payload
            parent_node_id = payload.get("parent_node_id")
            parent_gt = None
            if parent_node_id:
                parent_judge_path = self.work_dir / "nodes" / str(parent_node_id) / "judge.json"
                if parent_judge_path.exists():
                    parent_gt = json.loads(parent_judge_path.read_text(encoding="utf-8")).get("ground_truth_reward")
            ordered_node_ids = sorted(str(node_id) for node_id in payload.get("child_rewards", {}))
            sibling_scores = [float(payload["child_rewards"][node_id]) for node_id in ordered_node_ids if node_id in gt_by_node_id]
            sibling_gt = [float(gt_by_node_id[node_id]) for node_id in ordered_node_ids if node_id in gt_by_node_id]
            gt_reward_siblings = 0.0
            if len(sibling_scores) >= 2:
                mean_score = sum(sibling_scores) / len(sibling_scores)
                mean_gt = sum(sibling_gt) / len(sibling_gt)
                cov = sum((score - mean_score) * (gt - mean_gt) for score, gt in zip(sibling_scores, sibling_gt))
                var_score = sum((score - mean_score) ** 2 for score in sibling_scores)
                var_gt = sum((gt - mean_gt) ** 2 for gt in sibling_gt)
                if var_score > 0 and var_gt > 0:
                    gt_reward_siblings = cov / (var_score * var_gt) ** 0.5
            gt_reward_parent = 0.0
            parent_reward = payload.get("parent_reward")
            if parent_gt is not None and parent_reward is not None:
                agreements = [
                    float((float(payload["child_rewards"][node_id]) >= float(parent_reward)) == (float(gt_by_node_id[node_id]) >= float(parent_gt)))
                    for node_id in ordered_node_ids
                    if node_id in gt_by_node_id
                ]
                if agreements:
                    gt_reward_parent = sum(agreements) / len(agreements)
            payload["gt_reward_siblings"] = float(gt_reward_siblings)
            payload["gt_reward_parent"] = float(gt_reward_parent)
            payload["gt_by_rubric"] = {
                rubric_id: {
                    "parent_score": payload.get("parent_score_by_rubric", {}).get(rubric_id),
                    "child_scores": payload.get("child_score_by_rubric", {}).get(rubric_id, {}),
                    "ground_truth_by_node": (
                        ({str(parent_node_id): parent_gt} if parent_node_id is not None else {})
                        | {node_id: gt_by_node_id[node_id] for node_id in ordered_node_ids if node_id in gt_by_node_id}
                    ),
                }
                for rubric_id in sorted(
                    set(payload.get("parent_score_by_rubric", {}))
                    | set(payload.get("child_score_by_rubric", {}))
                )
            }
            _atomic_write_json(bundle.rubric_dir / "rubric.json", payload)
            if bundle.selected_for_round_summary and bundle.round_summary_path is not None and bundle.round_summary_path.exists():
                round_payload = json.loads(bundle.round_summary_path.read_text(encoding="utf-8"))
                round_payload["gt_by_rubric"] = payload["gt_by_rubric"]
                round_payload["gt_reward_siblings"] = payload["gt_reward_siblings"]
                round_payload["gt_reward_parent"] = payload["gt_reward_parent"]
                for sample_payload in round_payload.get("rubric_samples", []):
                    if sample_payload.get("sample_index") == payload.get("sample_index"):
                        sample_payload["gt_by_rubric"] = payload["gt_by_rubric"]
                        sample_payload["gt_reward_siblings"] = payload["gt_reward_siblings"]
                        sample_payload["gt_reward_parent"] = payload["gt_reward_parent"]
                        break
                _atomic_write_json(bundle.round_summary_path, round_payload)
        return rewards

    def submit_round(
        self,
        bundles: list[NodeArtifactBundle],
        write_future: Future | None,
        rubric_bundles: list[RubricArtifactBundle] | None = None,
    ) -> Future:
        previous_future = self._futures[-1] if self._futures else None
        future = self._executor.submit(self._evaluate_round, bundles, rubric_bundles, write_future, previous_future)
        self._futures.append(future)
        return future

    def wait(self) -> None:
        while self._futures:
            self._futures.pop(0).result()

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
