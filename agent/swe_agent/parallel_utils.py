from __future__ import annotations

import json
import os
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import numpy as np
from slime.swe_agent.contracts import ExportGroup, ExportSample, GRPOExportBundle


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
    prompt_payload: dict[str, Any]
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


class GRPOCollector:
    def __init__(
        self,
        *,
        instance_id: str,
        run_dir: Path,
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> None:
        self.instance_id = instance_id
        self.run_dir = run_dir
        self.alpha = alpha
        self.beta = beta
        self.bundle = GRPOExportBundle(
            instance_id=instance_id,
            run_dir=str(run_dir),
            policy_groups=[],
            rubric_groups=[],
        )

    def collect(
        self,
        bundles: list[NodeArtifactBundle],
        rubric_bundles: list[RubricArtifactBundle],
        group_id: str,
    ) -> None:
        policy_samples = self._build_policy_samples(bundles, group_id)
        rubric_samples = self._build_rubric_samples(rubric_bundles, group_id)
        if policy_samples:
            self.bundle.policy_groups.append(
                ExportGroup(
                    group_id=group_id,
                    samples=policy_samples,
                )
            )
        if rubric_samples:
            self.bundle.rubric_groups.append(
                ExportGroup(
                    group_id=group_id,
                    samples=rubric_samples,
                )
            )

    @staticmethod
    def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": message["role"],
            "content": message.get("content", message.get("message")),
        }

    @classmethod
    def _normalize_messages(cls, payload: dict[str, Any]) -> list[dict[str, Any]]:
        messages = [cls._normalize_message(message) for message in payload.get("messages")]
        if len(messages) > 1 and messages[-1]["role"] == "user":
            return messages[:-1]
        return messages

    @classmethod
    def _normalize_prompt(cls, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [cls._normalize_message(message) for message in payload.get("messages")]

    def _build_policy_samples(
        self,
        bundles: list[NodeArtifactBundle],
        group_id: str,
    ) -> list[ExportSample]:
        samples: list[ExportSample] = []
        for bundle in bundles:
            turns = self._normalize_messages(bundle.messages_payload)
            reward = bundle.judge_payload.get("overall_reward") # NOTE: if we want to add gt reward: + bundle.judge_payload.get("ground_truth_reward")
            if not turns or reward is None:
                continue
            samples.append(
                ExportSample(
                    sample_id=bundle.node_id,
                    group_id=group_id,
                    prompt=self._normalize_prompt(bundle.prompt_payload),
                    turns=turns,
                    reward=float(reward),
                )
            )
        return samples

    def _build_rubric_samples(
        self,
        rubric_bundles: list[RubricArtifactBundle],
        group_id: str,
    ) -> list[ExportSample]:
        samples: list[ExportSample] = []
        for bundle in rubric_bundles:
            payload = bundle.rubric_payload
            rubric_list_id = str(payload.get("rubric_list_id") or bundle.rubric_dir.name)
            conversation = self._normalize_messages(bundle.messages_payload)
            if not rubric_list_id or not conversation:
                continue
            generated = list(payload.get("generated"))
            variance_by_rubric = payload.get("variance_by_rubric")
            redundency_by_rubric = payload.get("redundency_by_rubric")
            turn_rewards = [
                (1.0 - self.alpha) * float(variance_by_rubric.get(rubric["rubric_id"]))
                + self.alpha * float(redundency_by_rubric.get(rubric["rubric_id"]))
                for rubric in generated
            ]
            scalar_reward = (
                (1.0 - self.beta) * float(payload.get("gt_reward_siblings"))
                + self.beta * float(payload.get("gt_reward_parent"))
            )
            samples.append(
                ExportSample(
                    sample_id=rubric_list_id,
                    group_id=group_id,
                    prompt=conversation[:1],
                    turns=conversation[1:],
                    reward=scalar_reward,
                    metadata={
                        "turn_rewards": turn_rewards,
                    },
                )
            )
        return samples


def _write_base_artifacts(
    bundles: list[NodeArtifactBundle],
    rubric_bundles: list[RubricArtifactBundle] | None = None,
    extra_json_writes: list[tuple[Path, Any]] | None = None,
) -> None:
    for bundle in bundles:
        bundle.node_dir.mkdir(parents=True, exist_ok=True)
        if bundle.raw_traj_payload is not None:
            _atomic_write_json(bundle.node_dir / "raw_traj.json", bundle.raw_traj_payload)
        _atomic_write_json(bundle.node_dir / "messages.json", bundle.messages_payload)
        if bundle.snapshot_payload is not None:
            _atomic_write_json(bundle.node_dir / "snapshot.json", bundle.snapshot_payload)
        _atomic_write_json(bundle.node_dir / "node.json", bundle.node_payload)
        if bundle.terminal_raw_traj_payload is not None:
            _atomic_write_json(bundle.node_dir / "terminal_raw_traj.json", bundle.terminal_raw_traj_payload)
        if bundle.terminal_messages_payload is not None:
            _atomic_write_json(bundle.node_dir / "terminal_messages.json", bundle.terminal_messages_payload)
        if bundle.terminal_patch_payload is not None:
            _atomic_write_json(bundle.node_dir / "terminal_patch.json", bundle.terminal_patch_payload)
    for bundle in rubric_bundles or []:
        bundle.rubric_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(bundle.rubric_dir / "messages.json", bundle.messages_payload)
        if bundle.raw_traj_payload is not None:
            _atomic_write_json(bundle.rubric_dir / "raw_traj.json", bundle.raw_traj_payload)
    for path, payload in extra_json_writes or []:
        _atomic_write_json(path, payload)


def _write_gt_artifacts(
    bundles: list[NodeArtifactBundle],
    rubric_bundles: list[RubricArtifactBundle] | None = None,
    extra_json_writes: list[tuple[Path, Any]] | None = None,
) -> None:
    for bundle in bundles:
        bundle.node_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(bundle.node_dir / "judge.json", bundle.judge_payload)
    for bundle in rubric_bundles or []:
        bundle.rubric_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(bundle.rubric_dir / "rubric.json", bundle.rubric_payload)
    for path, payload in extra_json_writes or []:
        _atomic_write_json(path, payload)


def _pairwise_diff(x):
    x = np.asarray(x, dtype=float)
    i, j = np.triu_indices(len(x), 1)
    return x[i] - x[j]


def gap_corr(a, b):
    da, db = _pairwise_diff(a), _pairwise_diff(b)
    if len(da) == 0:
        return 0.0
    return 1.0 - np.mean(np.abs(da - db))


def gap_redundancy(a, b):
    da, db = _pairwise_diff(a), _pairwise_diff(b)
    if len(da) == 0:
        return 0.0
    return 1.0 - min(
        np.mean(np.abs(da - db)),
        np.mean(np.abs(da + db)),
    )


class ArtifactWriter:
    def __init__(self, max_workers: int = 1, write_artifacts: bool = True) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="search-artifacts")
        self._futures: list[Future] = []
        self.write_artifacts = write_artifacts

    def write_round(
        self,
        bundles: list[NodeArtifactBundle],
        rubric_bundles: list[RubricArtifactBundle] | None = None,
        extra_json_writes: list[tuple[Path, Any]] | None = None,
    ) -> None:
        if self.write_artifacts:
            _write_base_artifacts(bundles, rubric_bundles, extra_json_writes)
            _write_gt_artifacts(bundles, rubric_bundles, None)

    def submit_round(
        self,
        bundles: list[NodeArtifactBundle],
        rubric_bundles: list[RubricArtifactBundle] | None = None,
        extra_json_writes: list[tuple[Path, Any]] | None = None,
        *,
        write_gt_files: bool = True,
    ) -> Future:
        if write_gt_files:
            future = self._executor.submit(self.write_round, bundles, rubric_bundles, extra_json_writes)
        else:
            future = self._executor.submit(_write_base_artifacts, bundles, rubric_bundles, extra_json_writes)
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
        collector: GRPOCollector | None = None,
        write_artifacts: bool = True,
        max_workers: int = 1,
    ) -> None:
        self.instance = instance
        self.task_id = task_id
        self.model_name = model_name
        self.namespace = namespace
        self.work_dir = work_dir
        self.evaluate_patches_fn = evaluate_patches_fn
        self.collector = collector
        self.write_artifacts = write_artifacts
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="search-gt-eval")
        self._futures: list[Future] = []

    def _evaluate_round(
        self,
        bundles: list[NodeArtifactBundle],
        rubric_bundles: list[RubricArtifactBundle] | None,
        extra_json_writes: list[tuple[Path, Any]] | None,
        previous_future: Future | None,
    ) -> dict[str, float]:
        if previous_future is not None:
            previous_future.result()

        patches_by_node_id: dict[str, str] = {}
        for bundle in bundles:
            if bundle.terminal_patch_payload is None:
                continue
            patch = bundle.terminal_patch_payload.get(self.task_id, {}).get("model_patch") or ""
            patches_by_node_id[bundle.node_id] = patch

        rewards: dict[str, float] = {}
        if patches_by_node_id:
            rewards = self.evaluate_patches_fn(
                instance=self.instance,
                patches_by_key=patches_by_node_id,
                model_name=self.model_name,
                max_workers=1,
                namespace=self.namespace,
                work_dir=self.work_dir,
            )
        for bundle in bundles:
            if bundle.node_id in rewards:
                bundle.judge_payload["ground_truth_reward"] = rewards[bundle.node_id]

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
            gt_reward_siblings = gap_corr(sibling_scores, sibling_gt)
            parent_reward = payload.get("parent_reward")
            gt_reward_parent = float(np.mean([
                1.0 - abs(float(payload["child_rewards"][node_id]) - float(parent_reward) - float(gt_by_node_id[node_id]) + float(parent_gt))
                for node_id in ordered_node_ids
            ]))
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

        if self.collector is not None:
            parent_id = bundles[0].node_payload.get("parent_id")
            self.collector.collect(bundles, rubric_bundles or [], parent_id)
        if self.write_artifacts:
            _write_gt_artifacts(bundles, rubric_bundles, extra_json_writes)
        return rewards

    def submit_round(
        self,
        bundles: list[NodeArtifactBundle],
        rubric_bundles: list[RubricArtifactBundle] | Future | None = None,
        extra_json_writes: list[tuple[Path, Any]] | None = None,
    ) -> Future:
        if isinstance(rubric_bundles, Future):
            previous_future = rubric_bundles
            rubric_bundles = None
        else:
            previous_future = self._futures[-1] if self._futures else None
        future = self._executor.submit(self._evaluate_round, bundles, rubric_bundles, extra_json_writes, previous_future)
        self._futures.append(future)
        return future

    def wait(self) -> None:
        while self._futures:
            self._futures.pop(0).result()

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
