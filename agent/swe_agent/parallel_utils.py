from __future__ import annotations

import json
import os
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
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
    raw_traj_payload: dict[str, Any]
    messages_payload: dict[str, Any]
    judge_payload: dict[str, Any]
    snapshot_payload: dict[str, Any] | None = None
    terminal_raw_traj_payload: dict[str, Any] | None = None
    terminal_messages_payload: dict[str, Any] | None = None
    terminal_patch_payload: dict[str, Any] | None = None


class ArtifactWriter:
    def __init__(self, max_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="search-artifacts")
        self._futures: list[Future] = []

    def write_round(self, bundles: list[NodeArtifactBundle]) -> None:
        for bundle in bundles:
            bundle.node_dir.mkdir(parents=True, exist_ok=True)
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

    def submit_round(self, bundles: list[NodeArtifactBundle]) -> Future:
        future = self._executor.submit(self.write_round, bundles)
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

    def _evaluate_round(self, bundles: list[NodeArtifactBundle], write_future: Future | None) -> dict[str, float]:
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
        return rewards

    def submit_round(self, bundles: list[NodeArtifactBundle], write_future: Future | None) -> Future:
        future = self._executor.submit(self._evaluate_round, bundles, write_future)
        self._futures.append(future)
        return future

    def wait(self) -> None:
        while self._futures:
            self._futures.pop(0).result()

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
