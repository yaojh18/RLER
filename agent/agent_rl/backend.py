from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from .protocol import RolloutResult, RolloutSessionSpec, RolloutSnapshot


@runtime_checkable
class RolloutSession(Protocol):
    spec: RolloutSessionSpec

    def is_finished(self) -> bool:
        ...

    def step(self) -> None:
        ...

    def run_until_pause(self, max_steps: Optional[int] = None) -> RolloutResult:
        ...

    def snapshot(self) -> RolloutSnapshot:
        ...

    def export_result(self) -> RolloutResult:
        ...


@runtime_checkable
class RolloutBackend(Protocol):
    def create_session(self, spec: RolloutSessionSpec) -> RolloutSession:
        ...

    def resume_session(self, snapshot: RolloutSnapshot) -> RolloutSession:
        ...
