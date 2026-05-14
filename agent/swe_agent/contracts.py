from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExportSample:
    sample_id: str
    group_id: str
    prompt: list[dict[str, Any]]
    turns: list[dict[str, Any]]
    reward: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportGroup:
    group_id: str
    samples: list[ExportSample]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SFTExportBundle:
    instance_id: str
    run_dir: str
    accepted_group_ids: list[str]
    policy_samples: list[ExportSample]
    rubric_samples: list[ExportSample]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            "SFTExportBundle("
            f"instance_id={self.instance_id!r}, "
            f"accepted_groups={len(self.accepted_group_ids)}, "
            f"policy_samples={len(self.policy_samples)}, "
            f"rubric_samples={len(self.rubric_samples)}, "
            f"run_dir={self.run_dir!r})"
        )


@dataclass
class GRPOExportBundle:
    instance_id: str
    run_dir: str
    policy_groups: list[ExportGroup]
    rubric_groups: list[ExportGroup]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            "GRPOExportBundle("
            f"instance_id={self.instance_id!r}, "
            f"policy_groups={len(self.policy_groups)}, "
            f"rubric_groups={len(self.rubric_groups)}, "
            f"run_dir={self.run_dir!r})"
        )
