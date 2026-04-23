from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_path: str
    default_split: str
    purpose: str


DATASET_SPECS: dict[str, DatasetSpec] = {
    "verified": DatasetSpec(
        name="verified",
        hf_path="princeton-nlp/SWE-Bench_Verified",
        default_split="test",
        purpose="benchmark",
    ),
    "rebench_v2": DatasetSpec(
        name="rebench_v2",
        hf_path="nebius/SWE-rebench-V2",
        default_split="train",
        purpose="training",
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    normalized = name.replace("-", "_")
    if normalized not in DATASET_SPECS:
        raise KeyError(f"Unknown dataset spec: {name}")
    return DATASET_SPECS[normalized]


def resolve_dataset_path(name: str) -> str:
    try:
        return get_dataset_spec(name).hf_path
    except KeyError:
        return name
