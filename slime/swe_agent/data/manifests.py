from __future__ import annotations

from datasets import load_dataset

from .datasets import resolve_dataset_path


def load_manifest_records(dataset_name: str, split: str) -> list[dict]:
    return list(load_dataset(resolve_dataset_path(dataset_name), split=split))


def build_instance_manifest(
    dataset_name: str,
    split: str,
    *,
    instance_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[dict]:
    records = load_manifest_records(dataset_name, split)
    if instance_ids is not None:
        allowed = set(instance_ids)
        records = [record for record in records if record.get("instance_id") in allowed]
    if limit is not None:
        records = records[:limit]
    return [
        {
            "dataset_name": dataset_name,
            "split": split,
            "instance_id": record.get("instance_id"),
            "repo": record.get("repo"),
            "base_commit": record.get("base_commit"),
            "problem_statement": record.get("problem_statement"),
            "image_name": record.get("image_name"),
            "install_config": record.get("install_config"),
        }
        for record in records
    ]
