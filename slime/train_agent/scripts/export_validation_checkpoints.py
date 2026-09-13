"""Export validation-marked Megatron checkpoints for direct SGLang loading.

This process is intentionally separate from the training Ray actors.  It
watches the shared checkpoint directory and converts each checkpoint carrying
an RLER validation marker into a HuggingFace/safetensors directory while the
next optimizer updates continue.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from slime.rollout.data_source import VALIDATION_CHECKPOINT_MARKER_PREFIX

MARKER_GLOB = f"{VALIDATION_CHECKPOINT_MARKER_PREFIX}*"
def _complete_export(path: Path) -> bool:
    return (path / "config.json").is_file() and (
        (path / "model.safetensors").is_file()
        or (path / "model.safetensors.index.json").is_file()
    )


def _export_one(args: argparse.Namespace, checkpoint: Path) -> None:
    destination = args.output_root / checkpoint.name
    if _complete_export(destination):
        return
    if destination.exists():
        raise RuntimeError(f"incomplete published HF checkpoint: {destination}")
    if not (checkpoint / "common.pt").is_file() or not (checkpoint / ".metadata").is_file():
        raise RuntimeError(f"Megatron checkpoint is not complete yet: {checkpoint}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{checkpoint.name}.export-",
            dir=args.output_root,
        )
    )
    command = [
        args.python,
        str(args.converter),
        "--input-dir",
        str(checkpoint),
        "--output-dir",
        str(staging),
        "--origin-hf-dir",
        str(args.origin_hf_dir),
        "--force",
        "--add-missing-from-origin-hf",
    ]
    try:
        subprocess.run(command, check=True)
        if not _complete_export(staging):
            raise RuntimeError(f"converted HF checkpoint is incomplete: {staging}")
        os.replace(staging, destination)
        print(f"exported {checkpoint} -> {destination}", flush=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _release_source_markers(checkpoint: Path) -> None:
    """Make a successfully exported source eligible for normal retention."""
    for marker in checkpoint.glob(MARKER_GLOB):
        if marker.is_file() and not marker.is_symlink():
            marker.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--origin-hf-dir", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--release-source-marker",
        action="store_true",
        help=(
            "release validation markers after a complete HF export so an "
            "older raw training checkpoint can be pruned normally"
        ),
    )
    args = parser.parse_args()
    failures: dict[str, str] = {}
    while True:
        pending = []
        marked_checkpoints = sorted(
            (
                path
                for path in args.checkpoint_root.glob(
                    "iter_[0-9][0-9][0-9][0-9][0-9][0-9][0-9]"
                )
                if path.is_dir() and any(path.glob(MARKER_GLOB))
            ),
            key=lambda path: path.name,
        )
        for checkpoint in marked_checkpoints:
            destination = args.output_root / checkpoint.name
            if _complete_export(destination):
                if args.release_source_marker:
                    _release_source_markers(checkpoint)
                continue
            try:
                _export_one(args, checkpoint)
                failures.pop(checkpoint.name, None)
                if args.release_source_marker:
                    _release_source_markers(checkpoint)
            except Exception as exc:
                pending.append(checkpoint)
                failures[checkpoint.name] = f"{type(exc).__name__}: {exc}"
                print(
                    f"validation HF export deferred for {checkpoint.name}: {failures[checkpoint.name]}",
                    flush=True,
                )

        if args.stop_file.exists() and not pending:
            break
        if args.stop_file.exists() and pending:
            # The training process has stopped, so one final failed pass is
            # enough.  Preserve training success and report the export error.
            break
        time.sleep(max(0.1, args.poll_seconds))

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
