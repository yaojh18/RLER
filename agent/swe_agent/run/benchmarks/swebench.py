#!/usr/bin/env python3

"""Run mini-SWE-agent on SWE-bench instances in batch mode."""
# Read this first: https://mini-swe-agent.com/latest/usage/swebench/  (usage docs)

import concurrent.futures
import contextlib
import json
import os
import random
import re
import subprocess
import threading
import time
import traceback
from pathlib import Path
from datasets import DownloadConfig, load_dataset

import docker
import typer
from jinja2 import StrictUndefined, Template
from rich.live import Live
from swebench.harness.docker_build import build_env_images, build_instance_image
from swebench.harness.test_spec.test_spec import make_test_spec

from swe_agent import Environment
from swe_agent.agents.default import DefaultAgent
from swe_agent.config import builtin_config_dir, get_config_from_spec
from swe_agent.environments.docker import docker_available
from swe_agent.environments import get_environment
from swe_agent.environments.singularity import resolve_singularity_image, singularity_available
from swe_agent.models import get_model
from swe_agent.run.benchmarks.deepswe_eval import (
    convert_deepswe_instance,
    is_deepswe_instance,
)
from swe_agent.run.benchmarks.r2egym_eval import (
    convert_r2egym_instance,
    is_r2egym_instance,
    r2egym_instance_id,
)
from swe_agent.run.benchmarks.swebench_pro_eval import (
    convert_swebench_pro_instance,
    is_swebench_pro_instance,
)
from swe_agent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from swe_agent.utils.log import add_file_handler, logger
from swe_agent.utils.serialize import UNSET, recursive_merge

_HELP_TEXT = """Run mini-SWE-agent on SWEBench instances.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

[bold red]IMPORTANT:[/bold red] [red]If you set this option, the default config file will not be used.[/red]
So you need to explicitly set it e.g., with [bold green]-c swebench.yaml <other options>[/bold green]

Multiple configs will be recursively merged.

Examples:

[bold red]-c model.model_kwargs.temperature=0[/bold red] [red]You forgot to add the default config file! See above.[/red]

[bold green]-c swebench.yaml -c model.model_kwargs.temperature=0.5[/bold green]

[bold green]-c swebench.yaml -c agent.max_iterations=50[/bold green]
"""

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench.yaml"

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
    "r2egym": "R2E-Gym/R2E-Gym-Subset",
    "swebench_pro": "ScaleAI/SWE-bench_Pro",
    "deepswe": "datacurve/deep-swe",
}

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
_OUTPUT_FILE_LOCK = threading.Lock()
_PREPARED_IMAGE_LOCK = threading.Lock()
_PREPARED_IMAGES: set[str] = set()
_IMAGE_RESOLUTION_LOCK = threading.Lock()
_IMAGE_RESOLUTION_CACHE: dict[str, tuple[str, str | None]] = {}
OFFICIAL_IMAGE_NAMESPACE = "swebench"
LOCAL_DATASET_DIRS = {
    "princeton-nlp/SWE-Bench_Verified": "swebench_verified",
    "ScaleAI/SWE-bench_Pro": "swebench_pro",
    "datacurve/deep-swe": "deepswe",
    "R2E-Gym/R2E-Gym-Subset": "r2egym",
}


def _load_dataset(*args, **kwargs):
    dataset_path = args[0] if args else kwargs.get("path")
    local_name = LOCAL_DATASET_DIRS.get(str(dataset_path))
    dataset_roots = []
    if configured_root := os.getenv("RLER_DATASETS_ROOT"):
        dataset_roots.append(Path(configured_root).expanduser())
    dataset_roots.extend(parent / "datasets" for parent in Path(__file__).resolve().parents)
    local_path = next(
        (
            root / local_name / "raw"
            for root in dataset_roots
            if local_name and (root / local_name / "raw").exists()
        ),
        None,
    )
    if local_path is not None:
        if args:
            return load_dataset(str(local_path), *args[1:], **kwargs)
        return load_dataset(**{**kwargs, "path": str(local_path)})
    if os.getenv("RLER_HF_DATASET_LOCAL_ONLY", "0") != "0":
        local_kwargs = {**kwargs, "download_config": DownloadConfig(local_files_only=True)}
        return load_dataset(*args, **local_kwargs)
    return load_dataset(*args, **kwargs)


class ProgressTrackingAgent(DefaultAgent):
    """Simple wrapper around DefaultAgent that provides progress updates."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        """Override step to provide progress updates."""
        self.progress_manager.update_instance_status(self.instance_id, f"Step {self.n_calls + 1:3d} (${self.cost:.2f})")
        return super().step()


def get_swebench_docker_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    return resolve_swebench_image(instance)[0]


def get_swebench_singularity_image_name(instance: dict) -> str:
    """Get the Apptainer/Singularity image for a SWEBench-style instance."""
    try:
        return resolve_singularity_image(str(instance.get("docker_image") or "local"), instance)
    except FileNotFoundError:
        return resolve_singularity_image(get_swebench_docker_image_name(instance), instance)


def get_swebench_harness_namespace(instance: dict) -> str | None:
    """Get the harness namespace that matches the resolved SWE-bench image."""
    return resolve_swebench_image(instance)[1]


def resolve_swebench_image(instance: dict) -> tuple[str, str | None]:
    """Resolve a SWE-bench instance image, preferring official published images."""
    instance_id = instance["instance_id"]
    with _IMAGE_RESOLUTION_LOCK:
        cached = _IMAGE_RESOLUTION_CACHE.get(instance_id)
    if cached is not None:
        return cached

    image_name = instance.get("image_name") or instance.get("docker_image")
    if image_name:
        resolved = (image_name, _infer_harness_namespace(image_name, instance))
    else:
        resolved = _resolve_generated_swebench_image(instance)

    with _IMAGE_RESOLUTION_LOCK:
        _IMAGE_RESOLUTION_CACHE[instance_id] = resolved
    return resolved


def _resolve_generated_swebench_image(instance: dict) -> tuple[str, str | None]:
    official_spec = make_test_spec(instance, namespace=OFFICIAL_IMAGE_NAMESPACE)
    if not docker_available():
        return official_spec.instance_image_key, OFFICIAL_IMAGE_NAMESPACE
    if _registry_image_exists(official_spec.instance_image_key):
        return official_spec.instance_image_key, OFFICIAL_IMAGE_NAMESPACE

    local_spec = make_test_spec(instance)
    image_name = local_spec.instance_image_key
    with _PREPARED_IMAGE_LOCK:
        if image_name not in _PREPARED_IMAGES:
            client = docker.from_env()
            try:
                build_env_images(
                    client,
                    [instance],
                    force_rebuild=False,
                    max_workers=1,
                    namespace=local_spec.namespace,
                    instance_image_tag=local_spec.instance_image_tag,
                    env_image_tag=local_spec.env_image_tag,
                )
                build_instance_image(local_spec, client, logger, nocache=False)
            finally:
                client.close()
            _PREPARED_IMAGES.add(image_name)
    return image_name, None


def select_container_environment_class(requested: str | None) -> str:
    if requested is not None:
        if requested == "docker" and docker_available():
            return "docker"
        if requested == "singularity" and singularity_available():
            return "singularity"
    if docker_available():
        return "docker"
    if singularity_available():
        return "singularity"
    raise RuntimeError(f"No supported container environment is available on this system")


def _registry_image_exists(image_name: str) -> bool:
    client = docker.from_env()
    try:
        client.images.get_registry_data(image_name)
        return True
    except docker.errors.NotFound:
        return False
    except docker.errors.APIError:
        logger.warning("Failed to query remote image %s; falling back to local build.", image_name)
        return False
    finally:
        client.close()


def _infer_harness_namespace(image_name: str, instance: dict) -> str | None:
    try:
        official_image = make_test_spec(instance, namespace=OFFICIAL_IMAGE_NAMESPACE).instance_image_key
    except Exception:
        return None
    if image_name == official_image:
        return OFFICIAL_IMAGE_NAMESPACE
    return None


def get_sb_environment(config: dict, instance: dict) -> Environment:
    env_config = config.setdefault("environment", {})
    env_config["environment_class"] = select_container_environment_class(env_config.get("environment_class", "docker"))
    if env_config["environment_class"] in ["docker", "swerex_modal"]:
        env_config["image"] = get_swebench_docker_image_name(instance)
    elif env_config["environment_class"] in ["singularity", "contree"]:
        env_config["image"] = get_swebench_singularity_image_name(instance)
    if is_r2egym_instance(instance):
        env_config["cwd"] = "/testbed"
        env_config.setdefault("dataset_name", "r2egym")
    elif is_swebench_pro_instance(instance) or is_deepswe_instance(instance):
        env_config["cwd"] = instance.get("swebench_workdir", "/app")
    else:
        env_config["cwd"] = env_config.get("cwd") or "/testbed"

    env = get_environment(env_config)
    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command, undefined=StrictUndefined).render(**instance)
        out = env.execute(startup_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str):
    """Update the output JSON file with results from a single instance."""
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, instance_id: str):
    """Remove an instance from the predictions file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id in output_data:
            del output_data[instance_id]
            output_path.write_text(json.dumps(output_data, indent=2))


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
) -> None:
    """Process a single SWEBench instance."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    # avoid inconsistent state if something here fails and there's leftover previous files
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)
    model = get_model(config=config.get("model", {}))
    task = instance["problem_statement"]

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pulling/starting environment")

    agent = None
    env = None
    exit_status = None
    result = None
    extra_info = {}

    try:
        env = get_sb_environment(config, instance)
        agent = ProgressTrackingAgent(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=instance_id,
            **config.get("agent", {}),
        )
        info = agent.run(task)
        exit_status = info.get("exit_status")
        result = info.get("submission")
    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, ""
        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}
    finally:
        if agent is not None:
            traj_path = instance_dir / f"{instance_id}.traj.json"
            agent.save(
                traj_path,
                {
                    "info": {
                        "exit_status": exit_status,
                        "submission": result,
                        **extra_info,
                    },
                    "instance_id": instance_id,
                },
            )
            logger.info(f"Saved trajectory to '{traj_path}'")
        if env is not None and hasattr(env, "cleanup"):
            env.cleanup()
        update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, result)
        progress_manager.on_instance_end(instance_id, exit_status)


def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[dict]:
    """Filter and slice a list of SWEBench instances."""
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]
    if (after_filter := len(instances)) != before_filter:
        logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != before_filter:
            logger.info(f"Instance slice: {before_filter} -> {after_slice} instances")
    return instances


def load_swebench_instances(subset: str, split: str) -> list[dict]:
    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    return [_normalize_dataset_row(dataset_path, row) for row in _load_dataset(dataset_path, split=split)]


def load_swebench_instances_slice(subset: str, split: str, offset: int, limit: int) -> list[dict]:
    if offset < 0 or limit < 1:
        raise ValueError("offset must be non-negative and limit must be positive")
    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading {limit} instance(s) from {dataset_path}, split {split}, offset {offset}...")
    rows = _load_dataset(dataset_path, split=split, streaming=True).skip(offset).take(limit)
    return [_normalize_dataset_row(dataset_path, row) for row in rows]


def load_swebench_instances_by_id(subset: str, split: str, instance_ids: list[str]) -> list[dict]:
    if not instance_ids:
        return []
    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading {len(instance_ids)} instance(s) from {dataset_path}, split {split}...")
    wanted = set(instance_ids)
    found: dict[str, dict] = {}
    for row in _load_dataset(dataset_path, split=split, streaming=True):
        instance_id = _row_instance_id(dataset_path, row)
        if instance_id in wanted and instance_id not in found:
            found[instance_id] = _normalize_dataset_row(dataset_path, row)
            if len(found) == len(wanted):
                break
    missing = wanted - set(found)
    if missing:
        raise RuntimeError(f"Instances not found in {subset}/{split}: {', '.join(sorted(missing))}")
    return [found[instance_id] for instance_id in instance_ids]


def _normalize_dataset_row(dataset_path: str, row: dict) -> dict:
    instance = dict(row)
    if dataset_path.startswith("R2E-Gym/"):
        return convert_r2egym_instance(instance)
    if dataset_path in {"ScaleAI/SWE-bench_Pro"}:
        return convert_swebench_pro_instance(instance)
    if dataset_path in {"datacurve/deep-swe"}:
        return convert_deepswe_instance(instance)
    return instance


def _row_instance_id(dataset_path: str, row: dict) -> str:
    if row.get("instance_id"):
        return str(row["instance_id"])
    if dataset_path.startswith("R2E-Gym/"):
        return r2egym_instance_id(dict(row))
    raise RuntimeError(f"Dataset {dataset_path} row does not have an instance_id")


def build_swebench_config(
    *,
    config_spec: list[str],
    model: str | None = None,
    model_class: str | None = None,
    environment_class: str | None = None,
    extra_overrides: dict | None = None,
) -> dict:
    logger.info(f"Building agent config from specs: {config_spec}")
    configs = [get_config_from_spec(spec) for spec in config_spec]
    merged_overrides = {
        "environment": {"environment_class": environment_class or UNSET},
        "model": {"model_name": model or UNSET, "model_class": model_class or UNSET},
    }
    if extra_overrides:
        merged_overrides = recursive_merge(merged_overrides, extra_overrides)
    configs.append(merged_overrides)
    return recursive_merge(*configs)


def run_swebench_instances(
    *,
    instances: list[dict],
    output_path: Path,
    config: dict,
    workers: int = 1,
    redo_existing: bool = False,
    show_live_progress: bool = True,
) -> list[dict]:
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "swe_agent.log")

    runnable_instances = instances
    if not redo_existing and (output_path / "preds.json").exists():
        existing_instances = list(json.loads((output_path / "preds.json").read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        runnable_instances = [
            instance for instance in runnable_instances if instance["instance_id"] not in existing_instances
        ]
    logger.info(f"Running on {len(runnable_instances)} instances...")

    progress_manager = RunBatchProgressManager(
        len(runnable_instances), output_path / f"exit_statuses_{time.time()}.yaml"
    )

    def process_futures(futures: dict[concurrent.futures.Future, str]) -> None:
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                logger.error(f"Error in future for instance {instance_id}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(instance_id, e)

    progress_context = Live(progress_manager.render_group, refresh_per_second=4) if show_live_progress else contextlib.nullcontext()
    with progress_context:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_instance, instance, output_path, config, progress_manager): instance[
                    "instance_id"
                ]
                for instance in runnable_instances
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)

    return runnable_instances


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use (e.g., 'anthropic' or 'swe_agent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT, rich_help_panel="Basic"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment type to use. Recommended are docker or singularity", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    output_path = Path(output)
    instances = load_swebench_instances(subset, split)
    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)
    config = build_swebench_config(
        config_spec=config_spec,
        model=model,
        model_class=model_class,
        environment_class=environment_class,
    )
    run_swebench_instances(
        instances=instances,
        output_path=output_path,
        config=config,
        workers=workers,
        redo_existing=redo_existing,
    )


if __name__ == "__main__":
    app()
