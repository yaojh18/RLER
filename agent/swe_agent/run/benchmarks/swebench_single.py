"""Run on a single SWE-Bench instance."""

from pathlib import Path

import typer

from swe_agent import global_config_dir
from swe_agent.agents import get_agent
from swe_agent.config import builtin_config_dir
from swe_agent.models import get_model
from swe_agent.run.benchmarks.swebench import (
    DATASET_MAPPING,
    build_swebench_config,
    get_sb_environment,
    load_swebench_instances_by_id,
    load_swebench_instances_slice,
)
from swe_agent.utils.log import logger
from swe_agent.utils.serialize import UNSET

DEFAULT_OUTPUT_FILE = global_config_dir / "last_swebench_single_run.traj.json"
DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench.yaml"

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

[bold red]IMPORTANT:[/bold red] [red]If you set this option, the default config file will not be used.[/red]
So you need to explicitly set it e.g., with [bold green]-c swebench.yaml <other options>[/bold green]

Multiple configs will be recursively merged.

Examples:

[bold red]-c model.model_kwargs.temperature=0[/bold red] [red]You forgot to add the default config file! See above.[/red]

[bold green]-c swebench.yaml -c model.model_kwargs.temperature=0.5[/bold green]

[bold green]-c swebench.yaml -c agent.mode=yolo[/bold green]
"""


# fmt: off
@app.command()
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    instance_spec: str = typer.Option(0, "-i", "--instance", help="SWE-Bench instance ID or index", rich_help_panel="Data selection"),
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use (e.g., 'anthropic' or 'swe_agent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    agent_class: str | None = typer.Option(None, "--agent-class", help="Agent class to use (e.g., 'interactive' or 'swe_agent.agents.interactive.InteractiveAgent')", rich_help_panel="Advanced"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment class to use (e.g., 'docker' or 'swe_agent.environments.docker.DockerEnvironment')", rich_help_panel="Advanced"),
    yolo: bool = typer.Option(False, "-y", "--yolo", help="Run without confirmation"),
    cost_limit: float | None = typer.Option(None, "-l", "--cost-limit", help="Cost limit. Set to 0 to disable."),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT, rich_help_panel="Basic"),
    exit_immediately: bool = typer.Option(False, "--exit-immediately", help="Exit immediately when the agent wants to finish instead of prompting.", rich_help_panel="Advanced"),
    output: Path | None = typer.Option(DEFAULT_OUTPUT_FILE, "-o", "--output", help="Output trajectory file", rich_help_panel="Basic"),
) -> None:
    # fmt: on
    """Run on a single SWE-Bench instance."""
    logger.info(f"Loading dataset from {DATASET_MAPPING.get(subset, subset)}, split {split}...")
    if instance_spec.isnumeric():
        instance = load_swebench_instances_slice(subset, split, int(instance_spec), 1)[0]
    else:
        instance = load_swebench_instances_by_id(subset, split, [instance_spec])[0]

    config = build_swebench_config(
        config_spec=config_spec,
        model=model_name,
        model_class=model_class,
        environment_class=environment_class,
        extra_overrides={
        "agent": {
            "agent_class": agent_class or UNSET,
            "mode": "yolo" if yolo else UNSET,
            "cost_limit": cost_limit or UNSET,
            "confirm_exit": False if exit_immediately else UNSET,
            "output_path": output or UNSET,
        },
    },
    )

    env = get_sb_environment(config, instance)
    agent = get_agent(
        get_model(config=config.get("model", {})),
        env,
        config.get("agent", {}),
        default_type="interactive",
    )
    agent.run(instance["problem_statement"])


if __name__ == "__main__":
    app()
