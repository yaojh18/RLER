#!/usr/bin/env python3

"""This is the central entry point to the mini-extra script. Use subcommands
to invoke other command line utilities like running on benchmarks, editing config,
inspecting trajectories, etc.
"""

import sys

from rich.console import Console
from swe_agent.run.benchmarks.swebench import app as swebench_app
from swe_agent.run.benchmarks.swebench_single import app as swebench_single_app
from swe_agent.run.utilities.config import app as config_app
from swe_agent.run.utilities.inspector import app as inspector_app

subcommands = [
    (config_app, ["config"], "Manage the global config file"),
    (inspector_app, ["inspect", "i", "inspector"], "Run inspector (browse trajectories)"),
    (swebench_app, ["swebench"], "Evaluate on SWE-bench (batch mode)"),
    (swebench_single_app, ["swebench-single"], "Evaluate on SWE-bench (single instance)"),
]


def get_docstring() -> str:
    lines = [
        "This is the [yellow]central entry point for all extra commands[/yellow] from mini-swe-agent.",
        "",
        "Available sub-commands:",
        "",
    ]
    for _, aliases, description in subcommands:
        alias_text = " or ".join(f"[bold green]{alias}[/bold green]" for alias in aliases)
        lines.append(f"  {alias_text}: {description}")
    return "\n".join(lines)


def main():
    args = sys.argv[1:]

    if len(args) == 0 or len(args) == 1 and args[0] in ["-h", "--help"]:
        return Console().print(get_docstring())

    for app, aliases, _ in subcommands:
        if args[0] in aliases:
            return app(args[1:], prog_name=f"mini-extra {aliases[0]}")

    return Console().print(get_docstring())


if __name__ == "__main__":
    main()
