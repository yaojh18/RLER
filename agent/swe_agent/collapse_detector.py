"""Conservative code-only detection of unrecovered rollout mode collapse."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


_COUNTER_SUFFIX = re.compile(
    r"(?i)\b((?:debug|test|issue|apply[_-]?fix|fix|script|attempt|try)"
    r"(?:[_-][a-z]+)*[_-]?)\d+(?=\.py\b)"
)


def _normalize_counter_suffixes(value: str) -> str:
    """Collapse only generated script counters, not task-relevant numbers."""

    value = _COUNTER_SUFFIX.sub(r"\1#", value)
    return re.sub(r"\s+", " ", value).strip()


def _has_sustained_recovery(
    cycles: list[dict[str, Any]],
    episode_end: int,
    *,
    minimum_stable_run: int = 4,
) -> bool:
    """Require a stable action suffix, not one lucky response, as recovery."""

    later = cycles[episode_end + 1 :]
    stable_run = 0
    for cycle in later:
        stable = bool(cycle["commands"]) and not (
            cycle["format_error"] or cycle["intra_response_repetition"]
        )
        stable_run = stable_run + 1 if stable else 0
        if stable_run >= minimum_stable_run:
            return True
    terminal_indices = [
        index for index, cycle in enumerate(later) if cycle["terminal_submission"]
    ]
    return bool(
        terminal_indices
        and sum(bool(cycle["commands"]) for cycle in later[: terminal_indices[-1]])
        >= 1
    )


def _best_run(values: list[Any]) -> tuple[int, int]:
    best_length = 0
    best_end = -1
    run_length = 0
    previous: Any = None
    for index, value in enumerate(values):
        if value and value == previous:
            run_length += 1
        else:
            run_length = 1 if value else 0
        previous = value or None
        if run_length > best_length:
            best_length = run_length
            best_end = index
    return best_length, best_end


def _assistant_cycles(
    messages: list[dict[str, Any]], assistant_step_limit: int | None
) -> list[dict[str, Any]]:
    cycles: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        if assistant_step_limit is not None and len(cycles) >= assistant_step_limit:
            break
        content = str(message.get("content", message.get("message", "")) or "")
        commands = tuple(
            re.sub(r"\s+", " ", str(call.get("command") or "")).strip()
            for call in (message.get("tool_calls") or [])
            if isinstance(call, dict) and str(call.get("command") or "").strip()
        )
        intra_response_repetition = ""
        if len(content) >= 8_000:
            character_run = re.search(r"(.)\1{2047,}", content, flags=re.DOTALL)
            if character_run is not None:
                intra_response_repetition = (
                    "intra_response_character_run:"
                    f"character={character_run.group(1)!r}:"
                    f"minimum_run=2048"
                )
            action_blocks = content.count("```mswea_bash_command")
            if (
                not intra_response_repetition
                and len(content) >= 40_000
                and not commands
                and action_blocks >= 4
            ):
                # A single unparsable assistant response that streams many
                # independent actions until the completion cap is the other
                # observed completion-collapse mode.
                intra_response_repetition = (
                    "intra_response_unparsed_action_stream:"
                    f"chars={len(content)}:action_blocks={action_blocks}"
                )
            # Code/source listings legitimately repeat tokens.  The detector
            # therefore measures only prose outside fenced code blocks.
            prose = re.sub(r"```.*?```", " ", content, flags=re.DOTALL)
            words = re.findall(r"[A-Za-z_][A-Za-z0-9_'-]*|\d+", prose.lower())
            if len(words) >= 800:
                window_size = 24
                windows = Counter(
                    tuple(words[start : start + window_size])
                    for start in range(len(words) - window_size + 1)
                )
                repeated_windows = sum(
                    count for count in windows.values() if count >= 2
                )
                repeated_ratio = repeated_windows / max(1, sum(windows.values()))
                peak_count = max(windows.values(), default=0)
                if (
                    not intra_response_repetition
                    and repeated_ratio >= 0.35
                    and peak_count >= 4
                ):
                    intra_response_repetition = (
                        "intra_response_repetition:"
                        f"ratio={repeated_ratio:.3f}:peak={peak_count}"
                    )
        next_user = ""
        for later in messages[index + 1 :]:
            if later.get("role") == "assistant":
                break
            if later.get("role") == "user":
                next_user = str(
                    later.get("content", later.get("message", "")) or ""
                )
                break
        cycles.append(
            {
                "response": re.sub(r"\s+", " ", content).strip(),
                "normalized_response": _normalize_counter_suffixes(content),
                "commands": commands,
                "normalized_commands": tuple(
                    _normalize_counter_suffixes(command) for command in commands
                ),
                "format_error": next_user.lstrip().startswith("Format error:"),
                "intra_response_repetition": intra_response_repetition,
                "terminal_submission": content.lstrip().startswith("diff --git ")
                or any(
                    "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in command
                    for command in commands
                ),
            }
        )
    return cycles


def trajectory_collapse_reason(
    messages: list[dict[str, Any]], *, assistant_step_limit: int | None = None
) -> str | None:
    """Return a collapse reason only for a visible, unrecovered loop.

    ``assistant_step_limit`` makes the observation boundary explicit.  A bare
    context/completion-limit status is deliberately not a collapse signal:
    productive but long exploration can hit a policy limit without exhibiting
    mode collapse.
    """

    cycles = _assistant_cycles(messages, assistant_step_limit)
    if not cycles:
        return None

    format_indices = [
        index for index, cycle in enumerate(cycles) if cycle["format_error"]
    ]
    max_format_streak = 0
    current_format_streak = 0
    for cycle in cycles:
        if cycle["format_error"]:
            current_format_streak += 1
            max_format_streak = max(max_format_streak, current_format_streak)
        else:
            current_format_streak = 0
    if format_indices:
        episode_end = format_indices[-1]
        recovered = _has_sustained_recovery(
            cycles,
            episode_end,
            # At a truncated observation boundary, three clean command turns
            # are enough to avoid a false positive.  The fourth clean turn is
            # unavailable in a real counterexample from this experiment even
            # though the rollout continues productively afterwards.  Full
            # trajectory adjudication retains the stricter four-turn rule.
            minimum_stable_run=3 if assistant_step_limit is not None else 4,
        )
        large_scale_episode = len(format_indices) >= 30 or max_format_streak >= 15
        if assistant_step_limit is None:
            parser_loop = large_scale_episode or (
                not recovered
                and (max_format_streak >= 7 or len(format_indices) >= 10)
            )
        else:
            # At the early observation boundary, do not punish a few
            # correctable parser mistakes.  Require a substantial visible
            # streak or a high error density, and honor a stable recovery that
            # is already visible before the boundary.
            density = len(format_indices) / len(cycles)
            parser_loop = not recovered and (
                max_format_streak >= 8
                or (len(format_indices) >= 10 and density >= 0.30)
            )
        if parser_loop:
            return (
                "format_error_parser_loop:"
                f"total={len(format_indices)}:streak={max_format_streak}"
            )

    repeated_message_indices = [
        index
        for index, cycle in enumerate(cycles)
        if cycle["intra_response_repetition"]
    ]
    if repeated_message_indices:
        last_repeated = repeated_message_indices[-1]
        recovered = _has_sustained_recovery(cycles, last_repeated)
        if not recovered:
            return str(cycles[last_repeated]["intra_response_repetition"])

    for key, label, minimum_run in (
        ("normalized_response", "repeated_counter_normalized_response", 6),
        ("normalized_commands", "repeated_counter_normalized_command", 6),
    ):
        best_length, best_end = _best_run([cycle[key] for cycle in cycles])
        recovered = _has_sustained_recovery(cycles, best_end)
        if best_length >= minimum_run and not recovered:
            return f"{label}:{best_length}"

    checkout_paths = Counter()
    checkout_last: dict[str, int] = {}
    for index, cycle in enumerate(cycles):
        for command in cycle["commands"]:
            match = re.search(
                r"(?:^|&&|;)\s*git\s+checkout(?:\s+--)?\s+([^\s;&|]+)",
                command,
            )
            if match is not None:
                path = match.group(1)
                checkout_paths[path] += 1
                checkout_last[path] = index
    if checkout_paths:
        path, count = checkout_paths.most_common(1)[0]
        if assistant_step_limit is not None:
            total_checkouts = sum(checkout_paths.values())
            last_checkout = max(checkout_last.values())
            # Twelve destructive resets already consume nearly a third of the
            # observable rollout.  Counting across touched files catches the
            # audited two-file variant without using task identity or result
            # fields.  This was the only added training-side signal separable
            # with zero false positives in the fully labelled cohort.
            if total_checkouts >= 12 and not _has_sustained_recovery(
                cycles, last_checkout
            ):
                return (
                    "repeated_edit_revert_cycle:"
                    f"paths={len(checkout_paths)}:checkouts={total_checkouts}"
                )
        elif count >= 20 and not _has_sustained_recovery(
            cycles, checkout_last[path]
        ):
            return f"repeated_edit_revert_cycle:path={path}:checkouts={count}"

    # Alternating short-period loops are invisible to consecutive-run checks.
    # Four complete periods plus no two-cycle recovery keeps this conservative.
    for key, label in (
        ("normalized_response", "periodic_assistant_response_loop"),
        ("normalized_commands", "periodic_tool_command_loop"),
    ):
        values = [cycle[key] for cycle in cycles]
        for period in range(2, 5):
            minimum_length = period * 4
            for start in range(len(values) - minimum_length + 1):
                end = start + period
                while (
                    end < len(values)
                    and values[end]
                    and values[end] == values[end - period]
                ):
                    end += 1
                length = end - start
                if length < minimum_length:
                    continue
                recovered = _has_sustained_recovery(cycles, end - 1)
                if not recovered:
                    return f"{label}:period={period}:length={length}"
    return None
