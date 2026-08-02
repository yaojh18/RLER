from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path

import ray

from slime.ray.placement_group import (
    create_placement_groups,
    create_rollout_manager,
    create_training_models,
)
from slime.rollout.data_source import (
    TRAIN_INSTANCE_BUDGET_EXHAUSTED_KEY,
    TRAIN_VALIDATION_BOUNDARY_KEY,
)
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import (
    configure_logger,
    finish_tracking,
    init_tracking,
)
from slime.utils.misc import should_run_periodic_action
logger = logging.getLogger(__name__)
_NO_ROLLOUT_DATA = object()
_CHECKPOINT_DIR_RE = re.compile(r"iter_(\d{7})")


def policy_version_for_checkpoint(checkpoint_id: int) -> str:
    from swe_agent.policy_version import policy_version_for_checkpoint as impl

    return impl(checkpoint_id)


def begin_policy_weight_update(target_version: str) -> str:
    from swe_agent.policy_version import begin_policy_weight_update as impl

    return impl(target_version)


def commit_policy_weight_update(
    target_version: str,
    transition_id: str,
) -> None:
    from swe_agent.policy_version import commit_policy_weight_update as impl

    impl(target_version, transition_id)


def fail_policy_weight_update(
    target_version: str,
    transition_id: str,
    error: BaseException,
) -> None:
    from swe_agent.policy_version import fail_policy_weight_update as impl

    impl(target_version, transition_id, error)


def _update_rollout_weights(actor_model, last_rollout_id: int) -> str:
    """Broadcast weights while publishing the exact SGLang policy epoch."""
    if not str(os.environ.get("RLER_POLICY_VERSION_STATE_PATH") or "").strip():
        # Preserve upstream Slime semantics for every run that did not opt in
        # to request-level validation versioning.
        actor_model.update_weights()
        return ""
    target_version = policy_version_for_checkpoint(last_rollout_id)
    transition_id = begin_policy_weight_update(target_version)
    try:
        actor_model.update_weights()
    except BaseException as exc:
        fail_policy_weight_update(
            target_version,
            transition_id,
            exc,
        )
        raise
    commit_policy_weight_update(target_version, transition_id)
    logger.info(
        "committed rollout policy version %s",
        target_version,
    )
    return target_version


def _uses_instance_attempt_eval(args) -> bool:
    return getattr(args, "eval_instance_interval", None) is not None


def _should_run_eval(rollout_id, args):
    """Run update-scheduled eval only when no instance cadence is active."""
    if _uses_instance_attempt_eval(args):
        return False
    return should_run_periodic_action(
        rollout_id,
        args.eval_interval,
        num_rollout=args.num_rollout,
    )


def _signal_attempt(rollout_data) -> int:
    attempted = rollout_data.get("attempted_instances")
    if attempted is None:
        attempted = (rollout_data.get("progress") or {}).get(
            "attempted_instances"
        )
    if attempted is None:
        raise RuntimeError(
            "source-instance control signal is missing attempted_instances"
        )
    return int(attempted)


def _stop_after_validation_attempt(args) -> int | None:
    value = getattr(args, "stop_after_validation_attempt", None)
    return None if value is None else int(value)


def _validate_chunk_stop_start(args, rollout_manager) -> None:
    """Require an intermediate stop target to be strictly in the future."""
    target = _stop_after_validation_attempt(args)
    if target is None:
        return
    progress = ray.get(
        rollout_manager.get_train_instance_progress.remote()
    )
    attempted = int(progress.get("attempted_instances", 0))
    acknowledged = int(progress.get("last_validation_attempt", 0))
    if target <= acknowledged or target < attempted:
        raise RuntimeError(
            "stop-after validation target must be a future unacknowledged "
            "boundary for this checkpoint: "
            f"target={target} attempted={attempted} "
            f"last_validation_attempt={acknowledged}"
        )


def _require_final_instance_budget_exhaustion(
    args,
    rollout_manager,
    *,
    intermediate_chunk_stop: bool,
) -> dict | None:
    """Fail before final validation/checkpoint on an incomplete formal run."""
    if (
        not getattr(
            args,
            "require_train_instance_budget_exhaustion",
            False,
        )
        or intermediate_chunk_stop
    ):
        return None
    configured_budget = getattr(args, "train_instance_budget", None)
    if configured_budget is None:
        raise RuntimeError(
            "require-train-instance-budget-exhaustion needs a configured "
            "train_instance_budget"
        )
    progress = ray.get(
        rollout_manager.get_train_instance_progress.remote()
    )
    attempted = int(progress.get("attempted_instances", 0))
    budget = int(configured_budget)
    if attempted != budget:
        raise RuntimeError(
            "final training invocation did not exhaust the source-instance "
            "budget; refusing final validation and checkpoint: "
            f"attempted={attempted} budget={budget}"
        )
    reported_budget = progress.get("instance_budget")
    if (
        reported_budget is not None
        and int(reported_budget) != budget
    ):
        raise RuntimeError(
            "rollout source reports a different instance budget than the "
            f"training contract: reported={reported_budget} configured={budget}"
        )
    return progress


def _run_instance_validation(
    rollout_manager,
    *,
    rollout_id: int,
    attempted_instances: int,
):
    """Drain validation only for an explicit final/chunk checkpoint."""
    progress = ray.get(
        rollout_manager.drain_train_validations.remote(
            int(rollout_id),
            int(attempted_instances),
        )
    )
    acknowledged = int(progress.get("last_validation_attempt", 0))
    if acknowledged < attempted_instances:
        raise RuntimeError(
            "rollout manager did not acknowledge the completed validation: "
            f"requested={attempted_instances}, acknowledged={acknowledged}"
        )
    return progress


def _checkpoint_retain_latest(args) -> int:
    value = getattr(args, "checkpoint_retain_latest", None)
    if value is None:
        value = os.environ.get("RLER_CHECKPOINT_RETAIN_LATEST", "0")
    try:
        retain = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "checkpoint retention must be a non-negative integer, got "
            f"{value!r}"
        ) from exc
    if retain < 0:
        raise ValueError(
            "checkpoint retention must be non-negative, got "
            f"{retain}"
        )
    return retain


def _prune_training_checkpoints(args, latest_rollout_id: int) -> list[int]:
    """Keep the newest N complete numeric checkpoints under ``args.save``.

    Pruning is deliberately fail-closed.  It only operates on exact
    ``iter_0000000`` directory names in the configured save directory, and
    only when the numeric Megatron tracker, newest checkpoint directory, and
    matching newest rollout data-source state all agree.  Dataset-state files
    paired with removed checkpoints are removed at the same time.
    """
    retain = _checkpoint_retain_latest(args)
    if retain == 0:
        return []

    save_root = Path(args.save)
    try:
        resolved_root = save_root.resolve(strict=True)
    except FileNotFoundError:
        logger.warning(
            "skip checkpoint pruning because save root does not exist: %s",
            save_root,
        )
        return []
    if (
        not resolved_root.is_dir()
        or resolved_root == Path("/")
        or len(resolved_root.parts) < 3
        or save_root.is_symlink()
    ):
        logger.warning(
            "skip checkpoint pruning for unsafe save root: %s",
            save_root,
        )
        return []

    tracker = resolved_root / "latest_checkpointed_iteration.txt"
    if tracker.is_symlink():
        logger.warning(
            "skip checkpoint pruning because tracker is a symlink: %s",
            tracker,
        )
        return []
    try:
        tracker_text = tracker.read_text().strip()
    except OSError:
        logger.warning(
            "skip checkpoint pruning because tracker is unreadable: %s",
            tracker,
        )
        return []
    if not tracker_text.isdigit():
        logger.warning(
            "skip checkpoint pruning because tracker is not numeric: %s",
            tracker_text,
        )
        return []
    tracker_id = int(tracker_text)
    if tracker_id != int(latest_rollout_id):
        logger.warning(
            "skip checkpoint pruning because tracker=%d does not match "
            "latest completed save=%d",
            tracker_id,
            latest_rollout_id,
        )
        return []

    rollout_state_root = resolved_root / "rollout"
    if (
        not rollout_state_root.is_dir()
        or rollout_state_root.is_symlink()
        or rollout_state_root.resolve().parent != resolved_root
    ):
        logger.warning(
            "skip checkpoint pruning for unsafe rollout-state root: %s",
            rollout_state_root,
        )
        return []
    latest_dir = resolved_root / f"iter_{tracker_id:07d}"
    latest_state = (
        rollout_state_root
        / f"global_dataset_state_dict_{tracker_id}.pt"
    )
    if (
        not latest_dir.is_dir()
        or latest_dir.is_symlink()
        or not latest_state.is_file()
        or latest_state.is_symlink()
    ):
        logger.warning(
            "skip checkpoint pruning because the latest model/data pair is "
            "incomplete: %s ; %s",
            latest_dir,
            latest_state,
        )
        return []

    checkpoints: list[tuple[int, Path]] = []
    for candidate in resolved_root.iterdir():
        match = _CHECKPOINT_DIR_RE.fullmatch(candidate.name)
        if (
            match is None
            or not candidate.is_dir()
            or candidate.is_symlink()
        ):
            continue
        # Guard against an unusual mount/symlink layout even though the
        # candidate itself is not a symlink.
        if candidate.resolve().parent != resolved_root:
            continue
        checkpoints.append((int(match.group(1)), candidate))

    if not checkpoints:
        return []
    checkpoints.sort()
    if checkpoints[-1][0] != tracker_id:
        logger.warning(
            "skip checkpoint pruning because newest numeric directory %d "
            "does not match tracker %d",
            checkpoints[-1][0],
            tracker_id,
        )
        return []

    to_remove = checkpoints[:-retain]
    removed: list[int] = []
    for checkpoint_id, checkpoint_dir in to_remove:
        # Revalidate every explicit target immediately before deletion.
        expected_dir = resolved_root / f"iter_{checkpoint_id:07d}"
        if (
            checkpoint_dir != expected_dir
            or not checkpoint_dir.is_dir()
            or checkpoint_dir.is_symlink()
            or checkpoint_dir.resolve().parent != resolved_root
        ):
            raise RuntimeError(
                f"refusing unsafe checkpoint prune target: {checkpoint_dir}"
            )
        state_path = (
            rollout_state_root
            / f"global_dataset_state_dict_{checkpoint_id}.pt"
        )
        shutil.rmtree(checkpoint_dir)
        if state_path.exists() or state_path.is_symlink():
            state_path.unlink()
        removed.append(checkpoint_id)
        logger.info(
            "pruned checkpoint/model-data pair for rollout %d",
            checkpoint_id,
        )
    return removed


def _rollout_dataset_state_path(
    args,
    rollout_id: int,
    *,
    staged: bool = False,
) -> Path:
    path = (
        Path(args.save)
        / "rollout"
        / f"global_dataset_state_dict_{int(rollout_id)}.pt"
    )
    return path.with_name(f"{path.name}.staged") if staged else path


def _stage_checkpoint_dataset_state(
    args,
    rollout_manager,
    *,
    rollout_id: int,
) -> bool:
    if not args.rollout_global_dataset:
        return False
    ray.get(
        rollout_manager.save.remote(
            int(rollout_id),
            staged=True,
        )
    )
    staged_path = _rollout_dataset_state_path(
        args,
        rollout_id,
        staged=True,
    )
    if not staged_path.is_file() or staged_path.is_symlink():
        raise RuntimeError(
            "rollout manager did not create a regular staged data-state "
            f"checkpoint: {staged_path}"
        )
    return True


def _commit_checkpoint_dataset_state(
    args,
    *,
    rollout_id: int,
) -> None:
    """Publish a staged cursor only after the matching model save succeeds."""
    staged_path = _rollout_dataset_state_path(
        args,
        rollout_id,
        staged=True,
    )
    final_path = _rollout_dataset_state_path(args, rollout_id)
    if not staged_path.is_file() or staged_path.is_symlink():
        raise RuntimeError(
            "cannot commit missing or unsafe staged rollout state: "
            f"{staged_path}"
        )
    save_root = Path(args.save)
    tracker = save_root / "latest_checkpointed_iteration.txt"
    checkpoint_dir = save_root / f"iter_{int(rollout_id):07d}"
    if tracker.is_symlink():
        raise RuntimeError(
            "cannot commit rollout state through a symlinked model "
            f"checkpoint tracker: {tracker}"
        )
    try:
        tracker_text = tracker.read_text().strip()
        tracker_id = int(tracker_text)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "cannot commit rollout state before the matching model "
            f"checkpoint tracker is durable: {tracker}"
        ) from exc
    if (
        tracker_id != int(rollout_id)
        or not checkpoint_dir.is_dir()
        or checkpoint_dir.is_symlink()
    ):
        raise RuntimeError(
            "cannot commit rollout state before the matching model "
            "checkpoint is durable: "
            f"requested={int(rollout_id)} tracker={tracker_id} "
            f"checkpoint_dir={checkpoint_dir}"
        )
    final_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged_path, final_path)
    _prune_training_checkpoints(args, int(rollout_id))


def _save_checkpoint(
    *,
    args,
    rollout_manager,
    actor_model,
    critic_model,
    rollout_id: int,
    save_model: bool,
    force_sync: bool,
    dataset_state_staged: bool = False,
) -> None:
    """Atomically pair the updated model with its exact future-data state."""
    if args.rollout_global_dataset and not dataset_state_staged:
        _stage_checkpoint_dataset_state(
            args,
            rollout_manager,
            rollout_id=rollout_id,
        )
    if save_model:
        if (not args.use_critic) or rollout_id >= args.num_critic_only_steps:
            actor_model.save_model(
                rollout_id,
                force_sync=(
                    force_sync
                    or bool(getattr(args, "async_save", False))
                ),
            )
        if args.use_critic:
            critic_model.save_model(
                rollout_id,
                force_sync=(
                    force_sync
                    or bool(getattr(args, "async_save", False))
                ),
            )
    if args.rollout_global_dataset:
        _commit_checkpoint_dataset_state(
            args,
            rollout_id=rollout_id,
        )


def _run_final_instance_validation_if_needed(
    args,
    rollout_manager,
    *,
    rollout_id: int,
) -> dict | None:
    if not _uses_instance_attempt_eval(args):
        return None
    progress = ray.get(
        rollout_manager.get_train_instance_progress.remote()
    )
    attempted = int(progress.get("attempted_instances", 0))
    last_validation = int(
        progress.get("last_validation_attempt", 0)
    )
    if attempted > last_validation:
        progress = _run_instance_validation(
            rollout_manager,
            rollout_id=rollout_id,
            attempted_instances=attempted,
        )
    return progress


def _refresh_existing_checkpoint_dataset_state(
    args,
    rollout_manager,
    *,
    checkpoint_id: int,
) -> bool:
    """Refresh an existing model checkpoint's cursor without rewriting it."""
    if not args.rollout_global_dataset or checkpoint_id < 0:
        return False
    save_root = Path(args.save)
    tracker = save_root / "latest_checkpointed_iteration.txt"
    checkpoint_dir = save_root / f"iter_{checkpoint_id:07d}"
    try:
        tracker_id = int(tracker.read_text().strip())
    except (OSError, ValueError):
        logger.warning(
            "cannot persist final data cursor without a numeric checkpoint "
            "tracker under save root %s",
            save_root,
        )
        return False
    if (
        tracker_id != checkpoint_id
        or not checkpoint_dir.is_dir()
        or checkpoint_dir.is_symlink()
    ):
        logger.warning(
            "cannot pair final data cursor with existing checkpoint %d "
            "under save root %s",
            checkpoint_id,
            save_root,
        )
        return False
    _stage_checkpoint_dataset_state(
        args,
        rollout_manager,
        rollout_id=checkpoint_id,
    )
    _commit_checkpoint_dataset_state(
        args,
        rollout_id=checkpoint_id,
    )
    return True


# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
def _train_upstream(args):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    # Always push actor weights to rollout once weights are loaded.
    _update_rollout_weights(
        actor_model,
        args.start_rollout_id - 1,
    )

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    # async train loop.
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = ray.get(rollout_data_next_future)

        # Start the next rollout early.
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        if args.use_critic:
            actor_trains_this_step = rollout_id >= args.num_critic_only_steps
            value_refs = critic_model.async_train(rollout_id, rollout_data_curr_ref)
            if actor_trains_this_step:
                ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            if (not args.use_critic) or rollout_id >= args.num_critic_only_steps:
                actor_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                )
            if args.use_critic:
                critic_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                )
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        if (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
            rollout_data_next_future = None
            _update_rollout_weights(actor_model, rollout_id)

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


def _train_instance_attempt_control(args):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    pgs = create_placement_groups(args)
    init_tracking(args)

    # The rollout manager must be initialized first because it can derive
    # num_rollout from the global dataset.
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(
        args,
        pgs["rollout"],
    )
    actor_model, critic_model = create_training_models(
        args,
        pgs,
        rollout_manager,
    )
    _validate_chunk_stop_start(args, rollout_manager)

    _update_rollout_weights(
        actor_model,
        args.start_rollout_id - 1,
    )
    last_weights_rollout_id = args.start_rollout_id - 1
    if args.check_weight_update_equal:
        ray.get(
            rollout_manager.check_weights.remote(action="compare")
        )

    rollout_id = args.start_rollout_id
    last_completed_rollout_id = rollout_id - 1
    last_model_save_rollout_id = None
    rollout_data_future = None
    ready_rollout_data = _NO_ROLLOUT_DATA
    stopped_by_instance_budget = False
    stopped_after_validation_boundary = False
    refill_preserved_validation_partial = False
    stop_after_attempt = _stop_after_validation_attempt(args)
    stop_boundary_acknowledged = False

    while rollout_id < args.num_rollout:
        if ready_rollout_data is _NO_ROLLOUT_DATA:
            if rollout_data_future is None:
                rollout_data_future = rollout_manager.generate.remote(
                    rollout_id
                )
            ready_rollout_data = ray.get(rollout_data_future)
            rollout_data_future = None

        signal = None
        if isinstance(ready_rollout_data, dict):
            if ready_rollout_data.get(TRAIN_VALIDATION_BOUNDARY_KEY):
                signal = "validation"
            elif ready_rollout_data.get(
                TRAIN_INSTANCE_BUDGET_EXHAUSTED_KEY
            ):
                signal = "budget"
        if signal == "validation":
            attempted = _signal_attempt(ready_rollout_data)
            boundary = int(
                ready_rollout_data.get("boundary", attempted)
            )
            if attempted != boundary:
                raise RuntimeError(
                    "validation control signal must stop at its exact "
                    f"boundary: attempted={attempted}, boundary={boundary}"
                )
            preserved_group_count = int(
                ready_rollout_data.get("preserved_group_count", 0) or 0
            )
            preserved_pending_count = int(
                ready_rollout_data.get("preserved_pending_count", 0) or 0
            )
            if preserved_group_count < 0:
                raise RuntimeError(
                    "validation control signal has negative "
                    f"preserved_group_count={preserved_group_count}"
                )
            if preserved_pending_count < 0:
                raise RuntimeError(
                    "validation control signal has negative "
                    f"preserved_pending_count={preserved_pending_count}"
                )
            preserved_partial = bool(
                ready_rollout_data.get("preserved_partial", False)
            )
            if preserved_partial != (
                preserved_group_count > 0
                or preserved_pending_count > 0
            ):
                raise RuntimeError(
                    "validation control signal has inconsistent preserved "
                    "partial metadata: "
                    f"preserved_partial={preserved_partial} "
                    f"preserved_group_count={preserved_group_count} "
                    f"preserved_pending_count={preserved_pending_count}"
                )
            if (
                refill_preserved_validation_partial
                and not preserved_partial
            ):
                raise RuntimeError(
                    "collector lost the optimizer partial while crossing a "
                    "later validation boundary during chunk-stop refill"
                )
            if (
                refill_preserved_validation_partial
                and stop_after_attempt is not None
                and boundary > stop_after_attempt
            ):
                # Do not let one nominal chunk silently consume and ACK the
                # next chunk's boundary. Failing before validation preserves
                # the previous checkpoint as the replay authority; the source
                # cursor and partial groups are regenerated on retry.
                raise RuntimeError(
                    "optimizer partial could not be refilled before the next "
                    "validation boundary: "
                    f"stop_target={stop_after_attempt} "
                    f"next_boundary={boundary} "
                    f"preserved_group_count={preserved_group_count}"
                )
            if (
                stop_after_attempt is not None
                and not refill_preserved_validation_partial
                and boundary > stop_after_attempt
            ):
                raise RuntimeError(
                    "training passed the requested intermediate validation "
                    f"boundary without stopping: target={stop_after_attempt} "
                    f"observed={boundary}"
                )
            # Use the pending optimizer-update id for logging. No optimizer
            # update is consumed, and generation retries this same id only
            # after validation has been acknowledged.
            _run_instance_validation(
                rollout_manager,
                rollout_id=rollout_id,
                attempted_instances=attempted,
            )
            ready_rollout_data = _NO_ROLLOUT_DATA
            if (
                stop_after_attempt is not None
                and boundary == stop_after_attempt
                and not refill_preserved_validation_partial
            ):
                stop_boundary_acknowledged = True
                if preserved_partial:
                    # The collector owns accepted groups that cannot be
                    # checkpointed or disposition-dropped at an intermediate
                    # boundary. Retry this same optimizer id after ACK, then
                    # stop immediately after its one normal update commits.
                    refill_preserved_validation_partial = True
                    logger.info(
                        "validation boundary %d preserved %d optimizer "
                        "groups and %d pending instances; refilling rollout "
                        "%d before chunk stop",
                        boundary,
                        preserved_group_count,
                        preserved_pending_count,
                        rollout_id,
                    )
                else:
                    stopped_after_validation_boundary = True
                    logger.info(
                        "stopping intermediate chunk at validation boundary "
                        "%d with no partial optimizer batch",
                        boundary,
                    )
                    break
            rollout_data_future = rollout_manager.generate.remote(
                rollout_id
            )
            continue
        if signal == "budget":
            if refill_preserved_validation_partial:
                raise RuntimeError(
                    "optimizer partial could not be refilled before the "
                    "terminal source-instance budget: "
                    f"stop_target={stop_after_attempt} "
                    f"attempted={_signal_attempt(ready_rollout_data)}"
                )
            refill_preserved_validation_partial = False
            stopped_by_instance_budget = True
            logger.info(
                "stopping normally at source-instance budget after %d "
                "attempts and %d completed optimizer updates",
                _signal_attempt(ready_rollout_data),
                max(0, last_completed_rollout_id - args.start_rollout_id + 1),
            )
            ready_rollout_data = _NO_ROLLOUT_DATA
            break

        rollout_data_curr_ref = ready_rollout_data
        ready_rollout_data = _NO_ROLLOUT_DATA

        should_save_checkpoint = should_run_periodic_action(
            rollout_id,
            args.save_interval,
            num_rollout_per_epoch,
            args.num_rollout,
        )
        required_final_budget_incomplete = False
        if (
            getattr(
                args,
                "require_train_instance_budget_exhaustion",
                False,
            )
            and stop_after_attempt is None
            and rollout_id == args.num_rollout - 1
        ):
            # Slime treats num_rollout-1 as an unconditional save point. If
            # that generic update ceiling is reached before the source budget,
            # train the already generated batch but suppress the misleading
            # "final" checkpoint; the post-loop exhaustion gate below raises
            # before validation or any replacement checkpoint.
            progress = ray.get(
                rollout_manager.get_train_instance_progress.remote()
            )
            attempted = int(progress.get("attempted_instances", 0))
            configured_budget = int(args.train_instance_budget)
            if attempted != configured_budget:
                required_final_budget_incomplete = True
                should_save_checkpoint = False
                logger.warning(
                    "suppressing final-step validation/checkpoint before budget "
                    "exhaustion: attempted=%d budget=%d",
                    attempted,
                    configured_budget,
                )
        dataset_state_staged = False
        if should_save_checkpoint and args.rollout_global_dataset:
            # Snapshot before N+1 is launched.  The collector snapshot includes
            # its accepted carry buffer and descriptors for every source
            # instance already drawn but still in flight, so the staged cursor
            # is the exact future-data state paired with model update N.
            dataset_state_staged = _stage_checkpoint_dataset_state(
                args,
                rollout_manager,
                rollout_id=rollout_id,
            )
        # Start N+1 early. A boundary/budget control message can be resolved
        # while update N trains, but is processed only after update N commits.
        completing_preserved_partial = (
            refill_preserved_validation_partial
        )
        if (
            not completing_preserved_partial
            and rollout_id + 1 < args.num_rollout
        ):
            rollout_data_future = rollout_manager.generate.remote(
                rollout_id + 1
            )

        if args.use_critic:
            actor_trains_this_step = (
                rollout_id >= args.num_critic_only_steps
            )
            value_refs = critic_model.async_train(
                rollout_id,
                rollout_data_curr_ref,
            )
            if actor_trains_this_step:
                ray.get(
                    actor_model.async_train(
                        rollout_id,
                        rollout_data_curr_ref,
                        external_data=value_refs,
                    )
                )
            else:
                ray.get(value_refs)
        else:
            ray.get(
                actor_model.async_train(
                    rollout_id,
                    rollout_data_curr_ref,
                )
            )

        last_completed_rollout_id = rollout_id
        if should_save_checkpoint:
            _save_checkpoint(
                args=args,
                rollout_manager=rollout_manager,
                actor_model=actor_model,
                critic_model=critic_model,
                rollout_id=rollout_id,
                save_model=True,
                force_sync=rollout_id == args.num_rollout - 1,
                dataset_state_staged=dataset_state_staged,
            )
            last_model_save_rollout_id = rollout_id

        if completing_preserved_partial:
            refill_preserved_validation_partial = False
            stopped_after_validation_boundary = True
            logger.info(
                "stopping intermediate chunk after rollout %d completed the "
                "optimizer batch preserved at validation boundary %d",
                rollout_id,
                stop_after_attempt,
            )
            break

        if (rollout_id + 1) % args.update_weights_interval == 0:
            # Do not wait for N+1. SGLang pause/flush makes an in-flight
            # request fail atomically; collectors discard that whole group,
            # refill it under the new weights, and enforce stale <= 1.
            _update_rollout_weights(actor_model, rollout_id)
            last_weights_rollout_id = rollout_id

        if (
            _should_run_eval(rollout_id, args)
            and not required_final_budget_incomplete
        ):
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_id += 1

    if (
        stop_after_attempt is not None
        and not stop_boundary_acknowledged
    ):
        raise RuntimeError(
            "training ended before the requested intermediate validation "
            f"boundary was acknowledged: target={stop_after_attempt}"
        )

    _require_final_instance_budget_exhaustion(
        args,
        rollout_manager,
        intermediate_chunk_stop=stopped_after_validation_boundary,
    )

    # Attempt-scheduled experiments validate once at the final cursor (for
    # example 1250), unless that cursor was already an exact interval boundary.
    if (
        _uses_instance_attempt_eval(args)
        and not stopped_after_validation_boundary
        and last_completed_rollout_id > last_weights_rollout_id
    ):
        _update_rollout_weights(
            actor_model,
            last_completed_rollout_id,
        )
        last_weights_rollout_id = last_completed_rollout_id
    final_eval_rollout_id = max(last_completed_rollout_id, 0)
    if not stopped_after_validation_boundary:
        _run_final_instance_validation_if_needed(
            args,
            rollout_manager,
            rollout_id=final_eval_rollout_id,
        )

    # A source budget can stop before args.num_rollout, so the ordinary
    # framework "final step" checkpoint condition is insufficient. Pair the
    # final model with the post-validation cursor. Avoid rewriting a 400+ GiB
    # model when this exact update was already saved synchronously; the
    # lightweight global-dataset state is still refreshed after validation.
    completed_new_update = (
        last_completed_rollout_id >= args.start_rollout_id
    )
    if completed_new_update:
        needs_model_save = (
            last_model_save_rollout_id != last_completed_rollout_id
        )
        if getattr(args, "async_save", False):
            # Re-enter save_model so Megatron finalizes an asynchronous save.
            needs_model_save = True
        needs_cursor_refresh = (
            stopped_by_instance_budget
            or _uses_instance_attempt_eval(args)
        )
        if needs_model_save or needs_cursor_refresh:
            _save_checkpoint(
                args=args,
                rollout_manager=rollout_manager,
                actor_model=actor_model,
                critic_model=critic_model,
                rollout_id=last_completed_rollout_id,
                save_model=needs_model_save,
                force_sync=True,
            )
    elif stopped_by_instance_budget:
        if not _refresh_existing_checkpoint_dataset_state(
            args,
            rollout_manager,
            checkpoint_id=args.start_rollout_id - 1,
        ):
            logger.warning(
                "source budget was exhausted without a new optimizer update, "
                "but no matching existing save checkpoint was available for "
                "the final data cursor"
            )
    elif stopped_after_validation_boundary:
        if not _refresh_existing_checkpoint_dataset_state(
            args,
            rollout_manager,
            checkpoint_id=args.start_rollout_id - 1,
        ):
            raise RuntimeError(
                "intermediate validation chunk stopped without a new "
                "optimizer update and could not pair its acknowledged source "
                "cursor with the existing checkpoint"
            )
    elif _uses_instance_attempt_eval(args):
        # This also covers a resumed run whose configured num_rollout has
        # already ended but whose final validation ACK was not yet persisted.
        _refresh_existing_checkpoint_dataset_state(
            args,
            rollout_manager,
            checkpoint_id=args.start_rollout_id - 1,
        )

    if stopped_after_validation_boundary:
        progress = ray.get(
            rollout_manager.get_train_instance_progress.remote()
        )
        attempted = int(progress.get("attempted_instances", 0))
        acknowledged = int(
            progress.get("last_validation_attempt", 0)
        )
        if (
            stop_after_attempt is None
            or attempted < stop_after_attempt
            or acknowledged < stop_after_attempt
        ):
            raise RuntimeError(
                "intermediate validation chunk did not finish at a "
                "checkpointable acknowledged cursor: "
                f"target={stop_after_attempt} attempted={attempted} "
                f"last_validation_attempt={acknowledged}"
            )

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


def _uses_instance_attempt_control(args) -> bool:
    return any(
        (
            getattr(args, "train_instance_budget", None) is not None,
            getattr(args, "eval_instance_interval", None) is not None,
            bool(
                getattr(
                    args,
                    "require_train_instance_budget_exhaustion",
                    False,
                )
            ),
            getattr(args, "stop_after_validation_attempt", None) is not None,
        )
    )


def train(args):
    if not _uses_instance_attempt_control(args):
        return _train_upstream(args)
    return _train_instance_attempt_control(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
