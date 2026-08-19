import abc
import copy
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import torch

from slime.utils.data import Dataset
from slime.utils.misc import load_function
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


TRAIN_INSTANCE_BUDGET_EXHAUSTED_KEY = (
    "__slime_train_instance_budget_exhausted__"
)
TRAIN_VALIDATION_BOUNDARY_KEY = "__slime_train_validation_boundary__"
ROLLOUT_CHECKPOINT_SCHEMA_VERSION = 2
ROLLOUT_COLLECTOR_STATE_METADATA_KEY = (
    "__rler_rollout_collector_state_v1__"
)


class TrainingInstanceBudgetExhausted(RuntimeError):
    """The configured source-instance traversal budget has been consumed."""

    def __init__(
        self,
        message: str | None = None,
        *,
        attempted_instances: int | None = None,
        budget: int | None = None,
    ):
        self.attempted_instances = (
            None if attempted_instances is None else int(attempted_instances)
        )
        self.budget = None if budget is None else int(budget)
        if message is None:
            message = (
                "training source-instance budget exhausted: "
                f"attempted={self.attempted_instances}, budget={self.budget}"
            )
        super().__init__(message)


class TrainingValidationBoundaryReached(RuntimeError):
    """Ask the rollout manager to schedule validation at an exact cursor."""

    def __init__(
        self,
        message: str | None = None,
        *,
        attempted_instances: int | None = None,
        boundary: int | None = None,
        preserved_partial: bool = False,
        preserved_group_count: int = 0,
        preserved_pending_count: int = 0,
    ):
        self.attempted_instances = (
            None if attempted_instances is None else int(attempted_instances)
        )
        self.boundary = None if boundary is None else int(boundary)
        # Custom collectors may already have accepted part of an optimizer
        # batch when the source cursor reaches a validation boundary.  The
        # exception is also the control-plane envelope used to tell
        # train_async whether it must retry this same rollout id after the
        # validation schedule instead of ending an intermediate chunk
        # immediately.
        self.preserved_partial = bool(preserved_partial)
        self.preserved_group_count = int(preserved_group_count)
        self.preserved_pending_count = int(preserved_pending_count)
        if self.preserved_group_count < 0:
            raise ValueError("preserved_group_count must be non-negative")
        if self.preserved_pending_count < 0:
            raise ValueError("preserved_pending_count must be non-negative")
        if self.preserved_partial != (
            self.preserved_group_count > 0
            or self.preserved_pending_count > 0
        ):
            raise ValueError(
                "preserved_partial must agree with preserved group/pending "
                "counts"
            )
        if message is None:
            message = (
                "training validation boundary reached: "
                f"attempted={self.attempted_instances}, "
                f"boundary={self.boundary}"
            )
        super().__init__(message)


class DataSource(abc.ABC):
    @abc.abstractmethod
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

    @abc.abstractmethod
    def add_samples(self, samples: list[list[Sample]]):
        """
        Add samples to the data source
        """

    @abc.abstractmethod
    def save(self, rollout_id, *, staged: bool = False):
        """
        Save the state of the data source
        """

    @abc.abstractmethod
    def load(self, rollout_id=None):
        """
        Load the state of the data source
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Length of the data source. May change when samples are added/fetched.
        """


# TODO may further refactor data-loading part later
class RolloutDataSource(DataSource):
    def __init__(self, args):
        self.args = args

        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        self.last_validation_scheduled_attempt = 0
        self.last_validation_attempt = 0
        # TODO remove this
        self.metadata = {}

        if args.rollout_global_dataset and args.prompt_data is not None:
            tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
            processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

            # TODO move (during the refactor)
            if (d := args.dump_details) is not None:
                tokenizer.save_pretrained(Path(d) / "tokenizer")
                if processor:
                    processor.save_pretrained(Path(d) / "processor")

            self.dataset = Dataset(
                args.prompt_data,
                tokenizer=tokenizer,
                processor=processor,
                max_length=args.rollout_max_prompt_len,
                prompt_key=args.input_key,
                multimodal_keys=args.multimodal_keys,
                label_key=args.label_key,
                metadata_key=args.metadata_key,
                tool_key=args.tool_key,
                apply_chat_template=args.apply_chat_template,
                apply_chat_template_kwargs=args.apply_chat_template_kwargs,
                seed=args.rollout_seed,
            )
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
        else:
            self.dataset = None

    def get_samples(self, num_samples):
        # TODO further improve code
        if self.dataset is not None:
            validation_interval = getattr(
                self.args, "eval_instance_interval", None
            )
            if validation_interval is not None:
                validation_interval = int(validation_interval)
                boundary = (
                    self.sample_group_index // validation_interval
                ) * validation_interval
                if (
                    boundary > 0
                    and boundary > self.last_validation_scheduled_attempt
                ):
                    raise TrainingValidationBoundaryReached(
                        attempted_instances=self.sample_group_index,
                        boundary=boundary,
                    )
                # get_samples must be atomic with respect to validation
                # boundaries. A batched caller at cursor 99 asking for two
                # groups must not consume attempts 100 and 101 in one draw,
                # because that would make exact attempt-100 validation
                # impossible. Current collectors request one source group,
                # so fail clearly instead of silently returning a short batch.
                next_boundary = boundary + validation_interval
                if (
                    next_boundary > self.last_validation_scheduled_attempt
                    and self.sample_group_index + num_samples > next_boundary
                ):
                    raise ValueError(
                        "get_samples request would cross an unscheduled "
                        "training validation boundary without drawing: "
                        f"attempted={self.sample_group_index} "
                        f"requested={num_samples} "
                        f"boundary={next_boundary}"
                    )
            budget = getattr(self.args, "train_instance_budget", None)
            if budget is not None:
                budget = int(budget)
                if self.sample_group_index + num_samples > budget:
                    raise TrainingInstanceBudgetExhausted(
                        attempted_instances=self.sample_group_index,
                        budget=budget,
                    )
            if self.sample_offset + num_samples <= len(self.dataset):
                prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_samples]
                self.sample_offset += num_samples
            else:
                prompt_samples = self.dataset.samples[self.sample_offset :]
                num_samples -= len(prompt_samples)
                self.epoch_id += 1
                if self.args.rollout_shuffle:
                    self.dataset.shuffle(self.epoch_id)
                prompt_samples += self.dataset.samples[:num_samples]
                self.sample_offset = num_samples
        else:
            prompt_samples = [Sample() for _ in range(num_samples)]

        samples = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def _checkpoint_path(
        self,
        rollout_id: int,
        *,
        staged: bool = False,
        load: bool = False,
    ) -> Path:
        root = self.args.load if load else self.args.save
        path = (
            Path(root)
            / "rollout"
            / f"global_dataset_state_dict_{int(rollout_id)}.pt"
        )
        return path.with_name(f"{path.name}.staged") if staged else path

    @staticmethod
    def _atomic_torch_save(state_dict: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            torch.save(state_dict, temporary)
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def save(self, rollout_id, *, staged: bool = False):
        if not self.args.rollout_global_dataset:
            return

        state_dict = {
            "checkpoint_schema_version": ROLLOUT_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_rollout_id": int(rollout_id),
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "last_validation_scheduled_attempt": (
                self.last_validation_scheduled_attempt
            ),
            "last_validation_attempt": self.last_validation_attempt,
            "metadata": self.metadata,
        }
        path = self._checkpoint_path(rollout_id, staged=staged)
        self._atomic_torch_save(state_dict, path)
        return str(path)

    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset:
            return

        if self.args.load is None:
            return

        path = self._checkpoint_path(rollout_id, load=True)
        if not path.exists():
            logger.info(f"Checkpoint {path} does not exist.")
            return

        logger.info(f"load metadata from {path}")
        state_dict = torch.load(path, weights_only=False)
        try:
            checkpoint_schema_version = int(
                state_dict.get("checkpoint_schema_version", -1)
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "rollout data-source checkpoint has an invalid schema "
                f"version: {state_dict.get('checkpoint_schema_version')!r}"
            ) from exc
        if checkpoint_schema_version != ROLLOUT_CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError(
                "unsupported rollout data-source checkpoint schema: "
                f"saved={checkpoint_schema_version} "
                f"required={ROLLOUT_CHECKPOINT_SCHEMA_VERSION}"
            )
        saved_rollout_id = state_dict.get("checkpoint_rollout_id")
        if (
            saved_rollout_id is not None
            and int(saved_rollout_id) != int(rollout_id)
        ):
            raise RuntimeError(
                "rollout data-source checkpoint id mismatch: "
                f"requested={rollout_id} saved={saved_rollout_id}"
            )
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.last_validation_scheduled_attempt = state_dict.get(
            "last_validation_scheduled_attempt",
            state_dict.get("last_validation_attempt", 0),
        )
        self.last_validation_attempt = state_dict.get(
            "last_validation_attempt", 0
        )
        self.metadata = state_dict.get("metadata", {})

        if self.args.rollout_global_dataset and self.args.rollout_shuffle and self.dataset is not None:
            self.dataset.shuffle(self.epoch_id)
        logger.info(
            "restored rollout data-source checkpoint id=%d schema=%d "
            "epoch_id=%d sample_offset=%d sample_group_index=%d "
            "sample_index=%d validation_completed=%d "
            "validation_scheduled=%d metadata_keys=%s",
            int(rollout_id),
            checkpoint_schema_version,
            int(self.epoch_id),
            int(self.sample_offset),
            int(self.sample_group_index),
            int(self.sample_index),
            int(self.last_validation_attempt),
            int(self.last_validation_scheduled_attempt),
            sorted(self.metadata),
        )

    def __len__(self) -> int:
        if self.dataset is None:
            return 0
        return len(self.dataset)

    def training_progress(self) -> dict[str, Any]:
        dataset_size = len(self.dataset) if self.dataset is not None else 0
        budget = getattr(self.args, "train_instance_budget", None)
        interval = getattr(self.args, "eval_instance_interval", None)
        return {
            "attempted_instances": int(self.sample_group_index),
            "instance_budget": None if budget is None else int(budget),
            "dataset_size": int(dataset_size),
            "epoch": (
                float(self.sample_group_index) / dataset_size
                if dataset_size
                else 0.0
            ),
            "last_validation_attempt": int(
                self.last_validation_attempt
            ),
            "last_validation_scheduled_attempt": int(
                self.last_validation_scheduled_attempt
            ),
            "eval_instance_interval": (
                None if interval is None else int(interval)
            ),
        }

    def mark_validation_scheduled(
        self,
        attempted_instances: int,
    ) -> None:
        attempted_instances = int(attempted_instances)
        if attempted_instances > self.sample_group_index:
            raise ValueError(
                "cannot schedule validation beyond the consumed source "
                f"cursor: {attempted_instances}>{self.sample_group_index}"
            )
        if attempted_instances < self.last_validation_attempt:
            raise ValueError(
                "cannot schedule validation before the completed validation "
                f"cursor: {attempted_instances}<"
                f"{self.last_validation_attempt}"
            )
        self.last_validation_scheduled_attempt = max(
            self.last_validation_scheduled_attempt,
            attempted_instances,
        )

    def acknowledge_validation(self, attempted_instances: int) -> None:
        attempted_instances = int(attempted_instances)
        if attempted_instances > self.sample_group_index:
            raise ValueError(
                "cannot acknowledge validation beyond the consumed source "
                f"cursor: {attempted_instances}>{self.sample_group_index}"
            )
        if attempted_instances > self.last_validation_scheduled_attempt:
            raise ValueError(
                "cannot acknowledge an unscheduled validation: "
                f"{attempted_instances}>"
                f"{self.last_validation_scheduled_attempt}"
            )
        self.last_validation_attempt = max(
            self.last_validation_attempt,
            attempted_instances,
        )


class RolloutDataSourceWithBuffer(RolloutDataSource):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = []
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

        samples += super().get_samples(num_samples=num_samples)
        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        if len(self.buffer) == 0 or num_samples == 0:
            return []

        samples = self.buffer_filter(self.args, None, self.buffer, num_samples)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        """
        Add a sample group to buffer.
        """
        if not samples:
            return
        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
        assert isinstance(samples[0], list), f"the elements of samples must be list, got {type(samples[0])}"
        for i in range(0, len(samples)):
            assert (
                len(samples[i]) == self.args.n_samples_per_prompt
            ), f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(samples[i])} != {self.args.n_samples_per_prompt}"
            group = samples[i]  # type: ignore
            self.buffer.append(group)

    # TODO remove
    def update_metadata(self, metadata: dict):
        self.metadata.update(metadata)

    # TODO remove
    def get_metadata(self):
        return self.metadata

    def get_buffer_length(self):
        return len(self.buffer)


def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
