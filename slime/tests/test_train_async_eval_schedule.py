from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_INSTANCE_BUDGET_EXHAUSTED_KEY = (
    "__slime_train_instance_budget_exhausted__"
)
TRAIN_VALIDATION_BOUNDARY_KEY = "__slime_train_validation_boundary__"


def _module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _load_train_async(monkeypatch):
    def should_run_periodic_action(
        rollout_id,
        interval,
        num_rollout_per_epoch=None,
        num_rollout=None,
    ):
        if interval is None:
            return False
        if num_rollout is not None and rollout_id == num_rollout - 1:
            return True
        step = rollout_id + 1
        return (step % interval == 0) or (
            num_rollout_per_epoch is not None
            and step % num_rollout_per_epoch == 0
        )

    stubs = {
        "ray": _module("ray"),
        "slime": _module("slime"),
        "slime.ray": _module("slime.ray"),
        "slime.ray.placement_group": _module(
            "slime.ray.placement_group",
            create_placement_groups=None,
            create_rollout_manager=None,
            create_training_models=None,
        ),
        "slime.rollout": _module("slime.rollout"),
        "slime.rollout.data_source": _module(
            "slime.rollout.data_source",
            TRAIN_INSTANCE_BUDGET_EXHAUSTED_KEY=(
                TRAIN_INSTANCE_BUDGET_EXHAUSTED_KEY
            ),
            TRAIN_VALIDATION_BOUNDARY_KEY=(
                TRAIN_VALIDATION_BOUNDARY_KEY
            ),
        ),
        "slime.utils": _module("slime.utils"),
        "slime.utils.arguments": _module("slime.utils.arguments", parse_args=None),
        "slime.utils.logging_utils": _module(
            "slime.utils.logging_utils",
            configure_logger=None,
            finish_tracking=None,
            init_tracking=None,
        ),
        "slime.utils.misc": _module(
            "slime.utils.misc",
            should_run_periodic_action=should_run_periodic_action,
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    path = REPO_ROOT / "slime/train_async.py"
    spec = importlib.util.spec_from_file_location("_test_train_async", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_rollout_manager_eval(**globals_):
    """Compile the real method without importing GPU runtime dependencies."""
    path = REPO_ROOT / "slime/slime/ray/rollout.py"
    tree = ast.parse(path.read_text())
    manager = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RolloutManager"
    )
    eval_method = next(
        node
        for node in manager.body
        if isinstance(node, ast.FunctionDef) and node.name == "eval"
    )
    namespace = dict(globals_)
    exec(
        compile(
            ast.Module(body=[eval_method], type_ignores=[]),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace["eval"]


def _load_rollout_manager_methods(method_names, **globals_):
    """Compile selected real manager methods without GPU/Ray imports."""
    path = REPO_ROOT / "slime/slime/ray/rollout.py"
    tree = ast.parse(path.read_text())
    manager = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RolloutManager"
    )
    selected = [
        node
        for node in manager.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    assert {node.name for node in selected} == set(method_names)
    namespace = dict(globals_)
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]),
            str(path),
            "exec",
        ),
        namespace,
    )
    return {name: namespace[name] for name in method_names}


def test_attempt_validation_schedule_is_nonblocking_and_drain_is_explicit():
    events = []

    class Ref:
        def __init__(self, value, *, ready=False):
            self.value = value
            self.ready = ready

    class FakeRay:
        @staticmethod
        def wait(refs, *, num_returns, timeout):
            assert len(refs) == num_returns == 1
            assert timeout == 0
            events.append(("wait", refs[0].ready))
            return (
                ([refs[0]], [])
                if refs[0].ready
                else ([], list(refs))
            )

        @staticmethod
        def get(ref):
            events.append(("get", ref.value["eval/train_instance_attempt"]))
            return ref.value

    class Logger:
        @staticmethod
        def info(message, *args):
            events.append(("log", message % args))

    class EvalRemote:
        def __init__(self):
            self.refs = {}

        def remote(
            self,
            rollout_id,
            attempted_instances,
            policy_version,
        ):
            events.append(
                (
                    "schedule",
                    rollout_id,
                    attempted_instances,
                    policy_version,
                )
            )
            ref = Ref(
                {
                    "eval/attempted": 50,
                    "eval/completed": 50,
                    "eval/incomplete": 0,
                    "eval/train_instance_attempt": attempted_instances,
                }
            )
            self.refs[attempted_instances] = ref
            return ref

    class ValidationManager:
        def __init__(self):
            self.eval = EvalRemote()

    class DataSource:
        def __init__(self):
            self.completed = 0
            self.scheduled = 0
            self.validation_rollout_ids = {}
            self.validation_policy_versions = {}

        def mark_validation_scheduled(
            self,
            attempted_instances,
            rollout_id,
            policy_version,
        ):
            events.append(
                (
                    "mark_scheduled",
                    attempted_instances,
                    rollout_id,
                    policy_version,
                )
            )
            self.scheduled = attempted_instances
            self.validation_rollout_ids[attempted_instances] = rollout_id
            self.validation_policy_versions[attempted_instances] = (
                policy_version
            )

        def acknowledge_validation(self, attempted_instances):
            events.append(("ack", attempted_instances))
            assert attempted_instances <= self.scheduled
            self.completed = attempted_instances

        def training_progress(self):
            return {
                "attempted_instances": 125,
                "last_validation_attempt": self.completed,
                "last_validation_scheduled_attempt": self.scheduled,
                "validation_rollout_ids": dict(
                    self.validation_rollout_ids
                ),
                "validation_policy_versions": dict(
                    self.validation_policy_versions
                ),
            }

    methods = _load_rollout_manager_methods(
        {
            "_schedule_train_validation",
            "_complete_train_validation",
            "_poll_train_validations",
            "drain_train_validations",
        },
        Any=object,
        logger=Logger,
        ray=FakeRay,
        committed_policy_version=lambda **_kwargs: "checkpoint-0000003",
        _require_complete_train_validation_metrics=lambda metrics, **_: metrics,
    )

    class Manager:
        _schedule_train_validation = methods[
            "_schedule_train_validation"
        ]
        _complete_train_validation = methods[
            "_complete_train_validation"
        ]
        _poll_train_validations = methods["_poll_train_validations"]
        drain_train_validations = methods["drain_train_validations"]

        def __init__(self):
            self.data_source = DataSource()
            self._train_validation_manager = ValidationManager()
            self._train_validation_refs = {}
            self._train_validation_rollout_ids = {}
            self._train_validation_policy_versions = {}

        def get_train_instance_progress(self):
            return self.data_source.training_progress()

    manager = Manager()
    manager._schedule_train_validation(
        attempted_instances=100,
        rollout_id=3,
    )
    assert manager.data_source.scheduled == 100
    assert manager.data_source.completed == 0
    assert not any(event[0] == "get" for event in events)

    manager._poll_train_validations()
    assert manager.data_source.completed == 0
    assert not any(event[0] == "get" for event in events)

    manager._train_validation_manager.eval.refs[100].ready = True
    manager._poll_train_validations()
    assert manager.data_source.completed == 100
    assert 100 not in manager._train_validation_refs

    # The terminal 125 cursor is the only deliberate wait.
    manager._train_validation_manager.eval.refs.clear()
    original_remote = manager._train_validation_manager.eval.remote

    def ready_remote(
        rollout_id,
        attempted_instances,
        policy_version,
    ):
        ref = original_remote(
            rollout_id,
            attempted_instances,
            policy_version,
        )
        ref.ready = True
        return ref

    manager._train_validation_manager.eval.remote = ready_remote
    progress = manager.drain_train_validations(7, 125)
    assert progress["last_validation_attempt"] == 125
    assert ("get", 125) in events


def test_generation_boundary_validation_uses_upcoming_policy_version():
    methods = _load_rollout_manager_methods(
        {"_validation_policy_version_for_generation"},
        policy_version_state_path=lambda: object(),
        policy_version_for_checkpoint=(
            lambda checkpoint_id: f"checkpoint-{checkpoint_id:07d}"
        ),
        committed_policy_version=(
            lambda **_kwargs: "checkpoint-0000003"
        ),
    )

    class Manager:
        _validation_policy_version_for_generation = methods[
            "_validation_policy_version_for_generation"
        ]

        def __init__(self, update_weights_interval):
            self.args = SimpleNamespace(
                update_weights_interval=update_weights_interval
            )

    assert (
        Manager(1)._validation_policy_version_for_generation(6)
        == "checkpoint-0000005"
    )
    assert (
        Manager(2)._validation_policy_version_for_generation(6)
        == "checkpoint-0000005"
    )
    assert (
        Manager(2)._validation_policy_version_for_generation(5)
        == "checkpoint-0000003"
    )


def test_rollout_manager_resume_uses_persisted_validation_rollout_id():
    events = []

    class EvalRemote:
        def remote(
            self,
            rollout_id,
            attempted_instances,
            policy_version,
        ):
            events.append(
                (
                    "schedule",
                    rollout_id,
                    attempted_instances,
                    policy_version,
                )
            )
            return object()

    class DataSource:
        def __init__(self):
            self.mapping = {100: 5}
            self.versions = {100: "checkpoint-0000004"}

        def load(self, rollout_id):
            events.append(("load", rollout_id))

        def training_progress(self):
            return {
                "attempted_instances": 156,
                "last_validation_attempt": 0,
                "last_validation_scheduled_attempt": 100,
                "validation_rollout_ids": dict(self.mapping),
                "validation_policy_versions": dict(self.versions),
                "eval_instance_interval": 100,
            }

        def mark_validation_scheduled(
            self,
            attempted_instances,
            rollout_id,
            policy_version,
        ):
            assert self.mapping[attempted_instances] == rollout_id
            assert self.versions[attempted_instances] == policy_version

    methods = _load_rollout_manager_methods(
        {"load", "_schedule_train_validation"},
        policy_version_state_path=lambda: object(),
        logger=SimpleNamespace(info=lambda *_args: None),
    )

    class Manager:
        load = methods["load"]
        _schedule_train_validation = methods[
            "_schedule_train_validation"
        ]

        def __init__(self):
            self.data_source = DataSource()
            self._train_validation_manager = SimpleNamespace(
                eval=EvalRemote()
            )
            self._train_validation_refs = {}
            self._train_validation_rollout_ids = {}
            self._train_validation_policy_versions = {}

        def get_train_instance_progress(self):
            return self.data_source.training_progress()

    manager = Manager()
    manager.load(7)

    assert events == [
        ("load", 7),
        ("schedule", 5, 100, "checkpoint-0000004"),
    ]


def test_rollout_manager_checkpoint_embeds_and_restores_collector_state():
    restored = []
    methods = _load_rollout_manager_methods(
        {"save", "load"},
        ROLLOUT_COLLECTOR_STATE_METADATA_KEY=(
            "__rler_rollout_collector_state_v1__"
        ),
    )

    class DataSource:
        def __init__(self):
            self.metadata = {}
            self.saved = []

        def save(self, rollout_id, *, staged=False):
            self.saved.append(
                (rollout_id, staged, dict(self.metadata))
            )

        def load(self, rollout_id):
            return None

    class Manager:
        save = methods["save"]
        load = methods["load"]

        def __init__(self):
            self.data_source = DataSource()
            self._train_validation_manager = None
            self._collector_checkpoint_hook = lambda name: {
                "checkpoint_state_dict": lambda rollout_id: {
                    "checkpoint_rollout_id": rollout_id,
                    "buffer": ["carry"],
                    "pending_tasks": ["instance-17"],
                },
                "load_checkpoint_state_dict": (
                    lambda state, rollout_id: restored.append(
                        (state, rollout_id)
                    )
                ),
            }[name]
            self._requires_collector_checkpoint_state = lambda: True

    manager = Manager()
    manager.save(3, staged=True)
    key = "__rler_rollout_collector_state_v1__"
    assert manager.data_source.saved[0][0:2] == (3, True)
    assert manager.data_source.saved[0][2][key]["buffer"] == ["carry"]

    manager.load(3)
    assert restored == [
        (manager.data_source.metadata[key], 3)
    ]


def test_rollout_manager_allows_fresh_negative_cursor_but_rejects_old_online_state():
    methods = _load_rollout_manager_methods({"load"})

    class DataSource:
        metadata = {}

        def load(self, rollout_id):
            return None

    class Manager:
        load = methods["load"]
        data_source = DataSource()
        _train_validation_manager = None
        _collector_checkpoint_hook = lambda self, name: lambda *_args: None
        _requires_collector_checkpoint_state = lambda self: True

    manager = Manager()
    manager.load(-1)
    with pytest.raises(
        RuntimeError,
        match="exact training position",
    ):
        manager.load(3)


def test_rollout_manager_resume_requires_persisted_validation_mapping():
    class DataSource:
        def load(self, _rollout_id):
            pass

        @staticmethod
        def training_progress():
            return {
                "attempted_instances": 156,
                "last_validation_attempt": 0,
                "last_validation_scheduled_attempt": 100,
                "validation_rollout_ids": {},
                "eval_instance_interval": 100,
            }

    methods = _load_rollout_manager_methods(
        {"load", "_schedule_train_validation"},
        policy_version_state_path=lambda: None,
        logger=SimpleNamespace(info=lambda *_args: None),
    )

    class Manager:
        load = methods["load"]
        _schedule_train_validation = methods[
            "_schedule_train_validation"
        ]

        def __init__(self):
            self.data_source = DataSource()
            self._train_validation_manager = object()
            self._train_validation_refs = {}
            self._train_validation_rollout_ids = {}
            self._train_validation_policy_versions = {}

        def get_train_instance_progress(self):
            return self.data_source.training_progress()

    with pytest.raises(
        RuntimeError,
        match="outstanding asynchronous validation.*without its exact policy",
    ):
        Manager().load(7)


def test_rollout_manager_eval_legacy_fast_path_does_not_touch_cursor():
    events = []
    args = SimpleNamespace(
        debug_train_only=False,
        eval_instance_interval=None,
    )
    result = SimpleNamespace(data={"legacy": "data"}, metrics={"m": 1})

    def call_rollout_fn(*call_args, **call_kwargs):
        events.append(("call_rollout_fn", call_args, call_kwargs))
        return result

    def log_eval(*call_args):
        events.append(("log", call_args))

    eval_method = _load_rollout_manager_eval(
        call_rollout_fn=call_rollout_fn,
        _log_eval_rollout_data=log_eval,
    )

    class Manager:
        eval_generate_rollout = "eval_fn"
        data_source = "data_source"

        def __init__(self):
            self.args = args

        def health_monitoring_resume(self):
            events.append(("health_resume",))

        def get_train_instance_progress(self):
            raise AssertionError("legacy eval must not read the attempt cursor")

        def _save_debug_rollout_data(
            self,
            data,
            *,
            rollout_id,
            evaluation,
        ):
            events.append(
                ("save_debug", data, rollout_id, evaluation)
            )

    assert eval_method(Manager(), 7) is None
    assert not hasattr(args, "eval_instance_attempt")
    assert events == [
        ("health_resume",),
        (
            "call_rollout_fn",
            ("eval_fn", args, 7, "data_source"),
            {"evaluation": True},
        ),
        ("save_debug", {"legacy": "data"}, 7, True),
        ("log", (7, args, {"legacy": "data"}, {"m": 1})),
    ]


def test_rollout_manager_eval_attempt_path_uses_exact_cursor():
    events = []
    args = SimpleNamespace(
        debug_train_only=False,
        eval_instance_interval=100,
    )
    result = SimpleNamespace(
        data={"attempt": "data"},
        metrics={"eval/completed": 50},
    )

    def call_rollout_fn(*call_args, **call_kwargs):
        events.append(("call_rollout_fn", call_args, call_kwargs))
        return result

    def log_eval(rollout_id, logged_args, data, metrics):
        events.append(("log", rollout_id, data, dict(metrics)))
        return metrics

    eval_method = _load_rollout_manager_eval(
        call_rollout_fn=call_rollout_fn,
        _log_eval_rollout_data=log_eval,
    )

    class Manager:
        eval_generate_rollout = "eval_fn"
        data_source = "data_source"

        def __init__(self):
            self.args = args

        def health_monitoring_resume(self):
            events.append(("health_resume",))

        def get_train_instance_progress(self):
            events.append(("progress",))
            return {"attempted_instances": 100}

        def _save_debug_rollout_data(
            self,
            data,
            *,
            rollout_id,
            evaluation,
        ):
            events.append(
                ("save_debug", data, rollout_id, evaluation)
            )

    returned = eval_method(Manager(), 9, 100)
    assert args.eval_instance_attempt == 100
    assert returned == {
        "eval/completed": 50,
        "eval/train_instance_attempt": 100,
    }
    assert events == [
        ("health_resume",),
        ("progress",),
        (
            "call_rollout_fn",
            ("eval_fn", args, 9, "data_source"),
            {"evaluation": True},
        ),
        ("save_debug", {"attempt": "data"}, 9, True),
        (
            "log",
            9,
            {"attempt": "data"},
            {
                "eval/completed": 50,
                "eval/train_instance_attempt": 100,
            },
        ),
    ]


def test_async_eval_runs_every_100_updates_and_at_final(monkeypatch):
    module = _load_train_async(monkeypatch)
    args = SimpleNamespace(eval_interval=100, num_rollout=1250)
    actual = [
        rollout_id + 1
        for rollout_id in range(args.num_rollout)
        if module._should_run_eval(rollout_id, args)
    ]
    assert actual == [*range(100, 1201, 100), 1250]


def test_async_eval_ignores_epoch_boundaries(monkeypatch):
    module = _load_train_async(monkeypatch)
    args = SimpleNamespace(eval_interval=100, num_rollout=1250)
    assert not module._should_run_eval(249, args)
    # Step 500 still evaluates because it is on the fixed 100-update cadence.
    assert module._should_run_eval(499, args)
    assert not module._should_run_eval(749, args)
    assert module._should_run_eval(1249, args)


def test_async_eval_remains_disabled_without_interval(monkeypatch):
    module = _load_train_async(monkeypatch)
    args = SimpleNamespace(eval_interval=None, num_rollout=1250)
    assert not module._should_run_eval(1249, args)


def test_instance_attempt_schedule_disables_update_schedule(monkeypatch):
    module = _load_train_async(monkeypatch)
    args = SimpleNamespace(
        eval_interval=1,
        eval_instance_interval=100,
        num_rollout=1250,
    )
    assert not module._should_run_eval(0, args)
    assert not module._should_run_eval(1249, args)


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("train_instance_budget", 1250),
        ("eval_instance_interval", 100),
        ("require_train_instance_budget_exhaustion", True),
        ("stop_after_validation_attempt", 100),
    ],
)
def test_each_source_attempt_contract_flag_selects_control_loop(
    monkeypatch,
    flag,
    value,
):
    module = _load_train_async(monkeypatch)
    args = SimpleNamespace(**{flag: value})
    assert module._uses_instance_attempt_control(args)


def test_unrelated_checkpoint_retention_does_not_select_control_loop(
    monkeypatch,
):
    module = _load_train_async(monkeypatch)
    args = SimpleNamespace(checkpoint_retain_latest=2)
    assert not module._uses_instance_attempt_control(args)


def test_legacy_train_matches_upstream_event_trace_and_epoch_semantics(
    monkeypatch,
):
    module = _load_train_async(monkeypatch)
    events = []

    class Ref:
        def __init__(self, label, value=None):
            self.label = label
            self.value = value

    class RemoteMethod:
        def __init__(self, name, result=None):
            self.name = name
            self.result = result

        def remote(self, *args, **kwargs):
            events.append(("remote", self.name, args, kwargs))
            value = (
                self.result(*args, **kwargs)
                if callable(self.result)
                else self.result
            )
            return Ref(self.name, value)

    class FakeRay:
        @staticmethod
        def get(ref):
            if isinstance(ref, list):
                return [FakeRay.get(item) for item in ref]
            assert isinstance(ref, Ref)
            events.append(("get", ref.label))
            return ref.value

    class RolloutManager:
        def __init__(self):
            self.generate = RemoteMethod(
                "generate",
                result=lambda rollout_id: f"rollout-{rollout_id}",
            )
            self.save = RemoteMethod("dataset_save")
            self.eval = RemoteMethod("eval")
            self.check_weights = RemoteMethod("check_weights")
            self.dispose = RemoteMethod("dispose")

    class Actor:
        def update_weights(self):
            events.append(("update_weights",))

        def async_train(self, rollout_id, rollout_data, external_data=None):
            events.append(
                (
                    "actor_train",
                    rollout_id,
                    rollout_data,
                    external_data,
                )
            )
            return Ref(f"actor_train_{rollout_id}")

        def save_model(self, rollout_id, force_sync=False):
            events.append(("actor_save", rollout_id, force_sync))

    rollout_manager = RolloutManager()
    actor = Actor()
    module.ray = FakeRay

    def periodic(
        rollout_id,
        interval,
        num_rollout_per_epoch=None,
        num_rollout=None,
    ):
        result = (
            interval is not None
            and (
                (
                    num_rollout is not None
                    and rollout_id == num_rollout - 1
                )
                or (rollout_id + 1) % interval == 0
                or (
                    num_rollout_per_epoch is not None
                    and (rollout_id + 1) % num_rollout_per_epoch == 0
                )
            )
        )
        events.append(
            (
                "periodic",
                rollout_id,
                interval,
                num_rollout_per_epoch,
                num_rollout,
                result,
            )
        )
        return result

    module.should_run_periodic_action = periodic
    module.configure_logger = lambda: events.append(("configure",))
    module.init_tracking = lambda args: events.append(("init_tracking",))
    module.finish_tracking = lambda args: events.append(
        ("finish_tracking",)
    )

    def create_pgs(args):
        events.append(("create_placement_groups",))
        return {"rollout": object()}

    module.create_placement_groups = create_pgs
    module.create_rollout_manager = (
        lambda args, pg: (
            events.append(("create_rollout_manager",))
            or (rollout_manager, 2)
        )
    )
    module.create_training_models = (
        lambda args, pgs, manager: (
            events.append(("create_training_models",))
            or (actor, None)
        )
    )
    args = SimpleNamespace(
        colocate=False,
        check_weight_update_equal=False,
        start_rollout_id=0,
        num_rollout=3,
        use_critic=False,
        save_interval=99,
        rollout_global_dataset=True,
        update_weights_interval=99,
        eval_interval=99,
    )

    module.train(args)

    assert events == [
        ("configure",),
        ("create_placement_groups",),
        ("init_tracking",),
        ("create_rollout_manager",),
        ("create_training_models",),
        ("update_weights",),
        ("remote", "generate", (0,), {}),
        ("get", "generate"),
        ("remote", "generate", (1,), {}),
        ("actor_train", 0, "rollout-0", None),
        ("get", "actor_train_0"),
        ("periodic", 0, 99, 2, 3, False),
        ("periodic", 0, 99, 2, None, False),
        ("get", "generate"),
        ("remote", "generate", (2,), {}),
        ("actor_train", 1, "rollout-1", None),
        ("get", "actor_train_1"),
        ("periodic", 1, 99, 2, 3, True),
        ("actor_save", 1, False),
        ("remote", "dataset_save", (1,), {}),
        ("get", "dataset_save"),
        ("periodic", 1, 99, 2, None, True),
        ("remote", "eval", (1,), {}),
        ("get", "eval"),
        ("get", "generate"),
        ("actor_train", 2, "rollout-2", None),
        ("get", "actor_train_2"),
        ("periodic", 2, 99, 2, 3, True),
        ("actor_save", 2, True),
        ("remote", "dataset_save", (2,), {}),
        ("get", "dataset_save"),
        ("periodic", 2, 99, 2, None, False),
        ("remote", "dispose", (), {}),
        ("get", "dispose"),
        ("finish_tracking",),
    ]


def test_attempt_path_checkpoint_restores_exact_cursor_without_skipping(
    monkeypatch,
):
    module = _load_train_async(monkeypatch)
    staged_snapshots = {}
    committed_snapshots = {}
    committed_prompts = []

    class Ref:
        def __init__(self, value=None, error=None):
            self.value = value
            self.error = error

    class FakeRay:
        @staticmethod
        def get(ref):
            if isinstance(ref, list):
                return [FakeRay.get(item) for item in ref]
            assert isinstance(ref, Ref)
            if ref.error is not None:
                raise ref.error
            return ref.value

    class GenerateMethod:
        def __init__(self, manager):
            self.manager = manager

        def remote(self, rollout_id):
            prompt_id = self.manager.cursor
            self.manager.cursor += 1
            return Ref((rollout_id, prompt_id))

    class NoopMethod:
        def remote(self, *args, **kwargs):
            return Ref()

    class RolloutManager:
        def __init__(self, cursor):
            self.cursor = cursor
            self.generate = GenerateMethod(self)
            self.save = NoopMethod()
            self.eval = NoopMethod()
            self.check_weights = NoopMethod()
            self.dispose = NoopMethod()

    class Actor:
        def __init__(self, fail_rollout_id=None):
            self.fail_rollout_id = fail_rollout_id
            self.pending = {}

        def update_weights(self):
            pass

        def async_train(self, rollout_id, rollout_data, external_data=None):
            if rollout_id == self.fail_rollout_id:
                return Ref(error=RuntimeError("simulated requeue"))
            self.pending[rollout_id] = rollout_data[1]
            return Ref()

        def save_model(self, rollout_id, force_sync=False):
            committed_prompts.append(self.pending.pop(rollout_id))

    def run(run_id, start_rollout_id, cursor, fail_rollout_id=None):
        manager = RolloutManager(cursor)
        actor = Actor(fail_rollout_id)
        module.ray = FakeRay
        module.configure_logger = lambda: None
        module.init_tracking = lambda args: None
        module.finish_tracking = lambda args: None
        module.create_placement_groups = lambda args: {"rollout": object()}
        module.create_rollout_manager = (
            lambda args, pg: (manager, args.num_rollout)
        )
        module.create_training_models = (
            lambda args, pgs, rollout_manager: (actor, None)
        )
        module._stage_checkpoint_dataset_state = (
            lambda args, rollout_manager, *, rollout_id: (
                staged_snapshots.__setitem__(
                    (run_id, rollout_id),
                    rollout_manager.cursor,
                )
                or True
            )
        )
        module._commit_checkpoint_dataset_state = (
            lambda args, *, rollout_id: committed_snapshots.__setitem__(
                (run_id, rollout_id),
                staged_snapshots[(run_id, rollout_id)],
            )
        )
        args = SimpleNamespace(
            colocate=False,
            check_weight_update_equal=False,
            start_rollout_id=start_rollout_id,
            num_rollout=3,
            train_instance_budget=3,
            use_critic=False,
            save_interval=1,
            rollout_global_dataset=True,
            update_weights_interval=1,
            eval_interval=None,
        )
        module.train(args)

    with pytest.raises(RuntimeError, match="simulated requeue"):
        run(
            run_id=0,
            start_rollout_id=0,
            cursor=0,
            fail_rollout_id=1,
        )

    assert committed_prompts == [0]
    # The checkpoint-0 snapshot is taken after rollout 0 has been selected but
    # before rollout 1 advances the source.  The staged checkpoint for failed
    # update 1 is never published.
    assert staged_snapshots[(0, 0)] == 1
    assert committed_snapshots == {(0, 0): 1}

    run(
        run_id=1,
        start_rollout_id=1,
        cursor=committed_snapshots[(0, 0)],
    )
    assert committed_prompts == [0, 1, 2]
    assert committed_snapshots[(1, 1)] == 2
    assert committed_snapshots[(1, 2)] == 3


def test_attempt_boundary_retries_same_update_and_budget_stops_normally(
    monkeypatch,
):
    module = _load_train_async(monkeypatch)
    events = []

    class Ref:
        def __init__(self, value=None):
            self.value = value

    class FakeRay:
        @staticmethod
        def get(ref):
            if isinstance(ref, list):
                return [FakeRay.get(item) for item in ref]
            assert isinstance(ref, Ref)
            return ref.value

    class RemoteMethod:
        def __init__(self, callback):
            self.callback = callback

        def remote(self, *args, **kwargs):
            return Ref(self.callback(*args, **kwargs))

    class RolloutManager:
        def __init__(self):
            self.attempted = 0
            self.last_validation = 0
            self.generate_calls = []
            self.generate_1_calls = 0
            self.generate = RemoteMethod(self._generate)
            self.eval = RemoteMethod(self._eval)
            self.acknowledge_train_validation = RemoteMethod(
                self._acknowledge
            )
            self.drain_train_validations = RemoteMethod(
                self._drain_train_validations
            )
            self.get_train_instance_progress = RemoteMethod(
                self._progress
            )
            self.save = RemoteMethod(self._save)
            self.check_weights = RemoteMethod(lambda *args, **kwargs: None)
            self.dispose = RemoteMethod(lambda: events.append(("dispose",)))

        def _generate(self, rollout_id):
            self.generate_calls.append(rollout_id)
            events.append(("generate", rollout_id))
            if rollout_id == 0:
                self.attempted = 100
                return "rollout-0"
            if rollout_id == 1:
                self.generate_1_calls += 1
                if self.generate_1_calls == 1:
                    return {
                        TRAIN_VALIDATION_BOUNDARY_KEY: True,
                        "attempted_instances": 100,
                        "boundary": 100,
                    }
                self.attempted = 125
                return "rollout-1"
            assert rollout_id == 2
            return {
                TRAIN_INSTANCE_BUDGET_EXHAUSTED_KEY: True,
                "attempted_instances": 125,
                "budget": 125,
            }

        def _eval(self, rollout_id, attempted):
            assert attempted == self.attempted
            events.append(("eval", rollout_id, attempted))
            return {
                "eval/attempted": 50,
                "eval/completed": 50,
                "eval/incomplete": 0,
            }

        def _acknowledge(self, attempted):
            assert attempted == self.attempted
            self.last_validation = attempted
            events.append(("ack", attempted))
            return self._progress()

        def _drain_train_validations(self, rollout_id, attempted):
            self._eval(rollout_id, attempted)
            return self._acknowledge(attempted)

        def _progress(self):
            return {
                "attempted_instances": self.attempted,
                "last_validation_attempt": self.last_validation,
                "instance_budget": 125,
                "eval_instance_interval": 100,
            }

        def _save(self, rollout_id):
            events.append(
                (
                    "dataset_save",
                    rollout_id,
                    self.attempted,
                    self.last_validation,
                )
            )

    class Actor:
        def update_weights(self):
            events.append(("update_weights",))

        def async_train(self, rollout_id, rollout_data, external_data=None):
            events.append(("train", rollout_id, rollout_data))
            return Ref()

        def save_model(self, rollout_id, force_sync=False):
            events.append(("model_save", rollout_id, force_sync))

    manager = RolloutManager()
    actor = Actor()
    module.ray = FakeRay
    module.configure_logger = lambda: None
    module.init_tracking = lambda args: None
    module.finish_tracking = lambda args: None
    module.create_placement_groups = lambda args: {"rollout": object()}
    module.create_rollout_manager = lambda args, pg: (manager, None)
    module.create_training_models = (
        lambda args, pgs, rollout_manager: (actor, None)
    )
    module._stage_checkpoint_dataset_state = (
        lambda args, rollout_manager, *, rollout_id: (
            FakeRay.get(rollout_manager.save.remote(rollout_id))
            is None
        )
    )
    module._commit_checkpoint_dataset_state = (
        lambda args, *, rollout_id: None
    )
    args = SimpleNamespace(
        colocate=False,
        check_weight_update_equal=False,
        start_rollout_id=0,
        num_rollout=10,
        use_critic=False,
        save_interval=10,
        rollout_global_dataset=True,
        update_weights_interval=1,
        eval_interval=1,
        eval_instance_interval=100,
        train_instance_budget=125,
        require_train_instance_budget_exhaustion=True,
        save="/does/not/matter/when-retention-is-disabled",
        checkpoint_retain_latest=0,
        async_save=False,
    )

    module.train(args)

    assert manager.generate_calls == [0, 1, 1, 2]
    assert [
        event for event in events if event[0] == "train"
    ] == [
        ("train", 0, "rollout-0"),
        ("train", 1, "rollout-1"),
    ]
    assert [
        event for event in events if event[0] == "eval"
    ] == [
        ("eval", 1, 100),
        ("eval", 1, 125),
    ]
    assert [
        event for event in events if event[0] == "ack"
    ] == [("ack", 100), ("ack", 125)]
    generate_1_indices = [
        index
        for index, event in enumerate(events)
        if event == ("generate", 1)
    ]
    assert len(generate_1_indices) == 2
    update_indices = [
        index
        for index, event in enumerate(events)
        if event == ("update_weights",)
    ]
    train_1_index = events.index(("train", 1, "rollout-1"))
    # Rollout N+1 was already dispatched.  Training and weight sync for N do
    # not wait for its control result; stale=1 permits that overlap.
    assert any(
        generate_1_indices[0] < index < generate_1_indices[1]
        for index in update_indices
    )
    assert not any(
        generate_1_indices[1] < index < train_1_index
        for index in update_indices
    )
    # The terminal model is paired with the cursor after final validation.
    assert (
        "dataset_save",
        1,
        125,
        125,
    ) in events
    assert ("model_save", 1, True) in events


def test_required_budget_exhaustion_fails_before_final_validation_or_checkpoint(
    monkeypatch, tmp_path
):
    module = _load_train_async(monkeypatch)
    events = []

    class Ref:
        def __init__(self, value=None):
            self.value = value

    class FakeRay:
        @staticmethod
        def get(ref):
            assert isinstance(ref, Ref)
            return ref.value

    class RemoteMethod:
        def __init__(self, name, callback):
            self.name = name
            self.callback = callback

        def remote(self, *args, **kwargs):
            events.append((self.name, *args))
            return Ref(self.callback(*args, **kwargs))

    class RolloutManager:
        def __init__(self):
            self.attempted = 1249
            self.generate = RemoteMethod(
                "generate", lambda rollout_id: "early-final-rollout"
            )
            self.get_train_instance_progress = RemoteMethod(
                "progress",
                lambda: {
                    "attempted_instances": self.attempted,
                    "last_validation_attempt": 1200,
                    "instance_budget": 1250,
                },
            )
            self.eval = RemoteMethod(
                "eval",
                lambda *args: {
                    "eval/attempted": 50,
                    "eval/completed": 50,
                    "eval/incomplete": 0,
                },
            )
            self.save = RemoteMethod("dataset_save", lambda *_args: None)
            self.check_weights = RemoteMethod(
                "check_weights", lambda *_args, **_kwargs: None
            )
            self.dispose = RemoteMethod("dispose", lambda: None)

    class Actor:
        def update_weights(self):
            events.append(("update_weights",))

        def async_train(self, rollout_id, rollout_data, external_data=None):
            events.append(("train", rollout_id, rollout_data))
            return Ref()

        def save_model(self, rollout_id, force_sync=False):
            events.append(("model_save", rollout_id, force_sync))

    manager = RolloutManager()
    module.ray = FakeRay
    module.configure_logger = lambda: None
    module.init_tracking = lambda args: None
    module.finish_tracking = lambda args: events.append(("finish",))
    module.create_placement_groups = lambda args: {"rollout": object()}
    module.create_rollout_manager = lambda args, pg: (manager, None)
    module.create_training_models = (
        lambda args, pgs, rollout_manager: (Actor(), None)
    )
    module._stage_checkpoint_dataset_state = (
        lambda args, rollout_manager, *, rollout_id: (
            FakeRay.get(rollout_manager.save.remote(rollout_id))
            is None
        )
    )
    module._commit_checkpoint_dataset_state = (
        lambda args, *, rollout_id: None
    )
    args = SimpleNamespace(
        colocate=False,
        check_weight_update_equal=False,
        start_rollout_id=0,
        num_rollout=1,
        use_critic=False,
        save_interval=1,
        rollout_global_dataset=True,
        update_weights_interval=1,
        eval_interval=1,
        eval_instance_interval=None,
        train_instance_budget=1250,
        require_train_instance_budget_exhaustion=True,
        stop_after_validation_attempt=None,
        save=tmp_path,
        checkpoint_retain_latest=0,
        async_save=False,
    )

    with pytest.raises(
        RuntimeError,
        match="did not exhaust the source-instance budget",
    ):
        module.train(args)

    assert ("train", 0, "early-final-rollout") in events
    assert not any(event[0] == "eval" for event in events)
    assert not any(event[0] == "dataset_save" for event in events)
    assert not any(event[0] == "model_save" for event in events)
    assert not any(event[0] == "dispose" for event in events)
    assert ("finish",) not in events


def test_exact_final_boundary_is_not_validated_twice(monkeypatch):
    module = _load_train_async(monkeypatch)
    eval_attempts = []

    class Ref:
        def __init__(self, value=None):
            self.value = value

    class FakeRay:
        @staticmethod
        def get(ref):
            assert isinstance(ref, Ref)
            return ref.value

    class RemoteMethod:
        def __init__(self, callback):
            self.callback = callback

        def remote(self, *args, **kwargs):
            return Ref(self.callback(*args, **kwargs))

    class RolloutManager:
        def __init__(self):
            self.last_validation = 0
            self.after_boundary = False
            self.generate = RemoteMethod(self._generate)
            self.eval = RemoteMethod(self._eval)
            self.acknowledge_train_validation = RemoteMethod(
                self._acknowledge
            )
            self.drain_train_validations = RemoteMethod(
                self._drain_train_validations
            )
            self.get_train_instance_progress = RemoteMethod(
                self._progress
            )
            self.save = RemoteMethod(lambda rollout_id: None)
            self.check_weights = RemoteMethod(lambda *args, **kwargs: None)
            self.dispose = RemoteMethod(lambda: None)

        def _generate(self, rollout_id):
            if not self.after_boundary:
                return {
                    TRAIN_VALIDATION_BOUNDARY_KEY: True,
                    "attempted_instances": 100,
                    "boundary": 100,
                }
            return {
                TRAIN_INSTANCE_BUDGET_EXHAUSTED_KEY: True,
                "attempted_instances": 100,
                "budget": 100,
            }

        def _eval(self, rollout_id, attempted):
            eval_attempts.append(attempted)
            return {
                "eval/attempted": 50,
                "eval/completed": 50,
                "eval/incomplete": 0,
            }

        def _acknowledge(self, attempted):
            self.last_validation = attempted
            self.after_boundary = True
            return self._progress()

        def _drain_train_validations(self, rollout_id, attempted):
            self._eval(rollout_id, attempted)
            return self._acknowledge(attempted)

        def _progress(self):
            return {
                "attempted_instances": 100,
                "last_validation_attempt": self.last_validation,
            }

    class Actor:
        def update_weights(self):
            pass

    manager = RolloutManager()
    module.ray = FakeRay
    module.configure_logger = lambda: None
    module.init_tracking = lambda args: None
    module.finish_tracking = lambda args: None
    module.create_placement_groups = lambda args: {"rollout": object()}
    module.create_rollout_manager = lambda args, pg: (manager, None)
    module.create_training_models = (
        lambda args, pgs, rollout_manager: (Actor(), None)
    )
    args = SimpleNamespace(
        colocate=False,
        check_weight_update_equal=False,
        start_rollout_id=0,
        num_rollout=10,
        use_critic=False,
        save_interval=10,
        rollout_global_dataset=True,
        update_weights_interval=1,
        eval_interval=1,
        eval_instance_interval=100,
    )

    module.train(args)
    assert eval_attempts == [100]


@pytest.mark.parametrize(
    "refill_outcome",
    ["success", "next_validation_boundary", "terminal_budget"],
)
def test_intermediate_chunk_partial_refill_is_checkpointed_or_fails_closed(
    monkeypatch, tmp_path, refill_outcome
):
    module = _load_train_async(monkeypatch)
    events = []

    class Ref:
        def __init__(self, value=None):
            self.value = value

    class FakeRay:
        @staticmethod
        def get(ref):
            assert isinstance(ref, Ref)
            return ref.value

    class RemoteMethod:
        def __init__(self, callback):
            self.callback = callback

        def remote(self, *args, **kwargs):
            return Ref(self.callback(*args, **kwargs))

    class RolloutManager:
        def __init__(self):
            self.attempted = 0
            self.last_validation = 0
            self.generate_calls = 0
            self.generate = RemoteMethod(self._generate)
            self.eval = RemoteMethod(self._eval)
            self.acknowledge_train_validation = RemoteMethod(self._ack)
            self.drain_train_validations = RemoteMethod(
                self._drain_train_validations
            )
            self.get_train_instance_progress = RemoteMethod(self._progress)
            self.save = RemoteMethod(self._save)
            self.check_weights = RemoteMethod(lambda *args, **kwargs: None)
            self.dispose = RemoteMethod(
                lambda: events.append(("dispose",))
            )

        def _generate(self, rollout_id):
            assert rollout_id == 0
            self.generate_calls += 1
            events.append(("generate", rollout_id))
            if self.generate_calls == 1:
                self.attempted = 100
                return {
                    TRAIN_VALIDATION_BOUNDARY_KEY: True,
                    "attempted_instances": 100,
                    "boundary": 100,
                    "preserved_partial": True,
                    "preserved_group_count": 1,
                    "preserved_group_kinds": {"root": 1},
                }
            assert self.last_validation == 100
            if refill_outcome == "next_validation_boundary":
                self.attempted = 200
                return {
                    TRAIN_VALIDATION_BOUNDARY_KEY: True,
                    "attempted_instances": 200,
                    "boundary": 200,
                    "preserved_partial": True,
                    "preserved_group_count": 1,
                    "preserved_group_kinds": {"root": 1},
                }
            if refill_outcome == "terminal_budget":
                self.attempted = 1250
                return {
                    TRAIN_INSTANCE_BUDGET_EXHAUSTED_KEY: True,
                    "attempted_instances": 1250,
                    "budget": 1250,
                }
            self.attempted = 103
            return "refilled-rollout-0"

        def _eval(self, rollout_id, attempted):
            events.append(("eval", rollout_id, attempted))
            return {
                "eval/attempted": 50,
                "eval/completed": 50,
                "eval/incomplete": 0,
            }

        def _ack(self, attempted):
            self.last_validation = attempted
            events.append(("ack", attempted))
            return self._progress()

        def _drain_train_validations(self, rollout_id, attempted):
            self._eval(rollout_id, attempted)
            return self._ack(attempted)

        def _progress(self):
            return {
                "attempted_instances": self.attempted,
                "last_validation_attempt": self.last_validation,
                "instance_budget": 1250,
                "eval_instance_interval": 100,
            }

        def _save(self, rollout_id):
            events.append(
                (
                    "dataset_save",
                    rollout_id,
                    self.attempted,
                    self.last_validation,
                )
            )

    class Actor:
        def update_weights(self):
            events.append(("update_weights",))

        def async_train(self, rollout_id, rollout_data, external_data=None):
            events.append(("train", rollout_id, rollout_data))
            return Ref()

        def save_model(self, rollout_id, force_sync=False):
            events.append(("model_save", rollout_id, force_sync))

    manager = RolloutManager()
    actor = Actor()
    module.ray = FakeRay
    module.configure_logger = lambda: None
    module.init_tracking = lambda args: None
    module.finish_tracking = lambda args: None
    module.create_placement_groups = lambda args: {"rollout": object()}
    module.create_rollout_manager = lambda args, pg: (manager, None)
    module.create_training_models = (
        lambda args, pgs, rollout_manager: (actor, None)
    )
    module._stage_checkpoint_dataset_state = (
        lambda args, rollout_manager, *, rollout_id: (
            FakeRay.get(rollout_manager.save.remote(rollout_id))
            is None
        )
    )
    module._commit_checkpoint_dataset_state = (
        lambda args, *, rollout_id: None
    )
    args = SimpleNamespace(
        colocate=False,
        check_weight_update_equal=False,
        start_rollout_id=0,
        num_rollout=10,
        use_critic=False,
        save_interval=10,
        rollout_global_dataset=True,
        update_weights_interval=1,
        eval_interval=1,
        eval_instance_interval=100,
        train_instance_budget=1250,
        stop_after_validation_attempt=100,
        save=tmp_path,
        checkpoint_retain_latest=0,
        async_save=False,
    )

    if refill_outcome != "success":
        error = (
            "could not be refilled before the next validation boundary"
            if refill_outcome == "next_validation_boundary"
            else "could not be refilled before the terminal"
        )
        with pytest.raises(
            RuntimeError,
            match=error,
        ):
            module.train(args)
        assert manager.generate_calls == 2
        assert [
            event for event in events if event[0] == "eval"
        ] == [("eval", 0, 100)]
        assert [
            event for event in events if event[0] == "ack"
        ] == [("ack", 100)]
        assert not any(event[0] == "train" for event in events)
        assert not any(event[0] == "dataset_save" for event in events)
        assert not any(event[0] == "model_save" for event in events)
        return

    module.train(args)

    assert manager.generate_calls == 2
    assert [
        event for event in events if event[0] == "eval"
    ] == [("eval", 0, 100)]
    assert [
        event for event in events if event[0] == "train"
    ] == [("train", 0, "refilled-rollout-0")]
    assert (
        "dataset_save",
        0,
        103,
        100,
    ) in events
    assert ("model_save", 0, True) in events
    assert events[-1] == ("dispose",)


def test_incomplete_validation_does_not_acknowledge_boundary(monkeypatch):
    module = _load_train_async(monkeypatch)
    acknowledgements = []

    class Ref:
        def __init__(self, value=None):
            self.value = value

    class FakeRay:
        @staticmethod
        def get(ref):
            assert isinstance(ref, Ref)
            return ref.value

    class RemoteMethod:
        def __init__(self, callback):
            self.callback = callback

        def remote(self, *args, **kwargs):
            return Ref(self.callback(*args, **kwargs))

    class Manager:
        drain_train_validations = RemoteMethod(
            lambda rollout_id, attempted: (_ for _ in ()).throw(
                RuntimeError(
                    "validation was incomplete: completed=49 attempted=50"
                )
            )
        )

    module.ray = FakeRay
    with pytest.raises(RuntimeError, match="validation was incomplete"):
        module._run_instance_validation(
            Manager(),
            rollout_id=7,
            attempted_instances=100,
        )
    assert acknowledgements == []


def test_budget_after_resume_persists_ack_with_existing_checkpoint(
    monkeypatch,
    tmp_path,
):
    module = _load_train_async(monkeypatch)
    checkpoint_id = 7
    (tmp_path / f"iter_{checkpoint_id:07d}").mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("7\n")
    (tmp_path / "rollout").mkdir()
    events = []

    class Ref:
        def __init__(self, value=None):
            self.value = value

    class FakeRay:
        @staticmethod
        def get(ref):
            assert isinstance(ref, Ref)
            return ref.value

    class RemoteMethod:
        def __init__(self, callback):
            self.callback = callback

        def remote(self, *args, **kwargs):
            return Ref(self.callback(*args, **kwargs))

    class Manager:
        def __init__(self):
            self.last_validation = 100
            self.generate = RemoteMethod(
                lambda rollout_id: {
                    TRAIN_INSTANCE_BUDGET_EXHAUSTED_KEY: True,
                    "attempted_instances": 125,
                    "budget": 125,
                }
            )
            self.eval = RemoteMethod(self._eval)
            self.acknowledge_train_validation = RemoteMethod(self._ack)
            self.drain_train_validations = RemoteMethod(
                self._drain_train_validations
            )
            self.get_train_instance_progress = RemoteMethod(self._progress)
            self.save = RemoteMethod(self._save)
            self.check_weights = RemoteMethod(lambda *args, **kwargs: None)
            self.dispose = RemoteMethod(lambda: None)

        def _eval(self, rollout_id, attempted):
            events.append(("eval", attempted))
            return {
                "eval/attempted": 50,
                "eval/completed": 50,
                "eval/incomplete": 0,
            }

        def _ack(self, attempted):
            self.last_validation = attempted
            events.append(("ack", attempted))
            return self._progress()

        def _drain_train_validations(self, rollout_id, attempted):
            self._eval(rollout_id, attempted)
            return self._ack(attempted)

        def _progress(self):
            return {
                "attempted_instances": 125,
                "last_validation_attempt": self.last_validation,
            }

        def _save(self, rollout_id):
            events.append(
                ("dataset_save", rollout_id, self.last_validation)
            )

    class Actor:
        def update_weights(self):
            pass

    manager = Manager()
    module.ray = FakeRay
    module.configure_logger = lambda: None
    module.init_tracking = lambda args: None
    module.finish_tracking = lambda args: None
    module.create_placement_groups = lambda args: {"rollout": object()}
    module.create_rollout_manager = lambda args, pg: (manager, None)
    module.create_training_models = (
        lambda args, pgs, rollout_manager: (Actor(), None)
    )
    module._stage_checkpoint_dataset_state = (
        lambda args, rollout_manager, *, rollout_id: (
            FakeRay.get(rollout_manager.save.remote(rollout_id))
            is None
        )
    )
    module._commit_checkpoint_dataset_state = (
        lambda args, *, rollout_id: None
    )
    args = SimpleNamespace(
        colocate=False,
        check_weight_update_equal=False,
        start_rollout_id=8,
        num_rollout=20,
        use_critic=False,
        save_interval=10,
        rollout_global_dataset=True,
        update_weights_interval=1,
        eval_interval=1,
        eval_instance_interval=100,
        save=tmp_path,
        checkpoint_retain_latest=0,
    )

    module.train(args)

    assert events == [
        ("eval", 125),
        ("ack", 125),
        ("dataset_save", 7, 125),
    ]


def test_checkpoint_pruning_keeps_latest_complete_pairs(
    monkeypatch,
    tmp_path,
):
    module = _load_train_async(monkeypatch)
    rollout_state = tmp_path / "rollout"
    rollout_state.mkdir()
    for checkpoint_id in (1, 2, 3):
        (tmp_path / f"iter_{checkpoint_id:07d}").mkdir()
        (
            rollout_state
            / f"global_dataset_state_dict_{checkpoint_id}.pt"
        ).touch()
    (tmp_path / "iter_not_numeric").mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("3\n")

    removed = module._prune_training_checkpoints(
        SimpleNamespace(save=tmp_path, checkpoint_retain_latest=2),
        latest_rollout_id=3,
    )

    assert removed == [1]
    assert not (tmp_path / "iter_0000001").exists()
    assert not (
        rollout_state / "global_dataset_state_dict_1.pt"
    ).exists()
    assert (tmp_path / "iter_0000002").is_dir()
    assert (tmp_path / "iter_0000003").is_dir()
    assert (tmp_path / "iter_not_numeric").is_dir()


def test_dataset_state_commit_publishes_only_staged_file(
    monkeypatch,
    tmp_path,
):
    module = _load_train_async(monkeypatch)

    class Ref:
        def __init__(self, value=None):
            self.value = value

    class FakeRay:
        @staticmethod
        def get(ref):
            assert isinstance(ref, Ref)
            return ref.value

    class Save:
        def remote(self, rollout_id, *, staged=False):
            assert staged
            path = (
                tmp_path
                / "rollout"
                / f"global_dataset_state_dict_{rollout_id}.pt.staged"
            )
            path.parent.mkdir()
            path.write_text("exact-state\n")
            return Ref(str(path))

    module.ray = FakeRay
    args = SimpleNamespace(
        save=tmp_path,
        rollout_global_dataset=True,
        checkpoint_retain_latest=0,
    )
    manager = SimpleNamespace(save=Save())

    assert module._stage_checkpoint_dataset_state(
        args,
        manager,
        rollout_id=4,
    )
    final_path = (
        tmp_path / "rollout" / "global_dataset_state_dict_4.pt"
    )
    assert not final_path.exists()

    (tmp_path / "iter_0000004").mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("4\n")
    module._commit_checkpoint_dataset_state(args, rollout_id=4)
    assert final_path.read_text() == "exact-state\n"
    assert not final_path.with_name(f"{final_path.name}.staged").exists()


def test_dataset_state_commit_rejects_mismatched_model_checkpoint(
    monkeypatch,
    tmp_path,
):
    module = _load_train_async(monkeypatch)
    state_root = tmp_path / "rollout"
    state_root.mkdir()
    staged_path = (
        state_root / "global_dataset_state_dict_4.pt.staged"
    )
    staged_path.write_text("exact-state\n")
    (tmp_path / "iter_0000003").mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("3\n")
    args = SimpleNamespace(
        save=tmp_path,
        rollout_global_dataset=True,
        checkpoint_retain_latest=0,
    )

    with pytest.raises(
        RuntimeError,
        match="matching model checkpoint",
    ):
        module._commit_checkpoint_dataset_state(args, rollout_id=4)

    assert staged_path.read_text() == "exact-state\n"
    assert not (
        state_root / "global_dataset_state_dict_4.pt"
    ).exists()


def test_model_save_failure_leaves_only_uncommitted_dataset_stage(
    monkeypatch,
    tmp_path,
):
    module = _load_train_async(monkeypatch)

    class Ref:
        def __init__(self, value=None):
            self.value = value

    class FakeRay:
        @staticmethod
        def get(ref):
            return ref.value

    class Save:
        def remote(self, rollout_id, *, staged=False):
            assert staged
            path = (
                tmp_path
                / "rollout"
                / f"global_dataset_state_dict_{rollout_id}.pt.staged"
            )
            path.parent.mkdir()
            path.write_text("future-data\n")
            return Ref(str(path))

    class Actor:
        def save_model(self, rollout_id, force_sync=False):
            raise RuntimeError("simulated model save failure")

    module.ray = FakeRay
    args = SimpleNamespace(
        save=tmp_path,
        rollout_global_dataset=True,
        checkpoint_retain_latest=0,
        use_critic=False,
        num_critic_only_steps=0,
        async_save=False,
    )
    with pytest.raises(RuntimeError, match="model save failure"):
        module._save_checkpoint(
            args=args,
            rollout_manager=SimpleNamespace(save=Save()),
            actor_model=Actor(),
            critic_model=None,
            rollout_id=5,
            save_model=True,
            force_sync=True,
        )

    state_root = tmp_path / "rollout"
    assert (
        state_root / "global_dataset_state_dict_5.pt.staged"
    ).is_file()
    assert not (
        state_root / "global_dataset_state_dict_5.pt"
    ).exists()
