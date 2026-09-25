"""Schedule values reach the actors, and bad schedule keys fail early.

Handlers apply each iteration's schedule to the stage actor (as attributes,
or through the actor's ``apply_schedule`` hook) before ``update``, and still
forward the same values to ``update`` as keyword arguments.
"""

import warnings
from pathlib import Path

import ase.io
import pytest
import yaml
from ase import Atoms

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import LocalFileSystem
from remote_handler.exceptions import ScheduleKeyWarning, WorkflowError
from remote_handler.workflows import trainers as trainers_mod
from remote_handler.workflows.active_learning_base import (
    ActiveLearningHandler,
    Schedule,
    apply_schedule_kwargs,
    validate_schedule_kwargs,
)
from remote_handler.workflows.evaluators import (
    EMTEvaluator,
    ORCAEvaluator,
    VASPEvaluator,
)
from remote_handler.workflows.selectors import StrideSelector
from remote_handler.workflows.staged_active_learning import (
    InputSource,
    RemoteSelector,
    SelectorActor,
    Stage,
    StagedActiveLearningHandler,
)
from remote_handler.workflows.trainers import (
    MaceModelTrainer,
    NeuralilILTrainer,
)


class _FnSchedule(Schedule):
    def __init__(self, num_iterations, fn):
        self._fn = fn
        super().__init__(num_iterations)

    def get_kwargs(self, iteration_idx):
        return self._fn(iteration_idx)


def _cfg(tmp_path):
    return ClusterConfig(
        remote_machine="m",
        partition="p",
        conda_env="e",
        conda_source="/x",
        savepath=str(tmp_path),
        template="t.j2",
    )


class _Actor:
    """Records the attribute value and update kwargs seen at each call."""

    def __init__(self, tmp_path: Path, name: str):
        self.temperature = 300
        self.out = Path(tmp_path) / f"{name}_out.xyz"
        self.out.write_text(name)
        self.seen: list[tuple[int, dict]] = []

    def update(self, calcdir, actor_input, **kwargs):
        self.seen.append((self.temperature, kwargs))

    def get_output(self):
        return self.out

    # Legacy getter names + hook, so the actor fits ActiveLearningHandler.
    get_sampled_configs = (
        get_evaluated_configs
    ) = get_trained_model = get_output

    def pending_manifest(self, calcdir):
        return None


def _seed(tmp_path: Path, name: str) -> Path:
    p = Path(tmp_path) / name
    p.write_text(name)
    return p


def _write_pool(path: Path, n: int) -> Path:
    frames = [
        Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7 + 0.05 * i]])
        for i in range(n)
    ]
    ase.io.write(path, frames)
    return path


# --------------------------------------------------------------------------
# The helpers
# --------------------------------------------------------------------------


def test_apply_sets_attributes(tmp_path):
    actor = _Actor(tmp_path, "a")
    apply_schedule_kwargs(actor, {"temperature": 500})
    assert actor.temperature == 500


def test_unknown_key_warns_and_is_not_set(tmp_path):
    actor = _Actor(tmp_path, "a")
    with pytest.warns(ScheduleKeyWarning, match="temprature"):
        validate_schedule_kwargs(actor, {"temprature": 500})
    apply_schedule_kwargs(actor, {"temprature": 500})
    assert not hasattr(actor, "temprature")


def test_known_keys_do_not_warn(tmp_path, recwarn):
    validate_schedule_kwargs(_Actor(tmp_path, "a"), {"temperature": 1})
    assert not recwarn.list


def test_apply_schedule_hook_replaces_setattr():
    class _Hooked:
        def __init__(self):
            self.x = 0
            self.received = None

        def apply_schedule(self, kwargs):
            self.received = kwargs

    actor = _Hooked()
    apply_schedule_kwargs(actor, {"x": 5})
    assert actor.received == {"x": 5}
    assert actor.x == 0


def test_unknown_schedule_keys_hook_replaces_hasattr_check(recwarn):
    class _Declares:
        def unknown_schedule_keys(self, kwargs):
            return [k for k in kwargs if k != "declared"]

    validate_schedule_kwargs(_Declares(), {"declared": 1})
    assert not recwarn.list
    with pytest.warns(ScheduleKeyWarning, match="other"):
        validate_schedule_kwargs(_Declares(), {"other": 1})


# --------------------------------------------------------------------------
# StagedActiveLearningHandler
# --------------------------------------------------------------------------


def test_staged_applies_attributes_and_forwards_kwargs(tmp_path):
    actor = _Actor(tmp_path, "sampler")
    schedule = _FnSchedule(3, lambda i: {"temperature": 100 * (i + 1)})
    StagedActiveLearningHandler(
        [
            Stage(
                "sampler",
                actor,
                schedule=schedule,
                initial_input=_seed(tmp_path, "model.pkl"),
            )
        ],
        calchome=tmp_path / "calc",
        num_iterations=3,
    ).run()

    assert actor.seen == [
        (100, {"temperature": 100}),
        (200, {"temperature": 200}),
        (300, {"temperature": 300}),
    ]


def _single_stage(tmp_path, actor, schedule, num_iterations=2):
    return StagedActiveLearningHandler(
        [
            Stage(
                "sampler",
                actor,
                schedule=schedule,
                initial_input=_seed(tmp_path, "model.pkl"),
            )
        ],
        calchome=tmp_path / "calc",
        num_iterations=num_iterations,
    )


def test_staged_bad_key_warns_once_at_construction(tmp_path):
    actor = _Actor(tmp_path, "sampler")
    schedule = _FnSchedule(2, lambda i: {"temprature": 100})
    with pytest.warns(ScheduleKeyWarning, match="temprature") as record:
        handler = _single_stage(tmp_path, actor, schedule)
    assert actor.seen == []
    assert len(record) == 1

    with warnings.catch_warnings():
        warnings.simplefilter("error", ScheduleKeyWarning)
        handler.run()  # no further warning per iteration


def test_bad_key_is_an_error_under_strict_filter(tmp_path):
    actor = _Actor(tmp_path, "sampler")
    schedule = _FnSchedule(2, lambda i: {"temprature": 100})
    with warnings.catch_warnings():
        warnings.simplefilter("error", ScheduleKeyWarning)
        with pytest.raises(ScheduleKeyWarning):
            _single_stage(tmp_path, actor, schedule)
    assert actor.seen == []


class _KwargsOnlySampler:
    """Reads schedule values only from update's kwargs (like lammps_al's
    LammpsSampler) -- the pre-existing contract, which must keep working."""

    def __init__(self, tmp_path):
        self.out = _seed(tmp_path, "samples.xyz")
        self.script_input: list[dict] = []

    def update(self, calcdir, actor_input, **kwargs):
        self.script_input.append(dict(kwargs))

    def get_output(self):
        return self.out


def test_kwargs_only_actor_still_receives_values(tmp_path):
    actor = _KwargsOnlySampler(tmp_path)
    schedule = _FnSchedule(2, lambda i: {"n_run": 10 * (i + 1)})
    with pytest.warns(ScheduleKeyWarning):
        handler = _single_stage(tmp_path, actor, schedule)
    handler.run()
    assert actor.script_input == [{"n_run": 10}, {"n_run": 20}]
    assert not hasattr(actor, "n_run")


def test_extra_inputs_clash_raises_at_construction(tmp_path):
    actor = _Actor(tmp_path, "sampler")
    actor.reference = None
    schedule = _FnSchedule(1, lambda i: {"reference": None})
    with pytest.raises(WorkflowError, match="collide"):
        StagedActiveLearningHandler(
            [
                Stage(
                    "sampler",
                    actor,
                    schedule=schedule,
                    initial_input=_seed(tmp_path, "model.pkl"),
                    extra_inputs={"reference": InputSource(stage="sampler")},
                )
            ],
            calchome=tmp_path / "calc",
            num_iterations=1,
        )


def test_extra_inputs_are_not_set_as_attributes(tmp_path):
    actor = _Actor(tmp_path, "sampler")
    schedule = _FnSchedule(2, lambda i: {"temperature": 7})
    StagedActiveLearningHandler(
        [
            Stage(
                "sampler",
                actor,
                schedule=schedule,
                initial_input=_seed(tmp_path, "model.pkl"),
                extra_inputs={"reference": InputSource(stage="sampler")},
            )
        ],
        calchome=tmp_path / "calc",
        num_iterations=2,
    ).run()

    assert not hasattr(actor, "reference")
    # ...but the extra input still reaches update as a keyword argument.
    assert all("reference" in kw for _, kw in actor.seen)


def test_remote_selector_consumes_n_configs_as_kwarg(tmp_path):
    pool = _write_pool(tmp_path / "pool.extxyz", 10)
    selector = RemoteSelector(
        selector_type="StrideSelector",
        selector_kwargs={"n_configs": 5},
        cluster_config=None,
        remote_fs=None,
        stub=True,
    )
    schedule = _FnSchedule(1, lambda i: {"n_configs": 2})
    calchome = tmp_path / "calc"
    handler = StagedActiveLearningHandler(
        [Stage("selector", selector, schedule=schedule, initial_input=pool)],
        calchome=calchome,
        num_iterations=1,
    )
    handler.run()

    out = calchome / handler.outputs["selector"][0][0]
    assert len(ase.io.read(out, index=":")) == 2
    # Not an attribute of RemoteSelector; it only travels through update.
    assert not hasattr(selector, "n_configs")


def test_remote_selector_warns_on_non_constructor_key():
    selector = RemoteSelector(
        selector_type="StrideSelector",
        selector_kwargs={"n_configs": 5},
        cluster_config=None,
        remote_fs=None,
        stub=True,
    )
    with pytest.warns(ScheduleKeyWarning, match="n_confgs"):
        validate_schedule_kwargs(selector, {"n_confgs": 2})


def test_selector_actor_forwards_to_wrapped_selector(tmp_path):
    pool = _write_pool(tmp_path / "pool.extxyz", 10)
    actor = SelectorActor(StrideSelector(n_configs=5))
    schedule = _FnSchedule(1, lambda i: {"n_configs": 3})
    calchome = tmp_path / "calc"
    handler = StagedActiveLearningHandler(
        [Stage("selector", actor, schedule=schedule, initial_input=pool)],
        calchome=calchome,
        num_iterations=1,
    )
    handler.run()

    assert actor.selector.n_configs == 3
    out = calchome / handler.outputs["selector"][0][0]
    assert len(ase.io.read(out, index=":")) == 3

    with pytest.warns(ScheduleKeyWarning, match="n_confgs"):
        validate_schedule_kwargs(actor, {"n_confgs": 3})


class _RecordingArrayTask:
    """Stands in for ArrayScriptTask; records what the trainer built."""

    instances: list["_RecordingArrayTask"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _RecordingArrayTask.instances.append(self)

    def allocate_data(self):
        pass


def test_scheduled_config_file_reaches_training_task(tmp_path, monkeypatch):
    initial = tmp_path / "initial.yaml"
    scheduled = tmp_path / "scheduled.yaml"
    initial.write_text("a: 1\n")
    scheduled.write_text("a: 2\n")

    _RecordingArrayTask.instances = []
    monkeypatch.setattr(trainers_mod, "ArrayScriptTask", _RecordingArrayTask)
    # Skip running the (mock) array job; only the built task matters here.
    monkeypatch.setattr(
        MaceModelTrainer,
        "get_trained_model",
        lambda self: _seed(tmp_path, "model.model"),
    )

    trainer = MaceModelTrainer(
        remote_fs=LocalFileSystem(),
        cluster_config=_cfg(tmp_path),
        config_file=initial,
        remote_mace_train_runner="/remote/run_train.py",
        num_models=2,
    )
    schedule = _FnSchedule(
        3, lambda i: {"config_file": scheduled if i == 1 else initial}
    )
    StagedActiveLearningHandler(
        [
            Stage(
                "trainer",
                trainer,
                schedule=schedule,
                output_extension="model",
                initial_input=_seed(tmp_path, "db.extxyz"),
            )
        ],
        calchome=tmp_path / "calc",
        num_iterations=3,
    ).run()

    used = [
        Path(t.kwargs["shared_ressources"][0])
        for t in _RecordingArrayTask.instances
    ]
    assert used == [initial, scheduled, initial]
    for task, cfg in zip(_RecordingArrayTask.instances, used):
        shared = task.kwargs["individual_ressources"][0][0].content
        assert shared["config_file"].endswith(cfg.name)


# --------------------------------------------------------------------------
# Legacy ActiveLearningHandler
# --------------------------------------------------------------------------


def _legacy(tmp_path, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return ActiveLearningHandler(
            calchome=tmp_path / "calc",
            initial_trainer_out=_seed(tmp_path, "model.pkl"),
            **kw,
        )


def test_legacy_applies_schedule(tmp_path):
    sampler = _Actor(tmp_path, "sampler")
    handler = _legacy(
        tmp_path,
        sampler=sampler,
        evaluator=_Actor(tmp_path, "evaluator"),
        trainer=_Actor(tmp_path, "trainer"),
        num_iterations=2,
        sampler_schedule=_FnSchedule(
            2, lambda i: {"temperature": 10 * (i + 1)}
        ),
    )
    handler.run()
    assert [t for t, _ in sampler.seen] == [10, 20]
    assert [kw for _, kw in sampler.seen] == [
        {"temperature": 10},
        {"temperature": 20},
    ]


def test_legacy_bad_key_warns_at_construction(tmp_path):
    with pytest.warns(ScheduleKeyWarning, match="temprature"):
        _legacy(
            tmp_path,
            sampler=_Actor(tmp_path, "sampler"),
            evaluator=_Actor(tmp_path, "evaluator"),
            trainer=_Actor(tmp_path, "trainer"),
            num_iterations=2,
            trainer_schedule=_FnSchedule(2, lambda i: {"temprature": 1}),
        )


# --------------------------------------------------------------------------
# Actors that derive state from a scheduled parameter
# --------------------------------------------------------------------------


def test_vasp_scheduled_incar_rebuilds_calc_config(tmp_path):
    ev = VASPEvaluator(
        remote_fs=LocalFileSystem(),
        cluster_config=_cfg(tmp_path),
        num_parallel=1,
        incar_kwargs={"ENCUT": 400},
        n_processors=8,
        stub=True,
    )
    apply_schedule_kwargs(
        ev, {"incar_kwargs": {"ENCUT": 520}, "n_processors": 16}
    )
    assert ev.calc_config["INCAR_kwargs"] == {"ENCUT": 520}
    assert ev.calc_config["launch_command"] == "mpirun -n 16 vasp_std"
    assert ev.n_processors == 16


def test_orca_scheduled_n_processors_rebuilds_calc_config(tmp_path):
    ev = ORCAEvaluator(
        cluster_config=_cfg(tmp_path),
        remote_fs=LocalFileSystem(),
        num_parallel=1,
        orcasimpleinput="B3LYP def2-SVP",
        n_processors=2,
        stub=True,
    )
    apply_schedule_kwargs(ev, {"n_processors": 4})
    assert ev.calc_config["orcablocks"] == "%pal nprocs 4 end"


def test_emt_scheduled_calculator_kwargs_rebuild_calc_config(tmp_path):
    ev = EMTEvaluator(
        remote_fs=LocalFileSystem(),
        cluster_config=_cfg(tmp_path),
        num_parallel=1,
        stub=True,
    )
    apply_schedule_kwargs(ev, {"calculator_kwargs": {"asap_cutoff": True}})
    assert ev.calc_config["calculator_kwargs"] == {"asap_cutoff": True}


def test_evaluator_calc_config_untouched_by_unrelated_key(tmp_path):
    ev = VASPEvaluator(
        remote_fs=LocalFileSystem(),
        cluster_config=_cfg(tmp_path),
        num_parallel=1,
        incar_kwargs={"ENCUT": 400},
        stub=True,
    )
    custom = {"INCAR_kwargs": {"ENCUT": 1}, "launch_command": "x"}
    apply_schedule_kwargs(ev, {"calc_config": custom})
    assert ev.calc_config == custom
    apply_schedule_kwargs(ev, {"required_fraction": 0.9})
    assert ev.calc_config == custom


def _neuralil(tmp_path, **kw):
    return NeuralilILTrainer(
        remote_fs=LocalFileSystem(),
        cluster_config=_cfg(tmp_path),
        training_script="train.py",
        stub=True,
        **kw,
    )


def test_neuralil_scheduled_hyperparameter(tmp_path):
    t = _neuralil(tmp_path)
    apply_schedule_kwargs(t, {"n_epochs": 7})
    assert t.training_kwargs["n_epochs"] == 7


def test_neuralil_scheduled_config_file_relayers(tmp_path):
    cfg = tmp_path / "nn.yaml"
    cfg.write_text(yaml.safe_dump({"n_epochs": 11, "n_batch": 8}))
    t = _neuralil(tmp_path, training_kwargs={"n_batch": 4})
    apply_schedule_kwargs(t, {"config_file": cfg})
    assert t.config_file == cfg
    assert t.training_kwargs["n_epochs"] == 11
    # Explicit training_kwargs still beat the (scheduled) config file.
    assert t.training_kwargs["n_batch"] == 4
    assert t.training_kwargs["stub"] is True


def test_neuralil_scheduled_training_kwargs_replace_overrides(tmp_path):
    t = _neuralil(tmp_path, training_kwargs={"n_batch": 4})
    apply_schedule_kwargs(t, {"training_kwargs": {"n_epochs": 3}})
    assert t.training_kwargs["n_epochs"] == 3
    assert t.training_kwargs["n_batch"] == 32  # back to the default


def test_neuralil_unknown_key_warns_and_is_dropped(tmp_path):
    t = _neuralil(tmp_path)
    with pytest.warns(ScheduleKeyWarning, match="n_epoch"):
        validate_schedule_kwargs(t, {"n_epoch": 3})
    apply_schedule_kwargs(t, {"n_epoch": 3})
    assert "n_epoch" not in t.training_kwargs
