"""Tests for the alternative stage-based active learning handler."""

import json
from pathlib import Path

import ase.io
import pytest
from ase import Atoms

from remote_handler.exceptions import WorkflowError
from remote_handler.workflows.active_learning_base import Selector
from remote_handler.workflows.selectors import ConfigSelector, StrideSelector
from remote_handler.workflows.staged_active_learning import (
    InputSource,
    SelectorActor,
    Stage,
    StageCompleted,
    StagedActiveLearningHandler,
    resolve_actor_output,
)


class _Actor:
    """Generic stage actor writing a single dummy output file.

    Records every ``update`` call so tests can assert which (iteration,
    stage) steps actually ran.
    """

    def __init__(self, tmp_path: Path, name: str, ext: str = "xyz"):
        self.name = name
        self.out = Path(tmp_path) / f"{name}_out.{ext}"
        self.out.write_text(name)
        self.calls: list[str] = []

    def update(self, calcdir, actor_input, **kwargs):
        inputs = (
            [actor_input]
            if isinstance(actor_input, (str, Path))
            else actor_input
        )
        for p in inputs:
            assert Path(p).exists(), f"missing input {p}"
        self.calls.append(Path(calcdir).name)

    def get_output(self):
        return self.out


class _LegacyGetterActor:
    """Actor exposing only a legacy getter name (no ``get_output``)."""

    def __init__(self, tmp_path: Path, name: str):
        self.out = Path(tmp_path) / f"{name}_out.xyz"
        self.out.write_text(name)

    def update(self, calcdir, actor_input, **kwargs):
        pass

    def get_sampled_configs(self):
        return self.out


def _seed(tmp_path: Path, name: str) -> Path:
    p = Path(tmp_path) / name
    p.write_text(name)
    return p


def _make_stages(tmp_path: Path):
    sampler = _Actor(tmp_path, "sampler")
    evaluator = _Actor(tmp_path, "evaluator")
    trainer = _Actor(tmp_path, "trainer", ext="pkl")
    stages = [
        Stage("sampler", sampler, initial_input=_seed(tmp_path, "model.pkl")),
        Stage("evaluator", evaluator),
        Stage("trainer", trainer, output_extension="pkl"),
    ]
    return stages, (sampler, evaluator, trainer)


def test_full_run_records_all_outputs(tmp_path):
    calchome = tmp_path / "calc"
    stages, (sampler, evaluator, trainer) = _make_stages(tmp_path)

    handler = StagedActiveLearningHandler(
        stages, calchome=calchome, num_iterations=3
    )
    handler.run()

    # 3 iterations x 3 stages all ran.
    assert len(sampler.calls) == 3
    assert len(evaluator.calls) == 3
    assert len(trainer.calls) == 3

    data = json.loads((calchome / "AL_progress.json").read_text())
    assert data["stages"] == ["sampler", "evaluator", "trainer"]
    for name in ("sampler", "evaluator", "trainer"):
        assert sorted(data["outputs"][name]) == ["0", "1", "2"]
        for it in ("0", "1", "2"):
            for rel in data["outputs"][name][it]:
                assert (calchome / rel).exists()


def test_exit_stage_truncates_last_iteration(tmp_path):
    calchome = tmp_path / "calc"
    stages, (sampler, evaluator, trainer) = _make_stages(tmp_path)

    handler = StagedActiveLearningHandler(
        stages,
        calchome=calchome,
        num_iterations=3,
        exit_stage="sampler",
    )
    handler.run()

    # Last iteration stops after the sampler.
    assert len(sampler.calls) == 3
    assert len(evaluator.calls) == 2
    assert len(trainer.calls) == 2


def test_entry_stage_skips_first_stages(tmp_path):
    calchome = tmp_path / "calc"
    sampler = _Actor(tmp_path, "sampler")
    evaluator = _Actor(tmp_path, "evaluator")
    trainer = _Actor(tmp_path, "trainer", ext="pkl")
    stages = [
        Stage("sampler", sampler, initial_input=_seed(tmp_path, "model.pkl")),
        Stage(
            "evaluator",
            evaluator,
            initial_input=_seed(tmp_path, "samples.extxyz"),
        ),
        Stage("trainer", trainer, output_extension="pkl"),
    ]

    handler = StagedActiveLearningHandler(
        stages,
        calchome=calchome,
        num_iterations=3,
        entry_stage="evaluator",
        exit_stage="trainer",
    )
    handler.run()

    # Iteration 0 skips the sampler.
    assert len(sampler.calls) == 2
    assert len(evaluator.calls) == 3
    assert len(trainer.calls) == 3


def test_entry_stage_without_seed_raises(tmp_path):
    sampler = _Actor(tmp_path, "sampler")  # no initial_input
    stages = [Stage("sampler", sampler)]
    with pytest.raises(WorkflowError, match="initial_input"):
        StagedActiveLearningHandler(stages, calchome=tmp_path / "c")


def test_duplicate_stage_names_raise(tmp_path):
    a = _Actor(tmp_path, "a")
    b = _Actor(tmp_path, "b")
    stages = [
        Stage("dup", a, initial_input=_seed(tmp_path, "s")),
        Stage("dup", b),
    ]
    with pytest.raises(WorkflowError, match="unique"):
        StagedActiveLearningHandler(stages, calchome=tmp_path / "c")


def test_resume_is_idempotent_and_extends(tmp_path):
    calchome = tmp_path / "calc"
    stages, (sampler, evaluator, trainer) = _make_stages(tmp_path)

    StagedActiveLearningHandler(
        stages, calchome=calchome, num_iterations=2
    ).run()
    assert len(sampler.calls) == 2

    # Re-run on the same calchome with no new iterations: nothing reruns.
    StagedActiveLearningHandler(
        stages, calchome=calchome, num_iterations=2
    ).run()
    assert len(sampler.calls) == 2

    # Extend to 4 iterations: only the two new ones run.
    StagedActiveLearningHandler(
        stages, calchome=calchome, num_iterations=4
    ).run()
    assert len(sampler.calls) == 4
    assert len(trainer.calls) == 4


def test_partial_resume_recomputes_missing_steps(tmp_path):
    calchome = tmp_path / "calc"
    stages, (sampler, evaluator, trainer) = _make_stages(tmp_path)

    StagedActiveLearningHandler(
        stages, calchome=calchome, num_iterations=2
    ).run()

    # Simulate a crash after the evaluator of iteration 1: drop the
    # trainer's iteration-1 output from the progress file.
    progress_file = calchome / "AL_progress.json"
    data = json.loads(progress_file.read_text())
    del data["outputs"]["trainer"]["1"]
    progress_file.write_text(json.dumps(data))

    trainer.calls.clear()
    StagedActiveLearningHandler(
        stages, calchome=calchome, num_iterations=2
    ).run()

    # Only the missing trainer step is recomputed.
    assert trainer.calls == ["1"]


def test_stage_mismatch_on_resume_raises(tmp_path):
    calchome = tmp_path / "calc"
    stages, _ = _make_stages(tmp_path)
    StagedActiveLearningHandler(
        stages, calchome=calchome, num_iterations=2
    ).run()

    other = [
        Stage(
            "sampler",
            _Actor(tmp_path, "s2"),
            initial_input=_seed(tmp_path, "m2"),
        ),
        Stage("trainer", _Actor(tmp_path, "t2")),
    ]
    with pytest.raises(WorkflowError, match="Stored stage list"):
        StagedActiveLearningHandler(other, calchome=calchome, num_iterations=2)


def test_combine_passes_all_producer_outputs(tmp_path):
    """A combine stage feeds all its historical outputs downstream."""
    calchome = tmp_path / "calc"

    seen: list[int] = []

    class _CountingEvaluator(_Actor):
        def update(self, calcdir, actor_input, **kwargs):
            inputs = (
                [actor_input]
                if isinstance(actor_input, (str, Path))
                else actor_input
            )
            seen.append(len(inputs))
            super().update(calcdir, actor_input, **kwargs)

    sampler = _Actor(tmp_path, "sampler")
    evaluator = _CountingEvaluator(tmp_path, "evaluator")
    stages = [
        Stage(
            "sampler",
            sampler,
            combine=True,
            initial_input=_seed(tmp_path, "model.pkl"),
        ),
        Stage("evaluator", evaluator),
    ]
    StagedActiveLearningHandler(
        stages, calchome=calchome, num_iterations=3, exit_stage="evaluator"
    ).run()

    # Evaluator sees a growing pool of sampler outputs: 1, 2, 3.
    assert seen == [1, 2, 3]


def _write_pool(path: Path, n: int) -> Path:
    frames = [
        Atoms(
            "H2",
            positions=[[0, 0, 0], [0, 0, 0.7 + 0.05 * i]],
            cell=[6, 6, 6],
            pbc=True,
        )
        for i in range(n)
    ]
    ase.io.write(path, frames)
    return path


def test_selector_stage_subsamples_pool(tmp_path):
    calchome = tmp_path / "calc"

    class _PoolSampler:
        def __init__(self, pool: Path):
            self.pool = pool

        def update(self, calcdir, actor_input, **kwargs):
            assert Path(actor_input).exists()

        def get_output(self):
            return self.pool

    pool = _write_pool(tmp_path / "pool.extxyz", 10)
    evaluator = _Actor(tmp_path, "evaluator")
    trainer = _Actor(tmp_path, "trainer", ext="pkl")

    stages = [
        Stage(
            "sampler",
            _PoolSampler(pool),
            initial_input=_seed(tmp_path, "model.pkl"),
        ),
        Stage("selector", SelectorActor(StrideSelector(n_configs=3))),
        Stage("evaluator", evaluator),
        Stage("trainer", trainer, output_extension="pkl"),
    ]

    handler = StagedActiveLearningHandler(
        stages, calchome=calchome, num_iterations=1
    )
    handler.run()

    data = json.loads((calchome / "AL_progress.json").read_text())
    sel_rel = data["outputs"]["selector"]["0"][0]
    selected = ase.io.read(calchome / sel_rel, index=":")
    assert len(selected) == 3


def test_from_actors_inserts_selector_stage(tmp_path):
    calchome = tmp_path / "calc"

    class _PoolSampler:
        def __init__(self, pool: Path):
            self.pool = pool

        def update(self, calcdir, actor_input, **kwargs):
            pass

        def get_sampled_configs(self):
            return self.pool

    pool = _write_pool(tmp_path / "pool.extxyz", 8)
    evaluator = _Actor(tmp_path, "evaluator")
    trainer = _Actor(tmp_path, "trainer", ext="pkl")

    handler = StagedActiveLearningHandler.from_actors(
        sampler=_PoolSampler(pool),
        evaluator=evaluator,
        trainer=trainer,
        selector=StrideSelector(n_configs=2),
        calchome=calchome,
        num_iterations=1,
        cycle_exit_point="training",
        initial_trainer_out=_seed(tmp_path, "model.pkl"),
    )
    assert [s.name for s in handler.stages] == [
        "sampler",
        "selector",
        "evaluator",
        "trainer",
    ]
    handler.run()

    data = json.loads((calchome / "AL_progress.json").read_text())
    sel_rel = data["outputs"]["selector"]["0"][0]
    assert len(ase.io.read(calchome / sel_rel, index=":")) == 2


@pytest.mark.parametrize(
    "cycle_entry_point", ["sampler", "evaluation", "training"]
)
def test_from_actors_matches_legacy_entry_points(tmp_path, cycle_entry_point):
    calchome = tmp_path / "calc"
    sampler = _Actor(tmp_path, "sampler")
    evaluator = _Actor(tmp_path, "evaluator")
    trainer = _Actor(tmp_path, "trainer", ext="pkl")

    handler = StagedActiveLearningHandler.from_actors(
        sampler=sampler,
        evaluator=evaluator,
        trainer=trainer,
        calchome=calchome,
        num_iterations=4,
        cycle_entry_point=cycle_entry_point,
        initial_trainer_out=_seed(tmp_path, "model.pkl"),
        initial_sampler_out=_seed(tmp_path, "samples.extxyz"),
        initial_evaluator_out=_seed(tmp_path, "db.extxyz"),
    )
    handler.run()
    # Completes and writes a progress file regardless of entry point.
    assert (calchome / "AL_progress.json").exists()


class _TopNSelector(Selector):
    """A custom first-class Selector actor (not built on ConfigSelector)."""

    def __init__(self, n_configs: int):
        self.n_configs = n_configs
        self._output = None

    def update(self, calcdir, selector_input, **kwargs):
        n = kwargs.get("n_configs", self.n_configs)
        frames = ase.io.read(selector_input, index=":")
        out_dir = Path(calcdir) / "selector"
        out_dir.mkdir(parents=True, exist_ok=True)
        self._output = out_dir / "selected.extxyz"
        ase.io.write(self._output, frames[:n])

    def get_selected_configs(self):
        return self._output


def test_selector_abc_is_implemented_by_selector_actor():
    assert issubclass(SelectorActor, Selector)


def test_custom_selector_actor_as_stage(tmp_path):
    calchome = tmp_path / "calc"

    class _PoolSampler:
        def __init__(self, pool):
            self.pool = pool

        def update(self, calcdir, actor_input, **kwargs):
            pass

        def get_output(self):
            return self.pool

    pool = _write_pool(tmp_path / "pool.extxyz", 10)
    stages = [
        Stage(
            "sampler",
            _PoolSampler(pool),
            initial_input=_seed(tmp_path, "model.pkl"),
        ),
        Stage("selector", _TopNSelector(n_configs=4)),
        Stage("evaluator", _Actor(tmp_path, "evaluator")),
    ]
    StagedActiveLearningHandler(
        stages, calchome=calchome, num_iterations=1, exit_stage="evaluator"
    ).run()

    data = json.loads((calchome / "AL_progress.json").read_text())
    sel_rel = data["outputs"]["selector"]["0"][0]
    assert len(ase.io.read(calchome / sel_rel, index=":")) == 4


def test_from_actors_accepts_selector_actor_directly(tmp_path):
    calchome = tmp_path / "calc"

    class _PoolSampler:
        def __init__(self, pool):
            self.pool = pool

        def update(self, calcdir, actor_input, **kwargs):
            pass

        def get_sampled_configs(self):
            return self.pool

    pool = _write_pool(tmp_path / "pool.extxyz", 10)
    handler = StagedActiveLearningHandler.from_actors(
        sampler=_PoolSampler(pool),
        evaluator=_Actor(tmp_path, "evaluator"),
        trainer=_Actor(tmp_path, "trainer", ext="pkl"),
        selector=_TopNSelector(n_configs=5),
        calchome=calchome,
        num_iterations=1,
        cycle_exit_point="training",
        initial_trainer_out=_seed(tmp_path, "model.pkl"),
    )
    # The provided Selector actor is used as-is (not re-wrapped).
    selector_stage = next(s for s in handler.stages if s.name == "selector")
    assert isinstance(selector_stage.actor, _TopNSelector)
    handler.run()

    data = json.loads((calchome / "AL_progress.json").read_text())
    sel_rel = data["outputs"]["selector"]["0"][0]
    assert len(ase.io.read(calchome / sel_rel, index=":")) == 5


class _RefRecordingSelector(ConfigSelector):
    """A QUESTS-like selector recording the reference size per select call.

    ``ref_calls`` holds, for each ``select`` invocation, the number of
    reference configs passed as the ``reference=`` argument (``None`` when no
    reference is supplied, i.e. the diversity-mode cold-start fallback). The
    reference is an argument now -- the ``set_reference`` mutator was removed.
    """

    def __init__(self, n_configs: int):
        self.n_configs = n_configs
        self.ref_calls: list = []

    def select(self, configs, reference=None):
        flat = (
            [a for g in configs for a in g]
            if configs and isinstance(configs[0], list)
            else list(configs)
        )
        self.ref_calls.append(None if reference is None else len(reference))
        return flat[: self.n_configs]


class _AtomsEvaluator:
    """Evaluator writing a real extxyz with ``n_frames`` per iteration."""

    def __init__(self, n_frames: int = 2):
        self.n_frames = n_frames
        self._out = None

    def update(self, calcdir, actor_input, **kwargs):
        frames = [
            Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
            for _ in range(self.n_frames)
        ]
        self._out = Path(calcdir) / "eval.extxyz"
        ase.io.write(self._out, frames)

    def get_evaluated_configs(self):
        return self._out


class _PoolSampler:
    def __init__(self, pool: Path):
        self.pool = pool

    def update(self, calcdir, actor_input, **kwargs):
        pass

    def get_sampled_configs(self):
        return self.pool


def _novelty_stages(tmp_path, selector, reference_initial=None):
    pool = _write_pool(tmp_path / "pool.extxyz", 6)
    return [
        Stage(
            "sampler",
            _PoolSampler(pool),
            initial_input=_seed(tmp_path, "model.pkl"),
        ),
        Stage(
            "selector",
            SelectorActor(selector),
            extra_inputs={
                "reference": InputSource(
                    "evaluator", initial=reference_initial
                )
            },
        ),
        Stage("evaluator", _AtomsEvaluator(n_frames=2)),
    ]


def test_novelty_reference_grows_across_iterations(tmp_path):
    selector = _RefRecordingSelector(n_configs=2)
    stages = _novelty_stages(tmp_path, selector)

    StagedActiveLearningHandler(
        stages,
        calchome=tmp_path / "calc",
        num_iterations=3,
        exit_stage="evaluator",
    ).run()

    # it0: no prior evaluator data -> diversity fallback (None).
    # it1: evaluator output of it0 (2). it2: it0+it1 (4).
    assert selector.ref_calls == [None, 2, 4]


def test_novelty_cold_start_seed_enables_first_iteration(tmp_path):
    ref_seed = _write_pool(tmp_path / "ref_seed.extxyz", 3)
    selector = _RefRecordingSelector(n_configs=2)
    stages = _novelty_stages(tmp_path, selector, reference_initial=ref_seed)

    StagedActiveLearningHandler(
        stages,
        calchome=tmp_path / "calc",
        num_iterations=3,
        exit_stage="evaluator",
    ).run()

    # Seed (3) is always included; evaluator outputs accumulate on top.
    assert selector.ref_calls == [3, 5, 7]


def test_novelty_reference_resumes_from_progress(tmp_path):
    calchome = tmp_path / "calc"
    selector = _RefRecordingSelector(n_configs=2)
    stages = _novelty_stages(tmp_path, selector)

    StagedActiveLearningHandler(
        stages, calchome=calchome, num_iterations=2, exit_stage="evaluator"
    ).run()
    assert selector.ref_calls == [None, 2]

    # Extend the loop: iteration 2 must read the accumulated reference
    # rebuilt from the persisted evaluator outputs (it0 + it1 = 4).
    StagedActiveLearningHandler(
        stages, calchome=calchome, num_iterations=3, exit_stage="evaluator"
    ).run()
    assert selector.ref_calls == [None, 2, 4]


def test_unknown_extra_input_stage_raises(tmp_path):
    stages = [
        Stage(
            "sampler",
            _Actor(tmp_path, "sampler"),
            initial_input=_seed(tmp_path, "model.pkl"),
        ),
        Stage(
            "selector",
            SelectorActor(StrideSelector(n_configs=2)),
            extra_inputs={"reference": InputSource("nope")},
        ),
    ]
    with pytest.raises(WorkflowError, match="unknown stage"):
        StagedActiveLearningHandler(stages, calchome=tmp_path / "c")


def test_from_actors_wires_novelty_reference(tmp_path):
    calchome = tmp_path / "calc"
    selector = _RefRecordingSelector(n_configs=2)

    handler = StagedActiveLearningHandler.from_actors(
        sampler=_PoolSampler(_write_pool(tmp_path / "pool.extxyz", 6)),
        evaluator=_AtomsEvaluator(n_frames=2),
        trainer=_Actor(tmp_path, "trainer", ext="pkl"),
        selector=selector,
        selector_reference_from="evaluator",
        calchome=calchome,
        num_iterations=2,
        cycle_exit_point="training",
        initial_trainer_out=_seed(tmp_path, "model.pkl"),
    )
    sel_stage = next(s for s in handler.stages if s.name == "selector")
    assert "reference" in sel_stage.extra_inputs
    assert sel_stage.extra_inputs["reference"].stage == "evaluator"

    handler.run()
    assert selector.ref_calls == [None, 2]


def test_directory_bundle_output_is_collected(tmp_path):
    """A stage emitting a directory bundle (e.g. a mace-jax model) is
    collected via copytree, not shutil.copy.
    """
    calchome = tmp_path / "calc"
    bundle = tmp_path / "Model-jax"
    bundle.mkdir()
    (bundle / "config.json").write_text("{}")
    (bundle / "params.msgpack").write_text("params")

    class _BundleTrainer:
        def update(self, calcdir, actor_input, **kwargs):
            pass

        def get_output(self):
            return bundle

    stages = [
        Stage(
            "sampler",
            _Actor(tmp_path, "sampler"),
            initial_input=_seed(tmp_path, "model.pkl"),
        ),
        Stage("trainer", _BundleTrainer(), output_extension="bundle"),
    ]
    StagedActiveLearningHandler(
        stages, calchome=calchome, num_iterations=1
    ).run()

    data = json.loads((calchome / "AL_progress.json").read_text())
    dest = calchome / data["outputs"]["trainer"]["0"][0]
    assert dest.is_dir()
    assert (dest / "config.json").exists()
    assert (dest / "params.msgpack").exists()


def test_directory_bundle_seed_is_ingested(tmp_path):
    """A directory bundle passed as initial_input is moved in via copytree."""
    calchome = tmp_path / "calc"
    bundle = tmp_path / "foundation-jax"
    bundle.mkdir()
    (bundle / "config.json").write_text("{}")
    (bundle / "params.msgpack").write_text("params")

    seen: list[bool] = []

    class _DirConsumer:
        def update(self, calcdir, actor_input, **kwargs):
            seen.append(Path(actor_input).is_dir())

        def get_output(self):
            return _seed(tmp_path, "out.xyz")

    stages = [Stage("sampler", _DirConsumer(), initial_input=bundle)]
    StagedActiveLearningHandler(
        stages, calchome=calchome, num_iterations=1
    ).run()

    # The sampler received the bundle as an existing directory.
    assert seen == [True]


def test_resolve_actor_output_legacy_getter(tmp_path):
    actor = _LegacyGetterActor(tmp_path, "sampler")
    assert resolve_actor_output(actor) == actor.out


def test_resolve_actor_output_missing_getter_raises(tmp_path):
    class _NoGetter:
        def update(self, *a, **k):
            pass

    with pytest.raises(WorkflowError, match="get_output"):
        resolve_actor_output(_NoGetter())


def test_stage_hooks_fire_after_each_recorded_step(tmp_path):
    calchome = tmp_path / "calc"
    stages, _ = _make_stages(tmp_path)
    events: list[StageCompleted] = []

    StagedActiveLearningHandler(
        stages,
        calchome=calchome,
        num_iterations=2,
        exit_stage="sampler",
        on_stage_complete=[events.append],
    ).run()

    assert [(e.stage, e.iteration) for e in events] == [
        ("sampler", 0),
        ("evaluator", 0),
        ("trainer", 0),
        ("sampler", 1),
    ]
    assert [e.is_last for e in events] == [False, False, False, True]
    # Outputs are the recorded copies, already on disk when the hook runs.
    for e in events:
        assert e.calchome == calchome
        assert e.outputs and all(p.exists() for p in e.outputs)
        assert all(p.parent.name == f"{e.stage}_output" for p in e.outputs)


def test_failing_hook_does_not_stop_the_loop(tmp_path):
    calchome = tmp_path / "calc"
    stages, (sampler, evaluator, trainer) = _make_stages(tmp_path)
    seen: list[str] = []

    def boom(event):
        raise RuntimeError("dashboard submit failed")

    StagedActiveLearningHandler(
        stages,
        calchome=calchome,
        num_iterations=2,
        on_stage_complete=[boom, lambda e: seen.append(e.stage)],
    ).run()

    # Every step ran, and the hook after the failing one still fired.
    assert len(trainer.calls) == 2
    assert len(seen) == 6


def test_resume_does_not_refire_hooks_for_done_steps(tmp_path):
    calchome = tmp_path / "calc"
    stages, _ = _make_stages(tmp_path)
    StagedActiveLearningHandler(
        stages, calchome=calchome, num_iterations=1
    ).run()

    events: list[StageCompleted] = []
    stages, _ = _make_stages(tmp_path)
    StagedActiveLearningHandler(
        stages,
        calchome=calchome,
        num_iterations=2,
        on_stage_complete=[events.append],
    ).run()

    assert {e.iteration for e in events} == {1}


def test_from_actors_forwards_stage_hooks(tmp_path):
    events: list[StageCompleted] = []
    handler = StagedActiveLearningHandler.from_actors(
        _Actor(tmp_path, "sampler"),
        _Actor(tmp_path, "evaluator"),
        _Actor(tmp_path, "trainer", ext="pkl"),
        calchome=tmp_path / "calc",
        num_iterations=1,
        initial_trainer_out=_seed(tmp_path, "model.pkl"),
        on_stage_complete=[events.append],
    )
    handler.run()
    assert [e.stage for e in events] == ["sampler"]
