"""Stage-based active learning handler (alternative implementation).

This is an *alternative* to :class:`ActiveLearningHandler` in
``active_learning_base.py``. The original handler hardcodes the
three-actor (sampler / evaluator / trainer) loop: its bookkeeping is a
fixed ``[sample, eval, train]`` done-flag triple and every actor's
config is tripled by hand in ``__init__``. Adding a fourth actor (e.g. a
configuration *selector* between sampling and evaluation) means widening
that triple and extending the hand-enumerated resume logic.

This module models the loop instead as an ordered list of uniform
:class:`Stage` objects, each wrapping an *actor* that satisfies the
:class:`ALActor` contract (``update(calcdir, actor_input, **kwargs)`` plus
an output getter). The loop, the file bookkeeping and the crash-resume
logic are all generic over the stage list, so adding the selector is just
inserting one ``Stage`` -- no changes to the orchestration code.

The original ``ActiveLearningHandler`` is left completely untouched; both
implementations coexist.

Data flow
---------
The stages form a ring: stage ``N`` consumes the output of stage ``N-1``
in the same iteration, and the first stage consumes the last stage's
output from the *previous* iteration. A stage's ``combine`` flag is a
*producer-side* setting: when true, that stage's consumer receives the
stage's outputs from *all* completed iterations (plus the consumer's
seed), rather than only the most recent one. This mirrors the
``combine_*_outputs`` semantics of the original handler.

Each stage may carry an ``initial_input`` *seed* -- the data fed into it
before its producer has run (e.g. the initial model handed to the sampler
on iteration 0). The entry stage of a fresh run must have a seed.

Beyond the ring, a stage may declare ``extra_inputs`` -- named secondary
streams pulled from *any* other stage's outputs (see :class:`InputSource`),
optionally accumulated across iterations. This turns the ring into a small
DAG and is what enables, for example, *novelty* selection: a selector
receives the evaluator's accumulated outputs from all previous iterations
as a ``reference`` and scores candidates by how novel they are relative to
the already-labelled data.

Resume
------
Progress is tracked as a cursor, not an enumerated flag combination: the
handler precomputes a deterministic *plan* of ``(iteration, stage)`` work
items from ``num_iterations`` and the entry/exit stages, then records each
completed item's outputs in ``AL_progress.json``. Resuming re-runs only
the plan items not yet recorded. ``entry_stage``/``exit_stage`` are fixed
at the first run and persisted; ``num_iterations`` may be increased on a
later run to extend the loop.
"""

import json
import logging
import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import ase.io
from ase import Atoms

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import FileSystemHandlerBase
from remote_handler.core.tasks import SingleScriptTask
from remote_handler.exceptions import WorkflowError
import remote_handler.workflows.selectors as selectors_mod
from remote_handler.workflows.active_learning_base import Schedule, Selector
from remote_handler.workflows.selectors import ConfigSelector

logger = logging.getLogger(__name__)

PROGRESS_FILENAME = "AL_progress.json"

# Remote payloads shipped with the package live in ``scripts/`` next to this
# module's package root (mirrors ``samplers.py``'s ``_SCRIPTS_DIR``).
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"

# Output getter names understood by the handler, in priority order. This
# lets the existing Sampler/GroundTruthEvaluator/ModelTrainer actors be
# used as stage actors *unchanged*, while new actors can simply implement
# ``get_output``.
_OUTPUT_GETTERS = (
    "get_output",
    "get_sampled_configs",
    "get_evaluated_configs",
    "get_trained_model",
    "get_selected_configs",
)


def resolve_actor_output(actor: object) -> Path | list[Path]:
    """Return an actor's output by trying the known getter names.

    Args:
        actor: Any stage actor.

    Returns:
        The path(s) produced by the actor.

    Raises:
        WorkflowError: If the actor exposes none of the known getters.
    """
    for name in _OUTPUT_GETTERS:
        getter = getattr(actor, name, None)
        if callable(getter):
            return getter()
    raise WorkflowError(
        f"Actor {actor!r} exposes none of the expected output getters "
        f"{_OUTPUT_GETTERS}. Implement 'get_output'."
    )


class SelectorActor(Selector):
    """Adapt a :class:`ConfigSelector` strategy into a `Selector` actor.

    Turns the pure ``select(configs) -> list[Atoms]`` interface into the
    ``update``/``get_selected_configs`` actor contract so a selector can
    sit between the sampler and the evaluator as a first-class stage. Each
    input file is treated as one *group* (e.g. one trajectory), so
    grouping-aware selectors such as ``LastFrameSelector`` keep working;
    flat selectors flatten automatically.

    Args:
        selector: The selection strategy to apply.
        output_filename: Name of the written extxyz file inside the
            stage's calcdir.
    """

    def __init__(
        self,
        selector: ConfigSelector,
        output_filename: str = "selected.extxyz",
    ):
        self.selector = selector
        self.output_filename = output_filename
        self._output: Optional[Path] = None

    def update(
        self,
        calcdir: Path,
        actor_input: Path | list[Path],
        reference: Optional[list[Path]] = None,
        **kwargs,
    ) -> None:
        paths = (
            [actor_input]
            if isinstance(actor_input, (str, Path))
            else list(actor_input)
        )
        grouped: list[list[Atoms]] = []
        for p in paths:
            frames = ase.io.read(p, index=":")
            if isinstance(frames, Atoms):
                frames = [frames]
            grouped.append(frames)

        selected = self.selector.select(
            grouped, reference=self._resolve_reference(reference)
        )

        out_dir = Path(calcdir) / "selector"
        os.makedirs(out_dir, exist_ok=True)
        out_path = out_dir / self.output_filename
        ase.io.write(out_path, selected)
        self._output = out_path
        logger.info(
            "SelectorActor: wrote %d selected configs to %s",
            len(selected),
            out_path,
        )

    def _resolve_reference(
        self, reference: Optional[list[Path]]
    ) -> Optional[list[Atoms]]:
        """Read reference config files into a flat ``list[Atoms]`` (or None).

        The result is passed to ``selector.select(grouped, reference=...)`` as
        an argument -- not via a mutator -- so a novelty selector
        (``QUESTSSelector``) scores against it while selectors that take no
        reference (``StrideSelector``) just ignore it, and a composed
        ``PipelineSelector`` forwards it to its members. An empty/absent
        reference yields None, so a novelty selector transparently falls back
        to diversity on the cold-start iteration.
        """
        if not reference:
            return None
        ref_atoms: list[Atoms] = []
        for p in reference:
            frames = ase.io.read(p, index=":")
            if isinstance(frames, Atoms):
                frames = [frames]
            ref_atoms.extend(frames)
        return ref_atoms or None

    def get_selected_configs(self) -> Path:
        if self._output is None:
            raise WorkflowError("SelectorActor has not produced output yet.")
        return self._output


class RemoteSelector(Selector):
    """Run a `ConfigSelector` strategy as a single remote SLURM job.

    Peer to `SelectorActor`, which runs the same `ConfigSelector.select()`
    call in-process on whatever host drives the loop. Use `RemoteSelector`
    instead when selection is heavy enough (or needs a package only
    available in a cluster env) to warrant its own job -- it ships the
    candidate pool (+ optional reference) and the selector's parameters to
    `cluster_config`, runs `scripts/selection/select_configs.py` there via a
    `SingleScriptTask`, and collects the written extxyz back. Mirrors the
    pattern used by `samplers.CMAESSampler` for a one-shot remote job.

    Args:
        selector_type: Name of a `ConfigSelector` subclass in
            `remote_handler.workflows.selectors` (e.g. "QUESTSSelector").
        selector_kwargs: Keyword arguments forwarded to the selector's
            constructor (must be JSON-plain values).
        cluster_config: Cluster to submit the selection job to.
        remote_fs: Filesystem handler used to ship files to/from that
            cluster (a `LocalFileSystem` when the driving process already
            runs on the cluster's login node, as with `VASPEvaluator`).
        output_filename: Name of the extxyz file the remote script writes.
        remote_savepath: Remote scratch root for the job (defaults to the
            executor's own default when None).
        loop_time: Seconds between job-completion polls.
        stub: When True, run the selector in-process instead of submitting
            a job (for dry-running the loop's wiring without a cluster).
        jobname: SLURM job name.
    """

    def __init__(
        self,
        selector_type: str,
        selector_kwargs: dict,
        cluster_config: ClusterConfig,
        remote_fs: FileSystemHandlerBase,
        output_filename: str = "selected.extxyz",
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30.0,
        stub: bool = False,
        jobname: str = "select_configs",
    ):
        self.selector_type = selector_type
        self.selector_kwargs = selector_kwargs
        self.cluster_config = cluster_config
        self.remote_fs = remote_fs
        self.output_filename = output_filename
        self.remote_savepath = remote_savepath
        self.loop_time = loop_time
        self.stub = stub
        self.jobname = jobname
        self._output: Optional[Path] = None

    def update(
        self,
        calcdir: Path,
        actor_input: Path | list[Path],
        reference: Optional[list[Path]] = None,
        **kwargs,
    ) -> None:
        paths = (
            [actor_input]
            if isinstance(actor_input, (str, Path))
            else list(actor_input)
        )
        self.select_dir = Path(calcdir) / "selector"
        self.results_dir = self.select_dir / "results"
        os.makedirs(self.select_dir, exist_ok=True)

        if self.stub:
            selector_cls = getattr(selectors_mod, self.selector_type)
            selector = selector_cls(**{**self.selector_kwargs, **kwargs})
            grouped: list[list[Atoms]] = []
            for p in paths:
                frames = ase.io.read(p, index=":")
                grouped.append(
                    [frames] if isinstance(frames, Atoms) else frames
                )
            ref_atoms = None
            if reference:
                ref_atoms = []
                for p in reference:
                    frames = ase.io.read(p, index=":")
                    ref_atoms.extend(
                        [frames] if isinstance(frames, Atoms) else frames
                    )
                ref_atoms = ref_atoms or None
            selected = selector.select(grouped, reference=ref_atoms)
            os.makedirs(self.results_dir, exist_ok=True)
            self._output = self.results_dir / self.output_filename
            ase.io.write(self._output, selected)
            logger.info(
                "RemoteSelector (stub): wrote %d selected configs to %s",
                len(selected),
                self._output,
            )
            return

        script_input = {
            "selector_type": self.selector_type,
            "selector_kwargs": {**self.selector_kwargs, **kwargs},
            "input_files": [Path(p).name for p in paths],
            "reference_files": (
                [Path(p).name for p in reference] if reference else None
            ),
            "output_filename": self.output_filename,
        }
        self.script_input_file = self.select_dir / "input_select.json"
        with open(self.script_input_file, "w") as f:
            json.dump(script_input, f, indent=2)

        self.select_task = SingleScriptTask(
            local_savepath=self.results_dir,
            input_file=self.script_input_file,
            cluster_config=self.cluster_config,
            remote_fs=self.remote_fs,
            script_file=_SCRIPTS_DIR / "selection" / "select_configs.py",
            remote_savepath=self.remote_savepath,
            jobfile_name="run_select_configs.sh",
            jobname=self.jobname,
            logfile="select_configs.log",
            n_processors=1,
            loop_time=self.loop_time,
        )
        self.select_task.allocate_data()
        for p in paths:
            self.select_task.copy_file(p)
        for p in reference or []:
            self.select_task.copy_file(p)

    def get_selected_configs(self) -> Path:
        if not self.stub:
            self.select_task.submit()
            if not self.select_task.finish_check():
                raise WorkflowError(
                    "Remote selection did not finish successfully "
                    "(no is_finished.info marker). Check the job logs in "
                    f"{self.select_task.calcdir}."
                )
            self.select_task.collect_results()
            self._output = self.results_dir / self.output_filename

        if self._output is None:
            raise WorkflowError("RemoteSelector has not produced output yet.")
        logger.info(
            "RemoteSelector: selected configs collected to %s", self._output
        )
        return self._output


@dataclass
class InputSource:
    """A secondary input stream pulled from another stage's outputs.

    The primary input of a stage always follows the ring (its
    predecessor's output). An ``InputSource`` declares an *additional*,
    named stream sourced from an arbitrary stage -- e.g. the accumulated
    evaluator outputs handed to a selector as a *reference* for novelty
    scoring. The resolved paths are passed to the actor's ``update`` as a
    keyword argument named after the dict key in :attr:`Stage.extra_inputs`.

    Args:
        stage: Name of the producer stage to pull outputs from.
        accumulate: If true, gather that stage's outputs from all eligible
            iterations; otherwise only the most recent one.
        include_current: If true, also consider the producer's output from
            the current iteration. Defaults to false, so a stage pulls only
            from *previous* iterations (the producer typically runs later
            in the ring and has no current-iteration output yet).
        initial: Optional seed data used before the producer has run (e.g.
            an initial training set as the cold-start reference).
    """

    stage: str
    accumulate: bool = True
    include_current: bool = False
    initial: Optional[Path | list[Path]] = None


@dataclass
class Stage:
    """A single actor in the active learning pipeline.

    Args:
        name: Unique stage name; also used for output dirs and progress
            keys.
        actor: The actor object (``update`` + an output getter).
        combine: If true, this stage's *consumer* receives the stage's
            outputs from all completed iterations (plus the consumer's
            seed) instead of only the latest.
        schedule: Optional per-iteration kwargs source for the actor.
        output_extension: Extension for the stage's collected output
            files.
        initial_input: Seed data fed into this stage before its producer
            has run. Required for the entry stage of a fresh run.
        extra_inputs: Optional named secondary input streams (see
            :class:`InputSource`). Each is resolved to a list of paths and
            passed to ``update`` as a keyword argument of the same name.
            Keys must not collide with the stage's schedule kwargs.
    """

    name: str
    actor: object
    combine: bool = False
    schedule: Optional[Schedule] = None
    output_extension: str = "extxyz"
    initial_input: Optional[Path | list[Path]] = None
    extra_inputs: dict[str, InputSource] = field(default_factory=dict)


def _as_path_list(value: Optional[Path | list[Path]]) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(v) for v in value]


class StagedActiveLearningHandler:
    """Generic, stage-list active learning loop with crash resume.

    See the module docstring for the data-flow and resume model.

    Args:
        stages: Ordered pipeline of :class:`Stage` objects (length >= 1).
        calchome: Working directory for the loop.
        num_iterations: Number of iterations to run.
        entry_stage: Name of the stage where iteration 0 starts
            (defaults to the first stage). Stages before it are skipped on
            the first iteration only.
        exit_stage: Name of the stage where the final iteration stops
            (defaults to the last stage). Stages after it are skipped on
            the last iteration only.

    Use :meth:`from_actors` for a drop-in mapping of the original
    ``ActiveLearningHandler`` arguments.
    """

    def __init__(
        self,
        stages: list[Stage],
        calchome: os.PathLike,
        num_iterations: int = 5,
        entry_stage: Optional[str] = None,
        exit_stage: Optional[str] = None,
    ) -> None:
        if not stages:
            raise WorkflowError("StagedActiveLearningHandler needs >=1 stage.")
        names = [s.name for s in stages]
        if len(set(names)) != len(names):
            raise WorkflowError(f"Stage names must be unique, got {names}.")

        self.stages = stages
        self.calchome = Path(calchome)
        os.makedirs(self.calchome, exist_ok=True)
        self.progress_file = self.calchome / PROGRESS_FILENAME

        # Per-stage collected outputs: {stage_name: {iter_idx: [rel str]}}
        self.outputs: dict[str, dict[int, list[str]]] = {}

        # Seeds moved into calchome once, stored as relative path strings.
        self.seeds: dict[str, list[str]] = {}
        for stage in self.stages:
            seed_rel = [
                str(self._move_into_home(p))
                for p in _as_path_list(stage.initial_input)
            ]
            self.seeds[stage.name] = seed_rel

        # Seeds for secondary (extra_inputs) streams, keyed by
        # (consumer stage name, keyword).
        self.extra_seeds: dict[tuple[str, str], list[str]] = {}
        name_set = set(names)
        for stage in self.stages:
            for kw, src in stage.extra_inputs.items():
                if src.stage not in name_set:
                    raise WorkflowError(
                        f"Stage '{stage.name}' declares extra input '{kw}' "
                        f"from unknown stage '{src.stage}'. Known stages: "
                        f"{names}."
                    )
                self.extra_seeds[(stage.name, kw)] = [
                    str(self._move_into_home(p))
                    for p in _as_path_list(src.initial)
                ]

        # Output collection dirs, one per stage.
        self.stage_output_dirs: dict[str, Path] = {}
        for stage in self.stages:
            out_dir = self.calchome / f"{stage.name}_output"
            os.makedirs(out_dir, exist_ok=True)
            self.stage_output_dirs[stage.name] = out_dir

        resuming = self.progress_file.exists()
        if resuming:
            self._load_progress(num_iterations)
        else:
            self.entry_index = self._index_of(entry_stage, default=0)
            self.exit_index = self._index_of(
                exit_stage, default=len(self.stages) - 1
            )
            self.num_iterations = num_iterations
            self._validate_entry_seed()

        self.plan = self._build_plan()
        self._save_progress()

        n_remaining = sum(
            1 for it, si in self.plan if not self._is_done(it, si)
        )
        logger.info(
            "%s active learning: %d stage-steps to run (%d total in plan).",
            "Resuming" if resuming else "Starting",
            n_remaining,
            len(self.plan),
        )

    # ---- setup helpers ----------------------------------------------------

    def _index_of(self, name: Optional[str], default: int) -> int:
        if name is None:
            return default
        for i, stage in enumerate(self.stages):
            if stage.name == name:
                return i
        raise WorkflowError(
            f"Unknown stage '{name}'. Known stages: "
            f"{[s.name for s in self.stages]}."
        )

    def _validate_entry_seed(self) -> None:
        entry = self.stages[self.entry_index]
        if not self.seeds[entry.name]:
            raise WorkflowError(
                f"Entry stage '{entry.name}' has no producer on iteration 0; "
                "it requires an 'initial_input' seed for a fresh run."
            )

    def _move_into_home(self, file: Path) -> Path:
        """Copy an external seed file/dir into calchome; return rel path."""
        file = Path(file)
        if file.is_relative_to(self.calchome):
            return file.relative_to(self.calchome)
        dest = self.calchome / file.name
        if file.is_dir():
            shutil.copytree(file, dest, dirs_exist_ok=True)
        else:
            shutil.copy(file, dest)
        return dest.relative_to(self.calchome)

    def _build_plan(self) -> list[tuple[int, int]]:
        """Deterministic list of (iteration, stage_index) work items."""
        plan: list[tuple[int, int]] = []
        last_iter = self.num_iterations - 1
        for it in range(self.num_iterations):
            start = self.entry_index if it == 0 else 0
            stop = self.exit_index if it == last_iter else len(self.stages) - 1
            for si in range(start, stop + 1):
                plan.append((it, si))
        return plan

    # ---- progress persistence --------------------------------------------

    def _load_progress(self, num_iterations: int) -> None:
        with open(self.progress_file, "r") as f:
            data = json.load(f)
        logger.info("Continuing run from %s ...", self.progress_file)

        stored_names = data.get("stages")
        if stored_names != [s.name for s in self.stages]:
            raise WorkflowError(
                "Stored stage list does not match the configured stages: "
                f"{stored_names} != {[s.name for s in self.stages]}."
            )
        self.entry_index = data["entry_index"]
        self.exit_index = data["exit_index"]
        # num_iterations may only grow on resume.
        self.num_iterations = max(data["num_iterations"], num_iterations)
        self.outputs = {
            name: {int(it): paths for it, paths in iters.items()}
            for name, iters in data.get("outputs", {}).items()
        }

    def _save_progress(self) -> None:
        data = {
            "stages": [s.name for s in self.stages],
            "entry_index": self.entry_index,
            "exit_index": self.exit_index,
            "num_iterations": self.num_iterations,
            "outputs": {
                name: {str(it): paths for it, paths in iters.items()}
                for name, iters in self.outputs.items()
            },
        }
        with open(self.progress_file, "w") as f:
            json.dump(data, f, indent=2)

    def _is_done(self, iter_idx: int, stage_idx: int) -> bool:
        name = self.stages[stage_idx].name
        return iter_idx in self.outputs.get(name, {})

    # ---- data flow --------------------------------------------------------

    def _gather_input(self, stage_idx: int, iter_idx: int) -> list[str]:
        """Resolve the relative input paths for a stage at an iteration."""
        stage = self.stages[stage_idx]
        if stage_idx > 0:
            producer = self.stages[stage_idx - 1]
            producer_iter = iter_idx
        else:
            producer = self.stages[-1]
            producer_iter = iter_idx - 1

        seed = list(self.seeds[stage.name])
        recorded = self.outputs.get(producer.name, {})
        available = sorted(it for it in recorded if it <= producer_iter)

        if producer.combine:
            paths = list(seed)
            for it in available:
                paths.extend(recorded[it])
        elif available:
            paths = list(recorded[available[-1]])
        else:
            paths = list(seed)

        if not paths:
            raise WorkflowError(
                f"No input available for stage '{stage.name}' at iteration "
                f"{iter_idx} (producer '{producer.name}'). Provide an "
                "'initial_input' seed."
            )
        return paths

    def _gather_extra(
        self, stage_name: str, kw: str, src: InputSource, iter_idx: int
    ) -> list[str]:
        """Resolve a secondary input stream's relative paths.

        Unlike :meth:`_gather_input`, this pulls from an arbitrary named
        stage and may cross iteration boundaries (e.g. a selector reading
        all *previous* iterations' evaluator outputs).
        """
        cutoff = iter_idx if src.include_current else iter_idx - 1
        recorded = self.outputs.get(src.stage, {})
        available = sorted(it for it in recorded if it <= cutoff)
        seed = list(self.extra_seeds.get((stage_name, kw), []))

        if src.accumulate:
            paths = list(seed)
            for it in available:
                paths.extend(recorded[it])
        elif available:
            paths = list(recorded[available[-1]])
        else:
            paths = list(seed)
        return paths

    # ---- execution --------------------------------------------------------

    @contextmanager
    def _reattach_if_pending(self, actor, calcdir: Path):
        """Reattach ``actor`` to an already-submitted job for this stage
        step, if one exists, instead of letting it resubmit.

        Duck-typed counterpart of ``ActiveLearningHandler``'s helper of the
        same name -- stage actors here need not subclass the
        ``Sampler``/``GroundTruthEvaluator``/``ModelTrainer``/``Selector``
        ABCs at all (see ``resolve_actor_output``'s own duck typing), so
        ``pending_manifest``/``collect_only`` are looked up with
        ``getattr`` rather than assumed present. See that helper's
        docstring for the full rationale.
        """
        pending_manifest = getattr(actor, "pending_manifest", None)
        manifest = pending_manifest(calcdir) if pending_manifest else None
        reattach = (
            manifest is not None
            and manifest.exists()
            and hasattr(actor, "collect_only")
        )
        if reattach:
            logger.info(
                "Found a pending manifest for stage %r at %s -- "
                "reattaching instead of resubmitting.",
                type(actor).__name__,
                manifest,
            )
            original_collect_only = actor.collect_only
            actor.collect_only = True
        try:
            yield reattach
        finally:
            if reattach:
                actor.collect_only = original_collect_only

    def _run_stage(self, iter_idx: int, stage_idx: int) -> None:
        stage = self.stages[stage_idx]
        calcdir = self.calchome / str(iter_idx)
        os.makedirs(calcdir, exist_ok=True)

        rel_inputs = self._gather_input(stage_idx, iter_idx)
        abs_inputs = [self.calchome / p for p in rel_inputs]
        input_arg = abs_inputs[0] if len(abs_inputs) == 1 else abs_inputs

        kwargs = (
            stage.schedule.get_kwargs(iter_idx)
            if stage.schedule is not None
            else {}
        )

        extra: dict[str, list[Path]] = {}
        for kw, src in stage.extra_inputs.items():
            rel = self._gather_extra(stage.name, kw, src, iter_idx)
            extra[kw] = [self.calchome / p for p in rel]
        clash = set(extra) & set(kwargs)
        if clash:
            raise WorkflowError(
                f"Stage '{stage.name}' extra_inputs keys {sorted(clash)} "
                "collide with schedule kwargs."
            )

        logger.info("[it %d] running stage '%s' ...", iter_idx, stage.name)
        with self._reattach_if_pending(stage.actor, calcdir):
            stage.actor.update(calcdir, input_arg, **extra, **kwargs)
            outputs = resolve_actor_output(stage.actor)
        outputs = (
            [outputs] if isinstance(outputs, (str, Path)) else list(outputs)
        )

        out_dir = self.stage_output_dirs[stage.name]
        new_rel: list[str] = []
        for i, produced in enumerate(outputs):
            dest = (
                out_dir
                / f"{stage.name}_it{iter_idx}_{i}.{stage.output_extension}"
            )
            self._copy(Path(produced), dest)
            new_rel.append(str(dest.relative_to(self.calchome)))

        self.outputs.setdefault(stage.name, {})[iter_idx] = new_rel
        self._save_progress()

    @staticmethod
    def _copy(src: Path, dest: Path) -> None:
        os.makedirs(dest.parent, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            shutil.copy(src, dest)

    def run(self) -> None:
        """Run all plan items not already recorded as complete."""
        for iter_idx, stage_idx in self.plan:
            if self._is_done(iter_idx, stage_idx):
                continue
            self._run_stage(iter_idx, stage_idx)
        logger.info("Active learning loop finished.")

    # ---- compatibility constructor ---------------------------------------

    @classmethod
    def from_actors(
        cls,
        sampler: object,
        evaluator: object,
        trainer: object,
        *,
        calchome: os.PathLike,
        selector: Optional[ConfigSelector | Selector] = None,
        selector_reference_from: Optional[str] = None,
        initial_selector_reference: Optional[Path | list[Path]] = None,
        num_iterations: int = 5,
        cycle_entry_point: str = "sampler",
        cycle_exit_point: str = "sampler",
        initial_trainer_out: Optional[Path | list[Path]] = None,
        initial_sampler_out: Optional[Path | list[Path]] = None,
        initial_evaluator_out: Optional[Path | list[Path]] = None,
        sampler_schedule: Optional[Schedule] = None,
        evaluator_schedule: Optional[Schedule] = None,
        trainer_schedule: Optional[Schedule] = None,
        selector_schedule: Optional[Schedule] = None,
        combine_sampler_outputs: bool = False,
        combine_evaluator_outputs: bool = True,
        combine_trainer_outputs: bool = False,
        sampler_output_extension: str = "extxyz",
        evaluator_output_extension: str = "extxyz",
        trainer_output_extension: str = "pkl",
        selector_output_extension: str = "extxyz",
    ) -> "StagedActiveLearningHandler":
        """Build a handler from the classic three-actor arguments.

        Mirrors :class:`ActiveLearningHandler`'s parameters so existing
        scripts can switch with minimal changes, and optionally inserts a
        ``selector`` stage between the sampler and the evaluator.

        ``cycle_entry_point`` / ``cycle_exit_point`` accept the legacy
        spellings ``"sampler"``, ``"evaluation"`` and ``"training"``.

        For novelty-based selection, set ``selector_reference_from`` to a
        stage name (typically ``"evaluator"``): the selector then receives
        that stage's accumulated outputs from all *previous* iterations as a
        ``reference`` and switches to novelty mode (see
        :class:`QUESTSSelector`). ``initial_selector_reference`` optionally
        seeds the reference (e.g. the starting training set) so novelty is
        active from the first iteration; otherwise iteration 0 falls back to
        diversity.
        """
        entry_map = {
            "sampler": "sampler",
            "evaluation": "evaluator",
            "training": "trainer",
        }
        if cycle_entry_point not in entry_map:
            raise WorkflowError(
                f"Unknown cycle_entry_point '{cycle_entry_point}'."
            )
        if cycle_exit_point not in entry_map:
            raise WorkflowError(
                f"Unknown cycle_exit_point '{cycle_exit_point}'."
            )

        stages = [
            Stage(
                name="sampler",
                actor=sampler,
                combine=combine_sampler_outputs,
                schedule=sampler_schedule,
                output_extension=sampler_output_extension,
                initial_input=initial_trainer_out,
            )
        ]
        if selector is not None:
            # A pure ConfigSelector strategy is wrapped into a Selector
            # actor; a ready-made Selector actor is used as-is.
            selector_actor = (
                SelectorActor(selector)
                if isinstance(selector, ConfigSelector)
                else selector
            )
            selector_extra: dict[str, InputSource] = {}
            if selector_reference_from is not None:
                selector_extra["reference"] = InputSource(
                    stage=selector_reference_from,
                    accumulate=True,
                    include_current=False,
                    initial=initial_selector_reference,
                )
            stages.append(
                Stage(
                    name="selector",
                    actor=selector_actor,
                    combine=False,
                    schedule=selector_schedule,
                    output_extension=selector_output_extension,
                    extra_inputs=selector_extra,
                )
            )
        stages.append(
            Stage(
                name="evaluator",
                actor=evaluator,
                combine=combine_evaluator_outputs,
                schedule=evaluator_schedule,
                output_extension=evaluator_output_extension,
                initial_input=initial_sampler_out,
            )
        )
        stages.append(
            Stage(
                name="trainer",
                actor=trainer,
                combine=combine_trainer_outputs,
                schedule=trainer_schedule,
                output_extension=trainer_output_extension,
                initial_input=initial_evaluator_out,
            )
        )

        return cls(
            stages=stages,
            calchome=calchome,
            num_iterations=num_iterations,
            entry_stage=entry_map[cycle_entry_point],
            exit_stage=entry_map[cycle_exit_point],
        )
