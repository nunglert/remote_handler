"""
Trainers for Active Learning Workflows.

Trainers fit machine learning models to evaluated configurations.
"""

import json
import logging
import os
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from remote_handler.core import remote_env
from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import FileSystemHandlerBase
from remote_handler.core.tasks import (
    SHARED_RESOURCE_PREFIX,
    ArrayScriptTask,
    DetachableTaskMixin,
    DictJSON,
    SingleScriptTask,
)
from remote_handler.core.utils import grep
from remote_handler.exceptions import ConfigurationError, WorkflowError

from .active_learning_base import ModelTrainer

logger = logging.getLogger(__name__)

# Remote payloads shipped with the package live in ``scripts/`` next to this
# module's package root. Each trainer defaults its ``training_script`` to the
# shipped payload it was written for, so the path need not be passed in.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _resolve_mace_train_runner(remote_fs, cluster_config):
    """Locate MACE's ``run_train.py`` in the *remote* conda env, or raise.

    Used when a trainer is given no explicit ``remote_mace_train_runner``: the
    path lives on the remote (the host is dependency-light), so it is probed
    there rather than imported locally.
    """
    path = remote_env.resolve_remote_module_file(
        remote_fs,
        getattr(cluster_config, "conda_source", ""),
        getattr(cluster_config, "conda_env", ""),
        "mace.cli.run_train",
    )
    if not path:
        env = getattr(cluster_config, "conda_env", "") or "(ambient)"
        raise ConfigurationError(
            "Could not locate MACE's run_train.py in the remote conda env "
            f"'{env}'. Install mace there, or pass remote_mace_train_runner "
            "explicitly."
        )
    logger.info("Resolved remote MACE run_train.py: %s", path)
    return path


def _resolve_executable_train_runner(
    remote_fs, cluster_config, executable, extra_name
):
    """Locate a CLI training tool's executable in the *remote* conda env.

    The CLI-tool counterpart of :func:`_resolve_mace_train_runner`: ACE
    (``pacemaker``) and GRACE (``gracemaker``) ship as bare console scripts
    rather than importable python modules, so their runner is probed via
    :func:`remote_env.resolve_remote_executable` instead of a module import.

    Args:
        executable: The console script's name (e.g. ``"pacemaker"``).
        extra_name: This package's optional-dependency extra that installs
            it (e.g. ``"ace"``), named in the error message.
    """
    path = remote_env.resolve_remote_executable(
        remote_fs,
        getattr(cluster_config, "conda_source", ""),
        getattr(cluster_config, "conda_env", ""),
        executable,
    )
    if not path:
        env = getattr(cluster_config, "conda_env", "") or "(ambient)"
        raise ConfigurationError(
            f"Could not locate '{executable}' in the remote conda env "
            f"'{env}'. Install it there (e.g. pip install '.[{extra_name}]'"
            "), or pass the runner path explicitly."
        )
    logger.info("Resolved remote %s: %s", executable, path)
    return path


class _EnsembleArrayTrainer(DetachableTaskMixin, ModelTrainer):
    """Shared machinery for training an ensemble of models via array jobs.

    Common base for every array-job trainer this package ships (MACE, ACE,
    GRACE): submit ``num_models`` independently-seeded jobs, poll, collect
    one artifact per job from ``calcdirs/*/models/`` (skipping non-job
    entries). What differs between tools is expressed entirely as class
    attributes / hook methods below -- how the training executable is
    resolved (:meth:`_resolve_train_runner`), extra per-job input/resources
    (:meth:`_extra_shared_input`/:meth:`_extra_shared_resources`), the
    per-job artifact name and how a stub one is written
    (:meth:`_artifact_name`/:meth:`_serialize_stub_model`).

    ``_MaceArrayTrainer`` layers the optional torch -> mace-jax conversion
    and MACE's module-import train-runner resolution on top; ``_AceArrayTrainer``
    /``_GraceArrayTrainer`` layer their own tool-specific hooks. All three
    share everything else -- data prep, array submission, finish-check,
    collection, stub scaffolding -- from here, once.
    """

    # Per-subclass constants -------------------------------------------------
    _script_name = "training.py"  # shipped payload; subclasses override
    _script_subdir = ""  # e.g. "mace", "ace", "grace"; subclasses override
    _jobname = "train"  # SLURM job name
    _tool_label = "Ensemble"  # e.g. "MACE", "ACE", "GRACE"; log/error text
    _action = "training"  # "{tool_label} {action} did not finish ..."
    _participle = "trained"  # "No models were successfully {participle}"
    _iteration_subdir = "train"  # calcdir/train -- see DetachableTaskMixin

    def __init__(
        self,
        remote_fs: FileSystemHandlerBase,
        cluster_config: ClusterConfig,
        config_file: os.PathLike,
        training_script: Optional[os.PathLike] = None,
        remote_train_runner: Optional[os.PathLike] = None,
        num_models: int = 5,
        validation_fraction: float = 0.1,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        stub: bool = False,
        n_processors: int = 1,
        collect_only: bool = False,
        reattach_max_wait: Optional[float] = None,
    ):
        super().__init__()
        if training_script is None:
            training_script = (
                _SCRIPTS_DIR / self._script_subdir / self._script_name
            )
        self.training_script = training_script
        self.remote_fs = remote_fs
        self.cluster_config = cluster_config
        self.config_file = config_file
        self.num_models = num_models
        self.validation_fraction = validation_fraction
        self.remote_train_runner = remote_train_runner
        self.remote_savepath = remote_savepath
        self.loop_time = loop_time
        self.stub = stub
        # Resources per training job; for GPU templates this is the GPU count
        # (``--gres=gpu:n``). One model per job -> one GPU by default.
        self.n_processors = n_processors
        self.model_name = "Model"  # Default model name
        self.collect_only = collect_only
        self.reattach_max_wait = reattach_max_wait

    # Hooks subclasses override --------------------------------------------
    def _resolve_train_runner(self) -> str:
        """Locate the remote training executable. Implemented per tool."""
        raise NotImplementedError

    def _extra_shared_input(self) -> dict:
        """Extra keys merged into the remote script's shared input JSON."""
        return {}

    def _extra_shared_resources(self) -> list:
        """Extra files staged into the shared resource dir (after config)."""
        return []

    def _artifact_name(self) -> str:
        """Per-job filename/dirname, under ``models/``, handed to the next
        sampler. Implemented per tool."""
        raise NotImplementedError

    def _stub_model_dict(self) -> dict:
        """Mock model payload written by stub mode."""
        return {
            "model_type": self._tool_label,
            "seed": np.random.randint(1e6),
            "training_loss": np.random.uniform(0.01, 0.05),
            "validation_loss": np.random.uniform(0.02, 0.08),
        }

    def _serialize_stub_model(self, model_dict: dict, path: Path) -> None:
        """Write the stub model payload to ``path``. Default: pickle it.

        Overridden where the real artifact isn't a pickle (ACE's is a YAML
        file, GRACE's is a directory) or needs extra side effects (MACE's
        optional mace-jax conversion).
        """
        with open(path, "wb") as f:
            pickle.dump(model_dict, f)

    # Shared implementation -------------------------------------------------
    def update(
        self,
        calcdir: Path,
        trainer_input: Optional[Path | list[Path]],
        **kwargs,
    ) -> None:
        """Prepare the trainer for a new iteration.

        Args:
            calcdir: Directory for this AL iteration
            trainer_input: Path(s) to database files with training data. Not
                read when ``collect_only`` -- a second process reconstructing
                an already-submitted array from its manifest needs none of
                the original training data, so ``None`` is expected there.
            **kwargs: Additional parameters
        """
        self.train_dir = self.iteration_dir(calcdir)
        self.results_dir = self.results_dir_for(calcdir)
        os.makedirs(self.train_dir, exist_ok=True)

        if not self.collect_only:
            self.db_files = (
                [trainer_input]
                if isinstance(trainer_input, Path)
                else trainer_input
            )

        # In stub mode, we don't need to create the task
        if self.stub:
            return

        def _build_training_task() -> ArrayScriptTask:
            # Auto-resolve the remote training executable if not given.
            if self.remote_train_runner is None:
                self.remote_train_runner = self._resolve_train_runner()

            # Shared resources live in rootpath/, but scripts run in a
            # subdir of it; reference them via SHARED_RESOURCE_PREFIX so
            # each script resolves them relative to its own calcpath.
            script_input_shared = {
                "database_files": [
                    f"{SHARED_RESOURCE_PREFIX}/{Path(p).name}"
                    for p in self.db_files
                ],
                "config_file": (
                    f"{SHARED_RESOURCE_PREFIX}/{Path(self.config_file).name}"
                ),
                "model_name": self.model_name,
                "validation_fraction": self.validation_fraction,
                "train_runner": Path(self.remote_train_runner).as_posix(),
                **self._extra_shared_input(),
            }

            # Create individual input files with different seeds
            individual_resources = [
                [
                    DictJSON(
                        content=script_input_shared
                        | {"seed": np.random.randint(1e6)},
                        filename="input_train.json",
                        remote_fs=self.remote_fs,
                    )
                    for i in range(self.num_models)
                ]
            ]

            return ArrayScriptTask(
                cluster_config=self.cluster_config,
                local_savepath=self.results_dir,
                remote_fs=self.remote_fs,
                python_script_file=self.training_script,
                individual_ressources=individual_resources,
                shared_ressources=(
                    [self.config_file]
                    + self._extra_shared_resources()
                    + self.db_files
                ),
                logfile="train.log",
                remote_savepath=self.remote_savepath,
                collect_only=False,
                num_parallel=min(self.num_models, 5),
                loop_time=self.loop_time,
                jobname=self._jobname,
                n_processors=self.n_processors,
            )

        self.training_task = self.build_or_resume_array_task(
            self.results_dir,
            _build_training_task,
            python_script_file=self.training_script,
        )

    def _create_stub_outputs(self) -> list[Path]:
        """Create mock per-job model artifacts for testing."""
        logger.info(
            f"Creating {self.num_models} stub {self._tool_label} models"
        )

        calcdirs = self.results_dir / "calcdirs"
        os.makedirs(calcdirs, exist_ok=True)

        model_paths = []
        for i in range(self.num_models):
            model_dir = calcdirs / f"job_{i}" / "models"
            os.makedirs(model_dir, exist_ok=True)

            artifact_path = model_dir / self._artifact_name()
            mock_model = self._stub_model_dict()
            self._serialize_stub_model(mock_model, artifact_path)
            model_paths.append(artifact_path)
            logger.info(f"Created stub model: {artifact_path}")

        return model_paths

    def get_trained_model(self) -> list[Path]:
        """Run the array job and collect the per-model artifacts.

        Raises:
            TaskError: If the job left no ``is_finished.info`` marker (or,
                when ``collect_only``, if that doesn't happen within
                ``reattach_max_wait``).
            WorkflowError: If no model artifacts were produced.
        """
        if self.stub:
            logger.info(
                f"Running in stub mode - creating mock {self._tool_label} models"
            )
            return self._create_stub_outputs()

        self.submit_if_needed(self.training_task)
        self.wait_if_needed(
            self.training_task,
            max_wait=self.reattach_max_wait,
            error_message=(
                f"{self._tool_label} {self._action} did not finish "
                "successfully (one or more array tasks left no "
                "is_finished.info marker). Check the training logs in: "
                f"{self.results_dir}"
            ),
        )
        self.training_task.collect_results()

        artifact = self._artifact_name()
        model_paths = []
        for calcpath in grep(str(self.results_dir / "calcdirs"), ""):
            models_dir = Path(calcpath) / "models"
            # calcdirs/ also holds non-job entries (logs/, .jax_cache/,
            # hlo_dumps/, ...); only entries with a models/ subdir are real
            # array-task dirs. Skip the rest silently -- warning for them is
            # noise -- but still warn for a genuine job dir missing its
            # artifact.
            if not models_dir.is_dir():
                continue
            model_path = models_dir / artifact
            if model_path.exists():
                model_paths.append(model_path)
            else:
                logger.warning(f"Model not found at {model_path}")

        if not model_paths:
            raise WorkflowError(
                f"No models were successfully {self._participle}\n"
                f"Check training logs in: {self.results_dir}"
            )

        logger.info(f"Collected {len(model_paths)} {self._participle} models")
        return model_paths


class _MaceArrayTrainer(_EnsembleArrayTrainer):
    """MACE specialisation of :class:`_EnsembleArrayTrainer`.

    Adds what's genuinely MACE-specific on top of the shared array-job
    machinery: the optional torch -> mace-jax conversion (``convert_to_jax``
    / ``jax_output_format``, a property of *training a MACE model*, not of
    finetuning specifically -- both shipped payloads run the conversion in
    the same job as training) and resolving the training executable as a
    python module import (``mace.cli.run_train``) rather than a bare CLI
    console script.

    ``MaceModelTrainer`` (train from scratch) and ``MaceFinetuningTrainer``
    (finetune a foundation model) differ only in a few spots: the default
    payload script, the SLURM job name, a few extra keys/resources for the
    remote input, and the stub contents -- expressed as the class
    attributes / hook overrides below and on those two subclasses.
    """

    # Per-subclass constants -----------------------------------------------
    _script_name = "mace_training.py"  # shipped payload under scripts/mace/
    _script_subdir = "mace"
    _jobname = "train"  # SLURM job name
    _tool_label = "MACE"
    _action = "training"  # "MACE {action} did not finish ..."
    _participle = "trained"  # "No models were successfully {participle}"

    def __init__(
        self,
        remote_fs: FileSystemHandlerBase,
        cluster_config: ClusterConfig,
        config_file: os.PathLike,
        training_script: Optional[os.PathLike] = None,
        remote_mace_train_runner: Optional[os.PathLike] = None,
        num_models: int = 5,
        validation_fraction: float = 0.1,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        stub: bool = False,
        n_processors: int = 1,
        convert_to_jax: bool = False,
        jax_output_format: str = "pkl",
        collect_only: bool = False,
        reattach_max_wait: Optional[float] = None,
    ):
        super().__init__(
            remote_fs=remote_fs,
            cluster_config=cluster_config,
            config_file=config_file,
            training_script=training_script,
            remote_train_runner=remote_mace_train_runner,
            num_models=num_models,
            validation_fraction=validation_fraction,
            remote_savepath=remote_savepath,
            loop_time=loop_time,
            stub=stub,
            n_processors=n_processors,
            collect_only=collect_only,
            reattach_max_wait=reattach_max_wait,
        )
        # When convert_to_jax: "pkl" (default, opaque) or "bundle" (a directory
        # with config.json + params.msgpack -- mace-jax's canonical format).
        if jax_output_format not in ("pkl", "bundle"):
            raise ConfigurationError(
                "jax_output_format must be 'pkl' or 'bundle', got "
                f"'{jax_output_format}'"
            )
        self.convert_to_jax = convert_to_jax
        self.jax_output_format = jax_output_format

    # Hooks: MACE-specific ---------------------------------------------------
    def _resolve_train_runner(self) -> str:
        return _resolve_mace_train_runner(self.remote_fs, self.cluster_config)

    def _extra_shared_input(self) -> dict:
        """Extra keys merged into the remote script's shared input JSON."""
        return {
            "convert_to_jax": self.convert_to_jax,
            "jax_output_format": self.jax_output_format,
        }

    # Shared: optional torch -> mace-jax conversion -------------------------
    def _jax_artifact_name(self) -> str:
        """Filename of the per-job artifact handed to the next sampler.

        A ``.model`` (no conversion), a ``-jax.pkl`` file, or a ``-jax``
        directory bundle, per ``convert_to_jax`` / ``jax_output_format``.
        """
        if not self.convert_to_jax:
            return f"{self.model_name}.model"
        if self.jax_output_format == "bundle":
            return f"{self.model_name}-jax"
        return f"{self.model_name}-jax.pkl"

    def _artifact_name(self) -> str:
        """Per-job filename, under ``models/``, handed to the next sampler."""
        return self._jax_artifact_name()

    def _stub_model_dict(self) -> dict:
        """Mock model payload written by stub mode."""
        return {
            "model_type": "MACE",
            "seed": np.random.randint(1e6),
            "num_interactions": 2,
            "hidden_irreps": "32x0e + 32x1o",
            "r_max": 4.0,
            "energy_weight": 1.0,
            "forces_weight": 100.0,
            "training_loss": np.random.uniform(0.01, 0.05),
            "validation_loss": np.random.uniform(0.02, 0.08),
        }

    def _serialize_stub_model(self, model_dict: dict, path: Path) -> None:
        """Write the stub payload: always the raw torch ``.model`` pickle
        (mirroring what real training always produces on top of any
        conversion), plus a mock mace-jax conversion at ``path`` when
        ``convert_to_jax`` makes that different from the base file.
        """
        model_dir = path.parent
        base_path = model_dir / f"{self.model_name}.model"
        with open(base_path, "wb") as f:
            pickle.dump(model_dict, f)
        if path == base_path:
            return
        if self.jax_output_format == "bundle":
            path.mkdir(parents=True, exist_ok=True)
            (path / "config.json").write_text("{}")
            (path / "params.msgpack").write_bytes(b"")
        else:
            with open(path, "wb") as f:
                pickle.dump(
                    {"config": model_dict, "params": b"", "head": None}, f
                )


class MaceModelTrainer(_MaceArrayTrainer):
    """
    Trainer for MACE models using array script tasks.

    Trains multiple MACE models in parallel using different random seeds.
    Supports stub mode for testing without actual training.

    Args:
        remote_fs: FileSystemHandlerBase for the machine where training runs
        config_file: Path to MACE YAML config file
        training_script: Remote payload; defaults to the shipped
            ``mace/mace_training.py``.
        remote_mace_train_runner: Path to MACE run_train.py on the remote
            machine. Optional -- if omitted, it is resolved automatically by
            probing the remote conda env (requires mace installed there).
        num_models: Number of models to train per generation
        validation_fraction: Fraction for train/validation split
        remote_savepath: Path on remote machine for calculation files
        loop_time: Interval (seconds) to poll for job completion
        stub: If True, creates mock model outputs without training
        convert_to_jax: If True, convert each trained torch model to a
            mace-jax bundle in the same job and return those paths from
            ``get_trained_model``. The training (cluster) environment must then
            have both the torch MACE stack and ``mace-jax``.
        jax_output_format: Bundle format when ``convert_to_jax`` -- ``"pkl"``
            (default; an opaque single-file pickle) or ``"bundle"`` (a
            directory with ``config.json`` + ``params.msgpack``, the canonical
            inspectable format mace-jax's loader / jaxrens consume).
        collect_only: If True, ``update`` reconstructs the array task
            submitted by an earlier, ``collect_only=False`` call (from the
            manifest written beside ``<calcdir>/train/results``) instead of
            allocating/submitting a new one, and ``get_trained_model`` skips
            the (already-done) submission. ``trainer_input`` is not read in
            this mode. Use this to collect a detached MACE training run from
            a fresh process.
        reattach_max_wait: Only used when ``collect_only``: how long (in
            seconds) to wait for the reattached job to finish before giving
            up. ``None`` (default) waits forever.
    """

    # Behaviour is entirely the base default (script mace_training.py, jobname
    # "train"); this subclass only fixes the public name and documentation.


class MaceFinetuningTrainer(_MaceArrayTrainer):
    """
    Trainer for finetuning MACE models from a foundation model.

    Like MaceModelTrainer but stages a pretrained foundation model as an extra
    shared resource. The optional torch -> mace-jax conversion is inherited
    from the shared base and works the same way here.

    Args:
        remote_fs: FileSystemHandlerBase for the machine where training runs
        config_file: Path to MACE YAML config file
        foundation_model: Path to the pretrained foundation model file
        training_script: Remote payload; defaults to the shipped
            ``mace/mace_finetuning.py``.
        remote_mace_train_runner: Path to MACE run_train.py on the remote
            machine. Optional -- if omitted, it is resolved automatically by
            probing the remote conda env (requires mace installed there).
        num_models: Number of models to train per generation
        validation_fraction: Fraction for train/validation split
        remote_savepath: Path on remote machine for calculation files
        loop_time: Interval (seconds) to poll for job completion
        stub: If True, creates mock model outputs without training
        convert_to_jax: If True, convert each trained torch model to a
            mace-jax bundle in the same job and return those paths from
            ``get_trained_model``. The training (cluster) environment must then
            have both the torch MACE stack and ``mace-jax``.
        jax_output_format: Bundle format when ``convert_to_jax`` -- ``"pkl"``
            (default; an opaque single-file pickle) or ``"bundle"`` (a
            directory with ``config.json`` + ``params.msgpack``, the canonical
            inspectable format mace-jax's loader / jaxrens consume).
        collect_only: If True, ``update`` reconstructs the array task
            submitted by an earlier, ``collect_only=False`` call (from the
            manifest written beside ``<calcdir>/train/results``) instead of
            allocating/submitting a new one, and ``get_trained_model`` skips
            the (already-done) submission. ``trainer_input`` is not read in
            this mode. Use this to collect a detached MACE training run from
            a fresh process.
        reattach_max_wait: Only used when ``collect_only``: how long (in
            seconds) to wait for the reattached job to finish before giving
            up. ``None`` (default) waits forever.
    """

    _script_name = "mace_finetuning.py"
    _jobname = "finetune"
    _action = "finetuning"
    _participle = "finetuned"

    def __init__(
        self,
        remote_fs: FileSystemHandlerBase,
        cluster_config: ClusterConfig,
        config_file: os.PathLike,
        foundation_model: os.PathLike,
        training_script: Optional[os.PathLike] = None,
        remote_mace_train_runner: Optional[os.PathLike] = None,
        num_models: int = 5,
        validation_fraction: float = 0.1,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        stub: bool = False,
        convert_to_jax: bool = False,
        n_processors: int = 1,
        jax_output_format: str = "pkl",
        collect_only: bool = False,
        reattach_max_wait: Optional[float] = None,
    ):
        super().__init__(
            remote_fs=remote_fs,
            cluster_config=cluster_config,
            config_file=config_file,
            training_script=training_script,
            remote_mace_train_runner=remote_mace_train_runner,
            num_models=num_models,
            validation_fraction=validation_fraction,
            remote_savepath=remote_savepath,
            loop_time=loop_time,
            stub=stub,
            n_processors=n_processors,
            convert_to_jax=convert_to_jax,
            jax_output_format=jax_output_format,
            collect_only=collect_only,
            reattach_max_wait=reattach_max_wait,
        )
        self.foundation_model = foundation_model

    # Hook overrides --------------------------------------------------------
    def _extra_shared_input(self) -> dict:
        return {
            **super()._extra_shared_input(),
            "foundation_model": (
                f"{SHARED_RESOURCE_PREFIX}/"
                f"{Path(self.foundation_model).name}"
            ),
        }

    def _extra_shared_resources(self) -> list:
        return [self.foundation_model]

    def _stub_model_dict(self) -> dict:
        return {
            "model_type": "MACE_finetuned",
            "seed": np.random.randint(1e6),
            "foundation_model": str(self.foundation_model),
            "training_loss": np.random.uniform(0.01, 0.05),
            "validation_loss": np.random.uniform(0.02, 0.08),
        }


class _AceArrayTrainer(_EnsembleArrayTrainer):
    """Ensemble trainer for ACE potentials, fit with ``pacemaker``."""

    _script_name = "ace_training.py"
    _script_subdir = "ace"
    _jobname = "ace_train"
    _tool_label = "ACE"
    _participle = "trained"

    def _resolve_train_runner(self) -> str:
        return _resolve_executable_train_runner(
            self.remote_fs, self.cluster_config, "pacemaker", "ace"
        )

    def _artifact_name(self) -> str:
        # pacemaker's B-basis potential -- a single YAML file.
        return f"{self.model_name}.yaml"

    def _stub_model_dict(self) -> dict:
        return {
            "model_type": "ACE",
            "seed": np.random.randint(1e6),
            "cutoff": 5.0,
            "embeddings": 9,
            "training_loss": np.random.uniform(0.01, 0.05),
            "validation_loss": np.random.uniform(0.02, 0.08),
        }

    def _serialize_stub_model(self, model_dict: dict, path: Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(model_dict, f)


class _GraceArrayTrainer(_EnsembleArrayTrainer):
    """Ensemble trainer for GRACE potentials, fit with ``gracemaker``."""

    _script_name = "grace_training.py"
    _script_subdir = "grace"
    _jobname = "grace_train"
    _tool_label = "GRACE"
    _participle = "trained"

    def _resolve_train_runner(self) -> str:
        return _resolve_executable_train_runner(
            self.remote_fs, self.cluster_config, "gracemaker", "grace"
        )

    def _artifact_name(self) -> str:
        # gracemaker's output is a directory (model.yaml + checkpoints/), not
        # a single file.
        return self.model_name

    def _stub_model_dict(self) -> dict:
        return {
            "model_type": "GRACE",
            "seed": np.random.randint(1e6),
            "cutoff": 5.0,
            "n_layers": 1,
            "training_loss": np.random.uniform(0.01, 0.05),
            "validation_loss": np.random.uniform(0.02, 0.08),
        }

    def _serialize_stub_model(self, model_dict: dict, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.yaml", "w") as f:
            yaml.safe_dump(model_dict, f)
        os.makedirs(path / "checkpoints", exist_ok=True)


class AceModelTrainer(_AceArrayTrainer):
    """
    Trainer for ACE (Atomic Cluster Expansion) models, fit with ``pacemaker``
    (https://github.com/ICAMS/python-ace) using array script tasks.

    Trains multiple ACE potentials in parallel using different random seeds,
    mirroring ``MaceModelTrainer``. Supports stub mode for testing without
    actual training. pacemaker (the ``python-ace``/``pyace`` package) is
    typically installed via conda-forge or built from source, since it ships
    compiled C++ extensions -- see the ``ace`` optional-dependency extra.

    Args:
        remote_fs: FileSystemHandlerBase for the machine where training runs
        config_file: Path to a YAML config with pacemaker's ``potential``/
            ``fit``/``backend`` sections (no ``data``/``seed`` needed there
            -- those are injected per job)
        training_script: Remote payload; defaults to the shipped
            ``ace/ace_training.py``.
        remote_train_runner: Path to the ``pacemaker`` executable on the
            remote machine. Optional -- if omitted, it is resolved
            automatically by probing the remote conda env's PATH.
        num_models: Number of models to train per generation
        validation_fraction: Fraction of data pacemaker splits off as its own
            internal test set
        remote_savepath: Path on remote machine for calculation files
        loop_time: Interval (seconds) to poll for job completion
        stub: If True, creates mock model outputs without training
        collect_only: If True, ``update`` reconstructs the array task
            submitted by an earlier, ``collect_only=False`` call (from the
            manifest written beside ``<calcdir>/train/results``) instead of
            allocating/submitting a new one, and ``get_trained_model`` skips
            the (already-done) submission. ``trainer_input`` is not read in
            this mode. Use this to collect a detached ACE training run from
            a fresh process.
        reattach_max_wait: Only used when ``collect_only``: how long (in
            seconds) to wait for the reattached job to finish before giving
            up. ``None`` (default) waits forever.
    """

    # Behaviour is entirely the base default (script ace_training.py,
    # jobname "ace_train"); this subclass only fixes the public name and
    # documentation.


class GraceModelTrainer(_GraceArrayTrainer):
    """
    Trainer for GRACE (Graph Atomic Cluster Expansion) models, fit with
    ``gracemaker`` (https://github.com/ICAMS/grace-tensorpotential) using
    array script tasks.

    Trains multiple GRACE potentials in parallel using different random
    seeds, mirroring ``MaceModelTrainer``. Supports stub mode for testing
    without actual training. gracemaker is installed via the ``grace``
    optional-dependency extra (the ``tensorpotential`` package).

    Args:
        remote_fs: FileSystemHandlerBase for the machine where training runs
        config_file: Path to a YAML config with gracemaker's ``cutoff``/
            ``potential``/``fit`` sections (no ``data``/``seed`` needed there
            -- those are injected per job)
        training_script: Remote payload; defaults to the shipped
            ``grace/grace_training.py``.
        remote_train_runner: Path to the ``gracemaker`` executable on the
            remote machine. Optional -- if omitted, it is resolved
            automatically by probing the remote conda env's PATH.
        num_models: Number of models to train per generation
        validation_fraction: Fraction of data gracemaker splits off as its
            own internal test set
        remote_savepath: Path on remote machine for calculation files
        loop_time: Interval (seconds) to poll for job completion
        stub: If True, creates mock model outputs without training
        collect_only: If True, ``update`` reconstructs the array task
            submitted by an earlier, ``collect_only=False`` call (from the
            manifest written beside ``<calcdir>/train/results``) instead of
            allocating/submitting a new one, and ``get_trained_model`` skips
            the (already-done) submission. ``trainer_input`` is not read in
            this mode. Use this to collect a detached GRACE training run
            from a fresh process.
        reattach_max_wait: Only used when ``collect_only``: how long (in
            seconds) to wait for the reattached job to finish before giving
            up. ``None`` (default) waits forever.
    """

    # Behaviour is entirely the base default (script grace_training.py,
    # jobname "grace_train"); this subclass only fixes the public name and
    # documentation.


class NeuralilILTrainer(DetachableTaskMixin, ModelTrainer):
    """
    Trainer for NeuralIL models using JAXNEST framework.

    Trains NeuralIL models with custom configuration.
    Supports stub mode for testing without actual training.

    Hyperparameters resolve in three layers (later overrides earlier):
    built-in defaults < ``config_file`` (a YAML file you author) <
    ``training_kwargs`` (an explicit dict). All optional -- with none given the
    built-in defaults are used.

    Args:
        training_script: Remote payload; defaults to the shipped
            ``neuralil/neuralil_training.py``.
        remote_fs: FileSystemHandlerBase for the machine where training runs
        remote_savepath: Path on remote machine for calculation files
        training_kwargs: Explicit hyperparameter overrides (highest precedence).
        stub: If True, creates mock model outputs without training
        loop_time: Interval (seconds) to poll for job completion
        config_file: Optional path to a YAML file of hyperparameters, layered
            on the built-in defaults (mirrors ``MaceModelTrainer``'s
            ``config_file``). Overridden by anything in ``training_kwargs``.
        collect_only: If True, ``update`` reconstructs the task submitted by
            an earlier, ``collect_only=False`` call (from the manifest
            written beside ``<calcdir>/training/results``) instead of
            allocating/submitting a new one, and ``get_trained_model`` skips
            the (already-done) submission. ``trainer_input`` is not read in
            this mode. Use this to collect a detached NeuralIL training run
            from a fresh process.
        reattach_max_wait: Only used when ``collect_only``: how long (in
            seconds) to wait for the reattached job to finish before giving
            up. ``None`` (default) waits forever.
    """

    _iteration_subdir = (
        "training"  # calcdir/training -- see DetachableTaskMixin
    )

    default_training_kwargs: dict = {
        "r_cut": 4.0,
        "training_fraction": 0.75,
        "supercell_diag": [3, 3, 3],
        "high_energy_penalty": -5.0,
        "n_ensemble": 30,
        "save_reduced": 5,
        "n_batch": 32,
        "n_max": 6,
        "embed_d": 1,
        "core_widths": [128, 64, 32],
        "morse": False,
        "n_eval_batch": 64,
        "n_epochs": 100,
        "energy_weight": 0.5,
        "log_cosh_parameter_energy": 0.200,
        "log_cosh_parameter_forces": 2.000,
        "seed": None,
        "stub": False,
        "optimizer_type": "VeLO",
        "learning_rate_kwargs": dict(
            max_learning_rate=1e-2,
            min_learning_rate=1e-3,
            fin_learning_rate=1e-5,
        ),
    }

    def __init__(
        self,
        remote_fs: FileSystemHandlerBase,
        cluster_config: ClusterConfig,
        training_script: Optional[os.PathLike] = None,
        remote_savepath: Optional[os.PathLike] = None,
        training_kwargs: dict = None,
        stub: bool = False,
        loop_time: float = 30,
        config_file: Optional[os.PathLike] = None,
        collect_only: bool = False,
        reattach_max_wait: Optional[float] = None,
    ):
        super().__init__()
        if training_script is None:
            training_script = (
                _SCRIPTS_DIR / "neuralil" / "neuralil_training.py"
            )
        self.training_script = training_script
        self.cluster_config = cluster_config
        self.collect_only = collect_only
        self.reattach_max_wait = reattach_max_wait
        self.config_file = config_file

        # Layer the hyperparameters: defaults < config_file (YAML) <
        # training_kwargs. deepcopy so the class-level defaults are never
        # mutated (otherwise overrides would leak across instances).
        merged = deepcopy(self.default_training_kwargs)
        if config_file is not None:
            import yaml

            with open(config_file) as f:
                merged.update(yaml.safe_load(f) or {})
        if training_kwargs:
            merged.update(training_kwargs)
        self.training_kwargs = merged

        self.remote_fs = remote_fs
        self.loop_time = loop_time
        self.remote_savepath = (
            Path(remote_savepath) if remote_savepath else None
        )
        self.stub = stub
        self.training_kwargs["stub"] = stub

        # The local staging input file is written lazily in update(),
        # under this iteration's training directory -- never in the cwd.
        self.input_file = None

    def update(
        self,
        calcdir: os.PathLike,
        trainer_input: Optional[Path | list[os.PathLike]],
        **kwargs,
    ) -> None:
        """
        Prepare trainer for a new iteration.

        Args:
            calcdir: Directory for this AL iteration
            trainer_input: Path(s) to database files with training data. Not
                read when ``collect_only`` -- see ``MaceModelTrainer``.
            **kwargs: Additional parameters
        """
        self.train_dir = self.iteration_dir(calcdir)
        self.results_dir = self.results_dir_for(calcdir)
        os.makedirs(self.train_dir, exist_ok=True)

        if not self.collect_only:
            self.db_files = (
                [trainer_input]
                if isinstance(trainer_input, Path)
                else trainer_input
            )

        # The remote pickle filename is deterministic from calcdir alone
        # (``model_<iteration>.pkl``), so it is computed here, unconditionally
        # -- not inside the build step below -- meaning a collect_only=True
        # trainer (which never runs that build step at all) still knows
        # where get_trained_model() should look, instead of depending on
        # self.training_kwargs["pickle_file_name"], which is only ever set
        # as a side effect of building the task in the *original* process.
        it = str(calcdir).split("/")[-1]
        self.pickle_filename = f"model_{it}.pkl"

        # In stub mode, we don't need to create the task
        if self.stub:
            return

        def _build_training_task() -> SingleScriptTask:
            # Create JSON input file for the training script inside this
            # iteration's training directory.
            self.input_file = self.train_dir / "input_nnff.json"
            with open(self.input_file, "w") as f:
                json.dump(self.training_kwargs, f)

            task = SingleScriptTask(
                cluster_config=self.cluster_config,
                script_file=self.training_script,
                input_file=self.input_file,
                local_savepath=self.results_dir,
                remote_fs=self.remote_fs,
                logfile="train.log",
                remote_savepath=self.remote_savepath,
                collect_only=False,
                loop_time=self.loop_time,
                jobname="NeuralIL_training",
                n_processors=1,
            )

            # Prepare training configuration
            # Copy database files to remote
            remote_dbs = [
                str(task.copy_file(db_file)) for db_file in self.db_files
            ]

            with open(self.input_file, "r") as f:
                self.training_kwargs = json.load(f)

            self.training_kwargs["structure_databases"] = remote_dbs
            self.training_kwargs["filepath"] = str(task.calcdir)
            self.training_kwargs["pickle_file_name"] = str(
                task.calcdir / self.pickle_filename
            )

            # Write JSON input file to remote
            with open(self.input_file, "w") as f:
                json.dump(self.training_kwargs, f)
            return task

        # Allocation (copies script and writes jobfile) happens inside
        # build_or_resume_single_task, uniformly for every non-collect_only
        # caller -- see DetachableTaskMixin.
        self.train_task = self.build_or_resume_single_task(
            self.results_dir, _build_training_task
        )

    def _create_stub_outputs(self) -> Path:
        """
        Create mock NeuralIL model file for testing.

        Generates fake trained model with realistic structure.

        Returns:
            Path to mock model file
        """
        logger.info("Creating stub NeuralIL model file")

        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)

        # Get pickle file name from training kwargs
        pickle_filename = self.training_kwargs.get(
            "pickle_file_name", "neuralil_model.pkl"
        )
        model_file = self.results_dir / os.path.basename(pickle_filename)

        # Mock NeuralIL model parameters (simplified)
        mock_model = {
            "model_type": "NeuralIL",
            "n_descriptors": 128,
            "n_layers": 3,
            "activation": "tanh",
            "learning_rate": 0.001,
            "num_epochs": 100,
            "final_train_loss": np.random.uniform(0.01, 0.05),
            "final_val_loss": np.random.uniform(0.02, 0.08),
            "training_time": np.random.uniform(100, 300),  # seconds
        }

        # Write as pickle
        with open(model_file, "wb") as f:
            pickle.dump(mock_model, f)

        logger.info(f"Created stub model: {model_file}")
        return model_file

    def get_trained_model(self) -> os.PathLike:
        """
        Run training and return model file path.

        Returns:
            Path to trained model file

        Raises:
            TaskError: If training doesn't complete successfully (or, when
                ``collect_only``, if that doesn't happen within
                ``reattach_max_wait``).
        """
        if self.stub:
            # In stub mode, create mock output directly
            logger.info("Running in stub mode - creating mock NeuralIL model")
            model_file = self._create_stub_outputs()
        else:
            # Submit (unless collect_only -- already submitted in another
            # process), wait, then collect -- rather than the single
            # task.run() convenience call, which unconditionally re-runs
            # allocate_data() and would be wrong for a collect_only task.
            self.submit_if_needed(self.train_task)

            self.wait_if_needed(
                self.train_task,
                max_wait=self.reattach_max_wait,
                error_message="Training did not complete successfully",
            )
            self.train_task.collect_results()

            # self.pickle_filename (set in update(), independent of
            # self.training_kwargs) is what makes this correct in
            # collect_only mode too -- see update()'s docstring note.
            model_file = self.results_dir / self.pickle_filename

            if not os.path.exists(model_file):
                raise WorkflowError(f"Model file not found: {model_file}")

        logger.info(f"Trained model available at: {model_file}")
        return model_file
