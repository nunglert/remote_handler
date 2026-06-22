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

from remote_handler.core import remote_env
from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import FileSystemHandlerBase
from remote_handler.core.tasks import (
    SHARED_RESOURCE_PREFIX,
    ArrayScriptTask,
    DictJSON,
    SingleScriptTask,
)
from remote_handler.core.utils import grep
from remote_handler.exceptions import (
    ConfigurationError,
    DependencyError,
    WorkflowError,
)

from .active_learning_base import ModelTrainer

logger = logging.getLogger(__name__)


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


class MaceModelTrainer(ModelTrainer):
    """
    Trainer for MACE models using array script tasks.

    Trains multiple MACE models in parallel using different random seeds.
    Supports stub mode for testing without actual training.

    Args:
        remote_fs: FileSystemHandlerBase for the machine where training runs
        training_script: Path to mace_training.py or compatible script
        config_file: Path to MACE YAML config file
        remote_mace_train_runner: Path to MACE run_train.py on the remote
            machine. Optional -- if omitted, it is resolved automatically by
            probing the remote conda env (requires mace installed there).
        num_models: Number of models to train per generation
        validation_fraction: Fraction for train/validation split
        remote_savepath: Path on remote machine for calculation files
        loop_time: Interval (seconds) to poll for job completion
        stub: If True, creates mock model outputs without training
    """

    def __init__(
        self,
        remote_fs: FileSystemHandlerBase,
        cluster_config: ClusterConfig,
        training_script: os.PathLike,
        config_file: os.PathLike,
        remote_mace_train_runner: Optional[os.PathLike] = None,
        num_models: int = 5,
        validation_fraction: float = 0.1,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        stub: bool = False,
        n_processors: int = 1,
    ):
        super().__init__()
        self.training_script = training_script
        self.remote_fs = remote_fs
        self.cluster_config = cluster_config
        self.config_file = config_file
        self.num_models = num_models
        self.validation_fraction = validation_fraction
        self.remote_mace_train_runner = remote_mace_train_runner
        self.remote_savepath = remote_savepath
        self.loop_time = loop_time
        self.stub = stub
        # Resources per training job; for GPU templates this is the GPU count
        # (``--gres=gpu:n``). One model per job -> one GPU by default.
        self.n_processors = n_processors
        self.model_name = "Model"  # Default model name

    def update(
        self, calcdir: Path, trainer_input: Path | list[Path], **kwargs
    ) -> None:
        """
        Prepare trainer for a new iteration.

        Args:
            calcdir: Directory for this AL iteration
            trainer_input: Path(s) to database files with training data
            **kwargs: Additional parameters
        """
        self.train_dir = calcdir / "train"
        self.results_dir = self.train_dir / "results"
        os.makedirs(self.train_dir, exist_ok=True)

        self.db_files = (
            [trainer_input]
            if isinstance(trainer_input, Path)
            else trainer_input
        )

        # In stub mode, we don't need to create the task
        if not self.stub:
            # Auto-resolve the remote run_train.py path if not given.
            if self.remote_mace_train_runner is None:
                self.remote_mace_train_runner = _resolve_mace_train_runner(
                    self.remote_fs, self.cluster_config
                )
            # Shared resources live in rootpath/, but scripts run in a
            # subdir of it; reference them via SHARED_RESOURCE_PREFIX so each
            # script resolves them relative to its own calcpath.
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
                "train_runner": Path(self.remote_mace_train_runner).as_posix(),
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

            self.training_task = ArrayScriptTask(
                cluster_config=self.cluster_config,
                local_savepath=self.results_dir,
                remote_fs=self.remote_fs,
                python_script_file=self.training_script,
                individual_ressources=individual_resources,
                shared_ressources=[self.config_file] + self.db_files,
                logfile="train.log",
                remote_savepath=self.remote_savepath,
                collect_only=False,
                num_parallel=min(self.num_models, 5),
                loop_time=self.loop_time,
                jobname="train",
                n_processors=self.n_processors,
            )

            self.training_task.allocate_data()

    def _create_stub_outputs(self) -> list[Path]:
        """
        Create mock MACE model files for testing.

        Generates fake trained models with realistic structure.

        Returns:
            List of paths to mock model files
        """
        logger.info(f"Creating {self.num_models} stub MACE model files")

        # Create mock calcdirs structure
        calcdirs = self.results_dir / "calcdirs"
        os.makedirs(calcdirs, exist_ok=True)

        model_paths = []
        for i in range(self.num_models):
            # Create calculation directory
            calcdir = calcdirs / f"job_{i}"
            model_dir = calcdir / "models"
            os.makedirs(model_dir, exist_ok=True)

            # Create mock model file
            model_path = model_dir / f"{self.model_name}.model"

            # Mock model content (simple dict with metadata)
            mock_model = {
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

            # Write as pickle (MACE models are typically pickle files)
            with open(model_path, "wb") as f:
                pickle.dump(mock_model, f)

            model_paths.append(model_path)
            logger.info(f"Created stub model: {model_path}")

        return model_paths

    def get_trained_model(self) -> list[Path]:
        """
        Run training and collect model files.

        Returns:
            List of paths to trained model files

        Raises:
            Exception: If training doesn't complete successfully
        """
        if self.stub:
            # In stub mode, create mock outputs directly
            logger.info("Running in stub mode - creating mock MACE models")
            model_paths = self._create_stub_outputs()
        else:
            # Run actual training
            self.training_task.submit()
            self.training_task.collect_results()

            # Collect trained models
            model_paths = []
            for calcpath in grep(str(self.results_dir / "calcdirs"), ""):
                model_path = Path(calcpath) / f"models/{self.model_name}.model"
                if model_path.exists():
                    model_paths.append(model_path)
                else:
                    logger.warning(f"Model not found at {model_path}")

            if not model_paths:
                raise WorkflowError(
                    "No models were successfully trained\n"
                    f"Check training logs in: {self.results_dir}"
                )

        logger.info(f"Collected {len(model_paths)} trained models")
        return model_paths


class MaceFinetuningTrainer(ModelTrainer):
    """
    Trainer for finetuning MACE models from a foundation model.

    Similar to MaceModelTrainer but takes a pretrained foundation model
    as an additional shared resource. Trains multiple finetuned models
    in parallel using different random seeds.

    Args:
        remote_fs: FileSystemHandlerBase for the machine where training runs
        training_script: Path to mace_finetuning.py or compatible script
        config_file: Path to MACE YAML config file
        foundation_model: Path to the pretrained foundation model file
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
    """

    def __init__(
        self,
        remote_fs: FileSystemHandlerBase,
        cluster_config: ClusterConfig,
        training_script: os.PathLike,
        config_file: os.PathLike,
        foundation_model: os.PathLike,
        remote_mace_train_runner: Optional[os.PathLike] = None,
        num_models: int = 5,
        validation_fraction: float = 0.1,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        stub: bool = False,
        convert_to_jax: bool = False,
        n_processors: int = 1,
        jax_output_format: str = "pkl",
    ):
        super().__init__()
        self.training_script = training_script
        self.remote_fs = remote_fs
        self.cluster_config = cluster_config
        self.config_file = config_file
        self.foundation_model = foundation_model
        self.num_models = num_models
        self.validation_fraction = validation_fraction
        self.remote_mace_train_runner = remote_mace_train_runner
        self.remote_savepath = remote_savepath
        self.loop_time = loop_time
        self.stub = stub
        self.convert_to_jax = convert_to_jax
        # When convert_to_jax: "pkl" (default, opaque) or "bundle" (a directory
        # with config.json + params.msgpack -- mace-jax's canonical format).
        if jax_output_format not in ("pkl", "bundle"):
            raise ConfigurationError(
                "jax_output_format must be 'pkl' or 'bundle', got "
                f"'{jax_output_format}'"
            )
        self.jax_output_format = jax_output_format
        # Resources per training job; for GPU templates this is the GPU count
        # (``--gres=gpu:n``). One model per job -> one GPU by default.
        self.n_processors = n_processors
        self.model_name = "Model"  # Default model name

    def update(
        self, calcdir: Path, trainer_input: Path | list[Path], **kwargs
    ) -> None:
        """
        Prepare trainer for a new iteration.

        Args:
            calcdir: Directory for this AL iteration
            trainer_input: Path(s) to database files with training data
            **kwargs: Additional parameters
        """
        self.train_dir = calcdir / "train"
        self.results_dir = self.train_dir / "results"
        os.makedirs(self.train_dir, exist_ok=True)

        self.db_files = (
            [trainer_input]
            if isinstance(trainer_input, Path)
            else trainer_input
        )

        # In stub mode, we don't need to create the task
        if not self.stub:
            # Auto-resolve the remote run_train.py path if not given.
            if self.remote_mace_train_runner is None:
                self.remote_mace_train_runner = _resolve_mace_train_runner(
                    self.remote_fs, self.cluster_config
                )
            # Shared resources live in rootpath/, but scripts run in a
            # subdir of it; reference them via SHARED_RESOURCE_PREFIX so each
            # script resolves them relative to its own calcpath.
            script_input_shared = {
                "database_files": [
                    f"{SHARED_RESOURCE_PREFIX}/{Path(p).name}"
                    for p in self.db_files
                ],
                "config_file": (
                    f"{SHARED_RESOURCE_PREFIX}/{Path(self.config_file).name}"
                ),
                "foundation_model": (
                    f"{SHARED_RESOURCE_PREFIX}/"
                    f"{Path(self.foundation_model).name}"
                ),
                "model_name": self.model_name,
                "validation_fraction": self.validation_fraction,
                "train_runner": Path(self.remote_mace_train_runner).as_posix(),
                "convert_to_jax": self.convert_to_jax,
                "jax_output_format": self.jax_output_format,
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

            self.training_task = ArrayScriptTask(
                cluster_config=self.cluster_config,
                local_savepath=self.results_dir,
                remote_fs=self.remote_fs,
                python_script_file=self.training_script,
                individual_ressources=individual_resources,
                shared_ressources=[self.config_file, self.foundation_model]
                + self.db_files,
                logfile="train.log",
                remote_savepath=self.remote_savepath,
                collect_only=False,
                num_parallel=min(self.num_models, 5),
                loop_time=self.loop_time,
                jobname="finetune",
                n_processors=self.n_processors,
            )

            self.training_task.allocate_data()

    def _create_stub_outputs(self) -> list[Path]:
        """
        Create mock finetuned MACE model files for testing.

        Returns:
            List of paths to mock model files
        """
        logger.info(f"Creating {self.num_models} stub finetuned MACE models")

        # Create mock calcdirs structure
        calcdirs = self.results_dir / "calcdirs"
        os.makedirs(calcdirs, exist_ok=True)

        model_paths = []
        for i in range(self.num_models):
            calcdir = calcdirs / f"job_{i}"
            model_dir = calcdir / "models"
            os.makedirs(model_dir, exist_ok=True)

            model_path = model_dir / f"{self.model_name}.model"

            mock_model = {
                "model_type": "MACE_finetuned",
                "seed": np.random.randint(1e6),
                "foundation_model": str(self.foundation_model),
                "training_loss": np.random.uniform(0.01, 0.05),
                "validation_loss": np.random.uniform(0.02, 0.08),
            }

            with open(model_path, "wb") as f:
                pickle.dump(mock_model, f)

            if self.convert_to_jax:
                jax_path = model_dir / self._jax_artifact_name()
                if self.jax_output_format == "bundle":
                    jax_path.mkdir(parents=True, exist_ok=True)
                    (jax_path / "config.json").write_text("{}")
                    (jax_path / "params.msgpack").write_bytes(b"")
                else:
                    with open(jax_path, "wb") as f:
                        pickle.dump(
                            {
                                "config": mock_model,
                                "params": b"",
                                "head": None,
                            },
                            f,
                        )
                model_paths.append(jax_path)
                logger.info(f"Created stub mace-jax bundle: {jax_path}")
            else:
                model_paths.append(model_path)
                logger.info(f"Created stub finetuned model: {model_path}")

        return model_paths

    def _jax_artifact_name(self) -> str:
        """Filename of the per-job artifact handed to the next sampler.

        A ``.model`` (no conversion), a ``-jax.pkl`` file, or a ``-jax``
        directory bundle, depending on ``convert_to_jax`` / ``jax_output_format``.
        """
        if not self.convert_to_jax:
            return f"{self.model_name}.model"
        if self.jax_output_format == "bundle":
            return f"{self.model_name}-jax"
        return f"{self.model_name}-jax.pkl"

    def get_trained_model(self) -> list[Path]:
        """
        Run finetuning and collect model files.

        Returns:
            List of paths to finetuned model files

        Raises:
            WorkflowError: If no models were successfully trained
        """
        if self.stub:
            logger.info(
                "Running in stub mode - creating mock finetuned MACE models"
            )
            model_paths = self._create_stub_outputs()
        else:
            self.training_task.submit()
            self.training_task.collect_results()

            # When converting, the artifact handed to the next sampler is the
            # mace-jax bundle produced alongside the torch .model: a ``.pkl``
            # file, or a ``-jax/`` directory bundle.
            model_filename = self._jax_artifact_name()
            model_paths = []
            for calcpath in grep(str(self.results_dir / "calcdirs"), ""):
                model_path = Path(calcpath) / "models" / model_filename
                if model_path.exists():
                    model_paths.append(model_path)
                else:
                    logger.warning(f"Model not found at {model_path}")

            if not model_paths:
                raise WorkflowError(
                    "No models were successfully finetuned\n"
                    f"Check training logs in: {self.results_dir}"
                )

        logger.info(f"Collected {len(model_paths)} finetuned models")
        return model_paths


class NeuralilILTrainer(ModelTrainer):
    """
    Trainer for NeuralIL models using JAXNEST framework.

    Trains NeuralIL models with custom configuration.
    Supports stub mode for testing without actual training.

    Hyperparameters resolve in three layers (later overrides earlier):
    built-in defaults < ``config_file`` (a YAML file you author) <
    ``training_kwargs`` (an explicit dict). All optional -- with none given the
    built-in defaults are used.

    Args:
        training_script: Path to NeuralIL training script
        remote_fs: FileSystemHandlerBase for the machine where training runs
        remote_savepath: Path on remote machine for calculation files
        training_kwargs: Explicit hyperparameter overrides (highest precedence).
        stub: If True, creates mock model outputs without training
        loop_time: Interval (seconds) to poll for job completion
        config_file: Optional path to a YAML file of hyperparameters, layered
            on the built-in defaults (mirrors ``MaceModelTrainer``'s
            ``config_file``). Overridden by anything in ``training_kwargs``.
    """

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
        training_script: os.PathLike,
        remote_savepath: Optional[os.PathLike] = None,
        training_kwargs: dict = None,
        stub: bool = False,
        loop_time: float = 30,
        config_file: Optional[os.PathLike] = None,
    ):
        super().__init__()
        self.training_script = training_script
        self.cluster_config = cluster_config
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
        trainer_input: Path | list[os.PathLike],
        **kwargs,
    ) -> None:
        """
        Prepare trainer for a new iteration.

        Args:
            calcdir: Directory for this AL iteration
            trainer_input: Path(s) to database files with training data
            **kwargs: Additional parameters
        """
        self.train_dir = Path(calcdir) / "training"
        self.results_dir = self.train_dir / "results"
        os.makedirs(self.train_dir, exist_ok=True)

        self.db_files = (
            [trainer_input]
            if isinstance(trainer_input, Path)
            else trainer_input
        )
        it = str(calcdir).split("/")[-1]

        # In stub mode, we don't need to create the task
        if not self.stub:
            # Create JSON input file for the training script inside this
            # iteration's training directory.
            self.input_file = self.train_dir / "input_nnff.json"
            with open(self.input_file, "w") as f:
                json.dump(self.training_kwargs, f)

            self.train_task = SingleScriptTask(
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
            remote_dbs = []
            for db_file in self.db_files:
                remote_dbs.append(str(self.train_task.copy_file(db_file)))

            with open(self.input_file, "r") as f:
                self.training_kwargs = json.load(f)

            self.training_kwargs["structure_databases"] = remote_dbs
            self.training_kwargs["filepath"] = str(self.train_task.calcdir)
            self.training_kwargs["pickle_file_name"] = str(
                self.train_task.calcdir / f"model_{it}.pkl"
            )

            # Write JSON input file to remote
            with open(self.input_file, "w") as f:
                json.dump(self.training_kwargs, f)
            # Allocate data (copies script and writes jobfile)
            self.train_task.allocate_data()

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
            Exception: If training doesn't complete successfully
        """
        if self.stub:
            # In stub mode, create mock output directly
            logger.info("Running in stub mode - creating mock NeuralIL model")
            model_file = self._create_stub_outputs()
        else:
            # Run actual training
            self.train_task.run()

            if not self.train_task.finish_check():
                raise WorkflowError("Training did not complete successfully")

            # Get model file path from training kwargs
            model_file = (
                self.results_dir
                / os.path.split(self.training_kwargs["pickle_file_name"])[-1]
            )

            if not os.path.exists(model_file):
                raise WorkflowError(f"Model file not found: {model_file}")

        logger.info(f"Trained model available at: {model_file}")
        return model_file
