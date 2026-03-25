"""
Trainers for Active Learning Workflows.

Trainers fit machine learning models to evaluated configurations.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import FileSystemHandlerBase
from remote_handler.core.tasks import (
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
from remote_handler.scripts.input_models import (
    MaceFinetuningInput,
    MaceTrainingInput,
)

from .active_learning_base import ModelTrainer

logger = logging.getLogger(__name__)


class MaceModelTrainer(ModelTrainer):
    """
    Trainer for MACE models using array script tasks.

    Trains multiple MACE models in parallel using different random seeds.
    Supports stub mode for testing without actual training.

    Args:
        remote_fs: FileSystemHandlerBase for the machine where training runs
        training_script: Path to mace_training.py or compatible script
        config_file: Path to MACE YAML config file
        remote_mace_train_runner: Path to MACE run_train.py on remote machine
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
        remote_mace_train_runner: os.PathLike,
        num_models: int = 5,
        validation_fraction: float = 0.1,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        stub: bool = False,
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
            # Shared resources live in rootpath/, but scripts run in
            # rootpath/calcdirs/N/. Prefix with ../../ so the script can
            # resolve them relative to its own calcpath.
            script_input_shared = {
                "database_files": [
                    f"../../{Path(p).name}" for p in self.db_files
                ],
                "config_file": f"../../{Path(self.config_file).name}",
                "model_name": self.model_name,
                "validation_fraction": self.validation_fraction,
                "train_runner": Path(self.remote_mace_train_runner).as_posix(),
            }

            # Validate input dict against the script's expected schema
            MaceTrainingInput.from_dict(script_input_shared | {"seed": 0})

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
        remote_mace_train_runner: Path to MACE run_train.py on remote machine
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
        foundation_model: os.PathLike,
        remote_mace_train_runner: os.PathLike,
        num_models: int = 5,
        validation_fraction: float = 0.1,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        stub: bool = False,
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
            # Shared resources live in rootpath/, but scripts run in
            # rootpath/calcdirs/N/. Prefix with ../../ so the script can
            # resolve them relative to its own calcpath.
            script_input_shared = {
                "database_files": [
                    f"../../{Path(p).name}" for p in self.db_files
                ],
                "config_file": f"../../{Path(self.config_file).name}",
                "foundation_model": f"../../{Path(self.foundation_model).name}",
                "model_name": self.model_name,
                "validation_fraction": self.validation_fraction,
                "train_runner": Path(self.remote_mace_train_runner).as_posix(),
            }

            # Validate input dict against the script's expected schema
            MaceFinetuningInput.from_dict(script_input_shared | {"seed": 0})

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

            model_paths.append(model_path)
            logger.info(f"Created stub finetuned model: {model_path}")

        return model_paths

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

            model_paths = []
            for calcpath in grep(str(self.results_dir / "calcdirs"), ""):
                model_path = Path(calcpath) / f"models/{self.model_name}.model"
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

    Args:
        training_script: Path to NeuralIL training script
        remote_fs: FileSystemHandlerBase for the machine where training runs
        remote_savepath: Path on remote machine for calculation files
        training_kwargs: Additional training parameters including pickle_file_name
        stub: If True, creates mock model outputs without training
        loop_time: Interval (seconds) to poll for job completion
    """

    def __init__(
        self,
        training_script: os.PathLike,
        remote_fs: FileSystemHandlerBase,
        cluster_config: ClusterConfig,
        remote_savepath: Optional[os.PathLike] = None,
        training_kwargs: dict = None,
        stub: bool = False,
        loop_time: float = 30,
    ):
        super().__init__()
        self.training_script = training_script
        self.cluster_config = cluster_config
        self.training_kwargs = training_kwargs or {}
        self.remote_fs = remote_fs
        self.loop_time = loop_time
        self.remote_savepath = (
            Path(remote_savepath) if remote_savepath else None
        )
        self.stub = stub
        self.training_kwargs["stub"] = stub

    def update(
        self, calcdir: os.PathLike, db_files: list[os.PathLike], **kwargs
    ) -> None:
        """
        Prepare trainer for a new iteration.

        Args:
            calcdir: Directory for this AL iteration
            db_files: List of database files with training data
            **kwargs: Additional parameters
        """
        self.train_dir = Path(calcdir) / "training"
        self.results_dir = self.train_dir / "results"

        if os.path.exists(self.train_dir):
            logger.warning("Training path already exists, take care!")

        os.makedirs(self.train_dir, exist_ok=True)
        self.db_files = db_files
        it = str(calcdir).split("/")[-1]

        # In stub mode, we don't need to create the task
        if not self.stub:
            # Create JSON input file for training script
            input_file = self.results_dir / "input_nnff.json"

            self.train_task = SingleScriptTask(
                cluster_config=self.cluster_config,
                script_file=self.training_script,
                input_file=input_file,
                local_savepath=self.results_dir,
                remote_fs=self.remote_fs,
                logfile="train.log",
                remote_savepath=self.remote_savepath,
                collect_only=False,
                loop_time=self.loop_time,
                jobname="NeuralIL_training",
            )

            # Prepare training configuration
            # Copy database files to remote
            remote_dbs = []
            for db_file in db_files:
                remote_dbs.append(str(self.train_task.copy_file(db_file)))

            self.training_kwargs["structure_databases"] = remote_dbs
            self.training_kwargs["filepath"] = str(self.train_task.calcdir)
            self.training_kwargs["pickle_file_name"] = str(
                self.train_task.calcdir / f"model_{it}.pkl"
            )

            # Write JSON input file to remote
            self.remote_fs.write_dict_to_json(
                self.train_task.calcdir / "input_nnff.json",
                self.training_kwargs,
            )

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
                / os.path.split(
                    self.train_task.training_kwargs["pickle_file_name"]
                )[-1]
            )

            if not os.path.exists(model_file):
                raise WorkflowError(f"Model file not found: {model_file}")

        logger.info(f"Trained model available at: {model_file}")
        return model_file
