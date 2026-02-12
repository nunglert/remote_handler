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

from remote_handler.core.file_handler import FileSystemHandlerBase
from remote_handler.core.tasks import (
    ArrayScriptTask,
    DictJSON,
    SingleScriptTask,
)
from remote_handler.core.utils import grep
from remote_handler.workflows.active_learning_base import ModelTrainer

logger = logging.getLogger(__name__)


class MaceModelTrainer(ModelTrainer):
    """
    Trainer for MACE models using array script tasks.

    Trains multiple MACE models in parallel using different random seeds.
    Supports stub mode for testing without actual training.

    Args:
        remote_fs: FileSystemHandlerBase for the machine where training runs
        training_script: Path to mace_training.py or compatible script
        template_path: Path to jinja2 template for mace_config.yaml
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
        training_script: os.PathLike,
        template_path: os.PathLike,
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
        self.template_path = template_path
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
            script_input_shared = {
                "database_files": [Path(p).name for p in self.db_files],
                "template_file": Path(self.template_path).name,
                "model_name": self.model_name,
                "validation_fraction": self.validation_fraction,
                "train_runner": Path(self.remote_mace_train_runner).as_posix(),
                "stub": self.stub,
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

            from remote_handler.core.cluster_handler import ClusterConfig

            cluster_config = ClusterConfig(
                jobfile_name="run_train.sh",
                n_processors=1,
            )

            self.training_task = ArrayScriptTask(
                cluster_config=cluster_config,
                local_savepath=self.results_dir,
                remote_fs=self.remote_fs,
                python_script_file=self.training_script,
                individual_ressources=individual_resources,
                shared_ressources=[self.template_path] + self.db_files,
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
            model_path = model_dir / f"{self.model_name}_stagetwo.model"

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
                model_path = (
                    Path(calcpath) / f"models/{self.model_name}_stagetwo.model"
                )
                if model_path.exists():
                    model_paths.append(model_path)
                else:
                    logger.warning(f"Model not found at {model_path}")

            if not model_paths:
                raise Exception("No models were successfully trained!")

        logger.info(f"Collected {len(model_paths)} trained models")
        return model_paths


class MACETrainer(ModelTrainer):
    """
    Trainer for MACE models using JAXNEST framework.

    Trains MACE models with gin configuration files.
    Supports stub mode for testing without actual training.

    Args:
        training_script: Path to MACE training script
        remote_fs: FileSystemHandlerBase for the machine where training runs
        config_file: Path to gin config file
        remote_savepath: Path on remote machine for calculation files
        training_fraction: Fraction of data to use for training
        validation_fraction: Fraction for train/validation split
        n_batch: Batch size for training
        training_kwargs: Additional training parameters
        stub: If True, creates mock model outputs without training
        loop_time: Interval (seconds) to poll for job completion
    """

    def __init__(
        self,
        training_script: os.PathLike,
        remote_fs: FileSystemHandlerBase,
        config_file: os.PathLike,
        remote_savepath: Optional[os.PathLike] = None,
        training_fraction: float = 0.9,
        validation_fraction: float = 0.2,
        n_batch: int = 32,
        training_kwargs: dict = None,
        stub: bool = False,
        loop_time: float = 30,
    ):
        super().__init__()
        self.training_script = training_script
        self.training_kwargs = training_kwargs or {
            "model.max_ell": 3,
            "model.num_interactions": 2,
            "model.hidden_irreps": "32x0e + 32x1o",
            "datasets.r_max": 4.0,
            "datasets.valid_fraction": 0.20,
            "datasets.n_edge": 200,
            "datasets.n_node": 50,
            "optimizer.max_num_intervals": 100,
            "optimizer.steps_per_interval": 128,
        }
        self.training_fraction = training_fraction
        self.validation_fraction = validation_fraction
        self.n_batch = n_batch
        self.remote_fs = remote_fs
        self.loop_time = loop_time
        self.config_file = config_file
        self.remote_savepath = (
            Path(remote_savepath) if remote_savepath else None
        )
        self.stub = stub

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
            try:
                from jaxnest.backends.mace import mace_train_utils
                from jaxnest.backends.mace.mace_train_utils import (
                    write_dict_to_gin,
                )
                from jaxnest.input_parser import update_default_args
            except ImportError:
                raise ImportError(
                    "MACETrainer requires 'jaxnest' package. "
                    "Install it or use stub=True for testing."
                )

            from math import ceil

            from remote_handler.core.cluster_handler import ClusterConfig
            from remote_handler.core.file_handler import LocalFileSystem

            # Validate config file name
            if not os.path.split(self.config_file)[1] == "config.gin":
                raise Exception("config_file needs to be named 'config.gin'!")

            # Parse gin config and merge with training_kwargs
            gin_kwargs = mace_train_utils.parse_gin_to_dict(self.config_file)
            update_default_args(self.training_kwargs, gin_kwargs)

            # Calculate learning rate schedule boundaries
            if "optimizer.max_num_intervals" in gin_kwargs:
                num_epochs = int(gin_kwargs["optimizer.max_num_intervals"])
                gin_kwargs[
                    "piecewise_constant_schedule.boundaries_and_scales"
                ] = {
                    ceil(num_epochs * 0.2): 0.1,
                    ceil(num_epochs * 0.8): 0.1,
                }

            cluster_config = ClusterConfig(
                jobfile_name="run_slurm.sh",
                n_processors=1,
                job_type_template="body_single.sh.j2",
            )

            # Create JSON input file for training script
            input_file = self.results_dir / "input_mace.json"

            self.train_task = SingleScriptTask(
                cluster_config=cluster_config,
                script_file=self.training_script,
                input_file=input_file,
                local_savepath=self.results_dir,
                remote_fs=self.remote_fs,
                logfile="train.log",
                remote_savepath=self.remote_savepath,
                collect_only=False,
                loop_time=self.loop_time,
                jobname="MACE_training",
            )

            # Prepare training configuration
            # Copy database files to remote
            remote_dbs = []
            for db_file in db_files:
                remote_dbs.append(str(self.train_task.copy_file(db_file)))

            # Write gin config to remote
            local_fs = LocalFileSystem()
            temp_config_file = Path(local_fs.create_tempdir()) / "config.gin"
            write_dict_to_gin(gin_kwargs, temp_config_file)
            self.remote_fs.put(
                temp_config_file, self.train_task.calcdir / "config.gin"
            )

            # Prepare JSON input dict
            json_dict = dict(
                filepath=str(self.train_task.calcdir),
                training_fraction=self.training_fraction,
                validation_fraction=self.validation_fraction,
                n_batch=self.n_batch,
                stub=self.stub,
                kwargs=gin_kwargs,
                structure_databases=remote_dbs,
            )

            # Store for later access
            self.json_dict = json_dict

            # Write JSON input file to remote
            self.remote_fs.write_dict_to_json(
                self.train_task.calcdir / "input_mace.json", json_dict
            )

            # Allocate data (copies script and writes jobfile)
            self.train_task.allocate_data()

    def _create_stub_outputs(self) -> Path:
        """
        Create mock MACE model file for testing.

        Generates fake trained model with gin configuration.

        Returns:
            Path to mock model file
        """
        logger.info("Creating stub MACE model file")

        # Create logs directory
        logs_dir = self.results_dir / "logs"
        os.makedirs(logs_dir, exist_ok=True)

        # Create mock model file
        model_file = logs_dir / "mace_model_stub.pkl"

        # Mock gin configuration string
        mock_gin_config = """
# Mock MACE Configuration
model.max_ell = 3
model.num_interactions = 2
model.hidden_irreps = '32x0e + 32x1o'
datasets.r_max = 4.0
datasets.valid_fraction = 0.20
optimizer.max_num_intervals = 100
optimizer.steps_per_interval = 128
"""

        # Mock model parameters (simplified)
        mock_params = {
            "energy_scale": 1.0,
            "force_scale": 1.0,
            "num_atoms_trained": 5000,
            "final_train_loss": np.random.uniform(0.01, 0.05),
            "final_val_loss": np.random.uniform(0.02, 0.08),
        }

        # MACE models store gin config and params in pickle
        with open(model_file, "wb") as f:
            pickle.dump(mock_gin_config, f)
            pickle.dump(mock_params, f)

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
            logger.info("Running in stub mode - creating mock MACE model")
            model_file = self._create_stub_outputs()
        else:
            # Run actual training
            self.train_task.run()

            if not self.train_task.finish_check():
                raise Exception("Training did not finish successfully!")

            # Find trained model
            logs_dir = (
                self.results_dir
                / self.train_task.json_dict["kwargs"]["logs.directory"]
            )
            model_files = grep(str(logs_dir), r".*\.pkl")

            if not model_files:
                raise Exception("No model file found after training!")

            model_file = model_files[0]

            # Update dataset paths in the model for portability
            try:
                self._fix_dataset_paths(model_file)
            except Exception as e:
                logger.warning(f"Could not fix dataset paths: {e}")

        logger.info(f"Trained model available at: {model_file}")
        return model_file

    def _fix_dataset_paths(self, model_file: Path) -> None:
        """
        Fix dataset paths in MACE model for portability.

        MACE models store absolute paths to train/val/test sets which break
        when moving from remote to local. This updates them to local paths.

        Args:
            model_file: Path to model file to fix
        """
        try:
            from jaxnest.backends.mace import mace_train_utils
        except ImportError:
            logger.warning(
                "Could not import mace_train_utils, skipping path fix"
            )
            return

        with open(model_file, "rb") as f:
            gin_str_init = pickle.load(f)
            params = pickle.load(f)

        keys = [
            "datasets.train_path",
            "datasets.valid_path",
            "datasets.test_path",
        ]
        vals = mace_train_utils.extract_values_from_gin_str(gin_str_init, keys)

        homepath = self.results_dir
        gin_str_res = mace_train_utils.update_dataset_paths(
            gin_str_init,
            homepath / os.path.split(vals[keys[0]])[-1],
            homepath / os.path.split(vals[keys[1]])[-1],
            homepath / os.path.split(vals[keys[2]])[-1],
        )

        # Backup original and write updated version
        import shutil

        backup_file = model_file.parent / f"old_{model_file.name}"
        shutil.move(model_file, backup_file)

        with open(model_file, "wb") as f:
            pickle.dump(gin_str_res, f)
            pickle.dump(params, f)

        logger.info(f"Updated dataset paths in {model_file}")


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
        remote_savepath: Optional[os.PathLike] = None,
        training_kwargs: dict = None,
        stub: bool = False,
        loop_time: float = 30,
    ):
        super().__init__()
        self.training_script = training_script
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
            from remote_handler.core.cluster_handler import ClusterConfig

            cluster_config = ClusterConfig(
                jobfile_name="run_slurm.sh",
                n_processors=1,
                job_type_template="body_single.sh.j2",
            )

            # Create JSON input file for training script
            input_file = self.results_dir / "input_nnff.json"

            self.train_task = SingleScriptTask(
                cluster_config=cluster_config,
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
                raise Exception("Training did not finish successfully!")

            # Get model file path from training kwargs
            model_file = (
                self.results_dir
                / os.path.split(
                    self.train_task.training_kwargs["pickle_file_name"]
                )[-1]
            )

            if not os.path.exists(model_file):
                raise Exception(f"Model file not found at {model_file}!")

        logger.info(f"Trained model available at: {model_file}")
        return model_file
