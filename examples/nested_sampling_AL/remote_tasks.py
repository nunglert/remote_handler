import logging
import os
from copy import deepcopy
from math import ceil
from pathlib import Path
from typing import Optional

from jaxnest import input_parser
from jaxnest.backends.mace.mace_train_utils import write_dict_to_gin
from jaxnest.input_parser import update_default_args

from remote_handler.core.file_handler import (
    FileSystemHandlerBase,
    LocalFileSystem,
)
from remote_handler.core.tasks import SingleScriptTask

os.environ["REMOTE_HANDLER_SLURM_TEMPLATES"] = os.getcwd()

logger = logging.getLogger(__name__)


class JAXNESTTask(SingleScriptTask):
    def __init__(
        self,
        ns_exe: os.PathLike,
        input_file: os.PathLike,
        local_savepath: os.PathLike,
        remote_fs: FileSystemHandlerBase,
        model_file: os.PathLike,
        n_processors: int,
        ns_kwargs: dict = {},
        label: str = "",
        logfile: os.PathLike = "task.log",
        remote_savepath: Optional[os.PathLike] = None,
        collect_only: bool = False,
        loop_time: float = 30,
    ):
        super().__init__(
            ns_exe,
            input_file,
            local_savepath,
            remote_fs,
            logfile,
            remote_savepath,
            collect_only,
            loop_time=loop_time,
            jobname=label,
            n_processors=n_processors,
            jobfile_name="run_slurm.sh",
            job_type_template="body_ns.sh.j2",
            # job_type_template="body_neuralil_train.sh.j2",
        )

        if not self.remote_fs.exists(ns_exe):
            raise Exception("Provided ns_exe does not exist on remote_fs!")

        self.label = label
        self.model_file = model_file
        self.ns_kwargs = ns_kwargs
        self.input_file = input_file

    def allocate_data(self):
        """ """
        self.model_file_remote = self.copy_file(self.model_file)
        # self.script_file_remote = self.copy_file(self.script_file)

        with open(self.input_file, "r") as f:
            input_kwargs = input_parser.input_str_to_dict(f.read())

        self.ns_kwargs["pickle_file"] = str(self.model_file_remote)
        update_default_args(self.ns_kwargs, input_kwargs)
        input_parser.write_ns_inputfile(self.input_file, input_kwargs)
        self.input_file_remote = self.copy_file(self.input_file)

        self.slurm_handler.write_jobfile(
            self.input_file_remote, self.script_file
        )


class NeuralILTrainingTask(SingleScriptTask):
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
        script_file: os.PathLike,
        input_file: os.PathLike,
        local_savepath: os.PathLike,
        remote_fs: FileSystemHandlerBase,
        training_databases: list[os.PathLike],
        label: str = "",
        training_kwargs: dict = {},
        logfile: os.PathLike = "task.log",
        remote_savepath: Optional[os.PathLike] = None,
        collect_only: bool = False,
        loop_time: float = 30,
    ):
        super().__init__(
            script_file,
            input_file,
            local_savepath,
            remote_fs,
            logfile,
            remote_savepath,
            collect_only,
            loop_time=loop_time,
            jobname="NeuralIL_training",
            n_processors=1,
            jobfile_name="run_slurm.sh",
            job_type_template="body_single.sh.j2",
            # job_type_template="body_neuralil_train.sh.j2",
        )
        update_default_args(training_kwargs, self.default_training_kwargs)
        self.training_kwargs = deepcopy(self.default_training_kwargs)

        logger.info(
            f"Starting training with training_kwargs:\n{self.training_kwargs}"
        )

        self.label = label
        self.training_databases = training_databases

    def allocate_data(self):
        """Sets up the json input file for the training script, copies both
        the training script and the input file to the remote calculation
        directory and writes the slurm batch script to start the training.
        """

        remote_dbs = []
        for db_file in self.training_databases:
            remote_dbs.append(str(self.copy_file(db_file)))
        self.training_kwargs["structure_databases"] = remote_dbs
        self.training_kwargs["filepath"] = str(self.calcdir)
        self.training_kwargs["pickle_file_name"] = str(
            self.calcdir / f"model{self.label}.pkl"
        )

        self.script_file_remote = self.copy_file(self.script_file)
        self.input_file_remote = self.remote_fs.write_dict_to_json(
            self.calcdir / "input_nnff.json", self.training_kwargs
        )
        self.slurm_handler.write_jobfile(".", self.script_file_remote)


class MACETrainingTask(SingleScriptTask):
    """
    Args:
        config_file: A gin config file used as a template for default
            parameters.
        training_kwargs: Dictionary for modifying specific parameters. These
            will overwerite the ones parsed from config_file
    """

    def __init__(
        self,
        script_file: os.PathLike,
        # input_file: os.PathLike,
        config_file: os.PathLike,
        local_savepath: os.PathLike,
        remote_fs: FileSystemHandlerBase,
        training_databases: list[os.PathLike],
        label: str = "",
        training_fraction=0.9,
        validation_fraction=0.2,
        n_batch=32,
        training_kwargs: dict = {
            "model.max_ell": 3,
            "model.num_interactions": 2,
            "model.hidden_irreps": "32x0e + 32x1o",
            "datasets.r_max": 4.0,
            "datasets.valid_fraction": 0.20,
            "datasets.n_edge": 200,
            "datasets.n_node": 50,
            "optimizer.max_num_intervals": 100,
            "optimizer.steps_per_interval": 128,
            # "piecewise_constant_schedule.boundaries_and_scales": {
            #     100: 0.1,  # divide the learning rate by 10 after 100 intervals
            #     1000: 0.1,  # divide the learning rate by 10 after 1000 intervals
            # },
        },
        logfile: os.PathLike = "task.log",
        remote_savepath: Optional[os.PathLike] = None,
        collect_only: bool = False,
        loop_time: float = 30,
        stub: bool = False,
    ):
        super().__init__(
            script_file,
            None,  # input_file
            local_savepath,
            remote_fs,
            logfile,
            remote_savepath,
            collect_only,
            loop_time=loop_time,
            jobname="MACE_training",
            n_processors=1,
            jobfile_name="run_slurm.sh",
            job_type_template="body_single.sh.j2",
            # job_type_template="body_neuralil_train.sh.j2",
        )

        self.label = label
        self.training_databases = training_databases
        self.training_kwargs = training_kwargs
        self.config_file = config_file

        if not os.path.split(self.config_file)[1] == "config.gin":
            raise Exception("config_file needs to be named 'config.gin'!")

        from jaxnest.backends.mace import mace_train_utils

        self.gin_kwargs = mace_train_utils.parse_gin_to_dict(self.config_file)
        update_default_args(self.training_kwargs, self.gin_kwargs)

        # assert "n_batch" in training_kwargs.keys()
        # assert "training_fraction" in training_kwargs.keys()
        # assert "validation_fraction" in training_kwargs.keys()

        if "optimizer.max_num_intervals" in self.gin_kwargs:
            num_epochs = int(self.gin_kwargs["optimizer.max_num_intervals"])
            self.gin_kwargs[
                "piecewise_constant_schedule.boundaries_and_scales"
            ] = {
                ceil(
                    num_epochs * 0.2
                ): 0.1,  # divide the learning rate by 10 after 100 intervals
                ceil(
                    num_epochs * 0.8
                ): 0.1,  # divide the learning rate by 10 after 1000 intervals
            }

        self.json_dict = dict(
            filepath=str(self.calcdir),
            training_fraction=training_fraction,
            validation_fraction=validation_fraction,
            n_batch=n_batch,
            stub=stub,
            kwargs=self.gin_kwargs,
        )
        write_dict_to_gin(self.gin_kwargs, self.config_file)

    def allocate_data(self):
        """Sets up the json input file for the training script, copies both
        the training script and the input file to the remote calculation
        directory and writes the slurm batch script to start the training.
        """

        remote_dbs = []
        for db_file in self.training_databases:
            remote_dbs.append(str(self.copy_file(db_file)))
        # self.copy_file(self.config_file)

        local_fs = LocalFileSystem()
        file = Path(local_fs.create_tempdir()) / "config.gin"
        write_dict_to_gin(self.gin_kwargs, file)
        self.remote_fs.put(file, self.calcdir / "config.gin")

        self.json_dict["structure_databases"] = remote_dbs

        # self.training_kwargs["structure_databases"] = remote_dbs
        # self.training_kwargs["filepath"] = str(self.calcdir)
        # self.training_kwargs["pickle_file_name"] = str(
        #     self.calcdir / f"model{self.label}.pkl"
        # )
        # assert "r_cut" in self.training_kwargs.keys()
        # assert "training_fraction" in self.training_kwargs.keys()
        # assert "kwargs" in self.training_kwargs.keys()
        # assert "stub" in self.training_kwargs.keys()
        # assert "n_batch" in self.training_kwargs.keys()

        self.script_file_remote = self.copy_file(self.script_file)

        self.input_file_remote = self.remote_fs.write_dict_to_json(
            self.calcdir / "input_mace.json", self.json_dict
        )
        self.slurm_handler.write_jobfile(".", self.script_file_remote)
