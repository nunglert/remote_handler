import os
from typing import Optional

from remote_handler.core.file_handler import FileSystemHandlerBase
from remote_handler.core.tasks import SingleScriptTask

os.environ["REMOTE_HANDLER_SLURM_TEMPLATES"] = os.getcwd()


class NeuralILTrainingTask(SingleScriptTask):
    def __init__(
        self,
        script_file: os.PathLike,
        input_file: os.PathLike,
        local_savepath: os.PathLike,
        remote_fs: FileSystemHandlerBase,
        training_databases: list[os.PathLike],
        label: str = "",
        training_kwargs: dict = {
            "r_cut": 4.0,
            "training_fraction": 0.75,
            "supercell_diag": [3, 3, 3],
            "high_energy_penalty": -5.0,
            "n_ensemble": 30,
            "save_reduced": 5,
            "n_batch": 32,
        },
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

        self.label = label
        self.training_databases = training_databases
        self.training_kwargs = training_kwargs

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
