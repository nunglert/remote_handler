import logging
import os
import shutil
from pathlib import Path
from random import sample
from typing import Optional

import ase.io
from ase import Atoms
from jaxnest import input_parser
from jaxnest.active_learning import remote_tasks
from jaxnest.active_learning.remote_tasks import NeuralILTrainingTask
from jaxnest.file_utils import grep
from jaxnest.input_parser import update_default_args

from remote_handler.core.active_learning_base import (
    ActiveLearningHandler,
    GroundTruthEvaluator,
    ModelTrainer,
    Sampler,
)
from remote_handler.core.file_handler import FileSystemHandlerBase
from remote_handler.core.tasks import SingleScriptTask
from remote_handler.dft_runner import VASPHandler

logger = logging.getLogger(__name__)


class NSTrajectorySampler(Sampler):
    def __init__(
        self,
        ns_exe: os.PathLike,
        remote_fs: FileSystemHandlerBase,
        initial_input_file: os.PathLike,
        n_gpu_parallel: int,
        sample_size: int = 100,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
    ):
        super().__init__()
        self.ns_exe = ns_exe
        self.remote_fs = remote_fs
        self.n_gpu_parallel = n_gpu_parallel
        self.remote_savepath = Path(remote_savepath)
        self.sample_size = sample_size
        self.loop_time = loop_time

        with open(initial_input_file, "r") as f:
            self.ns_params = input_parser.input_str_to_dict(f.read())

    def update(
        self, calcdir: os.PathLike, model_file: os.PathLike, **kwargs
    ) -> None:
        self.ns_dir = Path(calcdir) / "ns_sampler"
        self.results_dir = self.ns_dir / "results"
        os.makedirs(self.ns_dir, exist_ok=True)

        # remove results dir from crashed run
        if os.path.exists(self.results_dir):
            shutil.rmtree(self.results_dir)

        it = str(calcdir).split("/")[-1]
        inputfile_path = self.ns_dir / "ns.inp"

        kwargs["pickle_file"] = str(model_file)
        update_default_args(kwargs, self.ns_params)
        input_parser.write_ns_inputfile(inputfile_path, self.ns_params)

        # SingleScriptTask will create self.remote_savepath on the remote
        # machine and run the simulation there. When finished, it will copy the
        # results back to self.results_dir
        self.ns_task = SingleScriptTask(
            script_file=self.ns_exe,
            input_file=inputfile_path,
            local_savepath=self.results_dir,
            remote_fs=self.remote_fs,
            logfile="ns.log",
            remote_savepath=self.remote_savepath,
            jobname=f"AL_NS_{it}",
            n_processors=self.n_gpu_parallel,
            job_type_template="body_ns.sh.j2",
            loop_time=self.loop_time,
        )

    def get_sampled_configs(self) -> list[Atoms]:
        self.ns_task.run()
        ns_output_data = self.results_dir / "output_data"
        if not os.path.exists(ns_output_data):
            raise Exception(
                "NS did not run succesfully, couldn't find output_data dir!"
            )

        tfiles = grep(ns_output_data, "traj")

        # All trajs are same length, so we can compute the slice that leads
        # to the desired sample_size for the first one.
        n_runs = len(tfiles)
        assert n_runs > 0, "Could not successfully collect trajectory files!"
        n_samples = len(ase.io.read(tfiles[0], index=":"))
        assert n_samples > 0, "Trajectories are empty, check NS run!"

        sample_size_per_run = self.sample_size // n_runs
        every = n_samples // sample_size_per_run
        if sample_size_per_run > n_samples:
            logger.warning(
                f"{sample_size_per_run=} is smaller than {n_samples=}. Taking "
                "all samples. Are you sure this is intended?"
            )
            every = 1

        all_ats = []
        for run_id, tfile in enumerate(tfiles):
            ats = ase.io.read(tfile, index=f"::{every}")

            for at in ats:
                at.info["run_id"] = run_id

            all_ats.extend(ats)

        return all_ats


class VASPEvaluator(GroundTruthEvaluator):
    def __init__(
        self,
        singlepoint_script: os.PathLike,
        remote_fs: FileSystemHandlerBase,
        num_parallel: int,
        remote_savepath: Optional[os.PathLike] = None,
        launcher: str = "mpirun -n",
        incar_kwargs: dict = {
            # "istart": 0,
            # "icharg": 2,
            "xc": "PBE",
            "encut": 300,
            "prec": "Normal",
            "ismear": -1,
            "sigma": 0.025,
            "ibrion": -1,
            "ediff": 1e-5,
            # "isym": 0,
            "symprec": 1e-5,
            "lwave": False,
            "lcharg": False,
            "lreal": False,
            # "kspacing": 2.0
            "kspacing": 2.0,
        },
        stub: bool = False,  # just for testing
        loop_time: float = 30,
        required_fraction: float = 0.5,
    ):
        super().__init__()
        self.singlepoint_script = singlepoint_script
        self.incar_kwargs = incar_kwargs
        self.remote_fs = remote_fs
        self.num_parallel = num_parallel
        self.stub = stub
        self.launcher = launcher
        self.loop_time = loop_time
        self.required_fraction = required_fraction
        self.remote_savepath = Path(remote_savepath)

    def update(
        self, calcdir: os.PathLike, collected_file: os.PathLike, **kwargs
    ) -> None:
        self.dft_dir = Path(calcdir) / "dft"
        self.results_dir = self.dft_dir / "results"
        os.makedirs(self.dft_dir, exist_ok=True)
        it = str(calcdir).split("/")[-1]

        self.vasp_handler = VASPHandler(
            loop_time=self.loop_time,
            jobname="dft",
            every=1,
            start=None,
            end=None,
            logfile="dft.log",
            tag="dft",
            stub=self.stub,
            collect_only=False,
            remote_fs=self.remote_fs,
            savepath=self.remote_fs.slurm_config["savepath"],
            local_savepath=self.results_dir,
            traj_files=[collected_file],
            python_script_path=self.singlepoint_script,
            num_parallel=self.num_parallel,
            incar_kwargs=self.incar_kwargs,
            launcher=self.launcher,
        )

    def get_evaluated_configs(self) -> list[Atoms]:
        self.vasp_handler.run()

        if not self.vasp_handler.finish_check(self.required_fraction):
            raise Exception(
                f"Less than {self.required_fraction * 100}% ran successfully. "
                "Something probably went wrong. Otherwise, lower the "
                "required_fraction parameter."
            )

        all_ats = []
        for calcpath in grep(self.results_dir / "calcdirs", ""):
            try:
                at = ase.io.read(Path(calcpath) / "vasprun.xml")
                all_ats.append(at)
            except:
                logger.info(f"VASP run failed for {calcpath=}")

        return all_ats


class NeuralilILTrainer(ModelTrainer):
    def __init__(
        self,
        training_script: os.PathLike,
        remote_fs: FileSystemHandlerBase,
        remote_savepath: Optional[os.PathLike] = None,
        training_kwargs: dict = {
            "r_cut": 4.0,
            "training_fraction": 0.75,
            "supercell_diag": [3, 3, 3],
            "high_energy_penalty": -5.0,
            "n_ensemble": 30,
            "save_reduced": 5,
            "n_batch": 32,
        },
        stub: bool = False,  # just for testing
        loop_time: float = 30,
    ):
        super().__init__()
        self.training_script = training_script
        self.training_kwargs = training_kwargs
        self.remote_fs = remote_fs
        self.loop_time = loop_time
        self.remote_savepath = Path(remote_savepath)
        self.training_kwargs["stub"] = stub

    def update(
        self, calcdir: os.PathLike, db_files: list[os.PathLike], **kwargs
    ) -> None:
        self.train_dir = Path(calcdir) / "training"
        self.results_dir = self.train_dir / "results"
        os.makedirs(self.train_dir)
        it = str(calcdir).split("/")[-1]

        self.train_task = NeuralILTrainingTask(
            script_file=self.training_script,
            input_file=None,
            local_savepath=self.results_dir,
            remote_fs=self.remote_fs,
            remote_savepath=self.remote_savepath,
            logfile="train.log",
            training_databases=db_files,
            training_kwargs=self.training_kwargs,
            loop_time=self.loop_time,
        )

    def get_trained_model(self) -> os.PathLike:
        self.train_task.run()

        if not self.train_task.finish_check():
            raise Exception("Training not finished successfully!")

        model_file = (
            self.results_dir
            / os.path.split(
                self.train_task.training_kwargs["pickle_file_name"]
            )[-1]
        )
        return model_file
