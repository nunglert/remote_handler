import logging
import os
import re
import shutil
from pathlib import Path
from random import sample
from typing import Callable, Optional

import ase.io
import jax.numpy as jnp
import numpy as np
from ase import Atoms
from jaxnest import input_parser
from jaxnest.active_learning import remote_tasks, similarity_filter
from jaxnest.active_learning.remote_tasks import (
    JAXNESTTask,
    MACETrainingTask,
    NeuralILTrainingTask,
)
from jaxnest.backends.mace import mace_train_utils
from jaxnest.file_utils import grep, prepend_filename, secure_declare_path
from jaxnest.input_parser import update_default_args

from remote_handler.core.file_handler import FileSystemHandlerBase
from remote_handler.core.tasks import SingleScriptTask
from remote_handler.dft_runner import VASPHandler
from remote_handler.optimizer import RemoteOptimizationHandler
from remote_handler.workflows.active_learning_base import (
    ActiveLearningHandler,
    GroundTruthEvaluator,
    ModelTrainer,
    Sampler,
)

logger = logging.getLogger(__name__)


def sample_from_trajectories(
    ns_output_data: str,
    results_dir: Optional[Path] = None,
    model_file: Optional[str] = None,
    sample_size: int = 100,
    resample_size: Optional[int] = None,
    energy_uncty_weight: float = 0.0,
    force_uncty_weight: float = 1.0,
    optimize_traj: Optional[
        Callable[[list[str], os.PathLike], RemoteOptimizationHandler]
    ] = None,
    seed: Optional[int] = None,
) -> list[Atoms]:
    np.random.seed(seed)
    tfiles = grep(ns_output_data, "traj")

    # All trajs are same length, so we can compute the slice that leads
    # to the desired sample_size for the first one.
    n_runs = len(tfiles)
    assert n_runs > 0, "Could not successfully collect trajectory files!"
    n_samples = len(ase.io.read(tfiles[0], index=":"))
    assert n_samples > 0, "Trajectories are empty, check NS run!"

    sample_size_per_run = sample_size // n_runs
    every = n_samples // sample_size_per_run
    if sample_size_per_run > n_samples:
        logger.warning(
            f"{sample_size_per_run=} is smaller than {n_samples=}. Taking "
            "all samples. Are you sure this is intended?"
        )
        every = 1

    if optimize_traj is not None:
        assert model_file is not None, "Need model_file!"
        optimizer = optimize_traj(tfiles, model_file)
        optimizer.run()

    all_ats = []
    for run_id, tfile in enumerate(tfiles):
        ats = ase.io.read(tfile, index=f"::{every}")

        for at in ats:
            at.info["run_id"] = run_id

        all_ats.extend(ats)

    if resample_size is not None:
        assert results_dir is not None, "Need results_dir!"
        samples_per_run = len(all_ats) // n_runs
        indices = np.arange(len(all_ats))
        iterations = [at.info["iter"] for at in all_ats]
        index_pairs = np.vstack(
            [
                indices // samples_per_run,
                indices % samples_per_run,
                iterations,
            ]
        ).T

        funcs = [at.info["forces_uncert"] for at in all_ats]
        euncs = [at.info["energy_uncert"] for at in all_ats]
        funcs_norm = funcs / np.sum(funcs)
        euncs_norm = euncs / np.sum(euncs)
        score = (
            force_uncty_weight * funcs_norm + energy_uncty_weight * euncs_norm
        )
        score = score / np.sum(score)
        inds = np.random.choice(indices, resample_size, replace=False, p=score)
        inds.sort()
        all_ats = [all_ats[i] for i in inds]

        logger.info(f"Resampled following traj configs:\n {index_pairs[inds]}")

        np.savetxt(
            results_dir / "resampled_inds.csv",
            index_pairs[inds],
            header="run index, sample index, iteration",
        )
    return all_ats


class NSTrajectorySampler(Sampler):
    def __init__(
        self,
        ns_exe: os.PathLike,
        remote_fs: FileSystemHandlerBase,
        initial_input_file: os.PathLike,
        n_gpu_parallel: int,
        sample_size: int = 100,
        resample_size: Optional[int] = None,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        energy_uncty_weight: float = 1.0,
        force_uncty_weight: float = 1.0,
        optimize_traj: Optional[
            Callable[[list[str], os.PathLike], RemoteOptimizationHandler]
        ] = None,
        optimize_walkers: Optional[
            Callable[[list[str], os.PathLike], RemoteOptimizationHandler]
        ] = None,
    ):
        super().__init__()
        self.ns_exe = ns_exe
        self.remote_fs = remote_fs
        self.n_gpu_parallel = n_gpu_parallel
        self.remote_savepath = Path(remote_savepath)
        self.sample_size = sample_size
        self.resample_size = resample_size
        self.loop_time = loop_time
        self.optimize_traj = optimize_traj
        self.optimize_walkers = optimize_walkers
        self.energy_uncty_weight = energy_uncty_weight
        self.force_uncty_weight = force_uncty_weight

        assert self.resample_size <= self.sample_size

        with open(initial_input_file, "r") as f:
            self.ns_params = input_parser.input_str_to_dict(f.read())

        self.initial_input_file = initial_input_file

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
        # inputfile_path = self.ns_dir / "ns.inp"
        self.model_file = model_file

        self.ns_task = JAXNESTTask(
            ns_exe=self.ns_exe,
            input_file=self.initial_input_file,
            local_savepath=self.results_dir,
            remote_fs=self.remote_fs,
            logfile="ns.log",
            model_file=model_file,
            ns_kwargs=kwargs,
            remote_savepath=self.remote_savepath,
            label=f"AL_NS_{it}",
            n_processors=self.n_gpu_parallel,
            # job_type_template="body_ns.sh.j2",
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

        if self.optimize_traj is not None:
            optimizer = self.optimize_traj(tfiles, self.model_file)
            optimizer.run()

        all_ats = []
        for run_id, tfile in enumerate(tfiles):
            ats = ase.io.read(tfile, index=f"::{every}")

            for at in ats:
                at.info["run_id"] = run_id

            all_ats.extend(ats)

        if self.resample_size is not None:
            samples_per_run = len(all_ats) // n_runs
            indices = np.arange(len(all_ats))
            iterations = [at.info["iter"] for at in all_ats]
            index_pairs = np.vstack(
                [
                    indices // samples_per_run,
                    indices % samples_per_run,
                    iterations,
                ]
            ).T

            funcs = [at.info["forces_uncert"] for at in all_ats]
            euncs = [at.info["energy_uncert"] for at in all_ats]
            funcs_norm = funcs / np.sum(funcs)
            euncs_norm = euncs / np.sum(euncs)
            score = (
                self.force_uncty_weight * funcs_norm
                + self.energy_uncty_weight * euncs_norm
            )
            score = score / np.sum(score)
            inds = np.random.choice(
                indices, self.resample_size, replace=False, p=score
            )
            inds.sort()
            all_ats = [all_ats[i] for i in inds]

            logger.info(
                f"Resampled following traj configs:\n {index_pairs[inds]}"
            )

            np.savetxt(
                self.results_dir / "resampled_inds.csv",
                index_pairs[inds],
                header="run index, sample index, iteration",
            )

        return all_ats


class NSCollectorSampler(Sampler):
    def __init__(
        self,
        ns_exe: os.PathLike,
        remote_fs: FileSystemHandlerBase,
        initial_input_file: os.PathLike,
        n_gpu_parallel: int,
        sample_size: int = 100,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        optimize_traj: Optional[
            Callable[[list[str], os.PathLike], RemoteOptimizationHandler]
        ] = None,
        optimize_walkers: Optional[
            Callable[[list[str], os.PathLike], RemoteOptimizationHandler]
        ] = None,
    ):
        super().__init__()
        self.ns_exe = ns_exe
        self.remote_fs = remote_fs
        self.n_gpu_parallel = n_gpu_parallel
        self.remote_savepath = Path(remote_savepath)
        self.sample_size = sample_size
        self.loop_time = loop_time
        self.optimize_traj = optimize_traj
        self.optimize_walkers = optimize_walkers

        self.initial_input_file = initial_input_file

        with open(initial_input_file, "r") as f:
            self.ns_params = input_parser.input_str_to_dict(f.read())

        assert input_parser.str_to_logical(
            self.ns_params["do_active_learning"]
        ), "do_active_learning needs to be True!"

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
        self.model_file = model_file

        # kwargs["pickle_file"] = str(model_file)
        # update_default_args(kwargs, self.ns_params)
        # input_parser.write_ns_inputfile(inputfile_path, self.ns_params)

        # SingleScriptTask will create self.remote_savepath on the remote
        # machine and run the simulation there. When finished, it will copy the
        # results back to self.results_dir
        self.ns_task = JAXNESTTask(
            ns_exe=self.ns_exe,
            input_file=self.initial_input_file,
            local_savepath=self.results_dir,
            remote_fs=self.remote_fs,
            logfile="ns.log",
            model_file=model_file,
            ns_kwargs=kwargs,
            remote_savepath=self.remote_savepath,
            label=f"AL_NS_{it}",
            n_processors=self.n_gpu_parallel,
            # job_type_template="body_ns.sh.j2",
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

        if self.optimize_traj is not None:
            optimizer = self.optimize_traj(tfiles, self.model_file)
            optimizer.run()

        configs_file = self.results_dir / "AL_configs_final.db"

        import ase.db

        all_atoms = []
        with ase.db.connect(configs_file) as db:
            for row in db.select():
                all_atoms.append(row.toatoms())

        return all_atoms


def failed_convergence(outfile):
    with open(outfile, "r") as f:
        data = f.read()
    return bool(
        re.search(r"electronic self-consistency was not achieved", data)
    )


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
        slurm_header_add: str = "#SBATCH --time=0-02:00:00\nmodule purge\nmodule load vasp/6.4.2_intel\nsource /opt/intel/oneapi/setvars.sh --force\nexport VASP_PP_PATH=/home/nunglert/cellar/vasp/paw",
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
        self.slurm_header_add = slurm_header_add
        # self.remote_savepath = Path(remote_savepath)

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
            slurm_header_add=self.slurm_header_add,
        )

    def get_evaluated_configs(self) -> list[Atoms]:
        self.vasp_handler.run()

        if not self.vasp_handler.finish_check(self.required_fraction):
            logger.warning(
                f"Less than {self.required_fraction * 100}% ran successfully. "
                "Something probably went wrong. Otherwise, lower the "
                "required_fraction parameter."
            )

        all_ats = []
        for calcpath in grep(self.results_dir / "calcdirs", ""):
            try:
                vaspfile = Path(calcpath) / "vasprun.xml"
                outfile = Path(calcpath) / "OUTCAR"

                if failed_convergence(outfile):
                    raise Exception("VASP SCF not converged!")

                at = ase.io.read(vaspfile)
                at.get_potential_energy()
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
        training_kwargs: dict = {},
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
        if os.path.exists(self.train_dir):
            logger.warning("Training path already exists, take care!")
        os.makedirs(self.train_dir, exist_ok=True)
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
            raise Exception("Training did not finish successfully!")

        model_file = (
            self.results_dir
            / os.path.split(
                self.train_task.training_kwargs["pickle_file_name"]
            )[-1]
        )
        return model_file


class MACETrainer(ModelTrainer):
    """

    Args:
        training_script: MACE training script
        config_file: gin config file. A templated can be found in
            'test/data/active_learning/train_mace.py'
        remote_savepath: Path on the remote machine which should be the
            working directory.

        ...training params...

        stub: This is a boolean flag that can be passed to the training script.
            If True, the script will shortcut expensive computations and
            produce dummy results. This is just for testing created pipelines.
        loop_time: Time the RemoteTask for the training waits until it checks
            again for the job being finished.
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
        stub: bool = False,  # just for testing
        loop_time: float = 30,
    ):
        super().__init__()
        self.training_script = training_script
        self.training_kwargs = training_kwargs
        self.training_fraction = training_fraction
        self.validation_fraction = validation_fraction
        self.n_batch = n_batch
        self.remote_fs = remote_fs
        self.loop_time = loop_time
        self.config_file = config_file
        self.remote_savepath = Path(remote_savepath)
        self.stub = stub

    def update(
        self, calcdir: os.PathLike, db_files: list[os.PathLike], **kwargs
    ) -> None:
        self.train_dir = Path(calcdir) / "training"
        self.results_dir = self.train_dir / "results"
        if os.path.exists(self.train_dir):
            logger.warning("Training path already exists, take care!")
        os.makedirs(self.train_dir, exist_ok=True)
        it = str(calcdir).split("/")[-1]

        self.train_task = MACETrainingTask(
            script_file=self.training_script,
            # input_file=None,
            local_savepath=self.results_dir,
            remote_fs=self.remote_fs,
            remote_savepath=self.remote_savepath,
            logfile="train.log",
            training_databases=db_files,
            training_fraction=self.training_fraction,
            validation_fraction=self.validation_fraction,
            n_batch=self.n_batch,
            training_kwargs=self.training_kwargs,
            loop_time=self.loop_time,
            config_file=self.config_file,
            stub=self.stub,
        )

    def get_trained_model(self) -> os.PathLike:
        self.train_task.run()

        if not self.train_task.finish_check():
            raise Exception("Training not finished successfully!")

        model_file = grep(
            self.results_dir
            / self.train_task.json_dict["kwargs"]["logs.directory"],
            ".*\.pkl",
        )[0]

        # The following block is necessary, since JAX-MACE models can only be
        # loaded with the correct paths to the train, valid and test sets.
        # So once we move everything from a remote machine to a local machine
        # the paths are not reachable anymore and we need to reset them.
        import pickle

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

        shutil.move(model_file, prepend_filename(model_file, "old_"))
        with open(model_file, "wb") as f:
            pickle.dump(gin_str_res, f)
            pickle.dump(params, f)

        return model_file
