import os
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from re import T
from typing import Any, Optional, Union

from remote_handler.atoms_handler import RemoteAtomsHandler
from remote_handler.core import cluster_handler, file_handler, tasks


class RemoteOptimizationHandler(RemoteAtomsHandler):
    """

    Handler to optimize a set of trajectory files on a remote cluster.
    The trajectory files are sharded into batches of size `shard_size`, and each
    batch is optimized in a separate job.
    The optimization is done with either a NNFF or LJ potential, and the
    Steinhardt parameters are computed with either pyscal or quip.
    The optimized configurations are collected and redistributed into the shape
    of the original trajectory files.

    Args:
        traj_files (str): Path to the trajectory files to be optimized.
        remote_fs (file_handler.FileSystemHandlerBase): Remote file system handler.
        shard_size (int): Number of configurations per shard.
        savepath (str): Path to save the optimized trajectory files.
        num_parallel (int): Number of parallel jobs to submit.
        loop_time (float, optional): Time to wait between checking for job
            completion. Defaults to 30.
        jobname (str, optional): Name of the job. Defaults to "opt".
        every (Optional[int], optional): If provided, only every `every`-th
            configuration is optimized. Defaults to None.
        start (Optional[int], optional): If provided, only configurations
            starting from `start` are optimized. Defaults to None.
        end (Optional[int], optional): If provided, only configurations
            up to `end` are optimized. Defaults to None.
        local_savepath (Optional[os.PathLike], optional): If provided,
            a local copy of the optimized trajectory files is saved here.
            Defaults to None.
        logfile (str, optional): Path to the log file. Defaults to "opt.log".
        collect_only (bool, optional): If True, only collect the results
            from previous runs. Defaults to False.
        tag (str, optional): Tag to add to the job name. Defaults to "opt".
        python_script_path (str, optional): Path to the python script that
            performs the optimization and Steinhardt calculation.
            Defaults to
            "/home/aserafini/code/jaxnest_dev/scripts/utility/optimize_traj.py".
        stub (bool, optional): If True, run in stub mode. Defaults to False.
        pickle_file (str, optional): Path to the pickle file containing
            the NNFF model parameters. Required if optimization_backend is "NNFF".
        supercell_trafo (list[int], optional): Supercell transformation matrix
            for the calculator. Defaults to [3, 3, 3].
        max_neighbors_list (list[int], optional): List of maximum number of
            neighbors for each species for the optimization. Defaults to [20, 30, 40, 50, 60].
        max_neighbors_offset (int, optional): Offset to add to the maximum
            number of neighbors for each species for the optimization.
        fmax (float, optional): Maximum force tolerance for the optimization.
        steps (int, optional): Maximum number of optimization steps.
        rmax (float, optional): Maximum distance for the Steinhardt calculation.
        nbins (int, optional): Number of bins for the Steinhardt calculation.
        do_optimization (bool, optional): If True, perform the optimization.
        do_steinhardt (bool, optional): If True, compute the Steinhardt parameters.
        optimization_backend (str, optional): Backend to use for the optimization. 
            Either "NNFF" or "LJ". Defaults to "NNFF".
        steinhardt_backend (str, optional): Backend to use for the Steinhardt
            calculation. Either "pyscal" or "quip". Defaults to "quip".
        steinhardt_l_values (list[int], optional): List of l values to compute
            for the Steinhardt calculation. Defaults to [4, 6].

    """

    def __init__(
        self,
        traj_files: str,
        remote_fs: file_handler.FileSystemHandlerBase,
        shard_size: int,
        savepath: str,
        num_parallel: int,
        loop_time: float = 30,
        jobname: str = "opt",
        every: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        local_savepath: Optional[os.PathLike] = None,
        logfile: str = "opt.log",
        collect_only: bool = False,
        tag: str = "opt",
        python_script_path: str = "/home/aserafini/code/jaxnest_dev/scripts/utility/optimize_traj.py",
        stub: bool = False,
        pickle_file: str = "/home/nunglert/nnff_trainings/velo/si_bartok_gpu7/run11_r2scan_rerun/model_params_plain_ensemble_VELO_8E04F5B1.pkl",
        supercell_trafo: list[int] = [3, 3, 3],
        max_neighbors_list: list[int] = [20, 30, 40, 50, 60],
        max_neighbors_offset: int = 3,
        fmax: float = 0.1,
        steps: int = 200,
        rmax: float = 5.0,
        nbins: int = 200,
        do_optimization: bool = True,
        do_steinhardt: bool = True,
        optimization_backend: str = "NNFF",
        steinhardt_backend: str = "quip",
        steinhardt_l_values: list[int] = [4, 6],
    ):
        super().__init__(
            traj_files,
            remote_fs,
            shard_size,
            savepath,
            num_parallel,
            loop_time,
            jobname,
            every,
            start,
            end,
            local_savepath,
            logfile,
            collect_only,
            tag,
            2,  # n_processors
            slurm_header_add="module load openblas",
        )

        self.logger = tasks.SimpleLogger(logfile)
        self.do_optimization = do_optimization
        self.pickle_file = pickle_file

        self.json_args = {
            "STUB": stub,
            "tag": self.tag,
            # "do_rdf": do_rdf,
            "do_optimization": do_optimization,
            "do_steinhardt": do_steinhardt,
            "calculator_args": dict(
                # pickle_file=str(self.moved_pickle_file),
                supercell_trafo=supercell_trafo,
                max_neighbors_list=max_neighbors_list,
                max_neighbors_offset=max_neighbors_offset,
                optimization_backend=optimization_backend,
            ),
            "optimization_args": dict(
                fmax=fmax,
                steps=steps,
            ),
            "steinhardt_args": dict(
                nbins=nbins,
                cutoff=rmax,
                steinhardt_backend=steinhardt_backend,
                steinhardt_l_values=steinhardt_l_values,
            ),
        }

        # Since we did not initialize the python_script_file, shared_ressources
        # and individual_ressources in parent class, we need to do it here!
        self.python_script_file = python_script_path

        self.shared_ressources = []
        if optimization_backend == "NNFF":
            self.shared_ressources.append(pickle_file)

        if not collect_only:
            self.prepare_individual_data()

    def prepare_individual_data(self):
        config_files = []
        atoms_batches = []
        for calc_idx in range(self.num_array_tasks):
            json_config = deepcopy(self.json_args)

            # TODO: this is dirty, properly solve this
            json_config["DATABASE"] = f"opt.extxyz"
            pickle_file = f"{os.path.split(self.pickle_file)[1]}"
            json_config["calculator_args"]["pickle_file"] = (
                (Path(self.savepath) / pickle_file).resolve().__str__()
            )

            config_files.append(
                tasks.DictJSON(json_config, "input_opt.json", self.remote_fs)
            )
            atoms_batches.append(
                tasks.AtomsBatch(
                    self.shards[calc_idx], "opt.extxyz", self.remote_fs
                )
            )

        del self.shards

        self.individual_ressources = [config_files, atoms_batches]

    def run(self):
        self.allocate_data()
        self.submit()
        self.collect_and_harvest()
        # self.collect_results()

    # @property
    # def shared_ressources(self):
    #     return self._shared_ressources

    # @property
    # def individual_ressources(self):
    #     return self._individual_ressources

    # @property
    # def python_script_file(self):
    #     return self._python_script_file
