import os
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from re import T
from typing import Any, Optional, Union

from remote_handler import cluster_handler, file_handler, tasks
from remote_handler.atoms_handler import RemoteAtomsHandler


class RemoteOptimizationHandler(RemoteAtomsHandler):
    """

    Args:
        savepath: Path on the remote file system, where intermediate results
            are stored temporarily.

        optimization_backend: NNFF or LJ
        steinhardt_backend: pyscal or quip

    Example:
        Given a set of 3 trajectory files, containing 50 configs per file,
        with a shard_size of 20, this will shard the configs in 8 batches
        (arising from ceil(3 * 50 / 20)). For each batch a directory will be
        created in `savepath/calcdirs`. After the computations were run,
        the `savepath` should look like this:

        savepath
        |___calcdirs
            |___0
            |   |___0.extxyz
            |   |___input_opt.json
            |   |___opt_0.extxyz
            |
            |___1
                |___ ...

        The opt_X.extxyz files are collected and redistributed into the shape
        of the original trajectory files.


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
        python_script_path: str = "/home/nunglert/code/jaxnest_dev/scripts/utility/optimize_traj.py",
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
        )

        self.logger = tasks.SimpleLogger(logfile)
        self.do_optimization = do_optimization
        self.pickle_file = pickle_file

        # if do_optimization:
        #     assert steps is not None
        #     tag = "_"
        #     tag += "s" if start is None else f"s{start}"
        #     tag += "e" if end is None else f"e{end}"
        #     tag += "f" if every is None else f"f{every}"

        #     self.slurm_handler.jobname += tag

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
