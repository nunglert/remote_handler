import json
import logging
import math
import os
import random
import shutil
import string
import tempfile
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from itertools import groupby
from pathlib import Path
from re import T
from typing import Any, Optional, Union

import ase.db
import ase.io
import numpy as np
import paramiko
from ase import Atoms
from scipy import cluster

from remote_handler import cluster_handler, file_handler, tasks

logger = logging.getLogger(__name__)


def split_list(lst: list[Any], shard_size) -> list[list[Any]]:
    """
    Splits a list into smaller sublists of a specified size.

    Args:
        lst (list[Any]): The list to be split.
        shard_size (int): The size of each shard.

    Returns:
        list[list[Any]]: A list of sublists, where each sublist has a length
        of `shard_size`, except possibly the last one.
    """
    return [lst[i : i + shard_size] for i in range(0, len(lst), shard_size)]


def reduce_to_unique_order(arr):
    """
    Reduce an array to its unique elements while preserving the order
    of first appearance.

    Parameters:
        arr (iterable): The input array or list to process.

    Returns:
        list: A list of unique elements in the order of their first appearance.
    """
    seen = []
    for item in arr:
        if item not in seen:
            seen.append(item)
    return seen


def regroup(some_list, indices):
    """Group elements of a list based on an array of indices, preserving the
    order of first appearance of unique indices.

    Parameters:
        some_list (list): The input list of elements to group.
        indices (array-like): An array of indices specifying the grouping.
                              Consecutive occurrences of the same index
                              form a group.

    Returns:
        list of lists: A list of groups, where each group contains elements
                       from `some_list` corresponding to a unique index in
                       `indices`, in the order of their first appearance.

    """
    assert len(some_list) == len(
        indices
    ), "some_list and indices must be of same length"
    unique_order = reduce_to_unique_order(indices)
    lengths = [len(list(group)) for _, group in groupby(indices)]

    i = 0
    some_list_sets = []
    for group in lengths:
        ats = []
        for _ in range(group):
            ats.append(some_list[i])
            i += 1
        some_list_sets.append(ats)

    return [some_list_sets[j] for j in unique_order]


def front_tag(filepath, tag):
    connector = "" if tag == "" else "_"
    path, filename = os.path.split(filepath)
    return Path(path) / f"{tag}{connector}{filename}"


class RemoteAtomsHandler(tasks.ArrayScriptTask):
    """This is a Utility class for dealing with ase.Atoms trajectories. It can
    parse a list of `extxyz` files and split them in appropriate chunks that
    can conveniently be processed by a python script. For an example usage
    refer to the `RemoteOptimizationHandler` below.

    Args:
        savepath: Path on the remote file system, where intermediate results
            are stored temporarily.

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
        traj_files: list[str],
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
        tag: str = "opt_",
    ):
        super().__init__(
            local_savepath,
            remote_fs,
            individual_ressources=[],
            shared_ressources=[],
            logfile=logfile,
            remote_savepath=savepath,
            collect_only=collect_only,
            loop_time=loop_time,
            jobname=jobname,
            n_processors=2,
            jobfile_name="run_slurm.sh",
        )

        self.logger = tasks.SimpleLogger(logfile)
        self.num_parallel = num_parallel
        self.loop_time = loop_time
        self.traj_files = traj_files
        self.shard_size = shard_size
        self.map_indices = None
        self.indices = slice(start, end, every)

        self.tag = tag

        self.read_configs()

    def combine_trajs(
        self,
        traj_files,
        indices
        # self, traj_files, start=None, end=None, every=10
    ):
        """Combines configurations from all provided trajectories into a single
        list of Atoms objects.

        Args:
            traj_files: List of extxyz files containing atomic configurations.

        Returns:
            List of Atoms objects. Three things are stored as entries in the info
            dict:
                file_index: The index of the trajectory file from which the
                    configuration was taken.
                iteration_index: The index of the configuration in its
                    original file.
            This makes each Atoms object unambiguously attributable to its origin.
        """
        all_atoms = []
        for file_idx, file in enumerate(traj_files):
            ats = self.local_fs.read_ase(file, index=indices)
            for iteration_idx, at in enumerate(ats):
                at.info["file_index"] = file_idx
                at.info["iteration_index"] = iteration_idx * indices.step
            all_atoms.extend(ats)
        return all_atoms

    def read_configs(self):
        all_atoms = self.combine_trajs(self.traj_files, self.indices)
        self.num_configs_total = len(all_atoms)
        self.num_array_tasks = math.ceil(
            self.num_configs_total / self.shard_size
        )
        self.logger.print(
            f"Collected {self.num_configs_total} configs for processing"
        )
        self.logger.print(f"Processing in {self.num_array_tasks} batches")
        assert (
            self.num_array_tasks <= 1000
        ), "Max size for slurm array jobs is 1000!"
        self.shards = split_list(all_atoms, self.shard_size)
        self.all_shard_sizes = [len(shard) for shard in self.shards]

    def collect(
        self, only_finished: bool = True
    ) -> tuple[list[Atoms], list[int]]:
        """Scan temporary directories for extxyz files containing processed
        Atoms objects. The tag must be the same as the one assigned in the
        optimization driver script `optimize_traj.py`
        """
        all_atoms = []
        failed_indices = []
        for idx in range(self.num_array_tasks):
            if self.map_indices is None:
                calc_idx = idx
            else:
                calc_idx = self.map_indices[idx]

            calcpath = self.calcdir / str(calc_idx)
            database_path = front_tag(calcpath / "opt.extxyz", self.tag)
            if not self.remote_fs.exists(database_path):
                logger.warning(f"Processed traj missing for shard {calc_idx}!")
                failed_indices.append(calc_idx)
            else:
                ats = self.remote_fs.read_ase(database_path, index=":")
                all_finished = True
                if only_finished:
                    all_finished = bool(
                        np.all([at.info["is_finished"] for at in ats])
                    )
                size_preserved = len(ats) == self.all_shard_sizes[calc_idx]
                if not size_preserved or not all_finished:
                    logger.warning(
                        f"Shard {calc_idx} failed with {size_preserved=} "
                        f"and {all_finished=}"
                    )
                    failed_indices.append(calc_idx)
                all_atoms.extend(ats)

        return all_atoms, failed_indices

    def regroup(self, all_atoms) -> list[list[Atoms]]:
        """"""
        indices = [at.info["file_index"] for at in all_atoms]
        return regroup(all_atoms, indices)

    def redistribute(self, all_atoms_sets):
        for ats in all_atoms_sets:
            fname = self.traj_files[int(ats[-1].info["file_index"])]
            self.local_fs.write_ase(front_tag(fname, self.tag), ats)

    def collect_and_harvest(self):
        all_atoms, failed_indices = self.collect()
        self.check_and_harvest(all_atoms, failed_indices)

    def check_and_harvest(self, all_atoms, failed_indices):
        if failed_indices:
            raise Exception(
                f"Something went wrong during processing. Check slurm "
                f"output at {Path(self.savepath)/ 'slurm_output'}\n"
                f"{failed_indices=}"
            )

        if len(all_atoms) < self.num_configs_total:
            raise Exception(
                "Collected number of configurations < num_total_configs!"
            )

        all_atoms_sets = self.regroup(all_atoms)
        self.redistribute(all_atoms_sets)
        self.logger.print("Successfully harvested all processed configs!")

    def recompute_failed(self):
        """"""
        assert (
            self.collect_only
        ), "Recompute only makes sense in collect_only mode"
        self.logger.print("... Trying to recompute failed shards")
        all_atoms, failed_indices = self.collect()
        self.logger.print(f"found {len(failed_indices)} broken shards")
        self.logger.print("Broken shard indices:")
        self.logger.print(failed_indices)

        if failed_indices:
            self.size = len(failed_indices)
            self.slurm_handler.map_indices = failed_indices
            self.allocate_data(slurm_only=True)
            self.submit()

            ats_rerun, failed_indices = self.collect()
            all_atoms.extend(ats_rerun)

        self.check_and_harvest(all_atoms, failed_indices)

    # @property
    # @abstractmethod
    # def shared_ressources(self):
    #     pass

    # @property
    # @abstractmethod
    # def individual_ressources(self):
    #     pass

    # @property
    # @abstractmethod
    # def python_script_file(self):
    #     pass


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
        tag: str = "opt",
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
            json_config["DATABASE"] = f"opt.extxyz"
            pickle_file = f"../../{os.path.split(self.pickle_file)[1]}"
            json_config["calculator_args"]["pickle_file"] = pickle_file
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
        # self.collect_results()
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
