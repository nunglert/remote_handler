import json
import logging
import math
import os
import random
import shutil
from abc import ABC, abstractmethod
from copy import deepcopy
from itertools import groupby
from pathlib import Path
from re import T
from typing import Any, Optional, Union

import ase.db
import ase.io
import numpy as np
from ase import Atoms

from remote_handler.core import cluster_handler, file_handler, tasks

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
    # Get the order of unique indices based on their first appearance
    unique_order = reduce_to_unique_order(indices)
    # Group the indices and determine the lengths of each group
    lengths = [len(list(group)) for _, group in groupby(indices)]

    i = 0
    some_list_sets = []
    for group in lengths:
        ats = []
        for _ in range(group):
            # Append the corresponding element from some_list to the current group
            ats.append(some_list[i])
            i += 1
        # Append the current group to the list of grouped elements
        some_list_sets.append(ats)

    # Return the groups in the order of their first appearance
    return [some_list_sets[j] for j in unique_order]


def front_tag(filepath, tag):
    """
    Prepends a tag to the filename in a given filepath.
    Args:
        filepath (str or Path): The original file path.
        tag (str): The tag to prepend to the filename.
    Returns:
        Path: The modified file path with the prepended tag.
    """
    connector = "" if tag == "" else "_"
    path, filename = os.path.split(filepath)
    return Path(path) / f"{tag}{connector}{filename}"


class RemoteAtomsHandler(tasks.ArrayScriptTask):
    """
    This is a Utility subclass of tasks.ArrayScriptTask for dealing with 
    ase.Atoms trajectories. It can parse a list of `extxyz` files and 
    split them in appropriate chunks that can conveniently be processed
    by a python script.

    Args:
        traj_files: List of extxyz files containing atomic configurations.
        remote_fs: Instance of a RemoteFileSystem or LocalFileSystem
            (in case of running on a local machine with SLURM).
        shard_size: Number of configurations per task.
        savepath: Remote path where the temporary calculation directories
            are created and where the final results are written to.
        num_parallel: Number of parallel jobs to submit to the cluster, i.e.
            the size of the slurm array job.
        loop_time: Time in minutes between checking for finished jobs.
        jobname: Name of the job to be submitted to the cluster.
        every: If provided, only every `every`-th configuration is taken from
            each trajectory file.
        start: If provided, only configurations starting from this index are
            taken from each trajectory file.
        end: If provided, only configurations up to this index are taken from
            each trajectory file.
        local_savepath: Local path where the trajectory files are stored
            initially and where the final results are written to.
        logfile: Path to a logfile where the output of this script is written.
        collect_only: If True, only collect results from the remote machine,
            do not submit new jobs.
        tag: A string tag that is prepended to the filenames of the
            trajectory files when they are written to the local file system.
        n_processors: Number of processors to request per job.
        slurm_header_add: Additional lines to add to the SLURM header.
        
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
        remote_savepath: str,
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
        n_processors: int = 2,
        slurm_header_add: str = "",
    ):
        super().__init__(
            local_savepath,
            remote_fs,
            individual_ressources=[],
            shared_ressources=[],
            logfile=logfile,
            remote_savepath=remote_savepath,
            collect_only=collect_only,
            loop_time=loop_time,
            jobname=jobname,
            num_parallel=num_parallel,
            n_processors=n_processors,
            jobfile_name="run_slurm.sh",
            slurm_header_add=slurm_header_add,
        )

        # Initialize instance variables
        self.loop_time = loop_time
        self.traj_files = traj_files
        self.shard_size = shard_size
        self.map_indices = None
        self.indices = slice(start, end, every)

        self.tag = tag

        # Read and shard the trajectories
        self.read_configs()

    def combine_trajs(
        self,
        traj_files,
        indices
        
    ):
        """
        Combines configurations from all provided trajectories into a single
        list of Atoms objects.

        Args:
            traj_files: List of extxyz files containing atomic configurations.
            indices: Slice object specifying which configurations to read
                from each trajectory file.

        Returns:
            List of Atoms objects. Two attributes are stored as entries in the info
            dict:
                file_index: The index of the trajectory file from which the
                    configuration was taken.
                iteration_index: The index of the configuration in its
                    original file.
            This makes each Atoms object unambiguously attributable to its origin.
        """
        all_atoms = []
        for file_idx, file in enumerate(traj_files):
            # Read the configurations from the trajectory file
            ats = self.local_fs.read_ase(file, index=indices)
            
            for iteration_idx, at in enumerate(ats):
                at.info["file_index"] = file_idx # Save run index
                at.info["iteration_index"] = iteration_idx * indices.step # Save sample index
                
            all_atoms.extend(ats)
        return all_atoms

    def read_configs(self):
        """
        Reads all configurations from the provided trajectory files and
        shards them into batches of size `shard_size`.
        """
        # Combine all configurations from the trajectory files
        all_atoms = self.combine_trajs(self.traj_files, self.indices)
        # Determine the total number of configurations and the number of batches
        self.num_configs_total = len(all_atoms)
        self.num_array_tasks = math.ceil(
            self.num_configs_total / self.shard_size
        )
        logger.info(
            f"Collected {self.num_configs_total} configs for processing"
        )
        logger.info(f"Processing in {self.num_array_tasks} batches")
        assert (
            self.num_array_tasks <= 1000
        ), "Max size for slurm array jobs is 1000!"

        # Split the configurations into batches
        self.shards = split_list(all_atoms, self.shard_size)
        # Store the size of each batch for later verification
        self.all_shard_sizes = [len(shard) for shard in self.shards]

    def collect(
        self, only_finished: bool = True
    ) -> tuple[list[Atoms], list[int]]:
        """
        Scan temporary directories for extxyz files containing processed
        Atoms objects. The tag must be the same as the one assigned in the
        optimization driver script `optimize_traj.py`
        
        Args:
            only_finished: If True, only collect configurations that have
                the `is_finished` flag set to True in their info dict.
        Returns:
            A tuple containing:
                - A list of all collected Atoms objects.
                - A list of indices of batches that failed to process
        """
        result_filename = front_tag(f"opt.extxyz", self.tag)
        logger.info(f"Collecting results from: {str(result_filename)}")
        print(f"Collecting results from: {str(result_filename)}")
        all_atoms = []
        failed_indices = []
        for idx in range(self.num_array_tasks):
            # Determine the actual batch index
            if self.map_indices is None:
                calc_idx = idx
            else:
                calc_idx = self.map_indices[idx]
            # Determine the calculation path for this batch
            calcpath = self.calcdir / str(calc_idx)

            database_path = calcpath / result_filename
            if not self.remote_fs.exists(database_path):
                # Log a warning if the processed trajectory is missing meaning
                # the job likely failed
                logger.warning(f"Processed traj missing for shard {calc_idx}!")
                failed_indices.append(calc_idx)
            else:
                ats = self.remote_fs.read_ase(database_path, index=":")
                
                all_finished = True
                if only_finished:
                    # Check if all configurations in this batch are marked as finished
                    all_finished = bool(
                        np.all([at.info["is_finished"] for at in ats])
                    )

                # Verify that the number of collected configurations matches
                size_preserved = (len(ats) == self.all_shard_sizes[calc_idx])
                if not size_preserved or not all_finished:
                    logger.warning(
                        f"Shard {calc_idx} failed with {size_preserved=} "
                        f"and {all_finished=}"
                    )
                    failed_indices.append(calc_idx)

                # Collect the configurations from this batch
                all_atoms.extend(ats)

        return all_atoms, failed_indices

    def regroup(self, all_atoms) -> list[list[Atoms]]:
        """
        Regroup a flat list of Atoms objects into a list of lists, where each
        sublist corresponds to the configurations from a single original
        trajectory file.
        
        Args:
            all_atoms: A flat list of Atoms objects, each with an info dict
                containing a 'file_index' key that indicates the original
                trajectory file index.
        Returns:
            A list of lists, where each sublist contains Atoms objects from
            the same original trajectory file.
        """
        
        indices = [at.info["file_index"] for at in all_atoms]
        return regroup(all_atoms, indices)
    
    def redistribute(self, all_atoms_sets):
        """
        Redistribute the grouped Atoms objects back into their original
        trajectory files, writing them to the local file system with the
        specified tag prepended to the filenames."""

        for ats in all_atoms_sets:
            # Get the original filename using the file_index from the info dict
            fname = self.traj_files[int(ats[-1].info["file_index"])]
            # Write the configurations to the local file system with the tag
            self.local_fs.write_ase(front_tag(fname, self.tag), ats)

    def check_and_harvest(self, all_atoms, failed_indices):
        """
        Check for failed batches and redistribute collected configurations 
        into their original trajectory files.
        """
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
        logger.info("Successfully harvested all processed configs!")

    def collect_and_harvest(self):
        """
        Collect processed configurations from the remote machine and
        redistribute them into their original trajectory files.
        Raises an exception if any batch failed to process or if the
        number of collected configurations is less than the total expected.
        """
        all_atoms, failed_indices = self.collect()
        self.check_and_harvest(all_atoms, failed_indices)

    def recompute_failed(self):
        """
        Attempt to recompute any batches that failed during the initial
        processing. This method is only applicable when `collect_only` is
        set to True. It collects the failed batches, resubmits them for
        processing, and then collects the results again."""
        assert (
            self.collect_only
        ), "Recompute only makes sense in collect_only mode"

        logger.info("... Trying to recompute failed shards")
        all_atoms, failed_indices = self.collect()
        logger.info(f"Found {len(failed_indices)} broken shards")
        logger.info("Broken shard indices:")
        logger.info(failed_indices)

        if failed_indices:
            # Number of failed batches
            self.size = len(failed_indices)
            # Update the map_indices to only include the failed batches
            self.slurm_handler.map_indices = failed_indices
            # Avoid redistributing already collected configurations
            # and the creations of directories
            self.allocate_data(slurm_only=True) 
            self.submit()

            # Collect the results again after resubmission
            ats_rerun, failed_indices = self.collect()
            all_atoms.extend(ats_rerun)

        # Finally, check and harvest all collected configurations
        self.check_and_harvest(all_atoms, failed_indices)
