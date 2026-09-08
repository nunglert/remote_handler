"""Pure-domain sharding/collection logic for ASE Atoms trajectories.

Extracted from ``atoms_handler`` so the trajectory math (read, shard, collect,
regroup, redistribute) lives free of any SLURM/cluster dependency.
``RemoteAtomsHandler`` composes :class:`AtomsShardManager` from here.
"""

import logging
import math
import os
from itertools import groupby
from pathlib import Path
from typing import Any, Optional

import numpy as np
from ase import Atoms

from remote_handler.core import file_handler

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


class AtomsShardManager:
    """
    Manages sharding and collection of ASE Atoms trajectories.

    This class contains pure domain logic for handling atomic trajectory files:
    reading, sharding, collection, regrouping, and redistribution. It has no
    dependencies on SLURM or cluster infrastructure.

    Args:
        traj_files: List of extxyz files containing atomic configurations.
        shard_size: Number of configurations per shard. ``None`` or a
            negative value (e.g. ``-1``) means "no sharding": the entire set
            of configurations is processed as a single shard.
        indices: Slice object specifying which configurations to read (start:end:step).
    """

    def __init__(
        self,
        traj_files: list[str],
        shard_size: Optional[int],
        indices: slice,
    ):
        self.traj_files = traj_files
        self.shard_size = shard_size
        self.indices = indices

        # State populated by read_and_shard
        self.shards = None
        self.num_configs_total = None
        self.num_array_tasks = None
        self.all_shard_sizes = None

    def combine_trajs(
        self,
        traj_files: list[str],
        indices: slice,
        local_fs: file_handler.FileSystemHandlerBase,
    ) -> list[Atoms]:
        """
        Combines configurations from all provided trajectories into a single
        list of Atoms objects.

        Args:
            traj_files: List of extxyz files containing atomic configurations.
            indices: Slice object specifying which configurations to read
                from each trajectory file.
            local_fs: Filesystem handler for reading trajectory files.

        Returns:
            List of Atoms objects. Two attributes are stored as entries in the
            ``info`` dict, making each Atoms object unambiguously attributable
            to its origin:

            - ``file_index``: The index of the trajectory file from which the
              configuration was taken.
            - ``iteration_index``: The index of the configuration in its
              original file.
        """
        # A bare slice (e.g. slice(None, None, None)) leaves start/step as
        # None; normalize so iteration_index maps back to the original frame.
        step = indices.step if indices.step is not None else 1
        start = indices.start if indices.start is not None else 0

        all_atoms = []
        for file_idx, file in enumerate(traj_files):
            # Read the configurations from the trajectory file
            ats = local_fs.read_ase(file, index=indices)

            for iteration_idx, at in enumerate(ats):
                at.info["file_index"] = file_idx  # Save run index
                at.info["iteration_index"] = (
                    start + iteration_idx * step
                )  # Save sample index

            all_atoms.extend(ats)
        return all_atoms

    def read_and_shard(
        self, local_fs: file_handler.FileSystemHandlerBase
    ) -> list[list[Atoms]]:
        """
        Reads all configurations from the provided trajectory files and
        shards them into batches of size `shard_size`. A `shard_size` of
        ``None`` or a negative value processes the whole set as one shard.

        Args:
            local_fs: Filesystem handler for reading trajectory files.

        Returns:
            List of sharded Atoms lists.
        """
        # Combine all configurations from the trajectory files
        all_atoms = self.combine_trajs(self.traj_files, self.indices, local_fs)

        # Determine the total number of configurations and the number of batches
        self.num_configs_total = len(all_atoms)

        # A shard_size of None or a negative value (e.g. -1) means "no
        # sharding": process the entire set of configs as a single shard.
        # Resolve it here, where the total count is known, and store it back
        # so downstream consumers (all_shard_sizes, the handler's copies) see
        # the effective size.
        if self.shard_size is None or self.shard_size < 0:
            self.shard_size = max(self.num_configs_total, 1)

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

        return self.shards

    def collect(
        self,
        calcdir: Path,
        remote_fs: file_handler.FileSystemHandlerBase,
        num_array_tasks: int,
        map_indices: Optional[list[int]],
        tag: str,
        only_finished: bool = True,
    ) -> tuple[list[Atoms], list[int]]:
        """
        Scan calculation directories for extxyz files containing processed
        Atoms objects.

        Args:
            calcdir: Base calculation directory containing subdirectories 0, 1, 2, ...
            remote_fs: Filesystem handler for accessing remote files.
            num_array_tasks: Number of array tasks to collect from.
            map_indices: Optional mapping of task indices (for recompute scenarios).
            tag: Tag used to identify result files.
            only_finished: If True, only collect configurations that have
                the `is_finished` flag set to True in their info dict.

        Returns:
            A tuple containing:
                - A list of all collected Atoms objects.
                - A list of indices of batches that failed to process.
        """
        result_filename = front_tag(f"opt.extxyz", tag)
        logger.info(f"Collecting results from: {str(result_filename)}")
        print(f"Collecting results from: {str(result_filename)}")

        all_atoms = []
        failed_indices = []

        for idx in range(num_array_tasks):
            # Determine the actual batch index
            if map_indices is None:
                calc_idx = idx
            else:
                calc_idx = map_indices[idx]
            # Determine the calculation path for this batch
            calcpath = calcdir / str(calc_idx)

            database_path = calcpath / result_filename
            if not remote_fs.exists(database_path):
                # Log a warning if the processed trajectory is missing meaning
                # the job likely failed
                logger.warning(f"Processed traj missing for shard {calc_idx}!")
                failed_indices.append(calc_idx)
            else:
                ats = remote_fs.read_ase(database_path, index=":")

                all_finished = True
                if only_finished:
                    # Check if all configurations in this batch are marked as finished
                    all_finished = bool(
                        np.all([at.info["is_finished"] for at in ats])
                    )

                # Verify that the number of collected configurations matches
                size_preserved = len(ats) == self.all_shard_sizes[calc_idx]
                if not size_preserved or not all_finished:
                    logger.warning(
                        f"Shard {calc_idx} failed with {size_preserved=} "
                        f"and {all_finished=}"
                    )
                    failed_indices.append(calc_idx)

                # Collect the configurations from this batch
                all_atoms.extend(ats)

        return all_atoms, failed_indices

    def regroup(self, all_atoms: list[Atoms]) -> list[list[Atoms]]:
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

    def redistribute(
        self,
        all_atoms_sets: list[list[Atoms]],
        local_fs: file_handler.FileSystemHandlerBase,
        tag: str,
    ):
        """
        Redistribute the grouped Atoms objects back into their original
        trajectory files, writing them to the local file system with the
        specified tag prepended to the filenames.

        Args:
            all_atoms_sets: List of lists, where each sublist contains Atoms
                from the same original trajectory file.
            local_fs: Filesystem handler for writing trajectory files.
            tag: Tag to prepend to output filenames.
        """
        for ats in all_atoms_sets:
            # Get the original filename using the file_index from the info dict
            fname = self.traj_files[int(ats[-1].info["file_index"])]
            # Write the configurations to the local file system with the tag
            local_fs.write_ase(front_tag(fname, tag), ats)

    def check_and_harvest(
        self,
        all_atoms: list[Atoms],
        failed_indices: list[int],
        local_fs: file_handler.FileSystemHandlerBase,
        tag: str,
        savepath: Path,
    ):
        """
        Check for failed batches and redistribute collected configurations
        into their original trajectory files.

        Args:
            all_atoms: All collected Atoms objects.
            failed_indices: Indices of batches that failed.
            local_fs: Filesystem handler for writing results.
            tag: Tag to prepend to output filenames.
            savepath: Path to check for SLURM output (for error messages).

        Raises:
            Exception: If any batches failed or if collected count is insufficient.
        """
        if failed_indices:
            raise Exception(
                f"Something went wrong during processing. Check slurm "
                f"output at {savepath / 'slurm_output'}\n"
                f"{failed_indices=}"
            )

        if len(all_atoms) < self.num_configs_total:
            raise Exception(
                "Collected number of configurations < num_total_configs!"
            )

        all_atoms_sets = self.regroup(all_atoms)
        self.redistribute(all_atoms_sets, local_fs, tag)
        logger.info("Successfully harvested all processed configs!")
