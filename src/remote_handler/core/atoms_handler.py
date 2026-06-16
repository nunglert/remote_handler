"""Remote SLURM array driver for ASE Atoms trajectory processing.

``RemoteAtomsHandler`` couples :class:`~remote_handler.core.tasks.ArrayScriptTask`
(remote execution) with :class:`~remote_handler.core.atoms_shard.AtomsShardManager`
(trajectory domain logic) via composition.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from ase import Atoms

from remote_handler.core import file_handler, tasks
from remote_handler.core.atoms_shard import AtomsShardManager
from remote_handler.core.cluster_handler import ClusterConfig

logger = logging.getLogger(__name__)


class RemoteAtomsHandler(tasks.ArrayScriptTask):
    """
    Coordinates SLURM array job execution for processing ASE Atoms trajectories.

    This class combines ArrayScriptTask (for SLURM infrastructure) with
    AtomsShardManager (for trajectory handling) using composition. It manages
    the full workflow: reading trajectory files, sharding them into batches,
    submitting array jobs, collecting results, and redistributing them back
    into the original file structure.

    Architecture:
        - Inherits from ArrayScriptTask for SLURM job submission infrastructure
        - Composes AtomsShardManager for domain-specific Atoms handling logic
        - Delegates trajectory operations to shard_manager
        - Orchestrates the overall workflow

    Args:
        traj_files: List of extxyz files containing atomic configurations.
        cluster_config: ClusterConfig instance containing cluster-specific settings.
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
        cluster_config: ClusterConfig,
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
            cluster_config,
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
        self.map_indices = None
        self.tag = tag

        # Create the shard manager for domain logic
        self.shard_manager = AtomsShardManager(
            traj_files=traj_files,
            shard_size=shard_size,
            indices=slice(start, end, every),
        )

        # Read and shard the trajectories using the shard manager
        self.shards = self.shard_manager.read_and_shard(self.local_fs)

        # Expose properties for backwards compatibility with subclasses
        self.traj_files = self.shard_manager.traj_files
        self.shard_size = self.shard_manager.shard_size
        self.indices = self.shard_manager.indices
        self.num_configs_total = self.shard_manager.num_configs_total
        self.num_array_tasks = self.shard_manager.num_array_tasks
        self.all_shard_sizes = self.shard_manager.all_shard_sizes

    def combine_trajs(self, traj_files, indices):
        """
        Combines configurations from all provided trajectories into a single
        list of Atoms objects.

        Deprecated: This method is kept for backwards compatibility.
        Use shard_manager.combine_trajs() directly instead.
        """
        return self.shard_manager.combine_trajs(
            traj_files, indices, self.local_fs
        )

    def read_configs(self):
        """
        Reads all configurations from the provided trajectory files and
        shards them into batches of size `shard_size`.

        Deprecated: This method is kept for backwards compatibility.
        Sharding now happens automatically in __init__ via shard_manager.
        """
        self.shards = self.shard_manager.read_and_shard(self.local_fs)
        self.num_configs_total = self.shard_manager.num_configs_total
        self.num_array_tasks = self.shard_manager.num_array_tasks
        self.all_shard_sizes = self.shard_manager.all_shard_sizes

    def collect(
        self, only_finished: bool = True
    ) -> tuple[list[Atoms], list[int]]:
        """
        Scan temporary directories for extxyz files containing processed
        Atoms objects.

        Args:
            only_finished: If True, only collect configurations that have
                the `is_finished` flag set to True in their info dict.

        Returns:
            A tuple containing:
                - A list of all collected Atoms objects.
                - A list of indices of batches that failed to process.
        """
        return self.shard_manager.collect(
            calcdir=self.calcdir,
            remote_fs=self.remote_fs,
            num_array_tasks=self.num_array_tasks,
            map_indices=self.map_indices,
            tag=self.tag,
            only_finished=only_finished,
        )

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
        return self.shard_manager.regroup(all_atoms)

    def redistribute(self, all_atoms_sets):
        """
        Redistribute the grouped Atoms objects back into their original
        trajectory files, writing them to the local file system with the
        specified tag prepended to the filenames.

        Args:
            all_atoms_sets: List of lists, where each sublist contains Atoms
                from the same original trajectory file.
        """
        self.shard_manager.redistribute(
            all_atoms_sets, self.local_fs, self.tag
        )

    def check_and_harvest(self, all_atoms, failed_indices):
        """
        Check for failed batches and redistribute collected configurations
        into their original trajectory files.

        Args:
            all_atoms: All collected Atoms objects.
            failed_indices: Indices of batches that failed.

        Raises:
            Exception: If any batches failed or if collected count is insufficient.
        """
        self.shard_manager.check_and_harvest(
            all_atoms=all_atoms,
            failed_indices=failed_indices,
            local_fs=self.local_fs,
            tag=self.tag,
            savepath=Path(self.savepath),
        )

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
