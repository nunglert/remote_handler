import logging
import os
import random
import string
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from remote_handler.core import executors, file_handler, file_utils, remote_env
from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import (
    FileSystemHandlerBase,
    LocalFileSystem,
    RemoteFileSystem,
)
from remote_handler.core.file_utils import tag_path
from remote_handler.core.utils import SimpleLogger
from remote_handler.exceptions import FileSystemError, TaskError

logger = logging.getLogger(__name__)

# Array tasks execute in "<rootpath>/calcdirs/<run_id>" (see
# SlurmArrayHandler.calcdir and templates/body_array.sh.j2) -- two levels below
# the rootpath where shared resources are allocated. Reference shared resources
# from a task's input JSON with this prefix; it is derived from the run-subdir
# nesting so it stays correct if that layout ever changes.
_ARRAY_TASK_RUN_SUBDIR = os.path.join(remote_env.CALCDIRS_DIRNAME, "0")
SHARED_RESOURCE_PREFIX = os.path.relpath(os.curdir, _ARRAY_TASK_RUN_SUBDIR)


class RemoteTask(ABC):
    """This serves as an interface for tasks that should be performed on a
    remote machine. Tasks are always meant to be executed via `subprocess`.
    A `RemoteTask` can be a slurm job, but does not necessarily have to be one.
    Effectively, `RemoteTask` just provides the interface handling the
    questions:
    - What needs to be transferred to the remote machine?
    - How can I start your job?
    - Once it finished, how should I collect the results?
    and optionally:
    - how can I properly check (on top of slurm) whether your job finished?

    Methods that need to be implemented are:

    - ``allocate_data``: Logic for the allocation of files on the remote file
      system required for the task.
    - ``submit``: Logic for the submission of the job. This can be a simple
      ``subprocess.run(...)`` call, but can also include more elaborate slurm
      handling via ``remote_handler.core.cluster_handler``.
    - ``collect_results``: Logic for the collection of results. This can be a
      simple copy of files from the remote machine to the host, but can also
      include parsing of files and returning only the relevant information.
    - ``finish_check`` (optional): Logic for an additional check whether the
      job finished properly. This is useful, as slurm does not know anything
      about your job itself. Hence, this can be a good complementary way of
      checking whether a job has really finished.

    Args:
        logfile: A log file on the host machine where output is printed
        savepath: The path on the remote system where computations should be
            performed. Will try to use temporary dir if None.
        remote_fs: A FileSystemHandlerBase derived object.
        jobname: A label for the task, used in jobname and savepath tagging.
        collect_only: By default, RemoteTask creates a new tagged savepath if
            the provided savepath already exists. If you want RemoteTask act on
            a `savepath` that exists already, set this to True. This can be
            helpful if a task has crashed and you want to continue it or
            collect its results.
    """

    def __init__(
        self,
        logfile: str,
        savepath: os.PathLike,
        remote_fs: file_handler.FileSystemHandlerBase,
        jobname: str = "job",
        collect_only: bool = False,
    ):
        assert not "/" in str(logfile), "logfile should only be filename!"

        # Instatiate LocalFileSystem to handle local files without slurm_config_file
        self.local_fs = LocalFileSystem()
        # Save the provided remote_fs
        self.remote_fs = remote_fs

        self.collect_only = collect_only

        if savepath is not None:
            # Erase trailing slash if present
            savepath = Path(savepath)

            parent, _ = os.path.split(savepath)

            # Check if the parent directory exists on the remote system
            if not self.remote_fs.exists(parent):
                raise FileSystemError(
                    f"Parent directory must exist on remote system: {parent}\n"
                    f"Savepath: {savepath}"
                )

            if not self.remote_fs.exists(savepath):
                self.remote_fs.makedirs(savepath, exist_ok=True)

            # Append the specific step to the savepath
            self.savepath = Path(savepath) / jobname

            # Check if the directory already exists on the remote system
            if self.remote_fs.exists(self.savepath):
                if not collect_only:
                    logger.warning(
                        "Savepath already exists. A new directory with "
                        "a tag will be created to prevent overwriting!"
                    )

                    random_string = "".join(
                        random.choices(
                            string.ascii_letters + string.digits, k=10
                        )
                    )

                    working_dir, filename = os.path.split(self.savepath)
                    savetag = tag_path(
                        working_dir, filename, is_directory=True
                    )
                    self.savepath = Path(str(savetag) + random_string)

                    logger.info(f"Results saved to: {self.savepath}")
                    self.remote_fs.makedirs(self.savepath, exist_ok=True)

            else:
                if collect_only:
                    raise FileSystemError(
                        f"Cannot collect results: savepath does not exist on remote machine\n"
                        f"Path: {self.savepath}"
                    )
                logger.info("Creating the savepath at requested location")
                # Create the directory on the remote system
                self.remote_fs.makedirs(self.savepath, exist_ok=True)
        else:
            logger.info(
                "No savepath provided, a temporary directory will be created"
            )

            # Create a temporary directory on the remote system
            self.savepath = self.remote_fs.create_tempdir() / jobname
            # Print the path of the temporary directory
            logger.info(f"Temporary directory created at: {self.savepath}")

            if self.remote_fs.exists(self.savepath):
                if not collect_only:
                    logger.warning(
                        "Savepath already exists. A new directory with "
                        "a tag will be created to prevent overwriting!"
                    )

                    random_string = "".join(
                        random.choices(
                            string.ascii_letters + string.digits, k=10
                        )
                    )

                    working_dir, filename = os.path.split(savepath)
                    savetag = tag_path(
                        working_dir, filename, is_directory=True
                    )
                    self.savepath = Path(str(savetag) + random_string)

                    logger.info(f"Results saved to: {self.savepath}")
                    self.remote_fs.makedirs(self.savepath, exist_ok=True)

        # Instantiate the logger on the remote machine, at 'savepath' with 'logfile' name #TODO NU: what was this for?
        self.logger = SimpleLogger(Path(self.savepath) / logfile)

        super().__init__()

    @abstractmethod
    def allocate_data(self):
        """Function handling the allocation of files on the remote file
        system required for the task.

        Example:
            Say you want to do e.g. DFT calculations on another machine. Then
            this function would handle the logic of setting up the directories
            on the remote machine where the computations are supposed to be
            performed. Furthermore, it would transfer required input/submission
            files or create them in-place.
        """
        pass

    @abstractmethod
    def submit(self):
        """
        Function submitting the slurm job(s). This should depend on how
        the slurm logic was handled in `setup_slurm`.
        """
        pass

    @abstractmethod
    def collect_results(self):
        """Function controlling how the computed results are collected. Should
        files be copied/moved from the remote machine to the host or should the
        results only be parsed and returned?

        Example:
            Your task is a DFT run. It was succesfully run, but you now have a
            directory with all VASP related files (POSCAR, KPOINTS, POTCAR, ..)
            in the computatin path on the remote machine. This`function
            (`collect_results`) should provide the logic for the retrieval of
            the data that is required. Do you want the full directory? Is the
            vasprun.xml or OUTCAR enough?
        """
        pass

    def finish_check(self):
        """Optionally, the logic for an additional check can be implemented
        for a given task. Using only slurm to detect whether a job has finished
        can be error prone. Slurm does not know anything about our job itself,
        so this is a good complementary way of checking whether a job has
        really finished.

        Example:
            Your task consists of running a python script. You could write
            a small (empty) file `is_finished.info` at the end of your script.
            This function (`finish_check`) could then look for this file when
            called to confirm that the job is really finished.
            If your task is a VASP run, one could check whether the OUTCAR ends
            with the characteristic lines, etc.
        """
        pass


class SingleScriptTask(RemoteTask):
    """
    This class handles the submitting of a single script job, able to run a
    python script with an input file. The `savepath` defines the computation
    directory on the remote machine and all output files will be written there.

    This simple implementation just copies the whole computation directory
    to the `local_savepath` if the results are collected by `collect_results`.

    Args:
        script_file: Path to a python script that performs the computation.
        input_file: Path to an input file that is read by the script.
        local_savepath: Path where finished results will be copied from remote
            to local machine
        cluster_config: ClusterConfig instance containing cluster-specific settings.
        remote_fs: FileSystemHandlerBase derived object for file operations.
        logfile: Path to the logfile of the task.
        remote_savepath: Path on remote machine, where computation will be
            performed. If None, falls back to cluster_config.savepath.
        collect_only: If True, results are collected from `remote_savepath`
            if it already exists. If False a new savepath will be created with
            a tag to prevent overwriting.
        loop_time: The time interval which is used to check whether the job is
            completed.
        jobname: Name of the job used in slurm.
        n_processors: Number of processors that are requested per job. For CPU
            this refers to n_tasks, where as for GPU on datalab this refers to
            the number of GPU devices requested on a single node.
        jobfile_name: Name of the slurm job script that is created to submit the job.
        job_type_template: The template that is used to create the slurm job
            script. This needs to be present in one of the `TEMPLATE_DIRS` of
            `remote_handler.core.slurm_utils`.
    """

    def __init__(
        self,
        script_file: os.PathLike,
        input_file: os.PathLike,
        local_savepath: os.PathLike,
        cluster_config: ClusterConfig,
        remote_fs: FileSystemHandlerBase,
        logfile: os.PathLike = "task.log",
        additional_files: Optional[list[os.PathLike]] = [],
        remote_savepath: Optional[os.PathLike] = None,
        collect_only: bool = False,
        loop_time: float = 30.0,
        jobname: str = "job",
        n_processors: int = 2,
        jobfile_name: str = "run_slurm.sh",
        job_type_template: os.PathLike = "body_single.sh.j2",
    ):
        # If no remote_savepath is provided, fall back to the savepath from
        # the cluster_config
        if remote_savepath is None:
            remote_savepath = cluster_config.savepath

        # Use the base class to handle savepath creation and logging
        super().__init__(
            logfile=logfile,
            savepath=remote_savepath,
            collect_only=collect_only,
            remote_fs=remote_fs,
            jobname=jobname,
        )

        # The directory where the computation will be performed
        self.calcdir = Path(self.savepath)

        # Propagate this cluster's opt-in connection policy to the SFTP layer
        # so file transfers pause-and-retry on auth loss too (no-op if off).
        if hasattr(remote_fs, "connection_policy"):
            remote_fs.connection_policy = cluster_config.connection_policy

        self.script_file = script_file
        self.input_file = input_file
        self.additional_files = additional_files
        if not isinstance(self.additional_files, list):
            logger.warning(
                f"additional_files should be a list of paths, but got {type(self.additional_files)}. Converting to list."
            )
            self.additional_files = [self.additional_files]
        self.loop_time = loop_time
        self.local_savepath = local_savepath

        self.executor = executors.make_single_executor(
            cluster_config,
            jobname=jobname,
            savepath=self.calcdir,
            n_processors=n_processors,
            slurm_header_add="",
            remote_fs=remote_fs,
            jobfile_name=jobfile_name,
            loop_time=loop_time,
            job_type_template=job_type_template,
        )
        # Back-compat alias: external callers (e.g. RemoteAtomsHandler) still
        # reference ``.slurm_handler``. Same object as ``.executor``.
        self.slurm_handler = self.executor

        # Create composable components for file allocation and result collection
        self.file_allocator = FileAllocator(
            remote_fs=remote_fs, rootpath=self.calcdir
        )
        self.result_collector = ResultCollector(
            remote_fs=remote_fs, local_fs=self.local_fs
        )

        logger.info(f"SingleScriptTask")
        logger.info(f"calcdir: {str(self.calcdir)}")

    def copy_file(self, local_path):
        """
        Copies a local file to the remote calculation directory.
        Args:
            local_path: Path to the local file.
        Returns:
            The path to the file on the remote machine.
        """
        return self.file_allocator.copy_file(local_path)

    def copy_path(self, local_path):
        """Copy a local file *or* directory tree to the remote calcdir.

        Use this instead of :meth:`copy_file` when the artifact may be a
        directory (e.g. a mace-jax model bundle); files behave identically.

        Args:
            local_path: Path to the local file or directory.
        Returns:
            The path to the file/directory on the remote machine.
        """
        return self.file_allocator.copy_path(local_path)

    def allocate_data(self):
        """Allocate the script and input file on the remote machine and
        prepare the slurm job script.
        """
        if self.script_file is not None:
            self.script_file_remote = self.copy_file(self.script_file)
        else:
            self.script_file_remote = None

        self.input_file_remote = self.copy_file(self.input_file)

        if self.additional_files:
            self.remote_fs.makedirs(
                self.calcdir / "additional_files", exist_ok=True
            )
            for additional_file in self.additional_files:
                self.remote_fs.put(
                    additional_file,
                    self.calcdir
                    / "additional_files"
                    / os.path.basename(additional_file),
                )

        # After copying the files, write the job script on the remote machine
        self.executor.write_jobfile(
            self.input_file_remote, self.script_file_remote
        )

    def submit(self, only_start=False):
        """Submit the job and wait until it is finished."""
        self.executor.submit()
        if only_start:
            return
        self.executor.wait_until_finished()

    def collect_results(self):
        """Copy the entire calculation directory from the remote machine to
        the local machine using ResultCollector."""
        self.result_collector.collect_directory(
            self.calcdir, self.local_savepath, move=True
        )

    def finish_check(self):
        """Check for the presence of the 'is_finished.info' file in the
        calculation directory on the remote machine to confirm that the job
        has finished successfully."""
        return self.remote_fs.exists(self.calcdir / "is_finished.info")

    def node_problem(self):
        """If the Node has the requested resources there should be a file called
        'node_check.info' in the calculation directory on the remote machine.

        Returns:
            False, no problem on node. True, problem on node.
        """
        return self.remote_fs.exists(self.calcdir / "node_check.info")

    def run(self, only_start=False):
        """Run the entire workflow: allocate data, submit job, and collect results."""
        exclude_node_list = []
        self.allocate_data()
        while True:
            self.submit(only_start=only_start)
            if only_start:
                break

            # Node-exclusion retry is a scheduler capability (re-place the job
            # on another node). Backends without a notion of nodes run once.
            if not self.executor.supports_node_retry:
                break

            if not self.node_problem():
                break

            else:
                logger.warning(
                    "The job did not run on a node with the requested resources. "
                    "Resubmitting..."
                )

                last_used_node = (
                    self.remote_fs.read_str_from_file(
                        self.calcdir / "node_check.info"
                    )
                    .strip()
                    .split(".")[0]
                )
                logger.warning(f"Excluding node: {last_used_node}")

                exclude_node_list.append(last_used_node)

                self.executor.write_jobfile(
                    self.input_file_remote,
                    self.script_file,
                    exclude_nodes=",".join(exclude_node_list),
                )

        if only_start:
            return

        self.collect_results()


@dataclass
class Writeable(ABC):
    """This base class serves as an interface for one unit of data supplied to
    a slurm job.

    Example:
        You run DFT with an ArrayScriptTask. You have 1000 configs in total,
        you distribute them to 10 individual jobs in an array. Then each job of
        the array will require 100 configurations, which you will have to
        prepare in the directory of the job.

        Methods that need to be implemented are:

        - ``write``: Logic for writing the content to disk. This can be a
          simple write to a file, but can also include more elaborate logic
          like writing an ASE trajectory file.
    """

    content: Any
    filename: str
    remote_fs: file_handler.FileSystemHandlerBase

    @abstractmethod
    def write(self, calcpath):
        """Specify the logic to write content to disk"""
        pass


class DictJSON(Writeable):
    def write(self, calcpath):
        self.remote_fs.write_dict_to_json(
            Path(calcpath) / self.filename, self.content
        )


class AtomsBatch(Writeable):
    def write(self, calcpath):
        self.remote_fs.write_ase(Path(calcpath) / self.filename, self.content)


class Reader(ABC):
    @abstractmethod
    def read(self, calcpath):
        pass


class FileAllocator:
    """Handles file allocation and copying to remote filesystem.

    This component is responsible for:
    - Copying individual files to remote
    - Allocating shared resources that all jobs can access
    - Creating calculation directories and writing individual resources

    Args:
        remote_fs: FileSystemHandlerBase instance for remote operations
        rootpath: Root directory on remote where files are allocated
    """

    def __init__(
        self,
        remote_fs: file_handler.FileSystemHandlerBase,
        rootpath: os.PathLike,
    ):
        self.remote_fs = remote_fs
        self.rootpath = Path(rootpath)

    def copy_file(self, local_path: os.PathLike) -> Path:
        """Copy a single file from local to remote rootpath.

        Args:
            local_path: Path to the local file to copy

        Returns:
            Path to the file on the remote machine
        """
        moved_path = self.rootpath / os.path.split(local_path)[-1]
        self.remote_fs.put(local_path, moved_path)
        logger.debug(f"Copied {local_path} to {moved_path}")
        return moved_path

    def copy_path(self, local_path: os.PathLike) -> Path:
        """Copy a local file *or* directory tree to remote rootpath.

        Like :meth:`copy_file`, but if ``local_path`` is a directory (e.g. a
        mace-jax model bundle of ``config.json`` + ``params.msgpack``) it is
        uploaded recursively via ``put_dir`` rather than the single-file
        ``put``. A plain file is handled exactly as ``copy_file`` would.

        Args:
            local_path: Path to the local file or directory to copy

        Returns:
            Path to the file/directory on the remote machine
        """
        if not Path(local_path).is_dir():
            return self.copy_file(local_path)
        moved_path = self.rootpath / Path(local_path).name
        self.remote_fs.put_dir(local_path, moved_path)
        logger.debug(f"Copied directory {local_path} to {moved_path}")
        return moved_path

    def allocate_shared_resources(
        self, shared_resources: list[os.PathLike]
    ) -> list[Path]:
        """Copy shared resources to remote rootpath.

        Shared resources are files that all jobs in an array can access,
        such as model parameters or configuration files.

        Args:
            shared_resources: List of paths to files to copy

        Returns:
            List of paths to the copied files on remote
        """
        remote_paths = []
        for resource in shared_resources:
            remote_path = self.copy_file(resource)
            remote_paths.append(remote_path)
        logger.info(f"Allocated {len(shared_resources)} shared resources")
        return remote_paths

    def allocate_individual_resources(
        self,
        individual_resources: list[list],
        calcdir: Path,
    ) -> list[Path]:
        """Create calculation directories and write individual resources.

        For array jobs, each task needs its own calculation directory with
        its own input data. This method creates these directories and writes
        the Writeable objects to them.

        Args:
            individual_resources: List of lists of Writeable objects
            calcdir: Base calculation directory (will create subdirs 0, 1, 2, ...)

        Returns:
            List of paths to created calculation directories
        """
        all_calcpaths = []

        for idx, data_at_idx in enumerate(zip(*individual_resources)):
            # Create a directory for each job in the array
            calcpath = calcdir / f"{idx}"
            self.remote_fs.makedirs(calcpath)

            # For each Writeable object, call its write method
            for data in data_at_idx:
                data.write(calcpath)

            all_calcpaths.append(calcpath)

        logger.info(
            f"Allocated {len(all_calcpaths)} individual calculation directories"
        )
        return all_calcpaths


class ResultCollector:
    """Handles result collection from remote to local filesystem.

    This component is responsible for copying results back from the remote
    machine to the local machine after jobs complete. It handles directory
    creation and provides utilities for selective result collection.

    Args:
        remote_fs: FileSystemHandlerBase instance for remote operations
        local_fs: FileSystemHandlerBase instance for local operations
    """

    def __init__(
        self,
        remote_fs: file_handler.FileSystemHandlerBase,
        local_fs: file_handler.FileSystemHandlerBase,
    ):
        self.remote_fs = remote_fs
        self.local_fs = local_fs

    def collect_directory(
        self,
        remote_path: os.PathLike,
        local_path: os.PathLike,
        move: bool = True,
    ) -> None:
        """Copy an entire directory from remote to local.

        Args:
            remote_path: Path to directory on remote machine
            local_path: Path where directory should be copied on local machine
            move: If True, handle existing local_path by moving it aside
        """
        # Ensure local path exists and handle existing files
        file_utils.secure_declare_path(
            local_path, move=move, is_directory=True
        )

        logger.info(f"Collecting results from {remote_path} to {local_path}")
        self.remote_fs.get_dir(remote_path, local_path)
        logger.info("Results collected successfully")

    def collect_file(
        self,
        remote_path: os.PathLike,
        local_path: os.PathLike,
    ) -> None:
        """Copy a single file from remote to local.

        Args:
            remote_path: Path to file on remote machine
            local_path: Path where file should be copied on local machine
        """
        logger.debug(f"Collecting file from {remote_path} to {local_path}")
        self.remote_fs.get(remote_path, local_path)


class ArrayScriptTask(RemoteTask):
    """This is a task that consists of running the same script for a bunch of
    inputs using the slurm array functionality. The `remote_savepath` is the
    parent directory of all computations run on the remote machine. The idea
    behind this implementation is, that you prepare a indexed directory for
    each individual job of the array (like remote_savepath/0,
    remote_savepath/1, ...). The slurm array then iterates through all of these
    directories. The logic for each individual computation needs to be inferred
    inside the python script by parsing a config file.

    Hence, preparing an ArrayScriptTask is more involved than running a
    SingleScriptTask, as it requires a more elaborate preparation. However,
    it also allows you to efficiently handle a huge amount of data without
    overcharging slurm.

    Args:
        local_savepath: Path where finished results will be copied from remote
            to local machine
        cluster_config: ClusterConfig instance containing cluster-specific settings.
        remote_fs: FileSystemHandlerBase derived object for file operations.
        python_script_file: Python script that performs the computation.
        individual_ressources: A list of lists of Writeable objects that are
            needed for the individual jobs of the array.
        shared_ressources: A list of paths to files that are copied to the
            remote computation directory, but are not specific to one job of
            the array. This can be e.g. a potential file that is read by the
            script.
        logfile: Path to the logfile of the task.
        remote_savepath: Path on remote machine, where computation will be
            performed. If None, falls back to cluster_config.savepath.
        collect_only: If True, results are collected from `remote_savepath`
            if it already exists. If False a new savepath will be created with
            a tag to prevent overwriting.
        num_parallel: Number of array jobs that are submitted in parallel.
        loop_time: The time interval which is used to check whether the job is
            completed.
        jobname: Name of the job used in slurm.
        n_processors: Number of processors that are requested per job. For CPU
            this refers to n_tasks, where as for GPU on datalab this refers to
            the number of GPU devices requested on a single node.
        jobfile_name: Name of the slurm job script that is created to submit the job.
        slurm_header_add: Additional lines that are added to the header of the
            slurm job script.
    """

    def __init__(
        self,
        local_savepath: os.PathLike,
        cluster_config: ClusterConfig,
        remote_fs: FileSystemHandlerBase,
        python_script_file: os.PathLike = None,
        individual_ressources: list[list[Writeable]] = [],
        shared_ressources: list[os.PathLike] = [],
        logfile: os.PathLike = "task.log",
        remote_savepath: Optional[os.PathLike] = None,
        collect_only: bool = False,
        num_parallel: int = 10,
        loop_time: float = 30.0,
        jobname: str = "job",
        n_processors: int = 2,
        jobfile_name: str = "run_slurm.sh",
        slurm_header_add: str = "",
    ):
        if individual_ressources:
            # Save the number of individual jobs in the array
            self.num_array_tasks = len(individual_ressources[0])

        if remote_savepath is None:
            logger.info("Falling back to savepath from cluster_config")
            remote_savepath = cluster_config.savepath

        # Need to init base class first, since it handles savepath
        super().__init__(
            logfile=logfile,
            savepath=remote_savepath,
            collect_only=collect_only,
            remote_fs=remote_fs,
            jobname=jobname,
        )

        # The directory where the computation will be performed
        self.rootpath = Path(self.savepath)

        # Propagate this cluster's opt-in connection policy to the SFTP layer
        # so file transfers pause-and-retry on auth loss too (no-op if off).
        if hasattr(remote_fs, "connection_policy"):
            remote_fs.connection_policy = cluster_config.connection_policy

        # Store the provided arguments
        self.python_script_file = python_script_file
        self.individual_ressources = individual_ressources
        self.shared_ressources = shared_ressources
        self.loop_time = loop_time
        self.local_savepath = local_savepath

        # Instantiate the execution backend for array jobs
        self.executor = executors.make_array_executor(
            cluster_config,
            jobname=jobname,
            savepath=self.rootpath,
            n_processors=n_processors,
            remote_fs=remote_fs,
            slurm_header_add=slurm_header_add,
            jobfile_name=jobfile_name,
            loop_time=loop_time,
            num_parallel=num_parallel,
            job_type_template="body_array.sh.j2",
        )
        # Back-compat alias (see SingleScriptTask). Same object as ``.executor``.
        self.slurm_handler = self.executor

        self.calcdir = self.executor.calcdir

        # Create composable components for file allocation and result collection
        self.file_allocator = FileAllocator(
            remote_fs=remote_fs, rootpath=self.rootpath
        )
        self.result_collector = ResultCollector(
            remote_fs=remote_fs, local_fs=self.local_fs
        )

        logger.info(f"ArrayScriptTask")
        logger.info(f"rootpath: {str(self.rootpath)}")

        # Validate resources early if not in collect_only mode
        if not collect_only:
            self._validate_resources()

    def _validate_resources(self):
        """Validate python script and resources before allocation.

        Performs early validation to catch common errors before attempting
        to allocate resources on the remote machine.

        Raises:
            ValueError: If validation fails with descriptive error message
        """
        # 1. Validate python script
        if self.python_script_file is not None:
            if not os.path.exists(self.python_script_file):
                raise TaskError(
                    f"Python script not found: {self.python_script_file}"
                )
            if not str(self.python_script_file).endswith(".py"):
                logger.warning(
                    f"Python script does not have .py extension: "
                    f"{self.python_script_file}"
                )
            if not os.access(self.python_script_file, os.R_OK):
                raise TaskError(
                    f"Python script is not readable: {self.python_script_file}"
                )

        # 2. Validate shared resources
        if self.shared_ressources:
            shared_filenames = set()
            for resource in self.shared_ressources:
                # Check file exists
                if not os.path.exists(resource):
                    raise TaskError(f"Shared resource not found: {resource}")

                # Check for duplicate filenames
                filename = os.path.basename(resource)
                if filename in shared_filenames:
                    raise TaskError(
                        f"Duplicate shared resource filename: {filename}"
                    )
                shared_filenames.add(filename)

                # Warn about large files (> 100MB)
                file_size = os.path.getsize(resource)
                if file_size > 100 * 1024 * 1024:  # 100MB
                    logger.warning(
                        f"Large shared resource ({file_size / 1024 / 1024:.1f}MB): "
                        f"{filename}"
                    )

        # 3. Validate individual resources
        if self.individual_ressources:
            # Check that all sublists have same length
            lengths = [len(content) for content in self.individual_ressources]
            if len(set(lengths)) > 1:
                raise TaskError(
                    f"Individual resources have inconsistent lengths: {lengths}"
                )

            # Check that each item is a Writeable instance
            for i, content_list in enumerate(self.individual_ressources):
                for j, item in enumerate(content_list):
                    if not isinstance(item, Writeable):
                        raise TypeError(
                            f"Individual resource [{i}][{j}] is not a Writeable "
                            f"instance: {type(item)}. Currently there are "
                            "DictJSON and AtomsBatch, anything else needs a"
                            "(very simple) custom implementation."
                        )

                    # Validate Writeable attributes
                    if not hasattr(item, "filename") or not item.filename:
                        raise TaskError(
                            f"Individual resource [{i}][{j}] missing filename"
                        )
                    if not hasattr(item, "content"):
                        raise TaskError(
                            f"Individual resource [{i}][{j}] missing content"
                        )

            # Check for filename conflicts within each calcdir
            if self.individual_ressources:
                num_tasks = len(self.individual_ressources[0])
                for task_idx in range(num_tasks):
                    filenames = set()
                    for content_list in self.individual_ressources:
                        filename = content_list[task_idx].filename
                        if filename in filenames:
                            raise TaskError(
                                f"Duplicate filename in task {task_idx}: {filename}"
                            )
                        filenames.add(filename)

                        # Check for conflicts with shared resources
                        if self.shared_ressources:
                            for resource in self.shared_ressources:
                                shared_name = os.path.basename(resource)
                                if filename == shared_name:
                                    logger.warning(
                                        f"Task {task_idx} has individual resource "
                                        f"'{filename}' that conflicts with shared "
                                        f"resource name"
                                    )

        logger.debug("Resource validation passed")

    def copy_file(self, local_path):
        """Copies a local file to the remote rootpath directory.
        Args:
            local_path: Path to the local file.
        Returns:
            The path to the file on the remote machine.
        """
        return self.file_allocator.copy_file(local_path)

    def copy_path(self, local_path):
        """Copy a local file *or* directory tree to the remote rootpath.

        Use this instead of :meth:`copy_file` when the artifact may be a
        directory (e.g. a mace-jax model bundle); files behave identically.

        Args:
            local_path: Path to the local file or directory.
        Returns:
            The path to the file/directory on the remote machine.
        """
        return self.file_allocator.copy_path(local_path)

    def allocate_data(self, slurm_only: bool = False):
        """Allocate the script and input files on the remote machine and
        prepare the slurm job script.

        Args:
            slurm_only: Write only the SLURM file, don't allocate ressources.
        """
        # Copy the python script to the calculation directory on the remote machine
        self.script_file_remote = self.copy_file(self.python_script_file)
        # Write the job script on the remote machine
        self.executor.write_jobfile(
            self.script_file_remote, self.num_array_tasks
        )

        if not slurm_only:
            if self.shared_ressources:
                self.file_allocator.allocate_shared_resources(
                    self.shared_ressources
                )

            self.all_calcpaths = (
                self.file_allocator.allocate_individual_resources(
                    self.individual_ressources, self.calcdir
                )
            )

    def submit(self):
        # Submit the job and wait until it is finished.
        self.executor.submit()
        self.executor.wait_until_finished()

    def collect_results(self):
        """Copy the entire rootpath directory from the remote machine to
        the local machine using ResultCollector."""
        self.result_collector.collect_directory(
            self.rootpath, self.local_savepath, move=True
        )

    def finish_check(self):
        """Check for the presence of the 'is_finished.info' file in each
        calculation directory on the remote machine to confirm that all jobs
        have finished successfully."""
        assert self.all_calcpaths, "There are no calcpaths to check!"

        for calcpath in self.all_calcpaths:
            if not self.remote_fs.exists(Path(calcpath) / "is_finished.info"):
                return False

        return True
