import inspect
import logging
import os
import random
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from remote_handler.core import executors, file_handler, file_utils, remote_env
from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import (
    FileSystemHandlerBase,
    LocalFileSystem,
)
from remote_handler.core.file_utils import tag_path
from remote_handler.core.manifest import TASK_ARRAY, TASK_SINGLE, TaskManifest
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


def finalize_manifest_kwargs(
    cls: type, base_kwargs: dict, extra: dict
) -> dict:
    """Merge manifest-derived kwargs with caller-supplied ``extra`` overrides.

    Used by ``from_manifest()`` classmethods to build the final constructor
    call -- shared across modules (``ArrayScriptTask``/``SingleScriptTask``
    here, ``RemoteAtomsHandler`` in ``atoms_handler.py``), so it is public
    rather than module-private. Any ``extra`` key that is not a real
    parameter of ``cls.__init__`` is a typo and raises loudly; anything else
    (including a stale key that ends up in ``base_kwargs``) is silently
    dropped rather than raising a confusing ``TypeError`` from the
    constructor call.
    """
    sig_params = inspect.signature(cls.__init__).parameters
    unknown = sorted(key for key in extra if key not in sig_params)
    if unknown:
        raise TaskError(
            f"Unknown keyword argument(s) for {cls.__name__}.from_manifest(): "
            f"{', '.join(unknown)}"
        )
    merged = {**base_kwargs, **extra}
    return {key: value for key, value in merged.items() if key in sig_params}


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

    def _build_manifest(
        self, task_type: str, num_array_tasks: Optional[int] = None
    ) -> TaskManifest:
        """Snapshot the state needed to await/collect this task later.

        Reads the durable handle straight off the (already-submitted)
        executor -- job id, resolved remote dir, cluster config -- so a fresh
        process can rebuild it via
        :class:`~remote_handler.core.manifest.TaskHandle`. Subclasses expose
        this through :meth:`write_manifest`.
        """
        executor = self.executor
        cluster_config = executor.cluster_config
        backend = getattr(cluster_config, "executor", "slurm") or "slurm"
        job_id = getattr(executor, "pid", None)
        return TaskManifest(
            task_type=task_type,
            backend=backend,
            job_id=None if job_id is None else str(job_id),
            remote_savepath=str(self.savepath),
            local_savepath=str(self.local_savepath),
            jobname=executor.jobname,
            cluster_config=cluster_config.to_dict(),
            fs_type=(
                "local"
                if isinstance(self.remote_fs, LocalFileSystem)
                else "remote"
            ),
            n_processors=executor.n_processors,
            loop_time=self.loop_time,
            num_parallel=getattr(executor, "num_parallel", 10),
            num_array_tasks=num_array_tasks,
        )

    def _save_manifest(
        self,
        task_type: str,
        num_array_tasks: Optional[int],
        local_path: Optional[os.PathLike],
        write_remote: bool,
    ) -> Path:
        """Persist a manifest locally and (by default) in the remote job dir.

        The local copy defaults to ``<local_savepath>.manifest.json`` so it
        sits beside where results will land and stays unique per task.

        The local copy is the resume contract and is written first (its failure
        propagates); the remote copy is redundancy for resume-from-anywhere and
        is best-effort -- a failure there only warns, so an unreachable or
        read-only remote never defeats the local handle.
        """
        manifest = self._build_manifest(task_type, num_array_tasks)
        if local_path is None:
            local_path = Path(str(self.local_savepath) + ".manifest.json")
        manifest.save_local(local_path)
        self.manifest_path = Path(local_path)
        if write_remote:
            try:
                manifest.save_remote(self.remote_fs)
            except Exception as exc:  # noqa: BLE001 - bonus copy, never fatal
                logger.warning("Could not write remote manifest copy: %s", exc)
        return self.manifest_path

    def _maybe_write_manifest(
        self, task_type: str, num_array_tasks: Optional[int] = None
    ) -> None:
        """Best-effort manifest write from inside ``submit`` (default path).

        Called right after submission and *before* the wait, so the durable
        handle exists on disk while the job runs -- if the driver then dies in
        the wait loop, the job is recoverable. Fully best-effort: any failure
        (including the local write) only warns, so persisting the safety net
        can never take down the job it is protecting. Disabled per-task via
        ``write_manifest_on_submit=False``.
        """
        if not getattr(self, "write_manifest_on_submit", True):
            return
        try:
            self._save_manifest(task_type, num_array_tasks, None, True)
        except Exception as exc:  # noqa: BLE001 - safety net, never fatal
            logger.warning("Could not persist task manifest: %s", exc)


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
        write_manifest_on_submit: If True (default), ``submit`` persists a
            resume manifest right after submission (best-effort), so a driver
            crash during the wait does not orphan the running job. Set False
            to opt out.
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
        time_limit: Optional[str] = None,
        memory: Optional[str] = None,
        write_manifest_on_submit: bool = True,
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
        self.write_manifest_on_submit = write_manifest_on_submit

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
            time_limit=time_limit,
            memory=memory,
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
        """Submit the job and wait until it is finished.

        Persists a resume manifest right after submission and before the wait
        (best-effort; disable with ``write_manifest_on_submit=False``), so a
        driver crash during the wait does not orphan the running job.
        """
        self.executor.submit()
        self._maybe_write_manifest(TASK_SINGLE)
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

    def write_manifest(self, local_path=None, write_remote=True) -> Path:
        """Persist a resume handle for this (submitted) task.

        Call after ``submit``/``run(only_start=True)``. Reload it later with
        ``TaskHandle.from_file`` to poll and collect the job from another
        process.
        """
        return self._save_manifest(TASK_SINGLE, None, local_path, write_remote)

    def submit_detached(self, local_path=None, write_remote=True) -> Path:
        """Start the job and persist a manifest, without allocating first.

        The half of :meth:`detach` that starts after allocation; useful for a
        caller that has already run ``allocate_data()`` itself (e.g. a
        workflow actor that allocates as part of ``update()`` and only wants
        to detach at submit time).

        Suppresses ``submit``'s own manifest write and writes exactly once here
        instead, so the ``local_path`` / ``write_remote`` arguments are honored
        (and the remote copy is not written twice).
        """
        prev = self.write_manifest_on_submit
        self.write_manifest_on_submit = False
        try:
            self.submit(only_start=True)
        finally:
            self.write_manifest_on_submit = prev
        return self.write_manifest(
            local_path=local_path, write_remote=write_remote
        )

    def detach(self, local_path=None, write_remote=True) -> Path:
        """Allocate, start the job, persist a manifest, and return at once.

        The detach-on-submit path: fire the remote job and hand back the
        local manifest path without waiting, so submission feels like a local
        backgrounded command. Resume with ``TaskHandle.from_file``.
        """
        self.allocate_data()
        return self.submit_detached(
            local_path=local_path, write_remote=write_remote
        )

    @classmethod
    def from_manifest(
        cls,
        manifest_path: os.PathLike,
        remote_fs: Optional[FileSystemHandlerBase] = None,
        **extra,
    ) -> "SingleScriptTask":
        """Reconstruct a collect-only task from a persisted manifest.

        Rebuilds the task exactly as it ran (same resolved ``savepath``,
        cluster config, and filesystem), without re-allocating or
        re-submitting, so ``finish_check()`` / ``collect_results()`` can be
        called on it in a fresh process. ``script_file``/``input_file`` are
        never read in ``collect_only`` mode, so they are left ``None``.

        Args:
            manifest_path: Local ``*.manifest.json`` file written by
                ``write_manifest``/``detach``.
            remote_fs: Filesystem to use; rebuilt from the manifest if not
                given.
            **extra: Additional/overriding constructor kwargs (e.g.
                ``logfile``); must be valid ``SingleScriptTask`` parameters.
        """
        if cls is not SingleScriptTask:
            raise TaskError(
                "SingleScriptTask.from_manifest() only reconstructs the base "
                f"SingleScriptTask class, not {cls.__name__}."
            )
        manifest = TaskManifest.load(manifest_path)
        if manifest.task_type != TASK_SINGLE:
            raise TaskError(
                f"Manifest at {manifest_path} is a {manifest.task_type!r} "
                "task, not a single task."
            )
        fs = remote_fs or manifest.build_fs()
        remote_savepath = Path(manifest.remote_savepath).parent
        jobname = Path(manifest.remote_savepath).name

        base_kwargs = dict(
            script_file=None,
            input_file=None,
            local_savepath=manifest.local_savepath,
            cluster_config=manifest.build_cluster_config(),
            remote_fs=fs,
            remote_savepath=remote_savepath,
            jobname=jobname,
            collect_only=True,
            n_processors=manifest.n_processors,
            loop_time=manifest.loop_time,
        )
        return cls(**finalize_manifest_kwargs(cls, base_kwargs, extra))


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
        write_manifest_on_submit: If True (default), ``submit`` persists a
            resume manifest right after submission (best-effort), so a driver
            crash during the wait does not orphan the running array. Set False
            to opt out.
        num_array_tasks: Explicit number of array tasks. Only needed when
            ``individual_ressources`` is empty (e.g. reconstructing a
            ``collect_only`` task via :meth:`from_manifest`, where no
            allocation happens); inferred from ``individual_ressources``
            otherwise. Passing both must agree.
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
        time_limit: Optional[str] = None,
        memory: Optional[str] = None,
        write_manifest_on_submit: bool = True,
        num_array_tasks: Optional[int] = None,
    ):
        self.write_manifest_on_submit = write_manifest_on_submit

        inferred_num_array_tasks = (
            len(individual_ressources[0]) if individual_ressources else None
        )
        if (
            num_array_tasks is not None
            and inferred_num_array_tasks is not None
            and num_array_tasks != inferred_num_array_tasks
        ):
            raise TaskError(
                f"num_array_tasks={num_array_tasks} disagrees with the "
                f"length of individual_ressources ({inferred_num_array_tasks})."
            )
        # Save the number of individual jobs in the array: the explicit
        # argument wins, falling back to what individual_ressources implies,
        # else left unknown (None) -- e.g. a collect_only reconstruction that
        # was not told the count.
        self.num_array_tasks = (
            num_array_tasks
            if num_array_tasks is not None
            else inferred_num_array_tasks
        )

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
            time_limit=time_limit,
            memory=memory,
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

            self.file_allocator.allocate_individual_resources(
                self.individual_ressources, self.calcdir
            )

    @property
    def all_calcpaths(self) -> list[Path]:
        """The per-task calculation directories, ``calcdir/0``, ``calcdir/1``, ...

        Computed from ``num_array_tasks`` rather than stored, so it works
        equally whether this task just allocated its own directories or was
        reconstructed via :meth:`from_manifest` without allocating at all --
        as long as ``calcdir`` resolves to the same directory both times
        (guaranteed by reusing the manifest's resolved ``remote_savepath``).
        """
        if self.num_array_tasks is None:
            raise TaskError(
                "num_array_tasks is unknown for this ArrayScriptTask; pass "
                "it explicitly or via individual_ressources."
            )
        return [self.calcdir / str(i) for i in range(self.num_array_tasks)]

    def submit(self, only_start=False):
        """Submit the array job.

        Persists a resume manifest right after submission and before the wait
        (best-effort; disable with ``write_manifest_on_submit=False``), so a
        driver crash during the wait does not orphan the running array.

        Args:
            only_start: If True, return right after submission instead of
                blocking until the array finishes -- used by ``detach``.
        """
        self.executor.submit()
        self._maybe_write_manifest(
            TASK_ARRAY, getattr(self, "num_array_tasks", None)
        )
        if only_start:
            return
        self.executor.wait_until_finished()

    def collect_results(self):
        """Copy the entire rootpath directory from the remote machine to
        the local machine using ResultCollector."""
        self.result_collector.collect_directory(
            self.rootpath, self.local_savepath, move=True
        )

    def write_manifest(self, local_path=None, write_remote=True) -> Path:
        """Persist a resume handle for this (submitted) array task.

        See :meth:`SingleScriptTask.write_manifest`; records
        ``num_array_tasks`` so the array executor is rebuilt correctly.
        """
        return self._save_manifest(
            TASK_ARRAY,
            getattr(self, "num_array_tasks", None),
            local_path,
            write_remote,
        )

    def submit_detached(self, local_path=None, write_remote=True) -> Path:
        """Start the array job and persist a manifest, without allocating first.

        See :meth:`SingleScriptTask.submit_detached`; useful when
        ``allocate_data()`` has already run (e.g. a trainer that allocates
        during ``update()`` and only wants to detach at submit time).
        """
        prev = self.write_manifest_on_submit
        self.write_manifest_on_submit = False
        try:
            self.submit(only_start=True)
        finally:
            self.write_manifest_on_submit = prev
        return self.write_manifest(
            local_path=local_path, write_remote=write_remote
        )

    def detach(self, local_path=None, write_remote=True) -> Path:
        """Allocate, start the array job, persist manifest, return at once.

        See :meth:`SingleScriptTask.detach`.
        """
        self.allocate_data()
        return self.submit_detached(
            local_path=local_path, write_remote=write_remote
        )

    @classmethod
    def from_manifest(
        cls,
        manifest_path: os.PathLike,
        remote_fs: Optional[FileSystemHandlerBase] = None,
        **extra,
    ) -> "ArrayScriptTask":
        """Reconstruct a collect-only array task from a persisted manifest.

        Rebuilds the task exactly as it ran (same resolved ``savepath``,
        cluster config, filesystem, and ``num_array_tasks``), without
        re-allocating or re-submitting, so ``finish_check()`` /
        ``collect_results()`` can be called on it in a fresh process.

        Args:
            manifest_path: Local ``*.manifest.json`` file written by
                ``write_manifest``/``detach``.
            remote_fs: Filesystem to use; rebuilt from the manifest if not
                given.
            **extra: Additional/overriding constructor kwargs (e.g.
                ``python_script_file``, ``logfile``); must be valid
                ``ArrayScriptTask`` parameters.
        """
        if cls is not ArrayScriptTask:
            raise TaskError(
                "ArrayScriptTask.from_manifest() only reconstructs the base "
                f"ArrayScriptTask class, not {cls.__name__}."
            )
        manifest = TaskManifest.load(manifest_path)
        if manifest.task_type != TASK_ARRAY:
            raise TaskError(
                f"Manifest at {manifest_path} is a {manifest.task_type!r} "
                "task, not an array task."
            )
        fs = remote_fs or manifest.build_fs()
        remote_savepath = Path(manifest.remote_savepath).parent
        jobname = Path(manifest.remote_savepath).name

        base_kwargs = dict(
            local_savepath=manifest.local_savepath,
            cluster_config=manifest.build_cluster_config(),
            remote_fs=fs,
            remote_savepath=remote_savepath,
            jobname=jobname,
            collect_only=True,
            n_processors=manifest.n_processors,
            loop_time=manifest.loop_time,
            num_parallel=manifest.num_parallel,
            num_array_tasks=manifest.num_array_tasks,
        )
        return cls(**finalize_manifest_kwargs(cls, base_kwargs, extra))

    def finish_check(self):
        """Check for the presence of the 'is_finished.info' file in each
        calculation directory on the remote machine to confirm that all jobs
        have finished successfully."""
        assert self.all_calcpaths, "There are no calcpaths to check!"

        for calcpath in self.all_calcpaths:
            if not self.remote_fs.exists(Path(calcpath) / "is_finished.info"):
                return False

        return True


class DetachableTaskMixin:
    """Shared "submit now, collect later" logic for task-wrapping actors.

    Every actor that wraps one remote task per iteration (build, allocate,
    submit, later collect) -- a ``ModelTrainer``, a ``GroundTruthEvaluator``,
    or a downstream ``Sampler`` (concrete samplers live outside this
    package, in downstream repos, so this needs to be public) -- needs the
    same ``collect_only`` mode switch: build+allocate+submit on the original
    call, or reconstruct the already-submitted task from its manifest --
    with no allocation, no re-submit -- on a later, standalone collect call.
    Before this mixin, each actor hand-rolled that branch itself (see
    ``remote_handler.workflows.trainers._MaceArrayTrainer``'s git history);
    factoring it out here means any task-wrapping actor gets "detach one
    process, collect from another" for free instead of re-implementing it.

    Mix this into a class that already sets ``self.collect_only`` and
    ``self.remote_fs`` in its own ``__init__`` -- those stay declared
    per-class (they belong in each actor's own constructor signature/
    docstring, same as ``n_processors``, ``loop_time``, etc.), only the
    logic that *uses* them is shared here.

    Works for both task kinds an actor might wrap -- see
    :meth:`build_or_resume_array_task` (``ArrayScriptTask``, an ensemble/
    array of jobs) and :meth:`build_or_resume_single_task`
    (``SingleScriptTask``, one job). Both funnel through
    :meth:`build_or_resume_task`, identical apart from which class's
    ``from_manifest`` gets called.

    Note that "wrap the task-building step in a closure" only gets an actor
    this far, not all the way: an actor whose downstream code depends on
    something the build step computes (e.g. a per-run remote filename) needs
    that piece made available independent of the closure too, since the
    closure is never called in ``collect_only`` mode -- see
    ``NeuralilILTrainer``'s ``pickle_filename``, computed from ``calcdir``
    alone up front, precisely so ``get_trained_model()`` does not depend on
    the closure having run.
    """

    collect_only: bool
    remote_fs: FileSystemHandlerBase

    def manifest_path(self, results_dir: os.PathLike) -> Path:
        """Where ``write_manifest``/``detach`` puts this task's manifest.

        A fixed naming convention (``<results_dir>.manifest.json``, matching
        ``RemoteTask._save_manifest``'s own default) rather than a
        constructor argument -- so a later ``collect_only=True`` call can
        find it from ``results_dir`` alone, without the caller having to
        remember/pass a manifest path explicitly.
        """
        return Path(str(results_dir) + ".manifest.json")

    def build_or_resume_task(
        self,
        task_cls: type,
        results_dir: os.PathLike,
        build_task: Callable[[], "RemoteTask"],
        **from_manifest_extra,
    ) -> "RemoteTask":
        """Return this iteration's task, built fresh or resumed from a manifest.

        ``collect_only=False`` (the normal path): calls ``build_task()`` --
        supplied by the concrete actor, since only it knows how to construct
        its own task (payload script, inputs, shared resources, ...) -- then
        allocates it. Allocation stays here rather than inside ``build_task``
        so every caller gets it uniformly, the same way it's skipped
        uniformly below.

        ``collect_only=True``: reconstructs the task an earlier,
        ``collect_only=False`` call already submitted, via
        ``task_cls.from_manifest`` at :meth:`manifest_path`, with no
        allocation and no re-submit. ``build_task`` is never called in this
        branch -- constructing it would touch things a pure collect call
        shouldn't need (e.g. resolving a remote train-runner path, or the
        original training data).

        Args:
            task_cls: ``ArrayScriptTask`` or ``SingleScriptTask`` -- whichever
                this actor wraps.
            results_dir: This iteration's task directory (e.g.
                ``self.results_dir``); also anchors the manifest lookup.
            build_task: Zero-arg callable building this actor's task. Only
                invoked when not ``collect_only``.
            **from_manifest_extra: Extra constructor overrides for the
                ``collect_only`` path (e.g. ``python_script_file``); ignored
                otherwise.
        """
        if self.collect_only:
            return task_cls.from_manifest(
                self.manifest_path(results_dir),
                remote_fs=self.remote_fs,
                **from_manifest_extra,
            )
        task = build_task()
        task.allocate_data()
        return task

    def build_or_resume_array_task(
        self,
        results_dir: os.PathLike,
        build_task: Callable[[], "ArrayScriptTask"],
        **from_manifest_extra,
    ) -> "ArrayScriptTask":
        """:meth:`build_or_resume_task` for an ``ArrayScriptTask``."""
        return self.build_or_resume_task(
            ArrayScriptTask, results_dir, build_task, **from_manifest_extra
        )

    def build_or_resume_single_task(
        self,
        results_dir: os.PathLike,
        build_task: Callable[[], "SingleScriptTask"],
        **from_manifest_extra,
    ) -> "SingleScriptTask":
        """:meth:`build_or_resume_task` for a ``SingleScriptTask``."""
        return self.build_or_resume_task(
            SingleScriptTask, results_dir, build_task, **from_manifest_extra
        )

    def submit_if_needed(self, task: "RemoteTask") -> None:
        """Submit ``task``, unless it was already submitted in another process."""
        if not self.collect_only:
            task.submit()
