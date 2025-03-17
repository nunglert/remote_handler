import logging
import os
import random
import string
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

from remote_handler import cluster_handler, file_handler, slurm_utils
from remote_handler.file_handler import (
    FileSystemHandlerBase,
    LocalFileSystem,
    RemoteFileSystem,
    tag_path,
)

logger = logging.getLogger(__name__)


class SimpleLogger:
    def __init__(self, log_file=None):
        self.log_file = log_file

    def print(self, *args):
        # Join all arguments into a single string with spaces, like `print`
        message = " ".join(map(str, args))
        logger.info(message)

        if self.log_file is not None:
            # Write the message to the specified file
            with open(self.log_file, "a") as f:
                f.write(message + "\n")
        else:
            print(message)


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

    Args:
        logfile: A log file on the host machine where output is printed
        savepath: The path on the remote system where computations should be
            performed
        remote_fs: A FileSystemHandlerBase derived object.
        collect_only: By default, RemoteTask creates a new tagged savepath if
            the provided savepath already exists. If you want RemoteTask act on
            a `savepath` that exists already, set this to True. This can be
            helpful if a task has crashed and you want to continue it or
            collect its results.
    """

    def __init__(
        self,
        logfile: os.PathLike,
        savepath: os.PathLike,
        remote_fs: file_handler.FileSystemHandlerBase,
        collect_only: bool = False,
    ):
        self.logger = SimpleLogger(logfile)
        self.local_fs = LocalFileSystem()
        self.remote_fs = remote_fs

        self.savepath = savepath

        if savepath is not None:
            parent, child = os.path.split(savepath)
            if not self.remote_fs.exists(parent):
                raise Exception(
                    "The parent of savepath must already exist on the remote system!"
                )

            if self.remote_fs.exists(savepath):
                if not collect_only:
                    logger.warning(
                        "savepath is already existing. A new directory with "
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
                    self.savepath = Path(
                        str(
                            self.remote_fs.secure_path(
                                savetag, is_directory=True
                            )
                        )
                        + random_string
                    )
                    self.remote_fs.makedirs(self.savepath, exist_ok=True)

            else:
                if collect_only:
                    raise Exception(
                        "Can't collect if savepath does not exist on remote "
                        "machine"
                    )
                logger.info("Creating the savepath at requested location")
                self.remote_fs.makedirs(self.savepath, exist_ok=True)
        else:
            logger.info(
                "No savepath provided, a temporary director will be created"
            )
            self.savepath = self.remote_fs.create_tempdir()
            logger.info(f"{self.savepath=}")

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

    # @abstractmethod
    # def setup_slurm(self):
    #     """Function setting up the slurm logic. This function creates the
    #     necessary batch file(s) for slurm to execute the task(s).

    #     Example:
    #         You want to compute DFT for N configs. In `allocate_data` you
    #         prepared N directories with the required input files on the remote
    #         machine. `setup_slurm` is supposed to create e.g. either a separate
    #         slurm batch file for each config, running them as N individual
    #         slurm jobs or a single slurm batch file that runs them as an array.
    #     """
    #     pass

    @abstractmethod
    def submit(self):
        """Function submitting the slurm job(s). This should depend on how
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
        raise NotImplementedError(
            f"finish_check() is not implemented for this task!"
        )


class SingleScriptTask(RemoteTask):
    """This is a task that consists of running a python script with some input
    file using a single slurm job. The `savepath` defines the computation
    directory on the remote machine and all output files will be written there.
    This simple implementation just copies the whole computation directory
    to the `local_savepath` if the results are collected by `collect_results`.

    Args:
        script_file:
        input_file:
        local_savepath: Path where finished results will be copied from remote
            to local machine
        remote_fs: FileSystemHandlerBase derived object. Needs to be
            initialized with valid slurm_config.
        logfile: Path to the logfile of the task.
        remote_savepath: Path on remote machine, where computation will be
            performed
        collect_only: If True, results may be overwritten if the
            `remote_savepath` already exists. If False a new savepath will be
            created with a tag to prevent overwriting.
        loop_time: The time interval which is used to check whether the job is
            completed.
        jobname: Name of the job used in slurm.
        n_processors: Number of processors that are requested per job. For CPU
            this refers to n_tasks, where as for GPU on datalab this refers to
            the number of GPU devices requested on a single node.
        jobfile_name:
    """

    def __init__(
        self,
        script_file: os.PathLike,
        input_file: os.PathLike,
        local_savepath: os.PathLike,
        remote_fs: FileSystemHandlerBase,
        logfile: os.PathLike = "task.log",
        remote_savepath: Optional[os.PathLike] = None,
        collect_only: bool = False,
        loop_time: float = 30.0,
        jobname: str = "job",
        n_processors: int = 2,
        jobfile_name: str = "run_slurm.sh",
    ):
        assert (
            remote_fs.slurm_config is not None
        ), "remote_fs needs slurm_config!"

        if remote_savepath is None:
            remote_savepath = remote_fs.slurm_config["savepath"]

        # Need to init base classs first, since it handles savepath
        super().__init__(
            logfile=logfile,
            savepath=remote_savepath,
            collect_only=collect_only,
            remote_fs=remote_fs,
        )
        self.calcdir = Path(self.savepath)

        self.script_file = script_file
        self.input_file = input_file
        self.loop_time = loop_time
        self.local_savepath = local_savepath

        # self.slurm_path = None
        self.slurm_handler = cluster_handler.SlurmJobHandler(
            jobname=jobname,
            partition=remote_fs.slurm_config["partition"],
            savepath=self.calcdir,
            conda_env=remote_fs.slurm_config["conda_env"],
            n_processors=n_processors,
            slurm_header_add="",
            remote_fs=remote_fs,
            jobfile_name=jobfile_name,
            loop_time=loop_time,
            job_type_template="body_single.sh.j2",
        )

        self.logger.print(f"SingleScriptTask")
        self.logger.print(f"calcdir={str(self.calcdir)}")

    def copy_file(self, local_path):
        moved_path = self.calcdir / os.path.split(local_path)[-1]
        self.remote_fs.put(local_path, moved_path)
        return moved_path

    def allocate_data(self):
        self.script_file_remote = self.copy_file(self.script_file)
        self.input_file_remote = self.copy_file(self.input_file)
        self.slurm_handler.write_jobfile(
            self.input_file_remote, self.script_file_remote
        )

    def submit(self):
        self.slurm_handler.submit()
        self.slurm_handler.wait_until_finished()

    def collect_results(self):
        self.remote_fs.get_dir(self.calcdir, self.local_savepath)

    def finish_check(self):
        return self.remote_fs.exists(self.calcdir / "is_finished.info")


class ArrayScriptTask(RemoteTask):
    """This is a task that consists of running the same script for a bunch of
    inputs using the slurm array functionality. The `remote_savepath` is the
    parent directory of all computatins run on the remote machine. The idea
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
        script_file: Python script that performs the computation.
        local_savepath: Path where finished results will be copied from remote
            to local machine
        remote_fs: FileSystemHandlerBase derived object. Needs to be
            initialized with valid slurm_config.
        logfile: Path to the logfile of the task.
        remote_savepath: Path on remote machine, where computation will be
            performed
        collect_only: If True, results may be overwritten if the
            `remote_savepath` already exists. If False a new savepath will be
            created with a tag to prevent overwriting.
        loop_time: The time interval which is used to check whether the job is
            completed.
        jobname: Name of the job used in slurm.
        n_processors: Number of processors that are requested per job. For CPU
            this refers to n_tasks, where as for GPU on datalab this refers to
            the number of GPU devices requested on a single node.
        jobfile_name:
    """

    def __init__(
        self,
        script_file: os.PathLike,
        local_savepath: os.PathLike,
        remote_fs: FileSystemHandlerBase,
        logfile: os.PathLike = "task.log",
        remote_savepath: Optional[os.PathLike] = None,
        collect_only: bool = False,
        loop_time: float = 30.0,
        jobname: str = "job",
        n_processors: int = 2,
        jobfile_name: str = "run_slurm.sh",
    ):
        assert (
            remote_fs.slurm_config is not None
        ), "remote_fs needs slurm_config!"

        if remote_savepath is None:
            remote_savepath = remote_fs.slurm_config["savepath"]

        # Need to init base class first, since it handles savepath
        super().__init__(
            logfile=logfile,
            savepath=remote_savepath,
            collect_only=collect_only,
            remote_fs=remote_fs,
        )
        self.calcdir = Path(self.savepath)

        self.script_file = script_file
        # self.input_file = input_file
        self.loop_time = loop_time
        self.local_savepath = local_savepath

        self.slurm_handler = cluster_handler.SlurmArrayHandler(
            jobname=jobname,
            partition=remote_fs.slurm_config["partition"],
            savepath=self.calcdir,
            conda_env=remote_fs.slurm_config["conda_env"],
            n_processors=n_processors,
            slurm_header_add="",
            remote_fs=remote_fs,
            jobfile_name=jobfile_name,
            loop_time=loop_time,
            job_type_template="body_single.sh.j2",
        )

        self.logger.print(f"SingleScriptTask")
        self.logger.print(f"calcdir={str(self.calcdir)}")

    def copy_file(self, local_path):
        moved_path = self.calcdir / os.path.split(local_path)[-1]
        self.remote_fs.put(local_path, moved_path)
        return moved_path

    def allocate_data(self):
        self.script_file_remote = self.copy_file(self.script_file)
        self.input_file_remote = self.copy_file(self.input_file)
        self.slurm_handler.write_jobfile(
            self.input_file_remote, self.script_file_remote
        )

    def submit(self):
        self.slurm_handler.submit()
        self.slurm_handler.wait_until_finished()

    def collect_results(self):
        self.remote_fs.get_dir(self.calcdir, self.local_savepath)

    def finish_check(self):
        return self.remote_fs.exists(self.calcdir / "is_finished.info")


# class ArrayScriptTask(RemoteTask):
#     """This is a task that consists of running a script with some input file.
#     """

#     def __init__(
#         self,
#         logfile: str,
#         script_file: str,
#         input_file: str,
#         remote_fs: FileSystemHandlerBase,
#         remote_savepath: Optional[str] = None,
#         collect_only: bool = False
#     ):
#         self.script_file = script_file
#         self.input_file = input_file
#         # self.size = len(inputs)
#         super().__init__(
#             logfile=logfile,
#             savepath=remote_savepath,
#             collect_only=collect_only,
#             remote_fs=remote_fs
#         )

#     def allocate_data(self):
#         """
#         """


#     def setup_slurm(self):
#         pass

#     def submit(self):
#         pass

#     def collect_results(self):
#         pass
