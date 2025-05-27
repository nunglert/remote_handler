import logging
import os
import random
import string
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union
from xml.etree.ElementPath import prepare_star

from remote_handler.core import cluster_handler, file_handler, slurm_utils
from remote_handler.core.file_handler import (
    FileSystemHandlerBase,
    LocalFileSystem,
    RemoteFileSystem,
    tag_path,
)
from remote_handler.core.utils import SimpleLogger

logger = logging.getLogger(__name__)


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
            performed. Will try to use temporary dir if None.
        remote_fs: A FileSystemHandlerBase derived object.
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
        collect_only: bool = False,
    ):
        assert not "/" in str(logfile), "logfile should only be filename!"

        self.local_fs = LocalFileSystem()
        self.remote_fs = remote_fs

        self.savepath = savepath
        self.collect_only = collect_only

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
            performed. Will try to use temporary dir if None
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
        job_type_template: os.PathLike = "body_single.sh.j2",
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
            job_type_template=job_type_template,
        )

        logger.info(f"SingleScriptTask")
        logger.info(f"calcdir: {str(self.calcdir)}")

    def copy_file(self, local_path):
        moved_path = self.calcdir / os.path.split(local_path)[-1]
        self.remote_fs.put(local_path, moved_path)
        return moved_path

    def allocate_data(self):
        if self.script_file is not None:
            self.script_file_remote = self.copy_file(self.script_file)
        else:
            self.script_file_remote = None

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

    def run(self):
        self.allocate_data()
        self.submit()
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
        local_savepath: os.PathLike,
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
        if remote_fs.slurm_config is None:
            raise Exception("remote_fs needs slurm_config!")

        if individual_ressources:
            self.num_array_tasks = len(individual_ressources[0])

        if remote_savepath is None:
            logger.info("Falling back to savepath from your slurm_config_file")
            remote_savepath = remote_fs.slurm_config["savepath"]

        # Need to init base class first, since it handles savepath
        super().__init__(
            logfile=logfile,
            savepath=remote_savepath,
            collect_only=collect_only,
            remote_fs=remote_fs,
        )
        self.rootpath = Path(self.savepath)

        self.python_script_file = python_script_file
        self.individual_ressources = individual_ressources
        self.shared_ressources = shared_ressources
        self.loop_time = loop_time
        self.local_savepath = local_savepath

        self.slurm_handler = cluster_handler.SlurmArrayHandler(
            jobname=jobname,
            partition=remote_fs.slurm_config["partition"],
            savepath=self.rootpath,
            conda_env=remote_fs.slurm_config["conda_env"],
            n_processors=n_processors,
            map_indices=None,
            slurm_header_add=slurm_header_add,
            remote_fs=remote_fs,
            jobfile_name=jobfile_name,
            loop_time=loop_time,
            num_parallel=num_parallel,
            job_type_template="body_array.sh.j2",
        )

        self.calcdir = self.slurm_handler.calcdir

        logger.info(f"ArrayScriptTask")
        logger.info(f"rootpath: {str(self.rootpath)}")

    def copy_file(self, local_path):
        moved_path = self.rootpath / os.path.split(local_path)[-1]
        self.remote_fs.put(local_path, moved_path)
        return moved_path

    def allocate_data(self, slurm_only=False):
        assert (
            self.python_script_file is not None
        ), "Need self.python_script_file"
        assert os.path.exists(
            self.python_script_file
        ), "Did not find python script!"
        assert (
            len(self.individual_ressources) > 0
        ), "Specify some inputs for your script!"
        for content in self.individual_ressources:
            assert len(content) == len(
                self.individual_ressources[0]
            ), "args need to have same length!"

        self.script_file_remote = self.copy_file(self.python_script_file)
        self.slurm_handler.write_jobfile(
            self.script_file_remote, self.num_array_tasks
        )

        if not slurm_only:
            if self.shared_ressources:
                for ressource in self.shared_ressources:
                    self.copy_file(ressource)

            self.all_calcpaths = []
            for idx, data_at_idx in enumerate(
                zip(*self.individual_ressources)
            ):
                calcpath = self.calcdir / f"{idx}"
                self.remote_fs.makedirs(calcpath)
                for data in data_at_idx:
                    data.write(calcpath)
                self.all_calcpaths.append(calcpath)

    def submit(self):
        self.slurm_handler.submit()
        self.slurm_handler.wait_until_finished()

    def collect_results(self):
        self.remote_fs.get_dir(self.rootpath, self.local_savepath)

    def finish_check(self):
        assert self.all_calcpaths, "There are no calcpaths to check!"

        for calcpath in self.all_calcpaths:
            if not self.remote_fs.exists(Path(calcpath) / "is_finished.info"):
                return False

        return True
