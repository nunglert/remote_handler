import importlib.resources as pkg_resources
import logging
import os
import re
import subprocess
import time
from abc import ABC, abstractmethod
from importlib import resources
from pathlib import Path
from typing import Optional

import numpy as np
from jinja2 import Environment, FileSystemLoader

from remote_handler.core import file_handler, slurm_utils
from remote_handler.core.file_handler import FileSystemHandlerBase

logger = logging.getLogger(__name__)


class SlurmHandlerBase(ABC):
    """Class to handle file transfers and computations via slurm on a remote
    HPC cluster.

    Args:
        jobname: Name of the job used in slurm.
        partition: Name of the partition on the HPC setup.
        remote_machine: Name of remote machine (Needs to have a host entry in
            your .ssh/ config)
        savepath: Root path wherein computations are saved on remote directory.
        conda_env: Conda environment that you want to use on the remote machine
            in your slurm batch job.
        n_processors: Number of processors that are requested per job. For CPU
            this refers to n_tasks, where as for GPU on datalab this refers to
            the number of GPU devices requested on a single node.
        slurm_header_add: This allows for adding additional header lines to the
            slurm file. You may also consider creating you own jinja templates
            instead.An examples would be '# SBATCH --nodelist=SOME_NODE'
        remote_fs: A FileSystemHandlerBase derived object loaded with a valid
            `slurm_config`.
        loop_time: The time interval which is used to check whether the job is
            completed.
    """

    def __init__(
        self,
        jobname: str,
        partition: str,
        savepath: os.PathLike,
        conda_env: str,
        n_processors: int,
        slurm_header_add: str,
        remote_fs: file_handler.FileSystemHandlerBase,
        loop_time: float = 30.0,
    ):
        self.jobname = jobname
        self.partition = partition
        self.savepath = savepath
        self.conda_env = conda_env
        self.n_processors = n_processors
        self.slurm_header_add = slurm_header_add
        self.remote_fs = remote_fs
        self.loop_time = loop_time

        assert (
            remote_fs.slurm_config is not None
        ), "Need slurm_config set in remote_fs!"
        remote_machine = remote_fs.slurm_config["remote_machine"]
        self.ssh_command = f"ssh {remote_machine}"
        self.template_file = remote_fs.slurm_config["template"]

        self.is_written = False
        self.is_submitted = False
        super().__init__()

    @abstractmethod
    def submit(self):
        """
        Submit a job array to slurm.
        """
        pass

    @abstractmethod
    def write_jobfile(self):
        """
        Write a slurm jobfile.
        """
        pass

    @abstractmethod
    def check_status(self):
        """
        Check the status of the slurm job.
        """
        pass

    def wait_until_finished(self):
        while True:
            time.sleep(self.loop_time)
            status = self.check_status()
            if status:
                break


class SlurmJobHandler(SlurmHandlerBase):
    def __init__(
        self,
        jobname: str,
        partition: str,
        savepath: str,
        conda_env: str,
        n_processors: int,
        slurm_header_add: str,
        remote_fs: file_handler.FileSystemHandlerBase,
        jobfile_name: str = "run_slurm.sh",
        loop_time: float = 10.0,
        job_type_template: str = "body_single.sh.j2",
    ):
        self.jobfile_name = jobfile_name
        self.job_type_template = job_type_template

        self.calcdir = Path(savepath)
        if not remote_fs.exists(self.calcdir.parent):
            raise Exception("Parent dir of savepath needs to exist!")

        self.slurm_script_path = self.calcdir / self.jobfile_name

        super().__init__(
            jobname,
            partition,
            savepath,
            conda_env,
            n_processors,
            slurm_header_add,
            remote_fs,
            loop_time,
        )

    def write_jobfile(self, input_file, python_script_path):
        self.slurm_str = slurm_utils.get_slurm_str(
            input_file,
            python_script_path,
            self.jobname,
            self.partition,
            self.n_processors,
            self.conda_env,
            self.template_file,
            self.job_type_template,
            self.slurm_header_add,
        )

        self.remote_fs.write_str_to_file(
            self.slurm_str, self.slurm_script_path
        )
        logger.info("Succesfully wrote slurm file!")
        self.is_written = True

    def submit(self):
        assert (
            self.is_written
        ), "Can not submit job if batch file is not written yet!"
        self.pid = slurm_utils.submit_job(
            script_path=self.slurm_script_path,
            output_path=self.calcdir,
            ssh_command=self.ssh_command,
            cwd=self.calcdir,
        )
        logger.info(f"Submitted slurm job with pid {self.pid}")
        self.is_submitted = True

    def check_status(self):
        assert self.is_submitted, "Can not check status of unsubmitted job!"
        logger.debug("Checking slurm job status...")
        return slurm_utils.check_job_status_squeue(
            self.pid, ssh_command=self.ssh_command
        )


class SlurmArrayHandler(SlurmHandlerBase):
    def __init__(
        self,
        jobname: str,
        partition: str,
        savepath: str,
        conda_env: str,
        n_processors: int,
        remote_fs: file_handler.FileSystemHandlerBase,
        slurm_header_add: str = "",
        jobfile_name: str = "run_slurm.sh",
        loop_time: float = 10.0,
        job_type_template: str = "body_array.sh.j2",
        num_parallel: int = 10,
        map_indices: Optional[list[int]] = None,
    ):
        self.jobfile_name = jobfile_name
        self.job_type_template = job_type_template
        self.num_parallel = num_parallel
        self.map_indices = map_indices

        self.rootpath = Path(savepath)
        if not remote_fs.exists(self.rootpath.parent):
            raise Exception("Parent dir of savepath needs to exist!")

        self.slurm_script_path = self.rootpath / self.jobfile_name
        self.output_path = self.rootpath / "slurm_output"
        self.calcdir = self.rootpath / "calcdirs"
        remote_fs.makedirs(self.output_path, exist_ok=True)
        remote_fs.makedirs(self.calcdir, exist_ok=True)

        super().__init__(
            jobname,
            partition,
            savepath,
            conda_env,
            n_processors,
            slurm_header_add,
            remote_fs,
            loop_time,
        )

    def write_jobfile(self, python_script_path, num_array_tasks):
        self.num_array_tasks = num_array_tasks
        self.slurm_str = slurm_utils.get_array_slurm_str(
            calcdir=self.calcdir,
            python_script_path=python_script_path,
            num_array_tasks=num_array_tasks,
            num_parallel=self.num_parallel,
            jobname=self.jobname,
            partition=self.partition,
            n_processors=self.n_processors,
            conda_env=self.conda_env,
            map_indices=self.map_indices,
            header_template=self.template_file,
            job_type_template=self.job_type_template,
            slurm_header_add=self.slurm_header_add,
        )
        self.remote_fs.write_str_to_file(
            self.slurm_str, self.slurm_script_path
        )
        logger.info("Succesfully wrote slurm file!")

        self.is_written = True

    def submit(self):
        assert (
            self.is_written
        ), "Can not submit job if batch file is not written yet!"
        self.pid = slurm_utils.submit_job_array(
            script_path=self.slurm_script_path,
            output_path=self.output_path,
            ssh_command=self.ssh_command,
            cwd=self.calcdir,
            num_array_tasks=self.num_array_tasks,
            num_parallel=self.num_parallel,
        )
        logger.info(f"Submitted slurm job with pid {self.pid}")
        self.is_submitted = True

    def check_status(self):
        assert self.is_submitted, "Can not check status of unsubmitted job!"
        logger.debug("Checking slurm job status...")
        return slurm_utils.check_job_status_squeue(
            self.pid, ssh_command=self.ssh_command
        )
