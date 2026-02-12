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
from remote_handler.core.cluster_config import ClusterConfig
from remote_handler.core.file_handler import FileSystemHandlerBase

logger = logging.getLogger(__name__)


class SlurmHandlerBase(ABC):
    """
    Class to handle file transfers and computations via slurm on a remote
    HPC cluster.

    Methods to implement:
    - submit: Submit a job array to slurm.
    - write_jobfile: Write a slurm jobfile.
    - check_status: Check the status of the slurm job.


    Args:
        jobname: Name of the job used in slurm.
        cluster_config: ClusterConfig instance containing cluster-specific settings
            (partition, conda_env, template, remote_machine).
        savepath: Root path wherein computations are saved on remote directory.
        n_processors: Number of processors that are requested per job. For CPU
            this refers to n_tasks, where as for GPU on datalab this refers to
            the number of GPU devices requested on a single node.
        slurm_header_add: This allows for adding additional header lines to the
            slurm file. You may also consider creating you own jinja templates
            instead. An example would be '# SBATCH --nodelist=SOME_NODE'
        remote_fs: A FileSystemHandlerBase derived object for file operations.
        loop_time: The time interval which is used to check whether the job is
            completed.
    """

    def __init__(
        self,
        jobname: str,
        cluster_config: ClusterConfig,
        savepath: os.PathLike,
        n_processors: int,
        slurm_header_add: str,
        remote_fs: file_handler.FileSystemHandlerBase,
        loop_time: float = 30.0,
    ):
        self.jobname = jobname
        self.cluster_config = cluster_config
        self.partition = cluster_config.partition
        self.savepath = savepath
        self.conda_env = cluster_config.conda_env
        self.n_processors = n_processors
        self.slurm_header_add = slurm_header_add
        self.remote_fs = remote_fs
        self.loop_time = loop_time

        # Determine SSH command based on filesystem type
        if isinstance(remote_fs, file_handler.LocalFileSystem):
            self.ssh_command = None
        else:
            self.ssh_command = f"ssh {cluster_config.remote_machine}"

        # Get template file for the slurm scripts
        self.template_file = cluster_config.template
        self.conda_source = cluster_config.conda_source

        self.is_written = False
        self.is_submitted = False
        super().__init__()

        self._validate_conda()
        self._validate_savepath()

    def _validate_savepath(self):
        if not self.remote_fs.exists(self.savepath):
            raise Exception(
                f"Your provided {self.savepath=} does not exist on the "
                f"{self.cluster_config.remote_machine} filesystem!"
            )

    def _validate_conda(self):
        if self.conda_source.strip().endswith("miniconda3"):
            miniconda_home = Path(self.conda_source)
            self.conda_source = Path(self.conda_source) / "bin/activate"
        elif "bin/activate" in self.conda_source:
            miniconda_home = Path(self.conda_source).parents[1]
            self.conda_source = Path(self.conda_source)
        else:
            raise Exception(
                "I can not recognize your supplied conda_source "
                f"{self.conda_source}. Try sth like "
                "conda_source=/home/user/miniconda3/. Note that we only "
                "support miniconda3 at the moment."
            )

        if not self.remote_fs.exists(self.conda_source):
            raise Exception(
                f"{self.conda_source=} does not exist on the filesystem "
                f"{self.cluster_config.remote_machine}!"
            )
        if not self.remote_fs.exists(miniconda_home / "envs" / self.conda_env):
            raise Exception(
                f"The conda environment {self.conda_env} does not exist on the"
                f" {self.cluster_config.remote_machine} filesystem!"
            )

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
        """
        Wait until the job is finished by periodically checking the status.
        """
        while True:
            time.sleep(self.loop_time)
            status = self.check_status()
            # If status is not None, the job is finished
            if status:
                break


class SlurmJobHandler(SlurmHandlerBase):
    def __init__(
        self,
        jobname: str,
        cluster_config: ClusterConfig,
        savepath: str,
        n_processors: int,
        slurm_header_add: str,
        remote_fs: file_handler.FileSystemHandlerBase,
        jobfile_name: str = "run_slurm.sh",
        loop_time: float = 10.0,
        job_type_template: str = "body_single.sh.j2",
    ):
        # Name of the jobfile that will be created
        self.jobfile_name = jobfile_name
        # Template for single jobs
        self.job_type_template = job_type_template
        # Directory where the computation will be performed
        self.calcdir = Path(savepath)

        print(f"Using remote savepath: {self.calcdir}")
        # Check if in the remote machine the parent dir of 'savepath' exists
        if not remote_fs.exists(self.calcdir.parent):
            raise Exception("Parent dir of savepath needs to exist!")

        # Set the path for the slurm script
        self.slurm_script_path = self.calcdir / self.jobfile_name
        # TODO create output and calcdir directories?

        super().__init__(
            jobname,
            cluster_config,
            savepath,
            n_processors,
            slurm_header_add,
            remote_fs,
            loop_time,
        )

    def write_jobfile(
        self, input_file, python_script_path, exclude_nodes=None
    ):
        """
        Write the slurm jobfile for a single job.
        Args:
            input_file: Path to the input file that is used in the job.
            python_script_path: Path to the python script that is executed in
                the job.
        """
        self.slurm_str = slurm_utils.get_slurm_str(
            input_file,
            python_script_path,
            self.jobname,
            self.partition,
            self.n_processors,
            self.conda_env,
            self.conda_source,
            self.template_file,
            self.job_type_template,
            self.slurm_header_add,
            exclude_nodes,
        )

        self.remote_fs.write_str_to_file(
            self.slurm_str, self.slurm_script_path
        )
        logger.info("Succesfully wrote slurm file!")
        self.is_written = True

    def submit(self):
        """
        Submit the job to slurm.
        """
        # Check if the jobfile has been written
        assert (
            self.is_written
        ), "Can not submit job if batch file is not written yet!"
        # Submit the job and get the job id
        self.pid = slurm_utils.submit_job(
            script_path=self.slurm_script_path,
            output_path=self.calcdir,
            ssh_command=self.ssh_command,
            cwd=self.calcdir,
        )

        logger.info(f"Submitted slurm job with pid {self.pid}")
        self.is_submitted = True

    def check_status(self):
        """
        Check the status of the job via squeue.
        """
        assert self.is_submitted, "Can not check status of unsubmitted job!"
        logger.debug("Checking slurm job status...")
        return slurm_utils.check_job_status_squeue(
            self.pid, ssh_command=self.ssh_command
        )


class SlurmArrayHandler(SlurmHandlerBase):
    """
    Class to handle job arrays via slurm on a remote HPC cluster.

    Args:
        jobname: Name of the job used in slurm.
        cluster_config: ClusterConfig instance containing cluster-specific settings.
        savepath: Root path wherein computations are saved on remote directory.
        n_processors: Number of processors that are requested per job. For CPU
            this refers to n_tasks, where as for GPU on datalab this refers to
            the number of GPU devices requested on a single node.
        remote_fs: A FileSystemHandlerBase derived object for file operations.
        slurm_header_add: This allows for adding additional header lines to the
            slurm file. You may also consider creating you own jinja templates
            instead. An example would be '# SBATCH --nodelist=SOME_NODE'
        jobfile_name: Name of the jobfile that will be created.
        loop_time: The time interval which is used to check whether the job is
            completed.
        job_type_template: Template for array jobs.
        num_parallel: Maximum number of tasks that are submitted in parallel.
            It is the number after the '%' in the '#SBATCH --array=0-N%num_parallel'
        map_indices: Optional list of indices that are used to map the array
            job indices to specific tasks. For example if you have 100 tasks
            but only want to run tasks 0, 2, 5, you can set map_indices=[0, 2, 5].

    """

    def __init__(
        self,
        jobname: str,
        cluster_config: ClusterConfig,
        savepath: str,
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
        # Check if in the remote machine the parent dir of 'savepath' exists
        if not remote_fs.exists(self.rootpath.parent):
            raise Exception("Parent dir of savepath needs to exist!")
        # Set the path for the slurm script
        self.slurm_script_path = self.rootpath / self.jobfile_name
        # Create output and calcdir directories
        self.output_path = self.rootpath / "slurm_output"
        self.calcdir = self.rootpath / "calcdirs"
        remote_fs.makedirs(self.output_path, exist_ok=True)
        remote_fs.makedirs(self.calcdir, exist_ok=True)

        super().__init__(
            jobname,
            cluster_config,
            savepath,
            n_processors,
            slurm_header_add,
            remote_fs,
            loop_time,
        )

    def write_jobfile(self, python_script_path, num_array_tasks):
        """
        Write the slurm jobfile for the array job.
        Args:
            python_script_path: Path to the python script that is executed in
                each array job.
            num_array_tasks: Total number of array tasks.
        """
        self.num_array_tasks = num_array_tasks
        # Generate the slurm string
        self.slurm_str = slurm_utils.get_array_slurm_str(
            calcdir=self.calcdir,
            python_script_path=python_script_path,
            num_array_tasks=num_array_tasks,
            num_parallel=self.num_parallel,
            jobname=self.jobname,
            partition=self.partition,
            n_processors=self.n_processors,
            conda_env=self.conda_env,
            conda_source=self.conda_source,
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
