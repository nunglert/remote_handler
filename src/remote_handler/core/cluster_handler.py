"""Cluster and SLURM job management for remote HPC systems."""

import importlib.resources as pkg_resources
import json
import logging
import os
import re
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from importlib import resources
from pathlib import Path
from typing import Optional

import numpy as np
from jinja2 import Environment, FileSystemLoader

from remote_handler.config import get_config
from remote_handler.core import file_handler, slurm_utils
from remote_handler.core.file_handler import FileSystemHandlerBase
from remote_handler.exceptions import (
    ConfigurationError,
    DependencyError,
    FileSystemError,
)

logger = logging.getLogger(__name__)


@dataclass
class ClusterConfig:
    """
    Configuration for a remote cluster with SLURM job scheduling.

    This class encapsulates all cluster-specific settings needed for
    submitting and managing SLURM jobs on a remote HPC system.

    Args:
        remote_machine: SSH alias or hostname of the remote cluster
        partition: SLURM partition name to submit jobs to
        conda_env: Name of the conda environment to activate on the cluster
        savepath: Default path on the cluster where computations are saved
        template: Name of the Jinja2 template file for SLURM job headers

    Example:
        >>> config = ClusterConfig.from_file("my_cluster.json")
        >>> print(config.remote_machine)
        'my-hpc-cluster'
    """

    remote_machine: str
    partition: str
    conda_env: str
    conda_source: str
    savepath: str
    template: str

    @classmethod
    def from_file(cls, config_file: str) -> "ClusterConfig":
        """
        Load cluster configuration from a JSON file.

        Args:
            config_file: Path to the configuration file. Can be either:
                - An absolute path to a JSON file
                - A relative filename (will be searched in REMOTE_HANDLER_CONFIG_DIR)

        Returns:
            ClusterConfig instance with loaded configuration

        Raises:
            ConfigurationError: If the config file doesn't exist or config directory not set
                               for relative paths

        Example:
            >>> # Using absolute path
            >>> config = ClusterConfig.from_file("/home/user/configs/cluster.json")
            >>> # Using relative path (requires REMOTE_HANDLER_CONFIG_DIR)
            >>> config = ClusterConfig.from_file("cluster.json")
        """
        if os.path.isabs(config_file):
            config_path = Path(config_file)
            if not config_path.exists():
                raise ConfigurationError(
                    f"Configuration file does not exist: {config_file}"
                )
        else:
            # Use centralized configuration
            config = get_config()
            config_path = config.get_config_file(config_file)

        logger.info(f"Reading ClusterConfig from {config_path}")
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        known = cls.get_known(config_dict)

        return cls(**known)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ClusterConfig":
        """
        Create a ClusterConfig from a dictionary.

        Args:
            config_dict: Dictionary containing cluster configuration

        Returns:
            ClusterConfig instance

        Example:
            >>> config = ClusterConfig.from_dict({
            ...     "remote_machine": "my-cluster",
            ...     "partition": "normal",
            ...     "conda_env": "myenv",
            ...     "savepath": "/scratch/user/jobs",
            ...     "template": "slurm_template.sh.j2"
            ... })
        """
        known = cls.get_known(config_dict)
        return cls(**known)

    @classmethod
    def get_known(cls, args_dict):
        field_names = {f.name for f in fields(cls)}

        # Split known / unknown keys
        known = {}
        unknown = {}
        for k, v in args_dict.items():
            if k in field_names:
                known[k] = v
            else:
                unknown[k] = v

        if unknown:
            logger.warning(
                f"Ignoring unknown ClusterConfig keys: {list(unknown.keys())}"
            )

        return known


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

        self._validate_slurm()
        self._validate_conda()
        self._validate_savepath()

    def _validate_slurm(self):
        """Check that slurm (sbatch) is available on the target system."""
        cmd = "which sbatch"
        if self.ssh_command is not None:
            cmd = f"{self.ssh_command} '{cmd}'"
        try:
            subprocess.run(
                cmd, shell=True, capture_output=True, text=True, check=True
            )
        except subprocess.CalledProcessError:
            target = (
                "local machine"
                if self.ssh_command is None
                else self.cluster_config.remote_machine
            )
            raise DependencyError(
                f"SLURM is not available on {target}: 'sbatch' not found.\n"
                "Ensure SLURM is installed and accessible in PATH."
            )

    def _validate_savepath(self):
        if not self.remote_fs.exists(self.savepath):
            raise FileSystemError(
                f"Save path does not exist on remote filesystem: {self.savepath}\n"
                f"Remote machine: {self.cluster_config.remote_machine}"
            )

    def _validate_conda(self):
        if self.conda_source.strip().endswith("miniconda3"):
            miniconda_home = Path(self.conda_source)
            self.conda_source = Path(self.conda_source) / "bin/activate"
        elif "micromamba" in self.conda_source:
            miniconda_home = self.conda_source
            self.conda_source = 'eval "$(micromamba shell hook --shell bash)"' # run `eval "$(micromamba shell hook --shell bash)"` instead in the config, no path should be needed
        elif "bin/activate" in self.conda_source:
            miniconda_home = Path(self.conda_source).parents[1]
            self.conda_source = Path(self.conda_source)
        else:
            raise ConfigurationError(
                f"Invalid conda_source format: {self.conda_source}\n"
                "Expected format: /home/user/miniconda3/ or /path/to/bin/activate\n"
                "Note: Only miniconda3 is currently supported"
            )
        if "micromamba" not in str(self.conda_source):
            if not self.remote_fs.exists(self.conda_source):
                raise FileSystemError(
                    f"Conda activation script not found at: {self.conda_source}. "
                    "We currently do not support '~', write the full path!\n"
                    f"Remote machine: {self.cluster_config.remote_machine}"
                )
            if not self.remote_fs.exists(miniconda_home / "envs" / self.conda_env):
                raise ConfigurationError(
                    f"Conda environment '{self.conda_env}' not found on "
                    f"{self.cluster_config.remote_machine}\n"
                    f"Expected path: {miniconda_home / 'envs' / self.conda_env}"
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
            raise FileSystemError(
                f"Parent directory of savepath does not exist: {self.calcdir.parent}"
            )

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
            raise FileSystemError(
                f"Parent directory of savepath does not exist: {self.rootpath.parent}"
            )
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
