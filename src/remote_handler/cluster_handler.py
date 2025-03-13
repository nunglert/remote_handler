import re, os
import subprocess
from jinja2 import Environment, FileSystemLoader
import numpy as np
from importlib import resources
import importlib.resources as pkg_resources
from pathlib import Path
from abc import ABC, abstractmethod

from remote_handler import slurm_utils, file_handler
from remote_handler.file_handler import FileSystemHandlerBase


class SlurmHandlerBase(ABC):
    """Class to handle file transfers and computations via slurm on a remote
    HPC cluster.
    
    Args:
        jobname: 
        partition: 
        remote_machine: 
        savepath: Root path wherein computations are saved on remote directory.

    """

    def __init__(
        self,
        jobname,
        partition,
        savepath,
        conda_env,
        n_processors,
        slurm_header_add,
        remote_fs
    ):
        self.jobname = jobname
        self.partition = partition
        self.savepath = savepath
        self.conda_env = conda_env
        self.n_processors = n_processors
        self.slurm_header_add = slurm_header_add
        self.remote_fs = remote_fs

        if isinstance(remote_fs, file_handler.RemoteFileSystem):
            remote_machine = remote_fs.remote_alias
            self.ssh_command = f"ssh {remote_machine}"
        else:
            remote_machine = os.environ.get("REMOTE_HANDLER_LOCAL")
            if remote_machine is None:
                raise Exception(
                    "Set REMOTE_HANDLER_LOCAL environment variable to the "
                    "current system you are working on, such that SlurmHandler "
                    "can find the appropriate jinja template."
                )
            self.ssh_command = None
        self.template_file = f"{remote_machine}.sh.j2"


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
        jobfile_name: str = "run_slurm.sh"
    ):
        self.jobfile_name = jobfile_name

        self.calcdir = Path(savepath)

        self.slurm_script_path = self.calcdir / self.jobfile_name

        self.is_written = False
        self.is_submitted = False
        super().__init__(
            jobname,
            partition,
            savepath,
            conda_env,
            n_processors,
            slurm_header_add,
            remote_fs
        )


    def submit(self):
        assert self.is_written, "Can not submit job if batch file is not written yet!"
        self.pid = slurm_utils.submit_job(
            script_path=self.slurm_script_path, 
            output_path=self.calcdir, 
            ssh_command=self.ssh_command
        )
        self.is_submitted = True
    
    def write_jobfile(self, input_file, python_script_path):
        self.slurm_str = slurm_utils.get_slurm_str(
            input_file,
            python_script_path,
            self.jobname,
            self.partition,
            self.n_processors,
            self.conda_env,
            self.template_file,
            self.slurm_header_add,
        )

        self.remote_fs.write_str_to_file(
            self.slurm_str, self.slurm_script_path
        )
        self.is_written = True

    def check_status(self):
        assert self.is_submitted, "Can not check status of unsubmitted job!"
        return slurm_utils.check_job_status([self.pid])


# class SlurmArrayHandler(SlurmHandlerBase):

#     def __init__(
#         self, 
#         jobname: str, 
#         partition: str,
#         savepath: str,
#         conda_env: str,
#         n_processors: int,
#         slurm_header_add: str,
#         remote_fs: file_handler.FileSystemHandlerBase,
#     ):
#         # self.calcdir = self.

#         self.is_written = False
#         self.is_submitted = False
#         super().__init__(
#             jobname,
#             partition,
#             savepath,
#             conda_env,
#             n_processors,
#             slurm_header_add,
#             remote_fs
#         )

#     def submit(self):
#         return slurm_utils.submit_job_array(
#             script_path,
#             output_path,
#             num_array_tasks,
#             num_parallel=10,
#             ssh_command=None,
#         )
    
#     def write_jobfile(self, path):
#         slurm_str = slurm_utils.get_array_slurm_str(
#             calcdir,
#             python_script_path,
#             num_array_tasks,
#             num_parallel,
#             jobname,
#             partition,
#             n_cores,
#             conda_env,
#             map_indices,
#             template_file,
#             slurm_header_add,
#         )