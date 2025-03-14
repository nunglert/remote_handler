from abc import ABC, abstractmethod
from pathlib import Path
import os, random, string
from typing import Union, Any, Optional
import time

from remote_handler import slurm_utils
from remote_handler.file_handler import (
    LocalFileSystem,
    RemoteFileSystem,
    FileSystemHandlerBase,
    tag_path
)

from remote_handler import cluster_handler


class SimpleLogger:

    def __init__(self, log_file=None):
        self.log_file = log_file

    def print(self, *args):
        # Join all arguments into a single string with spaces, like `print`
        message = " ".join(map(str, args))

        if self.log_file is not None:
            # Write the message to the specified file
            with open(self.log_file, "a") as f:
                f.write(message + "\n")
        else:
            print(message)


class RemoteTask(ABC):
    """This serves as an interface for tasks that should be performed on a 
    remote machine.

    Args:
        logfile: A log file on the host machine where output is printed
        savepath: The path on the remote system where computations should be 
            performed
    """

    def __init__(self, logfile, savepath, collect_only, remote_fs):
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
                    random_string = ''.join(
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
                self.remote_fs.makedirs(self.savepath, exist_ok=True)
        else:
            self.savepath = self.remote_fs.create_tempdir()

        super().__init__()

    @abstractmethod
    def allocate_data(self):
        """Function handling the allocation of files on the remote file
        system required for the task. 
        
        Example:
            Say you want to do e.g. DFT calculations on another machine. Then 
            this function would handle the logic of setting up the directories
            on the remote machine where the computations are supposed to be 
            performed. Furthermore, it would transfer required input files or 
            create them in-place
        """
        pass

    @abstractmethod
    def setup_slurm(self):
        """Function setting up the slurm logic. This function creates the 
        necessary batch file(s) for slurm to execute the task(s). 

        Example:
            You want to compute DFT for N configs. In `allocate_data` you 
            prepared N directories with the required input files on the remote 
            machine. `setup_slurm` is supposed to create e.g. either a separate
            slurm batch file for each config, running them as N individual 
            slurm jobs or a single slurm batch file that runs them as an array.
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
        """
        pass


class SingleScriptTask(RemoteTask):
    """This is a task that consists of running a script with some input file.
    """

    def __init__(
        self, 
        logfile: os.PathLike,
        script_file: os.PathLike,
        input_file: os.PathLike,
        remote_fs: FileSystemHandlerBase,
        remote_savepath: Optional[os.PathLike] = None,
        collect_only: bool = False,
        loop_time: float = 30.,
        jobname: str = "job",
    ):
        assert remote_fs.slurm_config is not None, "remote_fs needs slurm_config!"

        if remote_savepath is None:
            remote_savepath = remote_fs.slurm_config["savepath"]

        super().__init__(
            logfile=logfile, 
            savepath=remote_savepath,
            collect_only=collect_only,
            remote_fs=remote_fs
        )
        self.calcdir = Path(self.savepath)

        self.script_file = script_file
        self.input_file = input_file
        self.loop_time = loop_time

        self.slurm_path = None
        self.slurm_handler = cluster_handler.SlurmJobHandler(
            jobname=jobname,
            partition="GPU-a40",
            savepath=self.calcdir,
            conda_env="newjax",
            n_processors=2,
            slurm_header_add="",
            remote_fs=remote_fs,
            jobfile_name="job",
        )


    def copy_file(self, local_path):
        moved_path = self.calcdir / os.path.split(local_path)[-1]
        self.remote_fs.copy(local_path, moved_path)
        return moved_path

    def allocate_data(self):
        """
        """
        self.script_file_remote = self.copy_file(self.script_file)
        self.input_file_remote = self.copy_file(self.input_file)

    def setup_slurm(self):
        self.slurm_handler.write_jobfile(
            self.input_file_remote, self.script_file_remote
        )
    
    def submit(self):
        self.slurm_handler.submit()

    def wait_until_finished(self):
        while True:
            time.sleep(self.loop_time)
            status = self.slurm_handler.check_status()
            if status:
                break

    def collect_results(self):
        self.remote_fs.get(self.calcdir, )


class ArrayScriptTask(RemoteTask):
    """This is a task that consists of running a script with some input file.
    """

    def __init__(
        self, 
        logfile: str,
        script_file: str,
        input_file: str,
        remote_fs: FileSystemHandlerBase,
        remote_savepath: Optional[str] = None,
        collect_only: bool = False
    ):
        self.script_file = script_file
        self.input_file = input_file
        # self.size = len(inputs)
        super().__init__(
            logfile=logfile, 
            savepath=remote_savepath,
            collect_only=collect_only,
            remote_fs=remote_fs
        )

    def allocate_data(self):
        """
        """
        

    def setup_slurm(self):
        pass
    
    def submit(self):
        pass

    def collect_results(self):
        pass


    