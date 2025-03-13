import logging
import os
import shutil
import math
import tempfile
import time
from itertools import groupby
from pathlib import Path
from re import T
from typing import Union, Any, Optional
import json
import paramiko
import random
import string

from ase import Atoms
import ase.db
import ase.io
import numpy as np
from datetime import datetime

from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

def tag_path(part0, part1, is_directory=False):
    date_str = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    if not is_directory:
        file_parts = part1.split(".")
        new_filename = (
            "".join(file_parts[:-1]) + "_" + date_str + "." + file_parts[-1]
        )
    else:
        new_filename = part1 + "_" + date_str

    save_file = Path(part0) / new_filename
    return save_file


def get_slurm_str(
    calcdir,
    python_script_path,
    num_array_tasks=1,
    num_parallel=10,
    jobname="",
    partition="normal",
    n_cores=8,
    conda_env="jaxnest",
    map_indices=None
):
    if map_indices is None:
        map_indices = np.arange(num_array_tasks)
    
    map_str = "" 
    for i in map_indices:
        map_str += f"{i} "
        
    slurm_str = f"""#!/bin/sh
#SBATCH --job-name={jobname}   # Job name
#SBATCH --ntasks={n_cores}          # Run a single task
# SBATCH --mem=0                      # Memory per processor
# SBATCH --time=3:0:0
#SBATCH --partition={partition}
# SBATCH --array=1-{num_array_tasks}%{num_parallel}          # Array range
ulimit -u 40000
ulimit -s unlimited

pwd; hostname; date

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate {conda_env}

# Print the task and run range
echo This is task $SLURM_ARRAY_TASK_ID

MAP=({map_str})

# Get the mapped index for this SLURM array task
MAPPED_INDEX=${{MAP[$SLURM_ARRAY_TASK_ID]}}

PER_TASK=1

# Calculate the starting and ending values for this task based
# on the SLURM task and the number of runs per task.
START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PER_TASK + 1 ))
END_NUM=$(( $SLURM_ARRAY_TASK_ID * $PER_TASK))

# Print the task and run range
echo This is task $SLURM_ARRAY_TASK_ID, which will do runs $START_NUM to $END_NUM

# Run the loop of runs for this task.
for (( run=$START_NUM; run<=END_NUM; run++ )); do
  echo This is SLURM task $SLURM_ARRAY_TASK_ID, run number $run
  run_id=$((run - 1))
  python {python_script_path} {calcdir}/$run_id
done
"""
    return slurm_str


class FileSystemHandlerBase(ABC):
    """Interface for handling file operations and directory management on a 
    file system.
    """
    
    @abstractmethod
    def copy(self, source, destination):
        """
        Copy a file or directory from the source path to the destination path.

        Parameters:
            source (str): The path of the file or directory to copy.
            destination (str): The path where the file or directory should be copied to.
        """
        pass

    @abstractmethod
    def makedirs(self, path, exist_ok=False):
        """
        Create a directory at the specified path, including any necessary
        parent directories.

        Parameters:
            path (str): The directory path to create.
            exist_ok (bool, optional): If True, do not raise an error if the
                directory already exists. Defaults to False.
        """
        pass

    @abstractmethod
    def mkdir(self, path):
        """
        Create a directory at the specified path.

        Parameters:
            path (str): The directory path to create.
        """
        pass

    @abstractmethod
    def write_str_to_file(self, string, filepath):
        """
        Write a string to a file at the specified file path.

        Parameters:
            string (str): The string content to write to the file.
            filepath (str): The path of the file where the string should be written.
        """
        pass

    @abstractmethod
    def write_dict_to_json(self, filepath, param_dict):
        """
        Write a dictionary to a JSON file at the specified file path.

        Parameters:
            filepath (str): The path of the file where the dictionary should be written.
            param_dict (dict): The dictionary to serialize and write to the JSON file.
        """
        pass

    @abstractmethod
    def read_ase(self, filepath, **kwargs):
        """
        Read an ASE Atoms object from a file at the specified file path.

        Parameters:
            filepath (str): The path of the file to read the ASE Atoms object from.
            **kwargs: Additional keyword arguments to customize the reading process.

        Returns:
            Atoms: An ASE Atoms object or a list of them read from the file.
        """
        pass

    @abstractmethod
    def write_ase(self, filepath, atoms_list, **kwargs):
        """
        Write a list of ASE Atoms objects to a file at the specified file path.

        Parameters:
            filepath (str): The path of the file where the ASE Atoms objects should be written.
            atoms_list (list of Atoms): A list of ASE Atoms objects to write to the file.
            **kwargs: Additional keyword arguments to customize the writing process.
        """
        pass

    @abstractmethod
    def exists(self, path):
        """
        Check if a file or directory exists at the specified path.

        Parameters:
            path (str): The path to check for existence.

        Returns:
            bool: True if the path exists, False otherwise.
        """
        pass

    @abstractmethod
    def create_tempdir(self):
        pass

    @abstractmethod
    def remove(self, path):
        """
        Remove a file at the specified path.

        Parameters:
            path (str): The path of the file to remove.
        """
        pass


    def tag_path(self, savepath):
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        working_dir, filename = os.path.split(savepath)
        savetag = tag_path(working_dir, filename, is_directory=True)
        return Path(str(savetag) + random_string)

    def secure_path(self, path, is_directory=False):
        if is_directory:  # delete trailing slash
            path = str(Path(path))

        parentpath, childname = os.path.split(path)
        if self.exists(path):
            path = tag_path(
                parentpath, childname, is_directory
            )
        
        return path

class LocalFileSystem(FileSystemHandlerBase):

    def copy(self, source, destination):
        shutil.copy(source, destination)

    def makedirs(self, path, exist_ok=False):
        os.makedirs(path, exist_ok=exist_ok)

    def mkdir(self, path):
        os.mkdir(path)

    def write_str_to_file(self, string, filepath):
        with open(filepath, "w") as f:
            f.write(string)

    def write_dict_to_json(self, filepath, param_dict):
        with open(filepath, "w") as f:
            json.dump(param_dict, f)

    def read_ase(self, filepath, **kwargs):
        return ase.io.read(filepath, **kwargs)
    
    def write_ase(self, filepath, atoms_list, **kwargs):
        return ase.io.write(filepath, atoms_list, **kwargs)

    def exists(self, path):
        return os.path.exists(path)
    
    def create_tempdir(self):
        """Create a temporary directory on the local filesystem and return its path."""
        return tempfile.mkdtemp()

    def remove(self, path):
        """
        Remove a file at the specified path.

        Parameters:
            path (str): The path of the file to remove.
        """
        os.remove(path)

class RemoteFileSystem(FileSystemHandlerBase):
    """
    RemoteFileSystem is a concrete implementation of FileSystemHandlerBase for
    handling file operations on remote systems via SSH and SFTP.

    Attributes:
        remote_alias (str): The alias for the remote host as defined in the SSH config.
        ssh_client (paramiko.SSHClient): The SSH client for managing connections.
        sftp_client (paramiko.SFTPClient): The SFTP client for file operations.
    """
    def __init__(
        self, 
        remote_alias: str, 
        remote_temp_path: Optional[str] = None,
        local_temp_path: Optional[str] = None
    ):
        self.remote_alias = remote_alias
        self.ssh_client = None
        self.sftp_client = None

        self._connect()

        if remote_temp_path is None:
            self.remote_tempdir = Path("/tmp")
        else:
            self.remote_tempdir = Path(remote_temp_path)
            if not self.exists(self.remote_tempdir):
                raise Exception(
                    "`remote_temp_path` must already exist!"
                )

        # local_tempdir is only required to temporarily create extxyz files
        # for reading and writing ase objects and succesively move them via 
        # sftp
        if local_temp_path is None:
            local_temp_path = Path("/tmp")
        else:
            if not os.path.exists(local_temp_path):
                raise Exception(
                    "`local_temp_path` must already exist!"
                )
            
        self.local_tempdir = self.tag_path(Path(local_temp_path) / "ase_temp")
        os.mkdir(self.local_tempdir)


    def _connect(self):
        ssh_config_file = os.path.expanduser("~/.ssh/config")
        ssh_config = paramiko.SSHConfig()
        with open(ssh_config_file) as f:
            ssh_config.parse(f)

        # Lookup the alias in SSH config
        host_config = ssh_config.lookup(self.remote_alias)

        hostname = host_config.get("hostname")
        username = host_config.get("user")
        identityfile = host_config.get("identityfile", [None])[0]  # Get the first IdentityFile if available

        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        self.ssh_client.connect(
            hostname,
            username=username,
            key_filename=identityfile,
        )
        self.sftp_client = self.ssh_client.open_sftp()

    def _disconnect(self):
        if self.sftp_client:
            self.sftp_client.close()
        if self.ssh_client:
            self.ssh_client.close()

    def copy(self, source, destination):
        source = str(source)
        destination = str(destination)
        self.sftp_client.put(source, destination)

    def makedirs(self, path, exist_ok=False):
        path = str(path)
        try:
            self.sftp_client.mkdir(path)
        except IOError:
            if not exist_ok:
                raise

    def mkdir(self, path):
        path = str(path)
        self.sftp_client.mkdir(path)

    def write_str_to_file(self, string, filepath):
        filepath = str(filepath)
        with self.sftp_client.file(filepath, "w") as f:
            f.write(string)

    def write_dict_to_json(self, filepath, param_dict):
        filepath = str(filepath)
        with self.sftp_client.file(filepath, "w") as f:
            f.write(json.dumps(param_dict))

    def read_ase(self, filepath, **kwargs):
        filepath = str(filepath)
        local_temp = self.local_tempdir / "temp_ase_file.extxyz"
        self.sftp_client.get(filepath, local_temp)
        atoms = ase.io.read(local_temp, **kwargs)
        os.remove(local_temp)
        return atoms

    def write_ase(self, filepath, atoms_list, **kwargs):
        # print(f"Writing to {filepath}")
        filepath = str(filepath)
        local_temp = self.local_tempdir / "temp_ase_file.extxyz"
        ase.io.write(local_temp, atoms_list, **kwargs)
        self.sftp_client.put(local_temp, filepath)
        os.remove(local_temp)

    def exists(self, path):
        path = str(path)
        try:
            self.sftp_client.stat(path)
            return True
        except FileNotFoundError:
            return False
        
    def create_tempdir(self, dirname="dir"):
        """Create a temporary directory on the remote filesystem and return its path."""
        tempdir = self.tag_path(self.remote_tempdir / dirname)
        self.makedirs(tempdir, exist_ok=False)
        return tempdir
    
    def remove(self, path):
        """
        Remove a file at the specified path on the remote system.

        Parameters:
            path (str): The path of the file to remove.
        """
        path = str(path)
        try:
            self.sftp_client.remove(path)
        except IOError as e:
            raise Exception(f"Failed to remove remote file: {path}") from e

