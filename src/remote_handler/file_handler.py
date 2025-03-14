import logging
from mimetypes import init
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
import functools
import posixpath
from stat import S_ISDIR

from ase import Atoms
import ase.db
import ase.io
import numpy as np
from datetime import datetime

from abc import ABC, abstractmethod

# from remote_handler import slurm_utils, cluster_handler

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


class FileSystemHandlerBase(ABC):
    """Interface for handling file operations and directory management on a 
    file system.
    """

    def __init__(self, slurm_config_file=None):
        self.slurm_config = None
        
        if slurm_config_file is not None:
            with open(slurm_config_file, "r") as f:
                self.slurm_config = json.load(f)
        
        super().__init__()
    
    @abstractmethod
    def get(self, remote, local):
        """
        Copy a file from a remote machine to the local machine.

        Parameters:
            source (str): The path of the file or directory to copy.
            destination (str): The path where the file or directory should be copied to.
        """
        pass

    @abstractmethod
    def put(self, local, remote):
        """
        Copy a file or directory from the local machine to a remote machine.

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

    def __init__(self, slurm_config_file=None):
        super().__init__(slurm_config_file)

    def copy(self, source, destination):
        shutil.copy(source, destination)

    def put(self, source, destination):
        self.copy(source, destination)

    def get(self, destination, source):
        self.copy(destination, source)
    
    def makedirs(self, path, exist_ok=False):
        os.makedirs(path, exist_ok=exist_ok)

    def mkdir(self, path):
        os.mkdir(path)

    def rmtree(self, path, ignore_errors=True):
        shutil.rmtree(path, ignore_errors)

    def put_dir(self, src_dir, dst_dir):
        shutil.copytree(src_dir, dst_dir)

    def get_dir(self, src_dir, dst_dir):
        shutil.copytree(src_dir, dst_dir)

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




def reconnect_on_failure(func):
    """Decorator to reconnect if SSH/SFTP operation fails."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except (paramiko.SSHException, EOFError, OSError):
            print("SFTP connection lost. Reconnecting...")
            self._reconnect()
            return func(self, *args, **kwargs)
    
    return wrapper

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
        remote_alias: Optional[str] = None, 
        remote_temp_path: Optional[str] = None,
        local_temp_path: Optional[str] = None,
        slurm_config_file: Optional[str] = None
    ):
        super().__init__(slurm_config_file)
        
        if slurm_config_file is None:
            if remote_alias is None:
                raise Exception("Need either slurm_config_file or remote_alias!")
            self.remote_alias = remote_alias

        else:
            self.remote_alias = self.slurm_config["remote_machine"]

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

    def _reconnect(self):
        self._disconnect()
        self._connect()

    @reconnect_on_failure
    def put(self, local, remote):
        local, remote = str(local), str(remote)
        self.sftp_client.put(local, remote)

    @reconnect_on_failure
    def get(self, remote, local):
        remote, local = str(remote), str(local)
        self.sftp_client.get(remote, local)

    @reconnect_on_failure
    def makedirs(self, path, exist_ok=False):
        path = str(path)
        try:
            self.sftp_client.mkdir(path)
        except IOError:
            if not exist_ok:
                raise
    
    @reconnect_on_failure
    def mkdir(self, path):
        path = str(path)
        self.sftp_client.mkdir(path)

    @reconnect_on_failure
    def rmtree(self, remotepath, level=0):
        for f in self.sftp_client.listdir_attr(remotepath):
            rpath = posixpath.join(remotepath, f.filename)
            if S_ISDIR(f.st_mode):
                self.rmtree(self.sftp_client, rpath, level=(level + 1))
            else:
                rpath = posixpath.join(remotepath, f.filename)
                print('removing %s%s' % ('    ' * level, rpath))
                self.sftp_client.remove(rpath)
        print('removing %s%s' % ('    ' * level, remotepath))
        self.sftp_client.rmdir(remotepath)

    @reconnect_on_failure
    def put_dir(self, local_dir, remote_dir):
        """
        Recursively copy local directory to remote host via SFTP.

        Args:
            local_dir: Local directory path
            remote_dir: Remote target path
        """
        local_dir = Path(local_dir)
        remote_dir = str(remote_dir).rstrip('/') + '/'

        if not local_dir.is_dir():
            raise ValueError(f"Local path '{local_dir}' is not a directory.")

        if not self.exists(remote_dir):
            self.sftp_client.mkdir(remote_dir)

        for item in local_dir.iterdir():
            remote_item = remote_dir + item.name
            if item.is_dir():
                # Create remote subdirectory if doesn't exist
                if not self.exists(remote_item):
                    self.sftp_client.mkdir(remote_item)
                # Recursively upload directory
                self.put_dir(item, remote_item)
            else:
                # Upload file
                self.sftp_client.put(str(item), remote_item)
                
    @reconnect_on_failure
    def get_dir(self, remote_dir, local_dir):
        """
        Recursively copy remote directory to local host via SFTP.

        Args:
            remote_dir (str): Remote directory path
            local_dir (str or Path): Local target path
        """
        remote_dir = str(remote_dir).rstrip('/') + '/'
        local_dir = Path(local_dir)

        if not self.exists(remote_dir):
            return

        local_dir.mkdir(parents=True, exist_ok=True)

        for filename in self.sftp_client.listdir(remote_dir):
            remote_path = remote_dir + filename
            local_path = local_dir / filename

            if S_ISDIR(self.sftp_client.stat(remote_path).st_mode):
                self.get_dir(remote_path + '/', local_path)
            else:
                if not local_path.is_file():
                    self.sftp_client.get(remote_path, str(local_path))

    @reconnect_on_failure
    def write_str_to_file(self, string, filepath):
        filepath = str(filepath)
        with self.sftp_client.file(filepath, "w") as f:
            f.write(string)

    @reconnect_on_failure
    def write_dict_to_json(self, filepath, param_dict):
        filepath = str(filepath)
        with self.sftp_client.file(filepath, "w") as f:
            f.write(json.dumps(param_dict))

    @reconnect_on_failure
    def read_ase(self, filepath, **kwargs):
        filepath = str(filepath)
        local_temp = self.local_tempdir / "temp_ase_file.extxyz"
        self.sftp_client.get(filepath, local_temp)
        atoms = ase.io.read(local_temp, **kwargs)
        os.remove(local_temp)
        return atoms

    @reconnect_on_failure
    def write_ase(self, filepath, atoms_list, **kwargs):
        # print(f"Writing to {filepath}")
        filepath = str(filepath)
        local_temp = self.local_tempdir / "temp_ase_file.extxyz"
        ase.io.write(local_temp, atoms_list, **kwargs)
        self.sftp_client.put(local_temp, filepath)
        os.remove(local_temp)

    @reconnect_on_failure
    def exists(self, path):
        path = str(path)
        try:
            self.sftp_client.stat(path)
            return True
        except FileNotFoundError:
            return False
    
    @reconnect_on_failure
    def create_tempdir(self):
        """Create a temporary directory on the remote filesystem and return its path."""
        tempdir = self.tag_path(self.remote_tempdir / "dir")
        self.makedirs(tempdir, exist_ok=False)
        return tempdir
    
    @reconnect_on_failure
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

