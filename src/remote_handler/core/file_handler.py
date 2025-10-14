import functools
import json
import logging
import math
import os
import posixpath
import random
import shutil
import string
import tempfile
import time
from abc import ABC, abstractmethod
from datetime import datetime

from pathlib import Path

from stat import S_ISDIR
from typing import Optional

import ase.io

import paramiko
from ase import Atoms

# Create a logger with the module name
logger = logging.getLogger(__name__)

# Load directory where config files for slurm are stored from environment variable
CONFIG_DIR = Path(os.environ.get("REMOTE_HANDLER_CONFIG_DIR"))


def tag_path(part0: str, part1: str, is_directory: bool = False):
    """
    Append a timestamp to a filename or directory name to create a unique path.
    Parameters:
        part0 (str): The directory path where the file or directory is located.
        part1 (str): The filename or directory name to which the timestamp will be appended.
        is_directory (bool, optional): If True, indicates that part1 is a directory name.
            Defaults to False.
        Returns:
            str: The new path with the timestamp appended to the filename or directory name.
    """
    # Get the current date and time
    date_str = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    
    if not is_directory:
        file_parts = part1.split(".")
        # If there's an extension, insert the date before it
        new_filename = (
            "".join(file_parts[:-1]) + "_" + date_str + "." + file_parts[-1]
        )
    else:
        # If it's a directory, just append the date
        new_filename = part1 + "_" + date_str

    save_file = Path(part0) / new_filename
    return save_file


class FileSystemHandlerBase(ABC):
    """
    Interface for handling file operations and directory management on a
    file system.
    Args:
        slurm_config_file (str, optional): Path to a SLURM configuration file.
            If provided, the configuration will be loaded and stored in the
            `slurm_config` attribute. Defaults to None.
    Attributes:
        slurm_config (dict or None): The loaded SLURM configuration if a
            configuration file is provided, otherwise None.
    Methods:
        get(remote, local): Abstract method to copy a file from a remote
            machine to the local machine.
        put(local, remote): Abstract method to copy a file or directory from
            the local machine to a remote machine.
        makedirs(path, exist_ok=False): Abstract method to create a directory
            at the specified path, including any necessary parent directories.
        mkdir(path): Abstract method to create a directory at the specified
            path.       
    """

    def __init__(self, slurm_config_file: Optional[str] = None):
        self.slurm_config = None

        if slurm_config_file is not None:
            if CONFIG_DIR is None:
                raise Exception(
                    "Set REMOTE_HANDLER_CONFIG_DIR environment variable, "
                    "where config files for different clusters are stored. See "
                    "e.g. in examples/config_files!"
                )
            slurm_config_path = CONFIG_DIR / slurm_config_file
            if not os.path.exists(slurm_config_path):
                raise Exception(
                    f"{slurm_config_file} does not exist in your "
                    f"REMOTE_HANDLER_CONFIG_DIR={CONFIG_DIR}"
                )

            with open(slurm_config_path, "r") as f:
                self.slurm_config = json.load(f)

        super().__init__()

    @abstractmethod
    def get(self, remote: os.PathLike, local: os.PathLike) -> None:
        """
        Copy a file from a remote machine to the local machine.

        Args:
            remote (str): The path of the file or directory to copy.
            local (str): The path where the file or directory should be copied to.
        """
        pass

    @abstractmethod
    def put(self, local: os.PathLike, remote: os.PathLike) -> None:
        """
        Copy a file or directory from the local machine to a remote machine.

        Args:
            local (str): The path of the file or directory to copy.
            remote (str): The path where the file or directory should be copied to. 
        """
        pass

    @abstractmethod
    def makedirs(self, path: os.PathLike, exist_ok: bool = False) -> None:
        """
        Create a directory at the specified path, including any necessary
        parent directories.

        Args:
            path (str): The directory path to create.
            exist_ok (bool, optional): If True, do not raise an error if the
                directory already exists. Defaults to False.
        """
        pass

    @abstractmethod
    def mkdir(self, path: os.PathLike) -> None:
        """
        Create a directory at the specified path.

        Args:
            path (str): The directory path to create.
        """
        pass

    @abstractmethod
    def write_str_to_file(self, string: str, filepath: os.PathLike) -> None:
        """
        Write a string to a file at the specified file path.

        Args:
            string (str): The string content to write to the file.
            filepath (str): The path of the file where the string should be written.
        """
        pass

    @abstractmethod
    def write_dict_to_json(
        self, filepath: os.PathLike, param_dict: dict
    ) -> None:
        """
        Write a dictionary to a JSON file at the specified file path.

        Args:
            filepath (str): The path of the file where the dictionary should be written.
            param_dict (dict): The dictionary to serialize and write to the JSON file.
        """
        pass

    @abstractmethod
    def read_ase(self, filepath: os.PathLike, **kwargs) -> list[Atoms]:
        """
        Read an ASE Atoms object from a file at the specified file path.

        Args:
            filepath (str): The path of the file to read the ASE Atoms object from.
            **kwargs: Additional keyword arguments to customize the reading process.

        Returns:
            Atoms: An ASE Atoms object or a list of them read from the file.
        """
        pass

    @abstractmethod
    def write_ase(
        self, filepath: os.PathLike, atoms_list: list[Atoms], **kwargs
    ) -> None:
        """
        Write a list of ASE Atoms objects to a file at the specified file path.

        Args:
            filepath (str): The path of the file where the ASE Atoms objects should be written.
            atoms_list (list of Atoms): A list of ASE Atoms objects to write to the file.
            **kwargs: Additional keyword arguments to customize the writing process.
        """
        pass

    @abstractmethod
    def exists(self, path: os.PathLike) -> bool:
        """
        Check if a file or directory exists at the specified path.

        Args:
            path (str): The path to check for existence.

        Returns:
            bool: True if the path exists, False otherwise.
        """
        pass

    @abstractmethod
    def create_tempdir(self) -> os.PathLike:
        """Create a temporary directory and return its path."""
        pass

    @abstractmethod
    def remove(self, path: os.PathLike) -> None:
        """
        Remove a file at the specified path.

        Args:
            path (str): The path of the file to remove.
        """
        pass

    def tag_path(self, savepath: os.PathLike) -> os.PathLike:
        """
        Append a random string to a filename or directory name to create a unique path.
        Parameters:
            savepath (str): The full path where the file or directory is located.
        Returns:
            str: The new path with the random string appended to the filename or directory name.    
        """

        # Generate a random string of fixed length
        random_string = "".join(
            random.choices(string.ascii_letters + string.digits, k=10)
        ) 
        # Append a timestamp to ensure uniqueness
        working_dir, filename = os.path.split(savepath)
        savetag = tag_path(working_dir, filename, is_directory=True)
        return Path(str(savetag) + random_string)

    def secure_path(
        self, path: os.PathLike, is_directory: bool = False
    ) -> os.PathLike:
        """
        Ensure that the provided path is unique by appending a timestamp if it
        already exists.
        Args:
            path (str): The original file or directory path.
            is_directory (bool, optional): If True, indicates that the path is a
                directory. Defaults to False.
        Returns:
            str: A unique file or directory path.
        """
        # Delete the last slash if present if is_directory is True
        if is_directory:  
            path = str(Path(path))

        parentpath, childname = os.path.split(path)
        # If the path already exists, append a timestamp to make it unique
        if self.exists(path):
            path = tag_path(parentpath, childname, is_directory)

        return path


class LocalFileSystem(FileSystemHandlerBase):
    """
    LocalFileSystem is a concrete implementation of FileSystemHandlerBase for
    handling file operations on the local filesystem.
    Args:
        slurm_config_file (str, optional): Path to a SLURM configuration file.
            If provided, the configuration will be loaded and stored in the
            `slurm_config` attribute. Defaults to None.
    """

    def __init__(self, slurm_config_file=None):
        super().__init__(slurm_config_file)

    def copy(self, source: os.PathLike, destination: os.PathLike) -> None:
        """
        Copy a file from source to destination on the local filesystem.
        Args:
            source (str): The path of the file to copy.
            destination (str): The path where the file should be copied.
        """
        shutil.copy(source, destination)

    def put(self, source: os.PathLike, destination: os.PathLike) -> None:
        """
        Copy a file from source to destination on the local filesystem.
        Args:
            source (str): The path of the file to copy.
            destination (str): The path where the file should be copied.
        """
        self.copy(source, destination)

    def get(self, destination: os.PathLike, source: os.PathLike) -> None:
        """
        Copy a file from source to destination on the local filesystem.
        Args:
            source (str): The path of the file to copy.
            destination (str): The path where the file should be copied.
        """
        self.copy(destination, source)

    def makedirs(self, path: os.PathLike, exist_ok=False) -> None:
        """
        Create a directory at the specified path, including any necessary
        parent directories.
        Args:
            path (str): The directory path to create.
            exist_ok (bool, optional): If True, do not raise an error if the
                directory already exists. Defaults to False.
        """
        os.makedirs(path, exist_ok=exist_ok)

    def mkdir(self, path: os.PathLike) -> None:
        """
        Create a directory at the specified path.
        Args:
            path (str): The directory path to create.
        """
        os.mkdir(path)

    def rmtree(self, path: os.PathLike, ignore_errors=True) -> None:
        """Recursively delete a directory on the local filesystem.
        Args:
            path (str): The path of the directory to delete.
            ignore_errors (bool, optional): If True, ignore errors during
                deletion. Defaults to True.
        """
        shutil.rmtree(path, ignore_errors)

    def put_dir(self, src_dir: os.PathLike, dst_dir: os.PathLike) -> None:
        """
        Recursively copy local directory to another location on the local filesystem.
        Args:
            src_dir (str): Local directory path
            dst_dir (str): Local target path
        """
        shutil.copytree(src_dir, dst_dir)

    def get_dir(self, src_dir: os.PathLike, dst_dir: os.PathLike) -> None:
        """
        Recursively copy local directory to another location on the local filesystem.
        Args:
            src_dir (str): Local directory path
            dst_dir (str): Local target path
        """
        shutil.copytree(src_dir, dst_dir)

    def write_str_to_file(self, string, filepath: os.PathLike) -> None:
        """
        Write a string to a file at the specified file path.
        Args:
            string (str): The string content to write to the file.
            filepath (str): The path of the file where the string should be written.
        """
        with open(filepath, "w") as f:
            f.write(string)

    def write_dict_to_json(self, filepath: os.PathLike, param_dict) -> None:
        """
        Write a dictionary to a JSON file at the specified file path.
        Args:
            filepath (str): The path of the file where the dictionary should be written.
            param_dict (dict): The dictionary to serialize and write to the JSON file.
        """
        filepath = str(filepath)
        with open(filepath, "w") as f:
            json.dump(param_dict, f, indent=2)

    def read_ase(self, filepath: os.PathLike, **kwargs) -> list[Atoms]:
        """
        Read an ASE file from the specified file path.
        Args:
            filepath (str): The path of the ASE file to read.
        Returns:
            Atoms: An ASE Atoms object or a list of them read from the file.
        """
        return ase.io.read(filepath, **kwargs)

    def write_ase(self, filepath: os.PathLike, atoms_list, **kwargs) -> None:
        """
        Write an ASE Atoms object or a list of them to the specified file path.
        Args:
            filepath (str): The path of the file where the Atoms object(s) should be written.
            atoms_list (list[Atoms]): A list of ASE Atoms objects to write to the file.
        """
        ase.io.write(filepath, atoms_list, **kwargs)

    def exists(self, path: os.PathLike) -> bool:
        """
        Check if a file or directory exists at the specified path.
        Args:
            path (str): The path to check for existence.
        Returns:
            bool: True if the path exists, False otherwise.
        """
        return os.path.exists(path)

    def create_tempdir(self) -> os.PathLike:
        """Create a temporary directory on the local filesystem and return its path."""
        return tempfile.mkdtemp()

    def remove(self, path: os.PathLike) -> None:
        """
        Remove a file at the specified path.

        Parameters:
            path (str): The path of the file to remove.
        """
        os.remove(path)

def reconnect_on_failure(func):
    """
    Decorator to reconnect if SSH/SFTP operation fails.
    This decorator wraps methods that perform SSH/SFTP operations. If an
    SSHException, EOFError, or OSError is raised during the execution of the
    decorated method, it attempts to reconnect and retries the operation once.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Attempt the operation
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

    Args:
        remote_alias (str): The alias for the remote host as defined in the SSH config.
        remote_temp_path (str, optional): Path on the remote system to use for
            temporary files.
        local_temp_path (str, optional): Path on the local system to use for
            temporary files.
        slurm_config_file (str, optional): Path to a SLURM configuration file.
            If provided, the configuration will be loaded and stored in the
            `slurm_config` attribute. 
    """

    def __init__(
        self,
        remote_alias: Optional[str] = None,
        remote_temp_path: Optional[str] = None,
        local_temp_path: Optional[str] = None,
        slurm_config_file: Optional[str] = None,
    ):
        super().__init__(slurm_config_file)

        if slurm_config_file is None:
            if remote_alias is None:
                raise Exception(
                    "Need either slurm_config_file or remote_alias!"
                )
            # Get remote alias from argument
            self.remote_alias = remote_alias
        else:
            # Get remote alias from slurm config file
            self.remote_alias = self.slurm_config["remote_machine"]

        self.ssh_client = None
        self.sftp_client = None

        self._connect()

        if remote_temp_path is None:
            self.remote_tempdir = Path("/tmp")
        else:
            self.remote_tempdir = Path(remote_temp_path)
            if not self.exists(self.remote_tempdir):
                raise Exception("`remote_temp_path` must already exist!")

        # local_tempdir is only required to temporarily create extxyz files
        # for reading and writing ase objects and succesively move them via
        # sftp
        if local_temp_path is None:
            local_temp_path = Path("/tmp")
        else:
            if not os.path.exists(local_temp_path):
                raise Exception("`local_temp_path` must already exist!")

        self.local_tempdir = self.tag_path(Path(local_temp_path) / "ase_temp")
        os.mkdir(self.local_tempdir)

    def _connect(self) -> None:
        ssh_config_file = os.path.expanduser("~/.ssh/config")
        ssh_config = paramiko.SSHConfig()
        with open(ssh_config_file) as f:
            ssh_config.parse(f)

        # Get the config option dictionary for the specified host
        host_config = ssh_config.lookup(self.remote_alias)

        # Extract relevant connection details
        hostname = host_config.get("hostname")
        username = host_config.get("user")
        # Get the first IdentityFile if available, else None
        identityfiles = host_config.get("identityfile", [None])
        if len(identityfiles) > 1:
            print("Warning: Found more than 1 key, using the first one: {}.".format(identityfiles[0]))
        identityfile = identityfiles[0]
        proxyjump = host_config.get("proxyjump")

        if proxyjump:
            # Step 1: Connect to jump host

            # Get jump host config
            jump_config = ssh_config.lookup(proxyjump)
            jump_host = jump_config.get("hostname")
            jump_user = jump_config.get("user")
            jump_key = jump_config.get("identityfile", [None])[0]
            # Connect to jump host
            jump_client = paramiko.SSHClient()

            # Automatically add the jump host's SSH key if missing
            jump_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Connect to jump host
            jump_client.connect(
                jump_host,
                username=jump_user,
                key_filename=jump_key,
            )

            # Step 2: Create a tunnel from jump host to target
            jump_transport = jump_client.get_transport()
            # Define the destination and local addresses
            dest_addr = (hostname, 22)
            local_addr = ("localhost", 0)
            # Open a channel through the jump host of type "direct-tcpip"
            channel = jump_transport.open_channel(
                "direct-tcpip", dest_addr, local_addr
            )

            # Step 3: Connect final client using this channel as a socket

            # Create the SSH client for the target host
            self.ssh_client = paramiko.SSHClient()
            # Automatically add the target host's SSH key if missing
            self.ssh_client.set_missing_host_key_policy(
                paramiko.AutoAddPolicy()
            )
            # Connect to the target host using the channel from the jump host
            self.ssh_client.connect(
                hostname,
                username=username,
                key_filename=identityfile,
                sock=channel,
            )
        else:
            # Direct connection
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(
                paramiko.AutoAddPolicy()
            )
            self.ssh_client.connect(
                hostname,
                username=username,
                key_filename=identityfile,
            )
        
        # Open an SFTP client session on the SSH server
        self.sftp_client = self.ssh_client.open_sftp()

    def _disconnect(self) -> None:
        """Close the SSH and SFTP connections."""
        # Check if clients are initialized before attempting to close
        if self.sftp_client:
            self.sftp_client.close()
        if self.ssh_client:
            self.ssh_client.close()

    def _reconnect(self) -> None:
        self._disconnect()
        self._connect()

    @reconnect_on_failure
    def put(self, local: os.PathLike, remote: os.PathLike):
        """SFTP put with automatic reconnect on failure.
        Args:
            local (str): The path of the file or directory to copy.
            remote (str): The path where the file or directory should be copied to. 
        """
        local, remote = str(local), str(remote)
        self.sftp_client.put(local, remote)

    @reconnect_on_failure
    def get(self, remote: os.PathLike, local: os.PathLike) -> None:
        """SFTP get with automatic reconnect on failure.
        Args:
            remote (str): The path of the file or directory to copy.
            local (str): The path where the file or directory should be copied to. 
        """
        remote, local = str(remote), str(local)
        self.sftp_client.get(remote, local)

    @reconnect_on_failure
    def makedirs(self, path: os.PathLike, exist_ok=False) -> None:
        """SFTP makedirs with automatic reconnect on failure.
        Args:
            path (str): The directory path to create.
            exist_ok (bool, optional): If True, do not raise an error if the
                directory already exists. Defaults to False."""
        path = str(path)
        try:
            self.sftp_client.mkdir(path)
        except IOError:
            if not exist_ok:
                raise

    @reconnect_on_failure
    def mkdir(self, path: os.PathLike) -> None:
        """SFTP mkdir with automatic reconnect on failure.
        
        Args:
            path (str): The directory path to create."""
        path = str(path)
        self.sftp_client.mkdir(path)

    @reconnect_on_failure
    def rmtree(self, remotepath: os.PathLike, level=0) -> None:
        """Recursively delete a directory on the remote system via SFTP.
        
        Args:
            remotepath (str): The path of the directory to delete.
            level (int, optional): The current recursion level (used for
                indentation in print statements). Defaults to 0.
        """
        for f in self.sftp_client.listdir_attr(remotepath):
            rpath = posixpath.join(remotepath, f.filename)
            if S_ISDIR(f.st_mode):
                self.rmtree(self.sftp_client, rpath, level=(level + 1))
            else:
                rpath = posixpath.join(remotepath, f.filename)
                print("removing %s%s" % ("    " * level, rpath))
                self.sftp_client.remove(rpath)
        print("removing %s%s" % ("    " * level, remotepath))
        self.sftp_client.rmdir(remotepath)

    @reconnect_on_failure
    def put_dir(self, local_dir: os.PathLike, remote_dir: os.PathLike) -> None:
        """
        Recursively copy local directory to remote host via SFTP.

        Args:
            local_dir: Local directory path
            remote_dir: Remote target path
        """
        local_dir = Path(local_dir)
        # Ensure remote_dir ends with a slash, removing any trailing slashes first if necessary
        remote_dir = str(remote_dir).rstrip("/") + "/"

        # Check if local_dir is a directory
        if not local_dir.is_dir():
            raise ValueError(f"Local path '{local_dir}' is not a directory.")

        # Create remote directory if it doesn't exist
        if not self.exists(remote_dir):
            self.sftp_client.mkdir(remote_dir)

        # Iterate over items in the local directory
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
    def get_dir(self, remote_dir: os.PathLike, local_dir: os.PathLike) -> None:
        """
        Recursively copy remote directory to local host via SFTP.

        Args:
            remote_dir (str): Remote directory path
            local_dir (str or Path): Local target path
        """
        # Ensure remote_dir ends with a slash, removing any trailing slashes first if necessary
        remote_dir = str(remote_dir).rstrip("/") + "/"
        local_dir = Path(local_dir)

        if not self.exists(remote_dir):
            return

        local_dir.mkdir(parents=True, exist_ok=True)

        for filename in self.sftp_client.listdir(remote_dir):
            remote_path = remote_dir + filename
            local_path = local_dir / filename
            # Check if it's a directory or file
            if S_ISDIR(self.sftp_client.stat(remote_path).st_mode):
                # Recursively download directory
                self.get_dir(remote_path + "/", local_path)
            else:
                # If the file doesn't exist locally, download it
                if not local_path.is_file():
                    self.sftp_client.get(remote_path, str(local_path))

    @reconnect_on_failure
    def write_str_to_file(self, string, filepath: os.PathLike) -> None:
        """
        Write a string to a file at the specified file path on the remote system.

        Args:
            string (str): The string content to write to the file.
            filepath (str): The path of the file where the string should be written.
        """
        filepath = str(filepath)
        with self.sftp_client.file(filepath, "w") as f:
            f.write(string)

    @reconnect_on_failure
    def write_dict_to_json(self, filepath: os.PathLike, param_dict) -> None:
        """
        Write a dictionary to a JSON file at the specified file path on the remote system.
        Args:
            filepath (str): The path of the file where the dictionary should be written.
            param_dict (dict): The dictionary to serialize and write to the JSON file.
        """
        filepath = str(filepath)
        with self.sftp_client.file(filepath, "w") as f:
            f.write(json.dumps(param_dict, indent=2))

    @reconnect_on_failure
    def read_ase(self, filepath: os.PathLike, **kwargs) -> list[Atoms]:
        """
        Read an ASE file from the specified file path on the remote system by 
        temporarily downloading it to the local system.
        Args:
            filepath (str): The path of the ASE file to read.
        """
        filepath = str(filepath)
        # Temporary local file for reading
        local_temp = self.local_tempdir / "temp_ase_file.extxyz"
        self.sftp_client.get(filepath, local_temp)
        atoms = ase.io.read(local_temp, **kwargs)
        # Clean up temporary file
        os.remove(local_temp)
        return atoms

    @reconnect_on_failure
    def write_ase(self, filepath: os.PathLike, atoms_list, **kwargs) -> None:
        """
        Write an ASE Atoms object to a file at the specified file path on the
        remote system by temporarily creating it on the local system.
        Args:
            filepath (str): The path of the file where the ASE Atoms object should be written
            atoms_list (list of Atoms): A list of ASE Atoms objects to write to the file.
        """
        filepath = str(filepath)
        # Temporary local file for writing
        local_temp = self.local_tempdir / "temp_ase_file.extxyz"
        ase.io.write(local_temp, atoms_list, **kwargs)
        self.sftp_client.put(local_temp, filepath)
        # Clean up temporary file
        os.remove(local_temp)

    @reconnect_on_failure
    def exists(self, path: os.PathLike) -> bool:
        """
        Check if a file or directory exists at the specified path on the remote system.
        Args:
            path (str): The path to check for existence.
        Returns:
            bool: True if the path exists, False otherwise.
        """
        path = str(path)
        try:
            self.sftp_client.stat(path)
            return True
        except FileNotFoundError:
            return False

    @reconnect_on_failure
    def create_tempdir(self) -> os.PathLike:
        """Create a temporary directory on the remote filesystem and return its path."""
        # Create a unique temporary directory path
        tempdir = self.tag_path(self.remote_tempdir / "dir")
        # Create the directory on the remote system
        self.makedirs(tempdir, exist_ok=False)
        return tempdir

    @reconnect_on_failure
    def remove(self, path: os.PathLike) -> None:
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
