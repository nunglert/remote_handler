import functools
import json
import logging
import os
import posixpath
import random
import shlex
import shutil
import socket
import string
import subprocess
import tarfile
import tempfile
import threading
from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path
from stat import S_ISDIR, S_ISLNK
from typing import Optional

import ase.io
import numpy as np
import paramiko
from ase import Atoms

from remote_handler.core import resilience
from remote_handler.core.cluster_config import (
    KNOWN_TRANSFERS,
    TRANSFER_SFTP,
    TRANSFER_TAR,
)
from remote_handler.core.file_utils import tag_path as tag_path_util
from remote_handler.core.ssh_connection import (
    ConfigBasedSSHClientProvider,
    SSHClientProvider,
)
from remote_handler.exceptions import (
    CommandTimeoutError,
    ConfigurationError,
    FileSystemError,
    RemoteConnectionError,
)

logger = logging.getLogger(__name__)

# Result of ``run_command`` -- a uniform shape across local/remote execution so
# callers (e.g. the scheduler-free DirectExecutor) don't branch on transport.
CommandResult = namedtuple("CommandResult", ["returncode", "stdout", "stderr"])

# Default limit (seconds) for ``run_command``. Without one, a command stalled
# on an overloaded login node blocks the caller forever with no error.
COMMAND_TIMEOUT = 120

# Suffix of a file still being downloaded; renamed away once complete.
PART_SUFFIX = ".part"


def _is_complete(local_path: Path, size: int) -> bool:
    """True if ``local_path`` is a file of exactly ``size`` bytes."""
    try:
        return local_path.is_file() and local_path.stat().st_size == size
    except OSError:
        return False


def _write_atomically(src, local_path: Path) -> None:
    """Copy the file object ``src`` to ``local_path`` via a ``.part`` file.

    The final name only ever holds a complete file: a transfer cut short
    leaves just the ``.part``, which the next attempt overwrites.
    """
    part = local_path.with_name(local_path.name + PART_SUFFIX)
    with open(part, "wb") as fh:
        shutil.copyfileobj(src, fh, 1 << 20)
    os.replace(part, local_path)


def _fetch_sftp_file(
    sftp: paramiko.SFTPClient, remote_path: str, local_path: Path, size: int
) -> None:
    """Download one file whose size is already known from the listing.

    Unlike ``SFTPClient.get`` this needs no extra ``stat`` round trip, and
    the read-ahead (``prefetch``) is sized up front.
    """
    with sftp.open(remote_path, "rb") as remote_file:
        # prefetch(0) never marks itself done, so a read would then block.
        if size > 0:
            remote_file.prefetch(size)
        _write_atomically(remote_file, local_path)


def _timeout_message(command: str, timeout: float, where: str) -> str:
    return (
        f"Command on {where} did not finish within {timeout:g}s:\n"
        f"{command.strip()}"
    )


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class FileSystemHandlerBase(ABC):
    """
    Interface for handling file operations and directory management on a
    file system.

    This class provides an abstract interface for common file operations
    that can work with both local and remote file systems. The concrete
    operations (``get``, ``put``, ``makedirs``, ``mkdir``, ...) are documented
    on their individual methods below.
    """

    def __init__(self):
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
        savetag = tag_path_util(working_dir, filename, is_directory=True)
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
            path = tag_path_util(parentpath, childname, is_directory)

        return path


class LocalFileSystem(FileSystemHandlerBase):
    """
    LocalFileSystem is a concrete implementation of FileSystemHandlerBase for
    handling file operations on the local filesystem.
    """

    def __init__(self):
        super().__init__()

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
        An existing ``dst_dir`` is completed (files are overwritten), matching
        the incremental ``RemoteFileSystem.get_dir``.
        Args:
            src_dir (str): Local directory path
            dst_dir (str): Local target path
        """
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

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
            json.dump(param_dict, f, indent=2, cls=NumpyEncoder)

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

    def run_command(
        self, command: str, timeout: Optional[float] = COMMAND_TIMEOUT
    ) -> CommandResult:
        """Run a shell command on the local machine.

        Counterpart to :meth:`RemoteFileSystem.run_command`, so executor code
        can launch work the same way regardless of where the filesystem lives.
        Raises :class:`CommandTimeoutError` if it runs longer than
        ``timeout`` seconds (``None`` waits indefinitely).
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise CommandTimeoutError(
                _timeout_message(command, timeout, "localhost")
            ) from exc
        return CommandResult(result.returncode, result.stdout, result.stderr)


def reconnect_on_failure(func):
    """
    Decorator to reconnect if SSH/SFTP operation fails.
    This decorator wraps methods that perform SSH/SFTP operations. If an
    SSHException, EOFError, or OSError is raised during the execution of the
    decorated method, it attempts to reconnect and retries the operation once.

    If that single reconnect+retry still fails with a connection/auth error
    *and* the filesystem's ``connection_policy`` is enabled (opt-in, off by
    default), the operation escalates to ``resilience.run_resilient``: it
    pauses and keeps reconnecting on an interval until the credential is
    refreshed, rather than crashing. Disabled is exactly the old behaviour.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Attempt the operation
        try:
            return func(self, *args, **kwargs)
        except (paramiko.SSHException, EOFError, OSError) as exc:
            logger.warning(
                "SFTP op '%s' failed (%s); reconnecting once...",
                func.__name__,
                exc,
            )
            try:
                self._reconnect()
                return func(self, *args, **kwargs)
            except (
                paramiko.SSHException,
                EOFError,
                OSError,
                RemoteConnectionError,
            ) as exc2:
                policy = getattr(
                    self, "connection_policy", resilience.DISABLED
                )
                if policy.enabled and resilience.is_connection_error(exc2):
                    return resilience.run_resilient(
                        lambda: func(self, *args, **kwargs),
                        policy,
                        before_retry=self._reconnect,
                        label=f"SFTP {func.__name__} -> {self.remote_alias}",
                    )
                raise

    return wrapper


class RemoteFileSystem(FileSystemHandlerBase):
    """
    RemoteFileSystem is a concrete implementation of FileSystemHandlerBase for
    handling file operations on remote systems via SSH and SFTP.

    Args:
        remote_alias (str): The alias for the remote host as defined in the SSH config.
        identityfile (str, optional): Path to SSH identity file to use.
        remote_temp_path (str, optional): Path on the remote system to use for
            temporary files.
        local_temp_path (str, optional): Path on the local system to use for
            temporary files.
        ssh_client_provider (SSHClientProvider, optional): Custom provider for
            SSH/SFTP clients. If None, uses ConfigBasedSSHClientProvider. This
            parameter is primarily for testing.
        connection_policy (ConnectionPolicy, optional): Pause-and-retry
            behaviour on connection/auth loss. Disabled by default.
        transfer (str, optional): How :meth:`get_dir` moves files:
            ``"sftp"`` (default) or ``"tar"``. See
            :class:`~remote_handler.core.cluster_config.ClusterConfig`.
    """

    def __init__(
        self,
        remote_alias: str,
        identityfile: Optional[str] = None,
        remote_temp_path: Optional[str] = None,
        local_temp_path: Optional[str] = None,
        ssh_client_provider: Optional[SSHClientProvider] = None,
        connection_policy: Optional[resilience.ConnectionPolicy] = None,
        transfer: str = TRANSFER_SFTP,
    ):
        super().__init__()

        if transfer not in KNOWN_TRANSFERS:
            raise ConfigurationError(
                f"Unknown transfer mode '{transfer}'. "
                f"Known modes: {', '.join(KNOWN_TRANSFERS)}."
            )
        self.transfer = transfer
        # Whether the remote has ``tar``; probed once, on first tar transfer.
        self._remote_has_tar_cache: Optional[bool] = None
        self.remote_alias = remote_alias
        self.identityfile = identityfile
        self.ssh_client = None
        self.sftp_client = None
        # Opt-in pause-and-retry on connection/auth loss; off by default.
        self.connection_policy = connection_policy or resilience.DISABLED

        # Store the client provider - use config-based by default
        if ssh_client_provider is None:
            # Default: use config-based provider (backward compatible)
            self.ssh_client_provider = ConfigBasedSSHClientProvider(
                remote_alias=remote_alias, identityfile=identityfile
            )
        else:
            # Test injection: use provided mock provider
            self.ssh_client_provider = ssh_client_provider

        # The initial connect goes through the same policy as every later
        # SFTP op, so e.g. an already-expired cert pauses here until it is
        # refreshed instead of crashing at startup (no-op when disabled).
        try:
            resilience.run_resilient(
                self._initial_connect,
                self.connection_policy,
                label=f"SSH connect -> {remote_alias}",
            )
        except socket.gaierror:
            hosts_str = ", ".join(self._available_hosts()) or "(none found)"
            raise RemoteConnectionError(
                f"Could not resolve hostname for alias '{remote_alias}'. "
                f"The host may be unreachable or misspelled.\n"
                f"Available SSH hosts: {hosts_str}"
            )
        except ConfigurationError:
            raise
        except Exception as e:
            logger.warning(
                "Failed connecting SSH client. Try providing the key file "
                "explicitly. Assure it does not have a passphrase!"
            )
            raise e

        if remote_temp_path is None:
            self.remote_tempdir = Path("/tmp")
        else:
            self.remote_tempdir = Path(remote_temp_path)
            if not self.exists(self.remote_tempdir):
                raise FileSystemError(
                    f"Remote temp path does not exist: {remote_temp_path}"
                )

        # local_tempdir is only required to temporarily create extxyz files
        # for reading and writing ase objects and succesively move them via
        # sftp
        if local_temp_path is None:
            local_temp_path = Path("/tmp")
        else:
            if not os.path.exists(local_temp_path):
                raise FileSystemError(
                    f"Local temp path does not exist: {local_temp_path}"
                )

        self.local_tempdir = self.tag_path(Path(local_temp_path) / "ase_temp")
        os.mkdir(self.local_tempdir)

    def _available_hosts(self) -> list[str]:
        """Host aliases defined in the SSH config (empty if unknown)."""
        if hasattr(self.ssh_client_provider, "config_parser"):
            return self.ssh_client_provider.config_parser.get_available_hosts()
        return []

    def _initial_connect(self) -> None:
        """One attempt of the constructor's (possibly retried) connect.

        Goes through ``_reconnect`` so each retry first releases whatever a
        failed attempt left half-open (e.g. a ProxyJump client). An
        unresolvable alias that is not in the SSH config is a misspelling,
        not a network blip, so it raises a non-transient
        ``ConfigurationError`` instead of pausing forever.
        """
        try:
            self._reconnect()
        except socket.gaierror as exc:
            available = self._available_hosts()
            if self.remote_alias in available:
                raise  # known alias: DNS hiccup, transient
            hosts_str = ", ".join(available) or "(none found)"
            raise ConfigurationError(
                f"Could not resolve hostname for alias "
                f"'{self.remote_alias}', and it is not defined in the SSH "
                f"config. Is it misspelled?\n"
                f"Available SSH hosts: {hosts_str}"
            ) from exc

    def _connect(self) -> None:
        """
        Establish SSH and SFTP connections using the configured client provider.
        """
        (
            self.ssh_client,
            self.sftp_client,
        ) = self.ssh_client_provider.create_connection()

    def _disconnect(self) -> None:
        """Close the SSH and SFTP connections."""
        # Check if clients are initialized before attempting to close
        if self.sftp_client:
            self.sftp_client.close()
        if self.ssh_client:
            self.ssh_client.close()

        # Cleanup provider resources (e.g., jump client)
        if hasattr(self.ssh_client_provider, "cleanup"):
            self.ssh_client_provider.cleanup()

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
                directory already exists. Defaults to False.
        """
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
    def rmtree(self, remotepath: os.PathLike) -> None:
        """Recursively delete a directory on the remote system via SFTP.

        Symlinks are removed, never followed, so a link pointing outside
        ``remotepath`` cannot cause anything there to be deleted. If the
        connection drops midway, the reconnect retry relists and continues
        with what is left.

        Args:
            remotepath (str): The path of the directory to delete.
        """
        self._rmtree(str(remotepath))

    def _rmtree(self, remotepath: str) -> None:
        # listdir_attr reports lstat-style attributes: a symlink to a
        # directory is S_ISLNK, not S_ISDIR, so it is removed as a file.
        for attr in self.sftp_client.listdir_attr(remotepath):
            path = posixpath.join(remotepath, attr.filename)
            if S_ISDIR(attr.st_mode):
                self._rmtree(path)
            else:
                logger.debug("removing %s", path)
                self.sftp_client.remove(path)
        logger.debug("removing %s", remotepath)
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
            raise FileSystemError(f"Path is not a directory: {local_dir}")

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
        """Recursively copy a remote directory to the local host.

        Incremental: a file that already exists locally with the remote size
        is skipped, so calling this again after an interruption only fetches
        what is missing or incomplete. Files are written via a ``.part``
        sibling and renamed when complete, so a dropped connection never
        leaves a truncated file under its final name. A missing
        ``remote_dir`` is a no-op.

        How the files travel depends on :attr:`transfer`. ``"sftp"`` fetches
        them one after another over SFTP. ``"tar"`` streams them all through
        a single remote ``tar`` over the same SSH connection; if the remote
        has no ``tar`` it falls back to the ``"sftp"`` path.

        Args:
            remote_dir (str): Remote directory path
            local_dir (str or Path): Local target path
        """
        remote_dir = str(remote_dir).rstrip("/") or "/"
        local_dir = Path(local_dir)

        if not self.exists(remote_dir):
            return

        subdirs, files = self._list_remote_tree(remote_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        for rel in subdirs:
            (local_dir / rel).mkdir(parents=True, exist_ok=True)

        pending = [
            (rel, size)
            for rel, size in files
            if not _is_complete(local_dir / rel, size)
        ]
        logger.debug(
            "get_dir %s: %d of %d files to fetch (transfer=%s)",
            remote_dir,
            len(pending),
            len(files),
            self.transfer,
        )
        if not pending:
            return

        if self.transfer == TRANSFER_TAR and self._remote_has_tar():
            self._get_files_tar(remote_dir, local_dir, pending)
        else:
            for rel, size in pending:
                _fetch_sftp_file(
                    self.sftp_client,
                    posixpath.join(remote_dir, rel),
                    local_dir / rel,
                    size,
                )

    def _list_remote_tree(
        self, remote_dir: str
    ) -> tuple[list[str], list[tuple[str, int]]]:
        """List a remote tree as (subdirs, [(file, size)]), paths relative.

        One ``listdir_attr`` per directory returns every entry's type and
        size, so there is no per-file ``stat`` round trip. ``subdirs`` lists
        parents before their children. Symlinks are followed, as the earlier
        ``stat``-based walk did.
        """
        subdirs: list[str] = []
        files: list[tuple[str, int]] = []
        stack = [""]
        while stack:
            rel_dir = stack.pop()
            base = posixpath.join(remote_dir, rel_dir)
            for attr in self.sftp_client.listdir_attr(base):
                name = attr.filename
                # A well-behaved server never sends these; refuse anything
                # that would place a file outside local_dir.
                if name in ("", ".", "..") or "/" in name:
                    logger.warning(
                        "Skipping suspicious remote entry %r in %s",
                        name,
                        base,
                    )
                    continue
                rel = posixpath.join(rel_dir, name) if rel_dir else name
                if attr.st_mode is not None and S_ISLNK(attr.st_mode):
                    attr = self.sftp_client.stat(
                        posixpath.join(remote_dir, rel)
                    )
                if attr.st_mode is not None and S_ISDIR(attr.st_mode):
                    subdirs.append(rel)
                    stack.append(rel)
                else:
                    files.append((rel, attr.st_size))
        return subdirs, files

    def _remote_has_tar(self) -> bool:
        """Whether ``tar`` is on the remote ``PATH`` (probed once)."""
        if self._remote_has_tar_cache is None:
            result = self.run_command("command -v tar")
            self._remote_has_tar_cache = result.returncode == 0
            if not self._remote_has_tar_cache:
                logger.warning(
                    "transfer='tar' but %s has no 'tar' on PATH; falling "
                    "back to per-file SFTP (slower).",
                    self.remote_alias,
                )
        return self._remote_has_tar_cache

    def _get_files_tar(
        self,
        remote_dir: str,
        local_dir: Path,
        pending: list[tuple[str, int]],
    ) -> None:
        """Fetch ``pending`` files in one ``tar`` stream over SSH exec.

        The file list goes to ``tar -T -`` on stdin; the archive comes back
        on stdout and is unpacked as it arrives (stdlib ``tarfile`` in
        stream mode), so nothing is buffered whole. Only regular files that
        were requested are written -- anything else in the stream is
        ignored -- and each lands via a ``.part`` file.
        """
        wanted = {rel for rel, _ in pending}
        stdin, stdout, stderr = self.ssh_client.exec_command(
            f"tar -cf - -C {shlex.quote(remote_dir)} -T -"
        )

        # Feed the list from a thread: tar starts writing before it has read
        # all names, and blocking on a full stdin window while nobody drains
        # stdout would deadlock. The './' prefix stops tar from parsing a
        # name that starts with '-' as an option.
        feed_errors: list[BaseException] = []

        def feed() -> None:
            try:
                stdin.write("".join(f"./{rel}\n" for rel, _ in pending))
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                feed_errors.append(exc)
            finally:
                stdin.channel.shutdown_write()

        feeder = threading.Thread(target=feed, daemon=True)
        feeder.start()
        with tarfile.open(fileobj=stdout, mode="r|") as archive:
            for member in archive:
                rel = posixpath.normpath(member.name)
                if not member.isfile() or rel not in wanted:
                    continue
                target = local_dir / rel
                _write_atomically(archive.extractfile(member), target)
                os.utime(target, (member.mtime, member.mtime))
        feeder.join()
        if feed_errors:
            raise feed_errors[0]

        status = stdout.channel.recv_exit_status()
        if status != 0:
            message = stderr.read().decode(errors="replace").strip()
            raise FileSystemError(
                f"Remote tar in {remote_dir} on {self.remote_alias} exited "
                f"with status {status}: {message}"
            )

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
            f.write(json.dumps(param_dict, indent=2, cls=NumpyEncoder))

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
    def read_str_from_file(self, filepath: os.PathLike) -> str:
        """
        Read a string from a file at the specified file path on the remote system.

        Args:
            filepath (str): The path of the file to read the string from.
        """
        filepath = str(filepath)
        with self.sftp_client.file(filepath, "r") as f:
            string = f.read().decode("utf-8")
        return string

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
            raise FileSystemError(
                f"Failed to remove remote file: {path}"
            ) from e

    @reconnect_on_failure
    def run_command(
        self, command: str, timeout: Optional[float] = COMMAND_TIMEOUT
    ) -> CommandResult:
        """Run a command on the remote host over the SSH connection.

        Uses the same authenticated paramiko client as the SFTP transport, so
        it honours whatever ssh-config/credentials the filesystem connected
        with (no separate ``ssh`` subprocess, no dependency on the default
        ``~/.ssh/config``). Detached launches (``nohup ... &``) return at once
        because the foreground statement exits immediately.

        Raises :class:`CommandTimeoutError` if the command has not exited
        within ``timeout`` seconds (``None`` waits indefinitely). The channel
        is closed then; the remote process may keep running.
        """
        _, stdout, stderr = self.ssh_client.exec_command(
            command, timeout=timeout
        )
        channel = stdout.channel
        # recv_exit_status() has no timeout of its own; wait on the event it
        # blocks on instead.
        if not channel.status_event.wait(timeout):
            channel.close()
            raise CommandTimeoutError(
                _timeout_message(command, timeout, self.remote_alias)
            )
        returncode = channel.recv_exit_status()
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        return CommandResult(returncode, out, err)
