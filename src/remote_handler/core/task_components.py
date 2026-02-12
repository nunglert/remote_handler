"""Composable components for building remote tasks.

This module provides modular components that can be composed together to build
complex remote task workflows. Breaking down tasks into these components makes
the code more maintainable and testable.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from remote_handler.core import file_handler, file_utils

logger = logging.getLogger(__name__)


class FileAllocator:
    """Handles file allocation and copying to remote filesystem.

    This component is responsible for:
    - Copying individual files to remote
    - Allocating shared resources that all jobs can access
    - Creating calculation directories and writing individual resources

    Args:
        remote_fs: FileSystemHandlerBase instance for remote operations
        rootpath: Root directory on remote where files are allocated
    """

    def __init__(
        self,
        remote_fs: file_handler.FileSystemHandlerBase,
        rootpath: os.PathLike,
    ):
        self.remote_fs = remote_fs
        self.rootpath = Path(rootpath)

    def copy_file(self, local_path: os.PathLike) -> Path:
        """Copy a single file from local to remote rootpath.

        Args:
            local_path: Path to the local file to copy

        Returns:
            Path to the file on the remote machine
        """
        moved_path = self.rootpath / os.path.split(local_path)[-1]
        self.remote_fs.put(local_path, moved_path)
        logger.debug(f"Copied {local_path} to {moved_path}")
        return moved_path

    def allocate_shared_resources(
        self, shared_resources: list[os.PathLike]
    ) -> list[Path]:
        """Copy shared resources to remote rootpath.

        Shared resources are files that all jobs in an array can access,
        such as model parameters or configuration files.

        Args:
            shared_resources: List of paths to files to copy

        Returns:
            List of paths to the copied files on remote
        """
        remote_paths = []
        for resource in shared_resources:
            remote_path = self.copy_file(resource)
            remote_paths.append(remote_path)
        logger.info(f"Allocated {len(shared_resources)} shared resources")
        return remote_paths

    def allocate_individual_resources(
        self,
        individual_resources: list[list],
        calcdir: Path,
    ) -> list[Path]:
        """Create calculation directories and write individual resources.

        For array jobs, each task needs its own calculation directory with
        its own input data. This method creates these directories and writes
        the Writeable objects to them.

        Args:
            individual_resources: List of lists of Writeable objects
            calcdir: Base calculation directory (will create subdirs 0, 1, 2, ...)

        Returns:
            List of paths to created calculation directories
        """
        all_calcpaths = []

        for idx, data_at_idx in enumerate(zip(*individual_resources)):
            # Create a directory for each job in the array
            calcpath = calcdir / f"{idx}"
            self.remote_fs.makedirs(calcpath)

            # For each Writeable object, call its write method
            for data in data_at_idx:
                data.write(calcpath)

            all_calcpaths.append(calcpath)

        logger.info(
            f"Allocated {len(all_calcpaths)} individual calculation directories"
        )
        return all_calcpaths


class ResultCollector:
    """Handles result collection from remote to local filesystem.

    This component is responsible for copying results back from the remote
    machine to the local machine after jobs complete. It handles directory
    creation and provides utilities for selective result collection.

    Args:
        remote_fs: FileSystemHandlerBase instance for remote operations
        local_fs: FileSystemHandlerBase instance for local operations
    """

    def __init__(
        self,
        remote_fs: file_handler.FileSystemHandlerBase,
        local_fs: file_handler.FileSystemHandlerBase,
    ):
        self.remote_fs = remote_fs
        self.local_fs = local_fs

    def collect_directory(
        self,
        remote_path: os.PathLike,
        local_path: os.PathLike,
        move: bool = True,
    ) -> None:
        """Copy an entire directory from remote to local.

        Args:
            remote_path: Path to directory on remote machine
            local_path: Path where directory should be copied on local machine
            move: If True, handle existing local_path by moving it aside
        """
        # Ensure local path exists and handle existing files
        file_utils.secure_declare_path(
            local_path, move=move, is_directory=True
        )

        logger.info(f"Collecting results from {remote_path} to {local_path}")
        self.remote_fs.get_dir(remote_path, local_path)
        logger.info("Results collected successfully")

    def collect_file(
        self,
        remote_path: os.PathLike,
        local_path: os.PathLike,
    ) -> None:
        """Copy a single file from remote to local.

        Args:
            remote_path: Path to file on remote machine
            local_path: Path where file should be copied on local machine
        """
        logger.debug(f"Collecting file from {remote_path} to {local_path}")
        self.remote_fs.get(remote_path, local_path)
