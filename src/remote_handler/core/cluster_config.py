"""Configuration management for cluster/SLURM setups."""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Load directory where config files are stored from environment variable
CONFIG_DIR = Path(os.environ.get("REMOTE_HANDLER_CONFIG_DIR", ""))


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
            Exception: If the config file doesn't exist or CONFIG_DIR is not set
                      for relative paths

        Example:
            >>> # Using absolute path
            >>> config = ClusterConfig.from_file("/home/user/configs/cluster.json")
            >>> # Using relative path (requires REMOTE_HANDLER_CONFIG_DIR)
            >>> config = ClusterConfig.from_file("cluster.json")
        """
        if os.path.isabs(config_file):
            if not os.path.exists(config_file):
                raise Exception(f"{config_file} does not exist!")
            config_path = config_file
        else:
            if CONFIG_DIR is None or CONFIG_DIR == Path(""):
                raise Exception(
                    "If config_file is not an absolute path, "
                    "set REMOTE_HANDLER_CONFIG_DIR environment variable, "
                    "where config files for different clusters are stored."
                )
            config_path = CONFIG_DIR / config_file
            if not os.path.exists(config_path):
                raise Exception(
                    f"{config_file} does not exist in your "
                    f"REMOTE_HANDLER_CONFIG_DIR={CONFIG_DIR}"
                )

        logger.info(f"Reading ClusterConfig from {config_path}")
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        return cls(**config_dict)

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
        return cls(**config_dict)
