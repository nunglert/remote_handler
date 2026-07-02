"""Backend-agnostic configuration for a remote compute target.

``ClusterConfig`` describes *where* and *as whom* work runs (SSH alias, conda
env, save path) plus cross-cutting policy (connection resilience, which
execution backend to use). It is consumed by every executor backend -- SLURM
and the scheduler-free ``direct`` one alike -- so it deliberately lives apart
from any single backend's handler code.
"""

import json
import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

from remote_handler.config import get_config
from remote_handler.core import resilience
from remote_handler.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class ClusterConfig:
    """
    Configuration for a remote compute target, independent of execution backend.

    Encapsulates the cluster-specific settings needed to run work on a remote
    machine -- where it connects, as whom, which conda env, where results are
    saved -- plus the choice of execution backend (``executor``). It is shared
    by all backends, so most fields are generic; ``partition``/``template`` are
    only consulted by the SLURM backend.

    Args:
        remote_machine: SSH alias or hostname of the remote machine
        partition: SLURM partition name to submit jobs to (SLURM backend only)
        conda_env: Name of the conda environment to activate on the remote
        savepath: Default path on the remote where computations are saved
        template: Jinja2 template file for SLURM job headers (SLURM backend only)
        resilient_connection: Opt-in. When True, SSH operations to this cluster
            pause-and-retry instead of crashing if the connection/credential
            drops (e.g. an expired ``step ssh login`` cert), resuming once it
            is refreshed. Off by default; most clusters don't need it.
        connection_retry_interval: Seconds between retries while paused.
        connection_max_wait: Give up after this many seconds paused; ``None``
            (default) waits indefinitely until the credential is refreshed.
        connection_command_timeout: Hard timeout (seconds) on each subprocess
            ``ssh`` call, so a hung ssh (e.g. blocked on an interactive QR/MFA
            prompt after the credential expired) fails with ``TimeoutExpired``
            instead of parking the driver forever. Applies regardless of
            ``resilient_connection``; ``None`` disables it. Defaults to 120s.
        executor: Which execution backend to run jobs through. ``"slurm"``
            (default) submits via sbatch; other backends are selected by the
            executor factory. Unset in a config JSON means ``"slurm"``.
        environment: Per-application shell setup for *this* cluster, keyed by
            application name (e.g. ``"vasp"``, ``"orca"``). Each value is shell
            lines injected into the SLURM header for jobs of that application
            (module loads, ``source`` of an env script, ``export`` of paths
            like ``VASP_PP_PATH``). This is where cluster- and site-specific
            setup belongs -- so it lives in the (per-user) cluster JSON rather
            than hardcoded in an actor. Actors that need it (e.g.
            ``VASPEvaluator`` -> ``"vasp"``) look up their key; anything passed
            as ``slurm_header_add`` is appended after it. Empty by default.

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
    resilient_connection: bool = False
    connection_retry_interval: float = 60.0
    connection_max_wait: Optional[float] = None
    connection_command_timeout: Optional[float] = 120.0
    executor: str = "slurm"
    environment: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # SFTP never expands '~', so a tilde savepath can never resolve. Reject
        # it at construction (covers from_dict / from_file / direct) instead of
        # letting it fail deep in a transfer as a bare "No such file".
        if str(self.savepath).startswith("~"):
            raise ConfigurationError(
                f"savepath '{self.savepath}' uses '~', which is not "
                "supported: SFTP does not expand '~'. Use an absolute remote "
                "path (e.g. '/home/<user>/...')."
            )

    @property
    def connection_policy(self) -> "resilience.ConnectionPolicy":
        """Build the ``ConnectionPolicy`` described by this config."""
        return resilience.ConnectionPolicy(
            enabled=self.resilient_connection,
            poll_interval=self.connection_retry_interval,
            max_wait=self.connection_max_wait,
            command_timeout=self.connection_command_timeout,
        )

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
    def from_name(cls, name: str) -> "ClusterConfig":
        """Load the cluster registered under ``name`` (e.g. ``"lugus"``).

        Looks for ``<name>.json`` in the resolved config directory -- where
        ``remote_handler configure cluster <name>`` writes it -- and loads it.
        A trailing ``.json`` in ``name`` is accepted and ignored, so both
        ``from_name("lugus")`` and ``from_name("lugus.json")`` work.

        Args:
            name: Registered cluster name (the ``<name>.json`` stem).

        Returns:
            ClusterConfig loaded from the registered JSON.

        Raises:
            ConfigurationError: If no cluster is registered under ``name``.
                The message lists the names that *are* available.

        Example:
            >>> config = ClusterConfig.from_name("lugus")
        """
        stem = name[:-5] if name.endswith(".json") else name
        config_dir = get_config().config_dir
        config_path = Path(config_dir) / f"{stem}.json" if config_dir else None

        if config_path is None or not config_path.exists():
            available = cls.available_names()
            hint = (
                "Available clusters: " + ", ".join(available)
                if available
                else "No clusters are configured yet -- create one with "
                f"'remote_handler configure cluster {stem}'."
            )
            raise ConfigurationError(
                f"No cluster named '{stem}' is configured in {config_dir}.\n"
                f"{hint}"
            )

        return cls.from_file(str(config_path))

    @classmethod
    def available_names(cls) -> list[str]:
        """Names of clusters registered in the resolved config directory.

        These are the ``<name>.json`` stems (excluding the global
        ``config.json``) that :meth:`from_name` accepts. Returns an empty
        list when nothing is configured.
        """
        config_dir = get_config().config_dir
        if not config_dir or not Path(config_dir).is_dir():
            return []
        return sorted(
            p.stem
            for p in Path(config_dir).glob("*.json")
            if p.name != "config.json"
        )

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
