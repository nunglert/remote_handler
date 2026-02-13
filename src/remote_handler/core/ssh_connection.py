"""
SSH Connection Components for Remote File System.

This module provides modular, testable components for establishing SSH and SFTP
connections. It separates connection logic into independently testable pieces:

- DTOs for connection parameters
- SSH config parsing
- Connection strategies (Direct, ProxyJump)
- Client provider abstraction for dependency injection
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import paramiko

from remote_handler.exceptions import RemoteConnectionError

logger = logging.getLogger(__name__)


@dataclass
class ProxyJumpParams:
    """
    Parameters for ProxyJump (jump host) connections.

    Attributes:
        hostname: The jump host hostname
        username: Username for jump host authentication
        identityfile: Optional SSH key file for jump host
    """

    hostname: str
    username: str
    identityfile: Optional[str] = None


@dataclass
class SSHConnectionParams:
    """
    Structured representation of SSH connection parameters.

    This serves as an intermediate data structure between SSH config
    parsing and actual connection establishment.

    Attributes:
        hostname: Target host hostname
        username: Username for authentication
        identityfile: Optional SSH key file path
        proxyjump: Optional ProxyJump parameters for jump host connections
    """

    hostname: str
    username: str
    identityfile: Optional[str] = None
    proxyjump: Optional[ProxyJumpParams] = None


class SSHConfigParser:
    """
    Parses SSH configuration files and extracts connection parameters.

    This class encapsulates all SSH config file I/O and parsing logic,
    making it independently testable without requiring actual SSH connections.

    Attributes:
        config_file: Path to SSH config file (default: ~/.ssh/config)
    """

    def __init__(self, config_file: str = "~/.ssh/config"):
        """
        Initialize SSH config parser.

        Args:
            config_file: Path to SSH config file
        """
        self.config_file = os.path.expanduser(config_file)

    def parse_host_config(
        self, remote_alias: str, identityfile_override: Optional[str] = None
    ) -> SSHConnectionParams:
        """
        Parse SSH config for a given host alias.

        Args:
            remote_alias: The SSH host alias to look up
            identityfile_override: Optional identity file to use instead of config

        Returns:
            SSHConnectionParams with all connection details

        Raises:
            FileNotFoundError: If SSH config file doesn't exist
            ValueError: If hostname not found for alias
        """
        ssh_config = paramiko.SSHConfig()
        with open(self.config_file) as f:
            ssh_config.parse(f)

        host_config = ssh_config.lookup(remote_alias)

        # Extract hostname and username
        hostname = host_config.get("hostname")
        username = host_config.get("user")

        if hostname is None:
            raise RemoteConnectionError(
                f"No hostname found for alias: {remote_alias}\n"
                f"Check your ~/.ssh/config file"
            )

        # Handle identity file with override logic
        if identityfile_override is None:
            identityfiles = host_config.get("identityfile", [None])
        else:
            identityfiles = [identityfile_override]

        if len(identityfiles) > 1:
            logger.warning(
                "Found more than 1 key for %s, using the first one: %s",
                remote_alias,
                identityfiles[0],
            )
        identityfile = identityfiles[0]

        # Log identity file usage
        if identityfile:
            logger.info(
                "SSH (%s): using identity file %s",
                remote_alias,
                os.path.expanduser(identityfile),
            )
        else:
            logger.info(
                "SSH (%s): no identity file specified (agent / default keys)",
                remote_alias,
            )

        # Handle ProxyJump recursively
        proxyjump_alias = host_config.get("proxyjump")
        proxyjump = None
        if proxyjump_alias:
            jump_config = ssh_config.lookup(proxyjump_alias)
            proxyjump = ProxyJumpParams(
                hostname=jump_config.get("hostname"),
                username=jump_config.get("user"),
                identityfile=jump_config.get("identityfile", [None])[0],
            )

        return SSHConnectionParams(
            hostname=hostname,
            username=username,
            identityfile=identityfile,
            proxyjump=proxyjump,
        )


class SSHConnectionStrategy(ABC):
    """
    Abstract base class for SSH connection strategies.

    This allows different connection methods (direct, ProxyJump, etc.)
    to be implemented as independent, testable classes.
    """

    @abstractmethod
    def connect(self, params: SSHConnectionParams) -> paramiko.SSHClient:
        """
        Establish SSH connection and return configured client.

        Args:
            params: Connection parameters

        Returns:
            Connected paramiko.SSHClient instance
        """
        pass


class DirectSSHConnection(SSHConnectionStrategy):
    """
    Direct SSH connection without any jump hosts.

    This is the simplest connection strategy for direct host-to-host connections.
    """

    def connect(self, params: SSHConnectionParams) -> paramiko.SSHClient:
        """
        Establish direct SSH connection.

        Args:
            params: Connection parameters

        Returns:
            Connected SSH client
        """
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        ssh_client.connect(
            params.hostname,
            username=params.username,
            key_filename=params.identityfile,
        )

        return ssh_client


class ProxyJumpSSHConnection(SSHConnectionStrategy):
    """
    SSH connection through a jump host (ProxyJump).

    This implements the SSH tunneling logic for bastion/jump hosts.
    Creates a tunnel: local -> jump_host -> target_host

    Attributes:
        jump_client: Reference to jump host client for cleanup
    """

    def __init__(self):
        """Initialize ProxyJump connection strategy."""
        self.jump_client = None

    def connect(self, params: SSHConnectionParams) -> paramiko.SSHClient:
        """
        Establish SSH connection through jump host.

        Args:
            params: Connection parameters including proxyjump details

        Returns:
            Connected SSH client to target host

        Raises:
            ValueError: If proxyjump params are missing
        """
        if params.proxyjump is None:
            raise RemoteConnectionError(
                "ProxyJump strategy requires proxyjump params in connection parameters"
            )

        jump_params = params.proxyjump

        # Step 1: Connect to jump host
        self.jump_client = paramiko.SSHClient()
        self.jump_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        self.jump_client.connect(
            jump_params.hostname,
            username=jump_params.username,
            key_filename=jump_params.identityfile,
        )

        # Step 2: Create tunnel from jump host to target
        jump_transport = self.jump_client.get_transport()
        dest_addr = (params.hostname, 22)
        local_addr = ("localhost", 0)

        channel = jump_transport.open_channel(
            "direct-tcpip", dest_addr, local_addr
        )

        # Step 3: Connect to target through tunnel
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        ssh_client.connect(
            params.hostname,
            username=params.username,
            key_filename=params.identityfile,
            sock=channel,
        )

        return ssh_client

    def cleanup_jump_client(self):
        """Close the jump client if it exists."""
        if self.jump_client:
            self.jump_client.close()
            self.jump_client = None


class SSHConnectionFactory:
    """
    Factory for creating appropriate connection strategy.

    Selects the right strategy based on connection parameters.
    """

    @staticmethod
    def create_strategy(params: SSHConnectionParams) -> SSHConnectionStrategy:
        """
        Select and instantiate the appropriate connection strategy.

        Args:
            params: Connection parameters

        Returns:
            Appropriate SSHConnectionStrategy instance
        """
        if params.proxyjump is not None:
            return ProxyJumpSSHConnection()
        else:
            return DirectSSHConnection()


class SSHClientProvider(ABC):
    """
    Abstract interface for providing SSH/SFTP clients.

    This abstraction enables dependency injection of mock clients in tests.
    """

    @abstractmethod
    def create_connection(
        self,
    ) -> tuple[paramiko.SSHClient, paramiko.SFTPClient]:
        """
        Create and return SSH and SFTP clients.

        Returns:
            Tuple of (ssh_client, sftp_client)
        """
        pass


class ConfigBasedSSHClientProvider(SSHClientProvider):
    """
    Standard SSH client provider that uses SSH config files.

    This is the default provider used in production. It reads SSH config,
    selects the appropriate connection strategy, and establishes connections.

    Attributes:
        remote_alias: SSH host alias
        identityfile: Optional identity file override
        config_parser: SSH config parser instance
        connection_strategy: Connection strategy instance (created during connection)
    """

    def __init__(
        self,
        remote_alias: str,
        identityfile: Optional[str] = None,
        config_file: str = "~/.ssh/config",
    ):
        """
        Initialize config-based SSH client provider.

        Args:
            remote_alias: SSH host alias from config
            identityfile: Optional identity file to use instead of config
            config_file: Path to SSH config file
        """
        self.remote_alias = remote_alias
        self.identityfile = identityfile
        self.config_parser = SSHConfigParser(config_file)
        self.connection_strategy = None

    def create_connection(
        self,
    ) -> tuple[paramiko.SSHClient, paramiko.SFTPClient]:
        """
        Parse config, select strategy, and establish connection.

        Returns:
            Tuple of (ssh_client, sftp_client)
        """
        # Parse SSH config
        params = self.config_parser.parse_host_config(
            self.remote_alias, self.identityfile
        )

        # Create and execute connection strategy
        self.connection_strategy = SSHConnectionFactory.create_strategy(params)
        ssh_client = self.connection_strategy.connect(params)

        # Open SFTP session
        sftp_client = ssh_client.open_sftp()

        return ssh_client, sftp_client

    def cleanup(self):
        """Cleanup any resources (e.g., jump client)."""
        if isinstance(self.connection_strategy, ProxyJumpSSHConnection):
            self.connection_strategy.cleanup_jump_client()


class MockSSHClientProvider(SSHClientProvider):
    """
    Test-only provider that returns pre-configured mock clients.

    This enables full control over SSH/SFTP behavior in tests without
    requiring actual network connections.

    Attributes:
        ssh_client: Pre-configured SSH client (typically a mock)
        sftp_client: Pre-configured SFTP client (typically a mock)
    """

    def __init__(self, ssh_client, sftp_client):
        """
        Initialize mock provider with pre-configured clients.

        Args:
            ssh_client: SSH client to return
            sftp_client: SFTP client to return
        """
        self.ssh_client = ssh_client
        self.sftp_client = sftp_client

    def create_connection(
        self,
    ) -> tuple[paramiko.SSHClient, paramiko.SFTPClient]:
        """
        Return pre-configured mock clients.

        Returns:
            Tuple of (ssh_client, sftp_client)
        """
        return self.ssh_client, self.sftp_client
