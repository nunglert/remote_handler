"""
Unit tests for SSH connection components.

These tests verify the modular SSH connection components without requiring
actual SSH connections, making them fast and reliable.
"""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import paramiko
import pytest

from remote_handler.core.ssh_connection import (
    ConfigBasedSSHClientProvider,
    DirectSSHConnection,
    MockSSHClientProvider,
    ProxyJumpParams,
    ProxyJumpSSHConnection,
    SSHConfigParser,
    SSHConnectionFactory,
    SSHConnectionParams,
    SSHConnectionStrategy,
)
from remote_handler.exceptions import RemoteConnectionError


class TestSSHConnectionParams:
    """Test SSH connection parameter dataclasses."""

    def test_connection_params_basic(self):
        """Test basic SSHConnectionParams creation."""
        params = SSHConnectionParams(
            hostname="test.example.com", username="testuser"
        )
        assert params.hostname == "test.example.com"
        assert params.username == "testuser"
        assert params.identityfile is None
        assert params.proxyjump is None

    def test_connection_params_with_identity(self):
        """Test SSHConnectionParams with identity file."""
        params = SSHConnectionParams(
            hostname="test.example.com",
            username="testuser",
            identityfile="~/.ssh/test_key",
        )
        assert params.identityfile == "~/.ssh/test_key"

    def test_connection_params_with_proxyjump(self):
        """Test SSHConnectionParams with ProxyJump."""
        jump_params = ProxyJumpParams(
            hostname="jump.example.com", username="jumpuser"
        )
        params = SSHConnectionParams(
            hostname="test.example.com",
            username="testuser",
            proxyjump=jump_params,
        )
        assert params.proxyjump.hostname == "jump.example.com"
        assert params.proxyjump.username == "jumpuser"


class TestSSHConfigParser:
    """Test SSH config file parsing."""

    def test_parse_direct_connection(self, tmp_path):
        """Test parsing SSH config for direct connection."""
        config_file = tmp_path / "ssh_config"
        config_file.write_text(
            """
Host testhost
    HostName test.example.com
    User testuser
    IdentityFile ~/.ssh/test_key
"""
        )

        parser = SSHConfigParser(str(config_file))
        params = parser.parse_host_config("testhost")

        assert params.hostname == "test.example.com"
        assert params.username == "testuser"
        # paramiko expands ~ in identity file paths
        import os

        assert params.identityfile == os.path.expanduser("~/.ssh/test_key")
        assert params.proxyjump is None

    def test_parse_proxyjump_connection(self, tmp_path):
        """Test parsing SSH config with ProxyJump."""
        config_file = tmp_path / "ssh_config"
        config_file.write_text(
            """
Host jumphost
    HostName jump.example.com
    User jumpuser
    IdentityFile ~/.ssh/jump_key

Host targethost
    HostName target.example.com
    User targetuser
    IdentityFile ~/.ssh/target_key
    ProxyJump jumphost
"""
        )

        parser = SSHConfigParser(str(config_file))
        params = parser.parse_host_config("targethost")

        import os

        assert params.hostname == "target.example.com"
        assert params.username == "targetuser"
        # paramiko expands ~ in identity file paths
        assert params.identityfile == os.path.expanduser("~/.ssh/target_key")
        assert params.proxyjump is not None
        assert params.proxyjump.hostname == "jump.example.com"
        assert params.proxyjump.username == "jumpuser"
        assert params.proxyjump.identityfile == os.path.expanduser(
            "~/.ssh/jump_key"
        )

    def test_parse_with_identityfile_override(self, tmp_path):
        """Test that identityfile override works."""
        config_file = tmp_path / "ssh_config"
        config_file.write_text(
            """
Host testhost
    HostName test.example.com
    User testuser
    IdentityFile ~/.ssh/config_key
"""
        )

        parser = SSHConfigParser(str(config_file))
        params = parser.parse_host_config(
            "testhost", identityfile_override="~/.ssh/override_key"
        )

        assert params.identityfile == "~/.ssh/override_key"

    def test_parse_nonexistent_config(self):
        """Test that nonexistent config file raises FileNotFoundError."""
        parser = SSHConfigParser("/nonexistent/config")
        with pytest.raises(FileNotFoundError):
            parser.parse_host_config("testhost")


class TestConnectionStrategies:
    """Test SSH connection strategies."""

    @patch("paramiko.SSHClient")
    def test_direct_connection_strategy(self, mock_ssh_class):
        """Test DirectSSHConnection strategy."""
        mock_client = MagicMock()
        mock_ssh_class.return_value = mock_client

        strategy = DirectSSHConnection()
        params = SSHConnectionParams(
            hostname="test.example.com",
            username="testuser",
            identityfile="~/.ssh/test_key",
        )

        result = strategy.connect(params)

        assert result == mock_client
        mock_client.set_missing_host_key_policy.assert_called_once()
        mock_client.connect.assert_called_once_with(
            "test.example.com",
            port=22,
            username="testuser",
            key_filename="~/.ssh/test_key",
        )

    @patch("paramiko.SSHClient")
    def test_proxyjump_connection_strategy(self, mock_ssh_class):
        """Test ProxyJumpSSHConnection strategy."""
        # Create two mock clients (jump and target)
        jump_client = MagicMock()
        target_client = MagicMock()
        mock_ssh_class.side_effect = [jump_client, target_client]

        # Setup jump client transport and channel
        jump_transport = MagicMock()
        channel = MagicMock()
        jump_client.get_transport.return_value = jump_transport
        jump_transport.open_channel.return_value = channel

        strategy = ProxyJumpSSHConnection()
        params = SSHConnectionParams(
            hostname="target.example.com",
            username="targetuser",
            identityfile="~/.ssh/target_key",
            proxyjump=ProxyJumpParams(
                hostname="jump.example.com",
                username="jumpuser",
                identityfile="~/.ssh/jump_key",
            ),
        )

        result = strategy.connect(params)

        # Verify jump client connection
        jump_client.connect.assert_called_once_with(
            "jump.example.com",
            port=22,
            username="jumpuser",
            key_filename="~/.ssh/jump_key",
        )

        # Verify channel creation
        jump_transport.open_channel.assert_called_once_with(
            "direct-tcpip", ("target.example.com", 22), ("localhost", 0)
        )

        # Verify target client connection through channel
        target_client.connect.assert_called_once_with(
            "target.example.com",
            port=22,
            username="targetuser",
            key_filename="~/.ssh/target_key",
            sock=channel,
        )

        assert result == target_client
        assert strategy.jump_client == jump_client

    def test_proxyjump_without_params_raises_error(self):
        """Test that ProxyJump strategy fails without proxyjump params."""
        strategy = ProxyJumpSSHConnection()
        params = SSHConnectionParams(
            hostname="target.example.com", username="targetuser"
        )

        with pytest.raises(
            RemoteConnectionError, match="ProxyJump strategy requires"
        ):
            strategy.connect(params)

    def test_proxyjump_cleanup(self):
        """Test ProxyJump jump client cleanup."""
        strategy = ProxyJumpSSHConnection()
        mock_jump_client = MagicMock()
        strategy.jump_client = mock_jump_client

        strategy.cleanup_jump_client()

        mock_jump_client.close.assert_called_once()
        assert strategy.jump_client is None

    def test_proxyjump_cleanup_when_none(self):
        """Test ProxyJump cleanup when jump_client is None."""
        strategy = ProxyJumpSSHConnection()
        strategy.jump_client = None

        # Should not raise error
        strategy.cleanup_jump_client()


class TestSSHConnectionFactory:
    """Test SSH connection factory."""

    def test_factory_selects_direct_connection(self):
        """Test factory selects DirectSSHConnection for simple params."""
        params = SSHConnectionParams(
            hostname="test.example.com", username="testuser"
        )

        strategy = SSHConnectionFactory.create_strategy(params)

        assert isinstance(strategy, DirectSSHConnection)

    def test_factory_selects_proxyjump_connection(self):
        """Test factory selects ProxyJumpSSHConnection when proxyjump present."""
        params = SSHConnectionParams(
            hostname="target.example.com",
            username="targetuser",
            proxyjump=ProxyJumpParams(
                hostname="jump.example.com", username="jumpuser"
            ),
        )

        strategy = SSHConnectionFactory.create_strategy(params)

        assert isinstance(strategy, ProxyJumpSSHConnection)


class TestSSHClientProviders:
    """Test SSH client provider implementations."""

    def test_mock_provider_returns_clients(self):
        """Test MockSSHClientProvider returns pre-configured clients."""
        mock_ssh = MagicMock()
        mock_sftp = MagicMock()

        provider = MockSSHClientProvider(mock_ssh, mock_sftp)
        ssh_client, sftp_client = provider.create_connection()

        assert ssh_client == mock_ssh
        assert sftp_client == mock_sftp

    @patch("remote_handler.core.ssh_connection.SSHConfigParser")
    @patch("remote_handler.core.ssh_connection.SSHConnectionFactory")
    def test_config_based_provider_creates_connection(
        self, mock_factory, mock_parser_class
    ):
        """Test ConfigBasedSSHClientProvider creates connection."""
        # Setup mocks
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser

        mock_params = SSHConnectionParams(
            hostname="test.example.com", username="testuser"
        )
        mock_parser.parse_host_config.return_value = mock_params

        mock_strategy = MagicMock()
        mock_factory.create_strategy.return_value = mock_strategy

        mock_ssh_client = MagicMock()
        mock_sftp_client = MagicMock()
        mock_strategy.connect.return_value = mock_ssh_client
        mock_ssh_client.open_sftp.return_value = mock_sftp_client

        # Create provider and connect
        provider = ConfigBasedSSHClientProvider(
            remote_alias="testhost", identityfile="/path/to/key"
        )
        ssh_client, sftp_client = provider.create_connection()

        # Verify config parsing
        mock_parser.parse_host_config.assert_called_once_with(
            "testhost", "/path/to/key"
        )

        # Verify strategy creation and connection
        mock_factory.create_strategy.assert_called_once_with(mock_params)
        mock_strategy.connect.assert_called_once_with(mock_params)

        # Verify SFTP session opened
        mock_ssh_client.open_sftp.assert_called_once()

        assert ssh_client == mock_ssh_client
        assert sftp_client == mock_sftp_client

    def test_config_based_provider_cleanup_direct(self):
        """Test cleanup with DirectSSHConnection (no-op)."""
        provider = ConfigBasedSSHClientProvider(remote_alias="testhost")
        provider.connection_strategy = DirectSSHConnection()

        # Should not raise error
        provider.cleanup()

    def test_config_based_provider_cleanup_proxyjump(self):
        """Test cleanup with ProxyJumpSSHConnection."""
        provider = ConfigBasedSSHClientProvider(remote_alias="testhost")
        mock_strategy = MagicMock(spec=ProxyJumpSSHConnection)
        provider.connection_strategy = mock_strategy

        provider.cleanup()

        mock_strategy.cleanup_jump_client.assert_called_once()


# ===== INTEGRATION TESTS WITH REAL SSH CONNECTIONS =====
# These tests use the actual remote_machine from config files
# They require SSH access and are marked with @pytest.mark.remote


@pytest.mark.remote
class TestSSHConnectionIntegration:
    """Integration tests using real SSH connections."""

    def test_parse_real_ssh_config(self, remote_cluster_config):
        """Test parsing real SSH config for remote machine."""
        parser = SSHConfigParser()  # Uses default ~/.ssh/config
        params = parser.parse_host_config(remote_cluster_config.remote_machine)

        # Verify we got valid connection params
        assert params.hostname is not None
        assert params.username is not None
        # ProxyJump may or may not be present depending on config
        print(f"Parsed SSH config for {remote_cluster_config.remote_machine}:")
        print(f"  Hostname: {params.hostname}")
        print(f"  Username: {params.username}")
        print(f"  Identity file: {params.identityfile}")
        print(f"  ProxyJump: {params.proxyjump is not None}")

    def test_connection_factory_with_real_config(self, remote_cluster_config):
        """Test that factory selects correct strategy for real config."""
        parser = SSHConfigParser()
        params = parser.parse_host_config(remote_cluster_config.remote_machine)

        strategy = SSHConnectionFactory.create_strategy(params)

        # Verify correct strategy selected
        if params.proxyjump is not None:
            assert isinstance(strategy, ProxyJumpSSHConnection)
            print(
                f"ProxyJump connection strategy selected for {remote_cluster_config.remote_machine}"
            )
        else:
            assert isinstance(strategy, DirectSSHConnection)
            print(
                f"Direct connection strategy selected for {remote_cluster_config.remote_machine}"
            )

    def test_config_based_provider_real_connection(
        self, remote_cluster_config
    ):
        """Test ConfigBasedSSHClientProvider with real SSH connection."""
        provider = ConfigBasedSSHClientProvider(
            remote_alias=remote_cluster_config.remote_machine
        )

        # Create real SSH connection
        ssh_client, sftp_client = provider.create_connection()

        try:
            # Verify clients are connected
            assert ssh_client is not None
            assert sftp_client is not None
            assert ssh_client.get_transport() is not None
            assert ssh_client.get_transport().is_active()

            # Test basic SFTP operation
            remote_home = sftp_client.normalize(".")
            print(
                f"Successfully connected to {remote_cluster_config.remote_machine}"
            )
            print(f"Remote home directory: {remote_home}")

        finally:
            # Cleanup
            sftp_client.close()
            ssh_client.close()
            provider.cleanup()

    def test_direct_connection_real(self, remote_cluster_config):
        """Test DirectSSHConnection with real SSH config."""
        parser = SSHConfigParser()
        params = parser.parse_host_config(remote_cluster_config.remote_machine)

        # Only test if direct connection (no proxyjump)
        if params.proxyjump is None:
            strategy = DirectSSHConnection()
            ssh_client = strategy.connect(params)

            try:
                assert ssh_client.get_transport().is_active()
                print(f"Direct connection successful to {params.hostname}")
            finally:
                ssh_client.close()
        else:
            pytest.skip(
                f"{remote_cluster_config.remote_machine} uses ProxyJump"
            )

    def test_proxyjump_connection_real(self, remote_cluster_config):
        """Test ProxyJumpSSHConnection with real SSH config."""
        parser = SSHConfigParser()
        params = parser.parse_host_config(remote_cluster_config.remote_machine)

        # Only test if proxyjump connection
        if params.proxyjump is not None:
            strategy = ProxyJumpSSHConnection()
            ssh_client = strategy.connect(params)

            try:
                assert ssh_client.get_transport().is_active()
                assert strategy.jump_client is not None
                assert strategy.jump_client.get_transport().is_active()
                print(
                    f"ProxyJump connection successful to {params.hostname} via {params.proxyjump.hostname}"
                )
            finally:
                ssh_client.close()
                strategy.cleanup_jump_client()
        else:
            pytest.skip(
                f"{remote_cluster_config.remote_machine} does not use ProxyJump"
            )

    def test_provider_reconnect(self, remote_cluster_config):
        """Test that provider can reconnect after disconnection."""
        provider = ConfigBasedSSHClientProvider(
            remote_alias=remote_cluster_config.remote_machine
        )

        # First connection
        ssh_client1, sftp_client1 = provider.create_connection()
        transport1 = ssh_client1.get_transport()
        assert transport1.is_active()

        # Close clients
        sftp_client1.close()
        ssh_client1.close()
        provider.cleanup()

        # Reconnect
        ssh_client2, sftp_client2 = provider.create_connection()
        transport2 = ssh_client2.get_transport()

        try:
            assert transport2.is_active()
            print(
                f"Reconnection successful to {remote_cluster_config.remote_machine}"
            )
        finally:
            sftp_client2.close()
            ssh_client2.close()
            provider.cleanup()
