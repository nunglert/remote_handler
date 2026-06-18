import os
import shutil
from pathlib import Path

import pytest

from remote_handler.core import file_handler


@pytest.fixture()
def local_dir(data_dir):
    """Directory containing test assets."""
    return data_dir / "assets" / "TestFileSystem"


@pytest.fixture()
def remote_target_dir(remote_fs, remote_cluster_config):
    """Target directory for remote filesystem tests.

    Verifies the configured savepath actually exists on the remote up front,
    so a misconfigured savepath fails with a clear message here instead of a
    bare "No such file" deep inside an SFTP transfer.
    """
    savepath = remote_cluster_config.savepath
    if not savepath:
        return remote_fs.create_tempdir()
    if not remote_fs.exists(savepath):
        pytest.fail(
            "Configured savepath does not exist on "
            f"{remote_cluster_config.remote_machine}: {savepath}\n"
            "Create it on the remote, or set an existing absolute path as "
            "the cluster config 'savepath'."
        )
    return Path(savepath)


class TestFileSystem:
    """Test filesystem operations for both local and remote filesystems."""

    def test_put(self, fs_with_target, local_dir):
        """Test uploading a single file."""
        fs, target_dir = fs_with_target
        filename = "single_file.txt"
        fs.put(local_dir / filename, target_dir / filename)
        assert fs.exists(target_dir / filename)

    def test_get(self, fs_with_target, local_dir):
        """Test downloading a single file."""
        fs, target_dir = fs_with_target
        filename = "single_file.txt"
        filename2 = "single_file2.txt"

        if os.path.exists(local_dir / filename2):
            os.remove(local_dir / filename2)

        fs.put(local_dir / filename, target_dir / filename2)
        fs.get(target_dir / filename2, local_dir / filename2)

        assert os.path.exists(local_dir / filename2)

    def test_put_dir(self, fs_with_target, local_dir):
        """Test uploading a directory."""
        fs, target_dir = fs_with_target
        dirname = "directory_with_contents"
        fs.put_dir(local_dir / dirname, target_dir / dirname)
        assert fs.exists(target_dir / dirname)

    def test_get_dir(self, fs_with_target, local_dir):
        """Test downloading a directory."""
        fs, target_dir = fs_with_target
        dirname = "directory_with_contents"
        dirname2 = "directory_with_contents2"

        if os.path.exists(local_dir / dirname2):
            shutil.rmtree(local_dir / dirname2)

        fs.put_dir(local_dir / dirname, target_dir / dirname2)
        fs.get_dir(target_dir / dirname2, local_dir / dirname2)

        assert os.path.exists(local_dir / dirname2)

    def test_create_tempdir(self, fs_with_target):
        """Test creating a temporary directory."""
        fs, _ = fs_with_target
        temppath = fs.create_tempdir()
        assert fs.exists(temppath)


@pytest.mark.remote
def test_reconnect(remote_fs, remote_target_dir, local_dir):
    """Test reconnection for remote filesystem."""
    filename = "single_file.txt"
    remote_fs.put(local_dir / filename, remote_target_dir / filename)
    assert remote_fs.exists(remote_target_dir / filename)

    remote_fs.sftp_client.close()

    remote_fs.put(local_dir / filename, remote_target_dir / filename)
    assert remote_fs.exists(remote_target_dir / filename)


# ===== NEW MOCK-BASED TESTS =====
# These tests demonstrate the new dependency injection capability
# for testing RemoteFileSystem with mock SSH/SFTP clients


from unittest.mock import MagicMock

import paramiko

from remote_handler.core.ssh_connection import MockSSHClientProvider


class TestRemoteFileSystemWithMocks:
    """Test RemoteFileSystem using mock SSH/SFTP clients."""

    def test_put_with_mock_client(self):
        """Test that put() correctly calls SFTP client's put method."""
        mock_ssh = MagicMock(spec=paramiko.SSHClient)
        mock_sftp = MagicMock(spec=paramiko.SFTPClient)
        mock_sftp.stat.side_effect = FileNotFoundError  # For exists() checks

        mock_provider = MockSSHClientProvider(mock_ssh, mock_sftp)

        fs = file_handler.RemoteFileSystem(
            remote_alias="test_host", ssh_client_provider=mock_provider
        )

        fs.put("/local/file.txt", "/remote/file.txt")

        mock_sftp.put.assert_called_once_with(
            "/local/file.txt", "/remote/file.txt"
        )

    def test_get_with_mock_client(self):
        """Test that get() correctly calls SFTP client's get method."""
        mock_ssh = MagicMock(spec=paramiko.SSHClient)
        mock_sftp = MagicMock(spec=paramiko.SFTPClient)
        mock_sftp.stat.side_effect = FileNotFoundError

        mock_provider = MockSSHClientProvider(mock_ssh, mock_sftp)

        fs = file_handler.RemoteFileSystem(
            remote_alias="test_host", ssh_client_provider=mock_provider
        )

        fs.get("/remote/file.txt", "/local/file.txt")

        mock_sftp.get.assert_called_once_with(
            "/remote/file.txt", "/local/file.txt"
        )

    def test_exists_with_mock_client(self):
        """Test exists() method with mock client."""
        mock_ssh = MagicMock(spec=paramiko.SSHClient)
        mock_sftp = MagicMock(spec=paramiko.SFTPClient)

        # Setup mock to return success for stat (file exists)
        mock_sftp.stat.return_value = MagicMock()

        mock_provider = MockSSHClientProvider(mock_ssh, mock_sftp)

        fs = file_handler.RemoteFileSystem(
            remote_alias="test_host", ssh_client_provider=mock_provider
        )

        result = fs.exists("/remote/file.txt")

        assert result is True
        mock_sftp.stat.assert_called_once_with("/remote/file.txt")

    def test_exists_file_not_found(self):
        """Test exists() returns False when file doesn't exist."""
        mock_ssh = MagicMock(spec=paramiko.SSHClient)
        mock_sftp = MagicMock(spec=paramiko.SFTPClient)

        # Setup mock to raise FileNotFoundError
        mock_sftp.stat.side_effect = FileNotFoundError

        mock_provider = MockSSHClientProvider(mock_ssh, mock_sftp)

        fs = file_handler.RemoteFileSystem(
            remote_alias="test_host", ssh_client_provider=mock_provider
        )

        result = fs.exists("/remote/file.txt")

        assert result is False

    def test_reconnect_on_failure_with_mock(self):
        """Test @reconnect_on_failure decorator with mock clients."""
        mock_ssh = MagicMock(spec=paramiko.SSHClient)
        mock_sftp = MagicMock(spec=paramiko.SFTPClient)
        mock_sftp.stat.side_effect = FileNotFoundError

        # Configure mock to fail once, then succeed
        mock_sftp.put.side_effect = [
            paramiko.SSHException("Connection lost"),
            None,  # Success on retry
        ]

        mock_provider = MockSSHClientProvider(mock_ssh, mock_sftp)

        fs = file_handler.RemoteFileSystem(
            remote_alias="test_host", ssh_client_provider=mock_provider
        )

        # Should succeed after automatic reconnect
        fs.put("/local/file.txt", "/remote/file.txt")

        # Verify it was called twice (initial + retry)
        assert mock_sftp.put.call_count == 2

    def test_mkdir_with_mock_client(self):
        """Test mkdir() with mock client."""
        mock_ssh = MagicMock(spec=paramiko.SSHClient)
        mock_sftp = MagicMock(spec=paramiko.SFTPClient)
        mock_sftp.stat.side_effect = FileNotFoundError

        mock_provider = MockSSHClientProvider(mock_ssh, mock_sftp)

        fs = file_handler.RemoteFileSystem(
            remote_alias="test_host", ssh_client_provider=mock_provider
        )

        fs.mkdir("/remote/newdir")

        mock_sftp.mkdir.assert_called_once_with("/remote/newdir")

    def test_disconnect_calls_provider_cleanup(self):
        """Test that disconnect calls provider cleanup."""
        mock_ssh = MagicMock(spec=paramiko.SSHClient)
        mock_sftp = MagicMock(spec=paramiko.SFTPClient)
        mock_sftp.stat.side_effect = FileNotFoundError

        mock_provider = MockSSHClientProvider(mock_ssh, mock_sftp)

        fs = file_handler.RemoteFileSystem(
            remote_alias="test_host", ssh_client_provider=mock_provider
        )

        fs._disconnect()

        # Verify SFTP and SSH clients were closed
        mock_sftp.close.assert_called_once()
        mock_ssh.close.assert_called_once()
