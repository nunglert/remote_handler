import os
import shutil
from pathlib import Path

import pytest
from pytest_lazyfixture import lazy_fixture

from remote_handler.core import file_handler


@pytest.fixture()
def remote_fs(config_dir):
    slurm_config_file = config_dir / "remote.json"
    fs = file_handler.RemoteFileSystem(slurm_config_file=slurm_config_file)
    target_dir = fs.slurm_config.get("savepath")
    if target_dir is None:
        target_dir = fs.create_tempdir()

    return fs, Path(target_dir)


@pytest.fixture()
def local_fs(data_dir):
    fs = file_handler.LocalFileSystem()
    target_dir = data_dir / "temp" / "TestFileSystem" / "temp"

    if fs.exists(target_dir):
        fs.rmtree(target_dir, ignore_errors=True)
    os.makedirs(target_dir, exist_ok=True)

    return fs, target_dir


@pytest.fixture()
def local_dir(data_dir):
    return data_dir / "assets" / "TestFileSystem"


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(
            lazy_fixture("remote_fs"),
            marks=[pytest.mark.remote],
        ),
        pytest.param(
            lazy_fixture("local_fs"),
            marks=[pytest.mark.local],
        ),
    ],
)
def instance(request):
    return request.param


class TestFileSystem:
    def test_put(self, instance, local_dir):
        fs, target_dir = instance
        filename = "single_file.txt"
        fs.put(local_dir / filename, target_dir / filename)
        assert fs.exists(target_dir / filename)

    def test_get(self, instance, local_dir):
        fs, target_dir = instance
        filename = "single_file.txt"
        filename2 = "single_file2.txt"

        if os.path.exists(local_dir / filename2):
            os.remove(local_dir / filename2)

        fs.put(local_dir / filename, target_dir / filename2)
        fs.get(target_dir / filename2, local_dir / filename2)

        assert os.path.exists(local_dir / filename2)

    def test_put_dir(self, instance, local_dir):
        fs, target_dir = instance
        dirname = "directory_with_contents"

        fs.put_dir(local_dir / dirname, target_dir / dirname)

        assert fs.exists(target_dir / dirname)

    def test_get_dir(self, instance, local_dir):
        fs, target_dir = instance
        dirname = "directory_with_contents"
        dirname2 = "directory_with_contents2"

        if os.path.exists(local_dir / dirname2):
            shutil.rmtree(local_dir / dirname2)

        fs.put_dir(local_dir / dirname, target_dir / dirname2)
        fs.get_dir(target_dir / dirname2, local_dir / dirname2)

        assert os.path.exists(local_dir / dirname2)

    def test_create_tempdir(self, instance):
        fs, target_dir = instance
        temppath = fs.create_tempdir()
        print(temppath)
        assert fs.exists(temppath)


def test_reconnect(remote_fs, local_dir):
    fs, target_dir = remote_fs
    filename = "single_file.txt"
    fs.put(local_dir / filename, target_dir / filename)
    assert fs.exists(target_dir / filename)

    fs.sftp_client.close()

    fs.put(local_dir / filename, target_dir / filename)
    assert fs.exists(target_dir / filename)


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
