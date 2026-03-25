import os
from pathlib import Path

import pytest

from remote_handler.config import get_config, reset_config
from remote_handler.core import file_handler
from remote_handler.core.cluster_handler import ClusterConfig

# Configuration for tests
# Tests require REMOTE_HANDLER_CONFIG_DIR environment variable to be set.
# This directory should contain cluster configuration files (e.g., remote.json, local.json).
#
# Example:
#   export REMOTE_HANDLER_CONFIG_DIR="/path/to/config/files"
#
# The remote.json file should contain configuration for a remote cluster that you
# want to test against. You can copy an existing cluster config (e.g., datalab.json)
# to remote.json for testing purposes.

if os.environ.get("REMOTE_HANDLER_CONFIG_DIR") is None:
    raise Exception(
        "Can't run tests without REMOTE_HANDLER_CONFIG_DIR environment variable set. "
        "Please set REMOTE_HANDLER_CONFIG_DIR to a directory containing cluster configs."
    )


@pytest.fixture(autouse=True, scope="function")
def reset_global_config():
    """Ensure clean config state for each test.

    This fixture automatically resets the global configuration singleton
    after each test to prevent state leakage between tests.
    """
    yield
    reset_config()


@pytest.fixture(scope="session")
def config_dir():
    """Get configuration directory from centralized config system."""
    config = get_config()
    return config.get_config_dir(required=True)


@pytest.fixture(scope="session")
def data_dir():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return Path(dir_path) / "data"


@pytest.fixture(scope="session")
def remote_cluster_config():
    """Cluster configuration for remote testing.

    Session-scoped since ClusterConfig objects are immutable.
    """
    return ClusterConfig.from_file("remote.json")


@pytest.fixture(scope="session")
def local_cluster_config():
    """Cluster configuration for local testing.

    Session-scoped since ClusterConfig objects are immutable.
    """
    return ClusterConfig.from_file("local.json")


@pytest.fixture()
def remote_fs(remote_cluster_config):
    """Remote filesystem using the remote cluster configuration."""
    return file_handler.RemoteFileSystem(
        remote_alias=remote_cluster_config.remote_machine
    )


@pytest.fixture()
def local_fs():
    """Local filesystem (no cluster config needed)."""
    return file_handler.LocalFileSystem()


@pytest.fixture(
    scope="module",
    params=[
        pytest.param("remote", marks=pytest.mark.remote, id="remote"),
        pytest.param("local", marks=pytest.mark.local, id="local"),
    ],
)
def env_type(request):
    """Type of test environment (remote or local).

    This fixture parametrizes tests to run against both remote and local environments.
    Use pytest markers to run only specific environments:
        pytest -m remote  # Run only remote tests
        pytest -m local   # Run only local tests
    """
    return request.param


@pytest.fixture
def filesystem(env_type, remote_fs, local_fs):
    """Get filesystem based on environment type.

    Automatically provides the appropriate filesystem (RemoteFileSystem or
    LocalFileSystem) based on the env_type parametrization.

    Note: Function-scoped to match remote_fs/local_fs scope, which may
    maintain stateful connections.
    """
    return remote_fs if env_type == "remote" else local_fs


@pytest.fixture
def cluster_config(env_type, remote_cluster_config, local_cluster_config):
    """Get cluster config based on environment type.

    Automatically provides the appropriate cluster configuration based on
    the env_type parametrization. Ensures filesystem and config are always
    properly paired.

    Note: Function-scoped to match filesystem scope for consistency.
    """
    return (
        remote_cluster_config if env_type == "remote" else local_cluster_config
    )


@pytest.fixture
def fs_with_target(
    env_type, remote_fs, remote_cluster_config, local_fs, tmp_path
):
    """Parametrized fixture providing (filesystem, target_dir) pairs.

    Returns a tuple of (filesystem, target_directory) appropriate for the
    current test environment. This ensures consistent handling of both
    remote and local test scenarios.

    For remote: Uses RemoteFileSystem with configured savepath or temp directory
    For local: Uses LocalFileSystem with pytest's tmp_path
    """
    if env_type == "remote":
        target_dir = (
            Path(remote_cluster_config.savepath)
            if remote_cluster_config.savepath
            else remote_fs.create_tempdir()
        )
        return remote_fs, target_dir
    return local_fs, tmp_path
