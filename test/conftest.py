import os
from pathlib import Path

import pytest
from pytest_lazyfixture import lazy_fixture

# In order to test the functionality of RemoteFileSystem, we need a path to
# configuration files set. You should set this as an environmental variable
# outside python, like:
#
#   export REMOTE_HANDLER_CONFIG_DIR="/path/to/config/files"
#
# I'm just hard coding it here for convenience to avoid setting an env variable
# for VScode debugger..
#
# The tests including a RemoteFileSystem will look for a `remote.json` inside
# REMOTE_HANDLER_CONFIG_DIR. Say, you are locally on machine1 (e.g. lugus) and
# want to run the tests, then you could run the tests for another machine2
# (e.g. datalab) by copying your config for machine1 to `remote.json`
# (e.g. cp datalab.json remote.json)

if os.environ.get("REMOTE_HANDLER_CONFIG_DIR") is None:
    # os.environ["REMOTE_HANDLER_CONFIG_DIR"] = "/home/nico.unglert/code/jaxnest_dev/experiments/remote_handler/examples/config_files"
    raise Exception(
        "Can't run tests without REMOTE_HANDLER_CONFIG_DIR environment variable set"
    )
    os.environ[
        "REMOTE_HANDLER_CONFIG_DIR"
    ] = "/home/nunglert/code/remote_handler/experiments/config_files"

from remote_handler.core import file_handler
from remote_handler.core.cluster_config import ClusterConfig


@pytest.fixture(scope="session")
def config_dir():
    config_path = os.environ.get("REMOTE_HANDLER_CONFIG_DIR")
    if config_path is None:
        raise Exception(
            "Set REMOTE_HANDLER_CONFIG_DIR environment variable, where config "
            "files for different clusters are stored. See e.g. in "
            "examples/config_files"
        )
    return Path(config_path)


@pytest.fixture(scope="session")
def data_dir():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return Path(dir_path) / "data"


@pytest.fixture()
def remote_cluster_config():
    """Cluster configuration for remote testing."""
    return ClusterConfig.from_file("remote.json")


@pytest.fixture()
def local_cluster_config():
    """Cluster configuration for local testing."""
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
        pytest.param(
            (
                "remote",
                lazy_fixture("remote_fs"),
                lazy_fixture("remote_cluster_config"),
            ),
            marks=[pytest.mark.remote],
            id="remote",
        ),
        pytest.param(
            (
                "local",
                lazy_fixture("local_fs"),
                lazy_fixture("local_cluster_config"),
            ),
            marks=[pytest.mark.local],
            id="local",
        ),
    ],
)
def test_environment(request):
    """Provides paired filesystem and cluster_config to avoid invalid combinations."""
    env_type, fs, config = request.param
    return {"type": env_type, "filesystem": fs, "cluster_config": config}


@pytest.fixture(scope="module")
def filesystem(test_environment):
    """Extract filesystem from the paired test environment."""
    return test_environment["filesystem"]


@pytest.fixture(scope="module")
def cluster_config(test_environment):
    """Extract cluster_config from the paired test environment."""
    return test_environment["cluster_config"]
