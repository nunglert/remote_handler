import os
from pathlib import Path

import pytest
from pytest_lazyfixture import lazy_fixture

from remote_handler import file_handler

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
    os.environ[
        "REMOTE_HANDLER_CONFIG_DIR"
    ] = "/home/nunglert/code/remote_handler/experiments/config_files"


# def pytest_collection_modifyitems(config, items):
#     if config.getoption("-m"):
#         # If marker explicitly provided, don't interfere.
#         return

#     remote = pytest.mark.skip(reason="Skipping remote tests by default.")
#     for item in items:
#         if "remote" in item.keywords:
#             item.add_marker(remote)


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
def remote_fs(config_dir):
    slurm_config_file = config_dir / "remote.json"
    return file_handler.RemoteFileSystem(slurm_config_file=slurm_config_file)


@pytest.fixture()
def local_fs(config_dir):
    slurm_config_file = config_dir / "local.json"
    return file_handler.LocalFileSystem(slurm_config_file=slurm_config_file)


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
def filesystem(request):
    return request.param
