import pytest
import os
from pathlib import Path

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
    os.environ["REMOTE_HANDLER_CONFIG_DIR"] = "/home/nico.unglert/code/jaxnest_dev/experiments/remote_handler/examples/config_files"
    os.environ["REMOTE_HANDLER_CONFIG_DIR"] = "/home/nunglert/code/remote_handler/experiments/config_files"


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

