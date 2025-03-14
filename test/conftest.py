import pytest
import os
from pathlib import Path

# In order to test the functionality of RemoteFileSystem, we need a path to
# configuration files set. You should set this as an environmental variable
# outside python, like:
#
#   export REMOTE_HANDLER_CONFIG_DIR="/path/to/config/files"
# 
# I'm just hard coding it here for convenience. 
if os.environ.get("REMOTE_HANDLER_CONFIG_DIR") is None:
    os.environ["REMOTE_HANDLER_CONFIG_DIR"] = "/home/nico.unglert/code/jaxnest_dev/experiments/remote_handler/examples/config_files"


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