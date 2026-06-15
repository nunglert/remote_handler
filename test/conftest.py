import json
import os
from pathlib import Path

import pytest

from remote_handler.config import get_config, reset_config
from remote_handler.core import file_handler
from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.exceptions import ConfigurationError

# -------------------------------------------------------------------------
# Test tiers
# -------------------------------------------------------------------------
# The suite is designed to run irrespective of whether the package has been
# configured:
#
#   * Unit tests   -- need no configuration; always run.
#   * Local tests  -- run against the local filesystem using a *synthesized*
#                     ClusterConfig (no config files required).
#   * Remote tests -- need a live cluster. They request the
#                     ``remote_cluster_config`` fixture, which SKIPS with a
#                     reason when REMOTE_HANDLER_CONFIG_DIR / remote.json is
#                     not available, so the rest of the suite still runs.
#   * Integration  -- marked ``integration`` and deselected by default
#                     (see pyproject.toml). Run with ``pytest -m integration``.
#
# To exercise the remote tier, point REMOTE_HANDLER_CONFIG_DIR at a directory
# containing a ``remote.json`` cluster config:
#
#     export REMOTE_HANDLER_CONFIG_DIR="/path/to/config/files"
#
# Without it, the remote-parametrized tests simply report as skipped.


@pytest.fixture(autouse=True, scope="function")
def reset_global_config(monkeypatch, tmp_path):
    """Ensure clean, hermetic config state for each test.

    Isolates the persistent-config location to a temp dir so tests never read
    a developer's real ``~/.config/remote_handler``, and resets the global
    configuration singleton before and after each test.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    reset_config()
    yield
    reset_config()


@pytest.fixture(scope="session")
def data_dir():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return Path(dir_path) / "data"


@pytest.fixture(scope="session")
def config_dir():
    """Configuration directory, or skip if the package is not configured."""
    try:
        return get_config().get_config_dir(required=True)
    except ConfigurationError as exc:
        pytest.skip(f"No config directory configured: {exc}")


def _remote_test_cluster_name() -> str:
    """Name of the cluster config the remote tier targets.

    Reads ``test_cluster`` from ``<config_dir>/config.json`` (set via
    ``remote_handler configure``), falling back to ``remote``. Read from the
    config dir rather than the global config singleton because the remote tier
    is driven by ``REMOTE_HANDLER_CONFIG_DIR`` (the autouse fixture isolates
    the XDG persistent config per-test).
    """
    try:
        config_dir = get_config().get_config_dir(required=True)
    except ConfigurationError:
        return "remote"
    cfgfile = config_dir / "config.json"
    if cfgfile.exists():
        try:
            name = json.loads(cfgfile.read_text()).get("test_cluster")
            if name:
                return name
        except (OSError, json.JSONDecodeError):
            pass
    return "remote"


@pytest.fixture(scope="session")
def remote_cluster_config():
    """Cluster configuration for the live-remote tier.

    The target cluster is configurable (``test_cluster`` in the global config,
    via ``remote_handler configure``); defaults to ``remote.json``. Skips with
    a reason when the package is not configured, so the rest of the suite still
    runs without a cluster. Session-scoped since ClusterConfig is immutable.
    """
    name = _remote_test_cluster_name()
    try:
        return ClusterConfig.from_file(f"{name}.json")
    except ConfigurationError as exc:
        pytest.skip(
            "Remote cluster config unavailable -- set "
            f"REMOTE_HANDLER_CONFIG_DIR to a directory containing {name}.json "
            f"to run remote tests ({exc})"
        )


@pytest.fixture(scope="session")
def local_cluster_config(tmp_path_factory):
    """Synthesized local cluster configuration -- needs no config files.

    ``savepath`` points at a session-scoped temp directory and ``template`` is
    a built-in one, so local-tier tests run irrespective of configuration.
    """
    savepath = tmp_path_factory.mktemp("local_cluster_savepath")
    return ClusterConfig(
        remote_machine="dirona",
        partition="normal",
        conda_env="base",
        conda_source="~/miniconda3/bin/activate",
        savepath=str(savepath),
        template="dirona.sh.j2",
    )


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

    Parametrizes tests over both environments. Filter with markers:
        pytest -m remote  # only remote (needs a configured cluster)
        pytest -m local   # only local
    """
    return request.param


@pytest.fixture
def filesystem(request, env_type):
    """Filesystem for the active env_type.

    Resolves only the relevant fixture via ``getfixturevalue`` so the ``local``
    parametrization never pulls in (and skips on) the remote fixtures.
    """
    if env_type == "remote":
        return request.getfixturevalue("remote_fs")
    return request.getfixturevalue("local_fs")


@pytest.fixture
def cluster_config(request, env_type):
    """Cluster config for the active env_type (resolved lazily)."""
    if env_type == "remote":
        return request.getfixturevalue("remote_cluster_config")
    return request.getfixturevalue("local_cluster_config")


@pytest.fixture
def fs_with_target(request, env_type, tmp_path):
    """Parametrized fixture providing (filesystem, target_dir) pairs.

    For remote: RemoteFileSystem with configured savepath or a temp directory.
    For local: LocalFileSystem with pytest's tmp_path.

    Only the active env_type's fixtures are resolved, so the local variant
    does not depend on (and skip with) the remote configuration.
    """
    if env_type == "remote":
        remote_fs = request.getfixturevalue("remote_fs")
        remote_cfg = request.getfixturevalue("remote_cluster_config")
        target_dir = (
            Path(remote_cfg.savepath)
            if remote_cfg.savepath
            else remote_fs.create_tempdir()
        )
        return remote_fs, target_dir
    return request.getfixturevalue("local_fs"), tmp_path
