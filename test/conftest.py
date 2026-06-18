import json
import os
from pathlib import Path
from typing import Optional

import pytest

from remote_handler.config import (
    get_config,
    get_persistent_config_dir,
    reset_config,
)
from remote_handler.core import file_handler
from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.exceptions import ConfigurationError

# Real persistent config dir, captured at import -- i.e. *before* the autouse
# ``reset_global_config`` fixture isolates XDG_CONFIG_HOME to a temp dir. This
# lets the remote tier fall back to a ``remote_handler configure`` setup when
# REMOTE_HANDLER_CONFIG_DIR is unset, so the env var is optional, not required.
_REAL_PERSISTENT_DIR = get_persistent_config_dir()

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
#                     reason when no cluster config is available, so the rest
#                     of the suite still runs.
#   * Integration  -- marked ``integration`` and deselected by default
#                     (see pyproject.toml). Run with ``pytest -m integration``.
#
# The remote tier finds its cluster config dir with this precedence:
#     1. REMOTE_HANDLER_CONFIG_DIR (an optional override), else
#     2. the real persistent config dir from ``remote_handler configure``.
# So once the package is configured, ``pytest -m remote`` runs with no env
# var; setting the env var still works to point elsewhere. Without either,
# the remote-parametrized tests simply report as skipped.


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


def _remote_config_dir() -> Optional[Path]:
    """Directory holding the remote-tier cluster JSONs, or ``None``.

    Precedence: ``REMOTE_HANDLER_CONFIG_DIR`` (optional override) > the real
    persistent config dir captured before the test harness isolated
    XDG_CONFIG_HOME. Resolved here (not via ``get_config()``) precisely so the
    isolation does not hide a ``remote_handler configure`` setup.
    """
    env = os.environ.get("REMOTE_HANDLER_CONFIG_DIR")
    if env:
        path = Path(env).expanduser()
        return path if path.is_dir() else None
    return _REAL_PERSISTENT_DIR if _REAL_PERSISTENT_DIR.is_dir() else None


def _remote_test_cluster_name(config_dir: Optional[Path]) -> str:
    """Cluster name the remote tier targets.

    Reads ``test_cluster`` from ``<config_dir>/config.json`` (set via
    ``remote_handler configure``), falling back to ``remote``.
    """
    if config_dir is None:
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

    Resolves the config dir from REMOTE_HANDLER_CONFIG_DIR or, when unset, the
    persistent ``remote_handler configure`` location; the target cluster is
    configurable via ``test_cluster`` (default ``remote.json``). Loads by
    absolute path so it is unaffected by the per-test XDG isolation. Skips with
    a reason when no config is available. Session-scoped (ClusterConfig is
    immutable).
    """
    config_dir = _remote_config_dir()
    name = _remote_test_cluster_name(config_dir)
    if config_dir is not None:
        path = config_dir / f"{name}.json"
        if path.exists():
            try:
                return ClusterConfig.from_file(str(path))
            except ConfigurationError as exc:
                pytest.skip(f"Cannot load remote cluster {path}: {exc}")
    pytest.skip(
        f"Remote cluster config unavailable -- run `remote_handler configure` "
        f"(or set REMOTE_HANDLER_CONFIG_DIR) so {name}.json is found, to run "
        "remote tests"
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
        if remote_cfg.savepath:
            if not remote_fs.exists(remote_cfg.savepath):
                pytest.fail(
                    "Configured savepath does not exist on "
                    f"{remote_cfg.remote_machine}: {remote_cfg.savepath}\n"
                    "Create it on the remote, or set an existing absolute "
                    "path as the cluster config 'savepath'."
                )
            target_dir = Path(remote_cfg.savepath)
        else:
            target_dir = remote_fs.create_tempdir()
        return remote_fs, target_dir
    return request.getfixturevalue("local_fs"), tmp_path
