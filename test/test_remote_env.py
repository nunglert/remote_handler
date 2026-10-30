"""Unit tests for conda resolution / validation in core.remote_env.

Run against a real ``LocalFileSystem`` (its ``run_command`` returns the local
``$HOME``), so the tilde-expansion path is exercised end to end without a
cluster: a ``~/...`` conda_source must validate by resolving against the home.
"""

import os
import sys
from pathlib import Path

import pytest

from remote_handler.core import remote_env
from remote_handler.core.file_handler import LocalFileSystem
from remote_handler.exceptions import (
    CommandTimeoutError,
    ConfigurationError,
    FileSystemError,
)


@pytest.fixture
def python_on_path(monkeypatch):
    """Ensure the bare ``python`` the probe calls is this interpreter."""
    bindir = str(Path(sys.executable).parent)
    monkeypatch.setenv(
        "PATH", os.pathsep.join([bindir, os.environ.get("PATH", "")])
    )


def _make_conda_tree(root: Path, env: str = "myenv") -> None:
    """Create a minimal miniconda3 layout: bin/activate + envs/<env>."""
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "activate").write_text("# activate\n")
    (root / "envs" / env).mkdir(parents=True)


def test_remote_home_uses_local_fs(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert remote_env.remote_home(LocalFileSystem()) == str(tmp_path)


def test_validate_conda_accepts_tilde(monkeypatch, tmp_path):
    """``~/miniconda3`` validates by expanding against the (home) tmp dir."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _make_conda_tree(tmp_path / "miniconda3")

    activate = remote_env.validate_conda_on_fs(
        "~/miniconda3", "myenv", LocalFileSystem(), remote_machine="local"
    )
    # The returned activate path stays as given (shell expands ~ at runtime).
    assert activate == str(Path("~/miniconda3") / "bin/activate")


def test_validate_conda_absolute_still_works(tmp_path):
    _make_conda_tree(tmp_path / "miniconda3")
    activate = remote_env.validate_conda_on_fs(
        str(tmp_path / "miniconda3"),
        "myenv",
        LocalFileSystem(),
        remote_machine="local",
    )
    assert activate == str(tmp_path / "miniconda3" / "bin/activate")


def test_validate_conda_missing_activate_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))  # no miniconda3 created
    with pytest.raises(FileSystemError, match="activation script not found"):
        remote_env.validate_conda_on_fs(
            "~/miniconda3", "myenv", LocalFileSystem(), remote_machine="local"
        )


def test_validate_conda_missing_env_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / "miniconda3"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "activate").write_text("# activate\n")  # no envs/<env>
    with pytest.raises(ConfigurationError, match="not found"):
        remote_env.validate_conda_on_fs(
            "~/miniconda3", "absent", LocalFileSystem(), remote_machine="local"
        )


def test_validate_conda_custom_env_dir(monkeypatch, tmp_path):
    """An env under ``conda_env_dir`` validates though 'envs/' is absent."""
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / "miniconda3"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "activate").write_text("# activate\n")  # no envs/<env>
    envs_elsewhere = tmp_path / "shared" / "conda-envs"
    (envs_elsewhere / "myenv").mkdir(parents=True)

    activate = remote_env.validate_conda_on_fs(
        "~/miniconda3",
        "myenv",
        LocalFileSystem(),
        remote_machine="local",
        conda_env_dir=str(envs_elsewhere),
    )
    assert activate == str(Path("~/miniconda3") / "bin/activate")


def test_validate_conda_env_dir_expands_tilde(monkeypatch, tmp_path):
    """``conda_env_dir`` may itself use ``~`` (expanded against home)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / "miniconda3"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "activate").write_text("# activate\n")
    (tmp_path / "conda-envs" / "myenv").mkdir(parents=True)

    remote_env.validate_conda_on_fs(
        "~/miniconda3",
        "myenv",
        LocalFileSystem(),
        remote_machine="local",
        conda_env_dir="~/conda-envs",
    )


def test_validate_conda_missing_env_hints_env_dir(monkeypatch, tmp_path):
    """Without ``conda_env_dir``, the error suggests setting it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / "miniconda3"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "activate").write_text("# activate\n")
    with pytest.raises(ConfigurationError, match="conda_env_dir"):
        remote_env.validate_conda_on_fs(
            "~/miniconda3", "absent", LocalFileSystem(), remote_machine="local"
        )


def test_expand_user_helper():
    assert remote_env._expand_user("~", "/home/u") == "/home/u"
    assert remote_env._expand_user("~/x/y", "/home/u") == "/home/u/x/y"
    assert remote_env._expand_user("/abs/path", "/home/u") == "/abs/path"


def test_resolve_remote_module_file_found(python_on_path):
    """Probing resolves an importable module's file (here, over LocalFS)."""
    path = remote_env.resolve_remote_module_file(
        LocalFileSystem(), "", "", "json"
    )
    assert path is not None
    assert path.replace(os.sep, "/").endswith("json/__init__.py")


def test_resolve_remote_module_file_missing(python_on_path):
    """A module that is not importable resolves to None (no raise)."""
    path = remote_env.resolve_remote_module_file(
        LocalFileSystem(), "", "", "no_such_module_xyz123"
    )
    assert path is None


# --------------------------------------------------------------------------
# conda_activation_script: activation + PATH hardening
# --------------------------------------------------------------------------
def test_activation_hardens_path_after_conda_activate():
    """The env's bin is forced to the front of PATH after activation, so a
    module-loaded interpreter (EESSI/Lmod) cannot shadow it -- and the user
    never has to hardcode the env path in their header."""
    script = remote_env.conda_activation_script(
        "/home/u/miniconda3/bin/activate", "jaxenv"
    )
    lines = [ln for ln in script.splitlines() if ln.strip()]
    assert lines == [
        "source /home/u/miniconda3/bin/activate",
        "conda activate jaxenv",
        '[ -n "$CONDA_PREFIX" ] && export PATH="$CONDA_PREFIX/bin:$PATH"',
    ]
    # The hardening comes *after* activation (so $CONDA_PREFIX is set).
    assert script.index("conda activate") < script.index("CONDA_PREFIX")


def test_activation_hardening_is_guarded_for_micromamba():
    script = remote_env.conda_activation_script("/opt/micromamba", "jaxenv")
    assert "micromamba activate jaxenv" in script
    assert '[ -n "$CONDA_PREFIX" ]' in script


def test_no_activation_no_hardening():
    """Empty/dummy config yields no activation and no PATH manipulation."""
    assert remote_env.conda_activation_script("", "env") == ""
    assert "CONDA_PREFIX" not in remote_env.conda_activation_script(
        "/nonsense/format", "env"
    )


class _TimingOutFS:
    """A filesystem whose every command times out."""

    def run_command(self, command, timeout=None):
        raise CommandTimeoutError(f"timed out: {command}")


@pytest.mark.parametrize(
    "probe",
    [
        lambda fs: remote_env.resolve_remote_module_file(
            fs, "", "", "mace.cli.run_train"
        ),
        lambda fs: remote_env.resolve_remote_executable(
            fs, "", "", "pacemaker"
        ),
        remote_env.remote_home,
    ],
    ids=["module", "executable", "home"],
)
def test_probe_timeout_propagates(probe):
    # A stalled probe must surface as a timeout, not as "module not found"
    # (or, for remote_home, a silent fallback to the local home).
    with pytest.raises(CommandTimeoutError):
        probe(_TimingOutFS())
