"""Unit tests for conda resolution / validation in core.remote_env.

Run against a real ``LocalFileSystem`` (its ``run_command`` returns the local
``$HOME``), so the tilde-expansion path is exercised end to end without a
cluster: a ``~/...`` conda_source must validate by resolving against the home.
"""

import os
from pathlib import Path

import pytest

from remote_handler.core import remote_env
from remote_handler.core.file_handler import LocalFileSystem
from remote_handler.exceptions import ConfigurationError, FileSystemError


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


def test_expand_user_helper():
    assert remote_env._expand_user("~", "/home/u") == "/home/u"
    assert remote_env._expand_user("~/x/y", "/home/u") == "/home/u/x/y"
    assert remote_env._expand_user("/abs/path", "/home/u") == "/abs/path"
