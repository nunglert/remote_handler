"""Unit tests for the remote-tier config-dir / cluster-name resolution.

These verify the behaviour that makes REMOTE_HANDLER_CONFIG_DIR optional:
the remote tier resolves its config dir from the env var when set, else from
the persistent ``remote_handler configure`` location, and reads the target
cluster name (``test_cluster``) from that dir's ``config.json``.
"""

import json

import conftest


def test_cluster_name_defaults_to_remote(tmp_path):
    assert conftest._remote_test_cluster_name(tmp_path) == "remote"
    assert conftest._remote_test_cluster_name(None) == "remote"


def test_cluster_name_from_config_json(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"test_cluster": "lugus"})
    )
    assert conftest._remote_test_cluster_name(tmp_path) == "lugus"


def test_cluster_name_falls_back_on_blank_or_bad_json(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"test_cluster": ""}))
    assert conftest._remote_test_cluster_name(tmp_path) == "remote"
    cfg.write_text("{not valid json")
    assert conftest._remote_test_cluster_name(tmp_path) == "remote"


def test_config_dir_prefers_env_when_set(tmp_path, monkeypatch):
    monkeypatch.setenv("REMOTE_HANDLER_CONFIG_DIR", str(tmp_path))
    assert conftest._remote_config_dir() == tmp_path


def test_config_dir_env_ignored_when_not_a_dir(tmp_path, monkeypatch):
    missing = tmp_path / "nope"
    monkeypatch.setenv("REMOTE_HANDLER_CONFIG_DIR", str(missing))
    assert conftest._remote_config_dir() != missing


def test_config_dir_without_env_uses_persistent(monkeypatch):
    monkeypatch.delenv("REMOTE_HANDLER_CONFIG_DIR", raising=False)
    # The real persistent dir captured at import (or None if absent), never
    # the per-test temp XDG dir -- that's what makes a configured setup work.
    expected = (
        conftest._REAL_PERSISTENT_DIR
        if conftest._REAL_PERSISTENT_DIR.is_dir()
        else None
    )
    assert conftest._remote_config_dir() == expected
