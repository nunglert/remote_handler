"""Unit tests for the remote_handler configuration CLI.

These are unit-tier (unmarked): no cluster, no network. Interactive prompts
are replaced with scripted fakes; SSH and the live connect probe are mocked.
The autouse ``reset_global_config`` fixture isolates XDG_CONFIG_HOME to a temp
dir, so the persistent-file read never touches a real ~/.config.
"""

import json

import pytest

from remote_handler import cli
from remote_handler.config import (
    get_config,
    get_persistent_config_dir,
    get_persistent_config_file,
    reset_config,
)
from remote_handler.core.cluster_handler import ClusterConfig


def _script_prompts(monkeypatch, answers):
    """Replace the three prompt helpers with a shared scripted reader."""
    it = iter(answers)
    monkeypatch.setattr(cli, "prompt_text", lambda *a, **k: next(it))
    monkeypatch.setattr(cli, "prompt_choice", lambda *a, **k: next(it))
    monkeypatch.setattr(cli, "prompt_yes_no", lambda *a, **k: next(it))


class _FakeSSH:
    """SSHConfigParser stub: alias always resolves, no real config read."""

    def __init__(self, *a, **k):
        pass

    def get_available_hosts(self):
        return ["lugus"]

    def parse_host_config(self, alias):
        return object()


# Cluster prompt order: remote_machine, partition, conda_env, conda_source,
# savepath, template.
_CLUSTER_ANSWERS = [
    "lugus",
    "normal",
    "jaxnest",
    "~/miniconda3/bin/activate",
    "/home/u/dump",
    "dirona.sh.j2",
]


def test_configure_cluster_writes_all_six_fields(monkeypatch):
    _script_prompts(monkeypatch, list(_CLUSTER_ANSWERS))
    monkeypatch.setattr(cli, "SSHConfigParser", _FakeSSH)

    rc = cli.main(["configure", "cluster", "lugus"])
    assert rc == 0

    target = get_persistent_config_dir() / "lugus.json"
    data = json.loads(target.read_text())
    assert set(data) == {
        "remote_machine",
        "partition",
        "conda_env",
        "conda_source",
        "savepath",
        "template",
    }
    # Regression guard: the emitted file must be loadable (unlike the
    # shipped examples that omit conda_source/template).
    reset_config()
    cc = ClusterConfig.from_file("lugus.json")
    assert cc.remote_machine == "lugus"
    assert cc.conda_source == "~/miniconda3/bin/activate"


def test_configure_cluster_rejects_empty_field(monkeypatch):
    answers = list(_CLUSTER_ANSWERS)
    answers[4] = ""  # empty savepath
    _script_prompts(monkeypatch, answers)
    monkeypatch.setattr(cli, "SSHConfigParser", _FakeSSH)

    rc = cli.main(["configure", "cluster", "lugus"])
    assert rc == 1
    assert not (get_persistent_config_dir() / "lugus.json").exists()


def test_persistent_file_precedence(monkeypatch, tmp_path):
    file_dir = tmp_path / "from_file"
    file_dir.mkdir()
    pfile = get_persistent_config_file()
    pfile.parent.mkdir(parents=True, exist_ok=True)
    pfile.write_text(json.dumps({"config_dir": str(file_dir)}))

    reset_config()
    assert get_config().config_dir == file_dir

    env_dir = tmp_path / "from_env"
    env_dir.mkdir()
    monkeypatch.setenv("REMOTE_HANDLER_CONFIG_DIR", str(env_dir))
    reset_config()
    assert get_config().config_dir == env_dir  # env wins over file


def test_resolve_defaults_to_persistent_dir(monkeypatch):
    monkeypatch.delenv("REMOTE_HANDLER_CONFIG_DIR", raising=False)
    reset_config()
    cfg = get_config()
    assert cfg.config_dir == get_persistent_config_dir()
    assert cfg.default_partition == "normal"


def test_check_fails_when_config_dir_missing(monkeypatch, capsys):
    monkeypatch.delenv("REMOTE_HANDLER_CONFIG_DIR", raising=False)
    reset_config()
    rc = cli.main(["check"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Config dir missing" in out
    assert "remote_handler configure" in out


def test_check_flags_unloadable_cluster_json(monkeypatch, capsys):
    cfg_dir = get_persistent_config_dir()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    # Example-style file missing conda_source/template -> TypeError on load.
    (cfg_dir / "broken.json").write_text(
        json.dumps(
            {
                "remote_machine": "lugus",
                "partition": "normal",
                "conda_env": "jaxnest",
                "savepath": "/x",
            }
        )
    )
    monkeypatch.setattr(cli, "SSHConfigParser", _FakeSSH)

    rc = cli.main(["check"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "broken.json: cannot load" in out


class _FakeStream:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return self._payload


class _FakeClient:
    def __init__(self, sbatch=b"/usr/bin/sbatch\n", env_ok=b"OK\n"):
        self._sbatch = sbatch
        self._env_ok = env_ok

    def exec_command(self, cmd):
        payload = self._sbatch if "sbatch" in cmd else self._env_ok
        return (None, _FakeStream(payload), _FakeStream(b""))

    def close(self):
        pass


def _make_fake_provider(exc=None, client=None):
    class _FakeProvider:
        def __init__(self, alias, *a, **k):
            pass

        def create_connection(self):
            if exc is not None:
                raise exc
            return (client or _FakeClient(), _FakeClient())

        def cleanup(self):
            pass

    return _FakeProvider


def _write_lugus(monkeypatch):
    cfg_dir = get_persistent_config_dir()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "lugus.json").write_text(
        json.dumps(
            {
                "remote_machine": "lugus",
                "partition": "normal",
                "conda_env": "jaxnest",
                "conda_source": "~/miniconda3/bin/activate",
                "savepath": "/x",
                "template": "dirona.sh.j2",
            }
        )
    )
    monkeypatch.setattr(cli, "SSHConfigParser", _FakeSSH)


def test_check_connect_reports_sbatch(monkeypatch, capsys):
    _write_lugus(monkeypatch)
    monkeypatch.setattr(
        cli, "ConfigBasedSSHClientProvider", _make_fake_provider()
    )
    rc = cli.main(["check", "--connect"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "sbatch at /usr/bin/sbatch" in out
    assert "conda env 'jaxnest' present" in out


def test_check_connect_degrades_on_cert_bug(monkeypatch, capsys):
    _write_lugus(monkeypatch)
    monkeypatch.setattr(
        cli,
        "ConfigBasedSSHClientProvider",
        _make_fake_provider(exc=AttributeError("public_blob")),
    )
    rc = cli.main(["check", "--connect"])
    out = capsys.readouterr().out
    assert rc == 0  # WARN, not FAIL
    assert "ssh-add -D" in out


def test_configure_persists_test_cluster(monkeypatch):
    # configure prompts (all prompt_text), in order:
    # config_dir, templates, partition, conda_env, conda_source, test_cluster
    _script_prompts(
        monkeypatch,
        ["", "", "normal", "base", "~/miniconda3/bin/activate", "lugus"],
    )
    rc = cli.main(["configure"])
    assert rc == 0
    data = json.loads(get_persistent_config_file().read_text())
    assert data["test_cluster"] == "lugus"
    # And it resolves through the global config singleton.
    assert get_config().test_cluster == "lugus"


def test_configure_blank_test_cluster_is_none(monkeypatch):
    _script_prompts(
        monkeypatch,
        ["", "", "normal", "base", "~/miniconda3/bin/activate", ""],
    )
    assert cli.main(["configure"]) == 0
    data = json.loads(get_persistent_config_file().read_text())
    assert data["test_cluster"] is None


def test_show_json_has_expected_keys(capsys):
    rc = cli.main(["show", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert set(data) == {
        "config_dir",
        "slurm_templates_dir",
        "default_partition",
        "default_conda_env",
        "default_conda_source",
        "test_cluster",
    }


@pytest.mark.parametrize(
    "answer,expected",
    [("", "default-val"), ("typed", "typed"), ("  x  ", "x")],
)
def test_prompt_text(answer, expected):
    from remote_handler.cli_prompts import prompt_text

    result = prompt_text("Label", default="default-val", ask=lambda _: answer)
    assert result == expected


def test_prompt_choice_index_literal_custom():
    from remote_handler.cli_prompts import prompt_choice

    choices = ["a.sh.j2", "b.sh.j2"]
    assert prompt_choice("L", choices, ask=lambda _: "2") == "b.sh.j2"
    assert prompt_choice("L", choices, ask=lambda _: "a.sh.j2") == "a.sh.j2"
    assert prompt_choice("L", choices, ask=lambda _: "custom") == "custom"


def test_prompt_yes_no():
    from remote_handler.cli_prompts import prompt_yes_no

    assert prompt_yes_no("L", default=True, ask=lambda _: "") is True
    assert prompt_yes_no("L", ask=lambda _: "y") is True
    assert prompt_yes_no("L", ask=lambda _: "no") is False
