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
# savepath, template, slurm_header_add (optional; blank -> not written).
_CLUSTER_ANSWERS = [
    "lugus",
    "normal",
    "jaxnest",
    "~/miniconda3/bin/activate",
    "/home/u/dump",
    "dirona.sh.j2",
    "",
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


def test_configure_cluster_writes_slurm_header_add_when_given(monkeypatch):
    answers = list(_CLUSTER_ANSWERS)
    answers[6] = "export TMPDIR=$HOME/tmp"  # cluster-wide header line
    _script_prompts(monkeypatch, answers)
    monkeypatch.setattr(cli, "SSHConfigParser", _FakeSSH)

    rc = cli.main(["configure", "cluster", "lugus"])
    assert rc == 0

    data = json.loads((get_persistent_config_dir() / "lugus.json").read_text())
    assert data["slurm_header_add"] == "export TMPDIR=$HOME/tmp"


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


def test_configure_test_sets_cluster(capsys):
    assert cli.main(["configure", "test", "lugus"]) == 0
    data = json.loads(get_persistent_config_file().read_text())
    assert data["test_cluster"] == "lugus"


def test_configure_test_preserves_other_fields():
    pfile = get_persistent_config_file()
    pfile.parent.mkdir(parents=True, exist_ok=True)
    pfile.write_text(
        json.dumps({"default_partition": "gpu", "test_cluster": "old"})
    )
    assert cli.main(["configure", "test", "lugus"]) == 0
    data = json.loads(pfile.read_text())
    assert data["test_cluster"] == "lugus"
    assert data["default_partition"] == "gpu"  # untouched


def test_configure_test_blank_clears(capsys):
    assert cli.main(["configure", "test", ""]) == 0
    data = json.loads(get_persistent_config_file().read_text())
    assert data["test_cluster"] is None


def test_configure_test_prompts_when_no_name(monkeypatch):
    _script_prompts(monkeypatch, ["lugus"])
    assert cli.main(["configure", "test"]) == 0
    data = json.loads(get_persistent_config_file().read_text())
    assert data["test_cluster"] == "lugus"


def test_ctrl_c_aborts_quietly(monkeypatch, capsys):
    def kbint(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "prompt_text", kbint)
    rc = cli.main(["configure"])
    assert rc == 130
    assert "Aborted" in capsys.readouterr().err


def test_ctrl_d_aborts_quietly(monkeypatch, capsys):
    def eof(*a, **k):
        raise EOFError

    monkeypatch.setattr(cli, "prompt_text", eof)
    rc = cli.main(["configure"])
    assert rc == 130
    assert "Aborted" in capsys.readouterr().err


def test_show_templates_lists_builtins_with_paths(capsys):
    rc = cli.main(["show", "templates"])
    assert rc == 0
    out = capsys.readouterr().out
    # built-in templates are always on the search path
    assert "dirona.sh.j2" in out
    # each line shows the resolved file path
    assert "/templates/dirona.sh.j2" in out


def test_show_clusters_lists_configs_with_paths(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("REMOTE_HANDLER_CONFIG_DIR", str(tmp_path))
    (tmp_path / "mycluster.json").write_text("{}")
    (tmp_path / "config.json").write_text("{}")  # must be excluded
    rc = cli.main(["show", "clusters"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "mycluster" in out
    assert str(tmp_path / "mycluster.json") in out
    assert str(tmp_path / "config.json") not in out


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


def test_show_extras_probes_imports(capsys, monkeypatch):
    """`show extras` reports per-extra status by actually importing modules.

    The imports are mocked so the test is fast and deterministic: everything
    resolves except ``mace.tools.freeze`` (i.e. stock mace, not the fork), so
    ``mace`` is present but ``mace_freeze`` is not.
    """
    import importlib

    def fake_import(name, *args, **kwargs):
        if name == "mace.tools.freeze":
            raise ModuleNotFoundError(name)
        return None  # pretend success; the command ignores the return value

    monkeypatch.setattr(importlib, "import_module", fake_import)

    rc = cli.main(["show", "extras"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "[x] mace" in out
    assert "[ ] mace_freeze  (missing: mace.tools.freeze)" in out
    # A plain (non-fork) extra resolves.
    assert "[x] dev" in out
    # The installed remote_handler version (and env) is shown at the top.
    assert "remote_handler" in out and "(local" in out


def test_local_version_helper_reports_installed():
    # Whatever is installed, the helper returns a non-empty string (the real
    # version, or "unknown" if metadata is somehow missing) -- never crashes.
    assert isinstance(cli._local_rh_version(), str)
    assert cli._local_rh_version()


def test_rh_version_remote_parses_line():
    fs = _FakeFS("noise\nRH_VERSION:0.3.0\n")
    assert cli._rh_version_remote(fs, _FakeCC()) == "0.3.0"
    # Probe runs inside the remote conda env.
    assert "importlib.metadata" in fs.commands[0]


def test_rh_version_remote_unknown_when_absent():
    fs = _FakeFS("ModuleNotFoundError: remote_handler\n")
    assert cli._rh_version_remote(fs, _FakeCC()) == "unknown"


# --------------------------------------------------------------------------
# show extras: remote / multi-machine
# --------------------------------------------------------------------------
from collections import namedtuple  # noqa: E402

_CR = namedtuple("CR", ["returncode", "stdout", "stderr"])


class _FakeFS:
    """Records commands and returns a canned ``run_command`` result."""

    def __init__(self, stdout):
        self._stdout = stdout
        self.commands = []

    def run_command(self, command):
        self.commands.append(command)
        return _CR(0, self._stdout, "")


class _FakeCC:
    conda_source = "/opt/conda/bin/activate"
    conda_env = "rh"
    remote_machine = "lugus"
    environment = {
        "vasp": "module load vasp/6.4.2_intel\nexport VASP_PP_PATH=/p",
    }


def test_probe_extras_remote_parses_json():
    fs = _FakeFS('noise\nRH_EXTRAS:{"mace": true, "mace_jax": false}\n')
    result = cli._probe_extras_remote(fs, _FakeCC())
    assert result == {"mace": True, "mace_jax": False}
    # The probe runs python in the remote conda env via a heredoc.
    assert "RH_PY" in fs.commands[0]


def test_probe_binaries_remote_uses_login_shell():
    fs = _FakeFS("vasp=yes\n")
    result = cli._probe_binaries_remote(fs, _FakeCC())
    assert result == {"vasp": True}
    # ``module`` needs a login shell, and the env setup must be sourced.
    assert "bash -l" in fs.commands[0]
    assert "module load vasp/6.4.2_intel" in fs.commands[0]


def test_probe_binaries_local_uses_which(monkeypatch):
    import shutil

    monkeypatch.setattr(
        shutil, "which", lambda b: "/usr/bin/" + b if b == "vasp_std" else None
    )
    result = cli._probe_binaries_local()
    assert result["vasp"] is True  # vasp_std resolved on PATH
    assert result["orca"] is False  # orca not found


def test_local_probe_includes_binaries(monkeypatch):
    # `local` is just another machine: it must report binaries, not skip them.
    import shutil

    monkeypatch.setattr(cli, "_probe_extras_local", lambda: {"mace": True})
    monkeypatch.setattr(shutil, "which", lambda b: None)
    status = cli._probe_machine("local")
    assert "vasp" in status and status["vasp"] is False
    assert "mace" in status


def test_probe_binaries_remote_empty_when_no_environment():
    class _NoEnv(_FakeCC):
        environment = {}

    fs = _FakeFS("")
    assert cli._probe_binaries_remote(fs, _NoEnv()) == {}
    assert fs.commands == []  # nothing to probe -> no SSH call


def test_show_extras_table_renders_grid(capsys, monkeypatch):
    canned = {
        "local": {"mace": True, "mace_jax": False},
        "lugus": {"mace": True, "mace_jax": True, "vasp": True},
        "down": None,
    }
    monkeypatch.setattr(cli, "_probe_machine", lambda t: canned[t])
    rc = cli._show_extras_table(["local", "lugus", "down"])
    out = capsys.readouterr().out

    # Header lists the machine columns in order.
    assert "local" in out and "lugus" in out and "down" in out
    # Cells: present / absent / unreachable.
    assert "✓" in out and "✗" in out and "?" in out
    # vasp is a binary-app row: applicable on lugus, n/a (—) on local.
    assert "vasp" in out and "—" in out
    # One machine was unreachable -> non-zero exit.
    assert rc == 1


def test_show_extras_table_all_ok_returns_zero(capsys, monkeypatch):
    monkeypatch.setattr(
        cli, "_probe_machine", lambda t: {"mace": True, "mace_jax": False}
    )
    rc = cli._show_extras_table(["local", "lugus"])
    assert rc == 0


def test_show_extras_table_shows_versions(capsys, monkeypatch):
    canned = {
        "local": {
            "mace": True,
            cli._RH_VERSION_KEY: "0.4.0",
            cli._RH_ENV_KEY: "rh_dev",
        },
        "lugus": {
            "mace": True,
            cli._RH_VERSION_KEY: "0.3.0",
            cli._RH_ENV_KEY: "jaxrens_al",
        },
        "down": None,
    }
    monkeypatch.setattr(cli, "_probe_machine", lambda t: canned[t])
    cli._show_extras_table(["local", "lugus", "down"])
    out = capsys.readouterr().out

    assert "remote_handler version" in out
    # Each machine's version is shown; the unreachable one renders as "?".
    assert "local" in out and "0.4.0" in out
    assert "lugus" in out and "0.3.0" in out
    # The conda env the version was read from is shown alongside.
    assert "(env: rh_dev)" in out and "(env: jaxrens_al)" in out
    # The reserved keys are a separate block, never raw ✓/✗ grid rows.
    assert cli._RH_VERSION_KEY not in out
    assert cli._RH_ENV_KEY not in out


# --------------------------------------------------------------------------
# status / collect  (resume a detached task from its manifest)
# --------------------------------------------------------------------------
def _detached_direct_task(tmp_path, finalizer=None):
    """Launch a real detached task on the direct/local backend and return the
    local manifest path. No SSH, no scheduler -- just the filesystem."""
    from remote_handler.core import file_handler, tasks

    remote = tmp_path / "remote"
    remote.mkdir()
    script = tmp_path / "single.py"
    script.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "x = json.load(open(sys.argv[1]))\n"
        "s = x['A'] + x['B']\n"
        'Path("result.json").write_text(json.dumps({"sum": s}))\n'
        'Path("is_finished.info").write_text("done")\n'
    )
    inp = tmp_path / "input.json"
    inp.write_text(json.dumps({"A": 2, "B": 40}))
    cfg = ClusterConfig(
        remote_machine="local",
        partition="p",
        conda_env="",
        conda_source="",
        savepath=str(remote),
        template="unused.sh.j2",
        executor="direct",
    )
    task = tasks.SingleScriptTask(
        script_file=script,
        input_file=inp,
        local_savepath=tmp_path / "results",
        cluster_config=cfg,
        remote_fs=file_handler.LocalFileSystem(),
        logfile="s.log",
        loop_time=0.1,
        remote_savepath=str(remote / "calc"),
    )
    if finalizer is not None:
        task.finalizer = finalizer
    return task.detach()


def test_collect_wait_then_status_finished(tmp_path, capsys):
    manifest = _detached_direct_task(tmp_path)

    # Collect blocks until the job finishes, then fetches the results.
    rc = cli.main(["collect", str(manifest), "--wait"])
    assert rc == 0
    result = json.loads((tmp_path / "results" / "result.json").read_text())
    assert result["sum"] == 42

    # After completion, status reports finished (exit 0).
    rc = cli.main(["status", str(manifest)])
    assert rc == 0
    assert "finished" in capsys.readouterr().out


def test_collect_discovers_manifest_in_directory(tmp_path):
    manifest = _detached_direct_task(tmp_path)
    # Pass the *directory*, not the file -- the common "cd back in" workflow.
    rc = cli.main(["collect", str(manifest.parent), "--wait"])
    assert rc == 0
    assert (tmp_path / "results" / "result.json").exists()


def test_status_missing_manifest_errors(tmp_path):
    rc = cli.main(["status", str(tmp_path)])
    assert rc == 1


def test_collect_ambiguous_manifest_errors(tmp_path):
    _detached_direct_task(tmp_path)
    # A second manifest in the same dir makes discovery ambiguous.
    (tmp_path / "other.manifest.json").write_text("{}")
    rc = cli.main(["collect", str(tmp_path)])
    assert rc == 1


# --------------------------------------------------------------------------
# collect: post-collect finalizer hook reporting (see core/manifest.py)
# --------------------------------------------------------------------------


def _mark_finalized(local_path):
    """Referenced by tests below via its dotted path ``"test_cli:..."``."""
    (local_path / "finalized.marker").write_text("ok")
    return local_path / "finalized.marker"


def test_collect_prints_finalized_output_on_success(tmp_path, capsys):
    manifest = _detached_direct_task(tmp_path, finalizer="test_cli:_mark_finalized")

    rc = cli.main(["collect", str(manifest), "--wait"])

    assert rc == 0
    assert (tmp_path / "results" / "finalized.marker").exists()
    out = capsys.readouterr().out
    assert "finalized:" in out
    assert "finalized.marker" in out


def test_collect_prints_warning_on_finalizer_failure(tmp_path, capsys):
    manifest = _detached_direct_task(
        tmp_path, finalizer="no_such_module_at_all:fn"
    )

    rc = cli.main(["collect", str(manifest), "--wait"])

    # A failing finalizer never hides an otherwise-successful raw collect.
    assert rc == 0
    assert (tmp_path / "results" / "result.json").exists()
    err = capsys.readouterr().err
    assert "[WARN] finalizer" in err
    assert "no_such_module_at_all:fn" in err
