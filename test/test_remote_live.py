"""Live SSH/SFTP round-trip tests against a real sshd.

These exercise the *real* transport path
(``ConfigBasedSSHClientProvider`` -> paramiko -> SFTP) without a cluster and
without SLURM, by pointing a temporary ssh-config at a live SSH server (the
user's real ``~/.ssh/config`` is never touched).

The SSH endpoint is provisioner-agnostic:

  * If ``SSHLIVE_HOST`` is set, connect to that already-running sshd. CI uses
    this -- it starts an sshd in the job itself, so the suite gets a genuine
    SSH/SFTP round-trip on every pipeline without needing a Docker daemon.
    Vars: ``SSHLIVE_HOST``, ``SSHLIVE_PORT`` (default 22), ``SSHLIVE_USER``
    (default ``tester``), ``SSHLIVE_KEY`` (path to the private key).
  * Otherwise boot a disposable ``linuxserver/openssh-server`` container
    (handy for local dev). Skips when Docker is unavailable.

Either way these auto-skip when no endpoint is reachable, so they are safe to
keep in the suite. Run just these with ``pytest -m sshlive``.
"""

import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase import Atoms

from remote_handler.core import tasks
from remote_handler.core.atoms_handler import RemoteAtomsHandler
from remote_handler.core.atoms_shard import front_tag
from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import RemoteFileSystem
from remote_handler.core.ssh_connection import ConfigBasedSSHClientProvider

pytestmark = [pytest.mark.sshlive]

SSH_IMAGE = "linuxserver/openssh-server:latest"
SSH_USER = "tester"


def _docker_available():
    """True if a usable Docker daemon is reachable."""
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            capture_output=True,
            timeout=10,
        )
        return True
    except (subprocess.SubprocessError, OSError):
        return False


def _free_port():
    """Grab a free localhost TCP port for the container to bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_tcp(port, timeout=60):
    """Block until something accepts TCP on 127.0.0.1:port."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except OSError:
            time.sleep(1)
    return False


def _write_ssh_config(workdir, host, port, user, key):
    """Write a throwaway ssh-config for the ``sshlive`` alias; return path."""
    config_file = workdir / "ssh_config"
    config_file.write_text(
        "Host sshlive\n"
        f"    HostName {host}\n"
        f"    Port {port}\n"
        f"    User {user}\n"
        f"    IdentityFile {key}\n"
    )
    return config_file


def _connect_fs(config_file, retries=20):
    """Connect a RemoteFileSystem via the config, retrying; skip on failure.

    A TCP-open port does not mean auth is ready (a container provisions its
    user asynchronously; a fresh sshd may still be starting), so retry.
    """
    last_err = None
    for _ in range(retries):
        try:
            provider = ConfigBasedSSHClientProvider(
                remote_alias="sshlive", config_file=str(config_file)
            )
            return RemoteFileSystem(
                remote_alias="sshlive", ssh_client_provider=provider
            )
        except Exception as err:  # noqa: BLE001 - report & skip below
            last_err = err
            time.sleep(1)
    pytest.skip(f"Could not connect to SSH endpoint: {last_err}")


@pytest.fixture(scope="session")
def ssh_container_fs(tmp_path_factory):
    """A RemoteFileSystem wired to a live sshd (env endpoint or Docker)."""
    workdir = tmp_path_factory.mktemp("sshlive")

    # Provisioner 1: an already-running sshd (CI starts one in the job).
    env_host = os.environ.get("SSHLIVE_HOST")
    if env_host:
        config_file = _write_ssh_config(
            workdir,
            host=env_host,
            port=int(os.environ.get("SSHLIVE_PORT", "22")),
            user=os.environ.get("SSHLIVE_USER", SSH_USER),
            key=os.environ["SSHLIVE_KEY"],
        )
        fs = _connect_fs(config_file)
        try:
            yield fs
        finally:
            fs._disconnect()
        return

    # Provisioner 2: a disposable openssh-server Docker container (local dev).
    if not _docker_available():
        pytest.skip(
            "No SSH endpoint: set SSHLIVE_HOST or make Docker available"
        )

    key = workdir / "id_ed25519"
    subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-N", "", "-q", "-f", str(key)],
        check=True,
    )
    pubkey = (workdir / "id_ed25519.pub").read_text().strip()

    port = _free_port()
    name = f"remote-handler-sshlive-{port}"
    started = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            "-p",
            f"127.0.0.1:{port}:2222",
            "-e",
            f"PUBLIC_KEY={pubkey}",
            "-e",
            f"USER_NAME={SSH_USER}",
            "-e",
            "PUID=1000",
            "-e",
            "PGID=1000",
            "-e",
            "PASSWORD_ACCESS=false",
            "-e",
            "SUDO_ACCESS=false",
            SSH_IMAGE,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if started.returncode != 0:
        pytest.skip(f"Could not start SSH container: {started.stderr.strip()}")

    try:
        if not _wait_for_tcp(port):
            pytest.skip("SSH container did not open its port in time")

        config_file = _write_ssh_config(
            workdir, "127.0.0.1", port, SSH_USER, key
        )
        fs = _connect_fs(config_file)
        try:
            yield fs
        finally:
            fs._disconnect()
    finally:
        subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True,
        )


def test_put_get_roundtrip(ssh_container_fs, tmp_path):
    """put a file, confirm it exists, get it back, compare, then remove."""
    fs = ssh_container_fs
    content = "hello remote_handler\n"

    local_src = tmp_path / "src.txt"
    local_src.write_text(content)
    remote = f"{fs.create_tempdir()}/hello.txt"

    fs.put(local_src, remote)
    assert fs.exists(remote)

    local_dst = tmp_path / "dst.txt"
    fs.get(remote, local_dst)
    assert local_dst.read_text() == content

    fs.remove(remote)
    assert not fs.exists(remote)


def test_makedirs(ssh_container_fs):
    """A freshly created remote directory reports as existing."""
    fs = ssh_container_fs
    remote_dir = f"{fs.create_tempdir()}/newdir"

    assert not fs.exists(remote_dir)
    fs.makedirs(remote_dir)
    assert fs.exists(remote_dir)


def test_write_read_ase(ssh_container_fs):
    """write_ase -> read_ase round-trips an Atoms object over SFTP."""
    fs = ssh_container_fs
    atoms = Atoms(
        "Si2",
        positions=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    remote = f"{fs.create_tempdir()}/atoms.extxyz"

    fs.write_ase(remote, [atoms])
    assert fs.exists(remote)

    loaded = fs.read_ase(remote, index=":")
    assert len(loaded) == 1
    assert loaded[0].get_chemical_formula() == atoms.get_chemical_formula()
    np.testing.assert_allclose(
        loaded[0].get_positions(), atoms.get_positions(), atol=1e-6
    )


# --------------------------------------------------------------------------
# Scheduler-free task execution over a real SSH connection.
# --------------------------------------------------------------------------
# These drive full SingleScriptTask/ArrayScriptTask pipelines through the
# ``direct`` executor against the live sshd: real SFTP transfer + a real remote
# ``python`` invocation + SFTP poll/collect, with NO scheduler and nothing
# mocked. This is the executor+transport integration the SLURM-gated task
# tests can't cover. (RemoteAtomsHandler adds ase-based sharding on top, which
# is unit-tested separately and would need ase installed on the remote.)
#
# Payloads are stdlib-only, but they still need a remote ``python`` and
# ``bash``: present on the CI ``SSHLIVE_HOST`` (localhost), absent on the bare
# openssh-server container used for local dev -- hence the capability skip.

# Writes its result + success marker into the cwd (the launcher cd's into the
# calcdir; the input file is passed as an absolute path).
_SINGLE_PAYLOAD = """
import json, sys
from pathlib import Path

data = json.load(open(sys.argv[1]))
Path("result.json").write_text(json.dumps({"sum": data["A"] + data["B"]}))
Path("is_finished.info").write_text("done")
"""

# Each array shard receives its own dir as argv[1]; all paths relative to it.
_ARRAY_PAYLOAD = """
import json, sys
from pathlib import Path

calc = Path(sys.argv[1])
data = json.load(open(calc / "input.json"))
(calc / "result.json").write_text(json.dumps({"sum": data["A"] + data["B"]}))
(calc / "is_finished.info").write_text("done")
"""


@pytest.fixture
def direct_remote(ssh_container_fs):
    """(fs, savepath, cluster_config) for a direct-executor run on the sshd.

    Skips when the remote lacks the ``python``/``bash`` the payload needs.
    """
    fs = ssh_container_fs
    probe = fs.run_command("command -v python && command -v bash")
    if probe.returncode != 0:
        pytest.skip("remote has no python/bash interpreter for direct exec")

    savepath = fs.create_tempdir()
    config = ClusterConfig(
        remote_machine="sshlive",
        partition="p",
        conda_env="",  # empty -> no activation -> ambient remote python
        conda_source="",
        savepath=str(savepath),
        template="unused.sh.j2",
        executor="direct",
    )
    return fs, Path(savepath), config


def test_direct_single_task_over_ssh(direct_remote, tmp_path):
    """SingleScriptTask runs end-to-end over real SSH, no scheduler."""
    fs, savepath, config = direct_remote

    script = tmp_path / "single.py"
    script.write_text(_SINGLE_PAYLOAD)
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps({"A": 4, "B": 5}))

    task = tasks.SingleScriptTask(
        script_file=script,
        input_file=input_file,
        local_savepath=tmp_path / "results",
        cluster_config=config,
        remote_fs=fs,
        loop_time=1,
        remote_savepath=str(savepath / "calc"),
    )

    task.allocate_data()
    task.submit()

    assert task.finish_check()  # marker written by the remote run
    local_result = tmp_path / "got.json"
    fs.get(str(Path(task.calcdir) / "result.json"), local_result)
    assert json.loads(local_result.read_text())["sum"] == 9


def test_direct_array_task_over_ssh(direct_remote, tmp_path):
    """ArrayScriptTask shards run end-to-end over real SSH, no scheduler."""
    fs, savepath, config = direct_remote

    script = tmp_path / "array.py"
    script.write_text(_ARRAY_PAYLOAD)

    n_shards = 3
    inputs = [
        tasks.DictJSON({"A": 100, "B": i}, "input.json", fs)
        for i in range(n_shards)
    ]

    task = tasks.ArrayScriptTask(
        local_savepath=tmp_path / "results",
        cluster_config=config,
        remote_fs=fs,
        python_script_file=script,
        individual_ressources=[inputs],
        shared_ressources=[],
        loop_time=1,
        remote_savepath=str(savepath),
    )

    task.allocate_data()
    task.submit()

    assert task.finish_check()
    for i in range(n_shards):
        local_result = tmp_path / f"got_{i}.json"
        fs.get(str(task.calcdir / str(i) / "result.json"), local_result)
        assert json.loads(local_result.read_text())["sum"] == 100 + i


# A shard processor: read the shipped configs, flag them finished, write the
# result file RemoteAtomsHandler.collect() expects (front_tag("opt.extxyz",
# tag)). Needs ase on the remote -- the test skips otherwise.
_ATOMS_PAYLOAD = """
import sys
from pathlib import Path
import ase.io

calc = Path(sys.argv[1])
atoms = ase.io.read(str(calc / "shard.extxyz"), index=":")
for at in atoms:
    at.info["is_finished"] = True
ase.io.write(str(calc / "proc_opt.extxyz"), atoms)
"""


class _MarkFinishedAtomsHandler(RemoteAtomsHandler):
    """Minimal concrete RemoteAtomsHandler for the test.

    RemoteAtomsHandler leaves the payload script + per-shard resources to a
    subclass (cf. RemoteOptimizationHandler). This ships each shard as
    ``shard.extxyz`` and runs the marker script above.
    """

    def __init__(self, *args, script, **kwargs):
        super().__init__(*args, **kwargs)
        self.python_script_file = script
        batches = [
            tasks.AtomsBatch(self.shards[i], "shard.extxyz", self.remote_fs)
            for i in range(self.num_array_tasks)
        ]
        del self.shards
        self.individual_ressources = [batches]


def test_direct_remote_atoms_handler_over_ssh(direct_remote, tmp_path):
    """RemoteAtomsHandler shards/runs/collects over real SSH, no scheduler.

    The faithful end-to-end: local traj files are sharded, shipped over SFTP,
    processed by a real remote ``python`` (via the direct executor), then
    collected and redistributed back into per-traj result files.
    """
    fs, savepath, config = direct_remote
    if fs.run_command("python -c 'import ase, ase.io'").returncode != 0:
        pytest.skip("remote python lacks ase for atoms processing")

    # Two local traj files, two configs each -> 4 configs.
    traj_files = []
    for fi in range(2):
        configs = [
            Atoms(
                "Si2",
                positions=[[0.0, 0.0, 0.0], [1.0 + ci + fi, 0.0, 0.0]],
                cell=[6.0, 6.0, 6.0],
                pbc=True,
            )
            for ci in range(2)
        ]
        path = tmp_path / f"traj_{fi}.extxyz"
        ase.io.write(str(path), configs)
        traj_files.append(str(path))

    script = tmp_path / "process.py"
    script.write_text(_ATOMS_PAYLOAD)

    handler = _MarkFinishedAtomsHandler(
        traj_files,
        config,
        fs,
        2,  # shard_size -> ceil(4 / 2) = 2 shards
        str(savepath),
        2,  # num_parallel
        every=1,
        tag="proc",
        loop_time=1,
        local_savepath=str(tmp_path),
        script=script,
    )

    handler.allocate_data()
    handler.submit()
    handler.collect_and_harvest()

    # Results are redistributed back into per-traj files, tag-prefixed.
    for original in traj_files:
        result = Path(front_tag(original, "proc"))
        assert result.exists(), f"missing redistributed result: {result}"
        ats = ase.io.read(str(result), index=":")
        assert len(ats) == 2
        assert all(bool(at.info.get("is_finished")) for at in ats)
