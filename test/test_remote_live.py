"""Live SSH/SFTP round-trip tests against a throwaway Docker container.

These exercise the *real* transport path
(``ConfigBasedSSHClientProvider`` -> paramiko -> SFTP) without a cluster and
without SLURM, by booting a disposable ``linuxserver/openssh-server``
container and pointing a temporary ssh-config at it (the user's real
``~/.ssh/config`` is never touched).

They auto-skip when Docker is unavailable, so they are safe to keep in the
suite. Run just these with ``pytest -m sshlive``.
"""

import shutil
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

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


@pytest.fixture(scope="session")
def ssh_container_fs(tmp_path_factory):
    """A RemoteFileSystem wired to a disposable openssh-server container."""
    if not _docker_available():
        pytest.skip("Docker not available")

    workdir = tmp_path_factory.mktemp("sshlive")
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

        config_file = workdir / "ssh_config"
        config_file.write_text(
            "Host sshlive\n"
            "    HostName 127.0.0.1\n"
            f"    Port {port}\n"
            f"    User {SSH_USER}\n"
            f"    IdentityFile {key}\n"
        )

        # TCP-open does not mean auth is ready (the image provisions the user
        # and authorized_keys asynchronously); retry the real connection.
        fs = None
        last_err = None
        for _ in range(20):
            try:
                provider = ConfigBasedSSHClientProvider(
                    remote_alias="sshlive", config_file=str(config_file)
                )
                fs = RemoteFileSystem(
                    remote_alias="sshlive", ssh_client_provider=provider
                )
                break
            except Exception as err:  # noqa: BLE001 - report & skip below
                last_err = err
                time.sleep(1)
        if fs is None:
            pytest.skip(f"Could not connect to SSH container: {last_err}")

        yield fs
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
