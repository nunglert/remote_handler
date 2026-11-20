"""``RemoteFileSystem`` tree operations: ``get_dir`` (sftp + tar transfer),
``rmtree``, and resumable collect.

No network: a fake SFTP client serves a local directory as "the remote", and
a fake SSH client runs ``exec_command`` as a local subprocess. So the tar
path runs the real ``tar`` binary in the plain unit tier; the ``sshlive``
tier repeats the round trip against a real sshd.
"""

import os
import shutil
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import paramiko
import pytest

from remote_handler.core import file_handler, file_utils, tasks
from remote_handler.core.cluster_config import (
    TRANSFER_SFTP,
    TRANSFER_TAR,
    ClusterConfig,
)
from remote_handler.core.ssh_connection import MockSSHClientProvider
from remote_handler.exceptions import ConfigurationError, FileSystemError

needs_tar = pytest.mark.skipif(
    shutil.which("tar") is None, reason="needs a local tar binary"
)


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------
class _FakeRemoteFile:
    def __init__(self, path, log):
        self._fh = open(path, "rb")
        self._log = log

    def prefetch(self, size):
        self._log.append(("prefetch", size))

    def read(self, n=-1):
        return self._fh.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._fh.close()


class FakeSFTP:
    """The subset of ``paramiko.SFTPClient`` get_dir uses, on local paths."""

    def __init__(self):
        self.log = []
        self.extra_entries = {}  # dir -> [SFTPAttributes] injected

    def listdir_attr(self, path):
        self.log.append(("listdir_attr", path))
        out = [
            paramiko.SFTPAttributes.from_stat(
                os.lstat(os.path.join(path, name)), name
            )
            for name in sorted(os.listdir(path))
        ]
        return out + self.extra_entries.get(path.rstrip("/"), [])

    def stat(self, path):
        self.log.append(("stat", path))
        return paramiko.SFTPAttributes.from_stat(os.stat(path))

    def open(self, path, mode="r"):
        self.log.append(("open", path))
        return _FakeRemoteFile(path, self.log)

    def remove(self, path):
        self.log.append(("remove", path))
        os.remove(path)

    def rmdir(self, path):
        self.log.append(("rmdir", path))
        os.rmdir(path)

    def opened(self):
        return [p for op, p in self.log if op == "open"]

    def close(self):
        pass


class _Stdin:
    def __init__(self, proc):
        self._proc = proc
        self.channel = SimpleNamespace(shutdown_write=proc.stdin.close)

    def write(self, data):
        self._proc.stdin.write(data.encode())


class _Stream:
    """stdout/stderr with the ``.channel`` surface paramiko provides."""

    def __init__(self, fh, proc, done):
        self._fh = fh
        self.channel = SimpleNamespace(
            status_event=done,
            recv_exit_status=proc.wait,
            close=lambda: None,
        )

    def read(self, n=-1):
        return self._fh.read(n)


class FakeSSH:
    """Runs ``exec_command`` locally.

    ``rewrite`` lets a test alter a command before it runs (e.g. to make tar
    emit an extra, unrequested member).
    """

    def __init__(self, has_tar=True, rewrite=None):
        self.has_tar = has_tar
        self.rewrite = rewrite or (lambda cmd: cmd)
        self.commands = []

    def exec_command(self, command, timeout=None):
        self.commands.append(command)
        if command == "command -v tar" and not self.has_tar:
            command = "exit 1"
        proc = subprocess.Popen(
            self.rewrite(command),
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        done = threading.Event()
        threading.Thread(
            target=lambda: (proc.wait(), done.set()), daemon=True
        ).start()
        return (
            _Stdin(proc),
            _Stream(proc.stdout, proc, done),
            _Stream(proc.stderr, proc, done),
        )

    def close(self):
        pass


def make_fs(tmp_path, transfer=TRANSFER_SFTP, ssh=None, sftp=None):
    ssh = ssh or FakeSSH()
    sftp = sftp or FakeSFTP()
    fs = file_handler.RemoteFileSystem(
        remote_alias="fake",
        ssh_client_provider=MockSSHClientProvider(ssh, sftp),
        local_temp_path=str(tmp_path),
        transfer=transfer,
    )
    return fs, ssh, sftp


def make_tree(root: Path) -> dict:
    """A small remote tree; returns {relpath: bytes}."""
    files = {
        "top.txt": b"top level\n",
        "system_00/frame_000.npz": os.urandom(70_000),
        "system_00/frame_001.npz": os.urandom(1_000),
        "system_01/deep/nested.dat": b"nested\n",
        "system_01/empty.txt": b"",
        "-starts-with-dash.txt": b"dash\n",
    }
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    (root / "empty_dir").mkdir()
    return files


def assert_tree(local: Path, files: dict):
    for rel, data in files.items():
        assert (local / rel).read_bytes() == data, rel
    assert (local / "empty_dir").is_dir()
    assert not list(local.rglob("*.part"))


@pytest.fixture
def remote_tree(tmp_path):
    root = tmp_path / "remote"
    files = make_tree(root)
    return root, files


# --------------------------------------------------------------------------
# sftp (default) path
# --------------------------------------------------------------------------
def test_sftp_copies_full_tree(tmp_path, remote_tree):
    root, files = remote_tree
    fs, _, sftp = make_fs(tmp_path)
    local = tmp_path / "local"

    fs.get_dir(str(root) + "/", local)

    assert_tree(local, files)
    # One listing per directory and one open per file; the only stat is
    # exists() on the root -- none per file.
    assert [p for op, p in sftp.log if op == "stat"] == [str(root)]
    assert len(sftp.opened()) == len(files)
    # Read-ahead sized from the listing; never prefetch(0) (it would hang).
    prefetched = sorted(n for op, n in sftp.log if op == "prefetch")
    assert prefetched == sorted(len(d) for d in files.values() if d)


def test_sftp_resume_skips_complete_and_repairs_truncated(
    tmp_path, remote_tree
):
    root, files = remote_tree
    local = tmp_path / "local"
    (local / "system_00").mkdir(parents=True)
    # Complete -> skipped; truncated -> refetched; stale .part -> replaced.
    (local / "top.txt").write_bytes(files["top.txt"])
    big = files["system_00/frame_000.npz"]
    (local / "system_00/frame_000.npz").write_bytes(big[: len(big) // 2])
    (local / "system_00/frame_001.npz.part").write_bytes(b"junk")

    fs, _, sftp = make_fs(tmp_path)
    fs.get_dir(root, local)

    assert_tree(local, files)
    opened = {Path(p).relative_to(root).as_posix() for p in sftp.opened()}
    assert "top.txt" not in opened
    assert "system_00/frame_000.npz" in opened


def test_sftp_second_call_fetches_nothing(tmp_path, remote_tree):
    root, files = remote_tree
    fs, _, sftp = make_fs(tmp_path)
    local = tmp_path / "local"
    fs.get_dir(root, local)
    sftp.log.clear()

    fs.get_dir(root, local)

    assert sftp.opened() == []


def test_missing_remote_dir_is_noop(tmp_path):
    fs, _, _ = make_fs(tmp_path)
    local = tmp_path / "local"
    fs.get_dir(tmp_path / "does-not-exist", local)
    assert not local.exists()


def test_symlinked_dir_is_followed(tmp_path, remote_tree):
    root, files = remote_tree
    os.symlink(root / "system_01", root / "linked")
    fs, _, _ = make_fs(tmp_path)
    local = tmp_path / "local"

    fs.get_dir(root, local)

    assert (local / "linked/deep/nested.dat").read_bytes() == b"nested\n"


def test_suspicious_remote_names_are_skipped(tmp_path, remote_tree):
    root, files = remote_tree
    fs, _, sftp = make_fs(tmp_path)
    evil = paramiko.SFTPAttributes()
    evil.filename, evil.st_mode, evil.st_size = "../escaped", 0o100644, 3
    sftp.extra_entries[str(root)] = [evil]

    fs.get_dir(root, tmp_path / "local")

    assert not (tmp_path / "escaped").exists()


def test_unknown_transfer_rejected(tmp_path):
    with pytest.raises(ConfigurationError, match="Unknown transfer"):
        make_fs(tmp_path, transfer="rsync")


# --------------------------------------------------------------------------
# tar path
# --------------------------------------------------------------------------
@needs_tar
def test_tar_copies_full_tree(tmp_path, remote_tree):
    root, files = remote_tree
    fs, ssh, sftp = make_fs(tmp_path, transfer=TRANSFER_TAR)
    local = tmp_path / "local"

    fs.get_dir(root, local)

    assert_tree(local, files)
    assert sftp.opened() == []  # all content came through the tar stream
    assert sum("tar -cf" in c for c in ssh.commands) == 1
    # mtime carried over from the archive
    src = (root / "top.txt").stat().st_mtime
    assert int((local / "top.txt").stat().st_mtime) == int(src)


@needs_tar
def test_tar_resume_only_streams_pending(tmp_path, remote_tree):
    root, files = remote_tree
    local = tmp_path / "local"
    (local / "system_00").mkdir(parents=True)
    (local / "top.txt").write_bytes(files["top.txt"])
    big = files["system_00/frame_000.npz"]
    (local / "system_00/frame_000.npz").write_bytes(big[:10])

    seen = []

    def spy(cmd):
        # Record the file list tar is fed by wrapping it in a tee.
        if cmd.startswith("tar "):
            listing = tmp_path / "fed.txt"
            seen.append(listing)
            return f"tee {listing} | {cmd}"
        return cmd

    fs, _, _ = make_fs(
        tmp_path, transfer=TRANSFER_TAR, ssh=FakeSSH(rewrite=spy)
    )
    fs.get_dir(root, local)

    assert_tree(local, files)
    fed = seen[0].read_text().split()
    assert "./top.txt" not in fed
    assert "./system_00/frame_000.npz" in fed


@needs_tar
def test_tar_nothing_pending_runs_no_tar(tmp_path, remote_tree):
    root, files = remote_tree
    fs, ssh, _ = make_fs(tmp_path, transfer=TRANSFER_TAR)
    local = tmp_path / "local"
    fs.get_dir(root, local)
    ssh.commands.clear()

    fs.get_dir(root, local)

    assert ssh.commands == []


@needs_tar
def test_tar_ignores_unrequested_members(tmp_path, remote_tree):
    root, files = remote_tree
    (root / "not_listed.txt").write_bytes(b"x")

    def add_member(cmd):
        # Make tar also emit a file that was never requested.
        if cmd.startswith("tar "):
            return f"{cmd} ./not_listed.txt"
        return cmd

    fs, _, sftp = make_fs(
        tmp_path, transfer=TRANSFER_TAR, ssh=FakeSSH(rewrite=add_member)
    )
    # Hide not_listed.txt from the listing so it is never requested.
    real = sftp.listdir_attr
    sftp.listdir_attr = lambda p: [
        a for a in real(p) if a.filename != "not_listed.txt"
    ]
    local = tmp_path / "local"

    fs.get_dir(root, local)

    assert_tree(local, files)
    assert not (local / "not_listed.txt").exists()


@needs_tar
def test_tar_failure_raises(tmp_path, remote_tree):
    root, _ = remote_tree
    fs, _, sftp = make_fs(tmp_path, transfer=TRANSFER_TAR)
    # A file that vanishes between listing and tar: tar exits non-zero.
    ghost = paramiko.SFTPAttributes()
    ghost.filename, ghost.st_mode, ghost.st_size = "ghost", 0o100644, 1
    sftp.extra_entries[str(root)] = [ghost]

    with pytest.raises(FileSystemError, match="Remote tar"):
        fs.get_dir(root, tmp_path / "local")


def test_tar_falls_back_to_sftp_without_remote_tar(tmp_path, remote_tree):
    root, files = remote_tree
    ssh = FakeSSH(has_tar=False)
    fs, _, sftp = make_fs(tmp_path, transfer=TRANSFER_TAR, ssh=ssh)
    local = tmp_path / "local"

    fs.get_dir(root, local)
    (local / "top.txt").unlink()
    fs.get_dir(root, local)

    assert_tree(local, files)
    assert len(sftp.opened()) == len(files) + 1
    # Probed once, never ran tar.
    assert ssh.commands == ["command -v tar"]


# --------------------------------------------------------------------------
# rmtree
# --------------------------------------------------------------------------
def test_rmtree_removes_nested_tree(tmp_path, remote_tree):
    root, _ = remote_tree
    fs, _, _ = make_fs(tmp_path)

    fs.rmtree(root)

    assert not root.exists()


def test_rmtree_does_not_follow_symlinks(tmp_path, remote_tree):
    root, _ = remote_tree
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious.txt").write_text("keep")
    os.symlink(outside, root / "link_to_outside")
    fs, _, _ = make_fs(tmp_path)

    fs.rmtree(str(root))

    assert not root.exists()
    assert (outside / "precious.txt").read_text() == "keep"


# --------------------------------------------------------------------------
# Config plumbing
# --------------------------------------------------------------------------
def _cfg(**kw):
    base = dict(
        remote_machine="m",
        partition="p",
        conda_env="e",
        conda_source="s",
        savepath="/x",
        template="t.j2",
    )
    return ClusterConfig(**{**base, **kw})


def test_cluster_config_transfer_default_and_validation():
    assert _cfg().transfer == TRANSFER_SFTP
    assert _cfg(transfer="tar").transfer == TRANSFER_TAR
    assert ClusterConfig.from_dict(_cfg().to_dict()).transfer == "sftp"
    with pytest.raises(ConfigurationError, match="Unknown transfer"):
        _cfg(transfer="rsync")


def test_cluster_config_remote_fs_passes_transfer(monkeypatch):
    seen = {}

    class _Stub:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(file_handler, "RemoteFileSystem", _Stub)
    _cfg(transfer="tar").remote_fs()
    assert seen["transfer"] == "tar"
    _cfg(transfer="tar").remote_fs(transfer="sftp")
    assert seen["transfer"] == "sftp"


def test_adopt_transfer_is_upgrade_only():
    fs = SimpleNamespace(transfer="sftp")
    tasks._adopt_transfer(fs, _cfg(transfer="tar"))
    assert fs.transfer == "tar"
    # An already non-default choice is kept.
    tasks._adopt_transfer(fs, _cfg(transfer="sftp"))
    assert fs.transfer == "tar"
    # Filesystems without the attribute (LocalFileSystem) are untouched.
    local = SimpleNamespace()
    tasks._adopt_transfer(local, _cfg(transfer="tar"))
    assert not hasattr(local, "transfer")


# --------------------------------------------------------------------------
# Resumable collect
# --------------------------------------------------------------------------
def _collector():
    local = file_handler.LocalFileSystem()
    return tasks.ResultCollector(remote_fs=local, local_fs=local)


def test_collect_directory_resume_keeps_existing(tmp_path, remote_tree):
    root, files = remote_tree
    dest = tmp_path / "results"
    dest.mkdir()
    (dest / "already_here.txt").write_text("keep me")

    _collector().collect_directory(root, dest, resume=True)

    assert (dest / "already_here.txt").read_text() == "keep me"
    assert (dest / "top.txt").read_bytes() == files["top.txt"]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "remote",
        "results",
    ]  # nothing moved aside


def test_collect_directory_default_moves_existing_aside(tmp_path, remote_tree):
    root, _ = remote_tree
    dest = tmp_path / "results"
    dest.mkdir()
    (dest / "old.txt").write_text("old")

    _collector().collect_directory(root, dest)

    assert not (dest / "old.txt").exists()
    aside = [p for p in tmp_path.iterdir() if p.name.startswith("results")]
    assert len(aside) == 2


def test_secure_declare_path_overwrite_handles_directories(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    (d / "f").write_text("x")
    # Used to crash with IsADirectoryError (os.remove on a directory).
    file_utils.secure_declare_path(str(d), move=False, is_directory=True)
    assert not d.exists()
