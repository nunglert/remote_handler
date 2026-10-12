"""How the connection policy reaches the SFTP layer (no network, no sleeping).

Covers the paths that used to run unprotected even with
``resilient_connection: true``: the ``RemoteFileSystem`` constructor's
initial connect, and the first SFTP calls a task makes while constructing.
"""

from unittest.mock import MagicMock

import paramiko
import pytest

from remote_handler.core import file_handler, resilience, tasks
from remote_handler.core.cluster_config import ClusterConfig
from remote_handler.core.ssh_connection import SSHClientProvider
from remote_handler.exceptions import ConfigurationError, RemoteConnectionError


class FakeClock:
    """Deterministic clock: sleeping advances virtual time, never blocks."""

    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.t += seconds

    def monotonic(self):
        return self.t


def _policy(clock, **kw):
    return resilience.ConnectionPolicy(
        enabled=True,
        poll_interval=5.0,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        **kw,
    )


class FlakyProvider(SSHClientProvider):
    """Raises the queued errors on successive connects, then succeeds."""

    def __init__(self, errors, known_hosts=()):
        self.errors = list(errors)
        self.connects = 0
        self.cleanups = 0
        self.config_parser = MagicMock()
        self.config_parser.get_available_hosts.return_value = list(known_hosts)
        self.ssh = MagicMock(spec=paramiko.SSHClient)
        self.sftp = MagicMock(spec=paramiko.SFTPClient)

    def create_connection(self):
        self.connects += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.ssh, self.sftp

    def cleanup(self):
        self.cleanups += 1


def _expired():
    return paramiko.AuthenticationException("certificate expired")


# --------------------------------------------------------------------------
# RemoteFileSystem constructor
# --------------------------------------------------------------------------


def test_constructor_pauses_on_expired_cert_until_refreshed():
    clock = FakeClock()
    provider = FlakyProvider([_expired(), _expired()])

    fs = file_handler.RemoteFileSystem(
        "musica",
        ssh_client_provider=provider,
        connection_policy=_policy(clock),
    )

    assert provider.connects == 3
    assert clock.sleeps == [5.0, 5.0]
    assert fs.sftp_client is provider.sftp
    # Each attempt first releases what the failed one left half-open.
    assert provider.cleanups == 3


def test_constructor_without_policy_fails_fast_as_before():
    provider = FlakyProvider([_expired()])

    with pytest.raises(paramiko.AuthenticationException):
        file_handler.RemoteFileSystem("musica", ssh_client_provider=provider)
    assert provider.connects == 1


def test_constructor_gives_up_after_max_wait():
    clock = FakeClock()
    provider = FlakyProvider([_expired()] * 10)

    with pytest.raises(paramiko.AuthenticationException):
        file_handler.RemoteFileSystem(
            "musica",
            ssh_client_provider=provider,
            connection_policy=_policy(clock, max_wait=10.0),
        )
    assert clock.t == 10.0


def test_constructor_unknown_alias_is_fatal_not_retried():
    """A misspelled alias must not loop forever under an enabled policy."""
    import socket

    clock = FakeClock()
    provider = FlakyProvider(
        [socket.gaierror("Name or service not known")] * 10,
        known_hosts=["musica"],
    )

    with pytest.raises(ConfigurationError, match="musicaa"):
        file_handler.RemoteFileSystem(
            "musicaa",
            ssh_client_provider=provider,
            connection_policy=_policy(clock),
        )
    assert provider.connects == 1
    assert clock.sleeps == []


def test_constructor_known_alias_dns_failure_is_retried():
    import socket

    clock = FakeClock()
    provider = FlakyProvider(
        [socket.gaierror("Temporary failure in name resolution")],
        known_hosts=["musica"],
    )

    file_handler.RemoteFileSystem(
        "musica",
        ssh_client_provider=provider,
        connection_policy=_policy(clock),
    )
    assert provider.connects == 2


def test_constructor_known_alias_dns_failure_without_policy():
    import socket

    provider = FlakyProvider(
        [socket.gaierror("Temporary failure in name resolution")],
        known_hosts=["musica"],
    )

    with pytest.raises(RemoteConnectionError, match="Could not resolve"):
        file_handler.RemoteFileSystem("musica", ssh_client_provider=provider)


# --------------------------------------------------------------------------
# ClusterConfig.remote_fs factory
# --------------------------------------------------------------------------


def _cluster_config(savepath="/scratch", **kw):
    return ClusterConfig(
        remote_machine="musica",
        partition="p",
        conda_env="",
        conda_source="",
        savepath=str(savepath),
        template="unused.sh.j2",
        **kw,
    )


def test_cluster_config_remote_fs_carries_policy():
    cfg = _cluster_config(
        resilient_connection=True, connection_retry_interval=7.0
    )
    provider = FlakyProvider([])

    fs = cfg.remote_fs(ssh_client_provider=provider)

    assert fs.remote_alias == "musica"
    assert fs.connection_policy.enabled
    assert fs.connection_policy.poll_interval == 7.0


def test_cluster_config_remote_fs_explicit_policy_wins():
    cfg = _cluster_config(resilient_connection=True)

    fs = cfg.remote_fs(
        ssh_client_provider=FlakyProvider([]),
        connection_policy=resilience.DISABLED,
    )

    assert fs.connection_policy is resilience.DISABLED


# --------------------------------------------------------------------------
# Tasks: policy attached before the first SFTP call, upgrade-only
# --------------------------------------------------------------------------


class _RecordingFS(file_handler.LocalFileSystem):
    """Local FS that records the policy in force at its first exists()."""

    def __init__(self, policy):
        super().__init__()
        self.connection_policy = policy
        self.policy_at_first_call = None

    def exists(self, path):
        if self.policy_at_first_call is None:
            self.policy_at_first_call = self.connection_policy
        return super().exists(path)


def _make_task(kind, tmp_path, cfg, fs):
    remote = tmp_path / "remote"
    (remote / "job").mkdir(parents=True, exist_ok=True)
    common = dict(
        local_savepath=tmp_path / "results",
        cluster_config=cfg,
        remote_fs=fs,
        remote_savepath=str(remote),
        jobname="job",
        collect_only=True,
    )
    if kind == "single":
        return tasks.SingleScriptTask(
            script_file=None, input_file=None, **common
        )
    return tasks.ArrayScriptTask(num_array_tasks=1, **common)


@pytest.mark.parametrize("kind", ["single", "array"])
def test_task_attaches_policy_before_first_sftp_call(kind, tmp_path):
    cfg = _cluster_config(
        savepath=tmp_path, resilient_connection=True, executor="direct"
    )
    fs = _RecordingFS(resilience.DISABLED)

    _make_task(kind, tmp_path, cfg, fs)

    assert fs.policy_at_first_call.enabled
    assert fs.connection_policy.enabled


@pytest.mark.parametrize("kind", ["single", "array"])
def test_task_does_not_downgrade_enabled_policy(kind, tmp_path):
    cfg = _cluster_config(savepath=tmp_path, executor="direct")
    assert not cfg.connection_policy.enabled
    explicit = resilience.ConnectionPolicy(enabled=True)
    fs = _RecordingFS(explicit)

    _make_task(kind, tmp_path, cfg, fs)

    assert fs.connection_policy is explicit
