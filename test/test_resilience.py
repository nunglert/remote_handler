"""Unit tests for the connection-resilience layer (no network, no sleeping)."""

import socket
import subprocess

import paramiko
import pytest

from remote_handler.core import resilience
from remote_handler.core.resilience import (
    ConnectionPolicy,
    is_connection_error,
    run_resilient,
)
from remote_handler.exceptions import RemoteConnectionError


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


def _calledproc(stderr):
    return subprocess.CalledProcessError(
        returncode=255, cmd="ssh host '...'", stderr=stderr
    )


# --------------------------------------------------------------------------
# is_connection_error
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "exc",
    [
        paramiko.ssh_exception.AuthenticationException("denied"),
        paramiko.SSHException("Server connection dropped"),
        RemoteConnectionError("unresolved alias"),
        EOFError(),
        socket.gaierror("name resolution"),
        socket.timeout("timed out"),
        ConnectionResetError("reset"),
        OSError("Socket is closed"),
        _calledproc("Permission denied (publickey)."),
        _calledproc("ssh: connect to host x port 22: Connection refused"),
        _calledproc("kex_exchange_identification: Connection closed"),
    ],
)
def test_connection_errors_are_classified_transient(exc):
    assert is_connection_error(exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        _calledproc("sbatch: error: invalid partition specified"),
        _calledproc("slurm_load_jobs error: Invalid job id specified"),
        FileNotFoundError("no such file"),  # OSError ENOENT must NOT loop
        ValueError("nonsense"),
        KeyError("x"),
    ],
)
def test_real_errors_are_not_transient(exc):
    assert is_connection_error(exc) is False


def test_calledprocesserror_with_bytes_stderr():
    exc = subprocess.CalledProcessError(
        255, "ssh ...", stderr=b"Permission denied (publickey)."
    )
    assert is_connection_error(exc) is True


# --------------------------------------------------------------------------
# run_resilient -- disabled policy is a pure pass-through
# --------------------------------------------------------------------------
def test_disabled_policy_calls_fn_once_and_returns():
    calls = []

    def fn():
        calls.append(1)
        return "value"

    assert run_resilient(fn, resilience.DISABLED) == "value"
    assert len(calls) == 1


def test_disabled_policy_reraises_immediately():
    def fn():
        raise EOFError()

    with pytest.raises(EOFError):
        run_resilient(fn, ConnectionPolicy(enabled=False))


# --------------------------------------------------------------------------
# run_resilient -- enabled policy pauses and resumes
# --------------------------------------------------------------------------
def test_enabled_retries_until_success():
    clock = FakeClock()
    policy = ConnectionPolicy(
        enabled=True,
        poll_interval=10,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    state = {"n": 0}
    pend, restore = [], []

    def fn():
        if state["n"] < 3:
            state["n"] += 1
            raise RemoteConnectionError("auth expired")
        return "done"

    result = run_resilient(
        fn,
        policy,
        on_pending=lambda exc: pend.append(exc),
        on_restored=lambda: restore.append(True),
    )
    assert result == "done"
    assert clock.sleeps == [10, 10, 10]  # paused 3 times
    assert len(pend) == 1  # announced exactly once
    assert len(restore) == 1  # restored exactly once


def test_enabled_non_connection_error_propagates_without_pausing():
    clock = FakeClock()
    policy = ConnectionPolicy(
        enabled=True, sleep=clock.sleep, monotonic=clock.monotonic
    )
    pend = []

    def fn():
        raise ValueError("real bug")

    with pytest.raises(ValueError):
        run_resilient(fn, policy, on_pending=lambda e: pend.append(e))
    assert clock.sleeps == []
    assert pend == []


def test_enabled_gives_up_after_max_wait():
    clock = FakeClock()
    policy = ConnectionPolicy(
        enabled=True,
        poll_interval=10,
        max_wait=25,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    def fn():
        raise EOFError()  # never recovers

    with pytest.raises(EOFError):
        run_resilient(fn, policy)
    # 0 -> sleep 10 -> 10 -> sleep 10 -> 20 -> sleep 10 -> 30 >= 25 -> raise
    assert clock.sleeps == [10, 10, 10]


def test_before_retry_runs_between_attempts():
    clock = FakeClock()
    policy = ConnectionPolicy(
        enabled=True,
        poll_interval=5,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    state = {"n": 0}
    reconnects = []

    def fn():
        if state["n"] < 2:
            state["n"] += 1
            raise paramiko.SSHException("dropped")
        return "ok"

    run_resilient(fn, policy, before_retry=lambda: reconnects.append(True))
    assert reconnects == [True, True]  # one reconnect before each retry


def test_before_retry_connection_failure_keeps_pausing():
    clock = FakeClock()
    policy = ConnectionPolicy(
        enabled=True,
        poll_interval=5,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    state = {"fn": 0, "rc": 0}

    def fn():
        state["fn"] += 1
        if state["fn"] <= 2:
            raise EOFError()
        return "recovered"

    def before_retry():
        # First reconnect attempt also fails (auth still down), then succeeds.
        state["rc"] += 1
        if state["rc"] == 1:
            raise RemoteConnectionError("still down")

    assert run_resilient(fn, policy, before_retry=before_retry) == "recovered"


def test_keyboard_interrupt_propagates():
    policy = ConnectionPolicy(enabled=True)

    def fn():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        run_resilient(fn, policy)
