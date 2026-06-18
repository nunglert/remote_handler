"""Connection resilience: pause-and-retry instead of crashing on auth loss.

Some clusters (e.g. ``lugus`` with ``step ssh login``) only grant SSH access
for a fixed window (24h); when the credential expires mid-run the host becomes
unreachable until a human refreshes it. The SLURM job itself keeps running on
the cluster -- only the driver loses contact -- so the right behaviour is to
*pause* the driver and retry until the credential is refreshed, not to crash.

This module provides that, gated behind an opt-in ``ConnectionPolicy`` so it is
a pure pass-through for the (vast majority of) machines that don't need it.

Two SSH paths in this package fail differently, and both are covered:
  * paramiko / SFTP (``RemoteFileSystem``) -- raises ``SSHException`` /
    ``RemoteConnectionError`` / ``EOFError`` / socket ``OSError``;
  * subprocess ``ssh <host> '...'`` (``slurm_utils``) -- raises
    ``CalledProcessError`` whose ``stderr`` carries the ssh error text.

``is_connection_error`` classifies both; ``run_resilient`` runs an operation
under a policy, pausing on connection/auth errors and resuming automatically.
"""

import errno
import logging
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Optional

import paramiko

from remote_handler.exceptions import RemoteConnectionError

logger = logging.getLogger(__name__)

# Substrings (lower-cased) that mark an *ssh-level* failure in the stderr of a
# ``ssh <host> '...'`` subprocess. These identify connectivity/auth problems as
# opposed to a remote command (e.g. sbatch/squeue) that ran but exited != 0.
_SSH_STDERR_MARKERS = (
    "permission denied",
    "connection refused",
    "connection closed",
    "connection reset",
    "connection timed out",
    "could not resolve hostname",
    "name or service not known",
    "host key verification failed",
    "no route to host",
    "network is unreachable",
    "operation timed out",
    "kex_exchange_identification",
    "broken pipe",
    "remote host identification has changed",
    "certificate",  # expired / invalid step certificate
    "too many authentication failures",
)

# OSError errnos that mean "the socket/connection died", as opposed to e.g.
# ENOENT (a missing remote file), which must NOT trigger an indefinite retry.
_CONN_ERRNOS = frozenset(
    {
        errno.ECONNRESET,
        errno.ECONNREFUSED,
        errno.ECONNABORTED,
        errno.EPIPE,
        errno.ETIMEDOUT,
        errno.EHOSTUNREACH,
        errno.ENETUNREACH,
        errno.ENOTCONN,
        errno.ESHUTDOWN,
    }
)

_OSERROR_MARKERS = ("socket is closed", "connection", "broken pipe")


def is_connection_error(exc: BaseException) -> bool:
    """True if ``exc`` is a transient connectivity/authentication failure.

    Deliberately conservative: a genuine remote error (missing file, rejected
    sbatch script) must return ``False`` so it propagates instead of looping
    forever.
    """
    # paramiko + socket level (AuthenticationException is an SSHException).
    if isinstance(
        exc,
        (
            paramiko.SSHException,
            RemoteConnectionError,
            EOFError,
            socket.gaierror,
            socket.timeout,
            ConnectionError,
        ),
    ):
        return True

    # subprocess ``ssh ...`` -- inspect stderr for ssh-level markers.
    if isinstance(exc, subprocess.CalledProcessError):
        stderr = exc.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        low = stderr.lower()
        return any(marker in low for marker in _SSH_STDERR_MARKERS)

    # Bare socket OSErrors (ConnectionError is already handled above).
    if isinstance(exc, OSError):
        if exc.errno in _CONN_ERRNOS:
            return True
        text = str(exc).lower()
        return any(marker in text for marker in _OSERROR_MARKERS)

    return False


@dataclass
class ConnectionPolicy:
    """How to behave when an SSH operation hits a connection/auth failure.

    Attributes:
        enabled: Master switch. When False, ``run_resilient`` is a no-op
            pass-through (today's behaviour). Off by default.
        poll_interval: Seconds to wait between retries while paused.
        max_wait: Give up (re-raise) after this many seconds of being paused.
            ``None`` means wait indefinitely until the credential is refreshed.
        sleep / monotonic: Injectable clock, for tests.
    """

    enabled: bool = False
    poll_interval: float = 60.0
    max_wait: Optional[float] = None
    sleep: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.monotonic


# Shared sentinel for "feature off". Never mutate this instance.
DISABLED = ConnectionPolicy()


def _safe_call(hook: Optional[Callable], *args) -> None:
    """Invoke an optional hook, never letting it break the retry loop."""
    if hook is None:
        return
    try:
        hook(*args)
    except Exception:  # noqa: BLE001 - a hook must never crash the loop
        logger.exception("connection-resilience hook raised; ignoring")


def run_resilient(
    fn: Callable,
    policy: ConnectionPolicy,
    *,
    before_retry: Optional[Callable[[], None]] = None,
    on_pending: Optional[Callable[[BaseException], None]] = None,
    on_restored: Optional[Callable[[], None]] = None,
    label: str = "remote operation",
):
    """Run ``fn`` under ``policy``, pausing on connection/auth failures.

    When ``policy.enabled`` is False this simply returns ``fn()`` -- identical
    to calling ``fn`` directly. When enabled, a connection/auth error pauses
    the call: it logs once, optionally runs ``before_retry`` (e.g. reconnect)
    after each wait, and retries every ``policy.poll_interval`` seconds until
    ``fn`` succeeds (then returns its result) or ``policy.max_wait`` elapses
    (then re-raises). Non-connection errors propagate immediately.

    Args:
        fn: The operation to run; its return value is returned on success.
        before_retry: Optional callable run after each wait, before retrying
            ``fn`` (used by the SFTP path to re-establish the paramiko client).
            Its own connection errors keep the loop paused; other errors raise.
        on_pending / on_restored: Optional notification hooks.
        label: Human-readable name for log messages.
    """
    if not policy.enabled:
        return fn()

    start = policy.monotonic()
    paused = False
    while True:
        try:
            result = fn()
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # noqa: BLE001
            if not is_connection_error(exc):
                raise
            if (
                policy.max_wait is not None
                and policy.monotonic() - start >= policy.max_wait
            ):
                logger.error(
                    "%s: still unreachable after %.0fs (max_wait); giving up.",
                    label,
                    policy.max_wait,
                )
                raise
            if not paused:
                paused = True
                logger.warning(
                    "%s: connection/auth lost (%s). Pausing -- retrying every "
                    "%.0fs until restored. If the credential expired, refresh "
                    "it (e.g. run `step ssh login` on the driver host).",
                    label,
                    exc,
                    policy.poll_interval,
                )
                _safe_call(on_pending, exc)
            policy.sleep(policy.poll_interval)
            if before_retry is not None:
                try:
                    before_retry()
                except KeyboardInterrupt:
                    raise
                except BaseException as rexc:  # noqa: BLE001
                    if not is_connection_error(rexc):
                        raise
                    # Reconnect still failing -- remain paused, try again.
            continue

        if paused:
            logger.info("%s: connection restored; resuming.", label)
            _safe_call(on_restored)
        return result
