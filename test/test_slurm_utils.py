import subprocess

import pytest

from remote_handler.core import resilience, slurm_utils
from remote_handler.exceptions import DependencyError, RemoteConnectionError


def test_validate_slurm_missing_sbatch_raises_dependency(monkeypatch):
    """`which sbatch` truly absent runs on the remote and exits 1; that is a
    genuine missing-dependency, not a connection failure."""

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(DependencyError, match="not found"):
        slurm_utils.validate_slurm_available(
            "ssh musica", "musica", resilience.DISABLED
        )


def test_validate_slurm_ssh_255_raises_connection_error(monkeypatch):
    """ssh's own connect/auth failure exits 255 -- an expired credential, not
    a missing sbatch. It must surface as RemoteConnectionError."""

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=255, cmd=cmd, stderr="Permission denied (publickey).\n"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RemoteConnectionError, match="expired"):
        slurm_utils.validate_slurm_available(
            "ssh musica", "musica", resilience.DISABLED
        )


def test_validate_slurm_timeout_raises_dependency(monkeypatch):
    """A hung ssh probe (killed by command_timeout) reports as a timed-out
    probe with a refresh hint, under a disabled policy."""

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(DependencyError, match="Timed out"):
        slurm_utils.validate_slurm_available(
            "ssh musica", "musica", resilience.DISABLED
        )


def test_validate_slurm_pauses_and_retries_under_enabled_policy(monkeypatch):
    """With resilient_connection on, an expired-credential 255 pauses-and-
    retries here (same as submit/status) instead of crashing; once the
    credential is refreshed the probe succeeds and construction proceeds."""
    calls = {"n": 0}

    def fake_run(cmd, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise subprocess.CalledProcessError(
                returncode=255,
                cmd=cmd,
                stderr="Permission denied (publickey).\n",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    slept = []
    policy = resilience.ConnectionPolicy(
        enabled=True, poll_interval=1, sleep=slept.append
    )
    slurm_utils.validate_slurm_available("ssh musica", "musica", policy)
    assert calls["n"] == 3  # failed twice, then succeeded
    assert slept  # it paused between attempts


def test_run_ssh_closes_stdin_and_bounds_the_call(monkeypatch):
    """The subprocess ssh call must never inherit the terminal (stdin closed)
    and must carry a finite timeout, so an expired credential can't park it at
    an interactive prompt -- the bug this guards against."""
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(cmd, 0, stdout="12345", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    policy = resilience.ConnectionPolicy(command_timeout=42.0)
    slurm_utils._run_ssh("ssh host 'sbatch job.sh'", policy)

    assert seen["stdin"] is subprocess.DEVNULL
    assert seen["timeout"] == 42.0
    # No controlling terminal -> an interactive helper (e.g. step) can't grab
    # /dev/tty to draw a QR prompt.
    assert seen["start_new_session"] is True


def test_run_ssh_retries_a_hung_ssh_under_enabled_policy(monkeypatch):
    """A ``TimeoutExpired`` from a hung ssh pauses-and-retries instead of
    propagating, once the policy is enabled."""
    calls = {"n": 0}

    def fake_run(cmd, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs["timeout"])
        return subprocess.CompletedProcess(cmd, 0, stdout="777", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    policy = resilience.ConnectionPolicy(
        enabled=True, poll_interval=1, sleep=lambda s: None
    )
    result = slurm_utils._run_ssh("ssh host 'squeue'", policy)
    assert result.stdout == "777"
    assert calls["n"] == 2  # hung once, then succeeded


def test_squeue_treats_purged_job_as_finished(monkeypatch):
    """A job that left the queue makes ``squeue -j`` exit non-zero with
    'Invalid job id specified'; that must read as *finished*, not crash -- the
    exact failure seen when a job completes while the connection is briefly
    down during a credential refresh."""

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=cmd,
            stderr="slurm_load_jobs error: Invalid job id specified\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert (
        slurm_utils.check_job_status_squeue(1518633, ssh_command="ssh musica")
        is True
    )


def test_squeue_reraises_other_errors(monkeypatch):
    """A non-'invalid job id' failure that persists through every retry is a
    real error and must propagate."""
    calls = {"n": 0}

    def fake_run(cmd, **kwargs):
        calls["n"] += 1
        raise subprocess.CalledProcessError(
            returncode=1, cmd=cmd, stderr="squeue: error: some other failure\n"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        slurm_utils.check_job_status_squeue(
            42, ssh_command="ssh musica", retries=3, retry_wait=0
        )
    assert calls["n"] == 4


def test_squeue_retries_transient_errors(monkeypatch):
    """An unresponsive controller makes ``squeue`` exit non-zero while the
    job is still running; a retry that succeeds must report the job's real
    state instead of crashing."""
    calls = {"n": 0}

    def fake_run(cmd, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=cmd,
                stderr="slurm_load_jobs error: Socket timed out on "
                "send/recv operation\n",
            )
        return subprocess.CompletedProcess(
            cmd, 0, stdout="JOBID PART NAME\n123 normal job\n\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert (
        slurm_utils.check_job_status_squeue(
            123, ssh_command="ssh musica", retry_wait=0
        )
        is False
    )
    assert calls["n"] == 3


def test_squeue_running_and_finished_by_empty_output(monkeypatch):
    """The normal (exit 0) paths: rows -> running; header-only -> finished."""
    outputs = {
        "running": "JOBID PART NAME\n123 normal job\nextra\n",
        "finished": "JOBID PART NAME\n",
    }

    def make_run(key):
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd, 0, stdout=outputs[key], stderr=""
            )

        return fake_run

    monkeypatch.setattr(subprocess, "run", make_run("running"))
    assert slurm_utils.check_job_status_squeue(1) is False
    monkeypatch.setattr(subprocess, "run", make_run("finished"))
    assert slurm_utils.check_job_status_squeue(1) is True


def test_get_slurm_str(cluster_config):
    """Test generating single SLURM job script."""
    calcdir = "kek"
    slurm_script = "lel"

    slurm_str = slurm_utils.get_slurm_str(
        calcdir,
        slurm_script,
        header_template=cluster_config.template,
    )

    print(slurm_str)


def test_get_array_slurm_str(cluster_config):
    """Test generating SLURM array job script."""
    calcdir = "kek"
    slurm_script = "lel"

    slurm_str = slurm_utils.get_array_slurm_str(
        calcdir,
        slurm_script,
        header_template=cluster_config.template,
    )

    print(slurm_str)
