import subprocess

import pytest

from remote_handler.core import resilience, slurm_utils


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
    """A non-'invalid job id' failure is a real error and must propagate."""

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1, cmd=cmd, stderr="squeue: error: some other failure\n"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        slurm_utils.check_job_status_squeue(42, ssh_command="ssh musica")


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
