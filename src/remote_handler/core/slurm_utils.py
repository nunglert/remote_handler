import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
from jinja2 import Environment, FileSystemLoader

from remote_handler.config import get_config
from remote_handler.core import remote_env, resilience
from remote_handler.exceptions import (
    DependencyError,
    RemoteConnectionError,
    TaskError,
)

logger = logging.getLogger(__name__)


def _run_ssh(cmd: str, policy: resilience.ConnectionPolicy):
    """Run a (possibly ssh-wrapped) shell command under a connection policy.

    Hardened against interactive-prompt hangs three ways: ``stdin`` is closed
    (``DEVNULL``); the child runs in its own session (``start_new_session``) so
    it has no controlling terminal and a helper like ``step`` cannot grab
    ``/dev/tty`` to draw a QR/MFA prompt (it fails fast instead); and a
    ``policy.command_timeout`` bounds the whole call so anything that still
    blocks surfaces as ``TimeoutExpired``. That timeout is classified transient,
    so with an enabled policy the call pauses and retries until the credential
    is refreshed; with a disabled policy it simply raises (still far better than
    an indefinite hang). A real non-zero exit (e.g. a bad sbatch) propagates as
    before.
    """
    return resilience.run_resilient(
        lambda: subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            timeout=policy.command_timeout,
        ),
        policy,
        label=f"ssh command ({cmd[:60]}...)",
    )


def validate_slurm_available(ssh_command, target, policy):
    """Probe that ``sbatch`` is available on the target system.

    Runs ``which sbatch`` through the same resilient wrapper as
    :func:`submit_job` / the status checks: with an enabled ``policy`` an
    expired/lost credential pauses-and-retries here instead of crashing, and
    ``policy.command_timeout`` still bounds a hung ssh.

    Args:
        ssh_command: The ``ssh <alias>`` prefix, or ``None`` to probe the
            local machine.
        target: Human-readable name of the machine, for error messages.
        policy: The cluster's :class:`resilience.ConnectionPolicy`.

    Raises:
        RemoteConnectionError: The probe failed at the ssh level (expired
            cert, auth, unreachable host) -- ``which sbatch`` truly absent
            exits 1 on the remote, whereas ssh's own connect/auth failure
            exits 255.
        DependencyError: ``sbatch`` is genuinely missing, or a bounded probe
            timed out.
    """
    cmd = "which sbatch"
    if ssh_command is not None:
        cmd = f"{ssh_command} '{cmd}'"
    try:
        _run_ssh(cmd, policy)
    except subprocess.TimeoutExpired:
        raise DependencyError(
            f"Timed out probing SLURM on {target}: the ssh connection did "
            "not respond. The credential may have expired -- refresh it "
            "(e.g. run `step ssh login` on the driver host) and retry."
        )
    except subprocess.CalledProcessError as exc:
        # An ssh-level failure is NOT a missing sbatch: `which sbatch` truly
        # absent exits 1 on the remote; ssh's own connect/auth failure exits
        # 255. `is_connection_error` also matches known stderr markers (e.g.
        # an expired step "certificate").
        if resilience.is_connection_error(exc) or exc.returncode == 255:
            raise RemoteConnectionError(
                f"Cannot reach {target} to probe SLURM (ssh exit "
                f"{exc.returncode}). The credential may have expired -- "
                "refresh it (e.g. run `step ssh login`). Set "
                "resilient_connection=True on this cluster to pause-and-"
                "retry instead of failing."
            ) from exc
        raise DependencyError(
            f"SLURM is not available on {target}: 'sbatch' not found.\n"
            "Ensure SLURM is installed and accessible in PATH."
        )


def submit_job(
    script_path,
    output_path,
    ssh_command=None,
    cwd=None,
    connection_policy: resilience.ConnectionPolicy = resilience.DISABLED,
) -> str:
    """
    Submit a slurm job.

    Args:
        script_path (os.PathLike): Path to the slurm script to be submitted.
        output_path (os.PathLike): Path to the directory where slurm output and error files
            will be written.
        ssh_command (Optional[str], optional): SSH command to be used for remote submission.
            Defaults to None.
        cwd (Optional[os.PathLike], optional): If provided, the command will be executed
            in this directory. Defaults to None.
    Returns:
        str: The job ID of the submitted job.
    """

    command = ""
    # Change to the specified directory if provided
    if cwd is not None:
        command += f"cd {cwd}; "
    # Construct the sbatch command with output and error file paths
    command += f"sbatch -e {output_path}/slurm.err -o {output_path}/slurm.out {script_path}"

    # Submit the job via SSH if a command is provided
    if ssh_command is not None:
        submit_cmd = f"{ssh_command} '{command}'"
    else:
        submit_cmd = command

    # Execute the submission command and capture the output
    result = _run_ssh(submit_cmd, connection_policy)
    # In the stdout, we expect to find the job ID
    match = re.search(r"(\d+)", result.stdout)
    if match is not None:
        return match.groups()[0]
    else:
        raise TaskError(
            f"SLURM job submission failed\n"
            f"Command: {submit_cmd}\n"
            f"Error: {result.stderr}"
        )


def submit_job_array(
    script_path: os.PathLike,
    output_path: os.PathLike,
    num_array_tasks: int,
    num_parallel: int = 10,
    ssh_command: Optional[str] = None,
    cwd: Optional[os.PathLike] = None,
    connection_policy: resilience.ConnectionPolicy = resilience.DISABLED,
) -> str:
    """
    Submit a slurm job array.

    Args:
        script_path (os.PathLike): Path to the slurm script to be submitted.
        output_path (os.PathLike): Path to the directory where slurm output and error files
            will be written.
        num_array_tasks (int): Number of tasks in the job array.
        num_parallel (int, optional): Maximum number of tasks to run in parallel. Defaults to 10.
        ssh_command (Optional[str], optional): SSH command to be used for remote submission.
            Defaults to None.
        cwd (Optional[os.PathLike], optional): If provided, the command will be executed
            in this directory. Defaults to None.
    Returns:
        str: The job ID of the submitted job array.
    """
    command = ""
    if cwd is not None:
        command += f"cd {cwd}; "

    # Construct the sbatch command for job arrays with output and error file paths: %A for job ID, %a for array index
    command += f"sbatch --output={output_path}/slurm_%A_%a.out --error={output_path}/slurm_%A_%a.err --array=1-{num_array_tasks}%{num_parallel} {script_path}"

    # If an ssh_command is provided, submit the job via SSH
    if ssh_command is not None:
        submit_cmd = f"{ssh_command} '{command}'"
    else:
        submit_cmd = command

    # Execute the submission command and capture the output
    result = _run_ssh(submit_cmd, connection_policy)

    # Extract and return the job ID from the output
    pid = re.search(r"(\d+)", result.stdout).groups()[0]
    return pid


def check_job_status_sacct(
    job_ids,
    ssh_command="ssh lugus",
    connection_policy: resilience.ConnectionPolicy = resilience.DISABLED,
):
    """
    Tries to retrieve information on the job using the slurm `sacct`
    functionality. I designed this for arrays. Somehow sacct does not work
    on all clusters, so you might want to use `check_job_status_squeue`.

    In principle this function would allow for more control, as it can also
    indicate whether a run failed or succesfully finished.

    Args:
        job_ids (list): List of job IDs to check.
        ssh_command (str, optional): SSH command to be used for remote checking.
            Defaults to "ssh lugus".
    Returns:
        list: List of booleans indicating whether each job has finished.
    """
    if len(job_ids) > 0:
        pid_str = ""
        for a in job_ids[:-1]:
            pid_str += str(a) + ","
        pid_str += str(job_ids[-1])

        if ssh_command is None:
            status_cmd = f"sacct -j {pid_str} --format=JobID,State"
        else:
            status_cmd = (
                f"{ssh_command} 'sacct -j {pid_str} --format=JobID,State'"
            )
        result = _run_ssh(status_cmd, connection_policy)
        logger.debug(result["stderr"])
        status = [s.split()[-1] for s in result.stdout.split("\n")[2:-1]]
        return [
            True if i in ["COMPLETED", "FAILED"] else False for i in status
        ]
    else:
        return []


def check_job_status_squeue(
    job_id,
    ssh_command="ssh lugus",
    connection_policy: resilience.ConnectionPolicy = resilience.DISABLED,
):
    """
    Tries to retrieve information on the job using the slurm `squeue`
    command. This is a quite naive method that just checks whether `squeue`
    returns and output for the given job id. If not, it considers the job as
    being finished.

    Once a job leaves the queue, ``squeue -j <id>`` may either return just its
    header (exit 0, no rows) *or*, on some SLURM builds, exit non-zero with
    ``slurm_load_jobs error: Invalid job id specified`` -- the controller has
    already purged the completed job. Both mean the same thing: finished. The
    non-zero-exit form is handled explicitly so a job that completes while the
    connection is briefly down (e.g. during a credential refresh) is reported
    as finished rather than crashing the driver. A genuine ssh/connection
    failure still propagates through ``_run_ssh``'s resilience handling.

    Args:
        job_id (str): The job ID to check.
        ssh_command (str, optional): SSH command to be used for remote checking.
            Defaults to "ssh lugus".
    Returns:
        bool: False if the job is still running, True if it has finished.
    """
    pid_str = str(job_id)
    if ssh_command is None:
        status_cmd = f"squeue -j {pid_str}"
    else:
        status_cmd = f"{ssh_command} 'squeue -j {pid_str}'"
    try:
        result = _run_ssh(status_cmd, connection_policy)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        # Job no longer known to the controller -> it has left the queue.
        if "invalid job id" in stderr.lower():
            return True
        raise
    res = result.stdout.split("\n")
    if len(res) > 2:
        return False
    else:
        return True


class LoggingFileSystemLoader(FileSystemLoader):
    """Custom Jinja2 file system loader that logs which template files are loaded."""

    def get_source(self, environment, template):
        """Override the get_source method to add logging."""
        source, filename, uptodate = super().get_source(environment, template)
        logger.info(f"{template} (resolved to {filename})")
        return source, filename, uptodate


def _load_header_template(header_template):
    """Load a SLURM header template by name *or* by explicit path.

    ``header_template`` (the cluster config's ``template`` field) may be:

    * a bare filename (e.g. ``"musica.sh.j2"``) resolved against the template
      search path -- ``REMOTE_HANDLER_SLURM_TEMPLATES`` then the built-ins; or
    * a path to a template file (absolute, or containing a path separator),
      which is loaded directly. Its directory is added to the search path so
      the body ``{% include job_type_template %}`` still resolves against the
      shipped ``body_*.sh.j2``.

    Jinja's ``get_template`` only ever does a *name* lookup within the search
    dirs, so an explicit path handed to it raises ``TemplateNotFound``; this
    bridges that gap.
    """
    template_dirs = list(get_config().get_template_dirs())
    name = header_template

    path = Path(header_template)
    if path.is_absolute() or os.sep in str(header_template):
        if not path.is_file():
            raise TaskError(
                "SLURM header template not found at the given path: "
                f"{header_template}"
            )
        template_dirs = [str(path.parent), *template_dirs]
        name = path.name

    env = Environment(loader=LoggingFileSystemLoader(template_dirs))
    return env.get_template(name)


def get_array_slurm_str(
    calcdir,
    python_script_path,
    num_array_tasks=1,
    num_parallel=10,
    jobname="job",
    partition="normal",
    n_processors=8,
    conda_env="jaxnest",
    conda_source="~/miniconda3",
    map_indices=None,
    header_template="lugus.sh.j2",
    job_type_template="body_array.sh.j2",
    slurm_header_add="",
    per_task=1,
    time_limit=None,
    memory=None,
):
    """
    Get the SLURM script for an array job as a string.
    The directories contained in TEMPLATE_DIRS will be scanned in order
    until the first succesful hit of a `header_template` file.

    Args:
        calcdir (os.PathLike): Path to the calculation directory.
        python_script_path (os.PathLike): Path to the python script to be executed.
        num_array_tasks (int, optional): Number of tasks in the job array. Defaults to 1.
        num_parallel (int, optional): Maximum number of tasks to run in parallel. Defaults to 10.
        jobname (str, optional): Name of the job used in slurm. Defaults to "job".
        partition (str, optional): Name of the partition on the HPC setup. Defaults to "normal".
        n_processors (int, optional): Number of CPU cores to request. Defaults to 8.
        conda_env (str, optional): Name of the conda environment to activate. Defaults to "jaxnest".
        map_indices (Optional[np.ndarray], optional): Specific indices for the array tasks.
            If None, defaults to np.arange(num_array_tasks). Defaults to None.
        header_template (str, optional): Template file for the slurm header. Defaults to "lugus.sh.j2".
        job_type_template (str, optional): Template file for the job body. Defaults to "body_array.sh.j2".
        slurm_header_add (str, optional): Additional lines to add to the slurm header. Defaults to "".
        per_task (int, optional): Number of array tasks per job. Defaults to 1.
    Returns:
        str: The rendered SLURM script as a string.
    """
    if map_indices is None:
        map_indices = np.arange(num_array_tasks)

    # Load the header template (by name on the search path, or explicit path).
    template = _load_header_template(header_template)
    # Render the template with the provided parameters
    slurm_str = template.render(
        calcdir=calcdir,
        python_script_path=python_script_path,
        num_array_tasks=num_array_tasks,
        num_parallel=num_parallel,
        jobname=jobname,
        partition=partition,
        n_cores=n_processors,
        conda_env=conda_env,
        conda_source=conda_source,
        conda_activation=remote_env.conda_activation_script(
            conda_source, conda_env
        ),
        map_indices=map_indices,
        slurm_header_add=slurm_header_add,
        job_type_template=job_type_template,
        per_task=per_task,
        time_limit=time_limit,
        memory=memory,
    )

    return slurm_str


def get_slurm_str(
    input_file,
    python_script_path,
    jobname="job",
    partition="normal",
    n_cores=8,
    conda_env="jaxnest",
    conda_source="/home/user/miniconda3",
    header_template="dirona.sh.j2",
    job_type_template="body_single.sh.j2",
    slurm_header_add="",
    exclude_nodes: str = None,
    time_limit=None,
    memory=None,
):
    """
    Get the SLURM script as a string.
    The directories contained in TEMPLATE_DIRS will be scanned in order
    until the first succesful hit of a `header_template` file.

    Args:
        input_file (os.PathLike): Path to the input file for the python script.
        python_script_path (os.PathLike): Path to the python script to be executed.
        jobname (str, optional): Name of the job used in slurm. Defaults to "job".
        partition (str, optional): Name of the partition on the HPC setup. Defaults to "normal".
        n_cores (int, optional): Number of CPU cores to request. Defaults to 8.
        conda_env (str, optional): Name of the conda environment to activate. Defaults to "jaxnest".
        header_template (str, optional): Template file for the slurm header. Defaults to "lugus.sh.j2".
        job_type_template (str, optional): Template file for the job body. Defaults to "body_single.sh.j2".
        slurm_header_add (str, optional): Additional lines to add to the slurm header. Defaults to "".
        exclude_nodes (str, optional): Comma-separated string of nodes to exclude. Defaults to None.
    Returns:
        str: The rendered SLURM script as a string.
    """

    # Load the header template (by name on the search path, or explicit path).
    template = _load_header_template(header_template)

    # Render the template with the provided parameters
    slurm_str = template.render(
        input_file=input_file,
        python_script_path=python_script_path,
        jobname=jobname,
        partition=partition,
        n_cores=n_cores,
        conda_env=conda_env,
        conda_source=conda_source,
        conda_activation=remote_env.conda_activation_script(
            conda_source, conda_env
        ),
        slurm_header_add=slurm_header_add,
        job_type_template=job_type_template,
        exclude_nodes=exclude_nodes,
        time_limit=time_limit,
        memory=memory,
    )

    return slurm_str
