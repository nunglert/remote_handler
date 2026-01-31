import logging
import os
import re
import subprocess
from typing import Optional

import numpy as np
from jinja2 import Environment, FileSystemLoader

from remote_handler import DEFAULT_TEMPLATE_DIR

logger = logging.getLogger(__name__)

CUSTOM_TEMPLATE_DIR = os.environ.get("REMOTE_HANDLER_SLURM_TEMPLATES")
TEMPLATE_DIRS = []
if CUSTOM_TEMPLATE_DIR is not None:
    TEMPLATE_DIRS.append(CUSTOM_TEMPLATE_DIR)
    # for template_path in CUSTOM_TEMPLATE_DIR:
# TEMPLATE_DIRS.append(DEFAULT_TEMPLATE_DIR)


def submit_job(script_path, output_path, ssh_command=None, cwd=None):
    # Submit job via SSH and capture job ID
    command = ""
    if cwd is not None:
        command += f"cd {cwd}; "

    command += f"sbatch -e {output_path}/slurm.err -o {output_path}/slurm.out {script_path}"

    if ssh_command is not None:
        submit_cmd = f"{ssh_command} '{command}'"
    else:
        submit_cmd = command

    result = subprocess.run(
        submit_cmd, shell=True, capture_output=True, text=True
    )
    match = re.search("(\d+)", result.stdout)
    if match is not None:
        return match.groups()[0]
    else:
        raise Exception(result.stderr)


def submit_job_array(
    script_path: os.PathLike,
    output_path: os.PathLike,
    num_array_tasks: int,
    num_parallel: int = 10,
    ssh_command: Optional[str] = None,
    cwd: Optional[os.PathLike] = None,
) -> str:
    command = ""
    if cwd is not None:
        command += f"cd {cwd}; "

    command += f"sbatch --output={output_path}/slurm_%A_%a.out --error={output_path}/slurm_%A_%a.err --array=1-{num_array_tasks}%{num_parallel} {script_path}"
    # slurm_command = f"sbatch -e {output_path}/slurm.err -o {output_path}/slurm.out {script_path}"

    if ssh_command is not None:
        submit_cmd = f"{ssh_command} '{command}'"
    else:
        submit_cmd = command

    result = subprocess.run(
        submit_cmd, shell=True, capture_output=True, text=True
    )
    # logger.debug(result["stderr"])

    pid = re.search(r"(\d+)", result.stdout).groups()[0]
    return pid


def check_job_status_sacct(job_ids, ssh_command="ssh lugus"):
    """Tries to retrieve information on the job using the slurm `sacct`
    functionality. I designed this for arrays. Somehow sacct does not work
    on all clusters, so you might want to use `check_job_status_squeue`.

    In principle this function would allow for more control, as it can also
    indicate whether a run failed or succesfully finished.
    """
    if len(job_ids) > 0:
        pid_str = ""
        for a in job_ids[:-1]:
            pid_str += str(a) + ","
        pid_str += str(job_ids[-1])

        status_cmd = f"{ssh_command} 'sacct -j {pid_str} --format=JobID,State'"
        result = subprocess.run(
            status_cmd, shell=True, capture_output=True, text=True
        )
        logger.debug(result["stderr"])
        status = [s.split()[-1] for s in result.stdout.split("\n")[2:-1]]
        return [
            True if i in ["COMPLETED", "FAILED"] else False for i in status
        ]
    else:
        return []


def check_job_status_squeue(job_id, ssh_command="ssh lugus"):
    """Tries to retrieve information on the job using the slurm `squeue`
    command. This is a quite naive method that just checks whether `squeue`
    returns and output for the given job id. If not, it considers the job as
    being finished.
    """
    pid_str = str(job_id)
    status_cmd = f"{ssh_command} 'squeue -j {pid_str}'"
    result = subprocess.run(
        status_cmd, shell=True, capture_output=True, text=True
    )
    res = result.stdout.split("\n")
    if len(res) > 2:
        return False
    else:
        return True


class LoggingFileSystemLoader(FileSystemLoader):
    def get_source(self, environment, template):
        source, filename, uptodate = super().get_source(environment, template)
        logger.info(f"{template} (resolved to {filename})")
        return source, filename, uptodate


def get_array_slurm_str(
    calcdir,
    python_script_path,
    num_array_tasks=1,
    num_parallel=10,
    jobname="job",
    partition="normal",
    n_processors=8,
    conda_env="jaxnest",
    map_indices=None,
    header_template="lugus.sh.j2",
    job_type_template="body_array.sh.j2",
    slurm_header_add="",
    per_task=1,
):
    """The directories contained in TEMPLATE_DIRS will be scanned in order
    until the first succesful hit of a `header_template` file."""
    if map_indices is None:
        map_indices = np.arange(num_array_tasks)

    env = Environment(loader=LoggingFileSystemLoader(TEMPLATE_DIRS))
    template = env.get_template(header_template)

    slurm_str = template.render(
        calcdir=calcdir,
        python_script_path=python_script_path,
        num_array_tasks=num_array_tasks,
        num_parallel=num_parallel,
        jobname=jobname,
        partition=partition,
        n_cores=n_processors,
        conda_env=conda_env,
        map_indices=map_indices,
        slurm_header_add=slurm_header_add,
        job_type_template=job_type_template,
        per_task=per_task,
    )

    return slurm_str


def get_slurm_str(
    input_file,
    python_script_path,
    jobname="job",
    partition="normal",
    n_cores=8,
    conda_env="jaxnest",
    header_template="lugus.sh.j2",
    job_type_template="body_single.sh.j2",
    slurm_header_add="",
):
    env = Environment(loader=LoggingFileSystemLoader(TEMPLATE_DIRS))
    template = env.get_template(header_template)

    slurm_str = template.render(
        input_file=input_file,
        python_script_path=python_script_path,
        jobname=jobname,
        partition=partition,
        n_cores=n_cores,
        conda_env=conda_env,
        slurm_header_add=slurm_header_add,
        job_type_template=job_type_template,
    )

    return slurm_str
