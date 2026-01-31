import logging
import os
import re
import subprocess
from typing import Optional

import numpy as np
from jinja2 import Environment, FileSystemLoader

from remote_handler import DEFAULT_TEMPLATE_DIR

logger = logging.getLogger(__name__)

TEMPLATE_DIRS = []

def submit_job(script_path, output_path, ssh_command=None, cwd=None) -> str:
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
    result = subprocess.run(
        submit_cmd, shell=True, capture_output=True, text=True
    )
    # In the stdout, we expect to find the job ID
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
    result = subprocess.run(
        submit_cmd, shell=True, capture_output=True, text=True
    )
    
    # Extract and return the job ID from the output
    pid = re.search(r"(\d+)", result.stdout).groups()[0]
    return pid


def check_job_status_sacct(job_ids, ssh_command="ssh lugus"):
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
    """
    Tries to retrieve information on the job using the slurm `squeue`
    command. This is a quite naive method that just checks whether `squeue`
    returns and output for the given job id. If not, it considers the job as
    being finished.

    Args:
        job_id (str): The job ID to check.
        ssh_command (str, optional): SSH command to be used for remote checking.
            Defaults to "ssh lugus".
    Returns:
        bool: True if the job is still running, False if it has finished.
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
    """Custom Jinja2 file system loader that logs which template files are loaded."""
    def get_source(self, environment, template):
        """Override the get_source method to add logging."""
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

    # Create a Jinja2 environment with the custom loader to have logging
    env = Environment(loader=LoggingFileSystemLoader(TEMPLATE_DIRS))
    template = env.get_template(header_template)
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
    header_template="dirona.sh.j2",
    job_type_template="body_single.sh.j2",
    slurm_header_add="",
    exclude_nodes: str = None,
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

    # Create a Jinja2 environment with the custom loader to have logging
    env = Environment(loader=LoggingFileSystemLoader(TEMPLATE_DIRS))

    # Load the specified header template
    template = env.get_template(header_template)

    # Render the template with the provided parameters
    slurm_str = template.render(
        input_file=input_file,
        python_script_path=python_script_path,
        jobname=jobname,
        partition=partition,
        n_cores=n_cores,
        conda_env=conda_env,
        slurm_header_add=slurm_header_add,
        job_type_template=job_type_template,
        exclude_nodes=exclude_nodes,
    )

    return slurm_str
