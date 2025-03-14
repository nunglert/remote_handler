import re
import subprocess
from jinja2 import Environment, FileSystemLoader
import numpy as np

from remote_handler import TEMPLATE_DIR

def submit_job(script_path, output_path, ssh_command=None):
    # Submit job via SSH and capture job ID
    slurm_command = f"sbatch -e {output_path}/slurm.err -o {output_path}/slurm.out {script_path}"

    if ssh_command is not None:
        submit_cmd = f"{ssh_command} '{slurm_command}'"
    else:
        submit_cmd = slurm_command

    result = subprocess.run(
        submit_cmd, shell=True, capture_output=True, text=True
    )
    print(result)
    match = re.search("(\d+)", result.stdout)
    if match is not None:
        return match.groups()[0]
    else:
        raise Exception(result.stderr)
    

def submit_job_array(
    script_path,
    output_path,
    num_array_tasks,
    num_parallel=10,
    ssh_command=None,
):
    # Submit job via SSH and capture job ID
    slurm_command = f"sbatch --output={output_path}/slurm_%A_%a.out --error={output_path}/slurm_%A_%a.err --array=1-{num_array_tasks}%{num_parallel} {script_path}"
    # slurm_command = f"sbatch -e {output_path}/slurm.err -o {output_path}/slurm.out {script_path}"

    if ssh_command is not None:
        submit_cmd = f"{ssh_command} '{slurm_command}'"
    else:
        submit_cmd = slurm_command

    result = subprocess.run(
        submit_cmd, shell=True, capture_output=True, text=True
    )
    pid = re.search(r"(\d+)", result.stdout).groups()[0]
    return pid


def check_job_status(job_ids, ssh_command="ssh lugus"):
    # Check job status via SSH
    if len(job_ids) > 0:
        pid_str = ""
        for a in job_ids[:-1]:
            pid_str += str(a) + ","
        pid_str += str(job_ids[-1])

        status_cmd = f"{ssh_command} 'sacct -j {pid_str} --format=JobID,State'"
        result = subprocess.run(
            status_cmd, shell=True, capture_output=True, text=True
        )
        status = [s.split()[-1] for s in result.stdout.split("\n")[2:-1]]
        # return status   # Adjust based on actual output when job completes
        return [
            True if i in ["COMPLETED", "FAILED"] else False for i in status
        ]
    else:
        return []

def check_array_status(job_id, ssh_command="ssh lugus"):
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
    

def get_array_slurm_str(
    calcdir,
    python_script_path,
    num_array_tasks=1,
    num_parallel=10,
    jobname="job",
    partition="normal",
    n_cores=8,
    conda_env="jaxnest",
    map_indices=None,
    template_file="lugus.sh.j2",
    slurm_header_add=""
):
    if map_indices is None:
        map_indices = np.arange(num_array_tasks)

    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template(template_file)

    slurm_str = template.render(
        calcdir=calcdir,
        python_script_path=python_script_path,
        num_array_tasks=num_array_tasks,
        num_parallel=num_parallel,
        jobname=jobname,
        partition=partition,
        n_cores=n_cores,
        conda_env=conda_env,
        map_indices=map_indices,
        slurm_header_add=slurm_header_add,
        job_type_template="body_array.sh.j2"
    )

    return slurm_str

def get_slurm_str(
    input_file,
    python_script_path,
    jobname="job",
    partition="normal",
    n_cores=8,
    conda_env="jaxnest",
    template_file="lugus.sh.j2",
    slurm_header_add=""
):
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template(template_file)

    slurm_str = template.render(
        input_file=input_file,
        python_script_path=python_script_path,
        jobname=jobname,
        partition=partition,
        n_cores=n_cores,
        conda_env=conda_env,
        slurm_header_add=slurm_header_add,
        job_type_template="body_single.sh.j2"
    )

    return slurm_str