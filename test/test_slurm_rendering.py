"""Characterization tests for the SLURM jobfile rendering.

These pin the *output artifact* of the SLURM execution backend -- the rendered
sbatch script -- independent of any live scheduler. They are the guardrail for
the executor-decoupling refactor (Phase 1): that change rewires how the
executor is constructed/injected but must not alter the script it produces.

Pure Jinja rendering, no ``sbatch`` needed, so they run in the default tier
(unlike the ``slurm``-gated end-to-end task tests, which skip without sbatch).
"""

from remote_handler.core.slurm_utils import get_array_slurm_str, get_slurm_str

HEADER = "musica.sh.j2"  # a shipped header template (no external deps)


def test_single_jobfile_render_is_stable():
    """A single job renders the conda activation + the one python call."""
    script = get_slurm_str(
        input_file="/remote/input.json",
        python_script_path="/remote/script.py",
        jobname="charjob",
        partition="charpart",
        n_cores=4,
        conda_env="charenv",
        conda_source="/home/u/miniconda3",
        header_template=HEADER,
        job_type_template="body_single.sh.j2",
    )

    assert "#SBATCH --job-name=charjob" in script
    assert "#SBATCH --partition=charpart" in script
    assert "conda activate charenv" in script
    # The defining behaviour of a single job: run the script on the input file.
    assert "python /remote/script.py /remote/input.json" in script


def test_array_jobfile_render_is_stable():
    """An array job renders the per-shard loop over ``calcdir/$run_id``."""
    script = get_array_slurm_str(
        calcdir="/remote/calc",
        python_script_path="/remote/script.py",
        num_array_tasks=3,
        num_parallel=10,
        jobname="charjob",
        partition="charpart",
        n_processors=4,
        conda_env="charenv",
        header_template=HEADER,
        job_type_template="body_array.sh.j2",
    )

    assert "#SBATCH --job-name=charjob" in script
    assert "conda activate charenv" in script
    # map_indices defaults to arange(num_array_tasks).
    assert "MAP=(0 1 2)" in script
    # The defining behaviour of an array shard: run the script on calcdir/$id.
    assert "python /remote/script.py /remote/calc/$run_id" in script
    # Array sizing (--array=1-N%P) is a submit-time sbatch CLI arg, NOT baked
    # into the jobfile body. Guards against that boundary shifting.
    assert "--array" not in script
