"""Characterization tests for the SLURM jobfile rendering.

These pin the *output artifact* of the SLURM execution backend -- the rendered
sbatch script -- independent of any live scheduler. They are the guardrail for
the executor-decoupling refactor (Phase 1): that change rewires how the
executor is constructed/injected but must not alter the script it produces.

Pure Jinja rendering, no ``sbatch`` needed, so they run in the default tier
(unlike the ``slurm``-gated end-to-end task tests, which skip without sbatch).
"""

import pytest

from remote_handler.core.slurm_utils import get_array_slurm_str, get_slurm_str
from remote_handler.exceptions import TaskError

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
    # The activation is rendered from the configured conda_source (not a
    # hard-coded ~/miniconda3): the resolved activate path must appear.
    assert "source /home/u/miniconda3/bin/activate" in script
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
        conda_source="/home/u/miniconda3",
        header_template=HEADER,
        job_type_template="body_array.sh.j2",
    )

    assert "#SBATCH --job-name=charjob" in script
    assert "source /home/u/miniconda3/bin/activate" in script
    assert "conda activate charenv" in script
    # The shard index comes from the MAP array; default map_indices = 0..N-1.
    assert "MAP=(0 1 2)" in script
    assert "run_id=${MAP[$((run - 1))]}" in script
    # The defining behaviour of an array shard: run the script on calcdir/$id.
    assert "python /remote/script.py /remote/calc/$run_id" in script
    # Array sizing (--array=1-N%P) is a submit-time sbatch CLI arg, NOT baked
    # into the jobfile body. Guards against that boundary shifting.
    assert "--array" not in script


def test_array_jobfile_honours_map_indices():
    """A non-contiguous map_indices maps array tasks to specific calcdirs.

    This is what a subset rerun (e.g. recompute only failed shards 0, 2, 5)
    relies on; the MAP array is what makes SLURM honour it like the direct
    executor does.
    """
    script = get_array_slurm_str(
        calcdir="/remote/calc",
        python_script_path="/remote/script.py",
        num_array_tasks=3,
        map_indices=[0, 2, 5],
        jobname="charjob",
        partition="charpart",
        n_processors=1,
        conda_env="charenv",
        conda_source="/home/u/miniconda3",
        header_template=HEADER,
        job_type_template="body_array.sh.j2",
    )
    # SLURM tasks 1,2,3 -> MAP[0,1,2] = calcdirs 0, 2, 5.
    assert "MAP=(0 2 5)" in script
    assert "run_id=${MAP[$((run - 1))]}" in script


def test_header_template_accepts_explicit_path(tmp_path):
    """``template`` may be a path to a custom header, not just a name.

    The custom header's directory is added to the search path so its body
    ``{% include %}`` still resolves against the shipped ``body_*`` templates.
    """
    custom = tmp_path / "custom.sh.j2"
    custom.write_text(
        "#!/bin/bash\n#SBATCH --job-name={{ jobname }}\n"
        "{{ conda_activation }}\nCUSTOM_HEADER_MARKER\n"
        "{% include job_type_template %}\n"
    )

    script = get_slurm_str(
        input_file="/remote/input.json",
        python_script_path="/remote/script.py",
        jobname="charjob",
        partition="charpart",
        n_cores=4,
        conda_env="charenv",
        conda_source="/home/u/miniconda3",
        header_template=str(custom),
        job_type_template="body_single.sh.j2",
    )

    assert "CUSTOM_HEADER_MARKER" in script
    assert "conda activate charenv" in script
    # The shipped body include still resolves from the package search path.
    assert "python /remote/script.py /remote/input.json" in script


def test_slurm_header_add_is_rendered_into_header():
    """The combined ``slurm_header_add`` lines land in the header verbatim.

    Guards the template boundary for the cluster-wide header: whatever
    ``SlurmHandlerBase`` combines (cluster config + per-call) must appear in
    the rendered script -- e.g. datalab's TMPDIR redirect.
    """
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
        slurm_header_add="export TMPDIR=$HOME/tmp\nmodule load vasp",
    )
    assert "export TMPDIR=$HOME/tmp" in script
    assert "module load vasp" in script


def test_header_template_missing_path_errors():
    """An explicit template path that does not exist fails clearly."""
    with pytest.raises(TaskError, match="not found at the given path"):
        get_slurm_str(
            input_file="i.json",
            python_script_path="s.py",
            jobname="j",
            partition="p",
            n_cores=2,
            conda_env="env",
            conda_source="/home/u/miniconda3",
            header_template="/no/such/dir/custom.sh.j2",
            job_type_template="body_single.sh.j2",
        )


def _single(time_limit, header=HEADER, memory=None):
    return get_slurm_str(
        input_file="i.json",
        python_script_path="s.py",
        jobname="j",
        partition="p",
        n_cores=2,
        conda_env="env",
        conda_source="/home/u/miniconda3",
        header_template=header,
        job_type_template="body_single.sh.j2",
        time_limit=time_limit,
        memory=memory,
    )


def _array(time_limit, header=HEADER, memory=None):
    return get_array_slurm_str(
        calcdir="/remote/calc",
        python_script_path="/remote/script.py",
        num_array_tasks=2,
        conda_env="env",
        conda_source="/home/u/miniconda3",
        header_template=header,
        job_type_template="body_array.sh.j2",
        time_limit=time_limit,
        memory=memory,
    )


def test_time_limit_renders_directive_when_set():
    assert "#SBATCH --time=0-02:00:00" in _single("0-02:00:00")
    assert "#SBATCH --time=0-02:00:00" in _array("0-02:00:00")


def test_no_time_limit_emits_no_directive():
    # The default: no --time line at all, so the job runs to the partition max.
    assert "--time" not in _single(None)
    assert "--time" not in _array(None)


@pytest.mark.parametrize(
    "header", ["dirona.sh.j2", "datalab.sh.j2", "musica.sh.j2"]
)
def test_time_limit_uniform_across_templates(header):
    # No shipped template hardcodes a --time anymore; it is driven solely by
    # the parameter, consistently across all of them.
    assert "--time" not in _array(None, header=header)
    assert "#SBATCH --time=1-00:00:00" in _array("1-00:00:00", header=header)


def test_memory_renders_directive_when_set():
    assert "#SBATCH --mem=16G" in _single(None, memory="16G")
    assert "#SBATCH --mem=16G" in _array(None, memory="16G")


def test_no_memory_emits_no_directive():
    # The default: no --mem line, so the job gets the partition's default.
    assert "--mem" not in _single(None)
    assert "--mem" not in _array(None)


@pytest.mark.parametrize(
    "header", ["dirona.sh.j2", "datalab.sh.j2", "musica.sh.j2"]
)
def test_memory_uniform_across_templates(header):
    # No shipped template hardcodes a --mem; it is driven solely by the
    # parameter, consistently across all of them.
    assert "--mem" not in _array(None, header=header)
    assert "#SBATCH --mem=8000M" in _array(None, header=header, memory="8000M")
