import pytest

from remote_handler import slurm_utils


@pytest.fixture(
    params=[
        # pytest.param("local", marks=[pytest.mark.local]),
        pytest.param("datalab"),
        pytest.param("lugus"),
    ]
)
def remote_machine(request):
    return request.param


def test_get_slurm_str(remote_machine):
    calcdir = "kek"
    slurm_script = "lel"

    slurm_str = slurm_utils.get_slurm_str(
        calcdir, slurm_script, template_file=f"{remote_machine}.sh.j2"
    )

    print(slurm_str)


def test_get_array_slurm_str(remote_machine):
    calcdir = "kek"
    slurm_script = "lel"

    slurm_str = slurm_utils.get_array_slurm_str(
        calcdir, slurm_script, template_file=f"{remote_machine}.sh.j2"
    )

    print(slurm_str)
