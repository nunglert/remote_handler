import pytest
from remote_handler import slurm_utils


@pytest.mark.parametrize(
    "remote_machine", ["datalab", "lugus"]
)
def test_get_slurm_str(remote_machine):
    calcdir = "kek"
    slurm_script = "lel"

    slurm_str = slurm_utils.get_slurm_str(
        calcdir,
        slurm_script,
        template_file=f"{remote_machine}.sh.j2"
    )

    print(slurm_str)
