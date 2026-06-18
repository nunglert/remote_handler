import pytest

from remote_handler.core import slurm_utils


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
