import json

import pytest

from remote_handler import slurm_utils


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(
            "remote",
            marks=[pytest.mark.remote],
        ),
        pytest.param(
            "local",
            marks=[pytest.mark.local],
        ),
    ],
)
def config(request, config_dir):
    with open(config_dir / f"{request.param}.json", "r") as f:
        config = json.load(f)
    return config


def test_get_slurm_str(config):
    calcdir = "kek"
    slurm_script = "lel"

    slurm_str = slurm_utils.get_slurm_str(
        calcdir,
        slurm_script,
        header_template=f"{config['remote_machine']}.sh.j2",
    )

    print(slurm_str)


def test_get_array_slurm_str(config):
    calcdir = "kek"
    slurm_script = "lel"

    slurm_str = slurm_utils.get_array_slurm_str(
        calcdir,
        slurm_script,
        header_template=f"{config['remote_machine']}.sh.j2",
    )

    print(slurm_str)
