"""Integration tests for VASPHandler and VASPEvaluator."""

import math
import os
import shutil
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase import Atoms

from remote_handler.scripts.input_models import DftSinglepointInput
from remote_handler.workflows.dft_runner import VASPHandler
from remote_handler.workflows.evaluators import VASPEvaluator

# The DFT (VASP) integration test runs only when the configured remote test
# cluster uses this template -- a proxy for "this is the known lugus setup with
# a working VASP installation". For any other template we can't assume VASP is
# present, so the test skips (via the ``vasp_cluster_config`` fixture) rather
# than submitting a job that would fail.
_VASP_TEMPLATE = "dirona.sh.j2"

# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture(scope="session")
def integration_temp_dir(data_dir):
    """Get base temp directory for DFT integration tests."""
    temp_dir = data_dir / "temp" / "integration" / "dft"
    temp_dir.mkdir(parents=True, exist_ok=True)
    yield temp_dir


@pytest.fixture
def vasp_cluster_config(remote_cluster_config):
    """The configured remote cluster, gated to the VASP setup.

    Couples this test to the actual cluster config (``remote_cluster_config``
    resolves the configured ``test_cluster``): it runs only when that config
    uses the ``dirona.sh.j2`` template, which on lugus implies the known VASP
    installation. Otherwise it skips.
    """
    if remote_cluster_config.template != _VASP_TEMPLATE:
        pytest.skip(
            "DFT integration runs only when the configured remote cluster "
            f"uses the {_VASP_TEMPLATE!r} template (known lugus+VASP setup); "
            f"configured template is {remote_cluster_config.template!r}."
        )
    return remote_cluster_config


@pytest.fixture
def dummy_trajectory(integration_temp_dir):
    """Create a small trajectory of Si structures for DFT testing."""
    np.random.seed(42)
    atoms_list = []
    for i in range(3):
        at = Atoms(
            "Si3",
            positions=np.random.uniform(0, 5, (3, 3)),
            cell=np.eye(3) * 7,
            pbc=True,
        )
        at.info["config_idx"] = i
        atoms_list.append(at)

    traj_path = integration_temp_dir / "test_traj.extxyz"
    ase.io.write(traj_path, atoms_list)
    return traj_path


@pytest.fixture
def scripts_dir():
    """Get the installed package scripts directory."""
    from remote_handler.scripts import input_models as input_models_module

    return Path(input_models_module.__file__).parent


# -------------------------------------------------------------------------
# VASPHandler tests
# -------------------------------------------------------------------------


@pytest.mark.remote
class TestVASPEvaluatorIntegration:
    """Integration tests for VASPEvaluator with actual SLURM submission.

    Requires SSH access to the configured cluster (lugus) and a config that
    uses the dirona.sh.j2 template (see ``vasp_cluster_config``).
    """

    def test_evaluator_update(
        self,
        integration_temp_dir,
        dummy_trajectory,
        remote_fs,
        vasp_cluster_config,
        scripts_dir,
    ):
        """Test VASPEvaluator update creates VASPHandler on lugus."""
        root_dir = integration_temp_dir / "test_evaluator_interface"
        if root_dir.exists():
            shutil.rmtree(root_dir)
        root_dir.mkdir()

        evaluator = VASPEvaluator(
            singlepoint_script=scripts_dir / "dft_singlepoint.py",
            remote_fs=remote_fs,
            cluster_config=vasp_cluster_config,
            num_parallel=5,
            incar_kwargs={
                "xc": "PBE",
                "encut": 300,
                "kspacing": 2.0,
                "ediff": 1e-5,
                "lwave": False,
                "lcharg": False,
            },
            remote_savepath=Path(vasp_cluster_config.savepath),
            loop_time=5,
            stub=False,
        )

        evaluator.update(calcdir=root_dir, evaluator_input=dummy_trajectory)

        # Verify VASPHandler was created with correct config
        assert hasattr(evaluator, "vasp_handler")
        assert evaluator.vasp_handler.num_configs_total == 3

        evaluator.get_evaluated_configs()
