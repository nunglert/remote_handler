"""Integration tests for VASPHandler and VASPEvaluator."""

import math
import os
import shutil
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase import Atoms

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import RemoteFileSystem
from remote_handler.dft_runner import VASPHandler
from remote_handler.scripts.input_models import DftSinglepointInput
from remote_handler.workflows.evaluators import VASPEvaluator

# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture(scope="session")
def integration_temp_dir(data_dir):
    """Get base temp directory for DFT integration tests."""
    temp_dir = data_dir / "temp" / "integration" / "dft"
    temp_dir.mkdir(parents=True, exist_ok=True)
    yield temp_dir


@pytest.fixture(scope="session")
def lugus_cluster_config():
    """Cluster configuration for lugus (DFT cluster)."""
    return ClusterConfig(
        remote_machine="lugus",
        partition="normal",
        conda_env="jaxnest",
        conda_source="/home/nunglert/miniconda3",
        savepath="/home/nunglert/dump/remote_handler",
        template="lugus.sh.j2",
    )


@pytest.fixture(scope="session")
def lugus_fs():
    """Remote filesystem for lugus."""
    return RemoteFileSystem(remote_alias="lugus")


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


@pytest.mark.integration
class TestVASPEvaluatorIntegration:
    """Integration tests for VASPEvaluator with actual SLURM submission.

    Requires SSH access to lugus.
    """

    def test_evaluator_update(
        self,
        integration_temp_dir,
        dummy_trajectory,
        lugus_fs,
        lugus_cluster_config,
        scripts_dir,
    ):
        """Test VASPEvaluator update creates VASPHandler on lugus."""
        root_dir = integration_temp_dir / "test_evaluator_interface"
        if root_dir.exists():
            shutil.rmtree(root_dir)
        root_dir.mkdir()

        evaluator = VASPEvaluator(
            singlepoint_script=scripts_dir / "dft_singlepoint.py",
            remote_fs=lugus_fs,
            cluster_config=lugus_cluster_config,
            num_parallel=5,
            incar_kwargs={
                "xc": "PBE",
                "encut": 300,
                "kspacing": 2.0,
                "ediff": 1e-5,
                "lwave": False,
                "lcharg": False,
            },
            remote_savepath=Path(lugus_cluster_config.savepath),
            loop_time=5,
            stub=False,
        )

        evaluator.update(calcdir=root_dir, evaluator_input=dummy_trajectory)

        # Verify VASPHandler was created with correct config
        assert hasattr(evaluator, "vasp_handler")
        assert evaluator.vasp_handler.num_configs_total == 3

        evaluator.get_evaluated_configs()
