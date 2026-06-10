"""Integration tests for mace_training.py script and MaceModelTrainer."""

import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest
from _foundation import foundation_model_path
from _mace import mace_available, mace_freeze_available, mace_run_train_script
from ase import Atoms

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.exceptions import ScriptInputError
from remote_handler.workflows.trainers import (
    MaceFinetuningTrainer,
    MaceModelTrainer,
)

# Trainer-workflow tests submit through SlurmJobHandler against the local
# machine; skip (rather than error with DependencyError) when SLURM is absent.
requires_local_slurm = pytest.mark.skipif(
    shutil.which("sbatch") is None,
    reason="SLURM (sbatch) not available on this machine",
)


@pytest.fixture(scope="session")
def integration_temp_dir(data_dir):
    """Get base temp directory for integration tests.

    Each test creates its own subdirectory within this base directory.
    Uses session scope so it's only created once for all tests.
    """
    temp_dir = data_dir / "temp" / "integration" / "mace_training"

    # Create base directory if it doesn't exist
    temp_dir.mkdir(parents=True, exist_ok=True)

    yield temp_dir

    # Optionally clean up entire directory after all tests (comment out to keep for inspection)
    # if temp_dir.exists():
    #     shutil.rmtree(temp_dir)


@pytest.fixture
def test_atoms_database():
    """Create test atoms with dummy energies and forces."""
    import numpy as np

    atoms_list = []
    for i in range(20):
        atoms = Atoms(
            "H2",
            positions=[[0, 0, 0], [0, 0, 0.7 + i * 0.01]],
            cell=[5, 5, 5],
            pbc=True,
        )
        # Add dummy DFT-like properties
        atoms.info["energy"] = -1.0 - i * 0.01
        atoms.info["REF_energy"] = -1.0 - i * 0.01
        atoms.arrays["forces"] = np.array([[0, 0, 0.1], [0, 0, -0.1]])
        atoms.arrays["REF_forces"] = np.array([[0, 0, 0.1], [0, 0, -0.1]])
        atoms_list.append(atoms)
    return atoms_list


@pytest.fixture
def mace_training_setup(integration_temp_dir, test_atoms_database, request):
    """Set up directory structure for mace_training.py test."""
    # Use test name for unique directory
    test_name = request.node.name
    root_dir = integration_temp_dir / test_name

    # Clean test directory if it exists from previous run
    if root_dir.exists():
        shutil.rmtree(root_dir)
    root_dir.mkdir()

    # Write database
    db_file = root_dir / "training_data.xyz"
    ase.io.write(db_file, test_atoms_database)

    # Create MACE config
    config_file = root_dir / "mace_config.yaml"
    config_file.write_text(
        """model: MACE
batch_size: 4
max_num_epochs: 2
hidden_irreps: 16x0e
r_max: 4.0
max_ell: 2
device: cpu
default_dtype: float32
"""
    )

    # Create calculation directory
    calc_dir = root_dir / "calculations" / "calc_001"
    calc_dir.mkdir(parents=True)

    return {
        "root_dir": root_dir,
        "calc_dir": calc_dir,
        "db_file": db_file,
        "config_file": config_file,
    }


@pytest.mark.integration
@pytest.mark.skipif(
    not mace_available(),
    reason="MACE (mace-torch) not installed",
)
class TestMaceTrainingIntegration:
    """Integration tests for full mace_training.py script execution."""

    def test_script_with_multiple_databases(
        self, integration_temp_dir, test_atoms_database
    ):
        """Test combining multiple database files."""
        root_dir = integration_temp_dir / "test_multiple_db"
        if root_dir.exists():
            shutil.rmtree(root_dir)
        root_dir.mkdir()

        train_runner = mace_run_train_script()
        assert os.path.exists(train_runner)

        # Create two database files
        db1 = root_dir / "db1.xyz"
        db2 = root_dir / "db2.xyz"
        ase.io.write(db1, test_atoms_database[:10])
        ase.io.write(db2, test_atoms_database[10:])

        # Create config
        config_file = root_dir / "mace_config.yaml"
        config_file.write_text(
            "model: MACE\nbatch_size: 4\nE0s: average\nmax_num_epochs: 5"
        )

        # Create calc directory
        calc_dir = root_dir / "calc"
        calc_dir.mkdir()

        # Create input JSON with multiple databases
        input_json = calc_dir / "input_train.json"
        input_data = {
            "database_files": [
                f"{calc_dir.parent}/db1.xyz",
                f"{calc_dir.parent}/db2.xyz",
            ],
            "train_runner": train_runner,
            "config_file": str(config_file),
            "model_name": "test",
            "seed": 42,
            "validation_fraction": 0.2,
        }
        input_json.write_text(json.dumps(input_data))

        # Run script
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "remote_handler.scripts.mace.mace_training",
                str(calc_dir),
            ],
            capture_output=True,
            text=True,
            cwd=root_dir,
            check=True,
        )
        print(result.stderr)

        # Should combine databases
        assert (calc_dir / "combined_db.extxyz").exists()
        combined = ase.io.read(calc_dir / "combined_db.extxyz", index=":")
        assert len(combined) == 20

    @pytest.mark.skipif(
        not mace_freeze_available(),
        reason="mace-freeze fork not installed "
        "(needs mace.tools.freeze; pip install .[mace_freeze])",
    )
    def test_frozen_script_with_multiple_databases(
        self, integration_temp_dir, test_atoms_database
    ):
        """Test combining multiple database files."""
        root_dir = integration_temp_dir / "test_frozen_multiple_db"
        if root_dir.exists():
            shutil.rmtree(root_dir)
        root_dir.mkdir()

        foundation_model = foundation_model_path()
        assert os.path.exists(foundation_model)
        train_runner = mace_run_train_script()
        assert os.path.exists(train_runner)

        # Create two database files
        db1 = root_dir / "db1.xyz"
        db2 = root_dir / "db2.xyz"
        ase.io.write(db1, test_atoms_database[:10])
        ase.io.write(db2, test_atoms_database[10:])

        # Create config
        config_file = root_dir / "mace_config.yaml"
        config_file.write_text(
            f"model: MACE\nbatch_size: 4\nE0s: average\nmax_num_epochs: 5\nfreeze: 6\nfoundation_model: {foundation_model}"
        )

        # Create calc directory
        calc_dir = root_dir / "calc"
        calc_dir.mkdir()

        # Create input JSON with multiple databases
        input_json = calc_dir / "input_train.json"
        input_data = {
            "database_files": [
                f"{calc_dir.parent}/db1.xyz",
                f"{calc_dir.parent}/db2.xyz",
            ],
            "train_runner": train_runner,
            "config_file": str(config_file),
            "model_name": "test",
            "seed": 42,
            "validation_fraction": 0.2,
        }
        input_json.write_text(json.dumps(input_data))

        # Run script
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "remote_handler.scripts.mace.mace_training",
                str(calc_dir),
            ],
            capture_output=True,
            text=True,
            cwd=root_dir,
            # check=True
        )
        print(result.stderr)

        # Should combine databases
        assert (calc_dir / "combined_db.extxyz").exists()
        combined = ase.io.read(calc_dir / "combined_db.extxyz", index=":")
        assert len(combined) == 20

    @pytest.mark.skipif(
        not mace_available(),
        reason="MACE (mace-torch) not installed",
    )
    def test_finetuning_script(
        self, integration_temp_dir, test_atoms_database
    ):
        """Test mace_finetuning.py script with foundation model."""
        root_dir = integration_temp_dir / "test_finetuning_script"
        if root_dir.exists():
            shutil.rmtree(root_dir)
        root_dir.mkdir()

        foundation_model = foundation_model_path()
        assert os.path.exists(foundation_model)
        train_runner = mace_run_train_script()
        assert os.path.exists(train_runner)

        # Write database
        db_file = root_dir / "training_data.xyz"
        ase.io.write(db_file, test_atoms_database)

        # Create config (no foundation_model here — the script injects it)
        config_file = root_dir / "mace_config.yaml"
        config_file.write_text(
            "model: MACE\n"
            "batch_size: 4\n"
            "E0s: average\n"
            "max_num_epochs: 5\n"
        )

        # Create calc directory
        calc_dir = root_dir / "calc"
        calc_dir.mkdir()

        # Create input JSON with foundation_model field
        input_json = calc_dir / "input_train.json"
        input_data = {
            "database_files": [str(db_file)],
            "train_runner": train_runner,
            "config_file": str(config_file),
            "foundation_model": foundation_model,
            "model_name": "finetuned_test",
            "seed": 42,
            "validation_fraction": 0.2,
        }
        input_json.write_text(json.dumps(input_data))

        # Run finetuning script
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "remote_handler.scripts.mace.mace_finetuning",
                str(calc_dir),
            ],
            capture_output=True,
            text=True,
            cwd=root_dir,
        )
        print(result.stderr)

        assert (
            result.returncode == 0
        ), f"Finetuning script failed:\n{result.stderr}"


@pytest.mark.integration
class TestMaceModelTrainerIntegration:
    """Integration tests for MaceModelTrainer workflow class."""

    @requires_local_slurm
    @pytest.mark.skipif(
        not mace_available(),
        reason="MACE (mace-torch) not installed",
    )
    def test_full_local_training(
        self, integration_temp_dir, test_atoms_database, local_fs
    ):
        """Test MaceModelTrainer with actual local MACE training.

        Needs MACE installed (run_train.py is located from the install) and a
        local SLURM (sbatch) for job submission.
        """
        from remote_handler.scripts import input_models

        root_dir = integration_temp_dir / "trainer_full"
        if root_dir.exists():
            shutil.rmtree(root_dir)
        root_dir.mkdir()

        train_runner = mace_run_train_script()

        # Write database
        db_file = root_dir / "training_data.xyz"
        ase.io.write(db_file, test_atoms_database)

        # Create MACE config template
        template = root_dir / "mace_config.yaml"
        template.write_text(
            "model: MACE\n"
            "batch_size: 4\n"
            "max_num_epochs: 2\n"
            "hidden_irreps: 16x0e\n"
            "r_max: 4.0\n"
            "max_ell: 2\n"
            "device: cpu\n"
            "default_dtype: float32\n"
            "E0s: average\n"
        )

        local_cluster_config = ClusterConfig(
            remote_machine="datalab",
            partition="GPU-a40",
            conda_env="clinamen",
            conda_source="/home/nico.unglert/miniconda3",
            savepath=str(root_dir / "remote_savepath"),
            template="datalab.sh.j2",
        )

        trainer = MaceModelTrainer(
            remote_fs=local_fs,
            cluster_config=local_cluster_config,
            training_script=Path(input_models.__file__).parent
            / "mace/mace_training.py",
            config_file=template,
            remote_mace_train_runner=train_runner,
            num_models=2,
            stub=False,
            loop_time=10,
        )

        calcdir = root_dir / "iteration_0"
        calcdir.mkdir()
        trainer.update(calcdir, db_file)

        model_files = trainer.get_trained_model()

        assert len(model_files) == 2
        for model_file in model_files:
            assert model_file.exists()


@pytest.mark.integration
class TestMaceFinetuningTrainerIntegration:
    """Integration tests for MaceFinetuningTrainer workflow class."""

    @requires_local_slurm
    @pytest.mark.skipif(
        not mace_freeze_available(),
        reason="mace-freeze fork not installed "
        "(needs mace.tools.freeze; pip install .[mace_freeze])",
    )
    def test_full_local_finetuning(
        self, integration_temp_dir, test_atoms_database, local_fs
    ):
        """Test MaceFinetuningTrainer with actual local MACE finetuning.

        Needs the mace-freeze fork (``mace.tools.freeze``) and a local SLURM
        (sbatch); the finetuning runner is located from the install and the
        MACE model comes from the shipped test asset.
        """
        from remote_handler.scripts import input_models

        root_dir = integration_temp_dir / "finetuning_trainer_full"
        if root_dir.exists():
            shutil.rmtree(root_dir)
        root_dir.mkdir()

        train_runner = mace_run_train_script()
        foundation_model = foundation_model_path()
        assert os.path.exists(foundation_model)

        # Write database
        db_file = root_dir / "training_data.xyz"
        ase.io.write(db_file, test_atoms_database)

        # Create MACE config template
        # For finetuning, architecture params (hidden_irreps, r_max, etc.)
        # come from the foundation model. dtype must match the foundation
        # model to avoid tensordot mismatches.
        template = root_dir / "mace_config.yaml"
        template.write_text(
            "model: MACE\n"
            "batch_size: 4\n"
            "max_num_epochs: 2\n"
            "device: cpu\n"
            "E0s: average\n"
            "freeze: 6\n"
        )

        local_cluster_config = ClusterConfig(
            remote_machine="datalab",
            partition="GPU-a40",
            conda_env="clinamen",
            conda_source="/home/nico.unglert/miniconda3",
            savepath=str(root_dir / "remote_savepath"),
            template="datalab.sh.j2",
        )

        trainer = MaceFinetuningTrainer(
            remote_fs=local_fs,
            cluster_config=local_cluster_config,
            training_script=Path(input_models.__file__).parent
            / "mace/mace_finetuning.py",
            config_file=template,
            foundation_model=foundation_model,
            remote_mace_train_runner=train_runner,
            num_models=2,
            stub=False,
            loop_time=1,
        )

        calcdir = root_dir / "iteration_0"
        calcdir.mkdir()
        trainer.update(calcdir, db_file)

        model_files = trainer.get_trained_model()

        assert len(model_files) == 2
        for model_file in model_files:
            assert model_file.exists()
