"""Integration tests for mace_training.py script."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import ase.io
import pytest
from ase import Atoms

from remote_handler.exceptions import ScriptInputError


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
    not os.getenv("MACE_TRAIN_RUNNER"),
    reason="MACE_TRAIN_RUNNER environment variable not set",
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

        train_runner = os.getenv("MACE_TRAIN_RUNNER")
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
                "remote_handler.scripts.mace_training",
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
        not os.getenv("FOUNDATION_MODEL"),
        reason="FOUNDATION_MODEL environment variable not set",
    )
    def test_frozen_script_with_multiple_databases(
        self, integration_temp_dir, test_atoms_database
    ):
        """Test combining multiple database files."""
        root_dir = integration_temp_dir / "test_frozen_multiple_db"
        if root_dir.exists():
            shutil.rmtree(root_dir)
        root_dir.mkdir()

        foundation_model = os.getenv("FOUNDATION_MODEL")
        assert os.path.exists(foundation_model)
        train_runner = os.getenv("MACE_TRAIN_RUNNER")
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
                "remote_handler.scripts.mace_training",
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
