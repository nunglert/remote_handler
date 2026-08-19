"""
Tests for Active Learning workflow components.

These tests verify evaluators, samplers, and trainers work correctly
in stub mode without requiring actual remote execution.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock

import ase.io
import numpy as np
import pytest
from ase import Atoms

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.workflows import (
    MaceModelTrainer,
    NeuralilILTrainer,
    VASPEvaluator,
)


@pytest.fixture
def mock_remote_fs():
    """Create a mock remote filesystem handler."""
    mock_fs = MagicMock()
    mock_fs.slurm_config = {"savepath": "/remote/path"}
    return mock_fs


@pytest.fixture
def mock_cluster_config():
    """Create a mock ClusterConfig for stub-mode tests."""
    return ClusterConfig(
        remote_machine="mock",
        partition="default",
        conda_env="test",
        conda_source="/opt/conda",
        savepath="/tmp/remote_handler",
        template="mock.sh.j2",
    )


@pytest.fixture(scope="session")
def workflow_test_dir():
    """
    Create a persistent test directory for workflow outputs.

    Creates test/data/temp for inspecting workflow folder structure.
    Cleans up the directory at the start of the test session.
    All test outputs persist after tests complete for inspection.
    """
    import shutil

    # Get the test directory (where this test file is located)
    test_dir = Path(__file__).parent.parent  # Go up to test/ directory
    temp_dir = test_dir / "data" / "temp"

    # Clean up old runs only once at session start
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    # Create fresh directory
    temp_dir.mkdir(parents=True, exist_ok=True)

    return temp_dir


@pytest.fixture
def temp_calcdir(tmp_path):
    """Create a temporary calculation directory."""
    calcdir = tmp_path / "iter_0"
    calcdir.mkdir()
    return calcdir


@pytest.fixture
def mock_xyz_file(tmp_path):
    """Create a mock XYZ file with test structures."""
    xyz_file = tmp_path / "test_configs.xyz"

    # Create simple test structures
    atoms_list = []
    for i in range(3):
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) + i * 0.1
        atoms = Atoms(
            symbols="H3", positions=positions, cell=[10, 10, 10], pbc=True
        )
        atoms_list.append(atoms)

    ase.io.write(xyz_file, atoms_list)
    return xyz_file


@pytest.fixture
def mock_model_files(tmp_path):
    """Create mock model files."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    model_files = []
    for i in range(3):
        model_file = model_dir / f"model_{i}.pkl"
        # Write simple mock content
        import pickle

        with open(model_file, "wb") as f:
            pickle.dump({"model_id": i, "type": "MACE"}, f)
        model_files.append(model_file)

    return model_files


class TestVASPEvaluator:
    """Test VASPEvaluator in stub mode."""

    def test_init_vasp_evaluator(
        self, mock_remote_fs, mock_cluster_config, tmp_path
    ):
        """Test VASPEvaluator initialization."""
        script = tmp_path / "dft_singlepoint.py"
        script.touch()

        evaluator = VASPEvaluator(
            singlepoint_script=script,
            remote_fs=mock_remote_fs,
            cluster_config=mock_cluster_config,
            num_parallel=5,
            incar_kwargs={"encut": 300, "xc": "pbe"},
            remote_savepath="/remote/calc",
            stub=True,
        )

        assert evaluator.stub is True
        assert evaluator.num_parallel == 5
        assert evaluator.incar_kwargs["encut"] == 300

    def test_vasp_stub_outputs(
        self, mock_remote_fs, mock_cluster_config, temp_calcdir, mock_xyz_file
    ):
        """Test VASPEvaluator creates stub outputs."""
        script = temp_calcdir / "dft_singlepoint.py"
        script.touch()

        evaluator = VASPEvaluator(
            singlepoint_script=script,
            remote_fs=mock_remote_fs,
            cluster_config=mock_cluster_config,
            num_parallel=5,
            incar_kwargs={"encut": 300},
            stub=True,
        )

        # Update evaluator with input file
        evaluator.update(temp_calcdir, mock_xyz_file)

        # Get evaluated configs (should create stub outputs)
        output_file = evaluator.get_evaluated_configs()

        assert output_file.exists()
        assert output_file.name == "evaluated_configs.extxyz"

        # Verify output structure
        evaluated_atoms = ase.io.read(output_file, index=":")
        assert len(evaluated_atoms) > 0

        # Check that stub outputs have required fields
        for atoms in evaluated_atoms:
            assert "REF_energy" in atoms.info
            assert "REF_forces" in atoms.arrays
            assert atoms.calc is None

    def test_vasp_create_stub_outputs(
        self, mock_remote_fs, mock_cluster_config, tmp_path
    ):
        """Test _create_stub_outputs method directly."""
        script = tmp_path / "dft_singlepoint.py"
        script.touch()

        evaluator = VASPEvaluator(
            singlepoint_script=script,
            remote_fs=mock_remote_fs,
            cluster_config=mock_cluster_config,
            num_parallel=5,
            incar_kwargs={"encut": 300},
            stub=True,
        )

        stub_atoms = evaluator._create_stub_outputs(num_configs=10)

        assert len(stub_atoms) == 10
        for atoms in stub_atoms:
            assert len(atoms) == 3  # Water molecule (O + 2H)
            assert "REF_energy" in atoms.info
            assert "REF_forces" in atoms.arrays
            assert atoms.arrays["REF_forces"].shape == (3, 3)


class TestMaceModelTrainer:
    """Test MaceModelTrainer in stub mode."""

    def test_init_mace_trainer(
        self, mock_remote_fs, mock_cluster_config, tmp_path
    ):
        """Test MaceModelTrainer initialization."""
        script = tmp_path / "mace_training.py"
        script.touch()
        template = tmp_path / "mace_config.yaml.template"
        template.touch()

        trainer = MaceModelTrainer(
            remote_fs=mock_remote_fs,
            cluster_config=mock_cluster_config,
            training_script=script,
            config_file=template,
            remote_mace_train_runner="/remote/mace/run_train.py",
            num_models=5,
            stub=True,
        )

        assert trainer.stub is True
        assert trainer.num_models == 5

    def test_mace_trainer_stub_outputs(
        self, mock_remote_fs, mock_cluster_config, temp_calcdir, mock_xyz_file
    ):
        """Test MaceModelTrainer creates stub model outputs."""
        script = temp_calcdir / "mace_training.py"
        script.touch()
        template = temp_calcdir / "mace_config.yaml.template"
        template.touch()

        trainer = MaceModelTrainer(
            remote_fs=mock_remote_fs,
            cluster_config=mock_cluster_config,
            training_script=script,
            config_file=template,
            remote_mace_train_runner="/remote/mace/run_train.py",
            num_models=3,
            stub=True,
        )

        # Update trainer with database file
        trainer.update(temp_calcdir, mock_xyz_file)

        # Get trained models (should create stub outputs)
        model_files = trainer.get_trained_model()

        assert len(model_files) == 3
        for model_file in model_files:
            assert model_file.exists()
            assert model_file.suffix == ".model"

            # Verify model content
            import pickle

            with open(model_file, "rb") as f:
                model_data = pickle.load(f)
                assert "model_type" in model_data
                assert model_data["model_type"] == "MACE"

    def test_mace_trainer_create_stub_outputs(
        self, mock_remote_fs, mock_cluster_config, tmp_path
    ):
        """Test _create_stub_outputs method directly."""
        script = tmp_path / "mace_training.py"
        script.touch()
        template = tmp_path / "mace_config.yaml.template"
        template.touch()

        trainer = MaceModelTrainer(
            remote_fs=mock_remote_fs,
            cluster_config=mock_cluster_config,
            training_script=script,
            config_file=template,
            remote_mace_train_runner="/remote/mace/run_train.py",
            num_models=4,
            stub=True,
        )

        # Need to set results_dir for stub outputs
        trainer.results_dir = tmp_path / "results"
        trainer.results_dir.mkdir()

        stub_models = trainer._create_stub_outputs()

        assert len(stub_models) == 4
        for model_path in stub_models:
            assert model_path.exists()

            import pickle

            with open(model_path, "rb") as f:
                model = pickle.load(f)
                assert "training_loss" in model
                assert "validation_loss" in model


class TestNeuralilILTrainer:
    """Tests for NeuralilIL model trainer."""

    def test_neuralil_trainer_initialization(
        self, mock_remote_fs, mock_cluster_config, tmp_path
    ):
        """Test NeuralilILTrainer can be initialized."""
        training_script = tmp_path / "train_neuralil.py"
        training_script.touch()

        trainer = NeuralilILTrainer(
            training_script=training_script,
            remote_fs=mock_remote_fs,
            cluster_config=mock_cluster_config,
            remote_savepath="/remote/train",
            training_kwargs={
                "pickle_file_name": "neuralil_model.pkl",
                "n_descriptors": 128,
            },
            stub=True,
            loop_time=30,
        )

        assert trainer.training_script == training_script
        assert trainer.stub is True
        assert trainer.training_kwargs["stub"] is True
        assert (
            trainer.training_kwargs["pickle_file_name"] == "neuralil_model.pkl"
        )

    def test_neuralil_trainer_stub_outputs(
        self, mock_remote_fs, mock_cluster_config, tmp_path
    ):
        """Test NeuralilILTrainer creates realistic stub outputs."""
        training_script = tmp_path / "train_neuralil.py"
        training_script.touch()

        trainer = NeuralilILTrainer(
            training_script=training_script,
            remote_fs=mock_remote_fs,
            cluster_config=mock_cluster_config,
            training_kwargs={"pickle_file_name": "neuralil_model.pkl"},
            stub=True,
        )

        # Create mock database file
        temp_calcdir = tmp_path / "iteration_0"
        temp_calcdir.mkdir()
        db_file = temp_calcdir / "training.db"
        db_file.touch()

        trainer.update(temp_calcdir, [db_file])
        model_file = trainer.get_trained_model()

        # Verify model file exists and has correct structure
        assert model_file.exists()
        assert model_file.name == "neuralil_model.pkl"

        # Verify mock model content
        import pickle

        with open(model_file, "rb") as f:
            model = pickle.load(f)

        assert model["model_type"] == "NeuralIL"
        assert "final_train_loss" in model
        assert "final_val_loss" in model
        assert model["final_train_loss"] > 0
