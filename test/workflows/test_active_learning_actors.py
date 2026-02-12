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

from remote_handler.workflows import (
    LammpsSampler,
    MaceModelTrainer,
    MACETrainer,
    NeuralilILTrainer,
    NSTrajectorySampler,
    VASPEvaluator,
)
from remote_handler.workflows.active_learning_base import ActiveLearningHandler


@pytest.fixture
def mock_remote_fs():
    """Create a mock remote filesystem handler."""
    mock_fs = MagicMock()
    mock_fs.slurm_config = {"savepath": "/remote/path"}
    return mock_fs


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

    def test_init_vasp_evaluator(self, mock_remote_fs, tmp_path):
        """Test VASPEvaluator initialization."""
        script = tmp_path / "dft_singlepoint.py"
        script.touch()

        evaluator = VASPEvaluator(
            singlepoint_script=script,
            remote_fs=mock_remote_fs,
            num_parallel=5,
            incar_kwargs={"encut": 300, "xc": "pbe"},
            remote_savepath="/remote/calc",
            stub=True,
        )

        assert evaluator.stub is True
        assert evaluator.num_parallel == 5
        assert evaluator.incar_kwargs["encut"] == 300

    def test_vasp_stub_outputs(
        self, mock_remote_fs, temp_calcdir, mock_xyz_file
    ):
        """Test VASPEvaluator creates stub outputs."""
        script = temp_calcdir / "dft_singlepoint.py"
        script.touch()

        evaluator = VASPEvaluator(
            singlepoint_script=script,
            remote_fs=mock_remote_fs,
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

    def test_vasp_create_stub_outputs(self, mock_remote_fs, tmp_path):
        """Test _create_stub_outputs method directly."""
        script = tmp_path / "dft_singlepoint.py"
        script.touch()

        evaluator = VASPEvaluator(
            singlepoint_script=script,
            remote_fs=mock_remote_fs,
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


class TestLammpsSampler:
    """Test LammpsSampler in stub mode."""

    def test_init_lammps_sampler(self, mock_remote_fs, tmp_path):
        """Test LammpsSampler initialization."""
        script = tmp_path / "lammps_sampling.py"
        script.touch()
        template = tmp_path / "lammps.in.template"
        template.touch()
        data_file = tmp_path / "structure.data"
        data_file.touch()

        sampler = LammpsSampler(
            remote_fs=mock_remote_fs,
            sampling_script=script,
            template_path=template,
            data_path=data_file,
            element_mapping={1: "Cu", 2: "Co"},
            num_samples=10,
            uncertainty_cutoff_radius=4.0,
            remote_lammps_executable_path="/remote/lammps/lmp",
            stub=True,
        )

        assert sampler.stub is True
        assert sampler.num_samples == 10
        assert sampler.element_mapping == {1: "Cu", 2: "Co"}

    def test_lammps_stub_outputs(
        self, mock_remote_fs, temp_calcdir, mock_model_files, tmp_path
    ):
        """Test LammpsSampler creates stub outputs."""
        script = tmp_path / "lammps_sampling.py"
        script.touch()
        template = tmp_path / "lammps.in.template"
        template.touch()
        data_file = tmp_path / "structure.data"
        data_file.touch()

        sampler = LammpsSampler(
            remote_fs=mock_remote_fs,
            sampling_script=script,
            template_path=template,
            data_path=data_file,
            element_mapping={1: "Cu", 2: "Co", 3: "H"},
            num_samples=15,
            uncertainty_cutoff_radius=4.0,
            remote_lammps_executable_path="/remote/lammps/lmp",
            stub=True,
        )

        # Update sampler with model files
        sampler.update(temp_calcdir, mock_model_files)

        # Get sampled configs (should create stub outputs)
        output_file = sampler.get_sampled_configs()

        assert output_file.exists()
        assert output_file.name == "samples.xyz"

        # Verify output structure
        sampled_atoms = ase.io.read(output_file, index=":")
        assert len(sampled_atoms) == 15

        # Check that stub outputs have uncertainty information
        for atoms in sampled_atoms:
            assert "max_uncertainty" in atoms.info
            assert "mean_uncertainty" in atoms.info
            assert "step" in atoms.info

    def test_lammps_create_stub_outputs(self, mock_remote_fs, tmp_path):
        """Test _create_stub_outputs method directly."""
        script = tmp_path / "lammps_sampling.py"
        script.touch()
        template = tmp_path / "lammps.in.template"
        template.touch()
        data_file = tmp_path / "structure.data"
        data_file.touch()

        sampler = LammpsSampler(
            remote_fs=mock_remote_fs,
            sampling_script=script,
            template_path=template,
            data_path=data_file,
            element_mapping={1: "Cu", 2: "Co"},
            num_samples=20,
            uncertainty_cutoff_radius=4.0,
            remote_lammps_executable_path="/remote/lammps/lmp",
            stub=True,
        )

        stub_samples = sampler._create_stub_outputs(num_samples=8)

        assert len(stub_samples) == 8
        for atoms in stub_samples:
            assert len(atoms) == 26  # 20 Cu + 4 Co + 2 H
            assert "max_uncertainty" in atoms.info
            assert atoms.info["max_uncertainty"] > 0


class TestNSTrajectorySampler:
    """Test NSTrajectorySampler in stub mode."""

    def test_init_ns_sampler(self, mock_remote_fs, tmp_path):
        """Test NSTrajectorySampler initialization."""
        ns_input = tmp_path / "ns.inp"
        ns_input.touch()

        sampler = NSTrajectorySampler(
            ns_exe="/path/to/ns",
            remote_fs=mock_remote_fs,
            initial_input_file=ns_input,
            n_gpu_parallel=4,
            sample_size=100,
            resample_size=50,
            stub=True,
        )

        assert sampler.stub is True
        assert sampler.sample_size == 100
        assert sampler.resample_size == 50

    def test_ns_stub_outputs(
        self, mock_remote_fs, temp_calcdir, mock_model_files
    ):
        """Test NSTrajectorySampler creates stub outputs."""
        ns_input = temp_calcdir / "ns.inp"
        ns_input.write_text("# Mock NS input")

        sampler = NSTrajectorySampler(
            ns_exe="/path/to/ns",
            remote_fs=mock_remote_fs,
            initial_input_file=ns_input,
            n_gpu_parallel=4,
            sample_size=80,
            resample_size=None,
            stub=True,
        )

        # Update sampler with model file
        sampler.update(temp_calcdir, mock_model_files[0])

        # Get sampled configs (should create stub outputs and write to file)
        sampled_configs_file = sampler.get_sampled_configs()

        assert sampled_configs_file.exists()
        sampled_atoms = ase.io.read(sampled_configs_file, index=":")
        assert len(sampled_atoms) == 80

        # Check NS-specific fields
        for atoms in sampled_atoms:
            assert "run_id" in atoms.info
            assert "iter" in atoms.info
            assert "energy_uncert" in atoms.info
            assert "forces_uncert" in atoms.info

    def test_ns_with_resampling(
        self, mock_remote_fs, temp_calcdir, mock_model_files
    ):
        """Test NSTrajectorySampler with resampling based on uncertainty."""
        ns_input = temp_calcdir / "ns.inp"
        ns_input.write_text("# Mock NS input")

        sampler = NSTrajectorySampler(
            ns_exe="/path/to/ns",
            remote_fs=mock_remote_fs,
            initial_input_file=ns_input,
            n_gpu_parallel=4,
            sample_size=100,
            resample_size=30,
            energy_uncty_weight=0.5,
            force_uncty_weight=1.0,
            stub=True,
        )

        sampler.update(temp_calcdir, mock_model_files[0])
        sampled_configs_file = sampler.get_sampled_configs()

        # After resampling, should have exactly resample_size configs
        assert sampled_configs_file.exists()
        sampled_atoms = ase.io.read(sampled_configs_file, index=":")
        assert len(sampled_atoms) == 30

        # Check that resampling indices were saved
        resample_file = temp_calcdir / "ns_sampler/results/resampled_inds.csv"
        assert resample_file.exists()

    def test_ns_create_stub_outputs(self, mock_remote_fs, tmp_path):
        """Test _create_stub_outputs method directly."""
        ns_input = tmp_path / "ns.inp"
        ns_input.touch()

        sampler = NSTrajectorySampler(
            ns_exe="/path/to/ns",
            remote_fs=mock_remote_fs,
            initial_input_file=ns_input,
            n_gpu_parallel=4,
            sample_size=60,
            stub=True,
        )

        stub_samples = sampler._create_stub_outputs()

        assert len(stub_samples) == 60
        for atoms in stub_samples:
            assert len(atoms) == 20  # 16 Cu + 4 O
            assert "energy_uncert" in atoms.info
            assert "forces_uncert" in atoms.info


class TestMaceModelTrainer:
    """Test MaceModelTrainer in stub mode."""

    def test_init_mace_trainer(self, mock_remote_fs, tmp_path):
        """Test MaceModelTrainer initialization."""
        script = tmp_path / "mace_training.py"
        script.touch()
        template = tmp_path / "mace_config.yaml.template"
        template.touch()

        trainer = MaceModelTrainer(
            remote_fs=mock_remote_fs,
            training_script=script,
            template_path=template,
            remote_mace_train_runner="/remote/mace/run_train.py",
            num_models=5,
            stub=True,
        )

        assert trainer.stub is True
        assert trainer.num_models == 5

    def test_mace_trainer_stub_outputs(
        self, mock_remote_fs, temp_calcdir, mock_xyz_file
    ):
        """Test MaceModelTrainer creates stub model outputs."""
        script = temp_calcdir / "mace_training.py"
        script.touch()
        template = temp_calcdir / "mace_config.yaml.template"
        template.touch()

        trainer = MaceModelTrainer(
            remote_fs=mock_remote_fs,
            training_script=script,
            template_path=template,
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

    def test_mace_trainer_create_stub_outputs(self, mock_remote_fs, tmp_path):
        """Test _create_stub_outputs method directly."""
        script = tmp_path / "mace_training.py"
        script.touch()
        template = tmp_path / "mace_config.yaml.template"
        template.touch()

        trainer = MaceModelTrainer(
            remote_fs=mock_remote_fs,
            training_script=script,
            template_path=template,
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


class TestMACETrainer:
    """Test MACETrainer (JAXNEST version) in stub mode."""

    def test_init_jaxnest_mace_trainer(self, mock_remote_fs, tmp_path):
        """Test MACETrainer initialization."""
        script = tmp_path / "mace_training.py"
        script.touch()
        config = tmp_path / "mace_config.gin"
        config.touch()

        trainer = MACETrainer(
            training_script=script,
            remote_fs=mock_remote_fs,
            config_file=config,
            stub=True,
        )

        assert trainer.stub is True
        assert trainer.training_fraction == 0.9

    def test_jaxnest_mace_trainer_stub_outputs(
        self, mock_remote_fs, temp_calcdir, mock_xyz_file
    ):
        """Test MACETrainer creates stub model output."""
        script = temp_calcdir / "mace_training.py"
        script.touch()
        config = temp_calcdir / "mace_config.gin"
        config.touch()

        trainer = MACETrainer(
            training_script=script,
            remote_fs=mock_remote_fs,
            config_file=config,
            stub=True,
        )

        # Update trainer with database files
        trainer.update(temp_calcdir, [mock_xyz_file])

        # Get trained model (should create stub output)
        model_file = trainer.get_trained_model()

        assert Path(model_file).exists()
        assert Path(model_file).suffix == ".pkl"

        # Verify model content
        import pickle

        with open(model_file, "rb") as f:
            gin_config = pickle.load(f)
            params = pickle.load(f)

            assert "model.max_ell" in gin_config
            assert "final_train_loss" in params

    def test_jaxnest_mace_create_stub_outputs(self, mock_remote_fs, tmp_path):
        """Test _create_stub_outputs method directly."""
        script = tmp_path / "mace_training.py"
        script.touch()
        config = tmp_path / "mace_config.gin"
        config.touch()

        trainer = MACETrainer(
            training_script=script,
            remote_fs=mock_remote_fs,
            config_file=config,
            stub=True,
        )

        # Need to set results_dir for stub outputs
        trainer.results_dir = tmp_path / "results"
        trainer.results_dir.mkdir()

        stub_model = trainer._create_stub_outputs()

        assert stub_model.exists()

        import pickle

        with open(stub_model, "rb") as f:
            gin_str = pickle.load(f)
            params = pickle.load(f)

            assert "model.max_ell" in gin_str
            assert "final_train_loss" in params


class TestNeuralilILTrainer:
    """Tests for NeuralilIL model trainer."""

    def test_neuralil_trainer_initialization(self, mock_remote_fs, tmp_path):
        """Test NeuralilILTrainer can be initialized."""
        training_script = tmp_path / "train_neuralil.py"
        training_script.touch()

        trainer = NeuralilILTrainer(
            training_script=training_script,
            remote_fs=mock_remote_fs,
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

    def test_neuralil_trainer_stub_outputs(self, mock_remote_fs, tmp_path):
        """Test NeuralilILTrainer creates realistic stub outputs."""
        training_script = tmp_path / "train_neuralil.py"
        training_script.touch()

        trainer = NeuralilILTrainer(
            training_script=training_script,
            remote_fs=mock_remote_fs,
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
