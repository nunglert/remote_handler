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
    LammpsSampler,
    MaceModelTrainer,
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
    temp_dir = test_dir / "data" / "temp" / "workflows"

    # Clean up old runs only once at session start
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    # Create fresh directory
    temp_dir.mkdir(parents=True, exist_ok=True)

    return temp_dir


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


class TestMACE_MD_ALWorkflow:
    """Test integrated workflow with multiple components using ActiveLearningHandler."""

    def test_full_al_cycle_stub(
        self,
        mock_remote_fs,
        mock_cluster_config,
        mock_xyz_file,
        workflow_test_dir,
    ):
        """Test MACE MD AL workflow: evaluation → training → LAMMPS sampling."""

        calchome = workflow_test_dir / "mace_md_al"
        calchome.mkdir(parents=True, exist_ok=True)

        # Create required script files
        dft_script = calchome / "dft_singlepoint.py"
        dft_script.touch()
        train_script = calchome / "mace_training.py"
        train_script.touch()
        template = calchome / "mace_config.yaml.template"
        template.touch()
        lammps_script = calchome / "lammps_sampling.py"
        lammps_script.touch()
        lammps_template = calchome / "lammps.in.template"
        lammps_template.touch()
        lammps_data = calchome / "structure.data"
        lammps_data.touch()

        # Initialize workflow components
        sampler = LammpsSampler(
            remote_fs=mock_remote_fs,
            sampling_script=lammps_script,
            template_path=lammps_template,
            data_path=lammps_data,
            element_mapping={1: "Cu", 2: "Co"},
            num_samples=10,
            uncertainty_cutoff_radius=4.0,
            remote_lammps_executable_path="/remote/lammps/lmp",
            stub=True,
        )

        evaluator = VASPEvaluator(
            singlepoint_script=dft_script,
            remote_fs=mock_remote_fs,
            cluster_config=mock_cluster_config,
            num_parallel=5,
            incar_kwargs={"encut": 300},
            stub=True,
        )

        trainer = MaceModelTrainer(
            remote_fs=mock_remote_fs,
            cluster_config=mock_cluster_config,
            training_script=train_script,
            config_file=template,
            remote_mace_train_runner="/remote/mace/run_train.py",
            num_models=3,
            stub=True,
        )

        # Create ActiveLearningHandler
        # Start with evaluation (we have initial configs), exit after one full cycle
        al_handler = ActiveLearningHandler(
            sampler=sampler,
            evaluator=evaluator,
            trainer=trainer,
            calchome=calchome,
            num_iterations=2,  # Run 2 iterations
            cycle_entry_point="evaluation",  # Start with evaluation of initial configs
            cycle_exit_point="sampler",  # Exit after sampling
            initial_sampler_out=mock_xyz_file,  # Initial configs to evaluate
            combine_evaluator_outputs=True,  # Accumulate training data
        )

        # Run the AL cycle
        al_handler.run()

        # Verify the workflow executed correctly
        assert (calchome / "AL_progress.json").exists()
        assert (calchome / "0").exists()  # Iteration 0 directory
        assert (calchome / "1").exists()  # Iteration 1 directory
        assert (calchome / "evaluator_output").exists()
        assert (calchome / "trainer_output").exists()
        assert (calchome / "sampler_output").exists()

        # Verify evaluator outputs
        eval_files = list((calchome / "evaluator_output").glob("*.extxyz"))
        assert len(eval_files) >= 1
        eval_atoms = ase.io.read(eval_files[0], index=":")
        assert len(eval_atoms) > 0
        assert all("REF_energy" in at.info for at in eval_atoms)

        # Verify trainer outputs
        model_files = list((calchome / "trainer_output").glob("*.pkl"))
        assert len(model_files) >= 1

        # Verify sampler outputs
        sampler_files = list((calchome / "sampler_output").glob("*.extxyz"))
        assert len(sampler_files) >= 1
        sample_atoms = ase.io.read(sampler_files[0], index=":")
        assert len(sample_atoms) == 10
        assert all("max_uncertainty" in at.info for at in sample_atoms)


class TestNestedSamplingALWorkflow:
    """Tests for full nested sampling active learning workflow using ActiveLearningHandler."""

    def test_nested_sampling_al_cycle(
        self, mock_remote_fs, mock_cluster_config, workflow_test_dir
    ):
        """Test NS AL workflow: sampling → evaluation → MACE training."""
        # This tests the workflow from examples/nested_sampling_AL/ns_active_learning.py

        calchome = workflow_test_dir / "ns_al_mace"
        calchome.mkdir(parents=True, exist_ok=True)

        # Create required script files
        ns_exe = "/remote/jaxnest/ns"
        ns_input = calchome / "ns_input.json"
        ns_input.touch()
        dft_script = calchome / "dft_singlepoint.py"
        dft_script.touch()
        training_script = calchome / "train_mace.py"
        training_script.touch()
        config_file = calchome / "mace_config.gin"
        config_file.touch()

        # Create initial model file
        import pickle

        initial_model = calchome / "initial_model.pkl"
        with open(initial_model, "wb") as f:
            pickle.dump({"model_type": "MACE", "iteration": -1}, f)

        # Initialize workflow components
        sampler = NSTrajectorySampler(
            ns_exe=ns_exe,
            remote_fs=mock_remote_fs,
            initial_input_file=ns_input,
            n_gpu_parallel=4,
            sample_size=100,
            resample_size=50,
            stub=True,
        )

        evaluator = VASPEvaluator(
            singlepoint_script=dft_script,
            remote_fs=mock_remote_fs,
            cluster_config=mock_cluster_config,
            num_parallel=5,
            incar_kwargs={"ENCUT": 400, "EDIFF": 1e-5},
            stub=True,
        )

        trainer = MaceModelTrainer(
            training_script=training_script,
            remote_fs=mock_remote_fs,
            cluster_config=mock_cluster_config,
            config_file=config_file,
            remote_mace_train_runner="/usr/bin/mace_run_train",
            stub=True,
        )

        # Create ActiveLearningHandler
        al_handler = ActiveLearningHandler(
            sampler=sampler,
            evaluator=evaluator,
            trainer=trainer,
            calchome=calchome,
            num_iterations=2,
            cycle_entry_point="sampler",  # Start with sampling
            cycle_exit_point="training",  # Complete full cycle
            initial_trainer_out=initial_model,  # Initial model for sampling
            combine_evaluator_outputs=True,  # Accumulate training data
        )

        # Run the AL cycle
        al_handler.run()

        # Verify the workflow executed correctly
        assert (calchome / "AL_progress.json").exists()
        assert (calchome / "0").exists()
        assert (calchome / "1").exists()

        # Verify sampler outputs (NS returns list of Atoms, gets written to file)
        # Since NS sampler returns list of Atoms, ActiveLearningHandler converts to file
        # The actual verification is that the cycle completed successfully
        # and files exist in the expected directories

        # Verify evaluator outputs
        eval_files = list((calchome / "evaluator_output").glob("*.extxyz"))
        assert len(eval_files) >= 1
        eval_atoms = ase.io.read(eval_files[0], index=":")
        assert all("REF_energy" in at.info for at in eval_atoms)

        # Verify trainer outputs
        model_files = list((calchome / "trainer_output").glob("*.pkl"))
        assert len(model_files) >= 1
        assert model_files[0].suffix == ".pkl"

    def test_nested_sampling_al_with_neuralil(
        self, mock_remote_fs, mock_cluster_config, workflow_test_dir
    ):
        """Test NS AL workflow with NeuralilILTrainer."""

        calchome = workflow_test_dir / "ns_al_neuralil"
        calchome.mkdir(parents=True, exist_ok=True)

        # Create required script files
        ns_exe = "/remote/jaxnest/ns"
        ns_input = calchome / "ns_input.json"
        ns_input.touch()
        dft_script = calchome / "dft_singlepoint.py"
        dft_script.touch()
        training_script = calchome / "train_neuralil.py"
        training_script.touch()

        # Create initial model file
        import pickle

        initial_model = calchome / "initial_model.pkl"
        with open(initial_model, "wb") as f:
            pickle.dump({"model_type": "NeuralIL", "iteration": -1}, f)

        # Initialize workflow components
        sampler = NSTrajectorySampler(
            ns_exe=ns_exe,
            remote_fs=mock_remote_fs,
            initial_input_file=ns_input,
            n_gpu_parallel=4,
            sample_size=80,
            stub=True,
        )

        evaluator = VASPEvaluator(
            singlepoint_script=dft_script,
            remote_fs=mock_remote_fs,
            cluster_config=mock_cluster_config,
            num_parallel=5,
            incar_kwargs={"ENCUT": 400},
            stub=True,
        )

        trainer = NeuralilILTrainer(
            training_script=training_script,
            remote_fs=mock_remote_fs,
            cluster_config=mock_cluster_config,
            training_kwargs={"pickle_file_name": "neuralil.pkl"},
            stub=True,
        )

        # Create ActiveLearningHandler
        al_handler = ActiveLearningHandler(
            sampler=sampler,
            evaluator=evaluator,
            trainer=trainer,
            calchome=calchome,
            num_iterations=1,
            cycle_entry_point="sampler",
            cycle_exit_point="training",
            initial_trainer_out=initial_model,
            combine_evaluator_outputs=True,
        )

        # Run the AL cycle
        al_handler.run()

        # Verify the workflow executed correctly
        assert (calchome / "AL_progress.json").exists()
        assert (calchome / "0").exists()

        # Verify NeuralIL model was trained
        model_files = list((calchome / "trainer_output").glob("*.pkl"))
        assert len(model_files) >= 1

        with open(model_files[0], "rb") as f:
            model = pickle.load(f)
        assert model["model_type"] == "NeuralIL"
