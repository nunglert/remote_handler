"""Integration tests for cmaes_sampling.py script and CMAESSampler interface."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from _foundation import foundation_model_available, foundation_model_path
from _markers import requires_local_slurm
from ase.build import bulk
from ase.io import write

import remote_handler.scripts.input_models as input_models_module
from remote_handler.scripts.input_models import CmaesSamplingInput
from remote_handler.workflows.samplers import BatchCMAESSampler, CMAESSampler


@pytest.fixture(scope="session")
def integration_temp_dir(data_dir):
    """Get base temp directory for cmaes integration tests."""
    temp_dir = data_dir / "temp" / "integration" / "cmaes_sampling"
    temp_dir.mkdir(parents=True, exist_ok=True)
    yield temp_dir


@pytest.fixture
def founder_poscar(integration_temp_dir):
    """Create a 63-atom silicon vacancy structure as founder.

    Builds a 2x2x2 supercell of diamond Si (64 atoms) and removes
    one atom to create a vacancy.
    """
    si_bulk = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)
    del si_bulk[-1]
    founder_path = integration_temp_dir / "POSCAR_si_vacancy"
    write(founder_path, si_bulk, format="vasp")
    return founder_path


@pytest.mark.integration
@pytest.mark.skipif(
    not foundation_model_available(),
    reason="MACE foundation model asset missing from test/data/assets",
)
class TestCmaesSamplingIntegration:
    """Integration tests for full cmaes_sampling.py script execution.

    Uses the MACE model shipped under ``test/data/assets``.
    """

    def test_script_energy_only(self, integration_temp_dir, founder_poscar):
        """Test a short energy-only evolution run."""
        root_dir = integration_temp_dir / "test_energy_only"
        if root_dir.exists():
            shutil.rmtree(root_dir)
        root_dir.mkdir()

        model = foundation_model_path()

        input_file = root_dir / "input.json"
        input_data = {
            "founder": str(founder_poscar),
            "label": "test_energy_only",
            "models": [model],
            "random_seed": 42,
            "step_size": 0.12,
            "generations": 3,
            "save_nth": 1,
            "batch": False,
            "loss_mode": "energy-only",
        }
        input_file.write_text(json.dumps(input_data))

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "remote_handler.scripts.clinamen2.cmaes_sampling",
                str(input_file),
            ],
            capture_output=True,
            text=True,
            cwd=root_dir,
            check=True,
        )
        print(result.stdout)
        print(result.stderr)

        assert result.returncode == 0, (
            f"Script failed with return code {result.returncode}:\n"
            f"{result.stderr}"
        )

        # Evolution output directory should exist
        evo_dir = root_dir / "test_energy_only"
        assert evo_dir.exists(), f"Evolution directory not found: {evo_dir}"

    def test_script_loss_threshold_batch(
        self, integration_temp_dir, founder_poscar
    ):
        """Test a short loss-threshold evolution run with batching."""
        root_dir = integration_temp_dir / "test_loss_threshold"
        if root_dir.exists():
            shutil.rmtree(root_dir)
        root_dir.mkdir()

        model = foundation_model_path()
        assert os.path.exists(model)

        input_file = root_dir / "input.json"
        input_data = {
            "founder": str(founder_poscar),
            "label": "test_loss_threshold",
            "models": [model],
            "random_seed": 42,
            "step_size": 0.12,
            "generations": 3,
            "save_nth": 1,
            "batch": True,
            "loss_mode": "loss-threshold",
            "loss_threshold": -1e6,
        }
        input_file.write_text(json.dumps(input_data))

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "remote_handler.scripts.clinamen2.cmaes_sampling",
                str(input_file),
            ],
            capture_output=True,
            text=True,
            cwd=root_dir,
            check=True,
        )
        print(result.stdout)
        print(result.stderr)

        assert result.returncode == 0, (
            f"Script failed with return code {result.returncode}:\n"
            f"{result.stderr}"
        )

        evo_dir = root_dir / "test_loss_threshold"
        assert evo_dir.exists(), f"Evolution directory not found: {evo_dir}"

    @requires_local_slurm
    def test_sampler_interface(
        self,
        integration_temp_dir,
        founder_poscar,
        local_fs,
        local_cluster_config,
    ):
        """Test the CMAESSampler interface end-to-end.

        Exercises the full CMAESSampler pipeline: update() sets up
        directories, writes input JSON, creates SingleScriptTask and
        allocates data. get_sampled_configs() submits via slurm and
        collects results.
        """
        root_dir = integration_temp_dir / "test_sampler_interface"
        if root_dir.exists():
            shutil.rmtree(root_dir)
        root_dir.mkdir()

        model = foundation_model_path()
        model_files = [Path(model)]

        # Resolve the actual script file path from the installed package
        script_path = (
            Path(input_models_module.__file__).parent
            / "clinamen2"
            / "cmaes_sampling.py"
        )

        # Use a temp dir as "remote" savepath for local execution
        remote_savepath = integration_temp_dir / "remote_cmaes"
        if remote_savepath.exists():
            shutil.rmtree(remote_savepath)
        remote_savepath.mkdir()

        sampler = CMAESSampler(
            remote_fs=local_fs,
            cluster_config=local_cluster_config,
            sampling_script=script_path,
            founder_path=founder_poscar,
            label="test_sampler_iface",
            generations=5,
            step_size=0.12,
            random_seed=42,
            save_nth=1,
            batch=False,
            loss_mode="energy-only",
            stub=False,
            remote_savepath=remote_savepath,
            loop_time=5,
        )

        # Full pipeline: update() + submit + collect
        sampler.update(calcdir=root_dir, sampler_input=model_files)

        # Verify update() created the expected structure
        assert sampler.sample_dir.exists()
        assert sampler.script_input_file.exists()

        # Validate the sampler-generated JSON
        config = CmaesSamplingInput.from_file(sampler.script_input_file)
        config.validate()

        # Run via slurm and collect results
        sampler.get_sampled_configs()

        # Verify evolution output was collected
        evo_dir = sampler.results_dir / "test_sampler_iface"
        assert evo_dir.exists(), f"Evolution directory not found: {evo_dir}"

    @requires_local_slurm
    def test_sample_multiple_interface(
        self,
        integration_temp_dir,
        founder_poscar,
        local_fs,
        local_cluster_config,
    ):
        """Test the CMAESSampler interface end-to-end.

        Exercises the full CMAESSampler pipeline: update() sets up
        directories, writes input JSON, creates SingleScriptTask and
        allocates data. get_sampled_configs() submits via slurm and
        collects results.
        """
        root_dir = integration_temp_dir / "test_multiple_sampler_interface"
        if root_dir.exists():
            shutil.rmtree(root_dir)
        root_dir.mkdir()

        model = foundation_model_path()
        model_files = [Path(model)]

        # Resolve the actual script file path from the installed package
        script_path = (
            Path(input_models_module.__file__).parent
            / "clinamen2"
            / "cmaes_sampling.py"
        )

        # Use a temp dir as "remote" savepath for local execution
        remote_savepath = integration_temp_dir / "remote_cmaes"
        if remote_savepath.exists():
            shutil.rmtree(remote_savepath)
        remote_savepath.mkdir()

        sampler = CMAESSampler(
            remote_fs=local_fs,
            cluster_config=local_cluster_config,
            sampling_script=script_path,
            founder_path=founder_poscar,
            label="test_sampler_iface",
            generations=5,
            step_size=0.12,
            random_seed=42,
            save_nth=1,
            batch=False,
            loss_mode="energy-only",
            stub=False,
            remote_savepath=remote_savepath,
            loop_time=5,
        )

        batch_size = 3
        random_seed = 42
        rng = np.random.default_rng(random_seed)
        child_seeds = rng.integers(0, 2**31, size=batch_size)

        for run_idx, seed in enumerate(child_seeds):
            calcdir = root_dir / f"{run_idx}"

            # Full pipeline: update() + submit + collect
            sampler.update(calcdir=calcdir, sampler_input=model_files)

            # Verify update() created the expected structure
            assert sampler.sample_dir.exists()
            assert sampler.results_dir.exists()
            assert sampler.script_input_file.exists()

            # Validate the sampler-generated JSON
            config = CmaesSamplingInput.from_file(sampler.script_input_file)
            config.validate()

            # Run via slurm and collect results
            sampler.get_sampled_configs()

            # Verify evolution output was collected
            evo_dir = sampler.results_dir / "test_sampler_iface"
            assert (
                evo_dir.exists()
            ), f"Evolution directory not found: {evo_dir}"

    @requires_local_slurm
    def test_batch_sampler_interface(
        self,
        integration_temp_dir,
        founder_poscar,
        local_fs,
        local_cluster_config,
    ):
        """Test the BatchCMAESSampler interface end-to-end.

        Exercises the full pipeline: update() creates N CMA-ES tasks with
        different seeds, get_sampled_configs() submits all evolutions and
        collects results.
        """
        root_dir = integration_temp_dir / "test_batch_sampler"
        if root_dir.exists():
            shutil.rmtree(root_dir)
        root_dir.mkdir()

        model = foundation_model_path()
        model_files = [Path(model)]

        scripts_dir = Path(input_models_module.__file__).parent
        sampling_script = scripts_dir / "clinamen2" / "cmaes_sampling.py"

        remote_savepath = integration_temp_dir / "remote_batch_cmaes"
        if remote_savepath.exists():
            shutil.rmtree(remote_savepath)
        remote_savepath.mkdir()

        batch_size = 2
        sampler = BatchCMAESSampler(
            batch_size=batch_size,
            remote_fs=local_fs,
            cluster_config=local_cluster_config,
            sampling_script=sampling_script,
            founder_path=founder_poscar,
            label="batch_test",
            generations=3,
            step_size=0.12,
            random_seed=42,
            save_nth=1,
            batch=False,
            loss_mode="energy-only",
            stub=False,
            remote_savepath=remote_savepath,
            loop_time=5,
            do_optimization=True,
            dof_mode="defect",
        )

        sampler.update(calcdir=root_dir, sampler_input=model_files)

        # Verify update() created tasks for each run
        assert len(sampler.sampling_tasks) == batch_size

        # Each run should have its own subdirectory and input file
        for run_idx in range(batch_size):
            run_dir = root_dir / f"run_{run_idx}" / "sample"
            assert run_dir.exists(), f"Run directory not found: {run_dir}"
            input_file = run_dir / "input_cmaes.json"
            assert input_file.exists(), f"Input file not found: {input_file}"

            config = CmaesSamplingInput.from_file(input_file)
            config.validate()
            assert config.label == f"batch_test_{run_idx}"

        # Run all evolutions and collect
        result_paths = sampler.get_sampled_configs()
        assert len(result_paths) == batch_size
        for p in result_paths:
            assert p.exists(), f"Evolution output not found: {p}"
