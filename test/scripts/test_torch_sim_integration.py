"""Integration tests for torch_sim_sampling.py script and TorchSimSampler interface."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.constraints import FixAtoms
from ase.io import read, write

import remote_handler.scripts.input_models as input_models_module
from remote_handler.scripts.input_models import TorchSimSamplingInput
from remote_handler.workflows.samplers import TorchSimSampler

# Sampler-interface tests submit through SlurmJobHandler against the local
# machine; skip (rather than error with DependencyError) when SLURM is absent.
requires_local_slurm = pytest.mark.skipif(
    shutil.which("sbatch") is None,
    reason="SLURM (sbatch) not available on this machine",
)


@pytest.fixture(scope="session")
def integration_temp_dir(data_dir):
    """Get base temp directory for torch_sim integration tests."""
    temp_dir = data_dir / "temp" / "integration" / "torch_sim_sampling"
    temp_dir.mkdir(parents=True, exist_ok=True)
    yield temp_dir


@pytest.fixture
def founder_with_frozen_layer(integration_temp_dir):
    """Create a silicon slab structure with frozen bottom layer.

    Builds a 2x2x2 supercell of diamond Si (64 atoms) and freezes
    atoms in the bottom half (z < midpoint) using ASE FixAtoms.
    This is required for NVT interface simulations.
    """
    si_bulk = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)
    positions = si_bulk.get_positions()
    z_mid = positions[:, 2].mean()
    frozen_indices = [i for i, pos in enumerate(positions) if pos[2] < z_mid]
    si_bulk.set_constraint(FixAtoms(indices=frozen_indices))

    founder_path = integration_temp_dir / "POSCAR_si_frozen"
    write(founder_path, si_bulk, format="vasp")
    return founder_path


@pytest.fixture
def founder_bulk(integration_temp_dir):
    """Create a simple bulk silicon structure (no frozen atoms).

    Suitable for NPT ensemble simulations that don't require
    frozen-atom constraints.
    """
    si_bulk = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)
    founder_path = integration_temp_dir / "POSCAR_si_bulk"
    write(founder_path, si_bulk, format="vasp")
    return founder_path


@pytest.mark.integration
@pytest.mark.skipif(
    not (
        os.getenv("FOUNDATION_MODEL")
        and os.path.isfile(os.getenv("FOUNDATION_MODEL"))
    ),
    reason="FOUNDATION_MODEL must point to an existing model file",
)
class TestTorchSimScriptIntegration:
    """Integration tests for torch_sim_sampling.py script execution.

    Requires environment variables:
        FOUNDATION_MODEL: Path to a MACE model file
    """

    def test_script_npt_ensemble(self, integration_temp_dir, founder_bulk):
        """Test NPT parallel tempering via the sampling script."""
        root_dir = integration_temp_dir / "test_script_npt"
        if root_dir.exists():
            shutil.rmtree(root_dir)
        root_dir.mkdir()

        model = os.getenv("FOUNDATION_MODEL")

        input_file = root_dir / "input_torchsim.json"
        input_data = {
            "founder": str(founder_bulk),
            "model_paths": [model],
            "ensemble": "npt",
            "temperatures": [300.0, 600.0],
            "dt": 0.001,
            "n_steps_npt": 0,
            "n_steps_pt": 5,
            "seed": 42,
            "device": "cuda",
            "dtype": "float64",
        }
        input_file.write_text(json.dumps(input_data))

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "remote_handler.scripts.torch_sim.torch_sim_sampling",
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

        # Verify output files
        out_dir = root_dir / "out"
        assert out_dir.exists(), f"Output directory not found: {out_dir}"

        samples_file = out_dir / "samples.extxyz"
        assert samples_file.exists(), f"Samples file not found: {samples_file}"

        # Read and validate sampled structures
        sampled = read(samples_file, index=":")
        assert len(sampled) == 2, f"Expected 2 replicas, got {len(sampled)}"

        for i, atoms in enumerate(sampled):
            assert atoms.info["replica"] == i
            assert atoms.info["ensemble"] == "npt"
            assert len(atoms) == len(read(founder_bulk))

        # Verify PT swap log exists
        assert (root_dir / "pt_swaps.json").exists()

    def test_script_nvt_ensemble(
        self, integration_temp_dir, founder_with_frozen_layer
    ):
        """Test NVT interface parallel tempering via the sampling script."""
        root_dir = integration_temp_dir / "test_script_nvt"
        if root_dir.exists():
            shutil.rmtree(root_dir)
        root_dir.mkdir()

        model = os.getenv("FOUNDATION_MODEL")

        input_file = root_dir / "input_torchsim.json"
        input_data = {
            "founder": str(founder_with_frozen_layer),
            "model_paths": [model],
            "ensemble": "nvt",
            "temperatures": [300.0, 600.0],
            "target_temperature": 300.0,
            "dt": 0.001,
            "n_steps_npt": 0,
            "n_steps_pt": 5,
            "seed": 42,
            "device": "cuda",
            "dtype": "float64",
        }
        input_file.write_text(json.dumps(input_data))

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "remote_handler.scripts.torch_sim.torch_sim_sampling",
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

        out_dir = root_dir / "out"
        samples_file = out_dir / "samples.extxyz"
        assert samples_file.exists(), f"Samples file not found: {samples_file}"

        sampled = read(samples_file, index=":")
        assert len(sampled) == 2, f"Expected 2 replicas, got {len(sampled)}"

        for atoms in sampled:
            assert atoms.info["ensemble"] == "nvt"

    def test_script_npt_with_equilibration(
        self, integration_temp_dir, founder_bulk
    ):
        """Test NPT with initial NPT equilibration steps."""
        root_dir = integration_temp_dir / "test_script_npt_equil"
        if root_dir.exists():
            shutil.rmtree(root_dir)
        root_dir.mkdir()

        model = os.getenv("FOUNDATION_MODEL")

        input_file = root_dir / "input_torchsim.json"
        input_data = {
            "founder": str(founder_bulk),
            "model_paths": [model],
            "ensemble": "npt",
            "temperatures": [300.0, 500.0],
            "dt": 0.001,
            "n_steps_npt": 3,
            "n_steps_pt": 5,
            "seed": 42,
            "device": "cuda",
            "dtype": "float64",
        }
        input_file.write_text(json.dumps(input_data))

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "remote_handler.scripts.torch_sim.torch_sim_sampling",
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

        samples_file = root_dir / "out" / "samples.extxyz"
        assert samples_file.exists()

        sampled = read(samples_file, index=":")
        assert len(sampled) == 2

    def test_script_npt_ensemble_model(
        self, integration_temp_dir, founder_bulk
    ):
        """Test NPT parallel tempering with a MaceEnsembleModel.

        Uses two copies of the same foundation model to form an ensemble.
        With identical members the mean should match single-model output
        and uncertainty should be near zero.  Also checks that uncertainty
        arrays are written to the h5md trajectory files.
        """
        root_dir = integration_temp_dir / "test_script_npt_ensemble_model"
        if root_dir.exists():
            shutil.rmtree(root_dir)
        root_dir.mkdir()

        model = os.getenv("FOUNDATION_MODEL")

        input_file = root_dir / "input_torchsim.json"
        input_data = {
            "founder": str(founder_bulk),
            "model_paths": [model, model],
            "ensemble": "npt",
            "temperatures": [300.0, 600.0],
            "dt": 0.001,
            "n_steps_npt": 0,
            "n_steps_pt": 5,
            "seed": 42,
            "device": "cuda",
            "dtype": "float64",
        }
        input_file.write_text(json.dumps(input_data))

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "remote_handler.scripts.torch_sim.torch_sim_sampling",
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

        # Verify output samples
        samples_file = root_dir / "out" / "samples.extxyz"
        assert samples_file.exists(), f"Samples file not found: {samples_file}"

        sampled = read(samples_file, index=":")
        assert len(sampled) == 2, f"Expected 2 replicas, got {len(sampled)}"

        # Verify h5md trajectory files contain uncertainty arrays
        from torch_sim.trajectory import TorchSimTrajectory

        for i in range(2):
            traj_path = root_dir / f"trajectory_{i}.h5md"
            assert (
                traj_path.exists()
            ), f"Trajectory file not found: {traj_path}"

            traj = TorchSimTrajectory(str(traj_path), mode="r")
            energy_unc = traj.get_array("energy_uncertainty")
            forces_unc = traj.get_array("forces_uncertainty")
            traj.close()

            assert (
                energy_unc is not None
            ), "energy_uncertainty not in trajectory"
            assert (
                forces_unc is not None
            ), "forces_uncertainty not in trajectory"

            # Identical ensemble members → uncertainty ≈ 0
            assert np.allclose(
                energy_unc, 0.0, atol=1e-4
            ), f"Energy uncertainty should be ~0 for identical models, got {energy_unc}"

        # Verify PT swap log exists
        assert (root_dir / "pt_swaps.json").exists()

    def test_script_foundation_model(self, integration_temp_dir, founder_bulk):
        """Test using the MACE-MP foundation model via 'foundation' keyword."""
        root_dir = integration_temp_dir / "test_script_foundation"
        if root_dir.exists():
            shutil.rmtree(root_dir)
        root_dir.mkdir()

        input_file = root_dir / "input_torchsim.json"
        input_data = {
            "founder": str(founder_bulk),
            "model_paths": ["foundation"],
            "ensemble": "npt",
            "temperatures": [300.0, 500.0],
            "dt": 0.001,
            "n_steps_npt": 0,
            "n_steps_pt": 5,
            "seed": 42,
            "device": "cuda",
            "dtype": "float64",
        }
        input_file.write_text(json.dumps(input_data))

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "remote_handler.scripts.torch_sim.torch_sim_sampling",
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

        samples_file = root_dir / "out" / "samples.extxyz"
        assert samples_file.exists()

        sampled = read(samples_file, index=":")
        assert len(sampled) == 2


@pytest.mark.integration
@pytest.mark.skipif(
    not (
        os.getenv("FOUNDATION_MODEL")
        and os.path.isfile(os.getenv("FOUNDATION_MODEL"))
    ),
    reason="FOUNDATION_MODEL must point to an existing model file",
)
class TestTorchSimSamplerIntegration:
    """Integration tests for TorchSimSampler workflow class.

    Requires environment variables:
        FOUNDATION_MODEL: Path to a MACE model file
    """

    @requires_local_slurm
    def test_sampler_npt_interface(
        self,
        integration_temp_dir,
        founder_bulk,
        local_fs,
        local_cluster_config,
    ):
        """Test the TorchSimSampler interface for NPT ensemble end-to-end.

        Exercises the full pipeline: update() sets up directories, writes
        input JSON, creates SingleScriptTask and allocates data.
        get_sampled_configs() submits via slurm and collects results.
        """
        root_dir = integration_temp_dir / "test_sampler_npt"
        if root_dir.exists():
            shutil.rmtree(root_dir)
        root_dir.mkdir()

        model = os.getenv("FOUNDATION_MODEL")
        model_files = [Path(model)]

        script_path = (
            Path(input_models_module.__file__).parent
            / "torch_sim"
            / "torch_sim_sampling.py"
        )

        remote_savepath = integration_temp_dir / "remote_torchsim_npt"
        if remote_savepath.exists():
            shutil.rmtree(remote_savepath)
        remote_savepath.mkdir()

        sampler = TorchSimSampler(
            remote_fs=local_fs,
            cluster_config=local_cluster_config,
            sampling_script=script_path,
            founder_path=founder_bulk,
            ensemble="npt",
            temperatures=[300.0, 600.0],
            model_paths=[Path(model).name],
            dt=0.001,
            n_steps_pt=5,
            seed=42,
            device="cuda",
            dtype="float64",
            stub=False,
            remote_savepath=remote_savepath,
            loop_time=5,
        )

        sampler.update(calcdir=root_dir, sampler_input=model_files)

        # Verify update() created the expected structure
        assert sampler.sample_dir.exists()
        assert sampler.results_dir.exists()
        assert sampler.script_input_file.exists()

        # Validate the generated JSON input
        config = TorchSimSamplingInput.from_file(sampler.script_input_file)
        config.validate()
        assert config.ensemble == "npt"
        assert config.temperatures == [300.0, 600.0]

        # Run via slurm and collect results
        output_path = sampler.get_sampled_configs()

        assert output_path.exists(), f"Samples output not found: {output_path}"

        sampled = read(output_path, index=":")
        assert len(sampled) == 2

    @requires_local_slurm
    def test_sampler_nvt_interface(
        self,
        integration_temp_dir,
        founder_with_frozen_layer,
        local_fs,
        local_cluster_config,
    ):
        """Test the TorchSimSampler interface for NVT ensemble end-to-end."""
        root_dir = integration_temp_dir / "test_sampler_nvt"
        if root_dir.exists():
            shutil.rmtree(root_dir)
        root_dir.mkdir()

        model = os.getenv("FOUNDATION_MODEL")
        model_files = [Path(model)]

        script_path = (
            Path(input_models_module.__file__).parent
            / "torch_sim"
            / "torch_sim_sampling.py"
        )

        remote_savepath = integration_temp_dir / "remote_torchsim_nvt"
        if remote_savepath.exists():
            shutil.rmtree(remote_savepath)
        remote_savepath.mkdir()

        sampler = TorchSimSampler(
            remote_fs=local_fs,
            cluster_config=local_cluster_config,
            sampling_script=script_path,
            founder_path=founder_with_frozen_layer,
            ensemble="nvt",
            temperatures=[300.0, 600.0],
            target_temperature=300.0,
            model_paths=[Path(model).name],
            dt=0.001,
            n_steps_pt=5,
            seed=42,
            device="cuda",
            dtype="float64",
            stub=False,
            remote_savepath=remote_savepath,
            loop_time=5,
        )

        sampler.update(calcdir=root_dir, sampler_input=model_files)

        assert sampler.sample_dir.exists()
        assert sampler.script_input_file.exists()

        config = TorchSimSamplingInput.from_file(sampler.script_input_file)
        config.validate()
        assert config.ensemble == "nvt"
        assert config.target_temperature == 300.0

        output_path = sampler.get_sampled_configs()

        assert output_path.exists(), f"Samples output not found: {output_path}"

        sampled = read(output_path, index=":")
        assert len(sampled) == 2

    def test_sampler_stub_mode(
        self,
        integration_temp_dir,
        founder_bulk,
        local_fs,
        local_cluster_config,
    ):
        """Test TorchSimSampler in stub mode (no GPU required).

        Stub mode creates mock sampling outputs without running the
        actual simulation, useful for testing the workflow pipeline.
        """
        root_dir = integration_temp_dir / "test_sampler_stub"
        if root_dir.exists():
            shutil.rmtree(root_dir)
        root_dir.mkdir()

        temperatures = [300.0, 450.0, 600.0]

        sampler = TorchSimSampler(
            remote_fs=local_fs,
            cluster_config=local_cluster_config,
            sampling_script="dummy_script.py",
            founder_path=founder_bulk,
            ensemble="npt",
            temperatures=temperatures,
            dt=0.001,
            n_steps_pt=10,
            seed=42,
            device="cpu",
            stub=True,
            loop_time=5,
        )

        sampler.update(calcdir=root_dir, sampler_input=[])

        # Verify directory structure created
        assert sampler.sample_dir.exists()
        assert sampler.results_dir.exists()

        # Run in stub mode and collect results
        output_path = sampler.get_sampled_configs()

        assert output_path.exists(), f"Stub output not found: {output_path}"

        sampled = read(output_path, index=":")
        assert len(sampled) == len(temperatures)

        for i, atoms in enumerate(sampled):
            assert atoms.info["replica"] == i
            assert atoms.info["ensemble"] == "npt"
            assert atoms.info["temperature"] == temperatures[i]

    @requires_local_slurm
    def test_sampler_foundation_model(
        self,
        integration_temp_dir,
        founder_bulk,
        local_fs,
        local_cluster_config,
    ):
        """Test TorchSimSampler with foundation model keyword."""
        root_dir = integration_temp_dir / "test_sampler_foundation"
        if root_dir.exists():
            shutil.rmtree(root_dir)
        root_dir.mkdir()

        script_path = (
            Path(input_models_module.__file__).parent
            / "torch_sim"
            / "torch_sim_sampling.py"
        )

        remote_savepath = integration_temp_dir / "remote_torchsim_foundation"
        if remote_savepath.exists():
            shutil.rmtree(remote_savepath)
        remote_savepath.mkdir()

        sampler = TorchSimSampler(
            remote_fs=local_fs,
            cluster_config=local_cluster_config,
            sampling_script=script_path,
            founder_path=founder_bulk,
            ensemble="npt",
            temperatures=[300.0, 500.0],
            model_paths=["foundation"],
            dt=0.001,
            n_steps_pt=5,
            seed=42,
            device="cuda",
            dtype="float64",
            stub=False,
            remote_savepath=remote_savepath,
            loop_time=5,
        )

        # Foundation model: sampler_input still passed but not copied to remote
        sampler.update(calcdir=root_dir, sampler_input=[])

        assert sampler.sample_dir.exists()
        assert sampler.script_input_file.exists()

        config = TorchSimSamplingInput.from_file(sampler.script_input_file)
        config.validate()
        assert config.model_paths == ["foundation"]

        output_path = sampler.get_sampled_configs()

        assert output_path.exists(), f"Samples output not found: {output_path}"

        sampled = read(output_path, index=":")
        assert len(sampled) == 2
