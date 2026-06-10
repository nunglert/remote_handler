"""Tests for MaceEnsembleModel (shared-body MACE ensemble)."""

import os

import pytest

# These tests need the optional torch / torch_sim extras. Skip the whole module
# (rather than erroring at collection) when they are not installed.
torch = pytest.importorskip("torch")
pytest.importorskip("torch_sim")

from _foundation import MODEL_ASSET
from ase.build import bulk
from torch_sim.models.mace import MaceModel

from remote_handler.scripts.torch_sim.mace_ensemble import MaceEnsembleModel


@pytest.fixture(scope="module")
def foundation_model_path():
    """Path to the MACE model shipped under test/data/assets."""
    path = str(MODEL_ASSET)
    if not os.path.isfile(path):
        pytest.skip(f"MACE model not found: {path}")
    return path


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def dtype():
    return torch.float64


@pytest.fixture(scope="module")
def si_state(device, dtype):
    """A batched SimState with 2 Si bulk systems."""
    import torch_sim as ts

    at1 = bulk("Si", "diamond", a=5.43, cubic=True)
    at2 = at1.copy()
    # perturb second system slightly so energies differ from first
    at2.positions += 0.05
    return ts.io.atoms_to_state([at1, at2], device=device, dtype=dtype)


@pytest.mark.integration
class TestMaceEnsembleModel:
    """Tests using the foundation model loaded N times (identical members)."""

    def test_construction(self, foundation_model_path, device, dtype):
        """Ensemble of 2 identical models loads without error."""
        model = MaceEnsembleModel(
            models=[foundation_model_path, foundation_model_path],
            device=device,
            dtype=dtype,
        )
        assert model.n_members == 2
        assert len(model.member_readouts) == 2
        assert len(model.member_atomic_energies) == 2

    def test_output_shapes(
        self, foundation_model_path, device, dtype, si_state
    ):
        """Forward returns correct tensor shapes."""
        model = MaceEnsembleModel(
            models=[foundation_model_path, foundation_model_path],
            device=device,
            dtype=dtype,
            compute_forces=True,
            compute_stress=True,
        )
        out = model(si_state)

        n_systems = int(si_state.n_systems)
        n_atoms = int(si_state.positions.shape[0])

        assert out["energy"].shape == (n_systems,)
        assert out["forces"].shape == (n_atoms, 3)
        assert out["stress"].shape == (n_systems, 3, 3)

    def test_mean_matches_single_model(
        self, foundation_model_path, device, dtype, si_state
    ):
        """Ensemble mean of identical models matches single-model prediction."""
        single = MaceModel(
            model=foundation_model_path,
            device=device,
            dtype=dtype,
            compute_forces=True,
            compute_stress=True,
        )
        ref = single(si_state)

        ensemble = MaceEnsembleModel(
            models=[foundation_model_path, foundation_model_path],
            device=device,
            dtype=dtype,
            compute_forces=True,
            compute_stress=True,
        )
        ens = ensemble(si_state)

        torch.testing.assert_close(
            ens["energy"], ref["energy"], atol=1e-6, rtol=1e-5
        )
        torch.testing.assert_close(
            ens["forces"], ref["forces"], atol=1e-6, rtol=1e-5
        )
        torch.testing.assert_close(
            ens["stress"], ref["stress"], atol=1e-6, rtol=1e-5
        )

    def test_identical_models_zero_uncertainty(
        self, foundation_model_path, device, dtype, si_state
    ):
        """Uncertainty should be ~0 when all members are identical copies."""
        model = MaceEnsembleModel(
            models=[foundation_model_path, foundation_model_path],
            device=device,
            dtype=dtype,
            compute_forces=True,
            compute_stress=True,
        )
        model(si_state)

        assert model.energy_uncertainty is not None
        assert model.forces_uncertainty is not None

        n_systems = int(si_state.n_systems)
        n_atoms = int(si_state.positions.shape[0])
        assert model.energy_uncertainty.shape == (n_systems,)
        assert model.forces_uncertainty.shape == (n_atoms, 3)

        assert torch.allclose(
            model.energy_uncertainty,
            torch.zeros_like(model.energy_uncertainty),
            atol=1e-6,
        )
        assert torch.allclose(
            model.forces_uncertainty,
            torch.zeros_like(model.forces_uncertainty),
            atol=1e-6,
        )

    def test_three_members(
        self, foundation_model_path, device, dtype, si_state
    ):
        """Ensemble with 3 identical members still works correctly."""
        model = MaceEnsembleModel(
            models=[foundation_model_path] * 3,
            device=device,
            dtype=dtype,
        )
        assert model.n_members == 3
        out = model(si_state)
        assert "energy" in out

    def test_forces_only_no_stress(
        self, foundation_model_path, device, dtype, si_state
    ):
        """compute_stress=False skips stress computation."""
        model = MaceEnsembleModel(
            models=[foundation_model_path, foundation_model_path],
            device=device,
            dtype=dtype,
            compute_forces=True,
            compute_stress=False,
        )
        out = model(si_state)
        assert "forces" in out
        assert "stress" not in out


class TestMaceEnsembleValidation:
    """Validation tests that don't require a model file."""

    def test_single_model_raises(self):
        """Need at least 2 models."""
        with pytest.raises(ValueError, match="at least 2"):
            MaceEnsembleModel(models=["dummy.model"])

    def test_empty_models_raises(self):
        """Empty list raises."""
        with pytest.raises(ValueError, match="at least 2"):
            MaceEnsembleModel(models=[])
