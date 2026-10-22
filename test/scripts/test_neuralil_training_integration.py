"""Tests for the NeuralIL training payload (scripts/neuralil/).

Covers three levels, all in the ``integration`` tier because they need the
NeuralIL stack (``pip install .[neuralil]``):

* the vendored ``_support`` helpers that replaced the old ``jaxnest`` imports
  (need only jax + ase);
* the pure helper functions in ``train_model`` (need the ``neuralil`` import to
  load the module);
* the end-to-end ``neuralil_training.py`` entry point, run as a subprocess in
  stub mode (fast, proves the import chain works without jaxnest) and in a
  minimal real-training mode.

All ``remote_handler.scripts.neuralil`` imports are done *inside* the tests so
that collecting this file never fails when the stack is absent -- the skipif
guards then deselect the tests with an informative reason.
"""

import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from _capabilities import jax_available, neuralil_available

NEURALIL_SCRIPT = "remote_handler.scripts.neuralil.neuralil_training"


def _make_database(n_structures=8, n_atoms=4, cell_size=8.0, seed=0):
    """Build a tiny synthetic training set of single-element structures.

    Each frame carries ``REF_energy`` / ``REF_forces`` (the keys the script
    parses as ground truth) and a cubic cell large enough to host the test
    cutoff without triggering a supercell expansion.
    """
    import ase

    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_structures):
        positions = rng.uniform(1.0, cell_size - 1.0, size=(n_atoms, 3))
        atoms = ase.Atoms(
            "H" * n_atoms,
            positions=positions,
            cell=[cell_size, cell_size, cell_size],
            pbc=True,
        )
        atoms.info["REF_energy"] = float(-1.0 - 0.01 * i)
        atoms.arrays["REF_forces"] = rng.normal(0.0, 0.1, size=(n_atoms, 3))
        frames.append(atoms)
    return frames


# --------------------------------------------------------------------------
# Vendored _support helpers (replaced the jaxnest imports). Need only jax.
# --------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.skipif(not jax_available(), reason="jax not installed")
class TestNeuralilSupport:
    def test_get_max_cutoff_cubic(self):
        from remote_handler.scripts.neuralil._support import structure_utils

        cell = np.eye(3) * 5.0
        # Largest sphere fitting a 5 A cubic cell has radius 2.5 A.
        assert np.isclose(
            float(structure_utils.get_max_cutoff(cell, nplib=np)), 2.5
        )

    def test_find_super_grows_small_cell(self):
        from remote_handler.scripts.neuralil._support import structure_utils

        # 3 A cubic cell cannot host a 4 A cutoff; the supercell must grow.
        cell = np.eye(3) * 3.0
        trafo = np.asarray(structure_utils.find_super(cell, 4.0, max_det=27))
        assert trafo.shape == (3,)
        assert np.all(trafo >= 1)
        # The returned supercell must actually host the cutoff.
        grown = cell * trafo[:, None]
        assert float(structure_utils.get_max_cutoff(grown, nplib=np)) > 4.0

    def test_find_super_noop_for_large_cell(self):
        from remote_handler.scripts.neuralil._support import structure_utils

        # A 10 A cell already hosts a 3 A cutoff -> no replication needed.
        cell = np.eye(3) * 10.0
        trafo = np.asarray(structure_utils.find_super(cell, 3.0, max_det=27))
        assert np.all(trafo == 1)

    def test_get_hnf_diagonals_determinant(self):
        from remote_handler.scripts.neuralil._support import structure_utils

        diags = structure_utils.get_HNF_diagonals_nb(12)
        diags = np.asarray(diags)
        assert diags.shape[1] == 3
        # Every diagonal transformation must have determinant 12.
        assert np.all(np.isclose(np.prod(diags, axis=1), 12))

    def test_batch_eval_with_remainder(self):
        import jax.numpy as jnp

        from remote_handler.scripts.neuralil._support import batching

        # 10 items, minibatch of 3 -> 3 groups + remainder of 1.
        data = jnp.arange(10).reshape(10, 1).astype(float)
        out = batching.evaluate_batch_func_pytree_args_remainder(
            3, lambda x: x * 2.0, data
        )
        assert out.shape == (10, 1)
        assert bool(jnp.allclose(out, data * 2.0))

    def test_batch_eval_pytree_argument(self):
        from collections import OrderedDict

        import jax.numpy as jnp

        from remote_handler.scripts.neuralil._support import batching

        pt = OrderedDict(a=jnp.arange(7).astype(float))
        out = batching.evaluate_batch_func_pytree_args_remainder(
            2, lambda d: OrderedDict(a=d["a"] + 1.0), pt
        )
        assert bool(jnp.allclose(out["a"], pt["a"] + 1.0))


# --------------------------------------------------------------------------
# Pure helpers in train_model. Need the neuralil import to load the module.
# --------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.skipif(
    not neuralil_available(), reason="neuralil not installed (.[neuralil])"
)
class TestNeuralilTrainModelHelpers:
    def test_transform_energy_forces_scaling(self):
        from remote_handler.scripts.neuralil import train_model

        energy = -3.0
        forces = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]])
        scaled_e, tiled_f = train_model.transform_energy_forces(
            energy, forces, [2, 1, 1]
        )
        # 2x1x1 supercell -> 2 copies: energy doubles, forces tile to 4 rows.
        assert np.isclose(scaled_e, -6.0)
        assert tiled_f.shape == (4, 3)
        assert np.allclose(tiled_f[:2], forces)
        assert np.allclose(tiled_f[2:], forces)

    def test_get_types_from_symbols(self):
        from remote_handler.scripts.neuralil import train_model

        symbols = [["H", "O", "H"], ["O", "O"]]
        types, n_types, sorted_elements = train_model.get_types_from_symbols(
            symbols
        )
        assert sorted_elements == ["H", "O"]
        assert n_types == 2
        # H -> 0, O -> 1 (alphabetical).
        assert types == [[0, 1, 0], [1, 1]]

    def test_padme_pads_to_uniform_size(self):
        import jax.numpy as jnp

        from remote_handler.scripts.neuralil import train_model

        positions = [np.zeros((2, 3)), np.ones((3, 3))]
        types = [np.array([0, 0]), np.array([1, 1, 1])]
        forces = [np.zeros((2, 3)), np.ones((3, 3))]
        p, t, f = train_model.padme(positions, types, forces)
        # Padded to the largest configuration (3 atoms).
        assert p.shape == (2, 3, 3)
        assert t.shape == (2, 3)
        assert f.shape == (2, 3, 3)
        # Padded type slots use the -1 sentinel.
        assert int(t[0, -1]) == -1

    def test_draw_urandom_int32_in_range(self):
        from remote_handler.scripts.neuralil import train_model

        values = {train_model.draw_urandom_int32() for _ in range(20)}
        assert len(values) > 1  # not constant
        for v in values:
            assert 0 <= int(v) < 2**32

    def test_transform_atoms_uses_vendored_find_super(self):
        import ase

        from remote_handler.scripts.neuralil import train_model

        # 10 A cell hosts a 3 A cutoff -> identity supercell transformation.
        atoms = ase.Atoms(
            "H2",
            positions=[[0, 0, 0], [0, 0, 1.0]],
            cell=[10, 10, 10],
            pbc=True,
        )
        pos, symbs, cell, trafo = train_model.transform_atoms(
            atoms, 3.0, [1, 1, 1]
        )
        assert np.all(np.asarray(trafo) == 1)
        assert len(symbs) == 2
        assert np.asarray(cell).shape == (3, 3)


# --------------------------------------------------------------------------
# End-to-end neuralil_training.py entry point (subprocess).
# --------------------------------------------------------------------------
def _base_input(structure_db, filepath, pickle_file):
    """Minimal-but-complete input dict for neuralil_training.py."""
    return {
        "structure_databases": [str(structure_db)],
        "r_cut": 3.0,
        "training_fraction": 0.7,
        "supercell_diag": [1, 1, 1],
        "filepath": str(filepath),
        "pickle_file_name": str(pickle_file),
        "n_batch": 2,
        "n_max": 4,
        "embed_d": 1,
        # Distinct, decreasing widths -> all ResNetDense (projecting) layers,
        # so the core accepts the descriptor+embedding input dim directly
        # (equal widths -> ResNetIdentity, which needs width == input dim).
        "core_widths": [32, 16],
        "n_ensemble": 2,
        "morse": False,
        "n_eval_batch": 4,
        "n_epochs": 1,
        "energy_weight": 0.5,
        "log_cosh_parameter_energy": 0.2,
        "log_cosh_parameter_forces": 2.0,
        "high_energy_penalty": -5.0,
        "save_reduced": 2,
        "optimizer_type": "onecycle",
        "learning_rate_kwargs": {
            "max_learning_rate": 1e-2,
            "min_learning_rate": 1e-3,
            "fin_learning_rate": 1e-5,
        },
        "seed": 42,
    }


def _run_script(input_json, cwd):
    return subprocess.run(
        [sys.executable, "-m", NEURALIL_SCRIPT, str(input_json)],
        capture_output=True,
        text=True,
        cwd=str(cwd),
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not neuralil_available(), reason="neuralil not installed (.[neuralil])"
)
class TestNeuralilTrainingScript:
    def test_stub_mode_writes_dummy_outputs(self, tmp_path):
        """Stub mode must run end-to-end and write the placeholder files.

        This is the cheap regression guard for the jaxnest removal: it forces
        a real import of the whole payload in a subprocess, so any leftover
        jaxnest dependency or broken vendored import would fail here.
        """
        out_dir = tmp_path / "stub_run"
        out_dir.mkdir()
        inp = _base_input(
            tmp_path / "does_not_need_to_exist.extxyz",
            out_dir,
            out_dir / "model_0.pkl",
        )
        inp["stub"] = True
        input_json = tmp_path / "input_nnff.json"
        input_json.write_text(json.dumps(inp))

        result = _run_script(input_json, cwd=tmp_path)
        assert result.returncode == 0, result.stderr

        # seed 42 -> run_seed_str "2A", instance_code "VELO_2A".
        for name in [
            "error_VELO_2A.txt",
            "train2A.npz",
            "val2A.npz",
            "data_VELO_2A.pickle",
            "model.pkl",
            "full_model.pkl",
            "is_finished.info",
        ]:
            assert (out_dir / name).exists(), f"missing stub output {name}"

    def test_real_training_writes_model(self, tmp_path):
        """A minimal real training run (2 epochs, 2-model ensemble, onecycle).

        Uses the local one-cycle optimizer (not VeLO) so there is no network
        download. Asserts the script completes and writes the reduced model
        pickle the trainer expects (``pickle_file_name``). Exercises the full
        pipeline end-to-end: DataProcessor -> ModelSetup -> Training -> pareto
        reduction -> pickle -> inference via the vendored ``load_neuralil``.
        """
        import ase.io

        out_dir = tmp_path / "real_run"
        out_dir.mkdir()
        db = tmp_path / "training_data.extxyz"
        ase.io.write(db, _make_database())

        pickle_file = out_dir / "model_0.pkl"
        inp = _base_input(db, out_dir, pickle_file)
        inp["stub"] = False
        # Two epochs so the per-epoch one-cycle schedule reset is exercised.
        inp["n_epochs"] = 2
        input_json = tmp_path / "input_nnff.json"
        input_json.write_text(json.dumps(inp))

        result = _run_script(input_json, cwd=tmp_path)
        assert result.returncode == 0, (
            f"training script failed:\nSTDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

        assert (out_dir / "is_finished.info").exists()
        # pickle_file_name set -> reduced model written there (+ full_ prefix).
        assert pickle_file.exists(), "reduced model pickle not written"
        with open(pickle_file, "rb") as f:
            model_info = pickle.load(f)
        assert model_info is not None
