"""Tests for the torch MACE -> mace-jax conversion path.

`MaceFinetuningTrainer(convert_to_jax=True)` trains a torch MACE model (with
the freeze fork) and then, in the same job, converts it to a mace-jax ``.pkl``
bundle via ``convert_torch_model_to_jax``. ``get_trained_model`` then hands the
``.pkl`` paths to the next sampler instead of the torch ``.model``.

* The stub-mode test runs by default (unit tier): the trainer imports are
  light and no ML stack is needed.
* The real-conversion test is ``integration`` and skips unless both the torch
  MACE stack and ``mace-jax`` are importable.
"""

import json
import pickle
from pathlib import Path

import pytest
from _capabilities import foundation_model_path, mace_jax_available


def test_finetuning_stub_convert_to_jax(tmp_path, local_fs):
    """Stub mode with convert_to_jax returns loadable ``.pkl`` bundles.

    Regression guard for the trainer wiring: the artifact handed on is the
    mace-jax ``-jax.pkl`` (not the torch ``.model``), one per ensemble member.
    """
    from remote_handler.core.cluster_handler import ClusterConfig
    from remote_handler.workflows.trainers import MaceFinetuningTrainer

    cluster_config = ClusterConfig(
        remote_machine="dummy",
        partition="dummy",
        conda_env="dummy",
        conda_source="dummy",
        savepath=str(tmp_path / "remote_savepath"),
        template="dummy.sh.j2",
    )

    trainer = MaceFinetuningTrainer(
        remote_fs=local_fs,
        cluster_config=cluster_config,
        training_script=tmp_path / "mace_finetuning.py",
        config_file=tmp_path / "mace_config.yaml",
        foundation_model=tmp_path / "foundation.model",
        remote_mace_train_runner="mace_run_train",
        num_models=2,
        stub=True,
        convert_to_jax=True,
    )

    db_file = tmp_path / "training_data.extxyz"
    db_file.write_text("")  # not read in stub mode
    calcdir = tmp_path / "iteration_0"
    calcdir.mkdir()

    trainer.update(calcdir, db_file)
    model_paths = trainer.get_trained_model()

    assert len(model_paths) == 2
    for model_path in model_paths:
        assert model_path.name == "Model-jax.pkl"
        assert model_path.exists()
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        assert "config" in bundle and "params" in bundle


@pytest.mark.integration
@pytest.mark.skipif(
    not mace_jax_available(),
    reason="needs torch MACE + mace-jax (.[mace_freeze,mace_jax])",
)
def test_convert_torch_model_to_jax(tmp_path):
    """Convert the shipped torch MACE model to a real mace-jax ``.pkl``."""
    from flax import serialization

    from remote_handler.scripts.mace.mace_utils import (
        convert_torch_model_to_jax,
    )

    out = convert_torch_model_to_jax(
        Path(foundation_model_path()),
        output_path=tmp_path / "model-jax.pkl",
    )
    assert out.exists()
    with open(out, "rb") as f:
        bundle = pickle.load(f)
    # Same {config, params, head} bundle that mace-jax-predict loads.
    assert "config" in bundle
    assert isinstance(bundle["params"], (bytes, bytearray))
    # Params must round-trip back to a non-empty param tree (guards that
    # state_to_serializable_dict produced something msgpack can restore).
    restored = serialization.msgpack_restore(bundle["params"])
    assert isinstance(restored, dict) and len(restored) > 0


@pytest.mark.integration
@pytest.mark.skipif(
    not mace_jax_available(),
    reason="needs torch MACE + mace-jax (.[mace_freeze,mace_jax])",
)
def test_converted_model_matches_torch():
    """The converted mace-jax model reproduces the torch MACE energies.

    This is the correctness proof for the conversion: convert the shipped
    torch model, then compare per-structure energies from the torch
    ``MACECalculator`` and the mace-jax ``MACEJAXCalculator`` built from the
    converted graph/params. They must agree to within float64 noise.
    """
    import numpy as np
    import torch
    from ase import Atoms
    from mace.calculators import MACECalculator
    from mace.tools.scripts_utils import extract_config_mace_model
    from mace_jax.calculators.mace import MACEJAXCalculator
    from mace_jax.cli.mace_jax_from_torch import convert_model
    from mace_jax.nnx_utils import state_to_pure_dict
    from mace_jax.tools import gin_model

    torch_model = torch.load(
        foundation_model_path(), map_location="cpu", weights_only=False
    )
    torch_model.eval()
    config = extract_config_mace_model(torch_model)
    config["torch_model_class"] = torch_model.__class__.__name__

    graphdef, state, _ = convert_model(torch_model, config)
    params = state_to_pure_dict(state)
    num_species = len(config["atomic_numbers"])

    def model_fn(p, graph, compute_force=False, compute_stress=False, **kw):
        out, _ = graphdef.apply(p)(
            gin_model._graph_to_data(graph, num_species=num_species),
            compute_force=compute_force,
            compute_stress=compute_stress,
        )
        return out

    calc_jax = MACEJAXCalculator(
        model_fn,
        params,
        float(torch_model.r_max),
        # The z_table maps atomic numbers -> model species indices; without it
        # the model sees raw Z and energies are wrong.
        atomic_numbers=list(config["atomic_numbers"]),
    )
    calc_torch = MACECalculator(models=torch_model)

    base = Atoms(
        numbers=[1, 1, 8],
        positions=[[0, 0, 0], [0, 0, 0.96], [0, 0.9, -0.3]],
        cell=np.eye(3) * 9.0,
        pbc=True,
    )
    max_diff = 0.0
    for i in range(4):
        atoms = base.copy()
        atoms.rattle(0.1, seed=i)
        e_torch = calc_torch.get_potential_energy(atoms)
        e_jax = calc_jax.get_potential_energy(atoms)
        max_diff = max(max_diff, abs(e_torch - e_jax))

    # float64 conversion agreement is ~1e-6 eV; 1e-3 is a safe ceiling.
    assert max_diff < 1e-3, f"torch vs jax energy mismatch: {max_diff} eV"


@pytest.mark.integration
@pytest.mark.skipif(
    not mace_jax_available(),
    reason="needs torch MACE + mace-jax (.[mace_freeze,mace_jax])",
)
def test_generate_jax_foundation_model(tmp_path, monkeypatch):
    """generate_jax_foundation_model glues download -> convert.

    The real network download is patched out (it points the downloader at the
    shipped foundation asset), so this tests our glue + the conversion without
    pulling ~100s of MB. The result must be a loadable mace-jax bundle.
    """
    import mace.calculators.foundations_models as fm

    import remote_handler.scripts.mace.mace_utils as mu

    monkeypatch.setattr(
        fm,
        "download_mace_mp_checkpoint",
        lambda model: foundation_model_path(),
    )

    out = tmp_path / "foundation-jax.pkl"
    result = mu.generate_jax_foundation_model(model="medium", output_path=out)

    assert result == out
    assert out.is_file()
    bundle = pickle.loads(out.read_bytes())
    assert "config" in bundle
    assert isinstance(bundle["params"], (bytes, bytearray))


@pytest.mark.integration
@pytest.mark.skipif(
    not mace_jax_available(),
    reason="needs torch MACE + mace-jax (.[mace_freeze,mace_jax])",
)
def test_convert_to_directory_bundle_roundtrips(tmp_path):
    """The directory bundle is readable by mace-jax's canonical loader.

    Proves the torch->bundle path (with the _macejax_patch shim) writes a
    config.json + params.msgpack that ``load_model_bundle`` consumes.
    """
    from mace_jax.tools.bundle import (
        DEFAULT_CONFIG_NAME,
        DEFAULT_PARAMS_NAME,
        load_model_bundle,
    )

    from remote_handler.scripts.mace.mace_utils import (
        convert_torch_model_to_jax_bundle,
    )

    out = convert_torch_model_to_jax_bundle(
        Path(foundation_model_path()), output_dir=tmp_path / "model-jax"
    )
    assert (out / DEFAULT_CONFIG_NAME).is_file()
    assert (out / DEFAULT_PARAMS_NAME).is_file()
    # config.json is human-readable JSON with class names resolved to strings.
    config = json.loads((out / DEFAULT_CONFIG_NAME).read_text())
    assert isinstance(config.get("interaction_cls"), str)
    # mace-jax's own loader reads it back.
    bundle = load_model_bundle(str(out), dtype="float64")
    assert bundle is not None
