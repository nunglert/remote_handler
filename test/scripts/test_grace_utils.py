import ase.io
import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from remote_handler.scripts.grace.grace_utils import (
    finalize_grace_artifact,
    merge_grace_config_with_paths,
    write_ase_standard_labels,
)


def test_write_ase_standard_labels_from_ref_keys(tmp_path):
    atoms = Atoms("Cu2", positions=[[0, 0, 0], [2.55, 0, 0]])
    forces = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    atoms.info["REF_energy"] = -1.0
    atoms.arrays["REF_forces"] = forces

    out = tmp_path / "labelled.extxyz"
    write_ase_standard_labels(out, [atoms])

    (readback,) = ase.io.read(out, index=":")
    assert readback.get_potential_energy() == pytest.approx(-1.0)
    assert np.allclose(readback.get_forces(), forces)


def test_write_ase_standard_labels_falls_back_to_calculator(tmp_path):
    atoms = Atoms("Cu2", positions=[[0, 0, 0], [2.55, 0, 0]])
    forces = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]])
    atoms.calc = SinglePointCalculator(atoms, energy=-2.0, forces=forces)

    out = tmp_path / "labelled.extxyz"
    write_ase_standard_labels(out, [atoms])

    (readback,) = ase.io.read(out, index=":")
    assert readback.get_potential_energy() == pytest.approx(-2.0)


def test_merge_grace_config_with_paths_injects_data_section(tmp_path):
    user_config = tmp_path / "config.yaml"
    user_config.write_text("cutoff: 5.0\npotential: {}\nfit: {}\n")

    merged = merge_grace_config_with_paths(
        user_config_file=user_config,
        seed=3,
        data_file=tmp_path / "train.extxyz",
        validation_fraction=0.2,
    )

    assert merged["seed"] == 3
    assert merged["data"]["filename"].endswith("train.extxyz")
    assert merged["data"]["test_size"] == 0.2
    assert merged["cutoff"] == 5.0


def test_merge_grace_config_preserves_user_test_size(tmp_path):
    user_config = tmp_path / "config.yaml"
    user_config.write_text(
        yaml.safe_dump(
            {"potential": {}, "fit": {}, "data": {"test_size": 0.4}}
        )
    )

    merged = merge_grace_config_with_paths(
        user_config_file=user_config,
        seed=1,
        data_file=tmp_path / "train.extxyz",
        validation_fraction=0.1,
    )

    assert merged["data"]["test_size"] == 0.4


def test_finalize_grace_artifact_moves_seed_dir(tmp_path):
    seed_dir = tmp_path / "seed" / "42"
    seed_dir.mkdir(parents=True)
    (seed_dir / "model.yaml").write_text("cutoff: 5.0\n")

    dest = finalize_grace_artifact(tmp_path, seed=42, model_name="Model")

    assert dest == tmp_path / "models" / "Model"
    assert (dest / "model.yaml").exists()
    assert not seed_dir.exists()


def test_finalize_grace_artifact_missing_dir_raises(tmp_path):
    with pytest.raises(RuntimeError, match="did not produce"):
        finalize_grace_artifact(tmp_path, seed=1, model_name="Model")
