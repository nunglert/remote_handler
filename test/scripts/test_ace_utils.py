import numpy as np
import pytest
import yaml
from _capabilities import pandas_available
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from remote_handler.scripts.ace.ace_utils import merge_ace_config_with_paths

requires_pandas = pytest.mark.skipif(
    not pandas_available(), reason="pandas not installed"
)


def _atoms_with_ref(energy=-1.0, forces=None):
    atoms = Atoms("Cu2", positions=[[0, 0, 0], [2.55, 0, 0]])
    if forces is None:
        forces = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    atoms.info["REF_energy"] = energy
    atoms.arrays["REF_forces"] = forces
    return atoms


@requires_pandas
def test_build_ace_dataframe_reads_ref_energy_forces(tmp_path):
    import ase.io

    from remote_handler.scripts.ace.ace_utils import build_ace_dataframe

    atoms = _atoms_with_ref()
    db_path = tmp_path / "db.extxyz"
    ase.io.write(db_path, [atoms])

    out_file = tmp_path / "ace_data.pckl.gzip"
    build_ace_dataframe(db_path, out_file)

    import pandas as pd

    df = pd.read_pickle(out_file, compression="gzip")
    assert len(df) == 1
    assert df.iloc[0]["energy"] == pytest.approx(-1.0)
    assert np.allclose(df.iloc[0]["forces"], atoms.arrays["REF_forces"])


@requires_pandas
def test_build_ace_dataframe_falls_back_to_calculator(tmp_path):
    import ase.io

    from remote_handler.scripts.ace.ace_utils import build_ace_dataframe

    atoms = Atoms("Cu2", positions=[[0, 0, 0], [2.55, 0, 0]])
    forces = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]])
    atoms.calc = SinglePointCalculator(atoms, energy=-2.0, forces=forces)
    db_path = tmp_path / "db.extxyz"
    ase.io.write(db_path, [atoms])

    out_file = tmp_path / "ace_data.pckl.gzip"
    build_ace_dataframe(db_path, out_file)

    import pandas as pd

    df = pd.read_pickle(out_file, compression="gzip")
    assert df.iloc[0]["energy"] == pytest.approx(-2.0)


def test_merge_ace_config_with_paths_injects_data_section(tmp_path):
    user_config = tmp_path / "config.yaml"
    user_config.write_text("potential: {elements: [Cu]}\nfit: {maxiter: 10}\n")

    merged = merge_ace_config_with_paths(
        user_config_file=user_config,
        seed=7,
        data_file=tmp_path / "train.pckl.gzip",
        validation_fraction=0.15,
    )

    assert merged["seed"] == 7
    assert merged["data"]["filename"].endswith("train.pckl.gzip")
    assert merged["data"]["test_size"] == 0.15
    assert merged["potential"] == {"elements": ["Cu"]}
    # A default backend is always injected (see the docstring): without it,
    # pacemaker's own CLI falls into a branch that crashes with
    # AttributeError on `args_parse.query_data` (a latent upstream bug).
    assert merged["backend"] == {
        "evaluator": "pyace",
        "parallel_mode": "process",
    }


def test_merge_ace_config_preserves_user_backend(tmp_path):
    """An explicit backend in the user's config wins over our default."""
    user_config = tmp_path / "config.yaml"
    user_config.write_text(
        yaml.safe_dump(
            {"potential": {}, "fit": {}, "backend": {"evaluator": "tensorpot"}}
        )
    )

    merged = merge_ace_config_with_paths(
        user_config_file=user_config,
        seed=1,
        data_file=tmp_path / "train.pckl.gzip",
    )

    assert merged["backend"] == {"evaluator": "tensorpot"}


def test_merge_ace_config_preserves_user_test_size(tmp_path):
    """An explicit data::test_size in the user's config wins over ours."""
    user_config = tmp_path / "config.yaml"
    user_config.write_text(
        yaml.safe_dump(
            {"potential": {}, "fit": {}, "data": {"test_size": 0.3}}
        )
    )

    merged = merge_ace_config_with_paths(
        user_config_file=user_config,
        seed=1,
        data_file=tmp_path / "train.pckl.gzip",
        validation_fraction=0.1,
    )

    assert merged["data"]["test_size"] == 0.3
