import os
import subprocess
from unittest.mock import patch

import numpy as np
import pytest
import yaml
from _capabilities import pandas_available
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from remote_handler.scripts.ace.ace_utils import (
    _tensorpot_gpu_ld_library_path,
    merge_ace_config_with_paths,
    run_ace_training,
)

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


def test_tensorpot_gpu_ld_library_path_none_without_nvidia_dir(tmp_path):
    with patch(
        "remote_handler.scripts.ace.ace_utils.sysconfig.get_paths",
        return_value={"purelib": str(tmp_path)},
    ):
        assert _tensorpot_gpu_ld_library_path() is None


def test_tensorpot_gpu_ld_library_path_finds_lib_dirs(tmp_path):
    nvidia_dir = tmp_path / "nvidia"
    (nvidia_dir / "cudnn" / "lib").mkdir(parents=True)
    (nvidia_dir / "cublas" / "lib").mkdir(parents=True)
    # Not a "lib" dir -- must be ignored.
    (nvidia_dir / "cudnn" / "include").mkdir(parents=True)

    with patch(
        "remote_handler.scripts.ace.ace_utils.sysconfig.get_paths",
        return_value={"purelib": str(tmp_path)},
    ):
        result = _tensorpot_gpu_ld_library_path()

    assert result is not None
    dirs = result.split(os.pathsep)
    assert len(dirs) == 2
    assert str(nvidia_dir / "cublas" / "lib") in dirs
    assert str(nvidia_dir / "cudnn" / "lib") in dirs


def _run_ace_training_capturing_env(tmp_path, backend, **run_kwargs):
    """Call run_ace_training with subprocess.run mocked; return its env kwarg."""
    return _run_ace_training_mocked(tmp_path, backend, **run_kwargs).kwargs[
        "env"
    ]


def _run_ace_training_mocked(tmp_path, backend, **run_kwargs):
    """Call run_ace_training with subprocess.run mocked; return its call."""
    config = {"potential": {}, "fit": {}}
    if backend is not None:
        config["backend"] = backend
    user_config = tmp_path / "config.yaml"
    user_config.write_text(yaml.safe_dump(config))

    data_file = tmp_path / "train.pckl.gzip"
    data_file.write_bytes(b"")
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()

    with patch(
        "remote_handler.scripts.ace.ace_utils.subprocess.run"
    ) as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"", stderr=b""
        )
        run_ace_training(
            config_file=tmp_path / "merged_config.yaml",
            user_config_file=user_config,
            seed=1,
            data_file=data_file,
            storage_dir=storage_dir,
            output_path=tmp_path / "models" / "test.yaml",
            train_runner="pacemaker",
            **run_kwargs,
        )

    return mock_run.call_args


def test_run_ace_training_injects_ld_library_path_for_tensorpot(tmp_path):
    nvidia_lib = tmp_path / "site-packages" / "nvidia" / "cudnn" / "lib"
    nvidia_lib.mkdir(parents=True)

    with patch(
        "remote_handler.scripts.ace.ace_utils.sysconfig.get_paths",
        return_value={"purelib": str(tmp_path / "site-packages")},
    ):
        env = _run_ace_training_capturing_env(
            tmp_path, backend={"evaluator": "tensorpot"}
        )

    assert str(nvidia_lib) in env["LD_LIBRARY_PATH"]


def test_run_ace_training_leaves_env_alone_for_default_backend(tmp_path):
    # No user-specified backend -> merge_ace_config_with_paths defaults to
    # the CPU "pyace" evaluator, which never needs LD_LIBRARY_PATH touched.
    env = _run_ace_training_capturing_env(tmp_path, backend=None)

    assert env.get("LD_LIBRARY_PATH") == os.environ.get("LD_LIBRARY_PATH")


def test_run_ace_training_passes_initial_potential(tmp_path):
    initial = tmp_path / "previous.yaml"
    initial.write_text("species: []\n")

    call = _run_ace_training_mocked(
        tmp_path, backend=None, initial_potential=initial
    )

    cmd = call.args[0]
    # -p (fit this potential), not -ip (start of a ladder): a full-size
    # -ip potential makes pacemaker skip the fit entirely.
    assert cmd[cmd.index("-p") + 1] == str(initial.resolve())
    assert "-ip" not in cmd


def test_run_ace_training_without_initial_potential_has_no_p(tmp_path):
    call = _run_ace_training_mocked(tmp_path, backend=None)

    assert "-p" not in call.args[0]
    assert "-ip" not in call.args[0]


def _merged_fit_config(tmp_path, fit, **run_kwargs):
    """Run run_ace_training (mocked) with ``fit``; return the written fit."""
    user_config = tmp_path / "config.yaml"
    user_config.write_text(yaml.safe_dump({"potential": {}, "fit": fit}))
    data_file = tmp_path / "train.pckl.gzip"
    data_file.write_bytes(b"")
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    merged = tmp_path / "merged_config.yaml"
    with patch(
        "remote_handler.scripts.ace.ace_utils.subprocess.run"
    ) as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"", stderr=b""
        )
        run_ace_training(
            config_file=merged,
            user_config_file=user_config,
            seed=1,
            data_file=data_file,
            storage_dir=storage_dir,
            output_path=tmp_path / "models" / "test.yaml",
            train_runner="pacemaker",
            **run_kwargs,
        )
    return yaml.safe_load(merged.read_text())["fit"]


LADDER_FIT = {
    "maxiter": 10,
    "ladder_step": [100, 0.2],
    "ladder_type": "power_order",
}


def test_warm_start_drops_ladder_keys(tmp_path):
    initial = tmp_path / "previous.yaml"
    initial.write_text("species: []\n")

    fit = _merged_fit_config(tmp_path, LADDER_FIT, initial_potential=initial)

    assert fit == {"maxiter": 10}


def test_fit_from_scratch_keeps_ladder_keys(tmp_path):
    fit = _merged_fit_config(tmp_path, LADDER_FIT)

    assert fit == LADDER_FIT
