import hashlib
import math
import os
import shutil

import ase.io
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import compare_atoms
from pytest_lazyfixture import lazy_fixture

from remote_handler import atoms_handler, dft_runner


def at_to_hash(at, length=8):
    arrays = [at.positions, at.cell]
    sha = hashlib.sha256()
    for array in arrays:
        sha.update(array.tobytes())

    return sha.hexdigest()[:length]


@pytest.fixture(scope="module")
def local_dir(data_dir):
    return data_dir / "temp/dft_runner/local_dir"


@pytest.fixture
def default_params(filesystem, data_dir, dummy_trajectory):
    """
    Variables:
        asset_dir: Path with static files required for test
        temp_dir: Path with temporary files created during the test
        local_dir: Location where the test is operating on the local machine
        local_results: Path where the directory from the remote machine will be
            copied to in order to investigate the resulting files.
    """

    asset_dir = data_dir / "assets/dft_runner/"
    temp_dir = data_dir / "temp/dft_runner"

    local_dir = temp_dir / "local_dir"
    local_results = temp_dir / "results"

    # These are static files the test needs to access
    # pickle_file = asset_dir / "model_params_plain_ensemble_VELO_8E04F5B1.pkl"
    python_script_path = asset_dir / "dft_singlepoint.py"
    # python_script_path = asset_dir / "optimize_traj.py"

    logfile = temp_dir / "test_slurm.log"

    for path in [local_dir, local_results]:
        if path is not None:
            if os.path.exists(path):
                shutil.rmtree(path)
    os.makedirs(local_dir, exist_ok=False)

    if logfile is not None:
        if os.path.exists(logfile):
            os.remove(logfile)

    tfile0 = local_dir / "traj0.extxyz"
    ase.io.write(tfile0, dummy_trajectory)
    tfile1 = local_dir / "traj1.extxyz"
    ase.io.write(tfile1, dummy_trajectory)
    traj_files = [tfile0, tfile1]

    params = dict(
        traj_files=traj_files,
        # savepath=remote_dir,
        savepath=filesystem.slurm_config["savepath"],
        num_parallel=5,
        python_script_path=python_script_path,
        loop_time=5,
        jobname="dft",
        every=1,
        start=None,
        end=None,
        remote_fs=filesystem,
        logfile=logfile,
        collect_only=False,
        local_savepath=local_results,
        tag="dft",
        stub=False,
        incar_kwargs={
            # "istart": 0,
            # "icharg": 2,
            "xc": "PBE",
            "encut": 300,
            "prec": "Normal",
            "ismear": -1,
            "sigma": 0.025,
            "ibrion": -1,
            "ediff": 1e-5,
            # "isym": 0,
            "symprec": 1e-7,
            "lwave": False,
            "lcharg": False,
            "lreal": False,
            # "kspacing": 2.0
            "kspacing": 2.0,
        },
    )

    return params


@pytest.fixture
def instance(default_params):
    params = default_params
    handler = dft_runner.VASPHandler(**params)
    return handler, params


def test_init(instance):
    handler, params = instance

    assert handler.remote_fs.exists(handler.savepath / "slurm_output")
    assert handler.remote_fs.exists(handler.savepath / "calcdirs")


def test_allocate_data(instance):
    handler, params = instance

    handler.allocate_data()

    handler.collect_results()

    n_shards = math.ceil(handler.num_configs_total / params["every"])
    for idx in range(n_shards):
        assert handler.remote_fs.exists(
            handler.savepath / "calcdirs" / f"{idx}"
        )


def test_submit(instance):
    handler, params = instance

    handler.allocate_data()

    handler.submit()

    handler.collect_results()

    n_shards = math.ceil(handler.num_configs_total / params["every"])
    for idx in range(n_shards):
        assert handler.remote_fs.exists(
            handler.savepath / "calcdirs" / f"{idx}"
        )


def test_collect_and_redistribute(instance):
    handler, params = instance

    # Modify tag such that the unprocessed trajs are collected
    # handler.tag = ""

    handler.allocate_data()

    handler.submit()

    handler.collect_results()

    all_atoms, failed_indices = handler.collect(only_finished=False)
    assert len(all_atoms) == handler.num_configs_total

    all_atoms_sets = handler.regroup(all_atoms)
    # check if regrouped sets of configurations are equal to initially
    # passed ones:
    for idx, tfile in enumerate(handler.traj_files):
        ats = ase.io.read(tfile, index=f"::{params['every']}")
        for at_0, at_1 in zip(ats, all_atoms_sets[idx]):
            assert np.allclose(at_0.positions, at_1.positions)
            assert np.allclose(at_0.cell[...], at_1.cell[...])

    # check if the files with the computed quantities were transferred back
    # to the local path where the original traj files are located
    handler.redistribute(all_atoms_sets)
    for idx, tfile in enumerate(handler.traj_files):
        # path, filename = os.path.split(tfile)
        # opt_tfile = Path(path) / f"{handler.tag}{filename}"
        opt_tfile = atoms_handler.front_tag(tfile, handler.tag)
        assert handler.local_fs.exists(opt_tfile)


def test_full_run(instance):
    handler, params = instance
    handler.allocate_data()
    handler.submit()

    handler.collect_and_harvest()

    handler.collect_results()
