import hashlib
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import compare_atoms
from pytest_lazyfixture import lazy_fixture

from remote_handler import optimizer


def at_to_hash(at, length=8):
    arrays = [at.positions, at.cell]
    sha = hashlib.sha256()
    for array in arrays:
        sha.update(array.tobytes())

    return sha.hexdigest()[:length]


@pytest.fixture(scope="module")
def local_dir(data_dir):
    return data_dir / "temp/optimizer/local_dir"


@pytest.fixture()
def dummy_trajectory():
    np.random.seed(0)
    n_configs = 20
    at0 = Atoms("Si3", cell=np.eye(3) * 7, pbc=True)
    ats = []
    for idx in range(n_configs):
        at = at0.copy()
        at.set_positions(np.random.random(size=(3, 3)) * 7)
        at.set_cell(at.cell[...] * np.random.random(size=(3, 3)))
        at.info["shard_idx"] = idx
        ats.append(at)
    return ats


@pytest.fixture
def traj_files(local_dir, dummy_trajectory):
    tfile0 = local_dir / "traj0.extxyz"
    ase.io.write(tfile0, dummy_trajectory)
    tfile1 = local_dir / "traj1.extxyz"
    ase.io.write(tfile1, dummy_trajectory)

    return [tfile0, tfile1]


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

    asset_dir = data_dir / "assets/optimizer/"
    temp_dir = data_dir / "temp/optimizer"

    local_dir = temp_dir / "local_dir"
    local_results = temp_dir / "results"

    # These are static files the test needs to access
    pickle_file = asset_dir / "model_params_plain_ensemble_VELO_8E04F5B1.pkl"
    python_script_path = asset_dir / "simple_at_processor.py"
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
        shard_size=10,
        # savepath=remote_dir,
        savepath=filesystem.slurm_config["savepath"],
        num_parallel=5,
        stub=False,
        python_script_path=python_script_path,
        pickle_file=pickle_file,
        supercell_trafo=[3, 3, 3],
        max_neighbors_list=[20, 30, 40, 50, 60],
        max_neighbors_offset=3,
        fmax=0.1,
        steps=200,
        rmax=5.0,
        nbins=200,
        loop_time=5,
        jobname="opt",
        every=2,
        start=None,
        end=None,
        do_optimization=False,
        do_steinhardt=True,
        # conda_env="cppjax",
        remote_fs=filesystem,
        logfile=logfile,
        # partition="normal",
        collect_only=False,
        optimization_backend="NNFF",
        steinhardt_backend="pyscal",
        steinhardt_l_values=[4, 6],
        local_savepath=local_results,
        tag="opt",
    )

    return params


@pytest.fixture
def instance(default_params):
    params = default_params
    handler = optimizer.RemoteOptimizationHandler(**params)
    return handler, params


def test_init(instance):
    handler, params = instance

    assert handler.remote_fs.exists(handler.savepath / "slurm_output")
    assert handler.remote_fs.exists(handler.savepath / "calcdirs")


def test_allocate_data(instance):
    handler, params = instance

    handler.allocate_data()

    handler.collect_results()

    n_shards = math.ceil(
        handler.num_configs_total / params["every"] / params["shard_size"]
    )
    for idx in range(n_shards):
        assert handler.remote_fs.exists(
            handler.savepath / "calcdirs" / f"{idx}"
        )


def test_submit(instance):
    handler, params = instance

    handler.allocate_data()

    handler.submit()

    handler.collect_results()

    n_shards = math.ceil(
        handler.num_configs_total / params["every"] / params["shard_size"]
    )
    for idx in range(n_shards):
        assert handler.remote_fs.exists(
            handler.savepath / "calcdirs" / f"{idx}"
        )


def test_collect_and_redistribute(instance):
    handler, params = instance

    # Modify tag such that the unprocessed trajs are collected
    handler.tag = ""

    handler.allocate_data()

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
        opt_tfile = optimizer.front_tag(tfile, handler.tag)
        assert handler.local_fs.exists(opt_tfile)


def test_full_run(instance):
    handler, params = instance
    handler.allocate_data()
    handler.submit()

    handler.collect_and_harvest()

    handler.collect_results()


def test_failure_incomplete(instance):
    """Tests if an error is raised if one of the processed trajectories does
    not have the expected length. Can happen e.g. when the python kernel
    crashes."""
    handler, params = instance
    handler.allocate_data()
    handler.submit()

    # emulate incomplete processed trajectory
    traj_file = Path(handler.savepath) / "calcdirs/0/opt.extxyz"
    optimized_file = optimizer.front_tag(traj_file, handler.tag)
    ats = handler.remote_fs.read_ase(optimized_file, index="1:")
    handler.remote_fs.write_ase(optimized_file, ats)

    with pytest.raises(Exception):
        handler.collect_and_harvest()

    handler.collect_results()


def test_failure_is_finished(instance):
    """Tests if an error is raised if the computations for one config crashed
    leading to an is_finished=False entry in the atoms info dict.
    """
    handler, params = instance
    handler.allocate_data()
    handler.submit()

    # emulate failure for one config in trajectory via is_finished=False
    traj_file = Path(handler.savepath) / "calcdirs/0/opt.extxyz"
    optimized_file = optimizer.front_tag(traj_file, handler.tag)
    ats = handler.remote_fs.read_ase(optimized_file, index=":")
    ats[4].info["is_finished"] = False
    handler.remote_fs.write_ase(optimized_file, ats)

    with pytest.raises(Exception):
        handler.collect_and_harvest()

    handler.collect_results()


def test_recompute_failed(instance):
    handler, params = instance
    handler.allocate_data()
    handler.submit()

    # emulate failure for one config in trajectory via is_finished=False
    handler.collect_only = True
    traj_file = Path(handler.savepath) / "calcdirs/0/opt.extxyz"
    optimized_file = optimizer.front_tag(traj_file, handler.tag)
    ats = handler.remote_fs.read_ase(optimized_file, index=":")
    ats[4].info["is_finished"] = False
    handler.remote_fs.write_ase(optimized_file, ats)

    handler.recompute_failed()

    handler.collect_results()


def test_slurm_normal(instance):
    handler, params = instance
    handler.run()
    all_atoms, failed_indices = handler.collect(only_finished=False)

    # We are skipping the actual computations, so Atoms should still be
    # identical
    for tfile in handler.traj_files:
        ats0 = ase.io.read(tfile, index=":")
        ats1 = ase.io.read(
            optimizer.front_tag(tfile, handler.jobname), index=":"
        )
        for at0, at1 in zip(ats0, ats1):
            if params["stub"]:
                assert at0 == at1


def test_slurm_recompute(instance):
    handler, params = instance
    handler.run()

    # now emulate a failed run by deleting one of the produced files
    handler.remote_fs.remove(
        handler.savepath / "calcdirs" / "0" / "opt_0.extxyz"
    )

    handler.recompute_failed()


# @pytest.fixture(
#     params=[
#         pytest.param("local", marks=[pytest.mark.local]),
#         pytest.param("datalab", marks=[pytest.mark.datalab]),
#         pytest.param("lugus", marks=[pytest.mark.lugus])
#     ]
# )
# def nonslurm_instance(request, traj_files, remote_dir, data_dir):
#     relative_path = Path("../../scripts/utility/optimize_traj.py")
#     python_script_path = (data_dir / relative_path).resolve()
#     pickle_file = data_dir / "systems/silicon/n16/model_params_plain_ensemble_VELO_8E04F5B1.pkl"

#     homepath = data_dir / "temp/optimizer/nonslurm"
#     savepath = homepath / "savepath"
#     if os.path.exists(homepath):
#         for root, dirs, files in os.walk(homepath, topdown=False):
#             for file in files:
#                 os.remove(os.path.join(root, file))
#             for dir in dirs:
#                 os.rmdir(os.path.join(root, dir))
#         os.rmdir(homepath)

#     os.makedirs(savepath, exist_ok=True)

#     if request.param == "lugus":
#         remote_fs = optimizer.RemoteFileSystem(
#             "lugus",
#             local_temp_path="/home/nico.unglert/dump/optimizations",
#             remote_temp_path="/home/nunglert/dump/optimizations"
#         )
#         savepath = remote_fs.secure_path(
#             "/home/nunglert/cellar/projects/replica_exchange/silicon/optimizations/pytest",
#             is_directory=True
#         )
#         print(savepath)
#     elif request.param == "datalab":
#         remote_fs = optimizer.RemoteFileSystem(
#             "datalab",
#             local_temp_path="/home/nico.unglert/dump/optimizations",
#             remote_temp_path="/home/nico.unglert/dump/optimizations"
#         )
#     elif request.param == "local":
#         remote_fs = optimizer.LocalFileSystem()
#     else:
#         raise Exception("Unknown remote machine!")

#     params = dict(
#         traj_files=traj_files,
#         shard_size=50,
#         savepath=savepath,
#         num_parallel=5,
#         every=5,
#         python_script_path=python_script_path,
#         pickle_file=pickle_file,
#         supercell_trafo=[3, 3, 3],
#         max_neighbors_list=[20, 30, 40, 50, 60],
#         max_neighbors_offset=3,
#         optimization_backend="LJ",
#         loop_time=5,
#         jobname="opt",
#         stub=False,
#         do_steinhardt=True,
#         do_optimization=False,
#         # remote_machine=request.param
#         remote_fs=remote_fs,
#         logfile=data_dir / "temp/optimizer/nonslurm/opt.log"
#     )
#     handler = optimizer.RemoteOptimizationHandler(**params)
#     return handler, params

# def test_nonslurm(nonslurm_instance):
#     handler, params = nonslurm_instance
#     handler.run_nonslurm()


# def test_nonslurm_broken(nonslurm_instance):
#     handler, params = nonslurm_instance
#     handler.run_nonslurm()

#     opt_file = Path(handler.calcdir) / "0" / "opt_0.extxyz"
#     ats = ase.io.read(opt_file, index=":")
#     ase.io.write(opt_file, ats[1:])

#     handler.run_nonslurm()
#     with open(Path(handler.savepath) / "nonslurm.log", "r") as f:
#         print(f.read())
