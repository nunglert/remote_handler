import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import compare_atoms

from remote_handler.core.atoms_shard import front_tag
from remote_handler.workflows import optimizer, run_optimizer


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


@pytest.fixture(
    params=[
        pytest.param("remote", marks=[pytest.mark.remote]),
        pytest.param("local", marks=[pytest.mark.local]),
    ]
)
def fs_config_savepath(
    request,
    remote_fs,
    remote_cluster_config,
    local_fs,
    local_cluster_config,
    tmp_path,
):
    """Provide resolved filesystem, cluster_config, and savepath for parametrized tests."""
    if request.param == "remote":
        return remote_fs, remote_cluster_config, remote_cluster_config.savepath
    else:
        return (
            local_fs,
            local_cluster_config,
            str(tmp_path / "local_slurm_test"),
        )


@pytest.fixture
def default_params(fs_config_savepath, data_dir, dummy_trajectory):
    """
    Variables:
        asset_dir: Path with static files required for test
        temp_dir: Path with temporary files created during the test
        local_dir: Location where the test is operating on the local machine
        local_results: Path where the directory from the remote machine will be
            copied to in order to investigate the resulting files.
    """
    filesystem, cluster_config, savepath = fs_config_savepath

    asset_dir = data_dir / "assets/optimizer/"
    temp_dir = data_dir / "temp/optimizer"

    local_dir = temp_dir / "local_dir"
    local_results = temp_dir / "results"

    # These are static files the test needs to access
    pickle_file = asset_dir / "model_params_plain_ensemble_VELO_8E04F5B1.pkl"
    python_script_path = asset_dir / "simple_at_processor.py"
    # python_script_path = asset_dir / "optimize_traj.py"

    logfile = "test_slurm.log"

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
        cluster_config=cluster_config,
        shard_size=10,
        savepath=savepath,
        num_parallel=5,
        n_processors=2,
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
        do_rdf=True,
        remote_fs=filesystem,
        logfile=logfile,
        collect_only=False,
        optimization_backend="NNFF",
        steinhardt_backend="pyscal",
        steinhardt_l_value=4,
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
        opt_tfile = front_tag(tfile, handler.tag)
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
    optimized_file = front_tag(traj_file, handler.tag)
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
    optimized_file = front_tag(traj_file, handler.tag)
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
    optimized_file = front_tag(traj_file, handler.tag)
    ats = handler.remote_fs.read_ase(optimized_file, index=":")
    ats[4].info["is_finished"] = False
    handler.remote_fs.write_ase(optimized_file, ats)

    handler.recompute_failed()

    handler.collect_results()


@pytest.fixture()
def argparser():
    return run_optimizer.get_parser()


def test_run_optimizer_parse_default(argparser):
    args = argparser.parse_args([])
    # for k,v in args.__dict__.items():
    #     print(k, " = ", v)


def test_run_optimizer_parse_default_process(argparser):
    args = argparser.parse_args([])

    with pytest.raises(Exception):
        args = run_optimizer.process_args(args, argparser)


def test_run_optimizer_parse_simple(argparser):
    args = argparser.parse_args(["--files", "lel.extxyz"])
    args = run_optimizer.process_args(args, argparser)

    assert isinstance(args.supercell_trafo, list)
    assert isinstance(args.max_neighbors_list, list)
    # Note: arg parser uses steinhardt_l_values (plural), but
    # RemoteOptimizationHandler expects steinhardt_l_value (singular)
    assert isinstance(args.steinhardt_l_values, list)


# @pytest.mark.parametrize("use_logfile", [True, False])
# @pytest.mark.parametrize("log_mode", ["stdout", "file"])
@pytest.mark.parametrize(
    "log_mode,use_logfile",
    [
        ("stdout", True),
        ("stdout", False),
        ("file", True),
    ],
)
def test_run_optimizer_parse_logger(
    argparser, log_mode, use_logfile, data_dir
):
    temp_dir = data_dir / "temp/run_optimizer/logger"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    log_file = temp_dir / "opt.log"

    arg_list = [
        "--files",
        "lel.extxyz",
        "--log_mode",
        log_mode,
    ]
    if use_logfile:
        arg_list.extend(["--logfile", str(log_file)])

    args = argparser.parse_args(arg_list)
    args = run_optimizer.process_args(args, argparser)


class ScratchDir:
    def __init__(self, data_dir, subfolder_name, dummy_trajectory):
        self.asset_dir = data_dir / "assets/run_optimizer/"
        self.temp_dir = data_dir / "temp/run_optimizer" / subfolder_name

        self.local_dir = self.temp_dir / "local_dir"
        self.local_results = self.temp_dir / "results"

        self.pickle_file = (
            self.asset_dir / "model_params_plain_ensemble_VELO_8E04F5B1.pkl"
        )
        self.python_script_path = self.asset_dir / "simple_at_processor.py"

        for path in [self.local_dir, self.local_results]:
            if path is not None:
                if os.path.exists(path):
                    shutil.rmtree(path)
        os.makedirs(self.local_dir, exist_ok=False)

        self.logfile = self.temp_dir / "opt.log"
        if os.path.exists(self.logfile):
            os.remove(self.logfile)

        self.tfile0 = str(self.local_dir / "traj0.extxyz")
        ase.io.write(self.tfile0, dummy_trajectory)
        self.tfile1 = str(self.local_dir / "traj1.extxyz")
        ase.io.write(self.tfile1, dummy_trajectory)


def test_setup_optimizer_case1(argparser, data_dir, dummy_trajectory):
    # Case 1: Explicitly providing extxyz files
    data = ScratchDir(data_dir, "case1", dummy_trajectory)
    args = argparser.parse_args(
        [
            "--files",
            data.tfile0,
            data.tfile1,
            "--log_mode",
            "file",
            "--logfile",
            data.logfile,
        ]
    )
    args = run_optimizer.process_args(args, argparser)
    handler = run_optimizer.setup_optimizer(args)


def test_setup_optimizer_case2(argparser, data_dir, dummy_trajectory):
    # Case 2: Parsing a configuration file
    data = ScratchDir(data_dir, "case2", dummy_trajectory)
    json_contents = {
        "files": [data.tfile0, data.tfile1],
        "log_mode": "file",
        "logfile": data.logfile,
        "every": 1,
    }

    config_file = str(data.temp_dir / "config.json")
    with open(config_file, "w") as f:
        json.dump(json_contents, f)

    args = argparser.parse_args(["--config", config_file, "--every", "2"])
    args = run_optimizer.process_args(args, argparser)
    assert args.every == 2, "Config is not overwritten by passed arg!"

    handler = run_optimizer.setup_optimizer(args)


def test_setup_optimizer_case3(argparser, data_dir, dummy_trajectory):
    # Case 3: Parsing an existing logfile
    data = ScratchDir(data_dir, "case3", dummy_trajectory)

    # we do one dummy run of process_args to generate a logfile
    args = argparser.parse_args(
        [
            "--files",
            data.tfile0,
            data.tfile1,
            "--log_mode",
            "file",
            "--logfile",
            data.logfile,
            "--every",
            "1",
        ]
    )
    args0 = run_optimizer.process_args(args, argparser)

    # now we try to parse from it
    args = argparser.parse_args(
        [
            "--read_out",
            str(data.temp_dir / "opt.log"),
            "--every",
            "2",
            "--logfile",
            data.logfile,
        ]
    )
    args1 = run_optimizer.process_args(args, argparser)
    assert args1.every == 2, "log params are not overwritten by passed arg!"
    handler = run_optimizer.setup_optimizer(args)

    for (k0, a0), (k1, a1) in zip(
        args0.__dict__.items(), args1.__dict__.items()
    ):
        # Obviously read_out may not be contained in args0, but we also have to
        # filter out remote_fs, since this is an object instance which will be
        # different for two separate initializations
        if not k0 in ["read_out", "every", "remote_fs"]:
            assert a0 == a1, f"{k0} not identical!"
