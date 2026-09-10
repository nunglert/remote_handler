"""Standalone submit-in-one-run, collect-in-a-later-run for workflow tasks.

``ArrayScriptTask`` used to store ``all_calcpaths``/``num_array_tasks`` only
as a side effect of ``allocate_data()`` (or, for ``RemoteAtomsHandler`` and
its subclasses, of the shard manager running in ``__init__``). A
``collect_only=True`` object built in a fresh process -- one that never
allocated -- therefore raised ``AttributeError`` from ``finish_check()``
instead of just checking the finished array. This module exercises the fix
for both a workflow handler (:class:`SinglePointHandler`) and the MACE
trainers, entirely against the scheduler-free ``direct`` backend on a
``LocalFileSystem`` -- no VASP/MACE/SLURM needed.
"""

import json
import pickle
from pathlib import Path

import ase.io
import pytest
from ase.build import bulk

from remote_handler.core.cluster_config import ClusterConfig
from remote_handler.core.file_handler import LocalFileSystem
from remote_handler.core.manifest import TaskHandle
from remote_handler.core.tasks import ArrayScriptTask
from remote_handler.workflows.dft_runner import SinglePointHandler
from remote_handler.workflows.trainers import (
    MaceFinetuningTrainer,
    MaceModelTrainer,
)


def _direct_config(savepath):
    return ClusterConfig(
        remote_machine="local",
        partition="p",
        conda_env="",
        conda_source="",
        savepath=str(savepath),
        template="unused.sh.j2",
        executor="direct",
    )


# --------------------------------------------------------------------------
# SinglePointHandler: finish_check() without allocation
# --------------------------------------------------------------------------


def test_singlepoint_handler_collect_only_finish_check_without_allocation(
    tmp_path,
):
    """A collect_only=True handler that never called allocate_data() can
    still finish_check() -- num_array_tasks/all_calcpaths come from the
    shard manager (recomputed fresh on every construction, collect_only or
    not), not from an allocation side effect."""
    remote = tmp_path / "remote"
    remote.mkdir()
    fs = LocalFileSystem()
    traj_file = tmp_path / "structures.extxyz"
    ase.io.write(traj_file, [bulk("Cu") for _ in range(3)])

    def build(collect_only):
        return SinglePointHandler(
            traj_files=[str(traj_file)],
            cluster_config=_direct_config(remote),
            remote_fs=fs,
            savepath=str(remote),
            num_parallel=2,
            python_script_path="unused_script.py",
            calc_config={},
            jobname="dft",
            collect_only=collect_only,
        )

    # Stand-in for process 1: build (but never submit), then mark every
    # calc dir as finished exactly as a real job would.
    handler1 = build(collect_only=False)
    assert len(handler1.all_calcpaths) == 3
    for calcpath in handler1.all_calcpaths:
        fs.makedirs(calcpath, exist_ok=True)
        (calcpath / "is_finished.info").write_text("done")

    # Stand-in for process 2: collect_only, no allocate_data() call anywhere.
    handler2 = build(collect_only=True)
    assert handler2.finish_check(required_fraction=1.0)


# --------------------------------------------------------------------------
# Bare ArrayScriptTask: full detach -> wait -> from_manifest() regression
# --------------------------------------------------------------------------

ARRAY_SCRIPT = """
import json, sys
from pathlib import Path

calc = Path(sys.argv[1])
data = json.load(open(calc / "input.json"))
(calc / "result.json").write_text(json.dumps({"sum": data["A"] + data["B"]}))
(calc / "is_finished.info").write_text("done")
"""


def test_array_task_detach_wait_from_manifest_collect(tmp_path):
    from remote_handler.core.tasks import DictJSON

    remote = tmp_path / "remote"
    remote.mkdir()
    script = tmp_path / "array.py"
    script.write_text(ARRAY_SCRIPT)
    fs = LocalFileSystem()

    n = 3
    inputs = [DictJSON({"A": 1, "B": i}, "input.json", fs) for i in range(n)]
    task = ArrayScriptTask(
        local_savepath=tmp_path / "results",
        cluster_config=_direct_config(remote),
        remote_fs=fs,
        python_script_file=script,
        individual_ressources=[inputs],
        loop_time=0.2,
        remote_savepath=str(remote),
    )

    manifest_path = task.detach()
    TaskHandle.from_file(manifest_path).wait()

    resumed = ArrayScriptTask.from_manifest(manifest_path, remote_fs=fs)
    assert resumed.finish_check()
    resumed.collect_results()

    from remote_handler.core.remote_env import CALCDIRS_DIRNAME

    for i in range(n):
        result = (
            tmp_path / "results" / CALCDIRS_DIRNAME / str(i) / "result.json"
        )
        assert json.loads(result.read_text())["sum"] == 1 + i


# --------------------------------------------------------------------------
# MACE trainers: update() -> submit_detached() -> wait -> collect_only
# update()/get_trained_model()
# --------------------------------------------------------------------------

FAKE_TRAIN_SCRIPT = """
import json, sys, pickle
from pathlib import Path

calc = Path(sys.argv[1])
data = json.load(open(calc / "input_train.json"))
models_dir = calc / "models"
models_dir.mkdir(parents=True, exist_ok=True)
mock_model = {"seed": data["seed"]}

if data.get("convert_to_jax"):
    if data.get("jax_output_format") == "bundle":
        jax_dir = models_dir / f"{data['model_name']}-jax"
        jax_dir.mkdir(parents=True, exist_ok=True)
        (jax_dir / "config.json").write_text("{}")
        (jax_dir / "params.msgpack").write_bytes(b"")
    else:
        with open(models_dir / f"{data['model_name']}-jax.pkl", "wb") as f:
            pickle.dump({"config": mock_model, "params": b"", "head": None}, f)
else:
    with open(models_dir / f"{data['model_name']}.model", "wb") as f:
        pickle.dump(mock_model, f)

(calc / "is_finished.info").write_text("done")
"""


def _mace_setup(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    script = tmp_path / "fake_train.py"
    script.write_text(FAKE_TRAIN_SCRIPT)
    config_file = tmp_path / "mace.yaml"
    config_file.write_text("dummy: true")
    db_file = tmp_path / "data.extxyz"
    db_file.write_text("dummy db")
    return remote, script, config_file, db_file, LocalFileSystem()


@pytest.mark.parametrize(
    "convert_to_jax,jax_output_format",
    [(False, "pkl"), (True, "pkl"), (True, "bundle")],
)
def test_mace_model_trainer_collect_only_round_trip(
    tmp_path, convert_to_jax, jax_output_format
):
    remote, script, config_file, db_file, fs = _mace_setup(tmp_path)
    calcdir = tmp_path / "iter0"

    trainer1 = MaceModelTrainer(
        remote_fs=fs,
        cluster_config=_direct_config(remote),
        config_file=str(config_file),
        training_script=str(script),
        remote_mace_train_runner="unused_run_train.py",
        num_models=2,
        loop_time=0.2,
        convert_to_jax=convert_to_jax,
        jax_output_format=jax_output_format,
    )
    trainer1.update(calcdir, [db_file])
    manifest_path = trainer1.training_task.submit_detached()
    TaskHandle.from_file(manifest_path).wait()

    trainer2 = MaceModelTrainer(
        remote_fs=fs,
        cluster_config=_direct_config(remote),
        config_file=str(config_file),
        training_script=str(script),
        remote_mace_train_runner="unused_run_train.py",
        num_models=2,
        loop_time=0.2,
        convert_to_jax=convert_to_jax,
        jax_output_format=jax_output_format,
        collect_only=True,
    )
    trainer2.update(calcdir, None)
    model_paths = trainer2.get_trained_model()

    assert len(model_paths) == 2
    for p in model_paths:
        assert p.exists()
        if convert_to_jax and jax_output_format == "bundle":
            assert p.is_dir()
            assert (p / "config.json").exists()


def test_mace_finetuning_trainer_forwards_collect_only(tmp_path):
    remote, script, config_file, db_file, fs = _mace_setup(tmp_path)
    foundation_model = tmp_path / "foundation.model"
    with open(foundation_model, "wb") as f:
        pickle.dump({"foundation": True}, f)
    calcdir = tmp_path / "iter0"

    def build(collect_only):
        return MaceFinetuningTrainer(
            remote_fs=fs,
            cluster_config=_direct_config(remote),
            config_file=str(config_file),
            foundation_model=str(foundation_model),
            training_script=str(script),
            remote_mace_train_runner="unused_run_train.py",
            num_models=2,
            loop_time=0.2,
            collect_only=collect_only,
        )

    trainer1 = build(collect_only=False)
    assert trainer1.collect_only is False
    trainer1.update(calcdir, [db_file])
    manifest_path = trainer1.training_task.submit_detached()
    TaskHandle.from_file(manifest_path).wait()

    trainer2 = build(collect_only=True)
    assert trainer2.collect_only is True
    trainer2.update(calcdir, None)
    model_paths = trainer2.get_trained_model()

    assert len(model_paths) == 2
    for p in model_paths:
        assert p.exists()
