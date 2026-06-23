"""Actors raise WorkflowError when their job leaves no is_finished marker.

Each sampler/trainer calls the task's ``finish_check`` between ``submit`` and
``collect_results``; a False result (no ``is_finished.info`` written by the
payload) must abort with a ``WorkflowError`` rather than silently collecting
partial/empty output. The task is mocked so no real cluster is needed.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import LocalFileSystem
from remote_handler.exceptions import WorkflowError
from remote_handler.workflows.samplers import (
    CMAESSampler,
    LammpsSampler,
    SSPDSampler,
    TorchSimSampler,
)
from remote_handler.workflows.trainers import (
    MaceFinetuningTrainer,
    MaceModelTrainer,
)


def _cfg(tmp_path):
    return ClusterConfig(
        remote_machine="m",
        partition="p",
        conda_env="e",
        conda_source="/x",
        savepath=str(tmp_path),
        template="t.j2",
    )


def _failed_task():
    """A task whose job ran but wrote no success marker."""
    task = MagicMock()
    task.finish_check.return_value = False
    task.calcdir = "/remote/calc"
    return task


def test_lammps_raises_on_unfinished():
    s = LammpsSampler(
        remote_fs=LocalFileSystem(),
        template_path="lammps.in.j2",
        data_path="lammps.data",
        element_mapping={1: "Cu"},
        num_samples=5,
        uncertainty_cutoff_radius=4.0,
        remote_lammps_executable_path="/usr/bin/lmp",
        stub=False,
    )
    s.sampling_task = _failed_task()
    with pytest.raises(WorkflowError, match="did not finish"):
        s.get_sampled_configs()
    s.sampling_task.collect_results.assert_not_called()


def test_cmaes_raises_on_unfinished(tmp_path):
    s = CMAESSampler(
        remote_fs=LocalFileSystem(),
        cluster_config=_cfg(tmp_path),
        founder_path="POSCAR",
        stub=False,
    )
    s.sampling_task = _failed_task()
    with pytest.raises(WorkflowError, match="did not finish"):
        s.get_sampled_configs()
    s.sampling_task.collect_results.assert_not_called()


def test_torch_sim_raises_on_unfinished(tmp_path):
    s = TorchSimSampler(
        remote_fs=LocalFileSystem(),
        cluster_config=_cfg(tmp_path),
        founder_path="founder.xyz",
        ensemble="nvt",
        temperatures=[300.0],
        target_temperature=300.0,
        stub=False,
    )
    s.sampling_task = _failed_task()
    with pytest.raises(WorkflowError, match="did not finish"):
        s.get_sampled_configs()
    s.sampling_task.collect_results.assert_not_called()


def test_sspd_raises_on_unfinished(tmp_path):
    s = SSPDSampler(
        remote_fs=LocalFileSystem(),
        data_path="start.xyz",
        cluster_config=_cfg(tmp_path),
    )
    s.sampling_task = _failed_task()
    with pytest.raises(WorkflowError, match="did not finish"):
        s.get_sampled_configs()
    s.sampling_task.collect_results.assert_not_called()


def test_mace_trainer_raises_on_unfinished(tmp_path):
    t = MaceModelTrainer(
        remote_fs=LocalFileSystem(),
        cluster_config=_cfg(tmp_path),
        config_file="mace.yaml",
        remote_mace_train_runner="run_train.py",
        stub=False,
    )
    t.results_dir = tmp_path
    t.training_task = _failed_task()
    with pytest.raises(WorkflowError, match="did not finish"):
        t.get_trained_model()
    t.training_task.collect_results.assert_not_called()


def test_mace_finetuning_raises_on_unfinished(tmp_path):
    t = MaceFinetuningTrainer(
        remote_fs=LocalFileSystem(),
        cluster_config=_cfg(tmp_path),
        config_file="mace.yaml",
        foundation_model="foundation.model",
        remote_mace_train_runner="run_train.py",
        stub=False,
    )
    t.results_dir = tmp_path
    t.training_task = _failed_task()
    with pytest.raises(WorkflowError, match="did not finish"):
        t.get_trained_model()
    t.training_task.collect_results.assert_not_called()


def test_finish_check_runs_before_collect(tmp_path):
    """The check gates the (move-)collect, so it must precede it."""
    s = SSPDSampler(
        remote_fs=LocalFileSystem(),
        data_path="start.xyz",
        cluster_config=_cfg(tmp_path),
    )
    s.sampling_task = _failed_task()
    with pytest.raises(WorkflowError):
        s.get_sampled_configs()
    # submit happened, finish_check happened, collect did NOT.
    s.sampling_task.submit.assert_called_once()
    s.sampling_task.finish_check.assert_called_once()
    s.sampling_task.collect_results.assert_not_called()
