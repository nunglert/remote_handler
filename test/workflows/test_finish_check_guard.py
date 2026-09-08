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
    t = MaceModelTrainer(
        remote_fs=LocalFileSystem(),
        cluster_config=_cfg(tmp_path),
        config_file="mace.yaml",
        remote_mace_train_runner="run_train.py",
        stub=False,
    )
    t.results_dir = tmp_path
    t.training_task = _failed_task()
    with pytest.raises(WorkflowError):
        t.get_trained_model()
    # submit happened, finish_check happened, collect did NOT.
    t.training_task.submit.assert_called_once()
    t.training_task.finish_check.assert_called_once()
    t.training_task.collect_results.assert_not_called()
