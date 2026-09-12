"""``get_trained_model`` collects only real array-task dirs from calcdirs/.

Mirrors ``test_trainer_collection.py`` (the MACE version) for the new
``AceModelTrainer``/``GraceModelTrainer``, which share the array-job
collection logic via ``_EnsembleArrayTrainer`` rather than
``_MaceArrayTrainer``. Pure logic -- no pacemaker/gracemaker needed.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import LocalFileSystem
from remote_handler.workflows.trainers import (
    AceModelTrainer,
    GraceModelTrainer,
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


def _finished_task():
    task = MagicMock()
    task.finish_check.return_value = True
    return task


def _make_ace_job_dir(calcdirs, name, model_name="Model"):
    models = calcdirs / name / "models"
    models.mkdir(parents=True)
    (models / f"{model_name}.yaml").write_text("potential: {}\n")


def _make_grace_job_dir(calcdirs, name, model_name="Model"):
    models = calcdirs / name / "models"
    models.mkdir(parents=True)
    model_dir = models / model_name
    model_dir.mkdir()
    (model_dir / "model.yaml").write_text("cutoff: 5.0\n")


TRAINERS = [
    (AceModelTrainer, _make_ace_job_dir),
    (GraceModelTrainer, _make_grace_job_dir),
]


@pytest.mark.parametrize("trainer_cls,make_job_dir", TRAINERS)
def test_collection_skips_non_job_dirs_without_warning(
    tmp_path, caplog, trainer_cls, make_job_dir
):
    t = trainer_cls(
        remote_fs=LocalFileSystem(),
        cluster_config=_cfg(tmp_path),
        config_file="config.yaml",
        remote_train_runner="runner",
        stub=False,
    )
    t.results_dir = tmp_path / "results"
    calcdirs = t.results_dir / "calcdirs"
    calcdirs.mkdir(parents=True)

    # Two genuine array-task dirs (numeric run ids, as collected for real runs)
    make_job_dir(calcdirs, "0")
    make_job_dir(calcdirs, "1")
    # Non-job entries that share calcdirs/ -- these must be ignored silently.
    (calcdirs / "logs").mkdir()
    (calcdirs / "train.log").write_text("…")

    t.training_task = _finished_task()

    with caplog.at_level(logging.WARNING):
        models = t.get_trained_model()

    assert sorted(Path(m).parent.parent.name for m in models) == ["0", "1"]
    assert not any("Model not found" in rec.message for rec in caplog.records)


@pytest.mark.parametrize("trainer_cls,make_job_dir", TRAINERS)
def test_collection_warns_for_real_job_dir_missing_artifact(
    tmp_path, caplog, trainer_cls, make_job_dir
):
    """A genuine job dir (has models/) but lacking the artifact still warns."""
    t = trainer_cls(
        remote_fs=LocalFileSystem(),
        cluster_config=_cfg(tmp_path),
        config_file="config.yaml",
        remote_train_runner="runner",
        stub=False,
    )
    t.results_dir = tmp_path / "results"
    calcdirs = t.results_dir / "calcdirs"
    calcdirs.mkdir(parents=True)

    make_job_dir(calcdirs, "0")  # good
    (calcdirs / "1" / "models").mkdir(parents=True)  # models/ but no artifact

    t.training_task = _finished_task()

    with caplog.at_level(logging.WARNING):
        models = t.get_trained_model()

    assert [Path(m).parent.parent.name for m in models] == ["0"]
    assert any("Model not found" in rec.message for rec in caplog.records)


@pytest.mark.parametrize("trainer_cls", [AceModelTrainer, GraceModelTrainer])
def test_stub_mode_creates_mock_models(tmp_path, trainer_cls):
    """stub=True needs no pacemaker/gracemaker anywhere."""
    t = trainer_cls(
        remote_fs=LocalFileSystem(),
        cluster_config=_cfg(tmp_path),
        config_file="config.yaml",
        num_models=3,
        stub=True,
    )
    t.update(tmp_path / "0", [tmp_path / "db.extxyz"])
    models = t.get_trained_model()

    assert len(models) == 3
    for m in models:
        assert Path(m).exists()
