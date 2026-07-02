"""``get_trained_model`` collects only real array-task dirs from calcdirs/.

A finished MACE array run leaves its per-model output under
``calcdirs/<id>/models/``, but the same ``calcdirs/`` directory also collects
non-job entries (``logs/``, ``.jax_cache/``, ``hlo_dumps/``, ...). Collection
must pick up the genuine model dirs and skip the rest silently -- without
emitting a spurious "Model not found" warning for every non-job entry.
"""

import logging
import pickle
from pathlib import Path
from unittest.mock import MagicMock

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import LocalFileSystem
from remote_handler.workflows.trainers import MaceModelTrainer


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


def _make_job_dir(calcdirs, name, model_name="Model"):
    models = calcdirs / name / "models"
    models.mkdir(parents=True)
    with open(models / f"{model_name}.model", "wb") as f:
        pickle.dump({"model_type": "MACE"}, f)


def test_collection_skips_non_job_dirs_without_warning(tmp_path, caplog):
    t = MaceModelTrainer(
        remote_fs=LocalFileSystem(),
        cluster_config=_cfg(tmp_path),
        config_file="mace.yaml",
        remote_mace_train_runner="run_train.py",
        stub=False,
    )
    t.results_dir = tmp_path / "results"
    calcdirs = t.results_dir / "calcdirs"
    calcdirs.mkdir(parents=True)

    # Two genuine array-task dirs (numeric run ids, as collected for real runs)
    _make_job_dir(calcdirs, "0")
    _make_job_dir(calcdirs, "1")
    # Non-job entries that share calcdirs/ -- these must be ignored silently.
    (calcdirs / "logs").mkdir()
    (calcdirs / ".jax_cache").mkdir()
    (calcdirs / "hlo_dumps").mkdir()
    (calcdirs / "train.log").write_text("…")

    t.training_task = _finished_task()

    with caplog.at_level(logging.WARNING):
        models = t.get_trained_model()

    assert sorted(Path(m).parent.parent.name for m in models) == ["0", "1"]
    assert not any("Model not found" in rec.message for rec in caplog.records)


def test_collection_warns_for_real_job_dir_missing_artifact(tmp_path, caplog):
    """A genuine job dir (has models/) but lacking the artifact still warns."""
    t = MaceModelTrainer(
        remote_fs=LocalFileSystem(),
        cluster_config=_cfg(tmp_path),
        config_file="mace.yaml",
        remote_mace_train_runner="run_train.py",
        stub=False,
    )
    t.results_dir = tmp_path / "results"
    calcdirs = t.results_dir / "calcdirs"
    calcdirs.mkdir(parents=True)

    _make_job_dir(calcdirs, "0")  # good
    (calcdirs / "1" / "models").mkdir(parents=True)  # models/ but no .model

    t.training_task = _finished_task()

    with caplog.at_level(logging.WARNING):
        models = t.get_trained_model()

    assert [Path(m).parent.parent.name for m in models] == ["0"]
    assert any("Model not found" in rec.message for rec in caplog.records)
