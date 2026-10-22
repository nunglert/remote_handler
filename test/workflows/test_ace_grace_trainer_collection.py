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


def _ace_update_capturing_task(tmp_path, monkeypatch, **update_kwargs):
    """Run ``AceModelTrainer.update`` with the array task recorded instead
    of built; returns ``(trainer, task_kwargs)``."""
    import remote_handler.workflows.trainers as trainers_mod

    recorded = {}

    class _RecordingTask:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

    monkeypatch.setattr(trainers_mod, "ArrayScriptTask", _RecordingTask)
    config = tmp_path / "ace.yaml"
    config.write_text("potential: {}\n")
    db = tmp_path / "db.extxyz"
    db.write_text("")
    trainer = AceModelTrainer(
        remote_fs=LocalFileSystem(),
        cluster_config=_cfg(tmp_path),
        config_file=config,
        remote_train_runner="pacemaker",
        num_models=3,
    )
    monkeypatch.setattr(
        trainer,
        "build_or_resume_array_task",
        lambda results_dir, build, **kw: build(),
    )
    trainer.update(tmp_path / "it", db, **update_kwargs)
    return trainer, recorded


def test_ace_warm_start_assigns_one_previous_member_each(
    tmp_path, monkeypatch
):
    previous = []
    for k in range(2):
        path = tmp_path / f"trainer_it1_{k}.yaml"
        path.write_text("species: []\n")
        previous.append(path)

    _, task = _ace_update_capturing_task(
        tmp_path, monkeypatch, initial_potentials=previous
    )

    inputs = [r.content for r in task["individual_ressources"][0]]
    assert [i["initial_potential"].split("/")[-1] for i in inputs] == [
        "trainer_it1_0.yaml",
        "trainer_it1_1.yaml",
        "trainer_it1_0.yaml",  # 3 members, 2 previous: cycles
    ]
    assert all(p in task["shared_ressources"] for p in previous)


@pytest.mark.parametrize("initial", [None, []])
def test_ace_without_initial_potentials_fits_from_scratch(
    tmp_path, monkeypatch, initial
):
    kwargs = {} if initial is None else {"initial_potentials": initial}

    _, task = _ace_update_capturing_task(tmp_path, monkeypatch, **kwargs)

    inputs = [r.content for r in task["individual_ressources"][0]]
    assert all("initial_potential" not in i for i in inputs)


def test_ace_warm_start_rejects_clashing_names(tmp_path, monkeypatch):
    from remote_handler.exceptions import ConfigurationError

    a = tmp_path / "a" / "Model.yaml"
    b = tmp_path / "b" / "Model.yaml"
    for p in (a, b):
        p.parent.mkdir()
        p.write_text("species: []\n")

    with pytest.raises(ConfigurationError, match="distinct file names"):
        _ace_update_capturing_task(
            tmp_path, monkeypatch, initial_potentials=[a, b]
        )


def _write_test_metrics(calcdirs, name, losses):
    """pacemaker-style test_metrics.txt: header + one row per iteration."""
    rows = ["ladder_step cycle_step iter_num loss rmse_f_comp"]
    rows += [f"0 0 {i} {loss} 0.1" for i, loss in enumerate(losses)]
    (calcdirs / name / "test_metrics.txt").write_text("\n".join(rows) + "\n")


def _ace_trainer(tmp_path, **kwargs):
    t = AceModelTrainer(
        remote_fs=LocalFileSystem(),
        cluster_config=_cfg(tmp_path),
        config_file="config.yaml",
        remote_train_runner="runner",
        stub=False,
        **kwargs,
    )
    t.results_dir = tmp_path / "results"
    (t.results_dir / "calcdirs").mkdir(parents=True)
    t.training_task = _finished_task()
    return t


def test_keep_best_hands_on_lowest_final_test_loss_first(tmp_path):
    t = _ace_trainer(tmp_path, num_models=4, keep_best=2)
    calcdirs = t.results_dir / "calcdirs"
    # final (last-row) test losses: 0 -> 0.03, 1 -> 4.3 (diverged),
    # 2 -> 0.02, 3 -> no metrics file
    for name, losses in (
        ("0", [12.0, 0.03]),
        ("1", [12.0, 0.01, 4.3]),  # best mid-fit, but its *final* is bad
        ("2", [12.0, 0.02]),
    ):
        _make_ace_job_dir(calcdirs, name)
        _write_test_metrics(calcdirs, name, losses)
    _make_ace_job_dir(calcdirs, "3")

    models = t.get_trained_model()

    assert [Path(m).parent.parent.name for m in models] == ["2", "0"]


def test_keep_best_ranks_members_without_metrics_last(tmp_path):
    t = _ace_trainer(tmp_path, num_models=3, keep_best=2)
    calcdirs = t.results_dir / "calcdirs"
    _make_ace_job_dir(calcdirs, "0")  # no metrics
    _make_ace_job_dir(calcdirs, "1")
    _write_test_metrics(calcdirs, "1", [0.5])
    _make_ace_job_dir(calcdirs, "2")
    _write_test_metrics(calcdirs, "2", ["nan"])

    models = t.get_trained_model()

    assert [Path(m).parent.parent.name for m in models] == ["1", "0"]


def test_without_keep_best_all_members_are_handed_on(tmp_path):
    t = _ace_trainer(tmp_path, num_models=2)
    calcdirs = t.results_dir / "calcdirs"
    for name, loss in (("0", 4.3), ("1", 0.02)):
        _make_ace_job_dir(calcdirs, name)
        _write_test_metrics(calcdirs, name, [loss])

    models = t.get_trained_model()

    assert sorted(Path(m).parent.parent.name for m in models) == ["0", "1"]


@pytest.mark.parametrize("keep_best", [0, 6])
def test_keep_best_out_of_range_is_rejected(tmp_path, keep_best):
    from remote_handler.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="keep_best"):
        _ace_trainer(tmp_path, num_models=5, keep_best=keep_best)


def test_keep_best_needs_a_test_loss_hook(tmp_path):
    from remote_handler.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="keep_best"):
        GraceModelTrainer(
            remote_fs=LocalFileSystem(),
            cluster_config=_cfg(tmp_path),
            config_file="config.yaml",
            num_models=3,
            keep_best=2,
        )


def test_stub_mode_keeps_best_count(tmp_path):
    t = AceModelTrainer(
        remote_fs=LocalFileSystem(),
        cluster_config=_cfg(tmp_path),
        config_file="config.yaml",
        num_models=4,
        keep_best=3,
        stub=True,
    )
    t.update(tmp_path / "0", [tmp_path / "db.extxyz"])

    assert len(t.get_trained_model()) == 3


@pytest.mark.parametrize("num_parallel,expected", [(None, 5), (8, 8)])
def test_num_parallel_defaults_to_the_old_cap(
    tmp_path, num_parallel, expected
):
    t = _ace_trainer(tmp_path, num_models=8, num_parallel=num_parallel)

    assert t.num_parallel == expected
