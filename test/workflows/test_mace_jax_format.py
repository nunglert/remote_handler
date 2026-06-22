"""MaceFinetuningTrainer jax_output_format switch (stub mode, no ML stack)."""

import pytest

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import LocalFileSystem
from remote_handler.exceptions import ConfigurationError
from remote_handler.workflows.trainers import MaceFinetuningTrainer


def _trainer(tmp_path, **kw):
    cfg = ClusterConfig(
        remote_machine="m",
        partition="p",
        conda_env="e",
        conda_source="/x",
        savepath=str(tmp_path),
        template="t.j2",
    )
    return MaceFinetuningTrainer(
        remote_fs=LocalFileSystem(),
        cluster_config=cfg,
        training_script="t.py",
        config_file="c.yaml",
        foundation_model="f.model",
        remote_mace_train_runner="rt.py",
        num_models=1,
        stub=True,
        convert_to_jax=True,
        **kw,
    )


def test_default_format_is_pkl(tmp_path):
    t = _trainer(tmp_path)
    assert t.jax_output_format == "pkl"
    assert t._jax_artifact_name().endswith("-jax.pkl")


def test_bundle_format_artifact_is_directory_name(tmp_path):
    t = _trainer(tmp_path, jax_output_format="bundle")
    assert t._jax_artifact_name() == "Model-jax"  # no extension -> a dir


def test_invalid_format_rejected(tmp_path):
    with pytest.raises(ConfigurationError, match="jax_output_format"):
        _trainer(tmp_path, jax_output_format="nonsense")


def test_stub_bundle_creates_directory(tmp_path):
    t = _trainer(tmp_path, jax_output_format="bundle")
    # update() sets results_dir; get_trained_model() drives the stub outputs.
    t.update(tmp_path / "iter0", [tmp_path / "db.extxyz"])
    paths = t.get_trained_model()
    assert paths and all(p.is_dir() for p in paths)
    assert all((p / "config.json").exists() for p in paths)
