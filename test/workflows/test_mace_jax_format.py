"""MACE trainers' jax_output_format switch (stub mode, no ML stack).

Both array trainers inherit the torch -> mace-jax conversion from their shared
base, so each case runs against ``MaceModelTrainer`` and
``MaceFinetuningTrainer`` alike.
"""

import pytest

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import LocalFileSystem
from remote_handler.exceptions import ConfigurationError
from remote_handler.workflows.trainers import (
    MaceFinetuningTrainer,
    MaceModelTrainer,
)

TRAINERS = [MaceModelTrainer, MaceFinetuningTrainer]


def _trainer(cls, tmp_path, **kw):
    cfg = ClusterConfig(
        remote_machine="m",
        partition="p",
        conda_env="e",
        conda_source="/x",
        savepath=str(tmp_path),
        template="t.j2",
    )
    # foundation_model is finetuning-only; the from-scratch trainer takes none.
    extra = (
        {"foundation_model": "f.model"} if cls is MaceFinetuningTrainer else {}
    )
    return cls(
        remote_fs=LocalFileSystem(),
        cluster_config=cfg,
        training_script="t.py",
        config_file="c.yaml",
        remote_mace_train_runner="rt.py",
        num_models=1,
        stub=True,
        **extra,
        **{"convert_to_jax": True, **kw},  # kw may override the default
    )


@pytest.mark.parametrize("cls", TRAINERS)
def test_default_format_is_pkl(cls, tmp_path):
    t = _trainer(cls, tmp_path)
    assert t.jax_output_format == "pkl"
    assert t._jax_artifact_name().endswith("-jax.pkl")


@pytest.mark.parametrize("cls", TRAINERS)
def test_bundle_format_artifact_is_directory_name(cls, tmp_path):
    t = _trainer(cls, tmp_path, jax_output_format="bundle")
    assert t._jax_artifact_name() == "Model-jax"  # no extension -> a dir


@pytest.mark.parametrize("cls", TRAINERS)
def test_no_conversion_keeps_torch_artifact(cls, tmp_path):
    t = _trainer(cls, tmp_path, convert_to_jax=False)
    assert t._jax_artifact_name() == "Model.model"


@pytest.mark.parametrize("cls", TRAINERS)
def test_invalid_format_rejected(cls, tmp_path):
    with pytest.raises(ConfigurationError, match="jax_output_format"):
        _trainer(cls, tmp_path, jax_output_format="nonsense")


@pytest.mark.parametrize("cls", TRAINERS)
def test_stub_bundle_creates_directory(cls, tmp_path):
    t = _trainer(cls, tmp_path, jax_output_format="bundle")
    # update() sets results_dir; get_trained_model() drives the stub outputs.
    t.update(tmp_path / "iter0", [tmp_path / "db.extxyz"])
    paths = t.get_trained_model()
    assert paths and all(p.is_dir() for p in paths)
    assert all((p / "config.json").exists() for p in paths)
