"""Hyperparameter resolution for NeuralilILTrainer.

Covers the three-layer merge (defaults < config_file YAML < training_kwargs),
backward compatibility of the ``training_kwargs`` dict, and a regression guard
that constructing a trainer never mutates the shared class-level defaults.

Construction is host-side only (no neuralil stack, no remote), so these are
plain unit tests.
"""

import yaml

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import LocalFileSystem
from remote_handler.workflows.trainers import NeuralilILTrainer


def _cfg(tmp_path):
    return ClusterConfig(
        remote_machine="m",
        partition="p",
        conda_env="e",
        conda_source="/x",
        savepath=str(tmp_path),
        template="t.j2",
    )


def _trainer(tmp_path, **kw):
    return NeuralilILTrainer(
        remote_fs=LocalFileSystem(),
        cluster_config=_cfg(tmp_path),
        training_script="train.py",
        stub=True,
        **kw,
    )


def test_defaults_used_when_nothing_given(tmp_path):
    t = _trainer(tmp_path)
    assert t.training_kwargs["n_epochs"] == 100  # a built-in default
    assert t.training_kwargs["stub"] is True


def test_training_kwargs_override_defaults(tmp_path):
    # Backward compatibility: the legacy dict still works.
    t = _trainer(tmp_path, training_kwargs={"n_epochs": 5})
    assert t.training_kwargs["n_epochs"] == 5


def test_config_file_overlays_defaults(tmp_path):
    cfg = tmp_path / "nnff.yaml"
    cfg.write_text(yaml.safe_dump({"n_epochs": 7, "r_cut": 6.0}))
    t = _trainer(tmp_path, config_file=cfg)
    assert t.training_kwargs["n_epochs"] == 7
    assert t.training_kwargs["r_cut"] == 6.0


def test_training_kwargs_beat_config_file(tmp_path):
    cfg = tmp_path / "nnff.yaml"
    cfg.write_text(yaml.safe_dump({"n_epochs": 7}))
    t = _trainer(tmp_path, config_file=cfg, training_kwargs={"n_epochs": 9})
    assert t.training_kwargs["n_epochs"] == 9


def test_no_cross_instance_pollution(tmp_path):
    """Constructing with overrides must NOT mutate the class defaults."""
    _trainer(tmp_path, training_kwargs={"n_epochs": 999})
    second = _trainer(tmp_path)
    assert second.training_kwargs["n_epochs"] == 100
    assert NeuralilILTrainer.default_training_kwargs["n_epochs"] == 100


def test_absolute_high_energy_penalty_is_default(tmp_path):
    t = _trainer(tmp_path)
    assert t.training_kwargs["high_energy_penalty"] == -5.0
    assert t.training_kwargs["high_energy_penalty_relative"] is None


def test_relative_penalty_drops_absolute_default(tmp_path):
    t = _trainer(
        tmp_path, training_kwargs={"high_energy_penalty_relative": 3.5}
    )
    assert t.training_kwargs["high_energy_penalty_relative"] == 3.5
    assert t.training_kwargs["high_energy_penalty"] is None


def test_relative_penalty_from_config_file(tmp_path):
    cfg = tmp_path / "nnff.yaml"
    cfg.write_text(yaml.safe_dump({"high_energy_penalty_relative": 3.0}))
    t = _trainer(tmp_path, config_file=cfg)
    assert t.training_kwargs["high_energy_penalty"] is None


def test_explicit_absolute_penalty_kept_alongside_relative(tmp_path):
    # Both explicit: passed through, and the training script rejects it.
    t = _trainer(
        tmp_path,
        training_kwargs={
            "high_energy_penalty": -4.0,
            "high_energy_penalty_relative": 3.0,
        },
    )
    assert t.training_kwargs["high_energy_penalty"] == -4.0


def test_scheduled_relative_penalty_drops_absolute(tmp_path):
    t = _trainer(tmp_path)
    t.apply_schedule({"high_energy_penalty_relative": 2.5})
    assert t.training_kwargs["high_energy_penalty_relative"] == 2.5
    assert t.training_kwargs["high_energy_penalty"] is None
