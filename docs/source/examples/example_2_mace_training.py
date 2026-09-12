"""Example 2 -- Training a MACE force field.

Configure a ``MaceModelTrainer`` and run one training step. In a full run the
``StagedActiveLearningHandler`` calls ``update`` / ``get_trained_model`` for
you each iteration; here we drive a single step directly.

MACE (``mace-torch``) must be installed in the *remote* conda env this
trainer targets (``convert_to_jax=True`` additionally needs ``mace-jax``
there). ``training_script`` and ``remote_mace_train_runner`` are both
optional and left unset below -- they default to the shipped payload script
and are auto-discovered by probing that remote env, respectively.

Included verbatim by ``docs/source/examples.rst`` and signature-checked by
``test/test_examples.py``.
"""

from pathlib import Path

from remote_handler.core import ClusterConfig
from remote_handler.core.file_handler import RemoteFileSystem
from remote_handler.workflows.trainers import MaceModelTrainer

# See mace_config.yaml (next to this script) for a minimal config.
CONFIG_FILE = Path(__file__).parent / "mace_config.yaml"

if __name__ == "__main__":
    config = ClusterConfig.from_name("datalab")
    remote_fs = RemoteFileSystem(remote_alias="datalab")

    trainer = MaceModelTrainer(
        remote_fs=remote_fs,
        cluster_config=config,
        config_file=CONFIG_FILE,
        num_models=3,
        validation_fraction=0.1,
    )

    # Train on DFT-labelled configs; calcdir is this iteration's work dir.
    trainer.update(Path("mace_run/0/training"), ["dft_labelled.extxyz"])
    model_paths = trainer.get_trained_model()
