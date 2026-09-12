"""Example 6 -- Training an ACE force field.

Configure an ``AceModelTrainer`` -- the same ensemble-array actor shape as
``MaceModelTrainer`` (Example 2), fitting a classical ACE potential with
``pacemaker`` instead of MACE.

``pacemaker`` must be installed in the *remote* conda env this trainer
targets (see the ``ace`` extra for the pieces needed to test this locally
-- it cannot coexist with the ``grace`` extra in one environment, see their
comments in ``pyproject.toml``).

Included verbatim by ``docs/source/examples.rst`` and signature-checked by
``test/test_examples.py``.
"""

from pathlib import Path

from remote_handler.core import ClusterConfig
from remote_handler.core.file_handler import RemoteFileSystem
from remote_handler.workflows.trainers import AceModelTrainer

# See ace_config.yaml (next to this script) for a minimal config.
CONFIG_FILE = Path(__file__).parent / "ace_config.yaml"

if __name__ == "__main__":
    config = ClusterConfig.from_name("datalab")
    remote_fs = RemoteFileSystem(remote_alias="datalab")

    trainer = AceModelTrainer(
        remote_fs=remote_fs,
        cluster_config=config,
        config_file=CONFIG_FILE,
        num_models=3,
        validation_fraction=0.1,
    )

    trainer.update(Path("ace_run/0/training"), ["dft_labelled.extxyz"])
    model_paths = trainer.get_trained_model()
