"""Example 7 -- Training a GRACE force field.

Configure a ``GraceModelTrainer`` -- the same ensemble-array actor shape as
``MaceModelTrainer`` (Example 2), fitting a GRACE potential with
``gracemaker`` instead of MACE.

``gracemaker`` must be installed in the *remote* conda env this trainer
targets (see the ``grace`` extra for the pieces needed to test this locally
-- it cannot coexist with the ``ace`` extra in one environment, see their
comments in ``pyproject.toml``).

Included verbatim by ``docs/source/examples.rst`` and signature-checked by
``test/test_examples.py``.
"""

from pathlib import Path

from remote_handler.core import ClusterConfig
from remote_handler.core.file_handler import RemoteFileSystem
from remote_handler.workflows.trainers import GraceModelTrainer

# See grace_config.yaml (next to this script) for a minimal config.
CONFIG_FILE = Path(__file__).parent / "grace_config.yaml"

if __name__ == "__main__":
    config = ClusterConfig.from_name("datalab")
    remote_fs = RemoteFileSystem(remote_alias="datalab")

    trainer = GraceModelTrainer(
        remote_fs=remote_fs,
        cluster_config=config,
        config_file=CONFIG_FILE,
        num_models=3,
        validation_fraction=0.1,
    )

    trainer.update(Path("grace_run/0/training"), ["dft_labelled.extxyz"])
    model_paths = trainer.get_trained_model()
