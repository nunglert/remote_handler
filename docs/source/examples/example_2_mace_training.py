"""Example 2 -- Training a MACE force field.

Configure a ``MaceModelTrainer`` and run one training step. In a full run the
``ActiveLearningHandler`` (Example 4) calls ``update`` / ``get_trained_model``
for you each iteration; here we drive a single step directly.

Included verbatim by ``docs/source/examples.rst`` and signature-checked by
``test/test_examples.py``.
"""

from pathlib import Path

import remote_handler
from remote_handler.core import ClusterConfig
from remote_handler.core.file_handler import RemoteFileSystem
from remote_handler.workflows.trainers import MaceModelTrainer

SCRIPTS = Path(remote_handler.__file__).parent / "scripts"

if __name__ == "__main__":
    config = ClusterConfig.from_name("datalab")
    remote_fs = RemoteFileSystem(remote_alias="datalab")

    trainer = MaceModelTrainer(
        remote_fs=remote_fs,
        cluster_config=config,
        training_script=str(SCRIPTS / "mace" / "mace_training.py"),
        config_file="config/mace_config.yaml",
        # MACE's run_train.py inside the remote conda environment:
        remote_mace_train_runner=(
            "/home/me/miniconda3/envs/mace/lib/python3.10/"
            "site-packages/mace/cli/run_train.py"
        ),
        num_models=3,
        validation_fraction=0.1,
    )

    # Train on DFT-labelled configs; calcdir is this iteration's work dir.
    trainer.update(Path("mace_run/0/training"), ["dft_labelled.extxyz"])
    model_paths = trainer.get_trained_model()
