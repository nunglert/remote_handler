"""Example 5 -- Training a NeuralIL model.

Configure a ``NeuralilILTrainer`` and run one training step -- the same
actor shape as Example 2 (MACE), for the JAXNEST-based NeuralIL model.
Hyperparameters layer built-in defaults < ``config_file`` < ``training_kwargs``.

The NeuralIL/JAXNEST stack must be installed in the *remote* conda env this
trainer targets (see the ``neuralil`` extra for the pieces needed to test
this locally).

Included verbatim by ``docs/source/examples.rst`` and signature-checked by
``test/test_examples.py``.
"""

from pathlib import Path

from remote_handler.core import ClusterConfig
from remote_handler.core.file_handler import RemoteFileSystem
from remote_handler.workflows.trainers import NeuralilILTrainer

if __name__ == "__main__":
    config = ClusterConfig.from_name("datalab")
    remote_fs = RemoteFileSystem(remote_alias="datalab")

    trainer = NeuralilILTrainer(
        remote_fs=remote_fs,
        cluster_config=config,
        training_kwargs={"n_epochs": 200, "n_ensemble": 10},
    )

    # Train on DFT-labelled configs; calcdir is this iteration's work dir.
    trainer.update(Path("neuralil_run/0/training"), ["dft_labelled.extxyz"])
    model_paths = trainer.get_trained_model()
