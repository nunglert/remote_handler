"""Example 4 -- A full active-learning loop.

Wire a sampler, a ground-truth evaluator and a trainer into the
``ActiveLearningHandler`` and run the Sampler -> Evaluator -> Trainer loop.
Each actor can target a different cluster.

Included verbatim by ``docs/source/examples.rst`` and signature-checked by
``test/test_examples.py``.
"""

from pathlib import Path

import remote_handler
from remote_handler.core import ClusterConfig
from remote_handler.core.file_handler import RemoteFileSystem
from remote_handler.workflows import VASPEvaluator
from remote_handler.workflows.active_learning_base import ActiveLearningHandler
from remote_handler.workflows.samplers import TorchSimSampler
from remote_handler.workflows.trainers import MaceModelTrainer

SCRIPTS = Path(remote_handler.__file__).parent / "scripts"

if __name__ == "__main__":
    # Each actor may run on its own machine.
    dft_fs = RemoteFileSystem(remote_alias="dirona")
    gpu_fs = RemoteFileSystem(remote_alias="datalab")
    dft_cfg = ClusterConfig.from_name("dirona")
    gpu_cfg = ClusterConfig.from_name("datalab")

    sampler = TorchSimSampler(
        remote_fs=gpu_fs,
        cluster_config=gpu_cfg,
        sampling_script=str(SCRIPTS / "torch_sim" / "torch_sim_sampling.py"),
        founder_path="inputs/founder.extxyz",
        ensemble="nvt",
        temperatures=[300.0, 450.0, 600.0],
        target_temperature=300.0,
    )
    evaluator = VASPEvaluator(
        singlepoint_script=str(SCRIPTS / "dft_singlepoint.py"),
        remote_fs=dft_fs,
        cluster_config=dft_cfg,
        num_parallel=25,
        incar_kwargs={"xc": "PBE", "encut": 400, "ediff": 1e-6},
    )
    trainer = MaceModelTrainer(
        remote_fs=gpu_fs,
        cluster_config=gpu_cfg,
        training_script=str(SCRIPTS / "mace" / "mace_training.py"),
        config_file="config/mace_config.yaml",
        remote_mace_train_runner=(
            "/home/me/miniconda3/envs/mace/lib/python3.10/"
            "site-packages/mace/cli/run_train.py"
        ),
    )

    al = ActiveLearningHandler(
        sampler=sampler,
        evaluator=evaluator,
        trainer=trainer,
        calchome=Path("al_run"),
        num_iterations=5,
        cycle_entry_point="sampler",
        cycle_exit_point="training",
        initial_trainer_out=["seed_model.model"],
        trainer_output_extension="model",
    )
    al.run()
