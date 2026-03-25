"""
Active learning workflow using torch_sim parallel tempering with MACE models.

This example demonstrates a three-step active learning loop:
  1. Initial sampling: TorchSimSampler runs NPT parallel tempering with the
                       MACE-MP foundation model to collect seed configurations
                       across a temperature ladder.
  2. Evaluation:       VASPEvaluator runs DFT single-point calculations on the
                       sampled structures to obtain ground-truth energies and forces.
  3. Training:         MaceFinetuningTrainer finetunes a committee of MACE models
                       on the accumulated DFT database.

After the initial data collection, the cycle repeats:
  sampling → evaluation → training → sampling → ...
with finetuned models replacing the foundation model for increasingly
accurate parallel tempering exploration.

Usage:
    Adjust the paths and parameters below to match your cluster setup,
    then run:
        python run_al.py
"""

import os

# Ensure writable TMPDIR (Slurm nodes may restrict /tmp)
_tmpdir = os.path.expanduser("~/tmp")
os.makedirs(_tmpdir, exist_ok=True)
os.environ["TMPDIR"] = _tmpdir

import logging
import sys
from pathlib import Path

import ase.io

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import RemoteFileSystem, LocalFileSystem
from remote_handler.scripts import input_models
from remote_handler.workflows.active_learning_base import ActiveLearningHandler
from remote_handler.workflows.evaluators import ORCAEvaluator
from remote_handler.workflows.samplers import SSPDSampler
from remote_handler.workflows.trainers import MaceModelTrainer
from remote_handler.workflows import active_learning_base


logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# ---------------------------------------------------------------------------
# Paths  (adjust to your setup)
# ---------------------------------------------------------------------------
calchome = Path(__file__).parent.resolve()
scripts_dir = Path(input_models.__file__).parent
config_dir = calchome / "config"
inputs_dir = calchome / "inputs"
inputs_dir.mkdir(parents=True, exist_ok=True)

# Cluster filesystems
gpu_fs = RemoteFileSystem(
    remote_alias="datalab", local_temp_path=_tmpdir, identityfile="/home/mketter/.ssh/datalab_key"
)  # GPU cluster for sampling + training
dft_fs = LocalFileSystem(
)  # CPU cluster for DFT

# Cluster configs
gpu_cluster_config_a40 = ClusterConfig(
    remote_machine="datalab",
    partition="GPU-a40",
    savepath="/home/michael.ketter/active_learning_SSPD/trainer",
    conda_env="remote_handler",
    conda_source="/home/michael.ketter/packages/miniconda3",
    template="datalab.sh.j2",
)

gpu_cluster_config_a100s = ClusterConfig(
    remote_machine="datalab",
    partition="GPU-a100s",
    savepath="/home/michael.ketter/active_learning_SSPD/sampler",
    conda_env="remote_handler",
    conda_source="/home/michael.ketter/packages/miniconda3",
    template="datalab.sh.j2",
)

dft_cluster_config = ClusterConfig(
    remote_machine="lugus",
    partition="normal",
    conda_env="remote_handler",
    conda_source="/home/mketter/packages/miniconda3",
    savepath="/home/mketter/active_learning_SSPD/evaluator",
    template="dirona.sh.j2",
)

# ---------------------------------------------------------------------------
# Inputs  (adjust to your system)
# ---------------------------------------------------------------------------
initial_database = calchome / "inputs" / "configs" / "InitialDatabase.extxyz"
mace_template = calchome / "config" / "templates" / "mace_config_l2.yaml.template"
initial_config = calchome / "inputs" / "configs" / "StartStruct.xyz"

# ---------------------------------------------------------------------------
# 1. Sampler: SSPD Sampler
# ---------------------------------------------------------------------------

SSPD_sampler = SSPDSampler(
    remote_fs=gpu_fs,
    cluster_config=gpu_cluster_config_a100s,
    data_path=initial_config,
    sampling_script=scripts_dir / "sspd" / "SSPD_sampling.py",
    remote_savepath=Path("/home/michael.ketter/active_learning_SSPD/sampler"),
    loop_time=30,
)

# ---------------------------------------------------------------------------
# 2. Evaluator: ORCAEvaluator
# ---------------------------------------------------------------------------

orca_evaluator = ORCAEvaluator(
    singlepoint_script=scripts_dir / "orca_singlepoint.py",
    remote_fs=dft_fs,
    cluster_config=dft_cluster_config,
    num_parallel=10,
    remote_savepath=Path("/home/mketter/active_learning_SSPD/evaluator"),
    orcasimpleinput="wB97X 6-31G(d) ENGRAD",
    n_processors=2,
    loop_time=30,
    required_fraction=1,
)


# ---------------------------------------------------------------------------
# 3. Trainer: MaceModelTrainer
# ---------------------------------------------------------------------------

mace_trainer = MaceModelTrainer(
    remote_fs=gpu_fs,
    cluster_config=gpu_cluster_config_a40,
    training_script=scripts_dir / "mace" / "mace_training.py",
    config_file=mace_template,
    remote_savepath=Path("/home/michael.ketter/active_learning_SSPD/trainer"),
    remote_mace_train_runner=Path(
        "/home/michael.ketter/packages/mace/mace/cli/run_train.py"
    ),
    loop_time=30,
    num_models=1,
)


# ---------------------------------------------------------------------------
# Active Learning Loop
# ---------------------------------------------------------------------------
num_iterations = 5

class SamplerSchedule(active_learning_base.Schedule):

    def __init__(self, num_iterations: int, starting_beta_inv: float):
        self.beta_inv = starting_beta_inv
        super().__init__(num_iterations)

    def get_kwargs(self, iteration_idx):
        kwargs = {
            "threshold": 0.99,
            "inv_beta": self.beta_inv + iteration_idx*0.05,
            "sampling_threshold": 0,
        }
        return kwargs

sampler_schedule = SamplerSchedule(num_iterations, 0.25)


al_handler = ActiveLearningHandler(
    sampler=SSPD_sampler,
    sampler_schedule=sampler_schedule,
    evaluator=orca_evaluator,
    trainer=mace_trainer,
    calchome=calchome,
    num_iterations=num_iterations,
    cycle_entry_point="evaluation",
    cycle_exit_point="training",
    initial_sampler_out=initial_database,
    initial_evaluator_out=None,
    initial_trainer_out=None,
    combine_evaluator_outputs=True,
    sampler_output_extension="extxyz",
    trainer_output_extension="model",)

al_handler.run()

