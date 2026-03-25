"""
Active learning workflow using CMA-ES evolution sampling with MACE models.

This example demonstrates a three-step active learning loop:
  1. Sampling:   BatchCMAESSampler runs N independent CMA-ES evolutions on a
                 GPU cluster, each exploring configuration space guided by MACE
                 energy evaluation. The winner structure from each evolution is
                 returned (optionally after gradient-based optimization).
  2. Evaluation: VASPEvaluator runs DFT single-point calculations on the
                 sampled structures to obtain ground-truth energies and forces.
  3. Training:   MaceModelTrainer retrains a committee of MACE models on the
                 accumulated DFT database.

The cycle repeats, with improved models guiding increasingly accurate CMA-ES
exploration of configuration space.

Usage:
    Adjust the paths and parameters below to match your cluster setup,
    then run:
        python run_al.py
"""

import logging
import os
import sys
from pathlib import Path

import ase.io

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import RemoteFileSystem
from remote_handler.scripts import input_models
from remote_handler.workflows.active_learning_base import ActiveLearningHandler
from remote_handler.workflows.evaluators import VASPEvaluator
from remote_handler.workflows.samplers import BatchCMAESSampler
from remote_handler.workflows.trainers import MaceFinetuningTrainer

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# ---------------------------------------------------------------------------
# Paths  (adjust to your setup)
# ---------------------------------------------------------------------------
# example_dir = Path(__file__).parent.resolve()
calchome = Path(
    "/home/nico.unglert/code/remote_handler/experiments/clinamen/run0/"
)
scripts_dir = Path(input_models.__file__).parent
config_dir = calchome / "config"
inputs_dir = calchome / "inputs"

# Cluster filesystems
gpu_fs = RemoteFileSystem(
    remote_alias="datalab"
)  # GPU cluster for sampling + training
dft_fs = RemoteFileSystem(remote_alias="lugus")  # CPU cluster for DFT

# Cluster configs (loaded from REMOTE_HANDLER_CONFIG_DIR)
gpu_cluster_config = ClusterConfig(
    remote_machine="datalab",
    partition="GPU-a40",
    savepath="/home/nunglert/dump/remote_handler",
    conda_env="clinamen",
    conda_source="/home/nico.unglert/miniconda3",
    template="datalab.sh.j2",
)

dft_cluster_config = ClusterConfig(
    remote_machine="lugus",
    partition="normal",
    conda_env="jaxnest",
    conda_source="/home/nunglert/miniconda3",
    savepath="/home/nunglert/dump/remote_handler",
    template="lugus.sh.j2",
)


# ---------------------------------------------------------------------------
# Inputs  (adjust to your system)
# ---------------------------------------------------------------------------
def create_founder(founder_path):
    from ase.build import bulk
    from ase.io import write

    si_bulk = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 1, 1)
    del si_bulk[-1]
    write(founder_path, si_bulk, format="vasp")
    return founder_path


founder_poscar = create_founder(inputs_dir / "POSCAR_si_vacancy")

# initial_db = example_dir / "inputs" / "initial_database.extxyz"
# initial_models = [
#     example_dir / "inputs" / "models" / f"model_{i}.model"
#     for i in range(5)
# ]
mace_template = config_dir / "mace_config.yaml"
foundation_model = inputs_dir / "2023-12-10-mace-128-L0_energy_epoch-249.model"

# Working directory for AL iterations
# calchome = example_dir / "al_runs"

# ---------------------------------------------------------------------------
# 1. Sampler: BatchCMAESSampler
# ---------------------------------------------------------------------------
cmaes_sampler = BatchCMAESSampler(
    batch_size=20,
    remote_fs=gpu_fs,
    cluster_config=gpu_cluster_config,
    sampling_script=scripts_dir / "clinamen2" / "cmaes_sampling.py",
    founder_path=founder_poscar,
    label="cmaes_evo",
    generations=5,
    step_size=0.12,
    random_seed=42,
    save_nth=2,
    batch=False,
    loss_mode="energy-only",
    remote_savepath=Path("/home/nico.unglert/dump/remote_handler"),
    loop_time=1,
    # Post-evolution optimization (optional)
    do_optimization=True,
    dof_mode="defect",
)

# ---------------------------------------------------------------------------
# 2. Evaluator: VASPEvaluator
# ---------------------------------------------------------------------------
vasp_evaluator = VASPEvaluator(
    singlepoint_script=scripts_dir / "dft_singlepoint.py",
    remote_fs=dft_fs,
    cluster_config=dft_cluster_config,
    num_parallel=25,
    n_processors=8,
    remote_savepath=Path("/home/nunglert/dump/dft"),
    incar_kwargs={
        "xc": "pbe",
        "encut": 400,
        "ediff": 1e-5,
        "kpts": [3, 3, 1],
        "gamma": True,
        "ismear": 0,
        "sigma": 0.05,
        "lwave": False,
        "lcharg": False,
        "lreal": False,
        "ncore": 8,
    },
    loop_time=1,
)

# ---------------------------------------------------------------------------
# 3. Trainer: MaceModelTrainer
# ---------------------------------------------------------------------------
mace_trainer = MaceFinetuningTrainer(
    remote_fs=gpu_fs,
    cluster_config=gpu_cluster_config,
    training_script=scripts_dir / "mace" / "mace_finetuning.py",
    config_file=mace_template,
    remote_mace_train_runner=Path(
        "/home/nico.unglert/code/mace-freeze/scripts/run_train.py"
    ),
    foundation_model=foundation_model,
    num_models=2,
    remote_savepath=Path("/home/nico.unglert/dump/remote_handler"),
    loop_time=1,
)

initial_sample_file = calchome / "initial_samples.extxyz"
if os.path.exists(initial_sample_file):
    initial_atoms = ase.io.read(initial_sample_file, index=":")
else:
    cmaes_sampler.update("initial_samples", [foundation_model])
    initial_atoms_paths = cmaes_sampler.get_sampled_configs()
    initial_atoms = []
    for p in initial_atoms_paths:
        initial_atoms.extend(ase.io.read(p, index=":"))

    ase.io.write(initial_sample_file, initial_atoms)


# ---------------------------------------------------------------------------
# Active Learning Loop
# ---------------------------------------------------------------------------
al_handler = ActiveLearningHandler(
    sampler=cmaes_sampler,
    evaluator=vasp_evaluator,
    trainer=mace_trainer,
    calchome=calchome,
    num_iterations=10,
    cycle_entry_point="evaluation",
    cycle_exit_point="training",
    initial_sampler_out=initial_sample_file,
    initial_evaluator_out=None,
    initial_trainer_out=None,
    combine_evaluator_outputs=True,
    sampler_output_extension="extxyz",
    trainer_output_extension="model",
)

al_handler.run()
