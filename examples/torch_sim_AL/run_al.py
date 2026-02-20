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

import logging
import sys
from pathlib import Path

import ase.io

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import RemoteFileSystem
from remote_handler.scripts import input_models
from remote_handler.workflows.active_learning_base import ActiveLearningHandler
from remote_handler.workflows.evaluators import VASPEvaluator
from remote_handler.workflows.samplers import TorchSimSampler
from remote_handler.workflows.trainers import MaceFinetuningTrainer

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
    remote_alias="datalab", local_temp_path=_tmpdir
)  # GPU cluster for sampling + training
dft_fs = RemoteFileSystem(
    remote_alias="lugus", local_temp_path=_tmpdir
)  # CPU cluster for DFT

# Cluster configs
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
    """Create a bulk Si 2x2x2 supercell (64 atoms) as founder structure."""
    from ase.build import bulk
    from ase.io import write

    si_bulk = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)
    write(founder_path, si_bulk, format="vasp")
    return founder_path


founder_poscar = create_founder(inputs_dir / "POSCAR_si_bulk")

mace_template = config_dir / "mace_config.yaml"
foundation_model = inputs_dir / "2023-12-10-mace-128-L0_energy_epoch-249.model"

# ---------------------------------------------------------------------------
# 1. Sampler: TorchSimSampler (NPT parallel tempering)
# ---------------------------------------------------------------------------
torchsim_sampler = TorchSimSampler(
    remote_fs=gpu_fs,
    cluster_config=gpu_cluster_config,
    sampling_script=scripts_dir / "torch_sim" / "torch_sim_sampling.py",
    founder_path=founder_poscar,
    ensemble="npt",
    temperatures=[
        300.0,
        450.0,
        600.0,
        750.0,
        900.0,
        1050,
        1200,
        1350,
        1500,
        1650,
        1800,
        1950,
        2100,
    ],
    model_path="foundation",
    dt=0.001,
    n_steps_pt=100,
    seed=42,
    device="cuda",
    dtype="float64",
    remote_savepath=Path("/home/nico.unglert/dump/remote_handler"),
    loop_time=1,
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
        "kspacing": 0.3,
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
# 3. Trainer: MaceFinetuningTrainer
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

# ---------------------------------------------------------------------------
# Initial data collection: PT sampling with foundation model
# ---------------------------------------------------------------------------
initial_sample_file = calchome / "initial_samples.extxyz"
if os.path.exists(initial_sample_file):
    initial_atoms = ase.io.read(initial_sample_file, index=":")
else:
    torchsim_sampler.update(calchome / "initial_samples", [foundation_model])
    output_path = torchsim_sampler.get_sampled_configs()
    initial_atoms = ase.io.read(output_path, index=":")
    ase.io.write(initial_sample_file, initial_atoms)

# ---------------------------------------------------------------------------
# Active Learning Loop
# ---------------------------------------------------------------------------
al_handler = ActiveLearningHandler(
    sampler=torchsim_sampler,
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
