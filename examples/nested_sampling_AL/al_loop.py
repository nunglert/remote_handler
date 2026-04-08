import logging
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import jax
from jaxnest import input_parser
from remote_handler.workflows import evaluators, samplers, trainers
from jaxnest.input_parser import update_default_args

import remote_handler
from remote_handler.workflows import active_learning_base
from remote_handler.core.file_handler import LocalFileSystem, RemoteFileSystem
from remote_handler.core.cluster_handler import ClusterConfig

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# Directory paths
data_dir = Path("/home/aserafini/code/remote_handler/src/remote_handler/scripts")
homepath = Path(os.getcwd())


# Add custom template directory for SLURM scripts
CONFIG_DIR = Path(homepath / "config_files")

os.environ["REMOTE_HANDLER_CONFIG_DIR"] = str(CONFIG_DIR)
os.environ["REMOTE_HANDLER_SLURM_TEMPLATES"] = str(CONFIG_DIR)

# Instance local file systems (with SLURM configurations)
local_fs = LocalFileSystem()

# Instance remote file system (with SLURM configurations)
remote_fs = RemoteFileSystem(
    remote_alias="datalab",
)

local_config_dict = {
    "remote_machine": "dirona",
    "partition": "normal",
    "savepath": "/home/aserafini/remote_jobs/al/test/",
    "conda_env": "jaxnest",
    "conda_source": "/home/aserafini/miniconda3",
    "template": "dirona.sh.j2"
}

local_cluster_config = ClusterConfig.from_dict(local_config_dict)

remote_config_dict = {
    "remote_machine": "datalab",
    "partition": "GPU-a100,GPU-a40,GPU-a100s",
    "savepath": "/home/alessandro.serafini/remote_jobs/al/test/",
    "conda_env": "jaxnest",
    "conda_source": "/home/alessandro.serafini/miniconda3",
    "template": "datalab.sh.j2"
}

remote_cluster_config = ClusterConfig.from_dict(remote_config_dict)

# Settings for the active learning loop
cycle_entry_point = "evaluation"
initial_model_file = None
initial_sample_file = homepath / "bartok_very_small_train.extxyz"
initial_db_file = None


stub = False
loop_time = 10

# Decide whether to use the stub executable or the real one
if stub:
    ns_exe = "/home/aserafini/code/jaxns/test/data/active_learning/ns_run_stub.py"
else:
    ns_exe = "/home/aserafini/code/jaxns/src/jaxnest/ns_run.py"

# Trial parse the ns.inp to avoid it crashing on the remote machine
n_gpu_parallel = 1
temp = input_parser.raw_parse(
    filename=homepath / "ns.inp",
    n_gpu_parallel=n_gpu_parallel,
    parse_only=True,
)

#################################################
# Set up the active learning components
#################################################

# SAMPLER

# Nested sampling trajectory sampler which runs jaxnest.ns_run.py on the remote machine
# and collects samples from the nested sampling trajectories for the active learning loop
ns_traj_sampler = samplers.NSTrajectorySampler(
    remote_fs=remote_fs,
    cluster_config=remote_cluster_config,
    sampling_script=ns_exe,
    initial_input_file=homepath / "ns.inp",
    n_gpu_parallel=n_gpu_parallel,
    sample_size=100,
    loop_time=loop_time,
    remote_savepath=remote_config_dict["savepath"]
)

# EVALUATOR

# VASP evaluator which runs single point DFT calculations on the remote machine
# to evaluate the energies of the collected samples
vasp_evaluator = evaluators.VASPEvaluator(
    singlepoint_script=data_dir / "dft_singlepoint.py",
    remote_fs=local_fs,
    cluster_config=local_cluster_config,
    num_parallel=25,
    incar_kwargs={
        # "istart": 0,
        # "icharg": 2,
        "xc": "R2SCAN",
        "encut": 300,
        "prec": "Normal",
        "ismear": -1,
        "sigma": 0.025,
        "ibrion": -1,
        "ediff": 1e-5,
        # "isym": 0,
        "symprec": 1e-7,
        "lwave": False,
        "lcharg": False,
        "lreal": False,
        # "kspacing": 2.0
        "kspacing": 0.3,
    },
    stub=stub,
    loop_time=loop_time,
    remote_savepath=local_config_dict["savepath"]
)

# TRAINER

# NeuralIL trainer which runs training of a NeuralIL model on the remote machine
# using the structures evaluated with DFT in the active learning loop
neuralil_trainer = trainers.NeuralilILTrainer(
    training_script=data_dir / "neuralil" / "neuralil_training.py",
    remote_fs=remote_fs,
    cluster_config=remote_cluster_config,   
    training_kwargs={
        "r_cut": 4.0,
        "training_fraction": 0.75,
        "supercell_diag": [3, 3, 3],
        "high_energy_penalty": -5.0,
        "n_ensemble": 30,
        "save_reduced": 5,
        "n_batch": 4,
    },
    stub=stub,
    loop_time=loop_time,
    remote_savepath=remote_config_dict["savepath"]
)

# SAMPLER SCHEDULE

# Schedule for the nested sampling trajectory sampler which changes the
# number of walkers and the MC_cell_P parameter in the ns.inp file
# as the active learning loop progresses
class SamplerSchedule(active_learning_base.Schedule):
    def get_kwargs(self, iteration_idx):
        if iteration_idx == 0:
            kwargs = {
                "n_walkers": 400,
                "gpu_n_batch": 200,
                "MC_cell_P": "2 4 6 8 10 12 14 16",
            }
        if iteration_idx > 0:
            kwargs = {
                "n_walkers": 400,
                "gpu_n_batch": 400,
                "MC_cell_P": "2 4 6 8 10 12 14 16",
            }
        if iteration_idx > 3:
            kwargs = {
                "n_walkers": 200,
                "gpu_n_batch": 400,
                "MC_cell_P": "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16",
            }
        return kwargs


num_iterations = 20
# Instantiate the sampler schedule with the desired number of iterations
sampler_schedule = SamplerSchedule(num_iterations)

# Function to update the ns.inp file with the new parameters from the schedule
if not stub:
    def kwargs_consumer(kwargs):
        ns_params = deepcopy(ns_traj_sampler.ns_params)
        update_default_args(kwargs, ns_params)

        input_parser.raw_parse(
            args_dict=ns_params, n_gpu_parallel=n_gpu_parallel, parse_only=True
        )

    # Test the kwargs consumer function and visualize the schedule
    sampler_schedule.test(kwargs_consumer)
    sampler_schedule.visualize()


al_handler = active_learning_base.ActiveLearningHandler(
    sampler=ns_traj_sampler,                 # Where to get new samples from (Nested sampling trajectories)
    evaluator=vasp_evaluator,                # How to evaluate the samples (DFT single point calculations)
    trainer=neuralil_trainer,                # How to train the model (NeuralIL)
    sampler_schedule=sampler_schedule,       # How to change the sampler parameters as the AL loop progresses
    calchome=homepath,                       # Home directory where the AL loop is run from
    initial_trainer_out=initial_model_file,  # Initial model file (None if starting from scratch)
    initial_sampler_out=initial_sample_file, # Initial database file (None if starting from scratch)
    initial_evaluator_out=initial_db_file,   # Initial sample file (extxyz with structures to add to the initial database)
    num_iterations=num_iterations,           # Number of active learning iterations
    cycle_entry_point=cycle_entry_point      # Entry point for the active learning cycle ("sampling" or "evaluation")
)

al_handler.run()
