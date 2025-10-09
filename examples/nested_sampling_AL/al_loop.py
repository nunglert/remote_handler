import logging
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import jax
from jaxnest import input_parser
from jaxnest.active_learning import ns_active_learning
from jaxnest.input_parser import update_default_args

import remote_handler
from remote_handler.core import active_learning_base
from remote_handler.core.file_handler import LocalFileSystem, RemoteFileSystem

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# Directory paths
data_dir = Path("/home/nico.unglert/code/jaxnest_dev/test/data")
homepath = Path(os.getcwd())

remote_dir = homepath / "remote_dirs" / "remote_dir"
os.makedirs(remote_dir.parent, exist_ok=True)

# Add custom template directory for SLURM scripts
remote_handler.core.slurm_utils.TEMPLATE_DIRS.append(
    str(data_dir / "active_learning/config_files")
)

# Instance local file systems (with SLURM configurations)
local_fs = LocalFileSystem(
    slurm_config_file=data_dir / "active_learning/config_files/datalab.json"
)

# Instance remote file system (with SLURM configurations)
remote_fs = RemoteFileSystem(
    slurm_config_file=data_dir / "active_learning/config_files/lugus.json"
)

# Settings for the active learning loop
cycle_entry_point = "evaluation"
initial_model_file = None
initial_sample_file = homepath / "exp_phases.extxyz"
initial_db_file = None

# initial_model_file = None
# initial_sample_file = "/home/nico.unglert/code/jaxnest_dev/test/data/active_learning/bartok_very_small_train.extxyz"
# initial_db_file = "/home/nico.unglert/code/jaxnest_dev/test/data/active_learning/bartok_very_small_train.extxyz"

stub = False
loop_time = 1

# Decide whether to use the stub executable or the real one
if stub:
    ns_exe = "/home/nico.unglert/code/jaxnest_dev/test/data/active_learning/ns_run_stub.py"
else:
    ns_exe = "/home/nico.unglert/code/jaxnest_dev/src/jaxnest/ns_run.py"

# Trial parse the ns.inp to avoid it crashing on the remote machine
n_gpu_parallel = 4
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
ns_traj_sampler = ns_active_learning.NSTrajectorySampler(
    ns_exe=ns_exe,
    remote_fs=local_fs,
    initial_input_file=homepath / "ns.inp",
    n_gpu_parallel=n_gpu_parallel,
    sample_size=100,
    remote_savepath=remote_dir,
    loop_time=loop_time,
)

# EVALUATOR
# VASP evaluator which runs single point DFT calculations on the remote machine
# to evaluate the energies of the collected samples
vasp_evaluator = ns_active_learning.VASPEvaluator(
    singlepoint_script=data_dir / "active_learning" / "dft_singlepoint.py",
    remote_fs=remote_fs,
    num_parallel=25,
    remote_savepath=remote_fs.slurm_config["savepath"],
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
)

neuralil_trainer = ns_active_learning.NeuralILTrainer(
    training_script=data_dir / "active_learning" / "train_nnff.py",
    remote_fs=local_fs,
    remote_savepath=remote_dir,
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
                "MC_cell_P": "3 7 11 15",
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
                "gpu_n_batch": 600,
                "MC_cell_P": "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16",
            }
        return kwargs


num_iterations = 20
# Instantiate the sampler schedule with the desired number of iterations
sampler_schedule = SamplerSchedule(num_iterations)


def kwargs_consumer(kwargs):
    ns_params = deepcopy(ns_traj_sampler.ns_params)
    update_default_args(kwargs, ns_params)

    input_parser.raw_parse(
        args_dict=ns_params, n_gpu_parallel=n_gpu_parallel, parse_only=True
    )


sampler_schedule.test(kwargs_consumer),
sampler_schedule.visualize()


al_handler = active_learning_base.ActiveLearningHandler(
    sampler=ns_traj_sampler,                 # Where to get new samples from (Nested sampling trajectories)
    evaluator=vasp_evaluator,                # How to evaluate the samples (DFT single point calculations)
    trainer=neuralil_trainer,                # How to train the model (NeuralIL)
    sampler_schedule=sampler_schedule,       # How to change the sampler parameters as the AL loop progresses
    calchome=homepath,                       # Home directory where the AL loop is run from
    initial_model_file=initial_model_file,   # Initial model file (None if starting from scratch)
    initial_db_file=initial_db_file,         # Initial database file (None if starting from scratch)
    initial_sample_file=initial_sample_file, # Initial sample file (extxyz with structures to add to the initial database)
    num_iterations=num_iterations,           # Number of active learning iterations
    cycle_entry_point=cycle_entry_point      # Entry point for the active learning cycle ("sampling" or "evaluation")
)

al_handler.run()
