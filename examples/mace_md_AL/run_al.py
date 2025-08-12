import logging
import os
import sys
from pathlib import Path

import cu_co_active_learning
from ase.atoms import Atoms
from ase.io import read, write

import remote_handler.core.slurm_utils
from remote_handler.core.file_handler import RemoteFileSystem

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

data_dir = Path(
    "/home/ffitsch/Documents/university/bsc_thesis/dirona-repo/active_learning/remote_handler"
)
homepath = Path(os.getcwd())

remote_handler.core.slurm_utils.TEMPLATE_DIRS.append(
    str(data_dir / "config/servers")
)

dirona_fs = RemoteFileSystem(slurm_config_file="dirona.json")

datalab_fs = RemoteFileSystem(slurm_config_file="datalab.json")

vasp_evaluator = cu_co_active_learning.VASPEvaluator(
    singlepoint_script=data_dir / "dft_singlepoint.py",
    remote_fs=dirona_fs,
    num_parallel=25,
    remote_savepath=Path("/home/ffitsch/dump"),
    incar_kwargs={
        "ncore": 1,
        "algo": "Normal",
        # "prec": "Accurate",
        "xc": "pbe",
        "gga": "RP",
        # "setups": {"H": "_h", "O": "_h"},
        "encut": 300,
        # "encut": 850,
        "ediff": 1e-6,
        "kpts": [11, 11, 11],
        "gamma": True,
        "ismear": 0,
        "sigma": 0.05,
        # "ispin": 2,
        # "ivdw": 11,
        "lwave": False,
        "lcharg": False,
        "lreal": False,
    },
    n_processors=2,
    stub=False,
    loop_time=1,
    required_fraction=1,
)

mace_trainer = cu_co_active_learning.MaceModelTrainer(
    remote_fs=datalab_fs,
    training_script=data_dir / "mace_training.py",
    template_path=data_dir / "config/templates/mace_config_l2.yaml.template",
    remote_savepath=Path("/home/e12218283/dump/dump"),
)

lammps_sampler = cu_co_active_learning.LammpsSampler(
    remote_fs=datalab_fs,
    sampling_script=data_dir / "lammps_sampling.py",
    template_path=data_dir / "config/templates/lammps.in.template",
    data_path=data_dir / "4-4-6_roughened_hydroxide.data",
    remote_savepath=Path("/home/e12218283/dump/dump"),
)


model_files = [
    data_dir / "models/71_v7_Model_stagetwo.model",
    data_dir / "models/72_v7_Model_stagetwo.model",
    data_dir / "models/73_v7_Model_stagetwo.model",
    data_dir / "models/74_v7_Model_stagetwo.model",
    data_dir / "models/75_v7_Model_stagetwo.model",
]

lammps_sampler.update(data_dir / "test", model_files)
lammps_sampler.get_sampled_configs()


vasp_evaluator.update(data_dir / "test", data_dir / "prim.xyz")
dft_evaluated_samples = vasp_evaluator.get_evaluated_configs()

for s in dft_evaluated_samples:
    s.info["REF_energy"] = s.get_potential_energy()
    s.arrays["REF_forces"] = s.get_forces()
    s.calc = None

db = read("al_test_db.extxyz", index=":")
if isinstance(db, list):
    db.extend(dft_evaluated_samples)
elif isinstance(db, Atoms):
    db = [db, dft_evaluated_samples]
write("al_test_db.extxyz", db, format="extxyz")

mace_trainer.update(
    calcdir=data_dir / "test", db_files=[data_dir / "al_test_db.extxyz"]
)
mace_trainer.get_trained_model()
