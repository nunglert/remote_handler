"""
Torch-Sim parallel tempering sampling script.

Entry point for remote execution via SingleScriptTask. Reads a JSON input
file (path passed as sys.argv[1]), runs a parallel tempering simulation
using torch_sim with a MACE model, and writes the final replica structures
to ``out/samples.extxyz``.

Supports two modes:
  - ``nvt``: NVT parallel tempering with optional initial NPT equilibration
    (interface_pt_simulation). Requires frozen-atom constraints on the
    founder structure.
  - ``npt``: NPT parallel tempering (pt_simulation_npt).
"""

import sys
from pathlib import Path

INPUTFILE = Path(sys.argv[1])
calcpath = INPUTFILE.parent

from remote_handler.scripts.input_models import TorchSimSamplingInput

input_config = TorchSimSamplingInput.from_file(INPUTFILE)
input_config.validate()

import os

import ase.io
import torch

from remote_handler.scripts.torch_sim.simulation_pipelines import (
    interface_pt_simulation,
    pt_simulation_npt,
)

# Load founder structure
at = ase.io.read(calcpath / input_config.founder)

# Load MACE model(s)
if len(input_config.model_paths) > 1:
    # Ensemble mode — pass list of paths; MaceEnsembleModel handles loading
    mace = [calcpath / p for p in input_config.model_paths]
elif input_config.model_paths[0] == "foundation":
    from mace.calculators.foundations_models import mace_mp

    mace = mace_mp(model="small", return_raw_model=True)
else:
    mace = torch.load(
        calcpath / input_config.model_paths[0], map_location="cpu"
    )

# Resolve dtype
dtype_map = {"float32": torch.float32, "float64": torch.float64}
dtype = dtype_map[input_config.dtype]

# Run simulation
common_kwargs = dict(
    at=at,
    calcpath=calcpath,
    temperatures=input_config.temperatures,
    mace=mace,
    dt=input_config.dt,
    gamma=input_config.gamma,
    alpha=input_config.alpha,
    cell_alpha=input_config.cell_alpha,
    n_steps_npt=input_config.n_steps_npt,
    n_steps_pt=input_config.n_steps_pt,
    seed=input_config.seed,
    device=input_config.device,
    dtype=dtype,
)

if input_config.ensemble == "nvt":
    state = interface_pt_simulation(
        target_temperature=input_config.target_temperature, **common_kwargs
    )
else:
    state = pt_simulation_npt(**common_kwargs)

# Extract final structures from all replicas and write to output
out_dir = calcpath / "out"
os.makedirs(out_dir, exist_ok=True)

final_atoms = state.to_atoms()
if not isinstance(final_atoms, list):
    final_atoms = [final_atoms]

for i, at in enumerate(final_atoms):
    at.info["replica"] = i
    at.info["ensemble"] = input_config.ensemble

ase.io.write(out_dir / "samples.extxyz", final_atoms)
