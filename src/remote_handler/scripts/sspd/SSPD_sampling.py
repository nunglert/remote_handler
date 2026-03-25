import json
import logging
import os
import pathlib
import subprocess
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional
import numpy as np

import ase
import jinja2
import numpy as np
import torch
from SSPD.SSPD import SSPD
import ase.io
import jax.numpy as jnp

import jax
jax.config.update("jax_platform_name", "cpu")
from SSPD.calculators.calculator import MACEModel


calcpath = Path(sys.argv[1]).parent
INPUTFILE = Path(sys.argv[1])
input_args = json.load(open(INPUTFILE, "r"))

outpath = calcpath / "out"
os.makedirs(outpath, exist_ok=False)

samples_file = outpath / "samples.xyz"


def get_SSPD_simulator(input_args):
    model_path = calcpath / input_args["model_file"]
    input_struct_path = calcpath / input_args["input_struct_path"]
    N = input_args["N"]

    input_struct = ase.io.read(input_struct_path)

    model = MACEModel(model_path, input_struct, N)

    positions = jnp.array(input_struct.get_positions()).ravel()
    natoms = len(input_struct)

    masses = jnp.array(input_struct.get_masses()).ravel()
    masses = jnp.repeat(masses, 3)
    masses = jnp.reshape(masses, jnp.shape(positions))

    simulator = SSPD(positions, 3*natoms, N, model.EvaluateEnergy, model.EvaluateEnergyGradient, model.EvaluateEnergyHessian, input_args["seed"], input_args["threshold"], input_args["inv_beta"], masses, max_acceptance=input_args["max_acceptance"], min_acceptance=input_args["min_acceptance"])

    return simulator

def run_SSPD(simulator, input_args):
    extracted_configs = []
    input_struct_path = calcpath / input_args["input_struct_path"]
    input_struct = ase.io.read(input_struct_path)

    np.random.seed(input_args["seed"])
    dt = input_args["initial_timestep"]
    for i in range(input_args["num_iterations"]):
        print("Currently at iteration " + str(i))
        dt, ratio = simulator.PerformIteration(dt)
        if (simulator.resampling >= input_args["sampling_threshold"]) and (i % input_args["sampling_frequency"] == 0):
            positions = np.array(simulator.X)
            chosen_pos = positions[np.random.randint(0, input_args["N"])]
            struct = input_struct.copy()
            struct.positions = chosen_pos.reshape((-1, 3))
            extracted_configs.append(struct)
    print("Extracted " + str(len(extracted_configs)) + " structures for adding to the database.")
    return extracted_configs



simulator = get_SSPD_simulator(input_args)

database = run_SSPD(simulator, input_args)

ase.io.write(samples_file, database, format="extxyz")
