import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import ase
import ase.io
import jax
import jax.numpy as jnp
import numpy as np
from SSPD.SSPD import SSPD

jax.config.update("jax_platform_name", "cpu")
import h5py
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

    model = MACEModel(
        model_path,
        input_struct,
        N,
        hessian_method=input_args["hessian_method"],
    )

    positions = jnp.array(input_struct.get_positions()).ravel()
    natoms = len(input_struct)

    masses = jnp.array(input_struct.get_masses()).ravel()
    masses = jnp.repeat(masses, 3)
    masses = jnp.reshape(masses, jnp.shape(positions))

    simulator = SSPD(
        positions,
        3 * natoms,
        N,
        model.EvaluateEnergy,
        model.EvaluateEnergyGradient,
        model.EvaluateEnergyHessian,
        input_args["seed"],
        input_args["threshold"],
        input_args["inv_beta"],
        masses,
        max_acceptance=input_args["max_acceptance"],
        min_acceptance=input_args["min_acceptance"],
    )

    return simulator, model


def run_SSPD(simulator, input_args, model):
    extracted_configs = []
    input_struct_path = calcpath / input_args["input_struct_path"]
    input_struct = ase.io.read(input_struct_path)
    num_iter = input_args["num_iterations"]

    ### FOR OUTPUT START

    # hardcoded settings start TODO
    E_max = None
    full_uncertainty = False
    track_evalues = False
    track_vhp_evaluations = False
    output_batch = 10
    N = input_args["N"]
    output = "SSPD.hdf5"
    natoms = len(input_struct)
    # hardcoded settings end

    iter_positions = []
    weights = []
    factors = []
    acceptance_ratio = []
    timesteps = []
    energies = []
    output_keys = [
        "positions",
        "weights",
        "factors",
        "acceptance_ratio",
        "timesteps",
        "energies",
    ]
    if track_evalues:
        eigenvalues = []
        output_keys.append("eigenvalues")
    if track_vhp_evaluations:
        vhp_evaluations = []
        output_keys.append("vhp_evaluations")
    if full_uncertainty:
        force_uncertainty = []
        energy_uncertainty = []
        output_keys.append("force_uncertainty")
        output_keys.append("energy_uncertainty")

    ### FOR OUTPUT END

    np.random.seed(input_args["seed"])
    dt = input_args["initial_timestep"]
    E_init = None
    t = -10
    written_file = False
    for i in range(num_iter):
        print("Currently at iteration " + str(i))
        dt, ratio = simulator.PerformIteration(dt)

        #### PUT THIS IN SEPARATE PLACE AND IMPROVE UPON IT
        # JUST FOR TESTING

        iter_positions.append(simulator.X)
        weights.append(simulator.weights)
        factors.append(simulator.factor)
        energies.append(simulator.energies)
        if E_init is None:
            E_init = simulator.energies[0]
        acceptance_ratio.append(ratio)
        timesteps.append(dt)
        if track_evalues:
            eigenvalues.append(simulator.eigvals)
        if track_vhp_evaluations:
            vhp_evaluations.append(simulator.vhp_evaluations / N)
        if full_uncertainty:
            energy_uncertainty.append(model.uncertainties["energies"])
            force_uncertainty.append(model.uncertainties["forces"])
        # if threshold is reached for any structure in the set, abort the run and output the current batch
        if E_max is not None:
            if np.sum((energies[-1] - E_init) > E_max * natoms) > 0:
                t = num_iter - 1
        if len(iter_positions) % output_batch == 0 or t == (num_iter - 1):
            if not written_file:
                with h5py.File(str(output) + ".hdf5", "w") as f:
                    f.create_dataset(
                        "positions",
                        data=iter_positions,
                        maxshape=(None, N, natoms * 3),
                    )
                    f.create_dataset(
                        "weights", data=weights, maxshape=(None, N)
                    )
                    f.create_dataset(
                        "energies", data=energies, maxshape=(None, N)
                    )
                    f.create_dataset("factors", data=factors, maxshape=(None,))
                    f.create_dataset(
                        "acceptance_ratio",
                        data=acceptance_ratio,
                        maxshape=(None,),
                    )
                    f.create_dataset(
                        "timesteps", data=timesteps, maxshape=(None,)
                    )
                    if track_evalues:
                        f.create_dataset(
                            "eigenvalues",
                            data=eigenvalues,
                            maxshape=(None, N, natoms * 3),
                        )
                    if track_vhp_evaluations:
                        f.create_dataset(
                            "vhp_evaluations",
                            data=vhp_evaluations,
                            maxshape=(None,),
                        )
                    if full_uncertainty:
                        f.create_dataset(
                            "force_uncertainty",
                            data=force_uncertainty,
                            maxshape=(None, N, natoms * 3),
                        )
                        f.create_dataset(
                            "energy_uncertainty",
                            data=energy_uncertainty,
                            maxshape=(None, N),
                        )
                written_file = True
            else:
                with h5py.File(str(output) + ".hdf5", "a") as f:
                    num_add = len(iter_positions)
                    for key in output_keys:
                        f[key].resize((f[key].shape[0] + num_add), axis=0)
                    f["positions"][-num_add:] = np.array(iter_positions)
                    f["weights"][-num_add:] = np.array(weights)
                    f["factors"][-num_add:] = np.array(factors)
                    f["acceptance_ratio"][-num_add:] = np.array(
                        acceptance_ratio
                    )
                    f["timesteps"][-num_add:] = np.array(timesteps)
                    f["energies"][-num_add:] = np.array(energies)
                    if track_evalues:
                        f["eigenvalues"][-num_add:] = np.array(eigenvalues)
                    if track_vhp_evaluations:
                        f["vhp_evaluations"][-output_batch:] = np.array(
                            vhp_evaluations
                        )
                    if full_uncertainty:
                        f["force_uncertainty"][-num_add:] = np.array(
                            force_uncertainty
                        )
                        f["energy_uncertainty"][-num_add:] = np.array(
                            energy_uncertainty
                        )
            if E_max is not None:
                if np.sum((energies[-1] - E_init) > E_max * natoms) > 0:
                    break
            iter_positions = []
            weights = []
            factors = []
            acceptance_ratio = []
            timesteps = []
            energies = []
            energy_uncertainty = []
            if track_evalues:
                eigenvalues = []
            if track_vhp_evaluations:
                vhp_evaluations = []
            if full_uncertainty:
                force_uncertainty = []

        #### CODE TO CHANGE END
        if (simulator.resampling >= input_args["sampling_threshold"]) and (
            i % input_args["sampling_frequency"] == 0
        ):
            positions = np.array(simulator.X)
            chosen_pos = positions[np.random.randint(0, input_args["N"])]
            struct = input_struct.copy()
            struct.positions = chosen_pos.reshape((-1, 3))
            extracted_configs.append(struct)
    print(
        "Extracted "
        + str(len(extracted_configs))
        + " structures for adding to the database."
    )
    return extracted_configs


simulator, model = get_SSPD_simulator(input_args)

database = run_SSPD(simulator, input_args, model)

ase.io.write(samples_file, database, format="extxyz")

# Success marker checked by SingleScriptTask.finish_check on the host.
(calcpath / "is_finished.info").write_text("SSPD sampling finished")
