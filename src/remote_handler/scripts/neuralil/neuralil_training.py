import json
import os
import sys
from pathlib import Path

import ase.db
import ase.io
import jax.random
import matplotlib.pyplot as plt
import natsort
import numpy as np

from remote_handler.scripts.neuralil.train_model import (
    DataProcessor,
    ModelSetup,
    Training,
    draw_urandom_int32,
    evaluate_data_processor,
)

inputfile = Path(sys.argv[1])
input_args = json.load(open(inputfile, "r"))

# Load the input arguments
# List of structure databases to use for training
structure_databases = natsort.natsorted(input_args["structure_databases"])
# Cutoff radius for neighbor list
r_cut = input_args["r_cut"]
# Fraction of data to use for training (rest is validation)
training_fraction = input_args["training_fraction"]
# Supercell diagonal for data augmentation
supercell_diag = input_args["supercell_diag"]
# Filepath to save outputs
filepath = input_args["filepath"]
# Pickle file name to save data processor and model
pickle_file_name = input_args["pickle_file_name"]

# Number of structures per batch
try:
    n_batch = input_args["n_batch"]
except:
    n_batch = 32

# Set up the random seed
run_seed = input_args.get("seed")
if run_seed is None:
    run_seed = draw_urandom_int32()

run_seed_str = f"{run_seed:0X}"
instance_code = f"VELO_{run_seed_str}"
rng = jax.random.PRNGKey(run_seed)

# Set up variables for the stub mode
if input_args["stub"]:
    dummy_files = [
        f"error_{instance_code}.txt",
        f"train{run_seed_str}.npz",
        f"val{run_seed_str}.npz",
        f"data_{instance_code}.pickle",
        "model.pkl",
        "full_model.pkl",
    ]
    for filename in dummy_files:
        with open(Path(input_args["filepath"]) / filename, "w") as f:
            f.write("keklelsos")
else:
    data_key, model_key, train_key = jax.random.split(rng, num=3)
    data_processor = DataProcessor(
        data_key, r_cut, training_fraction, supercell_diag, filepath=filepath
    )

    # Collect all atoms from the databases
    all_atoms_pre = []
    for database in structure_databases[:-1]:
        if ".extxyz" in database:
            atoms = ase.io.read(database, index=":")

        elif ".db" in database:
            atoms = []
            with ase.db.connect(database) as db:
                for row in db.select():
                    atoms.append(row.toatoms())
        else:
            raise Exception("Unknown database format!")

        all_atoms_pre.extend(atoms)

    all_atoms = []

    # Filter out energies that are above 10 eV/atom from the minimum energy
    if len(all_atoms_pre) > 0:
        # Divide by number of atoms to get energy per atom
        energies = np.array(
            [at.info["REF_energy"] / len(at) for at in all_atoms_pre]
        )
        energy_min = np.min(energies)
        keep_inds = np.where(energies < (energy_min + 6))[0]
        filtered_out = len(all_atoms_pre) - len(keep_inds)
        print(
            f"Filtered out {filtered_out} configurations due to E constraint"
        )

        for ind in keep_inds:
            all_atoms.append(all_atoms_pre[ind])

    # Always include all atoms from the last database
    all_atoms.extend(ase.io.read(structure_databases[-1], index=":"))

    if len(all_atoms) < n_batch:
        print(
            f"Specified {n_batch=} is larger than database Setting n_batch to size of database {len(all_atoms)}"
        )
        n_batch = len(all_atoms)

    # Set up the data processor
    data_processor.set_atoms(all_atoms)
    data_processor.transform_and_pad()
    data_processor.check_cutoff()
    data_processor.check_max_neighbors()
    data_processor.shuffle()
    data_processor.split_data()
    # Save the data processor
    data_processor.save(filename=f"data_{run_seed_str}.pickle")

    # Set up the model
    model_setup = ModelSetup(
        model_key,
        data_processor,
        n_max=input_args[
            "n_max"
        ],  # Maximum number of basis functions for the descriptor
        embed_d=input_args["embed_d"],  # Dimension of the embedding
        core_widths=input_args[
            "core_widths"
        ],  # Widths of the radial basis functions
        n_ensemble=input_args[
            "n_ensemble"
        ],  # Number of models in the ensemble
        morse=input_args["morse"],  # Use Morse-like variables
    )
    model_setup.init_params()

    model_setup.adjust_offsets()

    # Set up the Training instance
    training = Training(
        train_key,
        run_seed,
        model_setup,
        data_processor,
        n_batch=n_batch,
        n_eval_batch=input_args["n_eval_batch"],
        n_epochs=input_args["n_epochs"],
        energy_weight=input_args["energy_weight"],
        log_cosh_parameter_energy=input_args["log_cosh_parameter_energy"],
        log_cosh_parameter_forces=input_args["log_cosh_parameter_forces"],
        filepath=filepath,
        pickle_file_name=pickle_file_name,
        high_energy_penalty=input_args["high_energy_penalty"],
        save_reduced=input_args["save_reduced"],
        optimizer_type=input_args["optimizer_type"],
        learning_rate_kwargs=input_args["learning_rate_kwargs"],
    )

    training.run()

    # Evaluate the model on the validation set
    n_batch = 100

    # Make sure n_batch is not larger than the validation set size
    val_size = data_processor.val.cells.shape[0]
    if val_size < n_batch:
        n_batch = val_size

    evaluate_data_processor(
        training.reduced_pickle_file,
        data_processor,
        n_batch=n_batch,
        filename_label=run_seed_str,
        filepath=filepath,
    )

finished = True

with open(Path(filepath) / "is_finished.info", "w") as f:
    f.write(str(finished))
