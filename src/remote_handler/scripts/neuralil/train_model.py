import os
from typing import Optional

from remote_handler.scripts.neuralil._support import KeyArray

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import copy
import datetime
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import ase.db
import ase.io
import flax
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tqdm
import tqdm.auto
from ase.geometry import minkowski_reduce, wrap_positions
from jax import tree_util
from neuralil.bessel_descriptors import (
    PowerSpectrumGenerator,
    get_max_number_of_neighbors,
)
from neuralil.model import (
    NeuralIL,
    NeuralILModelInfo,
    NeuralILwithMorse,
    ResNetCore,
    update_energy_offset,
)
from neuralil.plain_ensembles.model import (
    PlainEnsemble,
    PlainEnsemblewithMorse,
)
from neuralil.plain_ensembles.training import (
    create_training_epoch,
    create_training_step,
    create_validation_step,
    emae_validation_statistic,
    ermse_validation_statistic,
    fmae_validation_statistic,
    frmse_validation_statistic,
    pack_params,
    unpack_params,
)
from neuralil.training import (
    create_elogcosh_validation_statistic,
    create_flogcosh_validation_statistic,
    create_log_cosh,
    create_one_cycle_minimizer,
    create_velo_minimizer,
)
from neuralil.utilities import create_array_shuffler, draw_urandom_int32

from remote_handler.scripts.neuralil._support import (
    batching,
    load_neuralil,
    structure_utils,
)


def transform_energy_forces(energy, forces, supercell_diag):
    """
    Scale energy and forces according to the supercell transformation.
    Args:
        energy (float): Original potential energy.
        forces (np.ndarray): Original forces.
        supercell_diag (list[int]): Supercell diagonal for transformations.
    Returns:
        tuple: Scaled energy and tiled forces.
    """
    multiplier = np.prod(
        supercell_diag
    )  # Get the number of unit cells in the supercell
    return (energy * multiplier, np.tile(forces, (multiplier, 1)))


def transform_atoms(atoms, r_cut, supercell_diag=[1, 1, 1]):
    """
    Transform an ASE Atoms object to a supercell that fits the cutoff.

    Args:
        atoms (ase.Atoms): ASE Atoms object.
        r_cut (float): Cutoff radius for neighbor calculations.
        supercell_diag (list[int], optional): Supercell diagonal for transformations. Defaults to [1, 1, 1].
    Returns:
        tuple: Transformed positions, symbols, cell, and supercell transformation.
    """
    # Multiply the cell by the supercell diagonal (initial scaling)
    sc_cell = np.dot(np.diag(np.array(supercell_diag)), atoms.cell[...])
    # Find a suitable supercell transformation
    super_trafo = structure_utils.find_super(sc_cell, r_cut, max_det=500)
    # Apply the transformation
    super_structure = atoms * super_trafo
    cell = super_structure.cell[...]
    return (
        wrap_positions(
            super_structure.positions, cell
        ),  # Wrap the positions to be within the cell
        super_structure.get_chemical_symbols(),
        cell,
        super_trafo,
    )


def transform_atoms_list(
    all_atoms,
    r_cut,
    supercell_diag=[1, 1, 1],
    isolated_trafo=(2, 2, 2),
    parse_ground_truth=True,
):
    """
    Transform a list of ASE Atoms objects to supercells that fit the cutoff.
    Args:
        all_atoms (list): List of ASE Atoms objects.
        r_cut (float): Cutoff radius for neighbor calculations.
        supercell_diag (list[int], optional): Supercell diagonal for transformations. Defaults to [1, 1, 1].
        isolated_trafo (tuple, optional): Supercell transformation to apply to isolated atoms. Defaults to (2, 2, 2).
        parse_ground_truth (bool, optional): Whether to parse forces and energies from the Atoms objects. Defaults to True.
    Returns:
        tuple: Transformed cells, symbols, positions, forces, energies, and transformation monitor.
    """

    # Lists to hold the transformed data
    length = len(all_atoms)
    forces = []
    energies = []
    cells = []
    symbols = []
    positions = []
    trafo_monitor = []

    print(f"Performing supercell transformations to fit {r_cut=}")
    for i in tqdm.auto.tqdm(range(length), total=length):
        atoms = all_atoms[i]
        # Extract ground truth data if required
        if parse_ground_truth:
            at_forces = atoms.arrays["REF_forces"]
            at_energy = atoms.info["REF_energy"]

        # Safety patch to avoid configs with only a single atom
        if len(atoms) == 1:
            # Create a first larger supercell
            atoms = atoms * isolated_trafo
            if parse_ground_truth:
                at_energy, at_forces = transform_energy_forces(
                    at_energy, at_forces, isolated_trafo
                )

        # Perform the actual transformation
        pos, symbs, cell, super_trafo = transform_atoms(
            atoms, r_cut, supercell_diag
        )

        # Scale the energy and forces if ground truth is parsed
        if parse_ground_truth:
            energy_scaled, forces_tiled = transform_energy_forces(
                at_energy, at_forces, super_trafo
            )
            forces.append(forces_tiled)
            energies.append(energy_scaled)

        # Store the transformed data
        cells.append(np.array(cell))
        symbols.append(symbs)
        positions.append(pos)
        # Monitor the size of the transformed structure
        trafo_monitor.append(len(pos))

    return (cells, symbols, positions, forces, energies, trafo_monitor)


def dont_transform_atoms(all_atoms):
    """
    Return positions, symbols, cells, forces, and energies from a list of ASE Atoms objects without transformation.
    Args:
        all_atoms (list): List of ASE Atoms objects.
    Returns:
        tuple: Cells, symbols, positions, forces, and energies.
    """

    forces = []
    energies = []
    cells = []
    symbols = []
    positions = []
    for at in all_atoms:
        forces.append(at.arrays["REF_forces"])
        energies.append(at.info["REF_energy"])
        cells.append(at.cell[...])
        symbols.append(at.get_chemical_symbols())
        positions.append(at.positions)

    return (cells, symbols, positions, forces, energies)


def padme(p, t, f, pad_size=None):
    """
    Pad positions, types, and forces to a uniform size.
    Args:
        p (list): List of positions arrays.
        t (list): List of types arrays.
        f (list): List of forces arrays.
        pad_size (int, optional): Size to pad the atomic configurations to. Defaults to None.
    Returns:
        tuple: Padded positions, types, and forces as jnp.ndarrays.
    """
    # Determine the maximum size if pad_size is not provided
    # For each configuration, get the number of atoms (first dimension of positions)
    max_pos = jnp.asarray(tree_util.tree_map(jnp.shape, p))[:, 0].max()
    if pad_size is None:
        pad_size = max_pos
    else:
        assert pad_size >= max_pos

    padded_positions = []
    padded_forces = []
    padded_types = []
    for idx in range(len(p)):
        # Calculate the required padding
        padding = pad_size - len(p[idx])
        padded_positions.append(
            jnp.pad(
                jnp.array(p[idx]),
                ((0, padding), (0, 0)),
                "constant",
                constant_values=1.0,
            )
        )
        padded_types.append(
            jnp.pad(
                jnp.array(t[idx]),
                ((0, padding)),
                "constant",
                constant_values=-1,
            )
        )
        if len(f) > 0:
            padded_forces.append(
                jnp.pad(jnp.array(f[idx]), ((0, padding), (0, 0)))
            )

    return (
        jnp.array(padded_positions),
        jnp.array(padded_types),
        jnp.array(padded_forces),
    )


@dataclass
class Data:
    positions: jnp.ndarray
    types: jnp.ndarray
    cells: jnp.ndarray
    energies: jnp.ndarray
    forces: jnp.ndarray


def get_types_from_symbols(symbols):
    """
    Map atomic symbols to integer types.
    """
    all_symbols = set()
    for s in symbols:
        all_symbols.update(s)
    sorted_elements = sorted(list(all_symbols))
    symbol_map = {s: i for i, s in enumerate(sorted_elements)}
    n_types = len(symbol_map)
    types = []
    for s in symbols:
        types.append([symbol_map[i] for i in s])
    return types, n_types, sorted_elements


class DataProcessor:
    """
    Class to handle data processing for NeuralIL models.
    It includes methods to transform atomic structures, pad data, and check consistency.

    Args:
        key (KeyArray): JAX random key for shuffling data.
        r_cut (float): Cutoff radius for neighbor calculations.
        training_fraction (float): Fraction of data to be used for training.
        supercell_diag (list[int], optional): Supercell diagonal for transformations. Defaults to [1, 1, 1].
        parse_ground_truth (bool, optional):
        filepath (os.PathLike, optional): Path to save/load the processor state. Defaults to ".".
    """

    def __init__(
        self,
        key: KeyArray,
        r_cut: float,
        training_fraction: float,
        supercell_diag: list[int] = [1, 1, 1],
        parse_ground_truth: bool = True,
        filepath: os.PathLike = ".",
    ) -> None:
        # Initialize attributes
        self.r_cut = r_cut
        self.training_fraction = training_fraction
        self.key = key
        self.supercell_diag = supercell_diag
        self.parse_ground_truth = parse_ground_truth
        self.filepath = Path(filepath)

    def run(self):
        self.transform_and_pad()
        self.check_cutoff()
        self.check_max_neighbors()
        self.shuffle()
        self.split_data()

    def atoms_from_db(self, db_file):
        """
        Load atoms from an ASE database file.
        Args:
            db_file (str): Path to the ASE database file.
        """
        self.all_atoms = []
        with ase.db.connect(db_file) as db:
            for row in db.select():
                self.all_atoms.append(row.toatoms())

    def atoms_from_aseio(self, db_file):
        """
        Load atoms from an ASE .extxyz file.

        Args:
        db_file (str): Path to the ASE .extxyz file.
        """
        self.all_atoms = ase.io.read(db_file, index=":")

    def set_atoms(self, atoms_list):
        """
        Load a list of ASE Atoms objects.
        Args:
            atoms_list (list): List of ASE Atoms objects.
        """
        self.all_atoms = atoms_list

    def transform_and_pad(self, transform=True, pad_size=None):
        """
        Transform and pad atomic structures.
        Args:
            transform (bool, optional): Whether to perform supercell transformations. Defaults to True.
            pad_size (int, optional): Size to pad the atomic configurations to. Defaults to None.
        """
        # Perform transformations if required
        if transform:
            (
                cells,
                symbols,
                positions,
                forces,
                energies,
                trafo_monitor,
            ) = transform_atoms_list(
                self.all_atoms,
                self.r_cut,
                self.supercell_diag,
                parse_ground_truth=self.parse_ground_truth,
            )

            # This are the numbers of atoms in the transformed structures
            self.trafo_monitor = jnp.array(trafo_monitor)
            # Create a frequency table of the sizes of the transformed structures
            arr = [[k, v] for k, v in Counter(trafo_monitor).items()]
            arr.sort()
            df = pd.DataFrame(arr, columns=["#atoms", "frequency"])
            print("size of transformed structures:")
            print(df.to_string(index=False))
        else:
            (
                cells,
                symbols,
                positions,
                forces,
                energies,
            ) = dont_transform_atoms(self.all_atoms)
        self.n_configurations = len(positions)

        # Map the types to their indices
        (types, self.n_types, self.sorted_elements) = get_types_from_symbols(
            symbols
        )

        # Pad the data
        self.positions, types, self.forces = padme(
            positions, types, forces, pad_size
        )
        self.cells = jnp.array(cells)
        self.energies = jnp.array(energies)
        self.types = jnp.array(types)

    def check_cutoff(self):
        """
        Check that the minimum image convention is fulfilled for all structures.
        """
        cells = jax.vmap(
            lambda c: jnp.dot(jnp.diag(jnp.array(self.supercell_diag)), c)
        )(self.cells)
        min_cut = np.min(list(map(structure_utils.get_max_cutoff, cells)))
        if min_cut < self.r_cut:
            raise Exception(
                f"WARNING: inconsistency in supercells. {min_cut=} at "
                f"{self.r_cut=}"
            )

    def check_max_neighbors(self):
        """
        Check the maximum number of neighbors that must be considered.
        """
        self.max_neighbors = max(
            [
                get_max_number_of_neighbors(
                    p, t, self.r_cut, c, *self.supercell_diag
                )
                for p, t, c in zip(self.positions, self.types, self.cells)
            ]
        )

        print(
            f"- Maximum number of neighbors that must be considered:"
            f"{self.max_neighbors}"
        )

    def shuffle(self):
        """Shuffle the data."""
        self.key, key = jax.random.split(self.key)
        shuffle = create_array_shuffler(key)
        self.cells = shuffle(self.cells)
        self.positions = shuffle(self.positions)
        self.energies = shuffle(self.energies)
        self.types = shuffle(self.types)
        self.forces = shuffle(self.forces)

    def split_data(self):
        """Split the data into training and validation sets."""
        n_train = round(self.n_configurations * self.training_fraction)
        n_validate = self.n_configurations - n_train

        print(f"- {self.n_configurations} configurations are available")
        print(f"\t- {n_train} will be used for training")
        print(f"\t- {n_validate} will be used for validation")

        def split_array(in_array):
            "Split an array in training and validation sections."
            return jnp.split(in_array, (n_train, n_train + n_validate))[:2]

        # Split all the necessary arrays
        cells_train, cells_validate = split_array(self.cells)
        positions_train, positions_validate = split_array(self.positions)
        types_train, types_validate = split_array(self.types)

        if self.parse_ground_truth:
            energies_train, energies_validate = split_array(self.energies)
            forces_train, forces_validate = split_array(self.forces)
        else:
            energies_train, energies_validate = [], []
            forces_train, forces_validate = [], []

        self.train = Data(
            positions_train,
            types_train,
            cells_train,
            energies_train,
            forces_train,
        )

        self.val = Data(
            positions_validate,
            types_validate,
            cells_validate,
            energies_validate,
            forces_validate,
        )

    def save(self, filename="data_processor.pickle"):
        """
        Save the data processor state to a file.
        Args:
            filename (str, optional): Filename to save the state. Defaults to "data_processor.pickle".
        """
        with open(self.filepath / filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename="data_processor.pickle"):
        """
        Load a data processor state from a file.
        Args:
            filename (str, optional): Filename to load the state from. Defaults to "data_processor.pickle".
        Returns:
            DataProcessor: Loaded data processor instance.
        """
        if not os.path.exists(filename):
            raise Exception("File does not exist!")

        with open(filename, "rb") as f:
            return pickle.load(f)

    @classmethod
    def from_atoms_list(
        cls,
        atoms_list,
        r_cut=4.0,
        training_fraction=1.0,
        supercell_diag=[3, 3, 3],
        seed=0,
    ):
        """
        Create a DataProcessor instance from a list of ASE Atoms objects.
        Args:
            atoms_list (list): List of ASE Atoms objects.
            r_cut (float, optional): Cutoff radius for neighbor calculations. Defaults to 4.0.
            training_fraction (float, optional): Fraction of data to be used for training. Defaults to 1.0.
            supercell_diag (list[int], optional): Supercell diagonal for transformations. Defaults to [3, 3, 3].
            seed (int, optional): Random seed for shuffling. Defaults to 0.
        Returns:
            DataProcessor: Initialized data processor instance."""
        key = jax.random.PRNGKey(seed)
        data_processor = cls(
            key,
            r_cut=r_cut,
            training_fraction=training_fraction,
            supercell_diag=supercell_diag,
            parse_ground_truth=False,
        )
        data_processor.set_atoms(atoms_list)
        data_processor.transform_and_pad()
        data_processor.check_cutoff()
        data_processor.check_max_neighbors()
        # data_processor.shuffle()
        data_processor.split_data()
        return data_processor


#########################################################################################################################


def compute_data(n_batch, batch_func, data):
    res = batching.evaluate_batch_func_pytree_args_remainder(
        n_batch, batch_func, data.positions, data.types, data.cells
    )
    return res


def save_data(label, e_pred, f_pred, data):
    data = dict(
        positions=data.positions,
        types=data.types,
        cells=data.cells,
        e_pred=e_pred,
        e_obs=data.energies,
        f_pred=f_pred,
        f_obs=data.forces,
    )
    np.savez(label, **data)
    return data


def evaluate_data_processor(
    pickle_file, data_processor, n_batch=100, filename_label="", filepath="."
):
    filepath = Path(filepath)
    train_label = "train"
    if data_processor.training_fraction == 1.0:
        train_label = "test"

    eval_energy, eval_forces = load_neuralil.create_evaluators(
        pickle_file,
        data_processor.max_neighbors,
        data_processor.supercell_diag,
    )
    e_batch = jax.jit(jax.vmap(eval_energy))
    f_batch = jax.jit(jax.vmap(eval_forces))

    e_train = compute_data(n_batch, e_batch, data_processor.train)
    f_train = compute_data(n_batch, f_batch, data_processor.train)
    train_data = save_data(
        filepath / f"{train_label}{filename_label}",
        e_train,
        f_train,
        data_processor.train,
    )

    val_data = None
    if data_processor.training_fraction < 1.0:
        e_val = compute_data(n_batch, e_batch, data_processor.val)
        f_val = compute_data(n_batch, f_batch, data_processor.val)
        val_data = save_data(
            filepath / f"val{filename_label}", e_val, f_val, data_processor.val
        )

    return train_data, val_data


class ModelSetup:
    """
    Class to set up a NeuralIL model with specified parameters.
    Args:
        key (KeyArray): JAX random key for model initialization.
        data_processor (DataProcessor): Data processor instance containing training data.
        n_max (int): Maximum number of basis functions for the descriptor.
        embed_d (int, optional): Dimension of the embedding. Defaults to 1.
        core_widths (list[int], optional): Widths of the radial basis functions. Defaults to [64, 32, 16].
        n_ensemble (int, optional): Number of models in the ensemble. Defaults to 5.
        morse (bool, optional): Whether to use Morse-like variables. Defaults to True.
    """

    def __init__(
        self,
        key,
        data_processor,
        n_max,
        embed_d=1,
        core_widths=[64, 32, 16],
        n_ensemble=5,
        morse=True,
    ):
        # Initialize attributes
        self.key = key
        self.n_max = n_max
        self.embed_d = embed_d
        self.core_widths = core_widths
        self.n_ensemble = n_ensemble
        self.r_cut = data_processor.r_cut
        self.data_processor = data_processor
        # The jaxrens neuralil fork takes max_neighbors at call time (in the
        # energy/force methods), so keep it handy for the apply/init calls.
        self.max_neighbors = data_processor.max_neighbors
        max_neighbors = self.max_neighbors

        # Set up the descriptor generator and model
        descriptor_generator = PowerSpectrumGenerator(
            n_max,
            data_processor.r_cut,
            data_processor.n_types,
            data_processor.supercell_diag,
        )
        # Set up the core model, individual model, and ensemble
        core_model = ResNetCore(core_widths)

        if morse:
            individual_model = NeuralILwithMorse(
                self.data_processor.n_types,
                embed_d,
                data_processor.r_cut,
                descriptor_generator,
                descriptor_generator.process_some_data,
                core_model,
                morse_type="RepulsiveMorse",
            )
            self.dynamics_model = PlainEnsemblewithMorse(
                individual_model, n_ensemble
            )
        else:
            individual_model = NeuralIL(
                self.data_processor.n_types,
                embed_d,
                data_processor.r_cut,
                descriptor_generator,
                descriptor_generator.process_some_data,
                core_model,
            )
            self.dynamics_model = PlainEnsemble(individual_model, n_ensemble)

        def individual_energy_calculator(params, positions, types, cell):
            return individual_model.apply(
                params,
                positions,
                types,
                cell,
                max_neighbors,
                method=individual_model.calc_potential_energy,
            )

        # JIT compile the vmapped individual energy calculator for efficiency
        # Mapped over positions, types, and cells
        self.vmapped_individual_energy_calculator = jax.jit(
            jax.vmap(individual_energy_calculator, in_axes=(None, 0, 0, 0))
        )

    def init_params(self):
        """
        Initialize model parameters using a sample from the training data.
        """
        self.key, key = jax.random.split(self.key)

        self.model_params = self.dynamics_model.init(
            key,
            self.data_processor.train.positions[0],
            self.data_processor.train.types[0],
            self.data_processor.train.cells[0],
            self.max_neighbors,
            method=self.dynamics_model.calc_forces,
        )

    def get_initial_evaluators(self, batch=False):
        """
        Get JIT-compiled evaluators for energy and forces.

        Args:
            batch (bool, optional): Whether to return batched evaluators. Defaults to False.
        Returns:
            tuple: JIT-compiled energy and forces evaluators.
        """
        calc_energy, calc_forces = load_neuralil.evaluators_from_model(
            self.dynamics_model, self.model_params, self.max_neighbors
        )
        if batch:
            return jax.jit(jax.vmap(calc_energy)), jax.jit(
                jax.vmap(calc_forces)
            )
        else:
            return jax.jit(calc_energy), jax.jit(calc_forces)

    def adjust_offsets(self, n_batch=64):
        """
        Adjust energy offsets for each model in the ensemble to match the training data.
        Args:
            n_batch (int, optional): Batch size for evaluation. Defaults to 64.
        """
        n_at = (self.data_processor.train.types >= 0).sum(axis=1)
        if n_at.shape[0] < n_batch:
            n_batch = n_at.shape[0]
        # Get the individual parameters for each model in the ensemble
        individual_params = unpack_params(self.model_params)
        updated_params = []
        # For each model, compute the predicted energies and adjust the offset
        for p in individual_params:
            energies = batching.evaluate_batch_func_pytree_args_remainder(
                n_batch,
                lambda pp, tt, cc: self.vmapped_individual_energy_calculator(
                    p, pp, tt, cc
                ),
                self.data_processor.train.positions,
                self.data_processor.train.types,
                self.data_processor.train.cells,
            )
            energies = jnp.asarray(energies)
            # Compute the offset to align predicted and observed energies
            # Each difference is divided by number of atoms in the configuration
            offset = -jnp.average(
                (self.data_processor.train.energies - energies) / n_at
            )
            updated_params.append(update_energy_offset(p, offset))
        # Pack the updated parameters back into the ensemble structure
        self.model_params = pack_params(updated_params)
        print("- Done")


def params1_dominate_params2(statistics1, statistics2):
    if (statistics1 > statistics2).any():
        return False
    return (statistics1 < statistics2).any()


def are_fronts_equal(front1, front2):
    if len(front1) != len(front2):
        return False
    return all([p1[1] == p2[1] for (p1, p2) in zip(front1, front2)])


def reduce_pareto_front(pareto_front):
    """
    Reduce a Pareto front by removing dominated points.
    Args:
        pareto_front (list): List of tuples containing (params, (energy_rmse, force_rmse)).
    Returns:
        list: Reduced Pareto front.
    """
    distilled_pareto_front = []
    for p1 in pareto_front:
        for p2 in pareto_front:
            if params1_dominate_params2(
                jnp.asarray(p2[-2:]), jnp.asarray(p1[-2:])
            ):
                break
        else:
            distilled_pareto_front.append(p1)
    return distilled_pareto_front


def get_high_energy_weight(e_obs_per_atom, thresh):
    w = jnp.where(
        e_obs_per_atom > thresh, 0.1 * jnp.exp(-0.01 * e_obs_per_atom), 1
    )
    return w


def create_gradient_evaluator(model, calc_loss_contribution, max_neighbors):
    def gradient_evaluator(
        model_params,
        positions_batch,
        types_batch,
        cells_batch,
        energies_batch,
        forces_batch,
    ):
        def calc_loss(model_params):
            def invoke_loss_contribution(
                positions, types, cell, obs_energy, obs_forces
            ):
                "Compute the contribution to the loss of a single data point."
                pred_energy, pred_forces = model.apply(
                    model_params,
                    positions,
                    types,
                    cell,
                    max_neighbors,
                    method=model.calc_potential_energy_and_forces,
                )
                pred_forces = jnp.where(
                    jnp.expand_dims(types, axis=-1) >= 0,
                    pred_forces,
                    jnp.zeros(3),
                )
                loss = jax.vmap(
                    calc_loss_contribution, in_axes=(0, None, 0, None, None)
                )(pred_energy, obs_energy, pred_forces, obs_forces, types)
                return loss

            return (
                jax.vmap(jax.checkpoint(invoke_loss_contribution))(
                    positions_batch,
                    types_batch,
                    cells_batch,
                    energies_batch,
                    forces_batch,
                )
                .mean(axis=0)
                .sum(axis=0)
            )

        loss_val = calc_loss(model_params)
        loss_grad = jax.jacrev(calc_loss)(model_params)
        return loss_val, loss_grad

    return gradient_evaluator


class Training:
    """
    Class to handle the training of a NeuralIL model.

    Args:
        key (KeyArray): JAX random key for training.
        run_seed (int): Seed for the training run.
        model_setup (ModelSetup): Model setup instance containing the model and parameters.
        data_processor (DataProcessor): Data processor instance containing training and validation data.
        n_batch (int, optional): Batch size for training. Defaults to 32.
        n_eval_batch (int, optional): Batch size for evaluation. Defaults to 64.
        n_epochs (int, optional): Number of training epochs. Defaults to 50.
        energy_weight (float, optional): Weighting factor for energy in the loss function. Defaults to 0.5.
        log_cosh_parameter_energy (float, optional): Parameter for log-cosh loss for energy. Defaults to 0.200.
        log_cosh_parameter_forces (float, optional): Parameter for log-cosh loss for forces. Defaults to 2.000.
        filepath (os.PathLike, optional): Path to save training outputs. Defaults to ".".
        pickle_file_name (os.PathLike, optional): Filename to save model parameters. Defaults to None.
        high_energy_penalty (float, optional): Threshold for high energy penalty in loss function. Defaults to None.
        save_reduced (int, optional): Number of ensemble members to save in a reduced model. Defaults to None.
        optimizer_type (str, optional): Type of optimizer to use ("VeLO" or "OneCycle"). Defaults to "VeLO".
        learning_rate_kwargs (dict, optional): Learning rate parameters for OneCycle optimizer. Defaults to None.
    """

    def __init__(
        self,
        key,
        run_seed,
        model_setup: ModelSetup,
        data_processor: DataProcessor,
        n_batch: int = 32,
        n_eval_batch: int = 64,
        n_epochs: int = 50,
        energy_weight: float = 0.5,
        log_cosh_parameter_energy: float = 0.200,
        log_cosh_parameter_forces: float = 2.000,
        filepath: os.PathLike = ".",
        pickle_file_name: Optional[os.PathLike] = None,
        high_energy_penalty: Optional[float] = None,
        save_reduced: Optional[int] = None,
        optimizer_type: str = "VeLO",
        learning_rate_kwargs: Optional[dict] = dict(
            max_learning_rate=1e-2,
            min_learning_rate=1e-3,
            fin_learning_rate=1e-5,
        ),
    ):
        # Initialize attributes
        self.run_seed = run_seed
        self.run_seed_str = f"{self.run_seed:0X}"
        self.instance_code = f"VELO_{self.run_seed_str}"

        self.n_batch = n_batch
        self.n_eval_batch = n_eval_batch
        self.n_epochs = n_epochs
        self.high_energy_penalty = high_energy_penalty

        self.model_setup = model_setup
        self.data_processor = data_processor

        self.filepath = Path(filepath)

        # File to save the ensemble model parameters
        if pickle_file_name is None:
            self.pickle_file = self.filepath / (
                f"model_params_plain_ensemble_{self.instance_code}.pkl"
            )
        else:
            self.pickle_file = pickle_file_name

        self.save_reduced = save_reduced
        if self.save_reduced is not None:
            if pickle_file_name is None:
                self.reduced_pickle_file = self.filepath / (
                    f"model_params_plain_ensemble_{self.instance_code}_reduced{self.save_reduced}.pkl"
                )
            else:
                self.reduced_pickle_file = pickle_file_name
                path, filename = os.path.split(pickle_file_name)
                # Build also the full model filename
                self.pickle_file = os.path.join(path, "full_" + filename)
            if self.save_reduced > self.model_setup.n_ensemble:
                raise Exception(
                    "Can not save a reduced model with more ensemble members"
                    " than the original model!"
                )

        # Create loss functions and validation statistics
        log_cosh_energy = create_log_cosh(1.0 / log_cosh_parameter_energy)
        log_cosh_forces = create_log_cosh(1.0 / log_cosh_parameter_forces)

        def calc_loss_contribution(
            pred_energy, obs_energy, pred_forces, obs_forces, types
        ):
            """Return the weighted log-cosh of the prediction deltas.
            1.0 - energy_weight is the weight for forces.

            Args:
                pred_energy (jnp.ndarray): Predicted energy.
                obs_energy (jnp.ndarray): Observed energy.
                pred_forces (jnp.ndarray): Predicted forces.
                obs_forces (jnp.ndarray): Observed forces.
                types (jnp.ndarray): Atomic types.
            Returns:
                jnp.ndarray: Computed loss contribution."""

            # Mask out the forces for padded atoms
            pred_forces = jnp.where(
                jnp.expand_dims(types, axis=-1) >= 0,
                pred_forces,
                jnp.zeros(3),
            )
            # Compute the per-atom energy difference
            delta_energy = (obs_energy - pred_energy) / (types >= 0).sum()
            energy_contribution = log_cosh_energy(delta_energy)
            # Compute the per-atom force differences
            delta_forces = jnp.sqrt(
                ((obs_forces - pred_forces) ** 2).sum(axis=-1)
            )
            forces_contribution = (
                log_cosh_forces(delta_forces).sum() / (types >= 0).sum()
            )
            # Combine the energy and force contributions
            loss = (
                energy_weight * energy_contribution
                + (1.0 - energy_weight) * forces_contribution
            )
            if high_energy_penalty is not None:
                loss *= get_high_energy_weight(
                    obs_energy / pred_forces.shape[0], high_energy_penalty
                )

            return loss

        # Store the loss function
        self.calc_loss_contribution = calc_loss_contribution
        # Initialize validation statistics to monitor during training
        self.validation_statistics = {
            "forces_MAE": fmae_validation_statistic,
            "energy_MAE": emae_validation_statistic,
            "energy_logcosh": create_elogcosh_validation_statistic(
                1.0 / log_cosh_parameter_energy
            ),
            "forces_RMSE": frmse_validation_statistic,
            "energy_RMSE": ermse_validation_statistic,
            "forces_logcosh": create_flogcosh_validation_statistic(
                1.0 / log_cosh_parameter_forces
            ),
        }
        # Units for the validation statistics
        self.validation_units = {
            "forces_MAE": "eV / Å",
            "energy_MAE": "eV / atom",
            "forces_RMSE": "eV / Å",
            "energy_RMSE": "eV / atom",
            "forces_logcosh": "eV / Å",
            "energy_logcosh": "eV / atom",
        }
        # Number of training samples
        n_train = self.data_processor.train.positions.shape[0]

        if optimizer_type.lower() == "velo":
            optimizer = create_velo_minimizer(
                n_train // self.n_batch, self.n_epochs
            )
        elif optimizer_type.lower() == "onecycle":
            assert (
                learning_rate_kwargs is not None
            ), "Need learning_rate_kwargs!"
            optimizer = create_one_cycle_minimizer(
                n_train // self.n_batch,
                learning_rate_kwargs["min_learning_rate"],
                learning_rate_kwargs["max_learning_rate"],
                learning_rate_kwargs["fin_learning_rate"],
            )
        else:
            raise Exception("Unknown optimizer. Try 'velo' or 'onecycle'!")
        # Initialize the optimizer state
        self.optimizer_state = optimizer.init(self.model_setup.model_params)

        # Create the training and validation steps and epochs
        self.training_step = create_training_step(
            self.model_setup.dynamics_model,
            optimizer,
            self.calc_loss_contribution,
            self.data_processor.max_neighbors,
        )
        # Create the training epoch function
        self.training_epoch = create_training_epoch(
            self.data_processor.train.positions,
            self.data_processor.train.types,
            self.data_processor.train.cells,
            self.data_processor.train.energies,
            self.data_processor.train.forces,
            self.n_batch,
            self.training_step,
            key,
        )
        # Create the validation step function
        self.validation_step = create_validation_step(
            self.model_setup.dynamics_model,
            self.validation_statistics,
            self.data_processor.val.positions,
            self.data_processor.val.types,
            self.data_processor.val.cells,
            self.data_processor.val.energies,
            self.data_processor.val.forces,
            self.n_eval_batch,
            self.data_processor.max_neighbors,
        )

    def debug_nans(self):
        gradient_evaluator = jax.jit(
            create_gradient_evaluator(
                self.model_setup.dynamics_model,
                self.calc_loss_contribution,
                self.data_processor.max_neighbors,
            )
        )

        import jax.random
        from neuralil.training import BatchedIterator

        batched_iterator = BatchedIterator(
            10,
            jax.random.PRNGKey(0),
            self.data_processor.train.positions,
            self.data_processor.train.types,
            self.data_processor.train.cells,
            self.data_processor.train.energies,
            self.data_processor.train.forces,
        )
        iterator = iter(batched_iterator)
        for (
            positions_batch,
            types_batch,
            cells_batch,
            energies_batch,
            forces_batch,
        ) in iterator:
            (loss_val, loss_grad) = gradient_evaluator(
                self.model_setup.model_params,
                positions_batch,
                types_batch,
                cells_batch,
                energies_batch,
                forces_batch,
            )
            print("Loss: ", loss_val)
            violated = tree_util.tree_map(
                lambda x: jnp.any(jnp.isnan(x)), loss_grad
            )
            conds, _ = tree_util.tree_flatten(violated)
            if jnp.any(jnp.array(conds)):
                print(violated)

    def run(self):
        """Run the training process."""
        # Get the initial model parameters
        model_params = self.model_setup.model_params
        # Get the initial parameters for each model in the ensemble
        initial_params = unpack_params(model_params)
        # Set up the Pareto fronts for each model in the ensemble
        pareto_fronts = [
            [(initial_params[i_model], -1, jnp.inf, jnp.inf)]
            for i_model in range(self.model_setup.n_ensemble)
        ]

        # Set up a file to log the errors during training
        error_io = open(self.filepath / f"error_{self.instance_code}.txt", "w")

        for i_epoch in range(self.n_epochs):
            # Run a full epoch.
            (self.optimizer_state, model_params) = self.training_epoch(
                self.optimizer_state, model_params
            )
            # Evaluate the results.
            statistics = self.validation_step(model_params)
            mae = statistics["forces_MAE"]
            rmse = statistics["forces_RMSE"]
            logcosh = statistics["forces_logcosh"]
            e_mae = statistics["energy_MAE"]
            e_rmse = statistics["energy_RMSE"]
            e_logcosh = statistics["energy_logcosh"]
            # Print the relevant statistics.
            candidates = []
            new_params = unpack_params(model_params)
            print("VALIDATION:")
            for i_model in range(self.model_setup.n_ensemble):
                print(
                    f"Model #{i_model+1}:"
                    f"F_RMSE = {rmse[i_model]} {self.validation_units['forces_RMSE']}. "
                    f"F_MAE = {mae[i_model]} {self.validation_units['forces_MAE']}. "
                    f"F_logcosh = {logcosh[i_model]} {self.validation_units['forces_logcosh']}. "
                    f"E_RMSE = {e_rmse[i_model]} {self.validation_units['energy_RMSE']}. "
                    f"E_MAE = {e_mae[i_model]} {self.validation_units['energy_MAE']}. "
                    f"E_logcosh = {e_logcosh[i_model]} {self.validation_units['energy_logcosh']}."
                )
                error_io.write(
                    f"{i_epoch=}\n"
                    f"Model #{i_model+1}:\n"
                    f"F_RMSE = {rmse[i_model]} {self.validation_units['forces_RMSE']}. \n"
                    f"F_MAE = {mae[i_model]} {self.validation_units['forces_MAE']}. \n"
                    f"F_logcosh = {logcosh[i_model]} {self.validation_units['forces_logcosh']}. \n"
                    f"E_RMSE = {e_rmse[i_model]} {self.validation_units['energy_RMSE']}. \n"
                    f"E_MAE = {e_mae[i_model]} {self.validation_units['energy_MAE']}. \n"
                    f"E_logcosh = {e_logcosh[i_model]} {self.validation_units['energy_logcosh']}.\n"
                )
                # Append the current model parameters and statistics to the candidates list
                # Each candidate is a tuple of (params, epoch, force_logcosh, energy_logcosh)
                candidates.append(
                    (
                        new_params[i_model],
                        i_epoch,
                        float(logcosh[i_model]),
                        float(e_logcosh[i_model]),
                    )
                )

            old_pareto_fronts = copy.deepcopy(pareto_fronts)

            updated = []
            for i_model in range(self.model_setup.n_ensemble):
                pareto_fronts[i_model].append(candidates[i_model])
                # Reduce the Pareto front to remove dominated points
                pareto_fronts[i_model] = reduce_pareto_front(
                    pareto_fronts[i_model]
                )
                if not are_fronts_equal(
                    pareto_fronts[i_model], old_pareto_fronts[i_model]
                ):
                    updated.append(i_model)

            if updated:
                # Compute the number of elements that costitute the Pareto fronts
                # across all models in the ensemble
                n_elements = sum([len(p) for p in pareto_fronts])
                print("Pareto fronts updated")
                print("There are currently", n_elements, "elements")
                for p in pareto_fronts:
                    # print the epoch and the errors for each element in the front
                    print([t[1:] for t in p])
                print(
                    "Only the element with the lowest force"
                    " loss in each front will be saved."
                )
                # Select the element with the lowest force loss in each Pareto front
                to_be_saved = [
                    min(p, key=lambda x: x[2]) for p in pareto_fronts
                ]

                # Create an object to hold all the information about the model.
                model_info = NeuralILModelInfo(
                    model_name=self.model_setup.dynamics_model.model_name,
                    model_version=self.model_setup.dynamics_model.model_version,
                    timestamp=datetime.datetime.now(),
                    r_cut=self.model_setup.r_cut,
                    n_max=self.model_setup.n_max,
                    sorted_elements=self.data_processor.sorted_elements,
                    embed_d=self.model_setup.embed_d,
                    core_widths=self.model_setup.core_widths,
                    constructor_kwargs=dict(),
                    random_seed=self.run_seed,
                    params=flax.serialization.to_state_dict(
                        pack_params([m[0] for m in to_be_saved])
                    ),
                    specific_info=None,
                )
                # Save the model information to a pickle file
                with open(self.pickle_file, "wb") as f:
                    print("- Saving the updated Pareto front")
                    pickle.dump(model_info, f, protocol=5)

        if self.save_reduced is not None:
            to_save = [t[1:] for t in to_be_saved]
            # Sort the models by force log-cosh error and select the best 'save_reduced' models
            model_idx_sorted = np.argsort(np.array(to_save)[:, 1])
            model_info = NeuralILModelInfo(
                model_name=self.model_setup.dynamics_model.model_name,
                model_version=self.model_setup.dynamics_model.model_version,
                timestamp=datetime.datetime.now(),
                r_cut=self.model_setup.r_cut,
                n_max=self.model_setup.n_max,
                sorted_elements=self.data_processor.sorted_elements,
                embed_d=self.model_setup.embed_d,
                core_widths=self.model_setup.core_widths,
                constructor_kwargs=dict(),
                random_seed=self.run_seed,
                params=flax.serialization.to_state_dict(
                    pack_params(
                        [
                            to_be_saved[model_idx_sorted[i]][0]
                            for i in range(self.save_reduced)
                        ]
                    )
                ),
                specific_info=None,
            )

            with open(self.reduced_pickle_file, "wb") as f:
                print("- Saving the updated Pareto front")
                pickle.dump(model_info, f, protocol=5)

            with open(self.filepath / "saved_params.pkl", "wb") as f:
                pickle.dump(to_be_saved, f, protocol=5)


def seed_from_datafile(datafile):
    run_seed_str = re.match(".*_([a-fA-F0-9]+).pickle", datafile).groups()[0]
    run_seed = int(run_seed_str, 16)  # conversion hex -> dec
    return run_seed, run_seed_str


def keys_from_datafile(datafile):
    run_seed, _ = seed_from_datafile(datafile)
    rng = jax.random.PRNGKey(run_seed)
    return jax.random.split(rng, num=3)


if __name__ == "__main__":
    STRUCTURE_DATABASE = "/home/nico.unglert/code/jaxnest_dev/test/data/active_learning/bartok_small_train.extxyz"

    R_CUT = 4.0
    TRAINING_FRACTION = 0.6

    run_seed = draw_urandom_int32()
    run_seed_str = f"{run_seed:0X}"
    instance_code = f"VELO_{run_seed_str}"
    rng = jax.random.PRNGKey(run_seed)

    rng, key = jax.random.split(rng)
    data_processor = DataProcessor(key, R_CUT, TRAINING_FRACTION)
    data_processor.atoms_from_aseio(STRUCTURE_DATABASE)
    data_processor.run()

    rng, key = jax.random.split(rng)
    model_setup = ModelSetup(
        key,
        data_processor,
        # n_types=1,
        n_max=6,
        # r_cut=R_CUT,
        embed_d=1,
        core_widths=[64, 32, 16],
        n_ensemble=5,
    )
    model_setup.init_params()
    # model_setup.adjust_offsets()

    rng, key = jax.random.split(rng)
    training = Training(
        key,
        run_seed,
        model_setup,
        data_processor,
        n_batch=32,
        n_eval_batch=64,
        n_epochs=50,
        energy_weight=0.5,
        log_cosh_parameter_energy=0.200,
        log_cosh_parameter_forces=2.000,
    )

    training.run()
