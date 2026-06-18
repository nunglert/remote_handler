"""NeuralIL model loading / evaluation glue needed by training.

Vendored from ``jaxnest.backends.neuralil.load_neuralil`` so that the
remote-side NeuralIL training payload no longer depends on the ``jaxnest``
package. Only the closure used by training is copied: ``load_neuralil_model``,
``create_evaluators`` and ``evaluators_from_model``.

These depend only on the third-party ``neuralil`` package (the actual NeuralIL
force-field library) plus :func:`get_max_cutoff` from the vendored
:mod:`structure_utils`.
"""

import pickle
from typing import Sequence

import jax.numpy as jnp
from neuralil.bessel_descriptors import _get_max_number_of_neighbors
from neuralil.model import ResNetCore
from neuralil.plain_ensembles.training import get_n_models

from .structure_utils import get_max_cutoff


def load_neuralil_model(pickle_file, max_neighbors, supercell_trafo):
    from neuralil.bessel_descriptors import PowerSpectrumGenerator
    from neuralil.model import NeuralIL, NeuralILwithMorse
    from neuralil.plain_ensembles.model import (
        PlainEnsemble,
        PlainEnsemblewithMorse,
    )

    model_info = pickle.load(open(pickle_file, "rb"))
    # NOTE: the jaxrens neuralil fork dropped the ``max_neighbors`` argument
    # from the descriptor-generator constructor (it is still passed at call
    # time, e.g. to ``process_some_data`` / the eval criteria below).
    descriptor_generator = PowerSpectrumGenerator(
        model_info.n_max,
        model_info.r_cut,
        len(model_info.sorted_elements),
        supercell_trafo,
    )

    core_model = ResNetCore(model_info.core_widths)

    N_ENSEMBLE = get_n_models(model_info.params)

    if "morse" in model_info.params["params"]["neuralil"].keys():
        individual_model = NeuralILwithMorse(
            len(model_info.sorted_elements),
            model_info.embed_d,
            model_info.r_cut,
            descriptor_generator,
            descriptor_generator.process_some_data,
            core_model,
            morse_type="RepulsiveMorse",
        )
        dynamics_model = PlainEnsemblewithMorse(individual_model, N_ENSEMBLE)
    else:
        individual_model = NeuralIL(
            len(model_info.sorted_elements),
            model_info.embed_d,
            model_info.r_cut,
            descriptor_generator,
            descriptor_generator.process_some_data,
            core_model,
        )
        dynamics_model = PlainEnsemble(individual_model, N_ENSEMBLE)

    return model_info, dynamics_model


def create_evaluators(
    pickle_file: str, max_neighbors: int, supercell_trafo: Sequence[int]
):
    """Creates evaluation functions for the potential energy and the forces
    using the NeuralIL Neural-Network Force Field. We use PlainEnsemble models
    since they are almost as cheap as a single model.

    Args:
        pickle_file: Path to ickle file containing the parameters of the model.
        max_neighbors: Maximum number of neighbors considered in the
            computation of the spherical Bessel descriptors. This parameter
            is critical for the computation time of the descriptors.
        supercell_trafo: Diagonal elements of the diagonal supercell transfor-
            mation that is used for the descriptor computation. This option
            allows us to run nested sampling on cells, which would not fit the
            desired cutoff. The supercell is then only built during the
            descriptor computation to compute all relevant neighbors.

    Returns:
        Potential energy and Force evaluation functions.
    """
    model_info, dynamics_model = load_neuralil_model(
        pickle_file, max_neighbors, supercell_trafo
    )

    def calc_energy(positions, types, cell):
        energy = dynamics_model.apply(
            model_info.params,
            positions,
            types,
            cell,
            max_neighbors,
            method=dynamics_model.calc_potential_energy,
        )
        return energy.mean()

    def eval_energy(
        positions: jnp.ndarray, types: jnp.ndarray, cell: jnp.ndarray
    ) -> float:
        """Returns the potential energy. At each evaluation, we check whether
        the configuration is compatible with the selected max_neighbors and
        r_cut parameters.

        Args:
            positions: Atomic positions, shape=(n_atoms, 3).
            types: Types of the atoms, shape=(n_atoms, ).
            cell: Simulation box, shape=(3, 3).

        Returns:
            Potential energy of the configuration. If incompatible with
            max_neighbors or r_cut, it returns jnp.nan.
        """
        cutoff_crit = (
            _get_max_number_of_neighbors(
                positions, types, model_info.r_cut, cell, *supercell_trafo
            )
            > max_neighbors
        )
        neighbor_crit = (
            get_max_cutoff(cell * jnp.array(supercell_trafo))
            < model_info.r_cut
        )

        energy = jnp.where(
            jnp.logical_or(cutoff_crit, neighbor_crit),
            jnp.nan,
            calc_energy(positions, types, cell),
        )

        return energy

    def calc_forces(positions, types, cell):
        forces = dynamics_model.apply(
            model_info.params,
            positions,
            types,
            cell,
            max_neighbors,
            method=dynamics_model.calc_forces,
        )
        return forces.mean(axis=0)

    def eval_forces(
        positions: jnp.ndarray, types: jnp.ndarray, cell: jnp.ndarray
    ) -> jnp.ndarray:
        """Returns the forces acting on the atoms of a configuration.
        At each evaluation, we check whether the configuration is compatible
        with the selected max_neighbors and r_cut parameters.

        Args:
            positions: Atomic positions, shape=(n_atoms, 3).
            types: Types of the atoms, shape=(n_atoms, ).
            cell: Simulation box, shape=(3, 3).

        Returns:
            Forces for the given configuration. If incompatible with
            max_neighbors or r_cut, it returns an equivalent array filled with
            jnp.nan.
        """
        cutoff_crit = (
            _get_max_number_of_neighbors(
                positions, types, model_info.r_cut, cell, *supercell_trafo
            )
            > max_neighbors
        )
        neighbor_crit = (
            get_max_cutoff(cell * jnp.array(supercell_trafo))
            < model_info.r_cut
        )

        forces = jnp.where(
            jnp.logical_or(cutoff_crit, neighbor_crit),
            jnp.nan,
            calc_forces(positions, types, cell),
        )
        return forces

    return eval_energy, eval_forces


def evaluators_from_model(dynamics_model, params, max_neighbors):
    """Creates evaluation functions for the potential energy and the forces
    using the NeuralIL Neural-Network Force Field. We use PlainEnsemble models
    since they are almost as cheap as a single model.

    Args:
        dynamics_model: The neuralil PlainEnsemble model.
        params: The parameters of the model.
        max_neighbors: Maximum number of neighbors passed to the model's
            energy/force methods (the jaxrens fork takes it at call time).
    Returns:
        Potential energy and Force evaluation functions.
    """

    def calc_energy(positions, types, cell):
        return dynamics_model.apply(
            params,
            positions,
            types,
            cell,
            max_neighbors,
            method=dynamics_model.calc_potential_energy,
        )

    def calc_forces(positions, types, cell):
        return dynamics_model.apply(
            params,
            positions,
            types,
            cell,
            max_neighbors,
            method=dynamics_model.calc_forces,
        )

    return calc_energy, calc_forces
