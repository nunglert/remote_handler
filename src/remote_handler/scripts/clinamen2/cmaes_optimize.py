"""Gradient-based structure optimization with MACE models."""

"""Additional classes and functions needed for slab handling."""

import importlib
import pathlib
from typing import Callable, List, NamedTuple, Optional, Tuple, Union

import ase.atoms
import numpy as np
import numpy.typing as npt
from ase.build import sort
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms
from ase.geometry import get_duplicate_atoms
from ase.io import read as ase_read
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE
from ase.optimize.optimize import Optimizer
from clinamen2.utils.file_handling import CMAFileHandler
from clinamen2.utils.structure_setup import (
    AnchorToImmutable,
    DofToAtoms,
    create_apply_weights,
    create_transform_dof,
)
from mace.calculators import MACECalculator


def get_overlayer_sr(*args):
    raise NotImplementedError


def resolve_optimizer(class_name: str):
    mod = importlib.import_module("ase.optimize")

    try:
        cls = getattr(mod, class_name)
    except AttributeError as e:
        raise ValueError(
            f"Unknown optimizer '{class_name}'. "
            f"Expected a class in ase.optimize."
        ) from e

    if not isinstance(cls, type) or not issubclass(cls, Optimizer):
        raise ValueError(
            f"'{class_name}' exists in ase.optimize but is not an Optimizer subclass."
        )

    return cls


class AnchorToImmutableWithDof:
    """Class that appends immutable atoms to an atoms object.

    Args:
        immutable_atoms: Atoms that always have to be added to complete the
            transformation of a CMA-ES sample to an atoms object to perform
            calculations on.
        sort_atoms: If True, ase.build.sort() will be applied to the resulting
            atoms object.
    """

    def __call__(
        self, inputs: Tuple[ase.atoms.Atoms, npt.ArrayLike]
    ) -> ase.atoms.Atoms:
        """Extend the given atoms by the immutable atoms of the class.

        atoms: Atoms object to be extended.
        """
        atoms, dof = inputs
        result = atoms.copy() + self.immutable_atoms.copy()
        result.wrap()

        return sort(result) if self.sort_atoms else result

    def __init__(
        self, immutable_atoms: ase.atoms.Atoms, sort_atoms: bool = False
    ) -> None:
        """Constructor"""

        self.immutable_atoms = immutable_atoms.copy()
        self.sort_atoms = sort_atoms


class LossThresholdException(Exception):
    """Exception to be raised for loss below threshold.

    This exception is raised instead of checking
    min distances.
    """

    pass


class LossThresholdException(Exception):
    """Exception to be raised for loss below threshold.

    This exception is raised instead of checking
    min distances.
    """

    pass


class UncertaintyException(Exception):
    """Exception to be raised for exceeding uncertainty threshold."""

    pass


class SlabPipeline(NamedTuple):
    """Functions to be applied to CMA samples in reverse order.

    Args:
        anchor_to_immutable: Combine dof atoms with immutable atoms.
            Default is None, None function will be skipped.
        apply_slab_symmetry: Build the opposite side of the slab.
            Default is None, None function will be skipped.
        add_mirror_plane_atoms: Add all the atoms that are exactly on
            the mirror plane and part of the overlayer.
        add_reflection: Add reflected atoms on same side of slab.
            Default is None, None function will be skipped.
        dof_to_atoms: Construct an atoms object from a CMA-ES sample.
            Default is None, None function will be skipped.
        apply_weights: Dampen or amplify variation of variables in CMA degrees
            of freedem vector. Default is None, None function will be skipped.
    """

    anchor_to_immutable: Callable = None
    apply_slab_symmetry: Callable = None
    add_mirror_plane_atoms: Callable = None
    add_reflection: Callable = None
    dof_to_atoms: Callable = None
    apply_weights: Callable = None


class DofToAtomsReturnTuple:
    """Class that creates an atoms object from a compatible CMA-ES sample.

    The degrees of freedom making up the sample are translated into the
    positions of atoms.

    Args:
        template_atoms: Template of an atoms object.
    """

    def __call__(self, dof: npt.ArrayLike) -> ase.atoms.Atoms:
        """Copy the atoms template and take positions from dof.

        Args:
            dof: Degrees of freedom sampled from the CMA-ES.
        """
        atoms = self.template_atoms.copy()
        positions = dof.reshape((-1, 3))
        atoms.positions[: positions.shape[0]] = positions
        atoms.wrap()

        return atoms, dof

    def __init__(self, template_atoms: ase.atoms.Atoms) -> None:
        """Constructor"""

        del template_atoms.constraints
        del template_atoms.constraints
        self.template_atoms = template_atoms.copy()


class DofToAtomsWithoutMirror:
    """Class that creates an atoms object from a compatible CMA-ES sample.

    The degrees of freedom making up the sample are translated into the
    positions of atoms. Dofs that are part of the overlayer and lie
    on the mirror plane get padded with the respective coordinate.

    Args:
        template_atoms: Template of an atoms object.
        mirror_pad_start: First index of fixed atom.
        mirror_axis: Index of the mirror axis.
        mirror_position: Position on mirror axis.
    """

    # TODO: Refactor!
    def __call__(self, dof: npt.ArrayLike) -> ase.atoms.Atoms:
        """Copy the atoms template and take positions from dof.

        Args:
            dof: Degrees of freedom sampled from the CMA-ES.
        """
        atoms = self.template_atoms.copy()

        main_dof = dof[: self.mirror_plane_start]
        main_pos = main_dof.reshape((-1, 3))
        atoms.positions = main_pos

        atoms.wrap()

        return atoms, dof

    def __init__(
        self,
        template_atoms: ase.atoms.Atoms,
        mirror_plane_atoms: ase.atoms.Atoms,
        mirror_plane_start: int,
        mirror_axis: int,
        mirror_position: float,
    ) -> None:
        """Constructor"""

        del template_atoms.constraints
        del template_atoms.constraints
        self.template_atoms = template_atoms.copy()
        self.mirror_plane_atoms = mirror_plane_atoms.copy()
        self.mirror_plane_start = mirror_plane_start
        self.mirror_axis = mirror_axis
        self.dof_axes = [0, 1, 2]
        self.dof_axes.remove(self.mirror_axis)
        self.mirror_position = mirror_position

        # the mirror plane atoms need to be exactly on the plane
        self.mirror_plane_atoms.positions[
            :, self.mirror_axis
        ] = self.mirror_position
        self.mirror_plane_atoms.positions[
            :, self.mirror_axis
        ] = self.mirror_position


def prepare_dof_and_pipeline(
    founder_filename: Union[str, ase.atoms.Atoms],
    surface_normal_axis: int = 2,
    cutoff: float = 0.6,
    bulk_range: Tuple[float, float] = (0.4, 0.6),
    srol_cutoff: float = None,
) -> Tuple[npt.ArrayLike, Callable]:
    """Wrapper for backwards compatibility"""
    cutoffs = [0.0, 0.0, 0.0]
    cutoffs[surface_normal_axis] = cutoff

    # ensures backwards compatibility (needed to change "<" to ">")
    if cutoff > 0.5:
        fun = prepare_dof_and_pipeline_legacy
    else:
        fun = prepare_dof_and_pipeline_with_reflection

    return fun(
        founder_filename=founder_filename,
        surface_normal_axis=surface_normal_axis,
        cutoffs=cutoffs,
        bulk_range=bulk_range,
        srol_cutoff=srol_cutoff,
    )


def prepare_dof_and_pipeline_legacy(
    founder_filename: Union[str, ase.atoms.Atoms],
    surface_normal_axis: int = 2,
    cutoffs: float = [0.0, 0.0, 0.6],
    bulk_range: Tuple[float, float] = (0.4, 0.6),
    srol_cutoff: float = None,
) -> Tuple[npt.ArrayLike, Callable]:
    """Setup of surface slab"""

    if isinstance(founder_filename, ase.atoms.Atoms):
        slab = founder_filename.copy()
    else:
        slab = ase_read(founder_filename)
    del slab[
        [
            atom.index
            for atom in slab
            if atom.scaled_position[surface_normal_axis]
            < cutoffs[surface_normal_axis]
        ]
    ]
    if srol_cutoff is not None:
        del slab[
            [
                atom.index
                for atom in slab
                if atom.scaled_position[surface_normal_axis] > srol_cutoff
                and atom.symbol == "Sr"
            ]
        ]
    if srol_cutoff is not None:
        del slab[
            [
                atom.index
                for atom in slab
                if atom.scaled_position[surface_normal_axis] > srol_cutoff
                and atom.symbol == "Sr"
            ]
        ]
    dof = slab.positions.flatten()

    if isinstance(founder_filename, ase.atoms.Atoms):
        bulk = founder_filename.copy()
    else:
        bulk = ase_read(founder_filename)
    bulk = get_bulk(
        atoms=bulk,
        bulk_range=bulk_range,
        surface_normal_axis=surface_normal_axis,
    )
    if srol_cutoff is not None:
        srol = ase_read(founder_filename)
        srol = get_overlayer_sr(
            atoms=srol,
            surface_normal_axis=surface_normal_axis,
            cutoff=srol_cutoff,
        )
        bulk.extend(srol)
    if srol_cutoff is not None:
        srol = ase_read(founder_filename)
        srol = get_overlayer_sr(
            atoms=srol,
            surface_normal_axis=surface_normal_axis,
            cutoff=srol_cutoff,
        )
        bulk.extend(srol)
    anchor_to_immutable = AnchorToImmutable(
        immutable_atoms=bulk, sort_atoms=True
    )
    apply_mirror_symmetry = create_apply_mirror_symmetry(
        rotation=np.eye(3) * (-1.0),
        translation=np.array([0.0, 0.0, 0.0]),  # no translation needed here
    )
    dof_to_atoms = DofToAtoms(template_atoms=slab)
    apply_weights = create_apply_weights(weights=np.ones(shape=dof.shape))
    pipeline = SlabPipeline(
        apply_weights=apply_weights,
        dof_to_atoms=dof_to_atoms,
        apply_slab_symmetry=apply_mirror_symmetry,
        anchor_to_immutable=anchor_to_immutable,
    )
    transform_dof = create_transform_dof(pipeline=pipeline)

    return dof, transform_dof, slab


def prepare_dof_and_pipeline_with_reflection(
    founder_filename: Union[str, ase.atoms.Atoms],
    surface_normal_axis: int = 2,
    cutoffs: list[float] = [0.0, 0.0, 0.4],
    bulk_range: Tuple[float, float] = (0.4, 0.6),
    scaled_plane_position: float = None,
    mirror_plane_position: float = None,
    mirror_axis: int = None,
    glide: bool = False,
    glide_axis: int = None,
    rotation: npt.ArrayLike = None,
    translation: npt.ArrayLike = np.asarray([0.0, 0.0, 0.0]),
    srol_cutoff: float = None,
) -> Tuple[npt.ArrayLike, Callable]:
    """Setup of surface slab

    - Handles one mirror or glide plane
    - Fix atoms on mirror plane (2 dofs instead of 3)
    """
    if isinstance(founder_filename, ase.atoms.Atoms):
        slab = founder_filename.copy()
    else:
        slab = ase_read(founder_filename)
    for c, cutoff in enumerate(cutoffs):
        if cutoff > 0.0:
            del slab[
                [
                    atom.index
                    for atom in slab
                    if atom.scaled_position[c] >= cutoff
                ]
            ]
    if glide:
        del slab[
            [
                atom.index
                for atom in slab
                if atom.scaled_position[glide_axis] < 0.0
            ]
        ]
    if srol_cutoff is not None:
        del slab[
            [
                atom.index
                for atom in slab
                if atom.scaled_position[surface_normal_axis] < srol_cutoff
                and atom.symbol == "Sr"
            ]
        ]
    if srol_cutoff is not None:
        del slab[
            [
                atom.index
                for atom in slab
                if atom.scaled_position[surface_normal_axis] < srol_cutoff
                and atom.symbol == "Sr"
            ]
        ]
    # specifically remove mirror plane atoms
    # those may be very close to the plane, not always fully on it
    if mirror_axis is not None:
        del slab[
            [
                atom.index
                for atom in slab
                if np.isclose(
                    atom.scaled_position[mirror_axis],
                    scaled_plane_position,
                    atol=1e-5,
                )
            ]
        ]
        # specifically remove boundary atoms (along mirror plane axis)
        del slab[
            [
                atom.index
                for atom in slab
                if np.isclose(
                    atom.scaled_position[mirror_axis],
                    0.0,
                    atol=1e-5,
                )
            ]
        ]
    dof = slab.positions.flatten()

    if mirror_axis is not None:
        # read the full struture again to avoid cumulative errors
        mirror_plane_atoms = ase_read(founder_filename).copy()
        for c, cutoff in enumerate(cutoffs):
            if cutoff > 0.0 and c != mirror_axis:
                del mirror_plane_atoms[
                    [
                        atom.index
                        for atom in mirror_plane_atoms
                        if atom.scaled_position[c] >= cutoff
                    ]
                ]
        if srol_cutoff is not None:
            del mirror_plane_atoms[
                [
                    atom.index
                    for atom in mirror_plane_atoms
                    if atom.scaled_position[surface_normal_axis] < srol_cutoff
                    and atom.symbol == "Sr"
                ]
            ]
        if srol_cutoff is not None:
            del mirror_plane_atoms[
                [
                    atom.index
                    for atom in mirror_plane_atoms
                    if atom.scaled_position[surface_normal_axis] < srol_cutoff
                    and atom.symbol == "Sr"
                ]
            ]

        # remove all atoms that are not on the boundary or mirror plane
        del mirror_plane_atoms[
            [
                atom.index
                for atom in mirror_plane_atoms
                if (
                    (
                        not np.isclose(
                            atom.scaled_position[mirror_axis],
                            0.0,
                            atol=1e-5,
                        )
                    )
                    and (
                        not np.isclose(
                            atom.scaled_position[mirror_axis],
                            scaled_plane_position,
                            atol=1e-5,
                        )
                    )
                )
            ]
        ]
        mirror_dof_from = dof.shape[0]
        mirror_dof_indices = [0, 1, 2]
        mirror_dof_indices.remove(mirror_axis)
        dof = np.concatenate(
            [
                dof,
                mirror_plane_atoms.positions[:, mirror_dof_indices].flatten(),
            ]
        )
        combined_slab = slab.copy()
        for atom in mirror_plane_atoms:
            combined_slab.append(atom)
    else:
        combined_slab = slab.copy()

    if isinstance(founder_filename, ase.atoms.Atoms):
        bulk = founder_filename.copy()
    else:
        bulk = ase_read(founder_filename)
    bulk = get_bulk(
        atoms=bulk,
        bulk_range=bulk_range,
        surface_normal_axis=surface_normal_axis,
    )
    if srol_cutoff is not None:
        srol = ase_read(founder_filename)
        srol = get_overlayer_sr(
            atoms=srol,
            surface_normal_axis=surface_normal_axis,
            cutoff=(
                1 - srol_cutoff
            ),  # srol_cutoff is a lower bound for mirror and an upper bound for full
        )
        bulk.extend(srol)
    anchor_to_immutable = AnchorToImmutableWithDof(
        immutable_atoms=bulk, sort_atoms=True
    )
    apply_mirror_symmetry = create_apply_mirror_symmetry(
        rotation=np.eye(3) * (-1.0),
        translation=np.array([0.0, 0.0, 0.0]),
        sort_atoms=True,
    )
    if mirror_axis is None:
        add_mirror_plane_atoms = None
        add_reflection = None
        if glide:
            add_reflection = create_apply_glide(
                rotation=rotation, translation=translation, sort_atoms=True
            )
    else:
        add_mirror_plane_atoms = create_add_mirror_plane_atoms(
            mirror_plane_atoms=mirror_plane_atoms,
            mirror_plane_start=mirror_dof_from,
            mirror_plane_position=mirror_plane_position,
            reflection_rotation_axis=mirror_axis,
        )
        add_reflection = create_apply_reflection(
            rotation=rotation,
            translation=translation,
            reflection_rotation_axis=mirror_axis,
            sort_atoms=True,
        )
    if mirror_axis is not None:
        dof_to_atoms = DofToAtomsWithoutMirror(
            template_atoms=slab,
            mirror_plane_atoms=mirror_plane_atoms,
            mirror_plane_start=mirror_dof_from,
            mirror_axis=mirror_axis,
            mirror_position=mirror_plane_position,
        )
    else:
        dof_to_atoms = DofToAtomsReturnTuple(template_atoms=slab)
    apply_weights = create_apply_weights(weights=np.ones(shape=dof.shape))
    pipeline = SlabPipeline(
        apply_weights=apply_weights,
        dof_to_atoms=dof_to_atoms,
        add_reflection=add_reflection,
        add_mirror_plane_atoms=add_mirror_plane_atoms,
        apply_slab_symmetry=apply_mirror_symmetry,
        anchor_to_immutable=anchor_to_immutable,
    )
    transform_dof = create_transform_dof(pipeline=pipeline)

    return dof, transform_dof, slab  # combined_slab


def create_add_mirror_plane_atoms(
    mirror_plane_atoms: ase.atoms.Atoms = None,
    mirror_plane_start: int = None,
    mirror_plane_position: float = None,
    mirror_dof_indices: npt.ArrayLike = None,
    reflection_rotation_axis: int = None,
    # sort_atoms: bool = False, # not optional in this step
) -> Callable:
    """Function to create closure that fixes atoms to mirror plane."""

    mirror_dof_indices = [0, 1, 2]
    mirror_dof_indices.remove(reflection_rotation_axis)

    def add_mirror_plane_atoms(
        inputs: Tuple[ase.atoms.Atoms, npt.ArrayLike],
    ) -> ase.atoms.Atoms:
        """Fix respective coordinate to mirror plane."""
        atoms, dof = inputs

        fixed_copy = mirror_plane_atoms.copy()

        mirror_plane_dof = dof[mirror_plane_start:]
        mirror_plane_pos = mirror_plane_dof.reshape((-1, 2))
        fixed_copy.positions[:, mirror_dof_indices] = mirror_plane_pos

        result = atoms.copy()
        for at in fixed_copy:
            result.append(at)

        result = sort(result)

        return result, dof

    return add_mirror_plane_atoms


def create_apply_mirror_symmetry(
    rotation: npt.ArrayLike,
    translation: npt.ArrayLike,
    invert_magmoms: bool = False,
    sort_atoms: bool = False,
) -> Callable:
    """Function to create closure that applies mirror symmetry to a slab.

    Args:
        rotation: Rotation matrix to define symmetric mirror positions,
            acting on scaled coordinates.
        translation: (Fractional) translation vector to define symmetric
            mirror positions, given in scaled coordinates.
        invert_magmoms: Whether the image atoms under glide plane symmetry
            that are added to the original input structure should have their
            magnetic moments inverted (multiplied by minus 1) compared to the
            original input structure's or not. While the resulting structure
            will satisfy the imposed symmetry either way, the two choices
            represent different physical systems.
    """

    def apply_mirror_symmetry(
        inputs: Tuple[ase.atoms.Atoms, npt.ArrayLike],
    ) -> ase.atoms.Atoms:
        """Copy all present atoms and adapt their positions.

        Args:
            atoms: Atoms object containing half a slab (or the outer layers of
                half a slab).
        """
        try:
            atoms, dof = inputs
        except ValueError:
            atoms = inputs
            dof = None

        # TODO: Do we need to copy here?
        result = atoms.copy()
        mirror = atoms.copy()

        mirror.set_scaled_positions(
            mirror.get_scaled_positions() @ rotation.T + translation
        )
        if invert_magmoms:
            mirror.set_initial_magnetic_moments(
                -mirror.get_initial_magnetic_moments()
            )
        result += mirror
        result.wrap()

        if sort_atoms:
            result = sort(result)

        if dof is not None:
            result = (result, dof)

        return result

    return apply_mirror_symmetry


def create_apply_glide(
    rotation: npt.ArrayLike,
    translation: npt.ArrayLike,
    sort_atoms: bool = False,
) -> Callable:
    """Wrapper to return mirror sym closure for glide plane."""

    apply_mirror_sym = create_apply_mirror_symmetry(
        rotation=rotation,
        translation=translation,
        invert_magmoms=False,
        sort_atoms=sort_atoms,
    )

    def apply_glide(
        inputs: Tuple[ase.atoms.Atoms, npt.ArrayLike],
    ) -> Tuple[ase.atoms.Atoms, npt.ArrayLike]:
        atoms, dof = apply_mirror_sym(inputs)
        get_duplicate_atoms(atoms, cutoff=1e-4, delete=True)

        return atoms, dof

    return apply_glide


def create_apply_reflection(
    rotation: npt.ArrayLike,
    translation: npt.ArrayLike,
    reflection_rotation_axis: int,
    sort_atoms: bool = False,
) -> Callable:
    """Closure to create function that adds reflections of atoms.

        Takes care of duplicates.

    Args:
        rotation: Rotation matrix to define symmetric mirror positions,
            acting on scaled coordinates.
        translation: (Fractional) translation vector to define symmetric
            mirror positions, given in scaled coordinates.
    """

    def apply_reflection(
        inputs: Tuple[ase.atoms.Atoms, npt.ArrayLike],
    ) -> Tuple[ase.atoms.Atoms, npt.ArrayLike]:
        """Copy all present atoms and adapt their positions.

        Args:
            atoms: Overlayer atoms.
        """
        atoms, dof = inputs

        result = atoms.copy()
        mirror = atoms.copy()

        mirror.set_scaled_positions(
            mirror.get_scaled_positions() @ rotation.T + translation
        )

        result += mirror
        result.wrap()

        if sort_atoms:
            result = sort(result)

        return result, dof

    return apply_reflection


def get_bulk(
    atoms: ase.atoms.Atoms,
    bulk_range: Tuple[float, float],
    surface_normal_axis: int,
) -> ase.atoms.Atoms:
    """Get bulk from atoms."""

    bulk = atoms.copy()
    del bulk[
        [
            atom.index
            for atom in bulk
            if atom.scaled_position[surface_normal_axis] > bulk_range[1]
        ]
    ]
    del bulk[
        [
            atom.index
            for atom in bulk
            if atom.scaled_position[surface_normal_axis] < bulk_range[0]
        ]
    ]

    return bulk


def create_uncertainty_filter(
    uncertainty_threshold: float = None,
    uncertainty_exception: Exception = BaseException,
) -> Callable:
    """Create function to filter configurations.

    The filter is applied with regards to relative uncertainty within
    the samples. An exception is added to the configurations over the
    threshold.

    Args:
        - uncertainty_threshold: Absolute value of uncertainty that may not be
            exceeded. Default is None, filters will be identity functions.
    """

    if uncertainty_threshold is None:
        return lambda *x: x, lambda *x: x

    else:

        def batch_uncertainty_filter(
            loss: float, additional: list, inputs: Tuple
        ) -> Tuple[float, list]:
            for a in additional:
                uncertainty = a["uncertainty"]
                if uncertainty > uncertainty_threshold:
                    a["exception"] = uncertainty_exception(
                        f"Uncertainty {uncertainty} "
                        f"over threshold of {uncertainty_threshold}."
                    )

            return loss, additional, inputs

        def single_uncertainty_filter(
            loss: float, additional: list, inputs: Tuple
        ):
            uncertainty = additional["uncertainty"]
            if uncertainty > uncertainty_threshold:
                additional["exception"] = uncertainty_exception(
                    f"Uncertainty {uncertainty} "
                    f"over threshold of {uncertainty_threshold}."
                )

            return loss, additional, inputs

        return single_uncertainty_filter, batch_uncertainty_filter


def create_lossthreshold_filter(
    loss_threshold: float = None,
    loss_threshold_exception: Exception = BaseException,
) -> Callable:
    """Create function to filter configurations.

    The filter is applied with regards to an absolute loss threshold.
    This is a lower bound that may not be exceeded.
    "Poor man's min distance exception."

    Args:
        - loss_threshold: Absolute value of lower loss bound.
            Default is None, filters will be identity functions.
    """

    if loss_threshold is None:
        return lambda *x: x, lambda *x: x

    else:

        def batch_lossthreshold_filter(
            loss: float, additional: list, inputs: Tuple
        ) -> Tuple[float, list]:
            for a in additional:
                check_val = a["loss"]
                if check_val < loss_threshold:
                    a["exception"] = loss_threshold_exception(
                        f"Loss {check_val} "
                        f"below threshold of {loss_threshold}."
                    )

            return loss, additional, inputs

        def single_lossthreshold_filter(
            loss: float, additional: list, inputs: Tuple
        ):
            if loss < loss_threshold:
                additional["exception"] = loss_threshold_exception(
                    f"Loss {loss} " f"below threshold of {loss_threshold}."
                )

            return loss, additional, inputs

        return single_lossthreshold_filter, batch_lossthreshold_filter


def create_optimize_fun(
    save_path: pathlib.Path,
    dof_to_atoms: Callable,
    calc: Calculator,
    surface_normal_axis: int = 2,
    fix_range: Tuple[float, float] = None,
    pbc: npt.ArrayLike = np.asarray([1, 1, 0]),
    additional_constraints: list[ase.constraints.FixConstraint] = None,
) -> Callable:
    """Return function that optimizes a given slab"""

    def optimize_fun(
        dof: npt.ArrayLike,
        label: str,
        fmax: float = 0.001,
        steps: int = 1000,
        write_poscars: bool = True,
        direct_atoms: ase.atoms.Atoms = None,
        bulk_terminate: bool = False,
        optimizer: str = "FIRE",
    ) -> pathlib.Path:
        """Recreates dof to atoms and optimizes"""
        if direct_atoms is None:
            atoms = dof_to_atoms(dof)
        else:
            atoms = direct_atoms
        if bulk_terminate:
            del atoms[
                [
                    atom.index
                    for atom in atoms
                    if atom.scaled_position[surface_normal_axis] < fix_range[0]
                ]
            ]

        atoms.calc = calc
        atoms.set_pbc(pbc)

        if write_poscars:
            write(
                filename=save_path / f"POSCAR_{label.replace('.', '')}_start",
                images=atoms,
                format="vasp",
                direct=True,
                vasp5=True,
            )

        if fix_range is not None:
            scaled_positions = atoms.get_scaled_positions()
            c = FixAtoms(
                mask=[
                    (
                        scaled_positions[i][surface_normal_axis] > fix_range[0]
                        and scaled_positions[i][surface_normal_axis]
                        < fix_range[1]
                    )
                    for i, _ in enumerate(atoms)
                ]
            )
            atoms.set_constraint(c)

        # does appending work if the constraints are empty? Would expect it to.
        if direct_atoms is not None and additional_constraints is not None:
            for ac in additional_constraints:
                atoms.constraints.append(ac)

        print(atoms.constraints)

        optimizer = resolve_optimizer(optimizer)
        print(optimizer)

        relax = optimizer(
            atoms=atoms,
            trajectory=str(save_path / f"{label}.{optimizer.__name__}.traj"),
            restart=str(save_path / f"{label}.{optimizer.__name__}.pckl"),
        )
        relax.run(fmax=fmax, steps=steps)
        write_atoms = Trajectory(
            save_path / f"{label}.{optimizer.__name__}.traj"
        )[-1]

        if write_poscars:
            output_file_path = (
                save_path / f"POSCAR_{label.replace('.', '')}_end"
            )
            write(
                filename=output_file_path,
                images=write_atoms,
                format="vasp",
                direct=True,
                vasp5=True,
            )
        else:
            output_file_path = None

        return output_file_path

    return optimize_fun


# very rudimentary way to get rid of any references to scratch
# TODO: have evolutions store the correct (=target) path from the get go
def get_rid_of_scratch(founder: str, data: str) -> str:
    if str(founder).startswith("/scratch"):
        return "/".join([str(data), "/".join((str(founder).split("/")[3:]))])
    else:
        return founder


def dof_to_atoms_from_evo(
    label,
    path,
    dof_mode,
    override_founder=None,
    choose_gen=None,
    choose_ind=-1,
    cutoff=0.7,
    bulk_range=(0.3, 0.7),
    handler=None,
    scaled_center=None,
    radius=None,
):
    if handler is None:
        handler = CMAFileHandler(
            label=label, target_dir=pathlib.Path(path / label)
        )

    try:
        evolution = handler.load_evolution()
        # founder = get_rid_of_scratch(str(evolution[-1]["founder"]), f"{path}") # TODO: NU
        founder = str(evolution[-1]["founder"])
        founder = path / pathlib.Path(founder)
    except KeyError:
        founder = pathlib.Path(override_founder)

    if dof_mode == "full-dof":
        # Must rebuild the SAME degrees of freedom the evolution used
        # (run_evolution -> clinamen2's prepare_dof_and_pipeline with
        # scaled_center/radius), not the local surface-slab wrapper -- otherwise
        # the reconstruction weights mismatch the optimized DOF.
        from clinamen2.utils.structure_setup import (
            prepare_dof_and_pipeline as prep_full,
        )

        _, transform_dof, _ = prep_full(
            founder,
            scaled_center=scaled_center,
            radius=radius,
        )
    elif dof_mode == "defect":
        # assert scaled_center is not None
        # assert radius is not None

        from clinamen2.utils.structure_setup import (
            prepare_dof_and_pipeline as prep_defect,
        )

        _, transform_dof, _ = prep_defect(
            founder,
            scaled_center=scaled_center,
            radius=radius,
        )
    else:
        raise NotImplementedError("No")

    evolution = handler.load_evolution()
    if choose_gen is None:
        last_gen = evolution[-1]["last_gen"]
    else:
        last_gen = choose_gen
    generation = handler.load_generation(last_gen)

    if choose_ind is None:
        atoms = transform_dof(generation[0].mean)
    elif choose_ind == -1:
        i = generation[2].argmin()
        atoms = transform_dof(generation[1][i])
    else:
        atoms = transform_dof(generation[1][choose_ind])

    return atoms, handler, last_gen, transform_dof


def create_E_surf_BaTiO3(
    A: float,
    N_1x1: int,
    E_BaTiO3: float,
    E_Ti: float,
    E_O: float,
):
    A_fact = 1.0 / A

    def E_surf(
        E_slab: float,
        delta_Ti: int,
        delta_O: int,
    ):
        return A_fact * (
            0.5 * E_slab - N_1x1 * E_BaTiO3 - delta_Ti * E_Ti - delta_O * E_O
        )

    return E_surf


def run_optimization(
    models: List[str],
    output_path: str,
    surface_axis: int = 2,
    label: str = "dummy",
    path: Optional[str] = None,
    dof_mode: str = "full-dof",
    choose_gen: Optional[int] = None,
    fix_range: Optional[list] = None,
    bulk_terminate: bool = False,
    optimizer: str = "FIRE",
    override_founder: Optional[str] = None,
):
    """Perform gradient-based structure optimization.

    Args:
        models: List of MACE model file paths.
        output_path: Path to save optimization results to.
        surface_axis: Surface normal axis (0, 1 or 2).
        label: Evolution label (C2) or output label (ASE).
        path: Evolution main path (required for C2 mode).
        dof_mode: Degree-of-freedom mode (C2 mode).
        choose_gen: Optimize the best individual of this generation (C2 mode).
        fix_range: Scaled range of atoms to fix along surface normal.
        bulk_terminate: If True, remove the overlayer from the bottom layer.
        optimizer: Name of ASE optimizer class (e.g. 'FIRE', 'BFGS').
        override_founder: Override founder structure path (C2 mode).
    """
    if fix_range is None:
        fix_range = np.asarray([0.3, 0.7])
    else:
        fix_range = np.asarray(fix_range)

    calc = MACECalculator(
        model_paths=[model for model in models],
        device="cuda",
        default_dtype="float32",
    )
    print(f"{calc.num_models} models loaded.")

    _, handler, last_gen, transform_dof = dof_to_atoms_from_evo(
        label=label,
        path=pathlib.Path(path),
        dof_mode=dof_mode,
        override_founder=override_founder,
        choose_gen=choose_gen,
    )
    dof_to_atoms = transform_dof

    optimace = create_optimize_fun(
        save_path=pathlib.Path(output_path),
        dof_to_atoms=dof_to_atoms,
        calc=calc,
        surface_normal_axis=surface_axis,
        fix_range=fix_range,
        pbc=np.asarray([1, 1, 0]),
    )

    if optimace is None:
        raise ValueError("Optimization function is None.")

    last_generation = handler.load_generation(generation=last_gen)
    winner_index = np.argmin(last_generation[2])
    winner_dof = last_generation[1][winner_index]
    print("Starting optimization")
    optimace(
        winner_dof,
        label=f"{label}_winner",
        bulk_terminate=bulk_terminate,
        optimizer=optimizer,
    )
    print("Optimization is done")


if __name__ == "__main__":
    import sys

    from remote_handler.scripts.input_models import CmaesOptimizeInput

    calcpath = pathlib.Path(sys.argv[1]).parent
    INPUTFILE = pathlib.Path(sys.argv[1])
    input_config = CmaesOptimizeInput.from_file(INPUTFILE)
    input_config.validate(calcpath)

    # Load models once for all evolutions
    calc = MACECalculator(
        model_paths=input_config.models,
        device="cuda",
        default_dtype="float32",
    )
    print(f"{calc.num_models} models loaded.")

    fix_range = (
        np.asarray(input_config.fix_range)
        if input_config.fix_range is not None
        else np.asarray([0.3, 0.7])
    )

    output_base = calcpath / input_config.output_dir
    output_base.mkdir(parents=True, exist_ok=True)

    for evo in input_config.evolutions:
        label = evo["label"]
        evo_path = pathlib.Path(evo["path"])
        dof_mode = evo.get("dof_mode", "full-dof")
        choose_gen = evo.get("choose_gen", None)
        override_founder = evo.get("override_founder", None)
        scaled_center = evo.get("scaled_center", None)
        radius = evo.get("radius", None)

        print(f"\n--- Optimizing evolution: {label} ---")

        try:
            _, handler, last_gen, transform_dof = dof_to_atoms_from_evo(
                label=label,
                path=evo_path,
                dof_mode=dof_mode,
                override_founder=override_founder,
                choose_gen=choose_gen,
                scaled_center=scaled_center,
                radius=radius,
            )

            evo_output = output_base / label
            evo_output.mkdir(parents=True, exist_ok=True)

            optimace = create_optimize_fun(
                save_path=evo_output,
                dof_to_atoms=transform_dof,
                calc=calc,
                surface_normal_axis=input_config.surface_axis,
                fix_range=fix_range,
                pbc=np.asarray([1, 1, 0]),
            )

            if optimace is None:
                raise ValueError(f"Optimization function is None for {label}")

            last_generation = handler.load_generation(generation=last_gen)
            winner_index = np.argmin(last_generation[2])
            winner_dof = last_generation[1][winner_index]

            optimace(
                winner_dof,
                label=f"{label}_winner",
                bulk_terminate=input_config.bulk_terminate,
                optimizer=input_config.optimizer,
            )
            print(f"Optimization done for {label}")

        except Exception as e:
            print(f"ERROR optimizing {label}: {e}")
            continue

    print("\nAll optimizations completed.")
