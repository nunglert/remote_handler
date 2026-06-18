"""Supercell / cutoff helpers needed by NeuralIL training.

Vendored from ``jaxnest.structure_utils`` so that the remote-side NeuralIL
training payload (:mod:`remote_handler.scripts.neuralil.train_model`) no longer
depends on the ``jaxnest`` package. Only the small closure used by training is
copied here -- the supercell search (``find_super`` and helpers) and the
maximum-cutoff calculation (``get_max_cutoff``).
"""

from types import ModuleType

import jax.numpy as jnp
import numpy as np
from ase.geometry import minkowski_reduce


def get_max_cutoff(cell: np.ndarray, nplib: ModuleType = jnp) -> float:
    """
    Get the maximum cutoff value that fits into a 3D cell.

    Args:
        cell: Cell, shape=(3, 3).

    Returns:
        Cutoff radius
    """
    return nplib.min(
        nplib.fabs(
            nplib.linalg.det(cell)  # Volume of the cell
            / nplib.array(
                [
                    nplib.linalg.norm(
                        nplib.cross(i, j)
                    )  # Compute area of face spanned by i and j
                    for i, j in zip(
                        cell, nplib.roll(cell, 1, axis=0)
                    )  # Create combinations (a,b), (b,c), (c,a)
                ]
            )
        )
        / 2.0
    )


def get_HNF_diagonals_nb(detS: int) -> np.ndarray:
    """Finds the diagonal transformation matrices which share the determinant
    value detS. E.g. for detS = 12, there are several transformations like
    [3,2,2], [2,2,3], [4,1,3], etc..

    Args:
        detS: Value for the transformation matrix determinant.

    Returns:
        Array containing the diagonals of the N diagonal supercell
        transformations for which the determinant equals detS. This has shape
        (N, 3)
    """
    id = -1
    tempDiag = np.zeros((3, detS * 3))
    for i in range(1, detS + 1):
        if not detS % i == 0:
            continue
        quotient = detS // i
        for j in range(1, quotient + 1):
            if quotient % j == 0:
                id += 1
                tempDiag[:, id] = [i, j, quotient // j]
    return tempDiag[:, 0 : id + 1].T


def find_super_var_aspect(
    cell: np.ndarray,
    target_cutoff: float,
    max_det: int = 72,
    minkowski: bool = False,
):
    """
    For a given cell, this finds a supercell that can host the target
    cutoff. We only consider diagonal transformations.
    Under the hood this iterates over increasing values of the determinant
    of diagonal supercell transformations detS. For each possible realization
    of detS (see `get_HNF_diagonals_nb` for more details), the cutoff is
    computed and checked whether the maximum of those cutoffs is larger than
    the provided `target_cutoff`.

    Args:
        cell: Array containing cell vectors C.T
        target_cutoff: Cutoff that should fit into the cell
        max_det: Maximum value for the determinant of the cell transformation.
            This corresponds to the number of replications that the
            transformation will cause. E.g. [3,3,3] -> detS = 27
        minkowski: This optionally performs a minkowski reduction on the
            transformed cell before its cutoff is determined. This might
            reduce the required size of the supercell.

    Returns:
        The diagonal elements of a diagonal supercell transformation of
        shape (3,).
    """
    for i in range(1, max_det + 1):
        rcuts = []
        # Get all possible diagonal transformations with determinant i
        permutations = get_HNF_diagonals_nb(i)
        for j in permutations:
            supercell = cell * np.array([j]).T

            if minkowski:
                op = minkowski_reduce(supercell)[1]
                new_cell = op @ supercell
            else:
                new_cell = supercell

            cutoff = get_max_cutoff(new_cell, nplib=np)
            rcuts.append(cutoff)

        # Find the maximum cutoff and its index
        maximum = np.argmax(np.array(rcuts))

        # If the maximum cutoff fits, return the transformation that caused it
        if rcuts[maximum] > target_cutoff:
            return np.array(permutations[maximum], dtype=np.int32)
        if i == max_det:
            raise ValueError(
                f"Could not find transformation within {max_det=}."
            )


def find_super_equal_aspect(
    cell: np.ndarray,
    target_cutoff: float,
    max_det: int = 72,
    minkowski: bool = False,
):
    """For a given cell, this finds a supercell that can host the target
    cutoff. This considers only diagonal supercell transformation [s,s,s].

    Args:
        cell: Array containing cell vectors C.T
        target_cutoff: Cutoff that should fit into the cell
        max_det: Maximum value for the determinant of the cell transformation.
            This corresponds to the number of replications that the
            transformation will cause. E.g. [3,3,3] -> detS = 27
        minkowski: This optionally performs a minkowski reduction on the
            transformed cell before its cutoff is determined. This might
            reduce the required size of the supercell.

    Returns:
        The diagonal elements of a diagonal supercell transformation of
        shape (3,).
    """
    # Get the maximum supercell dimension to consider
    sc_dim_max = int(np.floor(max_det ** (1 / 3)))
    for sc_dim in range(1, sc_dim_max + 1):
        # Create the supercell
        supercell = cell * sc_dim

        # Optionally apply minkowski reduction
        if minkowski:
            op = minkowski_reduce(supercell)[1]
            new_cell = op @ supercell
        else:
            new_cell = supercell

        # Compute the cutoff of the new cell
        cutoff = get_max_cutoff(new_cell, nplib=np)

        # If the cutoff fits, return the transformation
        if cutoff > target_cutoff:
            return np.ones(3, dtype=np.int32) * sc_dim

    raise ValueError(f"Could not find transformation within {max_det=}.")


def find_super(
    cell: np.ndarray,
    target_cutoff: float,
    max_det: int = 1000,
    minkowski: bool = False,
    equal_aspect: bool = False,
):
    """
    For a given cell, this finds a supercell that can host the target
    cutoff. This can either consider only diagonal transformations with
    equal aspect ratio [s,s,s] or variable aspect ratio [s1,s2,s3].

    Args:
        cell: Array containing cell vectors C.T
        target_cutoff: Cutoff that should fit into the cell
        max_det: Maximum value for the determinant of the cell transformation.
            This corresponds to the number of replications that the
            transformation will cause. E.g. [3,3,3] -> detS = 27
        minkowski: This optionally performs a minkowski reduction on the
            transformed cell before its cutoff is determined. This might
            reduce the required size of the supercell.
        equal_aspect: Whether to consider only diagonal transformations with
            equal aspect ratio [s,s,s] or variable aspect ratio [s1,s2,s3].

    Returns:
        The diagonal elements of a diagonal supercell transformation of
        shape (3,).
    """
    if equal_aspect:
        return find_super_equal_aspect(cell, target_cutoff, max_det, minkowski)
    else:
        return find_super_var_aspect(cell, target_cutoff, max_det, minkowski)
