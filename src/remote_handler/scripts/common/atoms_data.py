"""Shared ASE-database helpers, used by more than one training payload.

Originally lived only in ``scripts/mace/mace_utils.py``; extracted here since
none of it is actually MACE-specific -- it just combines/splits/labels
ASE-readable structure files, which the ACE and GRACE training payloads need
too (``scripts/ace/ace_utils.py``, ``scripts/grace/grace_utils.py``).
``mace_utils.py`` re-exports these names so existing imports keep working
unchanged.
"""

import logging
import pathlib
from typing import Optional

import ase.io
import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)


def make_dir(path: pathlib.Path, **kwargs) -> None:
    """Create a directory if it does not exist yet

    Args:
        path: Directory to create
    """
    try:
        path.mkdir(**kwargs)
    except FileExistsError:
        pass


def combine_databases(
    database_files: list[str], calcpath: pathlib.Path
) -> pathlib.Path:
    """Combine multiple database files into one, or return the single file.

    Args:
        database_files: List of database file paths (relative to calcpath)
        calcpath: Base path to resolve relative database paths

    Returns:
        Path to the (possibly combined) database file
    """
    if len(database_files) > 1:
        all_ats = []
        for p in database_files:
            ats = ase.io.read(calcpath / p, index=":")
            if ats is Atoms:
                all_ats.append(ats)
            else:
                all_ats.extend(ats)
        combined_path = calcpath / "combined_db.extxyz"
        ase.io.write(combined_path, all_ats)
        return combined_path
    else:
        return calcpath / database_files[0]


def write_with_keys(
    file_name: pathlib.Path,
    alist: list[Atoms],
    energy_key: str = "REF_energy",
    forces_key: str = "REF_forces",
) -> None:
    """
    starting from MACE 0.3.13, they disallowed energy and forces keys
    need to change to something else
    Make sure that energy_key and forces_key are in the config for the training script for lower versions!
    TODO: check actual MACE version to avoid compatibility issues

    Args:
        file_name: Path to file to write
        alist: List of Atoms objects to write
        energy_key: Key to use for energy, default is MACE default
        forces_key: Key to use for forces, default is MACE default
    """
    for atoms in alist:
        if energy_key not in atoms.info.keys():
            atoms.info[energy_key] = atoms.get_potential_energy()
        if forces_key not in atoms.arrays.keys():
            atoms.arrays[forces_key] = atoms.get_forces(apply_constraint=False)
    ase.io.write(file_name, alist)


def split_db(
    database: pathlib.Path,
    out_file: pathlib.Path,
    validation_fraction: float = 0.1,
    seed: Optional[int] = None,
    do_return_path: bool = False,
) -> Optional[tuple[pathlib.Path, pathlib.Path]]:
    """Split database into training and validation sets

    Args:
        database: Path to full database
        out_file: Path to files to write the split database to. Training and
            validation splits are marked by '_train' and '_val' respectively
        validation_fraction: Fraction of the database used for validation
        seed: Seed for setting up the NumPy random number generator
        do_return_path: Return a tuple of paths to train and validation sets
    """
    all_atoms = ase.io.read(database.resolve(), index=":")

    rng = np.random.default_rng(seed)
    n_validate = int(len(all_atoms) * validation_fraction)
    n_train = len(all_atoms) - n_validate
    indices = np.arange(len(all_atoms), dtype=int)
    rng.shuffle(indices)
    training_set = [all_atoms[i] for i in indices[:n_train]]
    validation_set = [all_atoms[i] for i in indices[n_train:]]

    out_stem = out_file.stem
    out_suffix = out_file.suffix
    train_file = out_file.parent / (out_stem + "_train" + out_suffix)
    valid_file = out_file.parent / (out_stem + "_val" + out_suffix)

    write_with_keys(train_file.resolve(), training_set)
    write_with_keys(valid_file.resolve(), validation_set)
    return (train_file, valid_file) if do_return_path else None
