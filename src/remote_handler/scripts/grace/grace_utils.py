"""Shared utilities for GRACE (Graph Atomic Cluster Expansion) training.

Fits GRACE potentials with ``gracemaker`` (the ``tensorpotential`` package,
https://github.com/ICAMS/grace-tensorpotential). Unlike ACE/pacemaker,
gracemaker reads extxyz directly, extracting energy/forces the plain ASE way
(``atoms.get_potential_energy()``/``get_forces()``) rather than
remote_handler's ``REF_energy``/``REF_forces`` convention -- so the training
data must be re-labelled onto a real ``SinglePointCalculator`` first (see
:func:`write_ase_standard_labels`).

gracemaker also fits its own internal train/test split (a ``data.test_size``
fraction, or an explicit ``data.test_filename``), so no separate split step
is needed here either (cf. MACE's ``split_db``).

Output layout: a run with ``seed: N`` (or ``--seed N``) writes into
``<cwd>/seed/<N>/`` (``model.yaml`` + ``checkpoints/``) -- a relative path,
so gracemaker must be run with ``cwd`` set to the calc dir.
:func:`finalize_grace_artifact` moves that directory to the conventional
``models/<model_name>/`` location once training finishes.
"""

import logging
import pathlib
import shutil
import subprocess
from typing import Optional

import ase.io
import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator


def write_ase_standard_labels(
    file_name: pathlib.Path,
    alist: list[Atoms],
    energy_key: str = "REF_energy",
    forces_key: str = "REF_forces",
) -> None:
    """Attach a real calculator carrying REF_energy/REF_forces, then write.

    gracemaker reads ``atoms.get_potential_energy()``/``get_forces()`` (the
    plain ASE convention that a normal extxyz round-trip restores), so the
    ground-truth values must sit behind a real calculator rather than as
    plain ``REF_energy``/``REF_forces`` info/array entries.

    Args:
        file_name: Path to write the labelled extxyz file to.
        alist: List of Atoms objects to write.
        energy_key: Info key to read ground-truth energy from, if no
            calculator is already attached.
        forces_key: Arrays key to read ground-truth forces from, if no
            calculator is already attached.
    """
    for atoms in alist:
        if energy_key in atoms.info and forces_key in atoms.arrays:
            energy = atoms.info[energy_key]
            forces = atoms.arrays[forces_key]
        else:
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces(apply_constraint=False)
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    ase.io.write(file_name, alist)


def merge_grace_config_with_paths(
    user_config_file: pathlib.Path,
    seed: int,
    data_file: pathlib.Path,
    validation_fraction: Optional[float] = None,
    **extra_config,
) -> dict:
    """Load the user's YAML config and merge in the run-managed keys.

    The user's config supplies the fitting hyperparameters (``cutoff``,
    ``potential``, ``fit``); this injects the top-level ``seed`` and the
    ``data`` section gracemaker needs to find the dataset for this run.

    Args:
        user_config_file: Path to the user's YAML config (cutoff/potential/
            fit hyperparameters; no ``data``/``seed`` needed there).
        seed: Random seed for this ensemble member.
        data_file: Path to the (relabelled) extxyz training file.
        validation_fraction: If given, sets ``data.test_size`` so gracemaker
            splits off its own internal test set. Any ``test_size`` or
            ``test_filename`` already present in the user's ``data`` section
            (if they authored one) takes precedence.
        **extra_config: Additional top-level key-value pairs to merge in.

    Returns:
        Dictionary with the merged configuration.
    """
    with open(user_config_file.resolve(), "r") as f:
        config = yaml.safe_load(f) or {}

    config["seed"] = seed

    data_config = config.get("data", {})
    data_config["filename"] = str(pathlib.Path(data_file).resolve())
    if validation_fraction and "test_size" not in data_config:
        data_config["test_size"] = validation_fraction
    data_config.setdefault("reference_energy", 0)
    config["data"] = data_config

    config.update(extra_config)
    return config


def run_grace_training(
    config_file: pathlib.Path,
    user_config_file: pathlib.Path,
    seed: int,
    data_file: pathlib.Path,
    storage_dir: pathlib.Path,
    train_runner: pathlib.Path,
    validation_fraction: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
    **extra_config,
) -> None:
    """Write the final gracemaker YAML config and run the fit.

    Runs with ``cwd=storage_dir`` since gracemaker writes its output
    (``seed/<seed>/model.yaml`` + ``checkpoints/``) relative to the working
    directory.

    Args:
        config_file: Path to write the merged gracemaker YAML config to.
        user_config_file: Path to the user's YAML config with fitting
            hyperparameters.
        seed: Random seed for this ensemble member (also passed via
            gracemaker's ``--seed``, which overrides the YAML value).
        data_file: Path to the (relabelled) extxyz training file.
        storage_dir: Working directory gracemaker runs in.
        train_runner: Path to the ``gracemaker`` executable.
        validation_fraction: See :func:`merge_grace_config_with_paths`.
        logger: Logger used to write additional output to.
        **extra_config: Passed through to
            :func:`merge_grace_config_with_paths`.
    """
    final_config = merge_grace_config_with_paths(
        user_config_file=user_config_file,
        seed=seed,
        data_file=data_file,
        validation_fraction=validation_fraction,
        **extra_config,
    )

    with open(config_file.resolve(), "w") as f:
        yaml.safe_dump(
            final_config, f, default_flow_style=False, sort_keys=False
        )

    cmd = [
        str(train_runner),
        str(config_file.resolve()),
        "--seed",
        str(seed),
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            cwd=str(storage_dir),
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as run_error:
        if logger is not None:
            logger.error(
                "GRACE (gracemaker) training returned with non-zero exit "
                f"code {run_error.returncode}"
            )
            logger.error(run_error.stderr.decode("utf-8"))
        raise RuntimeError(run_error)


def finalize_grace_artifact(
    storage_dir: pathlib.Path, seed: int, model_name: str
) -> pathlib.Path:
    """Move gracemaker's ``seed/<seed>/`` output to ``models/<model_name>/``.

    Args:
        storage_dir: The ``cwd`` gracemaker was run in (holds ``seed/``).
        seed: The seed the run used (names the ``seed/<seed>/`` subdir).
        model_name: Name tag to file the artifact under, under ``models/``.

    Returns:
        Path to the artifact directory under ``models/``.

    Raises:
        RuntimeError: If gracemaker did not produce the expected directory.
    """
    storage_dir = pathlib.Path(storage_dir)
    src = storage_dir / "seed" / str(seed)
    if not src.is_dir():
        raise RuntimeError(
            f"gracemaker did not produce the expected output directory: {src}"
        )

    models_dir = storage_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / model_name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(src), str(dest))
    return dest
