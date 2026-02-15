"""Shared utilities for MACE training and finetuning scripts."""

import logging
import pathlib
import subprocess
from typing import Optional

import ase.io
import numpy as np
import yaml
from ase import Atoms


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
    ase.io.write(train_file.resolve(), training_set)
    ase.io.write(valid_file.resolve(), validation_set)
    return (train_file, valid_file) if do_return_path else None


def merge_config_with_paths(
    user_config_file: pathlib.Path,
    model_name: str,
    seed: int,
    train_file: pathlib.Path,
    valid_file: pathlib.Path,
    storage_dir: pathlib.Path,
    **extra_config,
) -> dict:
    """Load user's YAML config and merge with generated paths.

    Args:
        user_config_file: Path to user's YAML config file
        model_name: Name tag of the model to be trained
        seed: Seed used for model training
        train_file: Path to file to read the training configurations from
        valid_file: Path to file to read the validation configurations from
        storage_dir: Directory to store the MACE training outputs in
        **extra_config: Additional key-value pairs to add to the config
            (e.g. foundation_model for finetuning)

    Returns:
        Dictionary with merged configuration
    """
    # Load user's config
    with open(user_config_file.resolve(), "r") as f:
        config = yaml.safe_load(f) or {}

    # Override/add paths managed by the script
    config.update(
        {
            "name": model_name,
            "seed": seed,
            "train_file": str(train_file.resolve()),
            "valid_file": str(valid_file.resolve()),
            "checkpoints_dir": str((storage_dir / "checkpoints").resolve()),
            "log_dir": str((storage_dir / "logs").resolve()),
            "model_dir": str((storage_dir / "models").resolve()),
            "results_dir": str((storage_dir / "results").resolve()),
        }
    )

    config.update(extra_config)

    return config


def run_training_from_config(
    config_file: pathlib.Path,
    user_config_file: pathlib.Path,
    model_name: str,
    seed: int,
    train_file: pathlib.Path,
    valid_file: pathlib.Path,
    train_runner: pathlib.Path,
    storage_dir: pathlib.Path,
    logger: Optional[logging.Logger] = None,
    **extra_config,
) -> None:
    """Prepare a yaml configuration file for MACE training and train a model.

    Args:
        config_file: Path to yaml file to store the final MACE configuration
        user_config_file: Path to user's YAML config with training parameters
        model_name: Name tag of the model to be trained
        seed: Seed used for model training
        train_file: Path to file to read the training configurations from
        valid_file: Path to file to read the validation configurations from
        train_runner: Path to MACE training runner script
        storage_dir: Directory to store the MACE training outputs in.
            Trained models are serialized in storage_dir/models
        logger: Logger used to write additional output to
        **extra_config: Additional key-value pairs passed to
            merge_config_with_paths (e.g. foundation_model for finetuning)
    """
    # Create model directory beforehand if it does not exist
    # (Only output directory MACE does not create itself)
    model_dir = storage_dir / "models"
    make_dir(model_dir)

    # Merge user config with generated paths
    final_config = merge_config_with_paths(
        user_config_file=user_config_file,
        model_name=model_name,
        seed=seed,
        train_file=train_file,
        valid_file=valid_file,
        storage_dir=storage_dir,
        **extra_config,
    )

    # Write final config
    with open(config_file.resolve(), "w") as f:
        yaml.safe_dump(
            final_config, f, default_flow_style=False, sort_keys=False
        )

    # Run training
    try:
        subprocess.run(
            [
                "python",
                str(train_runner),
                "--config",
                str(config_file.resolve()),
            ],
            check=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as run_error:
        if logger is not None:
            logger.error(
                f"MACE training returned with non-zero exit code "
                f"{run_error.returncode}"
            )
            logger.error(run_error.stderr.decode("utf-8"))
        raise RuntimeError(run_error)
