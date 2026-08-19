"""Shared utilities for MACE training and finetuning scripts."""

import logging
import pathlib
import subprocess
import sys
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
        # check if train_runner is the executable, not a full path
        if "mace_run_train" in str(train_runner):
            cmd = [str(train_runner), "--config", str(config_file.resolve())]
        else:
            cmd = [
                # Same interpreter running this script (works locally and on
                # the cluster where the env is activated); bare "python" could
                # resolve to an interpreter without torch.
                sys.executable,
                str(train_runner),
                "--config",
                str(config_file.resolve()),
            ]
        subprocess.run(
            cmd,
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


def convert_torch_model_to_jax(
    model_path: pathlib.Path,
    output_path: Optional[pathlib.Path] = None,
    head: Optional[str] = None,
) -> pathlib.Path:
    """Convert a trained torch MACE model to a mace-jax ``.pkl`` bundle.

    Reuses mace-jax's own conversion engine (``convert_model``) and writes the
    same ``{'config', 'params', 'head'}`` pickle that ``mace-jax-predict`` /
    ``MACEJAXCalculator`` load -- i.e. what ``mace-create-lammps-model``
    produces, but without building the (unused, dependency-heavy) LAMMPS-MLIAP
    wrapper.

    Runs in the same environment as training, so it requires both the torch
    MACE stack (``mace.tools.scripts_utils``) and ``mace-jax`` importable.

    Args:
        model_path: Path to the trained torch ``.model`` file.
        output_path: Destination ``.pkl``. Defaults to
            ``<model_path stem>-jax.pkl`` next to the input.
        head: Optional head name to record in the bundle.

    Returns:
        Path to the written ``.pkl`` bundle.
    """
    # Lazy imports: keep this module importable without torch / mace-jax (the
    # conversion only runs on the cluster env that has both stacks).
    import pickle

    import torch
    from flax import serialization
    from mace.tools.scripts_utils import extract_config_mace_model
    from mace_jax.cli.mace_jax_from_torch import convert_model
    from mace_jax.nnx_utils import state_to_serializable_dict

    model_path = pathlib.Path(model_path)
    bundle = torch.load(model_path, map_location="cpu", weights_only=False)
    torch_model = (
        bundle["model"]
        if isinstance(bundle, dict) and "model" in bundle
        else bundle
    )
    torch_model.eval()

    config = extract_config_mace_model(torch_model)
    if "error" in config:
        raise RuntimeError(config["error"])
    config["torch_model_class"] = torch_model.__class__.__name__

    graphdef, state, _ = convert_model(torch_model, config)
    # Use state_to_serializable_dict (not state_to_pure_dict): it converts the
    # ConfigDict leaves the MACE model carries into plain dicts, which msgpack
    # / flax.serialization can pack (pure_dict leaves a raw ConfigDict and
    # to_bytes then raises "can not serialize 'ConfigDict' object").
    variables = state_to_serializable_dict(state)

    artifact = {
        "config": config,
        "params": serialization.to_bytes(variables),
        "head": head,
    }

    if output_path is None:
        output_path = model_path.with_name(model_path.stem + "-jax.pkl")
    output_path = pathlib.Path(output_path)
    with output_path.open("wb") as f:
        pickle.dump(artifact, f)

    return output_path


def convert_torch_model_to_jax_bundle(
    model_path: pathlib.Path,
    output_dir: Optional[pathlib.Path] = None,
) -> pathlib.Path:
    """Convert a torch MACE model to a mace-jax *directory* bundle.

    Writes ``config.json`` (human-readable -- architecture, atomic energies,
    r_max) and ``params.msgpack`` into ``output_dir``. This is the format
    mace-jax's canonical loader (``mace_jax.tools.bundle.load_model_bundle``,
    which jaxrens delegates to) consumes -- unlike the opaque ``.pkl`` written
    by :func:`convert_torch_model_to_jax`.

    Reuses mace-jax's own conversion (``convert_model``) and config serializer,
    so it needs both the torch MACE stack and ``mace-jax``. Applies a small
    runtime shim for a mace-jax bug (see
    :func:`remote_handler.scripts.mace._macejax_patch.ensure_serialize_patched`).

    Args:
        model_path: Path to the trained torch ``.model`` file.
        output_dir: Destination directory. Defaults to ``<model stem>-jax``
            next to the input.

    Returns:
        Path to the written bundle directory.
    """
    import json

    import torch
    from flax import serialization
    from mace.tools.scripts_utils import extract_config_mace_model
    from mace_jax.cli.mace_jax_from_torch import (
        _serialize_for_json,
        convert_model,
    )
    from mace_jax.nnx_utils import state_to_serializable_dict
    from mace_jax.tools.bundle import DEFAULT_CONFIG_NAME, DEFAULT_PARAMS_NAME

    from remote_handler.scripts.mace._macejax_patch import (
        ensure_serialize_patched,
    )

    ensure_serialize_patched()

    model_path = pathlib.Path(model_path)
    bundle = torch.load(model_path, map_location="cpu", weights_only=False)
    torch_model = (
        bundle["model"]
        if isinstance(bundle, dict) and "model" in bundle
        else bundle
    )
    torch_model.eval()

    config = extract_config_mace_model(torch_model)
    if "error" in config:
        raise RuntimeError(config["error"])
    config["torch_model_class"] = torch_model.__class__.__name__

    _, state, _ = convert_model(torch_model, config)
    params_bytes = serialization.to_bytes(state_to_serializable_dict(state))

    if output_dir is None:
        output_dir = model_path.with_name(model_path.stem + "-jax")
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / DEFAULT_PARAMS_NAME).write_bytes(params_bytes)
    (output_dir / DEFAULT_CONFIG_NAME).write_text(
        json.dumps(_serialize_for_json(config), indent=2)
    )
    return output_dir


def generate_jax_foundation_model(
    model: str = "medium",
    output_path: Optional[pathlib.Path] = None,
    head: Optional[str] = None,
) -> pathlib.Path:
    """Download a MACE-MP foundation model and convert it to a mace-jax bundle.

    Bootstraps a mace-jax ``.pkl`` foundation model without having to obtain a
    torch ``.model`` and convert it by hand: it pulls the MACE-MP foundation
    checkpoint via MACE's own download helper (cached under MACE's cache dir)
    and feeds it to :func:`convert_torch_model_to_jax`.

    Runs wherever both stacks are importable (the torch MACE stack and
    ``mace-jax``) and where the download has network access -- typically a
    cluster login node. Needs the ``mace_jax`` extra.

    Only the MACE-MP (inorganic) foundations are supported, since MACE exposes
    a standalone checkpoint downloader only for those; MACE-OFF would need a
    separate path.

    Args:
        model: MACE-MP foundation size (``"small"`` / ``"medium"`` /
            ``"large"``), or a path/URL to a checkpoint.
        output_path: Destination ``.pkl``. Defaults to ``<checkpoint stem>``
            ``-jax.pkl`` next to the downloaded checkpoint.
        head: Optional head name to record in the bundle.

    Returns:
        Path to the written mace-jax ``.pkl`` bundle.
    """
    from mace.calculators.foundations_models import download_mace_mp_checkpoint

    torch_path = download_mace_mp_checkpoint(model)
    return convert_torch_model_to_jax(
        pathlib.Path(torch_path), output_path, head
    )
