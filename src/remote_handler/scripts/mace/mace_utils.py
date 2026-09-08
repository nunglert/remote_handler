"""Shared utilities for MACE training and finetuning scripts."""

import logging
import pathlib
import subprocess
import sys
from typing import Optional

import yaml

# combine_databases/split_db/write_with_keys/make_dir are not MACE-specific
# (they just combine/split/label ASE-readable structure files) -- they live
# in scripts/common/atoms_data.py and are re-exported here so existing
# imports of this module keep working unchanged.
from remote_handler.scripts.common.atoms_data import (  # noqa: F401
    combine_databases,
    make_dir,
    split_db,
    write_with_keys,
)


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


def maybe_convert_to_jax(
    calcpath: pathlib.Path,
    model_name: str,
    convert_to_jax: bool,
    jax_output_format: str = "pkl",
    logger=None,
) -> Optional[pathlib.Path]:
    """Post-training hook: convert ``models/<model_name>.model`` to mace-jax.

    Shared by the ``mace_training.py`` / ``mace_finetuning.py`` payloads so the
    conversion step is declared once. Runs in the same job/env as training, so
    that env needs both the torch MACE stack and ``mace-jax``.

    Args:
        calcpath: The array task's calc dir (holds ``models/``).
        model_name: Name tag the trained model was written under.
        convert_to_jax: No-op returning ``None`` when False.
        jax_output_format: ``"pkl"`` (opaque single file) or ``"bundle"``
            (a ``<model_name>-jax/`` directory with config.json +
            params.msgpack, mace-jax's canonical format).

    Returns:
        Path to the written bundle, or ``None`` if no conversion was asked for.
    """
    if not convert_to_jax:
        return None

    models_dir = pathlib.Path(calcpath) / "models"
    torch_model_path = models_dir / f"{model_name}.model"
    if not torch_model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found for conversion: {torch_model_path}"
        )

    if jax_output_format == "bundle":
        jax_model_path = convert_torch_model_to_jax_bundle(
            torch_model_path,
            output_dir=models_dir / f"{model_name}-jax",
        )
    else:
        jax_model_path = convert_torch_model_to_jax(
            torch_model_path,
            output_path=models_dir / f"{model_name}-jax.pkl",
        )

    (logger or logging.getLogger(__name__)).info(
        f"Converted torch model to mace-jax bundle: {jax_model_path}"
    )
    return jax_model_path


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
