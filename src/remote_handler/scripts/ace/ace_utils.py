"""Shared utilities for ACE (Atomic Cluster Expansion) training scripts.

Fits ACE potentials with ``pacemaker`` (the ``python-ace``/``pyace`` package,
https://github.com/ICAMS/python-ace). Unlike MACE/GRACE, pacemaker does not
read extxyz directly: its ``data.filename`` must be a pickled pandas
``DataFrame`` with an ``ase_atoms`` column plus ``energy`` (or
``energy_corrected``) and ``forces`` columns (see ``pyace.preparedata``).
:func:`build_ace_dataframe` does that conversion from remote_handler's
``REF_energy``/``REF_forces`` convention.

pacemaker fits its own internal train/test split (a ``data.test_size``
fraction, or an explicit ``data.test_filename``) rather than needing
pre-split files handed to it, so no separate split step is needed here (cf.
MACE's ``split_db``).
"""

import logging
import os
import pathlib
import subprocess
import sysconfig
from typing import Optional

import ase.io
import numpy as np
import yaml
from ase import Atoms

#: ``backend.evaluator`` value that routes pacemaker through TensorFlow
#: (GPU-capable) instead of its default CPU evaluator. See
#: :func:`_tensorpot_gpu_ld_library_path`.
_TENSORPOT_EVALUATOR = "tensorpot"


def build_ace_dataframe(
    database_path: pathlib.Path, out_file: pathlib.Path
) -> pathlib.Path:
    """Convert an ASE-readable structure file to pacemaker's dataframe format.

    Reads ``REF_energy``/``REF_forces`` (remote_handler's ground-truth
    convention) if present, falling back to any calculator already attached
    to the atoms (``get_potential_energy``/``get_forces``).

    Args:
        database_path: Path to an ASE-readable structure file (e.g. extxyz).
        out_file: Destination for the pickled dataframe (conventionally named
            ``*.pckl.gzip``, gzip-compressed via the ``.gzip`` suffix).

    Returns:
        Path to the written dataframe (``out_file``).
    """
    # Lazy import: pandas is a pacemaker dependency, not a core remote_handler
    # one -- keep this module importable without it (host stays
    # dependency-light; this only runs remotely, in the ace conda env).
    import pandas as pd

    atoms_list = ase.io.read(database_path, index=":")
    if isinstance(atoms_list, Atoms):
        atoms_list = [atoms_list]

    energies = []
    forces = []
    for atoms in atoms_list:
        if "REF_energy" in atoms.info:
            energies.append(atoms.info["REF_energy"])
        else:
            energies.append(atoms.get_potential_energy())
        if "REF_forces" in atoms.arrays:
            forces.append(np.asarray(atoms.arrays["REF_forces"]))
        else:
            forces.append(np.asarray(atoms.get_forces(apply_constraint=False)))

    df = pd.DataFrame(
        {"ase_atoms": atoms_list, "energy": energies, "forces": forces}
    )

    out_file = pathlib.Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if str(out_file).endswith("gzip") else "infer"
    df.to_pickle(out_file, compression=compression)
    return out_file


def merge_ace_config_with_paths(
    user_config_file: pathlib.Path,
    seed: int,
    data_file: pathlib.Path,
    validation_fraction: Optional[float] = None,
    **extra_config,
) -> dict:
    """Load the user's YAML config and merge in the run-managed keys.

    The user's config supplies the fitting hyperparameters (``potential``,
    ``fit``, optionally ``backend``); this injects the top-level ``seed`` and
    the ``data`` section pacemaker needs to find the dataset for this run,
    plus a default ``backend`` if the user's config didn't set one.

    The ``backend`` default is more than a convenience: pacemaker's own CLI
    (as of the current ``python-ace`` master, since a Nov 2023 regression --
    commit 29ba519) crashes with ``AttributeError: 'Namespace' object has no
    attribute 'query_data'`` when it falls through to its *own*
    default-backend branch, which only happens when ``backend`` is absent
    from the YAML. Supplying it here every time means that branch is never
    exercised, sidestepping the bug regardless of which pacemaker version is
    installed.

    Args:
        user_config_file: Path to the user's YAML config (potential/fit
            hyperparameters; no ``data``/``seed`` needed there).
        seed: Random seed for this ensemble member.
        data_file: Path to the pacemaker dataframe (see
            :func:`build_ace_dataframe`).
        validation_fraction: If given, sets ``data.test_size`` so pacemaker
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

    config.setdefault(
        "backend", {"evaluator": "pyace", "parallel_mode": "process"}
    )

    config.update(extra_config)
    return config


def _tensorpot_gpu_ld_library_path() -> Optional[str]:
    """Locate pip-installed NVIDIA CUDA libraries for TensorFlow's GPU path.

    Fitting ACE potentials via pacemaker's ``tensorpot`` backend (the
    ``ace_tensorpot_gpu`` extra) uses TensorFlow for GPU acceleration.
    ``tensorflow[and-cuda]`` installs its CUDA runtime as separate
    ``nvidia-*-cu12`` pip packages under ``site-packages/nvidia/*/lib``, but
    -- unlike a conda-provided CUDA toolkit -- does not add those
    directories to the dynamic linker's search path. Without them on
    ``LD_LIBRARY_PATH``, TensorFlow silently falls back to CPU (a log
    warning, not an error), silently defeating the entire point of this
    backend. This looks them up from the *running* interpreter's own
    site-packages (i.e. the same env pacemaker/tensorflow are installed
    into), so :func:`run_ace_training` can inject them into the training
    subprocess automatically -- no per-environment shell setup needed.

    Returns:
        A ``:``-joined string of lib directories, or ``None`` if none are
        found (e.g. the CPU-only default ``pyace`` evaluator, or a conda-
        provided CUDA toolkit that doesn't need this at all).
    """
    site_packages = pathlib.Path(sysconfig.get_paths()["purelib"])
    nvidia_dir = site_packages / "nvidia"
    if not nvidia_dir.is_dir():
        return None

    lib_dirs = sorted(
        str(path) for path in nvidia_dir.glob("*/lib") if path.is_dir()
    )
    if not lib_dirs:
        return None
    return os.pathsep.join(lib_dirs)


def run_ace_training(
    config_file: pathlib.Path,
    user_config_file: pathlib.Path,
    seed: int,
    data_file: pathlib.Path,
    storage_dir: pathlib.Path,
    output_path: pathlib.Path,
    train_runner: pathlib.Path,
    validation_fraction: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
    initial_potential: Optional[pathlib.Path] = None,
    **extra_config,
) -> None:
    """Write the final pacemaker YAML config and run the fit.

    Args:
        config_file: Path to write the merged pacemaker YAML config to.
        user_config_file: Path to the user's YAML config with fitting
            hyperparameters.
        seed: Random seed for this ensemble member.
        data_file: Path to the pacemaker dataframe (see
            :func:`build_ace_dataframe`).
        storage_dir: Working directory pacemaker runs in (it writes several
            side files -- logs, metrics, checkpoints -- into its cwd).
        output_path: Destination for the fitted B-basis potential
            (pacemaker's ``-o``/``--output``).
        train_runner: Path to the ``pacemaker`` executable.
        validation_fraction: See :func:`merge_ace_config_with_paths`.
        logger: Logger used to write additional output to.
        initial_potential: Potential to continue fitting from (pacemaker's
            ``-ip``/``--initial-potential``); ``None`` fits from scratch.
        **extra_config: Passed through to
            :func:`merge_ace_config_with_paths`.

    Note:
        If the merged config's ``backend.evaluator`` is ``"tensorpot"``
        (GPU-accelerated fitting, the ``ace_tensorpot_gpu`` extra), use a
        scipy-style ``fit.optimizer`` (``BFGS``/``L-BFGS-B``/``CG``/...),
        not ``Adam``/``SGD``: with a TF-native optimizer the fit runs and
        trains correctly but then crashes on completion
        (``AttributeError: 'NoneType' object has no attribute 'x'``) --
        pyace's ``fitadapter.py`` unconditionally reads ``self.res_opt.x``
        after fitting, but ``tensorpotential``'s ``FitTensorPotential`` only
        populates ``res_opt`` on its scipy-optimizer code path. This is an
        upstream bug in that version combination, not something
        remote_handler can work around from the config side.
    """
    final_config = merge_ace_config_with_paths(
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

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(train_runner),
        str(config_file.resolve()),
        "-o",
        str(output_path.resolve()),
    ]
    if initial_potential is not None:
        cmd += ["-ip", str(pathlib.Path(initial_potential).resolve())]

    env = os.environ.copy()
    backend_config = final_config.get("backend") or {}
    if backend_config.get("evaluator") == _TENSORPOT_EVALUATOR:
        gpu_lib_path = _tensorpot_gpu_ld_library_path()
        if gpu_lib_path:
            existing = env.get("LD_LIBRARY_PATH")
            env["LD_LIBRARY_PATH"] = (
                f"{gpu_lib_path}{os.pathsep}{existing}"
                if existing
                else gpu_lib_path
            )
            if logger is not None:
                logger.info(
                    "tensorpot backend: prepending pip NVIDIA CUDA libs to "
                    "LD_LIBRARY_PATH so TensorFlow can find the GPU: %s",
                    gpu_lib_path,
                )

    try:
        subprocess.run(
            cmd,
            check=True,
            cwd=str(storage_dir),
            env=env,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as run_error:
        if logger is not None:
            logger.error(
                "ACE (pacemaker) training returned with non-zero exit "
                f"code {run_error.returncode}"
            )
            logger.error(run_error.stderr.decode("utf-8"))
        raise RuntimeError(run_error)
