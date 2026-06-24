"""Capability and asset probes for the script-test tier.

Answers "is this optional stack / asset present?" so tests can skip with an
informative reason. Three groups:

  * MACE foundation model asset (shipped under ``test/data/assets``);
  * the MACE package (stock vs the freeze fork vs mace-jax);
  * the NeuralIL training stack (jax, and the full ``neuralil`` library).

Heavy imports (torch / mace / jax / neuralil) stay lazy -- inside the
functions -- so importing this module is cheap and never requires an extra.

Not a test module (leading underscore -> not collected by pytest); imported by
the sibling ``test_*`` files via pytest's prepend import mode. Pytest *markers*
live in the sibling ``_markers.py`` (skip infrastructure, a separate concern).
"""

import importlib.util
import os
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------
# MACE foundation model asset
# --------------------------------------------------------------------------
# Tests use the model shipped under ``test/data/assets`` so they run without
# any configuration.
MODEL_ASSET = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "assets"
    / "2023-12-10-mace-128-L0_energy_epoch-249.model"
)


def foundation_model_path() -> str:
    """Path to the MACE model shipped under ``test/data/assets``."""
    return str(MODEL_ASSET)


def foundation_model_available() -> bool:
    """True when a usable MACE foundation model is present."""
    return os.path.isfile(foundation_model_path())


# --------------------------------------------------------------------------
# MACE package probes
# --------------------------------------------------------------------------
# The mace-freeze fork (7radians/mace-freeze) and stock mace-torch both report
# distribution name ``mace-torch`` version ``0.3.14``, so they cannot be told
# apart by version, URL, or an env var. Probe the *capability* instead: the
# fork ships a ``mace.tools.freeze`` module (``freeze_layers`` /
# ``freeze_param``) that stock mace lacks -- robust to how it was installed.
def mace_available() -> bool:
    """True if the MACE package is importable (stock or fork)."""
    try:
        return importlib.util.find_spec("mace") is not None
    except Exception:
        return False


def mace_freeze_available() -> bool:
    """True if the freeze-capable MACE fork is installed."""
    try:
        from mace.tools.freeze import freeze_layers, freeze_param  # noqa: F401

        return True
    except Exception:
        return False


def clinamen_available() -> bool:
    """True if the Clinamen2 package is importable (``.[clinamen]``)."""
    try:
        return importlib.util.find_spec("clinamen2") is not None
    except Exception:
        return False


def mace_jax_available() -> bool:
    """True if both stacks for torch->mace-jax conversion are importable.

    The conversion (``convert_torch_model_to_jax``) reads a torch ``.model``
    via ``mace.tools.scripts_utils`` and writes mace-jax params, so it needs
    the torch MACE stack *and* ``mace_jax`` in the same process.
    """
    try:
        return (
            importlib.util.find_spec("mace_jax") is not None
            and importlib.util.find_spec("mace") is not None
        )
    except Exception:
        return False


def mace_run_train_script() -> Optional[str]:
    """Path to mace's ``run_train.py`` from the install, or None.

    Lets tests locate the training entry point from the installed package
    instead of a hard-coded env var; callers may still override with an env
    var for remote execution (where the path lives on another machine).
    """
    try:
        import mace.cli.run_train as run_train

        return run_train.__file__
    except Exception:
        return None


# --------------------------------------------------------------------------
# NeuralIL training stack probes
# --------------------------------------------------------------------------
# The remote-side NeuralIL payload
# (``remote_handler.scripts.neuralil.train_model``) needs the third-party
# ``neuralil`` library (``pip install .[neuralil]``), which pulls in
# jax/flax/optax. The vendored support helpers under
# ``scripts/neuralil/_support/`` only need jax (+ ase), so probe the two
# levels separately to skip with the most informative reason.
def jax_available() -> bool:
    """True if jax is importable (enough for the vendored _support helpers)."""
    try:
        return importlib.util.find_spec("jax") is not None
    except Exception:
        return False


def neuralil_available() -> bool:
    """True if the NeuralIL library is importable (full training stack)."""
    try:
        return importlib.util.find_spec("neuralil") is not None
    except Exception:
        return False
