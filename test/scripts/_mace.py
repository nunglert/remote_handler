"""Capability probes for the installed MACE package.

The mace-freeze fork (7radians/mace-freeze) and stock mace-torch both report
distribution name ``mace-torch`` version ``0.3.14``, so they cannot be told
apart by version, URL, or an env var. Probe the *capability* instead: the fork
ships a ``mace.tools.freeze`` module (``freeze_layers`` / ``freeze_param``)
that stock mace lacks. This is robust to how the fork was installed.

Not a test module (leading underscore -> not collected); imported by sibling
``test_*`` files via pytest's prepend import mode.
"""

import importlib.util
from typing import Optional


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


def mace_jax_available() -> bool:
    """True if both stacks for torch->mace-jax conversion are importable.

    The conversion (``convert_torch_model_to_jax``) reads a torch ``.model``
    via ``mace.tools.scripts_utils`` and writes mace-jax params, so it needs
    the torch MACE stack *and* ``mace_jax`` in the same process.
    """
    import importlib.util

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
