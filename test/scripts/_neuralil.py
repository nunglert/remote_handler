"""Capability probes for the NeuralIL training stack.

The remote-side NeuralIL training payload
(``remote_handler.scripts.neuralil.train_model``) needs the third-party
``neuralil`` force-field library (installed via ``pip install .[neuralil]``),
which pulls in jax/flax/optax. The vendored support helpers under
``scripts/neuralil/_support/`` only need jax (+ ase), so probe the two levels
separately to skip tests with the most informative reason.

Not a test module (leading underscore -> not collected); imported by sibling
``test_*`` files via pytest's prepend import mode.
"""

import importlib.util


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
