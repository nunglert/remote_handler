"""Self-contained support code for the NeuralIL training payload.

This subpackage vendors the small closure of helpers that the remote-side
NeuralIL training script used to import from ``jaxnest``. Keeping it here means
:mod:`remote_handler.scripts.neuralil.train_model` only depends on the
third-party ``neuralil`` force-field library and jax/flax -- never on jaxnest.
"""

import jax

# In jaxnest this was ``jaxnest.mcmc.KeyArray`` (an alias for the internal
# ``jax._src.prng.PRNGKeyArray``). It is only used in type hints / docstrings
# in the training code, so the public, forward-compatible ``jax.Array`` alias
# is sufficient.
KeyArray = jax.Array

__all__ = ["KeyArray", "batching", "load_neuralil", "structure_utils"]
