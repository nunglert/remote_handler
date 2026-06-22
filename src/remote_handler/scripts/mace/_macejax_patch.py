"""Runtime patch for mace-jax's torch -> JSON config serializer.

mace-jax's ``mace_jax.cli.mace_jax_from_torch._serialize_for_json`` is what
turns a torch-extracted MACE config into the JSON written to a directory
bundle's ``config.json``. Its logic is correct (it maps classes/callables to
their names), but it references ``torch`` -- which the module imports only
*locally inside* ``main()``, never at module scope -- so calling it as a
library function raises ``NameError: name 'torch' is not defined``.

Bind ``torch`` into the module namespace so the serializer works. Idempotent.
Remove once mace-jax imports ``torch`` at module scope upstream.
"""


def ensure_serialize_patched() -> None:
    """Make mace-jax's ``_serialize_for_json`` usable as a library function."""
    import mace_jax.cli.mace_jax_from_torch as mace_jax_from_torch
    import torch

    if getattr(mace_jax_from_torch, "torch", None) is None:
        mace_jax_from_torch.torch = torch
