"""Shared pytest markers for the script-test tier.

Not a test module (leading underscore -> not collected); imported by sibling
``test_*`` files via pytest's prepend import mode.
"""

import shutil

import pytest


def requires_local_slurm(test):
    """Tag a test as needing local SLURM, and skip when sbatch is absent.

    Applies the ``slurm`` marker -- so the scheduler tier is selectable, e.g.
    ``pytest -m "integration and not slurm"`` to run the ML tests that do not
    need a scheduler -- and skips with a clear reason when ``sbatch`` is not on
    PATH. Works on test functions/methods.
    """
    test = pytest.mark.slurm(test)
    return pytest.mark.skipif(
        shutil.which("sbatch") is None,
        reason="SLURM (sbatch) not available on this machine",
    )(test)


def _cuda_available():
    """True if a CUDA GPU is visible to torch (False if torch is absent)."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


# Skip a test that genuinely requires an NVIDIA GPU (no CPU code path).
requires_cuda = pytest.mark.skipif(
    not _cuda_available(), reason="needs an NVIDIA GPU (CUDA)"
)

# Device for *device-flexible* integration tests: the GPU when present (dev
# boxes) else CPU (CPU-only CI). The torch_sim sampling script runs on both,
# so these tests exercise it on whatever hardware is available.
TEST_DEVICE = "cuda" if _cuda_available() else "cpu"
