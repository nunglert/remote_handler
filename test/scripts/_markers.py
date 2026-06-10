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
