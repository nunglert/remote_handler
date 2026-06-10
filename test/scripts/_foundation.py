"""Shared resolver for the MACE foundation model used by script tests.

Tests use the model shipped under ``test/data/assets`` so they run without any
configuration.

Not a test module (leading underscore -> not collected by pytest); imported by
the sibling ``test_*`` files via pytest's prepend import mode.
"""

import os
from pathlib import Path

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
