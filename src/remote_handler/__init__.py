import importlib.resources as pkg_resources
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from remote_handler.logging_setup import configure_logging

try:
    # Single source of truth is pyproject.toml; report the installed version.
    __version__ = version("remote_handler")
except PackageNotFoundError:  # running from a source tree with no install
    __version__ = "0.0.0+unknown"

DEFAULT_TEMPLATE_DIR = Path(
    pkg_resources.files("remote_handler").joinpath("templates")
)

__all__ = ["__version__", "DEFAULT_TEMPLATE_DIR", "configure_logging"]
