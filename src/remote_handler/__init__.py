import importlib.resources as pkg_resources
from pathlib import Path

__version__ = "0.1.0"

DEFAULT_TEMPLATE_DIR = Path(
    pkg_resources.files("remote_handler").joinpath("templates")
)
