import importlib.resources as pkg_resources
from pathlib import Path

TEMPLATE_DIR = Path(
    pkg_resources.files("remote_handler").joinpath("templates")
)