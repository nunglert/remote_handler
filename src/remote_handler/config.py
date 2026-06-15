"""Configuration management for remote_handler.

This module provides centralized configuration handling with clear precedence:
1. Explicit parameters (highest priority)
2. Environment variables
3. Config files
4. Built-in defaults (lowest priority)

Example:
    >>> from remote_handler.config import get_config
    >>> config = get_config()
    >>> config_dir = config.get_config_dir()
    >>> template_dirs = config.get_template_dirs()
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from remote_handler.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


def get_persistent_config_dir() -> Path:
    """Fixed XDG location of the persistent global config directory.

    ``$XDG_CONFIG_HOME/remote_handler`` if set, else
    ``~/.config/remote_handler``. Read at call time so it honours a
    monkeypatched environment in tests.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "remote_handler"


def get_persistent_config_file() -> Path:
    """Path to the persistent global config JSON (may not exist)."""
    return get_persistent_config_dir() / "config.json"


@dataclass
class RemoteHandlerConfig:
    """Centralized configuration for remote_handler.

    Attributes:
        config_dir: Directory containing cluster configuration JSON files
        slurm_templates_dir: Directory containing custom SLURM job templates
        default_partition: Default SLURM partition for job submission
        default_conda_env: Default conda environment name
        default_conda_source: Default path to conda activation script
        test_cluster: Name (without ``.json``) of the per-cluster config used
            as the target for the ``remote`` test tier. ``None`` means fall
            back to ``remote.json``.
    """

    # Directories
    config_dir: Optional[Path] = None
    slurm_templates_dir: Optional[Path] = None

    # Default cluster settings
    default_partition: str = "normal"
    default_conda_env: str = "base"
    default_conda_source: str = "~/miniconda3/bin/activate"

    # Which cluster the ``remote`` test tier targets (name without ".json").
    test_cluster: Optional[str] = None

    @classmethod
    def from_env(cls) -> "RemoteHandlerConfig":
        """Load configuration from environment variables.

        Reads:
            REMOTE_HANDLER_CONFIG_DIR: Path to cluster config directory
            REMOTE_HANDLER_SLURM_TEMPLATES: Path to custom SLURM templates

        Returns:
            RemoteHandlerConfig with values from environment

        Raises:
            ConfigurationError: If environment variables point to invalid paths

        Example:
            >>> config = RemoteHandlerConfig.from_env()
        """
        config_dir = None
        config_dir_str = os.environ.get("REMOTE_HANDLER_CONFIG_DIR")

        if config_dir_str:
            config_dir = Path(config_dir_str).expanduser()
            if not config_dir.exists():
                raise ConfigurationError(
                    f"REMOTE_HANDLER_CONFIG_DIR points to non-existent directory: {config_dir}\n"
                    "Please create the directory or update the environment variable."
                )
            if not config_dir.is_dir():
                raise ConfigurationError(
                    f"REMOTE_HANDLER_CONFIG_DIR must be a directory, got: {config_dir}"
                )
            logger.info(
                f"Using config directory from environment: {config_dir}"
            )

        # Get templates directory
        templates_dir = None
        templates_str = os.environ.get("REMOTE_HANDLER_SLURM_TEMPLATES")

        if templates_str:
            templates_dir = Path(templates_str).expanduser()
            if not templates_dir.exists():
                logger.warning(
                    f"REMOTE_HANDLER_SLURM_TEMPLATES points to non-existent directory: {templates_dir}\n"
                    "Will fall back to built-in templates."
                )
                templates_dir = None
            elif not templates_dir.is_dir():
                logger.warning(
                    f"REMOTE_HANDLER_SLURM_TEMPLATES must be a directory: {templates_dir}\n"
                    "Will fall back to built-in templates."
                )
                templates_dir = None
            else:
                logger.info(
                    f"Using custom SLURM templates from: {templates_dir}"
                )

        return cls(
            config_dir=config_dir,
            slurm_templates_dir=templates_dir,
        )

    @classmethod
    def from_file(cls, config_file: Path) -> "RemoteHandlerConfig":
        """Load configuration from JSON file.

        Args:
            config_file: Path to JSON configuration file

        Returns:
            RemoteHandlerConfig with values from file

        Raises:
            ConfigurationError: If file doesn't exist, is invalid JSON, or missing required fields

        Example:
            >>> config = RemoteHandlerConfig.from_file(Path("config.json"))
        """
        config_file = Path(config_file).expanduser()

        if not config_file.exists():
            raise ConfigurationError(
                f"Configuration file not found: {config_file}"
            )

        try:
            with open(config_file) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(
                f"Invalid JSON in configuration file: {config_file}\n"
                f"Error at line {e.lineno}, column {e.colno}: {e.msg}"
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to read configuration file: {config_file}\n"
                f"Error: {e}"
            )

        # Validate schema
        if not isinstance(data, dict):
            raise ConfigurationError(
                f"Configuration file must contain a JSON object, got {type(data).__name__}"
            )

        # Convert string paths to Path objects
        if "config_dir" in data and data["config_dir"]:
            data["config_dir"] = Path(data["config_dir"]).expanduser()
        if "slurm_templates_dir" in data and data["slurm_templates_dir"]:
            data["slurm_templates_dir"] = Path(
                data["slurm_templates_dir"]
            ).expanduser()

        try:
            return cls(**data)
        except TypeError as e:
            raise ConfigurationError(
                f"Invalid fields in configuration file: {config_file}\n"
                f"Error: {e}"
            )

    def get_config_dir(self, required: bool = True) -> Optional[Path]:
        """Get configuration directory with validation.

        Args:
            required: If True, raises error when config_dir is not set

        Returns:
            Path to config directory, or None if not required and not set

        Raises:
            ConfigurationError: If required=True and config_dir is not set

        Example:
            >>> config = get_config()
            >>> config_dir = config.get_config_dir()
        """
        if self.config_dir is None:
            if required:
                raise ConfigurationError(
                    "Configuration directory not set. Please either:\n"
                    "1. Set REMOTE_HANDLER_CONFIG_DIR environment variable, or\n"
                    "2. Pass config_dir explicitly when creating RemoteHandlerConfig"
                )
            return None

        # Validate directory exists
        if not self.config_dir.exists():
            raise ConfigurationError(
                f"Configuration directory does not exist: {self.config_dir}"
            )

        return self.config_dir

    def get_config_file(self, filename: str) -> Path:
        """Get path to a specific config file.

        Args:
            filename: Name of config file (e.g., "cluster.json")

        Returns:
            Full path to config file

        Raises:
            ConfigurationError: If config_dir not set or file doesn't exist

        Example:
            >>> config = get_config()
            >>> cluster_config = config.get_config_file("my_cluster.json")
        """
        config_dir = self.get_config_dir(required=True)
        config_file = config_dir / filename

        if not config_file.exists():
            raise ConfigurationError(
                f"Configuration file not found: {config_file}\n"
                f"Available files in {config_dir}:\n"
                + "\n".join(f"  - {f.name}" for f in config_dir.glob("*.json"))
                if config_dir.exists()
                else "(directory does not exist)"
            )

        return config_file

    def get_template_dirs(self) -> list[Path]:
        """Get SLURM template search directories in priority order.

        Returns list of directories to search for templates:
        1. Custom templates directory (if set)
        2. Built-in package templates (always included as fallback)

        Returns:
            List of Path objects in search order

        Example:
            >>> config = get_config()
            >>> for template_dir in config.get_template_dirs():
            ...     print(f"Searching: {template_dir}")
        """
        dirs = []

        # Custom templates (highest priority)
        if self.slurm_templates_dir and self.slurm_templates_dir.exists():
            dirs.append(self.slurm_templates_dir)

        # Package defaults (fallback)
        from remote_handler import DEFAULT_TEMPLATE_DIR

        dirs.append(DEFAULT_TEMPLATE_DIR)

        return dirs

    def find_template(self, template_name: str) -> Optional[Path]:
        """Find a SLURM template file by name.

        Searches template directories in priority order and returns
        the first match found.

        Args:
            template_name: Name of template file (e.g., "body_single.sh.j2")

        Returns:
            Path to template file, or None if not found

        Example:
            >>> config = get_config()
            >>> template = config.find_template("body_single.sh.j2")
            >>> if template:
            ...     print(f"Found: {template}")
        """
        for template_dir in self.get_template_dirs():
            template_path = template_dir / template_name
            if template_path.exists():
                logger.debug(f"Found template: {template_path}")
                return template_path

        logger.warning(f"Template not found: {template_name}")
        return None

    def validate(self) -> None:
        """Validate configuration settings.

        Checks that all paths are valid and accessible.

        Raises:
            ConfigurationError: If any configuration is invalid

        Example:
            >>> config = RemoteHandlerConfig(config_dir=Path("/invalid"))
            >>> config.validate()  # Raises ConfigurationError
        """
        if self.config_dir is not None:
            if not self.config_dir.exists():
                raise ConfigurationError(
                    f"Configuration directory does not exist: {self.config_dir}"
                )
            if not self.config_dir.is_dir():
                raise ConfigurationError(
                    f"Configuration directory is not a directory: {self.config_dir}"
                )

        if self.slurm_templates_dir is not None:
            if not self.slurm_templates_dir.exists():
                raise ConfigurationError(
                    f"SLURM templates directory does not exist: {self.slurm_templates_dir}"
                )
            if not self.slurm_templates_dir.is_dir():
                raise ConfigurationError(
                    f"SLURM templates directory is not a directory: {self.slurm_templates_dir}"
                )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration

        Example:
            >>> config = get_config()
            >>> config_dict = config.to_dict()
            >>> print(json.dumps(config_dict, indent=2))
        """
        return {
            "config_dir": str(self.config_dir) if self.config_dir else None,
            "slurm_templates_dir": str(self.slurm_templates_dir)
            if self.slurm_templates_dir
            else None,
            "default_partition": self.default_partition,
            "default_conda_env": self.default_conda_env,
            "default_conda_source": self.default_conda_source,
            "test_cluster": self.test_cluster,
        }

    @classmethod
    def resolve(cls) -> "RemoteHandlerConfig":
        """Resolve configuration with full precedence.

        Precedence (highest first): environment variables > persistent
        config file (fixed XDG location) > built-in defaults. ``config_dir``
        defaults to the persistent config directory when neither the file nor
        the environment set it, so per-cluster JSONs live next to
        ``config.json`` with no prior setup.

        Returns:
            Merged RemoteHandlerConfig.
        """
        # 1. Persistent file (if any) provides the base, else defaults.
        pfile = get_persistent_config_file()
        base = cls.from_file(pfile) if pfile.exists() else cls()

        # 2. Environment overlays the file (env wins). from_env only sets the
        #    directories when the variables are actually present.
        env = cls.from_env()
        if env.config_dir is not None:
            base.config_dir = env.config_dir
        if env.slurm_templates_dir is not None:
            base.slurm_templates_dir = env.slurm_templates_dir

        # 3. Default config_dir to the persistent dir so cluster JSONs have a
        #    home even before anything is configured.
        if base.config_dir is None:
            base.config_dir = get_persistent_config_dir()

        return base

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"RemoteHandlerConfig(\n"
            f"  config_dir={self.config_dir},\n"
            f"  slurm_templates_dir={self.slurm_templates_dir},\n"
            f"  default_partition={self.default_partition!r},\n"
            f"  default_conda_env={self.default_conda_env!r}\n"
            f")"
        )


# Global configuration instance (singleton pattern)
_config: Optional[RemoteHandlerConfig] = None


def get_config() -> RemoteHandlerConfig:
    """Get the global configuration instance.

    Lazily resolves configuration on first access with precedence
    environment > persistent config file > built-in defaults.

    Returns:
        Global RemoteHandlerConfig instance

    Example:
        >>> config = get_config()
        >>> config_dir = config.get_config_dir()
    """
    global _config
    if _config is None:
        _config = RemoteHandlerConfig.resolve()
        logger.debug("Initialized configuration (env + persistent file)")
    return _config


def set_config(config: RemoteHandlerConfig) -> None:
    """Set the global configuration instance.

    Useful for testing or when you need to override configuration.

    Args:
        config: RemoteHandlerConfig instance to use globally

    Example:
        >>> # For testing
        >>> test_config = RemoteHandlerConfig(config_dir=Path("/tmp/test"))
        >>> set_config(test_config)
    """
    global _config
    _config = config
    logger.debug(f"Set global configuration: {config}")


def reset_config() -> None:
    """Reset global configuration to None.

    Useful for testing to ensure clean state.

    Example:
        >>> reset_config()
        >>> config = get_config()  # Will reinitialize from environment
    """
    global _config
    _config = None
    logger.debug("Reset global configuration")
