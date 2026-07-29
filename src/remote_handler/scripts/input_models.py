"""Input schemas for remote_handler scripts.

These dataclasses are the input contract for the scripts in ``scripts/``,
which run **on the remote machine**. Each script loads its JSON input into the
corresponding model and validates it at the top of execution:

    >>> from remote_handler.scripts.input_models import MaceTrainingInput
    >>> config = MaceTrainingInput.from_file(Path("input.json"))
    >>> config.validate()

Validation lives here (remote-side) on purpose: the checks that actually
matter -- chiefly that the referenced paths exist -- can only be done where the
paths are real and final, i.e. on the remote machine. The driver does not
pre-validate against these models; it just assembles the input JSON. Keep the
``validate`` methods as the single place that enforces a script's input
contract, and extend them with the path-existence / domain checks each script
needs.
"""

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional

from remote_handler.exceptions import ScriptInputError

logger = logging.getLogger(__name__)


def _require_path(root: Path, value, label: str) -> None:
    """Raise ``ScriptInputError`` if a referenced path does not exist.

    Relative paths are resolved against ``root`` -- the script's calc dir,
    where the transferred input bundle was unpacked; absolute paths are checked
    as-is. This is the existence check that can only run remotely, where the
    paths are final and reachable, which is why ``validate`` takes a ``root``.
    """
    p = Path(value)
    full = p if p.is_absolute() else Path(root) / p
    if not full.exists():
        raise ScriptInputError(f"{label} does not exist: {full}")


def _require_jax_format(value: str) -> None:
    """Shared check for the ``jax_output_format`` field of the MACE inputs."""
    if value not in ("pkl", "bundle"):
        raise ScriptInputError(
            f"jax_output_format must be 'pkl' or 'bundle', got '{value}'"
        )


@dataclass
class ScriptInput:
    """Base class for script input configurations.

    Provides shared ``from_file`` and ``from_dict`` class methods so that
    subclasses only need to declare their fields and a ``validate`` method.

    Subclasses can set ``_path_fields`` to a tuple of field names that should
    be converted from strings to ``Path`` objects when loading from a dict.
    """

    _path_fields: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def from_dict(cls, data: dict) -> "ScriptInput":
        """Create an instance from a dictionary, filtering unknown keys.

        Unknown keys are silently filtered with a warning log message.
        Fields listed in ``_path_fields`` are converted to ``Path`` objects.

        Args:
            data: Dictionary of input parameters

        Returns:
            Instance of the subclass

        Raises:
            ScriptInputError: If required fields are missing
        """
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            logger.warning(f"{cls.__name__}: ignoring unknown keys {unknown}")
        filtered = {k: v for k, v in data.items() if k in known}

        # Convert path fields
        for field in cls._path_fields:
            if field in filtered:
                filtered[field] = Path(filtered[field]).expanduser()

        try:
            return cls(**filtered)
        except TypeError as e:
            raise ScriptInputError(
                f"Invalid or missing fields for {cls.__name__}: {e}"
            )

    @classmethod
    def from_file(cls, input_file: Path) -> "ScriptInput":
        """Load configuration from a JSON file.

        Args:
            input_file: Path to JSON input file

        Returns:
            Instance of the subclass

        Raises:
            ScriptInputError: If file doesn't exist, invalid JSON, or
                missing fields
        """
        input_file = Path(input_file).expanduser()

        if not input_file.exists():
            raise ScriptInputError(f"Input file not found: {input_file}")

        try:
            with open(input_file) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ScriptInputError(
                f"Invalid JSON in input file: {input_file}\n"
                f"Error at line {e.lineno}, column {e.colno}: {e.msg}"
            )
        except Exception as e:
            raise ScriptInputError(
                f"Failed to read input file: {input_file}\n" f"Error: {e}"
            )

        return cls.from_dict(data)

    def validate(self, root: Optional[Path] = None) -> None:
        """Validate configuration parameters.

        Override in subclasses to add domain-specific checks. Data-shape checks
        (types, ranges, required combinations) always run. Path-existence
        checks run only when ``root`` is given -- pass the script's calc dir so
        in-bundle paths can be resolved and verified on the remote.

        Raises:
            ScriptInputError: If any parameter is invalid
        """
        pass

    def to_dict(self) -> dict:
        """Serialise to a plain dict, dropping unset (``None``) fields.

        The inverse of :meth:`from_dict` for the keys that matter: a host-side
        builder fills the schema and calls this to emit the payload, so the
        field names are declared once (on the dataclass) rather than hand-listed
        at every emit site.
        """
        return {
            k: v for k, v in dataclasses.asdict(self).items() if v is not None
        }


@dataclass
class MaceTrainingInput(ScriptInput):
    """Input configuration for mace_training.py script.

    Attributes:
        database_files: List of paths to database files (will be combined if multiple)
        train_runner: Path to MACE train_runner executable (e.g., mace_run_train)
        config_file: Path to user's YAML config with training parameters
        model_name: Name tag for the trained model
        seed: Random seed for database splitting and training
        validation_fraction: Fraction of data to use for validation (default: 0.1)
        convert_to_jax: If True, convert the trained torch model to a mace-jax
            bundle after training (needs mace-jax in the same env)
        jax_output_format: Bundle format when ``convert_to_jax`` -- ``"pkl"``
            or ``"bundle"``

    Example:
        >>> config = MaceTrainingInput.from_file(Path("input_train.json"))
        >>> config.validate()
    """

    _path_fields: ClassVar[tuple[str, ...]] = ("train_runner", "config_file")

    database_files: list[str]
    train_runner: Path
    config_file: Path
    model_name: str
    seed: int
    validation_fraction: float = 0.1
    convert_to_jax: bool = False
    # "pkl" (default) -> opaque pickle bundle; "bundle" -> a directory with
    # config.json + params.msgpack (mace-jax's canonical, inspectable format).
    jax_output_format: str = "pkl"

    def validate(self, root: Optional[Path] = None) -> None:
        """Validate configuration parameters.

        Raises:
            ScriptInputError: If any parameter is invalid
        """
        if not self.database_files:
            raise ScriptInputError("database_files cannot be empty")

        if not (0 < self.validation_fraction < 1):
            raise ScriptInputError(
                f"validation_fraction must be between 0 and 1, got {self.validation_fraction}"
            )

        if not self.model_name:
            raise ScriptInputError("model_name cannot be empty")

        _require_jax_format(self.jax_output_format)

        if root is not None:
            for f in self.database_files:
                _require_path(root, f, "database file")
            _require_path(root, self.config_file, "config_file")


@dataclass
class MaceFinetuningInput(ScriptInput):
    """Input configuration for mace_finetuning.py script.

    Extends the standard MACE training input with a foundation model
    that is used as the starting point for finetuning.

    Attributes:
        database_files: List of paths to database files (will be combined if multiple)
        train_runner: Path to MACE train_runner executable (e.g., mace_run_train)
        config_file: Path to user's YAML config with training parameters
        foundation_model: Path to the pretrained foundation model file
        model_name: Name tag for the trained model
        seed: Random seed for database splitting and training
        validation_fraction: Fraction of data to use for validation (default: 0.1)
        convert_to_jax: If True, convert the trained torch model to a mace-jax
            ``.pkl`` bundle after training (needs mace-jax in the same env)
    """

    _path_fields: ClassVar[tuple[str, ...]] = (
        "train_runner",
        "config_file",
        "foundation_model",
    )

    database_files: list[str]
    train_runner: Path
    config_file: Path
    foundation_model: Path
    model_name: str
    seed: int
    validation_fraction: float = 0.1
    convert_to_jax: bool = False
    # "pkl" (default) -> opaque pickle bundle; "bundle" -> a directory with
    # config.json + params.msgpack (mace-jax's canonical, inspectable format).
    jax_output_format: str = "pkl"

    def validate(self, root: Optional[Path] = None) -> None:
        """Validate configuration parameters.

        Raises:
            ScriptInputError: If any parameter is invalid
        """
        if not self.database_files:
            raise ScriptInputError("database_files cannot be empty")

        if not (0 < self.validation_fraction < 1):
            raise ScriptInputError(
                f"validation_fraction must be between 0 and 1, got {self.validation_fraction}"
            )

        if not self.model_name:
            raise ScriptInputError("model_name cannot be empty")

        _require_jax_format(self.jax_output_format)

        if root is not None:
            for f in self.database_files:
                _require_path(root, f, "database file")
            _require_path(root, self.config_file, "config_file")
            _require_path(root, self.foundation_model, "foundation_model")


@dataclass
class DftSinglepointInput(ScriptInput):
    """Input configuration for dft_singlepoint.py script.

    Attributes:
        DATABASE: Path to input structure file
        launch_command: Command to launch VASP
        INCAR_kwargs: Dictionary of VASP INCAR parameters

    Example:
        >>> config = DftSinglepointInput.from_file(Path("input_opt.json"))
        >>> config.validate()
    """

    DATABASE: str
    launch_command: str
    INCAR_kwargs: dict[str, Any]

    def validate(self, root: Optional[Path] = None) -> None:
        """Validate configuration parameters.

        Raises:
            ScriptInputError: If any parameter is invalid
        """
        if not self.DATABASE:
            raise ScriptInputError("DATABASE cannot be empty")

        if not self.launch_command:
            raise ScriptInputError("launch_command cannot be empty")

        if not isinstance(self.INCAR_kwargs, dict):
            raise ScriptInputError("INCAR_kwargs must be a dictionary")

        if root is not None:
            _require_path(root, self.DATABASE, "DATABASE")


@dataclass
class OrcaSinglepointInput(ScriptInput):
    """Input configuration for orca_singlepoint.py script.

    Attributes:
        DATABASE: Path to input structure file (extxyz or ASE-readable format)
        orca_binary: Full path to ORCA executable
        orcasimpleinput: Simple input line (functional, basis set, method)
        orcablocks: ORCA block input for advanced settings (e.g. parallelization)

    Example:
        >>> config = OrcaSinglepointInput.from_file(Path("input_dft.json"))
        >>> config.validate()
    """

    DATABASE: str
    orca_binary: str
    orcasimpleinput: str
    orcablocks: str

    def validate(self, root: Optional[Path] = None) -> None:
        """Validate configuration parameters.

        Raises:
            ScriptInputError: If any parameter is invalid
        """
        if not self.DATABASE:
            raise ScriptInputError("DATABASE cannot be empty")

        if not self.orca_binary:
            raise ScriptInputError("orca_binary cannot be empty")

        if not self.orcasimpleinput:
            raise ScriptInputError("orcasimpleinput cannot be empty")

        if not self.orcablocks:
            raise ScriptInputError("orcablocks cannot be empty")

        if root is not None:
            _require_path(root, self.DATABASE, "DATABASE")


@dataclass
class AseSinglepointInput(ScriptInput):
    """Input configuration for the generic ``ase_singlepoint.py`` script.

    Drives a single-point with any pure-ASE calculator (e.g. ``emt``,
    ``eam``), needing no external binary -- so it runs anywhere ASE is
    installed (locally, in CI, or on a cluster).

    Attributes:
        DATABASE: Path to the input structure file (ASE-readable).
        calculator: Calculator name understood by ``ase_singlepoint.py``
            (e.g. ``"emt"``).
        calculator_kwargs: Keyword arguments forwarded to the calculator.

    Example:
        >>> config = AseSinglepointInput.from_file(Path("input_dft.json"))
        >>> config.validate(Path("."))
    """

    DATABASE: str
    calculator: str
    calculator_kwargs: dict[str, Any] = field(default_factory=dict)

    def validate(self, root: Optional[Path] = None) -> None:
        """Validate configuration parameters.

        Raises:
            ScriptInputError: If any parameter is invalid
        """
        if not self.DATABASE:
            raise ScriptInputError("DATABASE cannot be empty")

        if not self.calculator:
            raise ScriptInputError("calculator cannot be empty")

        if not isinstance(self.calculator_kwargs, dict):
            raise ScriptInputError("calculator_kwargs must be a dictionary")

        if root is not None:
            _require_path(root, self.DATABASE, "DATABASE")
