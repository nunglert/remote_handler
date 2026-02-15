"""Input models for remote_handler scripts.

This module defines dataclasses for validating script inputs, providing:
- Type safety and IDE support
- Clear documentation of required/optional parameters
- Validation before script execution
- Consistent error handling

Each script has a corresponding Input dataclass that can be loaded from JSON:
    >>> from remote_handler.scripts.input_models import MaceTrainingInput
    >>> config = MaceTrainingInput.from_file(Path("input.json"))
    >>> config.validate()
"""

import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Optional

from remote_handler.exceptions import ScriptInputError

logger = logging.getLogger(__name__)


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

    def validate(self) -> None:
        """Validate configuration parameters.

        Override in subclasses to add domain-specific checks.

        Raises:
            ScriptInputError: If any parameter is invalid
        """
        pass


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

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ScriptInputError: If any parameter is invalid
        """
        if not self.database_files:
            raise ScriptInputError("database_files cannot be empty")

        if not self.config_file.exists():
            raise ScriptInputError(
                f"Config file does not exist: {self.config_file}"
            )

        if not (0 < self.validation_fraction < 1):
            raise ScriptInputError(
                f"validation_fraction must be between 0 and 1, got {self.validation_fraction}"
            )

        if not self.model_name:
            raise ScriptInputError("model_name cannot be empty")


@dataclass
class LammpsSamplingInput(ScriptInput):
    """Input configuration for lammps_sampling.py script.

    Attributes:
        lammps_path: Path to LAMMPS executable
        template_file: Name of LAMMPS input template file
        data_file: Name of initial structure data file
        model_files: List of paths to MACE model files (for committee)
        element_mapping: Dict mapping atom numbers to element symbols
        cutoff_radius: Cutoff radius for local uncertainty aggregation (Angstrom)
        bin_count: Number of trajectory segments for sampling
        seed: Random seed for LAMMPS
        lammps_config_name: Name for generated LAMMPS input file (default: "lammps.in")

    Example:
        >>> config = LammpsSamplingInput.from_file(Path("input.json"))
        >>> config.validate()
    """

    _path_fields: ClassVar[tuple[str, ...]] = ("lammps_path",)

    lammps_path: Path
    template_file: str
    data_file: str
    model_files: list[str]
    element_mapping: dict[str, str]
    cutoff_radius: float
    bin_count: int
    seed: int
    lammps_config_name: str = "lammps.in"

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ScriptInputError: If any parameter is invalid
        """
        if not self.lammps_path.exists():
            raise ScriptInputError(
                f"LAMMPS executable not found: {self.lammps_path}"
            )

        if not self.model_files:
            raise ScriptInputError("model_files cannot be empty")

        if self.cutoff_radius <= 0:
            raise ScriptInputError(
                f"cutoff_radius must be positive, got {self.cutoff_radius}"
            )

        if self.bin_count <= 0:
            raise ScriptInputError(
                f"bin_count must be positive, got {self.bin_count}"
            )


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

    def validate(self) -> None:
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


@dataclass
class OptimizeTrajInput(ScriptInput):
    """Input configuration for optimize_traj.py script.

    Attributes:
        tag: Tag to prepend to output files
        DATABASE: Path to input trajectory file
        do_optimization: Whether to perform structure optimization
        do_steinhardt: Whether to compute Steinhardt parameters
        do_rdf: Whether to compute radial distribution function
        general_args: General parameters (cutoff, nbins, etc.)
        calculator_args: Calculator configuration (optional, required if do_optimization=True)
        optimization_args: Optimization parameters (optional, required if do_optimization=True)
        steinhardt_args: Steinhardt parameters (optional, required if do_steinhardt=True)

    Example:
        >>> config = OptimizeTrajInput.from_file(Path("input_opt.json"))
        >>> config.validate()
    """

    tag: str
    DATABASE: str
    do_optimization: bool
    do_steinhardt: bool
    do_rdf: bool
    general_args: dict[str, Any]
    calculator_args: Optional[dict[str, Any]] = None
    optimization_args: Optional[dict[str, Any]] = None
    steinhardt_args: Optional[dict[str, Any]] = None

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ScriptInputError: If any parameter is invalid
        """
        if not self.DATABASE:
            raise ScriptInputError("DATABASE cannot be empty")

        if not isinstance(self.general_args, dict):
            raise ScriptInputError("general_args must be a dictionary")

        if self.do_optimization and not self.calculator_args:
            raise ScriptInputError(
                "calculator_args required when do_optimization=True"
            )

        if self.do_optimization and not self.optimization_args:
            raise ScriptInputError(
                "optimization_args required when do_optimization=True"
            )

        if self.do_steinhardt and not self.steinhardt_args:
            raise ScriptInputError(
                "steinhardt_args required when do_steinhardt=True"
            )


@dataclass
class CmaesSamplingInput(ScriptInput):
    """Input configuration for cmaes_sampling.py script.

    Attributes:
        founder: 'random' or filename of founder POSCAR (absolute path if not
            in default directory)
        label: Run label for the evolution
        random_seed: Random seed for evolution
        step_size: Initial CMA-ES step size
        generations: Number of generations to run
        pop_size: Population size (overrides CMA-ES default if set)
        save_nth: Save every nth generation
        batch: If True, NN eval is vmapped
        models: List of model file paths (space separated in original CLI)
        loss_mode: Loss function to use ('energy-only', 'with-uncertainties',
            'adversarial', 'loss-threshold')
        loss_threshold: Lower threshold for loss
        scaled_center: Scaled center as [x, y, z]
        radius: Radius of sphere around scaled_center
        c_r: c_r parameter for covariance matrix bias
        continue_evolution: Whether to continue a stopped evolution
        generation_checkpoint: Generation checkpoint to continue from
        repository_path: Path to ftmlip directory

    Example:
        >>> config = CmaesSamplingInput.from_file(Path("input.json"))
        >>> config.validate()
    """

    founder: str
    label: str = "dummy"
    random_seed: int = 0
    step_size: float = 1.0
    generations: int = 1000
    pop_size: Optional[int] = None
    save_nth: int = 1
    batch: bool = False
    models: Optional[list[str]] = None
    loss_mode: str = "energy-only"
    loss_threshold: float = -1e6
    scaled_center: Optional[list[float]] = None
    radius: Optional[float] = None
    c_r: float = 20.0
    continue_evolution: bool = False
    generation_checkpoint: Optional[int] = None
    repository_path: Optional[str] = None
    # Optional post-evolution optimization
    do_optimization: bool = False
    dof_mode: str = "full-dof"
    surface_axis: int = 2
    fix_range: Optional[list[float]] = None
    bulk_terminate: bool = False
    optimizer: str = "FIRE"

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ScriptInputError: If any parameter is invalid
        """
        if not self.founder:
            raise ScriptInputError("founder cannot be empty")

        if self.generations <= 0:
            raise ScriptInputError(
                f"generations must be positive, got {self.generations}"
            )

        if self.step_size <= 0:
            raise ScriptInputError(
                f"step_size must be positive, got {self.step_size}"
            )

        if self.save_nth <= 0:
            raise ScriptInputError(
                f"save_nth must be positive, got {self.save_nth}"
            )

        valid_loss_modes = {
            "energy-only",
            "with-uncertainties",
            "adversarial",
            "loss-threshold",
        }
        if self.loss_mode not in valid_loss_modes:
            raise ScriptInputError(
                f"loss_mode must be one of {valid_loss_modes}, got '{self.loss_mode}'"
            )

        if self.continue_evolution and self.models is None:
            raise ScriptInputError("models must be provided (cannot be None)")


@dataclass
class CmaesOptimizeInput(ScriptInput):
    """Input configuration for cmaes_optimize.py script.

    Attributes:
        evolutions: List of dicts, each describing one evolution to optimize.
            Required keys: 'label' (str), 'path' (str).
            Optional keys: 'dof_mode' (str), 'choose_gen' (int),
            'override_founder' (str), 'scaled_center' (list[float]),
            'radius' (float).
        models: List of MACE model file paths
        surface_axis: Surface normal axis (0, 1, or 2)
        fix_range: Scaled range of atoms to fix along surface normal
        bulk_terminate: If True, remove overlayer from bottom layer
        optimizer: Name of ASE optimizer class (e.g. 'FIRE', 'BFGS')
        output_dir: Subdirectory name for optimized output files

    Example:
        >>> config = CmaesOptimizeInput.from_file(Path("input_optimize.json"))
        >>> config.validate()
    """

    evolutions: list[dict]
    models: list[str]
    surface_axis: int = 2
    fix_range: Optional[list[float]] = None
    bulk_terminate: bool = False
    optimizer: str = "FIRE"
    output_dir: str = "optimized"

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ScriptInputError: If any parameter is invalid
        """
        if not self.evolutions:
            raise ScriptInputError("evolutions cannot be empty")

        if not self.models:
            raise ScriptInputError("models cannot be empty")

        for i, evo in enumerate(self.evolutions):
            if "label" not in evo:
                raise ScriptInputError(
                    f"evolutions[{i}] missing required key 'label'"
                )
            if "path" not in evo:
                raise ScriptInputError(
                    f"evolutions[{i}] missing required key 'path'"
                )
