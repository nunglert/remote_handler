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

        if self.jax_output_format not in ("pkl", "bundle"):
            raise ScriptInputError(
                "jax_output_format must be 'pkl' or 'bundle', got "
                f"'{self.jax_output_format}'"
            )

        if root is not None:
            for f in self.database_files:
                _require_path(root, f, "database file")
            _require_path(root, self.config_file, "config_file")
            _require_path(root, self.foundation_model, "foundation_model")


@dataclass
class LammpsSamplingInput(ScriptInput):
    """Input configuration for lammps/lammps_sampling.py script.

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

    def validate(self, root: Optional[Path] = None) -> None:
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

        if root is not None:
            _require_path(root, self.data_file, "data_file")
            _require_path(root, self.template_file, "template_file")
            for m in self.model_files:
                _require_path(root, m, "model file")


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


@dataclass
class CmaesSamplingInput(ScriptInput):
    """Input configuration for clinamen2/cmaes_sampling.py script.

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

    def validate(self, root: Optional[Path] = None) -> None:
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

        if root is not None:
            if self.founder and self.founder != "random":
                _require_path(root, self.founder, "founder")
            for m in self.models or []:
                _require_path(root, m, "model file")


@dataclass
class CmaesOptimizeInput(ScriptInput):
    """Input configuration for clinamen2/cmaes_optimize.py script.

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

    def validate(self, root: Optional[Path] = None) -> None:
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

        if root is not None:
            for m in self.models:
                _require_path(root, m, "model file")


@dataclass
class TorchSimSamplingInput(ScriptInput):
    """Input configuration for torch_sim/torch_sim_sampling.py script.

    Attributes:
        founder: Filename of the founder structure (ASE-readable format)
        model_paths: List of MACE model file paths.  Use ``["foundation"]``
            for the MACE-MP foundation model.  A single-element list runs a
            standard single-model simulation; multiple entries activate the
            ``MaceEnsembleModel`` (shared-body ensemble with uncertainty).
        ensemble: Ensemble type, "nvt" or "npt"
        temperatures: List of replica temperatures in K
        target_temperature: Target temperature in K (used for initial NPT
            equilibration in NVT mode). Required when ensemble="nvt".
        dt: Integration timestep in ps (default: 0.001)
        gamma: Friction coefficient for Langevin thermostat (optional)
        alpha: Friction coefficient for particle Langevin (NPT, optional)
        cell_alpha: Friction coefficient for cell Langevin (NPT, optional)
        n_steps_npt: Number of NPT equilibration steps (default: 0)
        n_steps_pt: Number of production parallel tempering steps (default: 100)
        seed: Random seed (default: 123)
        device: Torch device string (default: "cuda")
        dtype: Torch dtype string, "float32" or "float64" (default: "float32")

    Example:
        >>> config = TorchSimSamplingInput.from_file(Path("input.json"))
        >>> config.validate()
    """

    founder: str
    model_paths: list[str]
    ensemble: str
    temperatures: list[float]
    target_temperature: Optional[float] = None
    dt: float = 0.001
    gamma: Optional[float] = None
    alpha: Optional[float] = None
    cell_alpha: Optional[float] = None
    n_steps_npt: int = 0
    n_steps_pt: int = 100
    seed: int = 123
    device: str = "cuda"
    dtype: str = "float32"

    def validate(self, root: Optional[Path] = None) -> None:
        """Validate configuration parameters.

        Raises:
            ScriptInputError: If any parameter is invalid
        """
        if not self.founder:
            raise ScriptInputError("founder cannot be empty")

        if not self.model_paths:
            raise ScriptInputError("model_paths cannot be empty")

        if len(self.model_paths) > 1 and "foundation" in self.model_paths:
            raise ScriptInputError(
                "Cannot mix 'foundation' with other model paths in ensemble mode"
            )

        valid_ensembles = {"nvt", "npt"}
        if self.ensemble not in valid_ensembles:
            raise ScriptInputError(
                f"ensemble must be one of {valid_ensembles}, got '{self.ensemble}'"
            )

        if not self.temperatures:
            raise ScriptInputError("temperatures cannot be empty")

        if any(t <= 0 for t in self.temperatures):
            raise ScriptInputError("all temperatures must be positive")

        if self.ensemble == "nvt" and self.target_temperature is None:
            raise ScriptInputError(
                "target_temperature is required when ensemble='nvt'"
            )

        if self.dt <= 0:
            raise ScriptInputError(f"dt must be positive, got {self.dt}")

        if self.n_steps_pt <= 0:
            raise ScriptInputError(
                f"n_steps_pt must be positive, got {self.n_steps_pt}"
            )

        valid_dtypes = {"float32", "float64"}
        if self.dtype not in valid_dtypes:
            raise ScriptInputError(
                f"dtype must be one of {valid_dtypes}, got '{self.dtype}'"
            )

        if root is not None:
            _require_path(root, self.founder, "founder")
            for m in self.model_paths:
                if m != "foundation":  # sentinel for the MACE-MP foundation
                    _require_path(root, m, "model file")
