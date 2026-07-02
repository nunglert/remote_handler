"""The driver's overall input schema.

``ProcessTrajInput`` is the JSON contract the shard driver (``driver.py``) loads
and validates: the ordered ``steps``, their per-step ``args`` map, and the
downstream ``registration_modules`` to import. Per-step arg validation is *not*
here -- those contracts live with each processor and are enforced by the driver
via the registry's ``validate_args`` (see :mod:`remote_handler.processing.
registry`).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from remote_handler.exceptions import ScriptInputError
from remote_handler.scripts.input_models import ScriptInput, _require_path


@dataclass
class ProcessTrajInput(ScriptInput):
    """Input configuration for the trajectory-processing driver.

    Drives a pipeline of processors over a trajectory shard. Rather than a
    fixed set of ``do_X`` flags, it carries an ordered list of processor
    ``steps`` and a per-step ``args`` map -- so optimization and any structural
    analysis compose in one job, and adding a processor needs no schema change.

    Attributes:
        tag: Tag to prepend to output files
        DATABASE: Path to input trajectory file
        steps: Ordered processor names to run (e.g. ``["optimize",
            "rdf"]``); each must exist in the processor registry
        args: Map of processor name -> its argument dict
        STUB: If True, the remote payload runs in mock mode
        registration_modules: Import paths the driver imports (idempotently)
            before resolving ``steps``, so downstream processors register their
            ``run`` in the registry. Empty for built-in-only pipelines.

    Example:
        >>> config = ProcessTrajInput.from_file(Path("input_process.json"))
        >>> config.validate()
    """

    tag: str
    DATABASE: str
    steps: list[str]
    args: dict[str, Any]
    STUB: bool = False
    registration_modules: list[str] = field(default_factory=list)

    def validate(self, root: Optional[Path] = None) -> None:
        """Validate the *coarse* shape of a processor pipeline.

        This checks only what a generic schema can: a trajectory is named, at
        least one step is requested, and ``args`` is a per-step map. It
        deliberately does **not** validate each step's args -- those contracts
        belong to the processors (including downstream ones), which this generic
        input model must not import. The driver enforces them per step via the
        registry's ``validate_args`` right after importing the registration
        modules, so a malformed composed payload fails fast at job start.

        Raises:
            ScriptInputError: If any parameter is invalid
        """
        if not self.DATABASE:
            raise ScriptInputError("DATABASE cannot be empty")

        if not self.steps:
            raise ScriptInputError("steps cannot be empty")

        if not isinstance(self.args, dict):
            raise ScriptInputError("args must be a dictionary")

        if root is not None:
            _require_path(root, self.DATABASE, "DATABASE")
