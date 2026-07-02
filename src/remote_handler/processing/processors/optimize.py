"""The ``optimize`` processor -- one module, both ends of the contract.

This module holds the whole processor in one place:

* :class:`OptimizeArgs` -- the arg schema (single source of truth), used by the
  host to emit the payload and by the driver to validate it;
* :func:`run` -- the cluster-side compute (LBFGS relaxation; the calculator is
  built per backend, with heavy imports kept lazy);
* :class:`OptimizeProcessor` -- the host-side spec;
* the module-level :func:`register` call that wires ``run``/``OptimizeArgs`` into
  the registry.

The relaxation is plain ASE (LBFGS), owned here. Backends: ``NNFF`` / ``MACE``
build an ensemble NeuralIL calculator (from neuralil-jaxrens, imported lazily),
and ``LJ`` uses ASE's Lennard-Jones for quick smoke tests.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from remote_handler.exceptions import ScriptInputError
from remote_handler.processing.base import Processor
from remote_handler.processing.registry import register
from remote_handler.scripts.input_models import ScriptInput


# --- arg schema ------------------------------------------------------------
@dataclass
class OptimizeArgs(ScriptInput):
    """Args for the ``optimize`` processor.

    ``backend``/``fmax``/``steps`` are always required; the NNFF/MACE backends
    additionally need a ``pickle_file`` (the staged model) and carry the
    neighbour-bucket / supercell knobs. Fields left ``None`` are dropped by
    ``to_dict`` so an LJ payload stays minimal.
    """

    backend: str
    fmax: float
    steps: int
    pickle_file: Optional[str] = None
    max_neighbors: Optional[int] = None
    max_neighbors_list: Optional[list] = None
    neighbor_offset: Optional[int] = None
    supercell_diag: Optional[list] = None

    def validate(self, root: Optional[Path] = None) -> None:
        if self.backend not in ("NNFF", "MACE", "LJ"):
            raise ScriptInputError(
                "optimize: 'backend' must be one of ('NNFF', 'MACE', 'LJ'), "
                f"got {self.backend!r}"
            )
        if not isinstance(self.fmax, (int, float)):
            raise ScriptInputError("optimize: 'fmax' must be a number")
        if not isinstance(self.steps, int):
            raise ScriptInputError("optimize: 'steps' must be an int")
        if self.backend in ("NNFF", "MACE") and not self.pickle_file:
            raise ScriptInputError(
                f"optimize: backend={self.backend} requires a 'pickle_file'"
            )


# --- cluster-side compute --------------------------------------------------
def _build_calculator(args):
    backend = args["backend"]
    if backend in ("NNFF", "MACE"):
        # NeuralIL ASE calculator -- lives upstream in neuralil-jaxrens (migrated
        # out of the deprecated jaxnest.postprocess). One class serves single
        # models and ensembles; we load an ensemble (for uncertainties).
        # max_neighbors is a per-evaluation argument -- no compiled-function
        # cache. Imported lazily so importing this module host-side is cheap.
        from neuralil.ase_integration import NeuralILASECalculator

        return NeuralILASECalculator.from_pickle(
            args["pickle_file"],
            max_neighbors=args.get("max_neighbors"),
            max_neighbors_list=args.get("max_neighbors_list"),
            neighbor_offset=args.get("neighbor_offset", 0),
            supercell_diag=tuple(args.get("supercell_diag", (1, 1, 1))),
            ensemble=True,
        )
    if backend == "LJ":
        from ase.calculators.lj import LennardJones

        return LennardJones(sigma=1.0, epsilon=1.0, rc=3.0)
    raise ValueError(f"Unknown optimization backend: {backend!r}")


def run(atoms_list, args, calcpath=None):
    from ase.optimize import LBFGS

    # Build the calculator lazily on first use: a fully-resumed shard (every
    # config already converged) then never loads the model at all.
    calc = None
    out = []
    for at in atoms_list:
        if "is_converged" not in at.info:
            if calc is None:
                calc = _build_calculator(args)
            relaxed = at.copy()
            relaxed.calc = calc
            opt = LBFGS(relaxed, logfile=None)
            converged = opt.run(fmax=args["fmax"], steps=args["steps"])
            relaxed.info["is_converged"] = bool(converged)
            at = relaxed
        out.append(at)
    return out


# --- host-side spec --------------------------------------------------------
class OptimizeProcessor(Processor):
    """Relax each structure with an NNFF/MACE ensemble or an LJ calculator."""

    name = "optimize"
    registration_module = None  # built-in: run is always in the registry

    def __init__(
        self,
        pickle_file: Optional[str] = None,
        backend: str = "NNFF",
        fmax: float = 0.1,
        steps: int = 200,
        max_neighbors: int = 60,
        max_neighbors_list: Optional[list[int]] = None,
        neighbor_offset: int = 0,
        supercell_diag: tuple[int, int, int] = (1, 1, 1),
    ):
        if backend in ("NNFF", "MACE") and pickle_file is None:
            raise ValueError(
                f"optimize backend={backend!r} requires a pickle_file "
                "(the model parameters)."
            )
        self.pickle_file = pickle_file
        self.backend = backend
        self.fmax = fmax
        self.steps = steps
        # The NeuralIL calculator picks a neighbour bucket per structure. Pass a
        # single cap (max_neighbors) or a ladder (max_neighbors_list) to bound
        # recompiles; neighbor_offset adds headroom. An optional supercell
        # replicates small cells so the cutoff sphere fits.
        self.max_neighbors = max_neighbors
        self.max_neighbors_list = max_neighbors_list
        self.neighbor_offset = neighbor_offset
        self.supercell_diag = supercell_diag

    def build_args(self, savepath: str) -> dict:
        # Fill the shared schema so the field names come from one place; the
        # cluster driver validates the payload against this same schema.
        kwargs = dict(backend=self.backend, fmax=self.fmax, steps=self.steps)
        if self.backend in ("NNFF", "MACE"):
            remote_pickle = os.path.split(self.pickle_file)[1]
            kwargs.update(
                pickle_file=str((Path(savepath) / remote_pickle).resolve()),
                max_neighbors=self.max_neighbors,
                neighbor_offset=self.neighbor_offset,
                supercell_diag=list(self.supercell_diag),
            )
            if self.max_neighbors_list is not None:
                kwargs["max_neighbors_list"] = list(self.max_neighbors_list)
        return OptimizeArgs(**kwargs).to_dict()

    def shared_files(self) -> list:
        if self.backend in ("NNFF", "MACE"):
            return [self.pickle_file]
        return []


register("optimize", run, schema=OptimizeArgs)
