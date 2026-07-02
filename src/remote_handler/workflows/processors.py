"""Host-side processor specs -- the composable computation layer.

A :class:`Processor` is the *host* half of a processing step: it declares only
*what* to run, the remote arguments it needs, and any files to stage -- nothing
about sharding or transport (that is :class:`~remote_handler.workflows.optimizer.
RemoteTrajectoryProcessor`). Each spec has a counterpart on the *remote* side in
``scripts/trajectory/processors`` (the actual ``run`` executed on the cluster).
The two are matched by :attr:`Processor.name`; the parallel module naming is
deliberate -- ``workflows.processors`` (specs) vs ``scripts.trajectory.processors``
(implementations) are the two ends of one contract, kept apart because host
orchestration and remote payloads never share a process.

Built-in specs (:class:`OptimizeProcessor`, :class:`RDFProcessor`) are here;
heavier analyses (e.g. a jaxrens Steinhardt) live in downstream packages and set
:attr:`Processor.registration_module` so the stock driver imports them.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class Processor(ABC):
    """One processing step in the remote pipeline.

    A ``Processor`` contributes *only* its own remote arguments and any files
    that must be staged for it -- nothing about sharding or transport. Its
    :attr:`name` keys both the ordered ``steps`` list and the per-step ``args``
    map in the shard input JSON, and must match a processor in the remote
    registry (``scripts/trajectory/processors``).
    """

    #: Registry key; must match a remote processor module name.
    name: str

    #: Import path of the module whose import registers this step's ``run`` in
    #: the cluster-side ``REGISTRY`` (via ``register(name, run)``). ``None`` for
    #: built-in processors, which are already in the always-imported registry.
    #: The stock ``process_traj.py`` driver imports every declared module before
    #: resolving steps, so a downstream processor needs no custom driver.
    registration_module: Optional[str] = None

    @abstractmethod
    def build_args(self, savepath: str) -> dict:
        """Return this step's remote args.

        Args:
            savepath: Remote working directory, for resolving the paths of any
                files this processor stages (see :meth:`shared_files`).
        """

    def shared_files(self) -> list:
        """Files staged once and shared across all shards (default: none)."""
        return []


class OptimizeProcessor(Processor):
    """Relax each structure with an NNFF/MACE ensemble or an LJ calculator."""

    name = "optimize"

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
        args = {
            "backend": self.backend,
            "fmax": self.fmax,
            "steps": self.steps,
        }
        if self.backend in ("NNFF", "MACE"):
            remote_pickle = os.path.split(self.pickle_file)[1]
            args.update(
                pickle_file=str((Path(savepath) / remote_pickle).resolve()),
                max_neighbors=self.max_neighbors,
                neighbor_offset=self.neighbor_offset,
                supercell_diag=list(self.supercell_diag),
            )
            if self.max_neighbors_list is not None:
                args["max_neighbors_list"] = list(self.max_neighbors_list)
        return args

    def shared_files(self) -> list:
        if self.backend in ("NNFF", "MACE"):
            return [self.pickle_file]
        return []


class RDFProcessor(Processor):
    """Annotate structures with the radial distribution function."""

    name = "rdf"

    def __init__(self, cutoff: float = 5.0, nbins: int = 200):
        self.cutoff = cutoff
        self.nbins = nbins

    def build_args(self, savepath: str) -> dict:
        return {"cutoff": self.cutoff, "nbins": self.nbins}
