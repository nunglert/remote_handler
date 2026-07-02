"""Remote per-shard trajectory processing.

A trajectory (set of ``.extxyz`` files) is sharded into batches; each batch is
processed in its own array-job task on a remote cluster, and the results are
collected and reshaped back into the original layout.

The design separates the two concerns that pulled against each other:

* **Data movement** (shared): sharding, distributing the trajectory, and
  collecting/reshaping results -- done once, in :class:`RemoteTrajectoryProcessor`
  (and, beneath it, :class:`~remote_handler.core.atoms_handler.RemoteAtomsHandler`).
* **Computation** (composable): a *list* of :class:`Processor` steps. Each is a
  small, single-responsibility unit contributing only its own remote arguments
  and any files it needs staged. The remote payload (``process_traj.py``) reads
  a shard once and runs the requested steps in order via a processor registry
  (``scripts/trajectory/processors``), so a single job can e.g. optimize *then*
  characterize the relaxed structures, with the trajectory distributed only
  once. Each processor sources its science upstream (the NNFF calculator from
  neuralil-jaxrens, Steinhardt from jaxrens); dependency-light glue (LBFGS
  relaxation, the ASE RDF) is owned here.

``RemoteOptimizationHandler`` and ``RemoteAnalysisHandler`` are thin factories
over the same core -- they just assemble the processor list for the common
cases.
"""

import logging
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Optional

from remote_handler.core import file_handler, tasks
from remote_handler.core.atoms_handler import RemoteAtomsHandler
from remote_handler.core.cluster_handler import ClusterConfig

logger = logging.getLogger(__name__)

# Remote payloads shipped with the package live under ``scripts/``.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
_PROCESS_SCRIPT = _SCRIPTS_DIR / "trajectory" / "process_traj.py"

# The collector (``AtomsShardManager.collect``) locates each shard's processed
# output as ``front_tag("opt.extxyz", tag)``, so the per-shard working
# trajectory file must be named this for every processing type. It is the
# array task's data file, not specific to optimization.
_SHARD_ATOMS_NAME = "opt.extxyz"


# ---------------------------------------------------------------------------
# Processors -- the composable computation layer (host-side specs)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Handler -- the shared data-movement layer + pipeline wiring
# ---------------------------------------------------------------------------
class RemoteTrajectoryProcessor(RemoteAtomsHandler):
    """Run an ordered pipeline of :class:`Processor` steps over a trajectory.

    Shards the input trajectories, ships each shard a per-shard input JSON (the
    ordered ``steps`` + per-step ``args``) plus its atoms batch and any shared
    files, runs the array job, then collects and reshapes the results. The
    trajectory is distributed once regardless of how many steps run.

    Args:
        traj_files: Trajectory ``.extxyz`` file(s) to process.
        cluster_config: Target cluster configuration.
        remote_fs: Filesystem handler for the machine where jobs run.
        n_processors: Processors (or GPUs) requested per array task.
        shard_size: Number of configurations per shard.
        savepath: Remote working directory for the array job.
        num_parallel: Maximum array tasks to run concurrently.
        processors: Ordered processing steps to run on each shard.
        stub: If True, the remote payload runs in mock mode.
        python_script_path: Remote payload; defaults to the shipped
            ``trajectory/process_traj.py``.
        tag: Output-file tag (drives collection).
        loop_time: Seconds between completion polls.
        jobname: SLURM job name.
        every/start/end: Optional slice applied before sharding.
        local_savepath: Optional local directory to mirror results into.
        logfile: Per-task log filename.
        collect_only: If True, skip staging and only collect prior results.
        slurm_header_add: Extra SLURM header lines (default loads OpenBLAS,
            which the processor payloads link against).
    """

    _input_json_name = "input_process.json"

    def __init__(
        self,
        traj_files: str,
        cluster_config: ClusterConfig,
        remote_fs: file_handler.FileSystemHandlerBase,
        n_processors: int,
        shard_size: int,
        savepath: str,
        num_parallel: int,
        processors: list[Processor],
        *,
        stub: bool = False,
        python_script_path: Optional[os.PathLike] = None,
        tag: str = "proc",
        loop_time: float = 30,
        jobname: str = "proc",
        every: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        local_savepath: Optional[os.PathLike] = None,
        logfile: str = "proc.log",
        collect_only: bool = False,
        slurm_header_add: str = "module load openblas",
    ):
        if not processors:
            raise ValueError(
                "RemoteTrajectoryProcessor needs at least one processor."
            )
        names = [p.name for p in processors]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate processor names: {names}")
        self.processors = processors
        self._stub = stub

        super().__init__(
            traj_files,
            cluster_config,
            remote_fs,
            shard_size,
            savepath,
            num_parallel,
            loop_time=loop_time,
            jobname=jobname,
            every=every,
            start=start,
            end=end,
            local_savepath=local_savepath,
            logfile=logfile,
            collect_only=collect_only,
            tag=tag,
            n_processors=n_processors,
            slurm_header_add=slurm_header_add,
        )

        # Parent leaves these for subclasses to populate.
        self.python_script_file = (
            python_script_path
            if python_script_path is not None
            else _PROCESS_SCRIPT
        )
        self.shared_ressources = self._collect_shared_files()

        if not collect_only:
            self.prepare_individual_data()

    def _base_json(self) -> dict:
        """The per-shard input dict (``DATABASE`` is added per shard)."""
        return {
            "STUB": self._stub,
            "tag": self.tag,
            "steps": [p.name for p in self.processors],
            "args": {
                p.name: p.build_args(str(self.savepath))
                for p in self.processors
            },
        }

    def _collect_shared_files(self) -> list:
        files = []
        for p in self.processors:
            files.extend(p.shared_files())
        return files

    def prepare_individual_data(self):
        """Build the per-shard input JSON + atoms batch for every shard."""
        config_files = []
        atoms_batches = []

        for calc_idx in range(self.num_array_tasks):
            json_config = deepcopy(self._base_json())
            json_config["DATABASE"] = _SHARD_ATOMS_NAME

            config_files.append(
                tasks.DictJSON(
                    json_config, self._input_json_name, self.remote_fs
                )
            )
            atoms_batches.append(
                tasks.AtomsBatch(
                    self.shards[calc_idx], _SHARD_ATOMS_NAME, self.remote_fs
                )
            )

        del self.shards
        self.individual_ressources = [config_files, atoms_batches]

    def run(self):
        """Stage, submit, and collect+reshape the results."""
        self.allocate_data()  # From ArrayScriptTask
        self.submit()  # From ArrayScriptTask
        self.collect_and_harvest()  # From RemoteAtomsHandler


# ---------------------------------------------------------------------------
# Thin factories over the composable core
# ---------------------------------------------------------------------------
class RemoteOptimizationHandler(RemoteTrajectoryProcessor):
    """Optimize a sharded trajectory (a one-step :class:`OptimizeProcessor`)."""

    def __init__(
        self,
        traj_files: str,
        cluster_config: ClusterConfig,
        remote_fs: file_handler.FileSystemHandlerBase,
        n_processors: int,
        shard_size: int,
        savepath: str,
        num_parallel: int,
        pickle_file: Optional[str] = None,
        optimization_backend: str = "NNFF",
        fmax: float = 0.1,
        steps: int = 200,
        max_neighbors: int = 60,
        max_neighbors_list: Optional[list[int]] = None,
        neighbor_offset: int = 0,
        supercell_diag: tuple[int, int, int] = (1, 1, 1),
        python_script_path: Optional[os.PathLike] = None,
        stub: bool = False,
        loop_time: float = 30,
        jobname: str = "opt",
        every: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        local_savepath: Optional[os.PathLike] = None,
        logfile: str = "opt.log",
        collect_only: bool = False,
        tag: str = "opt",
    ):
        super().__init__(
            traj_files,
            cluster_config,
            remote_fs,
            n_processors,
            shard_size,
            savepath,
            num_parallel,
            processors=[
                OptimizeProcessor(
                    pickle_file=pickle_file,
                    backend=optimization_backend,
                    fmax=fmax,
                    steps=steps,
                    max_neighbors=max_neighbors,
                    max_neighbors_list=max_neighbors_list,
                    neighbor_offset=neighbor_offset,
                    supercell_diag=supercell_diag,
                )
            ],
            stub=stub,
            python_script_path=python_script_path,
            tag=tag,
            loop_time=loop_time,
            jobname=jobname,
            every=every,
            start=start,
            end=end,
            local_savepath=local_savepath,
            logfile=logfile,
            collect_only=collect_only,
        )


class RemoteAnalysisHandler(RemoteTrajectoryProcessor):
    """Compute structural descriptors on a sharded trajectory.

    A thin factory over :class:`RemoteTrajectoryProcessor` for the built-in
    analyses. Currently that is the RDF; Steinhardt lives in a downstream
    package (which depends on jaxrens) and registers itself as a processor --
    see :func:`~remote_handler.scripts.trajectory.processors.register`. For a
    custom pipeline, use :class:`RemoteTrajectoryProcessor` directly.
    """

    def __init__(
        self,
        traj_files: str,
        cluster_config: ClusterConfig,
        remote_fs: file_handler.FileSystemHandlerBase,
        n_processors: int,
        shard_size: int,
        savepath: str,
        num_parallel: int,
        do_rdf: bool = True,
        cutoff: float = 5.0,
        nbins: int = 200,
        python_script_path: Optional[os.PathLike] = None,
        stub: bool = False,
        loop_time: float = 30,
        jobname: str = "analysis",
        every: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        local_savepath: Optional[os.PathLike] = None,
        logfile: str = "analysis.log",
        collect_only: bool = False,
        tag: str = "analysis",
    ):
        procs: list[Processor] = []
        if do_rdf:
            procs.append(RDFProcessor(cutoff=cutoff, nbins=nbins))
        if not procs:
            raise ValueError(
                "RemoteAnalysisHandler has nothing to compute (enable do_rdf, "
                "or use RemoteTrajectoryProcessor with your own processors)."
            )
        super().__init__(
            traj_files,
            cluster_config,
            remote_fs,
            n_processors,
            shard_size,
            savepath,
            num_parallel,
            processors=procs,
            stub=stub,
            python_script_path=python_script_path,
            tag=tag,
            loop_time=loop_time,
            jobname=jobname,
            every=every,
            start=start,
            end=end,
            local_savepath=local_savepath,
            logfile=logfile,
            collect_only=collect_only,
        )
