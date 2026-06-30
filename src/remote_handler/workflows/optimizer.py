"""Remote per-shard trajectory processing handlers.

A trajectory (set of ``.extxyz`` files) is sharded into batches; each batch is
processed in its own array-job task on a remote cluster, and the results are
collected and reshaped back into the original trajectory layout.

The processing itself is split by responsibility into sibling handlers, each
carrying only its own parameters:

* :class:`RemoteOptimizationHandler` -- structure optimization with an NNFF/LJ
  calculator.
* :class:`RemoteAnalysisHandler` -- structural analysis (Steinhardt order
  parameters, radial distribution function).

Both share the sharding / staging / collection plumbing in
:class:`_RemoteTrajectoryHandler` (and, beneath it,
:class:`~remote_handler.core.atoms_handler.RemoteAtomsHandler`); they differ
only in the remote payload they ship, the per-shard input JSON they build, and
any files shared across shards.
"""

import logging
import os
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Optional

from remote_handler.core import file_handler, tasks
from remote_handler.core.atoms_handler import RemoteAtomsHandler
from remote_handler.core.cluster_handler import ClusterConfig

logger = logging.getLogger(__name__)

# Remote payloads shipped with the package live under ``scripts/``; each
# handler defaults its remote script to the one it was written for.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"

# The collector (``AtomsShardManager.collect``) locates each shard's processed
# output as ``front_tag("opt.extxyz", tag)``, so the per-shard working
# trajectory file must be named this for every processing type. It is the
# array task's data file, not specific to optimization.
_SHARD_ATOMS_NAME = "opt.extxyz"


class _RemoteTrajectoryHandler(RemoteAtomsHandler):
    """Shared plumbing for per-shard remote trajectory processing.

    Shards the input trajectories, ships each shard a per-shard input JSON plus
    its atoms batch, runs the array job, then collects and reshapes the results
    into the original trajectory layout. Subclasses decide *what* processing
    happens via three hooks -- the shipped remote payload
    (:meth:`_default_script`), the per-shard JSON payload (:meth:`_base_json`),
    and any files shared across all shards (:meth:`_shared_files`) -- so neither
    optimization nor analysis carries the other's parameters.

    Args:
        traj_files: Trajectory ``.extxyz`` file(s) to process.
        cluster_config: Target cluster configuration.
        remote_fs: Filesystem handler for the machine where jobs run.
        n_processors: Processors (or GPUs) requested per array task.
        shard_size: Number of configurations per shard.
        savepath: Remote working directory for the array job.
        num_parallel: Maximum array tasks to run concurrently.
        python_script_path: Remote payload; defaults to :meth:`_default_script`.
        tag: Output-file tag (drives collection); per processing type.
        loop_time: Seconds between completion polls.
        jobname: SLURM job name.
        every/start/end: Optional slice applied before sharding.
        local_savepath: Optional local directory to mirror results into.
        logfile: Per-task log filename.
        collect_only: If True, skip staging and only collect prior results.
        slurm_header_add: Extra SLURM header lines (default loads OpenBLAS,
            which the jaxnest payloads link against).
    """

    #: Per-shard input JSON filename (the remote payload reads this).
    _input_json_name = "input.json"

    def __init__(
        self,
        traj_files: str,
        cluster_config: ClusterConfig,
        remote_fs: file_handler.FileSystemHandlerBase,
        n_processors: int,
        shard_size: int,
        savepath: str,
        num_parallel: int,
        *,
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

        # Parent leaves these unset for subclasses to populate.
        self.python_script_file = (
            python_script_path
            if python_script_path is not None
            else self._default_script()
        )
        self.shared_ressources = self._shared_files()

        if not collect_only:
            self.prepare_individual_data()

    # -- hooks subclasses implement --------------------------------------
    @abstractmethod
    def _default_script(self) -> os.PathLike:
        """Path to the shipped remote payload for this processing type."""

    @abstractmethod
    def _base_json(self) -> dict:
        """Per-shard input dict (``DATABASE`` is added by the base class)."""

    def _shared_files(self) -> list:
        """Files staged once and shared across all shards (default: none)."""
        return []

    def _augment_shard_json(self, json_config: dict) -> None:
        """Per-shard tweak of the JSON (e.g. inject a resolved remote path)."""

    # -- shared implementation -------------------------------------------
    def prepare_individual_data(self):
        """Build the per-shard input JSON + atoms batch for every shard."""
        config_files = []
        atoms_batches = []

        for calc_idx in range(self.num_array_tasks):
            json_config = deepcopy(self._base_json())
            json_config["DATABASE"] = _SHARD_ATOMS_NAME
            self._augment_shard_json(json_config)

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


class RemoteOptimizationHandler(_RemoteTrajectoryHandler):
    """Optimize a sharded trajectory on a remote cluster.

    Each shard is relaxed with an NNFF ensemble (``"NNFF"``/``"MACE"``) or a
    Lennard-Jones (``"LJ"``) calculator. For NNFF/MACE the model ``pickle_file``
    is staged once as a shared resource and its resolved remote path is injected
    into each shard's input.

    Args:
        traj_files: Trajectory ``.extxyz`` file(s) to optimize.
        cluster_config: Target cluster configuration.
        remote_fs: Filesystem handler for the machine where jobs run.
        n_processors: Processors (or GPUs) requested per array task.
        shard_size: Number of configurations per shard.
        savepath: Remote working directory for the array job.
        num_parallel: Maximum array tasks to run concurrently.
        pickle_file: NNFF/MACE model parameters (required for those backends).
        optimization_backend: ``"NNFF"`` (default), ``"MACE"``, or ``"LJ"``.
        fmax: Force convergence tolerance.
        steps: Maximum optimization steps.
        supercell_trafo: Supercell transform for the NNFF calculator.
        max_neighbors_list: Per-species neighbour caps for the NNFF calculator.
        max_neighbors_offset: Offset added to the neighbour caps.
        python_script_path: Remote payload; defaults to the shipped
            ``jaxnest/optimize_traj.py``.
        stub: If True, the remote payload runs in mock mode.
        (plus the shared sharding/scheduling args of
        :class:`_RemoteTrajectoryHandler`.)
    """

    _input_json_name = "input_opt.json"

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
        supercell_trafo: list[int] = [3, 3, 3],
        max_neighbors_list: list[int] = [20, 30, 40, 50, 60],
        max_neighbors_offset: int = 3,
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
        self.pickle_file = pickle_file
        self.optimization_backend = optimization_backend
        self._json = {
            "STUB": stub,
            "tag": tag,
            "do_optimization": True,
            "calculator_args": dict(
                supercell_trafo=supercell_trafo,
                max_neighbors_list=max_neighbors_list,
                max_neighbors_offset=max_neighbors_offset,
                optimization_backend=optimization_backend,
            ),
            "optimization_args": dict(fmax=fmax, steps=steps),
        }
        super().__init__(
            traj_files,
            cluster_config,
            remote_fs,
            n_processors,
            shard_size,
            savepath,
            num_parallel,
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

    def _default_script(self) -> os.PathLike:
        return _SCRIPTS_DIR / "jaxnest" / "optimize_traj.py"

    def _base_json(self) -> dict:
        return self._json

    def _shared_files(self) -> list:
        if self.optimization_backend in ("NNFF", "MACE"):
            if self.pickle_file is None:
                raise ValueError(
                    f"optimization_backend={self.optimization_backend!r} "
                    "requires a pickle_file (the model parameters)."
                )
            return [self.pickle_file]
        return []

    def _augment_shard_json(self, json_config: dict) -> None:
        # The pickle is staged into savepath; point the calculator at its
        # resolved remote path (basename under the working directory).
        if self.pickle_file is not None:
            remote_pickle = os.path.split(self.pickle_file)[1]
            json_config["calculator_args"]["pickle_file"] = str(
                (Path(self.savepath) / remote_pickle).resolve()
            )


class RemoteAnalysisHandler(_RemoteTrajectoryHandler):
    """Compute structural descriptors on a sharded trajectory remotely.

    Runs Steinhardt bond-order parameters (via QUIP) and/or the radial
    distribution function on each shard. Needs no force-field model, so unlike
    :class:`RemoteOptimizationHandler` it stages no shared resources.

    Args:
        traj_files: Trajectory ``.extxyz`` file(s) to analyse.
        cluster_config: Target cluster configuration.
        remote_fs: Filesystem handler for the machine where jobs run.
        n_processors: Processors requested per array task.
        shard_size: Number of configurations per shard.
        savepath: Remote working directory for the array job.
        num_parallel: Maximum array tasks to run concurrently.
        do_steinhardt: Compute Steinhardt order parameters.
        do_rdf: Compute the radial distribution function.
        steinhardt_l_value: Spherical-harmonic order ``l`` for Steinhardt.
        cutoff: Neighbour cutoff (Angstrom) for both descriptors.
        nbins: Number of RDF bins.
        python_script_path: Remote payload; defaults to the shipped
            ``jaxnest/analyze_traj.py``.
        stub: If True, the remote payload runs in mock mode.
        (plus the shared sharding/scheduling args of
        :class:`_RemoteTrajectoryHandler`.)
    """

    _input_json_name = "input_analysis.json"

    def __init__(
        self,
        traj_files: str,
        cluster_config: ClusterConfig,
        remote_fs: file_handler.FileSystemHandlerBase,
        n_processors: int,
        shard_size: int,
        savepath: str,
        num_parallel: int,
        do_steinhardt: bool = False,
        do_rdf: bool = False,
        steinhardt_l_value: int = 4,
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
        if not (do_steinhardt or do_rdf):
            raise ValueError(
                "RemoteAnalysisHandler needs at least one of do_steinhardt / "
                "do_rdf enabled (nothing to compute otherwise)."
            )
        self._json = {
            "STUB": stub,
            "tag": tag,
            "do_steinhardt": do_steinhardt,
            "do_rdf": do_rdf,
            "steinhardt_args": dict(steinhardt_l_value=steinhardt_l_value),
            "general_args": dict(nbins=nbins, cutoff=cutoff),
        }
        super().__init__(
            traj_files,
            cluster_config,
            remote_fs,
            n_processors,
            shard_size,
            savepath,
            num_parallel,
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

    def _default_script(self) -> os.PathLike:
        return _SCRIPTS_DIR / "jaxnest" / "analyze_traj.py"

    def _base_json(self) -> dict:
        return self._json
