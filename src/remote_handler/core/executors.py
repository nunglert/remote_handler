"""Pluggable execution backends for :class:`~remote_handler.core.tasks.RemoteTask`.

A task describes *what* to transfer and collect; an ``Executor`` describes *how*
the per-job scripts actually run on the target -- via a scheduler (SLURM) or,
in a later phase, directly over SSH with no scheduler at all. Tasks build one
through the factories here instead of hardwiring a concrete handler, so the
same task code can run scheduler-free where appropriate.

Single and array jobs are distinct families (their ``write_jobfile``
signatures differ), each with its own backend implementation:

    SingleExecutor -> SlurmJobHandler
    ArrayExecutor  -> SlurmArrayHandler

The concrete SLURM handlers live in :mod:`remote_handler.core.cluster_handler`
and register against the ABCs defined here; this module owns only the
abstractions and the backend-selection factories.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from remote_handler.exceptions import ConfigurationError

if TYPE_CHECKING:  # avoid a runtime import cycle (cluster_handler imports us)
    from remote_handler.core.cluster_handler import ClusterConfig

# Backend names understood by the factory. SLURM is the default/back-compat
# backend; further backends (e.g. a scheduler-free "direct" one) are added by
# extending this tuple and the dispatch in the factories below.
SLURM = "slurm"
_KNOWN_BACKENDS = (SLURM,)


class Executor(ABC):
    """The contract a :class:`RemoteTask` depends on to run and await jobs.

    Concrete executors additionally expose a ``calcdir`` attribute (where the
    computation is performed) set during construction.
    """

    #: True if the backend can re-place a job to dodge a bad node (SLURM).
    #: Tasks consult this to decide whether to run their node-exclusion retry
    #: loop; backends without a notion of nodes leave it False.
    supports_node_retry = False

    @abstractmethod
    def submit(self):
        """Launch the job(s)."""

    @abstractmethod
    def wait_until_finished(self):
        """Block until the job(s) complete."""


class SingleExecutor(Executor):
    """Runs one script against one input file."""

    @abstractmethod
    def write_jobfile(
        self, input_file, python_script_path, exclude_nodes=None
    ):
        """Prepare the launch artifact for a single job."""


class ArrayExecutor(Executor):
    """Runs a script across ``num_array_tasks`` shards (an array job)."""

    @abstractmethod
    def write_jobfile(self, python_script_path, num_array_tasks):
        """Prepare the launch artifact for an array job."""


def _select_backend(cluster_config: "ClusterConfig") -> str:
    """Resolve and validate the execution backend for a cluster config.

    Pure: no instantiation and no remote I/O, so the selection logic is
    testable without a live scheduler. Falls back to SLURM when unset, which
    keeps existing cluster JSONs (with no ``executor`` key) working unchanged.
    """
    backend = getattr(cluster_config, "executor", SLURM) or SLURM
    if backend not in _KNOWN_BACKENDS:
        raise ConfigurationError(
            f"Unknown executor backend '{backend}'. "
            f"Known backends: {', '.join(_KNOWN_BACKENDS)}."
        )
    return backend


def make_single_executor(
    cluster_config: "ClusterConfig", **kwargs
) -> SingleExecutor:
    """Build the :class:`SingleExecutor` for this cluster's backend."""
    backend = _select_backend(cluster_config)
    # Local import: cluster_handler imports the ABCs above at module load, so
    # importing it here (not at module level) avoids a circular import.
    from remote_handler.core import cluster_handler

    if backend == SLURM:
        return cluster_handler.SlurmJobHandler(
            cluster_config=cluster_config, **kwargs
        )
    # Unreachable: _select_backend already rejected unknown backends.
    raise ConfigurationError(  # pragma: no cover
        f"No SingleExecutor registered for backend '{backend}'."
    )


def make_array_executor(
    cluster_config: "ClusterConfig", **kwargs
) -> ArrayExecutor:
    """Build the :class:`ArrayExecutor` for this cluster's backend."""
    backend = _select_backend(cluster_config)
    from remote_handler.core import cluster_handler

    if backend == SLURM:
        return cluster_handler.SlurmArrayHandler(
            cluster_config=cluster_config, **kwargs
        )
    raise ConfigurationError(  # pragma: no cover
        f"No ArrayExecutor registered for backend '{backend}'."
    )
