"""Trajectory processing -- a self-contained, extensible plugin framework.

Unlike the rest of the package (organized by layer: ``core`` / ``workflows`` /
``scripts``), this feature is organized *by feature*, because it is a different
kind of thing: an extensible registry of processors, each a tight trio of

* a host-side spec (:class:`~remote_handler.processing.base.Processor`),
* a cluster-side ``run``, and
* a shared arg schema,

wired together by a registry and a shard driver. A built-in processor keeps all
three in one module under ``processors/`` (e.g. ``processors/optimize.py``);
downstream processors do the same in their own package and self-register.

Public surface:

* :class:`RemoteTrajectoryProcessor` -- run an ordered list of processors over a
  sharded trajectory; :class:`RemoteOptimizationHandler` -- thin factory for the
  optimize step.
* :class:`Processor` -- base for a host-side spec; :class:`OptimizeProcessor`,
  :class:`RDFProcessor`, :class:`SteinhardtProcessor` -- the built-ins.
* :func:`register` -- the downstream extension point.
"""

from remote_handler.processing.base import Processor
from remote_handler.processing.handler import (
    RemoteOptimizationHandler,
    RemoteTrajectoryProcessor,
)
from remote_handler.processing.inputs import ProcessTrajInput
from remote_handler.processing.processors import (
    OptimizeArgs,
    OptimizeProcessor,
    RdfArgs,
    RDFProcessor,
    SteinhardtArgs,
    SteinhardtProcessor,
)
from remote_handler.processing.registry import (
    REGISTRY,
    register,
    validate_args,
)

__all__ = [
    "Processor",
    "RemoteTrajectoryProcessor",
    "RemoteOptimizationHandler",
    "OptimizeProcessor",
    "RDFProcessor",
    "SteinhardtProcessor",
    "OptimizeArgs",
    "RdfArgs",
    "SteinhardtArgs",
    "ProcessTrajInput",
    "REGISTRY",
    "register",
    "validate_args",
]
