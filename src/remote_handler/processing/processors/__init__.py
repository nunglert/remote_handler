"""Built-in processors, one module per step (each = Args + run + Processor).

Importing this package imports each built-in module, whose module-level
``register(...)`` call wires its ``run`` + arg schema into the registry. Adding a
built-in analysis is: drop in a module here and import it below. Downstream
analyses that pull heavier upstreams live in their own package and register
themselves the same way -- see
:func:`remote_handler.processing.registry.register`.
"""

from . import optimize, rdf, steinhardt  # noqa: F401  (runs register() calls)
from .optimize import OptimizeArgs, OptimizeProcessor
from .rdf import RdfArgs, RDFProcessor
from .steinhardt import SteinhardtArgs, SteinhardtProcessor

__all__ = [
    "optimize",
    "rdf",
    "steinhardt",
    "OptimizeArgs",
    "OptimizeProcessor",
    "RdfArgs",
    "RDFProcessor",
    "SteinhardtArgs",
    "SteinhardtProcessor",
]
