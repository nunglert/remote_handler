"""Built-in processors, one module per step (each = Args + run + Processor).

Importing this package imports each built-in module, whose module-level
``register(...)`` call wires its ``run`` + arg schema into the registry. Adding a
built-in analysis is: drop in a module here and import it below. Downstream
analyses that pull heavier upstreams (e.g. Steinhardt, which needs jaxrens) live
in their own package and register themselves the same way -- see
:func:`remote_handler.processing.registry.register`.
"""

from . import optimize, rdf  # noqa: F401  (import runs their register() calls)
from .optimize import OptimizeArgs, OptimizeProcessor
from .rdf import RdfArgs, RDFProcessor

__all__ = [
    "optimize",
    "rdf",
    "OptimizeArgs",
    "OptimizeProcessor",
    "RdfArgs",
    "RDFProcessor",
]
