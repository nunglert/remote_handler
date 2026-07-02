"""Registry of remote trajectory processors.

Each processor is a small module exposing ``run(atoms_list, args, calcpath)``
and doing one thing (optimize / rdf / ...). The ``process_traj.py`` driver
looks steps up here and runs them in order over a shard, so adding a new
analysis is: drop in a module + one entry below. No driver or handler change.

Built-in processors source dependency-light glue locally (the LBFGS
relaxation, the ASE RDF) and the NNFF calculator from neuralil-jaxrens. Analyses
that pull in heavier upstreams (e.g. Steinhardt, which needs jaxrens) live in
**downstream** packages and add themselves via :func:`register` -- so
remote_handler itself never depends on them. Those upstream imports are lazy
inside each ``run``, so importing this registry host-side (e.g. for tests)
needs none of them.
"""

from . import optimize, rdf

#: Step name -> callable(atoms_list, args, calcpath) -> atoms_list.
#: Names must match those a host-side ``Processor`` emits.
REGISTRY = {
    "optimize": optimize.run,
    "rdf": rdf.run,
}


def register(name, fn):
    """Add (or override) a processor in the registry.

    The extension point for downstream packages: a package that depends on both
    remote_handler and, say, jaxrens can register its own ``steinhardt``
    processor here. For the shipped ``process_traj.py`` driver to see it on the
    cluster, ensure the downstream module is imported before the driver runs --
    e.g. by pointing the handler's ``python_script_path`` at a thin driver that
    imports the downstream package first, then delegates to this registry.
    """
    REGISTRY[name] = fn
