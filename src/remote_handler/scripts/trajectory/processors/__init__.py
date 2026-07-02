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
REGISTRY = {}

#: Step name -> callable(args) that raises on a bad arg dict (optional). This is
#: how the composed-payload contract is enforced remote-side: each processor
#: owns the check for its own args, and the driver runs them before any compute.
_VALIDATORS = {}


def register(name, run, validate=None):
    """Add (or override) a processor in the registry.

    The extension point for downstream packages: a package that depends on both
    remote_handler and, say, jaxrens can register its own ``steinhardt``
    processor here. Call this at module import time, then set
    ``registration_module`` on the matching host-side ``Processor`` to that
    module's import path -- the stock ``process_traj.py`` driver imports every
    declared module before resolving steps, so nothing else is needed. (Pointing
    the handler's ``python_script_path`` at a thin driver still works, but is no
    longer required.) Registration must be idempotent -- re-importing the module
    simply re-registers the same ``run``.

    Args:
        name: Registry key; must match the host-side ``Processor.name``.
        run: ``callable(atoms_list, args, calcpath) -> atoms_list``.
        validate: Optional ``callable(args)`` that raises (e.g.
            ``ScriptInputError``) if this step's args are malformed. The driver
            runs it at job start, so a bad composed payload fails fast instead
            of deep inside a task. Omit for a permissive (unchecked) step.
    """
    REGISTRY[name] = run
    if validate is not None:
        _VALIDATORS[name] = validate


def validate_args(name, args):
    """Validate one step's args against its processor's contract, if any.

    No-op when the processor registered no validator (permissive). Raises
    whatever the processor's validator raises for a malformed ``args``.
    """
    validator = _VALIDATORS.get(name)
    if validator is not None:
        validator(args)


# Built-ins register through the same mechanism, each with its arg validator.
register("optimize", optimize.run, validate=optimize.validate_args)
register("rdf", rdf.run, validate=rdf.validate_args)
