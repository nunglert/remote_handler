"""The cluster-side processor registry.

Maps a step name to its ``run`` callable and (optionally) its arg schema. The
driver looks steps up here and runs them in order over a shard; ``validate_args``
enforces each step's arg contract at job start. This module holds only the
machinery -- the built-ins register themselves when ``processing.processors`` is
imported, and downstream packages call :func:`register` at their import time.
"""

#: Step name -> callable(atoms_list, args, calcpath) -> atoms_list.
REGISTRY = {}

#: Step name -> ``ScriptInput`` subclass declaring that step's arg contract
#: (optional). This is how the composed-payload contract is enforced remote-side
#: without a hand-written validator: the driver round-trips the step's args
#: through ``schema.from_dict(args).validate()`` before any compute.
_SCHEMAS = {}


def register(name, run, schema=None):
    """Add (or override) a processor in the registry.

    The extension point for downstream packages: a package that depends on both
    remote_handler and some heavier upstream can register its own custom
    processor here. Call this at module import time and define the matching
    host-side ``Processor`` subclass in the *same module* -- its
    ``registration_module`` then defaults to that module, and the stock driver
    imports it before resolving steps, so nothing else is needed. Registration
    must be idempotent -- re-importing the module simply re-registers the same
    ``run``.

    Args:
        name: Registry key; must match the host-side ``Processor.name``.
        run: ``callable(atoms_list, args, calcpath) -> atoms_list``.
        schema: Optional ``ScriptInput`` subclass declaring this step's arg
            contract. The driver validates the step's args against it at job
            start (via ``from_dict(...).validate()``), so a bad composed payload
            fails fast instead of deep inside a task. This is the same schema
            the host-side ``Processor.build_args`` fills, so the emit and check
            sides can't drift. Omit for a permissive (unchecked) step.
    """
    REGISTRY[name] = run
    if schema is not None:
        _SCHEMAS[name] = schema


def validate_args(name, args):
    """Validate one step's args against its processor's schema, if any.

    Round-trips ``args`` through the registered schema
    (``schema.from_dict(args).validate()``); raises ``ScriptInputError`` on a
    malformed payload. No-op when the processor registered no schema.
    """
    schema = _SCHEMAS.get(name)
    if schema is not None:
        schema.from_dict(args).validate()
