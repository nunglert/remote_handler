"""The ``Processor`` base -- host-side spec for one pipeline step.

A :class:`Processor` is the *host* half of a processing step: it declares only
*what* to run, the remote arguments it needs, and any files to stage -- nothing
about sharding or transport (that is :class:`~remote_handler.processing.handler.
RemoteTrajectoryProcessor`). Its cluster-side counterpart -- the ``run`` function
and the arg schema -- lives in the same processor module under ``processors/``
(built-ins) or in a downstream package (see :func:`~remote_handler.processing.
registry.register`).
"""

from abc import ABC, abstractmethod
from typing import Optional


class Processor(ABC):
    """One processing step in the remote pipeline.

    A ``Processor`` contributes *only* its own remote arguments and any files
    that must be staged for it -- nothing about sharding or transport. Its
    :attr:`name` keys both the ordered ``steps`` list and the per-step ``args``
    map in the shard input JSON, and must match a processor in the registry.

    Subclassing is low-ceremony: set a class-level :attr:`name` and implement
    :meth:`build_args`. :attr:`registration_module` defaults to the subclass's
    own module (so a downstream processor defined alongside its ``run`` +
    ``register`` call resolves on the cluster with nothing else to declare);
    built-ins, whose ``run`` is always loaded, set it to ``None``. A missing
    :attr:`name` is caught at class-definition time.
    """

    #: Registry key; must match a remote processor's registered name.
    name: str

    #: Import path of the module whose import registers this step's ``run`` in
    #: the cluster-side ``REGISTRY`` (via ``register(name, run)``). ``None`` for
    #: built-in processors, which are already in the always-loaded registry.
    #: Defaults (via ``__init_subclass__``) to the subclass's own module.
    registration_module: Optional[str] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Skip still-abstract intermediates; only concrete steps are checked.
        if getattr(cls, "__abstractmethods__", frozenset()):
            return
        if not getattr(cls, "name", None):
            raise TypeError(
                f"{cls.__name__} must set a class-level 'name' "
                "(its registry key)."
            )
        # Co-located downstream processors get their registration module for
        # free; a subclass that sets it explicitly (incl. None) is left alone.
        if "registration_module" not in cls.__dict__:
            cls.registration_module = cls.__module__

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
