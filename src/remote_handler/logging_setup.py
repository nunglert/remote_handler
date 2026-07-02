"""Opt-in logging setup for applications using ``remote_handler``.

The library itself never configures logging (it only ever calls
``logging.getLogger(__name__)``), so timestamps/levels are entirely the
application's choice. This module is a one-call convenience for that choice --
call :func:`configure_logging` once from your script/entry point. Importing
``remote_handler`` does **not** configure anything.
"""

import logging
import sys

#: Default record format -- timestamp, level, logger name, message.
DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"
#: Default timestamp format (no microseconds; second resolution is plenty).
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level=logging.INFO,
    *,
    stream=None,
    fmt: str = DEFAULT_FORMAT,
    datefmt: str = DEFAULT_DATEFMT,
    force: bool = True,
) -> None:
    """Install a timestamped logging handler on the root logger.

    Convenience wrapper around :func:`logging.basicConfig` with a sensible
    timestamped format, so every ``logger.info(...)`` across the workflow is
    stamped. Call once, from the application -- not on import.

    Args:
        level: Root log level (e.g. ``logging.INFO``, ``logging.DEBUG``).
        stream: Destination stream; defaults to ``sys.stdout``.
        fmt: ``logging`` format string (see :data:`DEFAULT_FORMAT`).
        datefmt: ``strftime`` format for ``%(asctime)s``.
        force: Replace any handlers already on the root logger (default True),
            so this wins even if something configured logging earlier.
    """
    logging.basicConfig(
        level=level,
        stream=stream if stream is not None else sys.stdout,
        format=fmt,
        datefmt=datefmt,
        force=force,
    )
