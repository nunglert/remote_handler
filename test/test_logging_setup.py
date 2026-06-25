"""``configure_logging`` is an opt-in timestamped logging setup."""

import io
import logging
import re

import remote_handler
from remote_handler import configure_logging
from remote_handler.logging_setup import DEFAULT_FORMAT

_TS = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")


def _with_clean_root(fn):
    """Run fn with the root logger state saved/restored (config is global)."""
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        return fn()
    finally:
        root.handlers[:] = saved_handlers
        root.level = saved_level


def test_exported_from_package():
    assert remote_handler.configure_logging is configure_logging
    assert "asctime" in DEFAULT_FORMAT


def test_configure_logging_stamps_records():
    def body():
        buf = io.StringIO()
        configure_logging(stream=buf, level=logging.INFO)
        logging.getLogger("remote_handler.test").info("hello world")
        out = buf.getvalue()
        assert "hello world" in out
        assert _TS.search(out), out  # a timestamp is present

    _with_clean_root(body)


def test_force_replaces_prior_config():
    def body():
        # A prior basicConfig (e.g. an example's) must not block our format.
        logging.basicConfig(stream=io.StringIO(), level=logging.WARNING)
        buf = io.StringIO()
        configure_logging(stream=buf, level=logging.INFO)
        logging.getLogger("remote_handler.test").info("after")
        assert _TS.search(buf.getvalue())

    _with_clean_root(body)
