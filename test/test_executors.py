"""Unit tests for the pluggable execution-backend seam.

These pin the *selection* logic introduced by the executor decoupling: which
backend a cluster config resolves to, and that the SLURM handlers register as
the single/array executor families. Pure -- no instantiation, no ``sbatch`` --
so they run in the default tier. End-to-end SLURM submission stays covered by
the ``slurm``-gated task tests.
"""

import pytest

from remote_handler.core import executors
from remote_handler.core.cluster_handler import (
    ClusterConfig,
    SlurmArrayHandler,
    SlurmJobHandler,
    _combine_headers,
)
from remote_handler.core.executors import ArrayExecutor, SingleExecutor
from remote_handler.exceptions import ConfigurationError


def _config(**overrides):
    base = dict(
        remote_machine="m",
        partition="p",
        conda_env="e",
        conda_source="s",
        savepath="/abs/path",
        template="t.sh.j2",
    )
    base.update(overrides)
    return ClusterConfig(**base)


def test_slurm_handlers_register_as_executor_families():
    """The SLURM handlers implement the single/array executor contracts."""
    assert issubclass(SlurmJobHandler, SingleExecutor)
    assert issubclass(SlurmArrayHandler, ArrayExecutor)
    # Fully concrete -- nothing left abstract.
    assert SlurmJobHandler.__abstractmethods__ == frozenset()
    assert SlurmArrayHandler.__abstractmethods__ == frozenset()


def test_node_retry_capability_is_slurm_single_only():
    """Only single SLURM jobs advertise node-exclusion retry."""
    assert SlurmJobHandler.supports_node_retry is True
    assert SlurmArrayHandler.supports_node_retry is False


def test_backend_defaults_to_slurm():
    """A config with no ``executor`` key resolves to the SLURM backend."""
    cfg = _config()
    assert cfg.executor == "slurm"
    assert executors._select_backend(cfg) == executors.SLURM


def test_unknown_backend_is_rejected():
    """An unrecognised backend fails fast with a clear error."""
    cfg = _config(executor="bogus")
    with pytest.raises(ConfigurationError, match="Unknown executor backend"):
        executors._select_backend(cfg)


def test_cluster_slurm_header_add_defaults_empty():
    """Absent in the JSON, the cluster-wide header is the empty string."""
    assert _config().slurm_header_add == ""


def test_cluster_slurm_header_add_round_trips_from_dict():
    cfg = ClusterConfig.from_dict(
        {
            "remote_machine": "m",
            "partition": "p",
            "conda_env": "e",
            "conda_source": "/x",
            "savepath": "/s",
            "template": "t.j2",
            "slurm_header_add": "export TMPDIR=$HOME/tmp",
        }
    )
    assert cfg.slurm_header_add == "export TMPDIR=$HOME/tmp"


def test_combine_headers_prepends_cluster_then_per_call():
    """Cluster-wide lines come first; the per-call header follows."""
    assert (
        _combine_headers("export TMPDIR=$HOME/tmp", "module load vasp")
        == "export TMPDIR=$HOME/tmp\nmodule load vasp"
    )


def test_combine_headers_drops_empty_parts():
    """A missing cluster (or per-call) header leaves the other untouched."""
    assert _combine_headers("", "module load vasp") == "module load vasp"
    assert _combine_headers("export TMPDIR=$HOME/tmp", "") == (
        "export TMPDIR=$HOME/tmp"
    )
    assert _combine_headers("  ", "\n") == ""
