"""Unit tests for ClusterConfig parse-time validation."""

import pytest

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.exceptions import ConfigurationError

_BASE = dict(
    remote_machine="alias",
    partition="normal",
    conda_env="env",
    conda_source="/home/u/miniconda3/bin/activate",
    savepath="/home/u/dump",
    template="t.sh.j2",
)


def test_absolute_savepath_ok():
    cc = ClusterConfig.from_dict(_BASE)
    assert cc.savepath == "/home/u/dump"


@pytest.mark.parametrize("bad", ["~/dump", "~user/dump", "~"])
def test_tilde_savepath_rejected_from_dict(bad):
    with pytest.raises(ConfigurationError, match="~"):
        ClusterConfig.from_dict({**_BASE, "savepath": bad})


def test_tilde_savepath_rejected_on_direct_construction():
    with pytest.raises(ConfigurationError):
        ClusterConfig(
            remote_machine="a",
            partition="p",
            conda_env="e",
            conda_source="/x",
            savepath="~/dump",
            template="t.j2",
        )
