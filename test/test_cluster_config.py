"""Unit tests for ClusterConfig parse-time validation."""

import json

import pytest

from remote_handler.config import get_config
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


def _register_cluster(name: str, **overrides) -> None:
    """Write ``<name>.json`` into the (isolated) config dir for from_name."""
    config_dir = get_config().config_dir
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / f"{name}.json").write_text(
        json.dumps({**_BASE, **overrides})
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


def test_from_name_loads_registered_cluster():
    _register_cluster("lugus", remote_machine="lugus-host")
    cc = ClusterConfig.from_name("lugus")
    assert cc.remote_machine == "lugus-host"


def test_from_name_accepts_trailing_json():
    _register_cluster("lugus")
    assert ClusterConfig.from_name("lugus.json").partition == "normal"


def test_from_name_unknown_lists_available():
    _register_cluster("lugus")
    _register_cluster("datalab")
    with pytest.raises(ConfigurationError) as exc:
        ClusterConfig.from_name("missing")
    msg = str(exc.value)
    assert "missing" in msg
    assert "lugus" in msg and "datalab" in msg


def test_from_name_none_configured_hints_cli():
    with pytest.raises(ConfigurationError, match="configure cluster nope"):
        ClusterConfig.from_name("nope")


def test_available_names_excludes_global_config():
    _register_cluster("lugus")
    _register_cluster("datalab")
    get_config().config_dir.joinpath("config.json").write_text("{}")
    assert ClusterConfig.available_names() == ["datalab", "lugus"]
