"""Per-cluster ``environment`` setup feeds the DFT SLURM header.

Cluster/site-specific setup (VASP module load, ``VASP_PP_PATH``, ...) lives in
``ClusterConfig.environment[<app>]`` -- not hardcoded in the evaluator. The DFT
handler injects the app's entry ahead of any per-call ``slurm_header_add``.
"""

import logging

import pytest

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.workflows.dft_runner import _resolve_environment_header
from remote_handler.workflows.evaluators import (
    AseCalculatorEvaluator,
    EMTEvaluator,
    ORCAEvaluator,
    VASPEvaluator,
)


def _cfg(environment=None):
    return ClusterConfig(
        remote_machine="lugus",
        partition="p",
        conda_env="e",
        conda_source="/x",
        savepath="/s",
        template="t.j2",
        environment=environment or {},
    )


def test_environment_defaults_empty():
    assert _cfg().environment == {}


def test_environment_round_trips_from_dict():
    cfg = ClusterConfig.from_dict(
        {
            "remote_machine": "m",
            "partition": "p",
            "conda_env": "e",
            "conda_source": "/x",
            "savepath": "/s",
            "template": "t.j2",
            "environment": {"vasp": "module load vasp/6.4.2"},
        }
    )
    assert cfg.environment["vasp"] == "module load vasp/6.4.2"


def test_no_key_returns_header_unchanged():
    cfg = _cfg({"vasp": "module load vasp"})
    # environment_key=None -> inject nothing (historical behaviour)
    assert _resolve_environment_header(cfg, None, "extra") == "extra"


def test_both_supplied_header_wins_and_warns():
    # Both sources set -> they are NOT combined (no duplication). The legacy
    # per-call header wins and a DeprecationWarning nudges migration.
    cfg = _cfg({"vasp": "module load vasp\nexport VASP_PP_PATH=/p"})
    with pytest.warns(DeprecationWarning, match="deprecated"):
        out = _resolve_environment_header(
            cfg, "vasp", "module load vasp/custom"
        )
    assert out == "module load vasp/custom"  # env entry ignored entirely


def test_header_only_no_env_does_not_warn(recwarn):
    # Header given, no matching env entry -> header used, no deprecation.
    cfg = _cfg({"orca": "module load orca"})  # no "vasp" entry
    out = _resolve_environment_header(cfg, "vasp", "#SBATCH --time=01:00:00")
    assert out == "#SBATCH --time=01:00:00"
    assert not [w for w in recwarn if w.category is DeprecationWarning]


def test_env_only_does_not_warn(recwarn):
    cfg = _cfg({"vasp": "module load vasp"})
    out = _resolve_environment_header(cfg, "vasp", "")
    assert out == "module load vasp"
    assert not [w for w in recwarn if w.category is DeprecationWarning]


def test_missing_env_and_no_extra_warns(caplog):
    cfg = _cfg({})  # nothing configured for vasp
    with caplog.at_level(logging.WARNING):
        out = _resolve_environment_header(cfg, "vasp", "")
    assert out == ""
    assert any(
        "vasp" in r.message and "environment" in r.message
        for r in caplog.records
    )


def test_preset_environment_keys():
    # Each DFT preset declares which environment entry it needs; pure-ASE and
    # the generic base inject nothing.
    assert VASPEvaluator._environment_key == "vasp"
    assert ORCAEvaluator._environment_key == "orca"
    assert EMTEvaluator._environment_key is None
    assert AseCalculatorEvaluator._environment_key is None
