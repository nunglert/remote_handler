"""The optimization and analysis handlers stay disentangled.

``RemoteOptimizationHandler`` and ``RemoteAnalysisHandler`` share the per-shard
staging/collection plumbing (``_RemoteTrajectoryHandler``) but must each build
*only* their own input JSON -- no optimization keys leak into an analysis job
and vice versa. These tests construct each handler against the local
filesystem (no cluster needed) and inspect the staged per-shard payloads,
shared resources, and default remote script.
"""

import ase.io
import numpy as np
import pytest
from ase import Atoms

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import LocalFileSystem
from remote_handler.workflows.optimizer import (
    RemoteAnalysisHandler,
    RemoteOptimizationHandler,
)


def _cfg(tmp_path):
    # ``direct`` executor: scheduler-free, so constructing the handler needs no
    # sbatch (we only inspect the staged payloads, never submit).
    return ClusterConfig(
        remote_machine="dirona",
        partition="normal",
        conda_env="base",
        conda_source="~/miniconda3/bin/activate",
        savepath=str(tmp_path / "save"),
        template="dirona.sh.j2",
        executor="direct",
    )


@pytest.fixture
def traj(tmp_path):
    """A small 6-config trajectory on local disk."""
    rng = np.random.default_rng(0)
    ats = []
    for _ in range(6):
        at = Atoms("Si2", cell=np.eye(3) * 6, pbc=True)
        at.set_positions(rng.random((2, 3)) * 6)
        ats.append(at)
    f = tmp_path / "traj.extxyz"
    ase.io.write(f, ats)
    return [str(f)]


def _common(tmp_path, traj):
    return dict(
        traj_files=traj,
        cluster_config=_cfg(tmp_path),
        remote_fs=LocalFileSystem(),
        n_processors=1,
        shard_size=3,  # 6 configs -> 2 shards
        savepath=str(tmp_path / "save"),
        num_parallel=2,
        every=1,
    )


def _shard_jsons(handler):
    """Per-shard input JSON dicts the handler staged."""
    config_files = handler.individual_ressources[0]
    return [c.content for c in config_files]


def _input_name(handler):
    return handler.individual_ressources[0][0].filename


# -- optimization -----------------------------------------------------------
def test_optimization_json_is_optimization_only(tmp_path, traj):
    h = RemoteOptimizationHandler(
        optimization_backend="LJ",
        fmax=0.05,
        steps=123,
        **_common(tmp_path, traj),
    )
    jsons = _shard_jsons(h)
    assert len(jsons) == 2
    for j in jsons:
        assert j["do_optimization"] is True
        assert "calculator_args" in j and "optimization_args" in j
        assert j["optimization_args"] == {"fmax": 0.05, "steps": 123}
        # No analysis concerns leak in.
        assert "do_steinhardt" not in j
        assert "do_rdf" not in j
        assert "steinhardt_args" not in j
        assert "general_args" not in j

    assert _input_name(h) == "input_opt.json"
    assert str(h.python_script_file).endswith("jaxnest/optimize_traj.py")
    # LJ needs no model, so nothing is staged as a shared resource.
    assert h.shared_ressources == []


def test_optimization_nnff_stages_pickle(tmp_path, traj):
    pickle = tmp_path / "model.pkl"
    pickle.write_bytes(b"")
    h = RemoteOptimizationHandler(
        optimization_backend="NNFF",
        pickle_file=str(pickle),
        **_common(tmp_path, traj),
    )
    # Model staged once, shared across shards...
    assert h.shared_ressources == [str(pickle)]
    # ...and each shard points its calculator at the resolved remote path.
    for j in _shard_jsons(h):
        assert j["calculator_args"]["pickle_file"].endswith("model.pkl")


def test_optimization_nnff_requires_pickle(tmp_path, traj):
    with pytest.raises(ValueError, match="requires a pickle_file"):
        RemoteOptimizationHandler(
            optimization_backend="NNFF", **_common(tmp_path, traj)
        )


# -- analysis ---------------------------------------------------------------
def test_analysis_json_is_analysis_only(tmp_path, traj):
    h = RemoteAnalysisHandler(
        do_steinhardt=True,
        do_rdf=True,
        steinhardt_l_value=6,
        cutoff=4.5,
        nbins=150,
        **_common(tmp_path, traj),
    )
    jsons = _shard_jsons(h)
    assert len(jsons) == 2
    for j in jsons:
        assert j["do_steinhardt"] is True and j["do_rdf"] is True
        assert j["steinhardt_args"] == {"steinhardt_l_value": 6}
        assert j["general_args"] == {"nbins": 150, "cutoff": 4.5}
        # No optimization concerns leak in.
        assert "do_optimization" not in j
        assert "calculator_args" not in j
        assert "optimization_args" not in j

    assert _input_name(h) == "input_analysis.json"
    assert str(h.python_script_file).endswith("jaxnest/analyze_traj.py")
    # Analysis needs no model -> no shared resources.
    assert h.shared_ressources == []


def test_analysis_requires_something_to_do(tmp_path, traj):
    with pytest.raises(ValueError, match="do_steinhardt / do_rdf"):
        RemoteAnalysisHandler(**_common(tmp_path, traj))
