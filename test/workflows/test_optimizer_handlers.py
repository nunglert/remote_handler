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
    OptimizeProcessor,
    RDFProcessor,
    RemoteAnalysisHandler,
    RemoteOptimizationHandler,
    RemoteTrajectoryProcessor,
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


# Every handler ships one driver + a per-shard steps/args JSON.
def test_all_handlers_use_the_single_driver(tmp_path, traj):
    h = RemoteOptimizationHandler(
        optimization_backend="LJ", **_common(tmp_path, traj)
    )
    assert _input_name(h) == "input_process.json"
    assert str(h.python_script_file).endswith("trajectory/process_traj.py")


# -- optimization factory ---------------------------------------------------
def test_optimization_builds_single_optimize_step(tmp_path, traj):
    h = RemoteOptimizationHandler(
        optimization_backend="LJ",
        fmax=0.05,
        steps=123,
        **_common(tmp_path, traj),
    )
    jsons = _shard_jsons(h)
    assert len(jsons) == 2
    for j in jsons:
        # A single ordered step, with only its own args.
        assert j["steps"] == ["optimize"]
        assert set(j["args"]) == {"optimize"}
        assert j["args"]["optimize"] == {
            "backend": "LJ",
            "fmax": 0.05,
            "steps": 123,
        }
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
        assert j["args"]["optimize"]["pickle_file"].endswith("model.pkl")


def test_optimization_nnff_requires_pickle(tmp_path, traj):
    with pytest.raises(ValueError, match="requires a pickle_file"):
        RemoteOptimizationHandler(
            optimization_backend="NNFF", **_common(tmp_path, traj)
        )


def test_optimization_forwards_neighbor_bucket_ladder(tmp_path, traj):
    """A neighbour-bucket ladder + offset reach the remote calculator args."""
    pickle = tmp_path / "model.pkl"
    pickle.write_bytes(b"")
    h = RemoteOptimizationHandler(
        optimization_backend="NNFF",
        pickle_file=str(pickle),
        max_neighbors_list=[20, 30, 40],
        neighbor_offset=2,
        **_common(tmp_path, traj),
    )
    for j in _shard_jsons(h):
        opt = j["args"]["optimize"]
        assert opt["max_neighbors_list"] == [20, 30, 40]
        assert opt["neighbor_offset"] == 2


# -- analysis factory -------------------------------------------------------
def test_analysis_builds_rdf_step(tmp_path, traj):
    h = RemoteAnalysisHandler(
        do_rdf=True,
        cutoff=4.5,
        nbins=150,
        **_common(tmp_path, traj),
    )
    jsons = _shard_jsons(h)
    assert len(jsons) == 2
    for j in jsons:
        assert j["steps"] == ["rdf"]
        assert j["args"]["rdf"] == {"cutoff": 4.5, "nbins": 150}
        # No optimization concern leaks in.
        assert "optimize" not in j["args"]
    # Analysis needs no model -> no shared resources.
    assert h.shared_ressources == []


def test_analysis_requires_something_to_do(tmp_path, traj):
    with pytest.raises(ValueError, match="nothing to compute"):
        RemoteAnalysisHandler(do_rdf=False, **_common(tmp_path, traj))


# -- composition: the whole point of the redesign ---------------------------
def test_pipeline_composes_optimize_then_analysis(tmp_path, traj):
    """One job, one trajectory upload: relax then characterize the results.

    The trajectory is distributed once; the ordered steps run in-sequence on
    each shard, and the optimizer's model is the only shared resource. (Uses a
    custom downstream-style step name to show the registry is open-ended.)
    """
    pickle = tmp_path / "model.pkl"
    pickle.write_bytes(b"")
    h = RemoteTrajectoryProcessor(
        processors=[
            OptimizeProcessor(pickle_file=str(pickle), backend="NNFF"),
            RDFProcessor(cutoff=5.0, nbins=100),
        ],
        **_common(tmp_path, traj),
    )
    for j in _shard_jsons(h):
        # Order preserved -> optimize before analysis.
        assert j["steps"] == ["optimize", "rdf"]
        assert set(j["args"]) == {"optimize", "rdf"}
        # New calculator design: a single max_neighbors, no compiled-set list.
        assert "max_neighbors" in j["args"]["optimize"]
        assert "max_neighbors_list" not in j["args"]["optimize"]
    # Only the model is shared; the trajectory itself is staged once as the
    # per-shard atoms batch (not a shared resource).
    assert h.shared_ressources == [str(pickle)]
    # Single driver + single per-shard input, regardless of step count.
    assert str(h.python_script_file).endswith("trajectory/process_traj.py")
    assert _input_name(h) == "input_process.json"


def test_duplicate_processor_names_rejected(tmp_path, traj):
    with pytest.raises(ValueError, match="duplicate processor names"):
        RemoteTrajectoryProcessor(
            processors=[RDFProcessor(), RDFProcessor()],
            **_common(tmp_path, traj),
        )
