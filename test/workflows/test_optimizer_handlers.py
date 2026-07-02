"""The composable trajectory processor + its optimize factory.

``RemoteTrajectoryProcessor`` runs an ordered list of ``Processor`` steps over a
shared sharding/collection base; ``RemoteOptimizationHandler`` is a thin factory
for the optimize step. Analysis has no dedicated handler -- it is just the core
with the processors you want (e.g. ``[RDFProcessor(), ...]``). These tests
construct handlers against the local filesystem (no cluster needed) and inspect
the staged per-shard payloads, shared resources, and default remote script,
checking that each processor contributes *only* its own args.
"""

import ase.io
import numpy as np
import pytest
from ase import Atoms

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import LocalFileSystem
from remote_handler.processing import (
    OptimizeProcessor,
    Processor,
    RDFProcessor,
    RemoteOptimizationHandler,
    RemoteTrajectoryProcessor,
)


class _DummyDownstreamProcessor(Processor):
    """Stand-in for a downstream (e.g. jaxrens Steinhardt) processor: declares
    a registration_module and stages nothing."""

    name = "dummy"
    registration_module = "some.test.module"

    def build_args(self, savepath: str) -> dict:
        return {"k": 1}


def test_processor_subclass_requires_name():
    """A concrete Processor without a name is rejected at definition time."""
    with pytest.raises(TypeError, match="must set a class-level 'name'"):

        class _NoName(Processor):
            def build_args(self, savepath):
                return {}


def test_processor_registration_module_defaults_to_own_module():
    """A downstream-style processor that co-locates its run+register gets its
    registration_module for free; an explicit value is left untouched."""

    class _Derived(Processor):
        name = "derived"

        def build_args(self, savepath):
            return {}

    assert _Derived.registration_module == _Derived.__module__
    # Explicitly-set value wins (the dummy above sets its own).
    assert _DummyDownstreamProcessor.registration_module == "some.test.module"
    # Built-ins opt out (their run is always in the registry).
    assert OptimizeProcessor.registration_module is None
    assert RDFProcessor.registration_module is None


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
    assert str(h.python_script_file).endswith("processing/driver.py")


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


# -- analysis is just the core with an RDF processor ------------------------
def test_rdf_only_pipeline_builds_rdf_step(tmp_path, traj):
    h = RemoteTrajectoryProcessor(
        processors=[RDFProcessor(cutoff=4.5, nbins=150)],
        **_common(tmp_path, traj),
    )
    jsons = _shard_jsons(h)
    assert len(jsons) == 2
    for j in jsons:
        assert j["steps"] == ["rdf"]
        assert j["args"]["rdf"] == {"cutoff": 4.5, "nbins": 150}
        # No optimization concern leaks in.
        assert "optimize" not in j["args"]
    # RDF needs no model -> no shared resources.
    assert h.shared_ressources == []


def test_analysis_pipeline_composes_rdf_then_downstream(tmp_path, traj):
    """RDF + a downstream descriptor compose in order, and the downstream
    processor's registration module rides along for the cluster driver."""
    h = RemoteTrajectoryProcessor(
        processors=[RDFProcessor(), _DummyDownstreamProcessor()],
        **_common(tmp_path, traj),
    )
    for j in _shard_jsons(h):
        assert j["steps"] == ["rdf", "dummy"]
        assert j["registration_modules"] == ["some.test.module"]


# -- registration modules travel in the shard JSON --------------------------
def test_builtin_only_pipeline_lists_no_registration_modules(tmp_path, traj):
    h = RemoteTrajectoryProcessor(
        processors=[OptimizeProcessor(backend="LJ"), RDFProcessor()],
        **_common(tmp_path, traj),
    )
    for j in _shard_jsons(h):
        assert j["registration_modules"] == []


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
    assert str(h.python_script_file).endswith("processing/driver.py")
    assert _input_name(h) == "input_process.json"


def test_duplicate_processor_names_rejected(tmp_path, traj):
    with pytest.raises(ValueError, match="duplicate processor names"):
        RemoteTrajectoryProcessor(
            processors=[RDFProcessor(), RDFProcessor()],
            **_common(tmp_path, traj),
        )
