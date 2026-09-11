"""Tests for RemoteSelector: running a ConfigSelector strategy as a job.

The non-stub tests drive a real ``update`` -> ``get_selected_configs`` cycle
against the scheduler-free ``direct`` executor backend on a
``LocalFileSystem`` (mirrors ``test_manifest.py``'s approach for
``SingleScriptTask``) -- so the whole remote path (JSON input building, file
shipping, ``scripts/selection/select_configs.py`` execution, results
collection) is exercised with nothing mocked and no ``sbatch`` needed.
"""

from pathlib import Path

import ase.io
import pytest
from ase import Atoms

from remote_handler.core.cluster_config import ClusterConfig
from remote_handler.core.file_handler import LocalFileSystem
from remote_handler.exceptions import WorkflowError
from remote_handler.workflows import selectors as selectors_mod
from remote_handler.workflows.active_learning_base import Selector
from remote_handler.workflows.staged_active_learning import RemoteSelector


def _direct_config(savepath):
    return ClusterConfig(
        remote_machine="local",
        partition="p",
        conda_env="",
        conda_source="",
        savepath=str(savepath),
        template="unused.sh.j2",
        executor="direct",
    )


def _write_pool(path: Path, n: int) -> Path:
    frames = [
        Atoms(
            "H2",
            positions=[[0, 0, 0], [0, 0, 0.7 + 0.05 * i]],
            cell=[6, 6, 6],
            pbc=True,
        )
        for i in range(n)
    ]
    ase.io.write(path, frames)
    return path


def test_remote_selector_is_selector_subclass():
    assert issubclass(RemoteSelector, Selector)


# --------------------------------------------------------------------------
# Stub mode: runs ConfigSelector.select() in-process, no cluster needed.
# --------------------------------------------------------------------------


def test_stub_mode_selects_in_process(tmp_path):
    pool = _write_pool(tmp_path / "pool.extxyz", 10)
    rs = RemoteSelector(
        selector_type="StrideSelector",
        selector_kwargs={"n_configs": 3},
        cluster_config=None,
        remote_fs=None,
        stub=True,
    )
    rs.update(tmp_path / "iter0", pool)
    out = rs.get_selected_configs()

    assert out == rs.results_dir / "selected.extxyz"
    assert len(ase.io.read(out, index=":")) == 3


def test_stub_mode_kwargs_override_selector_kwargs(tmp_path):
    """Per-iteration schedule kwargs (``update(**kwargs)``) win over the
    baseline ``selector_kwargs`` set at construction time."""
    pool = _write_pool(tmp_path / "pool.extxyz", 10)
    rs = RemoteSelector(
        selector_type="StrideSelector",
        selector_kwargs={"n_configs": 3},
        cluster_config=None,
        remote_fs=None,
        stub=True,
    )
    rs.update(tmp_path / "iter0", pool, n_configs=5)
    out = rs.get_selected_configs()

    assert len(ase.io.read(out, index=":")) == 5


class _RefRecordingSelector(selectors_mod.ConfigSelector):
    """Records the size of the ``reference`` seen by each ``select`` call.

    A stand-in for a novelty-based selector like ``QUESTSSelector``, kept
    dependency-free. ``calls`` is a class attribute (not instance) because
    ``RemoteSelector`` constructs a fresh selector instance per ``update``.
    """

    calls: list = []

    def __init__(self, n_configs: int):
        self.n_configs = n_configs

    def select(self, configs, reference=None):
        type(self).calls.append(None if reference is None else len(reference))
        flat = (
            [a for g in configs for a in g]
            if configs and isinstance(configs[0], list)
            else list(configs)
        )
        return flat[: self.n_configs]


def test_stub_mode_forwards_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(
        selectors_mod, "_RefRecordingSelector", _RefRecordingSelector, raising=False
    )
    _RefRecordingSelector.calls = []

    pool = _write_pool(tmp_path / "pool.extxyz", 6)
    ref = _write_pool(tmp_path / "ref.extxyz", 4)
    rs = RemoteSelector(
        selector_type="_RefRecordingSelector",
        selector_kwargs={"n_configs": 2},
        cluster_config=None,
        remote_fs=None,
        stub=True,
    )
    rs.update(tmp_path / "iter0", pool, reference=[ref])
    out = rs.get_selected_configs()

    assert _RefRecordingSelector.calls == [4]
    assert len(ase.io.read(out, index=":")) == 2


def test_stub_mode_cold_start_reference_is_none(tmp_path, monkeypatch):
    monkeypatch.setattr(
        selectors_mod, "_RefRecordingSelector", _RefRecordingSelector, raising=False
    )
    _RefRecordingSelector.calls = []

    pool = _write_pool(tmp_path / "pool.extxyz", 6)
    rs = RemoteSelector(
        selector_type="_RefRecordingSelector",
        selector_kwargs={"n_configs": 2},
        cluster_config=None,
        remote_fs=None,
        stub=True,
    )
    rs.update(tmp_path / "iter0", pool)  # no reference passed at all
    rs.get_selected_configs()

    assert _RefRecordingSelector.calls == [None]


def test_get_selected_configs_before_update_raises_in_stub_mode():
    rs = RemoteSelector(
        selector_type="StrideSelector",
        selector_kwargs={"n_configs": 1},
        cluster_config=None,
        remote_fs=None,
        stub=True,
    )
    with pytest.raises(WorkflowError, match="not produced output"):
        rs.get_selected_configs()


# --------------------------------------------------------------------------
# Non-stub: real submit -> collect cycle against the "direct" backend.
# --------------------------------------------------------------------------


def test_remote_selection_runs_as_direct_job(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    pool = _write_pool(tmp_path / "pool.extxyz", 10)

    rs = RemoteSelector(
        selector_type="StrideSelector",
        selector_kwargs={"n_configs": 4},
        cluster_config=_direct_config(remote),
        remote_fs=LocalFileSystem(),
        loop_time=0.2,
        remote_savepath=str(remote / "calc"),
    )
    rs.update(tmp_path / "iter0", pool)
    out = rs.get_selected_configs()

    assert out == rs.results_dir / "selected.extxyz"
    selected = ase.io.read(out, index=":")
    assert len(selected) == 4


def test_remote_selection_ships_reference_files(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    pool = _write_pool(tmp_path / "pool.extxyz", 6)
    ref = _write_pool(tmp_path / "ref.extxyz", 3)

    rs = RemoteSelector(
        selector_type="StrideSelector",
        selector_kwargs={"n_configs": 2},
        cluster_config=_direct_config(remote),
        remote_fs=LocalFileSystem(),
        loop_time=0.2,
        remote_savepath=str(remote / "calc"),
    )
    rs.update(tmp_path / "iter0", pool, reference=[ref])
    out = rs.get_selected_configs()

    # The reference file was actually shipped alongside the pool.
    assert (rs.select_task.calcdir / "ref.extxyz").exists()
    assert len(ase.io.read(out, index=":")) == 2


def test_remote_selection_failure_raises_workflow_error(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    pool = _write_pool(tmp_path / "pool.extxyz", 5)

    rs = RemoteSelector(
        selector_type="NoSuchSelector",
        selector_kwargs={},
        cluster_config=_direct_config(remote),
        remote_fs=LocalFileSystem(),
        loop_time=0.2,
        remote_savepath=str(remote / "calc"),
    )
    rs.update(tmp_path / "iter0", pool)

    with pytest.raises(WorkflowError, match="did not finish"):
        rs.get_selected_configs()
