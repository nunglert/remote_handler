"""Tests for durable task handles (detach-on-submit + crash resume).

The end-to-end tests drive a real ``detach`` -> reload-from-manifest ->
``wait`` -> ``collect`` cycle against the scheduler-free ``direct`` backend on
a ``LocalFileSystem`` -- i.e. the whole persistence path with nothing mocked
and no ``sbatch`` needed. A fresh :class:`TaskHandle` is built purely from the
JSON manifest, standing in for a second process that never saw the original
task object.
"""

import json
from pathlib import Path

import pytest

from remote_handler.core import file_handler, tasks
from remote_handler.core.cluster_config import ClusterConfig
from remote_handler.core.manifest import (
    MANIFEST_NAME,
    MANIFEST_VERSION,
    TaskHandle,
    TaskManifest,
)
from remote_handler.exceptions import TaskError

SINGLE_SCRIPT = """
import json, sys
from pathlib import Path

data = json.load(open(sys.argv[1]))
Path("result.json").write_text(json.dumps({"sum": data["A"] + data["B"]}))
Path("is_finished.info").write_text("done")
"""

ARRAY_SCRIPT = """
import json, sys
from pathlib import Path

calc = Path(sys.argv[1])
data = json.load(open(calc / "input.json"))
(calc / "result.json").write_text(json.dumps({"sum": data["A"] + data["B"]}))
(calc / "is_finished.info").write_text("done")
"""


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


# --------------------------------------------------------------------------
# Unit-level: serialization, version guard, attach
# --------------------------------------------------------------------------


def _sample_manifest(**over):
    base = dict(
        task_type="single",
        backend="slurm",
        job_id="12345",
        remote_savepath="/remote/calc",
        local_savepath="/local/results",
        jobname="job",
        cluster_config=_direct_config("/x").to_dict(),
        fs_type="remote",
    )
    base.update(over)
    return TaskManifest(**base)


def test_manifest_roundtrips_through_dict():
    m = _sample_manifest()
    assert TaskManifest.from_dict(m.to_dict()) == m


def test_manifest_roundtrips_through_file(tmp_path):
    m = _sample_manifest()
    path = m.save_local(tmp_path / "sub" / "task.json")
    assert path.exists()
    assert TaskManifest.load(path) == m


def test_manifest_rejects_unknown_version():
    data = _sample_manifest().to_dict()
    data["version"] = MANIFEST_VERSION + 1
    with pytest.raises(TaskError):
        TaskManifest.from_dict(data)


def test_build_fs_local():
    m = _sample_manifest(fs_type="local")
    assert isinstance(m.build_fs(), file_handler.LocalFileSystem)


class _FakeExecutor:
    def __init__(self):
        self.pid = None
        self.is_submitted = False

    # Borrow the real, concrete attach() from the ABC.
    attach = tasks.executors.Executor.attach


def test_attach_binds_job_without_submitting():
    ex = _FakeExecutor()
    returned = ex.attach("999")
    assert returned is ex
    assert ex.pid == "999"
    assert ex.is_submitted is True


# --------------------------------------------------------------------------
# End-to-end: detach -> reload -> wait -> collect (direct backend)
# --------------------------------------------------------------------------


def test_single_detach_resume_collect(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    script = tmp_path / "single.py"
    script.write_text(SINGLE_SCRIPT)
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps({"A": 1, "B": 2}))

    task = tasks.SingleScriptTask(
        script_file=script,
        input_file=input_file,
        local_savepath=tmp_path / "results",
        cluster_config=_direct_config(remote),
        remote_fs=file_handler.LocalFileSystem(),
        logfile="single.log",
        loop_time=0.2,
        remote_savepath=str(remote / "calc"),
    )

    # Detach: fire the job and get a durable handle back immediately.
    manifest_path = task.detach()
    assert manifest_path.exists()
    # A copy also lives beside the job on the "remote".
    assert (Path(task.savepath) / MANIFEST_NAME).exists()

    # Simulate a fresh process: rebuild purely from the manifest JSON.
    handle = TaskHandle.from_file(manifest_path)
    assert handle.manifest.task_type == "single"
    handle.wait()
    assert handle.is_finished()
    local = handle.collect()

    result = json.loads((local / "result.json").read_text())
    assert result["sum"] == 3


def test_array_detach_resume_collect(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    script = tmp_path / "array.py"
    script.write_text(ARRAY_SCRIPT)
    fs = file_handler.LocalFileSystem()

    n_shards = 3
    inputs = [
        tasks.DictJSON({"A": 10, "B": i}, "input.json", fs)
        for i in range(n_shards)
    ]

    task = tasks.ArrayScriptTask(
        local_savepath=tmp_path / "results",
        cluster_config=_direct_config(remote),
        remote_fs=fs,
        python_script_file=script,
        individual_ressources=[inputs],
        logfile="array.log",
        loop_time=0.2,
        remote_savepath=str(remote),
    )

    manifest_path = task.detach()
    handle = TaskHandle.from_file(manifest_path)
    assert handle.manifest.task_type == "array"
    assert handle.manifest.num_array_tasks == n_shards

    local = handle.wait_and_collect()

    from remote_handler.core.remote_env import CALCDIRS_DIRNAME

    for i in range(n_shards):
        shard_result = local / CALCDIRS_DIRNAME / str(i) / "result.json"
        assert shard_result.exists(), f"missing result for shard {i}"
        assert json.loads(shard_result.read_text())["sum"] == 10 + i


def test_blocking_submit_writes_manifest_by_default(tmp_path):
    """A plain blocking run persists a resume handle -- covering the original
    crash case (driver dies while waiting on a job it submitted)."""
    remote = tmp_path / "remote"
    remote.mkdir()
    script = tmp_path / "single.py"
    script.write_text(SINGLE_SCRIPT)
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps({"A": 1, "B": 2}))

    task = tasks.SingleScriptTask(
        script_file=script,
        input_file=input_file,
        local_savepath=tmp_path / "results",
        cluster_config=_direct_config(remote),
        remote_fs=file_handler.LocalFileSystem(),
        logfile="single.log",
        loop_time=0.2,
        remote_savepath=str(remote / "calc"),
    )
    task.run()  # blocking, no detach

    local_manifest = tmp_path / "results.manifest.json"
    assert local_manifest.exists()
    # The persisted handle round-trips and points at this job.
    handle = TaskHandle.from_file(local_manifest)
    assert handle.manifest.task_type == "single"
    assert handle.is_finished()


def test_write_manifest_on_submit_can_be_disabled(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    script = tmp_path / "single.py"
    script.write_text(SINGLE_SCRIPT)
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps({"A": 1, "B": 2}))

    task = tasks.SingleScriptTask(
        script_file=script,
        input_file=input_file,
        local_savepath=tmp_path / "results",
        cluster_config=_direct_config(remote),
        remote_fs=file_handler.LocalFileSystem(),
        logfile="single.log",
        loop_time=0.2,
        remote_savepath=str(remote / "calc"),
        write_manifest_on_submit=False,
    )
    task.run()

    assert not (tmp_path / "results.manifest.json").exists()


def test_manifest_write_failure_never_breaks_submit(tmp_path, monkeypatch):
    """The safety net must not take down the job it protects: a failing
    manifest write only warns, the run still completes."""
    remote = tmp_path / "remote"
    remote.mkdir()
    script = tmp_path / "single.py"
    script.write_text(SINGLE_SCRIPT)
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps({"A": 3, "B": 4}))

    task = tasks.SingleScriptTask(
        script_file=script,
        input_file=input_file,
        local_savepath=tmp_path / "results",
        cluster_config=_direct_config(remote),
        remote_fs=file_handler.LocalFileSystem(),
        logfile="single.log",
        loop_time=0.2,
        remote_savepath=str(remote / "calc"),
    )
    # Force every manifest write to blow up.
    monkeypatch.setattr(
        task,
        "_save_manifest",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    task.run()  # must still finish and collect

    result = json.loads((tmp_path / "results" / "result.json").read_text())
    assert result["sum"] == 7


# --------------------------------------------------------------------------
# Finalizer (post-collect hook registered on the RemoteTask before submit)
# --------------------------------------------------------------------------


def _double_sum_finalizer(local_path, multiplier=1):
    """Stand-in for a project-specific post-collect assembly step.

    Referenced from tests by its dotted path ``"test_manifest:..."`` -- exactly
    how a downstream project (e.g. a ``Sampler`` living outside this package)
    would register one on ``RemoteTask.finalizer``.
    """
    result = json.loads((Path(local_path) / "result.json").read_text())
    out = Path(local_path) / "finalized.json"
    out.write_text(json.dumps({"doubled": result["sum"] * multiplier}))
    return out


def test_finalizer_runs_after_collect(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    script = tmp_path / "single.py"
    script.write_text(SINGLE_SCRIPT)
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps({"A": 1, "B": 2}))

    task = tasks.SingleScriptTask(
        script_file=script,
        input_file=input_file,
        local_savepath=tmp_path / "results",
        cluster_config=_direct_config(remote),
        remote_fs=file_handler.LocalFileSystem(),
        logfile="single.log",
        loop_time=0.2,
        remote_savepath=str(remote / "calc"),
    )
    task.finalizer = "test_manifest:_double_sum_finalizer"
    task.finalizer_kwargs = {"multiplier": 2}

    manifest_path = task.detach()
    handle = TaskHandle.from_file(manifest_path)
    assert handle.manifest.finalizer == "test_manifest:_double_sum_finalizer"
    local = handle.wait_and_collect()

    assert handle.finalizer_error is None
    assert handle.finalized_output == local / "finalized.json"
    finalized = json.loads((local / "finalized.json").read_text())
    assert finalized["doubled"] == 6


def test_finalizer_failure_does_not_break_collect(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    script = tmp_path / "single.py"
    script.write_text(SINGLE_SCRIPT)
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps({"A": 1, "B": 2}))

    task = tasks.SingleScriptTask(
        script_file=script,
        input_file=input_file,
        local_savepath=tmp_path / "results",
        cluster_config=_direct_config(remote),
        remote_fs=file_handler.LocalFileSystem(),
        logfile="single.log",
        loop_time=0.2,
        remote_savepath=str(remote / "calc"),
    )
    task.finalizer = "no_such_module_at_all:fn"

    manifest_path = task.detach()
    handle = TaskHandle.from_file(manifest_path)
    local = handle.wait_and_collect()

    # Raw files are there regardless of the finalizer blowing up.
    result = json.loads((local / "result.json").read_text())
    assert result["sum"] == 3
    assert handle.finalized_output is None
    assert isinstance(handle.finalizer_error, TaskError)


def test_detach_can_skip_remote_manifest(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    script = tmp_path / "single.py"
    script.write_text(SINGLE_SCRIPT)
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps({"A": 4, "B": 5}))

    task = tasks.SingleScriptTask(
        script_file=script,
        input_file=input_file,
        local_savepath=tmp_path / "results",
        cluster_config=_direct_config(remote),
        remote_fs=file_handler.LocalFileSystem(),
        logfile="single.log",
        loop_time=0.2,
        remote_savepath=str(remote / "calc"),
    )

    manifest_path = task.detach(write_remote=False)
    assert manifest_path.exists()
    assert not (Path(task.savepath) / MANIFEST_NAME).exists()

    TaskHandle.from_file(manifest_path).wait_and_collect()
    result = json.loads((tmp_path / "results" / "result.json").read_text())
    assert result["sum"] == 9
