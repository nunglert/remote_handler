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
# Full task reconstruction: ArrayScriptTask.from_manifest /
# SingleScriptTask.from_manifest (collect_only, no re-allocation) -- distinct
# from TaskHandle above, which only rebuilds an executor+fs, not a full task
# with finish_check()/collect_results() semantics.
# --------------------------------------------------------------------------


def test_array_from_manifest_collects_without_reallocating(tmp_path):
    """The strongest end-to-end check: two objects standing in for two
    processes across a process boundary -- the second is built purely from
    the manifest JSON, never sees the original task object, and does not
    call allocate_data() before finish_check()/collect_results() succeed.

    Also confirms from_manifest() does not block: it is a plain, synchronous
    reconstruction -- waiting for the job is a separate step (TaskHandle, as
    used here before reconstructing the full task).
    """
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

    manifest_path = task.detach()  # process 1: submit and return at once
    TaskHandle.from_file(manifest_path).wait()  # await, generic handle

    # process 2: full task reconstruction, purely from the manifest file.
    resumed = tasks.ArrayScriptTask.from_manifest(manifest_path, remote_fs=fs)
    assert resumed.num_array_tasks == n_shards
    assert resumed.finish_check()  # no allocate_data() call anywhere above
    resumed.collect_results()

    from remote_handler.core.remote_env import CALCDIRS_DIRNAME

    for i in range(n_shards):
        shard_result = (
            tmp_path / "results" / CALCDIRS_DIRNAME / str(i) / "result.json"
        )
        assert shard_result.exists(), f"missing result for shard {i}"
        assert json.loads(shard_result.read_text())["sum"] == 10 + i


def test_single_from_manifest_collects_without_reallocating(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    script = tmp_path / "single.py"
    script.write_text(SINGLE_SCRIPT)
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps({"A": 1, "B": 2}))
    fs = file_handler.LocalFileSystem()

    task = tasks.SingleScriptTask(
        script_file=script,
        input_file=input_file,
        local_savepath=tmp_path / "results",
        cluster_config=_direct_config(remote),
        remote_fs=fs,
        logfile="single.log",
        loop_time=0.2,
        remote_savepath=str(remote / "calc"),
    )

    manifest_path = task.detach()
    TaskHandle.from_file(manifest_path).wait()

    resumed = tasks.SingleScriptTask.from_manifest(manifest_path, remote_fs=fs)
    assert resumed.finish_check()
    resumed.collect_results()

    result = json.loads((tmp_path / "results" / "result.json").read_text())
    assert result["sum"] == 3


def test_array_from_manifest_rejects_single_manifest(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    script = tmp_path / "single.py"
    script.write_text(SINGLE_SCRIPT)
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps({"A": 1, "B": 2}))
    fs = file_handler.LocalFileSystem()

    task = tasks.SingleScriptTask(
        script_file=script,
        input_file=input_file,
        local_savepath=tmp_path / "results",
        cluster_config=_direct_config(remote),
        remote_fs=fs,
        loop_time=0.2,
        remote_savepath=str(remote / "calc"),
    )
    manifest_path = task.detach()

    with pytest.raises(TaskError):
        tasks.ArrayScriptTask.from_manifest(manifest_path, remote_fs=fs)


def test_single_from_manifest_rejects_array_manifest(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    script = tmp_path / "array.py"
    script.write_text(ARRAY_SCRIPT)
    fs = file_handler.LocalFileSystem()
    inputs = [tasks.DictJSON({"A": 10, "B": 0}, "input.json", fs)]

    task = tasks.ArrayScriptTask(
        local_savepath=tmp_path / "results",
        cluster_config=_direct_config(remote),
        remote_fs=fs,
        python_script_file=script,
        individual_ressources=[inputs],
        loop_time=0.2,
        remote_savepath=str(remote),
    )
    manifest_path = task.detach()

    with pytest.raises(TaskError):
        tasks.SingleScriptTask.from_manifest(manifest_path, remote_fs=fs)


def test_from_manifest_rejects_subclasses(tmp_path):
    """from_manifest() is only for the two base classes -- a subclass (e.g.
    a workflow handler) must reconstruct itself another way."""

    class _NotABaseClass(tasks.ArrayScriptTask):
        pass

    with pytest.raises(TaskError):
        _NotABaseClass.from_manifest(tmp_path / "unused.manifest.json")


def test_from_manifest_rejects_unknown_extra_kwarg(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    script = tmp_path / "array.py"
    script.write_text(ARRAY_SCRIPT)
    fs = file_handler.LocalFileSystem()
    inputs = [tasks.DictJSON({"A": 10, "B": 0}, "input.json", fs)]

    task = tasks.ArrayScriptTask(
        local_savepath=tmp_path / "results",
        cluster_config=_direct_config(remote),
        remote_fs=fs,
        python_script_file=script,
        individual_ressources=[inputs],
        loop_time=0.2,
        remote_savepath=str(remote),
    )
    manifest_path = task.detach()

    with pytest.raises(TaskError):
        tasks.ArrayScriptTask.from_manifest(
            manifest_path, remote_fs=fs, this_is_not_a_real_kwarg=True
        )


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
