import json
import os
import shutil
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from remote_handler.core import file_handler, tasks
from remote_handler.core.cluster_config import ClusterConfig
from remote_handler.exceptions import TaskError

SCRIPT_STR = """
import sys
import time
import json

file = sys.argv[1]

with open(file, "r") as f:
    data = json.load(f)

print(data["A"] + data["B"])

time.sleep(1)

with open("is_finished.info", "w") as f:
    f.write("I'm finished af")
"""

ARRAY_SCRIPT_STR = """
import sys
import time
import json
from pathlib import Path

calcpath = Path(sys.argv[1])
INPUT_FILE = calcpath / "input.json"
CONFIG_FILE = calcpath / "config.json"

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

with open(CONFIG_FILE, "r") as f:
    config = json.load(f)
    print(config)

print(data["A"] + data["B"])

time.sleep(1)

with open(calcpath / "is_finished.info", "w") as f:
    f.write("I'm finished af")

"""


def write_json(A, B, filename):
    content = dict(A=A, B=B)
    with open(filename, "w") as f:
        json.dump(content, f)


@pytest.mark.integration
def test_SingleScriptTask_simple(
    data_dir, filesystem, cluster_config, tmp_path
):
    """This test runs a SingleScriptTask. It generates the required python
    script and a corresponding input file on-the-fly.

    A log file is written to:
        `data/temp/SingleScriptTask/simple`.

    The whole content of the computation directory on the remote system is
    transferred afterwards to:
        `data/temp/SingleScriptTask/simple/results`

    Note, that I perform the `finish_check` here after the results are
    collected. This allows us to comfortably check the output files in case
    something crashed.
    """
    # SLURM submission against the local machine needs sbatch; skip
    # (rather than error with DependencyError) when it is absent.
    if isinstance(filesystem, file_handler.LocalFileSystem) and (
        shutil.which("sbatch") is None
    ):
        pytest.skip("SLURM (sbatch) not available on this machine")

    # os.environ["REMOTE_HANDLER_LOCAL"] = "datalab"
    relative_path = Path("SingleScriptTask") / "simple"
    local_savepath = data_dir / "temp" / relative_path
    local_results_path = local_savepath / "results"

    # The script + input are generated on the fly; write them to a scratch
    # dir (NOT into test/data/assets, which is tracked -- writing there dirties
    # git on every run).
    script_path = tmp_path / "python_script_single.py"
    with open(script_path, "w") as f:
        f.write(SCRIPT_STR)

    if os.path.exists(local_savepath):
        shutil.rmtree(local_savepath)
    os.makedirs(local_savepath, exist_ok=True)

    input_file = tmp_path / "input.json"
    write_json(1, 2, input_file)

    task = tasks.SingleScriptTask(
        script_file=script_path,
        input_file=input_file,
        local_savepath=local_results_path,
        cluster_config=cluster_config,
        remote_fs=filesystem,
        logfile="test_SingleScriptTask.log",
        loop_time=1,
        remote_savepath=None,  # fall back to savepath in config
    )

    task.allocate_data()
    task.submit()
    task.collect_results()

    if not task.finish_check():
        # slurm_err = local_results_path / "slurm.err"
        # with open(slurm_err, "r") as f:
        #     raise Exception(f.read())
        raise Exception(
            "Run not finished? Only found these files: \n"
            f"{os.listdir(local_results_path)} \n"
            f"at {local_results_path}"
        )


@pytest.mark.integration
def test_ArrayScriptTask_simple(
    data_dir, filesystem, cluster_config, tmp_path
):
    """This test runs a ArrayScriptTask. It generates the required python
    script on-the-fly and sets up the directories on the remote machine with
    input and config files.

    A log file is written to:
        `data/temp/ArrayScriptTask/simple`.

    The whole content of the computation directory on the remote system is
    transferred afterwards to:
        `data/temp/ArrayScriptTask/simple/results`

    Note, that I perform the `finish_check` here after the results are
    collected. This allows us to comfortably check the output files in case
    something crashed.
    """
    # SLURM submission against the local machine needs sbatch; skip
    # (rather than error with DependencyError) when it is absent.
    if isinstance(filesystem, file_handler.LocalFileSystem) and (
        shutil.which("sbatch") is None
    ):
        pytest.skip("SLURM (sbatch) not available on this machine")

    # os.environ["REMOTE_HANDLER_LOCAL"] = "datalab"
    relative_path = Path("ArrayScriptTask") / "simple"
    local_savepath = data_dir / "temp" / relative_path
    local_results_path = local_savepath / "results"

    # Generated on the fly -> scratch dir, not the tracked test/data/assets.
    script_path = tmp_path / "python_script_array.py"
    with open(script_path, "w") as f:
        f.write(ARRAY_SCRIPT_STR)

    if os.path.exists(local_savepath):
        shutil.rmtree(local_savepath)
    os.makedirs(local_savepath, exist_ok=True)

    input_data = []
    config_data = []
    for i in range(4):
        mydict = dict(A=0, B=i)
        input_data.append(tasks.DictJSON(mydict, "input.json", filesystem))
        config_data.append(tasks.DictJSON(mydict, "config.json", filesystem))

    individual_ressources = [input_data, config_data]

    task = tasks.ArrayScriptTask(
        local_results_path,
        cluster_config,
        filesystem,
        script_path,
        individual_ressources,
        shared_ressources=[],
        logfile="test_ArrayScriptTask.log",
        loop_time=1,
        remote_savepath=None,  # fall back to savepath in config
    )

    task.allocate_data()
    task.submit()
    task.collect_results()

    if not task.finish_check():
        slurm_err = (
            local_results_path
            / "slurm_output"
            / f"slurm_{task.slurm_handler.pid}_1.err"
        )
        with open(slurm_err, "r") as f:
            raise Exception(f.read())


# --------------------------------------------------------------------------
# Allocation-independent finish_check/collect_results for collect_only
# ArrayScriptTask instances (no individual_ressources; num_array_tasks
# either explicit or reconstructed elsewhere via from_manifest, see
# test_manifest.py). Uses the scheduler-free "direct" backend on a
# LocalFileSystem, so no sbatch/cluster is needed.
# --------------------------------------------------------------------------


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


def _make_finished_array_calcdirs(remote, jobname, n):
    """Pre-populate <remote>/<jobname>/calcdirs/<i>/ as an already-finished
    array run, without ever constructing/running a real ArrayScriptTask."""
    from remote_handler.core.remote_env import CALCDIRS_DIRNAME

    calcdir = remote / jobname / CALCDIRS_DIRNAME
    for i in range(n):
        d = calcdir / str(i)
        d.mkdir(parents=True)
        (d / "is_finished.info").write_text("done")
        (d / "result.json").write_text(json.dumps({"i": i}))
    return calcdir


def test_array_collect_only_with_explicit_num_array_tasks(tmp_path):
    """collect_only=True + num_array_tasks (no individual_ressources) is
    enough for all_calcpaths/finish_check/collect_results -- no allocation
    needed."""
    remote = tmp_path / "remote"
    remote.mkdir()
    fs = file_handler.LocalFileSystem()
    n = 3
    jobname = "job"
    calcdir = _make_finished_array_calcdirs(remote, jobname, n)

    task = tasks.ArrayScriptTask(
        local_savepath=tmp_path / "results",
        cluster_config=_direct_config(remote),
        remote_fs=fs,
        remote_savepath=str(remote),
        jobname=jobname,
        collect_only=True,
        num_array_tasks=n,
    )

    assert task.all_calcpaths == [calcdir / str(i) for i in range(n)]
    assert task.finish_check()

    task.collect_results()
    from remote_handler.core.remote_env import CALCDIRS_DIRNAME

    for i in range(n):
        result = (
            tmp_path / "results" / CALCDIRS_DIRNAME / str(i) / "result.json"
        )
        assert json.loads(result.read_text())["i"] == i


def test_array_collect_only_without_num_array_tasks_raises(tmp_path):
    """No individual_ressources and no explicit num_array_tasks -> a clear
    TaskError, not an AttributeError, from all_calcpaths/finish_check."""
    remote = tmp_path / "remote"
    remote.mkdir()
    fs = file_handler.LocalFileSystem()
    jobname = "job"
    _make_finished_array_calcdirs(remote, jobname, 2)

    task = tasks.ArrayScriptTask(
        local_savepath=tmp_path / "results",
        cluster_config=_direct_config(remote),
        remote_fs=fs,
        remote_savepath=str(remote),
        jobname=jobname,
        collect_only=True,
    )

    with pytest.raises(TaskError):
        task.all_calcpaths
    with pytest.raises(TaskError):
        task.finish_check()


def test_array_num_array_tasks_mismatch_raises(tmp_path):
    """An explicit num_array_tasks that disagrees with individual_ressources
    is a construction-time TaskError, not a silently-wrong value."""
    remote = tmp_path / "remote"
    remote.mkdir()
    fs = file_handler.LocalFileSystem()
    script = tmp_path / "array.py"
    script.write_text("pass\n")
    inputs = [tasks.DictJSON({"A": 1}, "input.json", fs) for _ in range(3)]

    with pytest.raises(TaskError):
        tasks.ArrayScriptTask(
            local_savepath=tmp_path / "results",
            cluster_config=_direct_config(remote),
            remote_fs=fs,
            python_script_file=script,
            individual_ressources=[inputs],
            remote_savepath=str(remote),
            num_array_tasks=999,
        )


def test_array_num_array_tasks_explicit_matches_inferred(tmp_path):
    """An explicit num_array_tasks that agrees with individual_ressources is
    accepted (not just tolerated when they disagree)."""
    remote = tmp_path / "remote"
    remote.mkdir()
    fs = file_handler.LocalFileSystem()
    script = tmp_path / "array.py"
    script.write_text("pass\n")
    inputs = [tasks.DictJSON({"A": 1}, "input.json", fs) for _ in range(3)]

    task = tasks.ArrayScriptTask(
        local_savepath=tmp_path / "results",
        cluster_config=_direct_config(remote),
        remote_fs=fs,
        python_script_file=script,
        individual_ressources=[inputs],
        remote_savepath=str(remote),
        num_array_tasks=3,
    )
    assert task.num_array_tasks == 3


def test_single_collect_only_script_and_input_file_none(tmp_path):
    """collect_only=True + script_file=None/input_file=None -- the shape
    SingleScriptTask.from_manifest builds -- still supports finish_check and
    collect_results against an already-finished calc dir."""
    remote = tmp_path / "remote"
    remote.mkdir()
    fs = file_handler.LocalFileSystem()
    jobname = "job"
    calcdir = remote / jobname
    calcdir.mkdir(parents=True)
    (calcdir / "is_finished.info").write_text("done")
    (calcdir / "result.json").write_text(json.dumps({"sum": 42}))

    task = tasks.SingleScriptTask(
        script_file=None,
        input_file=None,
        local_savepath=tmp_path / "results",
        cluster_config=_direct_config(remote),
        remote_fs=fs,
        remote_savepath=str(remote),
        jobname=jobname,
        collect_only=True,
    )

    assert task.finish_check()
    task.collect_results()
    result = json.loads((tmp_path / "results" / "result.json").read_text())
    assert result["sum"] == 42


# ---------------------------------------------------------------------------
# DetachableTaskMixin: pure-logic unit tests, mocking away the actual task
# classes. Real end-to-end round trips through it already exist (MACE/
# NeuralilIL trainers and AseCalculatorEvaluator in
# test/workflows/test_standalone_collect.py and test_emt_evaluator.py); these
# instead pin down the mixin's own branching in isolation -- what gets
# called, with what, in each of collect_only True/False -- independent of
# any concrete actor.
# ---------------------------------------------------------------------------


class _Actor(tasks.DetachableTaskMixin):
    """Minimal stand-in for a trainer/evaluator: just the two attributes
    DetachableTaskMixin actually reads."""

    def __init__(self, collect_only, remote_fs=None):
        self.collect_only = collect_only
        self.remote_fs = remote_fs


def _fake_task_cls():
    """A fresh fake task class per test, recording from_manifest() calls on
    itself rather than a shared/class-level list."""

    class _FakeTask:
        from_manifest_calls = []

        @classmethod
        def from_manifest(cls, manifest_path, remote_fs=None, **extra):
            cls.from_manifest_calls.append((manifest_path, remote_fs, extra))
            return "resumed-task"

    return _FakeTask


def test_manifest_path_is_deterministic_from_results_dir(tmp_path):
    actor = _Actor(collect_only=False)
    results_dir = tmp_path / "iter0" / "train" / "results"
    assert actor.manifest_path(results_dir) == Path(
        str(results_dir) + ".manifest.json"
    )


def test_build_or_resume_task_builds_and_allocates_when_not_collect_only(
    tmp_path,
):
    actor = _Actor(collect_only=False)
    built_task = MagicMock()
    build_task = MagicMock(return_value=built_task)

    result = actor.build_or_resume_task(
        _fake_task_cls(), tmp_path / "results", build_task
    )

    build_task.assert_called_once_with()
    built_task.allocate_data.assert_called_once_with()
    assert result is built_task


def test_build_or_resume_task_resumes_from_manifest_when_collect_only(
    tmp_path,
):
    fs = object()
    actor = _Actor(collect_only=True, remote_fs=fs)
    build_task = MagicMock()
    fake_cls = _fake_task_cls()
    results_dir = tmp_path / "results"

    result = actor.build_or_resume_task(
        fake_cls, results_dir, build_task, extra_kwarg="value"
    )

    # The whole point of collect_only: never touch build_task -- it may
    # resolve a remote train-runner path or need the original input data,
    # neither of which a pure collect call should require.
    build_task.assert_not_called()
    assert result == "resumed-task"

    [(manifest_path, remote_fs, extra)] = fake_cls.from_manifest_calls
    assert manifest_path == actor.manifest_path(results_dir)
    assert remote_fs is fs
    assert extra == {"extra_kwarg": "value"}


def test_build_or_resume_array_task_dispatches_to_ArrayScriptTask(tmp_path):
    actor = _Actor(collect_only=True, remote_fs=object())
    with patch.object(tasks.ArrayScriptTask, "from_manifest") as mocked:
        mocked.return_value = "array-task"
        result = actor.build_or_resume_array_task(
            tmp_path / "results", MagicMock()
        )
    mocked.assert_called_once()
    assert result == "array-task"


def test_build_or_resume_single_task_dispatches_to_SingleScriptTask(
    tmp_path,
):
    actor = _Actor(collect_only=True, remote_fs=object())
    with patch.object(tasks.SingleScriptTask, "from_manifest") as mocked:
        mocked.return_value = "single-task"
        result = actor.build_or_resume_single_task(
            tmp_path / "results", MagicMock()
        )
    mocked.assert_called_once()
    assert result == "single-task"


def test_submit_if_needed_submits_when_not_collect_only():
    actor = _Actor(collect_only=False)
    task = MagicMock()
    actor.submit_if_needed(task)
    task.submit.assert_called_once_with()


def test_submit_if_needed_skips_submit_when_collect_only():
    """The other half of the point: a collect_only actor must never
    re-submit a job that another process already started."""
    actor = _Actor(collect_only=True)
    task = MagicMock()
    actor.submit_if_needed(task)
    task.submit.assert_not_called()
