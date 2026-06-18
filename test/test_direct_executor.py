"""Scheduler-free end-to-end tests for the ``direct`` execution backend.

These run real ``SingleScriptTask`` / ``ArrayScriptTask`` pipelines (allocate ->
submit -> collect) against a ``LocalFileSystem`` with
``ClusterConfig.executor == "direct"`` -- i.e. the whole task orchestration,
exercised with *no scheduler and nothing mocked*. This is the coverage the
SLURM-gated task tests cannot give without a live ``sbatch``.

Plain stdlib payload scripts, so no ML stack and no extra deps.
"""

import json
from pathlib import Path

from remote_handler.core import file_handler, tasks
from remote_handler.core.cluster_handler import ClusterConfig

# Writes its result + the success marker into the cwd, which the launcher sets
# to the calcdir (input file is passed as an absolute path).
SINGLE_SCRIPT = """
import json, sys
from pathlib import Path

data = json.load(open(sys.argv[1]))
Path("result.json").write_text(json.dumps({"sum": data["A"] + data["B"]}))
Path("is_finished.info").write_text("done")
"""

# Array shards receive their own dir as argv[1]; everything is relative to it.
ARRAY_SCRIPT = """
import json, sys
from pathlib import Path

calc = Path(sys.argv[1])
data = json.load(open(calc / "input.json"))
(calc / "result.json").write_text(json.dumps({"sum": data["A"] + data["B"]}))
(calc / "is_finished.info").write_text("done")
"""


def _direct_config(savepath):
    # Empty conda fields -> no activation -> the ambient ``python`` runs.
    return ClusterConfig(
        remote_machine="local",
        partition="p",
        conda_env="",
        conda_source="",
        savepath=str(savepath),
        template="unused.sh.j2",
        executor="direct",
    )


def test_direct_single_task_runs_without_scheduler(tmp_path):
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
        logfile="direct_single.log",
        loop_time=0.2,
        remote_savepath=str(remote / "calc"),
    )

    task.allocate_data()
    task.submit()

    assert task.finish_check()
    result = json.loads((Path(task.calcdir) / "result.json").read_text())
    assert result["sum"] == 3


def test_direct_array_task_runs_without_scheduler(tmp_path):
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
        shared_ressources=[],
        logfile="direct_array.log",
        loop_time=0.2,
        remote_savepath=str(remote),
    )

    task.allocate_data()
    task.submit()

    assert task.finish_check()
    for i in range(n_shards):
        shard_result = task.calcdir / str(i) / "result.json"
        assert shard_result.exists(), f"missing result for shard {i}"
        assert json.loads(shard_result.read_text())["sum"] == 10 + i
