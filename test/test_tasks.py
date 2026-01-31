import json
import os
import shutil
from pathlib import Path

import pytest
from pytest_lazyfixture import lazy_fixture

from remote_handler.core import file_handler, tasks

SCRIPT_STR = """
import sys
import time
import json

file = sys.argv[1]

with open(file, "r") as f:
    data = json.load(f)

print(data["A"] + data["B"])

time.sleep(5)

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

time.sleep(5)

with open(calcpath / "is_finished.info", "w") as f:
    f.write("I'm finished af")
"""


def write_json(A, B, filename):
    content = dict(A=A, B=B)
    with open(filename, "w") as f:
        json.dump(content, f)


def test_SingleScriptTask_simple(data_dir, filesystem):
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
    # os.environ["REMOTE_HANDLER_LOCAL"] = "datalab"
    relative_path = Path("SingleScriptTask") / "simple"
    assetpath = data_dir / "assets" / relative_path
    local_savepath = data_dir / "temp" / relative_path
    local_results_path = local_savepath / "results"

    script_path = assetpath / "python_script_single.py"
    with open(script_path, "w") as f:
        f.write(SCRIPT_STR)

    if os.path.exists(local_savepath):
        shutil.rmtree(local_savepath)
    os.makedirs(local_savepath, exist_ok=True)

    input_file = assetpath / "input.json"
    write_json(1, 2, input_file)

    task = tasks.SingleScriptTask(
        script_file=script_path,
        input_file=input_file,
        local_savepath=local_results_path,
        remote_fs=filesystem,
        logfile="test_SingleScriptTask.log",
        loop_time=1,
        remote_savepath=None,  # fall back to savepath in config
    )

    task.allocate_data()
    task.submit()
    task.collect_results()

    if not task.finish_check():
        slurm_err = local_results_path / "slurm.err"
        with open(slurm_err, "r") as f:
            raise Exception(f.read())


def test_ArrayScriptTask_simple(data_dir, filesystem):
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
    # os.environ["REMOTE_HANDLER_LOCAL"] = "datalab"
    relative_path = Path("ArrayScriptTask") / "simple"
    assetpath = data_dir / "assets" / relative_path
    local_savepath = data_dir / "temp" / relative_path
    local_results_path = local_savepath / "results"

    script_path = assetpath / "python_script_array.py"
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
