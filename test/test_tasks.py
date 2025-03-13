import pytest
import json
import os
from pathlib import Path
from remote_handler import tasks, file_handler

SCRIPT_STR = """
import sys
import json

file = sys.argv[1]

with open(file, "r") as f:
    data = json.load(f)

print(data["A"] + data["B"])
"""


@pytest.fixture()
def python_script(data_dir):
    script_path = data_dir / "simple_python_script.py"
    
    with open(script_path, "w") as f:
        f.write(SCRIPT_STR)
    
    return script_path
    
def write_json(A, B, filename):
    content = dict(A=A, B=B)
    with open(filename, "w") as f:
        json.dump(content, f)


@pytest.fixture()
def remote_fs():
    return file_handler.LocalFileSystem()

def test_SingleScriptTask_simple(python_script, data_dir, remote_fs):
    os.environ["REMOTE_HANDLER_LOCAL"] = "datalab"
    relative_path = Path("SingleScriptTask") / "simple"
    assetpath = data_dir / "assets" / relative_path
    savepath = data_dir / "temp" / relative_path
    os.makedirs(savepath, exist_ok=True)

    input_file = assetpath / "input.json"
    write_json(1, 2, input_file)

    task = tasks.SingleScriptTask(
        data_dir / "temp" / "test_SingleScriptTask.log",
        python_script,
        input_file,
        remote_fs,
        remote_savepath=savepath
    )

    task.allocate_data()
    task.setup_slurm()
    task.submit()


