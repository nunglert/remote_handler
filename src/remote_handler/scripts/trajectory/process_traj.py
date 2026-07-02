"""Remote payload: run a pipeline of processors over a trajectory shard.

This driver stays deliberately tiny: read the shard once, run each requested
step from the processor registry in order, write once. All the per-analysis
logic lives in ``processors/`` (one small module per step), so this file never
grows a ``do_X`` cascade -- adding an analysis touches only the registry.

Runs on the cluster in the model/analysis conda env; not imported host-side.
"""

import os
import sys
from pathlib import Path

import ase.io

from remote_handler.scripts.input_models import ProcessTrajInput
from remote_handler.scripts.trajectory.processors import REGISTRY

calcpath = Path(sys.argv[1])
INPUTFILE = calcpath / "input_process.json"
cfg = ProcessTrajInput.from_file(INPUTFILE)
cfg.validate(calcpath)

TAG = cfg.tag
DATABASE = calcpath / cfg.DATABASE

if not os.path.exists(DATABASE):
    raise Exception(
        f"Could not find {DATABASE=}. Current location is {os.getcwd()}"
    )


def front_tag(filepath, tag):
    connector = "" if tag == "" else "_"
    path, filename = os.path.split(filepath)
    return Path(path) / f"{tag}{connector}{filename}"


ats = ase.io.read(DATABASE, index=":")
processed_file = front_tag(DATABASE, TAG)

# Resume: fold any already-processed shard back in so finished configs persist.
if os.path.exists(processed_file):
    print(f"Reading existing {processed_file}")
    proc_ats = ase.io.read(processed_file, index=":")
    if "shard_idx" not in proc_ats[0].info.keys():
        print("No shard_idx, deleting existing file...")
    for proc_at in proc_ats:
        ats[proc_at.info["shard_idx"]] = proc_at
    os.remove(processed_file)

# Run each requested step in order over the whole shard.
for step in cfg.steps:
    if step not in REGISTRY:
        raise ValueError(
            f"Unknown processor {step!r}; known: {sorted(REGISTRY)}"
        )
    print(f"... Running processor: {step}")
    ats = REGISTRY[step](ats, cfg.args.get(step, {}), calcpath=calcpath)

for idx, at in enumerate(ats):
    at.info["shard_idx"] = idx
    at.info["is_finished"] = True
    ase.io.write(processed_file, at, append=True)
