import json
import os
import subprocess
import sys
import traceback
from multiprocessing import process
from pathlib import Path

import ase.io
import pyscal3 as pc
from ase.constraints import ExpCellFilter
from ase.geometry.analysis import Analysis
from ase.optimize import BFGS, FIRE, LBFGS
from jaxnest import file_utils, structure_utils, units
from jaxnest.postprocess import properties

calcpath = Path(sys.argv[1])
INPUTFILE = calcpath / "input_opt.json"
input_args = json.load(open(INPUTFILE, "r"))

TAG = input_args["tag"]
DATABASE = calcpath / input_args["DATABASE"]
STUB = input_args["STUB"]
# do_rdf = input_args["do_rdf"]
do_optimization = input_args["do_optimization"]
do_steinhardt = input_args["do_steinhardt"]


def front_tag(filepath, tag):
    connector = "" if tag == "" else "_"
    path, filename = os.path.split(filepath)
    return Path(path) / f"{tag}{connector}{filename}"


ats = ase.io.read(DATABASE, index=":")
processed_file = front_tag(DATABASE, TAG)

# If there is already a broken shard, read it and replace the corresponding
# entries in the unprocessed extxyz file. The loop will skip them
if os.path.exists(processed_file):
    print(f"Reading existing {processed_file}")
    proc_ats = ase.io.read(processed_file, index=":")
    if not "shard_idx" in proc_ats[0].info.keys():
        print("No shard_idx, deleting existing file...")

    for proc_at in proc_ats:
        idx = proc_at.info["shard_idx"]
        ats[idx] = proc_at
    os.remove(processed_file)

path, filename = os.path.split(DATABASE)


for idx, at in enumerate(ats):
    at_pre = at.copy()
    at_pre.info["is_finished"] = False
    try:
        if STUB:
            raise Exception("Skipping computations ...")

        is_finished = True

        at.info["shard_idx"] = idx
        at.info["is_finished"] = is_finished
        ase.io.write(processed_file, at, append=True)

    except Exception as e:
        full_traceback = traceback.format_exc()
        print("Full traceback captured:")
        print(full_traceback)

        ase.io.write(processed_file, at_pre, append=True)
        print(f"... Continuing despite crash for {at_pre}")
