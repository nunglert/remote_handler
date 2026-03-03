import json
import os
import sys
import numpy as np
from pathlib import Path
from mace.calculators import MACECalculator
from ase.filters import ExpCellFilter
from ase.optimize import LBFGS
import ase.io

calcpath = Path(sys.argv[1])
INPUTFILE = calcpath / "input_opt.json"

if not os.path.exists(INPUTFILE):
    raise Exception(
        f"Could not find {INPUTFILE=}. Current location is {os.getcwd()}"
    )


input_args = json.load(open(INPUTFILE, "r"))

TAG = input_args["tag"]
DATABASE = calcpath / input_args["DATABASE"] # Initial file with structures 

do_optimization = input_args["do_optimization"]
do_steinhardt = input_args["do_steinhardt"]
do_rdf = input_args["do_rdf"]


if not os.path.exists(DATABASE):
    raise Exception(
        f"Could not find {DATABASE=}. Current location is {os.getcwd()}"
    )

def front_tag(filepath, tag):
    connector = "" if tag == "" else "_"
    path, filename = os.path.split(filepath)
    return Path(path) / f"{tag}{connector}{filename}"


ats = ase.io.read(DATABASE, index=":")

processed_file = front_tag(DATABASE, TAG) # File to write the processed structures to

# If there is already a broken shard, read it and replace the corresponding
# entries in the unprocessed extxyz file. The loop will skip them
if os.path.exists(processed_file):
    os.remove(processed_file)

path, filename = os.path.split(DATABASE)

calc_args = input_args["calculator_args"]
calc = MACECalculator(model_paths=calc_args["pickle_file"], device="cpu")

for idx, at in enumerate(ats):
    at_pre = at.copy()
    at_pre.info["is_finished"] = False

    try:
        is_finished = False
        
        if do_optimization:
            if not "is_converged" in at.info:
                print(f"Optimizing...")
                # if calc_args.get("pressure", None) is not None:
                #     ecf = ExpCellFilter(at, pressure=calc_args["pressure"])
                # else:
                #     ecf = at
                at.calc = calc
                opt = LBFGS(at)
                is_converged = opt.run(fmax=0.01, steps=200)
                at.info["is_converged"] = is_converged
            is_finished = True

        at.info["shard_idx"] = idx
        at.info["is_finished"] = is_finished
        ase.io.write(processed_file, at, append=True)

    except Exception as e:
        print(f"Error processing structure {idx}: {e}")
        continue
