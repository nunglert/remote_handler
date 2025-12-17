import json
import os
import sys
import numpy as np
from pathlib import Path

import ase.io
from jaxnest.postprocess import properties

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


if do_optimization:
    calc_args = input_args["calculator_args"]
    if calc_args["optimization_backend"] == "NNFF":
        from jaxnest.postprocess import ensemble_ase_calculator

        pickle_file = calc_args["pickle_file"]
        if not os.path.exists(pickle_file):
            raise Exception(
                f"Could not find {pickle_file=}. Current location is {os.getcwd()}"
            )

        calc = ensemble_ase_calculator.EnsembleASECalculator(
            pickle_file=pickle_file,
            supercell_trafo=calc_args["supercell_trafo"],
            max_neighbors_list=calc_args["max_neighbors_list"],
            max_neighbors_offset=calc_args["max_neighbors_offset"],
            auto_sc=True,
        )

    elif calc_args["optimization_backend"] == "LJ":
        from ase.calculators.lj import LennardJones

        calc = LennardJones(sigma=1.0, epsilon=1.0, rc=3.0)

    else:
        raise Exception("Unknown optimization backend!")

ats = ase.io.read(DATABASE, index=":")

processed_file = front_tag(DATABASE, TAG) # File to write the processed structures to

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

# If using QUIP for steinhardt we compute the qs for the whole trajectory at
# once and set do_steinhardt False afterwards to avoid triggering steinhardt
# computation the loop below
if do_steinhardt:
    print("... Computing Steinhardt using QUIP ...")
    for l in input_args["steinhardt_args"]["steinhardt_l_values"]:
        properties.get_q_quip_inplace(
            ats, input_args["general_args"]["cutoff"], path=path, l=l
        )

if do_rdf:
    print("... Computing RDF ...")
    for at in ats:
        rdf, r = properties.get_rdf(
            at, 
            input_args["general_args"]["cutoff"],
            input_args["general_args"]["nbins"]
        )
        at.info["rdf"] = rdf
        at.info["nbins"] = input_args["general_args"]["nbins"]
        at.info["rmax"] = input_args["general_args"]["cutoff"]



for idx, at in enumerate(ats):
    at_pre = at.copy()
    at_pre.info["is_finished"] = False

    try:
        is_finished = False
        
        if do_rdf and "rdf" in at.info:
            is_finished = True
        if do_steinhardt:
            if np.all(
                [
                    f"q{l}_rcut{input_args['general_args']['cutoff']}" in at.arrays
                    for l in input_args["steinhardt_args"]["steinhardt_l_values"]
                ]
            ):
                is_finished = True

        if do_optimization:
            if not "is_converged" in at.info:
                print("... Optimizing")
                at = properties.optimize(
                    at,
                    calc,
                    # pressure=None,
                    # fmax=0.1,
                    trajectory_file=None,
                    # steps=100
                    **input_args["optimization_args"],
                )
            is_finished = True

        at.info["shard_idx"] = idx
        at.info["is_finished"] = is_finished
        ase.io.write(processed_file, at, append=True)

    except Exception as e:
        print(f"Error processing structure {idx}: {e}")
        continue
