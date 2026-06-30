"""Remote payload: compute structural descriptors on a trajectory shard.

Computes Steinhardt bond-order parameters (via QUIP) and/or the radial
distribution function for each structure in the shard and writes the annotated
structures back. No force-field model is involved (contrast
``optimize_traj.py``).

Runs on the cluster in the jaxnest conda env; not imported host-side.
"""

import os
import sys
from pathlib import Path

import ase.io
from jaxnest.postprocess import properties

from remote_handler.scripts.input_models import AnalyzeTrajInput

# Load and validate input configuration
calcpath = Path(sys.argv[1])
INPUTFILE = calcpath / "input_analysis.json"
input_config = AnalyzeTrajInput.from_file(INPUTFILE)
input_config.validate(calcpath)

TAG = input_config.tag
DATABASE = calcpath / input_config.DATABASE  # Initial file with structures

do_steinhardt = input_config.do_steinhardt
do_rdf = input_config.do_rdf

if not os.path.exists(DATABASE):
    raise Exception(
        f"Could not find {DATABASE=}. Current location is {os.getcwd()}"
    )


def front_tag(filepath, tag):
    connector = "" if tag == "" else "_"
    path, filename = os.path.split(filepath)
    return Path(path) / f"{tag}{connector}{filename}"


ats = ase.io.read(DATABASE, index=":")

processed_file = front_tag(DATABASE, TAG)  # Where processed structures go

# Fold any already-processed shard back in so finished configs are skipped.
if os.path.exists(processed_file):
    print(f"Reading existing {processed_file}")
    proc_ats = ase.io.read(processed_file, index=":")
    if "shard_idx" not in proc_ats[0].info.keys():
        print("No shard_idx, deleting existing file...")
    for proc_at in proc_ats:
        idx = proc_at.info["shard_idx"]
        ats[idx] = proc_at
    os.remove(processed_file)

path, _ = os.path.split(DATABASE)
cutoff = input_config.general_args["cutoff"]

# QUIP computes Steinhardt for the whole shard at once, in place.
if do_steinhardt:
    print("... Computing Steinhardt using QUIP ...")
    l = input_config.steinhardt_args["steinhardt_l_value"]
    properties.get_q_quip_inplace(ats, cutoff, path=path, l=l)

if do_rdf:
    print("... Computing RDF ...")
    nbins = input_config.general_args["nbins"]
    for at in ats:
        r, rdf_dict = properties.get_rdf(at, cutoff, nbins)
        at.info["r_rdf"] = r.tolist()
        for key in rdf_dict:
            at.info[f"rdf_{key}"] = rdf_dict[key].tolist()

for idx, at in enumerate(ats):
    try:
        is_finished = False
        if do_rdf and "r_rdf" in at.info:
            is_finished = True
        if do_steinhardt and f"q{l}_rcut{cutoff}" in at.arrays:
            is_finished = True

        at.info["shard_idx"] = idx
        at.info["is_finished"] = is_finished
        ase.io.write(processed_file, at, append=True)

    except Exception as e:
        print(f"Error processing structure {idx}: {e}")
        continue
