import json
import os
import sys
import traceback
from pathlib import Path

import ase.io
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


if do_optimization:
    calc_args = input_args["calculator_args"]
    if calc_args["optimization_backend"] == "NNFF":
        from jaxnest.postprocess import ensemble_ase_calculator

        calc = ensemble_ase_calculator.EnsembleASECalculator(
            pickle_file=calc_args["pickle_file"],
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
processed_file = front_tag(DATABASE, TAG)

# If there is already a broken shard, read it and replace the corresponding
# entries in the unprocessed extxyz file. The loop will skip them
if os.path.exists(processed_file):
    print(f"Reading existing {processed_file}")
    proc_ats = ase.io.read(processed_file, index=":")
    if not "shard_idx" in proc_ats[0].info.keys():
        print("No shard_idx, deleting existing file...")
        os.remove(processed_file)
    else:
        for proc_at in proc_ats:
            idx = proc_at.info["shard_idx"]
            ats[idx] = proc_at

path, filename = os.path.split(DATABASE)

# If using QUIP for steinhardt we compute the qs for the whole trajectory at
# once and set do_steinhardt False afterwards to avoid triggering steinhardt
# computation the loop below
if (
    do_steinhardt
    and input_args["steinhardt_args"]["steinhardt_backend"] == "quip"
):
    print("... Computing Steinhardt using QUIP ...")
    for l in input_args["steinhardt_args"]["steinhardt_l_values"]:
        properties.get_q_quip_inplace(
            ats, input_args["steinhardt_args"]["cutoff"], path=path, l=l
        )


for idx, at in enumerate(ats):
    at_pre = at.copy()
    at_pre.info["is_finished"] = False
    try:
        if STUB:
            raise Exception("Skipping computations (stub) ...")

        is_finished = False
        if do_steinhardt:
            sbackend = input_args["steinhardt_args"]["steinhardt_backend"]
            if sbackend == "pyscal":
                cutoff = input_args["steinhardt_args"]["cutoff"]
                nbins = input_args["steinhardt_args"]["nbins"]
                l_values = input_args["steinhardt_args"]["steinhardt_l_values"]
                q4_label = f"q4_rcut{cutoff}"
                q6_label = f"q6_rcut{cutoff}"

                if not (
                    q4_label in at.arrays
                    and q6_label in at.arrays
                    and "rdf" in at.info
                ):
                    print("... Computing Steinhardt")

                    qs, rdf, radii = properties.get_q(
                        at, cutoff, nbins, l_values=l_values
                    )

                    q4, q6 = qs
                    at.arrays[q4_label] = q4
                    at.arrays[q6_label] = q6

                    at.info["rdf"] = rdf
                    at.info["nbins"] = nbins
                    at.info["rmax"] = cutoff

                is_finished = True

            elif sbackend == "quip":
                is_finished = True

            else:
                raise Exception

        if do_optimization:
            if not "is_converged" in at.info:
                print("... Optimizing")
                at = properties.optimize(
                    at,
                    calc,
                    pressure=None,
                    # fmax=0.1,
                    trajectory_file=None,
                    # steps=100
                    **input_args["optimization_args"],
                )
            is_finished *= True

        at.info["shard_idx"] = idx
        at.info["is_finished"] = is_finished
        ase.io.write(processed_file, at, append=True)

    except Exception as e:
        full_traceback = traceback.format_exc()
        print("Full traceback captured:")
        print(full_traceback)

        ase.io.write(processed_file, at_pre, append=True)
        print(f"... Continuing despite crash for {at_pre}")
