import os
import sys
from pathlib import Path

import ase.io
from jaxnest.postprocess import properties

from remote_handler.scripts.input_models import OptimizeTrajInput

# Load and validate input configuration
calcpath = Path(sys.argv[1])
INPUTFILE = calcpath / "input_opt.json"
input_config = OptimizeTrajInput.from_file(INPUTFILE)
input_config.validate()

TAG = input_config.tag
DATABASE = calcpath / input_config.DATABASE  # Initial file with structures

do_optimization = input_config.do_optimization
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


if do_optimization:
    calc_args = input_config.calculator_args
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

processed_file = front_tag(
    DATABASE, TAG
)  # File to write the processed structures to

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
    l = input_config.steinhardt_args["steinhardt_l_value"]
    properties.get_q_quip_inplace(
        ats, input_config.general_args["cutoff"], path=path, l=l
    )

if do_rdf:
    print("... Computing RDF ...")
    for at in ats:
        r, rdf_dict = properties.get_rdf(
            at,
            input_config.general_args["cutoff"],
            input_config.general_args["nbins"],
        )

        at.info["r_rdf"] = r.tolist()
        for key in rdf_dict:
            at.info[f"rdf_{key}"] = rdf_dict[key].tolist()
        # rdf, r = properties.get_rdf(
        #     at,
        #     input_args["general_args"]["cutoff"],
        #     input_args["general_args"]["nbins"]
        # )
        # at.info["rdf"] = rdf
        # at.info["nbins"] = input_args["general_args"]["nbins"]
        # at.info["rmax"] = input_args["general_args"]["cutoff"]


for idx, at in enumerate(ats):
    at_pre = at.copy()
    at_pre.info["is_finished"] = False

    try:
        is_finished = False

        if do_rdf and "r_rdf" in at.info:
            is_finished = True
        if do_steinhardt:
            l = input_config.steinhardt_args["steinhardt_l_value"]
            if f"q{l}_rcut{input_config.general_args['cutoff']}" in at.arrays:
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
                    **input_config.optimization_args,
                )
            is_finished = True

        at.info["shard_idx"] = idx
        at.info["is_finished"] = is_finished
        ase.io.write(processed_file, at, append=True)

    except Exception as e:
        print(f"Error processing structure {idx}: {e}")
        continue
