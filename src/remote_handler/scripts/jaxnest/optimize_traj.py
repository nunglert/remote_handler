"""Remote payload: optimize a trajectory shard.

Relaxes each structure in the shard with an NNFF ensemble or a Lennard-Jones
calculator and writes the optimized structures back. Structural analysis
(Steinhardt/RDF) is a separate payload (``analyze_traj.py``).

Runs on the cluster in the jaxnest conda env; not imported host-side.
"""

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
input_config.validate(calcpath)

TAG = input_config.tag
DATABASE = calcpath / input_config.DATABASE  # Initial file with structures

if not os.path.exists(DATABASE):
    raise Exception(
        f"Could not find {DATABASE=}. Current location is {os.getcwd()}"
    )


def front_tag(filepath, tag):
    connector = "" if tag == "" else "_"
    path, filename = os.path.split(filepath)
    return Path(path) / f"{tag}{connector}{filename}"


# Build the calculator for the requested backend.
calc_args = input_config.calculator_args
if calc_args["optimization_backend"] in ("NNFF", "MACE"):
    from jaxnest.postprocess import ensemble_ase_calculator

    pickle_file = calc_args["pickle_file"]
    if not os.path.exists(pickle_file):
        raise Exception(
            f"Could not find {pickle_file=}. Current location is "
            f"{os.getcwd()}"
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
    raise Exception(
        f"Unknown optimization backend: {calc_args['optimization_backend']!r}"
    )

ats = ase.io.read(DATABASE, index=":")

processed_file = front_tag(DATABASE, TAG)  # Where processed structures go

# If there is already a (partially) processed shard, fold it back in so the
# loop skips configs that already converged.
if os.path.exists(processed_file):
    print(f"Reading existing {processed_file}")
    proc_ats = ase.io.read(processed_file, index=":")
    if "shard_idx" not in proc_ats[0].info.keys():
        print("No shard_idx, deleting existing file...")
    for proc_at in proc_ats:
        idx = proc_at.info["shard_idx"]
        ats[idx] = proc_at
    os.remove(processed_file)

for idx, at in enumerate(ats):
    try:
        if "is_converged" not in at.info:
            print("... Optimizing")
            at = properties.optimize(
                at,
                calc,
                trajectory_file=None,
                **input_config.optimization_args,
            )

        at.info["shard_idx"] = idx
        at.info["is_finished"] = True
        ase.io.write(processed_file, at, append=True)

    except Exception as e:
        print(f"Error processing structure {idx}: {e}")
        continue
