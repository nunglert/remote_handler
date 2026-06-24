"""
Sampler for using SSPD_torch-sim with MACE
"""

import json
import sys
from pathlib import Path

import ase.units
from torch_SSPD.runners.MACE_runners import run_SSPD_MACE


def main():
    spath = Path(sys.argv[1])
    calcpath = Path(spath).parent
    INPUTFILE = Path(spath)
    input_args = json.load(open(INPUTFILE, "r"))
    # outpath = calcpath / "out"
    # os.makedirs(outpath, exist_ok=False)
    model_path = calcpath / input_args["model_file"]
    input_struct_path = calcpath / input_args["input_struct_path"]
    N = input_args["N"]

    # samples_file = outpath / "samples.xyz"

    if "temperature" not in input_args.keys():
        temperature = input_args["inv_beta"] / ase.units.kB
    else:
        temperature = input_args["temperature"]
    if "num_equil" not in input_args.keys():
        num_equil = 0
    else:
        num_equil = input_args["num_equil"]
    run_SSPD_MACE(
        input_struct_path,
        model_file=model_path,
        n_equil=num_equil,
        n_run=input_args["num_iterations"],
        dt=input_args["initial_timestep"],
        N=N,
        seed=input_args["seed"],
        resampling_threshold=input_args["threshold"],
        n_traj_out=1,  # output every step
        temperature=temperature,
        out_path_dir="out",
    )

    # SAMPLING
    # sampling_frequency = input_args["sampling_frequency"]

    # ase.io.write(samples_file, database, format="extxyz")

    # Success marker checked by SingleScriptTask.finish_check on the host.
    (calcpath / "is_finished.info").write_text(
        "SSPD torch_sim sampling finished"
    )


if __name__ == "__main__":
    main()
