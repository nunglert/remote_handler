import json
import sys
from pathlib import Path

import ase.io
from ase.calculators.orca import ORCA
from ase.calculators.orca import OrcaProfile

from ase.calculators.singlepoint import SinglePointCalculator

calcpath = Path(sys.argv[1])
INPUTFILE = calcpath / "input_dft.json"
input_args = json.load(open(INPUTFILE, "r"))

DATABASE = input_args["DATABASE"]
orca_binary = input_args["orca_binary"]

orcasimpleinput = input_args["orcasimpleinput"]
orcablocks = input_args["orcablocks"]

atoms = ase.io.read(calcpath / DATABASE)

print(atoms)

profile = OrcaProfile(command=orca_binary)


atoms.calc = ORCA(profile=profile,
        directory=calcpath,
        orcasimpleinput=orcasimpleinput,
        orcablocks=orcablocks,
        )
atoms.calc = SinglePointCalculator(
    atoms, energy=atoms.get_potential_energy(), forces=atoms.get_forces()
)

atoms.info["is_finished"] = True
ase.io.write(calcpath / ("dft_" + DATABASE), atoms)

with open(calcpath / "is_finished.info", "w") as f:
    f.write("I'm a finished DFT run")
