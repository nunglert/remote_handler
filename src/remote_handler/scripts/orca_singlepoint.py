import sys
from pathlib import Path

import ase.io
from ase.calculators.orca import ORCA, OrcaProfile
from ase.calculators.singlepoint import SinglePointCalculator

from remote_handler.scripts.input_models import OrcaSinglepointInput

# ----------------------------------------------------------------------
# Load and validate input configuration
# ----------------------------------------------------------------------
calcpath = Path(sys.argv[1])
INPUTFILE = calcpath / "input_dft.json"
input_config = OrcaSinglepointInput.from_file(INPUTFILE)
input_config.validate()
# ----------------------------------------------------------------------

atoms = ase.io.read(calcpath / input_config.DATABASE)

print(atoms)

profile = OrcaProfile(command=input_config.orca_binary)

atoms.calc = ORCA(
    profile=profile,
    directory=calcpath,
    orcasimpleinput=input_config.orcasimpleinput,
    orcablocks=input_config.orcablocks,
)
atoms.calc = SinglePointCalculator(
    atoms, energy=atoms.get_potential_energy(), forces=atoms.get_forces()
)

atoms.info["is_finished"] = True
ase.io.write(calcpath / ("dft_" + input_config.DATABASE), atoms)

with open(calcpath / "is_finished.info", "w") as f:
    f.write("I'm a finished DFT run")
