import os
import sys
from pathlib import Path

import ase.io
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.vasp import Vasp

from remote_handler.scripts.input_models import DftSinglepointInput

# ----------------------------------------------------------------------
# Load and validate input configuration
# ----------------------------------------------------------------------
calcpath = Path(sys.argv[1])
INPUTFILE = calcpath / "input_dft.json"
input_config = DftSinglepointInput.from_file(INPUTFILE)
input_config.validate(calcpath)
# ----------------------------------------------------------------------

atoms = ase.io.read(calcpath / input_config.DATABASE)

try:
    at = ase.io.read(calcpath / "vasprun.xml")
    repeat_calculation = False
except:
    repeat_calculation = True

if not repeat_calculation:
    raise Exception("Calculation already ran succesfully. Skipping it.")

print(atoms)

VASP_calculator_args = {
    "command": input_config.launch_command,  # TODO: prun?
    "txt": "vasp.out",
}

VASP_config = {**VASP_calculator_args, **input_config.INCAR_kwargs}
paw_path = os.environ.get("VASP_PP_PATH", None)
print(f"Using PAW potentials from: {paw_path}")
Vasp.xc_defaults["r2scan"] = {"metagga": "R2SCAN"}
atoms.calc = Vasp(directory=calcpath, **VASP_config)

atoms.calc = SinglePointCalculator(
    atoms, energy=atoms.get_potential_energy(), forces=atoms.get_forces()
)

ase.io.write(calcpath / ("dft_" + input_config.DATABASE), atoms)

with open(calcpath / "is_finished.info", "w") as f:
    f.write("I'm a finished DFT run")
