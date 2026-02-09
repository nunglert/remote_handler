import json
import os
import sys
from pathlib import Path

import ase.io
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.vasp import Vasp

calcpath = Path(sys.argv[1])
INPUTFILE = calcpath / "input_opt.json"
input_args = json.load(open(INPUTFILE, "r"))

DATABASE = input_args["DATABASE"]
launch_command = input_args["launch_command"]
# CALC_IDX = input_args["CALC_IDX"]
# CALCDIR = input_args["CALCDIR"]
VASP_incar_kwargs = input_args["INCAR_kwargs"]
# print(f"Calculating index {CALC_IDX} of database {DATABASE}")

# with ase.db.connect(DATABASE) as db:
#     for idx, row in enumerate(db.select(index=CALC_IDX)):
#         atoms = row.toatoms()

atoms = ase.io.read(calcpath / DATABASE)

try:
    at = ase.io.read(calcpath / "vasprun.xml")
    repeat_calculation = False
except:
    repeat_calculation = True

if not repeat_calculation:
    raise Exception("Calculation already ran succesfully. Skipping it.")

print(atoms)

VASP_calculator_args = {
    "command": launch_command,  # TODO: prun?
    "txt": None,
}

# VASP_incar_kwargs = {
#     # "istart": 0,
#     # "icharg": 2,
#     "xc": "R2SCAN",
#     "encut": 300,
#     "prec": "Normal",
#     "ismear": -1,
#     "sigma": 0.025,
#     "ibrion": -1,
#     "ediff": 1e-5,
#     # "isym": 0,
#     "symprec": 1e-7,
#     "lwave": False,
#     "lcharg": False,
#     "lreal": False,
#     # "kspacing": 2.0
#     "kspacing": 0.3,
# }

VASP_config = {**VASP_calculator_args, **VASP_incar_kwargs}

# foldername = os.path.join(CALCDIR)
# if os.path.exists(foldername):
#     raise Exception(f"The path {foldername} exists already. Make sure that nothing is overwritten!")
# os.mkdir(foldername)

Vasp.xc_defaults["r2scan"] = {"metagga": "R2SCAN"}
atoms.calc = Vasp(directory=calcpath, **VASP_config)

atoms.calc = SinglePointCalculator(
    atoms, energy=atoms.get_potential_energy(), forces=atoms.get_forces()
)

ase.io.write(calcpath / ("dft_" + DATABASE), atoms)
################

# with ase.db.connect(DATABASE) as db:
#     for idx, row in enumerate(db.select(index=CALC_IDX)):
#         at = row.toatoms()
#         if not np.all(atoms.positions == at.positions):
#             raise Exception(
#                 "Configuration you write to is not the same as the one you calculated for!"
#             )
#         at.calc = SinglePointCalculator(
#             at,
#             energy=atoms.calc.get_potential_energy(),
#             forces=atoms.calc.get_forces(),
#         )
#         db.update(
#             row.id,
#             atoms=at,
#             data={"vasp_input": VASP_incar_kwargs},
#             vasp_finished=True,
#         )

# with open(Path(CALCDIR) / is_finished.info, "w") as f:
#     f.write(str(True))
