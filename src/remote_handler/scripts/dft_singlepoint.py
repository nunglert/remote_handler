"""VASP single-point payload: label one structure with energy and forces.

Writes ``dft_<DATABASE>`` and the ``is_finished.info`` success marker only
when the electronic (SCF) loop of the final ionic step converged. A run that
stopped at NELM without reaching EDIFF still returns an energy and forces,
but they are not a valid label, so it is reported as failed instead (an
``scf_unconverged.info`` note, no labelled output, non-zero exit) and the
evaluator's ``required_fraction`` decides whether the AL loop continues.

Import-safe (everything runs from :func:`main`), so the convergence check can
be unit-tested.
"""

import os
import re
import sys
from pathlib import Path

import ase.io
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.vasp import Vasp

from remote_handler.scripts.input_models import DftSinglepointInput

# VASP 6 prints one of these at the end of every electronic loop.
_SCF_CONVERGED = "aborting loop because EDIFF is reached"
_SCF_UNCONVERGED = "aborting loop EDIFF was not reached"
_NELM_LINE = re.compile(r"NELM\s*=\s*(\d+)")
_OSZICAR_STEP = re.compile(r"^\s*[A-Z]{2,4}:\s+(\d+)\s")


def scf_converged(calcpath: Path) -> tuple[bool, str]:
    """Whether the electronic loop of the last ionic step converged.

    Uses VASP 6's own end-of-loop message in ``OUTCAR``. VASP 5 prints
    neither message; then the last ionic step's electronic step count in
    ``OSZICAR`` is compared against ``NELM`` (reaching it means the loop was
    cut off). Anything that cannot be decided counts as not converged.

    Returns:
        ``(converged, reason)``; ``reason`` explains a ``False``.
    """
    outcar = Path(calcpath) / "OUTCAR"
    if not outcar.is_file():
        return False, "no OUTCAR"
    text = outcar.read_text(errors="replace")

    last_ok = text.rfind(_SCF_CONVERGED)
    last_bad = text.rfind(_SCF_UNCONVERGED)
    if last_ok >= 0 or last_bad >= 0:
        if last_ok > last_bad:
            return True, ""
        return False, "OUTCAR: EDIFF was not reached (unconverged)"

    nelm = _NELM_LINE.search(text)
    oszicar = Path(calcpath) / "OSZICAR"
    if nelm is None or not oszicar.is_file():
        return False, "cannot determine SCF convergence (no NELM/OSZICAR)"
    steps = 0
    for line in oszicar.read_text(errors="replace").splitlines():
        match = _OSZICAR_STEP.match(line)
        if match:
            steps = int(match.group(1))  # restarts at 1 every ionic step
    if steps == 0:
        return False, "OSZICAR has no electronic steps"
    if steps >= int(nelm.group(1)):
        return False, f"OSZICAR: {steps} electronic steps = NELM"
    return True, ""


def main(argv=None) -> None:
    argv = sys.argv if argv is None else argv

    # ------------------------------------------------------------------
    # Load and validate input configuration
    # ------------------------------------------------------------------
    calcpath = Path(argv[1])
    INPUTFILE = calcpath / "input_dft.json"
    input_config = DftSinglepointInput.from_file(INPUTFILE)
    input_config.validate(calcpath)
    # ------------------------------------------------------------------

    atoms = ase.io.read(calcpath / input_config.DATABASE)

    try:
        ase.io.read(calcpath / "vasprun.xml")
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

    energy = atoms.get_potential_energy()
    forces = atoms.calc.results["forces"]

    converged, reason = scf_converged(calcpath)
    if not converged:
        message = f"SCF not converged, not labelling this structure: {reason}"
        (calcpath / "scf_unconverged.info").write_text(message + "\n")
        sys.exit(message)

    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)

    ase.io.write(calcpath / ("dft_" + input_config.DATABASE), atoms)

    with open(calcpath / "is_finished.info", "w") as f:
        f.write("I'm a finished DFT run")


if __name__ == "__main__":
    main()
