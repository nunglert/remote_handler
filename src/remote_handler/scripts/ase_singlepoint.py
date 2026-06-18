"""Generic single-point payload: attach an ASE calculator and label E/F.

Runs **on the remote**. Reads ``input_dft.json`` (an ``AseSinglepointInput``),
builds the named ASE calculator, computes energy + forces, and writes the
labelled ``dft_<DATABASE>`` + ``is_finished.info``. Pure ASE -- no external
binary -- so it runs anywhere ASE is installed (locally, in CI, or on a
cluster), which makes it the cheap ground-truth used by ``EMTEvaluator``.

Add a calculator by extending :func:`build_calculator`.
"""

import sys
from pathlib import Path

import ase.io
from ase.calculators.singlepoint import SinglePointCalculator

from remote_handler.exceptions import ScriptInputError
from remote_handler.scripts.input_models import AseSinglepointInput


def build_calculator(name: str, kwargs: dict):
    """Construct an ASE calculator by name.

    Args:
        name: Calculator identifier (case-insensitive), e.g. ``"emt"``.
        kwargs: Keyword arguments forwarded to the calculator constructor.

    Raises:
        ScriptInputError: If ``name`` is not a known calculator.
    """
    key = name.lower()
    if key == "emt":
        from ase.calculators.emt import EMT

        return EMT(**kwargs)
    if key == "lj":
        from ase.calculators.lj import LennardJones

        return LennardJones(**kwargs)
    if key == "eam":
        from ase.calculators.eam import EAM

        return EAM(**kwargs)
    raise ScriptInputError(
        f"Unknown calculator '{name}'. Known: emt, lj, eam."
    )


def main(calcpath: Path) -> None:
    input_config = AseSinglepointInput.from_file(calcpath / "input_dft.json")
    input_config.validate(calcpath)

    atoms = ase.io.read(calcpath / input_config.DATABASE)
    atoms.calc = build_calculator(
        input_config.calculator, input_config.calculator_kwargs
    )
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)

    ase.io.write(calcpath / ("dft_" + input_config.DATABASE), atoms)
    (calcpath / "is_finished.info").write_text(
        "I'm a finished single-point run"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
