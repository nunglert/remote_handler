import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms

from remote_handler.scripts.mace.mace_utils import write_with_keys


def test_write_with_keys_preserves_forces_on_constrained_atoms(tmp_path):
    atoms = Atoms("Cu2", positions=[[0, 0, 0], [2.55, 0, 0]])
    atoms.set_constraint(FixAtoms(indices=[0]))
    forces = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    atoms.calc = SinglePointCalculator(atoms, energy=-1.0, forces=forces)
    output = tmp_path / "labelled.extxyz"

    write_with_keys(output, [atoms])

    assert np.allclose(atoms.arrays["REF_forces"], forces)
