"""Sampler payload: ASE Langevin MD, run *on the target* by a RemoteTask.

Contract (see ``DirectSingleExecutor``): invoked as ``python md_sample.py
<input.json>`` with the working directory set to the task's calcdir. Reads the
input JSON, runs a short NVT trajectory driven by the *current* model
(bootstrap EMT on the first iteration, the fitted Morse afterwards), writes the
sampled frames as ``sampled.extxyz`` in the cwd, and drops the
``is_finished.info`` success marker.

Pure ASE + numpy -- no ML stack, no external binary -- so it runs anywhere ASE
is installed. This is the deliberately-external sampling step cast as a
``RemoteTask``: swap this script for LAMMPS/torch_sim/... without touching the
rest of the loop.
"""

import json
import sys
from pathlib import Path

import ase.io
import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.neighborlist import neighbor_list


class Morse(Calculator):
    """Minimal pairwise Morse potential with a hard cutoff (PBC-correct).

    ``E = sum_{i<j} D (e^2 - 2 e)`` with ``e = exp(-alpha (r - r0))``. Uses
    ASE's neighbour list so periodic images are handled correctly -- the same
    calculator the trainer fits, kept identical on both sides of the loop.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, D, alpha, r0, rc=6.0, **kwargs):
        super().__init__(**kwargs)
        self.D, self.alpha, self.r0, self.rc = D, alpha, r0, rc

    def calculate(
        self, atoms=None, properties=("energy",), system_changes=all_changes
    ):
        super().calculate(atoms, properties, system_changes)
        i, j, d, Dvec = neighbor_list("ijdD", atoms, self.rc)
        e = np.exp(-self.alpha * (d - self.r0))
        energy = 0.5 * np.sum(self.D * (e**2 - 2.0 * e))
        # du/dd = 2 D alpha (e - e^2); force on i = -(du/dd) * (-Dvec/d).
        du_dd = 2.0 * self.D * self.alpha * (e - e**2)
        fpair = (du_dd / d)[:, None] * Dvec
        forces = np.zeros((len(atoms), 3))
        np.add.at(forces, i, fpair)
        self.results["energy"] = energy
        self.results["forces"] = forces


def build_calculator(model: dict):
    """Instantiate the sampling calculator from a small model dict."""
    kind = model.get("type", "morse").lower()
    if kind == "emt":
        from ase.calculators.emt import EMT

        return EMT()
    if kind == "morse":
        return Morse(
            D=model["D"],
            alpha=model["alpha"],
            r0=model["r0"],
            rc=model.get("rc", 6.0),
        )
    raise ValueError(f"unknown model type {kind!r}")


def main(input_path: Path) -> None:
    cfg = json.loads(Path(input_path).read_text())

    s = cfg["structure"]
    atoms = Atoms(
        numbers=s["numbers"],
        positions=s["positions"],
        cell=s["cell"],
        pbc=s["pbc"],
    )
    rng = np.random.default_rng(cfg.get("seed", 0))
    atoms.rattle(stdev=cfg.get("rattle", 0.0), seed=int(rng.integers(1 << 30)))
    atoms.calc = build_calculator(cfg["model"])

    MaxwellBoltzmannDistribution(
        atoms, temperature_K=cfg["temperature_K"], rng=rng
    )
    dyn = Langevin(
        atoms,
        timestep=cfg["timestep_fs"] * units.fs,
        temperature_K=cfg["temperature_K"],
        friction=cfg.get("friction", 0.01),
        rng=rng,
    )

    frames = []
    every = cfg["sample_every"]

    def grab():
        frames.append(atoms.copy())

    dyn.attach(grab, interval=every)
    dyn.run(cfg["n_steps"])

    ase.io.write(cfg.get("output", "sampled.extxyz"), frames)
    Path("is_finished.info").write_text("sampling finished")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
