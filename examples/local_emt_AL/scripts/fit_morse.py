"""Trainer payload: fit a pairwise Morse potential to the labelled data.

Contract (see ``DirectSingleExecutor``): invoked as ``python fit_morse.py
<input.json>`` with the working directory set to the task's calcdir. Reads the
accumulated labelled configurations, fits the three Morse parameters
(``D, alpha, r0``) to the reference energies and forces with
``scipy.optimize``, and writes the model as ``model.json`` in the cwd plus the
``is_finished.info`` marker.

This is the "simplest trainable model" stand-in for a heavy MLFF trainer: no
GPU, no ML framework, fits in milliseconds -- yet it exercises the exact same
``ModelTrainer`` -> ``RemoteTask`` path a real trainer uses.
"""

import json
import sys
from pathlib import Path

import ase.io
import numpy as np
from ase.neighborlist import neighbor_list
from scipy.optimize import least_squares


def _energy_forces(atoms, D, alpha, r0, rc):
    """Morse energy + forces (matches ``md_sample.Morse``)."""
    i, j, d, Dvec = neighbor_list("ijdD", atoms, rc)
    e = np.exp(-alpha * (d - r0))
    energy = 0.5 * np.sum(D * (e**2 - 2.0 * e))
    du_dd = 2.0 * D * alpha * (e - e**2)
    fpair = (du_dd / d)[:, None] * Dvec
    forces = np.zeros((len(atoms), 3))
    np.add.at(forces, i, fpair)
    return energy, forces


def main(input_path: Path) -> None:
    cfg = json.loads(Path(input_path).read_text())
    rc = cfg.get("rc", 6.0)
    ekey = cfg.get("energy_key", "REF_energy")
    fkey = cfg.get("forces_key", "REF_forces")
    fw = cfg.get("force_weight", 1.0)

    configs = []
    for rel in cfg["data"]:
        configs.extend(ase.io.read(rel, index=":"))
    if not configs:
        raise RuntimeError("no training configurations found")

    def residuals(params):
        D, alpha, r0 = params
        res = []
        for at in configs:
            e_pred, f_pred = _energy_forces(at, D, alpha, r0, rc)
            n = len(at)
            res.append((e_pred - at.info[ekey]) / n)  # per-atom energy
            res.extend((fw * (f_pred - at.arrays[fkey]) / np.sqrt(n)).ravel())
        return np.asarray(res)

    x0 = [cfg["init"]["D"], cfg["init"]["alpha"], cfg["init"]["r0"]]
    sol = least_squares(
        residuals,
        x0=x0,
        bounds=([1e-3, 0.3, 1.5], [10.0, 5.0, 3.5]),
        max_nfev=cfg.get("max_nfev", 200),
    )

    model = {
        "type": "morse",
        "D": float(sol.x[0]),
        "alpha": float(sol.x[1]),
        "r0": float(sol.x[2]),
        "rc": rc,
        "n_train": len(configs),
        "rms_residual": float(np.sqrt(np.mean(sol.fun**2))),
    }
    out = Path(cfg.get("output", "model.json"))
    out.write_text(json.dumps(model, indent=2))
    Path("is_finished.info").write_text("training finished")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
