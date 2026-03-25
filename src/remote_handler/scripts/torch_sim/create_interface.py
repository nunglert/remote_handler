"""
Build a commensurate Ti(0001) || 3C-SiC(111) interface (ASE).

What you get:
- Si-terminated or C-terminated SiC(111) slab
- Ti(0001) slab strained in-plane to match SiC
- stacked interface with a chosen lateral registry shift
- optional vacuum
- FixAtoms constraints on bottom N layers of each slab

Requires: ase, numpy
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import add_vacuum, bulk, make_supercell, surface
from ase.constraints import FixAtoms

# ----------------------------
# Helpers
# ----------------------------


def _unique_layers(z: np.ndarray, tol: float = 0.25) -> list[float]:
    """Cluster z-values into approximate layers."""
    z_sorted = np.sort(z)
    layers = [float(z_sorted[0])]
    for val in z_sorted[1:]:
        if abs(val - layers[-1]) > tol:
            layers.append(float(val))
    return layers


def _freeze_bottom_n_layers(
    atoms: Atoms, n_layers: int, *, tol: float = 0.25
) -> np.ndarray:
    """Return boolean mask of atoms belonging to bottom n_layers (by z-layer clustering)."""
    z = atoms.positions[:, 2]
    layers = _unique_layers(z, tol=tol)
    if n_layers <= 0:
        return np.zeros(len(atoms), dtype=bool)
    if n_layers >= len(layers):
        return np.ones(len(atoms), dtype=bool)
    z_cut = layers[n_layers - 1] + tol  # include that layer
    return z <= z_cut


def _shift_inplane(atoms: Atoms, frac_shift: tuple[float, float]) -> None:
    """Shift atoms laterally by fractional coords of the (a1,a2) lattice vectors."""
    a1 = atoms.cell[0].copy()
    a2 = atoms.cell[1].copy()
    dx = frac_shift[0] * a1 + frac_shift[1] * a2
    atoms.positions += dx  # shifts periodic in xy


def _match_inplane_cell(
    reference: Atoms, target: Atoms, *, scale_atoms: bool = True
) -> None:
    """
    Force target's in-plane lattice vectors to equal reference's in-plane vectors
    by applying an affine map in the xy plane.
    """
    A_ref = np.array([reference.cell[0][:2], reference.cell[1][:2]]).T  # 2x2
    A_tgt = np.array([target.cell[0][:2], target.cell[1][:2]]).T  # 2x2

    # Solve for M such that A_tgt @ M = A_ref  ->  M = A_tgt^{-1} A_ref
    M = np.linalg.solve(A_tgt, A_ref)  # 2x2

    # Apply M to target positions (xy) and cell (a1,a2) while keeping z as-is
    pos = target.positions.copy()
    xy = pos[:, :2] @ M
    pos[:, :2] = xy

    cell = target.cell.array.copy()
    cell[:2, :2] = (cell[:2, :2].T @ M).T  # apply to a1,a2 xy components

    # Overwrite in-plane cell vectors exactly to reference (robust)
    cell[0, :2] = reference.cell[0, :2]
    cell[1, :2] = reference.cell[1, :2]

    target.set_cell(cell, scale_atoms=False)
    if scale_atoms:
        target.positions[:] = pos


def _stack_slabs(bottom: Atoms, top: Atoms, gap: float) -> Atoms:
    """Stack top slab above bottom slab with a vacuum-free cell (vacuum can be added later)."""
    # Ensure same in-plane cell
    out = bottom.copy()

    # Place bottom so its minimum z is at 0
    out.positions[:, 2] -= out.positions[:, 2].min()

    top2 = top.copy()
    top2.positions[:, 2] -= top2.positions[:, 2].min()

    z_top_bottom = out.positions[:, 2].max()
    top2.positions[:, 2] += z_top_bottom + gap

    # Combine
    out += top2

    # Set cell: use bottom in-plane, set z just enough to contain atoms
    cell = out.cell.array.copy()
    zmax = out.positions[:, 2].max()
    cell[2, :] = [0.0, 0.0, zmax + 1.0]  # small padding; add_vacuum can extend
    out.set_cell(cell, scale_atoms=False)
    out.set_pbc((True, True, True))
    return out


# ----------------------------
# Builders
# ----------------------------


def build_sic_111_slab(
    *,
    a: float = 4.36,
    layers: int = 10,
    repeat_xy: tuple[int, int] = (2, 2),
    termination: str = "Si",
) -> Atoms:
    """
    Build 3C-SiC (zincblende) (111) slab.

    termination: "Si" or "C" selects which species is on the *top* surface.
    """
    # 3C-SiC zincblende primitive
    sic = bulk("SiC", "zincblende", a=a)

    # (111) surface; ASE 'surface' makes a slab with z as surface normal
    slab = surface(sic, (1, 1, 1), layers, vacuum=0.0)

    # Repeat in-plane
    slab = slab.repeat((repeat_xy[0], repeat_xy[1], 1))

    # Ensure top termination
    # Heuristic: if topmost atom is not desired element, flip slab in z (swap top/bottom).
    z = slab.positions[:, 2]
    top_idx = int(np.argmax(z))
    top_sym = slab[top_idx].symbol
    want = termination.strip().capitalize()
    if want not in ("Si", "C"):
        raise ValueError("termination must be 'Si' or 'C'")
    if top_sym != want:
        # flip along z: z -> zmax - z
        zmax = z.max()
        slab.positions[:, 2] = zmax - slab.positions[:, 2]

    # Put min z at 0
    slab.positions[:, 2] -= slab.positions[:, 2].min()

    # Set PBC in xy and a reasonable cell z
    cell = slab.cell.array.copy()
    zmax = slab.positions[:, 2].max()
    cell[2, :] = [0.0, 0.0, zmax + 1.0]
    slab.set_cell(cell, scale_atoms=False)
    slab.set_pbc((True, True, True))
    return slab


def build_ti_0001_slab(
    *,
    a: float = 2.95,
    c: float = 4.68,
    layers: int = 10,
    repeat_xy: tuple[int, int] = (2, 2),
) -> Atoms:
    """Build hcp-Ti (0001) slab."""
    ti = bulk("Ti", "hcp", a=a, c=c)
    slab = surface(ti, (0, 0, 1), layers, vacuum=0.0)
    slab = slab.repeat((repeat_xy[0], repeat_xy[1], 1))

    slab.positions[:, 2] -= slab.positions[:, 2].min()

    cell = slab.cell.array.copy()
    zmax = slab.positions[:, 2].max()
    cell[2, :] = [0.0, 0.0, zmax + 1.0]
    slab.set_cell(cell, scale_atoms=False)
    slab.set_pbc((True, True, True))
    return slab


def build_sic_ti_interface(
    *,
    sic_layers: int = 12,
    ti_layers: int = 12,
    sic_repeat_xy: tuple[int, int] = (2, 2),
    ti_repeat_xy: tuple[int, int] = (2, 2),
    termination: str = "Si",
    gap: float = 2.2,
    vacuum: float = 12.0,
    registry: tuple[float, float] = (0.0, 0.0),
    freeze_sic_bottom_layers: int = 2,
    freeze_ti_bottom_layers: int = 2,
) -> Atoms:
    """
    Create Ti(0001) || SiC(111) interface.

    registry: fractional shift (s,t) applied to the *Ti slab* in terms of the in-plane
              lattice vectors of the matched cell.
              Useful shifts for hex nets: (0,0), (1/3,2/3), (2/3,1/3).
    """
    sic = build_sic_111_slab(
        layers=sic_layers, repeat_xy=sic_repeat_xy, termination=termination
    )
    ti = build_ti_0001_slab(layers=ti_layers, repeat_xy=ti_repeat_xy)

    # Match Ti in-plane cell to SiC in-plane cell (strain Ti, keep SiC fixed)
    _match_inplane_cell(sic, ti, scale_atoms=True)

    # Use SiC cell for both (exact)
    ti.set_cell(sic.cell, scale_atoms=False)
    ti.set_pbc((True, True, True))

    # Apply registry shift to Ti
    _shift_inplane(ti, registry)

    # Stack: SiC at bottom, Ti on top
    interface = _stack_slabs(sic, ti, gap=gap)

    # Add vacuum on top (keeps interface intact)
    if vacuum and vacuum > 0:
        add_vacuum(interface, vacuum)

    # Freeze bottom layers of each slab:
    # Identify atoms from each slab by z position relative to the interface.
    # We stored SiC at bottom and Ti on top; a robust separator is the mid-gap plane.
    z = interface.positions[:, 2]
    z_min, z_max = z.min(), z.max()
    z_mid = 0.5 * (z_min + z_max)

    sic_mask = z < z_mid
    ti_mask = ~sic_mask

    freeze_mask = np.zeros(len(interface), dtype=bool)

    # Freeze bottom layers of SiC (global bottom of the whole structure)
    sic_atoms = interface[sic_mask]
    sic_freeze_local = _freeze_bottom_n_layers(
        sic_atoms, freeze_sic_bottom_layers, tol=0.25
    )
    freeze_mask[np.where(sic_mask)[0][sic_freeze_local]] = True

    # Freeze bottom layers of Ti slab (i.e., those closest to the interface within Ti slab):
    # For Ti, “bottom” means *minimum z within Ti atoms* (interface side).
    ti_atoms = interface[ti_mask]
    ti_freeze_local = _freeze_bottom_n_layers(
        ti_atoms, freeze_ti_bottom_layers, tol=0.25
    )
    freeze_mask[np.where(ti_mask)[0][ti_freeze_local]] = True

    interface.set_constraint(FixAtoms(mask=freeze_mask))
    interface.wrap()
    return interface


def build_registry_scan(
    *,
    terminations: tuple[str, ...] = ("Si", "C"),
    registries: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (1 / 3, 2 / 3),
        (2 / 3, 1 / 3),
    ),
    **kwargs,
) -> list[Atoms]:
    """Convenience: build multiple terminations and registries."""
    out: list[Atoms] = []
    for term in terminations:
        for reg in registries:
            atoms = build_sic_ti_interface(
                termination=term, registry=reg, **kwargs
            )
            atoms.info["termination"] = term
            atoms.info["registry"] = reg
            out.append(atoms)
    return out


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Build a small scan of candidates
    candidates = build_registry_scan(
        sic_layers=12,
        ti_layers=12,
        sic_repeat_xy=(2, 2),
        ti_repeat_xy=(2, 2),
        gap=2.2,
        vacuum=12.0,
        freeze_sic_bottom_layers=2,
        freeze_ti_bottom_layers=2,
    )

    # Write out for inspection (ASE can write .xyz/.traj; you can also use OVITO)
    from ase.io import write

    for i, at in enumerate(candidates):
        term = at.info["termination"]
        reg = at.info["registry"]
        write(
            f"SiC_Ti_interface_{i:02d}_{term}_reg{reg[0]:.2f}_{reg[1]:.2f}.xyz",
            at,
        )
