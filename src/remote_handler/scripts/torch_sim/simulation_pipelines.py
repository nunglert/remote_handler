import json
import os
import shutil
from pathlib import Path
from time import time
from typing import Any, Optional, Union

import ase.io
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch_sim as ts
from ase.build import bulk
from ase.constraints import FixAtoms as ASEFixAtoms
from mace.calculators import MACECalculator
from mace.calculators.foundations_models import mace_mp
from torch_sim.constraints import FixAtoms
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.models.mace import MaceModel
from torch_sim.trajectory import TorchSimTrajectory, TrajectoryReporter
from torch_sim.units import MetalUnits

# We need the following import to patch bugs in torch_sim. Even though it looks
# like it would note be used:
from remote_handler.scripts.torch_sim import (
    parallel_tempering,
    torchsim_patches,
)
from remote_handler.scripts.torch_sim.create_interface import (
    build_sic_ti_interface,
)

SerializedTorch = Any


def interface_pt_simulation(
    at,
    calcpath: Path,
    target_temperature: float,
    temperatures: list[float],
    mace: SerializedTorch = mace_mp(model="small", return_raw_model=True),
    dt: float = 0.001,
    gamma: Optional[float] = None,
    alpha: Optional[float] = None,
    cell_alpha: Optional[float] = None,
    n_steps_npt: int = 0,
    n_steps_pt: int = 100,
    seed: int = 123,
    device: str = "cuda",
    dtype=torch.float32,
):
    """
    mace (mace.MACECalculator): A MACE calculator, e.g. for a foundation
        model that is used to create the torch_sim model via ts.MaceModel
    temperatures (torch.Tensor): Target temperature in K with shape
        [n_systems]
    dt (torch.Tensor): Integration timestep, either scalar or shape
        [n_systems]
    gamma: Friction coefficient for Langevin thermostat,
        either scalar or with shape [n_systems]. Defaults to 1/(100*dt).
    alpha (torch.Tensor, optional): Friction coefficient for particle
        Langevin thermostat, either scalar or shape [n_systems]. Defaults
        to 1/(100*dt).
    cell_alpha (torch.Tensor, optional): Friction coefficient for cell
        Langevin thermostat, either scalar or shape [n_systems]. Defaults
        to same as alpha.
    seed (int, optional): Random seed for reproducibility. Defaults to
        None.
    n_steps_npt: Number of NPT steps to equilibrate the cell volume
    n_steps_pt: Number of production parallel tempering steps in an NVT
        ensemble
    """
    frozen_mask_np = None

    if getattr(at, "constraints", None) and ASEFixAtoms is not None:
        for c in at.constraints:
            if isinstance(c, ASEFixAtoms):
                # ASE FixAtoms stores either a mask or indices (depends how it was created)
                if getattr(c, "index", None) is not None:
                    frozen_mask_np = np.zeros(len(at), dtype=bool)
                    frozen_mask_np[np.asarray(c.index, dtype=int)] = True
                elif getattr(c, "mask", None) is not None:
                    frozen_mask_np = np.asarray(c.mask, dtype=bool).copy()
                break

    # Convert to torch mask
    device = torch.device(device)
    frozen_mask = torch.as_tensor(
        frozen_mask_np, device=device, dtype=torch.bool
    )

    n_batch = len(temperatures)

    assert frozen_mask.ndim == 1
    assert frozen_mask.shape[0] == len(at)
    assert (
        frozen_mask.any()
    ), "No atoms were marked frozen — check constraint logic."
    assert (
        ~frozen_mask
    ).any(), "All atoms were frozen — check constraint logic."

    model = MaceModel(
        model=mace,
        device=device,
        dtype=dtype,
        compute_forces=True,
        compute_stress=True,
    )

    constraint = FixAtoms(atom_mask=frozen_mask)

    # temperatures = torch.arange(1, n_batch+1, device=device) * 50  # K
    temperatures = torch.tensor(temperatures, device=device, dtype=dtype)
    # kT = temperatures * MetalUnits.temperature

    prop_calculators = {
        2: {"potential_energy": lambda state: state.energy},
        2: {
            "kinetic_energy": lambda state: ts.calc_kinetic_energy(
                momenta=state.momenta, masses=state.masses
            )
        },
    }
    # trajectory_file = calcpath / "trajectory.h5md"
    reporter_args = dict(
        filenames=calcpath / f"npt_equil.h5md",
        state_frequency=1,
        prop_calculators=prop_calculators,
    )
    reporter = TrajectoryReporter(**reporter_args)

    # ------------------------------------------------------------------
    # Start with an NPT equilibration (just isotropic to relax stress)
    # This is only done once for the "founder" structure at the target
    # temperature
    # ------------------------------------------------------------------
    if n_steps_npt > 0:
        state = ts.io.atoms_to_state(at, device=device, dtype=dtype)
        state = ts.npt_langevin_init(
            state=state,
            model=model,
            kT=target_temperature * MetalUnits.temperature,
            seed=seed,
            dt=dt,
            # alpha=alpha,
            # cell_alpha=cell_alpha,
        )

        state = ts.integrate(
            system=state,
            model=model,
            integrator=ts.Integrator.npt_langevin,
            n_steps=n_steps_npt,
            temperature=target_temperature,
            external_pressure=0,
            timestep=dt,
            trajectory_reporter=reporter,
        )

        # use volume equilibrated structure for replicas
        at_equil = state.to_atoms()
    else:
        at_equil = at

    state = ts.io.atoms_to_state(
        at_equil * n_batch, device=device, dtype=dtype
    )

    # ------------------------------------------------------------------
    # Run NVT simulation
    # ------------------------------------------------------------------
    model = MaceModel(
        model=mace,
        device=device,
        dtype=dtype,
        compute_forces=True,
        compute_stress=False,
    )
    state.constraints = [constraint]

    # state = ts.nvt_langevin_init(
    #     state=state,
    #     model=model,
    #     kT=kT,
    #     seed=seed,
    # )
    # state.constraints = [constraint]

    # state, kT, swap_log = parallel_tempering.parallel_tempering_nvt(
    #     state,
    #     model,
    #     kT,
    #     dt,
    #     gamma,
    #     n_total_steps=n_steps_pt,
    #     md_steps_per_exchange=5,
    #     reporter_args=reporter_args,
    #     start_step=n_steps_npt
    # )

    reporter_args = dict(
        filenames=[calcpath / f"trajectory_{i}.h5md" for i in range(n_batch)],
        state_frequency=1,
        prop_calculators=prop_calculators,
    )

    state, kT, P, swap_log = parallel_tempering.parallel_tempering(
        state,
        model,
        "nvt",
        temperatures,
        dt,
        n_total_steps=n_steps_pt,
        gamma=gamma,
        external_pressure=0,
        swap_temperature=True,
        swap_pressure=False,
        md_steps_per_exchange=5,
        reporter_args=reporter_args,
        start_step=n_steps_npt,
    )

    reporter.close()

    with Path(calcpath / "pt_swaps.json").open("w") as f:
        json.dump(swap_log, f)

    return state


def pt_simulation_npt(
    at,
    calcpath: Path,
    temperatures: list[float],
    mace: SerializedTorch = mace_mp(model="small", return_raw_model=True),
    dt: float = 0.001,
    gamma: Optional[float] = None,
    alpha: Optional[float] = None,
    cell_alpha: Optional[float] = None,
    n_steps_npt: int = 0,
    n_steps_pt: int = 100,
    seed: int = 123,
    device: str = "cuda",
    dtype=torch.float32,
):
    """
    mace (mace.MACECalculator): A MACE calculator, e.g. for a foundation
        model that is used to create the torch_sim model via ts.MaceModel
    temperatures (torch.Tensor): Target temperature in K with shape
        [n_systems]
    dt (torch.Tensor): Integration timestep, either scalar or shape
        [n_systems]
    gamma: Friction coefficient for Langevin thermostat,
        either scalar or with shape [n_systems]. Defaults to 1/(100*dt).
    alpha (torch.Tensor, optional): Friction coefficient for particle
        Langevin thermostat, either scalar or shape [n_systems]. Defaults
        to 1/(100*dt).
    cell_alpha (torch.Tensor, optional): Friction coefficient for cell
        Langevin thermostat, either scalar or shape [n_systems]. Defaults
        to same as alpha.
    seed (int, optional): Random seed for reproducibility. Defaults to
        None.
    n_steps_npt: Number of NPT steps to equilibrate the cell volume
    n_steps_pt: Number of production parallel tempering steps in an NVT
        ensemble
    """
    device = torch.device(device)

    n_batch = len(temperatures)
    state = ts.io.atoms_to_state([at] * n_batch, device=device, dtype=dtype)

    temperatures = torch.tensor(temperatures, device=device, dtype=dtype)

    prop_calculators = {
        2: {"potential_energy": lambda state: state.energy},
        2: {
            "kinetic_energy": lambda state: ts.calc_kinetic_energy(
                momenta=state.momenta, masses=state.masses
            )
        },
    }
    # trajectory_file = calcpath / "trajectory.h5md"
    reporter_args = dict(
        filenames=[calcpath / f"trajectory_{i}.h5md" for i in range(n_batch)],
        state_frequency=1,
        prop_calculators=prop_calculators,
    )

    # ------------------------------------------------------------------
    # Run NVT simulation
    # ------------------------------------------------------------------
    model = MaceModel(
        model=mace,
        device=device,
        dtype=dtype,
        compute_forces=True,
        compute_stress=True,
    )

    state, kT, P, swap_log = parallel_tempering.parallel_tempering(
        state,
        model,
        "npt",
        temperatures,
        dt,
        n_total_steps=n_steps_pt,
        gamma=gamma,
        alpha=alpha,
        cell_alpha=cell_alpha,
        external_pressure=0,
        swap_temperature=True,
        swap_pressure=False,
        md_steps_per_exchange=5,
        reporter_args=reporter_args,
        start_step=n_steps_npt,
    )

    with Path(calcpath / "pt_swaps.json").open("w") as f:
        json.dump(swap_log, f)

    return state
