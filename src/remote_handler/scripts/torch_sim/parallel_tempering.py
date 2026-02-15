import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_sim as ts
from torch_sim.trajectory import TrajectoryReporter
from torch_sim.units import MetalUnits


def _get_volume_per_system(state) -> torch.Tensor:
    """
    Returns volume per system, shape [n_systems].
    Tries state.cell first; falls back to state.current_cell if present.
    """
    return torch.linalg.det(state.cell)


def _rescale_atom_momenta_in_place(state, sys: int, scale: torch.Tensor):
    mask = state.system_idx == sys
    p = state.momenta.clone()
    p[mask] *= scale
    if hasattr(state, "set_constrained_momenta"):
        state.set_constrained_momenta(p)
    else:
        state.momenta = p


def _rescale_barostat_dofs_in_place(state, sys: int, scale: torch.Tensor):
    """
    Best-effort rescaling of barostat DOFs for Nose–Hoover style NPT states.
    Safe no-op for states that don't have these attributes.
    """
    # Nose–Hoover NPT in your code uses: cell_momentum [n_systems,1] and chain momenta
    if hasattr(state, "cell_momentum") and state.cell_momentum is not None:
        cm = state.cell_momentum.clone()
        # cm shape could be [n_systems] or [n_systems,1]
        cm[sys] *= scale
        state.cell_momentum = cm

    # If you ever store a full cell-velocity tensor (Langevin NPT), you can rescale it too
    if hasattr(state, "cell_velocities") and state.cell_velocities is not None:
        cv = state.cell_velocities.clone()
        cv[sys] *= scale
        state.cell_velocities = cv


def attempt_replica_exchange(
    state,
    *,
    i: int,
    j: int,
    rng: torch.Generator,
    kT: torch.Tensor,  # [n_systems]
    external_pressure: torch.Tensor
    | None = None,  # scalar or [n_systems] or [n_systems,3,3]
    swap_temperature: bool = True,
    swap_pressure: bool = False,
    rescale_momenta: bool = True,
    rescale_barostat: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, bool]:
    """
    General replica exchange move for NVT / NPT.

    Supports:
      - temperature swap (kT labels) for NVT and NPT
      - pressure swap (P labels) for NPT (optional)
      - both simultaneously

    Acceptance uses:
      S(x; beta, P) = beta * (U(x) + P*V(x))   (isotropic P)
    so log r = -[S(x_i; beta_j,P_j)+S(x_j; beta_i,P_i) - S(x_i; beta_i,P_i)-S(x_j; beta_j,P_j)].

    Returns updated (kT, external_pressure, accepted).
    """
    U = state.energy
    if U.ndim != 1:
        U = U.view(-1)

    n_sys = int(U.shape[0])

    # Handle pressure parameterization
    P = external_pressure
    if P is None:
        # NVT case
        P_i = P_j = None
    else:
        if not isinstance(P, torch.Tensor):
            P = torch.as_tensor(P, device=kT.device, dtype=kT.dtype)
        # Accept scalar or [n_systems]; (ignore tensor anisotropy here)
        if P.ndim == 0:
            P = P.expand(n_sys)
        elif P.ndim == 3:
            # if [n_systems,3,3], interpret as isotropic via trace/3
            P = torch.einsum("bii->b", P) / 3.0
        P_i = P[i]
        P_j = P[j]

    V = _get_volume_per_system(state) if P is not None else None

    beta = 1.0 / kT
    beta_i, beta_j = beta[i], beta[j]

    # current labels
    kTi, kTj = kT[i], kT[j]

    # proposed labels
    kT_i_new, kT_j_new = (kTj, kTi) if swap_temperature else (kTi, kTj)
    if P is not None:
        P_i_new, P_j_new = (P_j, P_i) if swap_pressure else (P_i, P_j)

    def S(Ua, Va, beta_a, Pa):
        if Pa is None:
            return beta_a * Ua
        return beta_a * (Ua + Pa * Va)

    # log r = -[S_i(new) + S_j(new) - S_i(old) - S_j(old)]
    if P is None:
        log_r = -(
            S(U[i], None, beta_j if swap_temperature else beta_i, None)
            + S(U[j], None, beta_i if swap_temperature else beta_j, None)
            - S(U[i], None, beta_i, None)
            - S(U[j], None, beta_j, None)
        )
    else:
        # old actions
        S_i_old = S(U[i], V[i], beta_i, P_i)
        S_j_old = S(U[j], V[j], beta_j, P_j)
        # new actions with swapped labels
        beta_i_new = 1.0 / kT_i_new
        beta_j_new = 1.0 / kT_j_new
        S_i_new = S(U[i], V[i], beta_i_new, P_i_new)
        S_j_new = S(U[j], V[j], beta_j_new, P_j_new)
        log_r = -(S_i_new + S_j_new - S_i_old - S_j_old)

    u = torch.rand((), generator=rng, device=kT.device, dtype=kT.dtype)
    accept = torch.log(u) < log_r
    if not bool(accept):
        return kT, external_pressure, False

    # apply swaps
    kT_new = kT.clone()
    if swap_temperature:
        kT_new[i], kT_new[j] = kTj, kTi

    P_new = external_pressure
    if P is not None and swap_pressure:
        P_tensor = P.clone()
        P_tensor[i], P_tensor[j] = P_tensor[j], P_tensor[i]
        P_new = P_tensor  # isotropic [n_systems]

    # rescale kinetic DOFs when temperature label changes
    if rescale_momenta and swap_temperature:
        si = torch.sqrt(kT_new[i] / kTi)
        sj = torch.sqrt(kT_new[j] / kTj)
        _rescale_atom_momenta_in_place(state, i, si)
        _rescale_atom_momenta_in_place(state, j, sj)
        if rescale_barostat:
            _rescale_barostat_dofs_in_place(state, i, si)
            _rescale_barostat_dofs_in_place(state, j, sj)

    return kT_new, P_new, True


import torch


def parallel_tempering(
    state,
    model,
    ensemble: str,
    temperatures: torch.Tensor,
    dt: float,
    n_total_steps: int,
    md_steps_per_exchange: int = 100,
    seed: int = 123,
    alpha: float | None = None,
    cell_alpha: float | None = None,
    gamma: float | None = None,
    external_pressure: torch.Tensor | float | None = None,
    swap_temperature: bool = True,
    swap_pressure: bool = False,
    rescale_momenta_on_swap: bool = True,
    rescale_barostat_on_swap: bool = True,
    start_step: int = 0,
    reporter_args=None,
):
    """
    Parallel tempering / replica exchange for batched simulations.

    Uses attempt_replica_exchange(...) (swap temperatures and/or pressures as labels)
    and supports:
      - NVT: temperature swaps
      - NPT: temperature swaps and/or pressure swaps

    Notes
    -----
    - We do NOT shuffle atoms between replicas. We swap labels (kT and/or P).
    - To avoid reporter / HDF5 handle issues, this function does not pass any
      trajectory_reporter into ts.integrate. Do reporting outside if desired.
    """
    if ensemble not in {"nvt", "npt"}:
        raise ValueError("ensemble must be 'nvt' or 'npt'")
    if ensemble == "nvt" and swap_pressure:
        raise ValueError("swap_pressure=True only supported for NPT")
    if ensemble == "nvt" and not swap_temperature:
        raise ValueError("NVT requires swap_temperature=True to do PT")

    device = temperatures.device
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    n_sys = int(state.n_systems)
    assert (
        temperatures.shape[0] == n_sys
    ), f"Expected temperatures shape [{n_sys}], got {tuple(temperatures.shape)}"

    # Labels (kT in energy units)
    kT = temperatures * MetalUnits.temperature

    # Pressure labels (only for NPT); allow scalar or per-replica
    P = external_pressure

    # --- propagate MD block at current labels
    # Prefer single integrate call per block to avoid re-init side effects.
    integrate_kwargs = dict(
        system=state,
        model=model,
        timestep=dt,
    )

    integrate_kwargs["temperature"] = temperatures
    if ensemble == "nvt":
        integrate_kwargs["gamma"] = gamma
        integrate_kwargs["integrator"] = ts.Integrator.nvt_langevin
    else:
        integrate_kwargs["external_pressure"] = P
        state.alpha = alpha
        state.cell_alpha = cell_alpha
        integrate_kwargs["integrator"] = ts.Integrator.npt_langevin

    stats = {
        "attempted": 0,
        "accepted": 0,
        "accept_rate": 0.0,
        "attempted_pairs": [],  # (step, i, j, accepted)
    }

    steps_done = 0
    exchange_round = 0

    while steps_done < n_total_steps:
        block = min(md_steps_per_exchange, n_total_steps - steps_done)

        integrate_kwargs["n_steps"] = block

        if reporter_args is not None:
            integrate_kwargs["trajectory_reporter"] = TrajectoryReporter(
                **reporter_args
            )
        else:
            integrate_kwargs["trajectory_reporter"] = None

        state = ts.integrate(**integrate_kwargs)
        steps_done += block

        # --- choose non-overlapping neighbor pairs
        if (exchange_round % 2) == 0:
            pairs = [(i, i + 1) for i in range(0, n_sys - 1, 2)]
        else:
            pairs = [(i, i + 1) for i in range(1, n_sys - 1, 2)]

        # --- attempt exchanges
        for i, j in pairs:
            stats["attempted"] += 1
            kT, P, acc = attempt_replica_exchange(
                state,
                i=i,
                j=j,
                rng=rng,
                kT=kT,
                external_pressure=(P if ensemble == "npt" else None),
                swap_temperature=swap_temperature,
                swap_pressure=(swap_pressure if ensemble == "npt" else False),
                rescale_momenta=rescale_momenta_on_swap,
                rescale_barostat=rescale_barostat_on_swap,
            )
            stats["accepted"] += int(acc)
            stats["attempted_pairs"].append(
                (start_step + steps_done, i, j, bool(acc))
            )

        # --- update temperature labels in Kelvin for next MD block
        temperatures = kT / MetalUnits.temperature

        exchange_round += 1

    stats["accept_rate"] = stats["accepted"] / max(1, stats["attempted"])

    # swap_log in your previous format
    swap_data = [
        {"step": int(step), "i": int(i), "j": int(j), "accepted": bool(acc)}
        for (step, i, j, acc) in stats["attempted_pairs"]
    ]
    swap_log = {
        "data": swap_data,
        "info": {
            "ensemble": ensemble,
            "swap_temperature": bool(swap_temperature),
            "swap_pressure": bool(swap_pressure)
            if ensemble == "npt"
            else False,
            "accept_rate": float(stats["accept_rate"]),
            "attempted": int(stats["attempted"]),
            "accepted": int(stats["accepted"]),
            "temperatures": temperatures.detach().cpu().tolist(),
            "external_pressure": (
                P.detach().cpu().tolist()
                if (ensemble == "npt" and isinstance(P, torch.Tensor))
                else None
            ),
        },
    }

    return state, temperatures, P, swap_log


def _attempt_swap_temperatures(
    state,
    kT: torch.Tensor,  # shape [n_systems] in energy units
    i: int,
    j: int,
    rng: torch.Generator,
    rescale_momenta: bool = True,
) -> tuple[torch.Tensor, bool]:
    """
    Replica exchange move between replicas i and j by swapping their temperatures.
    We keep configurations where they are (no atom shuffling); instead we swap kT labels.

    Acceptance (config swap equivalent):
        acc = min(1, exp((β_i - β_j) * (U_i - U_j))), β=1/kT
    where U is the potential energy (state.energy).

    If rescale_momenta=True, we also rescale momenta of each replica by
        p -> p * sqrt(kT_new / kT_old)
    which keeps the Maxwell distribution consistent after the temperature swap.
    """
    # Potential energies per replica (assumed shape [n_systems])
    U = state.energy
    if U.ndim != 1:
        U = U.view(-1)

    kTi = kT[i]
    kTj = kT[j]
    beta_i = 1.0 / kTi
    beta_j = 1.0 / kTj

    # log acceptance ratio for swapping temperatures between fixed configs
    # log r = (β_i - β_j) * (U_i - U_j)
    log_r = (beta_i - beta_j) * (U[i] - U[j])

    # accept if log(u) < log_r  (with u~Uniform(0,1))
    u = torch.rand((), generator=rng, device=kT.device, dtype=kT.dtype)
    accept = torch.log(u) < log_r

    if not bool(accept):
        return kT, False

    # Swap temperatures
    kT_new = kT.clone()
    kT_new[i], kT_new[j] = kTj, kTi

    if rescale_momenta:
        # rescale momenta in-place for atoms belonging to each system
        # p_i *= sqrt(kT_new_i / kT_old_i), same for j
        si = torch.sqrt(kT_new[i] / kTi)
        sj = torch.sqrt(kT_new[j] / kTj)

        # NOTE: system_idx is atom-level, shape [n_atoms]
        mask_i = state.system_idx == i
        mask_j = state.system_idx == j

        # momenta is atom-level, shape [n_atoms, 3]
        p = state.momenta
        p = p.clone()
        p[mask_i] *= si
        p[mask_j] *= sj

        # Apply constraints-aware setter if available
        if hasattr(state, "set_constrained_momenta"):
            state.set_constrained_momenta(p)
        else:
            state.momenta = p

    return kT_new, True


def parallel_tempering_nvt(
    state,
    model,
    temperatures: torch.Tensor,  # shape [n_batch]
    dt: float,
    gamma: float,
    n_total_steps: int,
    md_steps_per_exchange: int = 100,
    seed: int = 123,
    rescale_momenta_on_swap: bool = True,
    reporter_args=None,
    start_step: int = 0,
):
    """
    Parallel tempering / replica exchange for a batched NVT Langevin simulation.

    Strategy:
      - propagate each replica for md_steps_per_exchange steps at its current kT
      - attempt neighbor swaps (even/odd scheme): (0,1),(2,3),... then (1,2),(3,4),...
      - repeat until n_total_steps are done

    We do NOT shuffle atoms between replicas. We swap the temperatures (kT labels),
    which is equivalent, and avoids expensive atom reindexing.

    Returns:
      state, kT, stats dict
    """
    initial_temps = temperatures.clone()
    device = temperatures.device
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    kT = temperatures * MetalUnits.temperature

    n_sys = int(state.n_systems)
    assert (
        temperatures.shape[0] == n_sys
    ), f"Expected temperatures shape [{n_sys}], got {tuple(temperatures.shape)}"

    stats = {
        "attempted": 0,
        "accepted": 0,
        "accept_rate": 0.0,
        "attempted_pairs": [],  # list of (step, i, j, accepted)
    }

    steps_done = 0
    exchange_round = 0

    while steps_done < n_total_steps:
        # How many MD steps in this block
        block = min(md_steps_per_exchange, n_total_steps - steps_done)

        # propagate block steps
        for t in range(block):
            if reporter_args is not None:
                trajectory_reporter = TrajectoryReporter(**reporter_args)
            else:
                trajectory_reporter = None

            state = ts.integrate(
                system=state,
                model=model,
                integrator=ts.Integrator.nvt_langevin,
                n_steps=1,
                temperature=temperatures,
                timestep=dt,
                trajectory_reporter=trajectory_reporter,
                gamma=gamma,
            )

            # if trajectory_reporter is not None:
            #     trajectory_reporter.report(state, start_step + steps_done + t, model=model)

        steps_done += block

        # refresh energies for swap decisions (ensures U corresponds to current configs)
        # (Many integrators already updated state.energy; this is just to be safe.)
        # with torch.no_grad():
        #     out = model.forward(state)
        #     state.energy = out["energy"]

        # choose pairing: even/odd to avoid conflicts
        if (exchange_round % 2) == 0:
            pairs = [(i, i + 1) for i in range(0, n_sys - 1, 2)]
        else:
            pairs = [(i, i + 1) for i in range(1, n_sys - 1, 2)]

        kT = temperatures * MetalUnits.temperature
        # attempt swaps
        for i, j in pairs:
            stats["attempted"] += 1
            kT, acc = _attempt_swap_temperatures(
                state,
                kT,
                i,
                j,
                rng=rng,
                rescale_momenta=rescale_momenta_on_swap,
            )
            stats["accepted"] += int(acc)
            stats["attempted_pairs"].append(
                (start_step + steps_done, i, j, bool(acc))
            )

        temperatures = kT / MetalUnits.temperature

        exchange_round += 1

    swap_data = [
        {
            "step": int(step),
            "i": int(i),
            "j": int(j),
            "accepted": bool(acc),
        }
        for (step, i, j, acc) in stats["attempted_pairs"]
    ]
    swap_log = {
        "data": swap_data,
        "info": dict(
            temperatures=initial_temps.tolist(),
            accept_rate=float(stats["accepted"] / max(1, stats["attempted"])),
        ),
    }

    return state, temperatures, swap_log


def plot_acc_per_pair(swap_log, ax=None):
    """
    Plot diagnostics from a parallel tempering swap log.

    Parameters
    ----------
    swap_log : list of dict
        Each entry must contain:
        {"step": int, "i": int, "j": int, "accepted": bool}

    show : bool
        Whether to call plt.show().

    savepath : str | None
        If given, saves the figure to this path.
    """
    # ------------------------------------------------------------
    # Parse log
    # ------------------------------------------------------------
    swap_data = swap_log["data"]
    accepted = np.array([e["accepted"] for e in swap_data], dtype=int)
    pairs = [(e["i"], e["j"]) for e in swap_data]

    # ------------------------------------------------------------
    # Acceptance rate per pair
    # ------------------------------------------------------------
    pair_stats = defaultdict(lambda: [0, 0])  # attempted, accepted
    for (i, j), acc in zip(pairs, accepted):
        pair_stats[(i, j)][0] += 1
        pair_stats[(i, j)][1] += acc

    pair_labels = []
    pair_acc = []
    for i, j in sorted(pair_stats):
        att, acc = pair_stats[(i, j)]
        pair_labels.append(f"{i}-{j}")
        pair_acc.append(acc / max(1, att))

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))

    # --- Panel 2: acceptance per ladder pair
    ax.bar(pair_labels, pair_acc)
    ax.set_ylim(0, 1)
    ax.set_ylabel("acceptance rate")
    ax.set_xlabel("replica pair")
    ax.set_title("Acceptance rate per temperature pair")
    ax.axhline(0.2, linestyle="--")
    ax.axhline(0.5, linestyle="--")


def plot_temperature_diffusion(T_hist, ax=None):
    """
    Visualize diffusion / mixing in temperature space for parallel tempering.

    Parameters
    ----------
    T_hist : array-like
        Temperature history with shape [n_exchanges, n_replicas].
        This is typically (torch.stack(kT_history) / MetalUnits.temperature).cpu().numpy().

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    indices = np.argsort(T_hist, axis=1)
    ax.plot(indices)

    ax.set_ylabel("temperature index")
    ax.set_xlabel("iteration")
    ax.set_title("Replica diffusion in temperature space")
    # ax.set_yticks(np.arange(n_T))
    # ax.set_yticklabels([f"{t:g}" for t in temps_unique])


def reconstruct_kT_history_from_swap_log(
    swap_log: list[dict],
    kT0: torch.Tensor,  # shape [n_batch], energy units
    group_by: str = "step",
) -> torch.Tensor:
    """
    Reconstruct kT_history from swap_log for the 'swap temperatures' PT scheme.

    swap_log entries must have:
        {"step": int, "i": int, "j": int, "accepted": bool}

    Returns
    -------
    kT_history : torch.Tensor
        shape [n_groups+1, n_batch], containing kT after each exchange group.
        kT_history[0] = kT0.
    """
    # kT = kT0.detach().clone().to(device=device, dtype=dtype)
    kT = torch.tensor(kT0)

    # Sort by group key (step) but keep original order within same step
    # (Python sort is stable)
    swap_log_sorted = sorted(swap_log, key=lambda e: e[group_by])

    history = [kT.clone()]

    current_group = None
    for entry in swap_log_sorted:
        g = entry[group_by]
        if current_group is None:
            current_group = g

        # when group changes, snapshot kT
        if g != current_group:
            history.append(kT.clone())
            current_group = g

        if entry["accepted"]:
            i = int(entry["i"])
            j = int(entry["j"])
            # swap temperatures
            tmp = kT[i].clone()
            kT[i] = kT[j]
            kT[j] = tmp

    # final snapshot
    history.append(kT.clone())

    return torch.stack(history, dim=0)


def plot_swap_log(swap_log):
    temperatures = swap_log["info"]["temperatures"]
    T_hist = reconstruct_kT_history_from_swap_log(
        swap_log["data"], kT0=temperatures
    )

    fig, axs = plt.subplots(2, 1, figsize=(6, 6))
    ax = axs[0]
    plot_temperature_diffusion(T_hist, ax=ax)

    ax = axs[1]
    plot_acc_per_pair(swap_log, ax=ax)
    fig.tight_layout()


def plot_swap_log_from_json(json_file):
    with open(json_file, "r") as f:
        swap_log = json.load(f)
    plot_swap_log(swap_log)
