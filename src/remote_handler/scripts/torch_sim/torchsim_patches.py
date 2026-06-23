import torch_sim
import torch_sim.state
from torch_sim.state import *


# def _split_state_patched[T: SimState](state: T) -> list[T]:
def _split_state_patched(state: SimState) -> list[SimState]:
    """Split a SimState into a list of states, each containing a single system.

    Divides a multi-system state into individual single-system states, preserving
    appropriate properties for each system.

    Args:
        state (SimState): The SimState to split

    Returns:
        list[SimState]: A list of SimState objects, each containing a single
            system
    """
    system_sizes = state.n_atoms_per_system.tolist()

    split_per_atom = {}
    for attr_name, attr_value in get_attrs_for_scope(state, "per-atom"):
        if attr_name != "system_idx":
            split_per_atom[attr_name] = torch.split(
                attr_value, system_sizes, dim=0
            )

    split_per_system = {}
    for attr_name, attr_value in get_attrs_for_scope(state, "per-system"):
        if isinstance(attr_value, torch.Tensor):
            split_per_system[attr_name] = torch.split(attr_value, 1, dim=0)
        else:  # Non-tensor attributes are replicated for each split
            split_per_system[attr_name] = [attr_value] * state.n_systems

    global_attrs = dict(get_attrs_for_scope(state, "global"))

    # Create a state for each system
    states: list[SimState] = []
    n_systems = len(system_sizes)
    zero_tensor = torch.tensor([0], device=state.device, dtype=torch.int64)
    cumsum_atoms = torch.cat(
        (zero_tensor, torch.cumsum(state.n_atoms_per_system, dim=0))
    )
    for sys_idx in range(n_systems):
        system_attrs = {
            # Create a system tensor with all zeros for this system
            "system_idx": torch.zeros(
                system_sizes[sys_idx], device=state.device, dtype=torch.int64
            ),
            # Add the split per-atom attributes
            **{
                attr_name: split_per_atom[attr_name][sys_idx]
                for attr_name in split_per_atom
            },
            # Add the split per-system attributes
            **{
                attr_name: split_per_system[attr_name][sys_idx]
                for attr_name in split_per_system
            },
            # Add the global attributes
            **global_attrs,
        }

        atom_idx = torch.arange(
            cumsum_atoms[sys_idx],
            cumsum_atoms[sys_idx + 1],
            device=state.device,
        )
        new_constraints = [
            new_constraint
            for constraint in state.constraints
            if (
                new_constraint := constraint.select_sub_constraint(
                    atom_idx, sys_idx
                )
            )
        ]

        system_attrs["_constraints"] = new_constraints
        states.append(type(state)(**system_attrs))  # type: ignore[invalid-argument-type]

    return states


def _torch_sim_version() -> tuple[int, ...]:
    raw = getattr(torch_sim, "__version__", "0")
    parts = []
    for chunk in raw.split(".")[:2]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


# Why this patch is version-gated:
#   torch_sim < 0.6 computed per-system ``new_constraints`` in ``_split_state``
#   but never passed them to the split state, silently dropping constraints.
#   The copy above fixes that by setting ``_constraints`` before construction.
#   torch_sim >= 0.6 fixed this upstream AND added ``atom_extras``/
#   ``system_extras`` routing -- there our stale copy would crash (e.g. it
#   passes the ``node_energy`` atom-extra straight to ``MDState.__init__``,
#   which no longer accepts it). So only patch the buggy older versions.
if _torch_sim_version() < (0, 6):
    torch_sim.state._split_state = _split_state_patched
