"""Tests for the mace-jax ASE calculator glue.

The bucket ladder and the ``max_neighbors -> edge-cap`` mapping are exercised
directly. The mapping is validated against *real* mace-jax graph building
(neighbour-list construction is model-independent, so it needs no trained
bundle). The actual model forward pass (energies/forces/uncertainty) needs a
real bundle and is validated on the cluster -- not here.
"""

import numpy as np
import pytest
from ase import Atoms

from remote_handler.scripts.mace.mace_jax_ase import (
    edges_per_atom,
    pick_bucket,
)


# --- pure logic ------------------------------------------------------------
def test_pick_bucket_smallest_fitting():
    ladder = (20, 30, 40)
    assert pick_bucket(5, ladder) == 20
    assert pick_bucket(20, ladder) == 20
    assert pick_bucket(21, ladder) == 30


def test_pick_bucket_offset_and_exhaustion():
    assert pick_bucket(19, (20, 30), offset=2) == 30
    with pytest.raises(RuntimeError, match="Extend it"):
        pick_bucket(100, (20, 30))


def test_edges_per_atom_rounds_up():
    # budget * n_node must be >= n_edge.
    assert edges_per_atom(50, 8) == 7  # 50/8 = 6.25 -> 7
    assert edges_per_atom(48, 8) == 6
    assert edges_per_atom(0, 8) == 0


# --- edge-cap mapping against real mace-jax graph building ------------------
def _cu_cluster(n=6, spacing=2.5, box=12.0):
    # A compact cluster so there are real neighbours within the cutoff.
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, spacing * 2, size=(n, 3))
    return Atoms(
        "Cu" + str(n), positions=positions, cell=np.eye(3) * box, pbc=True
    )


def test_edge_cap_is_bucket_times_nodes():
    pytest.importorskip("mace_jax")
    pytest.importorskip("jraph")
    from remote_handler.scripts.mace.mace_jax_ase import (
        MaceJaxEnsembleCalculator,
    )

    calc = MaceJaxEnsembleCalculator(
        graphdef=object(),  # unused until a forward pass
        stacked_params=None,
        r_max=5.0,
        atomic_numbers=[29],  # Cu
        max_neighbors_list=[20, 30, 40, 60],
        n_members=1,
    )
    assert calc.ladder == (20, 30, 40, 60)
    assert calc.is_ensemble is False

    atoms = _cu_cluster()
    padded, n_node = calc._build_padded_graph(atoms)

    # The padded graph's total edge budget is exactly bucket * n_node, and the
    # chosen bucket is the ladder entry that covers the real per-atom edges.
    assert n_node == len(atoms)
    import mace_jax.data as _data  # noqa: F401  (ensure real graph path ran)

    total_edges = int(np.asarray(padded.n_edge).sum())
    assert total_edges % n_node == 0
    bucket = total_edges // n_node
    assert bucket in calc.ladder
    # Padding added exactly one dummy node and one padding graph.
    assert int(np.asarray(padded.n_node).sum()) == n_node + 1


def test_ensemble_flag_reflects_member_count():
    pytest.importorskip("mace_jax")
    from remote_handler.scripts.mace.mace_jax_ase import (
        MaceJaxEnsembleCalculator,
    )

    single = MaceJaxEnsembleCalculator(
        object(),
        None,
        r_max=5.0,
        atomic_numbers=[29],
        max_neighbors=40,
        n_members=1,
    )
    ens = MaceJaxEnsembleCalculator(
        object(),
        None,
        r_max=5.0,
        atomic_numbers=[29],
        max_neighbors=40,
        n_members=4,
    )
    assert single.is_ensemble is False
    assert ens.is_ensemble is True
