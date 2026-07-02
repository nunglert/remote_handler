"""ASE calculator for mace-jax models -- single model or ensemble.

The mace-jax counterpart of the NeuralIL calculator, but graph-based: a
structure becomes a jraph graph (via mace-jax's own utilities) padded to a
fixed edge budget before the jitted model runs. The bucket quantity is kept as
``max_neighbors`` (the per-atom edge budget) for parity with the NeuralIL
calculator and the ``OptimizeProcessor`` plumbing -- for MACE that maps to a
padded edge cap of ``max_neighbors * n_nodes`` (edges ~= max_neighbors per
atom). A ladder of ``max_neighbors`` values bounds recompiles.

An ensemble is N separately-trained bundles sharing one architecture: their
params are stacked along a leading axis and the model is ``vmap``-ed over them,
so every member runs in one call and the spread becomes an uncertainty
(``energy_uncert``/``energy_uncert_relative``/``forces_uncert`` in
``atoms.info``). A single bundle is an ensemble of one -- no uncertainty.

Runs on the cluster in the mace-jax conda env; not imported host-side.
"""

import math

from ase.calculators.calculator import Calculator, all_changes


def pick_bucket(true_max, ladder, offset=0):
    """Smallest ladder entry that accommodates ``true_max + offset``.

    Discretises the per-atom edge budget onto a fixed ladder so the number of
    distinct JIT compilations is bounded by ``len(ladder)``. Raises when the
    ladder is exhausted (extend ``max_neighbors_list``).
    """
    target = int(true_max) + int(offset)
    for b in ladder:
        if b >= target:
            return int(b)
    raise RuntimeError(
        f"Observed per-atom edge budget {int(true_max)} (+offset {offset}) "
        f"needs a bucket >= {target}, but max_neighbors_list={list(ladder)} "
        "has no entry that large. Extend it."
    )


def edges_per_atom(n_edge, n_node):
    """Per-atom edge budget needed to hold ``n_edge`` edges over ``n_node``
    atoms (round up so ``budget * n_node >= n_edge``)."""
    return math.ceil(int(n_edge) / max(int(n_node), 1))


class MaceJaxEnsembleCalculator(Calculator):
    """mace-jax ASE calculator (single model or stacked-params ensemble).

    Prefer :meth:`from_bundles` to construct from one or more bundle
    directories (``config.json`` + ``params.msgpack``).

    Args:
        graphdef: The shared nnx ``GraphDef`` (identical across members).
        stacked_params: Member params stacked along a leading axis (shape
            ``(n_members, ...)`` per leaf). One member => ``n_members == 1``.
        r_max: Model cutoff radius.
        atomic_numbers: Model's element list (for the type mapping).
        max_neighbors / max_neighbors_list: Per-atom edge budget, single cap or
            ascending ladder (mapped to an edge cap of ``budget * n_nodes``).
        neighbor_offset: Headroom added before bucketing.
        n_members: Number of ensemble members (leading axis of stacked_params).
    """

    implemented_properties = ("energy", "free_energy", "forces")
    excluded_properties = ("initial_charges", "initial_magmoms")

    def __init__(
        self,
        graphdef,
        stacked_params,
        r_max,
        atomic_numbers,
        max_neighbors=None,
        max_neighbors_list=None,
        neighbor_offset=0,
        n_members=1,
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        import jax
        from mace_jax.tools import gin_model

        if max_neighbors_list:
            self.ladder = tuple(sorted(int(b) for b in max_neighbors_list))
        elif max_neighbors is not None:
            self.ladder = (int(max_neighbors),)
        else:
            raise ValueError("provide max_neighbors or max_neighbors_list")
        self.neighbor_offset = int(neighbor_offset)

        self.graphdef = graphdef
        self.stacked_params = stacked_params
        self.r_max = float(r_max)
        self.atomic_numbers = [int(z) for z in atomic_numbers]
        self.num_species = len(self.atomic_numbers)
        self.n_members = int(n_members)
        self.is_ensemble = self.n_members > 1
        self.results = {}

        def _member(params, graph):
            out, _ = self.graphdef.apply(params)(
                gin_model._graph_to_data(graph, num_species=self.num_species),
                compute_force=True,
                compute_stress=False,
            )
            return out["energy"], out["forces"]

        # vmap over the stacked member params (graph shared), then jit. One
        # call yields every member's energy/forces.
        self._predict = jax.jit(jax.vmap(_member, in_axes=(0, None)))

    @classmethod
    def from_bundles(
        cls,
        bundle_paths,
        max_neighbors=None,
        max_neighbors_list=None,
        neighbor_offset=0,
        dtype="float64",
    ):
        """Build from one or more mace-jax bundle directories.

        All members must share one architecture (identical ``graphdef``); their
        params are stacked for the vmapped ensemble evaluation.
        """
        import jax
        import jax.numpy as jnp
        from mace_jax.tools.bundle import load_model_bundle

        if isinstance(bundle_paths, (str, bytes)):
            bundle_paths = [bundle_paths]
        bundle_paths = list(bundle_paths)
        if not bundle_paths:
            raise ValueError("from_bundles needs at least one bundle path")

        bundles = [load_model_bundle(str(p), dtype) for p in bundle_paths]
        graphdef = bundles[0].graphdef
        config = bundles[0].config

        # Stack member params along a new leading axis (requires matching
        # pytree structure -- i.e. one shared architecture).
        stacked = jax.tree_util.tree_map(
            lambda *leaves: jnp.stack(leaves), *[b.params for b in bundles]
        )
        return cls(
            graphdef,
            stacked,
            r_max=config["r_max"],
            atomic_numbers=config["atomic_numbers"],
            max_neighbors=max_neighbors,
            max_neighbors_list=max_neighbors_list,
            neighbor_offset=neighbor_offset,
            n_members=len(bundles),
        )

    def _build_padded_graph(self, atoms):
        import numpy as np
        from mace_jax import data
        from mace_jax.data.utils import (
            AtomicNumberTable,
            atomic_numbers_to_indices,
            graph_from_configuration,
        )

        config = data.config_from_atoms(atoms)
        table = AtomicNumberTable(self.atomic_numbers)
        config.atomic_numbers = atomic_numbers_to_indices(
            config.atomic_numbers, table
        )
        graph = graph_from_configuration(config, cutoff=self.r_max)

        n_edge = int(np.asarray(graph.n_edge).item())
        n_node = int(np.asarray(graph.n_node).item())
        bucket = pick_bucket(
            edges_per_atom(n_edge, n_node), self.ladder, self.neighbor_offset
        )
        import jraph

        padded = jraph.pad_with_graphs(
            graph,
            n_node=n_node + 1,  # one dummy padding node
            n_edge=bucket * n_node,  # edge cap = per-atom budget * atoms
            n_graph=2,  # one real graph + one padding graph
        )
        return padded, n_node

    def calculate(
        self, atoms=None, properties=None, system_changes=all_changes
    ):
        import jax
        import numpy as np

        Calculator.calculate(self, atoms)
        padded, n_node = self._build_padded_graph(self.atoms)

        # energies: (n_members, n_graph); forces: (n_members, n_nodes, 3)
        energies, forces = self._predict(self.stacked_params, padded)
        energies = np.asarray(jax.lax.stop_gradient(energies))[:, 0]
        forces = np.asarray(jax.lax.stop_gradient(forces))[:, :n_node, :]

        energy = float(energies.mean())
        self.results = {"energy": energy, "free_energy": energy}
        self.results["forces"] = forces.mean(axis=0)

        if self.is_ensemble:
            uncert = float(energies.std())
            self.results["energy_uncert"] = uncert
            self.results["energy_uncert_relative"] = (
                uncert / abs(energy) if energy else uncert
            )
            self.results["forces_uncert"] = float(np.std(forces, axis=0).sum())
            # Also surface uncertainty on the atoms for downstream selectors.
            self.atoms.info["energy_uncert"] = uncert
            self.atoms.info["forces_uncert"] = self.results["forces_uncert"]
