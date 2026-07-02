"""Optimization processor: relax each structure with an ASE optimizer.

The relaxation itself is plain ASE (LBFGS) and owned here. The calculator is
built per backend:

* ``NNFF`` / ``MACE`` -- an ensemble NeuralIL calculator provided upstream by
  neuralil-jaxrens (``neuralil.ase_integration``).
* ``LJ`` -- ASE's Lennard-Jones (for quick smoke tests).

Heavy imports are lazy so the registry stays importable host-side without those
deps. Expected ``args`` keys: ``backend``, ``fmax``, ``steps``, and for the NNFF
backends ``pickle_file``, ``max_neighbors``, ``supercell_diag``.
"""


def _build_calculator(args):
    backend = args["backend"]
    if backend in ("NNFF", "MACE"):
        # NeuralIL ASE calculator -- lives upstream in neuralil-jaxrens (migrated
        # out of the deprecated jaxnest.postprocess). One class serves single
        # models and ensembles; we load an ensemble (for uncertainties).
        # max_neighbors is a per-evaluation argument -- no compiled-function
        # cache.
        from neuralil.ase_integration import NeuralILASECalculator

        return NeuralILASECalculator.from_pickle(
            args["pickle_file"],
            max_neighbors=args.get("max_neighbors"),
            max_neighbors_list=args.get("max_neighbors_list"),
            neighbor_offset=args.get("neighbor_offset", 0),
            supercell_diag=tuple(args.get("supercell_diag", (1, 1, 1))),
            ensemble=True,
        )
    if backend == "LJ":
        from ase.calculators.lj import LennardJones

        return LennardJones(sigma=1.0, epsilon=1.0, rc=3.0)
    raise ValueError(f"Unknown optimization backend: {backend!r}")


def run(atoms_list, args, calcpath=None):
    from ase.optimize import LBFGS

    # Build the calculator lazily on first use: a fully-resumed shard (every
    # config already converged) then never loads the model at all.
    calc = None
    out = []
    for at in atoms_list:
        if "is_converged" not in at.info:
            if calc is None:
                calc = _build_calculator(args)
            relaxed = at.copy()
            relaxed.calc = calc
            opt = LBFGS(relaxed, logfile=None)
            converged = opt.run(fmax=args["fmax"], steps=args["steps"])
            relaxed.info["is_converged"] = bool(converged)
            at = relaxed
        out.append(at)
    return out
