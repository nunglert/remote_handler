"""RDF processor: annotate structures with the radial distribution function.

Lightweight and self-contained -- a PBC-aware neighbour-list histogram via ASE
(no ovito, no jaxnest). Stores ``r_rdf`` (bin centres) and ``rdf_total`` (g(r))
in each structure's info. Expected ``args`` keys: ``cutoff`` and ``nbins``.
"""

import numpy as np
from ase.neighborlist import neighbor_list


def _rdf(at, rmax, nbins):
    """Total g(r) for one structure, PBC-aware, normalised to 1 at large r."""
    dists = neighbor_list("d", at, rmax)  # all pair distances < rmax
    hist, edges = np.histogram(dists, bins=nbins, range=(0.0, rmax))
    r = 0.5 * (edges[1:] + edges[:-1])

    n = len(at)
    rho = n / at.get_volume()
    shell_vol = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    # ``neighbor_list`` counts each pair in both directions, so the ideal-gas
    # expectation is N * rho * shell_vol (each of N atoms sees rho*shell).
    norm = n * rho * shell_vol
    g = np.divide(hist, norm, out=np.zeros(nbins, dtype=float), where=norm > 0)
    return r, g


def run(atoms_list, args, calcpath=None):
    for at in atoms_list:
        r, g = _rdf(at, args["cutoff"], args["nbins"])
        at.info["r_rdf"] = r.tolist()
        at.info["rdf_total"] = g.tolist()
    return atoms_list
