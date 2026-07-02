"""The ``rdf`` processor -- one module, both ends of the contract.

Holds :class:`RdfArgs` (arg schema), :func:`run` (the cluster-side compute), the
host-side :class:`RDFProcessor` spec, and the :func:`register` call. The RDF
itself is lightweight and self-contained -- a PBC-aware neighbour-list histogram
via ASE (no ovito, no jaxnest). Stores ``r_rdf`` (bin centres) and ``rdf_total``
(g(r)) in each structure's info.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from ase.neighborlist import neighbor_list

from remote_handler.exceptions import ScriptInputError
from remote_handler.processing.base import Processor
from remote_handler.processing.registry import register
from remote_handler.scripts.input_models import ScriptInput


# --- arg schema ------------------------------------------------------------
@dataclass
class RdfArgs(ScriptInput):
    """Args for the ``rdf`` processor."""

    cutoff: float
    nbins: int

    def validate(self, root: Optional[Path] = None) -> None:
        if not isinstance(self.cutoff, (int, float)) or self.cutoff <= 0:
            raise ScriptInputError("rdf: 'cutoff' must be a positive number")
        if not isinstance(self.nbins, int) or self.nbins <= 0:
            raise ScriptInputError("rdf: 'nbins' must be a positive int")


# --- cluster-side compute --------------------------------------------------
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


# --- host-side spec --------------------------------------------------------
class RDFProcessor(Processor):
    """Annotate structures with the radial distribution function."""

    name = "rdf"
    registration_module = None  # built-in: run is always in the registry

    def __init__(self, cutoff: float = 5.0, nbins: int = 200):
        self.cutoff = cutoff
        self.nbins = nbins

    def build_args(self, savepath: str) -> dict:
        return RdfArgs(cutoff=self.cutoff, nbins=self.nbins).to_dict()


register("rdf", run, schema=RdfArgs)
