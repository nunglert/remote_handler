"""Unit tests for the owned trajectory processors (no jaxnest needed).

The RDF and the LBFGS relaxation are dependency-light glue owned by
remote_handler (the NNFF calculator comes from neuralil-jaxrens and Steinhardt
from a downstream package, so neither is exercised here). Runs against ASE only.
"""

import numpy as np
import pytest
from ase import Atoms

from remote_handler.scripts.trajectory.processors import (
    REGISTRY,
    optimize,
    rdf,
    register,
)


def test_registry_has_builtin_steps():
    # Steinhardt moved downstream (it needs jaxrens); the built-ins are the
    # dependency-light ones plus the NNFF optimizer.
    assert set(REGISTRY) == {"optimize", "rdf"}


def test_register_adds_downstream_processor():
    """Downstream packages plug in via register() (e.g. their steinhardt)."""
    sentinel = object()

    def _fake(atoms_list, args, calcpath=None):
        return sentinel

    try:
        register("steinhardt", _fake)
        assert REGISTRY["steinhardt"] is _fake
        assert REGISTRY["steinhardt"]([], {}) is sentinel
    finally:
        REGISTRY.pop("steinhardt", None)


def _periodic_gas(n=32, box=12.0, seed=0):
    rng = np.random.default_rng(seed)
    at = Atoms(f"Ar{n}", cell=np.eye(3) * box, pbc=True)
    at.set_positions(rng.random((n, 3)) * box)
    return at


def test_rdf_shape_and_normalisation():
    at = _periodic_gas()
    out = rdf.run([at], {"cutoff": 5.0, "nbins": 100})
    r = np.array(out[0].info["r_rdf"])
    g = np.array(out[0].info["rdf_total"])

    assert len(r) == 100 and len(g) == 100
    assert np.all(np.isfinite(g))
    assert np.all(g >= 0.0)
    # A near-ideal gas should give an order-1 g(r) (not, say, 0 or 1e3).
    assert 0.3 < g[-30:].mean() < 3.0


def test_rdf_bin_centres_within_cutoff():
    at = _periodic_gas()
    out = rdf.run([at], {"cutoff": 4.0, "nbins": 40})
    r = np.array(out[0].info["r_rdf"])
    assert r[0] > 0.0 and r[-1] < 4.0


def test_optimize_lj_sets_converged_flag():
    at = Atoms(
        "Ar2",
        positions=[[0, 0, 0], [0, 0, 1.05]],
        cell=np.eye(3) * 10,
        pbc=True,
    )
    out = optimize.run([at], {"backend": "LJ", "fmax": 0.05, "steps": 100})
    assert isinstance(out[0].info["is_converged"], bool)
    assert out[0].info["is_converged"] is True


def test_optimize_skips_already_converged():
    """An atoms already marked converged is passed through untouched."""
    at = Atoms("Ar", positions=[[0, 0, 0]], cell=np.eye(3) * 10, pbc=True)
    at.info["is_converged"] = True
    pos_before = at.get_positions().copy()
    # backend is irrelevant: the calculator must not even be built.
    out = optimize.run([at], {"backend": "does-not-exist"})
    assert np.allclose(out[0].get_positions(), pos_before)


def test_optimize_unknown_backend_raises():
    at = Atoms("Ar", positions=[[0, 0, 0]], cell=np.eye(3) * 10, pbc=True)
    with pytest.raises(ValueError, match="Unknown optimization backend"):
        optimize.run([at], {"backend": "nope", "fmax": 0.1, "steps": 10})
