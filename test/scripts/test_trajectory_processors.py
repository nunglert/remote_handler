"""Unit tests for the owned trajectory processors (no jaxnest needed).

The RDF and the LBFGS relaxation are dependency-light glue owned by
remote_handler (the NNFF calculator comes from neuralil-jaxrens and Steinhardt
from a downstream package, so neither is exercised here). Runs against ASE only.
"""

import importlib
import sys

import numpy as np
import pytest
from ase import Atoms

from remote_handler.exceptions import ScriptInputError
from remote_handler.scripts.trajectory.processors import (
    REGISTRY,
    optimize,
    rdf,
    register,
    validate_args,
)


def _import_registration_modules(modules):
    """Mirror the import step ``process_traj.py`` runs before resolving steps:
    import each module (its import populates REGISTRY), wrapping failures in a
    clear ImportError."""
    for mod in modules or []:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            raise ImportError(
                f"Could not import processor registration module {mod!r} on "
                f"the cluster. Is the downstream package installed in this env?"
            ) from exc


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


def test_driver_import_step_registers_downstream(tmp_path):
    """Importing a downstream module (as the driver does) runs its
    module-level register(...), making the step resolvable in REGISTRY."""
    pkg = tmp_path / "dummy_downstream.py"
    pkg.write_text(
        "from remote_handler.scripts.trajectory.processors import register\n"
        "def run(atoms_list, args, calcpath=None):\n"
        "    return atoms_list\n"
        "register('dummy_step', run)\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        assert "dummy_step" not in REGISTRY
        _import_registration_modules(["dummy_downstream"])
        assert "dummy_step" in REGISTRY
        # Idempotent: re-importing re-registers the same run without error.
        _import_registration_modules(["dummy_downstream"])
        assert REGISTRY["dummy_step"]([], {}) == []
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("dummy_downstream", None)
        REGISTRY.pop("dummy_step", None)


def test_driver_import_step_bad_module_raises_clear_error():
    with pytest.raises(
        ImportError, match="Is the downstream package installed"
    ):
        _import_registration_modules(["remote_handler.no_such_module_xyz"])


# -- per-step arg validation (the composed-payload contract) ----------------
def test_optimize_validate_args_accepts_valid():
    optimize.validate_args({"backend": "LJ", "fmax": 0.1, "steps": 100})
    optimize.validate_args(
        {"backend": "NNFF", "fmax": 0.1, "steps": 50, "pickle_file": "m.pkl"}
    )


@pytest.mark.parametrize(
    "args, match",
    [
        ({"fmax": 0.1, "steps": 10}, "backend"),
        ({"backend": "nope", "fmax": 0.1, "steps": 10}, "backend"),
        ({"backend": "LJ", "steps": 10}, "fmax"),
        ({"backend": "LJ", "fmax": 0.1}, "steps"),
        ({"backend": "LJ", "fmax": 0.1, "steps": 1.5}, "steps"),
        # NNFF without a model is the classic mid-run crash -- caught up front.
        ({"backend": "NNFF", "fmax": 0.1, "steps": 10}, "pickle_file"),
    ],
)
def test_optimize_validate_args_rejects(args, match):
    with pytest.raises(ScriptInputError, match=match):
        optimize.validate_args(args)


def test_rdf_validate_args_accepts_and_rejects():
    rdf.validate_args({"cutoff": 5.0, "nbins": 200})
    with pytest.raises(ScriptInputError, match="cutoff"):
        rdf.validate_args({"nbins": 200})
    with pytest.raises(ScriptInputError, match="cutoff"):
        rdf.validate_args({"cutoff": -1.0, "nbins": 200})
    with pytest.raises(ScriptInputError, match="nbins"):
        rdf.validate_args({"cutoff": 5.0, "nbins": 0})


def test_validate_args_dispatches_via_registry():
    """The driver calls validate_args(step, args); it routes to the processor's
    own validator and stays silent for good input."""
    validate_args("rdf", {"cutoff": 5.0, "nbins": 200})  # no raise
    with pytest.raises(ScriptInputError, match="cutoff"):
        validate_args("rdf", {"cutoff": 0.0, "nbins": 200})


def test_validate_args_permissive_for_processor_without_validator():
    """A processor registered with run only (no validator) is unchecked --
    validate_args is a no-op, never a KeyError."""
    try:
        register("nocheck", lambda a, args, calcpath=None: a)
        validate_args("nocheck", {"anything": 1})  # must not raise
    finally:
        REGISTRY.pop("nocheck", None)


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
