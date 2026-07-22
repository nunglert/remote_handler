"""Validate the built-in ``steinhardt`` processor and its JAX compute.

The compute is a JAX port of QUIP's ``get_qw`` (ported into remote_handler from
jaxrens). The reference values in ``test/data/steinhardt`` were produced by
QUIP's ``get_qw`` binary (``steinhardt_nelson_qw_module::calc_qw``) on a spread
of structures -- fcc/bcc/hcp/diamond crystals, a rattled crystal, rocksalt NaCl,
an isolated benzene molecule, and a 3-frame trajectory -- for several
``(l, r_cut, r_cut_min)`` settings.

For each atom we compare per-atom ``q_l`` to machine precision. ``w_l`` is
``Re(sum c^3 wigner3j) / (sum|c|^2)^1.5``, a 0/0 form for atoms whose bond-order
vector vanishes by symmetry (``q_l ~ 0``, e.g. ``q2`` on a perfectly cubic
site); QUIP and this port then emit uncorrelated round-off there, so those atoms
are excluded from the ``w`` comparison.

The jax-backed tests ``pytest.importorskip("jax")`` (the ``steinhardt`` extra);
the arg-schema, host-spec and ``wigner3j`` tests are pure-python and always run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from remote_handler.exceptions import ScriptInputError
from remote_handler.processing import validate_args
from remote_handler.processing.processors.steinhardt import (
    SteinhardtArgs,
    SteinhardtProcessor,
    calc_qw,
    neighbour_edges,
    qw_from_edges,
    run,
    wigner3j,
)

_DATA = Path(__file__).parent.parent / "data" / "steinhardt"
_MANIFEST = json.loads((_DATA / "get_qw_reference.json").read_text())
_TOL = _MANIFEST["tolerance"]

# Flatten (case, frame) into one parametrisation: each frame is its own test.
_PARAMS = []
for _case in _MANIFEST["cases"]:
    for _fi in range(len(_case["frames"])):
        _label = (
            f"{_case['structure']}-f{_fi}"
            f"-l{_case['l']}-rmin{_case['r_cut_min']}"
        )
        _PARAMS.append(pytest.param(_case, _fi, id=_label))


# --- numerical correctness vs QUIP get_qw (jax) ----------------------------
@pytest.mark.parametrize("case, frame_index", _PARAMS)
def test_calc_qw_matches_get_qw(case, frame_index):
    pytest.importorskip("jax")
    frames = read(_DATA / case["structure"], index=":")
    atoms = frames[frame_index]

    q, w = calc_qw(atoms, case["l"], case["r_cut"], case["r_cut_min"])

    ref = case["frames"][frame_index]
    q_ref = np.asarray(ref["q"])
    w_ref = np.asarray(ref["w"])

    assert q.shape == q_ref.shape
    np.testing.assert_allclose(q, q_ref, atol=_TOL, rtol=0.0)

    # exclude atoms whose bond-order vector vanishes (w = 0/0 is undefined)
    w_defined = q_ref > 1e-6
    if np.any(w_defined):
        np.testing.assert_allclose(
            w[w_defined], w_ref[w_defined], atol=_TOL, rtol=0.0
        )


# --- compute edge cases (jax) ----------------------------------------------
def test_no_neighbours_returns_zeros():
    pytest.importorskip("jax")
    atoms = Atoms("Cu", positions=[[0, 0, 0]], cell=[20, 20, 20], pbc=True)
    q, w = calc_qw(atoms, 6, r_cut=3.0)
    assert q.shape == (1,)
    assert np.all(q == 0.0) and np.all(w == 0.0)


def test_device_pinning_matches_default():
    # Passing an explicit device must not change the result (covers the
    # device_put path in qw_from_edges).
    jax = pytest.importorskip("jax")
    atoms = read(_DATA / "fcc_Cu.xyz")
    q0, w0 = calc_qw(atoms, 6, r_cut=3.0)
    q1, w1 = calc_qw(atoms, 6, r_cut=3.0, device=jax.devices()[0])
    np.testing.assert_allclose(q1, q0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(w1, w0, atol=1e-12, rtol=0.0)


def test_qw_from_edges_builds_default_tables():
    # Calling the core without precomputed tables exercises the
    # ``table is None`` / ``triples is None`` default-build branches.
    pytest.importorskip("jax")
    atoms = read(_DATA / "fcc_Cu.xyz")
    centers, vectors = neighbour_edges(atoms, 3.0)
    q, w = qw_from_edges(6, vectors, centers, len(atoms))
    q_ref, w_ref = calc_qw(atoms, 6, r_cut=3.0)
    np.testing.assert_allclose(np.asarray(q), q_ref, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(np.asarray(w), w_ref, atol=1e-12, rtol=0.0)


# --- wigner3j selection rules (pure python, always runs) -------------------
def test_wigner3j_selection_rules_return_zero():
    # m1 + m2 + m != 0
    assert wigner3j(2, 1, 2, 1, 2, 1) == 0.0
    # |m1| > j1 (with a vanishing m-sum)
    assert wigner3j(2, 3, 2, 0, 2, -3) == 0.0
    # triangle inequality violated: j > j1 + j2
    assert wigner3j(1, 0, 1, 0, 5, 0) == 0.0


# --- arg schema (pure python, always runs) ---------------------------------
def test_steinhardt_args_accepts_valid():
    validate_args(
        "steinhardt", {"l_values": [4, 6], "r_cut": 3.2, "r_cut_min": 0.0}
    )


@pytest.mark.parametrize(
    "args, match",
    [
        ({"l_values": [], "r_cut": 3.2}, "l_values"),
        ({"l_values": [4.0], "r_cut": 3.2}, "l_values"),
        ({"l_values": [4], "r_cut": 0.0}, "r_cut"),
        ({"l_values": [4], "r_cut": 3.2, "r_cut_min": -1.0}, "r_cut_min"),
        ({"l_values": [4], "r_cut": 3.2, "r_cut_min": 3.2}, "r_cut_min"),
    ],
)
def test_steinhardt_args_rejects(args, match):
    with pytest.raises(ScriptInputError, match=match):
        validate_args("steinhardt", args)


# --- host-side spec (pure python, always runs) -----------------------------
def test_processor_is_a_builtin_spec():
    proc = SteinhardtProcessor(r_cut=3.2, l_values=(4, 6))
    assert proc.name == "steinhardt"
    # built-in: run is always in the registry, so no registration module.
    assert proc.registration_module is None


def test_processor_empty_l_values_raises():
    with pytest.raises(ValueError, match="l_values"):
        SteinhardtProcessor(r_cut=3.2, l_values=())


def test_host_build_args_round_trips_through_schema():
    """What build_args emits is exactly what the driver's schema accepts."""
    proc = SteinhardtProcessor(r_cut=3.2, l_values=(4, 6), r_cut_min=0.5)
    args = proc.build_args("/s")
    assert args == {"l_values": [4, 6], "r_cut": 3.2, "r_cut_min": 0.5}
    validate_args("steinhardt", args)
    SteinhardtArgs.from_dict(args).validate()


# --- processor run over a real structure (jax) -----------------------------
def test_run_annotates_expected_keys():
    pytest.importorskip("jax")
    atoms = read(_DATA / "fcc_Cu.xyz")
    (out,) = run([atoms], {"l_values": [4, 6], "r_cut": 3.5})

    n = len(out)
    for lv in (4, 6):
        assert out.get_array(f"q{lv}").shape == (n,)
        assert out.get_array(f"w{lv}").shape == (n,)
        assert f"q{lv}_mean" in out.info
        assert f"w{lv}_mean" in out.info
        # per-atom values match the standalone compute for the same l/cutoff.
        q_ref, _ = calc_qw(atoms, lv, r_cut=3.5)
        np.testing.assert_allclose(
            out.get_array(f"q{lv}"), q_ref, atol=1e-12, rtol=0.0
        )
        assert out.info[f"q{lv}_mean"] == pytest.approx(float(q_ref.mean()))
