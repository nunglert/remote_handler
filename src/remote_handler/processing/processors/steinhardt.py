"""The ``steinhardt`` processor -- one module, both ends of the contract.

Holds :class:`SteinhardtArgs` (arg schema), :func:`run` (the cluster-side
compute), the host-side :class:`SteinhardtProcessor` spec, and the
:func:`register` call. It annotates each structure with per-atom Steinhardt
bond-orientational order parameters ``q<l>`` / ``w<l>`` (and their per-frame
means ``q<l>_mean`` / ``w<l>_mean`` in ``atoms.info``).

The compute is a JAX port of QUIP's ``steinhardt_nelson_qw_module::calc_qw``
(``src/libAtoms/steinhardt_nelson_qw.f95``) together with the angular machinery
it relies on (``SphericalYCartesian`` / ``SolidRCartesian`` and ``wigner3j`` in
``src/libAtoms/angular_functions.f95``).  Validated against QUIP's ``get_qw``
binary to machine precision; the reference data and comparison live in
``test/processing/test_steinhardt.py``.

For each central atom *i* and (even) degree *l*::

    c_m(i) = (1 / n_bonds(i)) * sum_{j in nbrs(i)} Y_lm(r_ij)
    q_l(i) = sqrt( (4 pi / (2l+1)) * sum_m |c_m|^2 )
    w_l(i) = Re( sum_{m1+m2+m3=0} c_m1 c_m2 c_m3 * wigner3j(l,m1,l,m2,l,m3) )
             / ( sum_m |c_m|^2 )^(3/2)

with ``Y_lm = SolidRCartesian(l, m, x) * sqrt((2l+1)/(4 pi)) * |x|^(-l)``.
Neighbours are all atoms within ``[r_cut_min, r_cut]``, periodic images
included, matching QUIP's ``calc_connect``.  Atoms with a vanishing bond-order
vector get ``q = w = 0`` (QUIP's driver maps the resulting NaNs to 0).

The Steinhardt parameters are inherently double-precision quantities (the
spherical-harmonic sums lose all signal in float32), so the ``jit``-compiled
core runs in float64 / complex128.  We opt those explicit wide dtypes in for the
scope of the computation via :func:`allow_explicit_x64` rather than flipping the
global ``jax_enable_x64`` flag, so any surrounding float32 pipeline is
unaffected.

``jax`` is an *optional* dependency (the ``steinhardt`` extra), imported lazily
inside :func:`run` and the functions it calls -- importing this module stays
cheap and jax-free host-side, exactly as the other built-in processors keep
their compute importable without heavy deps. Top-level imports are limited to
stdlib, numpy, and the remote_handler pieces below.
"""

from __future__ import annotations

import contextlib
import functools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from remote_handler.exceptions import ScriptInputError
from remote_handler.processing.base import Processor
from remote_handler.processing.registry import register
from remote_handler.scripts.input_models import ScriptInput


# --- precision helper (inlined from jaxrens._jax_init) ---------------------
@contextlib.contextmanager
def allow_explicit_x64():
    """Honour explicit float64 / complex128 dtypes for the enclosed block.

    Post-processing that legitimately needs double precision wraps its
    computation in this context manager: explicitly-requested float64 /
    complex128 arrays are honoured inside, while *default* float dtypes remain
    unchanged (so surrounding float32 code paths are unaffected), and the
    previous policy is restored on exit -- leaving no global precision state
    behind. Idempotent and re-entrant (nesting restores the outer policy).

    Inlined verbatim from ``jaxrens._jax_init.allow_explicit_x64`` so this
    processor carries no jaxrens dependency.
    """
    import jax

    prev = jax.config.jax_explicit_x64_dtypes
    jax.config.update("jax_explicit_x64_dtypes", "allow")
    try:
        yield
    finally:
        jax.config.update("jax_explicit_x64_dtypes", prev)


# --- cluster-side compute (JAX port of QUIP get_qw) ------------------------
def _factorial(n: int) -> float:
    """float64 factorial, matching QUIP's ``factorial`` (returns real(dp))."""
    return float(math.factorial(n))


def wigner3j(j1: int, m1: int, j2: int, m2: int, j: int, m: int) -> float:
    """Wigner 3j symbol, a faithful port of ``angular_functions::wigner3j``
    with ``denom = 1``."""
    # selection rules implied by the factorial ranges below
    if m1 + m2 + m != 0:
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m) > j:
        return 0.0
    if j > j1 + j2 or j < abs(j1 - j2):
        return 0.0

    pre_fac = (-1.0) ** (j1 - j2 - m)
    # triangle coefficient tc(j1,j2,j)
    tc = (
        _factorial(j1 + j2 - j)
        * _factorial(j1 - j2 + j)
        * _factorial(-j1 + j2 + j)
        / _factorial(j1 + j2 + j + 1)
    )
    triang_coeff = math.sqrt(tc)
    main_coeff = math.sqrt(
        _factorial(j1 + m1)
        * _factorial(j1 - m1)
        * _factorial(j2 + m2)
        * _factorial(j2 - m2)
        * _factorial(j + m)
        * _factorial(j - m)
    )

    kmin = max(j2 - j - m1, j1 + m2 - j, 0)
    kmax = min(j1 + j2 - j, j1 - m1, j2 + m2)
    sum_coeff = 0.0
    for k in range(kmin, kmax + 1):
        denom = (
            _factorial(k)
            * _factorial(j - j2 + m1 + k)
            * _factorial(j - j1 - m2 + k)
            * _factorial(j1 + j2 - j - k)
            * _factorial(j1 - m1 - k)
            * _factorial(j2 + m2 - k)
        )
        sum_coeff += ((-1.0) ** k) / denom

    return pre_fac * triang_coeff * main_coeff * sum_coeff


@dataclass(frozen=True)
class SolidRTable:
    """Static expansion of ``SolidRCartesian(l, m, x)`` for fixed ``l``.

    For each m, the solid harmonic is

        R_lm = factor_m * sum_p a^p b^(p-m) x3^(l-2p+m)
                                 / (p! (p-m)! (l-2p+m)!)

    with ``a = -x1/2 - i x2/2``, ``b = x1/2 - i x2/2``, summed over the p for
    which all three exponents are non-negative.  We pre-tabulate, per (m, p)
    term, the integer exponents (pa, pb, ps) and the real coefficient
    ``factor_m / (p!(p-m)!(l-2p+m)!)``.  ``m_index`` maps each term to its slot
    ``m + l`` in ``0..2l``.
    """

    l: int
    m_index: np.ndarray  # int, term -> column (m + l)
    pa: np.ndarray  # int, exponent of a
    pb: np.ndarray  # int, exponent of b
    ps: np.ndarray  # int, exponent of x3
    coeff: np.ndarray  # float64, term coefficient


def build_solidr_table(l: int) -> SolidRTable:
    """Pre-tabulate the ``SolidRCartesian(l, m, x)`` power expansion."""
    m_index, pa, pb, ps, coeff = [], [], [], [], []
    for m in range(-l, l + 1):
        factor_m = math.sqrt(_factorial(l + m) * _factorial(l - m))
        for p in range(0, l + 1):
            q = p - m
            s = l - p - q  # = l - 2p + m
            if q < 0 or s < 0:
                continue
            m_index.append(m + l)
            pa.append(p)
            pb.append(q)
            ps.append(s)
            coeff.append(
                factor_m / (_factorial(p) * _factorial(q) * _factorial(s))
            )
    return SolidRTable(
        l=l,
        m_index=np.asarray(m_index, dtype=np.int64),
        pa=np.asarray(pa, dtype=np.int64),
        pb=np.asarray(pb, dtype=np.int64),
        ps=np.asarray(ps, dtype=np.int64),
        coeff=np.asarray(coeff, dtype=np.float64),
    )


@dataclass(frozen=True)
class WignerTriples:
    """Flattened (m1, m2, m3) index triples with ``m3 = -m1-m2`` and the
    corresponding ``wigner3j(l,m1,l,m2,l,m3)`` value, for computing w_l."""

    l: int
    i1: np.ndarray  # int, column index m1 + l
    i2: np.ndarray  # int, column index m2 + l
    i3: np.ndarray  # int, column index m3 + l
    vals: np.ndarray  # float64, wigner3j value


def build_wigner_triples(l: int) -> WignerTriples:
    """Enumerate the non-zero ``wigner3j(l, ., l, ., l, .)`` triples (w_l)."""
    i1, i2, i3, vals = [], [], [], []
    for m1 in range(-l, l + 1):
        for m2 in range(-l, l + 1):
            m3 = -m1 - m2
            if m3 < -l or m3 > l:
                continue
            w = wigner3j(l, m1, l, m2, l, m3)
            if w == 0.0:
                continue
            i1.append(m1 + l)
            i2.append(m2 + l)
            i3.append(m3 + l)
            vals.append(w)
    return WignerTriples(
        l=l,
        i1=np.asarray(i1, dtype=np.int64),
        i2=np.asarray(i2, dtype=np.int64),
        i3=np.asarray(i3, dtype=np.int64),
        vals=np.asarray(vals, dtype=np.float64),
    )


def neighbour_edges(atoms, r_cut: float, r_cut_min: float = 0.0):
    """Build the neighbour list with the same convention as QUIP's
    ``calc_connect`` + ``neighbour(..., diff=...)``.

    Returns ``(centers, vectors)``: ``centers[k]`` is the central atom index
    of edge k and ``vectors[k] = r_j - r_i (+ periodic shift)``, i.e. the
    central->neighbour displacement.  Periodic self-images are included; the
    trivial zero-shift self pair is excluded -- exactly matching QUIP.
    Neighbours with distance ``< r_cut_min`` are dropped (the ``min_cutoff``
    guard in calc_qw).
    """
    from ase.neighborlist import neighbor_list

    i, _j, D = neighbor_list("ijD", atoms, r_cut)
    i = np.asarray(i, dtype=np.int64)
    D = np.asarray(D, dtype=np.float64)
    if r_cut_min > 0.0:
        dist = np.linalg.norm(D, axis=1)
        keep = dist >= r_cut_min
        i = i[keep]
        D = D[keep]
    return i, D


def _spherical_y(l, vectors, coeff, pa, pb, ps, m_index, n_cols):
    """Y_lm(x) for ``m = -l..l`` over edges. Complex, shape ``[E, 2l+1]``."""
    import jax.numpy as jnp

    x1 = vectors[:, 0]
    x2 = vectors[:, 1]
    x3 = vectors[:, 2]
    a = (-0.5 * x1) + (-0.5 * x2) * 1j
    b = (0.5 * x1) + (-0.5 * x2) * 1j

    # cumulative integer powers, p = 0..l (static l -> unrolled)
    a_pow = [jnp.ones_like(a)]
    b_pow = [jnp.ones_like(b)]
    x3_pow = [jnp.ones_like(x3)]
    for _ in range(1, l + 1):
        a_pow.append(a_pow[-1] * a)
        b_pow.append(b_pow[-1] * b)
        x3_pow.append(x3_pow[-1] * x3)
    a_pow = jnp.stack(a_pow)  # [l+1, E]
    b_pow = jnp.stack(b_pow)
    x3_pow = jnp.stack(x3_pow).astype(a.dtype)

    terms = coeff[:, None] * a_pow[pa] * b_pow[pb] * x3_pow[ps]  # [n_terms, E]
    E = vectors.shape[0]
    R = jnp.zeros((E, n_cols), dtype=a.dtype)
    R = R.at[:, m_index].add(terms.T)

    rnorm = jnp.sqrt(x1 * x1 + x2 * x2 + x3 * x3)
    pref = jnp.sqrt((2.0 * l + 1.0) / (4.0 * jnp.pi)) * rnorm ** (-l)
    return R * pref[:, None]


def _core_impl(
    l, vectors, centers, coeff, pa, pb, ps, n_atoms, m_index, i1, i2, i3, wvals
):
    import jax.numpy as jnp

    n_cols = 2 * l + 1
    y = _spherical_y(l, vectors, coeff, pa, pb, ps, m_index, n_cols)
    c = jnp.zeros((n_atoms, n_cols), dtype=y.dtype)
    c = c.at[centers].add(y)
    n_bonds = jnp.zeros(n_atoms).at[centers].add(1.0)

    denom = jnp.where(n_bonds > 0.0, n_bonds, 1.0)
    c = c / denom[:, None]

    sumsq = jnp.sum((c.conj() * c).real, axis=1)
    q = jnp.sqrt(sumsq * 4.0 * jnp.pi / (2.0 * l + 1.0))

    num = jnp.sum(c[:, i1] * c[:, i2] * c[:, i3] * wvals[None, :], axis=1).real
    w = num / sumsq**1.5

    # QUIP's driver replaces the NaN q/w of bondless atoms (0/0) by 0.
    q = jnp.where(jnp.isnan(q), 0.0, q)
    w = jnp.where(jnp.isnan(w), 0.0, w)
    return q, w


# ``_core_impl`` is jitted lazily (on first call) so that importing this module
# never touches jax; ``l`` (arg 0) and ``n_atoms`` (arg 7) are static.
_core_jitted = None


def _core(*args):
    global _core_jitted
    if _core_jitted is None:
        import jax

        _core_jitted = jax.jit(_core_impl, static_argnums=(0, 7))
    return _core_jitted(*args)


def qw_from_edges(
    l, vectors, centers, n_atoms, table=None, triples=None, device=None
):
    """Per-atom q_l, w_l from a precomputed neighbour list.

    ``vectors`` is ``[E, 3]`` central->neighbour displacements, ``centers`` is
    ``[E]`` central-atom indices.  ``table`` / ``triples`` are static tables
    (built here if omitted).  ``device`` optionally pins the inputs to a chosen
    JAX device (e.g. ``jax.devices('gpu')[0]``); otherwise the default backend
    is used.  Returns ``(q, w)`` JAX arrays of length ``n_atoms``.
    """
    import jax
    import jax.numpy as jnp

    if table is None:
        table = build_solidr_table(l)
    if triples is None:
        triples = build_wigner_triples(l)
    put = (
        (lambda x: jax.device_put(x, device))
        if device is not None
        else (lambda x: x)
    )
    with allow_explicit_x64():
        return _core(
            l,
            put(jnp.asarray(vectors, dtype=jnp.float64)),
            put(jnp.asarray(centers, dtype=jnp.int32)),
            jnp.asarray(table.coeff),
            jnp.asarray(table.pa, dtype=jnp.int32),
            jnp.asarray(table.pb, dtype=jnp.int32),
            jnp.asarray(table.ps, dtype=jnp.int32),
            n_atoms,
            jnp.asarray(table.m_index, dtype=jnp.int32),
            jnp.asarray(triples.i1, dtype=jnp.int32),
            jnp.asarray(triples.i2, dtype=jnp.int32),
            jnp.asarray(triples.i3, dtype=jnp.int32),
            jnp.asarray(triples.vals),
        )


def calc_qw(atoms, l, r_cut, r_cut_min=0.0, device=None):
    """Per-atom Steinhardt q_l and w_l for an ASE ``Atoms`` object.

    Builds the neighbour list (host-side, matching QUIP's periodic convention)
    and evaluates the order parameters.  ``device`` optionally pins the work to
    a specific JAX device (e.g. ``jax.devices('gpu')[0]``).  Returns ``(q, w)``
    NumPy arrays of length ``len(atoms)``.
    """
    table = build_solidr_table(l)
    triples = build_wigner_triples(l)
    centers, vectors = neighbour_edges(atoms, r_cut, r_cut_min)
    n_atoms = len(atoms)
    if len(centers) == 0:
        return np.zeros(n_atoms), np.zeros(n_atoms)
    q, w = qw_from_edges(
        l, vectors, centers, n_atoms, table, triples, device=device
    )
    return np.asarray(q), np.asarray(w)


# --- arg schema ------------------------------------------------------------
@dataclass
class SteinhardtArgs(ScriptInput):
    """Args for the ``steinhardt`` processor."""

    l_values: list = field(default_factory=list)
    r_cut: float = 0.0
    r_cut_min: float = 0.0

    def validate(self, root: Optional[Path] = None) -> None:
        if not self.l_values:
            raise ScriptInputError(
                "steinhardt: 'l_values' must be a non-empty list"
            )
        if not all(isinstance(v, int) for v in self.l_values):
            raise ScriptInputError("steinhardt: 'l_values' must be ints")
        if not isinstance(self.r_cut, (int, float)) or self.r_cut <= 0:
            raise ScriptInputError(
                "steinhardt: 'r_cut' must be a positive number"
            )
        if not isinstance(self.r_cut_min, (int, float)) or self.r_cut_min < 0:
            raise ScriptInputError(
                "steinhardt: 'r_cut_min' must be a non-negative number"
            )
        if self.r_cut_min >= self.r_cut:
            raise ScriptInputError(
                "steinhardt: 'r_cut_min' must be smaller than 'r_cut'"
            )


# --- cluster-side entry point ----------------------------------------------
def run(atoms_list, args, calcpath=None):
    """Annotate every structure with per-atom / per-frame Steinhardt q_l, w_l.

    Builds the per-``l`` static tables once and reuses them across all frames.
    Attaches per-atom arrays ``q<l>`` / ``w<l>`` (via ``atoms.new_array``) and
    per-frame means ``q<l>_mean`` / ``w<l>_mean`` in ``atoms.info``. Frames with
    no neighbours within the cutoff get all-zero arrays.
    """
    ls = [int(v) for v in args["l_values"]]
    r_cut = float(args["r_cut"])
    r_cut_min = float(args.get("r_cut_min", 0.0))
    tables = {v: (build_solidr_table(v), build_wigner_triples(v)) for v in ls}
    for atoms in atoms_list:
        n_atoms = len(atoms)
        for v in ls:
            table, triples = tables[v]
            centers, vectors = neighbour_edges(atoms, r_cut, r_cut_min)
            if len(centers) == 0:
                q = np.zeros(n_atoms)
                w = np.zeros(n_atoms)
            else:
                q, w = qw_from_edges(
                    v, vectors, centers, n_atoms, table, triples
                )
                q = np.asarray(q)
                w = np.asarray(w)
            atoms.new_array(f"q{v}", q)
            atoms.new_array(f"w{v}", w)
            atoms.info[f"q{v}_mean"] = float(q.mean())
            atoms.info[f"w{v}_mean"] = float(w.mean())
    return atoms_list


# --- host-side spec --------------------------------------------------------
class SteinhardtProcessor(Processor):
    """Annotate structures with Steinhardt bond-orientational order params."""

    name = "steinhardt"
    registration_module = None  # built-in: run is always in the registry

    def __init__(
        self,
        r_cut: float,
        *,
        l_values: Sequence[int] = (4, 6),
        r_cut_min: float = 0.0,
    ):
        if not l_values:
            raise ValueError("steinhardt: 'l_values' must be non-empty")
        self.r_cut = r_cut
        self.l_values = list(l_values)
        self.r_cut_min = r_cut_min

    def build_args(self, savepath: str) -> dict:
        return SteinhardtArgs(
            l_values=list(self.l_values),
            r_cut=self.r_cut,
            r_cut_min=self.r_cut_min,
        ).to_dict()


register("steinhardt", run, schema=SteinhardtArgs)
