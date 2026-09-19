"""Redundant internal coordinates (RIC) for JAX-native geometry optimization.

Builds a redundant internal coordinate set — bond stretches, valence bends,
linear-bend pairs, proper torsions, Wilson out-of-plane bends, and TRIC
translation/rotation coordinates for multi-fragment systems — evaluates it as a
vectorised ``jnp`` function, and provides the Wilson B matrix by automatic
differentiation together with the Pulay-Fogarasi back-transformation.

The coordinate *definition* (index arrays, masks, frozen reference axes) is
host-side NumPy and static; every geometry-dependent quantity is a traced
``jnp`` array.  The returned callables are therefore ``jit``- and ``vmap``-safe
for a batch of structures that share one topology, which is the common case for
a conformer ensemble.

Two choices here differ from a textbook implementation and are worth flagging:

* The Wilson B matrix is obtained by ``vmap(jacrev(kernel))`` over compact
  per-coordinate atom blocks rather than from hand-coded rows.  Each coordinate
  touches at most four atoms, so this costs ``O(n_int)`` rather than the
  ``O(n_int * 3N)`` of a whole-vector Jacobian, and — more importantly — B is
  then *exactly* the derivative of the ``q`` that is actually evaluated.  An
  ``O(eps)`` inconsistency between a hand-coded row and its coordinate is the
  usual cause of back-transformation divergence and is invisible until it bites.
* TRIC rotations use the linearised least-squares rotation vector rather than
  the quaternion exponential map, so their B rows are constant, there is no
  eigendecomposition, no ``theta -> pi`` branch cut, and linear or single-atom
  fragments need no special case.  The price is first-order accuracy in the
  rotation angle, paid for by refreshing the reference geometry.

References:
    Pulay & Fogarasi, J. Chem. Phys. 96 (1992) 2856.
    Baker, J. Comput. Chem. 14 (1993) 1085; Baker & Pulay, J. Chem. Phys. 105 (1996) 11100.
    Peng, Ayala, Schlegel & Frisch, J. Comput. Chem. 17 (1996) 49.
    Bakken & Helgaker, J. Chem. Phys. 117 (2002) 9160.
    Wang & Song, J. Chem. Phys. 144 (2016) 214108 (geomeTRIC / TRIC).
"""

from typing import Callable, List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .topology import _connected_components, build_bonded_lists
from .utils import detect_bonds

# ============================================================================
# Coordinate type tags
# ============================================================================

COORD_STRETCH = 0
COORD_BEND = 1
COORD_LINEAR = 2
COORD_TORSION = 3
COORD_OOP = 4
COORD_TRANS = 5
COORD_ROT = 6

_EPS2 = 1e-20


# ============================================================================
# Static coordinate definition
# ============================================================================


class CoordinateSpec(NamedTuple):
    """Static, host-side definition of one redundant internal coordinate set.

    Every array is NumPy and is closed over by the generated ``jnp`` kernels,
    so none of it appears as a traced value and no shape is data-dependent.
    One spec is valid for every structure sharing the same topology.

    Attributes:
        n_atoms: Number of atoms.
        n_internal: Total number of internal coordinates.
        stretch_idx: ``(n_str, 2)`` bonded and auxiliary atom pairs.
        stretch_r0: ``(n_str,)`` reference lengths in Angstrom (breakage tests).
        bend_idx: ``(n_bend, 3)`` ``(i, j, k)`` with *j* central.
        bend_has_linear: ``(n_bend,)`` True when a linear pair also covers this
            bend, so it may degenerate harmlessly.
        linear_idx: ``(n_lin, 3)`` near-linear angles carrying a coordinate pair.
        linear_axes: ``(n_lin, 2, 3)`` frozen orthonormal axes ``(u, w)``.
        torsion_idx: ``(n_tor, 4)`` proper torsions.
        oop_idx: ``(n_oop, 4)`` ``(i, j, k, l)`` with *j* the central atom.
        frag_atoms: ``(n_frag, max_frag)`` atom indices, zero-padded.
        frag_w: ``(n_frag, max_frag)`` weights, 0.0 on padding.
        rot_frag: ``(n_rot_frag,)`` indices of fragments that emit rotations.
        rot_pinv: ``(n_rot_frag, 3, 3)`` pseudo-inverse of the reference
            inertia tensor (rank 2 for linear fragments).
        coord_type: ``(n_internal,)`` one of the ``COORD_*`` tags.
        is_periodic: ``(n_internal,)`` True for torsions only.
        expected_rank: ``3N - 6`` (``-5`` linear, ``-3`` for a single atom).
        rank_at_build: Numerical rank of B at the reference geometry.
        is_linear_system: True when the whole system is collinear.
        fragments: Atom-index lists, one per fragment.
        n_aux_bonds: Number of auxiliary (non-covalent) stretches, at the tail
            of ``stretch_idx``.
        valid: False when the set cannot describe the system; the caller must
            then fall back to Cartesian coordinates.
        reason: Why ``valid`` is False, or None.
    """

    n_atoms: int
    n_internal: int
    stretch_idx: np.ndarray
    stretch_r0: np.ndarray
    bend_idx: np.ndarray
    bend_has_linear: np.ndarray
    linear_idx: np.ndarray
    linear_axes: np.ndarray
    torsion_idx: np.ndarray
    oop_idx: np.ndarray
    frag_atoms: np.ndarray
    frag_w: np.ndarray
    rot_frag: np.ndarray
    rot_pinv: np.ndarray
    coord_type: np.ndarray
    is_periodic: np.ndarray
    expected_rank: int
    rank_at_build: int
    is_linear_system: bool
    fragments: List[List[int]]
    n_aux_bonds: int
    valid: bool
    reason: Optional[str]

    @property
    def n_fragments(self) -> int:
        """Number of molecular fragments the coordinate set was built for."""
        return len(self.fragments)


# ============================================================================
# Host-side construction helpers
# ============================================================================


def _angle_deg(p, i, j, k) -> float:
    u = p[i] - p[j]
    v = p[k] - p[j]
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-8 or nv < 1e-8:
        return 0.0
    c = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _orthonormal_pair(axis: np.ndarray) -> np.ndarray:
    """Two unit vectors spanning the plane perpendicular to *axis*."""
    a = axis / (np.linalg.norm(axis) + 1e-12)
    cart = np.eye(3)
    u = cart[int(np.argmin(np.abs(a)))]
    u = u - np.dot(u, a) * a
    u /= np.linalg.norm(u) + 1e-12
    w = np.cross(a, u)
    w /= np.linalg.norm(w) + 1e-12
    return np.stack([u, w])


def _interfragment_bonds(
    positions: np.ndarray,
    fragments: List[List[int]],
    max_per_edge: int = 3,
) -> List[Tuple[int, int]]:
    """Auxiliary stretches connecting fragments along a minimum spanning tree.

    Used by ``interfragment="aux"``.  For each MST edge the closest atom pair
    is taken, plus up to ``max_per_edge - 1`` further short pairs so the edge
    constrains more than one degree of freedom.  The count stays
    ``O(n_fragments)``; a full pairwise scheme would be ``O(n_fragments**2)``.
    """
    n_frag = len(fragments)
    if n_frag < 2:
        return []
    # Closest pair between every fragment pair (n_frag is small for "aux").
    best = {}
    for a in range(n_frag):
        pa = positions[fragments[a]]
        for b in range(a + 1, n_frag):
            pb = positions[fragments[b]]
            d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=-1)
            flat = np.argsort(d, axis=None)[:max_per_edge]
            pairs = []
            for f in flat:
                ia, ib = np.unravel_index(f, d.shape)
                pairs.append((fragments[a][ia], fragments[b][ib], float(d[ia, ib])))
            best[(a, b)] = pairs
    # Prim's algorithm on the fragment graph, weighted by the closest distance.
    in_tree = {0}
    edges = []
    while len(in_tree) < n_frag:
        cand = None
        for a in in_tree:
            for b in range(n_frag):
                if b in in_tree:
                    continue
                key = (min(a, b), max(a, b))
                d = best[key][0][2]
                if cand is None or d < cand[0]:
                    cand = (d, key, b)
        edges.append(cand[1])
        in_tree.add(cand[2])
    out = []
    for key in edges:
        for i, j, _d in best[key]:
            out.append((min(i, j), max(i, j)))
    return out


def _hbond_stretches(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    bond_set: set,
    cutoff: float = 2.5,
    angle_min: float = 90.0,
) -> List[Tuple[int, int]]:
    """``H...A`` stretches for donor-hydrogen/acceptor pairs across fragments.

    Redundant on purpose: they carry chemistry that the TRIC rigid-body
    coordinates cannot express, and they give the Lindh model something
    distance-dependent to put stiffness on.
    """
    donors = {7, 8, 9, 15, 16, 17}
    h_idx = np.where(atomic_numbers == 1)[0]
    acc_idx = np.array([i for i in range(len(atomic_numbers)) if atomic_numbers[i] in donors])
    if len(h_idx) == 0 or len(acc_idx) == 0:
        return []
    # Covalent partner of each H (its nearest bonded heavy atom).
    partner = {}
    for i, j in bond_set:
        if atomic_numbers[i] == 1:
            partner[i] = j
        if atomic_numbers[j] == 1:
            partner[j] = i
    out = []
    for h in h_idx:
        d = partner.get(int(h))
        if d is None or atomic_numbers[d] not in donors:
            continue
        for a in acc_idx:
            a = int(a)
            if a == d:
                continue
            key = (min(int(h), a), max(int(h), a))
            if key in bond_set:
                continue
            dist = float(np.linalg.norm(positions[h] - positions[a]))
            if dist > cutoff:
                continue
            if _angle_deg(positions, d, int(h), a) < angle_min:
                continue
            out.append(key)
    return sorted(set(out))


def _rot_metric(ref: np.ndarray, w: np.ndarray, tol: float = 1e-8):
    """Pseudo-inverse of the reference inertia tensor and its rank.

    ``A = sum_i w_i (|y_i|^2 I - y_i y_i^T)``.  Rank 3 for a general fragment,
    2 for a linear one, 0 for a single atom — so the rank decides how many
    rotation coordinates the fragment can support without any special-casing.
    """
    a = np.zeros((3, 3))
    for y, wi in zip(ref, w):
        a += wi * (np.dot(y, y) * np.eye(3) - np.outer(y, y))
    evals, evecs = np.linalg.eigh(a)
    keep = evals > tol * max(float(evals[-1]), tol)
    inv = np.where(keep, 1.0 / np.where(keep, evals, 1.0), 0.0)
    return (evecs * inv) @ evecs.T, int(keep.sum())


def build_coordinates(
    positions,
    atomic_numbers,
    *,
    interfragment: str = "tric",
    bond_tolerance: float = 1.3,
    linear_threshold_deg: float = 175.0,
    near_linear_deg: float = 165.0,
    hbond_cutoff: float = 2.5,
    hbond_angle_deg: float = 90.0,
    include_oop: bool = True,
    max_aux_per_edge: int = 3,
) -> CoordinateSpec:
    """Build a redundant internal coordinate set for one topology.

    Host-side NumPy only; called once per topology (and again only when the
    coordinate set degenerates during an optimization).

    Args:
        positions: ``(n_atoms, 3)`` reference positions in Angstrom.
        atomic_numbers: ``(n_atoms,)`` atomic numbers.
        interfragment: How to connect disconnected fragments.  ``"tric"``
            (default) adds 3 translation and up to 3 rotation coordinates per
            fragment (Wang & Song 2016) — the only scheme whose coordinate
            count stays ``O(N)`` for many fragments.  ``"aux"`` adds
            Baker/Bakken-Helgaker auxiliary stretches, adequate for a
            two-fragment complex.  ``"hbond"`` adds hydrogen bonds only, which
            is *not* complete for fragments without one.  ``"none"`` adds
            nothing (single-molecule systems).
        bond_tolerance: Covalent-radius tolerance for bond detection.
        linear_threshold_deg: At or above this an angle is treated as linear
            and gets a linear-bend pair instead of an ordinary bend.
        near_linear_deg: Angles in ``[near_linear_deg, linear_threshold_deg)``
            get *both*, so the set stays complete if the angle straightens
            during the optimization.
        hbond_cutoff: Maximum ``H...A`` distance in Angstrom.
        hbond_angle_deg: Minimum ``D-H...A`` angle in degrees.
        include_oop: Add Wilson out-of-plane bends at 3-coordinate atoms.
        max_aux_per_edge: Cap on auxiliary stretches per fragment-graph edge.

    Returns:
        A :class:`CoordinateSpec`.  ``valid`` is False (with ``reason`` set)
        when the system is too small or the coordinate set cannot reach the
        expected rank, in which case the caller must use Cartesians.

    Example:
        >>> spec = build_coordinates(positions, numbers)
        >>> spec.n_internal, spec.rank_at_build, spec.expected_rank
        (98, 60, 60)
    """
    pos = np.asarray(positions, dtype=float)
    z = np.asarray(atomic_numbers, dtype=int)
    n_atoms = int(pos.shape[0])

    def _empty(reason):
        return CoordinateSpec(
            n_atoms=n_atoms,
            n_internal=0,
            stretch_idx=np.zeros((0, 2), int),
            stretch_r0=np.zeros(0),
            bend_idx=np.zeros((0, 3), int),
            bend_has_linear=np.zeros(0, bool),
            linear_idx=np.zeros((0, 3), int),
            linear_axes=np.zeros((0, 2, 3)),
            torsion_idx=np.zeros((0, 4), int),
            oop_idx=np.zeros((0, 4), int),
            frag_atoms=np.zeros((0, 1), int),
            frag_w=np.zeros((0, 1)),
            rot_frag=np.zeros(0, int),
            rot_pinv=np.zeros((0, 3, 3)),
            coord_type=np.zeros(0, int),
            is_periodic=np.zeros(0, bool),
            expected_rank=0,
            rank_at_build=0,
            is_linear_system=False,
            fragments=[],
            n_aux_bonds=0,
            valid=False,
            reason=reason,
        )

    if n_atoms < 3:
        return _empty("too_few_atoms")

    bonds, _ = detect_bonds(pos, z, bond_tolerance)
    bond_arr = np.asarray(bonds, dtype=int).reshape(-1, 2)
    fragments = _connected_components(n_atoms, [tuple(b) for b in bond_arr])
    bond_set = {(int(min(i, j)), int(max(i, j))) for i, j in bond_arr}

    # --- interfragment stretches -------------------------------------------
    aux: List[Tuple[int, int]] = []
    if len(fragments) > 1:
        if interfragment in ("aux", "hbond"):
            if interfragment == "aux":
                aux += _interfragment_bonds(pos, fragments, max_aux_per_edge)
            aux += _hbond_stretches(pos, z, bond_set, hbond_cutoff, hbond_angle_deg)
        elif interfragment == "tric":
            # Rigid-body coordinates carry the interfragment degrees of
            # freedom; the hydrogen bonds are added on top, redundantly.
            aux += _hbond_stretches(pos, z, bond_set, hbond_cutoff, hbond_angle_deg)
    aux = [p for p in sorted(set(aux)) if p not in bond_set]
    n_aux = len(aux)

    all_bonds = np.concatenate([bond_arr, np.asarray(aux, int).reshape(-1, 2)], axis=0)
    all_bonds = all_bonds.astype(int).reshape(-1, 2)

    # --- angles and torsions over the augmented bond graph -----------------
    _, angles_all, torsions_all = build_bonded_lists(pos, z, bonds=[tuple(b) for b in all_bonds])

    bends, linear, bend_has_lin = [], [], []
    for i, j, k in angles_all:
        th = _angle_deg(pos, i, j, k)
        if th < near_linear_deg:
            bends.append((i, j, k))
            bend_has_lin.append(False)
        elif th < linear_threshold_deg:
            # Redundant on purpose: the ordinary bend can go singular later,
            # and the pair is already there to carry the degrees of freedom.
            bends.append((i, j, k))
            bend_has_lin.append(True)
            linear.append((i, j, k))
        else:
            linear.append((i, j, k))

    linear_axes = np.zeros((len(linear), 2, 3))
    for n, (i, j, k) in enumerate(linear):
        linear_axes[n] = _orthonormal_pair(pos[k] - pos[i])

    torsions = [
        (i, j, k, l)
        for i, j, k, l in torsions_all
        if _angle_deg(pos, i, j, k) < linear_threshold_deg
        and _angle_deg(pos, j, k, l) < linear_threshold_deg
    ]

    oop = []
    if include_oop:
        nbrs = {a: [] for a in range(n_atoms)}
        for i, j in all_bonds:
            nbrs[int(i)].append(int(j))
            nbrs[int(j)].append(int(i))
        for j in range(n_atoms):
            if len(nbrs[j]) == 3:
                i, k, l = sorted(nbrs[j])
                oop.append((i, j, k, l))

    # --- TRIC fragments -----------------------------------------------------
    if interfragment == "tric" and len(fragments) > 1:
        tric_frags = fragments
    else:
        tric_frags = []
    n_frag = len(tric_frags)
    max_frag = max((len(f) for f in tric_frags), default=1)
    frag_atoms = np.zeros((n_frag, max_frag), dtype=int)
    frag_w = np.zeros((n_frag, max_frag))
    rot_frag, rot_pinv = [], []
    for fi, frag in enumerate(tric_frags):
        frag_atoms[fi, : len(frag)] = frag
        frag_w[fi, : len(frag)] = 1.0 / len(frag)
        ref = pos[frag] - pos[frag].mean(axis=0)
        pinv, rank = _rot_metric(ref, np.full(len(frag), 1.0 / len(frag)))
        if rank > 0:
            rot_frag.append(fi)
            rot_pinv.append(pinv)
    rot_frag = np.asarray(rot_frag, dtype=int)
    rot_pinv = np.asarray(rot_pinv).reshape(-1, 3, 3)

    stretch_idx = all_bonds
    bend_idx = np.asarray(bends, int).reshape(-1, 3)
    linear_idx = np.asarray(linear, int).reshape(-1, 3)
    torsion_idx = np.asarray(torsions, int).reshape(-1, 4)
    oop_idx = np.asarray(oop, int).reshape(-1, 4)

    counts = [
        (COORD_STRETCH, len(stretch_idx)),
        (COORD_BEND, len(bend_idx)),
        (COORD_LINEAR, 2 * len(linear_idx)),
        (COORD_TORSION, len(torsion_idx)),
        (COORD_OOP, len(oop_idx)),
        (COORD_TRANS, 3 * n_frag),
        (COORD_ROT, 3 * len(rot_frag)),
    ]
    coord_type = np.concatenate([np.full(c, t, dtype=int) for t, c in counts])
    is_periodic = coord_type == COORD_TORSION
    n_internal = int(coord_type.size)

    # --- expected rank and completeness ------------------------------------
    centred = pos - pos.mean(axis=0)
    _, sv, _ = np.linalg.svd(centred, full_matrices=False)
    is_linear_system = bool(n_atoms > 2 and sv[1] < 1e-4 * max(sv[0], 1e-12))
    expected_rank = 3 * n_atoms - (5 if is_linear_system else 6)

    spec = CoordinateSpec(
        n_atoms=n_atoms,
        n_internal=n_internal,
        stretch_idx=stretch_idx,
        stretch_r0=(
            np.linalg.norm(pos[stretch_idx[:, 1]] - pos[stretch_idx[:, 0]], axis=-1)
            if len(stretch_idx)
            else np.zeros(0)
        ),
        bend_idx=bend_idx,
        bend_has_linear=np.asarray(bend_has_lin, dtype=bool),
        linear_idx=linear_idx,
        linear_axes=linear_axes,
        torsion_idx=torsion_idx,
        oop_idx=oop_idx,
        frag_atoms=frag_atoms,
        frag_w=frag_w,
        rot_frag=rot_frag,
        rot_pinv=rot_pinv,
        coord_type=coord_type,
        is_periodic=is_periodic,
        expected_rank=expected_rank,
        rank_at_build=0,
        is_linear_system=is_linear_system,
        fragments=fragments,
        n_aux_bonds=n_aux,
        valid=True,
        reason=None,
    )

    if n_internal == 0:
        return spec._replace(valid=False, reason="no_internal_coordinates")

    fns = make_coordinate_fns(spec)
    xj = jnp.asarray(pos)
    ref0 = initial_frag_ref(spec, xj)
    b = np.asarray(fns.b_fn(xj, ref0))
    # Rank is measured after removing the global rigid-body modes, because
    # TRIC coordinates span those too and would otherwise inflate it to 3N.
    q_tr = np.asarray(global_tr_basis(xj))
    b_eff = b - (b @ q_tr) @ q_tr.T
    sv_b = np.linalg.svd(b_eff, compute_uv=False)
    rank = int(np.sum(sv_b > 1e-6 * max(sv_b[0], 1e-30))) if sv_b.size else 0
    spec = spec._replace(rank_at_build=rank)
    if rank < expected_rank:
        return spec._replace(valid=False, reason="rank_deficient")
    return spec


# ============================================================================
# Coordinate kernels
# ============================================================================


def _safe_norm(v, eps2: float = _EPS2):
    """Euclidean norm with a finite gradient at zero (the double-``where``)."""
    n2 = jnp.sum(v * v, axis=-1)
    safe = n2 > eps2
    n2s = jnp.where(safe, n2, 1.0)
    return jnp.where(safe, jnp.sqrt(n2s), 0.0)


def _unit(v, eps2: float = _EPS2):
    n = _safe_norm(v, eps2)
    safe = n > 0.0
    return jnp.where(safe, v / jnp.where(safe, n, 1.0)[..., None], 0.0)


def _q_stretch(p):
    return _safe_norm(p[1] - p[0])


def _q_bend(p):
    """Valence angle via ``atan2`` — stable at 0 and pi, unlike ``arccos``."""
    u, v = p[0] - p[1], p[2] - p[1]
    return jnp.arctan2(_safe_norm(jnp.cross(u, v)), jnp.dot(u, v))


def _q_linear(p, axes):
    """Co-linear bending pair: both components vanish smoothly at linearity."""
    a = _unit(p[0] - p[1])
    c = _unit(p[2] - p[1])
    s = a + c
    return jnp.stack([jnp.dot(axes[0], s), jnp.dot(axes[1], s)])


def _q_torsion(p):
    b1, b2, b3 = p[1] - p[0], p[2] - p[1], p[3] - p[2]
    n1, n2 = jnp.cross(b1, b2), jnp.cross(b2, b3)
    m = jnp.cross(n1, _unit(b2))
    return jnp.arctan2(jnp.dot(m, n2), jnp.dot(n1, n2))


def _q_oop(p):
    """Wilson out-of-plane angle of bond j->i against the j-k-l plane."""
    ji, jk, jl = _unit(p[0] - p[1]), _unit(p[2] - p[1]), _unit(p[3] - p[1])
    cross = jnp.cross(jk, jl)
    sin_phi = _safe_norm(cross)
    safe = sin_phi > 1e-6
    val = jnp.where(safe, jnp.dot(cross, ji) / jnp.where(safe, sin_phi, 1.0), 0.0)
    val = jnp.clip(val, -1.0 + 1e-8, 1.0 - 1e-8)
    return jnp.where(safe, jnp.arcsin(val), 0.0)


class CoordinateFunctions(NamedTuple):
    """Jit- and vmap-safe callables generated from a :class:`CoordinateSpec`.

    Attributes:
        q_fn: ``(positions, frag_ref) -> (n_internal,)`` coordinate values.
        b_fn: ``(positions, frag_ref) -> (n_internal, 3*n_atoms)`` Wilson B.
        dq_fn: ``(q_new, q_old) -> (n_internal,)`` difference with 2*pi
            wrapping applied to periodic coordinates.
        refresh_ref_fn: ``(positions, frag_ref) -> frag_ref``, refreshing the
            TRIC rotation reference of any fragment that has turned too far
            for the linearisation.
        spec: The spec these were generated from.
    """

    q_fn: Callable
    b_fn: Callable
    dq_fn: Callable
    refresh_ref_fn: Callable
    spec: CoordinateSpec


def wrap_periodic(dq: jnp.ndarray, is_periodic: jnp.ndarray) -> jnp.ndarray:
    """Wrap differences of periodic coordinates into ``(-pi, pi]``.

    Only *differences* need this: ``atan2`` makes the coordinate value
    discontinuous across the branch cut but leaves the derivative continuous,
    so B needs no wrapping.  ``round`` rather than ``mod`` keeps the branch
    symmetric about zero.

    Args:
        dq: ``(..., n_internal)`` differences of internal coordinates.
        is_periodic: ``(n_internal,)`` boolean mask (torsions).

    Returns:
        ``(..., n_internal)`` wrapped differences.
    """
    wrapped = dq - 2.0 * jnp.pi * jnp.round(dq / (2.0 * jnp.pi))
    return jnp.where(is_periodic, wrapped, dq)


def initial_frag_ref(spec: CoordinateSpec, positions: jnp.ndarray) -> jnp.ndarray:
    """Centroid-relative TRIC reference geometry, ``(n_frag, max_frag, 3)``."""
    if spec.frag_atoms.shape[0] == 0:
        return jnp.zeros((0, 1, 3), dtype=positions.dtype)
    atoms = jnp.asarray(spec.frag_atoms)
    w = jnp.asarray(spec.frag_w, dtype=positions.dtype)
    p = positions[atoms]
    c = jnp.sum(w[..., None] * p, axis=1, keepdims=True)
    return (p - c) * (w[..., None] > 0)


def make_coordinate_fns(spec: CoordinateSpec) -> CoordinateFunctions:
    """Generate jit/vmap-safe coordinate and B-matrix callables for a spec.

    Args:
        spec: Static coordinate definition from :func:`build_coordinates`.

    Returns:
        A :class:`CoordinateFunctions` bundle.  All index arrays are captured
        by closure, so ``jax.vmap(fns.q_fn, in_axes=(0, None))`` over a batch
        of geometries sharing this topology works unchanged.
    """
    n = spec.n_atoms
    s_idx = jnp.asarray(spec.stretch_idx)
    b_idx = jnp.asarray(spec.bend_idx)
    l_idx = jnp.asarray(spec.linear_idx)
    l_ax = jnp.asarray(spec.linear_axes)
    t_idx = jnp.asarray(spec.torsion_idx)
    o_idx = jnp.asarray(spec.oop_idx)
    f_atoms = jnp.asarray(spec.frag_atoms)
    f_w = jnp.asarray(spec.frag_w)
    r_frag = jnp.asarray(spec.rot_frag)
    r_pinv = jnp.asarray(spec.rot_pinv)
    periodic = jnp.asarray(spec.is_periodic)

    n_frag = spec.frag_atoms.shape[0]
    n_rot = spec.rot_frag.shape[0]

    def _rot_vectors(positions, frag_ref):
        """Linearised rotation vectors, ``(n_rot, 3)``; linear in *positions*."""
        if n_rot == 0:
            return jnp.zeros((0, 3), dtype=positions.dtype)
        atoms = f_atoms[r_frag]
        w = f_w[r_frag].astype(positions.dtype)
        ref = frag_ref[r_frag]
        p = positions[atoms]
        c = jnp.sum(w[..., None] * p, axis=1, keepdims=True)
        y = (p - c) * (w[..., None] > 0)
        b = jnp.sum(w[..., None] * jnp.cross(ref, y), axis=1)
        return jnp.einsum("fab,fb->fa", r_pinv.astype(positions.dtype), b)

    def q_fn(positions, frag_ref):
        parts = []
        if s_idx.shape[0]:
            parts.append(jax.vmap(_q_stretch)(positions[s_idx]))
        if b_idx.shape[0]:
            parts.append(jax.vmap(_q_bend)(positions[b_idx]))
        if l_idx.shape[0]:
            parts.append(jax.vmap(_q_linear)(positions[l_idx], l_ax).reshape(-1))
        if t_idx.shape[0]:
            parts.append(jax.vmap(_q_torsion)(positions[t_idx]))
        if o_idx.shape[0]:
            parts.append(jax.vmap(_q_oop)(positions[o_idx]))
        if n_frag:
            p = positions[f_atoms]
            w = f_w.astype(positions.dtype)
            parts.append(jnp.sum(w[..., None] * p, axis=1).reshape(-1))
        if n_rot:
            parts.append(_rot_vectors(positions, frag_ref).reshape(-1))
        if not parts:
            return jnp.zeros((0,), dtype=positions.dtype)
        return jnp.concatenate(parts)

    def _scatter(b, row0, idx, grads):
        """Place compact ``(n_coord, k, 3)`` gradients into the dense B."""
        rows = row0 + jnp.arange(idx.shape[0])
        return b.at[rows[:, None], idx].add(grads)

    def b_fn(positions, frag_ref):
        dt = positions.dtype
        b = jnp.zeros((spec.n_internal, n, 3), dtype=dt)
        row = 0
        # Each kernel touches at most four atoms, so jacrev of the *scalar*
        # over the compact block is one backward pass -- O(n_int) overall
        # instead of the O(n_int * 3N) a whole-vector Jacobian would cost.
        if s_idx.shape[0]:
            g = jax.vmap(jax.jacrev(_q_stretch))(positions[s_idx])
            b = _scatter(b, row, s_idx, g)
            row += s_idx.shape[0]
        if b_idx.shape[0]:
            g = jax.vmap(jax.jacrev(_q_bend))(positions[b_idx])
            b = _scatter(b, row, b_idx, g)
            row += b_idx.shape[0]
        if l_idx.shape[0]:
            g = jax.vmap(jax.jacrev(_q_linear))(positions[l_idx], l_ax)  # (nl, 2, 3, 3)
            nl = l_idx.shape[0]
            rows = row + jnp.arange(2 * nl).reshape(nl, 2)
            b = b.at[rows[..., None], l_idx[:, None, :]].add(g)
            row += 2 * nl
        if t_idx.shape[0]:
            g = jax.vmap(jax.jacrev(_q_torsion))(positions[t_idx])
            b = _scatter(b, row, t_idx, g)
            row += t_idx.shape[0]
        if o_idx.shape[0]:
            g = jax.vmap(jax.jacrev(_q_oop))(positions[o_idx])
            b = _scatter(b, row, o_idx, g)
            row += o_idx.shape[0]
        if n_frag:
            # d t_alpha / d x_{i beta} = w_i delta_{alpha beta} -- constant.
            w = f_w.astype(dt)
            eye = jnp.eye(3, dtype=dt)
            g = w[..., None, None] * eye  # (n_frag, max_frag, 3, 3)
            rows = row + jnp.arange(3 * n_frag).reshape(n_frag, 3)
            b = b.at[rows[:, :, None], f_atoms[:, None, :]].add(jnp.transpose(g, (0, 2, 1, 3)))
            row += 3 * n_frag
        if n_rot:
            # d v / d x_j = A^- w_j [y_j^ref]_x -- also constant in x, because
            # sum_i w_i y_i^ref = 0 kills the centroid term.
            atoms = f_atoms[r_frag]
            w = f_w[r_frag].astype(dt)
            ref = frag_ref[r_frag]
            zero = jnp.zeros_like(ref[..., 0])
            skew = jnp.stack(
                [
                    jnp.stack([zero, -ref[..., 2], ref[..., 1]], axis=-1),
                    jnp.stack([ref[..., 2], zero, -ref[..., 0]], axis=-1),
                    jnp.stack([-ref[..., 1], ref[..., 0], zero], axis=-1),
                ],
                axis=-2,
            )  # (n_rot, max_frag, 3, 3)
            g = jnp.einsum("fab,fkbc->fkac", r_pinv.astype(dt), skew) * w[..., None, None]
            rows = row + jnp.arange(3 * n_rot).reshape(n_rot, 3)
            b = b.at[rows[:, :, None], atoms[:, None, :]].add(jnp.transpose(g, (0, 2, 1, 3)))
            row += 3 * n_rot
        return b.reshape(spec.n_internal, 3 * n)

    def dq_fn(q_new, q_old):
        return wrap_periodic(q_new - q_old, periodic)

    def refresh_ref_fn(positions, frag_ref, threshold: float = 0.2):
        """Re-anchor fragments whose accumulated rotation left the linear regime."""
        if n_rot == 0:
            return frag_ref
        v = _rot_vectors(positions, frag_ref)
        too_far = jnp.linalg.norm(v, axis=-1) > threshold
        fresh = initial_frag_ref(spec, positions)
        mask = jnp.zeros((n_frag,), dtype=bool).at[r_frag].set(too_far)
        return jnp.where(mask[:, None, None], fresh, frag_ref)

    return CoordinateFunctions(q_fn, b_fn, dq_fn, refresh_ref_fn, spec)


# ============================================================================
# Generalized inverse and back-transformation
# ============================================================================


class GeneralizedInverse(NamedTuple):
    """Pseudo-inverse operators derived from one ``eigh`` of ``B^T B``.

    Attributes:
        b_pinv: ``(3N, n_internal)`` Moore-Penrose ``B^+``; also the
            back-transformation operator ``B^T G^-``.
        projector: ``(n_internal, n_internal)`` redundancy projector
            ``P = G G^- = B B^+``.
        rank: Numerical rank of B.
    """

    b_pinv: jnp.ndarray
    projector: jnp.ndarray
    rank: jnp.ndarray


def global_tr_basis(positions: jnp.ndarray) -> jnp.ndarray:
    """Orthonormal basis of the global translation/rotation subspace.

    Returns ``(3N, 6)``; columns spanning a direction the system does not have
    (the third rotation of a linear molecule, all rotations of a single atom)
    come back as zero, so the caller can use the result unconditionally.

    Args:
        positions: ``(n_atoms, 3)`` positions in Angstrom.

    Returns:
        ``(3*n_atoms, 6)`` orthonormal (or zero) columns.
    """
    n = positions.shape[0]
    dt = positions.dtype
    centred = positions - jnp.mean(positions, axis=0)
    cols = []
    for a in range(3):
        t = jnp.zeros((n, 3), dtype=dt).at[:, a].set(1.0)
        cols.append(t.reshape(-1))
    for a in range(3):
        axis = jnp.zeros((3,), dtype=dt).at[a].set(1.0)
        cols.append(jnp.cross(jnp.broadcast_to(axis, (n, 3)), centred).reshape(-1))
    basis = jnp.stack(cols, axis=1)

    # Modified Gram-Schmidt; a linearly dependent column normalises to zero.
    def _step(carry, i):
        q = carry
        v = basis[:, i]
        v = v - q @ (q.T @ v)
        nv = jnp.linalg.norm(v)
        v = jnp.where(nv > 1e-8, v / jnp.where(nv > 1e-8, nv, 1.0), 0.0)
        return q.at[:, i].set(v), None

    q0 = jnp.zeros_like(basis)
    q, _ = jax.lax.scan(_step, q0, jnp.arange(6))
    return q


def generalized_inverse(
    b_matrix: jnp.ndarray,
    rel_threshold: float = 1e-6,
    abs_floor: float = 1e-10,
    tr_basis: Optional[jnp.ndarray] = None,
) -> GeneralizedInverse:
    """Pseudo-inverse operators for a Wilson B matrix.

    Decomposes the ``(3N, 3N)`` matrix ``B^T B`` rather than the
    ``(n_internal, n_internal)`` matrix ``G = B B^T``.  Redundant coordinate
    sets have ``n_internal > 3N`` by construction, so this is the cheaper
    factorization, and it still yields every operator needed:
    ``B^+ = (B^T B)^+ B^T`` (back-transformation), ``(B^T)^+ = B^{+T}``
    (gradient transformation) and ``P = B B^+`` (redundancy projector).

    The cutoff is *relative* because the eigenvalues of ``G`` carry mixed
    units — stretch rows are dimensionless, bend rows scale as ``1/r``, TRIC
    translation rows as ``1/n_fragment`` — so an absolute threshold is
    fragile across system sizes.

    Args:
        b_matrix: ``(n_internal, 3N)`` Wilson B matrix.
        rel_threshold: Eigenvalues of ``B^T B`` below
            ``rel_threshold * lambda_max`` are treated as null.  1e-6 is right
            in float64; float32 would need ~1e-4, which is why the RIC path
            requires float64.
        abs_floor: Absolute floor on the eigenvalue cutoff.
        tr_basis: Optional ``(3N, 6)`` orthonormal global translation/rotation
            basis from :func:`global_tr_basis`.  When given, B is restricted
            to its orthogonal complement.  **This matters for TRIC:** the
            per-fragment translation and rotation coordinates span the
            *global* rigid-body modes as well, so without this the coordinate
            set has rank 3N, the model Hessian is null along those six
            directions, and the quasi-Newton step runs away along them.  For a
            single-fragment set B already annihilates them, so passing it is
            harmless and makes the null space exact rather than
            threshold-dependent.

    Returns:
        A :class:`GeneralizedInverse`.
    """
    if tr_basis is not None:
        # B_eff = B (I - Q Q^T); form M = B_eff^T B_eff without materialising
        # the projector twice.
        bq = b_matrix @ tr_basis
        b_eff = b_matrix - bq @ tr_basis.T
    else:
        b_eff = b_matrix
    m = b_eff.T @ b_eff
    m = 0.5 * (m + m.T)  # cheap insurance against scatter-add asymmetry
    evals, vecs = jnp.linalg.eigh(m)
    cut = jnp.maximum(rel_threshold * evals[-1], abs_floor)
    keep = evals > cut
    inv = jnp.where(keep, 1.0 / jnp.where(keep, evals, 1.0), 0.0)
    m_pinv = (vecs * inv) @ vecs.T
    b_pinv = m_pinv @ b_eff.T
    return GeneralizedInverse(
        b_pinv=b_pinv,
        projector=b_eff @ b_pinv,
        rank=jnp.sum(keep),
    )


def cartesian_to_internal_gradient(ginv: GeneralizedInverse, grad_cart: jnp.ndarray) -> jnp.ndarray:
    """Transform a Cartesian gradient to internals: ``g_q = (B^T)^+ g_x``."""
    return ginv.b_pinv.T @ grad_cart.reshape(-1)


class BackTransformResult(NamedTuple):
    """Outcome of the Pulay-Fogarasi back-transformation.

    Attributes:
        positions: ``(n_atoms, 3)`` new Cartesian positions in Angstrom.
        converged: True when the residual fell below the tolerance.
        n_iter: Micro-iterations taken.
        residual: Final ``max|dq_remaining|``.
        mode: 0 = converged, 1 = best iterate accepted, 2 = first-order
            fallback, 3 = bounded Cartesian fallback.
    """

    positions: jnp.ndarray
    converged: jnp.ndarray
    n_iter: jnp.ndarray
    residual: jnp.ndarray
    mode: jnp.ndarray


def _cap_step(dx, trust):
    """Scale a Cartesian displacement so no atom moves further than *trust*."""
    max_disp = jnp.max(_safe_norm(dx))
    return dx * jnp.minimum(1.0, trust / (max_disp + 1e-12))


def internal_to_cartesian(
    fns: CoordinateFunctions,
    positions: jnp.ndarray,
    frag_ref: jnp.ndarray,
    dq: jnp.ndarray,
    ginv: GeneralizedInverse,
    grad_cart: jnp.ndarray,
    *,
    trust_radius: float = 0.3,
    max_iter: int = 25,
    tol: float = 1e-6,
    dx_tol: float = 1e-7,
) -> BackTransformResult:
    """Back-transform an internal step to Cartesians (Pulay-Fogarasi).

    Iterates ``x <- x + B^+ dq_remaining`` with ``B`` and ``B^+`` frozen at the
    input geometry, as Pulay-Fogarasi, Gaussian and geomeTRIC all do —
    recomputing them each micro-iteration saves an iteration or two and costs a
    full B build plus factorization every time.

    Formulated on a *remaining difference* rather than an absolute target, so
    no absolute ``q_target`` is ever formed for a periodic coordinate: each
    micro-step is tiny, so wrapping ``q(x_{j+1}) - q(x_j)`` is unambiguous,
    whereas wrapping against a distant target could pick the wrong branch.

    The best iterate is tracked and returned: a diverging Pulay loop typically
    improves for two or three iterations and then blows up, so keeping the best
    turns a hard failure into a small step.

    Note:
        *dq* must already be projected with ``ginv.projector``.  An unprojected
        step has components outside the row space of B that are simply
        unreachable, and the iteration then chases an impossible target — the
        residual is then two orders of magnitude worse.

    Args:
        fns: Coordinate callables from :func:`make_coordinate_fns`.
        positions: ``(n_atoms, 3)`` starting positions in Angstrom.
        frag_ref: TRIC rotation reference for this iteration.
        dq: ``(n_internal,)`` requested (projected) internal displacement.
        ginv: Pseudo-inverse operators evaluated at *positions*.
        grad_cart: ``(3N,)`` Cartesian gradient, for the last-resort fallback.
        trust_radius: Cap on the per-atom Cartesian displacement in Angstrom.
        max_iter: Maximum micro-iterations (default 25; typical need is 3-8).
        tol: Tolerance on ``max|dq_remaining|`` below which the iteration is
            declared fully converged.  Note that a *finite* step can never
            drive this to zero: the coordinate manifold curves away from the
            projector evaluated at the starting point, so the residual floors
            out at roughly 2% of the step size.  Success is therefore judged
            on whether the iteration made progress and settled in Cartesian
            space, not on reaching *tol*.
        dx_tol: Per-atom displacement below which the Cartesian iteration is
            considered settled.  Reaching it early is the normal exit when the
            remaining residual lies outside the row space of B.

    Returns:
        A :class:`BackTransformResult`.
    """
    shape = positions.shape
    q0 = fns.q_fn(positions, frag_ref)

    def cond(c):
        _x, _dq, j, err, _bx, _be, dx_norm = c
        # The dx test is not a nicety: when the residual lies in the null
        # space of B^+ the iteration stalls, and without it every such step
        # would burn all max_iter micro-iterations.
        return jnp.logical_and(j < max_iter, jnp.logical_and(err > tol, dx_norm > dx_tol))

    def body(c):
        x, dq_rem, j, _err, best_x, best_err, _dxn = c
        dx = (ginv.b_pinv @ dq_rem).reshape(shape)
        x_new = x + dx
        achieved = fns.dq_fn(fns.q_fn(x_new, frag_ref), fns.q_fn(x, frag_ref))
        dq_new = dq_rem - achieved
        err = jnp.max(jnp.abs(dq_new))
        better = jnp.logical_and(err < best_err, jnp.all(jnp.isfinite(x_new)))
        best_x = jnp.where(better, x_new, best_x)
        best_err = jnp.where(better, err, best_err)
        return (x_new, dq_new, j + 1, err, best_x, best_err, jnp.max(jnp.abs(dx)))

    init = (
        positions,
        dq,
        jnp.zeros((), jnp.int32),
        jnp.max(jnp.abs(dq)),
        positions,
        jnp.asarray(jnp.inf, dtype=positions.dtype),
        jnp.asarray(1.0, dtype=positions.dtype),
    )
    init_err = jnp.max(jnp.abs(dq))
    _x, _dqr, n_iter, _err, best_x, best_err, dx_norm = jax.lax.while_loop(cond, body, init)

    finite = jnp.all(jnp.isfinite(best_x))
    # Progress, not perfection: a finite step's residual floors out at the
    # curvature of the coordinate manifold, so requiring best_err < tol would
    # reject every correct back-transformation of a realistic step.
    made_progress = best_err < init_err
    ok = jnp.logical_and(finite, made_progress)
    settled = dx_norm <= dx_tol

    dx_first = _cap_step((ginv.b_pinv @ dq).reshape(shape), trust_radius)
    first_finite = jnp.all(jnp.isfinite(dx_first))
    gmax = jnp.max(jnp.abs(grad_cart)) + 1e-12
    dx_cart = -trust_radius * grad_cart.reshape(shape) / gmax

    x_out = jnp.where(
        ok,
        best_x,
        jnp.where(first_finite, positions + dx_first, positions + dx_cart),
    )
    mode = jnp.where(
        ok,
        jnp.where(settled, 0, 1),
        jnp.where(first_finite, 2, 3),
    )
    return BackTransformResult(
        positions=x_out,
        converged=jnp.logical_and(ok, settled),
        n_iter=n_iter,
        residual=best_err,
        mode=mode,
    )


def cartesian_spec(n_atoms: int) -> CoordinateSpec:
    """A degenerate 'coordinate set' that *is* Cartesian coordinates.

    ``q(x) = x.reshape(-1)`` and ``B = I``.  This exists so the same
    quasi-Newton stepper can be run in Cartesians and in internals with
    nothing else changed, which is what makes the coordinate-system versus
    initial-Hessian comparison a controlled one rather than a comparison of
    two different optimizers.
    """
    n3 = 3 * n_atoms
    return CoordinateSpec(
        n_atoms=n_atoms,
        n_internal=n3,
        stretch_idx=np.zeros((0, 2), int),
        stretch_r0=np.zeros(0),
        bend_idx=np.zeros((0, 3), int),
        bend_has_linear=np.zeros(0, bool),
        linear_idx=np.zeros((0, 3), int),
        linear_axes=np.zeros((0, 2, 3)),
        torsion_idx=np.zeros((0, 4), int),
        oop_idx=np.zeros((0, 4), int),
        frag_atoms=np.zeros((0, 1), int),
        frag_w=np.zeros((0, 1)),
        rot_frag=np.zeros(0, int),
        rot_pinv=np.zeros((0, 3, 3)),
        coord_type=np.full(n3, -1, dtype=int),
        is_periodic=np.zeros(n3, bool),
        expected_rank=max(n3 - 6, 0),
        rank_at_build=max(n3 - 6, 0),
        is_linear_system=False,
        fragments=[list(range(n_atoms))],
        n_aux_bonds=0,
        valid=n_atoms >= 2,
        reason=None if n_atoms >= 2 else "too_few_atoms",
    )


def cartesian_coordinate_fns(n_atoms: int) -> CoordinateFunctions:
    """Coordinate callables for :func:`cartesian_spec`."""
    spec = cartesian_spec(n_atoms)
    n3 = 3 * n_atoms

    def q_fn(positions, frag_ref):
        return positions.reshape(-1)

    def b_fn(positions, frag_ref):
        return jnp.eye(n3, dtype=positions.dtype)

    def dq_fn(q_new, q_old):
        return q_new - q_old

    def refresh_ref_fn(positions, frag_ref, threshold: float = 0.2):
        return frag_ref

    return CoordinateFunctions(q_fn, b_fn, dq_fn, refresh_ref_fn, spec)
