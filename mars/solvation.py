"""
Solvation shell builder for MARS.

Places explicit solvent molecules around a solute in concentric layers using
Fibonacci-sphere sampling with van der Waals overlap rejection.

Layer-by-layer strategy
-----------------------
Each layer is built by treating the full system so far (solute + inner shells)
as the fixed reference and packing a fresh ring of solvent around it.  This
produces well-separated concentric solvation shells.

Between layers the solvent can optionally be relaxed with MARS potentials
while the inner atoms are frozen via gradient masking.

Automatic solvation
-------------------
:func:`auto_solvate` grows shells automatically until a convergence criterion
is met — either a target **padding distance** (Å of solvent around the solute)
or a target **total number of solvent molecules**.  The number of molecules per
layer is estimated from the surface area of the current system.

Solvent geometries
------------------
Pre-built solvent geometries live in :mod:`mars.solvents` (one XYZ file per
solvent in ``mars/solvents/library/``). The library currently ships ~30+
common solvents — water, alcohols (methanol/ethanol/propanol/butanol/...),
acetonitrile, acetone, halogenated (chloroform, DCM, CCl4), ethers (THF,
dioxane, diethyl ether), aromatics (benzene, toluene, pyridine), amides
(DMF, NMP, formamide), DMSO, n-alkanes, and several specialty solvents.
Use ``mars.solvents.list_solvents()`` for the full list, or
``register_solvent(...)`` to add your own.

Quick start
-----------
    from mars.utils import load_structure
    from mars.solvation import auto_solvate, solvate

    solute = load_structure("molecule.xyz")

    # Automatic: grow water shells until 5 Å padding is reached
    solvated = auto_solvate(solute, "water", padding=5.0)

    # Automatic: grow until at least 30 molecules are placed
    solvated = auto_solvate(solute, "water", n_molecules=30)

    # Manual: explicit layer counts
    solvated = solvate(solute, "water", layers=[6, 12])
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation

from .solvents import (  # noqa: F401  (get_solvent re-exported via mars.__init__)
    SOLVENT_LIBRARY,
    get_solvent,
    load_solvent,
)

# Backwards-compatible alias for code that imported ``SOLVENT_DB`` from this
# module. Prefer ``mars.solvents.SOLVENT_LIBRARY`` in new code.
SOLVENT_DB = SOLVENT_LIBRARY


# ============================================================================
# Structure dict helpers
# ============================================================================


def _symbols_to_numbers(symbols: List[str]) -> np.ndarray:
    from .utils import symbols_to_numbers

    return symbols_to_numbers(symbols)


def _make_struct(positions: np.ndarray, symbols: List[str]) -> Dict:
    """Create a MARS-compatible structure dict from numpy arrays."""
    import jax.numpy as jnp

    return {
        "positions": jnp.array(positions, dtype=jnp.float32),
        "symbols": list(symbols),
        "numbers": jnp.array(_symbols_to_numbers(symbols)),
    }


def _struct_pos(struct: Dict) -> np.ndarray:
    """Return positions as a plain numpy float64 array."""
    return np.array(struct["positions"], dtype=np.float64)


def _struct_sym(struct: Dict) -> List[str]:
    return list(struct["symbols"])


# ============================================================================
# Geometry helpers (pure numpy)
# ============================================================================


def _vdw(symbol: str) -> float:
    from .utils import get_vdw_radius

    return get_vdw_radius(symbol)


def _bounding_radius(pos: np.ndarray, syms: List[str]) -> float:
    """Bounding radius (Å): max over atoms of ``|p_i - centroid| + vdw_i``."""
    centroid = pos.mean(axis=0)
    return max(np.linalg.norm(p - centroid) + _vdw(s) for p, s in zip(pos, syms))


def _effective_radius(pos: np.ndarray, syms: List[str]) -> float:
    """Cross-sectional packing radius (shape-aware, not a sphere).

    Returns the maximum perpendicular distance of any atom (plus its vdW radius)
    to the molecule's **longest** principal axis. For a roughly spherical
    molecule this ≈ the bounding radius, but for a long thin solvent it is the
    molecular *width* (cross-section), not the *length* — so candidate spacing
    and shell offsets stay tight and elongated solvents pack along the surface
    rather than being held a full molecular-length apart.
    """
    pos = np.asarray(pos, dtype=float)
    if len(pos) <= 1:
        return _vdw(syms[0]) if syms else 0.0
    d = pos - pos.mean(axis=0)
    cov = d.T @ d
    axis = np.linalg.eigh(cov)[1][:, -1]  # eigenvector of largest eigenvalue
    perp = d - np.outer(d @ axis, axis)  # component perpendicular to the long axis
    perp_dist = np.linalg.norm(perp, axis=1)
    return float(max(pd + _vdw(s) for pd, s in zip(perp_dist, syms)))


def _fibonacci_sphere(n: int) -> np.ndarray:
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    pts = np.empty((n, 3))
    for i in range(n):
        theta = math.acos(1.0 - 2.0 * (i + 0.5) / n)
        phi = 2.0 * math.pi * i / golden
        pts[i] = (math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta))
    return pts


def _has_overlap(
    new_pos: np.ndarray,
    new_sym: List[str],
    ref_pos: np.ndarray,
    ref_sym: List[str],
    scale: float = 0.75,
) -> bool:
    for pn, sn in zip(new_pos, new_sym):
        rn = _vdw(sn)
        dists = np.linalg.norm(ref_pos - pn, axis=1)
        for d, se in zip(dists, ref_sym):
            if d < scale * (rn + _vdw(se)):
                return True
    return False


def _min_dist(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    dmin = np.inf
    for pa in pos_a:
        d = np.linalg.norm(pos_b - pa, axis=1).min()
        if d < dmin:
            dmin = d
    return float(dmin)


def _check_min_dist(pos_a: np.ndarray, pos_b: np.ndarray, min_d: float) -> bool:
    for pa in pos_a:
        if np.linalg.norm(pos_b - pa, axis=1).min() < min_d:
            return False
    return True


# ============================================================================
# Topology integrity (post-relaxation solvent-molecule check)
# ============================================================================


def _struct_numbers(struct: Dict) -> np.ndarray:
    """Atomic numbers of a structure dict as a plain numpy int array."""
    nums = struct.get("numbers")
    if nums is None:
        return _symbols_to_numbers(list(struct["symbols"]))
    return np.asarray(nums, dtype=int)


def _solvent_ref_bonds(solvent_data: Dict, tolerance: float):
    """Reference intramolecular bond set of one solvent molecule.

    Returns ``(bond_set, n_atoms)`` where ``bond_set`` holds ``(i, j)`` pairs
    (local indices, ``i < j``) detected on the reference geometry with
    :func:`mars.utils.detect_bonds`.
    """
    from .utils import detect_bonds

    pos = np.asarray(solvent_data["positions"], dtype=float)
    nums = _struct_numbers(solvent_data)
    bonds, _ = detect_bonds(pos, nums, tolerance=tolerance)
    return {(int(i), int(j)) for i, j in bonds}, len(solvent_data["symbols"])


def _find_broken_solvent(positions, numbers, n_solute, nm, ref_bonds, tolerance):
    """Indices of solvent molecules whose covalent topology changed.

    A molecule (block of *nm* atoms after the first *n_solute* solute atoms) is
    flagged **broken** when either:

    * a reference intramolecular bond is missing — a bond stretched past the
      covalent threshold or the molecule fragmented / an atom dissociated; or
    * one of its atoms formed a covalent bond to another molecule or to the
      solute — it reacted.

    Hydrogen bonds (e.g. O···H ≈ 1.8 Å) sit well above the covalent threshold
    and are not counted as bonds, so they do not trigger false positives.
    """
    from .utils import detect_bonds

    positions = np.asarray(positions, dtype=float)
    numbers = np.asarray(numbers, dtype=int)
    n_total = len(numbers)
    n_mol = (n_total - n_solute) // nm
    bonds, _ = detect_bonds(positions, numbers, tolerance=tolerance)
    bondset = {(int(i), int(j)) for i, j in bonds}  # i < j

    def mol_of(idx):
        return -1 if idx < n_solute else (idx - n_solute) // nm

    broken = set()
    # Reacted: a covalent bond crossing a molecule boundary (solvent–solvent
    # or solute–solvent). Only the solvent molecule(s) involved are flagged.
    for i, j in bondset:
        mi, mj = mol_of(i), mol_of(j)
        if mi != mj:
            if mi >= 0:
                broken.add(mi)
            if mj >= 0:
                broken.add(mj)
    # Fragmented: a reference intramolecular bond is no longer present.
    for m in range(n_mol):
        if m in broken:
            continue
        off = n_solute + m * nm
        for li, lj in ref_bonds:
            if (off + li, off + lj) not in bondset:
                broken.add(m)
                break
    return sorted(broken)


def _remove_solvent_molecules(system: Dict, n_solute: int, nm: int, broken) -> Dict:
    """Return a new system dict with the listed solvent molecule indices dropped."""
    if not broken:
        return system
    pos = _struct_pos(system)
    sym = _struct_sym(system)
    broken_set = set(broken)
    n_mol = (len(sym) - n_solute) // nm
    keep = list(range(n_solute))  # solute is always kept
    for m in range(n_mol):
        if m in broken_set:
            continue
        off = n_solute + m * nm
        keep.extend(range(off, off + nm))
    keep_arr = np.array(keep, dtype=int)
    return _make_struct(pos[keep_arr], [sym[i] for i in keep_arr])


def remove_broken_solvent(
    system: Dict, solvent: Union[str, Dict], n_solute: int, *, tolerance: float = 1.3
):
    """Remove solvent molecules whose covalent topology broke during relaxation.

    Detects, with :func:`_find_broken_solvent`, any solvent molecule that
    fragmented or reacted, and returns a copy of *system* with those molecules
    removed.

    Args:
        system:   Solvated MARS structure dict (solute first, then solvent).
        solvent:  Solvent name/path or data dict — defines the reference
                  topology and atom count per molecule.
        n_solute: Number of solute atoms (block boundary).
        tolerance: Covalent-bond tolerance for :func:`mars.utils.detect_bonds`
                  (a bond counts as broken past ``tolerance × (r_i + r_j)``).

    Returns:
        ``(clean_system, broken_indices)``.
    """
    solvent_data = load_solvent(solvent) if isinstance(solvent, str) else solvent
    ref_bonds, nm = _solvent_ref_bonds(solvent_data, tolerance)
    broken = _find_broken_solvent(
        _struct_pos(system), _struct_numbers(system), n_solute, nm, ref_bonds, tolerance
    )
    return _remove_solvent_molecules(system, n_solute, nm, broken), broken


# ============================================================================
# LayerSpec
# ============================================================================


@dataclass
class LayerSpec:
    """Parameters for one solvation layer.

    Attributes:
        n_solvent:       Solvent molecules in this layer.
        buffer:          Extra gap (Å) beyond vdW contact before placing candidates.
        vdw_scale:       vdW overlap rejection scale (< 1 = tighter packing).
        n_candidates:    Fibonacci-sphere candidates per atom per radial shell.
        n_shells:        Radial shells of candidates per atom.
        seed:            Random seed (each layer gets ``base_seed + layer_index``).
        min_solute_dist: Hard minimum atom–atom distance (Å) to inner system.
        min_solvent_dist:Hard minimum atom–atom distance (Å) between solvents.
        spread_weight:   Weight of angular spread vs proximity in greedy selection.
    """

    n_solvent: int
    buffer: float = 0.0
    vdw_scale: float = 0.75
    n_candidates: int = 200
    n_shells: int = 3
    seed: int = 42
    min_solute_dist: float = 0.0
    min_solvent_dist: float = 0.0
    spread_weight: float = 1.0


# ============================================================================
# Core placement — one layer
# ============================================================================


def _place_one_layer(inner_struct: Dict, solvent_data: Dict, spec: LayerSpec) -> Dict:
    """Overpack non-overlapping solvents around *inner_struct*, then greedily
    keep ``spec.n_solvent`` of them (close to the solute + well-spread).

    Candidate centres are sampled per inner atom (Fibonacci spheres offset
    outward), so the pool follows the molecular surface for solutes of any size.
    Returns a new structure dict: inner atoms + the selected solvents.
    """
    from .log import log_info

    rng = np.random.default_rng(spec.seed)

    solv_pos = solvent_data["positions"].astype(np.float64).copy()
    solv_sym = list(solvent_data["symbols"])

    # Centre the solvent at its own centroid for clean random rotations.
    solv_pos -= solv_pos.mean(axis=0)
    r_solvent = _bounding_radius(solv_pos, solv_sym)

    inner_pos = _struct_pos(inner_struct)
    inner_sym = _struct_sym(inner_struct)
    center = inner_pos.mean(axis=0)  # angular spread is measured around here

    all_cands = _generate_candidate_centres(
        inner_pos, inner_sym, r_solvent, spec.buffer, spec.n_shells, spec.n_candidates, rng
    )

    # --- attempt placement at each candidate ------------------------------
    placed_pos = list(inner_pos)
    placed_sym = list(inner_sym)
    placed_solv: list = []
    molecules: list = []  # (positions, symbols, min_dist_to_inner)

    for centre in all_cands:
        rot = Rotation.random(random_state=int(rng.integers(2**31)))
        trial_pos = rot.apply(solv_pos) + centre
        trial_sym = solv_sym

        if _has_overlap(
            trial_pos, trial_sym, np.array(placed_pos), placed_sym, scale=spec.vdw_scale
        ):
            continue

        if spec.min_solute_dist > 0.0:
            if not _check_min_dist(trial_pos, inner_pos, spec.min_solute_dist):
                continue

        if spec.min_solvent_dist > 0.0 and placed_solv:
            if not _check_min_dist(trial_pos, np.array(placed_solv), spec.min_solvent_dist):
                continue

        placed_pos.extend(trial_pos)
        placed_sym.extend(trial_sym)
        placed_solv.extend(trial_pos)
        molecules.append((trial_pos.copy(), list(trial_sym), _min_dist(trial_pos, inner_pos)))

    total = len(molecules)
    if total < spec.n_solvent:
        import warnings

        warnings.warn(
            f"Only packed {total} solvent molecules but {spec.n_solvent} "
            "requested. Try increasing n_candidates or n_shells.",
            RuntimeWarning,
            stacklevel=4,
        )
    log_info(f"  Packed {total} candidates, keeping {min(total, spec.n_solvent)}.")

    # --- greedy selection: close + well-spread ----------------------------
    if total == 0:
        selected = []
    elif spec.n_solvent == 1:
        selected = [min(molecules, key=lambda x: x[2])]
    else:
        # Solvent-molecule direction vectors measured from the solute centroid
        # (= *center*), so angular spread is around the solute.
        centroids = np.array([m[0].mean(axis=0) - center for m in molecules])
        distances = np.array([m[2] for m in molecules])
        d_range = max(distances.max() - distances.min(), 1e-12)
        proximity = 1.0 - (distances - distances.min()) / d_range

        sel_idx: list = [int(np.argmax(proximity))]
        sel_cen = [centroids[sel_idx[0]]]

        for _ in range(min(spec.n_solvent, total) - 1):
            best_score = -np.inf
            best_j = -1
            sel_arr = np.array(sel_cen)
            for j in range(total):
                if j in sel_idx:
                    continue
                cj = centroids[j]
                nj = np.linalg.norm(cj)
                if nj < 1e-12:
                    continue
                angles = []
                for sc in sel_arr:
                    ns = np.linalg.norm(sc)
                    if ns < 1e-12:
                        continue
                    cos_a = np.clip(np.dot(cj, sc) / (nj * ns), -1.0, 1.0)
                    angles.append(np.arccos(cos_a))
                min_ang = min(angles) if angles else math.pi
                score = proximity[j] + spec.spread_weight * min_ang / math.pi
                if score > best_score:
                    best_score = score
                    best_j = j
            sel_idx.append(best_j)
            sel_cen.append(centroids[best_j])

        selected = [molecules[i] for i in sel_idx]

    # --- build output structure -------------------------------------------
    final_pos = list(inner_pos)
    final_sym = list(inner_sym)
    for pos, sym, d in selected:
        final_pos.extend(pos)
        final_sym.extend(sym)
        log_info(f"    placed {sym[0]}-solvent  d_min = {d:.2f} Å")

    return _make_struct(np.array(final_pos), final_sym)


# ============================================================================
# Public placement API
# ============================================================================


def place_solvents_by_layers(
    solute: Dict,
    solvent: Union[str, Dict],
    layers: Union[List[int], List[LayerSpec]],
    buffer: float = 0.0,
    vdw_scale: float = 0.75,
    n_candidates: int = 200,
    n_shells: int = 3,
    seed: int = 42,
) -> Dict:
    """Build concentric solvation layers around *solute*.

    Each layer wraps around the full system so far (solute + inner solvents),
    producing naturally separated shells.

    Args:
        solute:       MARS structure dict of the bare solute.
        solvent:      Built-in name (``"water"``, ``"methanol"``, …) or a
                      raw data dict ``{"symbols": [...], "positions": ndarray}``.
        layers:       List of ints ``[n1, n2, …]`` (molecules per layer, sharing
                      global placement params), **or** a list of
                      :class:`LayerSpec` for per-layer control.
        buffer:       Extra gap (Å) beyond vdW contact (used when *layers* is
                      a list of ints).
        vdw_scale:    vdW overlap scale factor.
        n_candidates: Fibonacci candidates per atom per radial shell.
        n_shells:     Radial candidate shells per atom.
        seed:         Base random seed; layer *i* uses ``seed + i``.

    Returns:
        MARS structure dict: solute atoms first, then layer-1 solvents, …
    """
    from .log import log_header, log_info

    solvent_data = load_solvent(solvent) if isinstance(solvent, str) else solvent
    solvent_formula = "".join(
        f"{s}{solvent_data['symbols'].count(s)}" if solvent_data["symbols"].count(s) > 1 else s
        for s in dict.fromkeys(solvent_data["symbols"])
    )

    # Normalise to LayerSpec list
    specs: list[LayerSpec] = []
    for i, item in enumerate(layers):
        if isinstance(item, LayerSpec):
            specs.append(item)
        else:
            specs.append(
                LayerSpec(
                    n_solvent=int(item),
                    buffer=buffer,
                    vdw_scale=vdw_scale,
                    n_candidates=n_candidates,
                    n_shells=n_shells,
                    seed=seed + i,
                )
            )

    # Keep the solute in its input (lab) frame — not recentred.
    system = {k: v for k, v in solute.items()}
    n_solute = len(solute["symbols"])

    for i, spec in enumerate(specs):
        log_header(f"Solvation layer {i + 1}  " f"({spec.n_solvent} × {solvent_formula})")
        system = _place_one_layer(system, solvent_data, spec)
        n_solv = len(system["symbols"]) - n_solute
        log_info(
            f"  Layer {i + 1} done — total {len(system['symbols'])} atoms "
            f"({n_solute} solute + {n_solv} solvent)."
        )

    return system


def solvate(
    solute: Dict,
    solvent: Union[str, Dict] = "water",
    layers: Union[int, List[int], List[LayerSpec]] = 6,
    **kwargs,
) -> Dict:
    """Convenience wrapper for :func:`place_solvents_by_layers`.

    Args:
        solute:  MARS structure dict of the bare solute.
        solvent: Solvent name or raw data dict (default: ``"water"``).
        layers:  Molecules per layer — int (one shell) or list.
        **kwargs: Forwarded to :func:`place_solvents_by_layers`.

    Returns:
        Solvated MARS structure dict.

    Examples::

        solvated = solvate(mol, "water", layers=6)
        solvated = solvate(mol, "methanol", layers=[4, 8])
    """
    if isinstance(layers, int):
        layers = [layers]
    return place_solvents_by_layers(solute, solvent, layers, **kwargs)


# ============================================================================
# Automatic solvation — converge by padding or molecule count
# ============================================================================


def _generate_candidate_centres(
    inner_pos: np.ndarray,
    inner_sym: List[str],
    r_solvent: float,
    buffer: float,
    n_shells: int,
    n_candidates: int,
    rng,
) -> np.ndarray:
    """Per-atom surface-sampled candidate solvent centres.

    For each inner atom (vdW radius ``r_i``) and each radial layer
    ``L in 0..n_shells-1``, place ``n_candidates`` Fibonacci-sphere points at
    radius ``r_i + r_solvent + buffer + L * 2 * r_solvent`` around that atom.
    The union of per-atom spheres tracks the molecular surface (works on any
    geometry, including disconnected fragments). Near-duplicate centres (within
    one ``r_solvent`` of a kept centre) are merged, then the pool is shuffled
    with the seeded RNG.
    """
    inner_pos = np.asarray(inner_pos)
    fib_pts = _fibonacci_sphere(n_candidates)
    cands = []
    for p_atom, s_atom in zip(inner_pos, inner_sym):
        r_atom = _vdw(s_atom)
        for layer in range(n_shells):
            radius = r_atom + r_solvent + buffer + layer * 2.0 * r_solvent
            cands.append(p_atom + fib_pts * radius)
    cands = np.vstack(cands)

    # De-duplicate: drop centres within r_solvent of an already-kept centre.
    keep = np.ones(len(cands), dtype=bool)
    for i in range(len(cands)):
        if not keep[i]:
            continue
        d = np.linalg.norm(cands[i + 1 :] - cands[i], axis=1)
        keep[i + 1 :][d < r_solvent] = False
    cands = cands[keep]
    return cands[rng.permutation(len(cands))]


# ============================================================================
# Iterative fill→relax cover mode (shape-aware, multi-orientation)
# ============================================================================


def _shell_centres(solute_pos, solute_sym, r_pack, buffer, shell_idx, n_candidates, rng):
    """Candidate centres for ONE concentric shell, spaced by the shape-aware
    ``r_pack`` (cross-sectional radius) rather than the spherical bounding
    radius — so elongated solvents pack tightly."""
    fib = _fibonacci_sphere(n_candidates)
    cands = []
    for p, s in zip(solute_pos, solute_sym):
        radius = _vdw(s) + r_pack + buffer + shell_idx * 2.0 * r_pack
        cands.append(p + fib * radius)
    cands = np.vstack(cands)
    keep = np.ones(len(cands), dtype=bool)
    for i in range(len(cands)):
        if not keep[i]:
            continue
        dd = np.linalg.norm(cands[i + 1 :] - cands[i], axis=1)
        keep[i + 1 :][dd < r_pack] = False
    cands = cands[keep]
    return cands[rng.permutation(len(cands))]


def _fill_to_capacity(
    system,
    solvent_data,
    solute_pos,
    solute_sym,
    shell_idx,
    r_pack,
    *,
    buffer,
    vdw_scale,
    n_candidates,
    n_orient,
    min_solute_dist,
    min_solvent_dist,
    seed,
):
    """Place AS MANY solvents as fit (no clash) in one concentric shell around
    the solute, trying up to *n_orient* orientations per candidate (so a long
    solvent can find a tangent fit). Overlap is checked against the FULL current
    system, so this also fills gaps opened by a prior relaxation.

    Returns the list of ``(positions, symbols)`` newly placed (no greedy trim).
    """
    rng = np.random.default_rng(seed)
    solv0 = solvent_data["positions"].astype(np.float64).copy()
    solv0 -= solv0.mean(axis=0)
    solv_sym = list(solvent_data["symbols"])

    centres = _shell_centres(solute_pos, solute_sym, r_pack, buffer, shell_idx, n_candidates, rng)

    placed_arr = _struct_pos(system)
    placed_sym = _struct_sym(system)
    placed_solv = []  # solvent-only positions for the min-solvent-dist check
    molecules = []
    for centre in centres:
        chosen = None
        for _ in range(n_orient):
            rot = Rotation.random(random_state=int(rng.integers(2**31)))
            trial = rot.apply(solv0) + centre
            if _has_overlap(trial, solv_sym, placed_arr, placed_sym, scale=vdw_scale):
                continue
            if min_solute_dist > 0.0 and not _check_min_dist(trial, solute_pos, min_solute_dist):
                continue
            if (
                min_solvent_dist > 0.0
                and placed_solv
                and not _check_min_dist(trial, np.array(placed_solv), min_solvent_dist)
            ):
                continue
            chosen = trial
            break
        if chosen is None:
            continue
        placed_arr = np.vstack([placed_arr, chosen])
        placed_sym = placed_sym + solv_sym
        placed_solv.extend(chosen)
        molecules.append((chosen.copy(), list(solv_sym)))
    return molecules


def _solvate_cover(
    solute,
    solvent_data,
    *,
    padding,
    max_shells,
    buffer,
    vdw_scale,
    n_candidates,
    n_orient,
    max_refill,
    min_solute_dist,
    min_solvent_dist,
    relax_fn,
    seed,
    n_target=None,
    init_system=None,
    init_count=0,
    topology_repair=False,
    ref_bonds=None,
    topo_tolerance=1.3,
):
    """Iterative fill→relax cover build (dense, shape-aware).

    For each concentric shell (inner→outer): fill as many solvents as fit, relax
    (if *relax_fn* given), then refill — the relaxation opens space — repeating
    until a fill adds nothing (the shell is saturated), then advance.

    Stopping:
      * *n_target* given — grow shells until exactly *n_target* solvents are
        placed (the last fill is trimmed to the molecules closest to the solute);
      * *padding* given — grow saturated shells until the shell offset reaches it;
      * neither — stop after the first saturated contact shell (covers the solute).

    *relax_fn(system, n_solute)* returns a relaxed structure with the first
    *n_solute* atoms (the solute) frozen; pass ``None`` for pure-geometry fill.

    *init_system*/*init_count* seed the build from an existing solvated system
    (instead of the bare solute) with *init_count* solvent molecules already
    placed — used by the topology-repair loop to refill after removing broken
    molecules. The shell fill skips occupied space (overlap-checked) and fills
    the freed gaps plus outer shells until the target is reached.

    When *topology_repair* is set (with *ref_bonds* = the reference solvent bond
    set), each shell's relaxation is followed by a topology check: any solvent
    molecule that fragmented or reacted is removed in place, and the refill
    cycle / next shell re-places it — so broken molecules are caught layerwise
    during the build rather than only at the end.
    """
    from .log import log_header, log_info

    solute_pos = _struct_pos(solute)
    solute_sym = _struct_sym(solute)
    n_solute = len(solute_sym)
    r_pack = _effective_radius(
        solvent_data["positions"] - np.asarray(solvent_data["positions"]).mean(axis=0),
        list(solvent_data["symbols"]),
    )
    atoms_per_mol = len(solvent_data["symbols"])

    log_header("Layerwise cover solvation")
    log_info(f"  Solute        : {n_solute} atoms")
    log_info(f"  Packing radius: r_pack = {r_pack:.2f} Å (shape-aware)")
    log_info(f"  Relax per fill: {'on' if relax_fn is not None else 'off (geometry only)'}")

    if init_system is not None:
        system = init_system
        total = init_count
    else:
        system = {k: v for k, v in solute.items()}
        total = 0
    reached_target = False
    for shell in range(max_shells):
        log_header(f"Cover shell {shell + 1}")
        shell_added = 0
        for cycle in range(max_refill):
            new = _fill_to_capacity(
                system,
                solvent_data,
                solute_pos,
                solute_sym,
                shell,
                r_pack,
                buffer=buffer,
                vdw_scale=vdw_scale,
                n_candidates=n_candidates,
                n_orient=n_orient,
                min_solute_dist=min_solute_dist,
                min_solvent_dist=min_solvent_dist,
                seed=seed + shell * 97 + cycle,
            )
            if not new:
                break
            # Count mode: keep only the molecules closest to the solute, up to N.
            if n_target is not None and total + len(new) > n_target:
                keep = n_target - total
                new.sort(key=lambda m: _min_dist(m[0], solute_pos))
                new = new[:keep]
                reached_target = True
            pos = list(_struct_pos(system))
            sym = _struct_sym(system)
            for p, s in new:
                pos.extend(p)
                sym.extend(s)
            system = _make_struct(np.array(pos), sym)
            shell_added += len(new)
            total += len(new)
            log_info(
                f"  shell {shell + 1} fill {cycle + 1}: +{len(new)} molecules "
                f"(total {total}, {len(system['symbols'])} atoms)"
            )
            if relax_fn is not None:
                system = relax_fn(system, n_solute)
                # Layerwise topology check: drop any solvent molecule that
                # fragmented or reacted during this relaxation. The next refill
                # cycle (or shell) re-places it, so the count is maintained.
                if topology_repair and ref_bonds is not None:
                    broken = _find_broken_solvent(
                        _struct_pos(system),
                        _struct_numbers(system),
                        n_solute,
                        atoms_per_mol,
                        ref_bonds,
                        topo_tolerance,
                    )
                    if broken:
                        system = _remove_solvent_molecules(system, n_solute, atoms_per_mol, broken)
                        total = (len(system["symbols"]) - n_solute) // atoms_per_mol
                        reached_target = n_target is not None and total >= n_target
                        log_info(
                            f"  shell {shell + 1} topology check: removed "
                            f"{len(broken)} broken molecule(s) (total {total})"
                        )
                    else:
                        log_info(f"  shell {shell + 1} topology check: {total} intact")
            if reached_target:
                break
        if reached_target:
            log_info(f"  reached target of {n_target} molecules.")
            break
        if shell_added == 0:
            log_info("  shell added nothing — stopping.")
            break
        if n_target is not None:
            continue  # need more molecules — grow another shell
        if padding is None:
            break  # one saturated contact shell covers the solute
        if (r_pack + shell * 2.0 * r_pack) >= padding:
            break

    if n_target is not None and total < n_target:
        log_info(
            f"  Note: placed {total}/{n_target} (geometry-limited at " f"max_shells={max_shells})."
        )

    log_info("")
    log_info(
        f"  Cover done: {total} solvent molecules "
        f"({n_solute} solute + {len(system['symbols']) - n_solute} solvent)."
    )
    return system


def _topology_repair_loop(
    solute,
    solvent_data,
    system,
    *,
    n_solute,
    n_target,
    relax_fn,
    max_repair,
    tolerance,
    padding,
    max_shells,
    buffer,
    vdw_scale,
    n_candidates,
    n_orient,
    max_refill,
    seed,
):
    """Remove relaxation-broken solvent molecules and refill until converged.

    After the relaxed build, repeatedly: detect solvent molecules whose
    covalent topology broke (fragmented or reacted), remove them, refill the
    freed space — back to ``n_target`` in count mode, or back to the previous
    count in cover mode — and re-relax. Stops when a pass finds no broken
    molecules (converged) or after *max_repair* passes (any still-broken
    molecules are then dropped from the final structure).
    """
    from .log import log_header, log_info, log_warning

    ref_bonds, nm = _solvent_ref_bonds(solvent_data, tolerance)
    log_header("Topology repair")

    converged = False
    for rep in range(max_repair):
        n_current = (len(system["symbols"]) - n_solute) // nm
        broken = _find_broken_solvent(
            _struct_pos(system), _struct_numbers(system), n_solute, nm, ref_bonds, tolerance
        )
        if not broken:
            log_info(f"  Pass {rep + 1}: all {n_current} solvent molecules intact — converged.")
            converged = True
            break
        log_info(
            f"  Pass {rep + 1}: {len(broken)}/{n_current} solvent molecule(s) broken "
            f"— removing and refilling."
        )
        system = _remove_solvent_molecules(system, n_solute, nm, broken)
        n_after = (len(system["symbols"]) - n_solute) // nm
        restore = n_target if n_target is not None else n_current
        system = _solvate_cover(
            solute,
            solvent_data,
            padding=padding,
            max_shells=max_shells,
            buffer=buffer,
            vdw_scale=vdw_scale,
            n_candidates=n_candidates,
            n_orient=n_orient,
            max_refill=max_refill,
            min_solute_dist=0.0,
            min_solvent_dist=0.0,
            relax_fn=relax_fn,
            seed=seed + 1000 * (rep + 1),
            n_target=restore,
            init_system=system,
            init_count=n_after,
        )

    if not converged:
        broken = _find_broken_solvent(
            _struct_pos(system), _struct_numbers(system), n_solute, nm, ref_bonds, tolerance
        )
        if broken:
            log_warning(
                f"  Topology not fully converged after {max_repair} passes — dropping "
                f"{len(broken)} still-broken molecule(s) from the final structure."
            )
            system = _remove_solvent_molecules(system, n_solute, nm, broken)
    n_final = (len(system["symbols"]) - n_solute) // nm
    log_info(f"  Final solvent count: {n_final}")
    return system


def auto_solvate(
    solute: Dict,
    solvent: Union[str, Dict] = "water",
    padding: Optional[float] = None,
    n_molecules: Optional[int] = None,
    max_layers: int = 10,
    optimise_each_layer: bool = False,
    potential_name: str = "so3lr",
    charge: float = 0.0,
    fmax: float = 0.05,
    maxiter: int = 10000,
    method: str = "FIRE",
    max_stepsize: float = 0.15,
    lr_cutoff: float = 1000.0,
    vdw_scale: float = 0.75,
    buffer: float = 0.0,
    n_candidates: int = 300,
    n_shells_per_atom: int = 3,
    n_orient: int = 8,
    refill_cycles: int = 3,
    seed: int = 42,
    topology_repair: bool = True,
    max_repair: int = 3,
    topo_tolerance: float = 1.3,
    potential_kwargs: Optional[dict] = None,
) -> Dict:
    """Solvate a solute.

    Both paths run through :func:`_solvate_cover` (dense, shape-aware shell
    fill): fill each concentric shell with as many solvents as fit (shape-aware
    spacing + ``n_orient`` orientation tries), relax between fills when
    ``optimise_each_layer`` is set, refill until the shell is saturated
    (``refill_cycles`` cap), then advance.

    * **Cover** (default, or ``padding`` given) — the default stops at the first
      saturated contact shell (which covers the solute); ``padding`` grows
      further saturated shells until that thickness is reached.
    * **Count** (``n_molecules`` given) — keeps growing shells until exactly
      ``n_molecules`` are placed, trimming the final fill to the molecules
      closest to the solute. The dense fill packs the count tightly against the
      surface (much closer than the old greedy-spread placement).

    Args:
        solute: MARS structure dict (``positions``, ``symbols``,
            ``numbers``).
        solvent: Built-in solvent name resolved through
            :func:`mars.solvents.get_solvent` (case-insensitive aliases
            work — ``"DCM"``, ``"EtOH"``, ``"DMSO"``, …) **or** a raw
            data dict with ``symbols`` and ``positions``. Default
            ``"water"``.
        padding: Target solvent-shell thickness (Å) in cover mode — grows
            saturated shells until reached. ``None`` = one contact shell.
        n_molecules: Exact solvent-molecule count (count mode); the dense
            shell fill grows until reached, then trims to the N closest.
        max_layers: Safety cap on the number of shells (default 10).
        optimise_each_layer: When ``True`` (``--opt-mode layerwise``), relax with
            the solute frozen — between fills in cover mode, or once after
            placement in count mode.
        potential_name: Potential backend name (passed to
            :func:`mars.potentials.get_potential`).
        charge: Total system charge.
        fmax: Force-convergence threshold in eV/Å for the per-layer
            relaxation (default 0.05).
        maxiter: Max optimizer iterations per layer (default 10000).
        method: Optimizer — ``"FIRE"`` (default), ``"LBFGS"``,
            ``"GD"``, or ``"HYBRID"`` (FIRE coarse → LBFGS fine).
        max_stepsize: Max per-atom step in Å — LBFGS step bound, SP trust
            radius; not used by FIRE (default 0.15).
        lr_cutoff: SO3LR long-range cutoff in Å (default 1000).
        vdw_scale: vdW overlap-rejection scale factor (default 0.75 —
            lower values give tighter packing / more vdW overlap in the
            pre-relax structure).
        buffer: Extra gap in Å beyond the vdW contact distance
            (default 0).
        n_candidates: Fibonacci candidates per atom per radial shell
            (default 300).
        n_shells_per_atom: Starting number of radial candidate shells per
            atom (default 3).
        n_orient: Orientation tries per candidate site in cover mode (default 8).
        refill_cycles: Max fill→relax→refill cycles per shell in cover mode
            (default 3).
        seed: Random seed for reproducibility.
        topology_repair: After the relaxed build, check each solvent molecule's
            covalent topology; remove any that fragmented or reacted, refill to
            the target, re-relax, and repeat until none are broken (default
            ``True``). Only runs when relaxation is enabled
            (``optimise_each_layer=True``).
        max_repair: Maximum remove→refill→relax repair passes (default 3).
        topo_tolerance: Covalent-bond tolerance for the topology check — a bond
            counts as broken past ``topo_tolerance × (r_i + r_j)`` (default 1.3).
        potential_kwargs: Extra keyword arguments forwarded to the
            potential constructor.

    Returns:
        Solvated MARS structure dict — solute atoms first, then placed
        solvent atoms in layer order.

    Example:
        >>> from mars import load_structure, auto_solvate
        >>> solute = load_structure("solute.xyz")
        >>> solvated = auto_solvate(
        ...     solute,
        ...     solvent="water",
        ...     padding=5.0,
        ...     potential_name="so3lr",
        ...     optimise_each_layer=True,
        ... )
        >>> from mars.utils import save_structure
        >>> save_structure("solvated.xyz", solvated)
    """
    if potential_kwargs is None:
        potential_kwargs = {}

    solvent_data = load_solvent(solvent) if isinstance(solvent, str) else solvent

    # Both count (n_molecules) and cover (default / padding) use the same dense,
    # shape-aware iterative fill→relax build. Count just caps the total at N
    # (keeping the molecules closest to the solute).
    relax_fn = None
    if optimise_each_layer:

        def relax_fn(sys_struct, n_frozen):
            return optimize_solvation_shell(
                sys_struct,
                n_solute=n_frozen,
                potential_name=potential_name,
                freeze_mode="solute",
                charge=charge,
                lr_cutoff=lr_cutoff,
                fmax=fmax,
                maxiter=maxiter,
                method=method,
                max_stepsize=max_stepsize,
                **potential_kwargs,
            )

    # Topology check runs layerwise (after each shell relax) during the build,
    # and again as a final convergence loop below. Both share this reference
    # bond set. Only meaningful when relaxation actually runs.
    do_repair = topology_repair and relax_fn is not None
    ref_bonds = _solvent_ref_bonds(solvent_data, topo_tolerance)[0] if do_repair else None

    n_target = int(n_molecules) if n_molecules is not None else None
    system = _solvate_cover(
        solute,
        solvent_data,
        padding=padding,
        max_shells=max_layers,
        buffer=buffer,
        vdw_scale=vdw_scale,
        n_candidates=n_candidates,
        n_orient=n_orient,
        max_refill=refill_cycles,
        min_solute_dist=0.0,
        min_solvent_dist=0.0,
        relax_fn=relax_fn,
        seed=seed,
        n_target=n_target,
        topology_repair=do_repair,
        ref_bonds=ref_bonds,
        topo_tolerance=topo_tolerance,
    )

    # Final convergence pass: after the whole build, repeat remove→refill→relax
    # until no molecule is broken (the layerwise check above catches most cases;
    # this guarantees the returned structure is clean even if the last shell's
    # relaxation broke something).
    if do_repair:
        system = _topology_repair_loop(
            solute,
            solvent_data,
            system,
            n_solute=len(solute["symbols"]),
            n_target=n_target,
            relax_fn=relax_fn,
            max_repair=max_repair,
            tolerance=topo_tolerance,
            padding=padding,
            max_shells=max_layers,
            buffer=buffer,
            vdw_scale=vdw_scale,
            n_candidates=n_candidates,
            n_orient=n_orient,
            max_refill=refill_cycles,
            seed=seed,
        )

    return system


# ============================================================================
# Solvent-shell optimisation (MARS optimizer + potentials)
# ============================================================================


def optimize_solvation_shell(
    solvated: Dict,
    n_solute: int,
    potential_wrapper=None,
    potential_name: str = "so3lr",
    freeze_mode: str = "solute",
    charge: float = 0.0,
    lr_cutoff: float = 1000.0,
    fmax: float = 0.05,
    maxiter: int = 10000,
    method: str = "FIRE",
    max_stepsize: float = 0.15,
    multistage: bool = True,
    **potential_kwargs,
) -> Dict:
    """Relax the solvent shell with the MARS JAX optimizer.

    Frozen atoms are handled via gradient masking (zero-force constraint) so
    the optimiser never moves them.  Solvated structure must follow the
    convention: solute atoms first, then solvent atoms.

    When *multistage* is True (default), optimisation runs in two passes:

    - **Stage 1 (coarse)**: 40 % of *maxiter* with 3× looser *fmax* and
      the full *max_stepsize*.  Quickly resolves large clashes.
    - **Stage 2 (fine)**: remaining iterations with target *fmax* and
      0.3× *max_stepsize*.  Refines positions precisely.

    With ``method="HYBRID"`` (the default), stage 1 uses **FIRE** (clash-safe,
    velocity-damped) and stage 2 uses **LBFGS** (faster final convergence).
    HYBRID always runs both stages regardless of *multistage*.

    Args:
        solvated:         MARS structure dict of the solvated cluster.
        n_solute:         Number of solute atoms (determines freeze boundary).
        potential_wrapper: Pre-built :class:`~mars.potentials.PotentialWrapper`.
                          If None, one is built from *potential_name* + kwargs.
        potential_name:   Potential to use when building from scratch
                          (``"so3lr"``, ``"mace"``, ``"dxtb"``).
        freeze_mode:      Which atoms to freeze: ``"solute"`` (default),
                          ``"solvent"``, or ``"none"``.
        charge:           Total system charge.
        lr_cutoff:        SO3LR long-range cutoff (Å).
        fmax:             Force convergence criterion (eV/Å).
        maxiter:          Maximum optimisation steps (total across both stages).
        method:           Optimizer: ``"FIRE"`` (default), ``"LBFGS"``, ``"GD"``.
        max_stepsize:     Max per-atom step (LBFGS step bound; not used by FIRE).
        multistage:       Use two-stage coarse → fine optimisation (default True).
        **potential_kwargs: Extra kwargs forwarded to the potential constructor.

    Returns:
        Updated MARS structure dict with optimised positions.
    """
    import jax.numpy as jnp

    from .auto_config import EV_TO_KCALMOL
    from .cli._constraints import make_frozen_energy_fn
    from .log import log_header, log_info, log_step_end, log_step_start
    from .optimizer import optimize_single
    from .potentials import get_potential

    positions = jnp.array(solvated["positions"])
    symbols = list(solvated["symbols"])
    numbers = jnp.array(solvated["numbers"])
    n_total = len(symbols)

    log_header("Solvation shell optimisation")
    log_info(f"  Potential     : {potential_name}")
    log_info(f"  Total atoms   : {n_total}  ({n_solute} solute + " f"{n_total - n_solute} solvent)")
    log_info(f"  Freeze mode   : {freeze_mode}")
    is_hybrid = method.upper() == "HYBRID"
    if is_hybrid:
        multistage = True
        coarse_method, fine_method = "FIRE", "LBFGS"
        log_info(
            f"  Method        : HYBRID (FIRE coarse → LBFGS fine),  "
            f"fmax = {fmax} eV/Å,  maxiter = {maxiter}"
        )
    else:
        coarse_method = fine_method = method
        log_info(f"  Method        : {method},  fmax = {fmax} eV/Å,  " f"maxiter = {maxiter}")
    if multistage:
        log_info(f"  Multi-stage   : ON (coarse → fine)")

    # Build potential if not supplied
    if potential_wrapper is None:
        kwargs_pot: dict = dict(potential_kwargs)
        kwargs_pot["species"] = numbers
        if potential_name == "so3lr":
            kwargs_pot.setdefault("lr_cutoff", lr_cutoff)
            kwargs_pot.setdefault("charge", charge)
        elif potential_name in ("mace", "dxtb"):
            kwargs_pot.setdefault("charge", charge)
        potential_wrapper = get_potential(potential_name, **kwargs_pot)

    potential_wrapper.initialize(positions)
    energy_fn_raw = potential_wrapper.build_energy_fn()

    # Determine frozen indices
    if freeze_mode == "solute":
        frozen = list(range(n_solute))
    elif freeze_mode == "solvent":
        frozen = list(range(n_solute, n_total))
    else:
        frozen = []

    if frozen:
        log_info(f"  Frozen atoms  : {len(frozen)}")

    def _apply_freeze(efn, pos):
        if frozen:
            return make_frozen_energy_fn(efn, pos, frozen)
        return efn

    t0 = log_step_start("Geometry optimisation")

    if not multistage:
        # Single-pass optimisation (original behaviour)
        energy_fn = _apply_freeze(energy_fn_raw, positions)
        pos_opt, energy_opt, info = optimize_single(
            positions,
            energy_fn,
            fmax=fmax,
            maxiter=maxiter,
            method=method,
            max_stepsize=max_stepsize,
            potential_wrapper=potential_wrapper,
        )
        total_iters = info["iterations"]
        converged = info["converged"]
    else:
        # --- Stage 1: coarse — resolve major clashes ---
        coarse_maxiter = max(10, int(maxiter * 0.4))
        coarse_fmax = fmax * 5.0
        coarse_stepsize = max_stepsize

        energy_fn_coarse = _apply_freeze(energy_fn_raw, positions)
        pos_opt, energy_opt, info1 = optimize_single(
            positions,
            energy_fn_coarse,
            fmax=coarse_fmax,
            maxiter=coarse_maxiter,
            method=coarse_method,
            max_stepsize=coarse_stepsize,
            potential_wrapper=potential_wrapper,
        )
        log_info(
            f"  Stage 1 ({coarse_method:>5}): {info1['iterations']} steps, "
            f"fmax = {coarse_fmax:.3f} eV/Å, "
            f"converged = {info1['converged']}"
        )

        # --- Stage 2: fine — precise refinement ---
        fine_maxiter = maxiter - info1["iterations"]
        fine_stepsize = max_stepsize * 0.3

        # Rebuild frozen constraint with updated positions
        energy_fn_fine = _apply_freeze(energy_fn_raw, pos_opt)
        pos_opt, energy_opt, info2 = optimize_single(
            pos_opt,
            energy_fn_fine,
            fmax=fmax,
            maxiter=fine_maxiter,
            method=fine_method,
            max_stepsize=fine_stepsize,
            potential_wrapper=potential_wrapper,
        )
        log_info(
            f"  Stage 2 ({fine_method:>5}): {info2['iterations']} steps, "
            f"fmax = {fmax:.3f} eV/Å, "
            f"converged = {info2['converged']}"
        )

        total_iters = info1["iterations"] + info2["iterations"]
        converged = info2["converged"]
        info = info2

    log_step_end("Geometry optimisation", t0)

    if converged:
        log_info(
            f"  Converged in {total_iters} steps  "
            f"E = {float(energy_opt) * EV_TO_KCALMOL:.3f} kcal/mol"
        )
    else:
        from .log import log_warning

        log_warning(
            f"  ⚠️  NOT CONVERGED after {total_iters} steps  "
            f"E = {float(energy_opt) * EV_TO_KCALMOL:.3f} kcal/mol  "
            f"Fmax = {info['max_force']:.4f} eV/Å"
        )
        log_warning(
            "  Consider increasing --maxiter, reducing --max-stepsize, "
            "or checking the input structure."
        )

    return {
        "positions": pos_opt,
        "symbols": symbols,
        "numbers": numbers,
        "energy": energy_opt,
    }


# ============================================================================
# Layer-by-layer placement + optimisation workflow
# ============================================================================


def build_and_optimise_layers(
    solute: Dict,
    solvent: Union[str, Dict] = "water",
    layers: Union[List[int], List[LayerSpec]] = None,
    optimise_each_layer: bool = True,
    potential_name: str = "so3lr",
    charge: float = 0.0,
    fmax: float = 0.05,
    maxiter: int = 50,
    method: str = "FIRE",
    max_stepsize: float = 0.15,
    lr_cutoff: float = 1000.0,
    placement_kwargs: Optional[dict] = None,
    potential_kwargs: Optional[dict] = None,
) -> Dict:
    """Place solvation layers and optionally optimise the solvent after each.

    Workflow per layer *i*:

    1. Place layer *i* around the full system built so far.
    2. Optimise the new solvent while freezing all atoms placed before it.

    Args:
        solute:              MARS structure dict of the bare solute.
        solvent:             Solvent name or raw data dict.
        layers:              Molecules per layer or list of :class:`LayerSpec`.
                             Default: ``[6]``.
        optimise_each_layer: Run MARS optimisation after each layer.
        potential_name:      Potential for optimisation.
        charge:              Total system charge.
        fmax:                Force convergence criterion (eV/Å).
        maxiter:             Max optimisation steps per layer.
        method:              Optimizer (``"FIRE"``, ``"LBFGS"``, ``"GD"``).
        max_stepsize:        Max per-atom step (LBFGS step bound; not used by FIRE).
        lr_cutoff:           SO3LR long-range cutoff (Å).
        placement_kwargs:    Extra kwargs for :func:`place_solvents_by_layers`.
        potential_kwargs:    Extra kwargs for the potential constructor.

    Returns:
        Final solvated MARS structure dict (optimised if requested).
    """
    from .log import log_header, log_info

    if layers is None:
        layers = [6]
    if placement_kwargs is None:
        placement_kwargs = {}
    if potential_kwargs is None:
        potential_kwargs = {}

    solvent_data = load_solvent(solvent) if isinstance(solvent, str) else solvent

    # Normalise layers
    specs: list[LayerSpec] = []
    for i, item in enumerate(layers):
        if isinstance(item, LayerSpec):
            specs.append(item)
        else:
            _layer_fields = {
                "buffer",
                "vdw_scale",
                "n_candidates",
                "n_shells",
                "min_solute_dist",
                "min_solvent_dist",
                "spread_weight",
            }
            specs.append(
                LayerSpec(
                    n_solvent=int(item),
                    seed=42 + i,
                    **{k: v for k, v in placement_kwargs.items() if k in _layer_fields},
                )
            )

    system = {k: v for k, v in solute.items()}
    n_solute = len(solute["symbols"])

    for i, spec in enumerate(specs):
        n_inner = len(system["symbols"])
        solvent_formula = "".join(
            f"{s}{solvent_data['symbols'].count(s)}" if solvent_data["symbols"].count(s) > 1 else s
            for s in dict.fromkeys(solvent_data["symbols"])
        )
        log_header(f"Layer {i + 1} / {len(specs)}  " f"({spec.n_solvent} × {solvent_formula})")

        system = _place_one_layer(system, solvent_data, spec)
        log_info(
            f"  System size: {len(system['symbols'])} atoms "
            f"({n_solute} solute + "
            f"{len(system['symbols']) - n_solute} solvent)."
        )

        if optimise_each_layer:
            log_info(f"  Optimising layer {i + 1} " f"(freezing {n_inner} inner atoms) ...")
            system = optimize_solvation_shell(
                system,
                n_solute=n_inner,
                potential_name=potential_name,
                freeze_mode="solute",
                charge=charge,
                lr_cutoff=lr_cutoff,
                fmax=fmax,
                maxiter=maxiter,
                method=method,
                max_stepsize=max_stepsize,
                **potential_kwargs,
            )

    return system


# ============================================================================
# Multi-conformer & multi-seed solvation utilities
# ============================================================================


def transplant_solvent_shell(
    reference_solvated: Dict,
    new_solute: Dict,
    n_solute: int,
) -> Dict:
    """Create a solvated structure by swapping in different solute coordinates.

    The solvent atoms (indices ``n_solute:`` ) are copied verbatim from
    *reference_solvated*.  The solute atoms (indices ``:n_solute``) are
    taken from *new_solute*.

    Args:
        reference_solvated: Solvated structure dict (solute + solvent).
        new_solute: Structure dict with the replacement solute coordinates.
                    Must have exactly *n_solute* atoms with the same
                    element symbols as the solute portion of *reference*.
        n_solute: Number of solute atoms (determines splice boundary).

    Returns:
        New structure dict with swapped solute and unchanged solvent.

    Raises:
        ValueError: If *new_solute* has the wrong atom count or different
            element symbols from the reference solute portion.
    """
    if len(new_solute["symbols"]) != n_solute:
        raise ValueError(
            f"new_solute has {len(new_solute['symbols'])} atoms, " f"expected {n_solute}"
        )
    ref_solute_syms = list(reference_solvated["symbols"][:n_solute])
    new_syms = list(new_solute["symbols"])
    if new_syms != ref_solute_syms:
        raise ValueError("Element mismatch between reference solute and new_solute")

    new_pos = np.concatenate(
        [
            np.asarray(new_solute["positions"]),
            np.asarray(reference_solvated["positions"])[n_solute:],
        ],
        axis=0,
    )

    return {
        "positions": new_pos,
        "symbols": list(reference_solvated["symbols"]),
        "numbers": list(reference_solvated["numbers"]),
    }


def align_to_reference_np(positions: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Kabsch-align *positions* onto *reference* (numpy, no JAX).

    Both are centred, then positions are rotated to minimise RMSD with
    respect to reference.  The result is translated so its centroid
    coincides with that of *reference* (i.e. the aligned structure sits
    on top of the reference, not at the origin).

    ⚠️  **Disclaimer**: Kabsch alignment can introduce errors when:
      - The molecule has symmetry (ambiguous optimal rotation)
      - Large conformational changes occur (alignment may be chemically incorrect)
      - The reference structure is not representative of all structures in the set

    For reaction trajectories with significant geometry changes, consider
    disabling alignment (``--no-align``) or using per-structure solvation
    (``--all-conformers``) to avoid artificial solvation artifacts.

    Args:
        positions: ``(N, 3)`` array to align.
        reference: ``(N, 3)`` reference array.

    Returns:
        ``(N, 3)`` aligned positions (in the reference frame).
    """
    pos_c = positions - positions.mean(axis=0, keepdims=True)
    ref_c = reference - reference.mean(axis=0, keepdims=True)
    H = pos_c.T @ ref_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ S @ U.T
    aligned = pos_c @ R.T
    # translate to reference centroid
    aligned += reference.mean(axis=0, keepdims=True)
    return aligned


def solvate_multi_seed(
    solute: Dict,
    solvent: Union[str, Dict] = "water",
    n_replicas: int = 1,
    base_seed: int = 42,
    *,
    # auto-mode kwargs
    padding: Optional[float] = None,
    n_molecules: Optional[int] = None,
    max_layers: int = 10,
    # manual-mode kwargs
    layers: Optional[List[int]] = None,
    # placement kwargs
    vdw_scale: float = 0.75,
    buffer: float = 0.0,
    n_candidates: int = 300,
    n_shells_per_atom: int = 3,
    n_orient: int = 8,
    refill_cycles: int = 3,
    # optimisation kwargs (only used when opt_each_layer=True)
    optimise_each_layer: bool = False,
    potential_name: str = "so3lr",
    charge: float = 0.0,
    fmax: float = 0.05,
    maxiter: int = 10000,
    method: str = "FIRE",
    max_stepsize: float = 0.15,
    lr_cutoff: float = 1000.0,
    topology_repair: bool = True,
    max_repair: int = 3,
    topo_tolerance: float = 1.3,
    potential_kwargs: Optional[dict] = None,
) -> List[Dict]:
    """Run *n_replicas* independent solvation placements with different seeds.

    Each replica ``i`` uses ``seed = base_seed + i``.  Geometry placement
    is inherently sequential (numpy, not JAX), so replicas are built in a
    Python loop.

    To keep memory usage constant regardless of *n_replicas*, each
    completed replica is saved to a temporary XYZ file and freed from
    memory.  All replicas are reloaded at the end.

    Args:
        solute: Solute structure dict.
        solvent: Solvent name or data dict.
        n_replicas: Number of replicas to generate.
        base_seed: First seed; replica *i* uses ``base_seed + i``.
        layers: If given, use manual mode (:func:`place_solvents_by_layers`);
                otherwise use automatic mode (:func:`auto_solvate`). All other
                keyword arguments (placement and optimisation options) match
                those of the selected function.

    Returns:
        List of *n_replicas* solvated structure dicts.
    """
    import os
    import tempfile

    from .log import log_info
    from .utils import load_structure, save_structure

    with tempfile.TemporaryDirectory(prefix="mars_solv_") as tmpdir:
        tmp_paths: List[Tuple[str, Optional[float]]] = []

        for i in range(n_replicas):
            seed_i = base_seed + i
            if n_replicas > 1:
                log_info(f"  Solvation replica {i + 1}/{n_replicas}  (seed={seed_i})")

            if layers is not None:
                # Manual mode
                result = place_solvents_by_layers(
                    solute,
                    solvent,
                    layers,
                    buffer=buffer,
                    vdw_scale=vdw_scale,
                    n_candidates=n_candidates,
                    n_shells=n_shells_per_atom,
                    seed=seed_i,
                )
            else:
                # Automatic mode
                result = auto_solvate(
                    solute,
                    solvent,
                    padding=padding,
                    n_molecules=n_molecules,
                    max_layers=max_layers,
                    optimise_each_layer=optimise_each_layer,
                    potential_name=potential_name,
                    charge=charge,
                    fmax=fmax,
                    maxiter=maxiter,
                    method=method,
                    max_stepsize=max_stepsize,
                    lr_cutoff=lr_cutoff,
                    vdw_scale=vdw_scale,
                    buffer=buffer,
                    n_candidates=n_candidates,
                    n_shells_per_atom=n_shells_per_atom,
                    n_orient=n_orient,
                    refill_cycles=refill_cycles,
                    seed=seed_i,
                    topology_repair=topology_repair,
                    max_repair=max_repair,
                    topo_tolerance=topo_tolerance,
                    potential_kwargs=potential_kwargs,
                )

            # Save to disk and free memory
            tmp_path = os.path.join(tmpdir, f"replica_{i}.xyz")
            energy_i = result.get("energy", None)
            save_structure(tmp_path, result)
            del result
            tmp_paths.append((tmp_path, energy_i))

        # Reload all replicas from disk
        replicas: List[Dict] = []
        for path, energy in tmp_paths:
            struct = load_structure(path)
            if energy is not None:
                struct["energy"] = energy
            replicas.append(struct)

    return replicas


def uniformize_solvation_batch(
    solvated_list: List[Dict],
    n_solute: int,
    solvent: Union[str, Dict],
    n_target: Optional[int] = None,
    *,
    potential_name: str = "so3lr",
    charge: float = 0.0,
    lr_cutoff: float = 1000.0,
    fmax: float = 0.05,
    maxiter: int = 10000,
    method: str = "FIRE",
    max_stepsize: float = 0.15,
    max_repair_rounds: int = 2,
    seed: int = 42,
    topo_tolerance: float = 1.3,
    max_layers: int = 10,
    buffer: float = 0.0,
    vdw_scale: float = 0.75,
    n_candidates: int = 300,
    n_orient: int = 8,
    refill_cycles: int = 3,
    **potential_kwargs,
) -> List[Dict]:
    """Force every solvated structure to hold exactly ``n_target`` solvent molecules.

    ``vmap``-based batch optimisation (:func:`optimize_solvation_batch`) requires a
    uniform atom count, but the geometric / barostat builder and broken-solvent
    removal can leave conformers with different counts. This restores uniformity:

    * ``count == n_target`` → returned untouched (no cost).
    * ``count > n_target``  → trimmed to the ``n_target`` molecules closest to the
      solute.
    * ``count < n_target``  → repaired in up to ``max_repair_rounds`` rounds of
      *remove-broken → geometric refill to n_target → frozen-solute relax* (via
      :func:`_solvate_cover` with ``init_system``). ``n_target`` is treated as a
      hard requirement.

    If a structure still falls short after the repair rounds (genuinely
    geometry-limited), a warning is emitted and, as a final safety net, the whole
    batch is trimmed down to the minimum achieved count so the result is uniform
    and the batch optimiser cannot crash.

    Returns the uniformised list in the original input order.
    """
    import numpy as np

    from .log import log_header, log_info, log_warning

    if not solvated_list:
        return []

    solvent_data = load_solvent(solvent) if isinstance(solvent, str) else solvent
    ref_bonds, nm = _solvent_ref_bonds(solvent_data, topo_tolerance)

    def _count(struct: Dict) -> int:
        return (len(struct["symbols"]) - n_solute) // nm

    def _solute_of(struct: Dict) -> Dict:
        pos = _struct_pos(struct)
        sym = _struct_sym(struct)
        return _make_struct(pos[:n_solute], sym[:n_solute])

    def _trim_to(struct: Dict, target: int) -> Dict:
        n_cur = _count(struct)
        if n_cur <= target:
            return struct
        pos = _struct_pos(struct)
        solute_pos = pos[:n_solute]
        # Keep the molecules whose closest atom is nearest the solute.
        dists = []
        for m in range(n_cur):
            off = n_solute + m * nm
            mol = pos[off : off + nm]
            d = float(np.min(np.linalg.norm(mol[:, None, :] - solute_pos[None, :, :], axis=-1)))
            dists.append((d, m))
        dists.sort()
        keep = {m for _, m in dists[:target]}
        remove = [m for m in range(n_cur) if m not in keep]
        return _remove_solvent_molecules(struct, n_solute, nm, remove)

    def relax_fn(sys_struct: Dict, n_frozen: int) -> Dict:
        # One frozen-solute relaxation = the "barostat round" that seats the
        # freshly placed molecules without disturbing the solute.
        return optimize_solvation_shell(
            sys_struct,
            n_solute=n_frozen,
            potential_name=potential_name,
            freeze_mode="solute",
            charge=charge,
            lr_cutoff=lr_cutoff,
            fmax=fmax,
            maxiter=maxiter,
            method=method,
            max_stepsize=max_stepsize,
            **potential_kwargs,
        )

    counts = [_count(s) for s in solvated_list]
    # No explicit target (e.g. padding/auto build): uniformise to the most any
    # conformer achieved, refilling the others up to it.
    if n_target is None:
        n_target = max(counts)
    if all(c == n_target for c in counts):
        return list(solvated_list)

    log_header("Solvation batch uniformisation")
    log_info(f"  Target molecules : {n_target}")
    log_info(
        f"  Incoming counts  : min {min(counts)} / max {max(counts)} "
        f"over {len(counts)} structures"
    )

    repaired: List[Dict] = []
    achieved: List[int] = []
    for i, struct in enumerate(solvated_list):
        c = counts[i]
        if c == n_target:
            repaired.append(struct)
            achieved.append(c)
            continue
        if c > n_target:
            t = _trim_to(struct, n_target)
            repaired.append(t)
            achieved.append(_count(t))
            log_info(f"  Structure {i}: trimmed {c} -> {n_target} molecules.")
            continue

        # Deficient: up to max_repair_rounds of geometric refill + relax.
        system = struct
        for rnd in range(max_repair_rounds):
            cur = _count(system)
            if cur >= n_target:
                break
            log_info(f"  Structure {i}: repair round {rnd + 1} refilling {cur} -> {n_target}.")
            system = _solvate_cover(
                _solute_of(system),
                solvent_data,
                padding=None,
                max_shells=max_layers,
                buffer=buffer,
                vdw_scale=vdw_scale,
                n_candidates=n_candidates,
                n_orient=n_orient,
                max_refill=refill_cycles,
                min_solute_dist=0.0,
                min_solvent_dist=0.0,
                relax_fn=relax_fn,
                seed=seed + 1000 * (rnd + 1) + i,
                n_target=n_target,
                init_system=system,
                init_count=cur,
                topology_repair=True,
                ref_bonds=ref_bonds,
                topo_tolerance=topo_tolerance,
            )
        fc = _count(system)
        if fc < n_target:
            log_warning(
                f"  Structure {i}: reached {fc}/{n_target} solvent molecules after "
                f"{max_repair_rounds} repair rounds (geometry-limited)."
            )
        repaired.append(system)
        achieved.append(fc)

    # Final safety net: if any structure could not reach n_target, trim the whole
    # batch down to the minimum achieved count so shapes are uniform.
    final_target = min(achieved)
    if final_target != n_target:
        log_warning(
            f"  Not all structures reached {n_target} molecules; uniformising the "
            f"batch to the minimum achieved count ({final_target}) for batch optimisation."
        )
        repaired = [_trim_to(s, final_target) for s in repaired]

    return repaired


def optimize_solvation_batch(
    solvated_list: List[Dict],
    n_solute: int,
    *,
    potential_name: str = "so3lr",
    charge: float = 0.0,
    lr_cutoff: float = 1000.0,
    freeze_mode: str = "solute",
    fmax: float = 0.05,
    maxiter: int = 10000,
    method: str = "FIRE",
    max_stepsize: float = 0.15,
    parallel: bool = True,
    potential_wrapper=None,
    fire_dt_start: float = 0.05,
    fire_dt_max: float = 0.1,
    fire_n_min: int = 2,
    multistage: bool = True,
    **extra_potential_kwargs,
) -> List[Dict]:
    """Optimise a batch of solvated structures with frozen solute via vmap.

    All structures must have the **same** atom count and element order
    (different only in positions).  When *parallel* is True the batch is
    optimised in a single vmapped JAX call using
    :func:`~mars.optimizer.optimize_batch_parallel`.

    When *multistage* is True (default), each optimisation runs in two
    passes (coarse then fine), mirroring
    :func:`optimize_solvation_shell`.

    Args:
        solvated_list: List of solvated structure dicts.
        n_solute: Number of solute atoms (freeze boundary).
        potential_name: ML potential to use.
        charge: Total system charge.
        lr_cutoff: SO3LR long-range cutoff (Å).
        freeze_mode: ``"solute"`` (default), ``"solvent"``, or ``"none"``.
        fmax: Force convergence criterion (eV/Å).
        maxiter: Maximum optimisation steps.
        method: Optimiser: ``"FIRE"``, ``"LBFGS"``, ``"GD"``.
        max_stepsize: Step-size limit.
        parallel: If True, use JAX vmap batching.
        potential_wrapper: Pre-built potential (reused for all structures).
        fire_dt_start: FIRE initial timestep.
        fire_dt_max: FIRE max timestep.
        fire_n_min: FIRE min positive-power steps.
        multistage: Use two-stage coarse → fine optimisation (default True).

    Returns:
        List of optimised structure dicts with ``'energy'`` field.
    """
    import jax.numpy as jnp

    from .auto_config import EV_TO_KCALMOL
    from .cli._constraints import make_frozen_energy_fn, make_frozen_energy_fn_batched
    from .log import log_header, log_info, log_step_end, log_step_start, log_warning
    from .optimizer import optimize_batch_parallel, optimize_single
    from .potentials import get_potential

    if not solvated_list:
        return []

    n_batch = len(solvated_list)
    first = solvated_list[0]
    symbols = list(first["symbols"])
    numbers = jnp.array(first["numbers"])
    n_total = len(symbols)

    log_header("Batch solvation shell optimisation")
    log_info(f"  Batch size    : {n_batch}")
    log_info(f"  Total atoms   : {n_total}  ({n_solute} solute + " f"{n_total - n_solute} solvent)")
    log_info(f"  Freeze mode   : {freeze_mode}")
    log_info(f"  Parallel      : {parallel}")
    is_hybrid = method.upper() == "HYBRID"
    if is_hybrid:
        multistage = True
        coarse_method, fine_method = "FIRE", "LBFGS"
        log_info(
            f"  Method        : HYBRID (FIRE coarse → LBFGS fine),  "
            f"fmax = {fmax} eV/Å,  maxiter = {maxiter}"
        )
    else:
        coarse_method = fine_method = method
        log_info(f"  Method        : {method},  fmax = {fmax} eV/Å,  " f"maxiter = {maxiter}")
    if multistage:
        log_info(f"  Multi-stage   : ON (coarse → fine)")

    # Build potential once from first structure
    if potential_wrapper is None:
        kwargs_pot: dict = dict(extra_potential_kwargs)
        kwargs_pot["species"] = numbers
        if potential_name == "so3lr":
            kwargs_pot.setdefault("lr_cutoff", lr_cutoff)
            kwargs_pot.setdefault("charge", charge)
        elif potential_name in ("mace", "dxtb"):
            kwargs_pot.setdefault("charge", charge)
        potential_wrapper = get_potential(potential_name, **kwargs_pot)

    ref_positions = jnp.array(first["positions"])
    potential_wrapper.initialize(ref_positions)
    energy_fn = potential_wrapper.build_energy_fn()

    # Determine frozen indices
    if freeze_mode == "solute":
        frozen = list(range(n_solute))
    elif freeze_mode == "solvent":
        frozen = list(range(n_solute, n_total))
    else:
        frozen = []

    if frozen:
        log_info(f"  Frozen atoms  : {len(frozen)}")

    # Multi-stage parameters
    coarse_maxiter = max(10, int(maxiter * 0.4)) if multistage else maxiter
    coarse_fmax = fmax * 5.0 if multistage else fmax
    coarse_stepsize = max_stepsize
    fine_maxiter = maxiter - coarse_maxiter if multistage else 0
    fine_stepsize = max_stepsize * 0.3

    t0 = log_step_start(f"Batch optimisation ({n_batch} structures)")

    if parallel and n_batch > 1:
        # ---- vmap path ----
        if frozen:
            batched_efn = make_frozen_energy_fn_batched(energy_fn, frozen)
        else:
            batched_efn = energy_fn

        positions_batch = jnp.stack(
            [jnp.array(s["positions"]) for s in solvated_list]
        )  # (n_batch, n_atoms, 3)

        # Stage 1 (or single pass when multistage=False)
        pos_opt, energies_opt, converged = optimize_batch_parallel(
            positions_batch,
            batched_efn,
            fmax=coarse_fmax,
            maxiter=coarse_maxiter,
            max_stepsize=coarse_stepsize,
            method=coarse_method,
            potential_wrapper=potential_wrapper,
            fire_dt_start=fire_dt_start,
            fire_dt_max=fire_dt_max,
            fire_n_min=fire_n_min,
        )

        if multistage:
            n_conv_coarse = int(converged.sum())
            log_info(f"  Stage 1 ({coarse_method:>5}): " f"converged {n_conv_coarse}/{n_batch}")

            # Stage 2: fine refinement
            pos_opt, energies_opt, converged = optimize_batch_parallel(
                pos_opt,
                batched_efn,
                fmax=fmax,
                maxiter=fine_maxiter,
                max_stepsize=fine_stepsize,
                method=fine_method,
                potential_wrapper=potential_wrapper,
                fire_dt_start=fire_dt_start,
                fire_dt_max=fine_stepsize,
                fire_n_min=fire_n_min,
            )
            n_conv_fine = int(converged.sum())
            log_info(f"  Stage 2 ({fine_method:>5}): " f"converged {n_conv_fine}/{n_batch}")
        else:
            n_conv = int(converged.sum())
            log_info(f"  Converged: {n_conv}/{n_batch}")

        results = []
        for i in range(n_batch):
            results.append(
                {
                    "positions": pos_opt[i],
                    "symbols": symbols,
                    "numbers": list(first["numbers"]),
                    "energy": float(energies_opt[i]),
                }
            )

    else:
        # ---- sequential path ----
        results = []
        for i, struct in enumerate(solvated_list):
            positions_i = jnp.array(struct["positions"])

            if not multistage:
                if frozen:
                    efn_i = make_frozen_energy_fn(energy_fn, positions_i, frozen)
                else:
                    efn_i = energy_fn

                pos_opt_i, energy_i, info_i = optimize_single(
                    positions_i,
                    efn_i,
                    fmax=fmax,
                    maxiter=maxiter,
                    method=method,
                    max_stepsize=max_stepsize,
                    potential_wrapper=potential_wrapper,
                )
                converged_i = info_i["converged"]
                total_iters_i = info_i["iterations"]
            else:
                # Stage 1: coarse
                if frozen:
                    efn_coarse = make_frozen_energy_fn(energy_fn, positions_i, frozen)
                else:
                    efn_coarse = energy_fn

                pos_opt_i, energy_i, info1 = optimize_single(
                    positions_i,
                    efn_coarse,
                    fmax=coarse_fmax,
                    maxiter=coarse_maxiter,
                    method=coarse_method,
                    max_stepsize=coarse_stepsize,
                    potential_wrapper=potential_wrapper,
                )

                # Stage 2: fine (rebuild frozen constraint on updated pos)
                if frozen:
                    efn_fine = make_frozen_energy_fn(energy_fn, pos_opt_i, frozen)
                else:
                    efn_fine = energy_fn

                remaining = fine_maxiter
                pos_opt_i, energy_i, info2 = optimize_single(
                    pos_opt_i,
                    efn_fine,
                    fmax=fmax,
                    maxiter=remaining,
                    method=fine_method,
                    max_stepsize=fine_stepsize,
                    potential_wrapper=potential_wrapper,
                )
                converged_i = info2["converged"]
                total_iters_i = info1["iterations"] + info2["iterations"]

            status = "converged" if converged_i else "⚠️  NOT CONVERGED"
            log_msg = (
                f"  Structure {i + 1}/{n_batch}: {status} "
                f"({total_iters_i} steps)  "
                f"E = {float(energy_i) * EV_TO_KCALMOL:.3f} kcal/mol"
            )
            if converged_i:
                log_info(log_msg)
            else:
                log_warning(log_msg)
                log_warning("    Consider increasing --maxiter or reducing --max-stepsize.")

            results.append(
                {
                    "positions": pos_opt_i,
                    "symbols": symbols,
                    "numbers": list(first["numbers"]),
                    "energy": float(energy_i),
                }
            )

    log_step_end(f"Batch optimisation ({n_batch} structures)", t0)
    return results


# ============================================================================
# Spherical moving-wall "barostat" droplet + swap solvation (optional)
# ============================================================================


def _outer_wall_energy(pos, center, r_outer, k, exponent):
    """Soft spherical outer wall energy: penalises atoms beyond ``r_outer``.

    ``E = k · Σ_i max(0, |r_i - c|/r_outer - 1)^exponent``  (JAX, differentiable).
    """
    import jax.numpy as jnp

    d = jnp.linalg.norm(pos - jnp.asarray(center), axis=1)
    return k * jnp.sum(jnp.maximum(0.0, d / r_outer - 1.0) ** exponent)


def _shell_wall_barrier(center, r_inner, r_outer, k, exponent):
    """Build a ``positions -> scalar`` barrier = inner cavity wall + outer wall.

    * **Outer wall** keeps atoms inside ``r_outer`` (the moving piston).
    * **Inner wall** keeps atoms outside ``r_inner`` (the solute cavity); skipped
      when ``r_inner`` is ``None``/0.

    Suitable as the ``barrier_fn`` of :func:`mars.optimizer.optimize_single`.
    """
    import jax.numpy as jnp

    c = jnp.asarray(center)
    use_inner = r_inner is not None and r_inner > 0.0

    def barrier(pos):
        d = jnp.linalg.norm(pos - c, axis=1)
        e = k * jnp.sum(jnp.maximum(0.0, d / r_outer - 1.0) ** exponent)
        if use_inner:
            e = e + k * jnp.sum(jnp.maximum(0.0, 1.0 - d / r_inner) ** exponent)
        return e

    return barrier


def _wall_reaction_force(pos, center, r_outer, k, exponent):
    """Peak per-atom force (eV/Å) the outer wall exerts on the droplet.

    This is the barostat's "pressure" gauge: as the wall compresses the packed
    solvent, the reaction force rises; compression stops once it reaches the
    target ``--baro-max-force``.
    """
    import jax
    import jax.numpy as jnp

    grad = jax.grad(lambda p: _outer_wall_energy(p, center, r_outer, k, exponent))
    g = grad(jnp.asarray(pos))
    return float(jnp.max(jnp.linalg.norm(g, axis=1)))


def _place_droplet(
    solvent_data, n_target, center, r_inner, r_outer, *, vdw_scale, buffer, n_orient, seed
):
    """Place up to *n_target* solvent molecules in the spherical shell
    ``[r_inner, r_outer]`` around *center* (no solute present).

    Greedy placement with vdW-overlap rejection and ``n_orient`` orientation
    tries per candidate site. Returns ``(struct, n_placed)``.
    """
    rng = np.random.default_rng(seed)
    solv0 = np.asarray(solvent_data["positions"], dtype=np.float64).copy()
    solv0 -= solv0.mean(axis=0)
    solv_sym = list(solvent_data["symbols"])
    r_pack = _effective_radius(solv0, solv_sym)

    lo = (r_inner or 0.0) + r_pack + buffer
    hi = r_outer - r_pack
    centres = []
    if hi > lo:
        n_radii = max(1, int((hi - lo) / max(r_pack, 0.5)) + 1)
        for ri in np.linspace(lo, hi, n_radii):
            # ~ surface area / molecular cross-section, oversampled 1.5×
            n_pts = max(8, int(1.5 * 4.0 * ri * ri / max(r_pack * r_pack, 0.25)))
            centres.append(_fibonacci_sphere(n_pts) * ri + np.asarray(center))
    if centres:
        centres = np.vstack(centres)
        rng.shuffle(centres)
    else:
        centres = np.zeros((0, 3))

    placed_arr = np.zeros((0, 3))
    placed_sym: List[str] = []
    mols = []
    for centre in centres:
        if len(mols) >= n_target:
            break
        for _ in range(n_orient):
            rot = Rotation.random(random_state=int(rng.integers(2**31)))
            trial = rot.apply(solv0) + centre
            if placed_arr.shape[0] and _has_overlap(
                trial, solv_sym, placed_arr, placed_sym, scale=vdw_scale
            ):
                continue
            placed_arr = np.vstack([placed_arr, trial]) if placed_arr.shape[0] else trial
            placed_sym = placed_sym + solv_sym
            mols.append(trial)
            break

    if mols:
        pos = np.vstack(mols)
        sym = solv_sym * len(mols)
    else:
        pos, sym = np.zeros((0, 3)), []
    return _make_struct(pos, sym), len(mols)


def _relax_with_barrier(
    struct,
    n_frozen,
    barrier_fn,
    *,
    potential_name,
    charge,
    lr_cutoff,
    fmax,
    maxiter,
    method,
    max_stepsize,
    potential_kwargs,
):
    """Relax *struct* with an extra ``barrier_fn`` (e.g. the wall), optionally
    freezing the first *n_frozen* atoms. Returns ``(struct, energy)``."""
    import jax.numpy as jnp

    from .cli._constraints import make_frozen_energy_fn
    from .optimizer import optimize_single
    from .potentials import get_potential

    pos = jnp.array(struct["positions"])
    numbers = jnp.array(struct["numbers"])
    kw = dict(potential_kwargs or {})
    kw["species"] = numbers
    if potential_name == "so3lr":
        kw.setdefault("lr_cutoff", lr_cutoff)
        kw.setdefault("charge", charge)
    elif potential_name in ("mace", "dxtb"):
        kw.setdefault("charge", charge)
    pw = get_potential(potential_name, **kw)
    pw.initialize(pos)
    efn = pw.build_energy_fn()
    if n_frozen and n_frozen > 0:
        efn = make_frozen_energy_fn(efn, pos, list(range(n_frozen)))
    pos_opt, energy, _ = optimize_single(
        pos,
        efn,
        fmax=fmax,
        maxiter=maxiter,
        method=method,
        max_stepsize=max_stepsize,
        barrier_fn=barrier_fn,
        potential_wrapper=pw,
    )
    return _make_struct(np.asarray(pos_opt), list(struct["symbols"])), float(energy)


def _md_with_barrier(
    struct,
    barrier_fn,
    *,
    potential_name,
    charge,
    lr_cutoff,
    md_temp,
    md_time_ps,
    md_dt,
    seed,
    potential_kwargs,
):
    """Run a short NVT MD with the wall added to the energy, returning the final
    frame. Lets the solvent *flow* past each other while the wall compresses,
    rather than relaxing straight downhill. Used by the ``--baro-md`` variant."""
    import jax.numpy as jnp

    from .potentials import get_potential
    from .sampling import run_rotamer_md_jax
    from .utils import get_atomic_masses

    pos = jnp.array(struct["positions"])
    numbers = jnp.array(struct["numbers"])
    kw = dict(potential_kwargs or {})
    kw["species"] = numbers
    if potential_name == "so3lr":
        kw.setdefault("lr_cutoff", lr_cutoff)
        kw.setdefault("charge", charge)
    elif potential_name in ("mace", "dxtb"):
        kw.setdefault("charge", charge)
    pw = get_potential(potential_name, **kw)
    pw.initialize(pos)
    base = pw.build_energy_fn()

    def efn(p, **k):
        return base(p, **k) + barrier_fn(p)

    traj = run_rotamer_md_jax(
        pos,
        efn,
        T=md_temp,
        dt=md_dt,
        time_ps=md_time_ps,
        save_interval=100,
        mass=get_atomic_masses(numbers),
        random_seed=int(seed),
        potential_wrapper=pw,
    )
    last = traj[-1] if traj else np.asarray(pos)
    return _make_struct(np.asarray(last), list(struct["symbols"]))


def _barostat_compress(
    struct,
    center,
    r_inner,
    r_outer0,
    n_frozen,
    *,
    wall_k,
    wall_exponent,
    compress_step,
    fine_step,
    max_wall_force,
    max_steps,
    r_floor,
    r_min,
    potential_name,
    charge,
    lr_cutoff,
    fmax,
    step_maxiter,
    method,
    max_stepsize,
    compress_md,
    md_temp,
    md_time_ps,
    md_dt,
    seed,
    potential_kwargs,
    label,
):
    """Shrink the outer wall, relaxing between steps, until its reaction force
    reaches *max_wall_force*.

    Compression is **coarse** (``compress_step``) down to ``r_floor``, then
    **fine** (``fine_step``, e.g. 0.1 Å) so the wall creeps inward and stops
    right when the force target is met rather than overshooting. ``r_min`` is a
    hard collapse safety (never compress to/through the solute or cavity core).

    *r_inner* (or ``None``) is a fixed inner-cavity wall; *n_frozen* freezes the
    first N atoms (the solute, in the post-swap packing stage). MD compression
    only runs when nothing is frozen. Returns ``(struct, final_r_outer)``.
    """
    from .log import log_header, log_info, log_warning

    log_header(label)

    def _relax_at(struct, r_outer, idx):
        barrier = _shell_wall_barrier(center, r_inner, r_outer, wall_k, wall_exponent)
        if compress_md and not n_frozen:
            struct = _md_with_barrier(
                struct,
                barrier,
                potential_name=potential_name,
                charge=charge,
                lr_cutoff=lr_cutoff,
                md_temp=md_temp,
                md_time_ps=md_time_ps,
                md_dt=md_dt,
                seed=seed + idx,
                potential_kwargs=potential_kwargs,
            )
        struct, _ = _relax_with_barrier(
            struct,
            n_frozen,
            barrier,
            potential_name=potential_name,
            charge=charge,
            lr_cutoff=lr_cutoff,
            fmax=fmax,
            maxiter=step_maxiter,
            method=method,
            max_stepsize=max_stepsize,
            potential_kwargs=potential_kwargs,
        )
        f = _wall_reaction_force(struct["positions"], center, r_outer, wall_k, wall_exponent)
        return struct, f

    r_outer = float(r_outer0)
    idx = 0
    coarse_used = 0
    # Coarse steps (capped by max_steps) bring the wall down to r_floor; then
    # fine 0.1-Å steps creep inward until the force target is met — bounded only
    # by r_min (a hard collapse safety), so we don't stop short of the target.
    while r_outer > r_min:
        struct, f_wall = _relax_at(struct, r_outer, idx)
        idx += 1
        fine = (r_outer <= r_floor) or (coarse_used >= max_steps)
        log_info(
            f"  step {idx}: r_outer = {r_outer:.2f} Å, wall force = {f_wall:.3f} "
            f"eV/Å ({'fine' if fine else 'coarse'})"
        )
        if f_wall >= max_wall_force:
            log_info(f"  Target wall force reached (r_outer = {r_outer:.2f} Å).")
            return struct, r_outer
        step = fine_step if fine else compress_step
        if not fine:
            coarse_used += 1
        if r_outer - step < r_min:
            log_warning(
                f"  Reached the collapse-safety radius {r_min:.2f} Å before the "
                f"{max_wall_force} eV/Å target — stopping. Lower --baro-max-force "
                f"or raise --baro-k."
            )
            return struct, r_outer
        r_outer -= step
    log_warning(f"  Reached r_min before the force target (r_outer = {r_outer:.2f} Å).")
    return struct, r_outer


def solvate_barostat_batch(
    solutes: List[Dict],
    solvent: Union[str, Dict] = "water",
    *,
    n_molecules: int,
    reference_idx: int = 0,
    inner_buffer: float = 2.0,
    outer_radius: Optional[float] = None,
    fill_fraction: float = 0.35,
    wall_k: float = 5.0,
    wall_mode: str = "harmonic",
    wall_exponent: Optional[int] = None,
    compress_step: float = 0.5,
    fine_step: float = 0.1,
    max_wall_force: float = 1.0,
    max_compress_steps: int = 40,
    compress_md: bool = False,
    md_temp: float = 300.0,
    md_time_ps: float = 0.3,
    md_dt: float = 0.5,
    vdw_scale: float = 0.85,
    buffer: float = 0.0,
    n_orient: int = 8,
    seed: int = 42,
    potential_name: str = "so3lr",
    charge: float = 0.0,
    lr_cutoff: float = 1000.0,
    fmax: float = 0.05,
    step_maxiter: int = 400,
    final_maxiter: int = 4000,
    method: str = "FIRE",
    max_stepsize: float = 0.15,
    topology_repair: bool = True,
    topo_tolerance: float = 1.3,
    parallel: bool = True,
    potential_kwargs: Optional[dict] = None,
) -> List[Dict]:
    """Solvate a batch of (aligned) solutes via a spherical moving-wall barostat.

    **One** solute-free droplet is built and compressed (from ``solutes[
    reference_idx]``); the swap then drops *each* solute into the cavity, and the
    final packing compresses the solvent onto every solute **together**, as a
    vmapped batch, with a **per-structure moving wall**.

    Steps:

    1. **Droplet** — place *n_molecules* in the shell ``[r_inner, r_outer]`` (no
       solute) and compress it (inner + outer walls) until the solvent converges
       against the inner wall.
    2. **Swap** — transplant each solute into the droplet cavity → a batch
       (solute atoms first, then the shared solvent).
    3. **Batch final packing** — one shared wall-stepping loop over the whole
       batch: at each step relax all structures (solute frozen, each with its
       own outer wall) in a single vmapped call; any structure whose outer-wall
       reaction force reaches *max_wall_force* **freezes its wall**, the rest keep
       being compressed inward (coarse → ``fine_step``). Stops when all have
       converged. Each structure ends at its own radius.
    4. **Batch wall-free relaxation** — remove the wall and relax (solute frozen).
    5. **Topology check** per structure (broken molecules removed).

    Returns a list of solvated MARS structure dicts (same order as *solutes*).
    """
    import jax.numpy as jnp

    from .log import log_header, log_info, log_warning
    from .optimizer import optimize_wall_batch
    from .potentials import get_potential

    solvent_data = load_solvent(solvent) if isinstance(solvent, str) else solvent
    if n_molecules is None or int(n_molecules) <= 0:
        raise ValueError("solvate_barostat requires a positive n_molecules.")
    n_molecules = int(n_molecules)
    if wall_exponent is None:
        wall_exponent = 2 if wall_mode == "harmonic" else 12

    ref_solute = solutes[reference_idx]
    ref_pos = _struct_pos(ref_solute)
    solute_sym = _struct_sym(ref_solute)
    n_solute = len(solute_sym)
    center = ref_pos.mean(axis=0)
    r_solute = _bounding_radius(ref_pos, solute_sym)
    r_inner = r_solute + inner_buffer

    solv_c = np.asarray(solvent_data["positions"]) - np.asarray(solvent_data["positions"]).mean(
        axis=0
    )
    r_mol = _bounding_radius(solv_c, list(solvent_data["symbols"]))
    r_eff = _effective_radius(solv_c, list(solvent_data["symbols"]))
    r_floor = r_inner + 2.0 * r_eff
    if outer_radius is None:
        outer_radius = (r_inner**3 + n_molecules * (r_mol**3) / max(fill_fraction, 0.05)) ** (
            1.0 / 3.0
        )
    outer_radius = max(float(outer_radius), r_floor + 3.0)

    log_header("Barostat solvation (spherical moving-wall droplet)")
    log_info(f"  Structures    : {len(solutes)}  (reference = {reference_idx})")
    log_info(f"  Solute radius : {r_solute:.2f} Å  → cavity r_inner = {r_inner:.2f} Å")
    log_info(f"  Initial outer : r_outer = {outer_radius:.2f} Å  (fill {fill_fraction:.2f})")
    log_info(f"  Wall          : mode={wall_mode}, k={wall_k}, exponent={wall_exponent}")
    log_info(f"  Barostat stop : wall force ≥ {max_wall_force} eV/Å")

    # --- 1. Place + compress the solute-free droplet -----------------------
    droplet, n_placed = _place_droplet(
        solvent_data,
        n_molecules,
        center,
        r_inner,
        outer_radius,
        vdw_scale=vdw_scale,
        buffer=buffer,
        n_orient=n_orient,
        seed=seed,
    )
    if n_placed < n_molecules:
        log_warning(
            f"  Only placed {n_placed}/{n_molecules} in the initial droplet — "
            f"increase --baro-radius or --baro-fill for more room."
        )
    log_info(f"  Placed {n_placed} solvent molecules in the droplet.")

    droplet, r_outer = _barostat_compress(
        droplet,
        center,
        r_inner,
        outer_radius,
        0,
        wall_k=wall_k,
        wall_exponent=wall_exponent,
        compress_step=compress_step,
        fine_step=fine_step,
        max_wall_force=max_wall_force,
        max_steps=max_compress_steps,
        r_floor=r_floor,
        r_min=r_inner + 0.5 * r_eff,
        potential_name=potential_name,
        charge=charge,
        lr_cutoff=lr_cutoff,
        fmax=fmax,
        step_maxiter=step_maxiter,
        method=method,
        max_stepsize=max_stepsize,
        compress_md=compress_md,
        md_temp=md_temp,
        md_time_ps=md_time_ps,
        md_dt=md_dt,
        seed=seed,
        potential_kwargs=potential_kwargs,
        label="Barostat compression (droplet)",
    )

    # --- 2. Swap: transplant each solute into the cavity → batch -----------
    log_header("Swap solutes into droplet cavity")
    solv_pos = _struct_pos(droplet)
    solv_sym = _struct_sym(droplet)
    systems = [
        _make_struct(np.vstack([_struct_pos(s), solv_pos]), _struct_sym(s) + list(solv_sym))
        for s in solutes
    ]
    n_batch = len(systems)

    # --- 3. Batch final packing: per-structure moving wall onto the solute --
    log_header(f"Final packing — batch moving wall ({n_batch} structures)")
    numbers = jnp.array(systems[0]["numbers"])
    kwp = dict(potential_kwargs or {})
    kwp["species"] = numbers
    if potential_name == "so3lr":
        kwp.setdefault("lr_cutoff", lr_cutoff)
        kwp.setdefault("charge", charge)
    elif potential_name in ("mace", "dxtb"):
        kwp.setdefault("charge", charge)
    pw = get_potential(potential_name, **kwp)
    pw.initialize(jnp.array(systems[0]["positions"]))
    base_efn = pw.build_energy_fn()

    pos_batch = jnp.stack([jnp.array(s["positions"]) for s in systems])
    r_outer_b = jnp.full((n_batch,), float(r_outer), dtype=pos_batch.dtype)
    converged = jnp.zeros((n_batch,), dtype=bool)
    r_floor_final = r_solute + 1.5 * r_eff
    r_min_final = r_solute + 0.5
    span = max(0.0, r_floor_final - r_min_final)
    max_total = max_compress_steps + int(span / max(fine_step, 1e-3)) + 5

    for step in range(max_total):
        pos_batch, _e, _c, wforce = optimize_wall_batch(
            pos_batch,
            base_efn,
            r_outer_b,
            center=center,
            wall_k=wall_k,
            wall_exponent=wall_exponent,
            frozen=range(n_solute),
            fmax=fmax,
            maxiter=step_maxiter,
            potential_wrapper=pw,
        )
        reached = (wforce >= max_wall_force) | (r_outer_b <= r_min_final + 1e-6)
        converged = converged | reached
        log_info(
            f"  step {step + 1}: {int(converged.sum())}/{n_batch} converged, "
            f"r_outer∈[{float(r_outer_b.min()):.2f},{float(r_outer_b.max()):.2f}] Å, "
            f"force∈[{float(wforce.min()):.3f},{float(wforce.max()):.3f}] eV/Å"
        )
        if bool(converged.all()):
            break
        # Coarse step until each structure's wall reaches the refine zone, then
        # fine steps; frozen (converged) structures stop moving.
        sizes = jnp.where(r_outer_b <= r_floor_final, fine_step, compress_step)
        r_outer_b = jnp.where(converged, r_outer_b, jnp.maximum(r_outer_b - sizes, r_min_final))
    else:
        log_warning(
            f"  Hit step cap ({max_total}) with {int(converged.sum())}/{n_batch} converged."
        )

    systems = [
        _make_struct(np.asarray(pos_batch[i]), list(systems[i]["symbols"])) for i in range(n_batch)
    ]

    # --- 4. Batch wall-free relaxation (solute frozen) ---------------------
    log_header("Final relaxation (no wall, solute frozen)")
    systems = optimize_solvation_batch(
        systems,
        n_solute,
        potential_name=potential_name,
        charge=charge,
        lr_cutoff=lr_cutoff,
        freeze_mode="solute",
        fmax=fmax,
        maxiter=final_maxiter,
        method=method,
        max_stepsize=max_stepsize,
        parallel=(parallel and n_batch > 1),
        potential_wrapper=pw,
        **(potential_kwargs or {}),
    )

    # --- 5. Topology cleanup (removal only; no refill in barostat mode) ----
    results = []
    for i, s in enumerate(systems):
        if topology_repair:
            s, broken = remove_broken_solvent(s, solvent_data, n_solute, tolerance=topo_tolerance)
            if broken:
                log_warning(f"  Structure {i}: removed {len(broken)} broken solvent molecule(s).")
        n_final = (len(s["symbols"]) - n_solute) // len(solvent_data["symbols"])
        log_info(f"  Structure {i}: {n_final} solvent molecules.")
        results.append(s)
    return results


def solvate_barostat(solute: Dict, solvent: Union[str, Dict] = "water", **kwargs) -> Dict:
    """Single-solute spherical moving-wall **barostat** solvation.

    Thin wrapper over :func:`solvate_barostat_batch` (batch of one). Builds a
    solute-free solvent droplet, compresses it with an inner cavity wall + an
    outer "piston" wall, swaps the solute into the cavity, then compresses the
    solvent onto the solute (moving wall, until the wall reaction force target),
    relaxes without the wall, and cleans broken solvent molecules. See
    :func:`solvate_barostat_batch` for the full description and parameters.
    """
    return solvate_barostat_batch([solute], solvent, **kwargs)[0]
