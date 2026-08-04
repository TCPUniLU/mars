"""
RMSD pruning utilities for JAX structures.

Provides RMSD-based duplicate removal and rotational constant-based pruning
for structure dictionaries.
"""

from typing import Dict, List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .log import log_message
from .rmsd import rmsd_cv_jax
from .utils import ATOMIC_MASSES as _ATOMIC_MASSES_ARR
from .utils import COVALENT_RADII, BondTopology, _is_jax_oom


def compute_inertia_tensor(positions: jnp.ndarray, atomic_numbers: jnp.ndarray) -> jnp.ndarray:
    """Compute moment of inertia tensor.

    Args:
        positions: (N, 3) array of atomic positions in Angstrom
        atomic_numbers: (N,) array of atomic numbers

    Returns:
        inertia_tensor: (3, 3) moment of inertia tensor in amu*Angstrom^2
    """
    # Get masses from atomic numbers
    _z = np.asarray(atomic_numbers, dtype=int)
    masses = jnp.array(_ATOMIC_MASSES_ARR[np.clip(_z, 0, len(_ATOMIC_MASSES_ARR) - 1)])

    # Center at center of mass
    com = jnp.sum(positions * masses[:, None], axis=0) / jnp.sum(masses)
    coords = positions - com

    # Compute inertia tensor
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    Ixx = jnp.sum(masses * (y**2 + z**2))
    Iyy = jnp.sum(masses * (x**2 + z**2))
    Izz = jnp.sum(masses * (x**2 + y**2))
    Ixy = -jnp.sum(masses * x * y)
    Ixz = -jnp.sum(masses * x * z)
    Iyz = -jnp.sum(masses * y * z)

    I = jnp.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])

    return I


def compute_rotational_constants(
    positions: jnp.ndarray, atomic_numbers: jnp.ndarray
) -> jnp.ndarray:
    """Compute principal rotational constants (A, B, C) in GHz.

    Rotational constants are computed as B = h/(8π²I) where I is the
    principal moment of inertia. Convention: A >= B >= C.

    Args:
        positions: (N, 3) array of atomic positions in Angstrom
        atomic_numbers: (N,) array of atomic numbers

    Returns:
        rot_constants: (3,) array of rotational constants [A, B, C] in GHz
    """
    I_tensor = compute_inertia_tensor(positions, atomic_numbers)

    # Get principal moments (eigenvalues)
    principal_moments = jnp.linalg.eigvalsh(I_tensor)

    # Sort in ascending order (smallest moment -> largest constant)
    principal_moments = jnp.sort(principal_moments)

    # Convert to rotational constants in GHz
    # B [GHz] = h/(8π²I) = 505379.05 / I[amu*Angstrom^2]
    conversion = 505379.05  # MHz*amu*Angstrom^2

    # Handle zero or negative moments (linear/atomic cases)
    principal_moments = jnp.where(principal_moments > 1e-10, principal_moments, 1e-10)

    rot_constants = conversion / principal_moments

    # Return in descending order (A >= B >= C)
    return rot_constants[::-1]


def compare_rotational_constants(
    rot_const1: jnp.ndarray, rot_const2: jnp.ndarray, threshold: float = 0.02
) -> bool:
    """Compare two sets of rotational constants.

    Uses relative difference metric: |R1 - R2| / max(R1, R2) for each constant.
    Structures are considered similar if all three differences are below threshold.

    Args:
        rot_const1: (3,) array of rotational constants [A, B, C]
        rot_const2: (3,) array of rotational constants [A, B, C]
        threshold: Relative difference threshold (default 0.02 = 2%)

    Returns:
        similar: True if all relative differences are below threshold
    """
    rel_diffs = jnp.abs(rot_const1 - rot_const2) / jnp.maximum(rot_const1, rot_const2)
    return bool(jnp.all(rel_diffs < threshold))


def prune_by_rotational_constants(
    ensemble: List[Tuple[Dict, float]],
    threshold: float = 0.02,
    reference: Optional[List[Tuple[Dict, float]]] = None,
) -> List[Tuple[Dict, float]]:
    """Remove structures with similar rotational constants.

    The first structure is always kept (unless it duplicates one in reference).
    Each subsequent structure is compared to all kept structures AND the
    reference ensemble. If all three principal rotational constants are within
    the threshold (relative difference), the structure is discarded.

    Args:
        ensemble: List of (structure, energy) tuples (should be sorted by energy)
        threshold: Relative difference threshold for rotational constants (default 0.02)
        reference: Optional existing ensemble to compare against (e.g. accumulated
                   conformers from previous phases). New structures matching any
                   reference structure are discarded.

    Returns:
        pruned: List with similar structures removed
    """
    if not ensemble:
        return ensemble

    # Precompute rotational constants for reference ensemble
    ref_rot_consts = []
    if reference:
        for ref_struct, _ in reference:
            ref_rot_consts.append(
                compute_rotational_constants(ref_struct["positions"], ref_struct["numbers"])
            )

    kept = []
    kept_rot_consts = []

    for struct, e in ensemble:
        rot_const = compute_rotational_constants(struct["positions"], struct["numbers"])

        too_similar = False

        # Compare to reference ensemble
        for ref_rot in ref_rot_consts:
            if compare_rotational_constants(rot_const, ref_rot, threshold):
                too_similar = True
                break

        # Compare to already kept structures from this batch
        if not too_similar:
            for kept_rot in kept_rot_consts:
                if compare_rotational_constants(rot_const, kept_rot, threshold):
                    too_similar = True
                    break

        if not too_similar:
            kept.append((struct, e))
            kept_rot_consts.append(rot_const)

    removed = len(ensemble) - len(kept)
    log_message(
        f"[RotConst Prune] Removed {removed} similar structures (threshold={threshold:.3g})"
    )
    log_message(f"[RotConst Prune] Kept {len(kept)}/{len(ensemble)} structures")

    return kept


@jax.jit
def _rmsd_rows(row_pos: jnp.ndarray, all_pos: jnp.ndarray) -> jnp.ndarray:
    """Compute RMSD from row structures against all structures.

    Args:
        row_pos: (C, N, 3) row structures
        all_pos: (M, N, 3) all structures

    Returns:
        rmsd_block: (C, M) RMSD sub-matrix
    """

    def row_fn(pos):
        return jax.vmap(lambda ref: rmsd_cv_jax(pos, ref))(all_pos)

    return jax.vmap(row_fn)(row_pos)


@jax.jit
def _rmsd_rows_perm_inv(
    row_pos: jnp.ndarray, all_pos: jnp.ndarray, frag_perms: jnp.ndarray
) -> jnp.ndarray:
    """Permutation-invariant RMSD from row structures against all structures.

    For each (row, col) pair, computes
        min_{perm} RMSD(row, col[perm])
    over all fragment permutations in *frag_perms*.

    Args:
        row_pos:    (C, N, 3) row structures
        all_pos:    (M, N, 3) all structures
        frag_perms: (P, N) int32 atom-index permutations

    Returns:
        rmsd_block: (C, M) RMSD sub-matrix (minimum over permutations)
    """

    def _perm_inv_rmsd(pos, ref):
        """Min RMSD over all fragment permutations."""

        def _rmsd_for_perm(perm):
            return rmsd_cv_jax(pos, ref[perm])

        rmsds = jax.vmap(_rmsd_for_perm)(frag_perms)
        return jnp.min(rmsds)

    def row_fn(pos):
        return jax.vmap(lambda ref: _perm_inv_rmsd(pos, ref))(all_pos)

    return jax.vmap(row_fn)(row_pos)


def _pairwise_rmsd_chunked(
    row_pos: jnp.ndarray, all_pos: jnp.ndarray, frag_perms: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """Compute RMSD from row_pos against all_pos.

    Tries to allocate the full (C, M) block. On OOM, splits row_pos
    in half and retries — same binary-split strategy as optimizer.py.

    Args:
        row_pos:    (C, N, 3) row structures to compute
        all_pos:    (M, N, 3) all structures (columns, never split)
        frag_perms: (P, N) int32 atom-index permutations for identical-
                    fragment invariance.  ``None`` → standard RMSD.

    Returns:
        rmsd_block: (C, M) RMSD sub-matrix
    """
    C = row_pos.shape[0]
    _rmsd_fn = (
        _rmsd_rows if frag_perms is None else lambda r, a: _rmsd_rows_perm_inv(r, a, frag_perms)
    )

    try:
        return _rmsd_fn(row_pos, all_pos)
    except Exception as exc:
        if not _is_jax_oom(exc) or C < 2:
            raise

        mid = C // 2
        log_message(
            f"[RMSD Prune] OOM with {C} rows, splitting into " f"{mid} + {C - mid} and retrying"
        )
        jax.clear_caches()

        top = _pairwise_rmsd_chunked(row_pos[:mid], all_pos, frag_perms)
        bot = _pairwise_rmsd_chunked(row_pos[mid:], all_pos, frag_perms)
        return jnp.concatenate([top, bot], axis=0)


@jax.jit
def _greedy_prune_jax(
    pairwise_rmsds: jnp.ndarray, min_ref_rmsds: jnp.ndarray, threshold: float
) -> jnp.ndarray:
    """Greedy RMSD pruning on precomputed distances via lax.scan.

    Iterates through structures in order. A structure is kept only if its
    RMSD to all previously kept structures AND all reference structures
    exceeds the threshold.

    Args:
        pairwise_rmsds: (M, M) pairwise RMSD matrix within ensemble
        min_ref_rmsds: (M,) minimum RMSD to any reference (inf if none)
        threshold: RMSD threshold in Angstrom

    Returns:
        kept_mask: (M,) boolean mask of kept structures
    """

    def scan_fn(kept_mask, i):
        # Too close to any reference structure?
        too_close_ref = min_ref_rmsds[i] < threshold

        # Too close to any already-kept structure?
        # Masked entries (not kept) become inf and cannot trigger rejection
        rmsds_to_kept = jnp.where(kept_mask, pairwise_rmsds[i], jnp.inf)
        too_close_kept = jnp.min(rmsds_to_kept) < threshold

        keep = ~too_close_ref & ~too_close_kept
        new_mask = kept_mask.at[i].set(keep)
        return new_mask, keep

    M = pairwise_rmsds.shape[0]
    init_mask = jnp.zeros(M, dtype=bool)
    final_mask, _ = jax.lax.scan(scan_fn, init_mask, jnp.arange(M))
    return final_mask


def prune_by_rmsd(
    ensemble: List[Tuple[Dict, float]],
    threshold: float = 0.25,
    reference: Optional[List[Tuple[Dict, float]]] = None,
    frag_perms: Optional[jnp.ndarray] = None,
) -> List[Tuple[Dict, float]]:
    """Remove structures that are too similar (RMSD < threshold).

    Computes the full pairwise RMSD matrix in a single JIT-compiled,
    vmap-vectorized pass, then performs greedy selection via jax.lax.scan.

    Args:
        ensemble: List of (structure, energy) tuples (should be sorted by energy)
        threshold: RMSD threshold in Angstrom
        reference: Optional existing ensemble to compare against (e.g. accumulated
                   conformers from previous phases). New structures matching any
                   reference structure are discarded.
        frag_perms: (P, N) int32 atom-index permutations for identical-fragment
                    invariance (NCI mode).  ``None`` → standard RMSD.
    Returns:
        pruned: List with similar structures removed
    """
    if not ensemble:
        return ensemble

    M = len(ensemble)
    ensemble_pos = jnp.array([s["positions"] for s, _ in ensemble])

    # Pairwise RMSD within ensemble — JIT + double vmap (split on OOM)
    pairwise_rmsds = _pairwise_rmsd_chunked(ensemble_pos, ensemble_pos, frag_perms)

    # Sanity check: detect NaN/Inf in pairwise RMSD matrix
    n_nan = int(jnp.sum(jnp.isnan(pairwise_rmsds)))
    if n_nan > 0:
        log_message(
            f"[RMSD Prune] WARNING: {n_nan} NaN values in pairwise RMSD matrix — "
            f"replacing with 0.0 (structures treated as identical)"
        )
        pairwise_rmsds = jnp.where(jnp.isnan(pairwise_rmsds), 0.0, pairwise_rmsds)

    # Log RMSD statistics (off-diagonal only)
    if M > 1:
        offdiag_mask = ~jnp.eye(M, dtype=bool)
        offdiag = pairwise_rmsds[offdiag_mask]
        log_message(
            f"[RMSD Prune] Pairwise RMSD stats: "
            f"min={float(jnp.min(offdiag)):.4f}, "
            f"median={float(jnp.median(offdiag)):.4f}, "
            f"max={float(jnp.max(offdiag)):.4f} Å"
        )

    # Cross RMSD to reference ensemble
    if reference:
        ref_pos = jnp.array([s["positions"] for s, _ in reference])
        cross_rmsds = _pairwise_rmsd_chunked(ensemble_pos, ref_pos, frag_perms)
        min_ref_rmsds = jnp.min(cross_rmsds, axis=1)
    else:
        min_ref_rmsds = jnp.full(M, jnp.inf)

    # Greedy selection — fully JIT-compiled via lax.scan
    kept_mask = _greedy_prune_jax(pairwise_rmsds, min_ref_rmsds, threshold)

    kept = [ensemble[i] for i in range(M) if kept_mask[i]]
    removed = M - len(kept)
    log_message(f"[RMSD Prune] Removed {removed} similar structures (threshold={threshold:.3g} Å)")
    log_message(f"[RMSD Prune] Kept {len(kept)}/{M} structures")

    return kept


def compute_pairwise_rmsd_ensemble(ensemble: List[Tuple[Dict, float]]) -> jnp.ndarray:
    """Compute pairwise RMSD matrix for ensemble.

    Args:
        ensemble: List of (structure, energy) tuples

    Returns:
        rmsd_matrix: (N, N) symmetric matrix of pairwise RMSDs

    Example:
        >>> rmsd_mat = compute_pairwise_rmsd_ensemble(ensemble)
        >>> print(f"Max RMSD: {jnp.max(rmsd_mat):.2f} Å")
    """
    n = len(ensemble)
    positions_array = jnp.array([struct["positions"] for struct, _ in ensemble])

    from .rmsd import pairwise_rmsd_matrix

    return pairwise_rmsd_matrix(positions_array)


def find_most_diverse_subset(
    ensemble: List[Tuple[Dict, float]], n_keep: int, min_rmsd: float = 1.0
) -> List[Tuple[Dict, float]]:
    """Select most diverse subset of structures.

    Uses greedy algorithm: iteratively add structure that maximizes
    minimum RMSD to already selected structures.

    Args:
        ensemble: List of (structure, energy) tuples
        n_keep: Number of structures to keep
        min_rmsd: Minimum RMSD constraint (Angstrom)

    Returns:
        diverse_subset: Selected diverse structures
    """
    if len(ensemble) <= n_keep:
        return ensemble

    # Always keep lowest energy structure
    selected = [ensemble[0]]
    remaining = list(ensemble[1:])

    log_message(f"[Diversity] Selecting {n_keep} diverse structures (min RMSD={min_rmsd} Å)")

    while len(selected) < n_keep and remaining:
        # Find structure with maximum minimum RMSD to selected
        best_idx = -1
        best_min_rmsd = -1

        for i, (struct, e) in enumerate(remaining):
            pos = struct["positions"]

            # Compute RMSD to all selected structures
            min_rmsd_to_selected = float("inf")
            for sel_struct, _ in selected:
                rmsd = float(rmsd_cv_jax(pos, sel_struct["positions"]))
                min_rmsd_to_selected = min(min_rmsd_to_selected, rmsd)

            # Keep track of structure with best (maximum) minimum RMSD
            if min_rmsd_to_selected > best_min_rmsd:
                best_min_rmsd = min_rmsd_to_selected
                best_idx = i

        # Add best structure if it meets minimum RMSD requirement
        if best_idx >= 0 and best_min_rmsd >= min_rmsd:
            selected.append(remaining[best_idx])
            remaining.pop(best_idx)
        else:
            # No more structures meet criterion
            break

    log_message(f"[Diversity] Selected {len(selected)} structures")

    return selected


def cluster_by_rmsd(
    ensemble: List[Tuple[Dict, float]], threshold: float = 0.5
) -> List[List[Tuple[Dict, float]]]:
    """Cluster ensemble by RMSD similarity.

    Uses simple single-linkage clustering.

    Args:
        ensemble: List of (structure, energy) tuples
        threshold: RMSD threshold for clustering (Angstrom)

    Returns:
        clusters: List of clusters (each cluster is a list of structures)

    Example:
        >>> clusters = cluster_by_rmsd(ensemble, threshold=0.5)
        >>> print(f"Found {len(clusters)} clusters")
        >>> for i, cluster in enumerate(clusters):
        ...     print(f"Cluster {i}: {len(cluster)} structures")
    """
    if not ensemble:
        return []

    clusters = [[ensemble[0]]]  # First structure starts first cluster

    for struct, e in ensemble[1:]:
        pos = struct["positions"]
        assigned = False

        # Try to assign to existing cluster
        for cluster in clusters:
            # Check RMSD to cluster representative (first structure)
            rep_pos = cluster[0][0]["positions"]
            rmsd = float(rmsd_cv_jax(pos, rep_pos))

            if rmsd < threshold:
                cluster.append((struct, e))
                assigned = True
                break

        # Create new cluster if not assigned
        if not assigned:
            clusters.append([(struct, e)])

    log_message(f"[Clustering] Found {len(clusters)} clusters (threshold={threshold} Å)")
    for i, cluster in enumerate(clusters):
        log_message(f"  Cluster {i+1}: {len(cluster)} structures")

    return clusters


def get_cluster_representatives(
    clusters: List[List[Tuple[Dict, float]]],
) -> List[Tuple[Dict, float]]:
    """Get lowest-energy structure from each cluster.

    Args:
        clusters: List of clusters

    Returns:
        representatives: Lowest-energy structure from each cluster
    """
    representatives = []
    for cluster in clusters:
        # Cluster is already sorted by energy (from original ensemble)
        representatives.append(cluster[0])
    return representatives


class TopologyJAX(NamedTuple):
    """JAX-friendly topology for fully-batched pruning.

    Stores bonded and non-bonded pairs as flat index arrays so that
    jax.vmap can process an entire ensemble in a single JIT call.
    """

    bond_i: jnp.ndarray  # (B,) first atom index of each bond
    bond_j: jnp.ndarray  # (B,) second atom index of each bond
    bond_L0: jnp.ndarray  # (B,) reference bond lengths (Å)
    nonbond_i: jnp.ndarray  # (P,) first atom index of each non-bonded pair
    nonbond_j: jnp.ndarray  # (P,) second atom index of each non-bonded pair
    rcov_sum: jnp.ndarray  # (P,) sum of covalent radii for each non-bonded pair (Å)


def build_topology_jax(topology: BondTopology, atomic_numbers) -> TopologyJAX:
    """Convert a BondTopology into JAX arrays suitable for batched pruning.

    This is a one-time, Python-side setup step.  The returned TopologyJAX
    can then be consumed by prune_by_topology which runs fully in JAX.

    Args:
        topology       : BondTopology from detect_bonds
        atomic_numbers : (N,) array of atomic numbers

    Returns:
        TopologyJAX with precomputed index arrays and covalent-radii sums
    """
    bonds = topology.bonds
    bond_i = jnp.array([b[0] for b in bonds], dtype=jnp.int32)
    bond_j = jnp.array([b[1] for b in bonds], dtype=jnp.int32)
    bond_L0 = jnp.asarray(topology.distances)

    bond_set = {(b[0], b[1]) for b in bonds}
    atomic_numbers_np = np.asarray(atomic_numbers)
    n_atoms = int(atomic_numbers_np.shape[0])
    radii = COVALENT_RADII[np.clip(atomic_numbers_np, 0, len(COVALENT_RADII) - 1)]

    nb_pairs = [
        (i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms) if (i, j) not in bond_set
    ]

    if nb_pairs:
        nonbond_i = jnp.array([p[0] for p in nb_pairs], dtype=jnp.int32)
        nonbond_j = jnp.array([p[1] for p in nb_pairs], dtype=jnp.int32)
        rcov_sum = jnp.array([radii[p[0]] + radii[p[1]] for p in nb_pairs])
    else:
        nonbond_i = jnp.zeros(0, dtype=jnp.int32)
        nonbond_j = jnp.zeros(0, dtype=jnp.int32)
        rcov_sum = jnp.zeros(0)

    return TopologyJAX(bond_i, bond_j, bond_L0, nonbond_i, nonbond_j, rcov_sum)


def _make_topology_check_fn(
    topo_jax: TopologyJAX,
    relative_stretch: float,
    absolute_increase: float,
    hard_max: float,
    new_bond_tolerance: float,
):
    """Build a JIT+vmap topology validity checker from a TopologyJAX.

    All thresholds are precomputed here (Python scalars / static JAX arrays)
    so the returned function is a pure, traceable JAX computation.

    Args:
        topo_jax          : precomputed topology arrays
        relative_stretch  : multiplicative slack on bond length
        absolute_increase : additive slack on bond length (Å)
        hard_max          : absolute distance ceiling (Å); use jnp.inf to disable
        new_bond_tolerance: covalent-radius tolerance for new-bond detection

    Returns:
        check_batch: (M, N, 3) -> (M,) bool mask — True means topology is valid
    """
    # Precompute per-bond upper thresholds once
    bond_thresh = jnp.minimum(
        topo_jax.bond_L0 * relative_stretch,
        topo_jax.bond_L0 + absolute_increase,
    )
    bond_thresh = jnp.minimum(bond_thresh, hard_max)

    # Precompute per-non-bond lower thresholds once
    nonbond_thresh = new_bond_tolerance * topo_jax.rcov_sum

    bond_i = topo_jax.bond_i
    bond_j = topo_jax.bond_j
    nonbond_i = topo_jax.nonbond_i
    nonbond_j = topo_jax.nonbond_j

    def _check_one(pos: jnp.ndarray) -> jnp.ndarray:
        """Check a single (N, 3) structure. Returns a scalar bool."""
        # Broken bonds: any bonded distance exceeds threshold
        dij_bonds = jnp.linalg.norm(pos[bond_i] - pos[bond_j], axis=-1)
        no_broken = jnp.all(dij_bonds <= bond_thresh)

        # New bonds: any non-bonded distance falls below covalent threshold
        dij_nb = jnp.linalg.norm(pos[nonbond_i] - pos[nonbond_j], axis=-1)
        no_new = jnp.all(dij_nb >= nonbond_thresh)

        return no_broken & no_new

    @jax.jit
    def check_batch(positions_batch: jnp.ndarray) -> jnp.ndarray:
        """Batch topology check.

        Args:
            positions_batch: (M, N, 3) array of M structures

        Returns:
            valid_mask: (M,) bool array — True if structure passes topology check
        """
        return jax.vmap(_check_one)(positions_batch)

    return check_batch


def build_nci_cross_mask(fragments, n_atoms: int) -> np.ndarray:
    """Build a boolean (n_atoms, n_atoms) mask that is True for cross-fragment pairs.

    Works for any number of fragments.  Precomputed once and reused for every
    pruning call, so no fragment-level Python loops at query time.
    """
    labels = np.empty(n_atoms, dtype=np.int32)
    for frag_id, atoms in enumerate(fragments):
        labels[atoms] = frag_id
    return labels[:, None] != labels[None, :]  # (n_atoms, n_atoms)


def min_inter_fragment_dist(positions: np.ndarray, cross_mask: np.ndarray) -> float:
    """Minimum inter-fragment atom-atom distance for a single structure (Å).

    Fully vectorised: uses the BLAS-backed identity
      d²(i,j) = ||p_i||² + ||p_j||² - 2 p_i·p_j
    to build the full (n_atoms, n_atoms) squared-distance matrix in one matmul,
    then masks same-fragment pairs and takes the global minimum.
    Works for any number of fragments.
    """
    norms2 = np.einsum("ij,ij->i", positions, positions)  # (n_atoms,)
    d2 = norms2[:, None] + norms2[None, :] - 2.0 * (positions @ positions.T)
    d2_cross = np.where(cross_mask, d2, np.inf)
    return float(np.sqrt(np.min(d2_cross)))


def prune_by_nci_distance(ensemble, cross_mask: np.ndarray, d_max: float, label: str):
    """Discard structures where min inter-fragment contact distance > d_max.

    Fully vectorised over the entire ensemble: positions are stacked into a
    single (N, n_atoms, 3) array, the squared-distance matrices are computed
    with a batched matmul, and the per-structure minimum is found in one
    NumPy call — no Python loops over structures or atom pairs.
    Works for any number of fragments.
    """
    if not ensemble:
        return ensemble

    positions_batch = np.stack([np.asarray(s["positions"]) for s, _ in ensemble])
    # (N, n_atoms, 3)
    N = positions_batch.shape[0]

    # Batched pairwise squared distances via expansion trick (avoids full diff tensor)
    norms2 = np.einsum("nij,nij->ni", positions_batch, positions_batch)  # (N, n_atoms)
    dots = np.matmul(positions_batch, positions_batch.transpose(0, 2, 1))  # (N, n_atoms, n_atoms)
    d2 = norms2[:, :, None] + norms2[:, None, :] - 2.0 * dots  # (N, n_atoms, n_atoms)

    # Mask same-fragment pairs with inf, take per-structure minimum
    d2_cross = np.where(cross_mask[None], d2, np.inf)  # (N, n_atoms, n_atoms)
    d_min_sq = d2_cross.reshape(N, -1).min(axis=1)  # (N,)

    keep_mask = d_min_sq <= d_max * d_max
    kept = [item for item, keep in zip(ensemble, keep_mask) if keep]
    n_disc = N - len(kept)
    if n_disc:
        log_message(
            f"[{label}] NCI distance filter: discarded {n_disc} dissociated structures "
            f"(contact > {d_max:.2f} Å)"
        )
    return kept


def prune_by_topology(
    ensemble: List[Tuple[Dict, float]],
    topology: BondTopology,
    relative_stretch: float = 1.6,
    absolute_increase: float = 0.60,
    hard_max: Optional[float] = None,
    new_bond_tolerance: float = 1.2,
) -> List[Tuple[Dict, float]]:
    """Discard structures with broken bonds or newly formed bonds.

    Broken bond: any bonded pair exceeds the allowed stretch.
        allow d_ij <= min( L0*relative_stretch, L0+absolute_increase, hard_max? )

    New bond: any non-bonded pair satisfies the covalent bond criterion.
        d_ij < new_bond_tolerance * (r_cov[i] + r_cov[j])

    The check is fully batched: all structures are evaluated in a single
    JIT-compiled jax.vmap call.

    Defaults:
        relative_stretch   = 1.6   # ~60% relative elongation
        absolute_increase  = 0.60  # absolute slack in Å
        hard_max           = None  # set e.g. 2.5 Å for a global ceiling
        new_bond_tolerance = 1.2   # slightly tighter than detect_bonds (1.3)

    Args:
        ensemble          : list of (structure_dict, energy)
        topology          : BondTopology from detect_bonds
        relative_stretch  : multiplicative slack vs. initial bond length
        absolute_increase : additive slack in Å vs. initial bond length
        hard_max          : absolute hard cap in Å if you want a global ceiling
        new_bond_tolerance: covalent-radius tolerance for new-bond detection

    Returns:
        filtered_ensemble
    """
    if not ensemble:
        return ensemble

    # One-time Python setup: convert topology to JAX index arrays
    sample_numbers = ensemble[0][0]["numbers"]
    topo_jax = build_topology_jax(topology, sample_numbers)

    # Build JIT+vmap checker (thresholds baked in at trace time)
    hard_max_val = jnp.inf if hard_max is None else float(hard_max)
    check_batch = _make_topology_check_fn(
        topo_jax, relative_stretch, absolute_increase, hard_max_val, new_bond_tolerance
    )

    # Stack all positions into a single (M, N, 3) array and run in one shot
    M = len(ensemble)
    positions_batch = jnp.stack([jnp.asarray(s["positions"]) for s, _ in ensemble])
    valid_mask = np.array(check_batch(positions_batch))  # (M,) bool

    kept = [ensemble[i] for i in range(M) if valid_mask[i]]

    # Count each failure mode for logging (cheap Python pass on the mask)
    removed = M - len(kept)
    log_message(
        f"[Topology Prune] Removed {removed} invalid structures " f"(broken or newly formed bonds)"
    )
    log_message(f"[Topology Prune] Kept {len(kept)}/{M} structures")
    return kept
