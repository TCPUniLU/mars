"""
JAX-native RMSD calculations with automatic differentiation.

Provides Kabsch alignment and RMSD collective variable with gradients
computed via JAX autodiff.
"""

from typing import List, Tuple

import jax
import jax.numpy as jnp


def kabsch_jax(P: jnp.ndarray, Q: jnp.ndarray) -> jnp.ndarray:
    """Compute optimal rotation matrix aligning Q onto P using Kabsch algorithm.

    Args:
        P: (N, 3) array of reference positions (centered)
        Q: (N, 3) array of positions to align (centered)

    Returns:
        R: (3, 3) optimal rotation matrix
    """
    C = Q.T @ P
    U, S, Vt = jnp.linalg.svd(C, full_matrices=False)
    d = jnp.sign(jnp.linalg.det(U @ Vt))
    D = jnp.diag(jnp.array([1.0, 1.0, d]))
    R = U @ D @ Vt
    return R


def center_positions(positions: jnp.ndarray) -> jnp.ndarray:
    """Center positions by subtracting the centroid.

    Args:
        positions: (N, 3) array of positions

    Returns:
        centered: (N, 3) array of centered positions
    """
    return positions - jnp.mean(positions, axis=0, keepdims=True)


def align_to_reference(positions: jnp.ndarray, reference: jnp.ndarray) -> jnp.ndarray:
    """Align positions to reference frame using Kabsch algorithm.

    Both structures are centered, then positions are rotated to minimize
    RMSD with respect to reference.

    Args:
        positions: (N, 3) array of positions to align
        reference: (N, 3) array of reference positions

    Returns:
        aligned: (N, 3) array of aligned positions (centered and rotated)

    Note:
        Result is in the reference frame, suitable for use with MTD in aligned mode.
    """
    pos_centered = center_positions(positions)
    ref_centered = center_positions(reference)
    R = kabsch_jax(ref_centered, pos_centered)
    return pos_centered @ R


def rmsd_cv_jax(X: jnp.ndarray, Y: jnp.ndarray) -> float:
    """Compute RMSD between two sets of positions after Kabsch alignment.

    Args:
        X: (N, 3) array of current positions
        Y: (N, 3) array of reference positions

    Returns:
        rmsd: Root-mean-square deviation in Angstrom

    Note:
        Differentiable via JAX autodiff. Use jax.grad() to compute dRMSD/dX.
    """
    Xc = center_positions(X)
    Yc = center_positions(Y)
    R = kabsch_jax(Xc, Yc)
    Y_aligned = Yc @ R
    diff = Xc - Y_aligned
    N = X.shape[0]
    msd = jnp.sum(diff**2) / N
    return jnp.sqrt(msd + 1e-16)


def rmsd_and_grad_jax(X: jnp.ndarray, Y: jnp.ndarray) -> Tuple[float, jnp.ndarray]:
    """Compute RMSD and its gradient with respect to X.

    Args:
        X: (N, 3) array of current positions
        Y: (N, 3) array of reference positions

    Returns:
        rmsd: Root-mean-square deviation
        grad: (N, 3) array of gradient dRMSD/dX
    """

    def rmsd_fn(x):
        return rmsd_cv_jax(x, Y)

    rmsd, grad = jax.value_and_grad(rmsd_fn)(X)
    return rmsd, grad


def rmsd_to_multiple_refs(X: jnp.ndarray, refs: jnp.ndarray) -> jnp.ndarray:
    """Compute RMSD from X to multiple reference structures.

    Uses jax.vmap for efficient parallel computation.

    Args:
        X: (N, 3) array of current positions
        refs: (M, N, 3) array of M reference structures

    Returns:
        rmsds: (M,) array of RMSD values
    """
    batched_rmsd = jax.vmap(lambda ref: rmsd_cv_jax(X, ref))
    return batched_rmsd(refs)


def pairwise_rmsd_matrix(structures: jnp.ndarray) -> jnp.ndarray:
    """Compute pairwise RMSD matrix for a set of structures.

    Args:
        structures: (M, N, 3) array of M structures with N atoms

    Returns:
        rmsd_matrix: (M, M) symmetric matrix of pairwise RMSDs
    """
    M = structures.shape[0]

    def compute_row(i):
        return rmsd_to_multiple_refs(structures[i], structures)

    rmsd_matrix = jax.vmap(compute_row)(jnp.arange(M))
    return rmsd_matrix


@jax.jit
def radius_of_gyration(positions: jnp.ndarray) -> jnp.ndarray:
    """Compute unweighted radius of gyration.

    Rotation and translation invariant scalar shape descriptor.

    Args:
        positions: (N, 3) array of atomic positions

    Returns:
        rg: scalar radius of gyration in same units as positions
    """
    com = positions.mean(axis=0)
    centered = positions - com
    return jnp.sqrt(jnp.mean(jnp.sum(centered**2, axis=-1)))


def batch_radius_of_gyration(positions: jnp.ndarray) -> jnp.ndarray:
    """Compute radius of gyration for a batch of structures via vmap.

    Args:
        positions: (M, N, 3) array of M structures

    Returns:
        rgs: (M,) array of radii of gyration
    """
    return jax.vmap(radius_of_gyration)(positions)


# JIT-compiled versions
rmsd_cv_jax_jit = jax.jit(rmsd_cv_jax)
kabsch_jax_jit = jax.jit(kabsch_jax)
batch_radius_of_gyration_jit = jax.jit(batch_radius_of_gyration)


def compute_fragment_permutations(
    fragments: List[List[int]],
    atomic_numbers,
    max_permutations: int = 10_000,
) -> jnp.ndarray:
    """Compute atom-index arrays for all permutations of identical fragments.

    Two fragments are considered identical when they contain the same sorted
    list of atomic numbers.  For each group of identical fragments, all
    permutations within that group are enumerated.  The Cartesian product
    across groups gives the full set of permutations.

    Args:
        fragments:      List of atom-index lists (one per molecular fragment),
                        as returned by ``MoleculeTopology.fragments``.
        atomic_numbers: (n_atoms,) array of atomic numbers.
        max_permutations: Upper bound on the number of permutations to
            materialize. The natural count is the product of factorials of
            the identical-fragment group sizes — for a solute with 10
            identical solvents that's ``10! ≈ 3.6e6`` rows of a (P, N)
            int32 array (~1 GB just for the index buffer, plus Python
            list overhead during construction). When the natural count
            would exceed this cap, the function emits a warning and
            returns only the identity permutation so the caller falls
            back to standard (non-permutation-invariant) RMSD pruning
            rather than OOM-killing the process.

    Returns:
        perms: (n_perms, n_atoms) int32 JAX array.  Row 0 is always the
               identity permutation. When the cap is exceeded the array
               has shape (1, n_atoms).
    """
    import math
    import warnings
    from collections import defaultdict
    from itertools import permutations as _ip
    from itertools import product as _product

    atomic_numbers = jnp.asarray(atomic_numbers, dtype=jnp.int32)
    n_atoms_total = sum(len(f) for f in fragments)

    def _fingerprint(frag):
        return tuple(sorted(int(atomic_numbers[i]) for i in frag))

    # Group fragment indices by composition fingerprint
    groups = defaultdict(list)
    for i, frag in enumerate(fragments):
        groups[_fingerprint(frag)].append(i)

    # Upfront count: product of factorials of group sizes. Bail out before
    # enumerating if it would explode — the standard caller already
    # handles ``perms.shape[0] == 1`` as "no permutation invariance, use
    # plain RMSD".
    total_perms = 1
    largest_group = 0
    for group_idxs in groups.values():
        total_perms *= math.factorial(len(group_idxs))
        largest_group = max(largest_group, len(group_idxs))

    if total_perms > max_permutations:
        # Rebuild via fragments so the identity matches the same atom
        # ordering convention the enumerated path would have produced.
        atom_order = []
        for frag in fragments:
            atom_order.extend(frag)
        msg = (
            f"[NCI] {total_perms} fragment permutations would be required "
            f"(largest identical-fragment group: {largest_group}); this exceeds the "
            f"cap of {max_permutations} and would OOM. Falling back to standard "
            f"(non-permutation-invariant) RMSD pruning. Pass a larger "
            f"`max_permutations` if you need the full enumeration."
        )
        # Surface in both the user-facing MARS log and the Python warnings
        # stream so it shows up regardless of how the caller configured
        # logging.
        try:
            from .log import log_warning

            log_warning(msg)
        except Exception:
            pass
        warnings.warn(msg, stacklevel=2)
        return jnp.array([atom_order], dtype=jnp.int32)

    # Per-group permutations and their positions in the fragment list
    per_group_perms = []
    group_positions = []
    for _fp, group_idxs in groups.items():
        per_group_perms.append(list(_ip(group_idxs)))
        group_positions.append(group_idxs)

    # Cartesian product across groups → all fragment orderings
    atom_perms = []
    for combo in _product(*per_group_perms):
        frag_order = list(range(len(fragments)))
        for gpos, perm in zip(group_positions, combo):
            for slot, src in zip(gpos, perm):
                frag_order[slot] = src
        atom_order = []
        for frag_idx in frag_order:
            atom_order.extend(fragments[frag_idx])
        atom_perms.append(atom_order)

    return jnp.array(atom_perms, dtype=jnp.int32)


def check_rmsd_gradient(
    X: jnp.ndarray, Y: jnp.ndarray, epsilon: float = 1e-5
) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """Verify RMSD gradient using finite differences.

    Args:
        X: (N, 3) array of positions
        Y: (N, 3) array of reference
        epsilon: Step size for finite differences

    Returns:
        grad_analytical: Gradient from JAX autodiff
        grad_numerical: Gradient from finite differences
        max_error: Maximum absolute difference
    """
    rmsd_val, grad_analytical = rmsd_and_grad_jax(X, Y)

    grad_numerical = jnp.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(3):
            X_plus = X.at[i, j].set(X[i, j] + epsilon)
            rmsd_plus = rmsd_cv_jax(X_plus, Y)

            X_minus = X.at[i, j].set(X[i, j] - epsilon)
            rmsd_minus = rmsd_cv_jax(X_minus, Y)

            grad_numerical = grad_numerical.at[i, j].set((rmsd_plus - rmsd_minus) / (2 * epsilon))

    max_error = jnp.max(jnp.abs(grad_analytical - grad_numerical))

    return grad_analytical, grad_numerical, max_error
