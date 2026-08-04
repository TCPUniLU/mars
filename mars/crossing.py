"""
Genetic Z-matrix crossing (GC) for conformational search.

Implements the GC algorithm where structural elements from existing conformers
are combined in internal (Z-matrix) coordinate space to generate new structures.

The crossing formula in internal coordinates is:
    R_new = R_ref + (R_low - R_high)

where R_low is the lower-energy parent, R_high is the higher-energy parent,
and R_ref is the reference (typically the lowest-energy conformer). This
transfers structural character from lower-energy parents toward the reference,
producing new conformer candidates for geometry optimization.

Reference:
    P. Pracht, F. Bohle, S. Grimme, PCCP, 2020, 22, 7169-7192
"""

from collections import deque
from typing import Dict, List, Tuple

import numpy as np

from .utils import COVALENT_RADII

# ============================================================================
# CN-based clash detection
# ============================================================================

_CN_K = 16.0  # Steepness of the CN damping function
_CN_CTHR = 0.3  # Max allowed |ΔCN| per atom before a child is rejected
_CLASH_RTHR = 0.7  # Hard clash: reject if r < _CLASH_RTHR * (rcov_i + rcov_j)


def _compute_cn(positions: np.ndarray, atomic_numbers: np.ndarray) -> np.ndarray:
    """Smooth coordination numbers using a distance-based damping function.

    CN_i = sum_{j≠i} 1 / (1 + exp(-k * (rco_ij / r_ij - 1)))

    where rco_ij = r_cov_i + r_cov_j (sum of covalent radii) and k=16.
    Returns CN per atom (float array of shape (n_atoms,)).

    Values close to integer coordination numbers for well-formed geometries;
    increases when atoms clash (r < rco), decreases when bonds stretch (r >> rco).
    """
    n = len(positions)
    max_z = len(COVALENT_RADII) - 1
    rcov = np.array([COVALENT_RADII[min(int(z), max_z)] for z in atomic_numbers])
    cn = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-8:
                continue
            rco = rcov[i] + rcov[j]
            cn[i] += 1.0 / (1.0 + np.exp(-_CN_K * (rco / r - 1.0)))
    return cn


# ============================================================================
# Z-matrix construction
# ============================================================================


def _build_adjacency(positions, atomic_numbers, tolerance=1.3):
    """Build molecular adjacency list from covalent bond detection."""
    from .utils import detect_bonds

    bonds, _ = detect_bonds(positions, atomic_numbers, tolerance=tolerance)

    n_atoms = len(positions)
    neighbors = {i: set() for i in range(n_atoms)}
    for i, j in bonds:
        neighbors[i].add(j)
        neighbors[j].add(i)

    return neighbors


def _build_zmatrix_order(neighbors, n_atoms):
    """Build Z-matrix atom ordering via BFS spanning tree.

    Each atom (except the first three) is defined by three reference atoms:
      - bond_ref:  bonded atom (defines bond length)
      - angle_ref: second atom (defines bond angle)
      - dih_ref:   third atom (defines dihedral angle)

    Returns:
        List of (atom, bond_ref, angle_ref, dih_ref) tuples.
        For the first atoms, some refs are None.
    """
    visited = set()
    order = []
    parent = {}

    # BFS from atom 0
    queue = deque([0])
    visited.add(0)
    parent[0] = None

    while queue:
        atom = queue.popleft()
        order.append(atom)
        for nb in sorted(neighbors[atom]):
            if nb not in visited:
                visited.add(nb)
                parent[nb] = atom
                queue.append(nb)

    # Handle disconnected fragments
    for i in range(n_atoms):
        if i not in visited:
            order.append(i)
            parent[i] = order[0]
            visited.add(i)

    # Build Z-matrix entries from the BFS tree
    zmat = []
    placed = set()

    for idx, atom in enumerate(order):
        if idx == 0:
            zmat.append((atom, None, None, None))
            placed.add(atom)
        elif idx == 1:
            zmat.append((atom, order[0], None, None))
            placed.add(atom)
        elif idx == 2:
            bond_ref = parent[atom]
            # Pick angle_ref as another already-placed atom
            angle_ref = None
            for prev in order[:idx]:
                if prev != bond_ref:
                    angle_ref = prev
                    break
            zmat.append((atom, bond_ref, angle_ref, None))
            placed.add(atom)
        else:
            bond_ref = parent[atom]

            # angle_ref: parent of bond_ref, or any other placed atom
            angle_ref = parent.get(bond_ref)
            if angle_ref is None or angle_ref == atom:
                for prev in order[:idx]:
                    if prev not in (atom, bond_ref):
                        angle_ref = prev
                        break

            # dih_ref: parent of angle_ref, or any other placed atom
            dih_ref = parent.get(angle_ref)
            if dih_ref is None or dih_ref in (atom, bond_ref):
                for prev in order[:idx]:
                    if prev not in (atom, bond_ref, angle_ref):
                        dih_ref = prev
                        break

            zmat.append((atom, bond_ref, angle_ref, dih_ref))
            placed.add(atom)

    return zmat


# ============================================================================
# Internal coordinate computation
# ============================================================================


def _bond_length(pos, i, j):
    """Distance between atoms i and j."""
    return np.linalg.norm(pos[i] - pos[j])


def _bond_angle(pos, i, j, k):
    """Angle i-j-k in radians (j is the central/vertex atom)."""
    v1 = pos[i] - pos[j]
    v2 = pos[k] - pos[j]
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    return np.arccos(np.clip(cos_a, -1.0, 1.0))


def _dihedral_angle(pos, i, j, k, l):
    """Dihedral angle i-j-k-l in radians, using the atan2 sign convention."""
    b1 = pos[j] - pos[i]
    b2 = pos[k] - pos[j]
    b3 = pos[l] - pos[k]

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1_len = np.linalg.norm(n1)
    n2_len = np.linalg.norm(n2)
    if n1_len < 1e-10 or n2_len < 1e-10:
        return 0.0

    n1 /= n1_len
    n2 /= n2_len

    b2_hat = b2 / (np.linalg.norm(b2) + 1e-12)
    m1 = np.cross(n1, b2_hat)

    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    return -np.arctan2(y, x)


def _positions_to_internals(positions, zmat):
    """Convert Cartesian positions to Z-matrix internal coordinates.

    Returns:
        (n_atoms, 3) array with columns [bond_length, angle, dihedral].
        Entries that are not defined (first atoms) are NaN.
    """
    n = len(zmat)
    ic = np.full((n, 3), np.nan)

    for idx, (atom, br, ar, dr) in enumerate(zmat):
        if br is not None:
            ic[idx, 0] = _bond_length(positions, atom, br)
        if ar is not None:
            ic[idx, 1] = _bond_angle(positions, atom, br, ar)
        if dr is not None:
            ic[idx, 2] = _dihedral_angle(positions, atom, br, ar, dr)

    return ic


# ============================================================================
# NeRF: internal coordinates -> Cartesian
# ============================================================================


def _place_atom_nerf(pos_a, pos_b, pos_c, d, theta, phi):
    """Place a new atom using the Natural Extension Reference Frame (NeRF).

    The new atom is at distance d from pos_a, with angle theta (new-a-b)
    and dihedral phi (new-a-b-c).

    Args:
        pos_a: Bond reference position (new atom bonded to this).
        pos_b: Angle reference position.
        pos_c: Dihedral reference position.
        d: Bond length (Angstrom).
        theta: Bond angle (radians).
        phi: Dihedral angle (radians).

    Returns:
        New atom position (3,).
    """
    d_ab = pos_b - pos_a
    d_bc = pos_c - pos_b

    n_ab = d_ab / (np.linalg.norm(d_ab) + 1e-12)

    # Normal to the a-b-c plane
    n = np.cross(d_ab, d_bc)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-10:
        # Collinear: pick arbitrary perpendicular
        perp = np.array([1.0, 0.0, 0.0]) if abs(n_ab[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        n = np.cross(n_ab, perp)
    n = n / (np.linalg.norm(n) + 1e-12)

    # Right-handed local frame
    e_x = -n_ab
    e_z = n
    e_y = np.cross(e_z, e_x)

    # Spherical-to-local-Cartesian
    s_theta = np.sin(np.pi - theta)
    c_theta = np.cos(np.pi - theta)

    x = d * c_theta
    y = d * s_theta * np.cos(phi)
    z = d * s_theta * np.sin(phi)

    return pos_a + x * e_x + y * e_y + z * e_z


def _internals_to_positions(ic, zmat, ref_positions):
    """Reconstruct Cartesian positions from Z-matrix internal coordinates.

    The first atom is placed at its reference position. The second atom is
    placed along the reference bond direction at the new bond length. The
    third atom is placed using bond length + angle in the reference plane.
    All subsequent atoms use the full NeRF placement.

    Args:
        ic: (n_atoms, 3) internal coordinates [d, theta, phi].
        zmat: Z-matrix connectivity list.
        ref_positions: Reference Cartesian positions (sets global orientation).

    Returns:
        (n_atoms, 3) reconstructed positions.
    """
    n = len(zmat)
    pos = np.zeros((n, 3))

    # Atom 0: at reference position
    a0 = zmat[0][0]
    pos[a0] = ref_positions[a0]

    if n < 2:
        return pos

    # Atom 1: along reference bond direction, scaled to new bond length
    a1, br1, _, _ = zmat[1]
    direction = ref_positions[a1] - ref_positions[br1]
    d_norm = np.linalg.norm(direction)
    if d_norm > 1e-10:
        direction /= d_norm
    else:
        direction = np.array([1.0, 0.0, 0.0])
    pos[a1] = pos[br1] + ic[1, 0] * direction

    if n < 3:
        return pos

    # Atom 2: using bond length + angle (dihedral set to 0 in a dummy frame)
    a2, br2, ar2, _ = zmat[2]
    dummy = pos[br2] + np.array([0.0, 0.0, 1.0])
    pos[a2] = _place_atom_nerf(pos[br2], pos[ar2], dummy, ic[2, 0], ic[2, 1], 0.0)

    # Remaining atoms: full NeRF
    for idx in range(3, n):
        atom, br, ar, dr = zmat[idx]
        pos[atom] = _place_atom_nerf(pos[br], pos[ar], pos[dr], ic[idx, 0], ic[idx, 1], ic[idx, 2])

    return pos


# ============================================================================
# Z-matrix genetic crossing
# ============================================================================


def _wrap_dihedral(d):
    """Wrap angle difference to [-pi, pi]."""
    return (d + np.pi) % (2 * np.pi) - np.pi


def genetic_zmatrix_crossing(
    ensemble: List[Tuple[Dict, float]],
    reference_idx: int = 0,
    n_children: int = 20,
    random_seed: int = 42,
    boltzmann_weight: bool = True,
    temperature: float = 1.0,
) -> List[Dict]:
    """Generate new conformers via Z-matrix genetic crossing (GC).

    For each child, two parent structures are selected from the ensemble.
    The lower-energy parent (R_low) and higher-energy parent (R_high) are
    combined with the reference R_ref in internal coordinate space:

        R_new = R_ref + (R_low - R_high)

    Bond length and bond angle differences are applied directly. Dihedral
    angle differences are wrapped to [-pi, pi] before addition. The
    resulting internal coordinates are converted back to Cartesian for
    subsequent geometry optimization.

    Args:
        ensemble: List of (structure_dict, energy) tuples. Structures must
            share the same atom ordering and connectivity.
        reference_idx: Index of the reference structure (default: 0,
            i.e. the lowest-energy conformer in a sorted ensemble).
        n_children: Number of child structures to generate.
        random_seed: Random seed for reproducibility.
        boltzmann_weight: If True, select parents with Boltzmann weighting
            so lower-energy conformers are chosen more often.
        temperature: Temperature for Boltzmann weighting (in the same
            energy units as the ensemble energies, typically eV).

    Returns:
        List of structure dicts with Cartesian positions ready for
        geometry optimization.
    """
    if len(ensemble) < 2:
        return []

    rng = np.random.default_rng(random_seed)

    # Reference structure and topology
    ref_struct, _ = ensemble[reference_idx]
    ref_pos = np.asarray(ref_struct["positions"])
    atomic_numbers = np.asarray(ref_struct["numbers"])
    n_atoms = ref_pos.shape[0]

    if n_atoms < 4:
        return []  # Need >= 4 atoms for meaningful dihedral crossing

    # Build Z-matrix from reference topology
    neighbors = _build_adjacency(ref_pos, atomic_numbers)
    zmat = _build_zmatrix_order(neighbors, n_atoms)

    # Reference CN — used to screen children for clashes / broken bonds
    cn_ref = _compute_cn(ref_pos, atomic_numbers)

    # Compute internal coordinates for all ensemble members
    ref_ic = _positions_to_internals(ref_pos, zmat)

    all_ic = []
    for struct, _ in ensemble:
        pos = np.asarray(struct["positions"])
        all_ic.append(_positions_to_internals(pos, zmat))

    # Parent selection weights
    if boltzmann_weight:
        energies = np.array([e for _, e in ensemble])
        shifted = energies - energies.min()
        weights = np.exp(-shifted / max(temperature, 1e-8))
        weights /= weights.sum()
    else:
        weights = np.ones(len(ensemble)) / len(ensemble)

    # Generate children
    children = []
    n_rejected_cn = 0
    for _ in range(n_children):
        # Select two distinct parents
        idx_i = rng.choice(len(ensemble), p=weights)
        idx_j = rng.choice(len(ensemble), p=weights)
        while idx_j == idx_i and len(ensemble) > 2:
            idx_j = rng.choice(len(ensemble), p=weights)

        ic_i = all_ic[idx_i]
        ic_j = all_ic[idx_j]

        # Determine energy ordering: lower index in sorted ensemble = lower energy
        if idx_i < idx_j:
            ic_low, ic_high = ic_i, ic_j
        else:
            ic_low, ic_high = ic_j, ic_i

        # R_new = R_ref + (R_low - R_high)
        new_ic = ref_ic.copy()

        for k in range(n_atoms):
            # Bond length
            if not np.isnan(ref_ic[k, 0]):
                new_ic[k, 0] = ref_ic[k, 0] + ic_low[k, 0] - ic_high[k, 0]
                new_ic[k, 0] = max(new_ic[k, 0], 0.5)  # floor

            # Bond angle
            if not np.isnan(ref_ic[k, 1]):
                new_ic[k, 1] = ref_ic[k, 1] + ic_low[k, 1] - ic_high[k, 1]
                new_ic[k, 1] = np.clip(new_ic[k, 1], 0.01, np.pi - 0.01)

            # Dihedral angle (periodic wrapping)
            if not np.isnan(ref_ic[k, 2]):
                new_ic[k, 2] = ref_ic[k, 2] + _wrap_dihedral(ic_low[k, 2] - ic_high[k, 2])

        # Reconstruct Cartesian positions
        new_pos = _internals_to_positions(new_ic, zmat, ref_pos)

        # Hard distance clash check: reject if any atom pair is closer than
        # _CLASH_RTHR * (rcov_i + rcov_j), regardless of bonding.
        rcov = np.array(
            [COVALENT_RADII[min(int(z), len(COVALENT_RADII) - 1)] for z in atomic_numbers]
        )
        clashed = False
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r = np.linalg.norm(new_pos[i] - new_pos[j])
                if r < _CLASH_RTHR * (rcov[i] + rcov[j]):
                    clashed = True
                    break
            if clashed:
                break
        if clashed:
            n_rejected_cn += 1
            continue

        # CN clash / broken-bond check: discard if any atom's coordination
        # number deviates too far from the reference geometry.
        # A large increase means a steric clash; a large decrease means a
        # bond has been stretched beyond bonding range.
        cn_child = _compute_cn(new_pos, atomic_numbers)
        if np.any(np.abs(cn_child - cn_ref) > _CN_CTHR):
            n_rejected_cn += 1
            continue

        child = {
            "positions": new_pos,
            "symbols": ref_struct["symbols"],
            "numbers": ref_struct["numbers"],
        }
        children.append(child)

    if n_rejected_cn > 0:
        from .log import log_message

        log_message(
            f"[Crossing] CN check: rejected {n_rejected_cn}/{n_children} children "
            f"(clashes or broken bonds), kept {len(children)}"
        )

    return children
