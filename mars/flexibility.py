"""
Molecular flexibility metric.

Implements a bond-based flexibility score considering:

- Molecular branching (coordination)
- Ring structures (rigidity)
- Bond types (single vs multiple bonds)
- Molecular size normalization
"""

from typing import List, Optional, Tuple

import numpy as np


def calculate_flexibility(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    bonds: Optional[List[Tuple[int, int]]] = None,
    bond_orders: Optional[np.ndarray] = None,
) -> float:
    """Calculate molecular flexibility from the bond topology.

    Flexibility is a normalized metric (0 = rigid, 1 = flexible) based on:
    1. Branching factor: linear chains are more flexible than branched structures
    2. Ring penalty: rings reduce flexibility by ~50%
    3. Bond order: double/triple bonds reduce flexibility
    4. Coordination: sp2 carbons are more rigid than sp3

    Args:
        positions: (n_atoms, 3) atomic coordinates in Angstrom
        atomic_numbers: (n_atoms,) atomic numbers
        bonds: List of (i, j) bond tuples, or None to auto-detect
        bond_orders: (n_bonds,) bond orders (1=single, 2=double, 3=triple), or None for all single

    Returns:
        flexibility: Float between 0 (rigid) and 1 (flexible long chain)

    Example:
        >>> # Linear alkane (flexible)
        >>> flex = calculate_flexibility(positions, atomic_numbers)
        >>> print(f"Flexibility: {flex:.3f}")  # ~0.8-1.0 for long chains

        >>> # Aromatic ring (rigid)
        >>> flex = calculate_flexibility(positions, atomic_numbers)
        >>> print(f"Flexibility: {flex:.3f}")  # ~0.2-0.4 for rings
    """
    n_atoms = len(atomic_numbers)

    if n_atoms < 2:
        return 0.0  # Single atom has no flexibility

    # Auto-detect bonds if not provided
    if bonds is None:
        bonds = detect_bonds_simple(positions, atomic_numbers)

    if len(bonds) == 0:
        return 0.0  # No bonds = no flexibility

    # Calculate coordination numbers (number of bonds per atom)
    coordination = np.zeros(n_atoms, dtype=int)
    for i, j in bonds:
        coordination[i] += 1
        coordination[j] += 1

    # Calculate branching factors for each bond
    # branch = 2 / sqrt(coord_i * coord_j)
    # Linear segments (coord=2,2): branch = 1.0
    # Branching (coord=3,3): branch = 0.67
    # Terminal bonds (coord == 1 on either side) are excluded.

    branch_values = []
    active_bonds = []

    for idx, (i, j) in enumerate(bonds):
        coord_i = coordination[i]
        coord_j = coordination[j]

        # Skip terminal bonds.
        if coord_i == 1 or coord_j == 1:
            continue

        # Calculate branching factor
        branch = 2.0 / np.sqrt(coord_i * coord_j)
        branch_values.append(branch)
        active_bonds.append(idx)

    if len(branch_values) == 0:
        return 0.0  # No internal bonds

    # Ring detection (simplified heuristic): atoms with coordination > 2
    # are likely involved in a ring and get a 50% flexibility penalty.
    ring_factors = []
    for i, j in [bonds[idx] for idx in active_bonds]:
        is_ring = (coordination[i] > 2) or (coordination[j] > 2)
        ring_f = 0.5 if is_ring else 1.0
        ring_factors.append(ring_f)

    # Bond order factors: double / triple bonds reduce flexibility.
    if bond_orders is not None:
        bond_order_factors = []
        for idx in active_bonds:
            wbo = bond_orders[idx]
            # Simplified: single bonds (1.0) are flexible, double/triple reduce flexibility
            if wbo > 1.5:  # Double or triple bond
                bo_f = 0.3  # Significant rigidity
            elif wbo > 1.2:  # Partial double bond character
                bo_f = 0.6
            else:  # Single bond
                bo_f = 1.0
            bond_order_factors.append(bo_f)
    else:
        bond_order_factors = [1.0] * len(branch_values)

    # Hybridization factors (simplified): a carbon with coordination 3
    # is treated as sp2 and gets a 50% flexibility penalty.
    hyb_factors = []
    for i, j in [bonds[idx] for idx in active_bonds]:
        # sp2 carbon: atomic_number=6, coord=3
        is_sp2_i = (atomic_numbers[i] == 6) and (coordination[i] == 3)
        is_sp2_j = (atomic_numbers[j] == 6) and (coordination[j] == 3)

        if is_sp2_i or is_sp2_j:
            hyb_f = 0.5  # sp2 penalty
        else:
            hyb_f = 1.0
        hyb_factors.append(hyb_f)

    # Combine all per-bond factors.
    combined = (
        np.array(branch_values)
        * np.array(ring_factors)
        * np.array(bond_order_factors)
        * np.array(hyb_factors)
    )

    # Quadratic average: sqrt(sum(val^2) / m).
    flexibility = np.sqrt(np.mean(combined**2))

    return float(flexibility)


def detect_bonds_simple(
    positions: np.ndarray, atomic_numbers: np.ndarray, tolerance: float = 1.3
) -> List[Tuple[int, int]]:
    """Detect bonds using covalent radii with tolerance.

    Args:
        positions: (n_atoms, 3) atomic coordinates
        atomic_numbers: (n_atoms,) atomic numbers
        tolerance: Multiplicative factor for covalent radii sum (default: 1.3)

    Returns:
        bonds: List of (i, j) bond pairs
    """
    from .utils import COVALENT_RADII

    n_atoms = len(positions)
    bonds = []

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            zi = int(atomic_numbers[i])
            zj = int(atomic_numbers[j])
            r_i = float(COVALENT_RADII[zi]) if zi < len(COVALENT_RADII) else 1.0
            r_j = float(COVALENT_RADII[zj]) if zj < len(COVALENT_RADII) else 1.0

            max_dist = (r_i + r_j) * tolerance
            dist = np.linalg.norm(positions[i] - positions[j])

            if dist < max_dist:
                bonds.append((i, j))

    return bonds


def estimate_rotatable_bonds(
    positions: np.ndarray, atomic_numbers: np.ndarray, bonds: Optional[List[Tuple[int, int]]] = None
) -> int:
    """Estimate the number of rotatable bonds.

    A bond is rotatable if:
    - It's a single bond
    - Both atoms have coordination > 1 (not terminal)
    - Not in a ring (simplified heuristic)

    Args:
        positions: (n_atoms, 3) atomic coordinates
        atomic_numbers: (n_atoms,) atomic numbers
        bonds: Optional list of bonds

    Returns:
        n_rotatable: Number of rotatable bonds
    """
    if bonds is None:
        bonds = detect_bonds_simple(positions, atomic_numbers)

    n_atoms = len(atomic_numbers)
    coordination = np.zeros(n_atoms, dtype=int)
    for i, j in bonds:
        coordination[i] += 1
        coordination[j] += 1

    n_rotatable = 0
    for i, j in bonds:
        # Must be non-terminal
        if coordination[i] <= 1 or coordination[j] <= 1:
            continue

        # Heuristic: exclude if either atom is sp2-like (likely aromatic)
        is_sp2_i = (atomic_numbers[i] == 6) and (coordination[i] == 3)
        is_sp2_j = (atomic_numbers[j] == 6) and (coordination[j] == 3)

        if is_sp2_i or is_sp2_j:
            continue

        n_rotatable += 1

    return n_rotatable


def calculate_effective_atoms(flexibility: float, n_atoms: int) -> float:
    """Calculate effective atom count weighted by flexibility.

    Represents the "effective carbon chain length equivalent",
    computed as ``flexibility * n_atoms``.

    Args:
        flexibility: Molecular flexibility (0-1)
        n_atoms: Total number of atoms

    Returns:
        effective_atoms: Flexibility-weighted atom count
    """
    return flexibility * n_atoms


# ============================================================================
# Multi-Step MTD Workflow Configuration
# ============================================================================


def get_mtd_workflow_config(
    flexibility: float, n_atoms: int, mode: str = "normal", mtd_kscal: float = 1.0
) -> dict:
    """Generate multi-step MTD workflow parameters adapted to molecular flexibility.

    Creates a grid of (kpush, alpha) parameter combinations for multi-step
    MTD sampling. Each combination defines one MTD simulation step. The grid
    dimensions and parameter values scale with the search mode. Flexibility
    enters through the simulation time, not the bias parameters.

    Grid dimensions per mode:
      quick:    nk=2 x na=3 =  6 MTD steps
      normal:   nk=3 x na=4 = 12 MTD steps
      thorough: nk=4 x na=6 = 24 MTD steps

    kpush is scaled by the number of atoms:
      kpush = kstart (eV) x n_atoms x mtd_kscal

    Simulation time depends on flexibility:
      tmtd = 3.0 * exp(0.10 * flex * max(1, n_atoms - 8))
      (capped at [5, 500] ps)

    After the grid MTDs, a conformer refinement phase runs additional MTDs
    from diverse starting conformers (disabled for quick mode).

    Args:
        flexibility: Molecular flexibility (0-1)
        n_atoms: Number of atoms
        mode: 'quick', 'normal', or 'thorough'
        mtd_kscal: Global kpush scaling factor (default: 1.0)

    Returns:
        config: Dict with keys ``mtd_steps`` (list of {kpush, alpha, cvdump_fs}
            dicts), ``time_per_step_ps`` (MTD time per step in ps),
            ``n_conformer_starts`` (diverse starts for refinement),
            ``refinement_time_ps`` (time per refinement MTD in ps),
            ``refinement_params`` ({kpush, alpha, cvdump_fs} for refinement),
            and ``min_rmsd_threshold`` (RMSD threshold for diversity, Å).
    """
    import math

    # Mode-dependent grid dimensions and base parameters.
    #
    # kstart_ev is in eV per atom; multiplied by n_atoms and mtd_kscal.
    # alpha_start_ang is in A^-2.
    # cvdump = 1000 fs (Vbias snapshot every 1 ps).
    mode_config = {
        "quick": {
            "nk": 2,
            "na": 3,
            "kstart_ev": 0.054422772491976,
            "kinc": 2.0,
            "alpha_start_ang": 4.285277791311758,
            "alpinc": 2.0,
            "cvdump_fs": 1000.0,
            "rfac": 0.5,
            "n_conformer_starts": 0,
        },
        "normal": {
            "nk": 3,
            "na": 4,
            "kstart_ev": 0.081634158737964,
            "kinc": 2.0,
            "alpha_start_ang": 4.642384273921071,
            "alpinc": 5.0 / 3.0,
            "cvdump_fs": 1000.0,
            "rfac": 1.0,
            "n_conformer_starts": 3,
        },
        "thorough": {
            "nk": 4,
            "na": 6,
            "kstart_ev": 0.081634158737964,
            "kinc": 3.0 / 2.0,
            "alpha_start_ang": 4.642384273921071,
            "alpinc": 4.0 / 3.0,
            "cvdump_fs": 1000.0,
            "rfac": 1.0,
            "n_conformer_starts": 6,
        },
    }

    cfg = mode_config.get(mode, mode_config["normal"])

    kpush_base = cfg["kstart_ev"] * n_atoms * mtd_kscal
    alpha_base = cfg["alpha_start_ang"]

    # Generate kpush grid: kbase, kbase/kinc, kbase/kinc^2, ...
    kpush_values = [kpush_base / (cfg["kinc"] ** j) for j in range(cfg["nk"])]

    # Generate alpha grid: abase, abase/alpinc, abase/alpinc^2, ...
    alpha_values = [alpha_base / (cfg["alpinc"] ** i) for i in range(cfg["na"])]

    # Build step list (alpha outer loop, kpush inner loop).
    steps = []
    for alpha in alpha_values:
        for kpush in kpush_values:
            steps.append(
                {
                    "kpush": kpush,
                    "alpha": alpha,
                    "cvdump_fs": cfg["cvdump_fs"],
                }
            )

    # Compute time per step from molecular size and flexibility.
    # tmtd = 3.0 * exp(0.10 * flex * max(1, rednat - 8))
    # Clamped to [5.0, 500.0] ps. Quick mode applies an extra rfac = 0.5.
    rednat = max(1, n_atoms - 8)
    av1 = flexibility * rednat
    tmtd = 3.0 * math.exp(0.10 * av1)
    tmtd = max(5.0, min(500.0, tmtd))
    time_per_step = tmtd * cfg["rfac"]

    # Refinement parameters (moderate bias for conformer refinement phase)
    mid_kpush = kpush_values[len(kpush_values) // 2] * 0.75
    refinement_params = {
        "kpush": mid_kpush,
        "alpha": alpha_values[0],
        "cvdump_fs": cfg["cvdump_fs"],
    }
    # Rotamer-phase runtime is half the MTD runtime.
    refinement_time = time_per_step * 0.5

    # Adaptive RMSD threshold for diversity selection
    min_rmsd_threshold = 0.5 + 0.5 * flexibility + 0.05 * math.log(max(1, n_atoms - 8))
    min_rmsd_threshold = max(0.5, min(2.0, min_rmsd_threshold))

    return {
        "mtd_steps": steps,
        "time_per_step_ps": time_per_step,
        "n_conformer_starts": cfg["n_conformer_starts"],
        "refinement_time_ps": refinement_time,
        "refinement_params": refinement_params,
        "min_rmsd_threshold": min_rmsd_threshold,
    }
