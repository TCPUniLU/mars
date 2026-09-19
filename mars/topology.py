"""
Molecular topology analysis for MARS.

Provides tools to detect separate molecules (connected components) within
a system — distinguishing single-molecule from multi-molecule (e.g. solute +
solvent, non-covalent complexes, ion pairs) setups.

The bond graph is built with the same covalent-radii criterion used throughout
MARS (see ``utils.detect_bonds``).  Connected components in that graph
correspond to distinct molecular fragments.
"""

from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from .utils import detect_bonds

# ============================================================================
# Data structures
# ============================================================================


class MoleculeTopology(NamedTuple):
    """Topology of a (possibly multi-molecule) system.

    Attributes:
        n_molecules:    Number of distinct molecular fragments detected.
        fragments:      List of atom-index lists, one per fragment, ordered by
                        first atom index (ascending).
        n_atoms:        Total number of atoms in the system.
        bonds:          Full list of covalent bonds as (i, j) pairs (i < j).
        bond_distances: Reference bond lengths (Å) parallel to *bonds*.
        cell:           (3, 3) lattice matrix (rows = lattice vectors) used
                        when the system is periodic, or ``None`` for
                        non-periodic systems.
        is_periodic:    True when a unit cell was supplied.
    """

    n_molecules: int
    fragments: List[List[int]]
    n_atoms: int
    bonds: List[Tuple[int, int]]
    bond_distances: np.ndarray
    cell: Optional[np.ndarray]
    is_periodic: bool

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_multimolecular(self) -> bool:
        """True when the system contains more than one molecular fragment."""
        return self.n_molecules > 1

    @property
    def fragment_sizes(self) -> List[int]:
        """Number of atoms in each fragment."""
        return [len(f) for f in self.fragments]

    def fragment_atomic_numbers(self, atomic_numbers) -> List[np.ndarray]:
        """Atomic numbers partitioned by fragment.

        Args:
            atomic_numbers: (n_atoms,) array of atomic numbers.

        Returns:
            List of 1-D arrays, one per fragment.
        """
        z = np.asarray(atomic_numbers)
        return [z[frag] for frag in self.fragments]


# ============================================================================
# Internal helpers
# ============================================================================


def _connected_components(n_atoms: int, bonds: List[Tuple[int, int]]) -> List[List[int]]:
    """Find connected components of the bond graph using union-find (DSU).

    Args:
        n_atoms: Number of atoms (nodes).
        bonds:   Iterable of (i, j) edges.

    Returns:
        List of components, each a sorted list of atom indices.
        Ordered by the smallest atom index in each component (ascending).
    """
    parent = list(range(n_atoms))
    rank = [0] * n_atoms

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(x: int, y: int):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        # Union by rank
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1

    for i, j in bonds:
        union(i, j)

    # Group atoms by their root
    from collections import defaultdict

    groups: Dict[int, List[int]] = defaultdict(list)
    for atom in range(n_atoms):
        groups[find(atom)].append(atom)

    # Sort internally and return ordered by first atom
    components = [sorted(g) for g in groups.values()]
    components.sort(key=lambda c: c[0])
    return components


# ============================================================================
# Public API
# ============================================================================


def build_bonded_lists(
    positions,
    atomic_numbers,
    tolerance: float = 1.3,
    bonds: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate bonds, angles and proper torsions from the covalent bond graph.

    Pure NumPy, no JAX — this module stays importable without JAX so that
    ``mars viewer`` works on a bare install.  Mirrors the enumeration in
    :func:`mars.ir._build_topology` (which additionally assigns bond orders and
    labels and therefore has to live in ``ir.py``), so the two agree by
    construction on bonds and angles.

    Args:
        positions: ``(n_atoms, 3)`` positions in Angstrom.
        atomic_numbers: ``(n_atoms,)`` atomic numbers.
        tolerance: Covalent-radius tolerance passed to
            :func:`mars.utils.detect_bonds` (default 1.3, the MARS-wide value).
        bonds: Optional precomputed bond list, to avoid re-detecting.

    Returns:
        Three integer arrays ``(bonds, angles, torsions)`` of shapes
        ``(n_b, 2)``, ``(n_a, 3)`` and ``(n_t, 4)``.  Angles are ``(i, j, k)``
        with *j* central; torsions are ``(i, j, k, l)`` about the bond
        ``(j, k)``.  Empty inputs give correctly-shaped empty arrays.

    Example:
        >>> b, a, t = build_bonded_lists(positions, numbers)
        >>> b.shape, a.shape, t.shape
        ((21, 2), (36, 3), (41, 4))
    """
    positions = np.asarray(positions, dtype=float)
    atomic_numbers = np.asarray(atomic_numbers, dtype=int)
    n_atoms = int(positions.shape[0])

    if bonds is None:
        bonds, _ = detect_bonds(positions, atomic_numbers, tolerance)
    bond_arr = np.asarray(bonds, dtype=int).reshape(-1, 2)

    neighbors: Dict[int, set] = {i: set() for i in range(n_atoms)}
    for i, j in bond_arr:
        neighbors[int(i)].add(int(j))
        neighbors[int(j)].add(int(i))

    angles: List[Tuple[int, int, int]] = []
    for j in range(n_atoms):
        nbrs = sorted(neighbors[j])
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                angles.append((nbrs[a], j, nbrs[b]))

    torsions: List[Tuple[int, int, int, int]] = []
    for j, k in bond_arr:
        j, k = int(j), int(k)
        for i in sorted(neighbors[j]):
            if i == k:
                continue
            for l in sorted(neighbors[k]):
                if l == j or l == i:
                    continue
                torsions.append((i, j, k, l))

    return (
        bond_arr,
        np.asarray(angles, dtype=int).reshape(-1, 3),
        np.asarray(torsions, dtype=int).reshape(-1, 4),
    )


def analyse_topology(
    positions, atomic_numbers, tolerance: float = 1.3, cell=None
) -> MoleculeTopology:
    """Detect all molecular fragments in a system.

    Uses covalent radii to identify covalent bonds and then finds connected
    components in the resulting bond graph.  Isolated atoms (no bonds) each
    form their own fragment (e.g. noble-gas atoms, bare ions).

    Args:
        positions:      (n_atoms, 3) array of atomic positions in Angstrom.
        atomic_numbers: (n_atoms,) array of atomic numbers.
        tolerance:      Multiplicative tolerance applied to the sum of covalent
                        radii when deciding whether two atoms are bonded.
                        Default 1.3 is the same as ``detect_bonds``.
        cell:           Optional (3, 3) lattice matrix (rows = lattice vectors)
                        in Angstrom.  When provided, bond detection uses the
                        minimum image convention so that bonds across periodic
                        boundaries are found correctly.  This is intended for
                        crystal / periodic systems (e.g. IR Hessian of a
                        molecular crystal).  Pass ``None`` (default) for
                        gas-phase / cluster calculations.

    Returns:
        MoleculeTopology named-tuple with connectivity information.

    Example:
        >>> topo = analyse_topology(positions, numbers)
        >>> if topo.is_multimolecular:
        ...     print(f"System has {topo.n_molecules} fragments")
        ... else:
        ...     print("Single-molecule system")

        >>> # Crystal / periodic system
        >>> topo = analyse_topology(positions, numbers, cell=lattice_matrix)
        >>> print(f"{topo.n_molecules} molecules per unit cell")
    """
    positions = np.asarray(positions)
    atomic_numbers = np.asarray(atomic_numbers, dtype=int)
    n_atoms = int(positions.shape[0])

    cell_np = np.asarray(cell) if cell is not None else None
    bonds, bond_distances = detect_bonds(positions, atomic_numbers, tolerance, cell=cell_np)
    bond_distances = np.asarray(bond_distances)

    fragments = _connected_components(n_atoms, bonds)

    return MoleculeTopology(
        n_molecules=len(fragments),
        fragments=fragments,
        n_atoms=n_atoms,
        bonds=bonds,
        bond_distances=bond_distances,
        cell=cell_np,
        is_periodic=cell_np is not None,
    )


def is_multimolecular(positions, atomic_numbers, tolerance: float = 1.3, cell=None) -> bool:
    """Return True if the system contains more than one molecular fragment.

    Convenience wrapper around ``analyse_topology`` for quick checks.

    Args:
        positions:      (n_atoms, 3) array of atomic positions in Angstrom.
        atomic_numbers: (n_atoms,) array of atomic numbers.
        tolerance:      Bond-detection tolerance (default 1.3).
        cell:           Optional (3, 3) lattice matrix for periodic systems.

    Returns:
        bool
    """
    topo = analyse_topology(positions, atomic_numbers, tolerance, cell=cell)
    return topo.is_multimolecular


def count_molecules(positions, atomic_numbers, tolerance: float = 1.3, cell=None) -> int:
    """Return the number of distinct molecular fragments in the system.

    Args:
        positions:      (n_atoms, 3) array of atomic positions in Angstrom.
        atomic_numbers: (n_atoms,) array of atomic numbers.
        tolerance:      Bond-detection tolerance (default 1.3).
        cell:           Optional (3, 3) lattice matrix for periodic systems.

    Returns:
        int
    """
    topo = analyse_topology(positions, atomic_numbers, tolerance, cell=cell)
    return topo.n_molecules


def get_fragments(positions, atomic_numbers, tolerance: float = 1.3, cell=None) -> List[List[int]]:
    """Return atom-index lists for each molecular fragment.

    Args:
        positions:      (n_atoms, 3) array of atomic positions in Angstrom.
        atomic_numbers: (n_atoms,) array of atomic numbers.
        tolerance:      Bond-detection tolerance (default 1.3).
        cell:           Optional (3, 3) lattice matrix for periodic systems.

    Returns:
        List of fragments, each a sorted list of atom indices.
    """
    topo = analyse_topology(positions, atomic_numbers, tolerance, cell=cell)
    return topo.fragments


def split_structure(structure: Dict, tolerance: float = 1.3) -> List[Dict]:
    """Split a multi-molecule structure dict into per-fragment structure dicts.

    Each returned dict has the same keys as the input (``'positions'``,
    ``'symbols'``, ``'numbers'``) but contains only the atoms belonging to
    one fragment.

    Args:
        structure: MARS structure dictionary.
        tolerance: Bond-detection tolerance (default 1.3).

    Returns:
        List of structure dicts, one per fragment (same ordering as
        ``get_fragments``).  For a single-molecule system this is a list of
        length 1 containing a copy of the original structure.
    """
    import numpy as np

    positions = np.asarray(structure["positions"])
    numbers = np.asarray(structure["numbers"], dtype=int)
    symbols = list(structure["symbols"])

    topo = analyse_topology(positions, numbers, tolerance)

    result = []
    for frag in topo.fragments:
        frag_idx = np.array(frag)
        result.append(
            {
                "positions": positions[frag_idx],
                "numbers": numbers[frag_idx],
                "symbols": [symbols[i] for i in frag],
            }
        )
    return result
