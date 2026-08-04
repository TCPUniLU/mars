"""Tests for mars/functional_groups.py — MolGraph and FG identification."""

import numpy as np
import pytest

from mars.functional_groups import (
    MolGraph,
    identify_functional_groups,
    get_functional_group_for_atoms,
)
from mars.ir import _build_topology


def _topology_from_xyz(symbols, positions, charge=0):
    """Build a topology dict the same way ir.py does for FG detection."""
    numbers = np.array([{"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "S": 16}[s] for s in symbols])
    return _build_topology(np.array(positions), numbers, symbols, charge=charge)


# ── MolGraph basics ──────────────────────────────────────────────────────────


def test_molgraph_neighbors_and_bond_orders():
    # Methane: C with 4 H
    symbols = ["C", "H", "H", "H", "H"]
    topo = {
        "neighbors": {0: {1, 2, 3, 4}, 1: {0}, 2: {0}, 3: {0}, 4: {0}},
        "bond_orders": {(0, 1): 1, (0, 2): 1, (0, 3): 1, (0, 4): 1},
    }
    g = MolGraph(symbols, topo)
    assert g.n_atoms == 5
    assert g.sym(0) == "C"
    assert g.h_count(0) == 4
    assert g.heavy_neighbors(0) == []
    assert g.bo(0, 1) == 1
    assert g.has_bond_to(0, "H", 1) is True


def test_molgraph_handles_unknown_index():
    g = MolGraph(["C"], {"neighbors": {0: set()}, "bond_orders": {}})
    assert g.sym(99) == "?"
    # Default bond order is 1 for unknown pairs
    assert g.bo(0, 99) == 1


def test_molgraph_ring_detection():
    # Cyclopropane: 3-ring of carbons
    symbols = ["C", "C", "C", "H", "H", "H", "H", "H", "H"]
    topo = {
        "neighbors": {
            0: {1, 2, 3, 4},
            1: {0, 2, 5, 6},
            2: {0, 1, 7, 8},
            3: {0},
            4: {0},
            5: {1},
            6: {1},
            7: {2},
            8: {2},
        },
        "bond_orders": {
            (0, 1): 1,
            (0, 2): 1,
            (1, 2): 1,
            (0, 3): 1,
            (0, 4): 1,
            (1, 5): 1,
            (1, 6): 1,
            (2, 7): 1,
            (2, 8): 1,
        },
    }
    g = MolGraph(symbols, topo)
    assert g.is_in_ring(0)
    assert g.is_in_3ring(0)


# ── identify_functional_groups end-to-end ───────────────────────────────────


def test_identify_carbonyl_in_acetone():
    # Acetone: CH3-C(=O)-CH3
    symbols = ["O", "C", "C", "C", "H", "H", "H", "H", "H", "H"]
    topo = {
        "neighbors": {
            0: {1},  # O
            1: {0, 2, 3},  # carbonyl C
            2: {1, 4, 5, 6},  # methyl C
            3: {1, 7, 8, 9},  # methyl C
            4: {2},
            5: {2},
            6: {2},
            7: {3},
            8: {3},
            9: {3},
        },
        "bond_orders": {
            (0, 1): 2,
            (1, 2): 1,
            (1, 3): 1,
            (2, 4): 1,
            (2, 5): 1,
            (2, 6): 1,
            (3, 7): 1,
            (3, 8): 1,
            (3, 9): 1,
        },
    }
    fg_map = identify_functional_groups(symbols, topo)
    assert "ketone" in fg_map
    inst = fg_map["ketone"][0]
    # The carbonyl atoms (C=O) must be in the instance
    assert 0 in inst and 1 in inst


def test_identify_alcohol_in_methanol():
    # Methanol: CH3-OH
    symbols = ["O", "C", "H", "H", "H", "H"]
    topo = {
        "neighbors": {0: {1, 5}, 1: {0, 2, 3, 4}, 2: {1}, 3: {1}, 4: {1}, 5: {0}},
        "bond_orders": {(0, 1): 1, (0, 5): 1, (1, 2): 1, (1, 3): 1, (1, 4): 1},
    }
    fg_map = identify_functional_groups(symbols, topo)
    # Should find an alcohol-like / O-H context group
    has_alcohol_like = any("alcohol" in k or "OH" in k or "hydroxyl" in k for k in fg_map.keys())
    assert has_alcohol_like or "primary_alcohol" in fg_map or any("oh" in k.lower() for k in fg_map)


def test_identify_amine_in_methylamine():
    # CH3-NH2
    symbols = ["N", "C", "H", "H", "H", "H", "H"]
    topo = {
        "neighbors": {
            0: {1, 5, 6},
            1: {0, 2, 3, 4},
            2: {1},
            3: {1},
            4: {1},
            5: {0},
            6: {0},
        },
        "bond_orders": {(0, 1): 1, (0, 5): 1, (0, 6): 1, (1, 2): 1, (1, 3): 1, (1, 4): 1},
    }
    fg_map = identify_functional_groups(symbols, topo)
    # Some flavor of amine should be detected
    assert any("amine" in k.lower() or "nh" in k.lower() for k in fg_map.keys())


def test_identify_nitrile_in_acetonitrile():
    # CH3-C#N
    symbols = ["C", "N", "C", "H", "H", "H"]
    topo = {
        "neighbors": {0: {1, 2}, 1: {0}, 2: {0, 3, 4, 5}, 3: {2}, 4: {2}, 5: {2}},
        "bond_orders": {(0, 1): 3, (0, 2): 1, (2, 3): 1, (2, 4): 1, (2, 5): 1},
    }
    fg_map = identify_functional_groups(symbols, topo)
    assert "nitrile" in fg_map


# ── get_functional_group_for_atoms ──────────────────────────────────────────


def test_get_functional_group_for_atoms_returns_label():
    symbols = ["O", "C", "C", "C", "H", "H", "H", "H", "H", "H"]
    topo = {
        "neighbors": {
            0: {1},
            1: {0, 2, 3},
            2: {1, 4, 5, 6},
            3: {1, 7, 8, 9},
            4: {2},
            5: {2},
            6: {2},
            7: {3},
            8: {3},
            9: {3},
        },
        "bond_orders": {
            (0, 1): 2,
            (1, 2): 1,
            (1, 3): 1,
            (2, 4): 1,
            (2, 5): 1,
            (2, 6): 1,
            (3, 7): 1,
            (3, 8): 1,
            (3, 9): 1,
        },
    }
    fg_map = identify_functional_groups(symbols, topo)
    label = get_functional_group_for_atoms(fg_map, [0, 1], symbols=symbols, topology=topo)
    assert label is not None
    # The bond prefix is derived from atom_indices order (O=C or C=O); both are valid
    assert "=" in label and "ketone" in label


def test_get_functional_group_for_atoms_empty():
    assert get_functional_group_for_atoms({}, [0, 1]) is None
    assert get_functional_group_for_atoms({"ketone": [(0, 1)]}, []) is None


# ── End-to-end via _build_topology ──────────────────────────────────────────


def test_methanol_via_build_topology():
    # Use real geometry through _build_topology so the topology dict is built
    # the same way it is in production IR plotting.
    symbols = ["O", "C", "H", "H", "H", "H"]
    positions = [
        [-1.64319, 1.95355, -0.04369],
        [-1.85927, 0.55544, 0.01022],
        [-1.85858, 2.31045, 0.83476],
        [-1.62543, 0.12209, -0.96508],
        [-1.20770, 0.11406, 0.76832],
        [-2.90590, 0.35666, 0.25337],
    ]
    topo = _topology_from_xyz(symbols, positions)
    fg_map = identify_functional_groups(symbols, topo)
    # Must produce *some* non-empty mapping
    assert isinstance(fg_map, dict)
    assert len(fg_map) > 0
