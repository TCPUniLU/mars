"""Tests for mars/topology.py — fragment detection and structure splitting."""

import numpy as np
import pytest

from mars.topology import (
    MoleculeTopology,
    analyse_topology,
    is_multimolecular,
    count_molecules,
    get_fragments,
    split_structure,
)
from mars.utils import create_structure


@pytest.fixture
def two_methanes():
    """Two methane molecules separated by 6 Å — clearly two fragments."""
    methane1 = np.array(
        [
            [0.000, 0.000, 0.000],  # C
            [0.629, 0.629, 0.629],  # H
            [0.629, -0.629, -0.629],
            [-0.629, 0.629, -0.629],
            [-0.629, -0.629, 0.629],
        ]
    )
    methane2 = methane1 + np.array([6.0, 0.0, 0.0])
    positions = np.concatenate([methane1, methane2])
    numbers = np.array([6, 1, 1, 1, 1, 6, 1, 1, 1, 1])
    return positions, numbers


def test_analyse_topology_single_molecule():
    # Methane
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.629, 0.629, 0.629],
            [0.629, -0.629, -0.629],
            [-0.629, 0.629, -0.629],
            [-0.629, -0.629, 0.629],
        ]
    )
    numbers = np.array([6, 1, 1, 1, 1])
    topo = analyse_topology(positions, numbers)
    assert isinstance(topo, MoleculeTopology)
    assert topo.n_molecules == 1
    assert topo.is_multimolecular is False
    assert topo.fragment_sizes == [5]
    assert sorted(topo.fragments[0]) == [0, 1, 2, 3, 4]


def test_analyse_topology_two_methanes(two_methanes):
    pos, num = two_methanes
    topo = analyse_topology(pos, num)
    assert topo.n_molecules == 2
    assert topo.is_multimolecular is True
    assert sorted(topo.fragment_sizes) == [5, 5]


def test_is_multimolecular(two_methanes):
    pos, num = two_methanes
    assert is_multimolecular(pos, num) is True
    # Single molecule case
    assert is_multimolecular(pos[:5], num[:5]) is False


def test_count_molecules(two_methanes):
    pos, num = two_methanes
    assert count_molecules(pos, num) == 2


def test_get_fragments(two_methanes):
    pos, num = two_methanes
    frags = get_fragments(pos, num)
    assert len(frags) == 2
    assert all(isinstance(f, list) for f in frags)
    # All atom indices accounted for exactly once
    flat = sorted(idx for frag in frags for idx in frag)
    assert flat == list(range(len(pos)))


def test_split_structure_two_fragments(two_methanes):
    pos, num = two_methanes
    structure = create_structure(pos, ["C", "H", "H", "H", "H", "C", "H", "H", "H", "H"])
    parts = split_structure(structure)
    assert len(parts) == 2
    for p in parts:
        assert "positions" in p and "symbols" in p and "numbers" in p
        assert len(p["symbols"]) == 5


def test_split_structure_single_molecule_returns_one():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.629, 0.629, 0.629],
            [0.629, -0.629, -0.629],
            [-0.629, 0.629, -0.629],
            [-0.629, -0.629, 0.629],
        ]
    )
    structure = create_structure(positions, ["C", "H", "H", "H", "H"])
    parts = split_structure(structure)
    assert len(parts) == 1
    assert len(parts[0]["symbols"]) == 5


def test_isolated_atom_is_its_own_fragment():
    # Methane + a far-away argon atom
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.629, 0.629, 0.629],
            [0.629, -0.629, -0.629],
            [-0.629, 0.629, -0.629],
            [-0.629, -0.629, 0.629],
            [10.0, 10.0, 10.0],
        ]
    )
    numbers = np.array([6, 1, 1, 1, 1, 18])
    topo = analyse_topology(positions, numbers)
    assert topo.n_molecules == 2
    sizes = sorted(topo.fragment_sizes)
    assert sizes == [1, 5]


def test_fragment_atomic_numbers(two_methanes):
    pos, num = two_methanes
    topo = analyse_topology(pos, num)
    parts = topo.fragment_atomic_numbers(num)
    assert len(parts) == 2
    for p in parts:
        assert sorted(p.tolist()) == [1, 1, 1, 1, 6]
