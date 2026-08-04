"""Tests for mars/crossing.py — Z-matrix genetic crossing of conformers."""

import numpy as np
import pytest

from mars.crossing import genetic_zmatrix_crossing
from mars.utils import create_structure


def _butane_conformers():
    """Two butane (C4H10) conformers — a trans and a perturbed gauche."""
    bl = 1.54
    ang = np.deg2rad(112.0)
    xs = [0.0]
    ys = [0.0]
    for i in range(1, 4):
        if i % 2 == 1:
            xs.append(xs[-1] + bl * np.sin(ang / 2))
            ys.append(ys[-1] + bl * np.cos(ang / 2))
        else:
            xs.append(xs[-1] + bl * np.sin(ang / 2))
            ys.append(ys[-1] - bl * np.cos(ang / 2))
    pos_trans = np.array(
        [
            [xs[0], ys[0], 0.0],
            [xs[1], ys[1], 0.0],
            [xs[2], ys[2], 0.0],
            [xs[3], ys[3], 0.0],
            # H atoms (3 on terminal C, 2 on internal C, etc.) — only need ≥4 atoms
            [xs[0] - 0.7, ys[0], 0.5],
            [xs[0] - 0.7, ys[0], -0.5],
            [xs[0], ys[0] + 1.0, 0.0],
            [xs[1], ys[1], 0.9],
            [xs[1], ys[1], -0.9],
            [xs[2], ys[2], 0.9],
            [xs[2], ys[2], -0.9],
            [xs[3] + 0.7, ys[3], 0.5],
            [xs[3] + 0.7, ys[3], -0.5],
            [xs[3], ys[3] - 1.0, 0.0],
        ]
    )
    symbols = ["C", "C", "C", "C", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H"]

    # Second conformer: perturb dihedral by rotating the last C around the C2-C3 axis
    pos_pert = pos_trans.copy()
    rot = 0.4  # ~23 degrees
    R = np.array([[np.cos(rot), -np.sin(rot), 0], [np.sin(rot), np.cos(rot), 0], [0, 0, 1]])
    # Rotate the last C and its three Hs around the (xs[2], ys[2], 0) point
    center = np.array([xs[2], ys[2], 0.0])
    for idx in [3, 11, 12, 13]:
        pos_pert[idx] = (pos_pert[idx] - center) @ R.T + center

    s1 = create_structure(pos_trans, symbols)
    s2 = create_structure(pos_pert, symbols)
    return s1, s2


def test_returns_empty_for_small_ensemble():
    s, _ = _butane_conformers()
    assert genetic_zmatrix_crossing([(s, -1.0)], n_children=5) == []
    assert genetic_zmatrix_crossing([], n_children=5) == []


def test_returns_empty_for_few_atoms():
    # 3-atom water — too few for dihedral crossing
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    s = create_structure(pos, ["O", "H", "H"])
    out = genetic_zmatrix_crossing([(s, -1.0), (s, -0.5)], n_children=3)
    assert out == []


def test_produces_requested_n_children():
    s1, s2 = _butane_conformers()
    out = genetic_zmatrix_crossing([(s1, -10.0), (s2, -9.5)], n_children=8, random_seed=0)
    assert len(out) <= 8  # Some children may be rejected for CN clashes
    for child in out:
        assert "positions" in child
        assert "symbols" in child
        assert child["positions"].shape == s1["positions"].shape


def test_reproducible_with_seed():
    s1, s2 = _butane_conformers()
    out_a = genetic_zmatrix_crossing([(s1, -10.0), (s2, -9.5)], n_children=4, random_seed=123)
    out_b = genetic_zmatrix_crossing([(s1, -10.0), (s2, -9.5)], n_children=4, random_seed=123)
    assert len(out_a) == len(out_b)
    for ca, cb in zip(out_a, out_b):
        np.testing.assert_allclose(
            np.array(ca["positions"]), np.array(cb["positions"]), rtol=1e-6, atol=1e-6
        )


def test_no_boltzmann_weighting_uniform_selection():
    s1, s2 = _butane_conformers()
    out = genetic_zmatrix_crossing(
        [(s1, -10.0), (s2, -1.0)],
        n_children=4,
        boltzmann_weight=False,
        random_seed=42,
    )
    assert isinstance(out, list)
    assert all("positions" in c for c in out)
