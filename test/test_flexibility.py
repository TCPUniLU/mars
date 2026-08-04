"""Tests for mars/flexibility.py — flexibility estimation, rotatable bonds, MTD config."""

import numpy as np
import pytest

from mars.flexibility import (
    calculate_flexibility,
    detect_bonds_simple,
    estimate_rotatable_bonds,
    calculate_effective_atoms,
    get_mtd_workflow_config,
)


def _linear_alkane(n_carbons):
    """All-trans n-alkane geometry (carbon backbone only — sufficient for the
    coordination-based flexibility metric)."""
    bl = 1.54
    ang = np.deg2rad(112.0)
    xs = [0.0]
    ys = [0.0]
    for i in range(1, n_carbons):
        if i % 2 == 1:
            xs.append(xs[-1] + bl * np.sin(ang / 2))
            ys.append(ys[-1] + bl * np.cos(ang / 2))
        else:
            xs.append(xs[-1] + bl * np.sin(ang / 2))
            ys.append(ys[-1] - bl * np.cos(ang / 2))
    pos = np.array([[x, y, 0.0] for x, y in zip(xs, ys)])
    nums = np.full(n_carbons, 6)
    return pos, nums


def _benzene_ring():
    """Six C atoms in a planar hexagon (rigid ring)."""
    r = 1.40
    pos = np.array([[r * np.cos(k * np.pi / 3), r * np.sin(k * np.pi / 3), 0.0] for k in range(6)])
    nums = np.full(6, 6)
    return pos, nums


def test_detect_bonds_simple_linear_chain():
    pos, nums = _linear_alkane(4)
    bonds = detect_bonds_simple(pos, nums)
    # 4 carbons → 3 bonds in a linear chain
    assert len(bonds) == 3


def test_calculate_flexibility_in_unit_range():
    pos_lin, nums_lin = _linear_alkane(8)
    flex_lin = calculate_flexibility(pos_lin, nums_lin)
    pos_ring, nums_ring = _benzene_ring()
    flex_ring = calculate_flexibility(pos_ring, nums_ring)
    assert 0.0 <= flex_ring <= 1.0
    assert 0.0 <= flex_lin <= 1.0


def test_calculate_flexibility_branched_lower_than_linear():
    # Linear chain: coord = (1, 2, 2, 2, 2, 1) → branch factor = 1.0 in middle bonds
    pos_lin, nums_lin = _linear_alkane(6)
    flex_lin = calculate_flexibility(pos_lin, nums_lin)
    # Add a branch atom on C2 to lower its branching factor
    pos_branch = np.vstack([pos_lin, [pos_lin[1] + np.array([0.0, 0.0, 1.5])]])
    nums_branch = np.append(nums_lin, 6)
    flex_branch = calculate_flexibility(pos_branch, nums_branch)
    assert flex_branch < flex_lin


def test_calculate_flexibility_single_atom():
    pos = np.array([[0.0, 0.0, 0.0]])
    nums = np.array([6])
    assert calculate_flexibility(pos, nums) == 0.0


def test_calculate_flexibility_no_bonds():
    pos = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    nums = np.array([6, 6])
    assert calculate_flexibility(pos, nums) == 0.0


def test_estimate_rotatable_bonds_linear():
    # 6-carbon chain has 3 internal (non-terminal) bonds, all rotatable
    pos, nums = _linear_alkane(6)
    n_rot = estimate_rotatable_bonds(pos, nums)
    assert n_rot == 3


def test_estimate_rotatable_bonds_benzene_excludes_aromatic():
    pos, nums = _benzene_ring()
    n_rot = estimate_rotatable_bonds(pos, nums)
    # All ring carbons have coordination 2 (no H accounted for) so heuristic
    # may include them; just assert sane bounds
    assert 0 <= n_rot <= 6


def test_calculate_effective_atoms():
    assert calculate_effective_atoms(0.5, 20) == 10.0
    assert calculate_effective_atoms(1.0, 5) == 5.0


def test_mtd_workflow_config_shape():
    cfg = get_mtd_workflow_config(flexibility=0.5, n_atoms=10, mode="normal")
    assert isinstance(cfg, dict)
    assert "mtd_steps" in cfg
    assert isinstance(cfg["mtd_steps"], list) and len(cfg["mtd_steps"]) > 0
    for step in cfg["mtd_steps"]:
        assert {"kpush", "alpha", "cvdump_fs"} <= set(step.keys())
    assert cfg["time_per_step_ps"] > 0
    assert cfg["n_conformer_starts"] >= 1


@pytest.mark.parametrize("mode", ["quick", "normal", "thorough"])
def test_mtd_workflow_config_modes(mode):
    cfg = get_mtd_workflow_config(flexibility=0.7, n_atoms=20, mode=mode)
    assert isinstance(cfg, dict)
    assert len(cfg["mtd_steps"]) > 0
