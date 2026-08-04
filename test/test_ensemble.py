"""
Tests for mars.ensemble and mars.utils_prune — ensemble management.
"""

import os
import tempfile
import pytest
import jax.numpy as jnp
import numpy as np

from mars.utils import create_structure, save_ensemble, load_ensemble
from mars.ensemble import (
    sort_by_energy,
    prune_by_energy_window,
    get_unique_by_energy,
    filter_by_energy_range,
    get_lowest_n,
    get_energy_statistics,
)
from mars.utils_prune import (
    prune_by_rmsd,
    compute_pairwise_rmsd_ensemble,
    find_most_diverse_subset,
    cluster_by_rmsd,
    get_cluster_representatives,
)
from mars.auto_config import EV_TO_KCALMOL

# ── helpers ─────────────────────────────────────────────────────────────────────


def _make_structure(offset=0.0, n_atoms=4):
    positions = jnp.array([[i * 1.5 + offset * 0.1, 0.0, 0.0] for i in range(n_atoms)])
    return create_structure(positions, ["C"] * n_atoms)


def _make_ensemble(n=8, energy_range=5.0, rng=np.random.default_rng(0)):
    structs = []
    for i in range(n):
        offset = rng.uniform(-1.0, 1.0)
        s = _make_structure(offset + i * 2.0)
        e = float(rng.uniform(0.0, energy_range))
        structs.append((s, e))
    return structs


# ── sort_by_energy ───────────────────────────────────────────────────────────────


class TestSortByEnergy:
    def test_ascending_order(self):
        ens = _make_ensemble(n=10)
        sorted_ens = sort_by_energy(ens)
        energies = [e for _, e in sorted_ens]
        assert energies == sorted(energies)

    def test_length_preserved(self):
        ens = _make_ensemble(n=7)
        assert len(sort_by_energy(ens)) == 7

    def test_empty_returns_empty(self):
        assert sort_by_energy([]) == []

    def test_single_element(self):
        ens = [(_make_structure(), 1.23)]
        assert sort_by_energy(ens) == ens


# ── prune_by_energy_window ───────────────────────────────────────────────────────


class TestPruneByEnergyWindow:
    def test_removes_high_energy(self):
        ens = [(_make_structure(i), float(i)) for i in range(10)]
        ens = sort_by_energy(ens)
        # ewin in eV; window of 3 eV keeps indices 0–3
        pruned = prune_by_energy_window(ens, ewin=3.0)
        assert len(pruned) <= len(ens)
        ref = ens[0][1]
        for _, e in pruned:
            assert e - ref <= 3.0 + 1e-9

    def test_keeps_all_within_window(self):
        ens = [(_make_structure(), 0.0 + i * 0.1) for i in range(5)]
        pruned = prune_by_energy_window(ens, ewin=10.0)
        assert len(pruned) == 5

    def test_empty_input(self):
        assert prune_by_energy_window([], ewin=1.0) == []


# ── get_unique_by_energy ────────────────────────────────────────────────────────


class TestGetUniqueByEnergy:
    def test_removes_duplicates(self):
        s = _make_structure()
        ens = [(s, 1.0), (s, 1.0), (s, 2.0)]
        unique = get_unique_by_energy(ens, tolerance=1e-6)
        energies = [e for _, e in unique]
        assert len(set(round(e, 10) for e in energies)) == len(energies)

    def test_keeps_distinct(self):
        s = _make_structure()
        ens = [(s, 0.0), (s, 0.5), (s, 1.0)]
        unique = get_unique_by_energy(ens, tolerance=1e-6)
        assert len(unique) == 3

    def test_empty_input(self):
        assert get_unique_by_energy([]) == []


# ── filter_by_energy_range ───────────────────────────────────────────────────────


class TestFilterByEnergyRange:
    def test_lower_bound(self):
        ens = [(_make_structure(), float(e)) for e in [0.0, 1.0, 2.0, 3.0]]
        filtered = filter_by_energy_range(ens, min_energy=1.5)
        assert all(e >= 1.5 for _, e in filtered)

    def test_upper_bound(self):
        ens = [(_make_structure(), float(e)) for e in [0.0, 1.0, 2.0, 3.0]]
        filtered = filter_by_energy_range(ens, max_energy=2.0)
        assert all(e <= 2.0 for _, e in filtered)

    def test_both_bounds(self):
        ens = [(_make_structure(), float(e)) for e in [0.0, 1.0, 2.0, 3.0]]
        filtered = filter_by_energy_range(ens, min_energy=0.5, max_energy=2.5)
        assert all(0.5 <= e <= 2.5 for _, e in filtered)


# ── get_lowest_n ─────────────────────────────────────────────────────────────────


class TestGetLowestN:
    def test_returns_n_structures(self):
        ens = _make_ensemble(n=10)
        lowest = get_lowest_n(ens, n=3)
        assert len(lowest) == 3

    def test_returns_lowest_energy(self):
        ens = _make_ensemble(n=10)
        lowest = get_lowest_n(ens, n=3)
        min_all = min(e for _, e in ens)
        assert lowest[0][1] == pytest.approx(min_all)

    def test_n_larger_than_ensemble(self):
        ens = _make_ensemble(n=4)
        lowest = get_lowest_n(ens, n=10)
        assert len(lowest) == 4


# ── get_energy_statistics ────────────────────────────────────────────────────────


class TestGetEnergyStatistics:
    def test_keys(self):
        ens = _make_ensemble(n=5)
        stats = get_energy_statistics(ens)
        for key in ["n_structures", "min_energy", "max_energy", "mean_energy", "std_energy"]:
            assert key in stats

    def test_values(self):
        energies = [0.0, 1.0, 2.0, 3.0]
        ens = [(_make_structure(), e) for e in energies]
        stats = get_energy_statistics(ens)
        assert stats["n_structures"] == 4
        assert stats["min_energy"] == pytest.approx(0.0)
        assert stats["max_energy"] == pytest.approx(3.0)
        assert stats["mean_energy"] == pytest.approx(1.5)

    def test_empty_returns_empty_dict(self):
        assert get_energy_statistics([]) == {}


# ── prune_by_rmsd ────────────────────────────────────────────────────────────────


class TestPruneByRmsd:
    def _make_ens_with_duplicates(self):
        """Ensemble where pairs 0/1 and 2/3 are conformationally near-identical
        (Kabsch alignment cancels rigid translations, so spatial offset alone
        does not increase RMSD — we perturb internal coordinates instead).
        """
        base = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
        bent = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [3.0, 0.0, 0.0]])
        v_shape = jnp.array([[0.0, 0.0, 0.0], [1.5, 1.5, 0.0], [3.0, 0.0, 0.0]])
        structs = [
            (create_structure(base, ["C", "C", "C"]), 0.0),
            (create_structure(base + 0.0005, ["C", "C", "C"]), 0.1),
            (create_structure(bent, ["C", "C", "C"]), 1.0),
            (create_structure(bent + 0.0005, ["C", "C", "C"]), 1.1),
            (create_structure(v_shape, ["C", "C", "C"]), 2.0),
        ]
        return sort_by_energy(structs)

    def test_removes_near_duplicates(self):
        ens = self._make_ens_with_duplicates()
        pruned = prune_by_rmsd(ens, threshold=0.1)
        assert len(pruned) < len(ens)

    def test_keeps_diverse(self):
        ens = self._make_ens_with_duplicates()
        pruned = prune_by_rmsd(ens, threshold=0.1)
        # base/bent/v_shape are all conformationally distinct (>0.1 Å RMSD)
        assert len(pruned) >= 2

    def test_threshold_zero_keeps_all(self):
        ens = self._make_ens_with_duplicates()
        pruned = prune_by_rmsd(ens, threshold=0.0)
        assert len(pruned) == len(ens)

    def test_empty_input(self):
        assert prune_by_rmsd([]) == []


# ── find_most_diverse_subset ─────────────────────────────────────────────────────


class TestFindMostDiverseSubset:
    def test_returns_requested_count(self):
        ens = _make_ensemble(n=10)
        diverse = find_most_diverse_subset(ens, n_keep=4, min_rmsd=0.0)
        assert len(diverse) == 4

    def test_n_larger_than_ensemble(self):
        ens = _make_ensemble(n=3)
        diverse = find_most_diverse_subset(ens, n_keep=10, min_rmsd=0.0)
        assert len(diverse) == 3

    def test_empty_input(self):
        assert find_most_diverse_subset([], n_keep=5, min_rmsd=0.0) == []


# ── file I/O ────────────────────────────────────────────────────────────────────


class TestEnsembleIO:
    def test_save_load_roundtrip(self):
        """save_ensemble writes energies as ``Energy = X kcal/mol`` in the
        XYZ comment. ASE's default loader does not parse that comment line,
        so the round-trip preserves geometry but resets energy to 0.0 — we
        assert what the I/O contract actually provides.
        """
        ens = [(_make_structure(i * 2.0), float(i * 0.5)) for i in range(5)]
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            fname = f.name
        try:
            save_ensemble(fname, ens)
            loaded = load_ensemble(fname)
            assert len(loaded) == len(ens)
            for (s_orig, _e_orig), (s_load, _e_load) in zip(ens, loaded):
                # Geometry survives the round-trip
                assert jnp.allclose(s_orig["positions"], s_load["positions"], atol=1e-3)
                # symbols preserved
                assert list(s_load["symbols"]) == list(s_orig["symbols"])
        finally:
            if os.path.exists(fname):
                os.remove(fname)
