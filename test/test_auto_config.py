"""
Tests for mars.auto_config — automatic MTD parameter configuration.
"""

import pytest
import numpy as np
import jax.numpy as jnp

from mars.auto_config import (
    estimate_flexibility,
    compute_md_length,
    generate_mtd_parameters,
    EV_TO_KCALMOL,
    KCALMOL_TO_EV,
)

# ── unit constants ───────────────────────────────────────────────────────────────


class TestUnitConstants:
    def test_ev_to_kcal_approx(self):
        assert EV_TO_KCALMOL == pytest.approx(23.06, rel=0.01)

    def test_inverse_relation(self):
        assert EV_TO_KCALMOL * KCALMOL_TO_EV == pytest.approx(1.0, rel=1e-8)


# ── estimate_flexibility ─────────────────────────────────────────────────────────


class TestEstimateFlexibility:
    def test_returns_float(self):
        flex = estimate_flexibility(10)
        assert isinstance(flex, float)

    def test_bounded_01(self):
        for n in [3, 5, 10, 20, 50]:
            flex = estimate_flexibility(n)
            assert 0.0 <= flex <= 1.0, f"Flexibility out of bounds for n={n}: {flex}"

    def test_small_molecule_moderate(self):
        flex = estimate_flexibility(5)
        assert 0.0 < flex < 1.0

    def test_with_atomic_numbers(self):
        numbers = np.array([6, 6, 6, 1, 1, 1, 1, 1, 1, 1])  # propane-ish
        flex = estimate_flexibility(len(numbers), atomic_numbers=numbers)
        assert 0.0 <= flex <= 1.0


# ── compute_md_length ───────────────────────────────────────────────────────────


class TestComputeMdLength:
    def test_returns_positive(self):
        for mode in ["quick", "normal", "thorough"]:
            t = compute_md_length(10, mode=mode)
            assert t > 0.0, f"Non-positive time for mode={mode}"

    def test_thorough_ge_normal_ge_quick(self):
        t_q = compute_md_length(15, mode="quick")
        t_n = compute_md_length(15, mode="normal")
        t_t = compute_md_length(15, mode="thorough")
        assert t_q <= t_n <= t_t

    def test_larger_molecule_longer_time(self):
        t_small = compute_md_length(5)
        t_large = compute_md_length(50)
        assert t_large >= t_small

    def test_conformer_phase_shorter(self):
        t_expl = compute_md_length(20, phase="exploration")
        t_conf = compute_md_length(20, phase="conformer")
        assert t_conf <= t_expl

    def test_minimum_is_positive(self):
        # Even for tiny molecule, time should be at least 5 ps
        t = compute_md_length(3, mode="quick")
        assert t >= 1.0


# ── generate_mtd_parameters ──────────────────────────────────────────────────────


class TestGenerateMtdParameters:
    def test_returns_dict(self):
        config = generate_mtd_parameters(n_atoms=10)
        assert isinstance(config, dict)

    def test_top_level_keys(self):
        config = generate_mtd_parameters(n_atoms=15)
        for key in ["mtd_steps", "refinement_params", "min_rmsd_threshold", "n_conformer_starts"]:
            assert key in config, f"Missing top-level key '{key}'"

    def test_mtd_steps_is_nonempty_list(self):
        config = generate_mtd_parameters(n_atoms=10)
        assert isinstance(config["mtd_steps"], list)
        assert len(config["mtd_steps"]) > 0

    def test_mtd_step_required_keys(self):
        config = generate_mtd_parameters(n_atoms=15)
        for step in config["mtd_steps"]:
            for key in ["kpush", "alpha", "cvdump_fs"]:
                assert key in step, f"Missing key '{key}' in mtd_step"

    def test_mtd_step_values_positive(self):
        config = generate_mtd_parameters(n_atoms=20)
        for step in config["mtd_steps"]:
            assert step["kpush"] > 0.0
            assert step["alpha"] > 0.0
            assert step["cvdump_fs"] > 0.0

    def test_refinement_params_keys(self):
        config = generate_mtd_parameters(n_atoms=10)
        ref = config["refinement_params"]
        assert isinstance(ref, dict)
        for key in ["kpush", "alpha", "cvdump_fs"]:
            assert key in ref, f"Missing key '{key}' in refinement_params"

    def test_min_rmsd_threshold_positive(self):
        config = generate_mtd_parameters(n_atoms=10)
        assert float(config["min_rmsd_threshold"]) > 0.0

    def test_mtd_kscal_scales_kpush(self):
        base = generate_mtd_parameters(n_atoms=10, mtd_kscal=1.0)
        scaled = generate_mtd_parameters(n_atoms=10, mtd_kscal=2.0)
        for b, s in zip(base["mtd_steps"], scaled["mtd_steps"]):
            assert s["kpush"] == pytest.approx(b["kpush"] * 2.0, rel=1e-5)
