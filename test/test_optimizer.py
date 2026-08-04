"""
Tests for mars.optimizer — single-structure and batch geometry optimization.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from mars.optimizer import optimize_single, FMAX_MAP
from mars.utils import create_structure

# ── shared helpers ──────────────────────────────────────────────────────────────


def harmonic(positions):
    center = jnp.mean(positions, axis=0)
    return jnp.sum((positions - center) ** 2) * 0.1


def quadratic_min_at_origin(positions):
    """Global minimum at all-zeros."""
    return jnp.sum(positions**2) * 0.5


def _triangle(scale=1.0):
    return jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.75, 1.3, 0.0]]) * scale


# ── FMAX_MAP ────────────────────────────────────────────────────────────────────


class TestFmaxMap:
    def test_all_levels_present(self):
        for lvl in [-3, -2, -1, 0, 1, 2]:
            assert lvl in FMAX_MAP

    def test_tighter_for_higher_level(self):
        assert FMAX_MAP[-3] > FMAX_MAP[0] > FMAX_MAP[2]


# ── optimize_single ─────────────────────────────────────────────────────────────


class TestOptimizeSingle:
    def test_output_shapes(self):
        pos = _triangle()
        pos_opt, energy, info = optimize_single(pos, harmonic, fmax=0.05, maxiter=100)
        assert pos_opt.shape == pos.shape
        assert isinstance(float(energy), float)
        assert isinstance(info, dict)

    def test_energy_decreases(self):
        pos = _triangle() + jax.random.normal(jax.random.PRNGKey(0), (3, 3)) * 0.3
        E_init = float(harmonic(pos))
        pos_opt, E_opt, info = optimize_single(pos, harmonic, fmax=0.01, maxiter=500)
        assert E_opt <= E_init + 1e-8

    def test_info_keys_present(self):
        pos = _triangle()
        _, _, info = optimize_single(pos, harmonic, fmax=0.1, maxiter=50)
        for key in ["converged", "iterations", "final_energy", "max_force"]:
            assert key in info, f"Missing key: {key}"

    def test_converges_on_simple_potential(self):
        pos = jnp.array([[1.0, 0.5, 0.2], [-0.5, 0.8, -0.3]], dtype=jnp.float64)
        pos_opt, E_opt, info = optimize_single(pos, quadratic_min_at_origin, fmax=1e-4, maxiter=500)
        # minimum is at origin
        assert float(E_opt) < 0.01

    def test_gradient_descent_method(self):
        pos = _triangle()
        pos_opt, E_opt, info = optimize_single(pos, harmonic, fmax=0.05, maxiter=200, method="GD")
        assert pos_opt.shape == pos.shape
        assert E_opt <= float(harmonic(pos)) + 1e-8

    def test_barrier_fn_does_not_affect_reported_energy(self):
        """Barrier modifies the optimisation path, but returned energy is E only."""
        pos = _triangle() + 0.5

        def barrier(p):
            return jnp.sum(p**2) * 1000.0

        pos_opt, E_opt, _ = optimize_single(pos, harmonic, fmax=0.1, maxiter=50, barrier_fn=barrier)
        # reported energy should be from harmonic only (without barrier)
        E_direct = float(harmonic(pos_opt))
        assert float(E_opt) == pytest.approx(E_direct, rel=1e-4)

    def test_lbfgs_reaches_lower_energy_than_gd(self):
        """LBFGS typically converges faster than GD for smooth potentials.

        Use enough iterations so LBFGS converges while GD still has meaningful
        energy left; compare with a generous tolerance.
        """
        pos = _triangle() * 3.0 + 2.0  # Start far from minimum
        _, E_lbfgs, _ = optimize_single(pos, harmonic, fmax=1e-4, maxiter=200)
        _, E_gd, _ = optimize_single(
            pos, harmonic, fmax=1e-4, maxiter=20, method="GD"
        )  # GD with fewer iters
        # LBFGS with 200 iters should beat GD with only 20
        assert E_lbfgs <= E_gd + 1e-4

    def test_max_force_in_info(self):
        pos = _triangle() + 1.0
        _, _, info = optimize_single(pos, harmonic, fmax=0.01, maxiter=200)
        assert info["max_force"] >= 0.0

    def test_gradient_zero_at_minimum(self):
        """At the minimiser the gradient should be very small."""
        pos = jnp.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]], dtype=jnp.float64)
        pos_opt, _, info = optimize_single(pos, quadratic_min_at_origin, fmax=1e-5, maxiter=1000)
        grad = jax.grad(quadratic_min_at_origin)(pos_opt)
        assert float(jnp.max(jnp.abs(grad))) < 0.01
