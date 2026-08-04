"""
Tests for mars.mtd — MTD state management, bias, and hill deposition.
"""

import os
import tempfile
import pytest
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from mars.mtd import (
    MTDState,
    create_mtd_state,
    add_hill,
    increment_step,
    should_deposit_hill,
    compute_mtd_bias,
    get_mtd_statistics,
    save_mtd_state,
    load_mtd_state,
)

# ── helpers ─────────────────────────────────────────────────────────────────────


def _triangle():
    return jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.75, 1.3, 0.0]])


# ── create_mtd_state ────────────────────────────────────────────────────────────


class TestCreateMtdState:
    def test_initial_hills_zero(self):
        state = create_mtd_state(n_atoms=5, max_hills=100)
        assert int(state.n_hills) == 0

    def test_array_shapes(self):
        state = create_mtd_state(n_atoms=7, max_hills=50)
        assert state.cv_xyz.shape == (50, 7, 3)
        assert state.reference_xyz.shape == (7, 3)

    def test_parameters_stored(self):
        state = create_mtd_state(n_atoms=5, max_hills=100, kpush=0.3, alpha=1.2, cvdump_interval=20)
        assert float(state.kpush) == pytest.approx(0.3)
        assert float(state.alpha) == pytest.approx(1.2)
        assert int(state.cvdump_interval) == 20
        assert int(state.max_hills) == 100

    def test_step_counter_zero(self):
        state = create_mtd_state(n_atoms=3)
        assert int(state.step_counter) == 0

    def test_custom_reference(self):
        ref = jnp.ones((4, 3))
        state = create_mtd_state(n_atoms=4, reference_xyz=ref)
        assert jnp.allclose(state.reference_xyz, ref)


# ── add_hill ────────────────────────────────────────────────────────────────────


class TestAddHill:
    def test_n_hills_increments(self):
        state = create_mtd_state(n_atoms=3, max_hills=10)
        state = add_hill(state, _triangle())
        assert int(state.n_hills) == 1

    def test_positions_stored(self):
        pos = _triangle()
        state = create_mtd_state(n_atoms=3, max_hills=10)
        state = add_hill(state, pos)
        assert jnp.allclose(state.cv_xyz[0], pos)

    def test_multiple_hills(self):
        state = create_mtd_state(n_atoms=3, max_hills=10)
        pos1, pos2 = _triangle(), _triangle() + 0.5
        state = add_hill(state, pos1)
        state = add_hill(state, pos2)
        assert int(state.n_hills) == 2
        assert jnp.allclose(state.cv_xyz[0], pos1)
        assert jnp.allclose(state.cv_xyz[1], pos2)

    def test_immutability(self):
        state = create_mtd_state(n_atoms=3, max_hills=10)
        _ = add_hill(state, _triangle())
        assert int(state.n_hills) == 0  # original unchanged


# ── step counter / deposition logic ────────────────────────────────────────────


class TestStepCounter:
    def test_increment(self):
        state = create_mtd_state(n_atoms=3, cvdump_interval=10)
        state = increment_step(state)
        assert int(state.step_counter) == 1

    def test_deposit_at_interval(self):
        state = create_mtd_state(n_atoms=3, cvdump_interval=5)
        for _ in range(5):
            state = increment_step(state)
        assert should_deposit_hill(state)

    def test_no_deposit_between_intervals(self):
        state = create_mtd_state(n_atoms=3, cvdump_interval=10)
        for _ in range(3):
            state = increment_step(state)
        assert not should_deposit_hill(state)


# ── compute_mtd_bias ────────────────────────────────────────────────────────────


class TestComputeMtdBias:
    def test_zero_with_no_hills(self):
        state = create_mtd_state(n_atoms=3, max_hills=10, kpush=1.0, alpha=0.5)
        pos = _triangle()
        bias = compute_mtd_bias(pos, state)
        assert float(bias) == pytest.approx(0.0, abs=1e-10)

    def test_maximum_at_hill_center(self):
        """Bias is positive at the hill center once the logistic ramp has advanced."""
        pos = _triangle()
        state = create_mtd_state(n_atoms=3, max_hills=10, kpush=0.5, alpha=0.5)
        state = add_hill(state, pos)
        # At step_counter=0 the logistic ramp weight is exactly 0; advance one step.
        state = increment_step(state)
        bias = compute_mtd_bias(pos, state)
        # The ramp weight w ∈ (0, 1) so bias ∈ (0, kpush)
        assert float(bias) <= 0.5 + 1e-4
        assert float(bias) > 0.0

    def test_decreases_with_displacement(self):
        """Bias at a far position must be smaller than at the hill centre."""
        pos = _triangle()
        state = create_mtd_state(n_atoms=3, max_hills=10, kpush=1.0, alpha=0.5)
        state = add_hill(state, pos)
        # Advance step counter so ramp weight approaches 1
        for _ in range(50):
            state = increment_step(state)
        bias_center = float(compute_mtd_bias(pos, state))
        # Displace ONLY the test position while the hill stays at pos
        far = pos + 5.0
        bias_far = float(compute_mtd_bias(far, state))
        assert bias_center > bias_far

    def test_accumulates_with_multiple_hills(self):
        state = create_mtd_state(n_atoms=3, max_hills=20, kpush=0.1, alpha=0.5)
        pos = _triangle()
        for i in range(5):
            state = add_hill(state, pos + i * 0.05)
        bias = float(compute_mtd_bias(pos, state))
        assert bias > 0.0

    def test_positive_always(self):
        state = create_mtd_state(n_atoms=3, max_hills=10, kpush=0.8, alpha=0.5)
        pos = _triangle()
        for offset in [0.0, 0.5, 2.0]:
            state = add_hill(state, pos + offset)
        for test_pos in [pos, pos + 1.0, pos - 1.0]:
            assert float(compute_mtd_bias(test_pos, state)) >= 0.0

    def test_translation_invariant(self):
        """RMSD-based bias is invariant to global translation."""
        pos = _triangle()
        state1 = create_mtd_state(n_atoms=3, max_hills=10, kpush=0.5, alpha=0.5)
        state1 = add_hill(state1, pos)
        bias1 = float(compute_mtd_bias(pos + 0.3, state1))

        shift = jnp.array([10.0, 20.0, 30.0])
        state2 = create_mtd_state(n_atoms=3, max_hills=10, kpush=0.5, alpha=0.5)
        state2 = add_hill(state2, pos + shift)
        bias2 = float(compute_mtd_bias(pos + 0.3 + shift, state2))

        assert bias1 == pytest.approx(bias2, rel=1e-5)

    def test_differentiable(self):
        """compute_mtd_bias must be JAX-differentiable."""
        pos = _triangle()
        state = create_mtd_state(n_atoms=3, max_hills=10, kpush=0.5, alpha=0.5)
        state = add_hill(state, pos)
        grad_fn = jax.grad(lambda p: compute_mtd_bias(p, state))
        grad = grad_fn(pos + 0.2)
        assert grad.shape == pos.shape
        assert not jnp.any(jnp.isnan(grad))


# ── statistics ──────────────────────────────────────────────────────────────────


class TestMtdStatistics:
    def test_keys_present(self):
        state = create_mtd_state(n_atoms=5, max_hills=100, kpush=0.8, alpha=0.5)
        for _ in range(3):
            state = add_hill(state, jax.random.normal(jax.random.PRNGKey(0), (5, 3)))
        stats = get_mtd_statistics(state)
        for key in ["n_hills", "kpush", "alpha", "max_hills", "memory_mb"]:
            assert key in stats, f"Missing key: {key}"

    def test_n_hills_correct(self):
        state = create_mtd_state(n_atoms=4, max_hills=50)
        for i in range(7):
            state = add_hill(state, jax.random.normal(jax.random.PRNGKey(i), (4, 3)))
        stats = get_mtd_statistics(state)
        assert int(stats["n_hills"]) == 7

    def test_memory_positive(self):
        state = create_mtd_state(n_atoms=10, max_hills=200)
        stats = get_mtd_statistics(state)
        assert stats["memory_mb"] > 0.0


# ── save / load ──────────────────────────────────────────────────────────────────


class TestSaveLoad:
    def test_save_creates_file(self):
        """save_mtd_state must create a readable .npz file."""
        state = create_mtd_state(n_atoms=5, max_hills=20, kpush=0.7, alpha=1.2)
        for i in range(4):
            state = add_hill(state, jax.random.normal(jax.random.PRNGKey(i), (5, 3)))

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            fname = f.name
        try:
            save_mtd_state(state, fname)
            assert os.path.exists(fname), "File not created"
            # Verify key arrays are present in the archive
            data = np.load(fname)
            assert "cv_xyz" in data
            assert "n_hills" in data
            assert int(data["n_hills"]) == int(state.n_hills)
            assert float(data["kpush"]) == pytest.approx(float(state.kpush))
            assert float(data["alpha"]) == pytest.approx(float(state.alpha))
        finally:
            if os.path.exists(fname):
                os.remove(fname)

    def test_roundtrip_npz(self):
        """load_mtd_state should reconstruct the state."""
        state = create_mtd_state(n_atoms=5, max_hills=20, kpush=0.7, alpha=1.2)
        for i in range(4):
            state = add_hill(state, jax.random.normal(jax.random.PRNGKey(i), (5, 3)))

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            fname = f.name
        try:
            save_mtd_state(state, fname)
            loaded = load_mtd_state(fname)
            assert int(loaded.n_hills) == int(state.n_hills)
            assert float(loaded.kpush) == pytest.approx(float(state.kpush))
            assert int(loaded.max_hills) == int(state.max_hills)
            assert int(loaded.cvdump_interval) == int(state.cvdump_interval)
        finally:
            if os.path.exists(fname):
                os.remove(fname)
