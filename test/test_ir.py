"""Tests for mars/ir.py — Hessian, normal modes, IR spectrum.

These tests use the harmonic test potential so no ML weights are needed.
End-to-end ``compute_ir_spectrum`` / ``compute_ir_from_md`` exercises that
require SO3LR are kept in CLI / integration tests.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mars import ir as ir_mod
from mars.utils import get_atomic_masses


@pytest.fixture
def hh_positions():
    """Two-atom system in a 1-D harmonic well around (0, 0)."""
    return jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])


@pytest.fixture
def hh_numbers():
    return jnp.array([1, 1])


def _harmonic_energy(positions):
    """Quadratic potential pulling each atom toward the geometric centre."""
    centre = jnp.mean(positions, axis=0)
    return jnp.sum((positions - centre) ** 2) * 0.5


# ── Hessian computation ─────────────────────────────────────────────────────


def test_hessian_shape(hh_positions):
    H = ir_mod.compute_hessian(_harmonic_energy, hh_positions, use_jit=False)
    n_dof = hh_positions.size
    assert H.shape == (n_dof, n_dof)


def test_hessian_symmetric(hh_positions):
    H = ir_mod.compute_hessian(_harmonic_energy, hh_positions, use_jit=False)
    H = np.asarray(H)
    np.testing.assert_allclose(H, H.T, atol=1e-6)


def test_hessian_autodiff_matches_fd(hh_positions):
    H_ad = np.asarray(ir_mod.compute_hessian(_harmonic_energy, hh_positions, use_jit=False))
    H_fd = np.asarray(ir_mod.compute_hessian_fd(_harmonic_energy, hh_positions, displacement=1e-3))
    np.testing.assert_allclose(H_ad, H_fd, atol=1e-3)


# ── Normal modes ────────────────────────────────────────────────────────────


def test_normal_modes_shape_and_realness(hh_positions, hh_numbers):
    H = ir_mod.compute_hessian(_harmonic_energy, hh_positions, use_jit=False)
    masses = get_atomic_masses(hh_numbers)
    # Mass-weight the Hessian
    m_inv_sqrt = 1.0 / jnp.sqrt(jnp.repeat(masses, 3))
    H_mw = H * m_inv_sqrt[:, None] * m_inv_sqrt[None, :]
    freqs, modes, pops = ir_mod.compute_normal_modes(H_mw, masses, temperature=300.0)
    assert freqs.shape == (hh_positions.size,)
    assert modes.shape == (hh_positions.size, hh_positions.size)
    assert pops.shape == (hh_positions.size,)
    # No NaN / inf
    assert np.all(np.isfinite(np.asarray(freqs)))


def test_normal_modes_harmonic_oscillator():
    """A simple 1D harmonic potential should have a recognisable mode."""
    positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    numbers = jnp.array([6, 6])  # carbons
    masses = get_atomic_masses(numbers)

    def energy(p):
        # Harmonic restoring along +x with stiffness 10 eV/Å^2
        d = p[1, 0] - p[0, 0] - 1.0
        return 0.5 * 10.0 * d**2

    H = np.asarray(ir_mod.compute_hessian(energy, positions, use_jit=False))
    m_inv_sqrt = 1.0 / np.sqrt(np.repeat(np.asarray(masses), 3))
    H_mw = H * m_inv_sqrt[:, None] * m_inv_sqrt[None, :]
    freqs, _, _ = ir_mod.compute_normal_modes(jnp.array(H_mw), masses)
    # Among the 6 modes there must be at least one non-trivial vibrational
    # frequency (above the 50 cm⁻¹ translation/rotation threshold).
    high = np.abs(np.asarray(freqs)) > 50.0
    assert high.any()


# ── Save IR spectrum (file I/O) ─────────────────────────────────────────────


def test_save_ir_spectrum_roundtrip(tmp_output_dir):
    freqs = np.linspace(0, 4000, 200)
    intens = np.exp(-(((freqs - 1500.0) / 50.0) ** 2))
    results = {
        "frequencies": freqs,
        "intensities": intens,
        "intensities_normalized": intens,
        "temperature": 300.0,
        "energy": -10.0,
    }
    out = tmp_output_dir / "spec.dat"
    ir_mod.save_ir_spectrum(results, str(out))
    assert out.exists()
    data = np.loadtxt(str(out))
    assert data.shape[0] > 0
    assert data.shape[1] >= 2


# ── _select_conformer / _resolve_dtype helpers ──────────────────────────────


def test_select_conformer_picks_index_in_range():
    structures = [{"positions": np.zeros((1, 3))} for _ in range(3)]
    s, idx = ir_mod._select_conformer(structures, conformer_index=1)
    assert idx == 1
    assert s is structures[1]


def test_select_conformer_clamps_overflow():
    structures = [{"positions": np.zeros((1, 3))} for _ in range(3)]
    _, idx = ir_mod._select_conformer(structures, conformer_index=99)
    assert idx == 2


def test_select_conformer_default_to_zero():
    structures = [{"positions": np.zeros((1, 3))} for _ in range(2)]
    _, idx = ir_mod._select_conformer(structures, conformer_index=None)
    assert idx == 0


def test_resolve_dtype_float64():
    assert ir_mod._resolve_dtype(use_float64=True) == jnp.float64


def test_resolve_dtype_md_forces_float64():
    assert ir_mod._resolve_dtype(use_float64=False, force_float64_for_md=True) == jnp.float64


# ── _build_ir_potential routes potential_options ────────────────────────────


def test_build_ir_potential_harmonic():
    pot = ir_mod._build_ir_potential(
        "harmonic",
        numbers=np.array([1, 1]),
        lr_cutoff=10.0,
        charge=0.0,
        model_path=None,
        dtype=jnp.float32,
        compute_charges=False,
    )
    assert pot is not None
