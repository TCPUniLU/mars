"""Tests for the initial-Hessian models in mars.utils (identity and Lindh)."""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mars.utils import (  # noqa: E402
    HESSIAN_MODES,
    lindh_model_hessian_cartesian,
    model_hessian_cartesian,
)

# Staggered ethane, Angstrom.
_ETHANE_Z = np.array([6, 6, 1, 1, 1, 1, 1, 1])
_ETHANE_X = np.array(
    [
        [0.000, 0.000, 0.765],
        [0.000, 0.000, -0.765],
        [0.000, 1.017, 1.163],
        [0.881, -0.508, 1.163],
        [-0.881, -0.508, 1.163],
        [0.000, -1.017, -1.163],
        [0.881, 0.508, -1.163],
        [-0.881, 0.508, -1.163],
    ]
)


def test_modes_are_identity_and_lindh():
    assert HESSIAN_MODES == ("identity", "lindh")


def test_identity():
    h = model_hessian_cartesian("identity", _ETHANE_X, _ETHANE_Z)
    assert h.shape == (24, 24)
    assert np.allclose(np.asarray(h), np.eye(24))


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown init_hessian mode"):
        model_hessian_cartesian("schlegel", _ETHANE_X, _ETHANE_Z)


class TestLindh:
    def test_symmetric_and_positive_semidefinite(self):
        h = lindh_model_hessian_cartesian(_ETHANE_X, _ETHANE_Z)
        h = np.asarray(h)
        assert h.shape == (24, 24)
        assert np.allclose(h, h.T, atol=1e-10)
        assert float(np.linalg.eigvalsh(h)[0]) > -1e-8

    def test_six_soft_modes(self):
        """A model Hessian built from internal coordinates must be blind to
        the 6 rigid-body motions."""
        h = np.asarray(lindh_model_hessian_cartesian(_ETHANE_X, _ETHANE_Z))
        ev = np.linalg.eigvalsh(h)
        assert np.sum(np.abs(ev) < 1e-6) >= 6

    def test_cc_stretch_magnitude(self):
        """The C-C diagonal block should carry a chemically sane stiffness:
        a few tens of eV/A^2, not 1 and not 1e4."""
        h = np.asarray(lindh_model_hessian_cartesian(_ETHANE_X, _ETHANE_Z))
        # z is the C-C axis for this geometry.
        k = h[2, 2]
        assert 5.0 < k < 200.0

    def test_defined_for_nonbonded_pairs(self):
        """Lindh's damping is defined for any pair, which is what makes it
        usable for interfragment coordinates -- two well-separated waters
        must still give a finite, non-singular Hessian."""
        z = np.array([8, 1, 1, 8, 1, 1])
        x = np.array(
            [
                [0.000, 0.000, 0.000],
                [0.757, 0.586, 0.000],
                [-0.757, 0.586, 0.000],
                [0.000, 0.000, 6.000],
                [0.757, 0.586, 6.000],
                [-0.757, 0.586, 6.000],
            ]
        )
        h = np.asarray(lindh_model_hessian_cartesian(x, z))
        assert np.all(np.isfinite(h))
        assert float(np.linalg.eigvalsh(h)[0]) > -1e-8

    def test_stiffens_as_bond_shortens(self):
        """rho = exp[alpha (r_ref^2 - r^2)] means a compressed bond is stiffer."""
        x_short = _ETHANE_X.copy()
        x_short[0, 2], x_short[1, 2] = 0.65, -0.65
        k_ref = float(np.asarray(lindh_model_hessian_cartesian(_ETHANE_X, _ETHANE_Z))[2, 2])
        k_short = float(np.asarray(lindh_model_hessian_cartesian(x_short, _ETHANE_Z))[2, 2])
        assert k_short > k_ref

    def test_dtype_follows_input(self):
        h = lindh_model_hessian_cartesian(jnp.asarray(_ETHANE_X), _ETHANE_Z)
        assert isinstance(h, jnp.ndarray)
