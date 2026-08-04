"""
Tests for mars.rmsd — Kabsch alignment and RMSD collective variables.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from mars.rmsd import (
    kabsch_jax,
    center_positions,
    align_to_reference,
    rmsd_cv_jax,
    rmsd_and_grad_jax,
    rmsd_to_multiple_refs,
    pairwise_rmsd_matrix,
    check_rmsd_gradient,
)

# ── center_positions ────────────────────────────────────────────────────────────


class TestCenterPositions:
    def test_centroid_is_zero(self):
        X = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        Xc = center_positions(X)
        assert jnp.allclose(jnp.mean(Xc, axis=0), 0.0, atol=1e-10)

    def test_shape_preserved(self):
        X = jnp.ones((6, 3))
        assert center_positions(X).shape == (6, 3)

    def test_already_centered(self):
        X = jnp.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, -1.0, 0.0]])
        Xc = center_positions(X)
        assert jnp.allclose(Xc, X, atol=1e-10)


# ── kabsch_jax ──────────────────────────────────────────────────────────────────


class TestKabsch:
    def test_identity_for_same_structure(self):
        X = center_positions(jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        R = kabsch_jax(X, X)
        assert jnp.allclose(R, jnp.eye(3), atol=1e-6)

    def test_rotation_matrix_properties(self):
        key = jax.random.PRNGKey(0)
        X = center_positions(jax.random.normal(key, (5, 3)))
        Y = center_positions(jax.random.normal(jax.random.PRNGKey(1), (5, 3)))
        R = kabsch_jax(X, Y)
        # orthogonal
        assert jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-6)
        # proper rotation (det = +1)
        assert jnp.allclose(jnp.linalg.det(R), 1.0, atol=1e-6)

    def test_recovers_known_rotation(self):
        # Use more atoms (3 coplanar points are rank-2; add a 4th off-plane)
        X = center_positions(
            jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.5]])
        )
        theta = jnp.pi / 3
        Rz = jnp.array(
            [
                [jnp.cos(theta), -jnp.sin(theta), 0.0],
                [jnp.sin(theta), jnp.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        Y = X @ Rz
        # kabsch_jax(P, Q) returns R s.t. Q @ R ≈ P (aligns Q onto P)
        # So kabsch_jax(X, Y) → Y @ R ≈ X
        R = kabsch_jax(X, Y)
        assert jnp.allclose(Y @ R, X, atol=1e-4)


# ── rmsd_cv_jax ─────────────────────────────────────────────────────────────────


class TestRmsdCv:
    def test_identical_structures_zero(self):
        X = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        # sqrt regularisation gives ~1e-8 for identical structures
        assert float(rmsd_cv_jax(X, X)) < 1e-6

    def test_pure_translation_zero(self):
        """RMSD must be zero after optimal alignment of a translated copy."""
        X = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        Y = X + jnp.array([5.0, -3.0, 2.0])
        assert float(rmsd_cv_jax(X, Y)) < 1e-6

    def test_positive_for_different_structures(self):
        X = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        Y = jnp.array([[0.5, 0.5, 0.0], [1.5, 0.5, 0.0], [0.0, 1.5, 0.5]])
        assert float(rmsd_cv_jax(X, Y)) > 0.0

    def test_scalar_output(self):
        X = jnp.ones((4, 3))
        Y = X + 0.1
        assert rmsd_cv_jax(X, Y).shape == ()

    def test_symmetry(self):
        key = jax.random.PRNGKey(7)
        X = jax.random.normal(key, (5, 3))
        Y = jax.random.normal(jax.random.PRNGKey(8), (5, 3))
        assert jnp.allclose(rmsd_cv_jax(X, Y), rmsd_cv_jax(Y, X), atol=1e-6)


# ── rmsd_and_grad_jax ───────────────────────────────────────────────────────────


class TestRmsdAndGrad:
    def test_output_shapes(self):
        X = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        Y = X + 0.1
        rmsd, grad = rmsd_and_grad_jax(X, Y)
        assert rmsd.shape == ()
        assert grad.shape == X.shape

    def test_grad_nonzero_for_displaced(self):
        # Y must be a genuine deformation, not a rigid shift: RMSD is
        # translation/rotation invariant, so a uniform offset (X + c) gives
        # a ~zero gradient. Move a single atom to change the shape.
        X = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        Y = X.at[2].add(jnp.array([0.0, 0.3, 0.0]))
        _, grad = rmsd_and_grad_jax(X, Y)
        assert jnp.max(jnp.abs(grad)) > 1e-8

    def test_grad_matches_finite_diff(self):
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (4, 3)) * 2.0
        Y = jax.random.normal(jax.random.PRNGKey(43), (4, 3)) * 2.0
        _, grad_auto, max_err = check_rmsd_gradient(X, Y, epsilon=1e-6)
        assert max_err < 1e-4, f"Gradient error {max_err:.2e} too large"


# ── rmsd_to_multiple_refs ───────────────────────────────────────────────────────


class TestRmsdToMultipleRefs:
    def test_shape(self):
        X = jnp.zeros((3, 3))
        refs = jnp.stack([X, X + 0.1, X + 0.5])
        out = rmsd_to_multiple_refs(X, refs)
        assert out.shape == (3,)

    def test_first_is_zero_for_identical(self):
        X = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        refs = jnp.stack([X, X + 1.0, X + 2.0])
        rmsds = rmsd_to_multiple_refs(X, refs)
        assert float(rmsds[0]) < 1e-6  # sqrt regularisation

    def test_pure_translation_zero_for_all(self):
        """Translation-invariant RMSD should be near-zero for pure shifts."""
        X = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        refs = jnp.stack([X + d for d in [0.0, 1.0, 3.0]])
        rmsds = rmsd_to_multiple_refs(X, refs)
        # after optimal alignment a pure translation should be ~0 for all
        for r in rmsds:
            assert float(r) < 1e-6


# ── pairwise_rmsd_matrix ────────────────────────────────────────────────────────


class TestPairwiseRmsdMatrix:
    def test_shape(self):
        key = jax.random.PRNGKey(0)
        structs = jax.random.normal(key, (4, 5, 3))
        mat = pairwise_rmsd_matrix(structs)
        assert mat.shape == (4, 4)

    def test_diagonal_zero(self):
        key = jax.random.PRNGKey(1)
        structs = jax.random.normal(key, (3, 4, 3))
        mat = pairwise_rmsd_matrix(structs)
        # Diagonal should be very small (sqrt regularisation gives ~1e-8)
        assert jnp.allclose(jnp.diag(mat), 0.0, atol=1e-6)

    def test_symmetric(self):
        key = jax.random.PRNGKey(2)
        structs = jax.random.normal(key, (4, 5, 3))
        mat = pairwise_rmsd_matrix(structs)
        assert jnp.allclose(mat, mat.T, atol=1e-6)

    def test_off_diagonal_positive(self):
        key = jax.random.PRNGKey(3)
        # Make clearly distinct structures
        base = jax.random.normal(key, (5, 3))
        structs = jnp.stack([base + i * 5.0 for i in range(4)])
        mat = pairwise_rmsd_matrix(structs)
        # Off-diagonal should be positive
        off_diag = mat + jnp.eye(4) * 999.0
        assert jnp.all(mat[~jnp.eye(4, dtype=bool)] >= 0.0)
