"""Tests for mars.internal_coords — coordinate set, Wilson B, back-transformation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from mars.internal_coords import (  # noqa: E402
    build_coordinates,
    cartesian_coordinate_fns,
    generalized_inverse,
    global_tr_basis,
    initial_frag_ref,
    internal_to_cartesian,
    make_coordinate_fns,
    wrap_periodic,
)

# ── geometries ──────────────────────────────────────────────────────────────

WATER = (np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]), np.array([8, 1, 1]))
CO2 = (np.array([[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]]), np.array([6, 8, 8]))
ACETYLENE = (
    np.array([[0.0, 0, 0], [1.2, 0, 0], [-1.06, 0, 0], [2.26, 0, 0]]),
    np.array([6, 6, 1, 1]),
)
METHANE = (
    np.array(
        [
            [0.0, 0, 0],
            [0.63, 0.63, 0.63],
            [-0.63, -0.63, 0.63],
            [-0.63, 0.63, -0.63],
            [0.63, -0.63, -0.63],
        ]
    ),
    np.array([6, 1, 1, 1, 1]),
)
HF = (np.array([[0.0, 0, 0], [0.92, 0, 0]]), np.array([9, 1]))
WATER_DIMER = (
    np.array(
        [
            [0.0, 0, 0],
            [0.96, 0, 0],
            [-0.24, 0.93, 0],
            [2.85, 0, 0],
            [3.2, 0.9, 0],
            [3.2, -0.45, 0.78],
        ]
    ),
    np.array([8, 1, 1, 8, 1, 1]),
)


def _benzene():
    th = np.arange(6) * np.pi / 3
    pos = np.stack(
        [
            np.concatenate([1.39 * np.cos(th), 2.47 * np.cos(th)]),
            np.concatenate([1.39 * np.sin(th), 2.47 * np.sin(th)]),
            np.zeros(12),
        ],
        axis=1,
    )
    return pos, np.array([6] * 6 + [1] * 6)


BENZENE = _benzene()


def _alanine():
    from mars.utils import load_structure

    s = load_structure("examples/alanine/alanine_dipeptide.xyz")
    return np.asarray(s["positions"]), np.asarray(s["numbers"])


def _setup(pos, z, **kw):
    spec = build_coordinates(pos, z, **kw)
    fns = make_coordinate_fns(spec)
    x = jnp.asarray(pos)
    ref = initial_frag_ref(spec, x)
    return spec, fns, x, ref


# ── B matrix ────────────────────────────────────────────────────────────────


class TestBMatrix:
    @pytest.mark.parametrize(
        "case",
        [WATER, METHANE, BENZENE, CO2, WATER_DIMER],
        ids=["water", "methane", "benzene", "co2", "dimer"],
    )
    def test_matches_jacrev(self, case):
        """The scatter-built B must be exactly the Jacobian of the q actually used."""
        spec, fns, x, ref = _setup(*case)
        b = fns.b_fn(x, ref)
        b_ref = jax.jacrev(lambda xf: fns.q_fn(xf.reshape(-1, 3), ref))(x.reshape(-1))
        assert float(jnp.max(jnp.abs(b - b_ref))) < 1e-10

    def test_matches_finite_differences(self):
        """Independent anchor: central differences of q, no AD involved."""
        spec, fns, x, ref = _setup(*WATER)
        h = 1e-5
        b = np.asarray(fns.b_fn(x, ref))
        fd = np.zeros_like(b)
        flat = np.asarray(x).reshape(-1)
        for k in range(flat.size):
            for sgn, store in ((+1, 1), (-1, -1)):
                p = flat.copy()
                p[k] += sgn * h
                q = np.asarray(fns.q_fn(jnp.asarray(p.reshape(-1, 3)), ref))
                fd[:, k] += store * q / (2 * h)
        assert np.max(np.abs(b - fd)) < 1e-6

    def test_matches_ir_wilson_formulas(self):
        """Cross-check against the hand-coded rows in mars/ir.py."""
        from mars.ir import _project_mode_onto_bends, _project_mode_onto_stretches

        pos, z = _alanine()
        spec, fns, x, ref = _setup(pos, z)
        b = np.asarray(fns.b_fn(x, ref))
        rng = np.random.default_rng(0)
        disp = rng.normal(size=pos.shape) * 1e-3
        n_s = len(spec.stretch_idx)
        # Stretch rows: B @ disp must equal the projection ir.py computes.
        got = b[:n_s] @ disp.reshape(-1)
        _t, per_bond = _project_mode_onto_stretches(
            disp, pos, [tuple(map(int, p)) for p in spec.stretch_idx]
        )
        want = np.array([np.sqrt(v) for _b, v in per_bond]) * np.sign(got)
        assert np.allclose(np.abs(got), np.abs(want), atol=1e-10)

    @pytest.mark.parametrize(
        "case", [WATER, BENZENE, WATER_DIMER], ids=["water", "benzene", "dimer"]
    )
    def test_translation_nullspace(self, case):
        spec, fns, x, ref = _setup(*case)
        b = fns.b_fn(x, ref)
        ginv = generalized_inverse(b, tr_basis=global_tr_basis(x))
        for a in range(3):
            t = jnp.zeros_like(x).at[:, a].set(1.0).reshape(-1)
            assert float(jnp.max(jnp.abs(ginv.b_pinv.T @ t))) < 1e-10

    @pytest.mark.parametrize(
        "case", [WATER, BENZENE, WATER_DIMER], ids=["water", "benzene", "dimer"]
    )
    def test_rotation_nullspace(self, case):
        spec, fns, x, ref = _setup(*case)
        b = fns.b_fn(x, ref)
        ginv = generalized_inverse(b, tr_basis=global_tr_basis(x))
        centred = x - jnp.mean(x, axis=0)
        for a in range(3):
            axis = jnp.zeros(3).at[a].set(1.0)
            w = jnp.cross(jnp.broadcast_to(axis, x.shape), centred).reshape(-1)
            assert float(jnp.max(jnp.abs(ginv.b_pinv.T @ w))) < 1e-9


class TestInvariance:
    @pytest.mark.parametrize("case", [WATER, METHANE, BENZENE], ids=["water", "methane", "benzene"])
    def test_q_invariant_under_rigid_motion(self, case):
        """Values, as opposed to derivatives: q(Rx + t) == q(x)."""
        pos, z = case
        spec, fns, x, ref = _setup(pos, z)
        key = jax.random.PRNGKey(3)
        a = jax.random.normal(key, (3, 3))
        rot, _ = jnp.linalg.qr(a)
        rot = rot * jnp.sign(jnp.linalg.det(rot))
        shift = jnp.array([1.3, -0.7, 2.2])
        q0 = fns.q_fn(x, ref)
        q1 = fns.q_fn(x @ rot.T + shift, ref)
        # Fragment translations are not invariant under a global shift by
        # construction, so compare only the internal (non-TRIC) block. A
        # torsion sitting exactly at +-pi (benzene has several) flips branch
        # under rotation, which is invariance modulo 2*pi, hence the wrap.
        n_int_only = spec.n_internal - 3 * spec.frag_atoms.shape[0] - 3 * len(spec.rot_frag)
        diff = fns.dq_fn(q1, q0)[:n_int_only]
        assert float(jnp.max(jnp.abs(diff))) < 1e-9


class TestCoordinateCounts:
    @pytest.mark.parametrize(
        "case,linear",
        [
            (WATER, False),
            (CO2, True),
            (ACETYLENE, True),
            (METHANE, False),
            (BENZENE, False),
            (WATER_DIMER, False),
        ],
        ids=["water", "co2", "acetylene", "methane", "benzene", "dimer"],
    )
    def test_rank_equals_expected(self, case, linear):
        pos, z = case
        spec = build_coordinates(pos, z)
        n3 = 3 * len(z)
        assert spec.valid, spec.reason
        assert spec.expected_rank == n3 - (5 if linear else 6)
        assert spec.n_internal >= spec.expected_rank
        assert spec.rank_at_build == spec.expected_rank

    def test_diatomic_is_refused(self):
        spec = build_coordinates(*HF)
        assert not spec.valid
        assert spec.reason == "too_few_atoms"

    def test_alanine_dipeptide(self):
        pos, z = _alanine()
        spec = build_coordinates(pos, z)
        assert spec.valid and spec.rank_at_build == 3 * len(z) - 6


class TestLinearAngles:
    def test_no_ordinary_bend_at_linear_centre(self):
        spec = build_coordinates(*CO2)
        assert len(spec.bend_idx) == 0
        assert len(spec.linear_idx) == 1

    def test_near_linear_band_gets_both(self):
        """An angle at ~170 deg carries an ordinary bend *and* a linear pair."""
        pos = np.array([[0.0, 0, 0], [1.2, 0, 0], [-1.18, 0.21, 0]])
        spec = build_coordinates(pos, np.array([6, 8, 8]))
        assert len(spec.bend_idx) == 1
        assert len(spec.linear_idx) == 1
        assert bool(spec.bend_has_linear[0])


class TestTRIC:
    def test_dimer_gets_rigid_body_coordinates(self):
        spec = build_coordinates(*WATER_DIMER, interfragment="tric")
        assert spec.n_fragments == 2
        assert spec.frag_atoms.shape[0] == 2
        assert len(spec.rot_frag) == 2
        assert spec.rank_at_build == 12

    def test_aux_scheme_also_complete(self):
        spec = build_coordinates(*WATER_DIMER, interfragment="aux")
        assert spec.n_aux_bonds > 0
        assert spec.rank_at_build == 12

    def test_translation_rotation_rows_are_constant(self):
        """The linearised rotation coordinate is linear in x, so its B rows
        must be identical at two different geometries with the same reference."""
        pos, z = WATER_DIMER
        spec, fns, x, ref = _setup(pos, z, interfragment="tric")
        n_tric = 3 * spec.frag_atoms.shape[0] + 3 * len(spec.rot_frag)
        b0 = fns.b_fn(x, ref)[-n_tric:]
        x2 = x + jax.random.normal(jax.random.PRNGKey(5), x.shape) * 0.2
        b1 = fns.b_fn(x2, ref)[-n_tric:]
        assert float(jnp.max(jnp.abs(b0 - b1))) < 1e-12

    def test_single_atom_fragment_emits_no_rotation(self):
        pos = np.array([[0.0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0], [6.0, 0, 0]])
        spec = build_coordinates(pos, np.array([8, 1, 1, 18]), interfragment="tric")
        assert spec.frag_atoms.shape[0] == 2
        assert len(spec.rot_frag) == 1  # only the water rotates

    def test_linear_fragment_emits_two_rotations(self):
        from mars.internal_coords import _rot_metric

        ref = np.array([[-1.16, 0, 0], [0.0, 0, 0], [1.16, 0, 0]])
        _pinv, rank = _rot_metric(ref - ref.mean(0), np.full(3, 1 / 3))
        assert rank == 2


class TestWrapping:
    def test_wrap_takes_the_short_way(self):
        is_per = jnp.array([True])
        dq = jnp.array([np.deg2rad(-358.0)])
        assert abs(float(wrap_periodic(dq, is_per)[0]) - np.deg2rad(2.0)) < 1e-12

    def test_non_periodic_untouched(self):
        is_per = jnp.array([False])
        assert float(wrap_periodic(jnp.array([10.0]), is_per)[0]) == 10.0


class TestBackTransform:
    @pytest.mark.parametrize("scale,tol", [(0.005, 5e-3), (0.02, 2e-2), (0.08, 8e-2)])
    def test_roundtrip_projected(self, scale, tol):
        """A finite step's residual floors out at the manifold curvature, so
        the tolerance has to scale with the step, not be a fixed 1e-8."""
        pos, z = _alanine()
        spec, fns, x, ref = _setup(pos, z)
        ginv = generalized_inverse(fns.b_fn(x, ref), tr_basis=global_tr_basis(x))
        raw = jax.random.normal(jax.random.PRNGKey(1), (spec.n_internal,)) * scale
        dq = ginv.projector @ raw
        r = internal_to_cartesian(fns, x, ref, dq, ginv, jnp.zeros(3 * len(z)), trust_radius=1.0)
        achieved = fns.dq_fn(fns.q_fn(r.positions, ref), fns.q_fn(x, ref))
        rel = float(jnp.max(jnp.abs(achieved - dq)) / jnp.max(jnp.abs(dq)))
        assert int(r.mode) <= 1
        assert rel < tol

    def test_projection_matters(self):
        """An unprojected dq chases components outside the row space of B."""
        pos, z = _alanine()
        spec, fns, x, ref = _setup(pos, z)
        ginv = generalized_inverse(fns.b_fn(x, ref), tr_basis=global_tr_basis(x))
        raw = jax.random.normal(jax.random.PRNGKey(1), (spec.n_internal,)) * 0.02
        g = jnp.zeros(3 * len(z))
        r_proj = internal_to_cartesian(fns, x, ref, ginv.projector @ raw, ginv, g)
        r_raw = internal_to_cartesian(fns, x, ref, raw, ginv, g)
        assert float(r_proj.residual) < 0.2 * float(r_raw.residual)

    def test_never_nan_on_an_absurd_step(self):
        pos, z = _alanine()
        spec, fns, x, ref = _setup(pos, z)
        ginv = generalized_inverse(fns.b_fn(x, ref), tr_basis=global_tr_basis(x))
        dq = ginv.projector @ jnp.ones((spec.n_internal,)) * 50.0
        r = internal_to_cartesian(fns, x, ref, dq, ginv, jnp.ones(3 * len(z)), trust_radius=0.1)
        assert bool(jnp.all(jnp.isfinite(r.positions)))

    def test_no_centre_of_mass_drift(self):
        pos, z = _alanine()
        spec, fns, x, ref = _setup(pos, z)
        ginv = generalized_inverse(fns.b_fn(x, ref), tr_basis=global_tr_basis(x))
        dq = ginv.projector @ (jax.random.normal(jax.random.PRNGKey(2), (spec.n_internal,)) * 0.02)
        r = internal_to_cartesian(fns, x, ref, dq, ginv, jnp.zeros(3 * len(z)))
        drift = jnp.mean(r.positions, axis=0) - jnp.mean(x, axis=0)
        assert float(jnp.max(jnp.abs(drift))) < 1e-9


class TestCartesianSpec:
    def test_identity_b_and_exact_backtransform(self):
        pos, z = WATER
        fns = cartesian_coordinate_fns(len(z))
        x = jnp.asarray(pos)
        b = fns.b_fn(x, jnp.zeros((0, 1, 3)))
        assert float(jnp.max(jnp.abs(b - jnp.eye(3 * len(z))))) == 0.0


class TestVmap:
    def test_batch_matches_single(self):
        pos, z = _alanine()
        spec, fns, x, ref = _setup(pos, z)
        batch = jnp.stack([x, x + 0.01, x - 0.01])
        q_b = jax.vmap(fns.q_fn, in_axes=(0, None))(batch, ref)
        for i in range(3):
            assert float(jnp.max(jnp.abs(q_b[i] - fns.q_fn(batch[i], ref)))) < 1e-12


class TestTopologyAgreement:
    @pytest.mark.parametrize("case", [METHANE, BENZENE], ids=["methane", "benzene"])
    def test_bonds_and_angles_match_ir(self, case):
        from mars.ir import _build_topology
        from mars.utils import NUMBER_TO_SYMBOL

        pos, z = case
        symbols = [NUMBER_TO_SYMBOL[int(v)] for v in z]
        ref = _build_topology(pos, z, symbols, tolerance=1.3)
        spec = build_coordinates(pos, z)
        n_cov = len(spec.stretch_idx) - spec.n_aux_bonds
        got = sorted(tuple(map(int, b)) for b in spec.stretch_idx[:n_cov])
        assert got == sorted(tuple(b) for b in ref["bonds"])
