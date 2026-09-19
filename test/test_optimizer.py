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


# ── coordinate systems and initial Hessians ─────────────────────────────────

from mars.optimizer import (  # noqa: E402
    optimize_batch_parallel,
    optimize_single_with_logging,
    ric_gate,
)
from mars.potentials import get_potential  # noqa: E402

_ALANINE = "examples/alanine/alanine_dipeptide.xyz"


def _alanine():
    from mars.utils import load_structure

    s = load_structure(_ALANINE)
    return np.asarray(s["positions"]), np.asarray(s["numbers"])


def _valence(pos, z):
    return get_potential("valence", species=z, positions=pos).build_energy_fn()


def _butane_like():
    """A short chain: torsion-dominated, which is where RIC should show up."""
    pos = np.array(
        [
            [0.00, 0.00, 0.00],
            [1.53, 0.00, 0.00],
            [2.04, 1.44, 0.00],
            [3.57, 1.44, 0.10],
            [-0.38, 0.52, 0.89],
            [-0.38, 0.52, -0.89],
            [-0.38, -1.03, 0.02],
            [1.91, -0.52, -0.89],
            [1.91, -0.52, 0.89],
            [1.66, 1.96, 0.89],
            [1.66, 1.96, -0.89],
            [3.95, 0.93, -0.79],
            [3.95, 0.93, 1.00],
            [3.95, 2.47, 0.12],
        ]
    )
    z = np.array([6, 6, 6, 6] + [1] * 10)
    return pos, z


class TestBackwardCompatibility:
    def test_defaults_unchanged_without_atomic_numbers(self):
        """The single most important guard: an existing call must be
        bit-identical to what it produced before the coordinate axis existed."""
        pos = _triangle() + 0.3
        a, ea, ia = optimize_single(pos, harmonic, fmax=0.01, maxiter=200)
        b, eb, ib = optimize_single(
            pos, harmonic, fmax=0.01, maxiter=200, coords="cartesian", init_hessian=None
        )
        assert float(jnp.max(jnp.abs(a - b))) == 0.0
        assert float(ea) == float(eb)
        assert ia["iterations"] == ib["iterations"]

    def test_with_logging_forwards_every_kwarg(self):
        """This wrapper used to cherry-pick four kwargs and drop the rest."""
        seen = {}
        import mars.optimizer as mod

        real = mod.optimize_single

        def spy(*args, **kwargs):
            seen.update(kwargs)
            return real(*args, **kwargs)

        mod.optimize_single = spy
        try:
            optimize_single_with_logging(
                _triangle(),
                harmonic,
                fmax=0.05,
                maxiter=20,
                fire_dt_max=0.123,
                method="FIRE",
                atomic_numbers=np.array([6, 6, 6]),
            )
        finally:
            mod.optimize_single = real
        assert seen.get("fire_dt_max") == 0.123
        assert seen.get("method") == "FIRE"
        assert "atomic_numbers" in seen


class TestRICGate:
    def test_missing_atomic_numbers(self):
        ok, reason = ric_gate(_triangle(), None)
        assert not ok and reason == "no_atomic_numbers"

    def test_too_few_atoms(self):
        ok, reason = ric_gate(np.zeros((3, 3)), np.array([1, 1, 1]))
        assert not ok and reason == "too_few_atoms"

    def test_frozen_atoms_refused(self):
        pos, z = _alanine()
        ok, reason = ric_gate(pos, z, frozen_indices=[0, 1])
        assert not ok and reason == "frozen_atoms"

    def test_barrier_refused(self):
        pos, z = _alanine()
        ok, reason = ric_gate(pos, z, barrier_fn=lambda p: jnp.sum(p**2))
        assert not ok and reason == "barrier_fn"

    @pytest.mark.parametrize("method", ["FIRE", "GD", "SP", "HYBRID"])
    def test_unsupported_methods(self, method):
        pos, z = _alanine()
        ok, reason = ric_gate(pos, z, method=method)
        assert not ok and reason == "method_not_supported_in_internals"

    def test_system_too_large(self):
        pos, z = _alanine()
        ok, reason = ric_gate(pos, z, max_atoms=5)
        assert not ok and reason == "system_too_large"

    def test_clashed_geometry(self):
        pos, z = _alanine()
        pos = pos.copy()
        pos[1] = pos[0] + 0.2
        ok, reason = ric_gate(pos, z)
        assert not ok and reason == "clashed_geometry"

    def test_accepts_a_normal_molecule(self):
        pos, z = _alanine()
        ok, reason = ric_gate(pos, z)
        assert ok and reason == ""

    def test_internal_raises_rather_than_silently_falling_back(self):
        """coords='internal' is an assertion: the benchmark needs to know."""
        pos, z = _alanine()
        with pytest.raises(ValueError, match="frozen_atoms"):
            optimize_single(
                jnp.asarray(pos),
                _valence(pos, z),
                coords="internal",
                atomic_numbers=z,
                frozen_indices=[0],
            )


class TestRICOptimization:
    def test_same_minimum_as_cartesian(self):
        pos, z = _alanine()
        ef = _valence(pos, z)
        x0 = jnp.asarray(pos) + jax.random.normal(jax.random.PRNGKey(4), pos.shape) * 0.05
        _xc, ec, _ic = optimize_single(x0, ef, fmax=1e-3, maxiter=800)
        xr, er, ir = optimize_single(
            x0,
            ef,
            fmax=1e-3,
            maxiter=800,
            coords="internal",
            init_hessian="lindh",
            atomic_numbers=z,
        )
        assert ir["coords_used"] == "internal"
        assert ir["converged"]
        # RIC may legitimately find a *lower* basin; it must never find a worse one.
        assert float(er) <= float(ec) + 1e-5

    def test_not_more_iterations_than_cartesian(self):
        """Deliberately weak (<=, no factor) so it cannot flake; the strong
        claim belongs in the benchmark, not in CI."""
        pos, z = _butane_like()
        ef = _valence(pos, z)
        x0 = jnp.asarray(pos) + jax.random.normal(jax.random.PRNGKey(7), pos.shape) * 0.05
        _xc, _ec, ic = optimize_single(x0, ef, fmax=1e-3, maxiter=2000)
        _xr, _er, ir = optimize_single(
            x0,
            ef,
            fmax=1e-3,
            maxiter=2000,
            coords="internal",
            init_hessian="lindh",
            atomic_numbers=z,
        )
        assert ir["converged"]
        assert ir["iterations"] <= ic["iterations"]

    def test_cartesian_stepper_with_model_hessian(self):
        """--coords cartesian --init-hessian lindh is the 2x2's second arm."""
        pos, z = _alanine()
        ef = _valence(pos, z)
        x0 = jnp.asarray(pos) + jax.random.normal(jax.random.PRNGKey(8), pos.shape) * 0.05
        _x, _e, info = optimize_single(
            x0,
            ef,
            fmax=1e-3,
            maxiter=800,
            coords="cartesian",
            init_hessian="lindh",
            atomic_numbers=z,
        )
        assert info["coords_used"] == "cartesian"
        assert info["converged"]

    def test_reports_diagnostics(self):
        pos, z = _alanine()
        ef = _valence(pos, z)
        _x, _e, info = optimize_single(
            jnp.asarray(pos),
            ef,
            fmax=1e-2,
            maxiter=200,
            coords="internal",
            atomic_numbers=z,
        )
        for key in (
            "ric_n_internal",
            "ric_rank",
            "ric_expected_rank",
            "ric_fallback_steps",
            "ric_rejected_steps",
            "ric_init_hessian",
        ):
            assert key in info
        assert info["ric_rank"] == info["ric_expected_rank"]

    def test_never_worse_than_cartesian_on_a_bad_start(self):
        """The bail-to-Cartesian guarantee."""
        pos, z = _alanine()
        ef = _valence(pos, z)
        # Perturbed hard enough to be a genuinely bad start, but not into a
        # clash -- a clash is refused by the gate, which is a different test.
        x0 = jnp.asarray(pos) + jax.random.normal(jax.random.PRNGKey(11), pos.shape) * 0.12
        _xc, ec, _ic = optimize_single(x0, ef, fmax=1e-3, maxiter=1500)
        _xr, er, _ir = optimize_single(
            x0,
            ef,
            fmax=1e-3,
            maxiter=1500,
            coords="internal",
            init_hessian="lindh",
            atomic_numbers=z,
        )
        assert float(er) <= float(ec) + 1e-4


class TestBatch:
    def test_batch_ric_matches_single(self):
        pos, z = _alanine()
        ef = _valence(pos, z)
        x0 = jnp.asarray(pos) + jax.random.normal(jax.random.PRNGKey(9), pos.shape) * 0.03
        batch = jnp.stack([x0, x0, x0])
        _p, e_b, conv = optimize_batch_parallel(
            batch,
            ef,
            fmax=1e-2,
            maxiter=300,
            coords="internal",
            init_hessian="lindh",
            atomic_numbers=z,
        )
        _xs, e_s, _i = optimize_single(
            x0,
            ef,
            fmax=1e-2,
            maxiter=300,
            coords="internal",
            init_hessian="lindh",
            atomic_numbers=z,
        )
        assert bool(jnp.all(conv))
        for i in range(3):
            assert abs(float(e_b[i]) - float(e_s)) < 1e-8

    def test_batch_reports_real_iterations(self):
        """The CLI used to report iterations=maxiter for every conformer."""
        pos, z = _alanine()
        ef = _valence(pos, z)
        batch = jnp.stack([jnp.asarray(pos) + 0.01 * k for k in range(3)])
        _p, _e, conv, iters, mforce = optimize_batch_parallel(
            batch, ef, fmax=1e-2, maxiter=500, return_info=True
        )
        assert bool(jnp.all(conv))
        assert bool(jnp.all(iters > 0)) and bool(jnp.all(iters < 500))
        assert bool(jnp.all(mforce < 1e-2))

    def test_oom_retry_preserves_ric_kwargs(self):
        """The OOM recursion used to pass eleven positional arguments, so a
        halved sub-batch would silently revert to Cartesian."""
        import mars.optimizer as mod

        pos, z = _alanine()
        ef = _valence(pos, z)
        batch = jnp.stack([jnp.asarray(pos), jnp.asarray(pos) + 0.01])
        seen = []
        real = mod._optimize_batch_ric
        state = {"first": True}

        def flaky(*args, **kwargs):
            seen.append(kwargs.get("coords"))
            if state["first"]:
                state["first"] = False
                raise RuntimeError("RESOURCE_EXHAUSTED: Out of memory")
            return real(*args, **kwargs)

        mod._optimize_batch_ric = flaky
        try:
            mod.optimize_batch_parallel(
                batch,
                ef,
                fmax=1e-2,
                maxiter=100,
                coords="internal",
                init_hessian="lindh",
                atomic_numbers=z,
            )
        except Exception:
            pass
        finally:
            mod._optimize_batch_ric = real
        # Both halves must still have asked for internal coordinates.
        assert seen and all(c == "internal" for c in seen)


class TestFrozenAtoms:
    @pytest.mark.parametrize("coords", ["cartesian", "internal"])
    def test_frozen_atoms_do_not_move(self, coords):
        """A guard that should already have existed: the frozen-atom
        implementation pins coordinates inside the energy function, so a
        coordinate transformation could move them while the reported energy
        stayed plausible."""
        from mars.cli._constraints import make_frozen_energy_fn

        pos, z = _alanine()
        frozen = [0, 1, 2]
        ef = make_frozen_energy_fn(_valence(pos, z), pos, frozen)
        x0 = jnp.asarray(pos) + jax.random.normal(jax.random.PRNGKey(12), pos.shape) * 0.05
        kwargs = dict(fmax=1e-2, maxiter=200, atomic_numbers=z, frozen_indices=frozen)
        if coords == "internal":
            # The gate must refuse rather than silently produce a wrong geometry.
            with pytest.raises(ValueError, match="frozen_atoms"):
                optimize_single(x0, ef, coords="internal", **kwargs)
            return
        xo, _e, _i = optimize_single(x0, ef, coords="cartesian", **kwargs)
        fz = jnp.array(frozen)
        assert float(jnp.max(jnp.abs(xo[fz] - x0[fz]))) < 1e-10

    # GD with the default 0.2 A step diverges on this stiff valence potential
    # whether or not anything is frozen, so it gets a step it can survive.
    @pytest.mark.parametrize("method,max_stepsize", [("LBFGS", 0.2), ("FIRE", 0.2), ("GD", 0.01)])
    def test_frozen_indices_alone_pins_atoms(self, method, max_stepsize):
        """frozen_indices must freeze on its own: before this was honoured in
        the Cartesian path it reached only ric_gate, so an unwrapped energy
        function was optimized with no constraint at all -- silently."""
        pos, z = _alanine()
        frozen = [0, 1, 2]
        ef = _valence(pos, z)  # deliberately NOT wrapped by the caller
        x0 = jnp.asarray(pos) + jax.random.normal(jax.random.PRNGKey(3), pos.shape) * 0.05
        xo, _e, _i = optimize_single(
            x0,
            ef,
            fmax=1e-2,
            maxiter=200,
            method=method,
            max_stepsize=max_stepsize,
            atomic_numbers=z,
            frozen_indices=frozen,
        )
        fz = jnp.array(frozen)
        assert float(jnp.max(jnp.abs(xo[fz] - x0[fz]))) < 1e-10
        # and the rest of the molecule did move, i.e. this is a real relaxation
        free = jnp.array([i for i in range(len(z)) if i not in frozen])
        assert float(jnp.max(jnp.abs(xo[free] - x0[free]))) > 1e-4

    def test_frozen_indices_refused_for_hessian_steppers(self):
        """A zero gradient does not pin an atom when the step comes from a
        Hessian, so those routes must refuse rather than return a wrong
        geometry."""
        pos, z = _alanine()
        ef = _valence(pos, z)
        x0 = jnp.asarray(pos)
        for kw in ({"method": "SP"}, {"init_hessian": "lindh"}):
            with pytest.raises(ValueError, match="frozen_indices is not supported"):
                optimize_single(
                    x0,
                    ef,
                    fmax=1e-2,
                    maxiter=10,
                    atomic_numbers=z,
                    frozen_indices=[0, 1, 2],
                    **kw,
                )

    def test_frozen_indices_mask_covers_the_barrier(self):
        """The mask must wrap energy + barrier: a barrier applied afterwards
        would put a non-zero gradient back onto the frozen atoms."""
        pos, z = _alanine()
        frozen = [0, 1, 2]
        x0 = jnp.asarray(pos)

        def barrier(p):  # pulls every atom towards origin
            return 10.0 * jnp.sum(p**2)

        xo, _e, _i = optimize_single(
            x0,
            _valence(pos, z),
            fmax=1e-2,
            maxiter=50,
            method="LBFGS",
            barrier_fn=barrier,
            atomic_numbers=z,
            frozen_indices=frozen,
        )
        fz = jnp.array(frozen)
        assert float(jnp.max(jnp.abs(xo[fz] - x0[fz]))) < 1e-10
