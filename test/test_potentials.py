"""
Tests for mars.potentials — potential registry and built-in potentials.
"""

import warnings

import pytest
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from mars.potentials import (
    PotentialWrapper,
    get_potential,
    list_potentials,
    register_potential,
)

# ── registry ─────────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_list_potentials_nonempty(self):
        pots = list_potentials()
        assert len(pots) > 0

    def test_harmonic_registered(self):
        assert "harmonic" in list_potentials()

    def test_lj_registered(self):
        assert "lj" in list_potentials()

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown potential"):
            get_potential("__nonexistent_potential__")

    def test_register_custom(self):
        @register_potential("_test_custom_pot")
        class _TestPot(PotentialWrapper):
            def __init__(self, **kwargs):
                pass

            def _build_energy_fn(self):
                return lambda pos, **kw: jnp.sum(pos**2)

        assert "_test_custom_pot" in list_potentials()
        pot = get_potential("_test_custom_pot")
        energy_fn = pot.build_energy_fn()
        pos = jnp.ones((3, 3))
        assert float(energy_fn(pos)) == pytest.approx(9.0)


# ── HarmonicPotential ────────────────────────────────────────────────────────────


class TestHarmonicPotential:
    """HarmonicPotential requires initialize(positions) before build_energy_fn()."""

    @pytest.fixture
    def positions(self):
        return jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [0.75, 1.3, 0.0],
            ],
            dtype=jnp.float64,
        )

    @pytest.fixture
    def pot(self, positions):
        p = get_potential("harmonic")
        p.initialize(positions)  # required before build_energy_fn
        return p

    def test_energy_scalar(self, pot, positions):
        fn = pot.build_energy_fn()
        E = fn(positions)
        assert E.shape == ()

    def test_energy_real(self, pot, positions):
        fn = pot.build_energy_fn()
        E = fn(positions)
        assert not jnp.isnan(E)
        assert not jnp.isinf(E)

    def test_energy_nonnegative(self, pot, positions):
        fn = pot.build_energy_fn()
        assert float(fn(positions)) >= 0.0

    def test_minimum_at_centroid(self):
        """The harmonic centroid potential has minimum 0 when all atoms coincide."""
        coincident = jnp.zeros((4, 3), dtype=jnp.float64)
        p = get_potential("harmonic")
        p.initialize(coincident)
        fn = p.build_energy_fn()
        assert float(fn(coincident)) == pytest.approx(0.0, abs=1e-10)

    def test_gradient_not_zero(self, pot, positions):
        fn = pot.build_energy_fn()
        grad = jax.grad(fn)(positions)
        assert jnp.max(jnp.abs(grad)) > 1e-8

    def test_gradient_matches_finite_diff(self, pot, positions):
        fn = pot.build_energy_fn()
        grad_auto = jax.grad(fn)(positions)
        eps = 1e-5
        grad_fd = jnp.zeros_like(positions)
        for i in range(positions.shape[0]):
            for j in range(3):
                p_p = positions.at[i, j].add(eps)
                p_m = positions.at[i, j].add(-eps)
                grad_fd = grad_fd.at[i, j].set((fn(p_p) - fn(p_m)) / (2 * eps))
        assert jnp.allclose(grad_auto, grad_fd, atol=1e-4)

    def test_jit_compilable(self, pot, positions):
        fn = jax.jit(pot.build_energy_fn())
        E = fn(positions)
        assert not jnp.isnan(E)

    def test_build_energy_fn_with_charges_warns(self, pot):
        with pytest.warns(UserWarning, match="does not support partial charges"):
            energy_fn, charges_fn = pot.build_energy_fn(with_charges=True)
        assert charges_fn is None
        assert callable(energy_fn)

    def test_raises_without_initialize(self):
        """Calling build_energy_fn before initialize should raise RuntimeError."""
        p = get_potential("harmonic")
        with pytest.raises(RuntimeError, match="not initialized"):
            p.build_energy_fn()

    def test_uses_neighbor_lists_bool(self, pot):
        assert isinstance(pot.uses_neighbor_lists, bool)


# ── LJ Potential ──────────────────────────────────────────────────────────────────


def _lj_available():
    """Check if LJPotential can be initialized in this JAX-MD version."""
    try:
        pos = jnp.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=jnp.float32)
        p = get_potential("lj")
        p.initialize(pos)
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _lj_available(),
    reason="LJPotential incompatible with installed jax-md version (box=None unsupported)",
)
class TestLJPotential:
    """LJPotential also requires initialize(positions) before build_energy_fn()."""

    @pytest.fixture
    def dimer(self):
        """Two atoms at LJ σ distance."""
        return jnp.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=jnp.float64)

    @pytest.fixture
    def pot(self, dimer):
        p = get_potential("lj")
        p.initialize(dimer)
        return p

    def test_energy_scalar(self, pot, dimer):
        fn = pot.build_energy_fn()
        assert fn(dimer).shape == ()

    def test_energy_finite(self, pot, dimer):
        fn = pot.build_energy_fn()
        E = fn(dimer)
        assert not jnp.isnan(E)
        assert not jnp.isinf(E)

    def test_gradient_exists(self, pot, dimer):
        fn = pot.build_energy_fn()
        grad = jax.grad(fn)(dimer)
        assert grad.shape == dimer.shape

    def test_raises_without_initialize(self):
        p = get_potential("lj")
        with pytest.raises(RuntimeError, match="not initialized"):
            p.build_energy_fn()

    def test_energy_increases_at_close_contact(self, dimer):
        """LJ energy must be larger (repulsive) at close contact than at equilibrium."""
        p_close = get_potential("lj")
        close = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=jnp.float64)
        p_close.initialize(close)
        fn_close = p_close.build_energy_fn()

        p_far = get_potential("lj")
        far = jnp.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=jnp.float64)
        p_far.initialize(far)
        fn_far = p_far.build_energy_fn()

        assert float(fn_close(close)) > float(fn_far(far))


# ── SO3LR Potential (stable + developing packages) ─────────────────────────────────


def _so3lr_available():
    try:
        import so3lr  # noqa: F401

        return True
    except Exception:
        return False


def _so3lr_supports_model():
    """True when the installed SO3LR exposes v2 model selection (so3lr_dev)."""
    import inspect

    from so3lr import So3lrPotential

    params = inspect.signature(So3lrPotential).parameters
    return "model" in params or "workdir" in params


@pytest.mark.requires_so3lr
@pytest.mark.skipif(not _so3lr_available(), reason="SO3LR package not installed")
class TestSO3LRVersionAdaptive:
    """SO3LR must work against BOTH the stable and developing packages.

    The loader introspects the installed SO3LR API; these tests assert the
    default (v1) path works everywhere and that v2 model selection either
    loads (developing package) or falls back with a warning (stable package).
    """

    @pytest.fixture
    def water(self):
        return np.array([[0.0, 0.0, 0.0], [0.76, 0.59, 0.0], [-0.76, 0.59, 0.0]], dtype=float)

    @pytest.fixture
    def numbers(self):
        return np.array([8, 1, 1])

    def _energy(self, pot, positions):
        pot.initialize(jnp.asarray(positions))
        return float(pot.build_energy_fn()(jnp.asarray(positions)))

    def test_default_model_builds_and_evaluates(self, numbers, water):
        """The default v1 model must build and give a finite energy in either package."""
        pot = get_potential("so3lr", species=numbers, charge=0.0)
        e = self._energy(pot, water)
        assert np.isfinite(e)

    def test_model_selection_adapts_to_package(self, numbers, water):
        """Requesting a v2 model builds it on the dev package; on stable it warns
        and falls back to v1 (never crashes)."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pot = get_potential("so3lr", species=numbers, charge=0.0, model="so3lr-2-m")
            e = self._energy(pot, water)
        assert np.isfinite(e)

        model_warnings = [w for w in caught if "does not support model selection" in str(w.message)]
        if _so3lr_supports_model():
            # developing package: v2 model loaded, no fallback warning
            assert not model_warnings
        else:
            # stable package: model selection ignored, warned, v1 used
            assert model_warnings

    @pytest.mark.parametrize(
        "deprecated,current",
        [
            ("so3lr_v1", "so3lr-1"),
            ("so3lr", "so3lr-1"),
            ("so3lr-s", "so3lr-2-s"),
            ("so3lr-m", "so3lr-2-m"),
            ("so3lr-l", "so3lr-2-l"),
        ],
    )
    def test_deprecated_model_names_still_work(self, numbers, water, deprecated, current):
        """Pre-release names resolve to the same model as their current-name
        equivalent and still work, but raise DeprecationWarning; the current
        names raise nothing."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pot_dep = get_potential("so3lr", species=numbers, charge=0.0, model=deprecated)
            e_dep = self._energy(pot_dep, water)
        assert np.isfinite(e_dep)
        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert dep_warnings, f"{deprecated!r} should raise a DeprecationWarning"

        with warnings.catch_warnings(record=True) as caught2:
            warnings.simplefilter("always")
            pot_cur = get_potential("so3lr", species=numbers, charge=0.0, model=current)
            e_cur = self._energy(pot_cur, water)
        cur_dep_warnings = [w for w in caught2 if issubclass(w.category, DeprecationWarning)]
        assert not cur_dep_warnings, f"{current!r} should not raise DeprecationWarning"

        assert e_dep == pytest.approx(
            e_cur
        ), f"{deprecated!r} and {current!r} should resolve to the identical model"
