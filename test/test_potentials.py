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


# ============================================================================
# jax-md model zoo, MACE backends, CLI wiring
# ============================================================================

_WATER_DIMER = jnp.array(
    [
        [0.0, 0.0, 0.0],
        [0.757, 0.586, 0.0],
        [-0.757, 0.586, 0.0],
        [0.0, 0.0, 2.9],
        [0.757, 0.586, 2.9],
        [-0.757, 0.586, 2.9],
    ]
)
_WATER_Z = [8, 1, 1, 8, 1, 1]


class TestJaxMDRegistry:
    @pytest.mark.parametrize(
        "name",
        [
            "uma",
            "orb",
            "ani",
            "aimnet2",
            "aceff",
            "so3lr_jaxmd",
            "mm",
            "amber",
            "charmm",
            "opls",
            "amoeba",
        ],
    )
    def test_registered(self, name):
        assert name in list_potentials()

    def test_ani_rejects_unsupported_elements(self):
        with pytest.raises(ValueError, match="does not support element"):
            get_potential("ani", species=[26, 1])

    def test_ani_rejects_charge(self):
        with pytest.raises(ValueError, match="no charge input"):
            get_potential("ani", species=[8, 1, 1], charge=-1)

    def test_uma_rejects_unknown_task(self):
        with pytest.raises(ValueError, match="Unknown UMA task"):
            get_potential("uma", species=[8, 1, 1], task="nonsense")

    def test_mace_rejects_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown MACE backend"):
            get_potential("mace", species=[8, 1, 1], backend="nonsense")


def _jaxmd_model_available(module, checkpoint):
    import importlib
    from pathlib import Path

    try:
        importlib.import_module(module)
    except Exception:
        return False
    return (Path.home() / ".cache" / "jax_md" / checkpoint).is_file()


@pytest.mark.requires_jaxmd_ml
@pytest.mark.parametrize(
    "name, module, checkpoint",
    [
        ("ani", "jax_md._nn.ani", "ani2x_ensemble.eqx"),
        ("aimnet2", "jax_md._nn.aimnet2", "aimnet2.eqx"),
        ("aceff", "jax_md._nn.aceff", "aceff_v2.0.eqx"),
        ("so3lr_jaxmd", "jax_md._nn.so3lr", "so3lr.eqx"),
    ],
)
def test_jaxmd_model_contract(name, module, checkpoint):
    """Energy, gradient, jit and neighbor updates through the MARS interface."""
    if not _jaxmd_model_available(module, checkpoint):
        pytest.skip(f"{module} or its checkpoint is not available")
    jax.config.update("jax_enable_x64", True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pot = get_potential(name, species=_WATER_Z)
    pot.initialize(_WATER_DIMER)
    f = pot.build_energy_fn()
    nb = pot.allocate_neighbors(_WATER_DIMER)
    kw = pot.get_neighbor_kwargs(nb)
    e = jax.jit(f)(_WATER_DIMER, **kw)
    g = jax.jit(jax.grad(f))(_WATER_DIMER, **kw)
    assert np.isfinite(float(e))
    assert np.all(np.isfinite(np.asarray(g)))
    # Translation invariance and a neighbor update that keeps the energy.
    nb2 = pot.update_neighbors(_WATER_DIMER + 1.0, nb)
    e2 = f(_WATER_DIMER + 1.0, **pot.get_neighbor_kwargs(nb2))
    assert float(e2) == pytest.approx(float(e), abs=1e-6)
    overflow = nb2[0].did_buffer_overflow if isinstance(nb2, tuple) else nb2.did_buffer_overflow
    assert not bool(overflow)


class TestCLIPotentialSpec:
    def _args(self, **kw):
        from argparse import Namespace

        base = dict(
            potential="so3lr",
            charge=0.0,
            spin=1.0,
            float64=False,
            so3lr_backend="so3lr",
            mace_backend="auto",
            uma_model="uma-s-1p2",
            uma_task="omol",
            ml_model=None,
            prmtop=None,
            psf=None,
            ff_params=None,
            top=None,
            gmx_include=None,
            ff_xml=None,
            ff_pdb=None,
            ff_coords=None,
            amoeba_polarization="mutual",
            tinker_key=None,
            input="mol.xyz",
        )
        base.update(kw)
        return Namespace(**base)

    def test_default_so3lr_is_not_rerouted(self):
        from mars.cli._common import extra_potential_spec

        assert extra_potential_spec(self._args()) is None

    def test_so3lr_jaxmd_backend(self):
        from mars.cli._common import extra_potential_spec

        name, kw = extra_potential_spec(self._args(so3lr_backend="jaxmd", charge=-1.0))
        assert name == "so3lr_jaxmd" and kw["charge"] == -1.0

    def test_ml_model_path_vs_name(self):
        from mars.cli._common import extra_potential_spec

        _, kw = extra_potential_spec(self._args(potential="aceff", ml_model="aceff-jax-1.1"))
        assert kw["model"] == "aceff-jax-1.1"
        _, kw = extra_potential_spec(self._args(potential="orb", ml_model="/x/orb.eqx"))
        assert kw["model_path"] == "/x/orb.eqx"

    def test_mm_files_and_pdb_input(self):
        from mars.cli._common import extra_potential_spec

        name, kw = extra_potential_spec(
            self._args(potential="amoeba", input="w.pdb", amoeba_polarization="direct")
        )
        assert name == "amoeba" and kw["pdb"] == "w.pdb" and kw["polarization"] == "direct"
        name, kw = extra_potential_spec(self._args(potential="mm", prmtop="a.prmtop"))
        assert kw["family"] == "amber" and kw["prmtop"] == "a.prmtop"


def test_so3lr_jaxmd_is_closed_shell():
    """jax-md ships SO3LR v1, whose spin input was never trained."""
    with pytest.raises(ValueError, match="closed-shell only"):
        get_potential("so3lr_jaxmd", species=[6, 1, 1, 1], spin=2)


@pytest.mark.requires_so3lr
def test_so3lr_v1_refuses_spin():
    if not _so3lr_available():
        pytest.skip("so3lr not installed")
    with pytest.raises(ValueError, match="closed-shell only"):
        get_potential("so3lr", species=[6, 1, 1, 1], model="so3lr-1", spin=2)


class TestModelInputValidation:
    """Elements, charge and spin are checked against what each model supports."""

    @pytest.mark.parametrize(
        "name, kwargs, match",
        [
            ("ani", dict(species=[26, 1]), "does not support element"),
            ("ani", dict(species=[8, 1], charge=-1), "no charge input"),
            ("aimnet2", dict(species=[26, 8]), "does not support element"),
            ("aimnet2", dict(species=[6, 1, 1, 1], spin=2), "closed-shell only"),
            ("aceff", dict(species=[11, 17]), "does not support element"),
            ("orb", dict(species=[92, 8, 8]), "does not support element"),
            ("orb", dict(species=[6, 1, 1, 1]), "inconsistent with multiplicity"),
            ("uma", dict(species=[6, 1, 1, 1], spin=2, task="omat"), "closed-shell only"),
            ("uma", dict(species=[92, 8], task="omol"), "does not support element"),
            ("so3lr_jaxmd", dict(species=[35, 1]), "does not support element"),
        ],
    )
    def test_rejected(self, name, kwargs, match):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(ValueError, match=match):
                get_potential(name, **kwargs)

    @pytest.mark.parametrize(
        "name, kwargs",
        [
            ("aimnet2", dict(species=[8, 1], charge=-1)),
            ("orb", dict(species=[6, 1, 1, 1], spin=2)),
            ("uma", dict(species=[6, 1, 1, 1], spin=2, task="omol")),
            ("uma", dict(species=[92, 8], task="omat")),
        ],
    )
    def test_accepted(self, name, kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            get_potential(name, **kwargs)

    def test_validator_messages(self):
        from mars.potentials import validate_model_inputs

        with pytest.raises(ValueError, match=r"Supported: H \(Z=1\)"):
            validate_model_inputs("X", [1, 26], elements={1})
        validate_model_inputs("X", [8, 1], charge=-1, elements={1, 8})


@pytest.mark.requires_so3lr
def test_so3lr_v2_elements_follow_qcml():
    if not _so3lr_supports_model():
        pytest.skip("so3lr v2 models not installed")
    with pytest.raises(ValueError, match="does not support element"):
        get_potential("so3lr", species=[56, 8], model="so3lr-2-m")
    get_potential("so3lr", species=[26, 8], model="so3lr-2-m")


@pytest.mark.parametrize("name", ["aimnet2", "aceff", "so3lr_jaxmd"])
def test_jaxmd_models_with_charges_declare_a_charge_tap(name):
    """These models feed IR intensities; losing the charge tap would silently drop them."""
    from mars.potentials import _POTENTIAL_REGISTRY

    assert _POTENTIAL_REGISTRY[name].CHARGE_TAP is not None
