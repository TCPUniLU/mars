"""
Tests for mars.sampling — MTD and rotamer MD integration.
Uses the harmonic centroid potential (no ML dependencies).
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from mars.sampling import run_mtd_jax, run_rotamer_md_jax
from mars.potentials import PotentialWrapper

# ── helpers ─────────────────────────────────────────────────────────────────


def harmonic(positions, **kwargs):
    """Harmonic centroid potential. Accepts **kwargs so JAX-MD can pass kT etc."""
    center = jnp.mean(positions, axis=0)
    return jnp.sum((positions - center) ** 2) * 0.1


def _triangle():
    return jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.75, 1.3, 0.0],
        ],
        dtype=jnp.float64,
    )


MTD_PARAMS = {"kpush": 0.02, "alpha": 0.5, "cvdump_fs": 10.0}


# ── minimal potential wrapper for tests ─────────────────────────────────────
# The sampling loop checks:
#   if isinstance(nbr_state, tuple):   ← SO3LR dual-nbr path
#       overflow = nbr_state[0].did_buffer_overflow ...
#   else:                              ← single-nbr path (what we want)
#       overflow = nbr_state.did_buffer_overflow
#
# namedtuple IS a tuple subclass → triggers the wrong branch.
# Use a plain class registered as a JAX pytree so it threads through
# fori_loop while NOT being instanceof tuple.


class _DummyNbr:
    """Single-nbr dummy state compatible with sampling loop's overflow check."""

    def __init__(self, did_buffer_overflow=None):
        self.did_buffer_overflow = (
            did_buffer_overflow if did_buffer_overflow is not None else jnp.array(False)
        )


jax.tree_util.register_pytree_node(
    _DummyNbr,
    lambda s: ([s.did_buffer_overflow], None),
    lambda aux, children: _DummyNbr(children[0]),
)


class _HarmonicWrapper(PotentialWrapper):
    """Minimal PotentialWrapper backed by the harmonic energy function.

    Provides a JAX-pytree-compatible neighbor state required by the sampling
    loop (``did_buffer_overflow`` attribute), and passes **kwargs through so
    JAX-MD can supply extra arguments (e.g. ``kT``) to the energy function.
    """

    def allocate_neighbors(self, positions):
        return _DummyNbr(did_buffer_overflow=jnp.array(False))

    def update_neighbors(self, positions, nbr_state):
        return nbr_state

    def get_neighbor_kwargs(self, nbr_state):
        return {}

    def _build_energy_fn(self):
        return harmonic


def _make_wrapper(positions):
    pw = _HarmonicWrapper()
    pw.initialize(positions)
    return pw


# ── run_mtd_jax ──────────────────────────────────────────────────────────────


class TestRunMtdJax:
    def test_returns_trajectory_and_state(self):
        pos = _triangle()
        pw = _make_wrapper(pos)
        traj, state = run_mtd_jax(
            pos,
            harmonic,
            MTD_PARAMS,
            T=300,
            dt=1.0,
            time_ps=0.1,
            random_seed=0,
            potential_wrapper=pw,
        )
        assert isinstance(traj, list)
        assert hasattr(state, "n_hills")

    def test_trajectory_nonempty(self):
        pos = _triangle()
        pw = _make_wrapper(pos)
        traj, _ = run_mtd_jax(
            pos,
            harmonic,
            MTD_PARAMS,
            T=300,
            dt=1.0,
            time_ps=0.1,
            random_seed=0,
            potential_wrapper=pw,
        )
        assert len(traj) > 0

    def test_trajectory_frame_shape(self):
        pos = _triangle()
        pw = _make_wrapper(pos)
        traj, _ = run_mtd_jax(
            pos,
            harmonic,
            MTD_PARAMS,
            T=300,
            dt=1.0,
            time_ps=0.1,
            random_seed=0,
            potential_wrapper=pw,
        )
        for frame in traj:
            assert frame.shape == pos.shape

    def test_hills_deposited(self):
        pos = _triangle()
        pw = _make_wrapper(pos)
        _, state = run_mtd_jax(
            pos,
            harmonic,
            MTD_PARAMS,
            T=300,
            dt=1.0,
            time_ps=0.2,
            random_seed=0,
            potential_wrapper=pw,
        )
        assert int(state.n_hills) > 0

    def test_bias_grows_during_run(self):
        """More hills → higher hill count after longer run."""
        pos = _triangle()
        _, state_short = run_mtd_jax(
            pos,
            harmonic,
            MTD_PARAMS,
            T=300,
            dt=1.0,
            time_ps=0.1,
            random_seed=0,
            potential_wrapper=_make_wrapper(pos),
        )
        _, state_long = run_mtd_jax(
            pos,
            harmonic,
            MTD_PARAMS,
            T=300,
            dt=1.0,
            time_ps=0.5,
            random_seed=0,
            potential_wrapper=_make_wrapper(pos),
        )
        assert int(state_long.n_hills) >= int(state_short.n_hills)

    def test_reproducible_with_seed(self):
        pos = _triangle()
        traj1, _ = run_mtd_jax(
            pos,
            harmonic,
            MTD_PARAMS,
            T=300,
            dt=1.0,
            time_ps=0.1,
            random_seed=7,
            potential_wrapper=_make_wrapper(pos),
        )
        traj2, _ = run_mtd_jax(
            pos,
            harmonic,
            MTD_PARAMS,
            T=300,
            dt=1.0,
            time_ps=0.1,
            random_seed=7,
            potential_wrapper=_make_wrapper(pos),
        )
        assert jnp.allclose(traj1[-1], traj2[-1], atol=1e-5)

    def test_different_seeds_diverge(self):
        pos = _triangle()
        traj1, _ = run_mtd_jax(
            pos,
            harmonic,
            MTD_PARAMS,
            T=300,
            dt=1.0,
            time_ps=0.5,
            random_seed=1,
            potential_wrapper=_make_wrapper(pos),
        )
        traj2, _ = run_mtd_jax(
            pos,
            harmonic,
            MTD_PARAMS,
            T=300,
            dt=1.0,
            time_ps=0.5,
            random_seed=2,
            potential_wrapper=_make_wrapper(pos),
        )
        assert not jnp.allclose(traj1[-1], traj2[-1], atol=1e-6)

    def test_positions_are_finite(self):
        pos = _triangle()
        pw = _make_wrapper(pos)
        traj, _ = run_mtd_jax(
            pos,
            harmonic,
            MTD_PARAMS,
            T=300,
            dt=1.0,
            time_ps=0.2,
            random_seed=0,
            potential_wrapper=pw,
        )
        for frame in traj:
            assert jnp.all(jnp.isfinite(frame)), "Non-finite positions in trajectory"

    def test_custom_masses(self):
        pos = _triangle()
        pw = _make_wrapper(pos)
        masses = jnp.array([12.0, 12.0, 12.0], dtype=jnp.float64)
        traj, _ = run_mtd_jax(
            pos,
            harmonic,
            MTD_PARAMS,
            T=300,
            dt=1.0,
            time_ps=0.1,
            mass=masses,
            random_seed=0,
            potential_wrapper=pw,
        )
        assert len(traj) > 0


# ── run_rotamer_md_jax ───────────────────────────────────────────────────────


class TestRunRotamerMdJax:
    def test_returns_list(self):
        pos = _triangle()
        pw = _make_wrapper(pos)
        traj = run_rotamer_md_jax(
            pos,
            harmonic,
            T=300,
            dt=1.0,
            time_ps=0.1,
            save_interval=5,
            random_seed=0,
            potential_wrapper=pw,
        )
        assert isinstance(traj, list)

    def test_trajectory_nonempty(self):
        pos = _triangle()
        pw = _make_wrapper(pos)
        traj = run_rotamer_md_jax(
            pos,
            harmonic,
            T=300,
            dt=1.0,
            time_ps=0.1,
            save_interval=5,
            random_seed=0,
            potential_wrapper=pw,
        )
        assert len(traj) > 0

    def test_frame_shape(self):
        pos = _triangle()
        pw = _make_wrapper(pos)
        traj = run_rotamer_md_jax(
            pos,
            harmonic,
            T=300,
            dt=1.0,
            time_ps=0.1,
            save_interval=5,
            random_seed=0,
            potential_wrapper=pw,
        )
        for frame in traj:
            assert frame.shape == pos.shape

    def test_all_frames_finite(self):
        pos = _triangle()
        pw = _make_wrapper(pos)
        traj = run_rotamer_md_jax(
            pos,
            harmonic,
            T=300,
            dt=1.0,
            time_ps=0.2,
            save_interval=5,
            random_seed=0,
            potential_wrapper=pw,
        )
        for frame in traj:
            assert jnp.all(jnp.isfinite(frame))

    def test_higher_temperature_more_displacement(self):
        """Higher T → larger thermal displacements on average."""
        pos = _triangle()
        n_trials = 3
        disp_low, disp_high = 0.0, 0.0
        for seed in range(n_trials):
            traj_low = run_rotamer_md_jax(
                pos,
                harmonic,
                T=100,
                dt=0.5,
                time_ps=0.3,
                save_interval=5,
                random_seed=seed,
                potential_wrapper=_make_wrapper(pos),
            )
            traj_high = run_rotamer_md_jax(
                pos,
                harmonic,
                T=1000,
                dt=0.5,
                time_ps=0.3,
                save_interval=5,
                random_seed=seed,
                potential_wrapper=_make_wrapper(pos),
            )
            if traj_low and traj_high:
                disp_low += float(jnp.mean(jnp.abs(jnp.array(traj_low) - pos)))
                disp_high += float(jnp.mean(jnp.abs(jnp.array(traj_high) - pos)))
        assert disp_high >= disp_low - 1e-3  # allow tiny tolerance
