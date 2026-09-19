"""JAX-based structure optimization.

Provides geometry optimization (minimization and transition state search)
with JIT-compiled implementations for GPU acceleration.

- Single structure: LBFGS (default) via jaxopt, or FIRE via jax-md, GD via jaxopt
- Multiple structures: parallel via jax.vmap over LBFGS (or FIRE)
- Stationary point search: damped Newton eigenvector-following (JIT-compiled)
- All convergence criteria use max force (eV/Å) with default 0.01
"""

from typing import Callable, Dict, List, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space as jmd_space
from jax_md.minimize import fire_descent
from jaxopt import LBFGS, GradientDescent

from .auto_config import EV_TO_KCALMOL
from .internal_coords import build_coordinates
from .log import log_message, log_step_end, log_step_start
from .utils import _is_jax_oom

# ============================================================================
# Optimization Levels
# ============================================================================

FMAX_MAP = {
    -3: 0.05,  # Very loose
    -2: 0.03,  # Loose
    -1: 0.02,  # Normal loose
    0: 0.01,  # Default
    1: 0.005,  # Tight
    2: 0.002,  # Very tight
}


# ============================================================================
# Internal JIT Helpers
# ============================================================================


@jax.jit
def _compute_max_force(forces: jnp.ndarray) -> float:
    return jnp.max(jnp.abs(forces))


def _clip_atom_displacement(prev_pos: jnp.ndarray, new_pos: jnp.ndarray, max_disp: float):
    # Cap LBFGS step in physical Å units: prevents the first iteration's
    # raw-gradient direction from dragging an atom several Å on a clash.
    # Returns (clipped_pos, clipped_flag). Caller resets LBFGS state when the
    # flag is True so the optimizer's history/error stay consistent with the
    # actually-taken step.
    delta = new_pos - prev_pos
    max_norm = jnp.max(jnp.linalg.norm(delta, axis=-1))
    clipped = max_norm > max_disp
    scale = jnp.where(clipped, max_disp / (max_norm + 1e-12), 1.0)
    return prev_pos + delta * scale, clipped


def _select_state(clipped, fresh_state, kept_state):
    return jax.tree_util.tree_map(lambda a, b: jnp.where(clipped, a, b), fresh_state, kept_state)


# Conservative LBFGS line-search settings. Backtracking (vs jaxopt's default
# zoom) only shrinks from the initial trial step, so it can't accept a far step
# into an atomic clash — robust for large-magnitude / rough ML potentials, and
# neutral for well-behaved ones. The post-update displacement clip is the
# secondary safety net.
_LBFGS_KWARGS = dict(use_gamma=True, increase_factor=1.2, linesearch="backtracking")


def _make_force_and_energy_fn(energy_fn: Callable):
    @jax.jit
    def compute_forces(positions):
        return -jax.grad(energy_fn)(positions)

    @jax.jit
    def compute_energy(positions):
        return energy_fn(positions)

    return compute_forces, compute_energy


def _make_fire_fn(
    energy_fn: Callable,
    fmax: float = 0.01,
    maxiter: int = 1000,
    dt_start: float = 0.05,
    dt_max: float = 0.1,
    n_min: int = 2,
) -> Callable:
    """Build a fully-JIT FIRE minimizer (no neighbor list handling).

    Used by the batch/vmap optimizer for simple potentials without neighbor
    lists, and by ``optimize_single`` for the no-neighbor-list FIRE path.

    Returns ``pos_init -> (pos_opt, n_steps, converged)``.
    """
    _, shift_fn = jmd_space.free()
    fire_init, fire_apply = fire_descent(
        energy_fn, shift_fn, dt_start=dt_start, dt_max=dt_max, n_min=n_min
    )

    def run_fire(pos_init):
        state0 = fire_init(pos_init)
        max_f0 = jnp.max(jnp.abs(state0.force))

        def cond_fn(carry):
            _, step, max_f = carry
            return jnp.logical_and(step < maxiter, max_f >= fmax)

        def body_fn(carry):
            state, step, _ = carry
            new_state = fire_apply(state)
            new_max_f = jnp.max(jnp.abs(new_state.force))
            return new_state, step + 1, new_max_f

        final_state, n_steps, final_max_f = jax.lax.while_loop(
            cond_fn, body_fn, (state0, jnp.zeros((), dtype=jnp.int32), max_f0)
        )

        return final_state.position, n_steps, final_max_f < fmax

    return run_fire


def _safe_energy(energy_fn: Callable, large_val: float = 1e4) -> Callable:
    """Wrap energy_fn so NaN/Inf returns a large finite value.

    This prevents jaxopt's internal lax.while_loop from spinning
    on non-finite energies (which never satisfy the tolerance check).
    kwargs (e.g. neighbor=nbrs.idx) are forwarded to energy_fn unchanged.
    """

    def safe_fn(pos, **kwargs):
        e = energy_fn(pos, **kwargs)
        return jnp.where(jnp.isfinite(e), e, large_val)

    return safe_fn


def _mask_frozen_gradient(energy_fn: Callable, frozen_indices) -> Callable:
    """Wrap *energy_fn* so the atoms in *frozen_indices* carry zero gradient.

    ``jax.lax.stop_gradient`` is applied to those rows before the energy is
    evaluated: the returned value is unchanged, but the force on every frozen
    atom is identically zero.  No reference geometry is captured, so the
    wrapper depends only on *frozen_indices* and can be reused -- and
    JIT-compiled once -- across many geometries.
    """
    idx = jnp.asarray(frozen_indices, dtype=int)

    def frozen_energy_fn(positions, **kwargs):
        pinned = positions.at[idx].set(jax.lax.stop_gradient(positions[idx]))
        return energy_fn(pinned, **kwargs)

    return frozen_energy_fn


#: Cartesian steppers that hold an atom still once its gradient is zero: they
#: only ever move an atom along its own force component.  Hessian-based steps
#: (``"SP"``, and the RFO/BFGS stepper selected by ``init_hessian``) couple
#: coordinates and displace frozen atoms regardless.
_GRADIENT_ONLY_METHODS = ("LBFGS", "FIRE", "GD")


def _freeze_cartesian(energy_fn: Callable, frozen_indices, method: str) -> Callable:
    """Apply frozen-atom constraints to a Cartesian optimization.

    Returns *energy_fn* unchanged when nothing is frozen.  Refuses the
    Hessian-based steppers, where a zero gradient is not enough to hold an atom
    still -- the same reason :func:`ric_gate` refuses frozen atoms for internal
    coordinates.  Failing loudly beats returning a silently wrong geometry.
    """
    if frozen_indices is None or len(frozen_indices) == 0:
        return energy_fn
    if method not in _GRADIENT_ONLY_METHODS:
        raise ValueError(
            f"frozen_indices is not supported with method={method!r}: a "
            "Hessian-based step displaces atoms even when their gradient is "
            f"zero. Use one of {list(_GRADIENT_ONLY_METHODS)}."
        )
    return _mask_frozen_gradient(energy_fn, frozen_indices)


def _make_batch_optimizer(
    energy_fn: Callable,
    fmax: float = 0.01,
    maxiter: int = 100,
    barrier_fn: Callable = None,
    max_stepsize: float = 0.2,
    method: str = "LBFGS",
    potential_wrapper=None,
    fire_dt_start: float = 0.05,
    fire_dt_max: float = 0.1,
    fire_n_min: int = 2,
    frozen_indices=None,
):
    """JIT+vmap optimizer for parallel batch optimization.

    When ``potential_wrapper`` is provided, neighbor lists are updated inside
    the JIT-compiled optimization loop (vmapped across all structures).

    Returns ``f(pos_batch, nbr_batch) -> (pos_opt, energies, converged,
    n_iter, max_force)`` when neighbor lists are used, or
    ``f(pos_batch) -> (...)`` otherwise.  ``n_iter`` and ``max_force`` are
    per-structure so callers can report real convergence data instead of
    guessing (see :func:`optimize_batch_parallel`).
    """
    energy_jit = jax.jit(energy_fn)
    use_nbrs = potential_wrapper is not None and potential_wrapper.uses_neighbor_lists

    # Frozen atoms are masked on the *combined* energy: a barrier added
    # afterwards would put a non-zero gradient back on them.  ``energy_jit``
    # above stays unmasked so the reported energies are the true ones.
    if barrier_fn is not None:

        def _combined(pos, **kw):
            return energy_fn(pos, **kw) + barrier_fn(pos)

    else:
        _combined = energy_fn
    opt_energy = jax.jit(_safe_energy(_freeze_cartesian(_combined, frozen_indices, method)))

    if method == "FIRE":
        if use_nbrs:
            _, shift_fn = jmd_space.free()
            fire_init, fire_apply = fire_descent(
                opt_energy,
                shift_fn,
                dt_start=fire_dt_start,
                dt_max=fire_dt_max,
                n_min=fire_n_min,
            )
            get_nbr_kw = potential_wrapper.get_neighbor_kwargs

            def optimize_one(pos_init, nbr_state):
                nbr_kw = get_nbr_kw(nbr_state)
                state0 = fire_init(pos_init, **nbr_kw)
                max_f0 = jnp.max(jnp.abs(state0.force))

                def cond_fn(carry):
                    _, step, max_f, _ = carry
                    return jnp.logical_and(step < maxiter, max_f >= fmax)

                def body_fn(carry):
                    state, step, _, nbs = carry
                    kw = get_nbr_kw(nbs)
                    new_state = fire_apply(state, **kw)
                    nbs = potential_wrapper.update_neighbors(new_state.position, nbs)
                    new_max_f = jnp.max(jnp.abs(new_state.force))
                    return new_state, step + 1, new_max_f, nbs

                final_state, steps, final_max_f, final_nbs = jax.lax.while_loop(
                    cond_fn, body_fn, (state0, jnp.zeros((), dtype=jnp.int32), max_f0, nbr_state)
                )

                pos_opt = final_state.position
                final_kw = get_nbr_kw(final_nbs)
                return (
                    pos_opt,
                    energy_jit(pos_opt, **final_kw),
                    final_max_f < fmax,
                    steps,
                    final_max_f,
                )

            @jax.jit
            def optimize_batch(pos_batch, nbr_batch):
                return jax.vmap(optimize_one)(pos_batch, nbr_batch)

            return optimize_batch

        else:
            fire_fn = _make_fire_fn(
                opt_energy,
                fmax=fmax,
                maxiter=maxiter,
                dt_start=fire_dt_start,
                dt_max=fire_dt_max,
                n_min=fire_n_min,
            )

            @jax.jit
            def optimize_one(pos_init):
                pos_opt, steps, converged = fire_fn(pos_init)
                max_f = _compute_max_force(-jax.grad(opt_energy)(pos_opt))
                return pos_opt, energy_jit(pos_opt), converged, steps, max_f

            return jax.jit(jax.vmap(optimize_one))

    elif method == "GD":
        if use_nbrs:
            get_nbr_kw = potential_wrapper.get_neighbor_kwargs
            optimizer = GradientDescent(fun=opt_energy, tol=fmax, stepsize=max_stepsize)

            def optimize_one(pos_init, nbr_state):
                nbr_kw = get_nbr_kw(nbr_state)
                state = optimizer.init_state(pos_init, **nbr_kw)

                def cond_fn(carry):
                    _, state, step, _ = carry
                    return jnp.logical_and(step < maxiter, state.error >= fmax)

                def body_fn(carry):
                    pos, state, step, nbs = carry
                    kw = get_nbr_kw(nbs)
                    pos, state = optimizer.update(pos, state, **kw)
                    nbs = potential_wrapper.update_neighbors(pos, nbs)
                    return pos, state, step + 1, nbs

                pos, state, steps, final_nbs = jax.lax.while_loop(
                    cond_fn, body_fn, (pos_init, state, jnp.zeros((), dtype=jnp.int32), nbr_state)
                )
                final_kw = get_nbr_kw(final_nbs)
                converged = state.error < fmax
                max_f = _compute_max_force(-jax.grad(opt_energy)(pos, **final_kw))
                return pos, energy_jit(pos, **final_kw), converged, steps, max_f

            @jax.jit
            def optimize_batch(pos_batch, nbr_batch):
                return jax.vmap(optimize_one)(pos_batch, nbr_batch)

            return optimize_batch

        else:

            @jax.jit
            def optimize_one(pos_init):
                opt = GradientDescent(
                    fun=opt_energy, maxiter=maxiter, tol=fmax, stepsize=max_stepsize
                )
                result = opt.run(pos_init)
                converged = result.state.error < fmax
                max_f = _compute_max_force(-jax.grad(opt_energy)(result.params))
                return (
                    result.params,
                    energy_jit(result.params),
                    converged,
                    result.state.iter_num,
                    max_f,
                )

            return jax.jit(jax.vmap(optimize_one))

    elif method == "SP":
        sp_fn = _make_stationary_search_fn(
            opt_energy, fmax=fmax, maxiter=maxiter, trust_radius=max_stepsize
        )

        @jax.jit
        def optimize_one(pos_init):
            pos_opt, _, n_steps, converged = sp_fn(pos_init)
            max_f = _compute_max_force(-jax.grad(opt_energy)(pos_opt))
            return pos_opt, energy_jit(pos_opt), converged, n_steps, max_f

        return jax.jit(jax.vmap(optimize_one))

    else:  # LBFGS
        if use_nbrs:
            get_nbr_kw = potential_wrapper.get_neighbor_kwargs
            optimizer = LBFGS(fun=opt_energy, tol=fmax, max_stepsize=max_stepsize, **_LBFGS_KWARGS)

            def optimize_one(pos_init, nbr_state):
                nbr_kw = get_nbr_kw(nbr_state)
                state = optimizer.init_state(pos_init, **nbr_kw)

                def cond_fn(carry):
                    _, state, step, _ = carry
                    return jnp.logical_and(step < maxiter, state.error >= fmax)

                def body_fn(carry):
                    pos, state, step, nbs = carry
                    kw = get_nbr_kw(nbs)
                    new_pos, new_state = optimizer.update(pos, state, **kw)
                    pos, clipped = _clip_atom_displacement(pos, new_pos, max_stepsize)
                    nbs = potential_wrapper.update_neighbors(pos, nbs)
                    fresh_kw = get_nbr_kw(nbs)
                    fresh_state = optimizer.init_state(pos, **fresh_kw)
                    state = _select_state(clipped, fresh_state, new_state)
                    return pos, state, step + 1, nbs

                pos, state, steps, final_nbs = jax.lax.while_loop(
                    cond_fn, body_fn, (pos_init, state, jnp.zeros((), dtype=jnp.int32), nbr_state)
                )
                final_kw = get_nbr_kw(final_nbs)
                converged = state.error < fmax
                max_f = _compute_max_force(-jax.grad(opt_energy)(pos, **final_kw))
                return pos, energy_jit(pos, **final_kw), converged, steps, max_f

            @jax.jit
            def optimize_batch(pos_batch, nbr_batch):
                return jax.vmap(optimize_one)(pos_batch, nbr_batch)

            return optimize_batch

        else:
            optimizer = LBFGS(fun=opt_energy, tol=fmax, max_stepsize=max_stepsize, **_LBFGS_KWARGS)

            @jax.jit
            def optimize_one(pos_init):
                state = optimizer.init_state(pos_init)

                def cond_fn(carry):
                    _, state, step = carry
                    return jnp.logical_and(step < maxiter, state.error >= fmax)

                def body_fn(carry):
                    pos, state, step = carry
                    new_pos, new_state = optimizer.update(pos, state)
                    pos, clipped = _clip_atom_displacement(pos, new_pos, max_stepsize)
                    fresh_state = optimizer.init_state(pos)
                    state = _select_state(clipped, fresh_state, new_state)
                    return pos, state, step + 1

                pos, state, steps = jax.lax.while_loop(
                    cond_fn, body_fn, (pos_init, state, jnp.zeros((), dtype=jnp.int32))
                )
                converged = state.error < fmax
                max_f = _compute_max_force(-jax.grad(opt_energy)(pos))
                return pos, energy_jit(pos), converged, steps, max_f

            return jax.jit(jax.vmap(optimize_one))


# ============================================================================
# Single Structure Minimization
# ============================================================================


def optimize_single(
    positions_init: jnp.ndarray,
    energy_fn: Callable,
    fmax: float = 0.01,
    maxiter: int = 1000,
    method: str = "LBFGS",
    barrier_fn: Callable = None,
    max_stepsize: float = 0.2,
    potential_wrapper=None,
    fire_dt_start: float = 0.05,
    fire_dt_max: float = 0.1,
    fire_n_min: int = 2,
    *,
    coords: str = "cartesian",
    atomic_numbers=None,
    spec=None,
    init_hessian: str = None,
    interfragment: str = "tric",
    frozen_indices=None,
    ric_options: Dict = None,
) -> Tuple[jnp.ndarray, float, Dict]:
    """Minimize a single structure.

    Picks between LBFGS (default, via jaxopt), FIRE (via jax-md),
    plain gradient descent, or a damped-Newton stationary-point search.
    When ``potential_wrapper`` is supplied, the whole minimization runs in a
    single on-device ``jax.lax.while_loop`` with the neighbor list updated
    inside the loop (no per-step host sync), mirroring
    :func:`_make_batch_optimizer`.

    Args:
        positions_init: ``(n_atoms, 3)`` initial Cartesian positions in Å.
        energy_fn: JAX callable ``positions -> energy_eV`` (or
            ``positions, **neighbor_kwargs -> energy_eV`` when a
            ``potential_wrapper`` is in play).
        fmax: Force-convergence threshold in eV/Å (default 0.01).
        maxiter: Maximum optimizer iterations (default 1000).
        method: ``"LBFGS"`` (default), ``"FIRE"``, ``"GD"`` (gradient
            descent), or ``"SP"`` (damped-Newton stationary point).
        barrier_fn: Optional positions → energy callable added during
            optimization but not reported in the final energy
            (e.g. NCI confinement).
        max_stepsize: Max per-atom step in Å — LBFGS step bound, GD learning
            rate, and SP trust radius (default 0.2). Not used by FIRE, whose
            step is controlled by ``fire_dt_*``.
        potential_wrapper: Optional ``PotentialWrapper`` for systems
            that need neighbor-list management (e.g. SO3LR).
        fire_dt_start: FIRE initial timestep (only used with
            ``method="FIRE"``).
        fire_dt_max: FIRE max timestep.
        fire_n_min: FIRE min positive-power steps before dt increases.
        coords: ``"cartesian"`` (default, unchanged behaviour) or
            ``"internal"`` to run the RFO/BFGS stepper in redundant internal
            coordinates.  With ``"internal"`` the ``method`` argument selects
            only the Cartesian fallback.
        atomic_numbers: ``(n_atoms,)`` atomic numbers, needed to build a
            coordinate set.  Recovered from ``potential_wrapper.species``
            when omitted.
        spec: Precomputed ``CoordinateSpec``, to share one topology across a
            batch or across the cycles of a multilevel ladder.
        init_hessian: ``"identity"`` or ``"lindh"``.  Applies to *both*
            coordinate systems: ``coords="cartesian"`` with an explicit
            ``init_hessian`` runs the same RFO/BFGS stepper in Cartesians,
            which is what makes a coordinate-system comparison controlled.
            ``None`` (default) keeps the existing LBFGS/FIRE/GD/SP paths.
        interfragment: ``"tric"``, ``"aux"``, ``"hbond"`` or ``"none"``.
        frozen_indices: 0-based atom indices to hold fixed.  In Cartesians the
            gradient of those atoms is masked, which pins them for the
            gradient-only steppers (``"LBFGS"``, ``"FIRE"``, ``"GD"``); the
            energy function does not need to be wrapped by the caller.
            ``method="SP"``, ``init_hessian`` and ``coords="internal"`` are
            refused, because a Hessian-based or back-transformed step displaces
            frozen atoms even when their gradient is zero -- see
            :func:`ric_gate`.
        ric_options: Overrides for :data:`RIC_DEFAULTS`.

    Returns:
        Three-tuple ``(positions_opt, energy_opt, info)``:

        - ``positions_opt`` — optimized ``(n_atoms, 3)`` positions in Å.
        - ``energy_opt`` — final energy in eV (excluding any barrier).
        - ``info`` — dict with at least ``converged`` (bool), ``iterations`` (int), ``max_force`` (eV/Å), and ``grad_norm`` (eV/Å).

    Example:
        >>> from mars import load_structure, get_potential, optimize_single
        >>> structure = load_structure("input.xyz")
        >>> potential = get_potential("so3lr", species=structure["numbers"])
        >>> energy_fn = potential.build_energy_fn()
        >>> opt_pos, E, info = optimize_single(
        ...     structure["positions"], energy_fn,
        ...     fmax=0.005, maxiter=500, method="LBFGS",
        ... )
        >>> print(f"converged={info['converged']}  E={E:.4f} eV")
    """
    if coords == "internal" or init_hessian is not None:
        if coords != "internal" and frozen_indices is not None and len(frozen_indices) > 0:
            # init_hessian selects the RFO/BFGS stepper in Cartesians, which
            # displaces frozen atoms even when their gradient is zero (see
            # ric_gate).  Refuse rather than return a wrong geometry.
            raise ValueError(
                "frozen_indices is not supported with init_hessian: the RFO/BFGS "
                "stepper displaces atoms even when their gradient is zero. Use "
                "init_hessian=None to optimize with frozen atoms."
            )
        z = resolve_atomic_numbers(atomic_numbers, potential_wrapper)
        if coords == "internal":
            opts = dict(RIC_DEFAULTS)
            opts.update(ric_options or {})
            ok, reason = ric_gate(
                positions_init,
                z,
                method=method,
                barrier_fn=barrier_fn,
                frozen_indices=frozen_indices,
                spec=spec,
                max_atoms=opts["max_atoms"],
                clash_factor=opts["clash_factor"],
            )
            if not ok:
                # coords="internal" is an assertion, not a preference: the
                # benchmark and the tests need to know RIC actually ran.
                raise ValueError(
                    f"Internal coordinates unavailable for this system: {reason}. "
                    "Use coords='cartesian' to fall back explicitly."
                )
        return optimize_single_ric(
            positions_init,
            energy_fn,
            atomic_numbers=z,
            fmax=fmax,
            maxiter=maxiter,
            coords=coords,
            spec=spec,
            init_hessian=init_hessian or "lindh",
            interfragment=interfragment,
            max_stepsize=max_stepsize,
            potential_wrapper=potential_wrapper,
            ric_options=ric_options,
        )

    # Frozen atoms are masked on the *combined* energy: a barrier added
    # afterwards would put a non-zero gradient back on them.
    if barrier_fn is not None:

        def _combined(pos, **kw):
            return energy_fn(pos, **kw) + barrier_fn(pos)

    else:
        _combined = energy_fn
    opt_energy_fn = _safe_energy(_freeze_cartesian(_combined, frozen_indices, method))

    use_nbrs = potential_wrapper is not None and potential_wrapper.uses_neighbor_lists

    if method == "FIRE":
        if use_nbrs:
            # Whole minimization runs on-device in a single jax.lax.while_loop,
            # with the neighbor list updated inside the loop (mirrors the FIRE
            # path of _make_batch_optimizer). This avoids a per-step device->host
            # sync — the GPU is never blocked waiting for the Python loop.
            _, shift_fn = jmd_space.free()
            fire_init, fire_apply = fire_descent(
                opt_energy_fn,
                shift_fn,
                dt_start=fire_dt_start,
                dt_max=fire_dt_max,
                n_min=fire_n_min,
            )
            get_nbr_kw = potential_wrapper.get_neighbor_kwargs

            @jax.jit
            def _run_fire(pos_init, nbr_state):
                state0 = fire_init(pos_init, **get_nbr_kw(nbr_state))
                max_f0 = jnp.max(jnp.abs(state0.force))

                def cond_fn(carry):
                    _, step, max_f, _ = carry
                    return jnp.logical_and(step < maxiter, max_f >= fmax)

                def body_fn(carry):
                    state, step, _, nbs = carry
                    new_state = fire_apply(state, **get_nbr_kw(nbs))
                    nbs = potential_wrapper.update_neighbors(new_state.position, nbs)
                    return new_state, step + 1, jnp.max(jnp.abs(new_state.force)), nbs

                final_state, steps, final_max_f, final_nbs = jax.lax.while_loop(
                    cond_fn, body_fn, (state0, jnp.zeros((), jnp.int32), max_f0, nbr_state)
                )
                return final_state.position, steps, final_max_f, final_nbs

            nbr_state = potential_wrapper.allocate_neighbors(positions_init)
            positions_opt, steps, final_max_f, nbr_state = _run_fire(positions_init, nbr_state)
            iters = int(steps)
            converged = bool(final_max_f < fmax)
            grad_norm = float(final_max_f)
        else:
            # Fully JIT — no neighbor lists
            fire_fn = jax.jit(
                _make_fire_fn(
                    opt_energy_fn,
                    fmax=fmax,
                    maxiter=maxiter,
                    dt_start=fire_dt_start,
                    dt_max=fire_dt_max,
                    n_min=fire_n_min,
                )
            )
            positions_opt, iters, converged = fire_fn(positions_init)
            iters = int(iters)
            converged = bool(converged)
            grad_norm = float(_compute_max_force(-jax.grad(opt_energy_fn)(positions_opt)))

    elif method == "GD":
        if use_nbrs:
            optimizer = GradientDescent(fun=opt_energy_fn, tol=fmax, stepsize=max_stepsize)
            get_nbr_kw = potential_wrapper.get_neighbor_kwargs

            @jax.jit
            def _run_gd(pos_init, nbr_state):
                state0 = optimizer.init_state(pos_init, **get_nbr_kw(nbr_state))

                def cond_fn(carry):
                    _, state, step, _ = carry
                    return jnp.logical_and(step < maxiter, state.error >= fmax)

                def body_fn(carry):
                    pos, state, step, nbs = carry
                    pos, state = optimizer.update(pos, state, **get_nbr_kw(nbs))
                    nbs = potential_wrapper.update_neighbors(pos, nbs)
                    return pos, state, step + 1, nbs

                pos, state, steps, final_nbs = jax.lax.while_loop(
                    cond_fn, body_fn, (pos_init, state0, jnp.zeros((), jnp.int32), nbr_state)
                )
                return pos, steps, state.error, final_nbs

            nbr_state = potential_wrapper.allocate_neighbors(positions_init)
            positions_opt, steps, final_err, nbr_state = _run_gd(positions_init, nbr_state)
            iters = int(steps)
            converged = bool(final_err < fmax)
            grad_norm = float(final_err)
        else:
            _, opt_energy_jit = _make_force_and_energy_fn(opt_energy_fn)
            optimizer = GradientDescent(
                fun=opt_energy_jit, maxiter=maxiter, tol=fmax, stepsize=max_stepsize
            )
            result = optimizer.run(positions_init)
            positions_opt = result.params
            converged = result.state.error < fmax
            iters = result.state.iter_num
            grad_norm = float(result.state.error)

    elif method == "SP":
        sp_fn = _make_stationary_search_fn(
            opt_energy_fn, fmax=fmax, maxiter=maxiter, trust_radius=max_stepsize
        )
        positions_opt, _, n_steps, converged = sp_fn(positions_init)
        converged = bool(converged)
        iters = int(n_steps)
        grad_norm = float(_compute_max_force(-jax.grad(opt_energy_fn)(positions_opt)))
        if use_nbrs:
            nbr_state = potential_wrapper.allocate_neighbors(positions_opt)

    else:  # LBFGS
        if use_nbrs:
            optimizer = LBFGS(
                fun=opt_energy_fn, tol=fmax, max_stepsize=max_stepsize, **_LBFGS_KWARGS
            )
            get_nbr_kw = potential_wrapper.get_neighbor_kwargs

            @jax.jit
            def _run_lbfgs(pos_init, nbr_state):
                state0 = optimizer.init_state(pos_init, **get_nbr_kw(nbr_state))

                def cond_fn(carry):
                    _, state, step, _ = carry
                    return jnp.logical_and(step < maxiter, state.error >= fmax)

                def body_fn(carry):
                    pos, state, step, nbs = carry
                    new_pos, new_state = optimizer.update(pos, state, **get_nbr_kw(nbs))
                    pos, clipped = _clip_atom_displacement(pos, new_pos, max_stepsize)
                    nbs = potential_wrapper.update_neighbors(pos, nbs)
                    fresh_state = optimizer.init_state(pos, **get_nbr_kw(nbs))
                    state = _select_state(clipped, fresh_state, new_state)
                    return pos, state, step + 1, nbs

                pos, state, steps, final_nbs = jax.lax.while_loop(
                    cond_fn, body_fn, (pos_init, state0, jnp.zeros((), jnp.int32), nbr_state)
                )
                return pos, steps, state.error, final_nbs

            nbr_state = potential_wrapper.allocate_neighbors(positions_init)
            positions_opt, steps, final_err, nbr_state = _run_lbfgs(positions_init, nbr_state)
            iters = int(steps)
            converged = bool(final_err < fmax)
            grad_norm = float(final_err)
        else:
            _, opt_energy_jit = _make_force_and_energy_fn(opt_energy_fn)
            optimizer = LBFGS(
                fun=opt_energy_jit, tol=fmax, max_stepsize=max_stepsize, **_LBFGS_KWARGS
            )

            @jax.jit
            def _run_lbfgs_free(pos_init):
                state0 = optimizer.init_state(pos_init)

                def cond_fn(carry):
                    _, state, step = carry
                    return jnp.logical_and(step < maxiter, state.error >= fmax)

                def body_fn(carry):
                    pos, state, step = carry
                    new_pos, new_state = optimizer.update(pos, state)
                    pos, clipped = _clip_atom_displacement(pos, new_pos, max_stepsize)
                    fresh_state = optimizer.init_state(pos)
                    state = _select_state(clipped, fresh_state, new_state)
                    return pos, state, step + 1

                pos, state, steps = jax.lax.while_loop(
                    cond_fn, body_fn, (pos_init, state0, jnp.zeros((), jnp.int32))
                )
                return pos, steps, state.error

            positions_opt, steps, final_err = _run_lbfgs_free(positions_init)
            iters = int(steps)
            converged = bool(final_err < fmax)
            grad_norm = float(final_err)

    # Report energy and forces from original energy_fn (without barrier)
    if use_nbrs:
        nbr_state = potential_wrapper.update_neighbors(positions_opt, nbr_state)
        report_nbr_kw = potential_wrapper.get_neighbor_kwargs(nbr_state)
    else:
        report_nbr_kw = {}

    force_fn, energy_jit = _make_force_and_energy_fn(lambda pos: energy_fn(pos, **report_nbr_kw))
    energy_opt = energy_jit(positions_opt)
    forces = force_fn(positions_opt)
    max_force = _compute_max_force(forces)

    info = {
        "converged": converged,
        "iterations": iters,
        "final_energy": float(energy_opt),
        "max_force": float(max_force),
        "grad_norm": grad_norm,
    }

    return positions_opt, float(energy_opt), info


def optimize_single_with_logging(
    positions_init: jnp.ndarray,
    energy_fn: Callable,
    fmax: float = 0.01,
    structure_id: str = "",
    **kwargs,
) -> Tuple[jnp.ndarray, float]:
    """Minimize with progress logging.

    Args:
        positions_init: Initial positions
        energy_fn: Energy function
        fmax: Max force convergence (eV/Å)
        structure_id: Identifier for logging
        **kwargs: Forwarded verbatim to :func:`optimize_single` (``method``,
                  ``maxiter``, ``max_stepsize``, ``potential_wrapper``,
                  ``fire_dt_*``, ``coords``, ``atomic_numbers``, ...).

    Returns:
        (positions_opt, energy_opt)
    """
    start = log_step_start(f"Optimizing structure {structure_id}")

    # Forward everything. This used to cherry-pick four kwargs and silently
    # drop the rest, so fire_dt_* never reached the optimizer from the
    # conformer-search entry points.
    positions_opt, energy_opt, info = optimize_single(
        positions_init,
        energy_fn,
        fmax=fmax,
        **kwargs,
    )

    if info["converged"]:
        log_message(
            f"✓ Converged in {info['iterations']} steps, E={energy_opt * EV_TO_KCALMOL:.2f} kcal/mol"
        )
    else:
        log_message(
            f"⚠ Not converged after {info['iterations']} steps, E={energy_opt * EV_TO_KCALMOL:.2f} kcal/mol"
        )

    log_step_end(f"Optimizing structure {structure_id}", start)
    return positions_opt, energy_opt


# ============================================================================
# Redundant-Internal-Coordinate Minimization
# ============================================================================


class RICState(NamedTuple):
    """Loop state of the RIC minimizer; every field is a traced array.

    ``positions``/``energy``/``grad_cart`` always describe the current
    *accepted* point.  ``g_q_prev`` and ``dq_prev`` record the internal
    gradient and the step of the last accepted transition, which is what the
    BFGS update needs.
    """

    positions: jnp.ndarray
    energy: jnp.ndarray
    grad_cart: jnp.ndarray
    g_q_prev: jnp.ndarray
    dq_prev: jnp.ndarray
    can_update: jnp.ndarray
    hessian: jnp.ndarray
    frag_ref: jnp.ndarray
    trust: jnp.ndarray
    step: jnp.ndarray
    max_force: jnp.ndarray
    n_reject: jnp.ndarray
    n_reject_run: jnp.ndarray
    n_fallback: jnp.ndarray
    n_bt_fail: jnp.ndarray
    bt_iter_sum: jnp.ndarray
    rank: jnp.ndarray
    nbrs: object


def _rfo_lambda(b, f, n_bisect: int = 60):
    """Lowest root of the RFO secular equation, by fixed-count bisection.

    Solving ``lam = sum_i F_i^2 / (lam - b_i)`` this way avoids a second
    eigendecomposition (of the augmented Hessian) and has a deterministic,
    vmappable cost.  Sixty bisections give ~18 digits of the bracket.
    """
    lo = b[0] - jnp.linalg.norm(f) - 1.0
    hi = jnp.minimum(0.0, b[0]) - 1e-10

    def sec(lam):
        den = lam - b
        safe = jnp.abs(den) > 1e-14
        return lam - jnp.sum(jnp.where(safe, f**2 / jnp.where(safe, den, 1.0), 0.0))

    def step(_, bounds):
        lo_, hi_ = bounds
        mid = 0.5 * (lo_ + hi_)
        neg = sec(mid) < 0.0
        return (jnp.where(neg, mid, lo_), jnp.where(neg, hi_, mid))

    lo, hi = jax.lax.fori_loop(0, n_bisect, step, (lo, hi))
    return 0.5 * (lo + hi)


def _bfgs_update(h, dq, dg, active):
    """BFGS update of the internal Hessian, guarded against a bad pair."""
    sty = jnp.dot(dq, dg)
    hs = h @ dq
    sths = jnp.dot(dq, hs)
    ok = (
        active
        & (sty > 1e-8 * jnp.linalg.norm(dq) * jnp.linalg.norm(dg) + 1e-30)
        & (sths > 1e-12)
        & (jnp.linalg.norm(dq) > 1e-10)
        & jnp.all(jnp.isfinite(dg))
    )
    h_new = (
        h
        + jnp.outer(dg, dg) / jnp.where(sty != 0, sty, 1.0)
        - jnp.outer(hs, hs) / jnp.where(sths != 0, sths, 1.0)
    )
    return jnp.where(ok, h_new, h)


def _superpose(x_new, x_ref):
    """Kabsch-align *x_new* onto *x_ref*, removing net translation/rotation.

    The potential is invariant under this, so it cannot change the energy; it
    keeps the frozen linear-bend axes and the TRIC rotation reference valid
    over many steps.
    """
    cn = jnp.mean(x_new, axis=0)
    cr = jnp.mean(x_ref, axis=0)
    a, b = x_new - cn, x_ref - cr
    u, _s, vt = jnp.linalg.svd(a.T @ b)
    d = jnp.sign(jnp.linalg.det(u @ vt))
    rot = u @ jnp.diag(jnp.array([1.0, 1.0, 1.0], dtype=x_new.dtype).at[2].set(d)) @ vt
    return (a @ rot) + cr


def _make_ric_step(
    energy_fn: Callable,
    spec,
    fns,
    *,
    fmax: float,
    maxiter: int,
    trust_init: float,
    trust_min: float,
    trust_max: float,
    bt_iter: int,
    hessian_penalty: float,
    potential_wrapper=None,
    max_reject: int = 10,
):
    """Build the jit/vmap-safe RIC minimization loop for one coordinate spec.

    *spec* and *fns* are static (closed over), so the returned callable is a
    pure function of traced arrays and can be ``vmap``ped across a batch of
    structures sharing the topology.

    Returns:
        ``run(x0, frag_ref0, h0, nbrs0) -> RICState``.
    """
    from .internal_coords import (
        _safe_norm,
        generalized_inverse,
        global_tr_basis,
        internal_to_cartesian,
    )

    use_nbrs = potential_wrapper is not None and potential_wrapper.uses_neighbor_lists
    get_kw = potential_wrapper.get_neighbor_kwargs if use_nbrs else (lambda _s: {})
    n_int = spec.n_internal

    def _eg(x, nbrs):
        return jax.value_and_grad(lambda p: energy_fn(p, **get_kw(nbrs)))(x)

    def body(s: RICState) -> RICState:
        x, e, g = s.positions, s.energy, s.grad_cart
        b = fns.b_fn(x, s.frag_ref)
        ginv = generalized_inverse(b, tr_basis=global_tr_basis(x))
        g_q = ginv.b_pinv.T @ g

        # BFGS from the last accepted transition.
        h = _bfgs_update(s.hessian, s.dq_prev, g_q - s.g_q_prev, s.can_update)

        # Project the Hessian onto the non-redundant subspace. The large
        # penalty in the redundant complement stops the step trying to change
        # coordinate combinations the back-transformation cannot realise.
        p = ginv.projector
        eye = jnp.eye(n_int, dtype=h.dtype)
        h_proj = 0.5 * ((p @ h @ p) + (p @ h @ p).T) + hessian_penalty * (eye - p)
        evals, vecs = jnp.linalg.eigh(h_proj)
        f = vecs.T @ (p @ g_q)
        lam = _rfo_lambda(evals, f)
        den = lam - evals
        safe = jnp.abs(den) > 1e-10
        dq = p @ (vecs @ jnp.where(safe, f / jnp.where(safe, den, 1.0), 0.0))

        # Trust radius imposed on the Cartesian displacement: unit-clean, and
        # directly comparable with the existing --max-stepsize semantics.
        dx_est = (ginv.b_pinv @ dq).reshape(x.shape)
        max_disp = jnp.max(_safe_norm(dx_est))
        dq = dq * jnp.minimum(1.0, s.trust / (max_disp + 1e-12))
        pred_de = jnp.dot(dq, g_q) + 0.5 * jnp.dot(dq, h_proj @ dq)

        bt = internal_to_cartesian(
            fns, x, s.frag_ref, dq, ginv, g, trust_radius=s.trust, max_iter=bt_iter
        )
        dx_ric = bt.positions - x
        dx_cart = -s.trust * g.reshape(x.shape) / (jnp.max(jnp.abs(g)) + 1e-12)
        ric_ok = (
            jnp.all(jnp.isfinite(dx_ric))
            & (bt.mode <= 1)
            & (jnp.max(_safe_norm(dx_ric)) < 5.0 * s.trust)
        )
        dx = jnp.where(ric_ok, dx_ric, dx_cart)
        x_trial = _superpose(x + dx, x)

        nbrs = potential_wrapper.update_neighbors(x_trial, s.nbrs) if use_nbrs else s.nbrs
        e_trial, g_trial = _eg(x_trial, nbrs)

        accept = jnp.logical_and(jnp.isfinite(e_trial), e_trial <= e + 1e-12)
        ratio = jnp.where(jnp.abs(pred_de) > 1e-12, (e_trial - e) / pred_de, 1.0)
        disp = jnp.max(_safe_norm(dx))
        trust = jnp.where(
            accept,
            jnp.where(
                (ratio > 0.75) & (disp > 0.8 * s.trust),
                jnp.minimum(2.0 * s.trust, trust_max),
                jnp.where(ratio < 0.25, jnp.maximum(trust_min, 0.5 * s.trust), s.trust),
            ),
            jnp.maximum(trust_min, 0.25 * disp),
        )

        x_next = jnp.where(accept, x_trial, x)
        e_next = jnp.where(accept, e_trial, e)
        g_next = jnp.where(accept, g_trial.reshape(-1), g)
        frag_ref = fns.refresh_ref_fn(x_next, s.frag_ref)

        return s._replace(
            positions=x_next,
            energy=e_next,
            grad_cart=g_next,
            g_q_prev=jnp.where(accept, g_q, s.g_q_prev),
            dq_prev=jnp.where(accept, dq, s.dq_prev),
            can_update=accept,
            hessian=h,
            frag_ref=frag_ref,
            trust=trust,
            step=s.step + 1,
            max_force=jnp.max(jnp.abs(g_next)),
            n_reject=s.n_reject + (~accept).astype(s.n_reject.dtype),
            n_reject_run=jnp.where(accept, 0, s.n_reject_run + 1),
            n_fallback=s.n_fallback + (~ric_ok).astype(s.n_fallback.dtype),
            n_bt_fail=s.n_bt_fail + (bt.mode >= 2).astype(s.n_bt_fail.dtype),
            bt_iter_sum=s.bt_iter_sum + bt.n_iter,
            rank=ginv.rank.astype(jnp.int32),
            nbrs=nbrs,
        )

    def cond(s: RICState):
        # Bailing out after max_reject consecutive rejections guarantees the
        # RIC path is never worse than Cartesian LBFGS: the caller finishes
        # the job with the existing minimizer from wherever this stopped.
        return jnp.logical_and(
            s.step < maxiter,
            jnp.logical_and(s.max_force >= fmax, s.n_reject_run < max_reject),
        )

    def run(x0, frag_ref0, h0, nbrs0):
        e0, g0 = _eg(x0, nbrs0)
        s0 = RICState(
            positions=x0,
            energy=e0,
            grad_cart=g0.reshape(-1),
            g_q_prev=jnp.zeros((n_int,), dtype=x0.dtype),
            dq_prev=jnp.zeros((n_int,), dtype=x0.dtype),
            can_update=jnp.asarray(False),
            hessian=h0,
            frag_ref=frag_ref0,
            trust=jnp.asarray(trust_init, dtype=x0.dtype),
            step=jnp.zeros((), jnp.int32),
            max_force=jnp.max(jnp.abs(g0)),
            n_reject=jnp.zeros((), jnp.int32),
            n_reject_run=jnp.zeros((), jnp.int32),
            n_fallback=jnp.zeros((), jnp.int32),
            n_bt_fail=jnp.zeros((), jnp.int32),
            bt_iter_sum=jnp.zeros((), jnp.int32),
            rank=jnp.zeros((), jnp.int32),
            nbrs=nbrs0,
        )
        return jax.lax.while_loop(cond, body, s0)

    return run


# ============================================================================
# Public RIC entry points and the host-side gate
# ============================================================================


RIC_DEFAULTS = {
    "max_atoms": 150,
    "backtransform_iter": 25,
    "trust_init": 0.1,
    "trust_min": 0.005,
    "trust_max": 0.3,
    "hessian_penalty": 1000.0,
    "max_reject": 10,
    "rebuild_interval": 100,
    "max_rebuilds": 3,
    "clash_factor": 0.7,
}


def resolve_atomic_numbers(atomic_numbers, potential_wrapper):
    """Recover atomic numbers from an explicit argument or the potential."""
    if atomic_numbers is not None:
        return np.asarray(atomic_numbers, dtype=int)
    if potential_wrapper is not None:
        sp = getattr(potential_wrapper, "species", None)
        if sp is not None:
            return np.asarray(sp, dtype=int)
    return None


def ric_gate(
    positions,
    atomic_numbers,
    *,
    method: str = "LBFGS",
    barrier_fn=None,
    frozen_indices=None,
    spec=None,
    max_atoms: int = 150,
    clash_factor: float = 0.7,
) -> Tuple[bool, str]:
    """Decide host-side whether internal coordinates may be used.

    Returns a plain Python ``(bool, reason)`` so the caller can pick which
    traced function to *build* with an ordinary ``if``.  Nothing here is
    traced, so it is free to log, to raise, and to run NumPy.

    The refusals are deliberate, not conservatism:

    * **frozen atoms** — ``make_frozen_energy_fn`` pins coordinates inside the
      energy function, so the Cartesian gradient is zero there, but a
      back-transformed internal step still moves those atoms.  The reported
      energy stays correct because it is evaluated at the pinned coordinates,
      so the failure would be a silently wrong output geometry.  Constrained
      RIC needs Lagrange multipliers on every internal touching a frozen atom.
    * **barrier / wall potentials** — the minimum is a Cartesian confinement,
      not expressible in a bonded coordinate set, so the in-loop fallback
      would fire every iteration: all of the cost, none of the benefit.
    """
    from .utils import COVALENT_RADII

    if atomic_numbers is None and spec is None:
        return False, "no_atomic_numbers"
    if frozen_indices is not None and len(frozen_indices) > 0:
        return False, "frozen_atoms"
    if barrier_fn is not None:
        return False, "barrier_fn"
    if method in ("FIRE", "GD", "SP", "HYBRID"):
        return False, "method_not_supported_in_internals"
    if not jax.config.jax_enable_x64:
        return False, "float32_unavailable"

    pos = np.asarray(positions, dtype=float)
    n_atoms = pos.shape[0]
    if n_atoms < 4:
        return False, "too_few_atoms"
    if n_atoms > max_atoms:
        return False, "system_too_large"

    z = np.asarray(atomic_numbers, dtype=int)
    radii = np.asarray(COVALENT_RADII)[np.clip(z, 0, len(COVALENT_RADII) - 1)]
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    if np.any(d < clash_factor * (radii[:, None] + radii[None, :])):
        return False, "clashed_geometry"
    return True, ""


def optimize_single_ric(
    positions_init: jnp.ndarray,
    energy_fn: Callable,
    atomic_numbers=None,
    fmax: float = 0.01,
    maxiter: int = 300,
    *,
    coords: str = "internal",
    spec=None,
    init_hessian: str = "lindh",
    interfragment: str = "tric",
    max_stepsize: float = None,
    potential_wrapper=None,
    ric_options: Dict = None,
) -> Tuple[jnp.ndarray, float, Dict]:
    """Minimize one structure with the RFO/BFGS stepper.

    Runs rational-function-optimization steps on a projected BFGS Hessian,
    either in redundant internal coordinates (``coords="internal"``) or in
    Cartesians (``coords="cartesian"``, via :func:`~mars.internal_coords.
    cartesian_spec`).  Both use exactly the same stepper, so a comparison
    between them isolates the coordinate system rather than comparing two
    different optimizers.

    Convergence is tested on the Cartesian max-force in eV/Å, identically to
    :func:`optimize_single`, so results stay directly comparable with the
    LBFGS/FIRE paths and with ``FMAX_MAP``.  A converged *internal* gradient
    would not imply a converged Cartesian one when the coordinate set is
    incomplete, which is precisely the failure worth catching.

    Args:
        positions_init: ``(n_atoms, 3)`` positions in Angstrom.
        energy_fn: JAX callable ``positions -> energy_eV``.
        atomic_numbers: ``(n_atoms,)`` atomic numbers.  Recovered from
            ``potential_wrapper.species`` when omitted.
        fmax: Force convergence threshold in eV/Å.
        maxiter: Maximum iterations.
        coords: ``"internal"`` or ``"cartesian"``.
        spec: Precomputed :class:`~mars.internal_coords.CoordinateSpec`.
        init_hessian: ``"identity"`` or ``"lindh"``
            (:data:`mars.utils.HESSIAN_MODES`).
        interfragment: Passed to
            :func:`~mars.internal_coords.build_coordinates`.
        max_stepsize: Initial trust radius in Angstrom; defaults to the RIC
            default (0.1) so it keeps the same meaning as elsewhere in MARS.
        potential_wrapper: Optional ``PotentialWrapper`` for neighbor lists.
        ric_options: Overrides for :data:`RIC_DEFAULTS`.

    Returns:
        ``(positions_opt, energy_opt, info)`` — the same contract as
        :func:`optimize_single`, with extra ``ric_*`` diagnostic keys.
    """
    from .internal_coords import (
        build_coordinates,
        cartesian_coordinate_fns,
        cartesian_spec,
        generalized_inverse,
        global_tr_basis,
        initial_frag_ref,
        make_coordinate_fns,
    )
    from .utils import initial_internal_hessian

    opts = dict(RIC_DEFAULTS)
    opts.update(ric_options or {})
    trust_init = opts["trust_init"] if max_stepsize is None else float(max_stepsize)

    z = resolve_atomic_numbers(atomic_numbers, potential_wrapper)
    x = jnp.asarray(positions_init)
    n_atoms = int(x.shape[0])

    use_nbrs = potential_wrapper is not None and potential_wrapper.uses_neighbor_lists
    nbrs = potential_wrapper.allocate_neighbors(x) if use_nbrs else None

    info = {
        "coords_used": coords,
        "ric_fallback_reason": None,
        "ric_rebuilds": 0,
        "ric_init_hessian": init_hessian,
    }

    total_steps = 0
    rebuilds = 0
    max_rebuilds = opts["max_rebuilds"] if coords == "internal" else 0

    while True:
        if coords == "cartesian":
            spec_i = cartesian_spec(n_atoms)
            fns = cartesian_coordinate_fns(n_atoms)
        else:
            spec_i = (
                spec
                if spec is not None
                else build_coordinates(np.asarray(x), z, interfragment=interfragment)
            )
            if not spec_i.valid:
                info["ric_fallback_reason"] = spec_i.reason
                info["coords_used"] = "cartesian"
                return optimize_single(
                    x,
                    energy_fn,
                    fmax=fmax,
                    maxiter=maxiter,
                    potential_wrapper=potential_wrapper,
                )
            fns = make_coordinate_fns(spec_i)

        frag_ref = initial_frag_ref(spec_i, x)
        b0 = fns.b_fn(x, frag_ref)
        ginv0 = generalized_inverse(b0, tr_basis=global_tr_basis(x))
        h0 = initial_internal_hessian(
            init_hessian,
            x,
            z if z is not None else np.ones(n_atoms, dtype=int),
            ginv0,
            projector_penalty=opts["hessian_penalty"],
        )

        run = jax.jit(
            _make_ric_step(
                energy_fn,
                spec_i,
                fns,
                fmax=fmax,
                maxiter=(
                    min(maxiter - total_steps, opts["rebuild_interval"])
                    if coords == "internal"
                    else maxiter - total_steps
                ),
                trust_init=trust_init,
                trust_min=opts["trust_min"],
                trust_max=opts["trust_max"],
                bt_iter=opts["backtransform_iter"],
                hessian_penalty=opts["hessian_penalty"],
                potential_wrapper=potential_wrapper,
                max_reject=opts["max_reject"],
            )
        )
        state = run(x, frag_ref, h0, nbrs)
        x = state.positions
        nbrs = state.nbrs
        total_steps += int(state.step)
        converged = bool(state.max_force < fmax)
        stalled = bool(state.n_reject_run >= opts["max_reject"])

        info.update(
            {
                "ric_n_internal": int(spec_i.n_internal),
                "ric_rank": int(state.rank),
                "ric_expected_rank": int(spec_i.expected_rank),
                "ric_fallback_steps": int(state.n_fallback),
                "ric_bt_failures": int(state.n_bt_fail),
                "ric_rejected_steps": int(state.n_reject),
                "ric_trust_radius": float(state.trust),
                "ric_bt_iterations_mean": float(state.bt_iter_sum) / max(int(state.step), 1),
            }
        )

        if converged or stalled or total_steps >= maxiter or rebuilds >= max_rebuilds:
            break
        # The coordinate set is fixed for the life of the inner loop, so a
        # topology change is handled here on the host: rebuild and re-enter.
        if spec is None and z is not None:
            new_spec = build_coordinates(np.asarray(x), z, interfragment=interfragment)
            same = (
                new_spec.valid
                and new_spec.n_internal == spec_i.n_internal
                and np.array_equal(new_spec.stretch_idx, spec_i.stretch_idx)
            )
            if same:
                break
            log_message("[RIC] coordinate set changed; rebuilding and resetting the Hessian")
            rebuilds += 1
            info["ric_rebuilds"] = rebuilds
            continue
        break

    # Cartesian LBFGS finishes anything the RIC loop could not, which makes
    # the RIC path a strict improvement on the status quo rather than a bet.
    if not converged and total_steps < maxiter:
        info["ric_bailed_to_cartesian"] = True
        x, e, cart_info = optimize_single(
            x,
            energy_fn,
            fmax=fmax,
            maxiter=maxiter - total_steps,
            potential_wrapper=potential_wrapper,
        )
        info.update(
            converged=cart_info["converged"],
            iterations=total_steps + cart_info["iterations"],
            final_energy=e,
            max_force=cart_info["max_force"],
            grad_norm=cart_info["grad_norm"],
        )
        return x, float(e), info

    info.setdefault("ric_bailed_to_cartesian", False)
    info.update(
        converged=converged,
        iterations=total_steps,
        final_energy=float(state.energy),
        max_force=float(state.max_force),
        grad_norm=float(state.max_force),
    )
    return x, float(state.energy), info


def _build_ric_batch_inputs(positions_batch, atomic_numbers, spec, fns, init_hessian, opts):
    """Per-structure TRIC references and initial Hessians for a shared spec.

    The coordinate *definition* is shared across the batch (conformers of one
    molecule have one topology), but the model Hessian depends on the geometry,
    so it is built per structure.  That happens here on the host, once, outside
    the jit -- a few tens of milliseconds per structure against a whole
    optimization.
    """
    from .internal_coords import generalized_inverse, global_tr_basis, initial_frag_ref
    from .utils import initial_internal_hessian

    refs, hessians = [], []
    for i in range(positions_batch.shape[0]):
        x = positions_batch[i]
        ref = initial_frag_ref(spec, x)
        ginv = generalized_inverse(fns.b_fn(x, ref), tr_basis=global_tr_basis(x))
        hessians.append(
            initial_internal_hessian(
                init_hessian,
                x,
                atomic_numbers,
                ginv,
                projector_penalty=opts["hessian_penalty"],
            )
        )
        refs.append(ref)
    return jnp.stack(refs), jnp.stack(hessians)


def _optimize_batch_ric(
    positions_batch,
    energy_fn,
    atomic_numbers,
    *,
    fmax,
    maxiter,
    max_stepsize,
    coords,
    init_hessian,
    interfragment,
    spec,
    potential_wrapper,
    ric_options,
):
    """vmapped RFO/BFGS optimization of structures sharing one topology.

    The spec and the coordinate callables are closed over at trace time, so
    they become jit constants broadcast across the batch and ``optimize_one``
    still maps only over positions, references, Hessians and neighbor state.

    Structures whose covalent topology differs from the reference are not
    handled here: the caller checks that host-side and sends mismatched
    batches down the Cartesian path.
    """
    from .internal_coords import (
        build_coordinates,
        cartesian_coordinate_fns,
        cartesian_spec,
        make_coordinate_fns,
    )

    opts = dict(RIC_DEFAULTS)
    opts.update(ric_options or {})
    n_atoms = positions_batch.shape[1]
    init_hessian = init_hessian or "lindh"

    if coords == "cartesian":
        spec_i = cartesian_spec(n_atoms)
        fns = cartesian_coordinate_fns(n_atoms)
    else:
        spec_i = (
            spec
            if spec is not None
            else build_coordinates(
                np.asarray(positions_batch[0]), atomic_numbers, interfragment=interfragment
            )
        )
        if not spec_i.valid:
            raise ValueError(f"Internal coordinates unavailable: {spec_i.reason}")
        fns = make_coordinate_fns(spec_i)

    use_nbrs = potential_wrapper is not None and potential_wrapper.uses_neighbor_lists
    if use_nbrs:
        from .sampling import _allocate_nbrs_batch

        nbr_batch = _allocate_nbrs_batch(positions_batch, potential_wrapper)
    else:
        nbr_batch = None

    frag_refs, hessians = _build_ric_batch_inputs(
        positions_batch,
        atomic_numbers,
        spec_i,
        fns,
        init_hessian,
        opts,
    )

    run = _make_ric_step(
        energy_fn,
        spec_i,
        fns,
        fmax=fmax,
        maxiter=maxiter,
        trust_init=opts["trust_init"] if max_stepsize is None else float(max_stepsize),
        trust_min=opts["trust_min"],
        trust_max=opts["trust_max"],
        bt_iter=opts["backtransform_iter"],
        hessian_penalty=opts["hessian_penalty"],
        potential_wrapper=potential_wrapper,
        max_reject=opts["max_reject"],
    )

    if use_nbrs:
        batched = jax.jit(jax.vmap(run))
        state = batched(positions_batch, frag_refs, hessians, nbr_batch)
    else:
        batched = jax.jit(jax.vmap(lambda x, r, h: run(x, r, h, None)))
        state = batched(positions_batch, frag_refs, hessians)

    return (
        state.positions,
        state.energy,
        state.max_force < fmax,
        state.step,
        state.max_force,
        {
            "coords_used": coords,
            "ric_n_internal": int(spec_i.n_internal),
            "ric_expected_rank": int(spec_i.expected_rank),
            "ric_init_hessian": init_hessian,
            "ric_fallback_steps": np.asarray(state.n_fallback),
            "ric_rejected_steps": np.asarray(state.n_reject),
        },
    )


def ric_batch_topology_matches(positions_batch, atomic_numbers, spec, tolerance=1.3):
    """True when every structure in the batch has the reference bond set.

    Conformers of one molecule share a topology, so this is normally trivially
    true and costs one NumPy bond detection per structure.  Where it is false
    (a solvation batch with different shell arrangements) the caller must not
    reuse the spec.
    """
    from .utils import detect_bonds

    ref = {tuple(map(int, b)) for b in spec.stretch_idx[: len(spec.stretch_idx) - spec.n_aux_bonds]}
    for i in range(positions_batch.shape[0]):
        bonds, _ = detect_bonds(np.asarray(positions_batch[i]), atomic_numbers, tolerance)
        if {tuple(map(int, b)) for b in bonds} != ref:
            return False
    return True


# ============================================================================
# Stationary Point Search
# ============================================================================


def _make_stationary_search_fn(
    energy_fn: Callable,
    fmax: float = 0.01,
    maxiter: int = 1000,
    trust_radius: float = 0.3,
    step_tol: float = 1e-5,
    damping: float = 1e-3,
):
    """JIT-compiled damped Newton stationary point search (grad(E) = 0)."""
    value_and_grad = jax.value_and_grad(energy_fn)
    hess_fn = jax.hessian(energy_fn)

    @jax.jit
    def sp_search(positions_init):
        n_dof = positions_init.size

        def cond_fn(state):
            pos, step, converged, _mu = state
            return jnp.logical_and(~converged, step < maxiter)

        def body_fn(state):
            pos, step, _converged, mu = state

            E, g = value_and_grad(pos)
            H = hess_fn(pos)

            g_flat = g.reshape(n_dof)
            H_flat = H.reshape(n_dof, n_dof)

            # Damped Newton: (H + mu I) delta = -g
            I = jnp.eye(n_dof, dtype=H_flat.dtype)
            H_damped = H_flat + mu * I
            delta = jnp.linalg.solve(H_damped, -g_flat)

            # Trust radius
            step_norm = jnp.linalg.norm(delta)
            scale = jnp.minimum(1.0, trust_radius / (step_norm + 1e-12))
            delta = delta * scale

            new_pos = pos + delta.reshape(pos.shape)

            _, new_g = value_and_grad(new_pos)
            max_force = jnp.max(jnp.abs(-new_g))
            delta_norm = jnp.linalg.norm(delta)
            converged = jnp.logical_and(max_force < fmax, delta_norm < step_tol)

            # Adapt damping based on step acceptance
            mu_new = jnp.where(scale < 0.2, mu * 10.0, jnp.where(scale > 0.9, mu * 0.7, mu))
            mu_new = jnp.clip(mu_new, 1e-12, 1e6)

            return new_pos, step + 1, converged, mu_new

        init_state = (positions_init, jnp.array(0), jnp.array(False), jnp.array(damping))
        final_pos, n_steps, converged, _mu = jax.lax.while_loop(cond_fn, body_fn, init_state)
        energy = energy_fn(final_pos)
        return final_pos, energy, n_steps, converged

    return sp_search


def find_stationary_point(
    positions_init: jnp.ndarray,
    energy_fn: Callable,
    fmax: float = 0.01,
    tol: float = None,
    maxiter: int = 200,
    trust_radius: float = 0.3,
    verify: bool = False,
) -> Tuple[jnp.ndarray, float, Dict]:
    """Find a stationary point via damped Newton with trust-radius control.

    Unlike :func:`optimize_single`, this routine does **not** require the
    Hessian to be positive-definite, so it can converge on saddle points
    and transition states. Steps are bounded by ``trust_radius`` to keep
    the linearization local.

    Args:
        positions_init: ``(n_atoms, 3)`` starting positions in Å. Should
            already be reasonably close to the target stationary point.
        energy_fn: JAX callable ``positions -> energy_eV``.
        fmax: Force-convergence threshold in eV/Å (default 0.01).
        tol: Backwards-compatible alias for ``fmax`` — if set, it
            overrides ``fmax``.
        maxiter: Maximum iterations (default 200).
        trust_radius: Maximum step length in Å (default 0.3).
        verify: If True, run an extra Hessian eigenvalue analysis at the
            converged point (currently a no-op placeholder).

    Returns:
        Three-tuple ``(positions_sp, energy_sp_eV, info)`` where ``info``
        contains ``converged``, ``iterations``, ``final_energy``, and
        ``max_force`` (eV/Å).

    Example:
        >>> from mars import load_structure, get_potential, find_stationary_point
        >>> structure = load_structure("guess.xyz")
        >>> potential = get_potential("so3lr", species=structure["numbers"])
        >>> energy_fn = potential.build_energy_fn()
        >>> pos_sp, E_sp, info = find_stationary_point(
        ...     structure["positions"], energy_fn,
        ...     fmax=0.001, maxiter=200, trust_radius=0.2,
        ... )
    """
    if tol is not None:
        fmax = tol

    sp_fn = _make_stationary_search_fn(
        energy_fn, fmax=fmax, maxiter=maxiter, trust_radius=trust_radius
    )
    positions_sp, energy_sp, n_steps, converged = sp_fn(positions_init)

    force_fn, _ = _make_force_and_energy_fn(energy_fn)
    forces = force_fn(positions_sp)
    max_force = _compute_max_force(forces)

    info = {
        "converged": bool(converged),
        "iterations": int(n_steps),
        "final_energy": float(energy_sp),
        "max_force": float(max_force),
    }

    return positions_sp, float(energy_sp), info


def find_stationary_point_with_logging(
    positions_init: jnp.ndarray,
    energy_fn: Callable,
    fmax: float = 0.01,
    maxiter: int = 200,
    structure_id: str = "",
) -> Tuple[jnp.ndarray, float]:
    """Find stationary point with progress logging.

    Args:
        positions_init: Initial positions near stationary point
        energy_fn: Energy function
        fmax: Max force convergence (eV/Å)
        maxiter: Maximum iterations
        structure_id: Identifier for logging

    Returns:
        (positions_sp, energy_sp)
    """
    start = log_step_start(f"Finding stationary point {structure_id}")

    positions_sp, energy_sp, info = find_stationary_point(
        positions_init, energy_fn, fmax=fmax, maxiter=maxiter
    )

    if info["converged"]:
        log_message(
            f"Converged in {info['iterations']} steps, E={energy_sp * EV_TO_KCALMOL:.2f} kcal/mol"
        )
    else:
        log_message(f"Not converged after {info['iterations']} steps")

    log_step_end(f"Finding stationary point {structure_id}", start)
    return positions_sp, energy_sp


# ============================================================================
# Single-Point Energy Pre-Screening
# ============================================================================


def compute_single_point_energies(
    structures: List[Dict],
    energy_fn: Callable,
    potential_wrapper=None,
) -> List[float]:
    """Compute single-point energies for a list of structures using vmap.

    On OOM, automatically splits the batch in half and retries recursively.

    Returns:
        List of energies (eV) in the same order as structures.
    """
    if not structures:
        return []

    n = len(structures)
    use_nbrs = potential_wrapper is not None and potential_wrapper.uses_neighbor_lists
    positions_batch = jnp.stack([jnp.asarray(s["positions"]) for s in structures])

    try:
        if use_nbrs:
            from .sampling import _allocate_nbrs_batch

            nbr_batch = _allocate_nbrs_batch(positions_batch, potential_wrapper)
            get_nbr_kw = potential_wrapper.get_neighbor_kwargs
            energy_jit = jax.jit(energy_fn)

            def eval_one(pos, nbr_state):
                return energy_jit(pos, **get_nbr_kw(nbr_state))

            energies_batch = jax.jit(jax.vmap(eval_one))(positions_batch, nbr_batch)
        else:
            energies_batch = jax.jit(jax.vmap(energy_fn))(positions_batch)

        return [float(e) for e in energies_batch]

    except Exception as e:
        if not _is_jax_oom(e) or n < 2:
            raise

        mid = n // 2
        log_message(
            f"[SP energies] OOM with {n} structures, splitting into "
            f"{mid} + {n - mid} and retrying"
        )
        jax.clear_caches()

        energies_a = compute_single_point_energies(structures[:mid], energy_fn, potential_wrapper)
        energies_b = compute_single_point_energies(structures[mid:], energy_fn, potential_wrapper)
        return energies_a + energies_b


def prescreen_by_energy(
    structures: List[Dict],
    energy_fn: Callable,
    energy_ref: float,
    ewin_buffer: float,
    potential_wrapper=None,
    label: str = "",
) -> List[Dict]:
    """Discard structures whose single-point energy exceeds energy_ref + ewin_buffer.

    Cheap pre-filter before full geometry optimization to skip structures
    that are clearly outside the energy window regardless of optimization.

    Args:
        structures: List of structure dicts.
        energy_fn: JAX energy function.
        energy_ref: Reference energy (eV) — typically the lowest known conformer energy.
        ewin_buffer: Energy cutoff above energy_ref (eV). Structures with
                     SP energy > energy_ref + ewin_buffer are discarded.
        potential_wrapper: PotentialWrapper instance for neighbor-list potentials.
        label: Tag used in log messages (e.g. "MTD Grid").

    Returns:
        Filtered list of structure dicts.
    """
    if not structures:
        return structures

    energies = compute_single_point_energies(structures, energy_fn, potential_wrapper)
    cutoff = energy_ref + ewin_buffer
    filtered = [s for s, e in zip(structures, energies) if e <= cutoff]

    prefix = f"[{label}] " if label else ""
    if len(filtered) == 0:
        log_message(
            f"[{prefix}SP energy pre-screening] WARNING: energy window pruning (Ewin={ewin_buffer * EV_TO_KCALMOL:.1f} kcal/mol) "
            f"would discard ALL {len(structures)} conformers. "
            f"High-energy conformers will be sent to optimization as-is (pruning skipped)."
        )
        return structures
    else:
        log_message(
            f"{prefix}SP energy pre-screening: {len(structures)} → {len(filtered)} "
            f"(cutoff {cutoff * EV_TO_KCALMOL:.1f} kcal/mol)"
        )
        return filtered


# ============================================================================
# Multi-Structure Parallel Optimization
# ============================================================================


def optimize_multilevel_jax(
    structures: List[Dict],
    energy_fn: Callable,
    fmax: float = 0.01,
    maxiter: int = 1000,
    barrier_fn: Callable = None,
    parallel: bool = True,
    max_stepsize: float = 0.2,
    method: str = "LBFGS",
    return_converged: bool = False,
    potential_wrapper=None,
    fire_dt_start: float = 0.05,
    fire_dt_max: float = 0.1,
    fire_n_min: int = 2,
    *,
    return_info: bool = False,
    coords: str = "cartesian",
    atomic_numbers=None,
    spec=None,
    init_hessian: str = None,
    interfragment: str = "tric",
    frozen_indices=None,
    ric_options: Dict = None,
):
    """Optimize an ensemble of structures.

    Single structure: uses FIRE (default) with full diagnostics.
    Multiple structures: parallel execution via jax.vmap Fover FIRE.
    All structures must have the same number of atoms.

    Args:
        structures: List of structure dicts (with 'positions' key) or position arrays
        energy_fn: JAX energy function positions -> scalar
        fmax: Max force convergence (eV/Å)
        maxiter: Maximum iterations per structure
        barrier_fn: Optional barrier function positions -> scalar.
                    Added to energy during optimization; reported energies exclude it.
        return_info: When True, also return a list of per-structure info dicts
                     (``converged``, ``iterations``, ``final_energy``,
                     ``max_force``).  These are the real values from the
                     optimizer, including on the vmapped batch path.

    Returns:
        List of (structure_dict, energy) tuples; plus ``converged_list`` when
        *return_converged*, plus ``info_list`` when *return_info*.
    """
    start = log_step_start(f"Optimizing structures fmax={fmax:.3g} eV/Å")

    if not structures:
        import warnings

        warnings.warn(
            "[Optimizer] optimize_multilevel_jax called with empty structures list, returning []",
            stacklevel=2,
        )
        return []

    if isinstance(structures[0], dict):
        positions_list = [s["positions"] for s in structures]
        have_metadata = True
        if atomic_numbers is None and "numbers" in structures[0]:
            atomic_numbers = structures[0]["numbers"]
    else:
        positions_list = structures
        have_metadata = False

    n_structures = len(positions_list)
    use_parallel = parallel and n_structures > 1
    log_message(f"[Optimizer] {n_structures} structure(s), parallel={use_parallel}")

    converged_list = []
    info_list = []

    if not use_parallel:
        positions_opt_list = []
        energies_list = []
        for pos in positions_list:
            pos_opt, e_opt, info = optimize_single(
                jnp.asarray(pos),
                energy_fn,
                fmax=fmax,
                maxiter=maxiter,
                barrier_fn=barrier_fn,
                max_stepsize=max_stepsize,
                method=method,
                potential_wrapper=potential_wrapper,
                fire_dt_start=fire_dt_start,
                fire_dt_max=fire_dt_max,
                fire_n_min=fire_n_min,
                coords=coords,
                atomic_numbers=atomic_numbers,
                spec=spec,
                init_hessian=init_hessian,
                interfragment=interfragment,
                frozen_indices=frozen_indices,
                ric_options=ric_options,
            )
            positions_opt_list.append(pos_opt)
            energies_list.append(e_opt)
            converged_list.append(bool(info["converged"]))
            info_list.append(info)
    else:
        positions_batch = jnp.stack([jnp.asarray(p) for p in positions_list])
        (
            positions_opt_batch,
            energies_batch,
            converged_batch,
            n_iter_batch,
            max_force_batch,
        ) = optimize_batch_parallel(
            positions_batch,
            energy_fn,
            fmax=fmax,
            maxiter=maxiter,
            barrier_fn=barrier_fn,
            max_stepsize=max_stepsize,
            method=method,
            potential_wrapper=potential_wrapper,
            fire_dt_start=fire_dt_start,
            fire_dt_max=fire_dt_max,
            fire_n_min=fire_n_min,
            return_info=True,
            coords=coords,
            atomic_numbers=atomic_numbers,
            spec=spec,
            init_hessian=init_hessian,
            interfragment=interfragment,
            frozen_indices=frozen_indices,
            ric_options=ric_options,
        )
        positions_opt_list = [positions_opt_batch[i] for i in range(n_structures)]
        energies_list = [float(energies_batch[i]) for i in range(n_structures)]
        converged_list = [bool(converged_batch[i]) for i in range(n_structures)]
        info_list = [
            {
                "converged": bool(converged_batch[i]),
                "iterations": int(n_iter_batch[i]),
                "final_energy": float(energies_batch[i]),
                "max_force": float(max_force_batch[i]),
                "grad_norm": float(max_force_batch[i]),
            }
            for i in range(n_structures)
        ]

    optimized = []
    for i, (pos_opt, e_opt) in enumerate(zip(positions_opt_list, energies_list)):
        if have_metadata:
            struct_opt = structures[i].copy()
            struct_opt["positions"] = pos_opt
            struct_opt["energy"] = e_opt
        else:
            n_atoms = pos_opt.shape[0]
            struct_opt = {
                "positions": pos_opt,
                "symbols": ["C"] * n_atoms,
                "numbers": jnp.ones(n_atoms, dtype=int) * 6,
                "energy": e_opt,
            }
        optimized.append((struct_opt, e_opt))

    log_message(f"[Optimizer] Completed {n_structures} optimization(s)")
    log_step_end("Optimizing structures", start)

    if return_converged and return_info:
        return optimized, converged_list, info_list
    if return_converged:
        return optimized, converged_list
    if return_info:
        return optimized, info_list
    return optimized


def optimize_batch_parallel(
    positions_batch: jnp.ndarray,
    energy_fn: Callable,
    fmax: float = 0.01,
    maxiter: int = 100,
    barrier_fn: Callable = None,
    max_stepsize: float = 0.2,
    method: str = "LBFGS",
    potential_wrapper=None,
    fire_dt_start: float = 0.05,
    fire_dt_max: float = 0.1,
    fire_n_min: int = 2,
    *,
    return_info: bool = False,
    coords: str = "cartesian",
    atomic_numbers=None,
    spec=None,
    init_hessian: str = None,
    interfragment: str = "tric",
    frozen_indices=None,
    ric_options: Dict = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Optimize multiple structures in parallel using vmap.

    All structures must have the same number of atoms.
    On OOM, automatically splits the batch in half and retries.

    When ``potential_wrapper`` is provided with neighbor lists, structures are
    optimized serially (neighbor updates are not vmap-compatible).

    Args:
        positions_batch: (n_structures, n_atoms, 3) batch of positions
        energy_fn: Energy function
        fmax: Max force convergence (eV/Å)
        maxiter: Max iterations (keep small for parallel efficiency)
        barrier_fn: Optional barrier function positions -> scalar.
        potential_wrapper: Optional PotentialWrapper for neighbor list updates.

    Args (continued):
        return_info: When True, also return per-structure ``n_iter`` and
            ``max_force`` arrays.  Off by default so existing callers keep
            their three-tuple.

    Returns:
        positions_opt: (n_structures, n_atoms, 3)
        energies_opt: (n_structures,)
        converged: (n_structures,) boolean array

        and, when *return_info* is True, additionally
        ``n_iter`` (n_structures,) int and ``max_force`` (n_structures,) eV/Å.
    """
    n = positions_batch.shape[0]
    use_nbrs = potential_wrapper is not None and potential_wrapper.uses_neighbor_lists
    log_message(f"[Optimizer] Parallel batch: {n} structures")

    use_ric = coords == "internal" or init_hessian is not None
    if use_ric and coords != "internal" and frozen_indices is not None and len(frozen_indices) > 0:
        raise ValueError(
            "frozen_indices is not supported with init_hessian: the RFO/BFGS "
            "stepper displaces atoms even when their gradient is zero. Use "
            "init_hessian=None to optimize with frozen atoms."
        )
    if use_ric:
        z = resolve_atomic_numbers(atomic_numbers, potential_wrapper)
        opts = dict(RIC_DEFAULTS)
        opts.update(ric_options or {})
        ok, reason = (
            ric_gate(
                positions_batch[0],
                z,
                method=method,
                barrier_fn=barrier_fn,
                frozen_indices=frozen_indices,
                spec=spec,
                max_atoms=opts["max_atoms"],
                clash_factor=opts["clash_factor"],
            )
            if coords == "internal"
            else (True, "")
        )
        if ok and coords == "internal" and spec is None and z is not None:
            spec = build_coordinates(np.asarray(positions_batch[0]), z, interfragment=interfragment)
            if not spec.valid:
                ok, reason = False, spec.reason
            elif not ric_batch_topology_matches(positions_batch, z, spec):
                ok, reason = False, "batch_topology_mismatch"
        if not ok:
            log_message(f"[RIC] falling back to Cartesian for this batch: {reason}")
            use_ric = False

    if use_ric:
        try:
            pos_o, e_o, conv, iters, mforce, ric_info = _optimize_batch_ric(
                positions_batch,
                energy_fn,
                z,
                fmax=fmax,
                maxiter=maxiter,
                max_stepsize=max_stepsize,
                coords=coords,
                init_hessian=init_hessian,
                interfragment=interfragment,
                spec=spec,
                potential_wrapper=potential_wrapper,
                ric_options=ric_options,
            )
            jax.block_until_ready((pos_o, e_o, conv))
            log_message(
                f"[RIC] batch done: {ric_info['ric_n_internal']} internals, "
                f"{int(np.sum(ric_info['ric_fallback_steps']))} fallback step(s) total"
            )
            if return_info:
                return pos_o, e_o, conv, iters, mforce
            return pos_o, e_o, conv
        except Exception as exc:
            if _is_jax_oom(exc) and n >= 2:
                raise
            log_message(f"[RIC] batch path failed ({exc}); falling back to Cartesian")

    try:
        batch_fn = _make_batch_optimizer(
            energy_fn,
            frozen_indices=frozen_indices,
            fmax=fmax,
            maxiter=maxiter,
            barrier_fn=barrier_fn,
            max_stepsize=max_stepsize,
            method=method,
            potential_wrapper=potential_wrapper,
            fire_dt_start=fire_dt_start,
            fire_dt_max=fire_dt_max,
            fire_n_min=fire_n_min,
        )
        if use_nbrs:
            from .sampling import _allocate_nbrs_batch

            nbr_batch = _allocate_nbrs_batch(positions_batch, potential_wrapper)
            out = batch_fn(positions_batch, nbr_batch)
        else:
            out = batch_fn(positions_batch)

        # Block until computation completes so any OOM is raised here (inside the
        # try/except) rather than deferred to the caller when materializing values.
        jax.block_until_ready(out)

        log_message("[Optimizer] Batch optimization complete")
        positions_opt, energies_opt, converged, n_iter, max_force = out
        if return_info:
            return positions_opt, energies_opt, converged, n_iter, max_force
        return positions_opt, energies_opt, converged

    except Exception as e:
        if not _is_jax_oom(e) or n < 2:
            raise

        mid = n // 2
        log_message(
            f"[Optimizer] OOM with {n} structures, splitting into "
            f"{mid} + {n - mid} and retrying"
        )
        jax.clear_caches()

        # Keyword form on purpose: this recursion used to pass eleven
        # positional arguments, so any new parameter appended to the signature
        # would silently not reach the halved sub-batches.
        def _half(sl):
            return optimize_batch_parallel(
                positions_batch[sl],
                energy_fn,
                fmax=fmax,
                maxiter=maxiter,
                barrier_fn=barrier_fn,
                max_stepsize=max_stepsize,
                method=method,
                potential_wrapper=potential_wrapper,
                fire_dt_start=fire_dt_start,
                fire_dt_max=fire_dt_max,
                fire_n_min=fire_n_min,
                return_info=True,
                coords=coords,
                atomic_numbers=atomic_numbers,
                spec=spec,
                init_hessian=init_hessian,
                interfragment=interfragment,
                frozen_indices=frozen_indices,
                ric_options=ric_options,
            )

        a = _half(slice(None, mid))
        b = _half(slice(mid, None))
        merged = tuple(jnp.concatenate([a[i], b[i]], axis=0) for i in range(5))
        return merged if return_info else merged[:3]


def _spherical_wall_term(pos, center, r_outer, k, exponent):
    """Soft spherical outer-wall energy: ``k·Σ max(0, |r_i-c|/r_outer-1)^n``."""
    d = jnp.linalg.norm(pos - center, axis=1)
    return k * jnp.sum(jnp.maximum(0.0, d / r_outer - 1.0) ** exponent)


def optimize_wall_batch(
    positions_batch: jnp.ndarray,
    energy_fn: Callable,
    r_outer_batch: jnp.ndarray,
    *,
    center,
    wall_k: float,
    wall_exponent: int,
    frozen,
    fmax: float,
    maxiter: int,
    potential_wrapper,
    fire_dt_start: float = 0.05,
    fire_dt_max: float = 0.1,
    fire_n_min: int = 2,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Batched FIRE relaxation with a **per-structure** spherical outer wall.

    Each structure ``s`` is relaxed with ``energy = base(pos) + wall(pos,
    r_outer[s])`` in a single vmapped pass; ``frozen`` atoms (the solute) are
    held via stop-gradient. The per-structure wall radius is threaded through
    jax-md's kwargs, so structures with different ``r_outer`` relax in parallel.

    Used by the barostat's batch moving-wall final packing: call once per wall
    step with the current ``r_outer_batch`` (the caller freezes/steps each
    structure's radius between calls).

    Returns batched ``(pos_opt, base_energy, converged, wall_force)`` where
    ``wall_force[s]`` is the peak per-atom outer-wall reaction force (eV/Å) — the
    barostat's per-structure stop criterion.
    """
    from .sampling import _allocate_nbrs_batch

    center = jnp.asarray(center)
    frozen_idx = jnp.array(list(frozen), dtype=int)
    _, shift_fn = jmd_space.free()
    get_nbr_kw = potential_wrapper.get_neighbor_kwargs

    def total_energy(pos, r_outer=1.0, **nbr_kw):
        if frozen_idx.shape[0] > 0:
            pos_b = pos.at[frozen_idx].set(jax.lax.stop_gradient(pos[frozen_idx]))
        else:
            pos_b = pos
        return energy_fn(pos_b, **nbr_kw) + _spherical_wall_term(
            pos, center, r_outer, wall_k, wall_exponent
        )

    safe_energy = _safe_energy(total_energy)
    fire_init, fire_apply = fire_descent(
        safe_energy, shift_fn, dt_start=fire_dt_start, dt_max=fire_dt_max, n_min=fire_n_min
    )
    energy_jit = jax.jit(energy_fn)

    def optimize_one(pos_init, nbr_state, r_outer):
        state0 = fire_init(pos_init, r_outer=r_outer, **get_nbr_kw(nbr_state))
        max_f0 = jnp.max(jnp.abs(state0.force))

        def cond_fn(carry):
            _, step, max_f, _ = carry
            return jnp.logical_and(step < maxiter, max_f >= fmax)

        def body_fn(carry):
            state, step, _, nbs = carry
            new_state = fire_apply(state, r_outer=r_outer, **get_nbr_kw(nbs))
            nbs = potential_wrapper.update_neighbors(new_state.position, nbs)
            return new_state, step + 1, jnp.max(jnp.abs(new_state.force)), nbs

        final_state, _, final_max_f, final_nbs = jax.lax.while_loop(
            cond_fn, body_fn, (state0, jnp.zeros((), jnp.int32), max_f0, nbr_state)
        )
        pos_opt = final_state.position
        e_base = energy_jit(pos_opt, **get_nbr_kw(final_nbs))
        wall_force = jnp.max(
            jnp.linalg.norm(
                jax.grad(lambda p: _spherical_wall_term(p, center, r_outer, wall_k, wall_exponent))(
                    pos_opt
                ),
                axis=1,
            )
        )
        return pos_opt, e_base, final_max_f < fmax, wall_force

    batched = jax.jit(jax.vmap(optimize_one))

    def _run(pos_b, r_b):
        # On GPU OOM, split the batch in half and recurse — same strategy as
        # optimize_batch_parallel. block_until_ready surfaces the OOM here.
        n = pos_b.shape[0]
        try:
            nbr_b = _allocate_nbrs_batch(pos_b, potential_wrapper)
            out = batched(pos_b, nbr_b, r_b)
            jax.block_until_ready(out)
            return out
        except Exception as exc:
            if not _is_jax_oom(exc) or n < 2:
                raise
            mid = n // 2
            log_message(
                f"[wall-batch] OOM with {n} structures, splitting into "
                f"{mid} + {n - mid} and retrying"
            )
            jax.clear_caches()
            a = _run(pos_b[:mid], r_b[:mid])
            b = _run(pos_b[mid:], r_b[mid:])
            return tuple(jnp.concatenate([a[i], b[i]], axis=0) for i in range(4))

    return _run(positions_batch, r_outer_batch)
