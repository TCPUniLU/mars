"""JAX-based structure optimization.

Provides geometry optimization (minimization and transition state search)
with JIT-compiled implementations for GPU acceleration.

- Single structure: LBFGS (default) via jaxopt, or FIRE via jax-md, GD via jaxopt
- Multiple structures: parallel via jax.vmap over LBFGS (or FIRE)
- Stationary point search: damped Newton eigenvector-following (JIT-compiled)
- All convergence criteria use max force (eV/Å) with default 0.01
"""

from typing import Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
from jax_md import space as jmd_space
from jax_md.minimize import fire_descent
from jaxopt import LBFGS, GradientDescent

from .auto_config import EV_TO_KCALMOL
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
):
    """JIT+vmap optimizer for parallel batch optimization.

    When ``potential_wrapper`` is provided, neighbor lists are updated inside
    the JIT-compiled optimization loop (vmapped across all structures).

    Returns ``f(pos_batch, nbr_batch) -> (pos_opt, energies, converged)``
    when neighbor lists are used, or ``f(pos_batch) -> (...)`` otherwise.
    """
    energy_jit = jax.jit(energy_fn)
    use_nbrs = potential_wrapper is not None and potential_wrapper.uses_neighbor_lists

    if barrier_fn is not None:
        opt_energy = jax.jit(_safe_energy(lambda pos, **kw: energy_fn(pos, **kw) + barrier_fn(pos)))
    else:
        opt_energy = jax.jit(_safe_energy(energy_fn))

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

                final_state, _, final_max_f, final_nbs = jax.lax.while_loop(
                    cond_fn, body_fn, (state0, jnp.zeros((), dtype=jnp.int32), max_f0, nbr_state)
                )

                pos_opt = final_state.position
                final_kw = get_nbr_kw(final_nbs)
                return pos_opt, energy_jit(pos_opt, **final_kw), final_max_f < fmax

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
                pos_opt, _, converged = fire_fn(pos_init)
                return pos_opt, energy_jit(pos_opt), converged

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

                pos, state, _, final_nbs = jax.lax.while_loop(
                    cond_fn, body_fn, (pos_init, state, jnp.zeros((), dtype=jnp.int32), nbr_state)
                )
                final_kw = get_nbr_kw(final_nbs)
                converged = state.error < fmax
                return pos, energy_jit(pos, **final_kw), converged

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
                return result.params, energy_jit(result.params), converged

            return jax.jit(jax.vmap(optimize_one))

    elif method == "SP":
        sp_fn = _make_stationary_search_fn(
            opt_energy, fmax=fmax, maxiter=maxiter, trust_radius=max_stepsize
        )

        @jax.jit
        def optimize_one(pos_init):
            pos_opt, _, _, converged = sp_fn(pos_init)
            return pos_opt, energy_jit(pos_opt), converged

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

                pos, state, _, final_nbs = jax.lax.while_loop(
                    cond_fn, body_fn, (pos_init, state, jnp.zeros((), dtype=jnp.int32), nbr_state)
                )
                final_kw = get_nbr_kw(final_nbs)
                converged = state.error < fmax
                return pos, energy_jit(pos, **final_kw), converged

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

                pos, state, _ = jax.lax.while_loop(
                    cond_fn, body_fn, (pos_init, state, jnp.zeros((), dtype=jnp.int32))
                )
                converged = state.error < fmax
                return pos, energy_jit(pos), converged

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
    if barrier_fn is not None:
        opt_energy_fn = _safe_energy(lambda pos, **kw: energy_fn(pos, **kw) + barrier_fn(pos))
    else:
        opt_energy_fn = _safe_energy(energy_fn)

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
        **kwargs: Forwarded to ``optimize_single`` (max_stepsize, method,
                  potential_wrapper, etc.)

    Returns:
        (positions_opt, energy_opt)
    """
    start = log_step_start(f"Optimizing structure {structure_id}")
    max_stepsize = kwargs.get("max_stepsize", 0.2)
    method = kwargs.get("method", "LBFGS")
    potential_wrapper = kwargs.get("potential_wrapper", None)
    maxiter = kwargs.get("maxiter", 1000)

    positions_opt, energy_opt, info = optimize_single(
        positions_init,
        energy_fn,
        fmax=fmax,
        maxiter=maxiter,
        max_stepsize=max_stepsize,
        method=method,
        potential_wrapper=potential_wrapper,
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

    Returns:
        List of (structure_dict, energy) tuples
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
    else:
        positions_list = structures
        have_metadata = False

    n_structures = len(positions_list)
    use_parallel = parallel and n_structures > 1
    log_message(f"[Optimizer] {n_structures} structure(s), parallel={use_parallel}")

    converged_list = []

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
            )
            positions_opt_list.append(pos_opt)
            energies_list.append(e_opt)
            converged_list.append(bool(info["converged"]))
    else:
        positions_batch = jnp.stack([jnp.asarray(p) for p in positions_list])
        positions_opt_batch, energies_batch, converged_batch = optimize_batch_parallel(
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
        )
        positions_opt_list = [positions_opt_batch[i] for i in range(n_structures)]
        energies_list = [float(energies_batch[i]) for i in range(n_structures)]
        converged_list = [bool(converged_batch[i]) for i in range(n_structures)]

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

    if return_converged:
        return optimized, converged_list
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

    Returns:
        positions_opt: (n_structures, n_atoms, 3)
        energies_opt: (n_structures,)
        converged: (n_structures,) boolean array
    """
    n = positions_batch.shape[0]
    use_nbrs = potential_wrapper is not None and potential_wrapper.uses_neighbor_lists
    log_message(f"[Optimizer] Parallel batch: {n} structures")

    try:
        batch_fn = _make_batch_optimizer(
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
        )
        if use_nbrs:
            from .sampling import _allocate_nbrs_batch

            nbr_batch = _allocate_nbrs_batch(positions_batch, potential_wrapper)
            positions_opt, energies_opt, converged = batch_fn(positions_batch, nbr_batch)
        else:
            positions_opt, energies_opt, converged = batch_fn(positions_batch)

        # Block until computation completes so any OOM is raised here (inside the
        # try/except) rather than deferred to the caller when materializing values.
        jax.block_until_ready((positions_opt, energies_opt, converged))

        log_message("[Optimizer] Batch optimization complete")
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

        pos_a, eng_a, conv_a = optimize_batch_parallel(
            positions_batch[:mid],
            energy_fn,
            fmax,
            maxiter,
            barrier_fn,
            max_stepsize,
            method,
            potential_wrapper,
            fire_dt_start,
            fire_dt_max,
            fire_n_min,
        )
        pos_b, eng_b, conv_b = optimize_batch_parallel(
            positions_batch[mid:],
            energy_fn,
            fmax,
            maxiter,
            barrier_fn,
            max_stepsize,
            method,
            potential_wrapper,
            fire_dt_start,
            fire_dt_max,
            fire_n_min,
        )

        return (
            jnp.concatenate([pos_a, pos_b], axis=0),
            jnp.concatenate([eng_a, eng_b], axis=0),
            jnp.concatenate([conv_a, conv_b], axis=0),
        )


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
