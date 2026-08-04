"""
JAX-MD molecular dynamics sampling.

Provides Nose-Hoover dynamics (from jax-md) with metadynamics bias for conformational sampling.
"""

import math
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import simulate, space, units

from .auto_config import EV_TO_KCALMOL
from .log import log_message, log_step_end, log_step_start, progress_bar
from .mtd import (
    MTDState,
    add_hill,
    compute_mtd_bias,
    create_mtd_state,
    get_mtd_statistics,
    increment_step,
    should_deposit_hill,
)
from .rmsd import center_positions, kabsch_jax
from .utils import _is_jax_oom

unit = units.metal_unit_system()

# ============================================================================
# Metadynamics Sampling
# ============================================================================


def run_mtd_jax(
    positions_init: jnp.ndarray,
    energy_fn: Callable,
    mtd_params: Dict,
    T: float,
    dt: float,
    time_ps: float,
    trajectory_file: Optional[str] = None,
    save_interval: Optional[int] = None,
    mass: Optional[jnp.ndarray] = None,
    random_seed: int = 0,
    potential_wrapper=None,
) -> Tuple[List[jnp.ndarray], MTDState]:
    """Run metadynamics using jax-md Nose-Hoover dynamics.

    This implementation uses momentum-based dynamics (p = m*v) consistent with
    jax-md's Nose-Hoover integrator.

    Args:
        positions_init: (n_atoms, 3) initial positions in Angstrom
        energy_fn: JAX function positions -> energy (in eV)
        mtd_params: dict with keys 'kpush', 'alpha', 'cvdump_fs'
        T: Temperature in Kelvin
        dt: Timestep in femtoseconds
        time_ps: Simulation time in picoseconds
        trajectory_file: Optional path to save trajectory
        save_interval: Steps between trajectory saves (default: cvdump_interval)
        mass: (n_atoms,) per-atom masses in amu, or None for uniform mass 1.0
        random_seed: Random seed for reproducibility
        potential_wrapper: Optional PotentialWrapper instance for neighbor list support

    Returns:
        trajectory: List of position snapshots (numpy arrays)
        mtd_state: Final MTD state

    Example:
        >>> def harmonic_energy(pos):
        ...     return jnp.sum(pos**2) * 0.1
        >>> mtd_params = {'kpush': 0.8, 'alpha': 0.5, 'cvdump_fs': 50.0}
        >>> trajectory, mtd_state = run_mtd_jax(
        ...     positions_init, harmonic_energy, mtd_params, T=300, dt=0.5, time_ps=10.0
        ... )
    """
    start = log_step_start("JAX-MD Metadynamics")

    n_atoms = positions_init.shape[0]
    nsteps = int(time_ps * 1000 / dt)
    cvdump_interval = int(mtd_params["cvdump_fs"] / dt)

    if save_interval is None:
        save_interval = cvdump_interval

    # Force nsteps to be divisible by save_interval (no trailing steps lost)
    n_frames = nsteps // save_interval
    nsteps = n_frames * save_interval

    log_message(
        f"[MTD-JAX] Running {nsteps * dt / 1000:.3g} ps MD with {nsteps} steps, {n_frames} frames"
    )
    log_message(f"[MTD-JAX] Temperature: {T} K, timestep: {dt} fs")
    log_message(
        f"[MTD-JAX] Hill deposition every {cvdump_interval} steps ({mtd_params['cvdump_fs']} fs)"
    )
    log_message(f"[MTD-JAX] Inner JAX loop: {save_interval} steps/frame")
    log_message(
        f"[MTD-JAX] MTD parameters: kpush={mtd_params['kpush'] * EV_TO_KCALMOL:.3g} kcal/mol, alpha={mtd_params['alpha']:.2f} Å⁻²"
    )

    # Use JAX x64 config as the source of truth (positions may have been
    # created before the potential constructor enabled x64, so the array's
    # own dtype can be stale — the energy_fn and integrator follow x64).
    working_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

    # Set initial positions and establish reference frame (first frame, centered)
    positions = positions_init.astype(working_dtype)
    reference_frame = center_positions(positions)
    log_message(f"[MTD-JAX] Reference frame established from initial positions")

    # Initialize MTD state (n_hills and step_counter are jnp.int32 for tracing)
    max_hills = nsteps // cvdump_interval
    mtd_state = create_mtd_state(
        n_atoms=n_atoms,
        max_hills=max_hills,
        kpush=mtd_params["kpush"],
        alpha=mtd_params["alpha"],
        cvdump_interval=cvdump_interval,
        dtype=working_dtype,
        reference_xyz=reference_frame,  # Pass reference frame to MTD state
    )

    # Initialize JAX-MD Nose-Hoover integrator
    _, shift_fn = space.free()
    kT = T * unit["temperature"]
    dt_sim = dt * 1e-3  # fs -> ps

    # Per-atom mass array
    if mass is None:
        mass = jnp.ones(n_atoms)
    mass = jnp.asarray(mass, dtype=working_dtype)

    # Nose-Hoover chain parameters
    chain_length = 3
    chain_steps = 2

    # Initialize neighbor lists with initial positions
    log_message(f"[MTD-JAX] Allocating neighbor lists")
    nbr_state = potential_wrapper.allocate_neighbors(positions)

    # Wrapper function that aligns positions to reference, then computes energy + bias + constraints
    # This is fully differentiable - forces will include alignment transformation
    # Select bias computation method based on use_dynamic_bias flag
    bias_fn = compute_mtd_bias

    @jax.jit
    def base_total_energy_fn(pos, mtd_st=None, **kwargs):
        if mtd_st is None:
            mtd_st = mtd_state
        # Compute base energy and MTD bias on aligned positions
        # Forces will be computed via autodiff through the entire chain
        E_base = energy_fn(pos, **kwargs)
        E_bias = bias_fn(pos, mtd_st)  # Use selected bias function

        return E_base + E_bias

    total_energy_fn = base_total_energy_fn

    # Initialize JAX-MD Nose-Hoover integrator
    init_fn, apply_fn = simulate.nvt_nose_hoover(
        total_energy_fn,
        shift_fn,
        dt=dt_sim * unit["time"],
        kT=kT,
        chain_length=chain_length,
        chain_steps=chain_steps,
    )

    # Initialize state
    key = jax.random.PRNGKey(random_seed)
    state = init_fn(key, positions, mass=mass)

    # JIT-compiled inner loop: run save_interval steps
    @jax.jit
    def run_inner(jax_state, mtd_st, nbrs):
        def body(i, carry):
            s, mtd_st, nbrs = carry

            # Update neighbor lists
            nbrs = potential_wrapper.update_neighbors(s.position, nbrs)
            nbr_kwargs = potential_wrapper.get_neighbor_kwargs(nbrs)

            # Apply Nose-Hoover step with current MTD state and neighbor kwargs
            s = apply_fn(s, mtd_st=mtd_st, kT=kT, **nbr_kwargs)

            # Align to reference frame
            s = replace(s, position=center_positions(s.position))
            R = kabsch_jax(mtd_st.reference_xyz, s.position)
            s = replace(s, position=s.position @ R)

            # MTD: increment step and conditionally deposit hill
            mtd_st = increment_step(mtd_st)
            mtd_st = jax.lax.cond(
                should_deposit_hill(mtd_st),
                lambda st: add_hill(st, s.position),
                lambda st: st,
                mtd_st,
            )

            return (s, mtd_st, nbrs)

        return jax.lax.fori_loop(0, save_interval, body, (jax_state, mtd_st, nbrs))

    # Outer loop: collect trajectory frames (Python-level, one call per frame)
    trajectory = []
    for _ in progress_bar(range(n_frames), total=n_frames, desc="MTD simulation"):
        state, mtd_state, nbr_state = run_inner(state, mtd_state, nbr_state)
        if isinstance(nbr_state, tuple):
            overflow = nbr_state[0].did_buffer_overflow or nbr_state[1].did_buffer_overflow
        else:
            overflow = nbr_state.did_buffer_overflow

        if overflow:
            nbr_state = potential_wrapper.allocate_neighbors(state.position)
        trajectory.append(np.array(state.position))

    # Final statistics
    final_mtd_state = mtd_state
    stats = get_mtd_statistics(final_mtd_state)
    log_message(f"[MTD-JAX] Completed: {stats['n_hills']} hills deposited")
    log_message(f"[MTD-JAX] Memory used: {stats['memory_mb']:.3g} MB")
    log_message(
        f"[MTD-JAX] Utilization: {stats['n_hills']}/{stats['max_hills']} ({100*stats['n_hills']/stats['max_hills']:.1f}%)"
    )

    # Save trajectory if requested
    if trajectory_file:
        log_message(f"[MTD-JAX] Saving trajectory to {trajectory_file}")
        save_trajectory_hdf5(trajectory_file, trajectory, final_mtd_state)

    log_step_end("JAX-MD Metadynamics", start)

    return trajectory, final_mtd_state


# ============================================================================
# Neighbor-list batch allocation helper (shared by parallel MTD and rotamer MD)
# ============================================================================


def _allocate_nbrs_batch(positions_batch: jnp.ndarray, potential_wrapper) -> jnp.ndarray:
    """Allocate stacked neighbor lists for a batch of positions.

    JAX-MD's ``NeighborList.max_occupancy`` is static pytree metadata, so all
    replicas must share the same value before being stacked with
    ``jax.tree_util.tree_map``.  This helper normalises occupancy by directly
    padding the ``idx`` array of under-sized lists with the sentinel value
    (atom index == n_atoms).  This avoids re-allocating via JAX-MD, whose
    sparse-format ``extra_capacity`` is multiplied by N internally and can
    overshoot the target after capacity-limit clamping.

    Args:
        positions_batch: (n_replicas, n_atoms, 3) array of replica positions.
        potential_wrapper: PotentialWrapper instance.

    Returns:
        Stacked neighbor list(s) with leading batch axis of size n_replicas.
    """
    n_replicas = positions_batch.shape[0]
    n_atoms = positions_batch.shape[1]
    nbr_list = [potential_wrapper.allocate_neighbors(positions_batch[i]) for i in range(n_replicas)]

    def _pad_nl(nb, target_max):
        """Pad a single NeighborList's idx to exactly target_max entries."""
        if not hasattr(nb, "max_occupancy"):
            return nb  # Not a NeighborList; no padding needed
        if nb.max_occupancy == target_max:
            return nb
        if nb.max_occupancy > target_max:
            # Truncation: should not occur when target_max is the global max,
            # but handle gracefully just in case.
            new_idx = nb.idx[..., :target_max]
        else:
            pad = target_max - nb.max_occupancy
            sentinel = jnp.full(nb.idx.shape[:-1] + (pad,), n_atoms, dtype=nb.idx.dtype)
            new_idx = jnp.concatenate([nb.idx, sentinel], axis=-1)
        return replace(nb, idx=new_idx, max_occupancy=target_max)

    if isinstance(nbr_list[0], tuple):
        n_components = len(nbr_list[0])
        global_max = [max(nb[j].max_occupancy for nb in nbr_list) for j in range(n_components)]
        nbr_list = [
            tuple(_pad_nl(nb[j], global_max[j]) for j in range(n_components)) for nb in nbr_list
        ]
    else:
        if not hasattr(nbr_list[0], "max_occupancy"):
            return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *nbr_list)
        global_max = max(nb.max_occupancy for nb in nbr_list)
        nbr_list = [_pad_nl(nb, global_max) for nb in nbr_list]

    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *nbr_list)


# ============================================================================
# Rotamer MD Sampling
# ============================================================================


def run_rotamer_md_jax(
    positions_init: jnp.ndarray,
    energy_fn: Callable,
    T: float,
    dt: float,
    time_ps: float,
    save_interval: int = 100,
    mass: Optional[jnp.ndarray] = None,
    random_seed: int = 0,
    potential_wrapper=None,
) -> List[jnp.ndarray]:
    """Run standard NVT molecular dynamics using Nose-Hoover dynamics (no metadynamics bias).

    Used for rotamer sampling at different temperatures.

    Args:
        positions_init: (n_atoms, 3) initial positions
        energy_fn: JAX function positions -> energy
        T: Temperature in Kelvin
        dt: Timestep in femtoseconds
        time_ps: Simulation time in picoseconds
        save_interval: Steps between snapshots
        mass: (n_atoms,) per-atom masses in amu, or None for uniform mass 1.0
        random_seed: Random seed
        potential_wrapper: Optional PotentialWrapper instance for neighbor list support

    Returns:
        trajectory: List of position snapshots

    Example:
        >>> trajectory = run_rotamer_md_jax(
        ...     positions, energy_fn, T=400, dt=0.5, time_ps=5.0
        ... )
    """
    start = log_step_start(f"JAX-MD Rotamer MD (T={T}K)")

    n_atoms = positions_init.shape[0]
    nsteps = int(time_ps * 1000 / dt)

    # Force nsteps to be divisible by save_interval (no trailing steps lost)
    n_frames = nsteps // save_interval
    nsteps = n_frames * save_interval

    log_message(
        f"[Rotamer-JAX] T={T}K, {nsteps * dt / 1000:.3g}ps, {nsteps} steps, {n_frames} frames"
    )
    log_message(f"[Rotamer-JAX] Inner JAX loop: {save_interval} steps/frame")

    # Use JAX x64 config as source of truth (see run_mtd_jax for rationale).
    working_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    positions_init = jnp.asarray(positions_init, dtype=working_dtype)

    # Initialize neighbor lists
    log_message(f"[Rotamer-JAX] Allocating neighbor lists")
    nbr_state = potential_wrapper.allocate_neighbors(positions_init)

    # Initialize JAX-MD
    _, shift_fn = space.free()
    kT = T * unit["temperature"]

    # Per-atom mass array
    if mass is None:
        mass = jnp.ones(n_atoms)
    mass = jnp.asarray(mass, dtype=working_dtype)

    # Use Nose-Hoover dynamics for rotamer MD
    dt_sim = dt * 1e-3  # fs -> ps (JAX-MD uses ps)
    chain_length = 3  # Number of thermostats in the chain
    chain_steps = 2  # Number of chain integration steps per MD step
    init_fn, apply_fn = simulate.nvt_nose_hoover(
        energy_fn,
        shift_fn,
        dt=dt_sim * unit["time"],
        kT=kT,
        chain_length=chain_length,
        chain_steps=chain_steps,
    )

    # Initialize state (jax_md accepts per-atom mass array)
    key = jax.random.PRNGKey(random_seed)
    state = init_fn(key, positions_init, mass=mass)

    # JIT-compiled inner loop: run save_interval steps entirely in XLA
    @jax.jit
    def run_inner(state, nbrs):
        def body(i, carry):
            s, nbrs = carry
            # Update neighbor lists
            nbrs = potential_wrapper.update_neighbors(s.position, nbrs)
            nbr_kwargs = potential_wrapper.get_neighbor_kwargs(nbrs)
            # Apply Nose-Hoover step with neighbor kwargs
            s = apply_fn(s, kT=kT, **nbr_kwargs)
            return (s, nbrs)

        return jax.lax.fori_loop(0, save_interval, body, (state, nbrs))

    # Outer loop: collect trajectory frames
    trajectory = []

    for frame in progress_bar(range(n_frames), total=n_frames, desc=f"Rotamer MD T={T}K"):
        state, nbr_state = run_inner(state, nbr_state)
        # Check for overflow and reallocate if needed
        # Handle both single neighbor list (LJ) and dual neighbor lists (SO3LR)
        overflow = False
        if isinstance(nbr_state, tuple):
            # SO3LR: tuple of (nbrs, nbrs_lr)
            overflow = nbr_state[0].did_buffer_overflow or nbr_state[1].did_buffer_overflow
        else:
            # LJ or harmonic: single neighbor list
            overflow = nbr_state.did_buffer_overflow

        if overflow:
            log_message(f"[Rotamer-JAX] Neighbor list overflow at frame {frame}, reallocating...")
            nbr_state = potential_wrapper.allocate_neighbors(state.position)
        trajectory.append(np.array(state.position))

    log_step_end(f"JAX-MD Rotamer MD (T={T}K)", start)

    return trajectory


def run_rotamer_md_parallel(
    positions_batch: jnp.ndarray,
    energy_fn: Callable,
    temperatures,
    dt: float,
    time_ps: float,
    save_interval: int = 100,
    mass: Optional[jnp.ndarray] = None,
    random_seed: int = 0,
    potential_wrapper=None,
    n_batches: int = 1,
) -> Tuple[List[jnp.ndarray], int]:
    """Run NVT rotamer MD for multiple replicas in parallel using vmap.

    Each replica can start from a different position and run at a different
    temperature.  The inner integrator loop is fully vectorised with
    ``jax.vmap``, so all replicas advance simultaneously on the accelerator.

    Args:
        positions_batch: (n_replicas, n_atoms, 3) initial positions.
        energy_fn: JAX function positions -> energy (eV).
        temperatures: Per-replica temperatures in Kelvin — list or 1-D array
            of length n_replicas.
        dt: Timestep in femtoseconds.
        time_ps: Simulation time in picoseconds.
        save_interval: Steps between trajectory snapshots.
        mass: (n_atoms,) per-atom masses in amu, or None for uniform mass 1.0.
        random_seed: Base random seed; each replica receives a unique key via
            ``jax.random.split``.
        potential_wrapper: PotentialWrapper instance for neighbor list support.

    Returns:
        trajectory: List of (n_replicas, n_atoms, 3) numpy arrays, one per
            saved frame.
        n_batches_used: Actual number of sequential sub-batches executed.
            Returns 1 when all replicas fit in a single vmap call.

    Example:
        >>> traj = run_rotamer_md_parallel(
        ...     pos_batch, energy_fn, temperatures=[300, 400, 500],
        ...     dt=0.5, time_ps=10.0, potential_wrapper=pot
        ... )
    """
    n_replicas = positions_batch.shape[0]
    n_atoms = positions_batch.shape[1]

    # Pre-split path: run n_batches sequential sub-groups to stay within memory.
    if n_batches > 1:
        temps_list = list(temperatures)
        chunk_size = math.ceil(n_replicas / n_batches)
        all_trajs, nb_total = [], 0
        for i in range(0, n_replicas, chunk_size):
            t, nb = run_rotamer_md_parallel(
                positions_batch[i : i + chunk_size],
                energy_fn,
                temps_list[i : i + chunk_size],
                dt,
                time_ps,
                save_interval,
                mass,
                random_seed + i,
                potential_wrapper,
                n_batches=1,
            )
            all_trajs.append(t)
            nb_total += nb
        trajectory = [
            np.concatenate([t[f] for t in all_trajs], axis=0) for f in range(len(all_trajs[0]))
        ]
        return trajectory, nb_total

    nsteps = int(time_ps * 1000 / dt)
    n_frames = nsteps // save_interval
    nsteps = n_frames * save_interval

    # Use JAX x64 config as the source of truth. All MD state (positions,
    # mass, kT) must share the dtype of the energy_fn's outputs: jax-md's
    # update_chain_mass_fn recomputes Q = kT * tau**2 every step, and any
    # mismatch between the integrator state and the force dtype would widen
    # the NHC chain state mid-loop, breaking fori_loop's carry-type invariant.
    # The array's own dtype is unreliable because positions are often built
    # before the potential constructor flips on x64.
    working_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    positions_batch = jnp.asarray(positions_batch, dtype=working_dtype)

    temperatures_arr = jnp.asarray(temperatures, dtype=working_dtype)
    kT_vals = temperatures_arr * jnp.asarray(
        unit["temperature"], dtype=working_dtype
    )  # (n_replicas,)

    log_message(
        f"[Rotamer-JAX] Parallel MD: {n_replicas} replicas, "
        f"{nsteps * dt / 1000:.3f} ps, {n_frames} frames"
    )
    log_message(f"[Rotamer-JAX] Temperatures: {list(temperatures)}")
    log_message(f"[Rotamer-JAX] Inner JAX loop: {save_interval} steps/frame")

    _, shift_fn = space.free()

    if mass is None:
        mass = jnp.ones(n_atoms, dtype=working_dtype)
    mass = jnp.asarray(mass, dtype=working_dtype)

    dt_sim = dt * 1e-3
    # Build integrator using mean kT as reference; per-replica kT is passed
    # explicitly in apply_fn so the chain adapts to each replica's temperature.
    kT_ref = jnp.mean(kT_vals)
    init_fn, apply_fn = simulate.nvt_nose_hoover(
        energy_fn,
        shift_fn,
        dt=dt_sim * unit["time"],
        kT=kT_ref,
        chain_length=3,
        chain_steps=2,
    )

    def _init_single(key, pos):
        return init_fn(key, pos, mass=mass)

    def _apply_single(state, kT_val, **kwargs):
        return apply_fn(state, kT=kT_val, **kwargs)

    vmapped_init = jax.vmap(_init_single)
    vmapped_apply = jax.vmap(_apply_single)
    update_nbrs = jax.vmap(potential_wrapper.update_neighbors)

    @jax.jit
    def run_inner(state, kT_vals, nbrs):
        def body(_, carry):
            s, nbs = carry
            nbs = update_nbrs(s.position, nbs)
            nbr_kwargs = potential_wrapper.get_neighbor_kwargs(nbs)
            s = vmapped_apply(s, kT_vals, **nbr_kwargs)
            return (s, nbs)

        return jax.lax.fori_loop(0, save_interval, body, (state, nbrs))

    try:
        log_message(f"[Rotamer-JAX] Allocating neighbor lists")
        nbr_state = _allocate_nbrs_batch(positions_batch, potential_wrapper)

        keys = jax.random.split(jax.random.PRNGKey(random_seed), n_replicas)
        state = vmapped_init(keys, positions_batch)

        trajectory = []

        for _ in progress_bar(range(n_frames), total=n_frames, desc="Rotamer MD (parallel)"):
            state, nbr_state = run_inner(state, kT_vals, nbr_state)

            if isinstance(nbr_state, tuple):
                overflow = nbr_state[0].did_buffer_overflow | nbr_state[1].did_buffer_overflow
            else:
                overflow = nbr_state.did_buffer_overflow

            if overflow.any():
                log_message(f"[Rotamer-JAX] Neighbor list overflow, reallocating...")
                nbr_state = _allocate_nbrs_batch(state.position, potential_wrapper)

            trajectory.append(np.array(state.position))

        log_message(f"[Rotamer-JAX] Parallel rotamer MD complete")
        return trajectory, 1

    except Exception as e:
        if not _is_jax_oom(e) or n_replicas < 2:
            raise

        mid = n_replicas // 2
        import warnings

        warnings.warn(
            f"[Rotamer-JAX] JAX memory overflow with {n_replicas} replicas. "
            f"Splitting into {mid} + {n_replicas - mid} and retrying.",
            RuntimeWarning,
            stacklevel=2,
        )
        log_message(
            f"[Rotamer-JAX] Memory overflow: retrying with split batches "
            f"({mid} + {n_replicas - mid} replicas)"
        )
        jax.clear_caches()

        temps_list = list(temperatures)
        traj_a, nb_a = run_rotamer_md_parallel(
            positions_batch[:mid],
            energy_fn,
            temps_list[:mid],
            dt,
            time_ps,
            save_interval,
            mass,
            random_seed,
            potential_wrapper,
            n_batches=1,
        )
        traj_b, nb_b = run_rotamer_md_parallel(
            positions_batch[mid:],
            energy_fn,
            temps_list[mid:],
            dt,
            time_ps,
            save_interval,
            mass,
            random_seed + mid,
            potential_wrapper,
            n_batches=1,
        )
        return [np.concatenate([a, b], axis=0) for a, b in zip(traj_a, traj_b)], nb_a + nb_b


def run_multi_temperature_rotamer_md(
    structures: List[Dict],
    energy_fn: Callable,
    base_T: float,
    n_temps: int,
    dt: float,
    time_ps: float,
    n_structures: int = 5,
    mass: Optional[jnp.ndarray] = None,
    potential_wrapper=None,
    n_batches: int = 1,
    parallel: bool = True,
) -> List[List[jnp.ndarray]]:
    """Run rotamer MD on multiple structures at multiple temperatures.

    Implements multi-temperature rotamer sampling: each starting
    structure is propagated under unbiased MD at several temperatures,
    and every resulting trajectory contributes to the ensemble.

    Args:
        structures: List of structure dicts (from ensemble)
        energy_fn: Energy function
        base_T: Base temperature in Kelvin
        n_temps: Number of temperatures (T, T+100, T+200, ...)
        dt: Timestep in femtoseconds
        time_ps: Simulation time per run
        n_structures: Number of lowest-energy structures to sample
        mass: (n_atoms,) per-atom masses in amu, or None for uniform mass 1.0
        potential_wrapper: Optional PotentialWrapper instance for neighbor list support

    Returns:
        all_trajectories: List of trajectories for each (structure, temperature)

    Example:
        >>> ensemble = [...]  # List of (structure, energy)
        >>> all_traj = run_multi_temperature_rotamer_md(
        ...     ensemble, energy_fn, base_T=300, n_temps=3, dt=0.5, time_ps=10.0
        ... )
    """
    temperatures = [base_T + 100 * i for i in range(n_temps)]
    actual_structures = structures[:n_structures]
    n_actual = len(actual_structures)
    n_replicas = n_actual * n_temps

    log_message(
        f"[Rotamer-JAX] Multi-temperature rotamer MD "
        f"({'parallel' if parallel and n_replicas > 1 else 'sequential'})"
    )
    log_message(
        f"[Rotamer-JAX] {n_actual} structures × {n_temps} temperatures " f"= {n_replicas} replicas"
    )
    log_message(f"[Rotamer-JAX] Temperatures: {temperatures}")

    if not parallel or n_replicas == 1:
        # Sequential path: one run_rotamer_md_jax call per (structure, temperature) pair.
        all_trajectories = []
        for i, (structure, energy) in enumerate(actual_structures):
            log_message(
                f"[Rotamer-JAX] Structure {i+1}/{n_actual}, "
                f"E={energy * EV_TO_KCALMOL:.2f} kcal/mol"
            )
            for T in temperatures:
                traj = run_rotamer_md_jax(
                    structure["positions"],
                    energy_fn,
                    T=T,
                    dt=dt,
                    time_ps=time_ps,
                    mass=mass,
                    random_seed=int(i * 1000 + T),
                    potential_wrapper=potential_wrapper,
                )
                all_trajectories.append(traj)
        log_message(f"[Rotamer-JAX] Completed {len(all_trajectories)} rotamer MD runs")
        return all_trajectories

    # Parallel path: stack all (structure, temperature) combinations and vmap.
    positions_list = []
    temps_flat = []
    for i, (structure, energy) in enumerate(actual_structures):
        log_message(
            f"[Rotamer-JAX] Structure {i+1}/{n_actual}, " f"E={energy * EV_TO_KCALMOL:.2f} kcal/mol"
        )
        for T in temperatures:
            positions_list.append(structure["positions"])
            temps_flat.append(T)

    positions_batch = jnp.stack(positions_list)  # (n_replicas, n_atoms, 3)

    traj_frames, _ = run_rotamer_md_parallel(
        positions_batch,
        energy_fn,
        temps_flat,
        dt=dt,
        time_ps=time_ps,
        mass=mass,
        random_seed=0,
        potential_wrapper=potential_wrapper,
        n_batches=n_batches,
    )

    # traj_frames: list of (n_replicas, n_atoms, 3) arrays, one per frame.
    # Reshape to the expected output: one trajectory (list of frames) per replica.
    all_trajectories = [[] for _ in range(n_replicas)]
    for frame in traj_frames:
        for r in range(n_replicas):
            all_trajectories[r].append(frame[r])

    log_message(f"[Rotamer-JAX] Completed {n_replicas} rotamer MD runs")
    return all_trajectories


# ============================================================================
# Trajectory I/O
# ============================================================================


def save_trajectory_hdf5(
    filename: str, trajectory: List[jnp.ndarray], mtd_state: Optional[MTDState] = None
):
    """Save trajectory to HDF5 file.

    Args:
        filename: Output HDF5 file path
        trajectory: List of position arrays
        mtd_state: Optional MTD state to save alongside
    """
    import h5py

    with h5py.File(filename, "w") as f:
        # Save trajectory
        n_frames = len(trajectory)
        n_atoms, _ = trajectory[0].shape

        traj_data = np.array(trajectory)  # (n_frames, n_atoms, 3)
        f.create_dataset("trajectory", data=traj_data)

        # Metadata
        f.attrs["n_frames"] = n_frames
        f.attrs["n_atoms"] = n_atoms

        # Save MTD state if provided
        if mtd_state is not None:
            grp = f.create_group("mtd_state")
            grp.create_dataset("cv_xyz", data=np.array(mtd_state.cv_xyz))
            grp.create_dataset("reference_xyz", data=np.array(mtd_state.reference_xyz))
            grp.attrs["n_hills"] = mtd_state.n_hills
            grp.attrs["kpush"] = mtd_state.kpush
            grp.attrs["alpha"] = mtd_state.alpha


def load_trajectory_hdf5(filename: str) -> Tuple[List[jnp.ndarray], Optional[MTDState]]:
    """Load trajectory from HDF5 file.

    Args:
        filename: Input HDF5 file path

    Returns:
        trajectory: List of position arrays
        mtd_state: MTD state if saved, else None
    """
    import h5py

    with h5py.File(filename, "r") as f:
        traj_data = f["trajectory"][:]
        trajectory = [jnp.array(frame) for frame in traj_data]

        # Load MTD state if exists
        mtd_state = None
        if "mtd_state" in f:
            from .mtd import MTDState
            from .rmsd import center_positions

            grp = f["mtd_state"]
            cv_xyz = jnp.array(grp["cv_xyz"][:])
            if "reference_xyz" in grp:
                reference_xyz = jnp.array(grp["reference_xyz"][:])
            else:
                reference_xyz = center_positions(cv_xyz[0])
            mtd_state = MTDState(
                cv_xyz=cv_xyz,
                reference_xyz=reference_xyz,
                n_hills=int(grp.attrs["n_hills"]),
                kpush=float(grp.attrs["kpush"]),
                alpha=float(grp.attrs["alpha"]),
                cvdump_interval=100,  # Default
                step_counter=0,
            )

    return trajectory, mtd_state


# ============================================================================
# Wrapper for Backward Compatibility
# ============================================================================


def run_multi_mtd(
    positions_init: jnp.ndarray,
    energy_fn: Callable,
    mtd_params_list: List[Dict],
    T: float,
    dt: float,
    time_ps: float,
    mass: Optional[jnp.ndarray] = None,
    potential_wrapper=None,
    atomic_numbers: Optional[jnp.ndarray] = None,
) -> Tuple[List[jnp.ndarray], List[MTDState]]:
    """Run multiple MTD simulations with different (kpush, alpha) parameters.

    Each MTD uses a different bias strength/width to explore different
    conformational regimes — the union of trajectories covers a wider
    set of basins than any single parameter setting could.

    Args:
        positions_init: (n_atoms, 3) initial positions in Angstrom
        energy_fn: JAX function positions -> energy (in eV)
        mtd_params_list: List of dicts, each with 'kpush', 'alpha', 'cvdump_fs'
        T: Temperature in Kelvin
        dt: Timestep in femtoseconds
        time_ps: Simulation time per MTD in picoseconds
        mass: (n_atoms,) per-atom masses in amu, or None for uniform mass 1.0
        potential_wrapper: Optional PotentialWrapper instance for neighbor list support

    Returns:
        all_snapshots: Combined trajectory from all MTDs (deduplicated by step)
        mtd_states: List of final MTD states for each simulation

    Example:
        >>> from mars.auto_config import generate_mtd_parameters
        >>> params_list = generate_mtd_parameters(n_atoms=42, mode='normal')
        >>> snapshots, states = run_multi_mtd(
        ...     positions, energy_fn, params_list, T=300, dt=0.5, time_ps=10.0
        ... )
    """
    n_mtd = len(mtd_params_list)
    log_message(f"[Multi-MTD] Running {n_mtd} MTD simulations ({time_ps} ps each)")

    all_snapshots = []
    mtd_states = []

    for i, params in enumerate(mtd_params_list):
        log_message(
            f"[Multi-MTD] MTD {i+1}/{n_mtd}: "
            f"kpush={params['kpush']:.6f}, alpha={params['alpha']:.4f}"
        )

        trajectory, mtd_state = run_mtd_jax(
            positions_init,
            energy_fn,
            params,
            T=T,
            dt=dt,
            time_ps=time_ps,
            mass=mass,
            random_seed=i * 42 + 7,
            potential_wrapper=potential_wrapper,
        )

        all_snapshots.extend(trajectory)
        mtd_states.append(mtd_state)

        log_message(
            f"[Multi-MTD] MTD {i+1}: {len(trajectory)} snapshots, " f"{mtd_state.n_hills} hills"
        )

    log_message(f"[Multi-MTD] Total: {len(all_snapshots)} snapshots from {n_mtd} MTDs")

    return all_snapshots, mtd_states


def run_mtd_sampling(
    positions: jnp.ndarray,
    energy_fn: Callable,
    T: float = 400,
    dt: float = 0.5,
    time_ps: float = 2.0,
    cycle: int = 1,
    params: Optional[Dict] = None,
    mass: Optional[jnp.ndarray] = None,
    potential_wrapper=None,
) -> Tuple[List[jnp.ndarray], MTDState]:
    """Wrapper for run_mtd_jax with ASE-compatible defaults.

    Args:
        positions: Initial positions
        energy_fn: Energy function
        T: Temperature (default: 400 K)
        dt: Timestep (default: 0.5 fs)
        time_ps: Simulation time (default: 2.0 ps)
        cycle: Cycle number for trajectory naming
        params: Optional MTD parameters dict
        mass: (n_atoms,) per-atom masses in amu, or None for uniform mass 1.0
        potential_wrapper: Optional PotentialWrapper instance for neighbor list support

    Returns:
        trajectory: List of positions
        mtd_state: Final MTD state
    """
    mtd_params = {"kpush": 0.8, "alpha": 0.5, "cvdump_fs": 50.0}
    if params is not None:
        mtd_params.update(params)

    trajectory_file = f"mtd_{cycle}.h5" if cycle else None

    return run_mtd_jax(
        positions,
        energy_fn,
        mtd_params,
        T,
        dt,
        time_ps,
        trajectory_file=trajectory_file,
        mass=mass,
        random_seed=cycle * 42,
        potential_wrapper=potential_wrapper,
    )


# ============================================================================
# Diagnostics
# ============================================================================


def analyze_trajectory(trajectory: List[jnp.ndarray]) -> Dict:
    """Analyze trajectory and compute statistics.

    Args:
        trajectory: List of position arrays

    Returns:
        stats: Dictionary with trajectory statistics
    """
    n_frames = len(trajectory)
    n_atoms = trajectory[0].shape[0]

    # Compute RMSD to first frame
    from .rmsd import rmsd_cv_jax

    rmsds = [float(rmsd_cv_jax(frame, trajectory[0])) for frame in trajectory]

    # Compute radius of gyration
    def radius_of_gyration(pos):
        centered = pos - jnp.mean(pos, axis=0)
        return jnp.sqrt(jnp.mean(jnp.sum(centered**2, axis=1)))

    rg_values = [float(radius_of_gyration(frame)) for frame in trajectory]

    stats = {
        "n_frames": n_frames,
        "n_atoms": n_atoms,
        "rmsd_mean": np.mean(rmsds),
        "rmsd_std": np.std(rmsds),
        "rmsd_max": np.max(rmsds),
        "rg_mean": np.mean(rg_values),
        "rg_std": np.std(rg_values),
    }

    return stats


# ============================================================================
# Parallelization MTDs
# ============================================================================


def run_mtd_parallel(
    positions_init: jnp.ndarray,
    energy_fn: Callable,
    mtd_params: List[Dict],
    T: float,
    dt: float,
    time_ps: float,
    trajectory_file: Optional[str] = None,
    save_interval: Optional[int] = None,
    mass: Optional[jnp.ndarray] = None,
    random_seed: int = 0,
    potential_wrapper=None,
    n_batches: int = 1,
):
    """
    Parallel metadynamics using vmap over multiple MTD parameter sets.
    Each entry in mtd_params runs one independent MTD replica.
    """

    n_replicas = len(mtd_params)

    # Detect multi-start: positions_init may be (n_replicas, n_atoms, 3) with distinct
    # starting configurations, or the usual (n_atoms, 3) broadcast to all replicas.
    _multi_start = positions_init.ndim == 3

    # Pre-split path: run n_batches sequential sub-groups to stay within memory.
    if n_batches > 1:
        chunk_size = math.ceil(n_replicas / n_batches)
        trajs, states, nb_total = [], [], 0
        for i in range(0, n_replicas, chunk_size):
            chunk = mtd_params[i : i + chunk_size]
            pos_chunk = positions_init[i : i + len(chunk)] if _multi_start else positions_init
            t, s, nb = run_mtd_parallel(
                pos_chunk,
                energy_fn,
                chunk,
                T,
                dt,
                time_ps,
                None,
                save_interval,
                mass,
                random_seed + i,
                potential_wrapper,
                n_batches=1,
            )
            trajs.append(t)
            states.append(s)
            nb_total += nb
        trajectory = [np.concatenate([t[f] for t in trajs], axis=0) for f in range(len(trajs[0]))]
        final_mtd_state = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *states)
        if trajectory_file:
            save_trajectory_hdf5(trajectory_file, trajectory, final_mtd_state)
        return trajectory, final_mtd_state, nb_total

    # n_atoms and positions batch
    if _multi_start:
        n_atoms = positions_init.shape[1]
        positions = jnp.asarray(positions_init)  # already (n_replicas, n_atoms, 3)
    else:
        n_atoms = positions_init.shape[0]
        # Expand positions to shape (n_replicas, n_atoms, 3)
        positions = jnp.repeat(positions_init[None, :, :], repeats=n_replicas, axis=0)

    # Use JAX x64 config as source of truth (see run_rotamer_md_parallel for rationale).
    working_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    positions = jnp.asarray(positions, dtype=working_dtype)

    nsteps = int(time_ps * 1000 / dt)
    cvdump_intervals = jnp.array([p["cvdump_fs"] for p in mtd_params]) / dt

    # Ensure integer types
    cvdump_intervals = cvdump_intervals.astype(int)

    if save_interval is None:
        save_interval = int(cvdump_intervals.min())
    save_interval = int(save_interval)

    n_frames = nsteps // save_interval
    nsteps = n_frames * save_interval

    # Reference frames: one per replica (distinct starts get their own centred frame)
    if _multi_start:
        reference_frames = [center_positions(positions_init[i]) for i in range(n_replicas)]
    else:
        _ref = center_positions(positions_init)
        reference_frames = [_ref] * n_replicas

    log_message(
        f"[MTD-JAX] Running {n_replicas} x {nsteps * dt / 1000:.3g} ps parallel MD with {nsteps} steps, {n_frames} frames"
    )
    log_message(f"[MTD-JAX] Temperature: {T} K, timestep: {dt} fs")
    log_message(
        f"[MTD-JAX] Hill deposition every {cvdump_intervals[0]} steps ({mtd_params[0]['cvdump_fs']} fs)"
    )
    log_message(f"[MTD-JAX] Inner JAX loop: {save_interval} steps/frame")
    # ------------------------------
    # Build vectorized MTD states
    # ------------------------------
    max_hills = nsteps // int(cvdump_intervals.min())

    def _create_state(param, ref_frame):
        return create_mtd_state(
            n_atoms=n_atoms,
            max_hills=max_hills,
            kpush=param["kpush"],
            alpha=param["alpha"],
            cvdump_interval=int(param["cvdump_fs"] / dt),
            dtype=working_dtype,
            reference_xyz=ref_frame,
        )

    # shape: (n_replicas,)
    mtd_states = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs),
        *[_create_state(p, r) for p, r in zip(mtd_params, reference_frames)],
    )

    # ------------------------------
    # Single-replica energy + bias (NOT vmapped)
    # ------------------------------
    def total_energy_fn_single(pos, mtd_st=None, **kwargs):
        E_base = energy_fn(pos, **kwargs)
        if mtd_st is None:
            return E_base
        return E_base + compute_mtd_bias(pos, mtd_st)

    # ------------------------------
    # Build Nose–Hoover integrator, then vmap
    # ------------------------------
    _, shift_fn = space.free()
    kT = T * unit["temperature"]
    dt_sim = dt * 1e-3

    # Build per-atom mass (before closures that capture it)
    if mass is None:
        mass = jnp.ones(n_atoms)
    mass = mass.astype(working_dtype)

    # Pass un-vmapped energy fn so nvt_nose_hoover computes grad(f) per replica
    init_fn, apply_fn = simulate.nvt_nose_hoover(
        total_energy_fn_single,
        shift_fn,
        dt=dt_sim * unit["time"],
        kT=kT,
        chain_length=3,
        chain_steps=2,
    )

    # Vmap integrator — closures capture mass and kT so they are not batched
    def _init_single(key, pos):
        return init_fn(key, pos, mass=mass)

    def _apply_single(state, mtd_st, **kwargs):
        return apply_fn(state, mtd_st=mtd_st, kT=kT, **kwargs)

    vmapped_init = jax.vmap(_init_single)
    vmapped_apply = jax.vmap(_apply_single)

    # ------------------------------
    # Neighbor list helpers
    # ------------------------------
    # allocate_neighbors requires concrete values (int() for buffer sizes),
    # so it cannot be vmapped — use the module-level helper.
    def allocate_nbrs_batch(positions_batch):
        return _allocate_nbrs_batch(positions_batch, potential_wrapper)

    # update_neighbors only fills existing buffers with JAX ops — safe to vmap.
    update_nbrs = jax.vmap(potential_wrapper.update_neighbors)

    log_message(f"[MTD-JAX] Allocating neighbor lists")

    # ------------------------------
    # Inner integrator loop (vectorized)
    # ------------------------------
    @jax.jit
    def run_inner(state, mtd_st, nbrs):

        def body(_, carry):
            s, mtd_s, nbs = carry

            # Neighbor list update
            nbs = update_nbrs(s.position, nbs)
            nbr_kwargs = potential_wrapper.get_neighbor_kwargs(nbs)

            # One Nose–Hoover step
            s = vmapped_apply(s, mtd_s, **nbr_kwargs)

            # Alignment (vectorized)
            centered = jax.vmap(center_positions)(s.position)
            R = jax.vmap(kabsch_jax)(mtd_s.reference_xyz, centered)

            new_pos = jax.vmap(lambda x, r: x @ r)(centered, R)
            s = replace(s, position=new_pos)

            # MTD: step + deposit
            mtd_s = increment_step(mtd_s)
            mtd_s = jax.vmap(
                lambda st, pos: jax.lax.cond(
                    should_deposit_hill(st), lambda st2: add_hill(st2, pos), lambda st2: st2, st
                )
            )(mtd_s, s.position)

            return s, mtd_s, nbs

        return jax.lax.fori_loop(0, save_interval, body, (state, mtd_st, nbrs))

    # ------------------------------
    # Outer loop (Python) — wrapped for OOM recovery
    # ------------------------------
    try:
        nbr_state = allocate_nbrs_batch(positions)

        keys = jax.random.split(jax.random.PRNGKey(random_seed), n_replicas)
        state = vmapped_init(keys, positions)

        trajectory = []

        for _ in progress_bar(range(n_frames), total=n_frames, desc="MTD simulation"):
            state, mtd_states, nbr_state = run_inner(state, mtd_states, nbr_state)

            # Overflow handling (supports both single and tuple neighbor lists)
            if isinstance(nbr_state, tuple):
                overflow = nbr_state[0].did_buffer_overflow | nbr_state[1].did_buffer_overflow
            else:
                overflow = nbr_state.did_buffer_overflow

            if overflow.any():
                nbr_state = allocate_nbrs_batch(state.position)

            trajectory.append(np.array(state.position))

        final_mtd_state = mtd_states

        if trajectory_file:
            save_trajectory_hdf5(trajectory_file, trajectory, final_mtd_state)

        log_message(f"[MTD-JAX] Completed: {n_replicas}")

        return trajectory, final_mtd_state, 1

    except Exception as e:
        if not _is_jax_oom(e) or n_replicas < 2:
            raise

        mid = n_replicas // 2
        import warnings

        warnings.warn(
            f"[MTD-JAX] JAX memory overflow with {n_replicas} replicas. "
            f"Splitting into batches of {mid} and {n_replicas - mid} and retrying.",
            RuntimeWarning,
            stacklevel=2,
        )
        log_message(
            f"[MTD-JAX] Memory overflow: retrying with split batches "
            f"({mid} + {n_replicas - mid} replicas)"
        )
        jax.clear_caches()

        traj_a, state_a, nb_a = run_mtd_parallel(
            positions_init[:mid] if _multi_start else positions_init,
            energy_fn,
            mtd_params[:mid],
            T,
            dt,
            time_ps,
            None,
            save_interval,
            mass,
            random_seed,
            potential_wrapper,
            n_batches=1,
        )
        traj_b, state_b, nb_b = run_mtd_parallel(
            positions_init[mid:] if _multi_start else positions_init,
            energy_fn,
            mtd_params[mid:],
            T,
            dt,
            time_ps,
            None,
            save_interval,
            mass,
            random_seed + mid,
            potential_wrapper,
            n_batches=1,
        )

        # Merge: each trajectory is a list of (batch, n_atoms, 3) arrays
        trajectory = [np.concatenate([a, b], axis=0) for a, b in zip(traj_a, traj_b)]

        # Merge MTD states: pytree with (n_replicas, ...) leaves
        final_mtd_state = jax.tree_util.tree_map(
            lambda a, b: jnp.concatenate([a, b], axis=0),
            state_a,
            state_b,
        )

        if trajectory_file:
            save_trajectory_hdf5(trajectory_file, trajectory, final_mtd_state)

        return trajectory, final_mtd_state, nb_a + nb_b
