"""
Multi-step MTD workflow orchestrator (JAX implementation).

This module implements a multi-step MTD conformational search workflow:
  Phase 1: Multi-step MTD grid (different kpush/alpha per step)
  Phase 2: Conformer refinement from diverse starts (normal/thorough)
  Phase 3: Rotamer MD (optional)
  Phase 4: Genetic crossing (optional)
"""

import math
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np

from .auto_config import EV_TO_KCALMOL, _print_auto_config, auto_setup, run_trial_mtd
from .ensemble import prune_by_energy_window, sort_by_energy
from .log import log_energy_table, log_header, log_message, timer_end, timer_start
from .optimizer import (
    FMAX_MAP,
    optimize_multilevel_jax,
    optimize_single_with_logging,
    prescreen_by_energy,
)
from .sampling import (
    run_mtd_jax,
    run_mtd_parallel,
    run_multi_temperature_rotamer_md,
    run_rotamer_md_jax,
)
from .utils import create_structure, get_atomic_masses, save_ensemble, save_trajectory_xyz
from .utils_prune import (
    build_nci_cross_mask,
    min_inter_fragment_dist,
    prune_by_nci_distance,
    prune_by_rmsd,
    prune_by_rotational_constants,
    prune_by_topology,
)

# Adaptive distance threshold factors per optimization cycle (mirrors ewin scaling)
_NCI_DIST_FACTORS = {1: 2.0, 2: 1.5, 3: 1.0}


def _prune_energy_window_guarded(
    ensemble: List[Tuple[Dict, float]],
    ewin: float,
    is_last_cycle: bool,
    cycle_i: int,
    n_cycles: int,
    label: str,
) -> List[Tuple[Dict, float]]:
    """Energy-window prune with a coarse-cycle safety net.

    In the early (non-final) optimization cycles the geometries are only
    loosely optimized (small ``maxiter`` / large step caps), so their energies
    are far from converged. An energy window applied to those unreliable
    energies can collapse the whole ensemble down to the single lowest
    structure — which may then break or vanish in the next cycle and abort the
    run (observed with MACE-OFF on flexible molecules).

    When energy pruning would wipe out essentially everything in a *non-final*
    cycle, emit a warning and keep all structures so the next cycle
    re-optimizes them and prunes on better-converged geometries. The final
    cycle always prunes normally — by then the energies are trustworthy.
    """
    before = ensemble
    pruned = prune_by_energy_window(before, ewin=ewin)
    if (not is_last_cycle) and len(before) > 1 and len(pruned) <= 1:
        warnings.warn(
            f"[{label}] Cycle {cycle_i}/{n_cycles}: energy-window pruning "
            f"(Ewin={ewin * EV_TO_KCALMOL:.1f} kcal/mol) collapsed {len(before)} "
            f"coarsely-optimized structures to {len(pruned)}. Coarse-cycle energies "
            "are unreliable, so all structures are kept for the next optimization "
            "cycle, where pruning runs on better-converged geometries.",
            UserWarning,
            stacklevel=2,
        )
        log_message(
            f"[{label}] Cycle {cycle_i}/{n_cycles}: WARNING — energy pruning would keep "
            f"only {len(pruned)}/{len(before)} structures; keeping all {len(before)} "
            "for the next optimization cycle instead."
        )
        return before
    return pruned


def _ric_kwargs(
    coords, coords_coarse, init_hessian, interfragment, ric_options, numbers, cycle=None
):
    """Coordinate-system kwargs for one optimizer call in the ladder.

    Cycles 1-2 use *coords_coarse* (Cartesian by default).  They run on
    clashed metadynamics or genetic-crossing snapshots with 1.0 A / 0.5 A step
    caps and 10 / 100 iterations: a topology detected from such a geometry
    invents spurious bonds, a 1 A cap is ~30 degrees on a dihedral and far
    outside the back-transformation's local linearisation, and ten iterations
    cannot amortise the model-Hessian build.  Those cycles also never discard
    non-converged structures -- only the final one does -- so their job is
    just to get roughly downhill cheaply, which Cartesian already does
    clash-robustly.

    No coordinate spec is cached across cycles on purpose: letting each call
    rebuild from the *current* geometries is what makes the final cycle use a
    topology derived from the relaxed structure rather than the raw snapshot.
    """
    use = coords if (cycle is None or cycle >= 3) else coords_coarse
    return dict(
        coords=use,
        atomic_numbers=numbers,
        init_hessian=init_hessian,
        interfragment=interfragment,
        ric_options=ric_options,
    )


def run_conformer_search_jax(
    initial_structure: Dict,
    energy_fn: Callable,
    mode: str = "normal",
    temperature: Optional[float] = None,
    dt: Optional[float] = None,
    ewin: Optional[float] = None,
    optlevel: Optional[int] = None,
    rmsd_threshold: Optional[float] = None,
    mtd_params: Optional[Dict] = None,
    mtr_rotamer_number: Optional[int] = None,
    mtr_temps: Optional[int] = None,
    mtr_time_ps: Optional[float] = None,
    genetic_crossing: Optional[bool] = None,
    n_children: Optional[int] = None,
    mtd_kscal: float = 1.0,
    trial: bool = True,
    save_trajectory: bool = False,
    potential_wrapper=None,
    use_topology: bool = True,
    n_conformer_starts: Optional[int] = None,
    exploration_time_ps: Optional[float] = None,
    conformer_time_ps: Optional[float] = None,
    parallel: bool = True,
    nci: bool = False,
    nci_max_dist: float = 3.0,
    fire_dt_start: float = 0.05,
    fire_dt_max: float = 0.1,
    fire_n_min: int = 2,
    method: str = "LBFGS",
    final_fmax: Optional[float] = None,
    final_maxiter: Optional[int] = None,
    final_max_stepsize: Optional[float] = None,
    output_file: str = "auto_final_ensemble.xyz",
    coords: str = "cartesian",
    coords_coarse: str = "cartesian",
    init_hessian: Optional[str] = None,
    interfragment: str = "tric",
    ric_options: Optional[Dict] = None,
) -> List[Tuple[Dict, float]]:
    """Run Conformer MTD conformational search with automatic configuration.

    This is the main entry point. All parameters are auto-configured based on
    molecular properties and the chosen mode (quick/normal/thorough).

    Args:
        initial_structure: Structure dict with 'positions', 'symbols', 'numbers'
        energy_fn: JAX function positions -> energy (in eV)
        mode: Search mode - 'quick', 'normal', or 'thorough' (default: 'normal')
        temperature: Temperature in K (default: 300)
        dt: Initial timestep in fs (default: 0.5)
        ewin: Energy window in eV (None uses auto-determined value)
        optlevel: Optimization level -3 to 2 (None uses auto-determined value)
        rmsd_threshold: RMSD threshold in Angstrom (None uses auto-determined value)
        mtd_params: Manual MTD parameters dict with 'kpush' (eV), 'alpha' (1/A^2),
                    'cvdump_fs' (fs). Overrides auto-configured bias for both phases.
        mtr_rotamer_number: Number of structures for rotamer MD (None uses auto-determined)
        mtr_temps: Number of temperature replicas for rotamer MD (None uses auto-determined)
        mtr_time_ps: Rotamer MD time in ps (None uses auto-determined)
        genetic_crossing: Enable genetic crossing (None uses auto-determined from mode)
        n_children: Number of children for genetic crossing (None uses auto-determined)
        mtd_kscal: Global kpush scaling factor (default: 1.0)
        trial: Run trial MTD to validate settings (default: True)
        save_trajectory: Save MD trajectories as XYZ files (default: False)
        use_topology: Enable bond topology pruning (default: True)
        n_conformer_starts: Number of diverse conformer starting points (None uses auto-determined)
        exploration_time_ps: Exploration phase MTD time in ps (None uses auto-determined)
        conformer_time_ps: Conformer MTD time per run in ps (None uses auto-determined)

    Returns:
        ensemble: List of (structure, energy) tuples, sorted by energy
    """
    log_message("")

    return run_conformer_search_auto(
        initial_structure,
        energy_fn,
        mode=mode,
        temperature=temperature,
        dt=dt,
        ewin=ewin,
        optlevel=optlevel,
        rmsd_threshold=rmsd_threshold,
        mtd_params=mtd_params,
        mtr_rotamer_number=mtr_rotamer_number,
        mtr_temps=mtr_temps,
        mtr_time_ps=mtr_time_ps,
        genetic_crossing=genetic_crossing,
        n_children=n_children,
        mtd_kscal=mtd_kscal,
        trial=trial,
        use_topology=use_topology,
        save_trajectory=save_trajectory,
        potential_wrapper=potential_wrapper,
        n_conformer_starts=n_conformer_starts,
        exploration_time_ps=exploration_time_ps,
        conformer_time_ps=conformer_time_ps,
        parallel=parallel,
        nci=nci,
        nci_max_dist=nci_max_dist,
        fire_dt_start=fire_dt_start,
        fire_dt_max=fire_dt_max,
        fire_n_min=fire_n_min,
        method=method,
        final_fmax=final_fmax,
        final_maxiter=final_maxiter,
        final_max_stepsize=final_max_stepsize,
        output_file=output_file,
    )


# ============================================================================
# Simplified Workflow Variants
# ============================================================================


def run_mtd_only(
    initial_structure: Dict,
    energy_fn: Callable,
    mode: str = "normal",
    temperature: Optional[float] = None,
    dt: Optional[float] = None,
    optlevel: Optional[int] = None,
    ewin: Optional[float] = None,
    mtd_params: Optional[Dict] = None,
    mtd_kscal: float = 1.0,
    trial: bool = True,
    save_trajectory: bool = False,
    potential_wrapper=None,
    use_topology: bool = False,
    n_conformer_starts: Optional[int] = None,
    exploration_time_ps: Optional[float] = None,
    conformer_time_ps: Optional[float] = None,
    parallel: bool = True,
    nci: bool = False,
    nci_max_dist: float = 3.0,
    fire_dt_start: float = 0.05,
    fire_dt_max: float = 0.1,
    fire_n_min: int = 2,
    method: str = "LBFGS",
    final_fmax: Optional[float] = None,
    final_maxiter: Optional[int] = None,
    final_max_stepsize: Optional[float] = None,
    output_file: str = "auto_final_ensemble.xyz",
    coords: str = "cartesian",
    coords_coarse: str = "cartesian",
    init_hessian: Optional[str] = None,
    interfragment: str = "tric",
    ric_options: Optional[Dict] = None,
) -> List[Tuple[Dict, float]]:
    """Conformer MTD sampling only (no rotamer MD or genetic crossing).

    Uses the Conformer MTD workflow with automatic configuration,
    but disables rotamer MD and genetic crossing.

    Args:
        initial_structure: Initial structure dict
        energy_fn: Energy function
        mode: Search mode - 'quick', 'normal', or 'thorough' (default: 'normal')
        temperature: Temperature in K (default: 300)
        dt: Initial timestep in fs (default: 0.5)
        optlevel: Optimization level (None uses auto-determined)
        ewin: Energy window in eV (None uses auto-determined)
        mtd_params: Manual MTD parameters dict (None uses auto-determined)
        mtd_kscal: Global kpush scaling factor (default: 1.0)
        trial: Run trial MTD to validate settings (default: True)
        save_trajectory: Save MD trajectories as XYZ files (default: False)
        potential_wrapper: PotentialWrapper instance for neighbor list support
        use_topology: Enable bond topology constraints (default: False)
        n_conformer_starts: Number of diverse starting points (None uses auto-determined)
        exploration_time_ps: Exploration phase time in ps (None uses auto-determined)
        conformer_time_ps: Conformer MTD time per run in ps (None uses auto-determined)

    Returns:
        ensemble: Final ensemble
    """
    log_header("Conformer MTD-Only Workflow")
    log_message(f"[MTD-Only] Mode: {mode} (auto-configured)")
    log_message(f"[MTD-Only] Rotamer MD: disabled")
    log_message(f"[MTD-Only] Genetic crossing: disabled")
    log_message("")

    return run_conformer_search_auto(
        initial_structure,
        energy_fn,
        mode=mode,
        temperature=temperature,
        dt=dt,
        optlevel=optlevel,
        ewin=ewin,
        mtd_params=mtd_params,
        mtr_rotamer_number=0,  # Disable rotamer MD
        genetic_crossing=False,  # Disable genetic crossing
        mtd_kscal=mtd_kscal,
        trial=trial,
        save_trajectory=save_trajectory,
        potential_wrapper=potential_wrapper,
        use_topology=use_topology,
        n_conformer_starts=n_conformer_starts,
        exploration_time_ps=exploration_time_ps,
        conformer_time_ps=conformer_time_ps,
        parallel=parallel,
        nci=nci,
        nci_max_dist=nci_max_dist,
        fire_dt_start=fire_dt_start,
        fire_dt_max=fire_dt_max,
        fire_n_min=fire_n_min,
        method=method,
        final_fmax=final_fmax,
        final_maxiter=final_maxiter,
        final_max_stepsize=final_max_stepsize,
        output_file=output_file,
    )


def run_md_only(
    initial_structure: Dict,
    energy_fn: Callable,
    md_time_ps: Optional[float] = None,
    temperature: Optional[float] = None,
    dt: Optional[float] = None,
    optlevel: Optional[int] = None,
    ewin: Optional[float] = None,
    save_trajectory: bool = False,
    potential_wrapper=None,
    parallel: bool = True,
    nci: bool = False,
    nci_max_dist: float = 3.0,  # accepted for API parity with the other workflows
    fire_dt_start: float = 0.05,
    fire_dt_max: float = 0.1,
    fire_n_min: int = 2,
    method: str = "LBFGS",
    final_fmax: Optional[float] = None,
    final_maxiter: Optional[int] = None,
    final_max_stepsize: Optional[float] = None,
    output_file: str = "auto_final_ensemble.xyz",
    coords: str = "cartesian",
    coords_coarse: str = "cartesian",
    init_hessian: Optional[str] = None,
    interfragment: str = "tric",
    ric_options: Optional[Dict] = None,
) -> List[Tuple[Dict, float]]:
    """Simplified workflow: plain MD sampling only (no metadynamics, rotamer MD, or genetic crossing).

    Runs a single plain MD trajectory, extracts snapshots, optimizes them, and returns
    the pruned ensemble. Uses default values that can be overridden.

    Args:
        initial_structure: Initial structure dict
        energy_fn: Energy function
        md_time_ps: MD simulation time in ps (default: 10.0)
        temperature: Temperature in K (default: 300)
        dt: Timestep in fs (default: 0.5)
        optlevel: Optimization level (default: 0)
        ewin: Energy window in eV (default: 0.26 eV = 6 kcal/mol)
        save_trajectory: Save MD trajectory as XYZ file (default: False)
        potential_wrapper: PotentialWrapper instance for neighbor list support

    Returns:
        ensemble: Final ensemble of optimized structures
    """
    # Apply defaults
    if md_time_ps is None:
        md_time_ps = 10.0
    if temperature is None:
        temperature = 300.0
    if dt is None:
        dt = 0.5
    if optlevel is None:
        optlevel = 0
    if ewin is None:
        ewin = 0.26  # 6 kcal/mol

    fmax = FMAX_MAP[optlevel]
    if final_fmax is not None:
        log_message(
            f"[MD-Only] Overriding fmax: {final_fmax:.3g} eV/Å "
            f"(was {fmax:.3g} from optlevel={optlevel})"
        )
        fmax = final_fmax
    md_opt_maxiter = 1000 if final_maxiter is None else final_maxiter
    md_opt_max_step = 0.2 if final_max_stepsize is None else final_max_stepsize

    log_header("Plain MD Workflow (No Metadynamics)")
    log_message(f"[MD-Only] Configuration:")
    log_message(f"  - MD time: {md_time_ps:.1f} ps")
    log_message(f"  - Temperature: {temperature:.1f} K")
    log_message(f"  - Timestep: {dt:.3g} fs")
    log_message(f"  - Optimization level: {optlevel}")
    log_message(f"  - Energy window: {ewin * EV_TO_KCALMOL:.1f} kcal/mol")
    log_message("")

    positions_init = initial_structure["positions"]
    atomic_masses = get_atomic_masses(initial_structure["numbers"])

    # Bond topology detection — warn if system is multimolecular
    from .topology import analyse_topology

    mol_topo = analyse_topology(np.array(positions_init), np.array(initial_structure["numbers"]))
    if mol_topo.is_multimolecular:
        if nci:
            log_message(
                f"[MD-Only] NCI mode: system has {mol_topo.n_molecules} molecular fragments "
                f"(fragment sizes: {mol_topo.fragment_sizes}). "
                "Confinement potential active."
            )
        else:
            warnings.warn(
                f"[MD-Only] The system contains {mol_topo.n_molecules} molecular fragments "
                f"(fragment sizes: {mol_topo.fragment_sizes}). "
                "MD-Only sampling is designed for single-molecule systems. "
                "For non-covalent complexes or multi-molecule systems, use the -nci option.",
                UserWarning,
                stacklevel=2,
            )

    # Fragment permutations for permutation-invariant RMSD pruning (NCI mode)
    frag_perms = None
    if nci and mol_topo.is_multimolecular:
        from .rmsd import compute_fragment_permutations

        frag_perms = compute_fragment_permutations(
            mol_topo.fragments, np.array(initial_structure["numbers"])
        )
        if frag_perms.shape[0] > 1:
            log_message(
                f"[NCI] {frag_perms.shape[0]} fragment permutations for "
                f"permutation-invariant RMSD pruning"
            )
        else:
            frag_perms = None

    # Initial optimization
    log_message("[MD-Only] Initial structure optimization...")
    timer_start("Initial Optimization")
    positions_opt, energy_init = optimize_single_with_logging(
        positions_init,
        energy_fn,
        fmax=fmax,
        structure_id="initial",
        potential_wrapper=potential_wrapper,
        fire_dt_start=fire_dt_start,
        fire_dt_max=fire_dt_max,
        fire_n_min=fire_n_min,
        method=method,
        maxiter=md_opt_maxiter,
        max_stepsize=md_opt_max_step,
        **_ric_kwargs(
            coords,
            coords_coarse,
            init_hessian,
            interfragment,
            ric_options,
            initial_structure["numbers"],
        ),
    )
    timer_end("Initial Optimization")

    log_message(f"[MD-Only] Initial energy: {energy_init * EV_TO_KCALMOL:.2f} kcal/mol")

    # Run plain MD
    log_header(f"Plain MD Sampling ({md_time_ps} ps at {temperature} K)")
    timer_start("MD Sampling")
    trajectory = run_rotamer_md_jax(
        positions_opt,
        energy_fn,
        T=temperature,
        dt=dt,
        time_ps=md_time_ps,
        save_interval=200,
        mass=atomic_masses,
        random_seed=42,
        potential_wrapper=potential_wrapper,
    )

    timer_end("MD Sampling")
    log_message(f"[MD-Only] Sampled {len(trajectory)} snapshots")

    # Save trajectory if requested
    if save_trajectory:
        traj_file = f"md_trajectory.xyz"
        save_trajectory_xyz(traj_file, trajectory, initial_structure["symbols"])
        log_message(f"[MD-Only] Saved trajectory to {traj_file}")

    # Extract and optimize snapshots
    log_header("Optimizing MD Snapshots")
    timer_start("Snapshot Optimization")

    # Take snapshots at regular intervals (max 100 snapshots)
    n_snapshots = min(100, len(trajectory))
    stride = max(1, len(trajectory) // n_snapshots)
    snapshots = trajectory[::stride]

    log_message(
        f"[MD-Only] Optimizing {len(snapshots)} snapshots (stride={stride}, level={optlevel})"
    )

    # Create structure list from snapshots
    structures = []
    for pos in snapshots:
        struct = create_structure(
            np.array(pos), initial_structure["symbols"], initial_structure["numbers"]
        )
        structures.append(struct)

    # Pre-optimization pruning
    n_before = len(structures)
    dummy_ensemble = [(s, 0.0) for s in structures]
    dummy_ensemble = prune_by_rotational_constants(dummy_ensemble, threshold=0.02)
    dummy_ensemble = prune_by_rmsd(dummy_ensemble, threshold=0.25, frag_perms=frag_perms)
    structures = [s for s, _ in dummy_ensemble]
    log_message(f"[MD-Only] Pre-opt pruning: {n_before} -> {len(structures)} structures")

    # Single-point energy pre-screening: discard structures clearly outside the window
    structures = prescreen_by_energy(
        structures,
        energy_fn,
        energy_init,
        ewin_buffer=4 * ewin,
        potential_wrapper=potential_wrapper,
        label="MD-Only",
    )

    # Optimize all structures
    ensemble = optimize_multilevel_jax(
        structures,
        energy_fn,
        fmax=fmax,
        maxiter=md_opt_maxiter,
        max_stepsize=md_opt_max_step,
        parallel=parallel,
        potential_wrapper=potential_wrapper,
        fire_dt_start=fire_dt_start,
        fire_dt_max=fire_dt_max,
        fire_n_min=fire_n_min,
        method=method,
        **_ric_kwargs(
            coords,
            coords_coarse,
            init_hessian,
            interfragment,
            ric_options,
            initial_structure["numbers"],
        ),
    )

    # Sort by energy
    ensemble = sort_by_energy(ensemble)
    log_message(f"[MD-Only] Optimized {len(ensemble)} structures")

    # Prune by energy window
    ensemble = prune_by_energy_window(ensemble, ewin)
    log_message(
        f"[MD-Only] After energy pruning (ewin={ewin * EV_TO_KCALMOL:.1f} kcal/mol): {len(ensemble)} structures"
    )

    # Prune by rotational constants
    ensemble = prune_by_rotational_constants(ensemble, threshold=0.02)
    log_message(f"[MD-Only] After rotational constant pruning: {len(ensemble)} structures")

    # Prune by RMSD
    rmsd_threshold = 0.25  # Angstrom
    ensemble = prune_by_rmsd(ensemble, threshold=rmsd_threshold, frag_perms=frag_perms)
    log_message(
        f"[MD-Only] After RMSD pruning (threshold={rmsd_threshold} Å): {len(ensemble)} unique structures"
    )

    timer_end("Snapshot Optimization")
    log_header("MD-Only Workflow Complete")
    log_message(f"[MD-Only] Found {len(ensemble)} unique conformers")

    if len(ensemble) > 0:
        energies = [e for _, e in ensemble]
        e_min = min(energies)
        log_message(
            f"[MD-Only] Energy span: {(max(energies) - e_min) * EV_TO_KCALMOL:.1f} kcal/mol"
        )

    save_ensemble(output_file, ensemble)
    log_message(f"[MD-Only] Saved ensemble to {output_file}")

    return ensemble


# ============================================================================
# Auto-Configured Workflow
# ============================================================================


def run_conformer_search_auto(
    initial_structure: Dict,
    energy_fn: Callable,
    mode: str = "normal",
    temperature: Optional[float] = None,
    dt: Optional[float] = None,
    ewin: Optional[float] = None,
    optlevel: Optional[int] = None,
    rmsd_threshold: Optional[float] = None,
    mtd_params: Optional[Dict] = None,
    mtr_rotamer_number: Optional[int] = None,
    mtr_temps: Optional[int] = None,
    mtr_time_ps: Optional[float] = None,
    genetic_crossing: Optional[bool] = None,
    n_children: Optional[int] = None,
    mtd_kscal: float = 1.0,
    trial: bool = True,
    save_trajectory: bool = False,
    potential_wrapper=None,
    use_topology: bool = True,
    n_conformer_starts: Optional[int] = None,
    exploration_time_ps: Optional[float] = None,
    conformer_time_ps: Optional[float] = None,
    parallel: bool = True,
    nci: bool = False,
    nci_max_dist: float = 3.0,
    fire_dt_start: float = 0.05,
    fire_dt_max: float = 0.1,
    fire_n_min: int = 2,
    method: str = "LBFGS",
    final_fmax: Optional[float] = None,
    final_maxiter: Optional[int] = None,
    final_max_stepsize: Optional[float] = None,
    output_file: str = "auto_final_ensemble.xyz",
    coords: str = "cartesian",
    coords_coarse: str = "cartesian",
    init_hessian: Optional[str] = None,
    interfragment: str = "tric",
    ric_options: Optional[Dict] = None,
) -> List[Tuple[Dict, float]]:
    """Run multi-step metadynamics with automatic parameter configuration.

    Implements the top-level conformational-sampling workflow:

    1. Auto-determine MTD parameters from molecular flexibility
       (covalent topology, sp²/sp³ hybridization, rotatable-bond count).
    2. Optional trial MTD to validate the timestep.
    3. Multi-step MTD grid (different `kpush`/`alpha` per step).
    4. Refinement from diverse starts (`normal`/`thorough` modes).
    5. Optional rotamer MD on the lowest-energy structures.
    6. Optional Z-matrix genetic crossing (`normal`/`thorough` modes).

    The result is pruned by energy window + RMSD + (optionally) bond
    topology before being returned sorted by energy.

    Args:
        initial_structure: Structure dict with ``positions`` (n_atoms, 3)
            in Å, ``symbols`` (list of str), and ``numbers``
            (n_atoms,) atomic numbers.
        energy_fn: JAX callable ``positions -> energy`` returning eV.
        mode: ``"quick"``, ``"normal"`` (default), or ``"thorough"``.
        temperature: Temperature in K (default 300).
        dt: Initial MD timestep in fs (default 0.5; may be lowered by
            the trial MTD).
        ewin: Energy window in eV for ensemble pruning. ``None`` uses
            the auto value derived from `mode`.
        optlevel: Optimization tightness level, ``-3..2``. ``None``
            uses the auto value.
        rmsd_threshold: RMSD cutoff in Å for duplicate pruning. ``None``
            uses the auto value.
        mtd_params: Manual override with keys ``kpush`` (eV),
            ``alpha`` (1/Å²), ``cvdump_fs`` (fs). When given, every
            grid step uses this single parameter set.
        mtr_rotamer_number: Number of structures fed to rotamer MD.
        mtr_temps: Number of temperature replicas for rotamer MD.
        mtr_time_ps: Rotamer MD time in ps.
        genetic_crossing: Enable Z-matrix genetic crossing. ``None``
            defers to the auto setting for `mode`.
        n_children: Number of crossing children per generation.
        mtd_kscal: Global multiplicative factor on every `kpush`
            (default 1.0).
        trial: Run a short trial MTD to validate the timestep
            (default ``True``).
        save_trajectory: Persist MD trajectories as multi-frame XYZ.
        potential_wrapper: Optional ``PotentialWrapper`` for
            neighbor-list bookkeeping (auto-detected for SO3LR).
        use_topology: Enable bond-topology pruning (default ``True``).
        n_conformer_starts: Diverse starting points for refinement.
        exploration_time_ps: MTD time per grid step (ps).
        conformer_time_ps: Refinement MTD time per run (ps).
        parallel: Vmap parallelism for batch optimization.
        nci: Apply spherical confinement during all MTD/MD runs.
        nci_max_dist: Max inter-fragment drift in Å before a snapshot
            is discarded (only used when ``nci`` is True).
        fire_dt_start: FIRE initial timestep.
        fire_dt_max: FIRE max timestep.
        fire_n_min: FIRE min positive-power steps before dt increases.

    Returns:
        Sorted list of ``(structure_dict, energy_eV)`` tuples — one
        per unique conformer surviving all pruning stages.

    Example:
        >>> from mars import load_structure, get_potential
        >>> from mars.conf_sampling import run_conformer_search_auto
        >>> structure = load_structure("input.xyz")
        >>> potential = get_potential(
        ...     "so3lr", species=structure["numbers"], charge=0.0
        ... )
        >>> energy_fn = potential.build_energy_fn()
        >>> ensemble = run_conformer_search_auto(
        ...     structure, energy_fn, mode="thorough",
        ...     genetic_crossing=True, n_children=30,
        ... )
        >>> for i, (struct, e) in enumerate(ensemble[:5]):
        ...     print(f"{i+1}: {e:.4f} eV")
    """
    # Apply defaults for non-overridden input params
    if temperature is None:
        temperature = 300.0
    if dt is None:
        dt = 0.5

    positions = initial_structure["positions"]
    n_atoms = positions.shape[0]

    atomic_masses = get_atomic_masses(initial_structure["numbers"])

    # ========================================================================
    # Step 0: Bond topology detection
    # Always run so we can warn about multi-molecule systems regardless of
    # whether topology-based pruning is enabled.
    # ========================================================================
    from .log import log_topology
    from .topology import analyse_topology
    from .utils import BondTopology

    log_message("[Auto] Detecting bond topology...")
    mol_topo = analyse_topology(np.array(positions), np.array(initial_structure["numbers"]))

    if mol_topo.is_multimolecular:
        if nci:
            log_message(
                f"[Auto] NCI mode: system has {mol_topo.n_molecules} molecular fragments "
                f"(fragment sizes: {mol_topo.fragment_sizes}). "
                "Confinement potential active."
            )
        else:
            warnings.warn(
                f"[Auto] The system contains {mol_topo.n_molecules} molecular fragments "
                f"(fragment sizes: {mol_topo.fragment_sizes}). "
                "This workflow is designed for single-molecule conformational search. "
                "For non-covalent complexes or multi-molecule systems, use the -nci option.",
                UserWarning,
                stacklevel=2,
            )

    n_bonds = len(mol_topo.bonds)
    topology = None
    if use_topology:
        log_topology(
            mol_topo.bonds,
            mol_topo.bond_distances,
            atomic_numbers=list(np.array(initial_structure["numbers"])),
            n_molecules=mol_topo.n_molecules,
            fragment_sizes=mol_topo.fragment_sizes,
        )
        topology = BondTopology(bonds=mol_topo.bonds, distances=mol_topo.bond_distances)

    # Fragment permutations for permutation-invariant RMSD pruning (NCI mode)
    frag_perms = None
    if nci and mol_topo.is_multimolecular:
        from .rmsd import compute_fragment_permutations

        frag_perms = compute_fragment_permutations(
            mol_topo.fragments, np.array(initial_structure["numbers"])
        )
        if frag_perms.shape[0] > 1:
            log_message(
                f"[NCI] {frag_perms.shape[0]} fragment permutations for "
                f"permutation-invariant RMSD pruning"
            )
        else:
            frag_perms = None  # no identical fragments → standard RMSD

    # NCI inter-fragment distance filter setup
    nci_cross_mask = None
    nci_d_max = None
    if nci and mol_topo.is_multimolecular:
        nci_cross_mask = build_nci_cross_mask(mol_topo.fragments, n_atoms)
        initial_d_min = min_inter_fragment_dist(np.array(positions), nci_cross_mask)
        nci_d_max = np.maximum(initial_d_min, nci_max_dist)
        log_message(
            f"[NCI] Inter-fragment distance filter: {mol_topo.n_molecules} fragments, "
            f"initial contact {initial_d_min:.2f} Å, "
            f"max allowed {nci_d_max:.2f} Å"
        )

    # ========================================================================
    # Step 1: Automatic configuration
    # ========================================================================
    config = auto_setup(
        n_atoms=n_atoms,
        mode=mode,
        atomic_numbers=initial_structure.get("numbers"),
        positions=np.array(positions),
        n_bonds=n_bonds,
        temperature=temperature,
        dt=dt,
        mtd_kscal=mtd_kscal,
    )

    # Apply user overrides to auto config (non-None values take priority)
    if ewin is not None:
        config["ewin"] = ewin
    if optlevel is not None:
        config["optlevel"] = optlevel
    if rmsd_threshold is not None:
        config["rmsd_threshold"] = rmsd_threshold
    # Override rotamer MD settings
    if mtr_rotamer_number is not None:
        log_message(f"[Auto] Overriding rotamer structures: {mtr_rotamer_number}")
        config["rotamer_md"]["n_structures"] = mtr_rotamer_number
    if mtr_temps is not None:
        log_message(f"[Auto] Overriding rotamer temps: {mtr_temps}")
        config["rotamer_md"]["n_temps"] = mtr_temps
    if mtr_time_ps is not None:
        log_message(f"[Auto] Overriding rotamer time: {mtr_time_ps} ps")
        config["rotamer_md"]["time_ps"] = mtr_time_ps

    # Override genetic crossing settings
    if genetic_crossing is not None:
        config["genetic_crossing"]["enabled"] = genetic_crossing
        if genetic_crossing:
            log_message(f"[Auto] Enabling genetic crossing")
        else:
            log_message(f"[Auto] Disabling genetic crossing")
    if n_children is not None:
        log_message(f"[Auto] Overriding number of children: {n_children}")
        config["genetic_crossing"]["n_children"] = n_children

    # Override MTD workflow settings
    if n_conformer_starts is not None:
        log_message(f"[Auto] Overriding number of conformer starts: {n_conformer_starts}")
        config["conformer_mtd"]["n_conformer_starts"] = n_conformer_starts
    if exploration_time_ps is not None:
        log_message(f"[Auto] Overriding time per MTD step: {exploration_time_ps} ps")
        config["conformer_mtd"]["time_per_step_ps"] = exploration_time_ps
    if conformer_time_ps is not None:
        log_message(f"[Auto] Overriding refinement time: {conformer_time_ps} ps")
        config["conformer_mtd"]["refinement_time_ps"] = conformer_time_ps

    # Override MTD bias parameters (manual --kpush/--alpha/--cvdump-fs)
    if mtd_params is not None:
        log_message(
            f"[Auto] Manual MTD parameters: "
            f"kpush={mtd_params['kpush'] * EV_TO_KCALMOL:.3g} kcal/mol, "
            f"alpha={mtd_params['alpha']:.2f} Å⁻², "
            f"cvdump_fs={mtd_params['cvdump_fs']:.1f} fs"
        )
        config["conformer_mtd"]["mtd_steps"] = [mtd_params]
        config["conformer_mtd"]["refinement_params"] = mtd_params

    # Safety check for very small molecules
    if n_atoms <= 2:
        log_message("[Auto] Molecule has <= 2 atoms, nothing to sample")
        return []

    # ========================================================================
    # Configuration Summary (single print, after all user overrides)
    # ========================================================================
    _print_auto_config(config)

    fmax = FMAX_MAP.get(config["optlevel"], 0.01)
    if final_fmax is not None:
        log_message(
            f"[Auto] Overriding final-cycle fmax: "
            f"{final_fmax:.3g} eV/Å "
            f"(coarse cycles scaled 50× / 10×)"
        )
        fmax = final_fmax

    final_cycle_maxiter = 1000 if final_maxiter is None else final_maxiter
    final_cycle_max_step = 0.2 if final_max_stepsize is None else final_max_stepsize
    if final_maxiter is not None:
        log_message(f"[Auto] Overriding final-cycle maxiter: {final_cycle_maxiter}")
    if final_max_stepsize is not None:
        log_message(f"[Auto] Overriding final-cycle max_stepsize: {final_cycle_max_step} Å")
    if method != "LBFGS":
        log_message(f"[Auto] Using optimizer method: {method}")

    # 3-cycle multilevel optimization parameters (used in all phases)
    opt_cycles = [
        (50 * fmax, 2.0 * config["ewin"], 10, 1.0),  # Cycle 1: coarse opt, large steps
        (10 * fmax, 1.5 * config["ewin"], 100, 0.5),  # Cycle 2: medium opt, medium steps
        (fmax, config["ewin"], final_cycle_maxiter, final_cycle_max_step),  # Cycle 3: fine
    ]

    # ========================================================================
    # Step 2: Initial optimization
    # ========================================================================
    log_header("Initial Structure Optimization")
    timer_start("Initial Optimization")
    positions_opt, energy_init = optimize_single_with_logging(
        positions,
        energy_fn,
        fmax=fmax * 50,
        structure_id="initial",
        potential_wrapper=potential_wrapper,
        fire_dt_start=fire_dt_start,
        fire_dt_max=fire_dt_max,
        fire_n_min=fire_n_min,
        method=method,
        **_ric_kwargs(
            coords,
            coords_coarse,
            init_hessian,
            interfragment,
            ric_options,
            initial_structure["numbers"],
        ),
    )
    timer_end("Initial Optimization")

    initial_structure_opt = initial_structure.copy()
    initial_structure_opt["positions"] = positions_opt
    initial_structure_opt["energy"] = energy_init

    log_message(f"[Auto] Initial energy: {energy_init * EV_TO_KCALMOL:.2f} kcal/mol")

    # Check if the molecule broke during initial optimization
    opt_topo = analyse_topology(np.array(positions_opt), np.array(initial_structure["numbers"]))
    if opt_topo.n_molecules > mol_topo.n_molecules:
        log_message("")
        log_message(
            "[Auto] ERROR: The molecule fragmented during initial optimization "
            f"({opt_topo.n_molecules} fragments detected, expected {mol_topo.n_molecules}). "
            "The molecule is not stable with this model."
        )
        log_message("")
        return []

    # ========================================================================
    # Step 3: Trial MTD (optional)
    # ========================================================================
    effective_dt = config["dt"]

    if trial:
        log_header("Trial Metadynamics")
        timer_start("Trial MTD")
        trial_result = run_trial_mtd(
            positions_opt,
            energy_fn,
            n_atoms,
            T=temperature,
            dt=config["dt"],
            mass=atomic_masses,
            potential_wrapper=potential_wrapper,
        )
        timer_end("Trial MTD")

        effective_dt = trial_result["dt"]
        log_message(f"[Auto] Trial MTD passed (dt={effective_dt:.3g} fs)")

    # ========================================================================
    # Step 4: Multi-Step MTD Grid
    # ========================================================================
    conformer_mtd_cfg = config.get("conformer_mtd", {})
    ensemble = []
    # Tracks all pre-opt-pruned structures ever sent to the optimizer, across all
    # phases.  Subsequent phases prune incoming snapshots against this pool to
    # avoid re-optimising starting points that were already explored (and may
    # have been discarded from the final ensemble for other reasons).
    seen_pool: List[Tuple[Dict, float]] = []

    mtd_steps = conformer_mtd_cfg["mtd_steps"]
    time_per_step = conformer_mtd_cfg["time_per_step_ps"]
    n_steps = len(mtd_steps)

    log_message("")
    log_message(f"  ═══ Phase 1: Multi-Step MTD Grid ({n_steps} steps) ═══")
    log_message("")
    timer_start("Phase 1: MTD Grid")
    log_message(f"[MTD Grid] Running {n_steps} MTD simulations with different bias parameters")
    log_message(f"[MTD Grid] Time per step: {time_per_step:.3g} ps")
    log_message(f"[MTD Grid] Total MTD time: {n_steps * time_per_step:.3g} ps")
    log_message("")

    all_grid_snapshots = []
    n_batches_phase1 = 1

    if parallel and n_steps > 1:
        # Parallel path: vmap all MTD steps simultaneously.
        # run_mtd_parallel returns (trajectory, final_mtd_state).
        # trajectory is a list of (n_replicas, n_atoms, 3) arrays, one per saved frame.
        traj_batch, _, n_batches_phase1 = run_mtd_parallel(
            positions_opt,
            energy_fn,
            mtd_steps,
            T=temperature,
            dt=effective_dt,
            time_ps=time_per_step,
            mass=atomic_masses,
            potential_wrapper=potential_wrapper,
            save_interval=200,
        )

        for i, frame_batch in enumerate(traj_batch):
            for r in range(frame_batch.shape[0]):
                all_grid_snapshots.append(frame_batch[r])
        if save_trajectory:
            for r in range(n_steps):
                replica_traj = [frame_batch[r] for frame_batch in traj_batch]
                save_trajectory_xyz(
                    f"mtd_{r+1}_trajectory.xyz", replica_traj, initial_structure["symbols"]
                )

    else:
        # Sequential path: one run_mtd_jax call per MTD step.
        for step_idx, step_params in enumerate(mtd_steps):
            log_message(
                f"[MTD Grid] Step {step_idx+1}/{n_steps}: "
                f"kpush={step_params['kpush'] * EV_TO_KCALMOL:.4f} kcal/mol, "
                f"alpha={step_params['alpha']:.4f} Å⁻², "
                f"cvdump={step_params['cvdump_fs']:.0f} fs"
            )
            trajectory, mtd_state = run_mtd_jax(
                positions_opt,
                energy_fn,
                step_params,
                T=temperature,
                dt=effective_dt,
                time_ps=time_per_step,
                mass=atomic_masses,
                random_seed=42 + step_idx,
                potential_wrapper=potential_wrapper,
                save_interval=200,
            )
            log_message(
                f"[MTD Grid] Step {step_idx+1}: {len(trajectory)} snapshots, "
                f"{mtd_state.n_hills} hills deposited"
            )
            all_grid_snapshots.extend(trajectory)

            if save_trajectory:
                save_trajectory_xyz(
                    f"mtd_{step_idx+1}_trajectory.xyz", trajectory, initial_structure["symbols"]
                )

    log_message(f"[MTD Grid] Total snapshots from grid: {len(all_grid_snapshots)}")

    # Convert grid snapshots to structures
    grid_structures = []
    for pos in all_grid_snapshots:
        struct = create_structure(
            positions=pos,
            symbols=initial_structure["symbols"],
            numbers=initial_structure["numbers"],
        )
        grid_structures.append(struct)

    # Pre-optimization pruning (topology checked first)
    n_before = len(grid_structures)
    dummy_ensemble = [(s, 0.0) for s in grid_structures]
    if use_topology:
        dummy_ensemble = prune_by_topology(dummy_ensemble, topology)
    dummy_ensemble = prune_by_rotational_constants(dummy_ensemble, threshold=0.02)
    dummy_ensemble = prune_by_rmsd(
        dummy_ensemble, threshold=config["rmsd_threshold"], frag_perms=frag_perms
    )
    grid_structures = [s for s, _ in dummy_ensemble]
    log_message(f"[MTD Grid] Pre-opt pruning: {n_before} -> {len(grid_structures)} snapshots")

    # Single-point energy pre-screening: discard structures clearly outside the window
    grid_structures = prescreen_by_energy(
        grid_structures,
        energy_fn,
        energy_init,
        ewin_buffer=4 * config["ewin"],
        potential_wrapper=potential_wrapper,
        label="MTD Grid",
    )

    seen_pool.extend([(s, 0.0) for s in grid_structures])
    log_message(f"[MTD Grid] Seen pool: {len(seen_pool)} structures tracked")

    current_structures = grid_structures
    for cycle_i, (cycle_fmax, cycle_ewin, cycle_maxiter, cycle_max_step) in enumerate(
        opt_cycles, 1
    ):
        if not current_structures:
            log_message(f"[MTD Grid] Cycle {cycle_i}/3: No structures to optimize. Skipping...")
            break
        log_message(
            f"[MTD Grid] Cycle {cycle_i}/3: Optimizing {len(current_structures)} structures "
            f"(fmax={cycle_fmax:.3g} eV/Å, ewin={cycle_ewin*EV_TO_KCALMOL:.1f} kcal/mol, "
            f"maxiter={cycle_maxiter}, max_step={cycle_max_step} Å)..."
        )
        is_last_cycle = cycle_i == len(opt_cycles)
        grid_ensemble, converged_flags = optimize_multilevel_jax(
            current_structures,
            energy_fn,
            fmax=cycle_fmax,
            maxiter=cycle_maxiter,
            parallel=parallel,
            max_stepsize=cycle_max_step,
            return_converged=True,
            potential_wrapper=potential_wrapper,
            fire_dt_start=fire_dt_start,
            fire_dt_max=fire_dt_max,
            fire_n_min=fire_n_min,
            method=method,
            **_ric_kwargs(
                coords,
                coords_coarse,
                init_hessian,
                interfragment,
                ric_options,
                initial_structure["numbers"],
                cycle_i,
            ),
        )
        if is_last_cycle:
            n_nc = sum(1 for c in converged_flags if not c)
            if n_nc:
                log_message(f"[MTD Grid] Discarding {n_nc} non-converged structures in final cycle")
            grid_ensemble = [(s, e) for (s, e), c in zip(grid_ensemble, converged_flags) if c]
        grid_ensemble = sort_by_energy(grid_ensemble)
        if use_topology:
            grid_ensemble = prune_by_topology(grid_ensemble, topology)
        if nci_cross_mask is not None:
            grid_ensemble = prune_by_nci_distance(
                grid_ensemble, nci_cross_mask, nci_d_max * _NCI_DIST_FACTORS[cycle_i], "MTD Grid"
            )
        grid_ensemble = _prune_energy_window_guarded(
            grid_ensemble, cycle_ewin, is_last_cycle, cycle_i, len(opt_cycles), "MTD Grid"
        )
        grid_ensemble = prune_by_rotational_constants(grid_ensemble, threshold=0.02)
        grid_ensemble = prune_by_rmsd(
            grid_ensemble, threshold=config["rmsd_threshold"], frag_perms=frag_perms
        )
        log_message(
            f"[MTD Grid] Cycle {cycle_i}/3: {len(current_structures)} -> {len(grid_ensemble)} conformers"
        )
        current_structures = [s for s, _ in grid_ensemble]
        if not current_structures:
            log_message("[MTD Grid] No structures to optimize. Skiping the optimization...")
            break

    ensemble = grid_ensemble
    log_message(f"[MTD Grid] After optimization and pruning: {len(ensemble)} conformers")
    timer_end("Phase 1: MTD Grid")

    if len(ensemble) == 0:
        log_message("")
        log_message(
            "[MTD Grid] ERROR: All structures generated by Phase 1 MTD were discarded "
            "(broken bonds or outside energy window). "
            "The molecule is not stable with this model."
        )
        log_message("")
        return []

    # ========================================================================
    # Step 4b: Conformer Refinement from Diverse Starts (normal/thorough)
    # ========================================================================
    n_refinement_starts = conformer_mtd_cfg["n_conformer_starts"]

    if n_refinement_starts > 0 and len(ensemble) > 0:
        from .utils_prune import find_most_diverse_subset

        log_message("")
        log_message("  ═══ Phase 2: Conformer Refinement from Diverse Starts ═══")
        log_message("")
        timer_start("Phase 2: Conformer Refinement")

        min_rmsd = conformer_mtd_cfg.get("min_rmsd_threshold", 0.5)
        diverse_conformers = find_most_diverse_subset(
            ensemble, n_keep=min(n_refinement_starts, len(ensemble)), min_rmsd=min_rmsd
        )

        log_message(f"[Refinement] Selected {len(diverse_conformers)} diverse starting points")
        log_energy_table(diverse_conformers, title="Diverse starting conformers")

        save_ensemble("diverse_conformers.xyz", diverse_conformers)

        ref_params = conformer_mtd_cfg["refinement_params"]
        ref_time = conformer_mtd_cfg["refinement_time_ps"]

        n_diverse = len(diverse_conformers)
        log_message(f"[Refinement] Running {n_diverse} MTDs: {ref_time:.3g} ps each")
        log_message(
            f"[Refinement] Bias: kpush={ref_params['kpush'] * EV_TO_KCALMOL:.3g} kcal/mol, "
            f"alpha={ref_params['alpha']:.2f} Å⁻²"
        )

        all_refinement_snapshots = []

        if parallel and n_diverse > 1:
            # Parallel path: vmap all diverse starting conformers simultaneously.
            starts_batch = jnp.stack(
                [jnp.asarray(s["positions"]) for s, _ in diverse_conformers]
            )  # (n_diverse, n_atoms, 3)
            ref_params_list = [ref_params] * n_diverse

            traj_batch, ref_mtd_state_batch, _ = run_mtd_parallel(
                starts_batch,
                energy_fn,
                ref_params_list,
                T=temperature,
                dt=effective_dt,
                time_ps=ref_time,
                mass=atomic_masses,
                random_seed=200,
                potential_wrapper=potential_wrapper,
                save_interval=200,
            )
            # traj_batch: list of (n_diverse, n_atoms, 3) arrays, one per saved frame
            for frame_batch in traj_batch:
                for r in range(frame_batch.shape[0]):
                    all_refinement_snapshots.append(frame_batch[r])

            avg_hills = int(jnp.mean(ref_mtd_state_batch.n_hills))
            log_message(
                f"[Refinement] Parallel MTD complete: {len(all_refinement_snapshots)} snapshots "
                f"from {n_diverse} conformers, {avg_hills} hills avg"
            )

            if save_trajectory:
                for r in range(n_diverse):
                    replica_traj = [frame_batch[r] for frame_batch in traj_batch]
                    save_trajectory_xyz(
                        f"refinement_mtd_{r+1}_trajectory.xyz",
                        replica_traj,
                        initial_structure["symbols"],
                    )
        else:
            for i, (start_struct, start_energy) in enumerate(diverse_conformers):
                log_message(
                    f"[Refinement] MTD {i+1}/{n_diverse}: "
                    f"E={start_energy * EV_TO_KCALMOL:.2f} kcal/mol"
                )

                ref_trajectory, ref_mtd_state = run_mtd_jax(
                    start_struct["positions"],
                    energy_fn,
                    ref_params,
                    T=temperature,
                    dt=effective_dt,
                    time_ps=ref_time,
                    mass=atomic_masses,
                    random_seed=200 + i,
                    potential_wrapper=potential_wrapper,
                    save_interval=200,
                )

                log_message(
                    f"[Refinement] MTD {i+1}: {len(ref_trajectory)} snapshots, "
                    f"{ref_mtd_state.n_hills} hills"
                )

                all_refinement_snapshots.extend(ref_trajectory)

                if save_trajectory:
                    save_trajectory_xyz(
                        f"refinement_mtd_{i+1}_trajectory.xyz",
                        ref_trajectory,
                        initial_structure["symbols"],
                    )

        log_message(f"[Refinement] Total snapshots: {len(all_refinement_snapshots)}")

        # Convert and optimize refinement snapshots
        ref_structures = []
        for pos in all_refinement_snapshots:
            struct = create_structure(
                positions=pos,
                symbols=initial_structure["symbols"],
                numbers=initial_structure["numbers"],
            )
            ref_structures.append(struct)

        # Pre-optimization pruning in two steps (topology first, then):
        # 1. Prune against ensemble (optimized conformers) — discard anything in a known basin
        # 2. Prune survivors against seen_pool (previously explored starting points) — avoid
        #    re-optimising starting geometries that were already tried and discarded
        n_before = len(ref_structures)
        dummy_ensemble = [(s, 0.0) for s in ref_structures]
        if use_topology:
            dummy_ensemble = prune_by_topology(dummy_ensemble, topology)
        dummy_ensemble = prune_by_rotational_constants(
            dummy_ensemble, threshold=0.02, reference=ensemble
        )
        dummy_ensemble = prune_by_rmsd(
            dummy_ensemble,
            threshold=config["rmsd_threshold"],
            reference=ensemble,
            frag_perms=frag_perms,
        )
        dummy_ensemble = prune_by_rotational_constants(
            dummy_ensemble, threshold=0.02, reference=seen_pool
        )
        dummy_ensemble = prune_by_rmsd(
            dummy_ensemble,
            threshold=config["rmsd_threshold"],
            reference=seen_pool,
            frag_perms=frag_perms,
        )
        ref_structures = [s for s, _ in dummy_ensemble]
        log_message(f"[Refinement] Pre-opt pruning: {n_before} -> {len(ref_structures)}")

        # Single-point energy pre-screening: discard structures clearly outside the window
        ref_energy_ref = min((e for _, e in ensemble), default=energy_init)
        ref_structures = prescreen_by_energy(
            ref_structures,
            energy_fn,
            ref_energy_ref,
            ewin_buffer=4 * config["ewin"],
            potential_wrapper=potential_wrapper,
            label="Refinement",
        )

        seen_pool.extend([(s, 0.0) for s in ref_structures])
        log_message(f"[Refinement] Seen pool: {len(seen_pool)} structures tracked")

        # 3-cycle optimization with progressive refinement
        current_structures = ref_structures
        ref_ensemble = []
        for cycle_i, (cycle_fmax, cycle_ewin, cycle_maxiter, cycle_max_step) in enumerate(
            opt_cycles, 1
        ):
            if not current_structures:
                log_message(
                    f"[Refinement] Cycle {cycle_i}/3: No structures to optimize. Skipping..."
                )
                break
            log_message(
                f"[Refinement] Cycle {cycle_i}/3: Optimizing {len(current_structures)} structures "
                f"(fmax={cycle_fmax:.3g} eV/Å, ewin={cycle_ewin*EV_TO_KCALMOL:.1f} kcal/mol, "
                f"maxiter={cycle_maxiter}, max_step={cycle_max_step} Å)..."
            )
            is_last_cycle = cycle_i == len(opt_cycles)
            ref_ensemble, converged_flags = optimize_multilevel_jax(
                current_structures,
                energy_fn,
                fmax=cycle_fmax,
                maxiter=cycle_maxiter,
                parallel=parallel,
                max_stepsize=cycle_max_step,
                return_converged=True,
                potential_wrapper=potential_wrapper,
                fire_dt_start=fire_dt_start,
                fire_dt_max=fire_dt_max,
                fire_n_min=fire_n_min,
                method=method,
                **_ric_kwargs(
                    coords,
                    coords_coarse,
                    init_hessian,
                    interfragment,
                    ric_options,
                    initial_structure["numbers"],
                    cycle_i,
                ),
            )
            if is_last_cycle:
                n_nc = sum(1 for c in converged_flags if not c)
                if n_nc:
                    log_message(
                        f"[Refinement] Discarding {n_nc} non-converged structures in final cycle"
                    )
                ref_ensemble = [(s, e) for (s, e), c in zip(ref_ensemble, converged_flags) if c]
            ref_ensemble = sort_by_energy(ref_ensemble)
            if use_topology:
                ref_ensemble = prune_by_topology(ref_ensemble, topology)
            if nci_cross_mask is not None:
                ref_ensemble = prune_by_nci_distance(
                    ref_ensemble,
                    nci_cross_mask,
                    nci_d_max * _NCI_DIST_FACTORS[cycle_i],
                    "Refinement",
                )
            ref_ensemble = _prune_energy_window_guarded(
                ref_ensemble, cycle_ewin, is_last_cycle, cycle_i, len(opt_cycles), "Refinement"
            )
            ref_ensemble = prune_by_rotational_constants(
                ref_ensemble, threshold=0.02, reference=ensemble
            )
            ref_ensemble = prune_by_rmsd(
                ref_ensemble,
                threshold=config["rmsd_threshold"],
                reference=ensemble,
                frag_perms=frag_perms,
            )
            log_message(
                f"[Refinement] Cycle {cycle_i}/3: {len(current_structures)} -> {len(ref_ensemble)} conformers"
            )
            current_structures = [s for s, _ in ref_ensemble]
            if not current_structures:
                log_message("[Refinement] No structures to optimize. Skiping the optimization...")
                break

        # Merge refined ensemble with existing
        ensemble.extend(ref_ensemble)
        ensemble = sort_by_energy(ensemble)
        ensemble = prune_by_energy_window(ensemble, ewin=config["ewin"])
        ensemble = prune_by_rotational_constants(ensemble, threshold=0.02)
        ensemble = prune_by_rmsd(
            ensemble, threshold=config["rmsd_threshold"], frag_perms=frag_perms
        )

        log_message(f"[Refinement] After refinement: {len(ensemble)} total conformers")
        timer_end("Phase 2: Conformer Refinement")

    # Save intermediate ensemble
    save_ensemble("mtd_ensemble.xyz", ensemble)
    log_energy_table(ensemble, title="Ensemble after MTD workflow")

    # ========================================================================
    # Step 5: Rotamer MD
    # ========================================================================
    rot_cfg = config["rotamer_md"]
    if rot_cfg["enabled"] and len(ensemble) > 0:
        log_header("Rotamer MD Sampling")
        timer_start("Rotamer MD")

        n_strucs = min(rot_cfg["n_structures"], len(ensemble))
        rot_time = rot_cfg["time_ps"]

        log_message(
            f"[Auto] Rotamer MD: {n_strucs} structures × "
            f"{rot_cfg['n_temps']} temps, {rot_time:.1f} ps each"
        )

        # Scale n_batches from Phase 1 to account for the larger replica count
        # (n_strucs * n_temps vs len(mtd_steps)).
        n_rot_replicas = n_strucs * rot_cfg["n_temps"]
        n_rot_batches = max(1, math.ceil(n_rot_replicas * n_batches_phase1 / len(mtd_steps)))
        if n_rot_batches > 1:
            log_message(
                f"[Rotamer MD] Pre-splitting into {n_rot_batches} batches "
                f"(inferred from Phase 1 memory limit)"
            )

        all_trajectories = run_multi_temperature_rotamer_md(
            ensemble[:n_strucs],
            energy_fn,
            base_T=temperature,
            n_temps=rot_cfg["n_temps"],
            dt=effective_dt,
            time_ps=rot_time,
            n_structures=n_strucs,
            mass=atomic_masses,
            potential_wrapper=potential_wrapper,
            n_batches=n_rot_batches,
            parallel=parallel,
        )

        # Collect all rotamer structures
        rotamer_structures = []
        for trajectory in all_trajectories:
            for pos in trajectory:
                struct = create_structure(
                    positions=pos,
                    symbols=initial_structure["symbols"],
                    numbers=initial_structure["numbers"],
                )
                rotamer_structures.append(struct)

        # Pre-optimization pruning in two steps (topology first, then):
        # 1. Prune against ensemble (optimized conformers) — discard anything in a known basin
        # 2. Prune survivors against seen_pool (previously explored starting points) — avoid
        #    re-optimising starting geometries that were already tried and discarded
        n_before = len(rotamer_structures)
        dummy_ensemble = [(s, 0.0) for s in rotamer_structures]
        if use_topology:
            dummy_ensemble = prune_by_topology(dummy_ensemble, topology)
        dummy_ensemble = prune_by_rotational_constants(
            dummy_ensemble, threshold=0.02, reference=ensemble
        )
        dummy_ensemble = prune_by_rmsd(
            dummy_ensemble,
            threshold=config["rmsd_threshold"],
            reference=ensemble,
            frag_perms=frag_perms,
        )
        dummy_ensemble = prune_by_rotational_constants(
            dummy_ensemble, threshold=0.02, reference=seen_pool
        )
        dummy_ensemble = prune_by_rmsd(
            dummy_ensemble,
            threshold=config["rmsd_threshold"],
            reference=seen_pool,
            frag_perms=frag_perms,
        )
        rotamer_structures = [s for s, _ in dummy_ensemble]
        log_message(
            f"[Rotamer MD] Pre-opt pruning: {n_before} -> {len(rotamer_structures)} structures"
        )

        # Single-point energy pre-screening: discard structures clearly outside the window
        rot_energy_ref = min((e for _, e in ensemble), default=energy_init)
        rotamer_structures = prescreen_by_energy(
            rotamer_structures,
            energy_fn,
            rot_energy_ref,
            ewin_buffer=4 * config["ewin"],
            potential_wrapper=potential_wrapper,
            label="Rotamer MD",
        )

        seen_pool.extend([(s, 0.0) for s in rotamer_structures])
        log_message(f"[Rotamer MD] Seen pool: {len(seen_pool)} structures tracked")

        # 3-cycle optimization with progressive refinement
        current_structures = rotamer_structures
        rot_ensemble = []
        for cycle_i, (cycle_fmax, cycle_ewin, cycle_maxiter, cycle_max_step) in enumerate(
            opt_cycles, 1
        ):
            if not current_structures:
                log_message(
                    f"[Rotamer MD] Cycle {cycle_i}/3: No structures to optimize. Skipping..."
                )
                break
            log_message(
                f"[Rotamer MD] Cycle {cycle_i}/3: Optimizing {len(current_structures)} structures "
                f"(fmax={cycle_fmax:.3g} eV/Å, ewin={cycle_ewin*EV_TO_KCALMOL:.1f} kcal/mol, "
                f"maxiter={cycle_maxiter}, max_step={cycle_max_step} Å)..."
            )
            is_last_cycle = cycle_i == len(opt_cycles)
            rot_ensemble, converged_flags = optimize_multilevel_jax(
                current_structures,
                energy_fn,
                fmax=cycle_fmax,
                maxiter=cycle_maxiter,
                parallel=parallel,
                max_stepsize=cycle_max_step,
                return_converged=True,
                potential_wrapper=potential_wrapper,
                fire_dt_start=fire_dt_start,
                fire_dt_max=fire_dt_max,
                fire_n_min=fire_n_min,
                method=method,
                **_ric_kwargs(
                    coords,
                    coords_coarse,
                    init_hessian,
                    interfragment,
                    ric_options,
                    initial_structure["numbers"],
                    cycle_i,
                ),
            )
            if is_last_cycle:
                n_nc = sum(1 for c in converged_flags if not c)
                if n_nc:
                    log_message(
                        f"[Rotamer MD] Discarding {n_nc} non-converged structures in final cycle"
                    )
                rot_ensemble = [(s, e) for (s, e), c in zip(rot_ensemble, converged_flags) if c]
            rot_ensemble = sort_by_energy(rot_ensemble)
            if use_topology:
                rot_ensemble = prune_by_topology(rot_ensemble, topology)
            if nci_cross_mask is not None:
                rot_ensemble = prune_by_nci_distance(
                    rot_ensemble,
                    nci_cross_mask,
                    nci_d_max * _NCI_DIST_FACTORS[cycle_i],
                    "Rotamer MD",
                )
            rot_ensemble = _prune_energy_window_guarded(
                rot_ensemble, cycle_ewin, is_last_cycle, cycle_i, len(opt_cycles), "Rotamer MD"
            )
            rot_ensemble = prune_by_rotational_constants(
                rot_ensemble, threshold=0.02, reference=ensemble
            )
            rot_ensemble = prune_by_rmsd(
                rot_ensemble,
                threshold=config["rmsd_threshold"],
                reference=ensemble,
                frag_perms=frag_perms,
            )
            log_message(
                f"[Rotamer MD] Cycle {cycle_i}/3: {len(current_structures)} -> {len(rot_ensemble)} conformers"
            )
            current_structures = [s for s, _ in rot_ensemble]
            if not current_structures:
                log_message("[Rotamer MD] No structures to optimize. Skiping the optimization...")
                break

        log_message(f"[Rotamer MD] Produced {len(rot_ensemble)} optimized structures")

        # Merge and prune
        ensemble.extend(rot_ensemble)
        ensemble = sort_by_energy(ensemble)
        ensemble = prune_by_energy_window(ensemble, ewin=config["ewin"])
        ensemble = prune_by_rotational_constants(ensemble, threshold=0.02)
        ensemble = prune_by_rmsd(
            ensemble, threshold=config["rmsd_threshold"], frag_perms=frag_perms
        )

        log_message(f"[Auto] After rotamer MD: {len(ensemble)} conformers")
        timer_end("Rotamer MD")

    # ========================================================================
    # Step 6: Genetic Crossing
    # ========================================================================
    gc_cfg = config["genetic_crossing"]
    if gc_cfg["enabled"] and len(ensemble) >= 2:
        log_header("Genetic Z-matrix Crossing")
        timer_start("Genetic Crossing")

        from .crossing import genetic_zmatrix_crossing

        n_children = gc_cfg["n_children"]
        log_message(f"[Auto] Generating {n_children} children via Z-matrix crossing")
        log_message(f"[Auto] Reference: lowest-energy conformer (index 0)")

        children = genetic_zmatrix_crossing(
            ensemble,
            reference_idx=0,
            n_children=n_children,
            boltzmann_weight=True,
            temperature=config["ewin"],
        )

        # Pre-optimization pruning in two steps:
        # 1. Prune against ensemble (optimized conformers) — discard anything in a known basin
        # 2. Prune survivors against seen_pool (previously explored starting points) — avoid
        #    re-optimising starting geometries that were already tried and discarded
        n_before = len(children)
        dummy_children = [(c, 0.0) for c in children]
        dummy_children = prune_by_rotational_constants(
            dummy_children, threshold=0.02, reference=ensemble
        )
        dummy_children = prune_by_rmsd(
            dummy_children,
            threshold=config["rmsd_threshold"],
            reference=ensemble,
            frag_perms=frag_perms,
        )
        dummy_children = prune_by_rotational_constants(
            dummy_children, threshold=0.02, reference=seen_pool
        )
        dummy_children = prune_by_rmsd(
            dummy_children,
            threshold=config["rmsd_threshold"],
            reference=seen_pool,
            frag_perms=frag_perms,
        )
        children = [c for c, _ in dummy_children]
        if len(children) < n_before:
            log_message(f"[Genetic] Pre-opt pruning: {n_before} -> {len(children)} children")

        # Pre-opt topology check: discard structures with severely broken bonds
        # before sending to the GPU optimizer (broken geometries can cause hangs).
        # Uses looser thresholds than the post-opt check since crossing can produce
        # mildly distorted geometries that the optimizer can still recover.
        if use_topology:
            n_before_topo = len(children)
            dummy_topo = [(c, 0.0) for c in children]
            dummy_topo = prune_by_topology(
                dummy_topo, topology, relative_stretch=2.0, absolute_increase=1.0
            )
            children = [c for c, _ in dummy_topo]
            if len(children) < n_before_topo:
                log_message(
                    f"[Genetic] Pre-opt topology check: {n_before_topo} -> {len(children)} children"
                )
        seen_pool.extend([(c, 0.0) for c in children])
        log_message(f"[Genetic] Seen pool: {len(seen_pool)} structures tracked")

        # 3-cycle optimization with progressive refinement
        current_structures = children
        gc_ensemble = []
        for cycle_i, (cycle_fmax, cycle_ewin, cycle_maxiter, cycle_max_step) in enumerate(
            opt_cycles, 1
        ):
            if not current_structures:
                log_message(f"[Genetic] Cycle {cycle_i}/3: No structures to optimize. Skipping...")
                break
            log_message(
                f"[Genetic] Cycle {cycle_i}/3: Optimizing {len(current_structures)} structures "
                f"(fmax={cycle_fmax:.3g} eV/Å, ewin={cycle_ewin*EV_TO_KCALMOL:.1f} kcal/mol, "
                f"maxiter={cycle_maxiter}, max_step={cycle_max_step} Å)..."
            )
            is_last_cycle = cycle_i == len(opt_cycles)
            gc_ensemble, converged_flags = optimize_multilevel_jax(
                current_structures,
                energy_fn,
                fmax=cycle_fmax,
                maxiter=cycle_maxiter,
                parallel=parallel,
                max_stepsize=cycle_max_step,
                return_converged=True,
                potential_wrapper=potential_wrapper,
                fire_dt_start=fire_dt_start,
                fire_dt_max=fire_dt_max,
                fire_n_min=fire_n_min,
                method=method,
                **_ric_kwargs(
                    coords,
                    coords_coarse,
                    init_hessian,
                    interfragment,
                    ric_options,
                    initial_structure["numbers"],
                    cycle_i,
                ),
            )
            if is_last_cycle:
                n_nc = sum(1 for c in converged_flags if not c)
                if n_nc:
                    log_message(
                        f"[Genetic] Discarding {n_nc} non-converged structures in final cycle"
                    )
                gc_ensemble = [(s, e) for (s, e), c in zip(gc_ensemble, converged_flags) if c]
            gc_ensemble = sort_by_energy(gc_ensemble)
            if use_topology:
                gc_ensemble = prune_by_topology(gc_ensemble, topology)
            if nci_cross_mask is not None:
                gc_ensemble = prune_by_nci_distance(
                    gc_ensemble, nci_cross_mask, nci_d_max * _NCI_DIST_FACTORS[cycle_i], "Genetic"
                )
            gc_ensemble = _prune_energy_window_guarded(
                gc_ensemble, cycle_ewin, is_last_cycle, cycle_i, len(opt_cycles), "Genetic"
            )
            gc_ensemble = prune_by_rotational_constants(
                gc_ensemble, threshold=0.02, reference=ensemble
            )
            gc_ensemble = prune_by_rmsd(
                gc_ensemble,
                threshold=config["rmsd_threshold"],
                reference=ensemble,
                frag_perms=frag_perms,
            )
            log_message(
                f"[Genetic] Cycle {cycle_i}/3: {len(current_structures)} -> {len(gc_ensemble)} conformers"
            )
            current_structures = [s for s, _ in gc_ensemble]
            if not current_structures:
                log_message("[Genetic] No structures to optimize. Skiping the optimization...")
                break

        log_message(f"[Genetic] {len(gc_ensemble)} children optimized")

        # Merge and prune
        ensemble.extend(gc_ensemble)
        ensemble = sort_by_energy(ensemble)
        ensemble = prune_by_energy_window(ensemble, ewin=config["ewin"])
        ensemble = prune_by_rotational_constants(ensemble, threshold=0.02)
        ensemble = prune_by_rmsd(
            ensemble, threshold=config["rmsd_threshold"], frag_perms=frag_perms
        )

        log_message(f"[Auto] After genetic crossing: {len(ensemble)} conformers")
        timer_end("Genetic Crossing")

    # ========================================================================
    # Step 7: Final output
    # ========================================================================
    log_header("Final Ensemble")
    log_message(f"[Auto] Final ensemble: {len(ensemble)} conformers")

    save_ensemble(output_file, ensemble)
    log_message(f"[Auto] Saved ensemble to {output_file}")
    log_energy_table(ensemble, title="Final auto-configured ensemble")

    if len(ensemble) > 0:
        energies = [e for _, e in ensemble]
        e_min = min(energies)
        log_message(f"[Auto] Energy span:  {(max(energies) - e_min) * EV_TO_KCALMOL:.1f} kcal/mol")

    log_header("Conformer Search Auto Complete")

    return ensemble
