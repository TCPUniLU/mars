"""
Automatic configuration for multi-step MTD conformational search.

Determines MTD parameters based on molecular properties:
  Phase 1: Multi-step MTD grid (different kpush/alpha per step)
  Phase 2: Conformer refinement from diverse starts (normal/thorough)

Optionally followed by:
  Phase 3: Rotamer MD
  Phase 4: Genetic Crossing

JAX-native implementation built on the MARS sampler.

Units: eV and Angstrom.
"""

import math
from typing import Dict, Optional

import numpy as np

from .log import log_header, log_message

# ============================================================================
# Unit Conversion Constants
# ============================================================================

KCALMOL_TO_EV = 0.043364104241833  # 1 kcal/mol = 0.04336 eV
EV_TO_KCALMOL = 1.0 / KCALMOL_TO_EV  # 1 eV = 23.0605 kcal/mol


# ============================================================================
# Flexibility Estimation
# ============================================================================


def estimate_flexibility(
    n_atoms: int,
    atomic_numbers: Optional[np.ndarray] = None,
    positions: Optional[np.ndarray] = None,
    n_bonds: Optional[int] = None,
) -> float:
    """Estimate molecular flexibility from structure properties.

    Uses a covalent-topology flexibility measure when structure is
    available. The flexibility combines:
      - Bond branching factor (linear chains are more flexible)
      - Ring rigidity (rings reduce flexibility by ~50%)
      - Hybridization (sp2 carbons are more rigid)
      - Bond orders (double/triple bonds reduce flexibility)

    When structure data is unavailable, falls back to atom count heuristics.

    Args:
        n_atoms: Number of atoms
        atomic_numbers: Optional array of atomic numbers
        positions: Optional (n_atoms, 3) positions for distance-based estimation
        n_bonds: Optional number of covalent bonds from topology detection

    Returns:
        flexibility: Scalar flexibility measure (0 = rigid, 1 = flexible long chain)
    """
    # Base flexibility from atom count
    if n_atoms <= 3:
        return 0.5

    # If we have full structure information, use the topology-based flexibility calculation
    if positions is not None and atomic_numbers is not None:
        try:
            from .flexibility import calculate_flexibility

            flex = calculate_flexibility(
                np.array(positions),
                np.array(atomic_numbers),
                bonds=None,  # Auto-detect bonds
                bond_orders=None,  # Assume single bonds
            )
            return flex
        except Exception as e:
            log_message(
                f"[Auto Setup] Warning: Molecular flexibility calculation failed ({e}), using heuristic"
            )
            # Fall through to heuristic method

    # Fallback: Heuristic flexibility from atom count and connectivity
    # Heavy atoms approximation (exclude H if atomic_numbers provided)
    if atomic_numbers is not None:
        n_heavy = int(np.sum(atomic_numbers > 1))
    else:
        # Assume ~60% heavy atoms for organic molecules
        n_heavy = max(2, int(0.6 * n_atoms))

    # If bond count is available, use it to improve estimate
    if n_bonds is not None and n_bonds > 0:
        linear_bonds = max(1, n_atoms - 1)
        bond_excess = (n_bonds - linear_bonds) / max(1, n_atoms)
        n_rot_approx = max(1, 0.65 * n_bonds * (n_heavy / max(1, n_atoms)))
        connectivity_factor = 1.0 + 0.3 * bond_excess
    else:
        n_rot_approx = max(1, n_heavy / 3.0)
        connectivity_factor = 1.0

    flex_per_bond = 0.7
    flexibility = flex_per_bond * math.sqrt(n_rot_approx / max(1, n_heavy)) * connectivity_factor
    flexibility = max(0.3, min(1.0, flexibility))

    return flexibility


# ============================================================================
# MD Length Determination
# ============================================================================


def compute_md_length(
    n_atoms: int,
    flexibility: Optional[float] = None,
    mode: str = "normal",
    phase: str = "exploration",
) -> float:
    """Compute MTD simulation time per step.

    Uses a flexibility-scaled exponential formula::

        av1   = flexibility * max(1, n_atoms - 8)
        tmtd  = 3.0 * exp(0.10 * av1)          [normal/thorough]
        tmtd  = tmtd * 0.5                      [quick: rfac = 0.5]
        tmtd  = max(5.0, min(lenthr, tmtd))     [clamp: lenthr = 500 ps]

    The ``phase`` argument selects an additional time scale factor:
      - 'exploration': full per-step time (rfac already applied for quick)
      - 'conformer':   0.5× the exploration time

    Args:
        n_atoms:     Number of atoms
        flexibility: Molecular flexibility (0-1). If None, estimated.
        mode:        'quick', 'normal', or 'thorough'
        phase:       'exploration' (full MTD step) or 'conformer' (0.5× time)

    Returns:
        md_time_ps: Simulation time per step in picoseconds
    """
    if flexibility is None:
        flexibility = estimate_flexibility(n_atoms)

    rednat = max(1, n_atoms - 8)
    av1 = flexibility * rednat

    # Same base and exponent for normal and thorough modes; 500 ps cap.
    tmtd = 3.0 * math.exp(0.10 * av1)
    tmtd = max(5.0, min(500.0, tmtd))

    # Quick mode runs at half the time.
    if mode == "quick":
        tmtd *= 0.5

    # Conformer refinement uses half the exploration time.
    if phase == "conformer":
        tmtd *= 0.5

    return tmtd


# ============================================================================
# MTD Parameter Generation
# ============================================================================


def generate_mtd_parameters(
    n_atoms: int, mode: str = "normal", mtd_kscal: float = 1.0, flexibility: Optional[float] = None
) -> Dict:
    """Generate multi-step MTD workflow parameters.

    Delegates to the flexibility module which creates a grid of (kpush, alpha)
    parameter combinations adapted to molecular flexibility and search mode.

    Args:
        n_atoms: Number of atoms
        mode: 'quick', 'normal', or 'thorough'
        mtd_kscal: Global kpush scaling factor (default: 1.0)
        flexibility: Molecular flexibility (0-1). If None, uses 0.5.

    Returns:
        config: Dict with 'mtd_steps', 'time_per_step_ps',
                'n_conformer_starts', 'refinement_time_ps',
                'refinement_params', 'min_rmsd_threshold'
    """
    from .flexibility import get_mtd_workflow_config

    if flexibility is None:
        flexibility = 0.5

    return get_mtd_workflow_config(flexibility, n_atoms, mode, mtd_kscal)


# ============================================================================
# Trial MTD Configuration
# ============================================================================


def get_trial_mtd_params(n_atoms: int) -> Dict:
    """Get parameters for a short trial MTD to validate settings.

    A 1 ps test MTD is run before the full search to verify that the
    simulation is stable with the chosen timestep and parameters.

    Args:
        n_atoms: Number of atoms

    Returns:
        trial_params: Dict with trial MTD configuration.
                      kpush in eV, alpha in 1/A^2.
    """
    return {
        "kpush": 1.0,  # eV
        "alpha": 1.8,  # 1/A^2
        "cvdump_fs": 20.0,  # Dump every 20 fs
        "time_ps": 1.0,  # 1 ps trial
        "dt": 0.5,  # Starting timestep in fs
        "max_retries": 6,  # Max retry attempts
        "min_dt": 0.25,  # Minimum timestep before giving up
    }


def run_trial_mtd(
    positions,
    energy_fn,
    n_atoms: int,
    T: float = 300.0,
    dt: float = 0.5,
    mass=None,
    potential_wrapper=None,
) -> Dict:
    """Run a short trial MTD to validate simulation settings.

    Uses an iterative timestep-validation pattern:

    1. Run 1 ps MTD with test parameters
    2. If simulation fails (NaN energies), reduce timestep and retry
    3. Return validated timestep and estimated runtime

    Args:
        positions: (n_atoms, 3) initial positions in Angstrom
        energy_fn: JAX energy function (eV)
        n_atoms: Number of atoms
        T: Temperature in Kelvin
        dt: Initial timestep in fs
        mass: (n_atoms,) per-atom masses in amu, or None for uniform mass 1.0

    Returns:
        result: Dict with 'success', 'dt', 'message' keys
    """
    import jax.numpy as jnp

    trial_params = get_trial_mtd_params(n_atoms)

    current_dt = dt
    attempt = 0

    while attempt < trial_params["max_retries"]:
        attempt += 1
        log_message(
            f"[Trial MTD] Attempt {attempt}: dt={current_dt:.3g} fs, "
            f"kpush={trial_params['kpush'] * EV_TO_KCALMOL:.3g} kcal/mol, "
            f"alpha={trial_params['alpha']:.2f} Å⁻²"
        )

        try:
            from .sampling import run_mtd_jax

            mtd_params = {
                "kpush": trial_params["kpush"],
                "alpha": trial_params["alpha"],
                "cvdump_fs": trial_params["cvdump_fs"],
            }

            trajectory, mtd_state = run_mtd_jax(
                positions,
                energy_fn,
                mtd_params,
                T=T,
                dt=current_dt,
                time_ps=trial_params["time_ps"],
                mass=mass,
                random_seed=42,
                potential_wrapper=potential_wrapper,
            )

            # Check for NaN in trajectory
            has_nan = False
            for frame in trajectory[-5:]:  # Check last few frames
                if jnp.any(jnp.isnan(frame)):
                    has_nan = True
                    break

            if has_nan:
                raise RuntimeError("NaN detected in trajectory")

            # Check for explosion (atoms too far apart)
            last_frame = trajectory[-1]
            max_dist = jnp.max(jnp.abs(last_frame - jnp.mean(last_frame, axis=0)))
            if max_dist > 100.0:  # Angstrom
                raise RuntimeError("Atoms dispersed too far (possible explosion)")

            log_message(
                f"[Trial MTD] Success with dt={current_dt:.3g} fs "
                f"({len(trajectory)} frames, {mtd_state.n_hills} hills)"
            )

            return {
                "success": True,
                "dt": current_dt,
                "n_frames": len(trajectory),
                "n_hills": mtd_state.n_hills,
                "message": f"Trial MTD succeeded with dt={current_dt:.3g} fs",
            }

        except Exception as e:
            last_error = e
            log_message(f"[Trial MTD] Failed (attempt {attempt}): {e}")

            # Halve the timestep and retry
            new_dt = current_dt / 2.0
            if new_dt >= trial_params["min_dt"]:
                current_dt = new_dt
                log_message(f"[Trial MTD] Reducing timestep to {current_dt:.3g} fs")
            else:
                log_message(
                    f"[Trial MTD] Cannot reduce timestep further (min={trial_params['min_dt']:.3g} fs)"
                )
                break

    raise RuntimeError(
        f"Trial MTD failed after {attempt} attempt(s). "
        f"Last timestep tried: {current_dt:.3g} fs. "
        f"Last error: {last_error}"
    )


# ============================================================================
# Main Automatic Configuration
# ============================================================================

# Default energy window: 10.0 kcal/mol ~ 0.433 eV
_DEFAULT_EWIN_EV = 0.433


def auto_setup(
    n_atoms: int,
    mode: str = "normal",
    atomic_numbers: Optional[np.ndarray] = None,
    positions: Optional[np.ndarray] = None,
    temperature: float = 300.0,
    dt: float = 0.5,
    mtd_kscal: float = 1.0,
    n_bonds: Optional[int] = None,
) -> Dict:
    """Compute automatic configuration for multi-step MTD search.

    Configures a grid of MTD simulations with varying bias parameters,
    adapted to molecular flexibility:
      Phase 1: Multi-step MTD grid (different kpush/alpha per step)
      Phase 2: Conformer refinement from diverse starts (normal/thorough)

    Followed optionally by rotamer MD and genetic crossing.

    Args:
        n_atoms: Number of atoms
        mode: 'quick', 'normal', or 'thorough'
        atomic_numbers: Optional atomic number array
        positions: Optional (n_atoms, 3) positions in Angstrom
        temperature: Simulation temperature in K (default: 300)
        dt: Timestep in fs (default: 0.5)
        mtd_kscal: Global kpush scaling factor (default: 1.0)
        n_bonds: Optional number of covalent bonds

    Returns:
        config: Configuration dict with keys:
            - 'mode', 'n_atoms', 'flexibility'
            - 'dt', 'temperature', 'ewin', 'rmsd_threshold', 'optlevel'
            - 'conformer_mtd': Conformer MTD workflow settings
            - 'rotamer_md': Rotamer MD settings
            - 'genetic_crossing': Genetic crossing settings
            - 'trial_params': Trial MTD parameters
    """
    log_header("Automatic Configuration")

    # --- Step 1: Flexibility ---
    flexibility = estimate_flexibility(n_atoms, atomic_numbers, positions, n_bonds)
    if n_bonds is not None:
        log_message(
            f"[Auto Setup] Molecular flexibility: {flexibility:.3g} (using {n_bonds} bonds)"
        )
    else:
        log_message(f"[Auto Setup] Molecular flexibility: {flexibility:.3g}")

    # --- Step 2: Multi-step MTD parameters (grid adapted to flexibility) ---
    conformer_mtd = generate_mtd_parameters(n_atoms, mode, mtd_kscal, flexibility=flexibility)
    log_message(
        f"[Auto Setup] MTD grid: {len(conformer_mtd['mtd_steps'])} steps, "
        f"{conformer_mtd['time_per_step_ps']:.1f} ps each"
    )
    if conformer_mtd["n_conformer_starts"] > 0:
        log_message(
            f"[Auto Setup] Refinement: {conformer_mtd['n_conformer_starts']} diverse starts, "
            f"{conformer_mtd['refinement_time_ps']:.1f} ps each"
        )

    # --- Step 3: Rotamer MD ---
    # nrotammds = max(1, nmetadyn / 4), 2 temperature replicas,
    # runtime = MTD_time * 0.5.
    n_mtd_steps = len(conformer_mtd["mtd_steps"])
    rotamer_time = conformer_mtd["refinement_time_ps"]  # = time_per_step * 0.5
    n_rotamer_strucs = max(1, n_mtd_steps // 4)
    rotamer_md = {
        "enabled": True,
        "n_structures": n_rotamer_strucs,
        "n_temps": 2,
        "time_ps": rotamer_time,
    }

    # --- Step 4: Genetic crossing ---
    # n_children = min(round(mdtime * 50.0), 5000); quick halves it.
    gc_time = conformer_mtd["time_per_step_ps"]
    gc_children = min(int(gc_time * 50.0), 5000)
    gc_settings = {
        "quick": {"enabled": False, "n_children": 0},
        "normal": {"enabled": True, "n_children": gc_children},
        "thorough": {"enabled": True, "n_children": gc_children},
    }
    genetic_crossing = gc_settings.get(mode, gc_settings["normal"])

    # --- General settings ---
    ewin = _DEFAULT_EWIN_EV
    rmsd_threshold = 0.250
    optlevel = {"quick": -1, "normal": 0, "thorough": 1}.get(mode, 0)

    config = {
        "mode": mode,
        "n_atoms": n_atoms,
        "flexibility": flexibility,
        "dt": dt,
        "temperature": temperature,
        "ewin": ewin,
        "rmsd_threshold": rmsd_threshold,
        "optlevel": optlevel,
        "conformer_mtd": conformer_mtd,
        "rotamer_md": rotamer_md,
        "genetic_crossing": genetic_crossing,
        "trial_params": get_trial_mtd_params(n_atoms),
    }

    return config


def _print_auto_config(config: Dict):
    """Print formatted Conformer MTD configuration summary."""
    log_message(f"")
    log_message(f"  ┍{'━'*70}┑")
    log_message(f"  │{'MARS Automatic Configuration':^70}│")
    log_message(f"  ┕{'━'*70}┙")
    log_message(f"")
    log_message(f"  Molecular Properties:")
    log_message(f"    Atoms:              {config['n_atoms']}")
    log_message(f"    Flexibility:        {config['flexibility']:.3g}")
    log_message(f"    Mode:               {config['mode']}")
    log_message(f"")
    log_message(f"  General Settings:")
    log_message(f"    Temperature:        {config['temperature']} K")
    log_message(f"    Timestep:           {config['dt']} fs")
    log_message(f"    Energy window:      {config['ewin'] * EV_TO_KCALMOL:.1f} kcal/mol")
    log_message(f"    RMSD threshold:     {config['rmsd_threshold']:.2f} Å")
    log_message(f"    Opt level:          {config['optlevel']}")
    log_message(f"")

    # Multi-step MTD Workflow
    cmtd = config["conformer_mtd"]
    mtd_steps = cmtd["mtd_steps"]
    log_message(f"  ╔{'═'*68}╗")
    log_message(f"  ║{'  Workflow: Multi-Step MTD':^68}║")
    log_message(f"  ╚{'═'*68}╝")
    log_message(f"")

    log_message(f"  Phase 1 - Multi-Step MTD Grid:")
    log_message(f"    MTD steps:        {len(mtd_steps)}")
    log_message(f"    Time per step:    {cmtd['time_per_step_ps']:.3g} ps")
    log_message(f"    Total MTD time:   {len(mtd_steps) * cmtd['time_per_step_ps']:.3g} ps")
    log_message(f"")
    log_message(f"    Step parameters:")
    for i, step in enumerate(mtd_steps):
        log_message(
            f"      [{i+1:2d}] kpush = {step['kpush'] * EV_TO_KCALMOL:.3g} kcal/mol, "
            f"alpha = {step['alpha']:.2f} Å⁻², "
            f"cvdump = {step['cvdump_fs']:.3g} fs"
        )
    log_message(f"")

    if cmtd["n_conformer_starts"] > 0:
        log_message(f"  Phase 2 - Conformer Refinement:")
        log_message(f"    Diverse starts:   {cmtd['n_conformer_starts']}")
        log_message(f"    Time each:        {cmtd['refinement_time_ps']:.3g} ps")
        ref = cmtd["refinement_params"]
        log_message(f"    Bias strength:    kpush = {ref['kpush'] * EV_TO_KCALMOL:.3g} kcal/mol")
        log_message(f"    Bias width:       alpha = {ref['alpha']:.2f} Å⁻²")
        log_message(f"    RMSD threshold:   >= {cmtd['min_rmsd_threshold']:.2f} Å")
    else:
        log_message(f"  Conformer Refinement: disabled (quick mode)")
    log_message(f"")

    # Rotamer MD
    rot = config["rotamer_md"]
    phase_num = 3 if cmtd["n_conformer_starts"] > 0 else 2
    if rot["enabled"]:
        log_message(f"  Phase {phase_num} - Rotamer MD:")
        log_message(f"    Structures:       {rot['n_structures']} lowest-energy conformers")
        log_message(f"    Temperatures:     {rot['n_temps']} replicas")
        log_message(f"    Time each:        {rot['time_ps']:.3g} ps")
    else:
        log_message(f"  Rotamer MD:         disabled")

    # Genetic crossing
    log_message(f"")
    gc = config["genetic_crossing"]
    phase_num += 1
    if gc["enabled"]:
        log_message(f"  Phase {phase_num} - Genetic Crossing:")
        log_message(f"    Children:         {gc['n_children']} offspring structures")
    else:
        log_message(f"  Genetic Crossing:   disabled")

    log_message(f"")
    log_message(f"  {'─'*70}")
    log_message(f"")
