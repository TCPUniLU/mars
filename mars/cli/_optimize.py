"""
Structure optimization CLI workflow.
"""

import sys

from ..auto_config import EV_TO_KCALMOL
from ..log import (
    log_error,
    log_info,
    log_timing_summary,
    log_warning,
    timer_end,
    timer_reset,
    timer_start,
)
from ..utils import load_structure, save_ensemble
from ._common import build_potential, setup_logging, setup_precision, validate_input_file
from ._constraints import log_constraints, make_frozen_energy_fn


def run_optimize_workflow(args):
    """Run structure optimization workflow."""
    from ..optimizer import optimize_multilevel_jax, optimize_single

    setup_logging(args)
    timer_reset()
    timer_start("Initialization")
    input_path = validate_input_file(args.input)
    setup_precision(args)

    if getattr(args, "task", "minimize") == "stationary":
        if args.method != "SP":
            log_info(f"--task stationary: overriding --method {args.method} with SP")
        args.method = "SP"

    # Load structure(s)
    log_info(f"Loading structure from {args.input}")
    try:
        # Try to load all structures from multi-frame XYZ
        try:
            from ase import io

            from ..utils import ase_atoms_to_structure

            atoms_list = io.read(str(input_path), index=":")
            if not isinstance(atoms_list, list):
                atoms_list = [atoms_list]
            structures = [ase_atoms_to_structure(atoms) for atoms in atoms_list]
        except ImportError:
            # Fallback: single structure only
            structures = [load_structure(str(input_path))]

        n_structures = len(structures)
        log_info(f"Loaded {n_structures} structure(s)")

        # Handle conformer selection
        if args.conformer is not None:
            if args.conformer >= n_structures:
                log_error(f"Conformer index {args.conformer} out of range (max: {n_structures-1})")
                sys.exit(1)
            structures = [structures[args.conformer]]
            log_info(f"Selected conformer {args.conformer}")
        elif not args.all_conformers and n_structures > 1:
            log_info(
                f"Multiple conformers found. Using first conformer. Use --conformer N or --all-conformers."
            )
            structures = [structures[0]]

    except Exception as e:
        log_error(f"Failed to load structure: {e}")
        sys.exit(1)

    # Build energy function using first structure for initialization
    # (build_potential logs the potential + model and its citation)
    try:
        potential_wrapper, energy_fn = build_potential(args, structures[0])
    except Exception as e:
        log_error(f"Failed to initialize potential: {e}")
        sys.exit(1)

    # Apply frozen-atom constraints (config file only, not a CLI flag)
    constraint_atoms = getattr(args, "constraint_atoms", None)
    if constraint_atoms is not None:
        apply_to = getattr(args, "constraint_apply_to", "all")
        if apply_to in ("all", "optimization"):
            log_constraints(log_info, constraint_atoms, apply_to)
            energy_fn = make_frozen_energy_fn(
                energy_fn, structures[0]["positions"], constraint_atoms
            )

    timer_end("Initialization")
    timer_start("Optimization")

    # Process each structure
    optimized_structures = []
    all_info = []

    parallel = not args.not_parallel
    use_batch = parallel and len(structures) > 1

    try:
        if use_batch:
            # Parallel path: optimize all structures in a single vmapped call.
            log_info(
                f"Optimizing {len(structures)} structures in parallel "
                f"(method={args.method}, fmax={args.fmax} eV/Å, "
                f"maxiter={args.maxiter}, max_stepsize={args.max_stepsize})"
            )
            ensemble, converged_flags = optimize_multilevel_jax(
                structures,
                energy_fn,
                fmax=args.fmax,
                maxiter=args.maxiter,
                method=args.method,
                max_stepsize=args.max_stepsize,
                parallel=True,
                return_converged=True,
                potential_wrapper=potential_wrapper,
                fire_dt_start=args.fire_dt_start,
                fire_dt_max=args.fire_dt_max,
                fire_n_min=args.fire_n_min,
            )
            for idx, ((struct_opt, energy_opt), conv) in enumerate(zip(ensemble, converged_flags)):
                optimized_structures.append(struct_opt)
                all_info.append(
                    {
                        "converged": conv,
                        "iterations": args.maxiter,
                        "max_force": 0.0,
                        "grad_norm": 0.0,
                    }
                )
                if not conv:
                    log_warning(
                        f"Conformer {idx} not fully converged after "
                        f"{args.maxiter} iterations (fmax={args.fmax:.3g} eV/Å not reached)"
                    )
        else:
            # Sequential path (used when --not-parallel or single structure).
            for idx, structure in enumerate(structures):
                if len(structures) > 1:
                    log_info("")
                    log_info(f"{'='*70}")
                    log_info(f"Processing conformer {idx}")
                    log_info(f"{'='*70}")

                positions_init = structure["positions"]
                n_atoms = positions_init.shape[0]
                log_info(f"  Atoms: {n_atoms}")
                log_info(
                    f"  Starting optimization (method={args.method}, fmax={args.fmax:.3g} eV/Å, "
                    f"maxiter={args.maxiter}, max_stepsize={args.max_stepsize})"
                )

                positions_opt, energy_opt, info = optimize_single(
                    positions_init,
                    energy_fn,
                    fmax=args.fmax,
                    maxiter=args.maxiter,
                    method=args.method,
                    max_stepsize=args.max_stepsize,
                    potential_wrapper=potential_wrapper,
                    fire_dt_start=args.fire_dt_start,
                    fire_dt_max=args.fire_dt_max,
                    fire_n_min=args.fire_n_min,
                )
                if info["converged"]:
                    log_info(f"  ✓ Converged in {info['iterations']} iterations")
                else:
                    log_info(f"  ⚠ Not converged after {info['iterations']} iterations")
                log_info(f"  Final energy: {energy_opt * EV_TO_KCALMOL:.2f} kcal/mol")
                log_info(f"  Max force: {info['max_force']:.3g} eV/Å")

                opt_structure = structure.copy()
                opt_structure["positions"] = positions_opt
                opt_structure["energy"] = energy_opt
                optimized_structures.append(opt_structure)
                all_info.append(info)

        # Save optimized structure(s)
        if len(optimized_structures) > 0:
            ensemble = [(struct, struct["energy"]) for struct in optimized_structures]
            save_ensemble(args.output, ensemble)

        # Summary
        log_info("")
        log_info("=" * 70)
        log_info("Optimization Complete")
        log_info("=" * 70)
        log_info(f"Method: {args.method}")
        log_info(f"Structures processed: {len(optimized_structures)}")

        # Show summary for each structure
        for idx, (opt_struct, info) in enumerate(zip(optimized_structures, all_info)):
            if len(optimized_structures) > 1:
                log_info(f"\nConformer {idx}:")
            log_info(f"  Converged: {info['converged']}")
            log_info(f"  Iterations: {info['iterations']}")
            log_info(f"  Final energy: {opt_struct['energy'] * EV_TO_KCALMOL:.2f} kcal/mol")
            if "is_stationary_point" in info:
                log_info(f"  Is SP: {info['is_stationary_point']}")

        log_info("")
        if len(optimized_structures) > 1:
            log_info(f"Output: {args.output} (multi-frame, {len(optimized_structures)} structures)")
        else:
            log_info(f"Output: {args.output}")
        timer_end("Optimization")
        log_info("")
        log_info("Success!")
        log_info("=" * 70)

        log_timing_summary()

    except Exception as e:
        log_error(f"Optimization failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    return 0
