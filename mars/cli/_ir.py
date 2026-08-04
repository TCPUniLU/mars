"""
IR spectrum calculation CLI workflow.
"""

import sys

from ..log import (
    log_error,
    log_info,
    log_timing_summary,
    log_warning,
    timer_end,
    timer_reset,
    timer_start,
)
from ._common import log_potential_and_citation, setup_logging, validate_input_file
from ._config import parse_atom_indices


def run_ir_workflow(args):
    """Run IR spectrum calculation workflow."""
    from .. import ir

    setup_logging(args)
    timer_reset()
    input_path = validate_input_file(args.input)

    n_replicas = getattr(args, "n_replicas", 1)
    parallel = not getattr(args, "not_parallel", False)
    window_arg = args.window if args.window != "none" else None
    optimize = not getattr(args, "no_optimize", False)

    atoms_raw = getattr(args, "atoms", None)
    if atoms_raw:
        # argparse delivers a list of tokens with nargs="+". Stitch them
        # back together so the user can write either "--atoms 0-3,7" or
        # "--atoms 0 1 2" or "--atoms 0-3 7 10-12".
        if isinstance(atoms_raw, (list, tuple)):
            joined = ",".join(str(x) for x in atoms_raw)
        else:
            joined = str(atoms_raw)
        try:
            atom_indices = parse_atom_indices(joined)
        except (ValueError, TypeError) as exc:
            log_error(f"Invalid --atoms specification {atoms_raw!r}: {exc}")
            sys.exit(1)
    else:
        atom_indices = None

    potential_options = {}
    if args.potential == "mace":
        potential_options["mace_foundation"] = getattr(args, "mace_foundation", None)
        potential_options["mace_model"] = getattr(args, "mace_model", None)
        potential_options["mace_cache_dir"] = getattr(args, "mace_cache_dir", None)
        potential_options["mace_spin"] = getattr(args, "spin", None)
    elif args.potential == "dxtb":
        potential_options["dxtb_method"] = getattr(args, "dxtb_method", None)

    # Announce the potential + model and its citation (IR builds the potential
    # internally rather than through build_potential).
    log_potential_and_citation(args)

    try:
        timer_start("IR Calculation")
        if args.all_conformers and not args.md:
            # Hessian-based IR for all conformers (no MD)
            all_results = ir.compute_ir_all_conformers(
                str(input_path),
                potential_name=args.potential,
                model_path=args.so3lr_model,
                temperature=args.temperature,
                charge=args.charge,
                lr_cutoff=args.lr_cutoff,
                use_float64=args.float64,
                output_prefix=args.output_prefix,
                optimize=optimize,
                fmax=args.fmax,
                fd_hessian=args.fd_hessian,
                fd_displacement=args.fd_displacement,
                potential_options=potential_options,
                atom_indices=atom_indices,
                classify=not args.no_analysis,
                save_mode_xyz=args.mode_xyz,
                save_hessian=args.save_hessian,
                save_structure=args.save_structure,
                save_analysis=not args.no_analysis,
            )

            log_info("")
            log_info("=" * 70)
            log_info("IR Calculation Complete - All Conformers")
            log_info("=" * 70)
            log_info(f"Processed {len(all_results)} conformers")
            log_info(f"Output files:")
            log_info(f"  - {args.output_prefix}_N_ir_spectrum.dat (IR spectrum data per conformer)")
            log_info(f"  - {args.output_prefix}_N_ir_spectrum.png (IR spectrum plot per conformer)")
            if args.mode_xyz:
                log_info(
                    f"  - {args.output_prefix}_N_normal_modes.xyz (normal modes per conformer)"
                )
            if args.save_hessian:
                log_info(
                    f"  - {args.output_prefix}_N_ir_hessian_data.txt (Hessian data per conformer)"
                )
            if args.save_structure:
                log_info(
                    f"  - {args.output_prefix}_N_optimized_structure.xyz "
                    f"(optimized geometry per conformer)"
                )
            if not args.no_analysis:
                log_info(
                    f"  - {args.output_prefix}_N_mode_analysis.txt (mode analysis per conformer)"
                )
            if args.log_file:
                log_info(f"  - {args.log_file} (log file)")

        elif args.md and (args.all_conformers or n_replicas > 1):
            # MD-based IR with multiple conformers and/or replicas
            conformer_indices = (
                None
                if args.all_conformers
                else ([args.conformer] if args.conformer is not None else [0])
            )
            results = ir.compute_ir_from_md_parallel(
                str(input_path),
                potential_name=args.potential,
                model_path=args.so3lr_model,
                temperature=args.temperature,
                charge=args.charge,
                lr_cutoff=args.lr_cutoff,
                use_float64=args.float64,
                conformer_indices=conformer_indices,
                n_replicas=n_replicas,
                nvt_time_ps=args.nvt_time,
                nve_time_ps=args.nve_time,
                dt_fs=args.md_timestep,
                dipole_save_fs=args.dipole_save_fs,
                chop=args.chop,
                window=window_arg,
                parallel=parallel,
                potential_options=potential_options,
                atom_indices=atom_indices,
            )

            # Save averaged spectrum
            ir.save_ir_spectrum(results, args.output)
            if args.plot:
                ir.save_ir_plot(
                    results,
                    output_file=args.plot_output,
                    freq_range=tuple(args.freq_range),
                    broadening=args.broadening,
                    gaussian_sigma=args.gaussian_sigma,
                )

            n_total = results.get("n_total_trajectories", n_replicas)
            log_info("")
            log_info("=" * 70)
            log_info("IR Calculation Complete - Parallel MD")
            log_info("=" * 70)
            log_info(f"Trajectories averaged: {n_total}")
            log_info(f"Output files:")
            log_info(f"  - {args.output} (IR spectrum data)")
            if args.plot:
                log_info(f"  - {args.plot_output} (IR spectrum plot)")
            if args.log_file:
                log_info(f"  - {args.log_file} (log file)")

        else:
            # Single conformer / single replica
            if args.md:
                # MD-based IR calculation using dipole autocorrelation
                results = ir.compute_ir_from_md(
                    str(input_path),
                    potential_name=args.potential,
                    model_path=args.so3lr_model,
                    temperature=args.temperature,
                    charge=args.charge,
                    lr_cutoff=args.lr_cutoff,
                    use_float64=args.float64,
                    conformer_index=args.conformer,
                    nvt_time_ps=args.nvt_time,
                    nve_time_ps=args.nve_time,
                    dt_fs=args.md_timestep,
                    dipole_save_fs=args.dipole_save_fs,
                    chop=args.chop,
                    window=window_arg,
                    trajectory_file=args.trajectory_file,
                    random_seed=0,
                    potential_options=potential_options,
                    atom_indices=atom_indices,
                )
            else:
                # Hessian-based IR calculation (analytical or finite differences)
                results = ir.compute_ir_spectrum(
                    str(input_path),
                    potential_name=args.potential,
                    model_path=args.so3lr_model,
                    temperature=args.temperature,
                    charge=args.charge,
                    lr_cutoff=args.lr_cutoff,
                    use_float64=args.float64,
                    conformer_index=args.conformer,
                    optimize=optimize,
                    fmax=args.fmax,
                    classify=not args.no_analysis,
                    fd_hessian=args.fd_hessian,
                    fd_displacement=args.fd_displacement,
                    potential_options=potential_options,
                    atom_indices=atom_indices,
                )

            # Save results
            ir.save_ir_spectrum(results, args.output)

            # Save detailed Hessian data if requested (only for Hessian-based approach)
            if args.save_hessian and not args.md:
                ir.save_ir_hessian_data(results, args.hessian_output)
            elif args.save_hessian and args.md:
                log_warning("--save-hessian flag ignored for MD-based IR (Hessian approach only)")

            # Save optimized structure if requested (only for Hessian-based approach)
            if args.save_structure and not args.md:
                ir.save_optimized_structure(results, args.structure_output)
            elif args.save_structure and args.md:
                log_warning("--save-structure flag ignored for MD-based IR (Hessian approach only)")

            # Save normal mode XYZ file if requested (only for Hessian-based approach)
            if args.mode_xyz and not args.md:
                ir.save_normal_mode_xyz(results, output_file=args.mode_output)
            elif args.mode_xyz and args.md:
                log_warning("--mode-xyz flag ignored for MD-based IR (Hessian approach only)")

            # Save mode analysis by default (only for Hessian-based approach)
            if not args.no_analysis and not args.md:
                ir.save_mode_analysis(results, args.analysis_output)

            # Generate plot if requested
            if args.plot:
                ir.save_ir_plot(
                    results,
                    output_file=args.plot_output,
                    freq_range=tuple(args.freq_range),
                    broadening=args.broadening,
                    gaussian_sigma=args.gaussian_sigma,
                )

            log_info("")
            log_info("=" * 70)
            log_info("IR Calculation Complete")
            log_info("=" * 70)
            log_info(f"Output files:")
            log_info(f"  - {args.output} (IR spectrum data)")
            if args.save_hessian and not args.md:
                log_info(f"  - {args.hessian_output} (detailed Hessian data)")
            if args.save_structure and not args.md:
                log_info(f"  - {args.structure_output} (optimized structure)")
            if args.mode_xyz and not args.md:
                log_info(f"  - {args.mode_output} (all normal modes with eigenvectors)")
            if not args.no_analysis and not args.md:
                log_info(f"  - {args.analysis_output} (vibrational mode analysis)")
            if args.plot:
                log_info(f"  - {args.plot_output} (IR spectrum plot)")
            if args.log_file:
                log_info(f"  - {args.log_file} (log file)")

        timer_end("IR Calculation")
        log_info("")
        log_info("Success!")
        log_info("=" * 70)

        log_timing_summary()

    except Exception as e:
        log_error(f"IR calculation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    return 0
