"""
Conformational search CLI workflow (default command).
"""

import sys

from ..auto_config import EV_TO_KCALMOL, KCALMOL_TO_EV
from ..conf_sampling import run_conformer_search_jax, run_md_only, run_mtd_only
from ..log import (
    log_error,
    log_info,
    log_message,
    log_timing_summary,
    log_warning,
    timer_end,
    timer_reset,
    timer_start,
)
from ..utils import load_structure
from ._common import build_potential, setup_logging, setup_precision, validate_input_file
from ._constraints import log_constraints, make_frozen_energy_fn


def _log_crest_citation():
    """Note that the conformational search is inspired by CREST (please cite)."""
    log_message("")
    log_message(
        "This conformational search workflow is inspired by CREST " "(Grimme and co-workers)."
    )
    log_message("If you use it, please cite the CREST papers:")
    log_message(
        "  P. Pracht, F. Bohle, S. Grimme, "
        "Phys. Chem. Chem. Phys. 22 (2020) 7169-7192. "
        "doi:10.1039/C9CP06869D"
    )
    log_message(
        "  P. Pracht, S. Grimme, C. Bannwarth, et al., "
        "J. Chem. Phys. 160 (2024) 114110. "
        "doi:10.1063/5.0197592"
    )
    log_message("")


def run_conformer_search_workflow(args):
    """Run conformational search workflow (default CLI command)."""
    setup_logging(args)
    _log_crest_citation()
    timer_reset()
    timer_start("Initialization")
    input_path = validate_input_file(args.input)
    setup_precision(args)

    # Load structure
    log_info(f"Loading structure from {args.input}")
    try:
        structure = load_structure(str(input_path))
        n_atoms = structure["positions"].shape[0]
        log_info(f"Loaded structure with {n_atoms} atoms")
    except Exception as e:
        log_error(f"Failed to load structure: {e}")
        sys.exit(1)

    # In NCI mode, pre-centre the input positions by their *centroid*
    # BEFORE the potential is initialized. This matters because the
    # MTD/MD inner loop centres by centroid at every step
    # (`mars/sampling.py:center_positions` = `p - mean(p, axis=0)`).
    # If we anchored the confinement wall in lab coordinates and the
    # MD then shifted the molecule to a centroid at origin, the wall
    # would float at `centroid_input` away from the molecule — the
    # offset reduces the effective buffer on one side and lets atoms
    # cross the wall under thermal drift, triggering the polynomial-12
    # force blow-up. By moving the input to a frame where the centroid
    # is already at origin AND anchoring the wall at origin, the wall
    # and the MD frame are consistent and every atom truly has the
    # requested buffer of clearance.
    if args.nci:
        import numpy as np

        from ..rmsd import center_positions

        positions_lab = structure["positions"]
        centroid_lab = np.asarray(np.mean(np.asarray(positions_lab), axis=0))
        log_info(
            f"NCI: pre-centring positions by centroid "
            f"({centroid_lab[0]:.3f}, {centroid_lab[1]:.3f}, {centroid_lab[2]:.3f}) Å "
            f"so the MD frame and the confinement wall stay consistent."
        )
        # Use the same centring helper the MTD inner loop uses
        # (`mars.rmsd.center_positions`) so the input frame matches the
        # frame the MD operates in — guarantees the wall (anchored at
        # the origin below) coincides with the MD-frame centroid.
        structure = dict(structure)
        structure["positions"] = center_positions(positions_lab)

    # Build energy function from potential
    # (build_potential logs the potential + model and its citation)
    if args.potential in ("harmonic", "lj"):
        log_warning(
            "Using test potential (harmonic/lj). For production use, the default so3lr potential is recommended."
        )
    try:
        potential_wrapper, energy_fn = build_potential(args, structure)
    except ImportError as e:
        log_error(f"Failed to load potential '{args.potential}': {e}")
        sys.exit(1)
    except Exception as e:
        log_error(f"Failed to initialize potential '{args.potential}': {e}")
        sys.exit(1)

    # Wrap energy function with confinement potential in NCI mode
    if args.nci:
        import numpy as np

        from ..potentials import get_potential

        positions = np.asarray(structure["positions"])  # already centred
        # The wall is anchored at the origin of the centred frame; this
        # coincides with the centroid of the molecule, which the MTD/MD
        # loop also pins to the origin at every step. Radius =
        # max_dist_from_origin + buffer guarantees that the farthest
        # atom has exactly `buffer` Å of clearance and every other atom
        # has more.
        center = np.zeros(3, dtype=positions.dtype)
        distances = np.linalg.norm(positions - center, axis=-1)
        max_dist = float(distances.max())
        conf_radius = max_dist + float(args.nci_buffer)

        min_clearance = float(conf_radius - distances.max())
        log_info(
            f"NCI mode: spherical confinement ({args.nci_mode}) "
            f"radius={conf_radius:.2f} Å, centre=(0.000, 0.000, 0.000) Å "
            f"(coincides with MD-frame centroid), "
            f"extent={max_dist:.2f} Å, buffer={args.nci_buffer:.1f} Å, "
            f"min atom-to-wall clearance={min_clearance:.2f} Å"
        )
        # Defensive assertion: every atom must start inside the wall.
        if distances.max() >= conf_radius:
            log_warning(
                f"NCI: input geometry has an atom outside the confinement "
                f"(dist {distances.max():.2f} Å ≥ radius {conf_radius:.2f} Å). "
                f"This will cause large initial forces — check the input."
            )

        conf_potential = get_potential(
            "confinement", radii=conf_radius, center=center, mode=args.nci_mode
        )
        conf_potential.initialize(positions)
        conf_energy_fn = conf_potential.build_energy_fn()
        _base_energy_fn = energy_fn

        def energy_fn(pos, **kwargs):  # noqa: F811
            return _base_energy_fn(pos, **kwargs) + conf_energy_fn(pos)

    # Apply frozen-atom constraints (config file only, not a CLI flag)
    constraint_atoms = getattr(args, "constraint_atoms", None)
    if constraint_atoms is not None:
        apply_to = getattr(args, "constraint_apply_to", "all")
        if apply_to in ("all", "mtd"):
            log_constraints(log_info, constraint_atoms, apply_to)
            energy_fn = make_frozen_energy_fn(energy_fn, structure["positions"], constraint_atoms)

    timer_end("Initialization")

    # Determine search mode
    if args.mode:
        search_mode = args.mode
    elif args.quick:
        search_mode = "quick"
    elif args.thorough:
        search_mode = "thorough"
    else:
        search_mode = "normal"

    # Convert CLI inputs from kcal/mol to internal eV (None stays None)
    ewin_ev = args.ewin * KCALMOL_TO_EV if args.ewin is not None else None

    # Build manual MTD params if user specified non-default values
    mtd_params = None
    if args.kpush != 20.0 or args.alpha != 0.5 or args.cvdump_fs != 50.0:
        mtd_params = {
            "kpush": args.kpush * KCALMOL_TO_EV,
            "alpha": args.alpha,
            "cvdump_fs": args.cvdump_fs,
        }
        log_info(
            f"Using manual MTD parameters: kpush={args.kpush} kcal/mol, "
            f"alpha={args.alpha}, cvdump_fs={args.cvdump_fs} fs"
        )

    # Run workflow based on mode
    try:
        parallel = not args.not_parallel

        opt_kwargs = dict(
            method=getattr(args, "method", "LBFGS"),
            final_fmax=getattr(args, "fmax", None),
            final_maxiter=getattr(args, "maxiter", None),
            final_max_stepsize=getattr(args, "max_stepsize", None),
        )

        if args.md_only:
            log_info("Running plain MD workflow (no metadynamics)")
            ensemble = run_md_only(
                structure,
                energy_fn,
                md_time_ps=args.mtd_time if args.mtd_time != 10.0 else None,
                temperature=args.temperature if args.temperature != 300.0 else None,
                dt=args.mtd_dt if args.mtd_dt != 0.5 else None,
                optlevel=args.optlevel,
                ewin=ewin_ev,
                save_trajectory=args.save_trajectory,
                potential_wrapper=potential_wrapper,
                parallel=parallel,
                nci=args.nci,
                nci_max_dist=args.nci_max_dist,
                fire_dt_start=args.fire_dt_start,
                fire_dt_max=args.fire_dt_max,
                fire_n_min=args.fire_n_min,
                output_file=args.output,
                **opt_kwargs,
            )

        elif args.mtd_only:
            log_info(f"Running Conformer MTD-only workflow (mode={search_mode})")
            ensemble = run_mtd_only(
                structure,
                energy_fn,
                mode=search_mode,
                temperature=args.temperature if args.temperature != 300.0 else None,
                dt=args.mtd_dt if args.mtd_dt != 0.5 else None,
                optlevel=args.optlevel,
                ewin=ewin_ev,
                mtd_params=mtd_params,
                mtd_kscal=args.kscal,
                trial=not args.no_trial,
                save_trajectory=args.save_trajectory,
                potential_wrapper=potential_wrapper,
                use_topology=not args.no_topology,
                n_conformer_starts=args.n_conformer_starts,
                exploration_time_ps=args.exploration_time,
                conformer_time_ps=args.conformer_time,
                parallel=parallel,
                nci=args.nci,
                nci_max_dist=args.nci_max_dist,
                fire_dt_start=args.fire_dt_start,
                fire_dt_max=args.fire_dt_max,
                fire_n_min=args.fire_n_min,
                output_file=args.output,
                **opt_kwargs,
            )

        else:
            # Default: full Conformer MTD workflow
            log_info(f"Running Conformer MTD workflow (mode={search_mode})")

            ensemble = run_conformer_search_jax(
                structure,
                energy_fn,
                mode=search_mode,
                temperature=args.temperature if args.temperature != 300.0 else None,
                dt=args.mtd_dt if args.mtd_dt != 0.5 else None,
                ewin=ewin_ev,
                optlevel=args.optlevel,
                mtd_params=mtd_params,
                mtr_rotamer_number=(
                    args.rotamer_structures if args.rotamer_structures != 5 else None
                ),
                mtr_temps=args.rotamer_temps if args.rotamer_temps != 3 else None,
                mtr_time_ps=args.rotamer_time if args.rotamer_time != 10.0 else None,
                genetic_crossing=args.genetic_crossing if args.genetic_crossing else None,
                n_children=args.n_children if args.n_children != 20 else None,
                mtd_kscal=args.kscal,
                trial=not args.no_trial,
                save_trajectory=args.save_trajectory,
                potential_wrapper=potential_wrapper,
                use_topology=not args.no_topology,
                n_conformer_starts=args.n_conformer_starts,
                exploration_time_ps=args.exploration_time,
                conformer_time_ps=args.conformer_time,
                parallel=parallel,
                nci=args.nci,
                nci_max_dist=args.nci_max_dist,
                fire_dt_start=args.fire_dt_start,
                fire_dt_max=args.fire_dt_max,
                fire_n_min=args.fire_n_min,
                output_file=args.output,
                **opt_kwargs,
            )

    except Exception as e:
        log_error(f"Workflow failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Summary
    log_info("")
    log_info("=" * 70)
    log_info("Workflow Complete")
    log_info("=" * 70)
    log_info(f"Found {len(ensemble)} unique conformers")

    if len(ensemble) > 0:
        energies = [e for _, e in ensemble]
        e_min = min(energies)
        log_info(f"Energy span: {(max(energies) - e_min) * EV_TO_KCALMOL:.1f} kcal/mol")

        log_info("")
        log_info("Lowest 10 conformers ΔE (kcal/mol):")
        for i, (_, e) in enumerate(ensemble[:10]):
            log_info(f"  {i+1:3d}. {(e - e_min) * EV_TO_KCALMOL:.2f} kcal/mol")

    if args.nci:
        import numpy as np

        # In NCI mode `structure["positions"]` were pre-centred earlier
        # so the centroid sits at the origin and the wall is anchored
        # there. Reflect that here so the post-run summary matches what
        # was actually built.
        positions = np.asarray(structure["positions"])
        max_dist = float(np.max(np.linalg.norm(positions, axis=-1)))
        conf_radius = max_dist + float(args.nci_buffer)
        log_info("")
        log_info("NCI Mode Settings:")
        log_info(f"  Confinement type   : {args.nci_mode}")
        log_info(
            "  Centre             : (0.00, 0.00, 0.00) Å "
            "(positions pre-centred by their input centroid; wall at the MD-frame centroid)"
        )
        log_info(f"  Molecular extent   : {max_dist:.2f} Å")
        log_info(f"  Buffer             : {args.nci_buffer:.1f} Å")
        log_info(f"  Wall radius        : {conf_radius:.2f} Å")
        log_info(f"  Force constant (k) : {10.0 * EV_TO_KCALMOL:.1f} kcal/mol")
        if args.nci_mode == "wall":
            log_info(f"  Wall exponent (n)  : 12")
            log_info(f"  Energy form        : k · max(0, d-1)^n")
        else:
            log_info(f"  Energy form        : k · max(0, d-1)²")

    log_info("")
    log_info("Output files:")
    log_info(f"  - {args.output} (final ensemble)")
    if not args.md_only:
        log_info(f"  - mtd_ensemble.xyz (after MTD cycles)")

    if args.save_trajectory:
        log_info(f"  - mtd_*_trajectory.xyz (MTD grid trajectories)")
        log_info(f"  - refinement_mtd_*_trajectory.xyz (refinement MTD trajectories)")
    if args.log_file:
        log_info(f"  - {args.log_file} (log file)")

    log_info("")
    log_info("Success!")
    log_info("=" * 70)

    log_timing_summary()

    return 0
