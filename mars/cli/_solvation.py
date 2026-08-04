"""
Solvation CLI workflow.

Implements:
    mars solvation input.xyz --solvent water --padding 5.0 [options]
    mars solvation input.xyz --solvent water --n-molecules 30 [options]
    mars solvation input.xyz --solvent water --layers 6 12 [options]

Two modes
---------
**Automatic** (``--padding`` or ``--n-molecules``): shells are grown one by one
until the convergence criterion is met.  Molecules-per-layer are estimated from
the surface area.

**Manual** (``--layers N [N …]``): explicit molecule counts per layer.

Opt modes (apply to both auto and manual)
-----------------------------------------
none        — pure geometry placement, no energy minimisation
after-all   — optimise the full solvent shell once after all layers are placed
              (freeze solute, relax all solvents together)
layerwise   — after placing each layer, relax it while freezing all inner atoms
              (most accurate but more expensive)

Multi-conformer modes
---------------------
``--solute-swap``   solvate a reference frame, RMSD-align each conformer,
                       transplant the solvent shell, then batch-optimise.
``--all-conformers``   solvate each conformer independently (no transplant).
``--n-solvations N``   generate N independent placements with different seeds.

Post-optimisation
-----------------
``--relax-solute``     also move solute atoms during constrained opt.
``--final-relax [FMAX]`` run a final unconstrained relaxation (all atoms free);
                       optional FMAX tightens the final force threshold (else
                       the process ``--fmax`` is used).
"""

import sys

from ..log import (
    log_error,
    log_header,
    log_info,
    log_parameters,
    log_timing_summary,
    log_warning,
    timer_end,
    timer_reset,
    timer_start,
)
from ..utils import load_structure, save_ensemble, save_structure
from ._common import build_potential, setup_logging, setup_precision, validate_input_file


def _uniformize_batch(batch, n_solute, solvent_data, args):
    """Repair a ``(c_idx, r_idx, struct)`` batch to a uniform solvent count.

    No-op unless the batch actually contains mixed atom counts. Shared by every
    solvation build path (independent / swap-transplant / barostat) and by the
    final topology cleanup, so the vmap batch optimiser always sees identical
    shapes and the saved ensemble has a consistent number of solvent molecules.
    """
    if not batch or len(batch) < 2:
        return batch
    if len({len(s["symbols"]) for _, _, s in batch}) <= 1:
        return batch

    from ..log import timer_end, timer_start
    from ..solvation import uniformize_solvation_batch

    timer_start("Batch uniformisation")
    uniform = uniformize_solvation_batch(
        [s for _, _, s in batch],
        n_solute,
        solvent_data,
        getattr(args, "n_molecules", None),
        potential_name=args.potential,
        charge=args.charge,
        lr_cutoff=getattr(args, "lr_cutoff", 1000.0),
        fmax=args.fmax,
        maxiter=args.maxiter,
        method=args.method,
        max_stepsize=args.max_stepsize,
        seed=getattr(args, "seed", 42),
        topo_tolerance=getattr(args, "topo_tolerance", 1.3),
    )
    out = [(c, r, u) for (c, r, _), u in zip(batch, uniform)]
    timer_end("Batch uniformisation")
    return out


def run_solvation_workflow(args):
    """Main solvation workflow called from the CLI."""
    import numpy as np

    from ..auto_config import EV_TO_KCALMOL
    from ..solvation import (
        LayerSpec,
        align_to_reference_np,
        auto_solvate,
        optimize_solvation_batch,
        place_solvents_by_layers,
        solvate_barostat,
        solvate_barostat_batch,
        solvate_multi_seed,
        transplant_solvent_shell,
    )

    setup_logging(args)
    timer_reset()
    timer_start("Initialization")

    # -----------------------------------------------------------------------
    # Input — multi-frame aware loading
    # -----------------------------------------------------------------------
    input_path = validate_input_file(args.input)
    setup_precision(args)

    log_info(f"Loading structure(s) from {args.input}")
    try:
        from ase import io as ase_io

        from ..utils import ase_atoms_to_structure

        atoms_list = ase_io.read(str(input_path), index=":")
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
        all_solutes = [ase_atoms_to_structure(a) for a in atoms_list]
    except ImportError:
        all_solutes = [load_structure(str(input_path))]
    except Exception as exc:
        log_error(f"Failed to load structure: {exc}")
        sys.exit(1)

    n_frames = len(all_solutes)
    log_info(f"  Loaded {n_frames} frame(s)")

    # -----------------------------------------------------------------------
    # Flags
    # -----------------------------------------------------------------------
    n_replicas = getattr(args, "n_solvations", 1)
    solute_swap = getattr(args, "solute_swap", False)
    all_conformers = getattr(args, "all_conformers", False)
    barostat = getattr(args, "barostat", False)
    if barostat and not args.n_molecules:
        log_error("--barostat requires a target count: pass --n-molecules N.")
        sys.exit(1)
    if barostat and n_replicas > 1:
        log_info("  Note: --barostat builds one droplet; --n-solvations is ignored.")
        n_replicas = 1
    do_align = not getattr(args, "no_align", False)
    relax_solute = getattr(args, "relax_solute", False)
    # --final-relax: None = off; -1.0 (flag alone) = use the process --fmax;
    # a positive float = use that tighter fmax for the final relax only.
    _final_relax_arg = getattr(args, "final_relax", None)
    final_relax = _final_relax_arg is not None
    final_relax_fmax = (
        args.fmax if (_final_relax_arg is None or _final_relax_arg <= 0) else _final_relax_arg
    )

    ref_idx = args.conformer if args.conformer is not None else 0

    if ref_idx >= n_frames:
        log_error(
            f"Conformer index {ref_idx} out of range "
            f"(file has {n_frames} frame(s), max index {n_frames - 1})"
        )
        sys.exit(1)

    if solute_swap and all_conformers:
        log_error("--solute-swap and --all-conformers are mutually exclusive.")
        sys.exit(1)

    # Decide which conformers to process and how
    if solute_swap:
        # Swap mode: solvate reference, transplant shell onto all others
        conformer_indices = list(range(n_frames))
        mode_multi = "swap"
    elif all_conformers:
        # Independent solvation for each conformer
        conformer_indices = list(range(n_frames))
        mode_multi = "independent"
    else:
        # Single conformer
        conformer_indices = [ref_idx]
        mode_multi = "single"

    reference_solute = all_solutes[ref_idx]
    n_solute = len(reference_solute["symbols"])

    # Validate all conformers have same atoms (required for swap, good
    # practice for independent too)
    for idx in conformer_indices:
        s = all_solutes[idx]
        if len(s["symbols"]) != n_solute:
            log_error(
                f"Conformer {idx} has {len(s['symbols'])} atoms but "
                f"reference (frame {ref_idx}) has {n_solute}. "
                f"All must match."
            )
            sys.exit(1)
        if list(s["symbols"]) != list(reference_solute["symbols"]):
            log_error(
                f"Conformer {idx} has different element order than " f"reference (frame {ref_idx})."
            )
            sys.exit(1)

    log_info(f"  Reference conformer: frame {ref_idx}  ({n_solute} atoms)")
    if mode_multi == "swap":
        log_info(f"  Solute-swap: transplanting shell onto {n_frames} conformers")
        if do_align:
            log_info("  RMSD alignment: ON (disable with --no-align)")
            log_warning(
                "  ⚠️  Kabsch alignment may introduce errors for large "
                "conformational changes or symmetries. "
                "Use --no-align for reaction trajectories if uncertain."
            )
        else:
            log_info("  RMSD alignment: OFF")
    elif mode_multi == "independent":
        log_info(f"  Independent solvation for {n_frames} conformers")
    if n_replicas > 1:
        log_info(
            f"  Solvation replicas: {n_replicas}  "
            f"(seeds {args.seed}..{args.seed + n_replicas - 1})"
        )
    if relax_solute:
        log_info("  Relax solute: ON (solute atoms will also be optimised)")
    if final_relax:
        log_info(f"  Final unconstrained relax: ON (fmax = {final_relax_fmax} eV/Å)")

    # Validate solvent — accepts a library name/alias OR a path to a user .xyz
    from ..solvents import load_solvent

    try:
        solvent_data = load_solvent(args.solvent)
    except (KeyError, FileNotFoundError) as exc:
        log_error(str(exc))
        sys.exit(1)
    src = solvent_data.get("source", "")
    src_note = "  [user file]" if src == "user-file" else ""
    log_info(
        f"  Solvent: {args.solvent}  " f"({len(solvent_data['symbols'])} atoms/molecule){src_note}"
    )

    # -----------------------------------------------------------------------
    # Decide auto vs manual mode
    # -----------------------------------------------------------------------
    use_auto = args.layers is None

    if use_auto:
        mode_desc = "automatic"
        if args.padding is not None:
            log_info(f"  Mode: auto (padding = {args.padding} Å)")
        elif args.n_molecules is not None:
            log_info(f"  Mode: auto (n_molecules = {args.n_molecules})")
        else:
            log_info("  Mode: auto (default padding = 5.0 Å)")
    else:
        mode_desc = "manual"
        log_info(
            f"  Mode: manual, layers = {args.layers}  "
            f"(total planned solvents: {sum(args.layers)})"
        )

    # Determine effective freeze_mode for constrained opt
    if relax_solute:
        effective_freeze = "none"
    else:
        effective_freeze = args.freeze_mode

    n_total_batch = len(conformer_indices) * n_replicas
    log_parameters(
        {
            "Mode": "barostat" if barostat else mode_desc,
            "Multi-conformer": mode_multi,
            "Solvent": args.solvent,
            "Barostat": (
                f"wall={args.baro_wall_mode}, k={args.baro_k}, "
                f"max_force={args.baro_max_force} eV/Å, step={args.baro_step} Å"
                if barostat
                else False
            ),
            "Padding (Å)": args.padding if use_auto else "—",
            "Target molecules": args.n_molecules if use_auto else "—",
            "Layers": args.layers if not use_auto else "auto",
            "Max auto layers": args.max_layers if use_auto else "—",
            "Potential": args.potential,
            "Opt mode": args.opt_mode,
            "Freeze mode": effective_freeze,
            "Relax solute": relax_solute,
            "Final relax": f"fmax={final_relax_fmax} eV/Å" if final_relax else False,
            "RMSD align": do_align and mode_multi == "swap",
            "Charge": args.charge,
            "fmax": f"{args.fmax} eV/Å",
            "maxiter": args.maxiter,
            "Method": args.method,
            "vdW scale": args.vdw_scale,
            "Buffer (Å)": args.buffer,
            "Candidates/atom": args.n_candidates,
            "Radial shells": args.n_shells,
            "Seed": args.seed,
            "Conformers": len(conformer_indices),
            "Replicas": n_replicas,
            "Total batch": n_total_batch,
        },
        title="Solvation parameters",
    )

    timer_end("Initialization")

    # -----------------------------------------------------------------------
    # Common solvation kwargs
    # -----------------------------------------------------------------------
    def _baro_kwargs(seed):
        """Barostat parameters shared by the single- and batch-solute paths."""
        return dict(
            n_molecules=args.n_molecules,
            inner_buffer=args.baro_inner_buffer,
            outer_radius=args.baro_radius,
            fill_fraction=args.baro_fill,
            wall_k=args.baro_k,
            wall_mode=args.baro_wall_mode,
            compress_step=args.baro_step,
            fine_step=args.baro_fine_step,
            max_wall_force=args.baro_max_force,
            max_compress_steps=args.baro_max_steps,
            compress_md=args.baro_md,
            md_temp=args.baro_md_temp,
            md_time_ps=args.baro_md_time,
            md_dt=args.baro_md_dt,
            vdw_scale=args.vdw_scale,
            buffer=args.buffer,
            n_orient=args.n_orient,
            seed=seed,
            potential_name=args.potential,
            charge=args.charge,
            lr_cutoff=args.lr_cutoff,
            fmax=args.fmax,
            method=args.method,
            max_stepsize=args.max_stepsize,
            topology_repair=getattr(args, "topology_repair", True),
            topo_tolerance=getattr(args, "topo_tolerance", 1.3),
            parallel=not getattr(args, "not_parallel", False),
        )

    def _barostat_one(solute_struct, seed):
        """Build a solvated structure via the spherical moving-wall barostat."""
        return solvate_barostat(solute_struct, solvent=solvent_data, **_baro_kwargs(seed))

    def _solvate_one(solute_struct, seed):
        """Solvate a single solute structure (barostat, auto, or manual)."""
        if barostat:
            return _barostat_one(solute_struct, seed)
        if use_auto:
            return auto_solvate(
                solute_struct,
                solvent=solvent_data,
                padding=args.padding,
                n_molecules=args.n_molecules,
                max_layers=args.max_layers,
                optimise_each_layer=(args.opt_mode == "layerwise"),
                potential_name=args.potential,
                charge=args.charge,
                fmax=args.fmax,
                maxiter=args.maxiter,
                method=args.method,
                max_stepsize=args.max_stepsize,
                lr_cutoff=args.lr_cutoff,
                vdw_scale=args.vdw_scale,
                buffer=args.buffer,
                n_candidates=args.n_candidates,
                n_shells_per_atom=args.n_shells,
                n_orient=args.n_orient,
                refill_cycles=args.refill_cycles,
                seed=seed,
                topology_repair=getattr(args, "topology_repair", True),
                max_repair=getattr(args, "max_repair", 3),
                topo_tolerance=getattr(args, "topo_tolerance", 1.3),
            )
        else:
            specs = [
                LayerSpec(
                    n_solvent=n,
                    buffer=args.buffer,
                    vdw_scale=args.vdw_scale,
                    n_candidates=args.n_candidates,
                    n_shells=args.n_shells,
                    seed=seed + i,
                )
                for i, n in enumerate(args.layers)
            ]
            if args.opt_mode == "layerwise":
                from ..solvation import _place_one_layer

                system = {k: v for k, v in solute_struct.items()}
                for i, spec in enumerate(specs):
                    n_inner = len(system["symbols"])
                    system = _place_one_layer(system, solvent_data, spec)
                    system = _run_layer_opt(args, system, n_inner, EV_TO_KCALMOL)
                return system
            else:
                return place_solvents_by_layers(solute_struct, solvent_data, specs)

    # -----------------------------------------------------------------------
    # BRANCH A: solute-swap mode
    # -----------------------------------------------------------------------
    if mode_multi == "swap" and barostat:
        # Barostat swap: align every conformer to the reference frame, build ONE
        # compressed droplet, and pack the solute onto all conformers together as
        # a single vmapped batch (per-structure moving wall). No transplant /
        # after-all opt — the barostat already does placement + packing + relax.
        timer_start("Barostat batch solvation")
        from ..utils import center_of_mass

        _ref_com = center_of_mass(reference_solute["positions"], reference_solute["numbers"])
        ref_pos = np.asarray(reference_solute["positions"]) - _ref_com
        aligned, order = [], []
        for c_idx in conformer_indices:
            cs = all_solutes[c_idx]
            if c_idx == ref_idx:
                ap = ref_pos
            elif do_align:
                ap = align_to_reference_np(np.asarray(cs["positions"]), ref_pos)
            else:
                ap = np.asarray(cs["positions"]) - center_of_mass(cs["positions"], cs["numbers"])
            aligned.append({**cs, "positions": ap})
            order.append(c_idx)
        try:
            solvated_list = solvate_barostat_batch(
                aligned,
                solvent=solvent_data,
                reference_idx=order.index(ref_idx),
                **_baro_kwargs(args.seed),
            )
        except Exception as exc:
            log_error(f"Barostat batch solvation failed: {exc}")
            import traceback

            traceback.print_exc()
            sys.exit(1)
        batch = [(order[i], 0, solvated_list[i]) for i in range(len(order))]
        n_total_atoms = len(batch[0][2]["symbols"])
        n_solvent_atoms = n_total_atoms - n_solute
        timer_end("Barostat batch solvation")

    elif mode_multi == "swap":
        # Step 1: solvate the reference conformer (with replicas)
        timer_start("Reference placement")
        try:
            replica_solvated = solvate_multi_seed(
                reference_solute,
                solvent=solvent_data,
                n_replicas=n_replicas,
                base_seed=args.seed,
                padding=args.padding,
                n_molecules=args.n_molecules,
                max_layers=args.max_layers,
                layers=args.layers if not use_auto else None,
                vdw_scale=args.vdw_scale,
                buffer=args.buffer,
                n_candidates=args.n_candidates,
                n_shells_per_atom=args.n_shells,
                n_orient=args.n_orient,
                refill_cycles=args.refill_cycles,
                optimise_each_layer=(args.opt_mode == "layerwise"),
                potential_name=args.potential,
                charge=args.charge,
                fmax=args.fmax,
                maxiter=args.maxiter,
                method=args.method,
                max_stepsize=args.max_stepsize,
                lr_cutoff=args.lr_cutoff,
                topology_repair=getattr(args, "topology_repair", True),
                max_repair=getattr(args, "max_repair", 3),
                topo_tolerance=getattr(args, "topo_tolerance", 1.3),
                potential_kwargs=getattr(args, "potential_kwargs", None),
            )
        except Exception as exc:
            log_error(f"Reference solvation failed: {exc}")
            import traceback

            traceback.print_exc()
            sys.exit(1)
        timer_end("Reference placement")

        # Step 2: RMSD-align conformers to reference & transplant.
        # The solvation routines recenter the solute to its center of mass, so
        # the solvent shell in `replica_solvated` lives in that COM-centred
        # frame. Align conformers to the SAME recentred reference frame, else
        # the transplanted shell would be offset by the COM translation.
        from ..utils import center_of_mass

        timer_start("Conformer expansion")
        _ref_com = center_of_mass(reference_solute["positions"], reference_solute["numbers"])
        ref_pos = np.asarray(reference_solute["positions"]) - _ref_com

        batch = []  # (conformer_idx, replica_idx, structure_dict)
        for r_idx, ref_solvated in enumerate(replica_solvated):
            for c_idx in conformer_indices:
                if c_idx == ref_idx:
                    batch.append((c_idx, r_idx, ref_solvated))
                else:
                    conf_solute = all_solutes[c_idx]
                    if do_align:
                        # Kabsch-align this conformer onto the reference
                        aligned_pos = align_to_reference_np(
                            np.asarray(conf_solute["positions"]),
                            ref_pos,
                        )
                        conf_aligned = dict(conf_solute)
                        conf_aligned["positions"] = aligned_pos
                    else:
                        conf_aligned = conf_solute

                    transplanted = transplant_solvent_shell(
                        ref_solvated,
                        conf_aligned,
                        n_solute,
                    )
                    batch.append((c_idx, r_idx, transplanted))

        n_total_atoms = len(batch[0][2]["symbols"])
        n_solvent_atoms = n_total_atoms - n_solute

        if len(batch) > 1:
            log_info(
                f"  Expanded to {len(batch)} structures  "
                f"({len(conformer_indices)} conformers "
                f"× {n_replicas} replicas)"
            )
        if len(batch) > 50:
            log_warning(
                f"Large batch ({len(batch)} structures). "
                f"Memory usage may be high. Consider --not-parallel."
            )
        timer_end("Conformer expansion")

    # -----------------------------------------------------------------------
    # BRANCH B: independent solvation (--all-conformers or single)
    # -----------------------------------------------------------------------
    else:
        timer_start("Placement")
        batch = []
        for c_idx in conformer_indices:
            solute_i = all_solutes[c_idx]
            for r_idx in range(n_replicas):
                seed_i = args.seed + r_idx
                if len(conformer_indices) > 1 or n_replicas > 1:
                    log_header(f"Conformer {c_idx}, replica {r_idx}")
                try:
                    solvated = _solvate_one(solute_i, seed_i)
                except Exception as exc:
                    log_error(f"Solvation failed for conformer {c_idx}, " f"replica {r_idx}: {exc}")
                    import traceback

                    traceback.print_exc()
                    sys.exit(1)
                batch.append((c_idx, r_idx, solvated))

        n_total_atoms = len(batch[0][2]["symbols"])
        n_solvent_atoms = n_total_atoms - n_solute
        timer_end("Placement")

    # -----------------------------------------------------------------------
    # Enforce a uniform solvent count across the whole batch (covers every
    # build path: independent / swap-transplant / barostat). The geometric and
    # barostat builders can be geometry-limited at different counts per
    # conformer, which would crash the vmap batch optimiser (jnp.stack needs
    # identical shapes). Repair deficient structures up to the hard target
    # (--n-molecules) with geometric refill + relax, and trim over-filled ones.
    # -----------------------------------------------------------------------
    batch = _uniformize_batch(batch, n_solute, solvent_data, args)
    if batch:
        n_total_atoms = len(batch[0][2]["symbols"])
        n_solvent_atoms = n_total_atoms - n_solute

    # -----------------------------------------------------------------------
    # Constrained optimisation (after-all or transplanted conformers)
    # -----------------------------------------------------------------------
    needs_opt = False

    if barostat:
        # The barostat already did placement + compression + packing + a
        # wall-free relaxation, so no extra constrained optimisation here.
        needs_opt = False
    elif args.opt_mode == "after-all":
        needs_opt = True
    elif mode_multi == "swap" and args.opt_mode == "layerwise":
        # Reference was layerwise-optimised during placement, but
        # transplanted conformers still need optimisation.
        needs_opt = True
        log_info(
            "  Note: transplanted conformers receive after-all "
            "optimisation (layerwise applied to reference only)"
        )

    if needs_opt and len(batch) > 0:
        timer_start("Constrained optimisation")
        parallel = not getattr(args, "not_parallel", False)

        # Validate charge before optimization
        if args.charge != 0.0:
            log_info(f"  Charge validation: total system charge = {args.charge}")
        else:
            log_info(f"  Charge validation: neutral system (charge = 0)")

        structures_to_opt = [item[2] for item in batch]
        try:
            multistage = not getattr(args, "no_multistage", False)
            optimized = optimize_solvation_batch(
                structures_to_opt,
                n_solute,
                potential_name=args.potential,
                charge=args.charge,
                lr_cutoff=args.lr_cutoff,
                freeze_mode=effective_freeze,
                fmax=args.fmax,
                maxiter=args.maxiter,
                method=args.method,
                max_stepsize=args.max_stepsize,
                parallel=parallel,
                fire_dt_start=getattr(args, "fire_dt_start", 0.05),
                fire_dt_max=getattr(args, "fire_dt_max", 0.1),
                fire_n_min=getattr(args, "fire_n_min", 2),
                multistage=multistage,
            )
        except Exception as exc:
            log_error(f"Constrained optimisation failed: {exc}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

        batch = [(c, r, opt) for (c, r, _), opt in zip(batch, optimized)]
        timer_end("Constrained optimisation")

    # -----------------------------------------------------------------------
    # Final unconstrained relaxation (all atoms free)
    # -----------------------------------------------------------------------
    if final_relax and len(batch) > 0:
        timer_start("Final unconstrained relax")
        parallel = not getattr(args, "not_parallel", False)
        log_header("Final unconstrained relaxation (all atoms free)")

        # Validate charge before optimization
        if args.charge != 0.0:
            log_info(f"  Charge validation: total system charge = {args.charge}")
        else:
            log_info(f"  Charge validation: neutral system (charge = 0)")

        structures_to_relax = [item[2] for item in batch]
        try:
            multistage = not getattr(args, "no_multistage", False)
            relaxed = optimize_solvation_batch(
                structures_to_relax,
                n_solute,
                potential_name=args.potential,
                charge=args.charge,
                lr_cutoff=args.lr_cutoff,
                freeze_mode="none",  # all atoms free
                fmax=final_relax_fmax,
                maxiter=args.maxiter,
                method=args.method,
                max_stepsize=args.max_stepsize,
                parallel=parallel,
                fire_dt_start=getattr(args, "fire_dt_start", 0.05),
                fire_dt_max=getattr(args, "fire_dt_max", 0.1),
                fire_n_min=getattr(args, "fire_n_min", 2),
                multistage=multistage,
            )
        except Exception as exc:
            log_error(f"Final relaxation failed: {exc}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

        batch = [(c, r, opt) for (c, r, _), opt in zip(batch, relaxed)]
        timer_end("Final unconstrained relax")

    # -----------------------------------------------------------------------
    # Final topology cleanup (every mode): drop any solvent molecule that
    # fragmented or reacted during the constrained / unconstrained relaxation,
    # then refill back to the target so the saved ensemble stays uniform. The
    # layerwise/auto build already repairs during the build, but the after-all
    # and --final-relax steps run afterwards, so this guarantees clean, uniform
    # output.
    # -----------------------------------------------------------------------
    if getattr(args, "topology_repair", True) and len(batch) > 0:
        from ..solvation import remove_broken_solvent

        tol = getattr(args, "topo_tolerance", 1.3)
        total_removed = 0
        cleaned_batch = []
        for c_idx, r_idx, struct in batch:
            clean, broken = remove_broken_solvent(struct, solvent_data, n_solute, tolerance=tol)
            total_removed += len(broken)
            cleaned_batch.append((c_idx, r_idx, clean))
        batch = cleaned_batch
        if total_removed:
            log_warning(
                f"  Topology cleanup: removed {total_removed} broken solvent "
                f"molecule(s); refilling to restore a uniform count."
            )
            # Refill the structures whose count dipped back up to the target
            # (geometric placement + relax), keeping the ensemble uniform.
            batch = _uniformize_batch(batch, n_solute, solvent_data, args)
        else:
            log_info("  Topology cleanup: all solvent molecules intact.")

    # -----------------------------------------------------------------------
    # Save output
    # -----------------------------------------------------------------------
    timer_start("Output")

    def _make_comment(c_idx, r_idx, struct):
        comment = f"Solvated with {args.solvent}"
        if not use_auto:
            comment += f", layers={args.layers}"
        else:
            if args.padding is not None:
                comment += f", padding={args.padding}"
            if args.n_molecules is not None:
                comment += f", n_molecules={args.n_molecules}"
        comment += f", opt={args.opt_mode}"
        if mode_multi == "swap":
            comment += ", swap"
        if n_total_batch > 1:
            comment += f", conformer={c_idx}, replica={r_idx}"
        energy = struct.get("energy")
        if energy is not None:
            comment += f", energy={float(energy):.8f}"
        return comment

    try:
        if len(batch) == 1:
            c_idx, r_idx, final = batch[0]
            out_path = args.output or "solvated.xyz"
            save_structure(out_path, final, comment=_make_comment(c_idx, r_idx, final))
            log_info(f"  Saved: {out_path}")
        else:
            prefix = getattr(args, "output_prefix", "solvated")
            ensemble = [(item[2], float(item[2].get("energy", 0.0))) for item in batch]
            all_fname = args.output or f"{prefix}_all.xyz"
            save_ensemble(all_fname, ensemble)

            log_info(f"  Saved multi-frame ensemble: {all_fname}  " f"({len(batch)} frames)")
    except Exception as exc:
        log_error(f"Failed to save output: {exc}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    log_info("")
    log_info("=" * 70)
    log_info("Solvation complete")
    log_info("=" * 70)
    log_info(f"  Solute atoms   : {n_solute}")
    log_info(f"  Solvent atoms  : {n_solvent_atoms}")
    log_info(f"  Total atoms    : {n_total_atoms}")
    log_info(f"  Multi-conf mode: {mode_multi}")
    log_info(f"  Conformers     : {len(conformer_indices)}")
    log_info(f"  Replicas       : {n_replicas}")
    log_info(f"  Total structures: {len(batch)}")
    if not use_auto:
        log_info(f"  Layers         : {args.layers}")
    else:
        log_info(f"  Mode           : auto")
    timer_end("Output")
    log_timing_summary()

    return 0


# ============================================================================
# Internal helpers for optimisation steps (kept for layerwise manual mode)
# ============================================================================


def _run_after_all_opt(args, solute, solvated, n_solute, EV_TO_KCALMOL):
    """Optimise full solvent shell (freeze solute or solvent)."""
    import jax.numpy as jnp

    from ..cli._constraints import make_frozen_energy_fn
    from ..optimizer import optimize_single

    try:
        potential_wrapper, energy_fn = build_potential(args, solvated)
    except Exception as exc:
        log_error(f"Failed to build potential: {exc}")
        sys.exit(1)

    positions = jnp.array(solvated["positions"])
    n_total = len(solvated["symbols"])

    if args.freeze_mode == "solute":
        frozen = list(range(n_solute))
    elif args.freeze_mode == "solvent":
        frozen = list(range(n_solute, n_total))
    else:
        frozen = []

    if frozen:
        energy_fn = make_frozen_energy_fn(energy_fn, positions, frozen)
        log_info(f"  Frozen atoms: {len(frozen)} ({args.freeze_mode})")

    try:
        pos_opt, energy_opt, info = optimize_single(
            positions,
            energy_fn,
            fmax=args.fmax,
            maxiter=args.maxiter,
            method=args.method,
            max_stepsize=args.max_stepsize,
            potential_wrapper=potential_wrapper,
        )
    except Exception as exc:
        log_error(f"Optimisation failed: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    if info["converged"]:
        log_info(
            f"  Converged in {info['iterations']} steps  "
            f"E = {float(energy_opt) * EV_TO_KCALMOL:.3f} kcal/mol"
        )
    else:
        log_warning(
            f"  Not converged after {info['iterations']} steps  "
            f"Fmax = {info['max_force']:.4f} eV/Å"
        )

    result = dict(solvated)
    result["positions"] = pos_opt
    result["energy"] = energy_opt
    return result


def _run_layer_opt(args, system, n_inner, EV_TO_KCALMOL):
    """Optimise a single freshly-placed layer (freeze inner atoms)."""
    import jax.numpy as jnp

    from ..cli._constraints import make_frozen_energy_fn
    from ..optimizer import optimize_single

    try:
        potential_wrapper, energy_fn = build_potential(args, system)
    except Exception as exc:
        log_error(f"Layer optimisation failed to build potential: {exc}")
        sys.exit(1)

    positions = jnp.array(system["positions"])
    frozen = list(range(n_inner))
    energy_fn_frozen = make_frozen_energy_fn(energy_fn, positions, frozen)

    try:
        pos_opt, energy_opt, info = optimize_single(
            positions,
            energy_fn_frozen,
            fmax=args.fmax,
            maxiter=args.maxiter,
            method=args.method,
            max_stepsize=args.max_stepsize,
            potential_wrapper=potential_wrapper,
        )
    except Exception as exc:
        log_error(f"Layer optimisation failed: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    if info["converged"]:
        log_info(
            f"  Converged in {info['iterations']} steps  "
            f"E = {float(energy_opt) * EV_TO_KCALMOL:.3f} kcal/mol"
        )
    else:
        log_warning(
            f"  Not converged after {info['iterations']} steps  "
            f"Fmax = {info['max_force']:.4f} eV/Å"
        )

    result = dict(system)
    result["positions"] = pos_opt
    result["energy"] = energy_opt
    return result
