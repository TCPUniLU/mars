"""
Shared utilities for MARS CLI commands.
"""

import sys
from pathlib import Path

from ..log import init_log, log_error, log_info
from ..utils import check_charge, enable_float64


def setup_debug_mode(args):
    """Enable debug mode if --debug flag is set."""
    if args.debug:
        from .. import enable_debug

        enable_debug()
        args.log_level = "DEBUG"


def setup_logging(args):
    """Initialize logging from CLI arguments."""
    init_log(logfile=args.log_file, level=args.log_level, console=True)


def setup_precision(args):
    """Configure JAX floating point precision and device.

    ``--coords internal`` requires ``--float64`` and is refused without it.
    """
    needs_x64 = (
        getattr(args, "coords", "cartesian") == "internal"
        or getattr(args, "coords_coarse", "cartesian") == "internal"
    )
    if needs_x64 and not args.float64:
        log_error(
            "--coords internal requires 64-bit precision: re-run with --float64. "
            "In float32 the back-transformation residual (~1e-4 rad) produces a "
            "spurious force of ~4e-3 eV/A, above the tightest convergence "
            "threshold, and the optimizer oscillates without reporting an error."
        )
        sys.exit(1)

    if args.float64:
        enable_float64()
        log_info("Using 64-bit floating point precision")
    else:
        log_info("Using 32-bit floating point precision")

    if hasattr(args, "cpu") and args.cpu:
        log_info("Forcing CPU execution")
        import jax

        jax.config.update("jax_platform_name", "cpu")


def validate_input_file(input_path_str):
    """Check that the input file exists, exit if not."""
    input_path = Path(input_path_str)
    if not input_path.exists():
        log_error(f"Input file not found: {input_path_str}")
        sys.exit(1)
    return input_path


# Track which (potential, model) combinations have already been announced so
# the banner + citation print once even when build_potential is called
# repeatedly in a single run (e.g. per-layer solvation).
_ANNOUNCED_POTENTIALS = set()


def _potential_model_label(args):
    """Human-readable model label for the selected potential (``None`` if n/a)."""
    p = args.potential
    if p == "so3lr":
        return getattr(args, "so3lr_model", None) or "so3lr-1"
    if p == "mace":
        foundation = getattr(args, "mace_foundation", None) or "off"
        model = getattr(args, "mace_model", None) or "small"
        if foundation == "off24" and model in (None, "small", "medium"):
            # Mirror MACEPotential._resolve_off24: OFF24 publishes one size, and
            # "small" is only ever the CLI-wide default leaking through.
            model = "medium"
        return f"{foundation}/{model}"
    if p == "dxtb":
        return getattr(args, "dxtb_method", None) or "gfn1"
    if p == "uma":
        model = getattr(args, "uma_model", None) or "uma-s-1p2"
        return f"{model}/{getattr(args, 'uma_task', None) or 'omol'}"
    if p in ("orb", "ani", "aimnet2", "aceff"):
        return getattr(args, "ml_model", None) or "default"
    if p == "amoeba":
        return f"polarization={getattr(args, 'amoeba_polarization', None) or 'mutual'}"
    return None


def _potential_citations(args):
    """Citation line(s) for the selected potential (matches the Citation docs)."""
    p = args.potential
    if p == "so3lr":
        return [
            "A. Kabylda, J. T. Frank, S. Suárez-Dou, et al., "
            "J. Am. Chem. Soc. 147 (2025) 33723-33734. doi:10.1021/jacs.5c09558"
        ]
    if p == "mace":
        return [
            "D. P. Kovács, J. H. Moore, N. J. Browning, et al., "
            "J. Am. Chem. Soc. 147 (2025) 17598-17611. doi:10.1021/jacs.4c07099",
            "I. Batatia, D. P. Kovács, G. N. C. Simm, C. Ortner, G. Csányi, "
            "MACE: Higher Order Equivariant Message Passing Neural Networks, "
            "Adv. Neural Inf. Process. Syst. 35 (2022) 11423-11436.",
        ]
    if p == "dxtb":
        return [
            "M. Friede, C. Hölzer, S. Ehlert, S. Grimme, "
            "J. Chem. Phys. 161 (2024) 062501. doi:10.1063/5.0216715"
        ]
    if p == "uma":
        return [
            "B. M. Wood, M. Dzamba, X. Fu, et al., UMA: A Family of Universal Models "
            "for Atoms, Adv. Neural Inf. Process. Syst. 38 (2025)."
        ]
    if p == "orb":
        return [
            "B. Rhodes, S. Vandenhaute, V. Šimkus, et al., Orb-v3: atomistic simulation "
            "at scale, arXiv:2504.06231 (2025)."
        ]
    if p == "ani":
        return [
            "C. Devereux, J. S. Smith, K. K. Huddleston, et al., "
            "J. Chem. Theory Comput. 16 (2020) 4192-4202. doi:10.1021/acs.jctc.0c00121",
            "X. Gao, F. Ramezanghorbani, O. Isayev, J. S. Smith, A. E. Roitberg, "
            "J. Chem. Inf. Model. 60 (2020) 3408-3415. doi:10.1021/acs.jcim.0c00451",
        ]
    if p == "aimnet2":
        return [
            "D. M. Anstine, R. Zubatyuk, O. Isayev, "
            "Chem. Sci. 16 (2025) 10228-10244. doi:10.1039/D4SC08572H"
        ]
    if p == "aceff":
        return [
            "S. E. Farr, S. Doerr, A. Mirarchi, et al., AceFF: A State-of-the-Art Machine "
            "Learning Potential for Small Molecules, arXiv:2601.00581 (2026).",
            "R. P. Pelaez, G. Simeon, R. Galvelis, et al., "
            "J. Chem. Theory Comput. 20 (2024) 4076-4087. doi:10.1021/acs.jctc.4c00253",
        ]
    if p in ("amber", "charmm", "opls", "amoeba", "mm"):
        return [
            "P. Eastman, R. Galvelis, R. P. Peláez, et al. (OpenMM 8), "
            "J. Phys. Chem. B 128 (2024) 109-116. doi:10.1021/acs.jpcb.3c06662",
            "Please also cite the specific force field whose parameter files you loaded "
            "(e.g. ff19SB, GAFF2, CHARMM36m, OPLS-AA, AMOEBA).",
        ]
    return []


#: Who implemented the JAX code behind each potential (upstream PRs / authors).
_IMPLEMENTATION_CREDITS = {
    "uma": "jax-md UMA port (jax-md PRs #376, #394, #400, #401: @ekurtulus, "
    "@abhijeetgangan; #399: @teddykoker) of FAIR-Chem UMA",
    "orb": "jax-md port of orb-models (jax-md PR #411: @mitkotak, @abhijeetgangan; bio-mlff)",
    "ani": "jax-md port of TorchANI (jax-md PR #411: @mitkotak, @abhijeetgangan; bio-mlff)",
    "aimnet2": "jax-md port of aimnetcentral (jax-md PR #411: @mitkotak, @abhijeetgangan; "
    "bio-mlff)",
    "aceff": "jax-md port of torchmd-net AceFF (jax-md PR #411: @mitkotak, @abhijeetgangan; "
    "bio-mlff)",
    "so3lr_jaxmd": "jax-md SO3LR port (jax-md PR #411: @mitkotak, @abhijeetgangan; bio-mlff)",
    "mace_jaxmd": "jax-md MACE interface (jax-md PR #390: @DRosen285, @abhijeetgangan, "
    "@beitom) on ACEsuit mace-jax",
    "mm": "jax-md mm_forcefields (jax-md PRs #319: @alecbetancourt; #350: @DRosen285, @beitom; "
    "@abhijeetgangan) + MARS Lepton compiler / AMOEBA; parameters read by OpenMM",
}


def _potential_credits(args):
    """Implementation credit line for the selected potential, or ``None``."""
    p = args.potential
    if p == "so3lr" and getattr(args, "so3lr_backend", "so3lr") == "jaxmd":
        return _IMPLEMENTATION_CREDITS["so3lr_jaxmd"]
    if p == "mace" and getattr(args, "mace_backend", "auto") == "jaxmd":
        return _IMPLEMENTATION_CREDITS["mace_jaxmd"]
    if p in ("amber", "charmm", "opls", "amoeba"):
        return _IMPLEMENTATION_CREDITS["mm"]
    return _IMPLEMENTATION_CREDITS.get(p)


def log_potential_and_citation(args):
    """Announce the loaded potential + model and its citation (once per combo)."""
    from ..log import log_message

    potential = args.potential
    model_label = _potential_model_label(args)
    key = (potential, model_label)
    if key in _ANNOUNCED_POTENTIALS:
        return
    _ANNOUNCED_POTENTIALS.add(key)

    if model_label is not None:
        log_info(f"Using potential: {potential}  |  model: {model_label}")
    else:
        log_info(f"Using potential: {potential}")

    credit = _potential_credits(args)
    if credit:
        log_message(f"  Implementation: {credit}")
    citations = _potential_citations(args)
    if citations:
        log_message(f"  If you use the {potential} potential, please cite:")
        for line in citations:
            log_message(f"    {line}")
        log_message("")


def extra_potential_spec(args):
    """Registry name and constructor kwargs (without ``species``) for the potentials
    added with the jax-md integration, or ``None`` for the others.

    Shared by :func:`build_potential` and the IR command, which builds its
    potential inside :mod:`mars.ir`.
    """
    import jax.numpy as jnp

    p = args.potential
    kw = {}
    if p == "so3lr" and getattr(args, "so3lr_backend", "so3lr") == "jaxmd":
        # jax-md's SO3LR port: same model, no so3lr package needed.
        kw["charge"] = args.charge
        if getattr(args, "spin", None) is not None:
            kw["spin"] = args.spin
        if args.float64:
            kw["dtype"] = jnp.float64
        return "so3lr_jaxmd", kw
    if p in ("uma", "orb", "ani", "aimnet2", "aceff"):
        kw["charge"] = getattr(args, "charge", 0.0)
        if getattr(args, "spin", None) is not None:
            kw["spin"] = args.spin
        if p == "uma":
            kw["model"] = getattr(args, "uma_model", None) or "uma-s-1p2"
            kw["task"] = getattr(args, "uma_task", None) or "omol"
        elif getattr(args, "ml_model", None):
            model = args.ml_model
            kw["model_path" if model.endswith(".eqx") else "model"] = model
        if args.float64:
            kw["dtype"] = jnp.float64
        return p, kw
    if p in ("amber", "charmm", "opls", "amoeba", "mm"):
        kw = _mm_file_kwargs(args)
        kw["charge"] = getattr(args, "charge", 0.0)
        kw["spin"] = getattr(args, "spin", 1.0) or 1.0
        return p, kw
    return None


def _mm_file_kwargs(args):
    """Force-field input files from the CLI, for the amber/charmm/opls/amoeba potentials."""
    inp = getattr(args, "input", None)
    inp_ext = Path(inp).suffix.lower() if inp else ""
    pdb = getattr(args, "ff_pdb", None) or (inp if inp_ext == ".pdb" else None)
    kw = {
        "prmtop": getattr(args, "prmtop", None),
        "coordinates": getattr(args, "ff_coords", None),
        "psf": getattr(args, "psf", None),
        "params": getattr(args, "ff_params", None),
        "top": getattr(args, "top", None),
        "include_dir": getattr(args, "gmx_include", None),
        "xml": getattr(args, "ff_xml", None),
        "pdb": pdb,
        "polarization": getattr(args, "amoeba_polarization", None),
        "tinker_key": getattr(args, "tinker_key", None),
    }
    if args.potential == "amoeba" and inp_ext == ".xyz" and kw["tinker_key"]:
        kw["tinker_xyz"] = inp
    if args.potential == "mm" and not (kw["xml"] or kw["prmtop"] or kw["psf"] or kw["top"]):
        raise SystemExit(
            "--potential mm needs force-field files: --ff-xml ... --ff-pdb, or use "
            "--potential amber/charmm/opls/amoeba."
        )
    if args.potential == "mm":
        kw["family"] = (
            "amber"
            if kw["prmtop"]
            else "charmm" if kw["psf"] else "opls" if kw["top"] else "openmm"
        )
    return {k: v for k, v in kw.items() if v is not None}


def build_potential(args, structure, compute_charges=False):
    """Build potential energy function from CLI arguments.

    Args:
        args: Parsed CLI arguments
        structure: Loaded structure dict with 'positions', 'symbols', 'numbers'
        compute_charges: If True, also return a charges function for IR.

    Returns:
        When *compute_charges* is False:
            ``(potential, energy_fn)``
        When *compute_charges* is True:
            ``(potential, energy_fn, charges_fn)`` — *charges_fn* may be
            ``None`` if the potential does not support partial charges.
    """
    import jax.numpy as jnp

    from ..potentials import get_potential, list_potentials

    potential_kwargs = {}

    registry_name = args.potential
    spec = extra_potential_spec(args)
    if spec is not None:
        registry_name, extra = spec
        potential_kwargs.update(extra)
        potential_kwargs["species"] = structure["numbers"]

    elif args.potential == "so3lr":
        potential_kwargs.update(
            species=structure["numbers"],
            lr_cutoff=args.lr_cutoff,
            charge=args.charge,
            model=args.so3lr_model,
        )
        if getattr(args, "spin", None) is not None:
            potential_kwargs["spin"] = args.spin
        if args.float64:
            potential_kwargs["dtype"] = jnp.float64
        if compute_charges:
            potential_kwargs["compute_charges"] = True

    elif args.potential == "mace":
        potential_kwargs["species"] = structure["numbers"]
        if hasattr(args, "mace_foundation") and args.mace_foundation:
            potential_kwargs["foundation"] = args.mace_foundation
        if hasattr(args, "mace_model") and args.mace_model:
            potential_kwargs["model"] = args.mace_model
        if getattr(args, "mace_cache_dir", None):
            potential_kwargs["cache_dir"] = args.mace_cache_dir
        # charge/spin are used by charge/spin-aware models (e.g. omol)
        potential_kwargs["charge"] = getattr(args, "charge", 0.0)
        if getattr(args, "spin", None) is not None:
            potential_kwargs["spin"] = args.spin
        potential_kwargs["backend"] = getattr(args, "mace_backend", None) or "auto"
        if args.float64:
            potential_kwargs["dtype"] = jnp.float64

    elif args.potential == "valence":
        # Test force field: needs the elements and a reference geometry to
        # derive its topology from (initialize() is called below anyway, but
        # the constructor signature requires species up front).
        potential_kwargs["species"] = structure["numbers"]

    elif args.potential == "dxtb":
        potential_kwargs["species"] = structure["numbers"]
        potential_kwargs["charge"] = args.charge
        if getattr(args, "spin", None) is not None:
            potential_kwargs["spin"] = args.spin
        if hasattr(args, "dxtb_method") and args.dxtb_method:
            potential_kwargs["method"] = args.dxtb_method
        if args.float64:
            potential_kwargs["dtype"] = jnp.float64

    try:
        check_charge(structure, charge=args.charge, multiplicity=getattr(args, "spin", 1) or 1)
    except ValueError as exc:
        log_error(str(exc))
        sys.exit(1)

    log_potential_and_citation(args)

    if registry_name not in list_potentials():
        available = ", ".join(list_potentials())
        log_error(f"Unknown potential '{args.potential}'. Available: {available}")
        sys.exit(1)
    try:
        potential = get_potential(registry_name, **potential_kwargs)
    except (ValueError, FileNotFoundError) as exc:
        # Report what the constructor actually objected to. This used to be
        # reported as "Unknown potential", which hid every configuration error
        # (a bad model name, an unsupported size) behind a wrong message.
        log_error(f"Could not build potential '{args.potential}': {exc}")
        sys.exit(1)

    try:
        potential.initialize(structure["positions"])
    except ValueError as exc:
        # e.g. a structure that does not match the force-field topology
        log_error(f"Could not initialise potential '{args.potential}': {exc}")
        sys.exit(1)

    if compute_charges:
        result = potential.build_energy_fn(with_charges=True)
        if isinstance(result, tuple):
            energy_fn, charges_fn = result
        else:
            energy_fn, charges_fn = result, None
        return potential, energy_fn, charges_fn

    energy_fn = potential.build_energy_fn()
    return potential, energy_fn
