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
    """Configure JAX floating point precision and device."""
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
        return getattr(args, "so3lr_model", None) or "so3lr_v1"
    if p == "mace":
        foundation = getattr(args, "mace_foundation", None) or "off"
        model = getattr(args, "mace_model", None) or "small"
        return f"{foundation}/{model}"
    if p == "dxtb":
        return getattr(args, "dxtb_method", None) or "gfn1"
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
    return []


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

    citations = _potential_citations(args)
    if citations:
        log_message(f"  If you use the {potential} potential, please cite:")
        for line in citations:
            log_message(f"    {line}")
        log_message("")


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

    if args.potential == "so3lr":
        potential_kwargs.update(
            species=structure["numbers"],
            lr_cutoff=args.lr_cutoff,
            charge=args.charge,
            model=args.so3lr_model,
        )
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
        if args.float64:
            potential_kwargs["dtype"] = jnp.float64

    elif args.potential == "dxtb":
        potential_kwargs["species"] = structure["numbers"]
        potential_kwargs["charge"] = args.charge
        if hasattr(args, "dxtb_method") and args.dxtb_method:
            potential_kwargs["method"] = args.dxtb_method
        if args.float64:
            potential_kwargs["dtype"] = jnp.float64

    try:
        check_charge(structure, charge=args.charge)
    except ValueError as exc:
        log_error(str(exc))
        sys.exit(1)

    log_potential_and_citation(args)

    try:
        potential = get_potential(args.potential, **potential_kwargs)
    except ValueError:
        available = ", ".join(list_potentials())
        log_error(f"Unknown potential '{args.potential}'. Available: {available}")
        sys.exit(1)

    potential.initialize(structure["positions"])

    if compute_charges:
        result = potential.build_energy_fn(with_charges=True)
        if isinstance(result, tuple):
            energy_fn, charges_fn = result
        else:
            energy_fn, charges_fn = result, None
        return potential, energy_fn, charges_fn

    energy_fn = potential.build_energy_fn()
    return potential, energy_fn
