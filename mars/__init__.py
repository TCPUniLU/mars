"""
MARS — Machine-Learned Force Field Framework for Automated Conformational
Sampling, Vibrational Spectroscopy, and Microsolvation

A high-performance conformational search package using JAX for GPU acceleration.

Top-level names are imported lazily (PEP 562): importing :mod:`mars` itself is
cheap and does not pull in JAX / jax-md / the ML potentials. The heavy
submodules are only imported the first time one of their symbols is accessed.
This lets lightweight, JAX-free tools (e.g. ``mars viewer``) run in
environments without a working JAX/GPU stack.
"""

__version__ = "0.1.1"

# ============================================================================
# Suppress external package warnings by default
# ============================================================================
import warnings
import logging
import os
import importlib

warnings.filterwarnings("ignore")
logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

# Suppress TF/JAX C++ level logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("JAX_PLATFORMS", os.environ.get("JAX_PLATFORMS", ""))


def enable_debug():
    """Re-enable all warnings and verbose logging for debugging."""
    from .log import init_log

    warnings.filterwarnings("default")
    logging.getLogger("jax").setLevel(logging.DEBUG)
    logging.getLogger("absl").setLevel(logging.DEBUG)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    init_log(level="DEBUG")


# ============================================================================
# Lazy attribute resolution (PEP 562)
# ============================================================================
# Map each public name to the submodule that provides it. Submodules are
# imported on first access only, so `import mars` stays JAX-free.

_LAZY_SUBMODULES = {
    "conf_sampling": (
        "run_conformer_search_auto",
        "run_mtd_only",
        "run_md_only",
    ),
    "sampling": (
        "run_mtd_jax",
        "run_multi_mtd",
        "run_rotamer_md_jax",
        "run_rotamer_md_parallel",
        "run_multi_temperature_rotamer_md",
        "run_mtd_sampling",
        "analyze_trajectory",
    ),
    "optimizer": (
        "optimize_single",
        "optimize_single_ric",
        "optimize_batch_parallel",
        "find_stationary_point",
        "find_stationary_point_with_logging",
        "ric_gate",
        "RIC_DEFAULTS",
    ),
    "internal_coords": (
        "CoordinateSpec",
        "build_coordinates",
        "make_coordinate_fns",
        "generalized_inverse",
        "internal_to_cartesian",
        "global_tr_basis",
        "initial_frag_ref",
        "wrap_periodic",
    ),
    "mtd": (
        "MTDState",
        "create_mtd_state",
        "add_hill",
        "compute_mtd_bias",
        "get_mtd_statistics",
        "save_mtd_state",
        "load_mtd_state",
    ),
    "rmsd": (
        "kabsch_jax",
        "center_positions",
        "align_to_reference",
        "rmsd_cv_jax",
        "rmsd_and_grad_jax",
        "rmsd_to_multiple_refs",
        "pairwise_rmsd_matrix",
        "check_rmsd_gradient",
        "radius_of_gyration",
        "batch_radius_of_gyration",
    ),
    "ensemble": (
        "sort_by_energy",
        "prune_by_energy_window",
        "get_unique_by_energy",
        "filter_by_energy_range",
        "get_lowest_n",
        "get_energy_statistics",
        "print_ensemble_summary",
    ),
    "utils_prune": (
        "prune_by_rmsd",
        "compute_pairwise_rmsd_ensemble",
        "find_most_diverse_subset",
        "cluster_by_rmsd",
        "get_cluster_representatives",
    ),
    "crossing": ("genetic_zmatrix_crossing",),
    "utils": (
        "create_structure",
        "structure_to_arrays",
        "load_structure",
        "save_structure",
        "save_ensemble",
        "load_ensemble",
        "ase_atoms_to_structure",
        "structure_to_ase_atoms",
        "enable_float64",
        "get_device_info",
        "to_device",
        "check_charge",
        "KB_EV_PER_K",
        "detect_bonds",
        "COVALENT_RADII",
        "ATOMIC_MASSES",
        "VDW_RADII",
        "ELEMENT_SYMBOLS",
        "SYMBOL_TO_NUMBER",
        "NUMBER_TO_SYMBOL",
        "get_vdw_radius",
        "get_covalent_radius",
        "get_atomic_mass",
        "symbol_to_number",
        "number_to_symbol",
        "symbols_to_numbers",
        "lindh_model_hessian_cartesian",
        "model_hessian_cartesian",
        "initial_internal_hessian",
        "HESSIAN_MODES",
    ),
    "potentials": (
        "PotentialWrapper",
        "get_potential",
        "list_potentials",
        "register_potential",
    ),
    "auto_config": (
        "auto_setup",
        "estimate_flexibility",
        "compute_md_length",
        "generate_mtd_parameters",
        "run_trial_mtd",
        "EV_TO_KCALMOL",
        "KCALMOL_TO_EV",
    ),
    "log": (
        "init_log",
        "log_message",
        "log_header",
        "log_energy_table",
        "log_topology",
    ),
    "topology": (
        "MoleculeTopology",
        "analyse_topology",
        "is_multimolecular",
        "count_molecules",
        "get_fragments",
        "split_structure",
    ),
    "solvation": (
        "solvate",
        "auto_solvate",
        "place_solvents_by_layers",
        "build_and_optimise_layers",
        "optimize_solvation_shell",
        "remove_broken_solvent",
        "solvate_barostat",
        "solvate_barostat_batch",
        "transplant_solvent_shell",
        "solvate_multi_seed",
        "optimize_solvation_batch",
        "align_to_reference_np",
        "LayerSpec",
        "get_solvent",
        "load_solvent",
        "SOLVENT_DB",
    ),
}

# Names that are exposed under a different alias than in their submodule.
_LAZY_ALIASES = {
    "run_conformer_search": ("conf_sampling", "run_conformer_search_jax"),
    "optimize_multilevel": ("optimizer", "optimize_multilevel_jax"),
}

# Submodules exposed as attributes themselves (e.g. ``mars.ir``).
_LAZY_MODULE_ATTRS = ("ir",)

# Reverse map: public name -> submodule name.
_NAME_TO_SUBMODULE = {name: sub for sub, names in _LAZY_SUBMODULES.items() for name in names}


def __getattr__(name):
    """Lazily import the submodule that provides *name* on first access."""
    if name in _LAZY_ALIASES:
        sub, real = _LAZY_ALIASES[name]
        value = getattr(importlib.import_module(f".{sub}", __name__), real)
    elif name in _NAME_TO_SUBMODULE:
        sub = _NAME_TO_SUBMODULE[name]
        value = getattr(importlib.import_module(f".{sub}", __name__), name)
    elif name in _LAZY_MODULE_ATTRS:
        value = importlib.import_module(f".{name}", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__():
    return sorted(
        list(globals().keys())
        + list(_NAME_TO_SUBMODULE)
        + list(_LAZY_ALIASES)
        + list(_LAZY_MODULE_ATTRS)
    )


# ============================================================================
# Convenience Functions
# ============================================================================


def quick_conformer_search(xyz_file: str, energy_fn, **kwargs):
    """Quick conformational search from XYZ file.

    Args:
        xyz_file: Path to input XYZ file
        energy_fn: JAX energy function
        **kwargs: Additional parameters for run_conformer_search_auto

    Returns:
        ensemble: List of (structure, energy) tuples
    """
    from .utils import load_structure
    from .conf_sampling import run_conformer_search_auto

    structure = load_structure(xyz_file)
    return run_conformer_search_auto(structure, energy_fn, mode="quick", **kwargs)


def thorough_conformer_search(xyz_file: str, energy_fn, **kwargs):
    """Thorough conformational search from XYZ file.

    Args:
        xyz_file: Path to input XYZ file
        energy_fn: JAX energy function
        **kwargs: Additional parameters for run_conformer_search_auto

    Returns:
        ensemble: List of (structure, energy) tuples
    """
    from .utils import load_structure
    from .conf_sampling import run_conformer_search_auto

    structure = load_structure(xyz_file)
    return run_conformer_search_auto(structure, energy_fn, mode="thorough", **kwargs)


# ============================================================================
# Module Metadata
# ============================================================================

__all__ = [
    # Main workflow
    "run_conformer_search",
    "run_conformer_search_auto",
    "run_mtd_only",
    "run_md_only",
    # Sampling
    "run_mtd_jax",
    "run_multi_mtd",
    "run_rotamer_md_jax",
    "run_rotamer_md_parallel",
    "run_multi_temperature_rotamer_md",
    # Optimization
    "optimize_single",
    "optimize_single_ric",
    "build_coordinates",
    "make_coordinate_fns",
    "generalized_inverse",
    "internal_to_cartesian",
    "lindh_model_hessian_cartesian",
    "initial_internal_hessian",
    "optimize_multilevel",
    "optimize_batch_parallel",
    "find_stationary_point",
    "find_stationary_point_with_logging",
    # Metadynamics
    "MTDState",
    "create_mtd_state",
    "add_hill",
    "compute_mtd_bias",
    "get_mtd_statistics",
    "save_mtd_state",
    "load_mtd_state",
    # RMSD
    "kabsch_jax",
    "center_positions",
    "align_to_reference",
    "rmsd_cv_jax",
    "rmsd_and_grad_jax",
    "rmsd_to_multiple_refs",
    "pairwise_rmsd_matrix",
    "check_rmsd_gradient",
    # Ensemble
    "sort_by_energy",
    "prune_by_energy_window",
    "prune_by_rmsd",
    # Genetic crossing
    "genetic_zmatrix_crossing",
    # Utilities
    "load_structure",
    "save_structure",
    "save_ensemble",
    "load_ensemble",
    "create_structure",
    "check_charge",
    "enable_float64",
    "enable_debug",
    # Auto-Configuration
    "auto_setup",
    "estimate_flexibility",
    "compute_md_length",
    "generate_mtd_parameters",
    "run_trial_mtd",
    # Topology
    "MoleculeTopology",
    "analyse_topology",
    "is_multimolecular",
    "count_molecules",
    "get_fragments",
    "split_structure",
    # Potentials
    "PotentialWrapper",
    "get_potential",
    "list_potentials",
    "register_potential",
    "detect_bonds",
    "COVALENT_RADII",
    # IR submodule
    "ir",
    # Convenience
    "quick_conformer_search",
    "thorough_conformer_search",
]
