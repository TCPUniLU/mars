"""
Ensemble management for JAX structures.

Provides sorting, pruning, and I/O operations for ensembles of structure dictionaries.
"""

from typing import Dict, List, Tuple

from .auto_config import EV_TO_KCALMOL
from .log import log_message


def sort_by_energy(ensemble: List[Tuple[Dict, float]]) -> List[Tuple[Dict, float]]:
    """Sort ensemble by energy (ascending).

    Args:
        ensemble: List of (structure, energy) tuples

    Returns:
        sorted_ensemble: Sorted list
    """
    return sorted(ensemble, key=lambda x: x[1])


def prune_by_energy_window(
    ensemble: List[Tuple[Dict, float]], ewin: float = 6.0
) -> List[Tuple[Dict, float]]:
    """Remove structures outside energy window from lowest.

    Args:
        ensemble: List of (structure, energy) tuples (should be sorted)
        ewin: Energy window in eV

    Returns:
        pruned: Structures within energy window
    """
    if not ensemble:
        return ensemble

    ref_energy = ensemble[0][1]
    pruned = [(struct, e) for struct, e in ensemble if e - ref_energy <= ewin]

    removed = len(ensemble) - len(pruned)
    log_message(
        f"[Ensemble] Energy window pruning: kept {len(pruned)}, removed {removed} (Ewin={ewin * EV_TO_KCALMOL:.1f} kcal/mol)"
    )

    return pruned


def get_unique_by_energy(
    ensemble: List[Tuple[Dict, float]], tolerance: float = 1e-6
) -> List[Tuple[Dict, float]]:
    """Remove duplicate structures with identical energies.

    Args:
        ensemble: List of (structure, energy) tuples
        tolerance: Energy tolerance for considering duplicates (eV)

    Returns:
        unique: List without energy duplicates
    """
    if not ensemble:
        return ensemble

    unique = [ensemble[0]]

    for struct, e in ensemble[1:]:
        # Check if energy is different from all kept structures
        is_unique = all(abs(e - e_kept) > tolerance for _, e_kept in unique)
        if is_unique:
            unique.append((struct, e))

    removed = len(ensemble) - len(unique)
    if removed > 0:
        log_message(
            f"[Ensemble] Removed {removed} energy duplicates (tol={tolerance * EV_TO_KCALMOL:.3g} kcal/mol)"
        )

    return unique


def filter_by_energy_range(
    ensemble: List[Tuple[Dict, float]], min_energy: float = None, max_energy: float = None
) -> List[Tuple[Dict, float]]:
    """Filter ensemble to specific energy range.

    Args:
        ensemble: List of (structure, energy) tuples
        min_energy: Minimum energy (inclusive), None = no lower bound
        max_energy: Maximum energy (inclusive), None = no upper bound

    Returns:
        filtered: Structures within range
    """
    filtered = []

    for struct, e in ensemble:
        if min_energy is not None and e < min_energy:
            continue
        if max_energy is not None and e > max_energy:
            continue
        filtered.append((struct, e))

    return filtered


def get_lowest_n(ensemble: List[Tuple[Dict, float]], n: int) -> List[Tuple[Dict, float]]:
    """Get N lowest-energy structures.

    Args:
        ensemble: List of (structure, energy) tuples
        n: Number of structures to keep

    Returns:
        lowest: N lowest-energy structures
    """
    sorted_ensemble = sort_by_energy(ensemble)
    return sorted_ensemble[:n]


def get_energy_statistics(ensemble: List[Tuple[Dict, float]]) -> Dict:
    """Compute statistics about ensemble energies.

    Args:
        ensemble: List of (structure, energy) tuples

    Returns:
        stats: Dictionary with energy statistics
    """
    import numpy as np

    if not ensemble:
        return {}

    energies = [e for _, e in ensemble]

    stats = {
        "n_structures": len(ensemble),
        "min_energy": min(energies),
        "max_energy": max(energies),
        "mean_energy": np.mean(energies),
        "std_energy": np.std(energies),
        "range": max(energies) - min(energies),
    }

    return stats


def print_ensemble_summary(ensemble: List[Tuple[Dict, float]], title: str = "Ensemble Summary"):
    """Print summary of ensemble.

    Args:
        ensemble: List of (structure, energy) tuples
        title: Title for summary
    """
    log_message(f"\n{'='*60}")
    log_message(title)
    log_message(f"{'='*60}")

    if not ensemble:
        log_message("Empty ensemble")
        return

    stats = get_energy_statistics(ensemble)

    log_message(f"Number of structures: {stats['n_structures']}")
    e_min = stats["min_energy"]
    log_message(f"Energy span: {stats['range'] * EV_TO_KCALMOL:.1f} kcal/mol")
    log_message(f"Std energy: {stats['std_energy'] * EV_TO_KCALMOL:.2f} kcal/mol")

    # Show top 10 relative to minimum
    if len(ensemble) > 0:
        log_message(f"\nLowest 10 ΔE (kcal/mol):")
        for i, (_, e) in enumerate(ensemble[:10]):
            log_message(f"  {i+1:3d}. {(e - e_min) * EV_TO_KCALMOL:.2f} kcal/mol")

    log_message(f"{'='*60}\n")
