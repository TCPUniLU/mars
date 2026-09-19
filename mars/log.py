"""
Logging utilities for MARS using Python's logging package.

Provides structured logging with configurable output levels and formats.
Console output is clean and user-friendly; file output includes timestamps.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Create logger
logger = logging.getLogger("MARS")
logger.propagate = False  # Prevent duplicate messages from root logger

# Track whether the banner has been displayed
_banner_displayed = False

MARS_CITATION = (
    "S. Suárez-Dou et al., MARS: Machine-Learned Force Field\n"
    "Framework for Automated Conformational Sampling, Vibrational Spectroscopy,\n"
    "and Microsolvation, ChemRxiv (2026). doi:10.26434/chemrxiv.15007087/v1"
)


# ============================================================================
# Global Timing Tracker
# ============================================================================


class TimingTracker:
    """Tracks cumulative wall-clock time spent in named workflow stages."""

    def __init__(self):
        self._stages = []  # [(name, total_seconds)]
        self._stage_idx = {}  # name -> index in _stages
        self._current = {}  # name -> start_time (float)
        self._total_start = time.time()

    def reset(self):
        self._stages.clear()
        self._stage_idx.clear()
        self._current.clear()
        self._total_start = time.time()

    def start(self, stage: str):
        self._current[stage] = time.time()
        if stage not in self._stage_idx:
            self._stage_idx[stage] = len(self._stages)
            self._stages.append([stage, 0.0])

    def end(self, stage: str):
        if stage in self._current:
            elapsed = time.time() - self._current.pop(stage)
            if stage in self._stage_idx:
                self._stages[self._stage_idx[stage]][1] += elapsed

    def total(self) -> float:
        return time.time() - self._total_start

    def get_stages(self):
        """Return list of (name, total_seconds) in insertion order."""
        return [(name, t) for name, t in self._stages]


_global_timer = TimingTracker()


def timer_reset():
    """Reset the global timing tracker. Call at the start of each workflow run."""
    _global_timer.reset()


def timer_start(stage: str):
    """Start timing a named workflow stage."""
    _global_timer.start(stage)


def timer_end(stage: str):
    """Stop timing a named stage and accumulate the elapsed time."""
    _global_timer.end(stage)


def log_timing_summary():
    """Print a formatted timing summary table for all recorded stages."""
    stages = _global_timer.get_stages()
    total = _global_timer.total()
    if not stages:
        return

    def _fmt(t: float) -> str:
        if t >= 3600:
            return f"{t / 3600:.2f}h"
        if t >= 60:
            return f"{t / 60:.1f}min"
        return f"{t:.1f}s"

    logger.info("")
    logger.info("=" * 70)
    logger.info("Timing Summary")
    logger.info("=" * 70)
    for name, t in stages:
        pct = 100.0 * t / total if total > 0 else 0.0
        logger.info(f"  {name:<42} {_fmt(t):>10}   {pct:5.1f}%")
    logger.info(f"  {'-' * 60}")
    logger.info(f"  {'Total wall time':<42} {_fmt(total):>10}")
    logger.info("=" * 70)
    logger.info("")


# ============================================================================
# Console-friendly formatter (no timestamp, clean output)
# ============================================================================


class _ConsoleFormatter(logging.Formatter):
    """Clean console formatter: no timestamps, level shown only for warnings/errors."""

    def format(self, record):
        if record.levelno >= logging.WARNING:
            return f"  {record.levelname}: {record.getMessage()}"
        return f"  {record.getMessage()}"


# File formatter (with timestamps for post-run analysis)
_FILE_FORMAT = "[%(asctime)s] %(levelname)s - %(message)s"
_FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"


def init_log(
    logfile: Optional[str] = None,
    level: str = "INFO",
    console: bool = True,
    format_str: Optional[str] = None,
):
    """Initialize logging configuration.

    Args:
        logfile: Optional path to log file. If None, log to console only.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: Whether to log to console (default: True)
        format_str: Optional custom format string (used for file handler only)

    Example:
        >>> init_log("MARS.log", level="DEBUG")
        >>> init_log(console=True)  # Console only
    """
    global _banner_displayed

    # Clear any existing handlers
    logger.handlers.clear()

    # Set level
    logger.setLevel(getattr(logging, level.upper()))

    # Console handler (clean, no timestamps)
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(_ConsoleFormatter())
        logger.addHandler(console_handler)

    # File handler (with timestamps)
    if logfile:
        file_formatter = logging.Formatter(format_str or _FILE_FORMAT, datefmt=_FILE_DATEFMT)
        file_handler = logging.FileHandler(logfile, mode="w")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Log initialization banner (only once)
    if not _banner_displayed:
        _banner_displayed = True
        _RED = "\033[91m"
        _RESET = "\033[0m"
        logger.info("")
        logger.info("=" * 70)
        logger.info(_RED + """

             ▒▒░░░░░░░▒▒▒░░░░▒
          ▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░▒▒
       ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▒▒▒▒░░░░░░░░
      ▒▒▒▒▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░▒▒▒░░░░
     ▒▒▒▒▒▓▒▓▒▒▒▓▓▒▒▒▒▒▒▒▒▒░░░▒▒▒▒▒░░▒
    ▒▒▓▓▓▓▓▓▒▒▓▒▒▒▒▒▒▒▒░░▒░░░░▒▒▒▒▒▒░░▒▒
   ▒▒▒▓▓▓▓▓▓▒▓▓▓▒▒░░▒▒░░▒▓▒▒░░░░░▒▒▒░░░▒
  ▒▓▒▒▒▓▓▒▒▒▓▓▓  ██████   ██████   █████████   ███████████    █████████
  ░▒▒▒▒▒▒▒▒▓▒▓▓ ▒▒██████ ██████   ███▒▒▒▒▒███ ▒▒███▒▒▒▒▒███  ███▒▒▒▒▒███
 ░░░▒▓▒▒▒▒▓▒▒▓▓  ▒███▒█████▒███  ▒███    ▒███  ▒███    ▒███ ▒███    ▒▒▒
 ░░░▒░░░░▒▒▓▒▒▓  ▒███▒▒███ ▒███  ▒███████████  ▒██████████  ▒▒██████████
 ░░▒░▒░░░░▒▓▓▓▒  ▒███ ▒▒▒  ▒███  ▒███▒▒▒▒▒███  ▒███▒▒▒▒▒███  ▒▒▒▒▒▒▒▒███
  ▒▒▒░░░░░░░░▒▒  ▒███      ▒███  ▒███    ▒███  ▒███    ▒███  ███    ▒███
  ▒▒▒░▒░░░░░░░▒  █████     █████ █████   █████ █████   █████▒▒█████████
   ▒▒░░░▒░░░░░▒ ▒▒▒▒▒     ▒▒▒▒▒ ▒▒▒▒▒   ▒▒▒▒▒ ▒▒▒▒▒   ▒▒▒▒▒  ▒▒▒▒▒▒▒▒▒
    ▒▒░░▒▒░░░░░▒▒▒▒▒▒▒░░▒▒▒░▒▒▒░▒░░▒▒▒▒▒
     ▒░░░▒▒▒░░░░░░▒▒▒░░░▒▒▒▒▒▒▒▒▒░░░░▒
      ▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒
       ▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░▒▒░▒▒
          ▒▒▒▒▒░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░
             ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                   ▒▒▒▒▒

""" + _RESET)
        logger.info(
            "Machine-Learned Force Field Framework for Automated Conformational".center(70)
            + "\n"
            + "Sampling, Vibrational Spectroscopy, and Microsolvation".center(70)
        )
        logger.info("=" * 70)
        logger.info("")
        logger.info("If you use MARS, please cite:")
        logger.info("  " + MARS_CITATION)
        logger.info("")


def log_header(title: str):
    """Log a section header with visual separation.

    Args:
        title: Section title
    """
    logger.info("")
    logger.info("")
    logger.info("-" * 60)
    logger.info(title)
    logger.info("-" * 60)
    logger.info("")


def log_message(msg: str, level: str = "INFO"):
    """Log a message at specified level.

    Args:
        msg: Message to log
        level: Log level (DEBUG, INFO, WARNING, ERROR)
    """
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(msg)


def log_debug(msg: str):
    """Log debug message."""
    logger.debug(msg)


def log_info(msg: str):
    """Log info message."""
    logger.info(msg)


def log_warning(msg: str):
    """Log warning message."""
    logger.warning(msg)


def log_error(msg: str):
    """Log error message."""
    logger.error(msg)


def log_topology(
    bonds: List[Tuple[int, int]],
    distances: List[float],
    atomic_numbers: Optional[List[int]] = None,
    n_show: int = 10,
    n_molecules: Optional[int] = None,
    fragment_sizes: Optional[List[int]] = None,
):
    """Log bond topology information.

    Args:
        bonds: List of (i, j) tuples representing bonded atom pairs
        distances: List of bond distances in Angstrom
        atomic_numbers: Optional list of atomic numbers for element info
        n_show: Maximum number of bonds to show in detail (default: 10)
        n_molecules: Optional number of distinct molecular fragments
        fragment_sizes: Optional list of atom counts per fragment
    """
    logger.info("")
    logger.info("Bond Topology Summary")
    logger.info("-" * 60)
    if n_molecules is not None:
        if n_molecules == 1:
            logger.info("Molecular fragments: 1 (single-molecule system)")
        else:
            sizes_str = ", ".join(str(s) for s in (fragment_sizes or []))
            logger.info(f"Molecular fragments: {n_molecules} (sizes: {sizes_str})")
    logger.info(f"Total bonds detected: {len(bonds)}")

    if len(bonds) == 0:
        logger.warning("No bonds detected - topology constraints disabled")
        return

    # Bond statistics
    import numpy as np

    distances_arr = np.array(distances)
    logger.info(f"Bond distance range: {distances_arr.min():.3f} - {distances_arr.max():.3f} Å")
    logger.info(f"Mean bond distance: {distances_arr.mean():.3f} Å")

    # Count bonds by approximate type (based on distance)
    short_bonds = np.sum(distances_arr < 1.2)  # e.g., H-X, C=C
    medium_bonds = np.sum((distances_arr >= 1.2) & (distances_arr < 1.6))  # e.g., C-C, C-N
    long_bonds = np.sum(distances_arr >= 1.6)  # e.g., C-O, C-S

    logger.info("")
    logger.info("Bond distribution:")
    if short_bonds > 0:
        logger.info(f"  Short bonds (< 1.2 Å): {short_bonds} (likely H-X or multiple bonds)")
    if medium_bonds > 0:
        logger.info(f"  Medium bonds (1.2-1.6 Å): {medium_bonds} (likely C-C, C-N, C-O)")
    if long_bonds > 0:
        logger.info(f"  Long bonds (> 1.6 Å): {long_bonds} (likely involving heavy atoms)")

    # Show example bonds
    if len(bonds) > 0:
        n_display = min(n_show, len(bonds))
        logger.info("")
        logger.info(f"Example bonds (showing {n_display} of {len(bonds)}):")

        from .utils import number_to_symbol

        for idx in range(n_display):
            i, j = bonds[idx]
            dist = distances[idx]

            if atomic_numbers is not None:
                elem_i = number_to_symbol(int(atomic_numbers[i]))
                elem_j = number_to_symbol(int(atomic_numbers[j]))
                logger.info(
                    f"  Bond {idx+1:3d}: {elem_i:>2s}({i:3d}) - {elem_j:>2s}({j:3d})  "
                    f"r₀ = {dist:5.3f} Å"
                )
            else:
                logger.info(f"  Bond {idx+1:3d}: atom {i:3d} - atom {j:3d}  r₀ = {dist:5.3f} Å")

        if len(bonds) > n_show:
            logger.info(f"  ... and {len(bonds) - n_show} more bonds")

    logger.info("")
    logger.info("-" * 60)


def log_step_start(name: str) -> float:
    """Log the start of a computation step.

    Args:
        name: Step name

    Returns:
        start_time: Timestamp for duration calculation
    """
    t = time.time()
    logger.info(f">> {name} ...")
    return t


def log_step_end(name: str, start_time: float):
    """Log the end of a computation step with duration.

    Args:
        name: Step name
        start_time: Start timestamp from log_step_start
    """
    dt = time.time() - start_time
    logger.info(f"<< {name} done ({dt:.1f}s)")
    logger.info("")


def log_energy_table(
    ensemble: List[Tuple[Dict, float]], title: str = "Energies", max_rows: int = 20
):
    """Log a formatted energy table.

    Energies are stored internally in eV but displayed in kcal/mol.

    Args:
        ensemble: List of (structure, energy) tuples (energy in eV)
        title: Table title
        max_rows: Maximum rows to display
    """
    from .auto_config import EV_TO_KCALMOL

    if not ensemble:
        logger.info(f"  {title}: no structures")
        return

    # Find minimum energy for relative display
    e_min = min(energy for _, energy in ensemble)

    logger.info("")
    logger.info(f"  {title}")
    logger.info(f"  {'Index':<8} {'ΔE (kcal/mol)':>18}")
    logger.info(f"  {'-' * 28}")

    for i, (struct, energy) in enumerate(ensemble[:max_rows]):
        rel_energy = (energy - e_min) * EV_TO_KCALMOL
        logger.info(f"  {i+1:<8d} {rel_energy:18.2f}")

    if len(ensemble) > max_rows:
        logger.info(f"  ... ({len(ensemble) - max_rows} more structures)")

    logger.info(f"  {'-' * 28}")
    logger.info("")


# ============================================================================
# Progress bars (tqdm-based)
# ============================================================================


def progress_bar(iterable, total=None, desc="", **kwargs):
    """Create a progress bar using tqdm if available, otherwise a simple fallback.

    Args:
        iterable: Iterable to wrap
        total: Total number of items (optional)
        desc: Description for the progress bar
        **kwargs: Extra kwargs passed to tqdm

    Returns:
        Wrapped iterable with progress display
    """
    if tqdm is not None:
        return tqdm(
            iterable,
            total=total,
            desc=f"  {desc}",
            bar_format="  {desc}: {bar:30} {percentage:3.0f}% [{n_fmt}/{total_fmt}] {elapsed}<{remaining}",
            **kwargs,
        )
    else:
        # Fallback: just return iterable, log start/end
        logger.info(f"  {desc} ({total or '?'} items)")
        return iterable


# Legacy progress functions (kept for compatibility but now minimal)
def log_progress_init(title: str, total: int):
    """Initialize progress tracking (legacy, prefer progress_bar())."""
    pass  # Now handled by tqdm


def log_progress_update(current: int, total: int, bar_length: int = 30):
    """Log progress update (legacy, prefer progress_bar())."""
    pass  # Now handled by tqdm


def log_progress_end():
    """Mark progress as complete (legacy, prefer progress_bar())."""
    pass  # Now handled by tqdm


def log_statistics(stats: Dict, title: str = "Statistics"):
    """Log dictionary of statistics.

    Args:
        stats: Dictionary of statistics
        title: Section title
    """
    logger.info("")
    logger.info(f"  {title}")
    logger.info(f"  {'-' * 40}")
    for key, value in stats.items():
        if isinstance(value, float):
            logger.info(f"    {key}: {value:.3g}")
        else:
            logger.info(f"    {key}: {value}")
    logger.info(f"  {'-' * 40}")
    logger.info("")


def log_parameters(params: Dict, title: str = "Parameters"):
    """Log workflow parameters.

    Args:
        params: Dictionary of parameters
        title: Section title
    """
    logger.info("")
    logger.info(f"  {title}")
    logger.info(f"  {'-' * 40}")
    for key, value in params.items():
        logger.info(f"    {key}: {value}")
    logger.info(f"  {'-' * 40}")
    logger.info("")


# Convenience context manager for timing
class LogTimer:
    """Context manager for timing code blocks.

    Example:
        >>> with LogTimer("Optimization"):
        ...     optimize_structure()
    """

    def __init__(self, name: str):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = log_step_start(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        log_step_end(self.name, self.start_time)
        return False


# Configure basic logging if not already configured
def _setup_default_logging():
    """Setup a minimal console handler if none exists yet.

    This is called at module import time as a safety net so that any log
    messages emitted before the workflow calls ``init_log`` (e.g. during
    early imports) are not silently dropped.  Importantly it does *not*
    display the banner or set ``_banner_displayed``, so that the banner is
    shown (to both console *and* any log file) the first time ``init_log``
    is explicitly called by a workflow.
    """
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(_ConsoleFormatter())
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)


# Setup on import
_setup_default_logging()
