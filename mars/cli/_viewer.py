"""
Interactive results viewer CLI workflow (`mars viewer`).

Three modes:

* ``mars viewer input.xyz`` — plain 3D molecule viewer.
* ``mars viewer --ir [--dir DIR | --modes ... ]`` — interactive IR explorer.
  Loads saved harmonic-IR outputs (``normal_modes.xyz`` for geometry +
  eigenvectors + frequencies, optional ``ir_spectrum.dat`` for intensities, and
  optional ``mode_analysis.txt`` for the vibrational classification).
* ``mars viewer input.xyz --nci [--nci-buffer X]`` — preview the spherical
  confinement wall used by ``mars ... --nci``.

The viewer reads files; it never recomputes anything.  Reading the IR output
formats lives here (``mars/ir.py`` only *writes* them).
"""

import re
import sys
from pathlib import Path

import numpy as np

from ..log import log_error, log_info, log_warning
from ._common import setup_logging, validate_input_file

# Default filenames written by `mars ir` (see mars/ir.py savers).
_DEFAULT_MODES = "normal_modes.xyz"
_DEFAULT_SPECTRUM = "ir_spectrum.dat"
_DEFAULT_ANALYSIS = "mode_analysis.txt"


def run_viewer_workflow(args):
    """Dispatch to the plain / NCI / IR viewer based on the parsed flags."""
    from .. import viewer

    setup_logging(args)

    if args.ir and args.nci:
        log_error("--ir and --nci cannot be combined.")
        sys.exit(1)

    save = getattr(args, "save", None)

    try:
        if args.ir:
            if args.input is not None:
                log_warning(
                    f"Positional input '{args.input}' is ignored in --ir mode; "
                    "IR data is loaded from result files (see --dir / --modes)."
                )
            ir_data = _resolve_ir_data(args)
            viewer.view_ir(
                ir_data,
                arrow_scale=args.arrow_scale,
                broadening=args.broadening,
                freq_range=tuple(args.freq_range),
                amplitude=args.amplitude,
                atom_scale=args.atom_scale,
                bond_width=args.bond_width,
                show_grid=args.grid,
                save=save,
            )
        elif args.nci:
            structures = _load_structures_or_exit(args)
            viewer.view_nci(
                structures,
                buffer=args.nci_buffer,
                atom_scale=args.atom_scale,
                bond_width=args.bond_width,
                show_grid=args.grid,
                save=save,
            )
        else:
            structures = _load_structures_or_exit(args)
            viewer.view_molecule(
                structures,
                atom_scale=args.atom_scale,
                bond_width=args.bond_width,
                show_grid=args.grid,
                save=save,
            )
    except RuntimeError as exc:
        # matplotlib missing (or other clean viewer error).
        log_error(str(exc))
        sys.exit(1)

    return 0


def _load_structures_or_exit(args):
    """Load all frames of the positional input as a list of structure dicts.

    Multi-frame XYZ files yield one entry per conformer (navigable in the
    viewer); single-frame files yield a one-element list. Exits cleanly on
    missing input or parse errors.
    """
    from ..utils import load_ensemble

    if args.input is None:
        log_error(
            "An input .xyz file is required for this viewer mode "
            "(e.g. `mars viewer molecule.xyz`)."
        )
        sys.exit(1)
    input_path = validate_input_file(args.input)
    ensemble = load_ensemble(str(input_path))
    structures = [structure for structure, _energy in ensemble]
    if not structures:
        log_error(f"No structures found in '{input_path}'.")
        sys.exit(1)
    return structures


# ---------------------------------------------------------------------------
# IR result discovery + parsing
# ---------------------------------------------------------------------------


def _resolve_ir_data(args):
    """Locate IR result files (explicit flags override directory discovery),
    parse them, and return the ``ir_data`` dict consumed by ``viewer.view_ir``.
    """
    search_dir = Path(args.dir) if args.dir else Path.cwd()

    def _pick(explicit, default):
        if explicit:
            return Path(explicit)
        candidate = search_dir / default
        return candidate if candidate.exists() else None

    modes_path = _pick(args.modes, _DEFAULT_MODES)
    spectrum_path = _pick(args.spectrum, _DEFAULT_SPECTRUM)
    analysis_path = _pick(args.mode_analysis, _DEFAULT_ANALYSIS)

    if modes_path is None or not Path(modes_path).exists():
        log_error(
            f"Could not find normal-mode data ('{_DEFAULT_MODES}') in "
            f"'{search_dir}'. Generate it with `mars ir input.xyz --mode-xyz`, "
            "or point at it with --modes FILE (and optionally --dir DIR)."
        )
        sys.exit(1)

    log_info(f"Loading normal modes from {modes_path}")
    symbols, numbers, positions, displacements, frequencies, mode_indices = _parse_normal_modes_xyz(
        str(modes_path)
    )

    # Intensities are optional: a model without a dipole (no charges) yields
    # frequencies + eigenvectors but no intensities. Leave them at zero in that
    # case — the viewer detects this and draws uniform sticks for every mode.
    intensities = np.zeros(len(frequencies), dtype=float)
    if spectrum_path and Path(spectrum_path).exists():
        log_info(f"Loading IR intensities from {spectrum_path}")
        spec_freq, spec_int = _parse_ir_spectrum_dat(str(spectrum_path))
        intensities = _align_intensities(frequencies, spec_freq, spec_int)
        if not np.any(intensities > 0):
            log_warning(
                f"'{spectrum_path}' has no positive intensities (dipole-free "
                "model?) — showing all modes with uniform sticks."
            )
    else:
        log_warning(
            f"No '{_DEFAULT_SPECTRUM}' found — intensities not available; "
            "showing all modes with uniform sticks. "
            "Use --spectrum FILE to provide intensities."
        )

    classification = {}
    if analysis_path and Path(analysis_path).exists():
        log_info(f"Loading mode classification from {analysis_path}")
        classification = _parse_mode_analysis(str(analysis_path))

    return {
        "positions": positions,
        "numbers": numbers,
        "symbols": symbols,
        "frequencies": frequencies,
        "intensities": intensities,
        "displacements": displacements,
        "mode_indices": mode_indices,
        "classification": classification,
    }


def _parse_normal_modes_xyz(path):
    """Parse the multi-frame ``normal_modes.xyz`` written by
    ``ir.save_normal_mode_xyz``.

    Each frame: line 1 = N, line 2 = ``Mode {i}, Freq: {f} cm⁻¹, ...``, then N
    atom lines ``symbol x y z vx vy vz``.

    Returns:
        ``(symbols, numbers, positions, displacements, frequencies, mode_indices)``
        where positions is (N,3) (geometry, identical across frames),
        displacements is (M, N, 3), frequencies/mode_indices are length-M.
    """
    from ..utils import symbols_to_numbers

    with open(path, "r") as f:
        lines = f.readlines()

    freq_re = re.compile(r"Mode\s+(\d+).*Freq:\s*([-+0-9.eE]+)")

    symbols = None
    positions = None
    displacements = []
    frequencies = []
    mode_indices = []

    i = 0
    n_lines = len(lines)
    while i < n_lines:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        try:
            n_atoms = int(line)
        except ValueError:
            i += 1
            continue

        comment = lines[i + 1] if i + 1 < n_lines else ""
        match = freq_re.search(comment)
        if match:
            mode_indices.append(int(match.group(1)))
            frequencies.append(float(match.group(2)))
        else:
            mode_indices.append(len(mode_indices))
            frequencies.append(0.0)

        frame_syms = []
        frame_pos = []
        frame_disp = []
        for k in range(n_atoms):
            parts = lines[i + 2 + k].split()
            frame_syms.append(parts[0])
            frame_pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if len(parts) >= 7:
                frame_disp.append([float(parts[4]), float(parts[5]), float(parts[6])])
            else:
                frame_disp.append([0.0, 0.0, 0.0])

        if symbols is None:
            symbols = frame_syms
            positions = np.asarray(frame_pos, dtype=float)
        displacements.append(np.asarray(frame_disp, dtype=float))

        i += 2 + n_atoms

    if symbols is None:
        raise RuntimeError(f"No frames found in normal-mode file '{path}'.")

    numbers = np.asarray(symbols_to_numbers(symbols), dtype=int)
    return (
        symbols,
        numbers,
        positions,
        np.asarray(displacements, dtype=float),
        np.asarray(frequencies, dtype=float),
        np.asarray(mode_indices, dtype=int),
    )


def _parse_ir_spectrum_dat(path):
    """Parse the 2-column ``ir_spectrum.dat`` (freq, intensity; '#' comments)."""
    data = np.loadtxt(path, comments="#")
    data = np.atleast_2d(data)
    if data.size == 0:
        return np.zeros(0), np.zeros(0)
    return data[:, 0], data[:, 1]


def _align_intensities(frequencies, spec_freq, spec_int):
    """Map spectrum intensities onto the mode list.

    The savers write modes and spectrum in the same order, so use a direct
    index map when the lengths match; otherwise fall back to nearest-frequency
    matching.
    """
    intensities = np.zeros(len(frequencies), dtype=float)
    if len(spec_freq) == len(frequencies):
        return np.asarray(spec_int, dtype=float)
    if len(spec_freq) == 0:
        return intensities
    for i, f in enumerate(frequencies):
        j = int(np.argmin(np.abs(spec_freq - f)))
        intensities[i] = spec_int[j]
    return intensities


def _parse_mode_analysis(path):
    """Parse the 'ALL MODES' table in ``mode_analysis.txt``.

    Returns ``{mode_index: {'type', 'label', 'functional_group'}}``.

    The table is written with fixed-width columns by ``ir.save_mode_analysis``
    (idx[0:6], freq[7:19], intensity[22:32], type[35:47], label[48:78],
    functional group[79:]). We slice by column so that label/group fields that
    themselves contain spaces are recovered correctly, falling back to a
    whitespace split for any short/irregular row. Classification is cosmetic
    for the viewer, so parsing stays tolerant of absence.
    """
    result = {}
    try:
        with open(path, "r") as f:
            lines = f.readlines()
    except OSError:
        return result

    in_table = False
    for raw in lines:
        line = raw.rstrip("\n")
        if "ALL MODES" in line:
            in_table = True
            continue
        if not in_table:
            continue
        if "DETAILED VIBRATIONAL" in line:
            break
        stripped = line.strip()
        if not stripped or stripped.startswith("-") or stripped.startswith("Mode"):
            continue

        # Fixed-width column parse (matches the writer's format string).
        try:
            mode_idx = int(line[0:6])
        except (ValueError, IndexError):
            continue
        if len(line) >= 47:
            mtype = line[35:47].strip()
            label = line[48:78].strip() if len(line) > 48 else ""
            functional_group = line[79:].strip() if len(line) > 79 else ""
        else:
            # Fallback for irregular rows: whitespace split.
            parts = stripped.split()
            mtype = parts[3] if len(parts) > 3 else ""
            label = parts[4] if len(parts) > 4 else ""
            functional_group = " ".join(parts[5:]) if len(parts) > 5 else ""

        result[mode_idx] = {
            "type": mtype,
            "label": label,
            "functional_group": functional_group,
        }
    return result
