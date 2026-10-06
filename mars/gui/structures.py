"""
Lightweight, JAX-free structure parsing for the MARS web GUI.

The GUI server must start instantly and validate uploads without importing
JAX / jax-md (``mars.utils`` pulls those in), so this module carries small
pure-Python readers for the formats the GUI accepts:

* XYZ (single or multi-frame, e.g. a conformer ensemble),
* PDB (``ATOM`` / ``HETATM`` records, first ``MODEL`` only),
* MDL MOL / SDF (V2000 atom block, first record only),
* Tripos MOL2 (``@<TRIPOS>ATOM`` section).

Anything else is handed to ASE (``ase.io.read``) when it is installed.
Every reader returns a list of frames, each a dict with ``symbols``,
``positions`` (list of ``[x, y, z]`` in Å) and ``comment``.
"""

import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

STRUCTURE_EXTENSIONS = (".xyz", ".pdb", ".sdf", ".mol", ".mol2", ".extxyz")

# Element symbols (Z = 1..118) used to normalise / validate atom labels.
_ELEMENTS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga "
    "Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd "
    "Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac "
    "Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
_SYMBOL_BY_LOWER = {s.lower(): s for s in _ELEMENTS}

_ENERGY_RE = re.compile(r"Energy\s*=\s*([-+0-9.eE]+)\s*(kcal/mol|eV)?", re.IGNORECASE)


class StructureError(ValueError):
    """Raised when an uploaded file cannot be parsed as a molecular structure."""


def normalize_symbol(label: str) -> str:
    """Map an atom label (``"CL"``, ``"c"``, ``"6"``, ``"C1"``) to an element symbol.

    Args:
        label: Raw label from a structure file.

    Returns:
        The canonical element symbol.

    Raises:
        StructureError: If the label does not correspond to an element.
    """
    raw = label.strip()
    if raw.isdigit():
        z = int(raw)
        if 1 <= z <= len(_ELEMENTS):
            return _ELEMENTS[z - 1]
        raise StructureError(f"invalid atomic number {raw!r}")
    letters = re.match(r"[A-Za-z]+", raw)
    if not letters:
        raise StructureError(f"invalid element label {label!r}")
    word = letters.group(0)
    for n in (2, 1):  # prefer a valid two-letter symbol ("Cl"), else one letter ("C1")
        if len(word) >= n and word[:n].lower() in _SYMBOL_BY_LOWER:
            return _SYMBOL_BY_LOWER[word[:n].lower()]
    raise StructureError(f"unknown element {label!r}")


def _frame(symbols, positions, comment="") -> Dict:
    return {"symbols": list(symbols), "positions": positions, "comment": comment}


def parse_xyz(text: str) -> List[Dict]:
    """Parse a (multi-frame) XYZ string.

    Args:
        text: File content.

    Returns:
        List of frames.

    Raises:
        StructureError: On malformed input.
    """
    lines = text.splitlines()
    frames = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        try:
            n = int(lines[i].split()[0])
        except (ValueError, IndexError):
            raise StructureError(f"line {i + 1}: expected an atom count, got {lines[i]!r}")
        if n <= 0:
            raise StructureError(f"line {i + 1}: atom count must be positive")
        comment = lines[i + 1] if i + 1 < len(lines) else ""
        symbols, positions = [], []
        for k in range(n):
            ln = i + 2 + k
            if ln >= len(lines):
                raise StructureError(f"frame {len(frames) + 1}: expected {n} atoms, file ended")
            parts = lines[ln].split()
            if len(parts) < 4:
                raise StructureError(f"line {ln + 1}: expected 'symbol x y z', got {lines[ln]!r}")
            symbols.append(normalize_symbol(parts[0]))
            try:
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                raise StructureError(f"line {ln + 1}: non-numeric coordinates")
        frames.append(_frame(symbols, positions, comment))
        i += 2 + n
    if not frames:
        raise StructureError("no atoms found")
    return frames


def parse_pdb(text: str) -> List[Dict]:
    """Parse ATOM/HETATM records of a PDB string (first MODEL only)."""
    symbols, positions = [], []
    for line in text.splitlines():
        rec = line[:6].strip().upper()
        if rec == "ENDMDL" and symbols:
            break
        if rec not in ("ATOM", "HETATM"):
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            raise StructureError(f"bad PDB coordinate record: {line!r}")
        elem = line[76:78].strip() if len(line) >= 78 else ""
        if not elem:
            name = line[12:16].strip()
            elem = name.lstrip("0123456789")[:2]
            if len(elem) == 2 and elem.lower() not in _SYMBOL_BY_LOWER:
                elem = elem[0]
        symbols.append(normalize_symbol(elem))
        positions.append([x, y, z])
    if not symbols:
        raise StructureError("no ATOM/HETATM records found")
    return [_frame(symbols, positions, "")]


def parse_sdf(text: str) -> List[Dict]:
    """Parse the first record of an MDL MOL / SDF (V2000) string."""
    lines = text.splitlines()
    if len(lines) < 4:
        raise StructureError("MOL/SDF file too short")
    counts = lines[3]
    if "V3000" in counts:
        raise StructureError("V3000 MOL files are not supported; convert to V2000 or XYZ")
    try:
        n = int(counts[:3])
    except ValueError:
        raise StructureError(f"bad MOL counts line: {counts!r}")
    symbols, positions = [], []
    for k in range(n):
        if 4 + k >= len(lines):
            raise StructureError("MOL atom block truncated")
        parts = lines[4 + k].split()
        if len(parts) < 4:
            raise StructureError(f"bad MOL atom line: {lines[4 + k]!r}")
        positions.append([float(parts[0]), float(parts[1]), float(parts[2])])
        symbols.append(normalize_symbol(parts[3]))
    if not symbols:
        raise StructureError("no atoms found")
    return [_frame(symbols, positions, lines[0].strip())]


def parse_mol2(text: str) -> List[Dict]:
    """Parse the ``@<TRIPOS>ATOM`` section of a MOL2 string."""
    symbols, positions = [], []
    in_atoms = False
    name = ""
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        s = line.strip()
        if s.upper().startswith("@<TRIPOS>MOLECULE") and idx + 1 < len(lines):
            name = lines[idx + 1].strip()
        if s.startswith("@<TRIPOS>"):
            if in_atoms:
                break
            in_atoms = s.upper() == "@<TRIPOS>ATOM"
            continue
        if not in_atoms or not s:
            continue
        parts = s.split()
        if len(parts) < 6:
            raise StructureError(f"bad MOL2 atom line: {line!r}")
        positions.append([float(parts[2]), float(parts[3]), float(parts[4])])
        symbols.append(normalize_symbol(parts[5].split(".")[0]))
    if not symbols:
        raise StructureError("no @<TRIPOS>ATOM records found")
    return [_frame(symbols, positions, name)]


def _parse_with_ase(path: Path) -> List[Dict]:
    try:
        from ase import io
    except ImportError:
        raise StructureError(f"unsupported format {path.suffix!r} (install ASE for more formats)")
    try:
        images = io.read(str(path), index=":")
    except Exception as exc:  # ASE raises many exception types
        raise StructureError(f"ASE could not read the file: {exc}")
    return [_frame(a.get_chemical_symbols(), a.get_positions().tolist(), "") for a in images]


_PARSERS = {
    ".xyz": parse_xyz,
    ".extxyz": parse_xyz,
    ".pdb": parse_pdb,
    ".sdf": parse_sdf,
    ".mol": parse_sdf,
    ".mol2": parse_mol2,
}


def read_structure_file(path) -> List[Dict]:
    """Read a structure file into a list of frames.

    Args:
        path: File path; the format is chosen from the extension.

    Returns:
        List of frames (at least one).

    Raises:
        StructureError: If the file cannot be parsed.
    """
    path = Path(path)
    parser = _PARSERS.get(path.suffix.lower())
    if parser is None:
        return _parse_with_ase(path)
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        raise StructureError(f"cannot read file: {exc}")
    return parser(text)


def frames_to_xyz(frames: List[Dict]) -> str:
    """Serialise frames to a multi-frame XYZ string."""
    out = []
    for fr in frames:
        out.append(str(len(fr["symbols"])))
        out.append(fr.get("comment", "").replace("\n", " "))
        for s, (x, y, z) in zip(fr["symbols"], fr["positions"]):
            out.append(f"{s:<2s} {x:14.8f} {y:14.8f} {z:14.8f}")
    return "\n".join(out) + "\n"


def formula(symbols: List[str]) -> str:
    """Hill-order chemical formula (C first, then H, then alphabetical)."""
    counts = Counter(symbols)
    order = []
    if "C" in counts:
        order += ["C"] + (["H"] if "H" in counts else [])
    order += sorted(s for s in counts if s not in order)
    return "".join(f"{s}{counts[s] if counts[s] > 1 else ''}" for s in order)


def parse_energy(comment: str):
    """Extract an energy (kcal/mol) from a MARS XYZ comment line, or ``None``.

    MARS ensembles use ``"Structure i, Energy = E kcal/mol"``; eV values are
    converted to kcal/mol.
    """
    m = _ENERGY_RE.search(comment or "")
    if not m:
        return None
    value = float(m.group(1))
    if (m.group(2) or "").lower() == "ev":
        value *= 23.060547830619
    return value


def summarize(frames: List[Dict]) -> Dict:
    """Return a JSON-friendly summary of parsed frames for the GUI."""
    first = frames[0]
    return {
        "n_atoms": len(first["symbols"]),
        "n_frames": len(frames),
        "formula": formula(first["symbols"]),
        "elements": sorted(set(first["symbols"])),
        "energies": [parse_energy(f.get("comment", "")) for f in frames],
    }
