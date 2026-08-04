"""
Solvent library for MARS.

Bundles a collection of pre-built solvent geometries that the
:mod:`mars.solvation` shell builder draws from. Each solvent is stored as a
plain XYZ file under ``mars/solvents/library/`` together with a
``manifest.toml`` carrying metadata (formula, density, dielectric, aliases).

Public API
----------

.. autofunction:: get_solvent
.. autofunction:: list_solvents
.. autofunction:: register_solvent
.. autofunction:: get_solvent_metadata

The library is loaded lazily on first access so importing this module is
cheap.

Adding a new solvent at runtime
-------------------------------

    >>> from mars.solvents import register_solvent
    >>> register_solvent("propylene_carbonate", "/path/to/pc.xyz",
    ...                  formula="C4H6O3", density=1.21, dielectric=66.1)
    >>> from mars.solvents import get_solvent
    >>> data = get_solvent("propylene_carbonate")
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_LIBRARY_DIR = Path(__file__).parent / "library"
_MANIFEST_PATH = Path(__file__).parent / "manifest.toml"

# Lazy state — populated on first access.
_SOLVENTS: Optional[Dict[str, Dict]] = None
_ALIAS_MAP: Optional[Dict[str, str]] = None


def _load_manifest() -> Dict[str, Dict]:
    """Parse the manifest TOML using the stdlib parser when available."""
    if not _MANIFEST_PATH.exists():
        return {}
    try:
        import tomllib  # Python 3.11+

        with open(_MANIFEST_PATH, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        try:
            import tomli  # backport

            with open(_MANIFEST_PATH, "rb") as f:
                return tomli.load(f)
        except ImportError:
            return _load_manifest_fallback()


def _load_manifest_fallback() -> Dict[str, Dict]:
    """Minimal TOML reader for the small manifest schema we ship.

    Handles only the subset used by ``manifest.toml``: top-level tables
    of the form ``[name]`` followed by ``key = value`` lines with
    string, float, or list-of-strings values. Used when neither
    ``tomllib`` (Python 3.11+) nor ``tomli`` is available.
    """
    data: Dict[str, Dict] = {}
    current: Optional[str] = None
    with open(_MANIFEST_PATH, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1].strip()
                data[current] = {}
                continue
            if "=" not in line or current is None:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.split("#", 1)[0].strip()
            if value.startswith("[") and value.endswith("]"):
                items = value[1:-1].split(",")
                data[current][key] = [s.strip().strip('"') for s in items if s.strip()]
            elif value.startswith('"') and value.endswith('"'):
                data[current][key] = value[1:-1]
            else:
                try:
                    data[current][key] = float(value)
                except ValueError:
                    data[current][key] = value
    return data


def _load_solvent_xyz(path: Path) -> Dict:
    """Read an XYZ file and return ``{symbols, positions}`` arrays."""
    with open(path, "r") as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    symbols: List[str] = []
    positions = []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        symbols.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return {"symbols": symbols, "positions": np.array(positions)}


def _ensure_loaded() -> None:
    """Build the solvent registry lazily on first access."""
    global _SOLVENTS, _ALIAS_MAP
    if _SOLVENTS is not None:
        return

    manifest = _load_manifest()
    solvents: Dict[str, Dict] = {}
    aliases: Dict[str, str] = {}

    for xyz_path in sorted(_LIBRARY_DIR.glob("*.xyz")):
        name = xyz_path.stem
        meta = manifest.get(name, {})
        entry = _load_solvent_xyz(xyz_path)
        entry.update(
            {
                "formula": meta.get("formula", ""),
                "smiles": meta.get("smiles", ""),
                "density": meta.get("density", float("nan")),
                "dielectric": meta.get("dielectric", float("nan")),
                "source": meta.get("source", ""),
                "aliases": list(meta.get("aliases", [])),
            }
        )
        solvents[name] = entry
        aliases[name.lower()] = name
        for alias in entry["aliases"]:
            aliases[alias.lower()] = name

    _SOLVENTS = solvents
    _ALIAS_MAP = aliases


def list_solvents() -> List[str]:
    """Return a sorted list of canonical solvent names available in the library."""
    _ensure_loaded()
    assert _SOLVENTS is not None
    return sorted(_SOLVENTS.keys())


def _resolve_name(name: str) -> str:
    """Resolve an alias or canonical name; raise KeyError on unknown input."""
    _ensure_loaded()
    assert _ALIAS_MAP is not None and _SOLVENTS is not None
    if name in _SOLVENTS:
        return name
    canonical = _ALIAS_MAP.get(name.lower())
    if canonical is None:
        available = ", ".join(sorted(_SOLVENTS.keys()))
        raise KeyError(f"Unknown solvent '{name}'. Available: {available}")
    return canonical


def get_solvent(name: str) -> Dict:
    """Return the solvent entry for *name* (resolving aliases case-insensitively).

    The returned dict has at minimum ``symbols`` (list[str]) and ``positions``
    (numpy float array of shape ``(N, 3)``), plus the metadata fields
    populated from the manifest (``formula``, ``density``, ``dielectric``,
    ``source``, ``aliases``).
    """
    canonical = _resolve_name(name)
    return _SOLVENTS[canonical]


def get_solvent_metadata(name: str) -> Dict:
    """Return only the metadata fields for *name* (no positions/symbols)."""
    entry = get_solvent(name)
    return {
        k: entry[k] for k in ("formula", "smiles", "density", "dielectric", "source", "aliases")
    }


def load_solvent(name_or_path: str) -> Dict:
    """Resolve a solvent from either the built-in library or a user XYZ file.

    If *name_or_path* points to an existing file (or ends in ``.xyz``), it is
    read as a custom solvent geometry and returned as a solvent entry dict
    (``symbols`` + ``positions`` plus best-effort metadata, with
    ``source="user-file"``). Otherwise it is resolved against the built-in
    library by canonical name or alias (case-insensitive), exactly like
    :func:`get_solvent`.

    This is the single entry point the CLI uses for ``--solvent``, so users can
    pass either a library name (``water``, ``EtOH``, …) or a path to their own
    ``.xyz`` (``./my_solvent.xyz``).

    Args:
        name_or_path: A library name/alias, or a path to an ``.xyz`` file.

    Returns:
        Solvent entry dict with at least ``symbols`` and ``positions``.

    Raises:
        FileNotFoundError: if *name_or_path* looks like a path but does not exist.
        KeyError: if it is neither a file nor a known library name/alias.
    """
    candidate = Path(name_or_path)
    looks_like_path = (
        candidate.suffix.lower() == ".xyz"
        or candidate.exists()
        or any(sep in name_or_path for sep in ("/", "\\"))
    )
    if looks_like_path:
        if not candidate.exists():
            raise FileNotFoundError(f"Solvent file not found: {name_or_path}")
        entry = _load_solvent_xyz(candidate)
        entry.update(
            {
                "formula": "",
                "smiles": "",
                "density": float("nan"),
                "dielectric": float("nan"),
                "source": "user-file",
                "aliases": [],
            }
        )
        return entry
    return get_solvent(name_or_path)


def register_solvent(name: str, xyz_path: str, **metadata) -> None:
    """Register a user-supplied solvent geometry at runtime.

    The XYZ file is read immediately. ``metadata`` may include any of
    ``aliases`` (list of strings), ``formula``, ``density``, ``dielectric``,
    ``source``. Aliases collide-check against existing entries.
    """
    _ensure_loaded()
    assert _SOLVENTS is not None and _ALIAS_MAP is not None
    if name in _SOLVENTS:
        raise ValueError(f"Solvent '{name}' is already registered.")
    entry = _load_solvent_xyz(Path(xyz_path))
    entry.update(
        {
            "formula": metadata.get("formula", ""),
            "smiles": metadata.get("smiles", ""),
            "density": metadata.get("density", float("nan")),
            "dielectric": metadata.get("dielectric", float("nan")),
            "source": metadata.get("source", "user-registered"),
            "aliases": list(metadata.get("aliases", [])),
        }
    )
    _SOLVENTS[name] = entry
    _ALIAS_MAP[name.lower()] = name
    for alias in entry["aliases"]:
        if alias.lower() in _ALIAS_MAP and _ALIAS_MAP[alias.lower()] != name:
            raise ValueError(f"Alias '{alias}' already maps to '{_ALIAS_MAP[alias.lower()]}'.")
        _ALIAS_MAP[alias.lower()] = name


class _SolventLibraryProxy:
    """Read-only mapping that proxies into the lazy registry.

    Exposed as ``mars.solvents.SOLVENT_LIBRARY`` and as the legacy
    ``mars.solvation.SOLVENT_DB`` alias for backwards compatibility.
    """

    def __getitem__(self, key):
        return get_solvent(key)

    def __contains__(self, key):
        try:
            _resolve_name(key)
            return True
        except KeyError:
            return False

    def __iter__(self):
        return iter(list_solvents())

    def keys(self):
        return list_solvents()

    def __len__(self):
        return len(list_solvents())


SOLVENT_LIBRARY = _SolventLibraryProxy()


__all__ = [
    "SOLVENT_LIBRARY",
    "get_solvent",
    "load_solvent",
    "list_solvents",
    "register_solvent",
    "get_solvent_metadata",
]
