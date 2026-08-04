#!/usr/bin/env python3
"""Regenerate MARS solvent geometries with Open Babel.

For every entry in ``mars/solvents/manifest.toml`` that has a ``smiles`` field
and whose ``source`` is **not** ``"experimental"``, this rebuilds
``mars/solvents/library/<name>.xyz`` from the SMILES using
``obabel -:"<smiles>" -oxyz --gen3d`` (MMFF94 cleanup). Experimental
geometries are curated and left untouched.

Standalone — depends only on the Python stdlib and the ``obabel`` executable
(Open Babel ≥ 3.0), not on the ``mars`` package.

Usage::

    python tools/regen_solvents.py            # non-experimental entries
    python tools/regen_solvents.py --all      # every entry with a SMILES
    python tools/regen_solvents.py --only thf dmf   # selected names
    python tools/regen_solvents.py --dry-run  # show what would change
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST = _REPO_ROOT / "mars" / "solvents" / "manifest.toml"
_LIBRARY = _REPO_ROOT / "mars" / "solvents" / "library"
_GEN_SOURCE = "obabel --gen3d (MMFF94)"


def _obabel_xyz(smiles: str) -> list[str]:
    """Return the atom lines (``symbol x y z``) for *smiles* via obabel."""
    proc = subprocess.run(
        ["obabel", f"-:{smiles}", "-oxyz", "--gen3d"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"obabel failed for SMILES {smiles!r}:\n{proc.stderr}")
    lines = proc.stdout.splitlines()
    if len(lines) < 3:
        raise RuntimeError(f"obabel produced no atoms for SMILES {smiles!r}")
    n = int(lines[0].strip())
    atom_lines = [ln for ln in lines[2 : 2 + n] if ln.strip()]
    if len(atom_lines) != n:
        raise RuntimeError(f"obabel atom count mismatch for SMILES {smiles!r}")
    return atom_lines


def _write_xyz(name: str, formula: str, atom_lines: list[str]) -> None:
    comment = f"{name} ({formula}) — {_GEN_SOURCE}"
    out = _LIBRARY / f"{name}.xyz"
    with open(out, "w") as f:
        f.write(f"{len(atom_lines)}\n{comment}\n")
        for ln in atom_lines:
            f.write(ln.rstrip() + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="regenerate every entry with a SMILES")
    ap.add_argument("--only", nargs="+", metavar="NAME", help="regenerate only these names")
    ap.add_argument("--dry-run", action="store_true", help="list targets, do not write files")
    args = ap.parse_args(argv)

    if shutil.which("obabel") is None:
        print("error: 'obabel' not found on PATH (install Open Babel >= 3.0).", file=sys.stderr)
        return 2

    with open(_MANIFEST, "rb") as f:
        manifest = tomllib.load(f)

    regenerated, skipped = [], []
    for name, meta in manifest.items():
        smiles = meta.get("smiles", "")
        source = meta.get("source", "")
        if args.only and name not in args.only:
            continue
        if not smiles:
            skipped.append((name, "no smiles"))
            continue
        if not args.all and not args.only and source == "experimental":
            skipped.append((name, "experimental (kept)"))
            continue
        if args.dry_run:
            print(f"would regenerate {name:22s} from {smiles!r}")
            regenerated.append(name)
            continue
        atom_lines = _obabel_xyz(smiles)
        _write_xyz(name, meta.get("formula", ""), atom_lines)
        print(f"regenerated {name:22s} ({len(atom_lines)} atoms) from {smiles!r}")
        regenerated.append(name)

    print(f"\n{len(regenerated)} regenerated, {len(skipped)} skipped.")
    for name, why in skipped:
        print(f"  skip {name:22s} — {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
