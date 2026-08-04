#!/usr/bin/env python3
"""Extract the first (lowest-energy) frame of a multi-frame XYZ ensemble.

MARS writes conformer ensembles sorted by energy, so frame 0 is the global
minimum. Used by the walkthrough to render / re-use the best conformer.

Usage
-----
    python lowest_conformer.py ensemble.xyz best.xyz
"""

import sys


def first_frame(in_path: str, out_path: str) -> None:
    with open(in_path) as fh:
        lines = fh.readlines()
    n_atoms = int(lines[0].split()[0])
    block = lines[: n_atoms + 2]  # count line + comment + n_atoms coordinate lines
    with open(out_path, "w") as fh:
        fh.writelines(block)
    print(f"Wrote {out_path} (lowest-energy conformer, {n_atoms} atoms).")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "conformers.xyz"
    dst = sys.argv[2] if len(sys.argv) > 2 else "best.xyz"
    first_frame(src, dst)
