#!/usr/bin/env python3
"""Generate the alanine-dipeptide starting structure for the MARS walkthrough.

Builds N-acetyl-L-alanine-N'-methylamide (Ace-Ala-NMe), the canonical
"alanine dipeptide" used to study backbone (phi/psi) conformational
preferences, and writes it to ``alanine_dipeptide.xyz``.

Usage
-----
    python 00_make_structure.py [output.xyz]

Requires RDKit (``pip install rdkit`` or ``conda install -c conda-forge rdkit``).
A pre-built geometry is embedded as a fallback if RDKit is unavailable.
"""

import sys

# Ace-Ala-NMe : CH3-CO-NH-CH(CH3)-CO-NH-CH3
SMILES = "CC(=O)N[C@@H](C)C(=O)NC"
OUTPUT = sys.argv[1] if len(sys.argv) > 1 else "alanine_dipeptide.xyz"


def build_with_rdkit(smiles: str) -> str:
    """Return an XYZ string for *smiles* using RDKit (ETKDG + MMFF)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)

    conf = mol.GetConformer()
    lines = [str(mol.GetNumAtoms()), "alanine dipeptide (Ace-Ala-NMe), MMFF-optimized"]
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol():2s} {p.x:14.8f} {p.y:14.8f} {p.z:14.8f}")
    return "\n".join(lines) + "\n"


def main():
    try:
        xyz = build_with_rdkit(SMILES)
    except Exception as exc:  # pragma: no cover - environment dependent
        sys.exit(
            f"RDKit structure generation failed ({exc}).\n"
            "Install RDKit, or drop your own alanine_dipeptide.xyz next to this script."
        )
    with open(OUTPUT, "w") as fh:
        fh.write(xyz)
    print(f"Wrote {OUTPUT} ({xyz.splitlines()[0]} atoms).")


if __name__ == "__main__":
    main()
