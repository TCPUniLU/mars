# Classical force fields

MARS runs AMBER/GAFF, CHARMM, OPLS-AA and AMOEBA through every workflow
(sampling, optimisation, IR, MD), with analytical forces and Hessians in JAX.

!!! info "You bring the parameters"
    MARS ships **no force-field parameters**. You give it your own parameter
    and topology files. [OpenMM](https://openmm.org) and
    [ParmEd](https://parmed.github.io/ParmEd/html/index.html) read them, and
    MARS evaluates every term in JAX.

```bash
pip install "mars[mm]"        # openmm, parmed
```

## Inputs

=== "AMBER / GAFF"

    ```bash
    mars optimize ala2.xyz --potential amber --prmtop ala2.prmtop
    mars ir ligand.xyz --potential amber --prmtop lig.prmtop --ff-coords lig.inpcrd
    ```

    | Option | File |
    |--------|------|
    | `--prmtop` | Amber parameter-topology file (tleap / antechamber / ParmEd) |
    | `--ff-coords` *(optional)* | `inpcrd` / `rst7` / PDB matching the prmtop |

=== "CHARMM"

    ```bash
    mars optimize pep.pdb --potential charmm --psf pep.psf \
         --ff-params par_all36m_prot.prm top_all36_prot.rtf toppar_water_ions.str
    ```

    | Option | File |
    |--------|------|
    | `--psf` | CHARMM / X-PLOR PSF |
    | `--ff-params` | Every `.prm` / `.par` / `.rtf` / `.top` / `.str` the PSF needs |
    | `--ff-coords` *(optional)* | `crd` or PDB |

    Urey-Bradley terms, harmonic impropers, CMAP and NBFIX are all supported.

=== "OPLS-AA"

    ```bash
    mars optimize mol.gro --potential opls --top mol.top --gmx-include /path/to/gromacs/top
    mars optimize mol.pdb --potential opls --ff-xml mol_opls.xml     # e.g. LigParGen OpenMM XML
    ```

    | Option | File |
    |--------|------|
    | `--top` (+ `--gmx-include`) | GROMACS topology; `#include` files are looked up in `--gmx-include` (default `$GMXDATA/top`) |
    | `--ff-xml` (+ `--ff-pdb`) | OpenMM ForceField XML, plus a PDB with matching residue and atom names |

    Ryckaert-Bellemans torsions and GROMACS combination rule 3 (geometric
    mixing) are supported.

=== "AMOEBA"

    ```bash
    mars ir water3.pdb --potential amoeba --amoeba-polarization mutual
    mars ir mol.xyz --potential amoeba --tinker-key mol.key amoebabio18.prm    # Tinker files
    ```

    | Option | File / value |
    |--------|--------------|
    | `--ff-xml` | AMOEBA XML (default `amoeba2018.xml`, bundled with OpenMM) + a PDB input |
    | `--tinker-key` | Tinker `.key` / `.prm`, with a Tinker `.xyz` input (OpenMM ≥ 8.3) |
    | `--amoeba-polarization` | `mutual` *(default; exact solve)*, `direct`, `extrapolated` (OPT) |

=== "Any OpenMM System"

    ```bash
    mars optimize mol.pdb --potential mm --ff-xml my_forcefield.xml
    ```

    Any OpenMM `ForceField` XML works, as long as it uses only the forces
    listed below.

By default the geometry is the MARS input structure, and its atom order must
match the topology exactly. MARS checks this, see
[Error messages](#error-messages). Use `--ff-coords` to take the coordinates
that belong to the parameter files instead.

From Python:

```python
from mars.forcefields import load_charmm
from mars.potentials import get_potential

system = load_charmm("pep.psf", ["par_all36m_prot.prm", "top_all36_prot.rtf"], "pep.crd")
pot = get_potential("mm", system=system)
pot.initialize(system.positions)
energy_fn = pot.build_energy_fn()            # positions in Å -> eV
pot.energy_components(system.positions)      # per-term energies (eV)
```

## Charge, spin and IR

- **Charge** comes from the parameters. `--charge` is only checked against
  the force field's net charge.
- **Spin**: force fields are closed-shell, so `--spin` must be 1.
- **IR intensities** use the fixed partial charges. For AMOEBA the molecular
  dipole also includes the permanent atomic dipoles (lab frame) and the
  induced dipoles, μ = Σ (qᵢrᵢ + dᵢ + μᵢⁱⁿᵈ). It matches OpenMM's
  `getSystemMultipoleMoments` to 10⁻⁶ D.

## Functional forms

Each term is evaluated by one of three engines:

- **jax-md**: [jax-md](https://github.com/jax-md/jax-md) `mm_forcefields`.
- **Lepton**: MARS's compiler for OpenMM's custom-force expressions. It covers
  any `Custom*Force` generically.
- **MARS native**: written in JAX in MARS.

=== "AMBER / GAFF"

    | Term | Engine |
    |------|--------|
    | Harmonic bonds | jax-md |
    | Harmonic angles | MARS native (jax-md's angle helper loses accuracy near 180°) |
    | Periodic proper and improper torsions | jax-md |
    | LJ 12-6 (Lorentz-Berthelot) + Coulomb, scaled 1-4 pairs | jax-md |

=== "CHARMM"

    | Term | Engine |
    |------|--------|
    | Bonds, Urey-Bradley, torsions, CMAP | jax-md |
    | Angles | MARS native |
    | Harmonic impropers | Lepton |
    | NBFIX pair tables, special 1-4 LJ | Lepton |
    | Coulomb | jax-md |

=== "OPLS-AA"

    | Term | Engine |
    |------|--------|
    | Bonds, Fourier torsions, Coulomb | jax-md |
    | Angles, Ryckaert-Bellemans torsions | MARS native |
    | Geometric-mean LJ and its 1-4 pairs (GROMACS rule 3) | Lepton |

=== "AMOEBA"

    | Term | Engine |
    |------|--------|
    | Quartic bonds, sextic angles, in-plane angles, out-of-plane bends, stretch-bend, stretch-torsion, angle-torsion, pi-torsion | Lepton |
    | Urey-Bradley, torsions | jax-md |
    | Torsion-torsion (bicubic grids) | MARS native |
    | Buffered 14-7 vdW with H reduction | MARS native |
    | Permanent multipoles in local frames (charge, dipole, quadrupole) | MARS native |
    | Thole-damped polarization: mutual, direct, extrapolated (OPT) | MARS native |

An OpenMM force that none of these handles raises an error; no term is
ever silently dropped.

## Error messages

MARS checks the inputs before any energy is computed, and stops with a
message that says what is wrong and how to fix it:

| Situation | What MARS reports |
|-----------|-------------------|
| A file is missing | `CHARMM PSF file (--psf) not found: x.psf` |
| A file is not in the expected format, or is truncated | `Could not build the AMBER system ... Hint: the file is not in the expected format` |
| Incomplete parameters (an atom, bond, angle or dihedral type is not defined) | `MissingParameter: Missing bond type for <Bond ...>`, `no parameters for type 'ZZ'`, plus a hint to add the `.str` / `.prm` / include files |
| PDB residue or atom names that match no XML template | `No template found for residue 0 (LIG) ...`, plus a hint about hydrogens, residue names or ligand XML |
| A topology bond with no bond-stretching term | `N bonds of the topology have no bond-stretching parameters` |
| Non-finite parameters, r₀ ≤ 0, LJ σ ≤ 0 with ε > 0, zero masses | `Invalid force-field parameters: ...` (the first 10 problems) |
| Negative force constants (intended in some force fields, e.g. AMOEBA water) | warning only |
| Non-integer net charge | warning |
| The structure has a different number of atoms | `expected (22, 3) ... Missing hydrogens or extra solvent/ions` |
| A different atom order (e.g. an XYZ not in PSF/prmtop order) | `Atom order differs from the force-field topology`, or `bonds ... stretched beyond twice their equilibrium length` |
| Coordinates in nm instead of Å | `the coordinates look like nm ... Multiply them by 10` |
| Overlapping atoms, or NaN coordinates | `Atoms ... overlap (0.17 A apart)` / `NaN or infinite coordinates` |
| Non-finite initial energy or forces | names the offending term |
| Very large initial forces (> 10⁴ kcal/mol/Å) | warning: minimise before MD |
| Cutoffs, PBC, constraints, virtual sites, implicit solvent | explains the free-space-only scope and how to rebuild the system |
| `--spin` ≠ 1, or `--charge` that contradicts the parameters | `closed-shell` / `does not match the force-field topology` |

## Scope and limitations

??? note "Free space only"
    All nonbonded sums are all-pairs with no cutoff, and there is no
    periodic box, Ewald or PME, which suits molecules and clusters up to a few
    thousand atoms. Implicit solvent (GB, AMOEBA GK) is not supported.

??? note "No constraints or virtual sites"
    Systems are built with `constraints=None`, so X–H bonds are flexible;
    use a 0.5 fs MD time step. TIP4P-style extra points are rejected.

## Validation

Every family is checked against OpenMM's Reference platform, per force and
in total. In spot checks:

| Family | Test system | Largest force error (kcal/mol/Å) |
|--------|-------------|----------------------------------|
| AMBER (amber14) | Alanine dipeptide | 1.3×10⁻⁷ |
| CHARMM36 (incl. CMAP, NBFIX) | Trp-cage | 2×10⁻⁵ |
| OPLS (RB torsions, rule 3) | Butane | 7×10⁻⁹ |
| AMOEBA, mutual / direct / extrapolated | Alanine dipeptide | 1.6×10⁻⁹ / 7×10⁻¹⁰ / 7×10⁻¹⁰ |


Full results, including MD stability, are in
[Validation and credits](credits.md).
