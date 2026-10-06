# Choosing a potential

Every MARS command takes `--potential NAME`. Pick a row, then follow the
link for installation and options.

| Potential | Kind | Charge | Spin | Best for | IR intensities | Install |
|-----------|------|:------:|:----:|----------|:--------------:|---------|
| [`so3lr`](ml.md) *(default)* | ML | ✅ | v2 models only | Organic and biomolecules, long-range interactions | ✅ | source |
| [`mace`](ml.md) | ML | OMol only | OMol only | Organic molecules (OFF), materials (MP), broad chemistry (OMol) | — | `[mace]` |
| [`uma`](ml.md) | ML | ✅ | ✅ | Broad chemistry (`omol`), materials (`omat`) | — | `[uma]` |
| [`orb`](ml.md) | ML | ✅ | ✅ | Broad molecular chemistry (OMol25) | — | `[jaxmd-ml]` |
| [`aimnet2`](ml.md) | ML | ✅ | — | Neutral and charged closed-shell molecules | ✅ | `[jaxmd-ml]` |
| [`aceff`](ml.md) | ML | ✅ | — | Drug-like small molecules | ✅ (effective charges) | `[jaxmd-ml]` |
| [`ani`](ml.md) | ML | — | — | Neutral H, C, N, O, F, S, Cl | — | `[jaxmd-ml]` |
| [`dxtb`](other.md#dxtb) | Tight binding | ✅ | ✅ | Any element; GFN1/GFN2-xTB | ✅ (FD Hessian) | `[dxtb]` |
| [`amber`](forcefields.md) | Force field | from parameters | — | Your AMBER/GAFF `prmtop` | ✅ | `[mm]` |
| [`charmm`](forcefields.md) | Force field | from parameters | — | Your CHARMM PSF and parameters | ✅ | `[mm]` |
| [`opls`](forcefields.md) | Force field | from parameters | — | Your GROMACS `.top` or OpenMM XML | ✅ | `[mm]` |
| [`amoeba`](forcefields.md) | Polarizable force field | from parameters | — | AMOEBA (amoeba2018 or Tinker files) | ✅ (multipoles + induced dipoles) | `[mm]` |
| [`harmonic`, `lj`, `valence`, `confinement`](other.md#test-potentials) | Test / helper | — | — | Tests, CI, NCI mode | — | none |

Notes on the columns:

- **Charge and spin**: `--charge Q` and `--spin M`, where M is the
  multiplicity 2S+1 (1 = singlet, 2 = doublet, ...). MARS converts M to each
  model's own convention. A potential with no spin input refuses `--spin` ≠ 1
  rather than silently ignoring it. See
  [Charge and spin](ml.md#charge-and-spin).
- **IR intensities** need the molecular dipole, which comes from the
  potential's partial charges (and, for AMOEBA, its atomic multipoles).
  Without them MARS computes frequencies only. See
  [IR intensities](ml.md#ir-intensities).
- **Analytical Hessians** are available for every potential except dxtb and
  MACE-OMol on its PyTorch backend; those fall back to finite differences.

## Quick examples

```bash
mars input.xyz                                            # SO3LR (default)
mars input.xyz --potential mace --mace-foundation off --mace-model small
mars input.xyz --potential uma --uma-task omol --charge -1
mars input.xyz --potential orb --spin 2                    # doublet radical
mars input.xyz --so3lr-model so3lr-2-m --spin 3            # triplet with SO3LR v2
mars optimize ala2.xyz --potential amber --prmtop ala2.prmtop
mars ir water3.pdb --potential amoeba --amoeba-polarization mutual
```

## In this section

- [ML potentials](ml.md): SO3LR, MACE, UMA, ORB, AIMNet2, AceFF and ANI, with
  backends, weights, and charge and spin.
- [Classical force fields](forcefields.md): AMBER, CHARMM, OPLS-AA, AMOEBA;
  input files, terms and error messages.
- [Other potentials](other.md): dxtb and the test potentials.
- [Custom potentials](custom.md): plug in your own model.
- [Validation and credits](credits.md): how each potential is checked, and who
  implemented it.
