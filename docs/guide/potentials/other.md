# Other potentials

## dxtb

[dxtb](https://github.com/grimme-lab/dxtb) is a differentiable PyTorch
implementation of GFN1-xTB and GFN2-xTB. It covers any element, and is
charge- and spin-aware: MARS passes the multiplicity as unpaired electrons
(M − 1). Its Mulliken charges drive IR intensities. Hessians use finite
differences, because dxtb runs through a PyTorch callback.

```bash
mars input.xyz --potential dxtb --dxtb-method gfn2 --charge 0 --spin 1
```

**Install:** `pip install "dxtb[libcint]>=0.4.0"`. See
[Dependency conflicts](../../install.md#dependency-conflicts-dxtb-vs-the-jax-stack).

## Test potentials

These need no ML dependencies, which makes them useful for tests, CI and
trying out workflows.

| Name | What it is |
|------|------------|
| `harmonic` | Spring to the centroid |
| `lj` | Lennard-Jones pair potential (`epsilon`, `sigma`, `cutoff`) |
| `valence` | Generic bonds, angles, torsions and soft LJ, from detected bonds |
| `confinement` | Spherical or ellipsoidal wall, used by [NCI mode](../nci_mode.md) |

```bash
mars optimize input.xyz --potential harmonic
```
