# Validation and credits

## How potentials are validated

Many of the JAX ports MARS relies on are new, and their upstream tests are
thin. So every potential is re-validated in MARS, using the suite in
`validation/` at the repository root (outside the package and CI):

- **ML parity**: energies and forces on a curated set of molecules (neutral,
  charged, open-shell, peptides, non-covalent dimers), compared with the
  original package's ASE calculator in float64.
- **Force-field parity**: energies per force and in total, and forces,
  against OpenMM's Reference platform.
- **MD stability**: NVE energy drift, plus NVT temperature and topology
  preservation, for every potential.

Until a model passes, MARS prints a one-time "experimental" warning when it
loads. The current results are in `validation/README.md`.

| Potential | Upstream validation (jax-md) | MARS validation |
|-----------|------------------------------|-----------------|
| `uma` | Layer parity; energy-only end-to-end check at atol 0.05 | vs FAIR-Chem (needs the gated weights) |
| `orb`, `ani`, `aimnet2`, `aceff`, `so3lr_jaxmd` | float64 parity on one molecule (toluene) | vs orb-models, TorchANI, aimnet, torchmd-net, so3lr |
| `mace --mace-backend jaxmd` | Synthetic models only | vs mace-torch |
| `amber`, `charmm`, `opls` | vs OpenMM | vs OpenMM (see [Classical force fields](forcefields.md#validation)) |
| `amoeba` | not upstream | vs OpenMM, all three polarization modes |

## Implementation credits

| Potential | Implemented by (jax-md PRs) | Original model code |
|-----------|-----------------------------|---------------------|
| `uma` | #376 @ekurtulus, @abhijeetgangan; #394, #400, #401 @abhijeetgangan; #399 @teddykoker | [FAIR-Chem](https://github.com/FAIR-Chem/fairchem) |
| `orb`, `ani`, `aimnet2`, `aceff`, `--so3lr-backend jaxmd` | #411 @mitkotak, @abhijeetgangan (ported from [bio-mlff](https://github.com/mitkotak/bio-mlff)); #416, #419 @abhijeetgangan | [orb-models](https://github.com/orbital-materials/orb-models), [TorchANI](https://github.com/aiqm/torchani), [aimnetcentral](https://github.com/isayevlab/aimnetcentral), [torchmd-net](https://github.com/torchmd/torchmd-net), [so3lr](https://github.com/general-molecular-simulations/so3lr) |
| `--mace-backend jaxmd` | #390 @DRosen285, @abhijeetgangan, @beitom; #418 @xyf5432 | [mace](https://github.com/ACEsuit/mace), [mace-jax](https://github.com/ACEsuit/mace-jax) |
| `amber`, `charmm`, `opls` | #319 @alecbetancourt; #350 @DRosen285, @beitom; @abhijeetgangan | Parameters read by [OpenMM](https://openmm.org) and [ParmEd](https://parmed.github.io/ParmEd/html/index.html) |
| `amoeba` | MARS (native JAX) | [OpenMM](https://openmm.org), [Tinker](https://dasher.wustl.edu/tinker/) |

The MARS command line prints the implementation and the articles to cite each
time a potential is loaded. The BibTeX entries are in
[Citation](../../about/citation.md).
