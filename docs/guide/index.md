# User guide

Conceptual overview of each MARS workflow. Each page links to the
matching copy-pasteable [recipes](../examples/index.md) and the underlying
[API reference](../api/index.md). Every workflow can also be run from the
[graphical interface](../gui/index.md) (`mars gui`), whose task forms carry
the same options as the subcommands below.

| Page | Subcommand | What it covers |
|------|------------|----------------|
| [Conformational search](conformer_search.md) | `mars` | The default mode — full sampler, MTD-only, MD-only |
| [IR spectroscopy](ir_spectroscopy.md)        | `mars ir`        | Analytical Hessian, finite differences, MD-based dipole ACF |
| [Optimization](optimization.md)              | `mars optimize`  | LBFGS / FIRE / GD minimization and stationary-point search |
| [Explicit solvation](solvation.md)           | `mars solvation` | Layer-by-layer shell construction with optional relaxation |
| [NCI mode](nci_mode.md)                      | flag             | Spherical confinement for host–guest / multi-fragment inputs |
| [Potentials](potentials/index.md)                  | flag             | SO3LR, MACE, UMA, ORB, AIMNet2, AceFF, ANI, dxtb, AMBER/CHARMM/OPLS/AMOEBA, custom |
| [Logging](logging.md)                        | flag             | Console / file output, timing tracker, debug mode |
