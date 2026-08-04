# API reference

Auto-generated reference for every public module. Each page renders the
module's docstrings via [`mkdocstrings`](https://mkdocstrings.github.io/),
so the source files in `mars/` are the single point of truth.

## Module map

| Module | Purpose |
|--------|---------|
| [`mars.utils`](utils.md) | Periodic table, structure I/O, bond detection, JAX helpers |
| [`mars.potentials`](potentials.md) | Potential registry (SO3LR, MACE, dxtb, harmonic, LJ, confinement) and `PotentialWrapper` ABC |
| [`mars.ir`](ir.md) | IR spectrum (Hessian / MD-based), normal modes, peak allocation |
| [`mars.solvation`](solvation.md) | Layer-by-layer solvation shell builder |
| [`mars.solvents`](solvents.md) | Built-in solvent library + runtime registration |
| [`mars.conf_sampling`](conf_sampling.md) | Top-level conformational search workflows |
| [`mars.optimizer`](optimizer.md) | LBFGS / FIRE / GD / stationary-point optimization |
| [`mars.ensemble`](ensemble.md) | Sort, prune, and analyse conformer ensembles |
| [`mars.rmsd`](rmsd.md) | RMSD CV with Kabsch alignment and JAX autodiff |
| [`mars.mtd`](mtd.md) | `MTDState` and Gaussian-hill bias |
| [`mars.topology`](topology.md) | Fragment detection and structure splitting |
| [`mars.flexibility`](flexibility.md) | Molecular flexibility metric and MTD parameter grid |
| [`mars.functional_groups`](functional_groups.md) | AccFG-style FG identification for IR peak allocation |
| [`mars.log`](log.md) | Logging, timing tracker, progress bars |
