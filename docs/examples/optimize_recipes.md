# Optimization recipes

## Local minimum

```bash
# LBFGS with default convergence (0.01 eV/Å)
mars optimize input.xyz

# Tighter convergence + custom output
mars optimize input.xyz --fmax 0.001 --maxiter 500 -o opt.xyz

# Pick a different minimizer
mars optimize input.xyz --method FIRE --fire-dt-max 0.05
mars optimize input.xyz --method GD --max-stepsize 0.05
```

## Stationary point (saddle / transition state)

```bash
# Damped-Newton search — no gradient sign enforced
mars optimize input.xyz --task stationary
mars optimize input.xyz --method SP   # equivalent
```

## Batch (multi-frame input)

```bash
# Optimize the 3rd conformer only (0-indexed)
mars optimize ensemble.xyz --conformer 2

# Optimize every frame in parallel (vmap)
mars optimize ensemble.xyz --all-conformers --output-prefix opt

# Sequential fallback (lower memory, easier to debug)
mars optimize ensemble.xyz --all-conformers --not-parallel
```

## With frozen-atom constraints

`[constraints]` is only available via the config file:

```toml
# opt.toml
[optimize]
task = "minimize"
method = "LBFGS"
fmax = 0.001
maxiter = 500
output = "optimized.xyz"

[constraints]
atoms = "0-3,7"          # CSV/range strings work too: → [0,1,2,3,7]
k = 100.0
apply_to = "optimization"
```

```bash
mars optimize input.xyz --config opt.toml
```

See the [`[constraints]` config section](../config/constraints_section.md).

## Switching potentials

```bash
mars optimize input.xyz --potential mace --mace-foundation off --mace-model small
mars optimize input.xyz --potential dxtb --dxtb-method gfn2
mars optimize input.xyz --potential harmonic   # toy potential (CI / tests)
```

## Python API

```python
from mars import (
    load_structure, get_potential,
    optimize_single, find_stationary_point,
)

structure = load_structure("input.xyz")
potential = get_potential("so3lr", species=structure["numbers"])
energy_fn = potential.build_energy_fn()

# Local minimum
opt_pos, energy, info = optimize_single(
    structure["positions"], energy_fn,
    fmax=0.01, maxiter=1000, method="LBFGS",
)
print(f"Converged: {info['converged']}, E = {energy:.4f} eV")

# Stationary point
opt_pos, energy = find_stationary_point(structure["positions"], energy_fn)
```

See also: [User guide](../guide/optimization.md),
[CLI reference](../cli/optimize.md),
[`mars.optimizer`](../api/optimizer.md).
