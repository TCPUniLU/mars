# `mars optimize`

Find local minima or stationary points.

```bash
mars optimize <input.xyz> [OPTIONS]
```

See the [user guide](../guide/optimization.md) for concepts and
[recipes](../examples/optimize_recipes.md) for copy-pasteable command
lines.

## Task & method

| Flag | Default | Description |
|------|---------|-------------|
| `--task {minimize,stationary}` | `minimize` | Optimization task |
| `--method {LBFGS,GD,FIRE,SP}` | `LBFGS` | Minimizer |
| `--fmax eV/Å` | 0.01 | Force-convergence threshold |
| `--maxiter N` | 1000 | Maximum iterations |
| `--max-stepsize Å` | 0.2 | Per-atom step cap (LBFGS step bound, SP trust radius; not used by FIRE) |

`--task stationary` forces `--method SP` (damped Newton).

## FIRE-specific

| Flag | Default | Description |
|------|---------|-------------|
| `--fire-dt-start` | 0.05 | Initial FIRE timestep |
| `--fire-dt-max` | 0.1 | Maximum FIRE timestep |
| `--fire-n-min` | 2 | Minimum positive-power steps before dt increase |

## Multi-conformer

| Flag | Description |
|------|-------------|
| `--conformer N` | Optimize a single frame (0-indexed) |
| `--all-conformers` | Vmap-parallel optimization of every frame |
| `--output-prefix STR` | Filename prefix for per-conformer output (default: `optimized`) |

## Output

| Flag | Default | Description |
|------|---------|-------------|
| `-o`, `--output FILE` | `optimized.xyz` | Output structure file |

## Examples

```bash
mars optimize input.xyz --fmax 0.001
mars optimize input.xyz --task stationary
mars optimize input.xyz --method FIRE --fire-dt-max 0.05
mars optimize ensemble.xyz --all-conformers
mars optimize ensemble.xyz --all-conformers --not-parallel
mars optimize input.xyz --config opt.toml
```
