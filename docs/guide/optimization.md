# Optimization

`mars optimize input.xyz` finds local minima or stationary points.

## Tasks

- **Minimization** (`--task minimize`, default): converge to the nearest
  energy minimum using `--method`.
- **Stationary point** (`--task stationary`): damped-Newton search that
  doesn't enforce a gradient sign, useful for locating transition states
  or saddle points. Equivalent to `--method SP`.

## Minimizers

| Method | Backend | Notes |
|--------|---------|-------|
| `LBFGS` *(default)* | jaxopt | Limited-memory BFGS; fast for most molecules. Uses a backtracking line search to avoid clashing steps on strained geometries |
| `FIRE`  | jax-md | Robust against bad starting geometries; tune `--fire-dt-max` |
| `GD`    | jaxopt | Simple gradient descent, mostly diagnostic |
| `SP`    | custom | Damped-Newton; respects `--max-stepsize` as trust radius |

Convergence is controlled by `--fmax` (eV/Å) and `--maxiter`.

## Coordinate systems

`--coords` is an axis orthogonal to `--method`, and is an opt-in add-on to
`mars optimize` only — the conformational-search CLI is unchanged.

| | |
|--------|-------|
| `cartesian` *(default)* | Everything above, unchanged |
| `internal` | RFO/BFGS in redundant internal coordinates, with `--init-hessian {identity,lindh}` |

**`--coords internal` requires `--float64`** and exits with an error without
it — in float32 the back-transformation residual alone exceeds a tight
`--fmax`, so the optimizer would oscillate without ever reporting a failure.

```bash
mars optimize small.xyz --coords internal --float64 --fmax 0.002
```

## Batch optimization

For multi-frame XYZ inputs:

- `--conformer N` — pick a single frame (0-indexed).
- `--all-conformers` — vmap-parallel optimization over the entire ensemble
  (use `--not-parallel` to fall back to a sequential loop, useful for
  debugging or when memory is tight).

Output goes to `<output-prefix>_<i>.xyz` per conformer; sorted by final
energy in the multi-frame XYZ when `-o ensemble.xyz` is set.

## Frozen-atom constraints

The optimizer reads a `[constraints]` block from the
[`--config FILE`](../config/constraints_section.md):

```toml
[constraints]
atoms = [0, 1, 2]
k = 100.0
apply_to = "optimization"
```

This zeroes the gradient on the listed atoms, holding them fixed during
minimization.

## See also

- [Recipes](../examples/optimize_recipes.md).
- [`[optimize]` config section](../config/optimize_section.md).
- API reference: [`mars.optimizer`](../api/optimizer.md).
