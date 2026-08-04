# `[constraints]` section

Frozen-atom restraints. The `[constraints]` block is **only** available
via a config file — there is no equivalent CLI flag. It applies to the
default conformer-search mode and `mars optimize`.

## Example

```toml
[constraints]
atoms = [0, 1, 2]           # 0-indexed atom indices
k = 100.0                   # spring constant in eV/Å² (optional, default 100)
apply_to = "optimization"   # "all" | "optimization" | "sampling"
```

## Keys

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `atoms` | `[int, ...]` or string | required | Atoms to freeze. Strings accept CSV (`"0,1,2"`), ranges (`"0-5"`), and mixes (`"0-3,7,10-12"`). |
| `k` | float (eV/Å²) | 100.0 | Harmonic spring strength used when the constraint is applied as a restraint rather than a hard freeze. |
| `apply_to` | `"all"` \| `"optimization"` \| `"sampling"` | `"all"` | Which workflow stages honour the constraints. |

## How it's applied

For optimization, the gradient on the listed atoms is zeroed out, so
those positions don't move. For sampling, the same gradient mask is
applied inside the MD step, holding the atoms fixed while the rest of
the system explores configuration space.

## Example combined with `[optimize]`

```toml
[optimize]
method = "LBFGS"
fmax = 0.001
maxiter = 500

[constraints]
atoms = "0-3,7"
k = 100.0
apply_to = "optimization"
```

```bash
mars optimize input.xyz --config opt.toml
```

Watch for the line `Positional constraints: N atoms frozen ...` in the
log output — it confirms the constraints were loaded.

## See also

- [Optimization guide](../guide/optimization.md).
- API: [`mars.cli._config.get_constraints_from_config`](../api/log.md).
