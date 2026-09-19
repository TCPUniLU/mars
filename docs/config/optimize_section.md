# `[optimize]` section

Defaults for `mars optimize`. Pair with a `[constraints]` section to
freeze atoms.

## Example

```toml
# opt.toml — LBFGS minimization with constraints
[optimize]
task = "minimize"           # or "stationary"
method = "LBFGS"
fmax = 0.001                # eV/Å
maxiter = 500
output = "optimized.xyz"

[constraints]
atoms = "0-3,7"             # CSV/range strings work too: → [0,1,2,3,7]
k = 100.0
apply_to = "optimization"
```

```bash
mars optimize input.xyz --config opt.toml
```

## Selected keys

| Key | Type | CLI equivalent |
|-----|------|----------------|
| `task` | `"minimize"` \| `"stationary"` | `--task` |
| `method` | `"LBFGS"` \| `"GD"` \| `"FIRE"` \| `"SP"` | `--method` |
| `fmax` | float (eV/Å) | `--fmax` |
| `maxiter` | int | `--maxiter` |
| `max_stepsize` | float (Å) | `--max-stepsize` |
| `fire_dt_start` / `fire_dt_max` / `fire_n_min` | various | FIRE-specific |
| `conformer` | int | `--conformer` |
| `all_conformers` | bool | `--all-conformers` |
| `output_prefix` | str | `--output-prefix` |
| `output` | str | `-o` |
| `charge` | float | `--charge` |

See the [optimize CLI reference](../cli/optimize.md) for the full list.

## Coordinate system keys

These need no special handling — the config layer turns any scalar key into
the matching flag:

```toml
[optimize]
coords = "internal"              # --coords internal   (default: cartesian)
init_hessian = "lindh"           # --init-hessian lindh
interfragment = "tric"           # --interfragment tric
ric_max_atoms = 150              # --ric-max-atoms 150
ric_backtransform_iter = 25      # --ric-backtransform-iter 25
```

These keys belong to `[optimize]` only. The conformational-search CLI does not
expose internal coordinates; drive
[`mars.conf_sampling`](../api/conf_sampling.md) directly if you need them
there.

See the [optimization guide](../guide/optimization.md).
