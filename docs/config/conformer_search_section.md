# `[conformer_search]` section

Defaults for the default `mars input.xyz` mode. Every key maps to the
matching long CLI flag with underscores in place of dashes.

## Example

```toml
# run.toml — auto-configured thorough sampling with genetic crossing
[conformer_search]
mode = "thorough"
genetic_crossing = true
n_children = 30
ewin = 6.0                  # kcal/mol energy window
charge = -1.0
log_level = "INFO"
output = "ensemble.xyz"

# Optimizer choice (defaults to LBFGS). Switch to FIRE / GD / SP if needed.
method = "LBFGS"
fmax = 0.005                # final-cycle force threshold (eV/Å)
# maxiter = 2000
# max_stepsize = 0.1

[constraints]
atoms = [0, 1, 2]           # freeze the substrate during sampling
k = 100.0
apply_to = "all"            # "all" | "optimization" | "sampling"
```

```bash
mars input.xyz --config run.toml
# CLI flags still win on overlap:
mars input.xyz --config run.toml --mode quick
```

## Selected keys

| Key | Type | CLI equivalent |
|-----|------|----------------|
| `mode` | `"quick"` \| `"normal"` \| `"thorough"` | `--mode` |
| `quick` / `thorough` | bool | `--quick` / `--thorough` |
| `mtd_only` / `md_only` | bool | `--mtd-only` / `--md-only` |
| `genetic_crossing` | bool | `--genetic-crossing` |
| `n_children` | int | `--n-children` |
| `mtd_cycles` | int | `--mtd-cycles` |
| `mtd_time` | float (ps) | `--mtd-time` |
| `mtd_dt` | float (fs) | `--mtd-dt` |
| `kpush` | float | `--kpush` |
| `alpha` | float | `--alpha` |
| `cvdump_fs` | float | `--cvdump-fs` |
| `no_trial` | bool | `--no-trial` |
| `no_topology` | bool | `--no-topology` |
| `temperature` | float (K) | `-T` / `--temperature` |
| `charge` | float | `--charge` |
| `ewin` | float (kcal/mol) | `--ewin` |
| `rotamer_structures` | int | `--rotamer-structures` |
| `rotamer_temps` | int | `--rotamer-temps` |
| `rotamer_time` | float (ps) | `--rotamer-time` |
| `nci` | bool | `--nci` |
| `nci_buffer` | float (Å) | `--nci-buffer` |
| `nci_mode` | `"wall"` \| `"harmonic"` | `--nci-mode` |
| `nci_max_dist` | float (Å) | `--nci-max-dist` |
| `method` | `"LBFGS"` \| `"FIRE"` \| `"GD"` \| `"SP"` | `--method` |
| `fmax` | float (eV/Å) | `--fmax` |
| `maxiter` | int | `--maxiter` |
| `max_stepsize` | float (Å) | `--max-stepsize` |
| `fire_dt_start` / `fire_dt_max` / `fire_n_min` | float / float / int | `--fire-dt-*` |
| `output` | str | `-o` / `--output` |
| `save_trajectory` | bool | `--save-trajectory` |

For the full flag list see the
[CLI reference](../cli/conformer_search.md).
