# `[solvation]` section

Defaults for `mars solvation`.

## Example: manual layers with layerwise relax

```toml
# solv.toml
[solvation]
solvent = "water"
layers = [4, 8]             # → --layers 4 8
opt_mode = "layerwise"
method = "FIRE"
fmax = 0.05
vdw_scale = 0.75
seed = 42
output = "solvated.xyz"

[global]
charge = 0.0
```

```bash
mars solvation solute.xyz --config solv.toml
```

## Example: automatic mode

For automatic-mode solvation, replace `layers = [...]` with **one** of:

```toml
[solvation]
solvent = "water"
padding = 5.0               # grow shells until ≥5 Å padding
```

```toml
[solvation]
solvent = "water"
n_molecules = 30            # keep exactly 30 solvent molecules
```

## Selected keys

| Key | Type | CLI equivalent |
|-----|------|----------------|
| `solvent` | str | `--solvent` |
| `padding` | float (Å) | `--padding` |
| `n_molecules` | int | `--n-molecules` |
| `layers` | `int` or `[int, ...]` | `--layers` |
| `max_layers` | int | `--max-layers` |
| `n_orient` | int | `--n-orient` |
| `refill_cycles` | int | `--refill-cycles` |
| `opt_mode` | `"none"` \| `"after-all"` \| `"layerwise"` | `--opt-mode` |
| `method` | `"FIRE"` \| `"LBFGS"` \| `"GD"` \| `"HYBRID"` | `--method` |
| `freeze_mode` | `"solute"` \| `"solvent"` \| `"none"` | `--freeze-mode` |
| `fmax` | float (eV/Å) | `--fmax` |
| `maxiter` | int | `--maxiter` |
| `vdw_scale` | float | `--vdw-scale` |
| `buffer` | float (Å) | `--buffer` |
| `n_candidates` | int | `--n-candidates` |
| `n_shells` | int | `--n-shells` |
| `topology_repair` | bool | `--no-topology-repair` (set `false` to disable) |
| `max_repair` | int | `--max-repair` |
| `topo_tolerance` | float | `--topo-tolerance` |
| `barostat` | bool | `--barostat` (needs `n_molecules`) |
| `baro_inner_buffer` | float (Å) | `--baro-inner-buffer` |
| `baro_radius` | float (Å) | `--baro-radius` |
| `baro_fill` | float | `--baro-fill` |
| `baro_step` | float (Å) | `--baro-step` |
| `baro_fine_step` | float (Å) | `--baro-fine-step` |
| `baro_k` | float (eV/Å²) | `--baro-k` |
| `baro_wall_mode` | `"harmonic"` \| `"wall"` | `--baro-wall-mode` |
| `baro_max_force` | float (eV/Å) | `--baro-max-force` |
| `baro_max_steps` | int | `--baro-max-steps` |
| `baro_md` | bool | `--baro-md` |
| `baro_md_temp` | float (K) | `--baro-md-temp` |
| `baro_md_time` | float (ps) | `--baro-md-time` |
| `baro_md_dt` | float (fs) | `--baro-md-dt` |
| `seed` | int | `--seed` |
| `conformer` | int | `--conformer` |
| `all_conformers` | bool | `--all-conformers` |
| `solute_swap` | bool | `--solute-swap` |
| `no_align` | bool | `--no-align` |
| `relax_solute` | bool | `--relax-solute` |
| `final_relax` | bool or float | `--final-relax` (`true` = process `--fmax`; a float = final-relax fmax) |
| `n_solvations` | int | `--n-solvations` |
| `output` | str | `-o` |
| `output_prefix` | str | `--output-prefix` |
| `charge` | float | `--charge` |

See the [solvation CLI reference](../cli/solvation.md) for the full list.
