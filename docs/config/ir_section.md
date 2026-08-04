# `[ir]` section

Defaults for `mars ir`. Keys mirror the long CLI flags.

## Example: Hessian-based with peak allocation

```toml
# ir.toml
[ir]
plot = true
freq_range = [400, 3500]    # cm⁻¹ — list expands to `--freq-range 400 3500`
broadening = 12.0
save_hessian = true
mode_xyz = true
output = "ir_spectrum.dat"
plot_output = "ir.png"
temperature = 300.0

[global]
log_file = "ir_run.log"
```

```bash
mars ir input.xyz --config ir.toml
```

## Example: MD-based dipole autocorrelation

```toml
[ir]
md = true
nvt_time = 5.0
nve_time = 25.0
n_replicas = 4
chop = 10
window = "hann"
plot = true
```

## Example: partial Hessian on a solute in explicit solvent

When the input XYZ contains both a solute and surrounding solvent
molecules, the `atoms` key restricts the IR calculation to the solute.
The Hessian path only displaces the selected atoms (forces still see
the whole system → partial 3K×3K Hessian); the MD path tracks the
dipole only over the selected atoms.

```toml
[ir]
atoms = [0, 1, 2, 3, 4, 5, 6, 7, 8]   # solute = first nine atoms
plot = true
save_hessian = true
mode_xyz = true
```

Alternative compact forms (all equivalent):

```toml
atoms = "0-8"              # range
atoms = "0-3,7,10-12"      # mixed
atoms = 5                  # single atom
```

## Selected keys

| Key | Type | CLI equivalent |
|-----|------|----------------|
| `plot` | bool | `--plot` |
| `freq_range` | `[float, float]` | `--freq-range` |
| `broadening` | float | `--broadening` |
| `gaussian_sigma` | float | `--gaussian-sigma` |
| `no_optimize` | bool | `--no-optimize` |
| `fmax` | float (eV/Å) | `--fmax` |
| `fd_hessian` | bool | `--fd-hessian` |
| `fd_displacement` | float (Å) | `--fd-displacement` |
| `md` | bool | `--md` |
| `nvt_time` | float (ps) | `--nvt-time` |
| `nve_time` | float (ps) | `--nve-time` |
| `md_timestep` | float (fs) | `--md-timestep` |
| `dipole_save_fs` | float (fs) | `--dipole-save-fs` |
| `n_replicas` | int | `--n-replicas` |
| `chop` | int | `--chop` |
| `window` | str | `--window` |
| `conformer` | int | `--conformer` |
| `all_conformers` | bool | `--all-conformers` |
| `save_hessian` | bool | `--save-hessian` |
| `save_structure` | bool | `--save-structure` |
| `mode_xyz` | bool | `--mode-xyz` |
| `no_analysis` | bool | `--no-analysis` |
| `output` | str | `-o` |
| `plot_output` | str | `--plot-output` |
| `temperature` | float (K) | `-t` |
| `charge` | float | `--charge` |
| `atoms` | list of int, CSV/range string, or int | `--atoms` |

See the [IR CLI reference](../cli/ir.md) for the full flag list.
