# `mars ir`

Compute the IR spectrum from a structure (or every frame in a multi-frame
XYZ).

```bash
mars ir <input.xyz> [OPTIONS]
```

See the [user guide](../guide/ir_spectroscopy.md) for concepts and
[recipes](../examples/ir_recipes.md) for copy-pasteable command lines.

## Method

| Flag | Default | Description |
|------|---------|-------------|
| *(none)* | analytical Hessian | Default — JAX autodiff |
| `--fd-hessian` | off | Approximate the Hessian by finite differences |
| `--fd-displacement Å` | 0.01 | Displacement for FD Hessian |
| `--md` | off | MD-based IR via dipole autocorrelation |
| `--nvt-time PS` | 5.0 | NVT equilibration time (MD-based) |
| `--nve-time PS` | 25.0 | NVE production time (MD-based) |
| `--md-timestep FS` | 0.5 | MD timestep (MD-based) |
| `--dipole-save-fs FS` | 2.5 | Save dipole every N fs (MD-based) |
| `--n-replicas N` | 1 | Independent MD replicas with different seeds |
| `--chop N` | 10 | Split trajectory into N segments for averaging |
| `--window {hann,hamming,blackman,bartlett,none}` | `hann` | Autocorrelation window |
| `--trajectory-file FILE` | — | Save NVE trajectory as HDF5 |

## Geometry

| Flag | Default | Description |
|------|---------|-------------|
| `--no-optimize` | off | Skip pre-Hessian geometry optimization |
| `--fmax eV/Å` | 0.01 | Force-convergence for pre-Hessian optimization (ignored if `--no-optimize`) |

## Multi-conformer

| Flag | Description |
|------|-------------|
| `--conformer N` | Compute IR for a single frame (0-indexed) |
| `--all-conformers` | Compute IR for every frame |
| `--output-prefix STR` | Prefix for per-conformer output files (default: `conformer`) |

## Atom selection (partial Hessian / partial dipole)

| Flag | Description |
|------|-------------|
| `--atoms IDX [IDX ...]` | Restrict the IR calculation to a subset of atoms (0-based). Accepts CSV `0,1,2`, range `0-5`, mixed `0-3,7,10-12`, or whitespace-separated tokens. |

When `--atoms` is given:

- **Hessian path** — only the selected atoms are displaced. The forces and
  charges still see every atom in the system, so the partial Hessian
  captures how the selected atoms move under the influence of the full
  environment. The diagonalized matrix is `3K × 3K` (`K = len(atoms)`)
  and the reported `positions / symbols / masses / numbers / charges`
  in the result dict refer to the selected subset (the full structure
  is preserved under `full_positions` / `full_symbols` / `full_numbers`
  / `full_masses` / `full_charges` and the selection itself under
  `atom_indices`).
- **MD path** (`--md`) — the dynamics run on the full system; the tracked
  dipole is `μ(t) = Σ_{i ∈ atoms} q_i(t) r_i(t)`. Useful for isolating
  the IR signal of a solute in explicit solvent.

## Output

| Flag | Default | Description |
|------|---------|-------------|
| `-o`, `--output FILE` | `ir_spectrum.dat` | Spectrum data file |
| `--plot` | off | Save a PNG plot (`--plot-output`) |
| `--plot-output FILE` | `ir_spectrum.png` | Plot path |
| `--freq-range MIN MAX` | `400 4000` | Frequency window for the plot (cm⁻¹) |
| `--broadening` | 10.0 | Lorentzian broadening (cm⁻¹) |
| `--gaussian-sigma` | 2.0 | Gaussian smoothing for MD plots |
| `--save-hessian` | off | Save Hessian matrix + frequencies + force constants |
| `--hessian-output FILE` | `ir_hessian_data.txt` | |
| `--save-structure` | off | Save the pre-Hessian optimized geometry |
| `--structure-output FILE` | `optimized_structure.xyz` | |
| `--mode-xyz` | off | Save all normal modes as a multi-frame XYZ |
| `--mode-output FILE` | `normal_modes.xyz` | |
| `--no-analysis` | off | Skip vibrational-mode classification + FG allocation |
| `--analysis-output FILE` | `mode_analysis.txt` | |

## Examples

```bash
mars ir input.xyz --plot
mars ir input.xyz --fd-hessian --plot
mars ir input.xyz --md --n-replicas 4 --chop 10 --plot
mars ir input.xyz --plot --save-hessian --mode-xyz
mars ir ensemble.xyz --all-conformers --plot
mars ir input.xyz --potential mace --mace-foundation off --mace-model small --plot

# Partial Hessian — solute in explicit solvent (solute = atoms 0..8)
mars ir solvated.xyz --atoms 0-8 --plot
mars ir solvated.xyz --atoms "0-2,15" --plot         # noncontiguous subset
mars ir solvated.xyz --atoms 0-8 --md --plot         # partial dipole in MD
```

## Interactive viewer

The saved outputs (`normal_modes.xyz`, `ir_spectrum.dat`, `mode_analysis.txt`)
can be explored interactively with [`mars viewer --ir`](viewer.md): a rotatable
3D molecule with per-mode displacement arrows and animation, alongside a
clickable IR spectrum.

```bash
mars ir input.xyz --mode-xyz --save-structure --plot
mars viewer --ir              # auto-discovers the files in the current directory
```
