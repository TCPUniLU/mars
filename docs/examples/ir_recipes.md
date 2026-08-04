# IR recipes

## Hessian-based

```bash
# Analytical Hessian (default; recommended for small molecules)
mars ir input.xyz --plot

# Finite-differences Hessian (use when autodiff is unstable)
mars ir input.xyz --fd-hessian --fd-displacement 0.01 --plot

# Save Hessian matrix + normal modes + per-mode analysis
mars ir input.xyz --plot --save-hessian --mode-xyz
```

## MD-based dipole autocorrelation

```bash
# Single replica, default 5 ps NVT + 25 ps NVE
mars ir input.xyz --md --plot

# Multiple replicas, segment-averaged spectrum
mars ir input.xyz --md --n-replicas 4 --chop 10 --window hann --plot

# Custom MD parameters
mars ir input.xyz --md --nvt-time 10 --nve-time 50 --md-timestep 0.5 \
    --dipole-save-fs 2.5
```

!!! warning "Partial charges required"
    MD-based IR needs a potential that provides partial charges. Currently
    only `--potential so3lr` qualifies in the built-in registry.

## Multi-conformer

```bash
# Single conformer from a multi-frame XYZ
mars ir ensemble.xyz --conformer 2 --plot

# Compute IR for every conformer (per-conformer output files)
mars ir ensemble.xyz --all-conformers --plot --output-prefix ir
```

## Partial-system IR (solute in explicit solvent)

Use `--atoms` to restrict the IR calculation to a 0-based subset of the
input atoms — typically the solute of a `solute+solvent` XYZ. The
Hessian path only displaces the selected atoms (forces still come from
the full system); the MD path tracks only the selected atoms' dipole.

```bash
# Partial Hessian over the first nine atoms (solute)
mars ir solvated.xyz --atoms 0-8 --plot

# Noncontiguous selection
mars ir solvated.xyz --atoms "0-2,15" --plot

# Save the partial 3K×3K Hessian and the selected normal modes
mars ir solvated.xyz --atoms 0-8 --plot --save-hessian --mode-xyz

# MD-based IR with the dipole summed over the selected atoms only
mars ir solvated.xyz --atoms 0-8 --md --plot
```

Accepted formats are CSV (`0,1,2`), range (`0-5`), mixed
(`0-3,7,10-12`), or whitespace-separated tokens (`0-3 7 10-12`).

```toml
# ir_partial.toml
[ir]
atoms = [0, 1, 2, 3, 4, 5, 6, 7, 8]    # or "0-8", or "0-3,7,10-12"
plot = true
save_hessian = true
mode_xyz = true
```

```bash
mars ir solvated.xyz --config ir_partial.toml
```

## Switching potentials

```bash
# MACE (requires mace-torch + mace-jax installed)
mars ir input.xyz --potential mace --mace-foundation off \
    --mace-model small --plot

# dxtb GFN2-xTB
mars ir input.xyz --potential dxtb --dxtb-method gfn2 --plot
```

## Plot tuning

```bash
# Frequency range + broadening
mars ir input.xyz --plot --freq-range 400 3500 --broadening 12.0

# Gaussian smoothing on MD-based plots
mars ir input.xyz --md --plot --gaussian-sigma 3.0
```

## Driving from a config file

```toml
# ir.toml
[ir]
plot = true
freq_range = [400, 3500]
broadening = 12.0
save_hessian = true
mode_xyz = true
output = "ir_spectrum.dat"
plot_output = "ir.png"
temperature = 300.0
```

```bash
mars ir input.xyz --config ir.toml
```

For an MD-based run, swap the `[ir]` body for:

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

## Python API

```python
from mars.ir import compute_ir_spectrum, compute_ir_from_md

# Analytical Hessian
result = compute_ir_spectrum(
    "input.xyz",
    potential_name="so3lr",
    fmax=0.005,
    broadening=10.0,
)

# MD-based IR
result = compute_ir_from_md(
    "input.xyz",
    potential_name="so3lr",
    nvt_time_ps=5.0,
    nve_time_ps=25.0,
)

# Pass extra potential options for MACE / dxtb
result = compute_ir_spectrum(
    "input.xyz",
    potential_name="mace",
    potential_options={"mace_foundation": "off", "mace_model": "small"},
)

# Partial Hessian on a subset (e.g. solute in explicit solvent)
result = compute_ir_spectrum(
    "solvated.xyz",
    potential_name="so3lr",
    atom_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8],
)
# result["positions"], ["symbols"], ["masses"], ["numbers"], ["charges"]
# refer to the selected subset. The original full structure lives under
# result["full_positions"], ["full_symbols"], ["full_numbers"],
# ["full_masses"], ["full_charges"]; result["atom_indices"] records the
# selection.

# Same idea for MD-based IR — restricts the dipole to selected atoms
result = compute_ir_from_md(
    "solvated.xyz",
    potential_name="so3lr",
    atom_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8],
)
```

## Explore the results interactively

```bash
# Produce the result files, then open the interactive IR explorer
mars ir input.xyz --mode-xyz --save-structure --plot
mars viewer --ir                       # auto-discovers normal_modes.xyz etc.

# Point at a specific directory or files
mars viewer --ir --dir results/
mars viewer --ir --modes normal_modes.xyz --spectrum ir_spectrum.dat

# Tune the look (also adjustable live with the in-window sliders/buttons)
mars viewer --ir --atom-scale 1.5 --bond-width 4 --arrow-scale 2.0 --amplitude 0.8

# Render a snapshot headlessly (no display required)
mars viewer --ir --save ir_view.png
```

The explorer shows a rotatable 3D molecule with per-mode displacement arrows
and an oscillation animation, an info panel (frequency / intensity /
classification), and a clickable IR spectrum with a marker on the selected
mode. See the [`mars viewer` reference](../cli/viewer.md).

See also: [User guide](../guide/ir_spectroscopy.md),
[CLI reference](../cli/ir.md), [`mars viewer`](../cli/viewer.md),
[`mars.ir`](../api/ir.md),
[`mars.functional_groups`](../api/functional_groups.md).
