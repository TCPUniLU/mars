# `mars viewer`

Interactive 3D visualization for molecules, harmonic-IR results, and the `--nci`
confinement wall. Built on matplotlib (an optional dependency) — install it with
`pip install matplotlib`.

```bash
mars viewer [input.xyz] [OPTIONS]
```

The viewer **reads** files only; it never recomputes anything. It has three
modes:

| Mode | Command | What it shows |
|------|---------|---------------|
| Molecule | `mars viewer input.xyz` | Plain interactive 3D structure (rotate/zoom) |
| IR explorer | `mars viewer --ir` | 3D molecule + per-mode displacement arrows/animation, info panel, and a clickable IR spectrum |
| NCI wall | `mars viewer input.xyz --nci` | Molecule inside the spherical confinement wall used by `mars … --nci` |

In the molecule and NCI modes a multi-frame XYZ (e.g. a conformer ensemble) is
loaded in full — **◄ Conf / Conf ►** buttons and the ←/→ keys step through the
conformers. All three modes share the live atom-size, bond-width, and grid/axes
controls described below.

## Mode selection

| Flag | Default | Description |
|------|---------|-------------|
| *(positional `input.xyz`)* | — | Structure file. Required for the molecule and `--nci` modes; ignored in `--ir` mode |
| `--ir` | off | Interactive IR explorer (loads saved IR result files) |
| `--nci` | off | Show the spherical confinement wall |
| `--nci-buffer Å` | 3.0 | Clearance between the outermost atom and the wall (matches the conformer-search default) |

## IR file selection (`--ir` mode)

The IR explorer loads the files written by `mars ir … --mode-xyz --save-structure`
(and `--plot` for intensities). By default it auto-discovers them in the current
directory; explicit flags override discovery.

| Flag | Default | Description |
|------|---------|-------------|
| `--dir DIR` | current directory | Directory to search for result files |
| `--modes FILE` | `normal_modes.xyz` | Normal modes with eigenvectors (**required** — geometry, frequencies, displacements) |
| `--spectrum FILE` | `ir_spectrum.dat` | IR intensities (optional — without it, or for dipole-free models, all modes show as uniform sticks) |
| `--mode-analysis FILE` | `mode_analysis.txt` | Vibrational classification for colours/labels (optional) |

!!! note
    Only `normal_modes.xyz` is required. Generate it with
    `mars ir input.xyz --mode-xyz`. Add `--plot` (writes `ir_spectrum.dat`) for
    intensities and keep the default analysis output for mode labels/colours.

## Display options

All display options are tunable from the command line.

| Flag | Default | Description |
|------|---------|-------------|
| `--atom-scale` | 1.0 | Atom marker-size multiplier (larger = bigger atoms) |
| `--bond-width` | 2.5 | Bond line width in points |
| `--grid` | off | Show the 3D grid, axes and panes (hidden by default for a clean view; toggleable live in `--ir`) |
| `--arrow-scale` | 1.0 | Initial displacement-arrow length multiplier (`--ir`) |
| `--amplitude Å` | 0.5 | Initial vibration-animation amplitude (`--ir`) |
| `--broadening` | 10.0 | Lorentzian broadening (cm⁻¹) for the spectrum envelope (`--ir`) |
| `--freq-range MIN MAX` | `400 4000` | Frequency window for the spectrum panel (cm⁻¹) (`--ir`) |
| `--save FILE` | — | Render the view to a PNG (headless, no window) instead of opening one |

Atoms use Jmol/CPK colours and are sized by covalent radius; the 3D molecule is
always drawn with an equal (cubic) aspect ratio so geometry is not distorted.
`--arrow-scale` and `--amplitude` are also adjustable live with the sliders in
the `--ir` window.

`--save` selects the non-interactive `Agg` backend, so it works over plain SSH
and in CI where no display is available.

## Controls (IR explorer)

<figure markdown>
  ![The mars viewer --ir IR explorer stepping through methanol normal modes](../assets/viewer/ir_viewer.gif){ width="720" }
  <figcaption><code>mars viewer --ir</code> on the methanol case study: the 3D
  molecule with per-mode displacement arrows and vibration animation, an info
  panel (frequency, intensity, mode type/label/functional group), and the
  clickable IR spectrum with the selected peak marked. Stepping through the
  strongest modes.</figcaption>
</figure>

| Input | Action |
|-------|--------|
| **◄ Prev / Next ►** buttons | Step through vibrational modes |
| Click a spectrum peak | Select the nearest mode |
| **←** / **→** (or ↑/↓) | Previous / next mode |
| **space** | Toggle the vibration animation |
| **Animate** button | Toggle the vibration animation |
| **Grid** button | Toggle the 3D grid/axes on and off |
| **Arrow** slider | Adjust displacement-arrow length |
| **Amp** slider | Adjust animation amplitude (Å) |
| **Atom** slider | Adjust atom marker size |
| **Bond** slider | Adjust bond line width |
| Mouse drag / scroll (3D panel) | Rotate / zoom the molecule |

Every display option (atom size, bond width, grid/axes, arrow length, animation
amplitude) is adjustable live from the window, and each has a matching CLI flag
to set its initial value. **All** modes are shown and selectable — including
low-frequency translation/rotation modes and negative (imaginary) modes; the
spectrum x-axis widens automatically to include them and a dashed line marks
ν = 0.

### Modes without intensities (dipole-free models)

A potential without a dipole (no partial charges) yields frequencies and
eigenvectors but no IR intensities. The explorer still works: if no
`ir_spectrum.dat` is found — or it contains no positive intensities — every mode
is drawn as a **uniform-height stick** (no Lorentzian envelope), the info panel
shows `I = (not computed)`, and you can still navigate, animate, and inspect
every mode. Provide `--spectrum FILE` to overlay real intensities when available.

## Controls (molecule & NCI viewers)

<figure markdown>
  ![The mars viewer --nci confinement-wall viewer on a benzene dimer](../assets/viewer/nci_viewer.gif){ width="620" }
  <figcaption><code>mars viewer --nci</code> on a benzene dimer: the non-covalent
  complex inside its translucent spherical confinement wall (the same wall used
  by <code>mars … --nci</code>), stepping through conformers with the Buffer /
  Atom / Bond sliders and Conf ◄/► navigation.</figcaption>
</figure>

| Input | Action |
|-------|--------|
| **Grid** button | Toggle the 3D grid/axes on and off |
| **Atom** slider | Adjust atom marker size |
| **Bond** slider | Adjust bond line width |
| **◄ Conf / Conf ►** buttons | Step through conformers (multi-frame input only) |
| **←** / **→** | Previous / next conformer (multi-frame input only) |
| **Buffer (Å)** slider | *(NCI only)* Resize the confinement wall live |
| Mouse drag / scroll | Rotate / zoom the molecule |

For a multi-frame input, each conformer is shown in turn; in the NCI viewer
every conformer is independently centroid-centred and gets its own
`radius = max(|rᵢ|) + buffer`, so you can check the clearance frame by frame.

## NCI wall

`mars viewer input.xyz --nci` reproduces exactly what the conformer-search
solver enforces: positions are centred by their centroid, the wall is anchored
at the origin, and the radius is `max(|rᵢ|) + buffer`. The molecule is drawn
inside a translucent sphere, and the centre, radius, buffer, and closest
atom-to-wall clearance are printed so you can decide whether to raise
`--nci-buffer`. See the [NCI mode guide](../guide/nci_mode.md).

The window has a **Buffer (Å)** slider that redraws the wall live (the title
updates with the new radius/clearance), plus the same **Atom**, **Bond**, and
**Grid** controls as the other modes — so you can sweep buffer values without
restarting. `--nci-buffer` sets the slider's initial value.

## Examples

```bash
# Plain 3D molecule viewer
mars viewer molecule.xyz

# Multi-frame XYZ (conformer ensemble) — step through with ◄/► or arrow keys
mars viewer ensemble.xyz
mars viewer ensemble.xyz --nci          # each conformer in its own wall

# Generate IR results, then explore them interactively (auto-discovery)
mars ir molecule.xyz --mode-xyz --save-structure --plot
mars viewer --ir

# Point at a results directory or explicit files
mars viewer --ir --dir results/
mars viewer --ir --modes normal_modes.xyz --spectrum ir_spectrum.dat

# Preview the NCI confinement wall and check the clearance
mars viewer complex.xyz --nci
mars viewer complex.xyz --nci --nci-buffer 5.0

# Tune the look (initial values; all are also adjustable live in the window)
mars viewer molecule.xyz --atom-scale 1.5 --bond-width 4   # bigger atoms, thicker bonds
mars viewer molecule.xyz --grid                            # keep the grid/axes
mars viewer --ir --arrow-scale 2.0 --amplitude 0.8 --broadening 15

# Render headlessly to a PNG (no display needed)
mars viewer molecule.xyz --save view.png
mars viewer --ir --save ir_view.png
mars viewer complex.xyz --nci --nci-buffer 5 --save wall.png
```
