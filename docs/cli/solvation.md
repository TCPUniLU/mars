# `mars solvation`

Build explicit solvation shells around a solute.

```bash
mars solvation <solute.xyz> [OPTIONS]
```

Candidates are generated **per solute atom** so the shell **hugs the molecular
surface** (works for molecules of any size, including disconnected fragments).
By **default** (`--opt-mode layerwise`) the build is iterative: fill a shell with
as many solvents as fit → relax (solute frozen) → refill (the relaxation opens
space) → repeat until the shell is saturated, wrapping the solute in a dense,
relaxed first solvation shell. Spacing is **shape-aware** (cross-sectional radius,
not a sphere) so linear/elongated solvents pack tightly. The solute is left in its
input frame (solvent atoms appended after it).

`--n-molecules N` instead keeps **exactly N**: the same dense shape-aware fill
grows shells until N are placed, then trims to the N closest to the solute (so
the count packs tightly against the surface). `--opt-mode none` is fast
placement-only (no potential).

See the [user guide](../guide/solvation.md) for concepts and
[recipes](../examples/solvation_recipes.md) for copy-pasteable command
lines.

## Solvent selection

| Flag | Default | Description |
|------|---------|-------------|
| `--solvent NAME\|FILE` | `water` | A built-in solvent name/alias, **or a path to your own `.xyz`** |

`--solvent` accepts either:

- a built-in solvent name or alias (case-insensitive — `DCM`, `EtOH`, `DMSO`,
  `hexane`, …; see [`mars.solvents.list_solvents()`](../api/solvents.md)), or
- a **path to your own single-molecule `.xyz` file**, e.g.
  `--solvent ./my_solvent.xyz`. The value is treated as a file when it ends in
  `.xyz`, exists on disk, or contains a path separator.

```bash
# Use a custom solvent geometry from your own XYZ file
mars solvation solute.xyz --solvent ./propylene_carbonate.xyz --padding 5.0
```

!!! info "Built-in geometries"
    Built-in solvents marked `experimental` in the manifest ship curated
    geometries; all others are generated from SMILES with Open Babel
    (`obabel --gen3d`, MMFF94). Regenerate them with
    `python tools/regen_solvents.py`. The shell builder relaxes solvent
    geometries anyway when `--opt-mode != none`.

## Mode

Default (no target flag) = **cover**: fill+relax the contact shell until it
saturates, wrapping the solute. Otherwise provide one of:

| Flag | Default | Description |
|------|---------|-------------|
| *(none)* | cover | Saturate the first contact shell (wraps the solute) |
| `--padding Å` | — | Grow saturated shells until this thickness is reached |
| `--n-molecules N` | — | Keep exactly N solvents (dense shell fill → trim to N closest) |
| `--layers N [N ...]` | — | Manual shell counts |
| `--max-layers N` | 10 | Safety cap on the number of shells |

## Optimization

| Flag | Default | Description |
|------|---------|-------------|
| `--opt-mode {none,after-all,layerwise}` | `layerwise` | `layerwise` = iterative fill→relax→cover (default, needs a potential); `after-all` = one relax at the end; `none` = placement only (fast) |
| `--freeze-mode {solute,solvent,none}` | `solute` | Which atoms to freeze (used with `after-all`) |
| `--method {FIRE,LBFGS,GD,HYBRID}` | `FIRE` | Optimizer |
| `--fmax eV/Å` | 0.05 | Force-convergence threshold |
| `--maxiter N` | 10000 | Maximum iterations per layer |
| `--max-stepsize Å` | 0.15 | Per-atom step cap (LBFGS step bound, SP trust radius; not used by FIRE) |
| `--fire-dt-start` / `--fire-dt-max` / `--fire-n-min` | 0.05 / 0.1 / 2 | FIRE tuning |

## Placement parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--vdw-scale` | 0.75 | vdW overlap scale factor (lower = tighter packing / more vdW overlap pre-relax) |
| `--buffer Å` | 0.0 | Extra gap beyond vdW contact |
| `--n-candidates` | 300 | Fibonacci candidate points per atom per radial band |
| `--n-shells` | 3 | Radial candidate bands per atom |
| `--n-orient` | 8 | Orientation tries per site (cover mode) — helps elongated solvents fit |
| `--refill-cycles` | 3 | Max fill→relax→refill cycles per shell (cover mode) |
| `--seed` | 42 | Base random seed (layer `i` uses `seed + i`) |

## Topology repair

After relaxation, each solvent molecule's covalent topology is checked against
the reference: a molecule is **broken** if a reference bond is missing (it
fragmented / an atom dissociated) or if it formed a covalent bond to another
molecule or the solute (it reacted). Hydrogen bonds sit above the covalent
threshold, so they are not flagged.

In the **layerwise/auto** build the check runs **after each shell's relaxation**
(you'll see a `shell N topology check: …` line per shell), removing broken
molecules in place so the refill re-places them. After the build a final
convergence loop — remove broken → refill to the target → re-relax → recheck —
repeats until none are broken (or `--max-repair` passes). In **every** mode a
final per-structure removal pass also cleans the output (after `after-all` /
`--final-relax`); any structure whose count dips is then **refilled back to the
target** (geometric placement + relax) so the saved ensemble keeps a uniform
solvent count — see *Batch uniformization* below.

## Batch uniformization

When multiple conformers are solvated together (e.g. `--all-conformers`,
`--n-solvations`, or barostat batches), the vmap batch optimizer requires every
structure to have the **same** atom count. The geometric and barostat builders
can be geometry-limited at different counts per conformer, and broken-solvent
removal can lower counts — leaving a ragged batch that would otherwise crash the
optimizer (`jnp.stack` shape mismatch).

To prevent this, the workflow uniformizes the batch before optimization (and
again after the final topology cleanup), covering **every** build path
(independent / swap-transplant / barostat):

- `count == target` → left untouched;
- `count > target` → trimmed to the molecules closest to the solute;
- `count < target` → repaired in up to **2 rounds** of *remove-broken → geometric
  refill to the target → frozen-solute relax*.

The target is `--n-molecules` (treated as a hard requirement). If a structure is
genuinely geometry-limited and can't reach it after the repair rounds, a warning
is logged and the whole batch is trimmed down to the minimum achieved count so
the result stays uniform.

| Flag | Default | Description |
|------|---------|-------------|
| `--no-topology-repair` | (on) | Disable the post-relaxation topology check entirely |
| `--max-repair N` | 3 | Max remove→refill→relax repair passes (layerwise/auto) |
| `--topo-tolerance T` | 1.3 | Covalent-bond tolerance — a bond counts as broken past `T × (r_i + r_j)` |

## Barostat mode (spherical moving-wall droplet)

`--barostat` builds the solvent shell a different way: instead of placing solvent
on the solute surface, it packs a **solute-free spherical droplet** and
compresses it with a moving wall, then swaps the solute in. Requires
`--n-molecules N`.

1. **Place** N solvent molecules loosely in a spherical shell `[r_inner, r_outer]`
   (no solute). `r_inner = solute_radius + --baro-inner-buffer` reserves a central
   cavity; `r_outer` is sized from `--baro-fill` (or set with `--baro-radius`).
2. **Compress the droplet** (barostat): relax inside an inner cavity wall + an
   outer "piston" wall, measure the outer-wall **reaction force**, then move the
   wall inward — `--baro-step` (coarse) down to the refine zone, then
   `--baro-fine-step` (0.1 Å) — until the force reaches `--baro-max-force` (the
   target "pressure"). It does **not** stop at a fixed minimum radius; a low
   collapse-safety radius is the only hard backstop.
3. **Swap**: drop the solute into the cavity (solute atoms first).
4. **Final packing**: remove the inner wall and keep compressing the outer wall
   inward with the **solute frozen** (coarse then fine steps) — this constrains
   the solvent toward the solute, closing the cavity gap and packing the first
   shell onto the solute surface — until the wall force is reached again.
5. **Final relaxation without the wall**: the wall is removed and the solvent is
   relaxed to its true minimum (solute still frozen), so the structure is not
   biased by the barostat.
6. **Topology check**: broken solvent molecules are removed. Counts are then
   restored to the target by the workflow's batch uniformization step (geometric
   refill + relax), so barostat batches stay uniform.

With `--solute-swap` the droplet is built **once** from the reference, every
conformer is aligned and swapped into the cavity, and the final packing (step 4)
runs as a **single vmapped batch with a per-structure moving wall**: at each step
all structures are relaxed together; any structure whose wall reaction force
reaches `--baro-max-force` **freezes its wall** while the rest keep compressing,
until all have converged. The wall-free relax (step 5) is also batched. (No
`after-all` re-optimization is applied in barostat mode — it already optimizes.)
A conformer batch that exceeds GPU memory is **automatically split** into
sub-batches and recombined (recursive halving, same as the other batch paths) —
no flag needed.

Compression is optimization-driven (FIRE) by default; with `--baro-md` each step
first runs a short NVT MD (the wall is added to the energy) so the solvent
flows/packs like a liquid, then settles with FIRE (droplet stage only).

| Flag | Default | Description |
|------|---------|-------------|
| `--barostat` | off | Enable barostat-droplet mode (needs `--n-molecules`) |
| `--baro-inner-buffer Å` | 2.0 | Cavity clearance beyond the solute bounding radius |
| `--baro-radius Å` | auto | Initial outer-wall radius (auto-sized from `--baro-fill`) |
| `--baro-fill` | 0.35 | Initial fill fraction used to size the outer radius (lower = looser start) |
| `--baro-step Å` | 0.5 | Coarse outer-wall inward step per barostat iteration |
| `--baro-fine-step Å` | 0.1 | Fine step near the target — the wall creeps in by this until the force is met |
| `--baro-k eV/Å²` | 5.0 | Wall force constant |
| `--baro-wall-mode {harmonic,wall}` | harmonic | Wall form: harmonic (n=2, soft) or wall (n=12, steep) |
| `--baro-max-force eV/Å` | 1.0 | Stop when the outer-wall reaction force reaches this (target pressure) |
| `--baro-max-steps N` | 40 | Maximum compression steps |
| `--baro-md` | off | Run a short NVT MD (wall in energy) at each step before the FIRE settle — solvent flows like a liquid |
| `--baro-md-temp K` | 300 | Temperature for `--baro-md` |
| `--baro-md-time ps` | 0.3 | MD time per compression step (`--baro-md`) |
| `--baro-md-dt fs` | 0.5 | MD timestep (`--baro-md`) |

## Multi-conformer / replica

| Flag | Description |
|------|-------------|
| `--conformer N` | Reference frame (0-indexed) |
| `--all-conformers` | Solvate every frame independently |
| `--solute-swap` | Solvate the reference, then transplant the shell onto every other conformer and re-optimize |
| `--no-align` | Disable RMSD alignment of conformers to the reference (transplant mode) |
| `--n-solvations N` | Independent solvation replicas (different seeds) |
| `--relax-solute` | Also relax solute atoms during constrained optimisation |
| `--final-relax [FMAX]` | Run an unconstrained relax after the constrained one. Flag alone uses the process `--fmax`; give a value (e.g. `--final-relax 0.01`) to tighten the final force threshold only |
| `--no-multistage` | Single-pass optimisation (skip coarse→fine staging) |

## Output

| Flag | Default | Description |
|------|---------|-------------|
| `-o`, `--output FILE` | `solvated.xyz` / `<prefix>_all.xyz` | Output structure |
| `--output-prefix STR` | `solvated` | Per-conformer/replica prefix |

## Examples

```bash
mars solvation solute.xyz --solvent water --padding 5.0
mars solvation solute.xyz --solvent methanol --layers 4 8 --opt-mode layerwise
mars solvation solute.xyz --solvent DMSO --n-molecules 30 --potential mace
mars solvation ensemble.xyz --solvent water --padding 5.0 --solute-swap
mars solvation solute.xyz --config solv.toml
```
