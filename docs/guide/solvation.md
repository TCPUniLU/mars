# Explicit solvation

`mars solvation solute.xyz` builds an explicit solvent environment around a
solute. MARS ships **two independent builders**:

- **Layered (surface packing)** — the default. Grows solvent shells outward from
  the solute surface, relaxing each shell (`--opt-mode layerwise`). Covered next.
- **[Barostat (droplet compression)](#barostat-mode-spherical-moving-wall-droplet-barostat)**
  (`--barostat`) — packs a solute-free droplet, compresses it with a moving wall,
  then swaps the solute in. Denser, more bulk-liquid-like.

See [Two ways to solvate the same molecule](../examples/solvation_recipes.md#two-ways-to-solvate-the-same-molecule)
for a side-by-side demo of both on one molecule.

## Layered build — iterative fill→relax cover (`--opt-mode layerwise`)

The default builder places solvent by per-atom Fibonacci-sphere candidate
sampling with vdW-overlap rejection, running an iterative *fill → relax → refill*
cover that wraps the solute in a dense, relaxed first solvation shell.

### How the fill→relax→refill cover works

For each concentric shell (inner → outer):

1. **Fill to capacity** — place as many solvents as fit without clashing (per-atom
   surface candidates; each candidate is tried in several orientations,
   `--n-orient`, so elongated solvents find a tangent fit).
2. **Relax** — minimise the cluster with the chosen potential, the **solute frozen**.
3. **Refill** — the relaxation opens space, so fill again; repeat (up to
   `--refill-cycles`) until a fill adds nothing — the shell is **saturated**.
4. Advance to the next shell only if more thickness is requested.

By default this stops after the **first saturated contact shell**, which covers the
solute surface. `--padding P` keeps growing saturated shells until the shell reaches
`P` Å; `--n-molecules N` instead keeps exactly N (see drivers below).

### Shape-aware packing for linear/elongated solvents
Candidate spacing and shell offsets use an **effective cross-sectional radius**
(the molecule's *width*, from its principal axes), not the spherical bounding
radius (its *length*). So a long thin solvent (ether, alkane) packs tightly along
the surface instead of being held a full molecular-length apart. For compact
solvents (water, DMSO) the effective radius ≈ the bounding radius, so packing is
unchanged. The full-atom vdW check still prevents clashes.

## Drivers

| Driver | Meaning | Flags |
|--------|---------|-------|
| **Cover** (default) | fill+relax until the contact shell is saturated (wraps the solute) | *(none)* |
| **Padding** | grow saturated shells until `P` Å of solvent thickness | `--padding 5.0` |
| **Count** | keep exactly N solvents (dense shell fill → trim to N closest) | `--n-molecules 30` |
| **Manual** | explicit per-shell counts (sequential shells) | `--layers 4 8 12` |

**Count mode** (`--n-molecules N`) runs the same dense, shape-aware shell fill
as cover mode, growing concentric shells until at least N molecules are placed,
then trimming the final fill to the N closest to the solute. The result packs N
tightly against the molecular surface (liquid-like spacing), and works for any
solvent size or shape.

## Optimization modes (`--opt-mode`)

| Mode | Behaviour |
|------|-----------|
| `layerwise` | **DEFAULT** — iterative fill→relax→refill cover (above). Needs a working potential (SO3LR/MACE); slower. |
| `after-all` | fill all shells, then one global relax at the end |
| `none`      | placement only — no relax (fast, pure geometry) |

`--method` picks the optimizer (`FIRE`, `LBFGS`, `GD`, or `HYBRID` =
FIRE coarse → LBFGS fine). `--freeze-mode {solute,solvent,none}` controls
what's held fixed.

## Topology repair (post-relaxation integrity)

Relaxation can occasionally **break** a solvent molecule — a bond stretches past
the covalent threshold (it fragments / an atom dissociates), or it forms a
covalent bond to a neighbour or the solute (it reacts). After relaxation each
solvent molecule's bond graph (covalent radii, via `detect_bonds`) is compared
to the reference solvent; hydrogen bonds sit above the covalent threshold and
are **not** counted, so they never trigger a false positive.

In the **layerwise/auto** build the check runs at two points:

1. **Layerwise** — after *each shell's* relaxation, broken molecules are removed
   in place and the refill cycle / next shell re-places them. You'll see a
   `shell N topology check: M intact` (or `removed K broken …`) line per shell.
2. **Final convergence loop** — after the whole build:

   > detect broken molecules → remove them → refill to the target (count mode) or
   > back to the previous count (cover mode) → re-relax → recheck

   repeating until a pass finds nothing broken (or `--max-repair` passes, after
   which any stragglers are dropped).

This keeps the requested molecule count while guaranteeing every solvent
molecule in the output is chemically intact. A final per-structure cleanup pass
also runs in **every** mode (so molecules broken by `after-all` or
`--final-relax` are removed too); any structure whose count dips is then
**refilled back to the target** so the saved ensemble keeps a uniform solvent
count (see *Batch uniformization* below).

## Batch uniformization

vmap batch optimization requires every conformer in a batch to share the same
atom count. Because the geometric/barostat builders can be geometry-limited at
different counts per conformer (and broken-solvent removal lowers counts), the
workflow uniformizes the batch before optimization and after the final cleanup,
across **all** build paths (independent / swap / barostat):

- `count == target` → untouched;
- `count > target` → trimmed to the molecules closest to the solute;
- `count < target` → repaired in up to **2 rounds** of remove-broken → geometric
  refill to the target (`--n-molecules`) → frozen-solute relax.

`--n-molecules` is the hard target. If a conformer is genuinely geometry-limited
and can't reach it, a warning is logged and the batch is trimmed to the minimum
achieved count so shapes stay uniform.

Disable with `--no-topology-repair`; tune with `--max-repair N` and
`--topo-tolerance T` (a bond is broken past `T × (r_i + r_j)`, default 1.3).
The API exposes [`remove_broken_solvent`](../api/solvation.md) for a one-shot
removal pass.

## Barostat mode — spherical moving-wall droplet (`--barostat`)

An alternative to surface placement: build a **pre-packed, solute-free solvent
droplet** and compress it with a moving wall (a crude barostat), then swap the
solute in. Requires `--n-molecules N`.

```
   ( outer "piston" wall — moves inward each step )
        . . . . . . . . . . .
      .   solvent  solvent    .
     .  solv   [ cavity ]  solv .     inner wall = solute cavity
      .   solvent  solvent    .       (solute swapped in afterwards)
        . . . . . . . . . . .
```

**Why.** The droplet is equilibrated *independently of the solute*, so one
compressed droplet can be transplanted across every conformer (`--solute-swap`)
— the equilibration cost is paid once. It also gives a more bulk-liquid-like
local environment than surface-by-surface placement.

**Swap = batch.** With `--solute-swap`, the droplet is built once from the
reference, all conformers are aligned and swapped into the cavity, and the final
packing runs as a **single vmapped batch with a per-structure moving wall**: each
structure's wall moves inward independently and **freezes when that structure
reaches `--baro-max-force`**; the loop continues until all have converged. The
wall-free relaxation is batched too. If the conformer batch is too large for GPU
memory it is split into sub-batches automatically (recursive halving), so large
ensembles just work without a batch-size flag.

**How it works.**

1. **Place** N solvent molecules loosely in a spherical shell `[r_inner, r_outer]`
   around the solute centroid (solute absent). `r_inner = solute_radius +
   --baro-inner-buffer`; `r_outer` is auto-sized from `--baro-fill` (loose start)
   or set with `--baro-radius`.
2. **Compress** with two walls combined as a `barrier_fn` on the relaxation:
   - **inner wall** `k·Σ max(0, 1 − rᵢ/r_inner)ⁿ` keeps the cavity clear;
   - **outer wall** `k·Σ max(0, rᵢ/r_outer − 1)ⁿ` is the piston.

   Each step: relax (FIRE) → measure the outer-wall **reaction force**
   `maxᵢ|∇ᵢE_outer|` → if it's below `--baro-max-force`, shrink `r_outer` and
   repeat. Stepping is **coarse** (`--baro-step`) down to a refine zone, then
   **fine** (`--baro-fine-step`, 0.1 Å) so the wall creeps inward and stops right
   at the force target rather than overshooting — it is **not** stopped by a
   fixed minimum radius (only a low collapse-safety backstop).
3. **Swap** the solute into the cavity (solute atoms first).
4. **Final packing** — drop the inner wall and keep compressing the outer wall
   inward with the **solute frozen** (coarse then fine). This is the stage that
   constrains the solvent *toward the solute*: the wall drives the first shell
   onto the solute surface, closing the cavity gap, until the wall reaction force
   is reached again.
5. **Final relaxation without the wall** — the wall is removed and the solvent is
   relaxed to its true minimum (solute still frozen), so the result is not biased
   by the barostat.
6. **Topology check** removes any broken solvent molecules; the workflow's batch
   uniformization then refills counts back to the target (geometric placement +
   relax), so barostat batches stay uniform.

`--baro-wall-mode harmonic` (n=2, soft, smooth force) is the default; `wall`
(n=12) is a stiffer boundary. Compression is optimization-driven (FIRE) by
default; `--baro-md` runs a short NVT MD with the wall in the energy at each step
(then a FIRE settle) so the solvent flows past itself and packs more like a
liquid — tune with `--baro-md-temp`, `--baro-md-time`, `--baro-md-dt`. See
[`solvate_barostat`](../api/solvation.md) for the Python API.

## Solvent library

`mars/solvents/library/` ships ~30 pre-built solvents. See the full list:

```bash
python -c "import mars.solvents; print(mars.solvents.list_solvents())"
```

Aliases work case-insensitively — `DCM`, `EtOH`, `DMSO`, `hexane`, etc.
all resolve to their canonical entry.

**Geometries.** A handful of solvents (water, methanol, acetonitrile, acetone,
benzene, n-hexane, DMSO, HF) ship curated `experimental` geometries; every other
entry is generated from its SMILES with Open Babel (`obabel --gen3d`, MMFF94),
recorded in `mars/solvents/manifest.toml`. To rebuild the generated geometries:

```bash
python tools/regen_solvents.py          # all non-experimental entries
python tools/regen_solvents.py --only thf dmf    # just a few
```

### Using your own solvent

Pass a path to a single-molecule `.xyz` anywhere a solvent name is accepted —
on the CLI (`--solvent ./my_solvent.xyz`) or in the API:

```python
from mars.solvents import load_solvent
from mars.solvation import solvate

# load_solvent resolves a library name/alias OR a file path
entry = load_solvent("./propylene_carbonate.xyz")   # source == "user-file"

# solvate()/auto_solvate() also accept a path string directly
solvated = solvate(solute, solvent="./propylene_carbonate.xyz", layers=[6, 12])
```

To reuse a custom solvent under a name (with metadata), register it at runtime:

```python
from mars.solvents import register_solvent
register_solvent(
    "propylene_carbonate",
    "/path/to/pc.xyz",
    formula="C4H6O3",
    density=1.21,
    dielectric=66.1,
    aliases=["pc"],
)
```

See [`mars.solvents`](../api/solvents.md) for the full API.

## Multi-conformer solvation

| Mode | Flag | Behaviour |
|------|------|-----------|
| Independent | `--all-conformers` | Re-solvate each frame from scratch |
| Transplant  | `--solute-swap --conformer 0` | Solvate the reference, then transplant the same shell onto every other conformer and re-optimize |

Transplant mode is much faster and gives lower between-conformer variance
than independent solvation when the reference is representative.

## Placement parameters

| Flag | Default | Effect |
|------|---------|--------|
| `--vdw-scale`    | 0.75 | vdW overlap threshold (lower = tighter packing / more vdW overlap pre-relax) |
| `--buffer`       | 0.0  | Extra gap (Å) beyond vdW contact |
| `--n-candidates` | 300  | Fibonacci candidate points per atom per radial band |
| `--n-shells`     | 3    | Number of radial candidate bands per atom |
| `--n-orient`     | 8    | Orientation tries per site (cover mode) — helps elongated solvents fit |
| `--refill-cycles`| 3    | Max fill→relax→refill cycles per shell (cover mode) |
| `--seed`         | 42   | Base random seed; layer *i* uses `seed + i` |

## See also

- [Recipes](../examples/solvation_recipes.md).
- [`[solvation]` config section](../config/solvation_section.md).
- API reference: [`mars.solvation`](../api/solvation.md),
  [`mars.solvents`](../api/solvents.md).
