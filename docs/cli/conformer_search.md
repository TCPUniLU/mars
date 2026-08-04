# `mars` (conformational search)

The default subcommand. Runs the multi-step MTD + rotamer MD + optional
genetic-crossing workflow.

```bash
mars <input.xyz> [OPTIONS]
```

See the [user guide](../guide/conformer_search.md) for concepts and
[recipes](../examples/conformer_search_recipes.md) for copy-pasteable
command lines.

## Search mode

| Flag | Default | Description |
|------|---------|-------------|
| `--quick` / `--thorough` | (mutually exclusive) | Pick a preset mode |
| `--mode {quick,normal,thorough}` | `normal` | Explicit mode (overrides --quick/--thorough) |
| `--n-conformer-starts N` | auto | Number of diverse conformer starting points |
| `--exploration-time PS` | auto | Exploration phase MTD time |
| `--conformer-time PS` | auto | Conformer-MTD time per run |

## Workflow variants

| Flag | Description |
|------|-------------|
| `--mtd-only` | Skip rotamer MD and genetic crossing |
| `--md-only` | Plain NVT MD (no MTD bias, no rotamers, no crossing) |
| `--nci` | Apply spherical confinement (host–guest / multi-fragment) |
| `--nci-buffer Å` (3.0) | Distance from outermost atom to confinement wall |
| `--nci-mode {wall,harmonic}` (`wall`) | Confinement potential shape |
| `--nci-max-dist Å` (3.0) | Max inter-fragment drift before structure is discarded |

## Metadynamics parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--mtd-cycles N` | 2 | Number of MTD cycles |
| `--mtd-time PS` | 10.0 | MTD time per cycle |
| `--mtd-dt FS` | 0.5 | MTD timestep |
| `--kpush KCAL` | 20.0 | Gaussian hill height |
| `--alpha` | 0.5 | Gaussian width parameter |
| `--cvdump-fs FS` | 50.0 | Hill deposition interval |
| `--kscal` | 1.0 | Global kpush scaling factor (autoconf) |
| `--no-trial` | off | Skip trial-MTD validation |
| `--no-topology` | off | Disable bond-topology pruning |

## General parameters

| Flag | Default | Description |
|------|---------|-------------|
| `-T`, `--temperature K` | 300 | Temperature |
| `--optlevel {-3..2}` | auto | Optimization tightness level |
| `--ewin KCAL_MOL` | auto | Energy window for pruning |
| `--charge Q` | 0.0 | Total molecular charge |

## Optimizer

The workflow runs many local optimizations: an initial relax, a 3-cycle
multilevel ladder after every MTD/rotamer/genetic-crossing phase. The
following flags pick the optimizer and tune the **final** (fine) cycle.
Coarse pre-cycles stay scaled at 50× / 10× the final `fmax`, with 10 /
100 iterations and 1.0 / 0.5 Å step caps.

| Flag | Default | Description |
|------|---------|-------------|
| `--method {LBFGS,FIRE,GD,SP}` | `LBFGS` | Minimizer used in every optimization cycle. `LBFGS` (jaxopt) is the default; `FIRE` (jax-md) and `GD` are alternatives; `SP` runs a damped-Newton stationary-point search and is mostly relevant if you want to study transition-state-like geometries inside the workflow (for true saddle searches, prefer `mars optimize --task stationary`). |
| `--fmax eV/Å` | auto from `--optlevel` | Final-cycle force threshold. Coarse pre-cycles scale to 50× and 10× this value. |
| `--maxiter N` | 1000 | Final-cycle iteration cap. |
| `--max-stepsize Å` | 0.2 | Final-cycle per-atom step cap (LBFGS line-search bound / GD learning rate / SP trust radius; not used by FIRE). |

### FIRE-specific parameters (only used when `--method FIRE`)

| Flag | Default | Description |
|------|---------|-------------|
| `--fire-dt-start` | 0.05 | FIRE initial timestep |
| `--fire-dt-max` | 0.1 | FIRE maximum timestep |
| `--fire-n-min` | 2 | FIRE minimum positive-power steps |

## Rotamer MD

| Flag | Default | Description |
|------|---------|-------------|
| `--rotamer-structures N` | 5 | Number of starting structures for rotamer MD |
| `--rotamer-temps N` | 3 | Number of temperature replicas |
| `--rotamer-time PS` | 10.0 | Rotamer MD time |

## Genetic crossing

| Flag | Default | Description |
|------|---------|-------------|
| `--genetic-crossing` | off | Enable genetic crossing |
| `--n-children N` | 20 | Number of children per generation |

## Output

| Flag | Default | Description |
|------|---------|-------------|
| `-o`, `--output FILE` | `auto_final_ensemble.xyz` | Output ensemble file |
| `--save-trajectory` | off | Save MD trajectories as XYZ |

## Examples

```bash
mars input.xyz
mars input.xyz --thorough --genetic-crossing --n-children 30
mars input.xyz --mtd-only --mtd-cycles 3
mars input.xyz --md-only --mtd-time 20.0
mars input.xyz --nci --nci-buffer 3.0 --nci-mode wall
mars input.xyz --config run.toml

# Pick a different optimizer / tighten the final cycle
mars input.xyz --method FIRE
mars input.xyz --method GD --max-stepsize 0.05
mars input.xyz --method LBFGS --fmax 0.005 --maxiter 2000
```
