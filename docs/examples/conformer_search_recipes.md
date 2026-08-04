# Conformational search recipes

## Modes

```bash
# Auto-configured normal search (default)
mars input.xyz

# Quick mode (less thorough, faster)
mars input.xyz --quick

# Thorough mode (more cycles, more replicas)
mars input.xyz --thorough

# Full conformational sampling with genetic crossing
mars input.xyz --thorough --genetic-crossing --n-children 30
```

## Workflow variants

```bash
# Metadynamics only — no rotamer MD, no crossing
mars input.xyz --mtd-only

# Plain MD — no metadynamics bias
mars input.xyz --md-only --mtd-time 20.0
```

## Charged / non-covalent systems

```bash
# Charged molecule with a tight energy window
mars input.xyz --charge -1.0 --ewin 4.0

# Host–guest / ion-pair — spherical confinement during sampling
mars complex.xyz --nci --nci-buffer 3.0 --nci-mode wall
mars complex.xyz --nci --nci-mode harmonic --nci-max-dist 4.0
```

See [NCI mode](../guide/nci_mode.md) for context.

## Tuning the sampler

```bash
# Override MTD parameters explicitly
mars input.xyz --mtd-cycles 3 --mtd-time 15.0 --mtd-dt 0.5 \
    --kpush 18.0 --alpha 0.6 --cvdump-fs 50.0

# Skip the trial-MTD validation step (faster but riskier)
mars input.xyz --no-trial

# Disable bond-topology pruning (allows bond-breaking conformers through)
mars input.xyz --no-topology
```

## Choosing the optimizer

Every optimization cycle in the workflow (initial relax + the 3-cycle
multilevel ladder after each sampling phase) honours the same method
selector as `mars optimize`. The flags below tune the **final** cycle;
coarse pre-cycles stay scaled at 50× / 10× of the final `fmax` with
10 / 100 iterations and 1.0 / 0.5 Å step caps.

```bash
# FIRE (jax-md) — robust when forces are noisy (e.g. dxtb pure callbacks)
mars input.xyz --method FIRE

# Gradient descent with a tight step cap
mars input.xyz --method GD --max-stepsize 0.05

# LBFGS (default) with tighter convergence in the final cycle
mars input.xyz --method LBFGS --fmax 0.005 --maxiter 2000

# Combine with FIRE-specific knobs (only consulted when --method FIRE)
mars input.xyz --method FIRE --fire-dt-start 0.05 --fire-dt-max 0.1 --fire-n-min 3
```

```toml
# run.toml — set the optimizer through the config file
[conformer_search]
mode = "thorough"
method = "FIRE"
fmax = 0.005
maxiter = 2000
```

## Output and diagnostics

```bash
# Custom output file + save every MD trajectory as XYZ
mars input.xyz -o my_ensemble.xyz --save-trajectory

# Debug logging
mars input.xyz --log-level DEBUG --log-file run.log
mars input.xyz --debug                # alias for DEBUG + warnings
```

## Driving everything from a config file

```toml
# run.toml
[conformer_search]
mode = "thorough"
genetic_crossing = true
n_children = 30
ewin = 6.0
charge = -1.0
output = "ensemble.xyz"

[constraints]
atoms = [0, 1, 2]
k = 100.0
apply_to = "all"
```

```bash
mars input.xyz --config run.toml
mars input.xyz --config run.toml --mode quick   # CLI overrides config
```

## Python API

```python
from mars import (
    load_structure, get_potential,
    run_conformer_search_auto,
)

structure = load_structure("input.xyz")
potential = get_potential("so3lr", species=structure["numbers"], charge=0.0)
energy_fn = potential.build_energy_fn()

ensemble = run_conformer_search_auto(structure, energy_fn, mode="normal")
print(f"Found {len(ensemble)} conformers")
for i, (struct, energy) in enumerate(ensemble[:5]):
    print(f"  {i+1}: E = {energy:.4f} eV")
```

See also: [User guide](../guide/conformer_search.md),
[CLI reference](../cli/conformer_search.md),
[`mars.conf_sampling`](../api/conf_sampling.md).
