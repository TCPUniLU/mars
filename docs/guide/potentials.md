# Potentials

Every MARS subcommand accepts `--potential NAME`. The registered backends:

| Name | Source | Notes |
|------|--------|-------|
| `so3lr` *(default)* | SO3LR ML potential | Realistic chemistry; provides partial charges (needed for IR intensities); pick model via `--so3lr-model` |
| `mace`  | MACE family | JAX-native (via mace-jax); pick model via `--mace-foundation` / `--mace-model` |
| `dxtb`  | dxtb / GFNx-TB | Tight binding; `--dxtb-method gfn1` or `gfn2` |
| `nci`   | spherical confinement only | Used internally by `--nci` mode |
| `harmonic` | toy centroid spring | Cheap, no ML dependency — ideal for tests and CI |
| `lj`    | Lennard-Jones | Pair potential for diagnostic tests |

## Picking and configuring

```bash
mars input.xyz --potential mace --mace-foundation off --mace-model small
mars input.xyz --potential dxtb --dxtb-method gfn2
mars optimize input.xyz --potential harmonic   # no ML deps required
```

SO3LR is the default potential. MARS works with **two SO3LR packages**, and
detects which one is installed automatically:

- **Stable `so3lr`** (v0.1.x) — ships a single **v1** model. This is the safe
  default; `--so3lr-model` is ignored (with a warning) since no other model is
  available.
- **Developing `so3lr` package** (`so3lr_dev`, ≥ 0.2) — adds a **v2** model
  registry selectable with `--so3lr-model`. The v2 models are still under active
  development.

| `--so3lr-model` | Model | Requires |
|-----------------|-------|----------|
| `so3lr-1` *(default)* | legacy v1 | either package |
| `so3lr-2-s`  | v2 small | developing package |
| `so3lr-2-m`  | v2 medium (recommended) | developing package |
| `so3lr-2-l`  | v2 large (5 Å short-range cutoff) | developing package |
| *(a path)* | custom / fine-tuned model workdir | developing package |

The pre-release names `so3lr_v1`, `so3lr`, `so3lr-s`, `so3lr-m`, `so3lr-l`
still resolve to the same models above but are deprecated and emit a
`DeprecationWarning`; use the names in the table for new work.

```bash
mars input.xyz --potential so3lr                             # v1 (works with either package)
mars input.xyz --potential so3lr --so3lr-model so3lr-2-l     # v2 (developing package only)
mars input.xyz --potential so3lr --so3lr-model /path/to/finetuned
```

If you request a v2 model while only the stable package is installed, MARS
prints a warning and falls back to the v1 model rather than failing.

`--lr-cutoff Å` controls the long-range cutoff (1000 Å for gas phase, ~12 Å
for periodic systems).

!!! warning "Supported MACE variants"
    The JAX-native MACE backend converts the PyTorch checkpoint with
    `mace-jax`. Supported variants:

    | Foundation | Backend | Status |
    |------------|---------|--------|
    | `off`  | JAX-native | ✅ MACE-OFF23 — `small`, `medium`, `large` |
    | `off24`| JAX-native | ✅ MACE-OFF24 — `medium` only (the only size published upstream) |
    | `mp`   | JAX-native | ✅ `small`, `medium`, `large` (and `medium-mpa-0`, …) |
    | `anicc`| JAX-native | ✅ supported |
    | `omol` | **PyTorch (ASE)** | ✅ via fallback (see below) |

    `off24` needs no extra setup: `--mace-model` defaults to `medium` and is
    resolved to the upstream checkpoint, which is downloaded once and cached
    as converted JAX weights like any other family. A local checkpoint path or
    URL may be passed instead.

    ```bash
    mars input.xyz --potential mace --mace-foundation off24
    ```

    The JAX-native families are validated against the PyTorch reference
    (energies match to ~1e-6 eV, forces to ~1e-7 eV/Å). **`omol` uses NonLinear
    interaction blocks that `mace-jax` does not yet convert faithfully (~1 eV
    per atom drift), so it runs on MACE's native PyTorch (ASE) calculator
    instead**, with forces bridged into JAX via `jax.custom_vjp`. Consequences
    for `omol`:

    - Requires `mace-torch` + `torch` installed.
    - Analytical Hessians are unavailable, so IR automatically uses finite
      differences (`--fd-hessian`).
    - It is charge/spin-aware: pass `--charge` and `--spin` (spin multiplicity,
      default 1) — these are forwarded to the calculator.

## Custom potentials

Register your own with the `@register_potential` decorator. It only
needs to implement `_build_energy_fn()` returning a positions → eV
callable:

```python
import jax.numpy as jnp
from mars import register_potential, PotentialWrapper, get_potential

@register_potential("my_potential")
class MyPotential(PotentialWrapper):
    def __init__(self, species, k=0.1, **kwargs):
        self.k = k

    def _build_energy_fn(self):
        k = self.k
        def energy_fn(positions, **kwargs):
            centre = jnp.mean(positions, axis=0)
            return jnp.sum((positions - centre) ** 2) * k
        return energy_fn

pot = get_potential("my_potential", species=numbers, k=0.05)
energy_fn = pot.build_energy_fn()
```

Once registered the new name is usable from the CLI: `mars input.xyz
--potential my_potential` (subject to the wrapper accepting whatever
flags the CLI passes through).

!!! question "Want to plug in your own potential? We're happy to help"
    MARS is model-agnostic by design and we actively welcome new backends. If
    you'd like to wrap your own force field / ML potential and hit anything —
    the neighbour-list interface, partial charges for IR, `vmap` batching, or
    getting it registered — reach out and we'll help you get it working (and
    happily link or upstream it):

    - Issues / discussions: <https://github.com/TCPUniLU/mars/issues>

## See also

- API reference: [`mars.potentials`](../api/potentials.md).
- [NCI mode](nci_mode.md) for the confinement potential.
