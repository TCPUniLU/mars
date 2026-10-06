# Custom potentials

Register your own potential with the `@register_potential` decorator. It
only has to implement `_build_energy_fn()`, returning a callable that maps
positions (Å) to an energy (eV):

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

Once registered, the name works from the CLI: `mars input.xyz --potential
my_potential`, as long as the wrapper accepts whatever flags the CLI passes
through.

Optional extras:

- **Neighbor lists**: override `uses_neighbor_lists`, `allocate_neighbors`,
  `update_neighbors` and `get_neighbor_kwargs`. See the `lj`, `so3lr` and
  jax-md wrappers in `mars/potentials*.py`.
- **IR intensities**: make `build_energy_fn(with_charges=True)` return
  `(energy_fn, charges_fn)`. If the molecular dipole is more than Σ qᵢrᵢ,
  return a `mars.potentials.ChargesFunction(charges_fn, atomic_dipoles)`, where
  `atomic_dipoles(positions) -> (n_atoms, 3)` gives each atom's contribution in
  e·Å. AMOEBA does this.
- **Charge and spin**: accept `charge` and `spin` (multiplicity), and raise a
  `ValueError` for values the model cannot represent.

!!! question "Want to plug in your own potential? We're happy to help"
    MARS is model-agnostic by design, and we welcome new backends. If you get
    stuck wrapping your own force field or ML potential (the neighbor-list
    interface, partial charges for IR, `vmap` batching, or registering it),
    reach out and we'll help you get it working, and happily link or upstream
    it: <https://github.com/TCPUniLU/mars/issues>.
