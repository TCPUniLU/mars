"""
Positional constraint utility for MARS CLI.

Implements frozen-atom constraints by masking out force contributions.
The approach replaces the actual positions of the frozen atoms with their
reference positions *before* evaluating the base energy function.  As a
result, JAX's automatic differentiation yields identically-zero gradients
(forces) for those atoms — their energy and force contributions are
effectively masked to zero regardless of where the integrator would place
them.

This works transparently for both:
- Optimization: zero gradient → the optimizer never updates frozen atoms.
- Metadynamics / MD: zero force → frozen atoms do not accelerate.
  (For MD, frozen atoms also need zero initial velocity; the MTD runner
  initialises velocities from a Maxwell-Boltzmann distribution, so atoms
  with zero force will still drift slightly from residual thermal velocity.
  If hard freezing during dynamics is required, the velocity mask should be
  applied inside the MD integrator, which is left as a future extension.)
"""

import jax.numpy as jnp

from ..optimizer import _mask_frozen_gradient


def make_frozen_energy_fn(energy_fn, reference_positions, frozen_indices):
    """Return a wrapped energy function that freezes *frozen_indices* atoms.

    Before evaluating *energy_fn* the frozen atoms are replaced with their
    coordinates in *reference_positions*.  The base energy function therefore
    sees those atoms as stationary, and JAX autodiff propagates zero gradient
    back to the original position array for those indices.

    Args:
        energy_fn: Callable ``(positions, **kwargs) -> scalar_eV``.
        reference_positions: Array ``(n_atoms, 3)`` in Angstrom — the
            positions that frozen atoms are pinned to.
        frozen_indices: Sequence of int — 0-based atom indices to freeze.

    Returns:
        A new callable with the same signature as *energy_fn*.
    """
    idx = jnp.array(frozen_indices, dtype=int)
    ref = jnp.array(reference_positions)

    def frozen_energy_fn(positions, **kwargs):
        # Pin frozen atoms to reference → their gradient is identically 0
        pinned = positions.at[idx].set(ref[idx])
        return energy_fn(pinned, **kwargs)

    return frozen_energy_fn


def make_frozen_energy_fn_batched(energy_fn, frozen_indices):
    """Return a vmap-compatible energy function that freezes *frozen_indices*.

    Unlike :func:`make_frozen_energy_fn` (which pins frozen atoms to a single
    external reference array), this version uses ``jax.lax.stop_gradient`` to
    zero out gradients for frozen atoms *in-place*.  Each element in a vmapped
    batch therefore keeps its own frozen-atom coordinates — no shared reference
    is needed.

    Args:
        energy_fn: Callable ``(positions, **kwargs) -> scalar_eV``.
        frozen_indices: Sequence of int — 0-based atom indices to freeze.

    Returns:
        A new callable with the same signature as *energy_fn*.
    """
    return _mask_frozen_gradient(energy_fn, frozen_indices)


def log_constraints(log_fn, constraint_atoms, apply_to):
    """Emit a concise log line about active constraints."""
    log_fn(
        f"Positional constraints: {len(constraint_atoms)} atoms frozen "
        f"(indices {constraint_atoms[:5]}"
        f"{'...' if len(constraint_atoms) > 5 else ''}"
        f", apply_to='{apply_to}')"
    )
