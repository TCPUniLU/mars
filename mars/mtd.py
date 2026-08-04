"""
JAX-native metadynamics with RMSD-based Gaussian hills bias.

Uses aligned reference frame for efficient bias calculation.
Fully JIT-compiled and differentiable via JAX autodiff.
"""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp


@dataclass
class MTDState:
    """MTD state with hill positions and parameters.

    Attributes:
        cv_xyz: (max_hills, n_atoms, 3) deposited hill positions
        reference_xyz: (n_atoms, 3) reference structure
        n_hills: Current number of hills
        kpush: Gaussian height (eV)
        alpha: Gaussian width (1/Å²)
        cvdump_interval: Steps between depositions
        step_counter: Current MD step
        max_hills: Allocated array size
    """

    cv_xyz: jnp.ndarray  # (max_hills, n_atoms, 3) centered positions
    reference_xyz: jnp.ndarray  # (n_atoms, 3) reference (first frame, centered)
    n_hills: int
    kpush: float
    alpha: float
    cvdump_interval: int
    step_counter: int
    max_hills: int


def mtdstate_flatten(state):
    # Arrays, traced integers, and per-replica parameters are dynamic children.
    # Only max_hills (used for array shapes) stays static.
    children = (
        state.cv_xyz,
        state.reference_xyz,
        state.n_hills,
        state.step_counter,
        state.kpush,
        state.alpha,
        state.cvdump_interval,
    )
    aux = (state.max_hills,)
    return children, aux


def mtdstate_unflatten(aux, children):
    cv_xyz, reference_xyz, n_hills, step_counter, kpush, alpha, cvdump_interval = children
    (max_hills,) = aux
    return MTDState(
        cv_xyz, reference_xyz, n_hills, kpush, alpha, cvdump_interval, step_counter, max_hills
    )


jax.tree_util.register_pytree_node(MTDState, mtdstate_flatten, mtdstate_unflatten)


def create_mtd_state(
    n_atoms: int,
    max_hills: int = 10000,
    kpush: float = 0.8,
    alpha: float = 0.5,
    cvdump_interval: int = 100,
    dtype=None,
    reference_xyz=None,
) -> MTDState:
    """Initialize MTD state.

    Args:
        n_atoms: Number of atoms
        max_hills: Maximum hills to store
        kpush: Gaussian height (eV)
        alpha: Gaussian width (1/Å²)
        cvdump_interval: Steps between depositions
        dtype: Float dtype (default: float32)
        reference_xyz: (n_atoms, 3) reference structure
    Returns:
        Initialized MTD state
    """
    if dtype is None:
        dtype = jnp.float32

    if reference_xyz is None:
        reference_xyz = jnp.zeros((n_atoms, 3), dtype=dtype)
    else:
        reference_xyz = jnp.asarray(reference_xyz, dtype=dtype)

    return MTDState(
        cv_xyz=jnp.zeros((max_hills, n_atoms, 3), dtype=dtype),
        reference_xyz=reference_xyz,
        n_hills=jnp.int32(0),
        kpush=jnp.array(kpush, dtype=dtype),
        alpha=jnp.array(alpha, dtype=dtype),
        cvdump_interval=jnp.int32(cvdump_interval),
        step_counter=jnp.int32(0),
        max_hills=max_hills,
    )


def add_hill(mtd_state: MTDState, positions: jnp.ndarray) -> MTDState:
    """Add a Gaussian hill at current positions.

    Args:
        mtd_state: Current MTD state
        positions: (n_atoms, 3) positions in reference frame

    Returns:
        Updated MTD state
    """
    new_cv_xyz = mtd_state.cv_xyz.at[mtd_state.n_hills].set(positions)
    return replace(mtd_state, cv_xyz=new_cv_xyz, n_hills=mtd_state.n_hills + 1)


def increment_step(mtd_state: MTDState) -> MTDState:
    """Increment step counter."""
    return replace(mtd_state, step_counter=mtd_state.step_counter + 1)


def should_deposit_hill(mtd_state: MTDState) -> bool:
    """Check if hill should be deposited (at cvdump_interval)."""
    return (mtd_state.step_counter % mtd_state.cvdump_interval) == 0


# ============================================================================
# Bias Computation
# ============================================================================


@jax.jit
def compute_mtd_bias(positions: jnp.ndarray, mtd_state: MTDState) -> float:
    """Compute MTD bias from all hills (with logistic damping on the newest).

    Optimized with fused operations and vectorized masking.
    Fully differentiable via jax.grad for forces.

    Args:
        positions: (n_atoms, 3) positions in reference frame
        mtd_state: Current MTD state

    Returns:
        Total bias energy (eV)
    """
    max_hills = mtd_state.cv_xyz.shape[0]
    n_atoms = positions.shape[0]
    hill_idx = jnp.arange(max_hills)

    is_active = hill_idx < mtd_state.n_hills

    diff = mtd_state.cv_xyz - positions
    msd = jnp.sum(diff * diff, axis=(1, 2)) / n_atoms
    rmsd = jnp.sqrt(msd)
    energies = mtd_state.kpush * jnp.exp(-mtd_state.alpha * (rmsd * rmsd))

    # Logistic ramp: w goes 0→1 since the newest hill was deposited
    # r = log(3) / (0.5 * cvdump_interval) so w = 0.5 at half the interval
    Ndep = jnp.maximum(mtd_state.cvdump_interval, 1)
    r = jnp.maximum(jnp.log(3.0) / (0.5 * Ndep), 0.03)
    t = jnp.asarray(mtd_state.step_counter % mtd_state.cvdump_interval, dtype=energies.dtype)
    w_newest = 2.0 / (1.0 + jnp.exp(-r * t)) - 1.0  # in (0, 1)

    # Apply weight ONLY to the newest active hill; others stay at 1.0
    newest_mask = (hill_idx == (mtd_state.n_hills - 1)) & is_active
    weights = jnp.where(newest_mask, w_newest, 1.0)  # (H,)

    # # Mask out inactive hills and sum
    masked_energies = jnp.where(is_active, weights * energies, 0.0)

    return jnp.sum(masked_energies)


# ============================================================================
# Statistics and Diagnostics
# ============================================================================


def get_mtd_statistics(mtd_state: MTDState) -> dict:
    """Get MTD statistics for monitoring and diagnostics.

    Args:
        mtd_state: MTD state

    Returns:
        stats: Dictionary with statistics
    """
    total_memory = mtd_state.cv_xyz.nbytes
    return {
        "n_hills": mtd_state.n_hills,
        "kpush": mtd_state.kpush,
        "alpha": mtd_state.alpha,
        "cvdump_interval": mtd_state.cvdump_interval,
        "step_counter": mtd_state.step_counter,
        "max_hills": mtd_state.max_hills,
        "array_size": mtd_state.cv_xyz.shape[0],
        "memory_mb": total_memory / 1024**2,
        "hills_remaining": mtd_state.max_hills - mtd_state.n_hills,
    }


# ============================================================================
# Persistence (Save/Load Hills)
# ============================================================================


def save_mtd_state(mtd_state: MTDState, filename: str):
    """Save MTD state to file.

    Args:
        mtd_state: MTD state to save
        filename: Output file path (HDF5 or NPZ)
    """
    import numpy as np

    if filename.endswith(".npz"):
        np.savez(
            filename,
            cv_xyz=np.array(mtd_state.cv_xyz),
            reference_xyz=np.array(mtd_state.reference_xyz),
            n_hills=mtd_state.n_hills,
            kpush=mtd_state.kpush,
            alpha=mtd_state.alpha,
            cvdump_interval=mtd_state.cvdump_interval,
            step_counter=mtd_state.step_counter,
            max_hills=mtd_state.max_hills,
        )
    else:
        # Assume HDF5
        import h5py

        with h5py.File(filename, "w") as f:
            f.create_dataset("cv_xyz", data=np.array(mtd_state.cv_xyz))
            f.create_dataset("reference_xyz", data=np.array(mtd_state.reference_xyz))
            f.attrs["n_hills"] = mtd_state.n_hills
            f.attrs["kpush"] = mtd_state.kpush
            f.attrs["alpha"] = mtd_state.alpha
            f.attrs["cvdump_interval"] = mtd_state.cvdump_interval
            f.attrs["step_counter"] = mtd_state.step_counter
            f.attrs["max_hills"] = mtd_state.max_hills


def load_mtd_state(filename: str) -> MTDState:
    """Load MTD state from file.

    Args:
        filename: Input file path

    Returns:
        mtd_state: Loaded MTD state
    """
    import numpy as np

    if filename.endswith(".npz"):
        data = np.load(filename)
        cv_xyz = jnp.array(data["cv_xyz"])
        reference_xyz = jnp.array(data["reference_xyz"])
        max_hills = int(data["max_hills"]) if "max_hills" in data else cv_xyz.shape[0]

        return MTDState(
            cv_xyz=cv_xyz,
            reference_xyz=reference_xyz,
            n_hills=int(data["n_hills"]),
            kpush=float(data["kpush"]),
            alpha=float(data["alpha"]),
            cvdump_interval=int(data["cvdump_interval"]),
            step_counter=int(data["step_counter"]),
            max_hills=max_hills,
        )
    else:
        # Assume HDF5
        import h5py

        with h5py.File(filename, "r") as f:
            cv_xyz = jnp.array(f["cv_xyz"][:])
            reference_xyz = jnp.array(f["reference_xyz"][:])
            max_hills = int(f.attrs["max_hills"]) if "max_hills" in f.attrs else cv_xyz.shape[0]

            return MTDState(
                cv_xyz=cv_xyz,
                reference_xyz=reference_xyz,
                n_hills=int(f.attrs["n_hills"]),
                kpush=float(f.attrs["kpush"]),
                alpha=float(f.attrs["alpha"]),
                cvdump_interval=int(f.attrs["cvdump_interval"]),
                step_counter=int(f.attrs["step_counter"]),
                max_hills=max_hills,
            )
