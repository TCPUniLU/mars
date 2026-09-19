"""
ML-potential for MARS.

Provides a registry-based system for ML and test potentials.
New potentials can be added by:
    1. Subclassing PotentialWrapper
    2. Decorating with @register_potential("name")
    3. Implementing _build_energy_fn() (and optionally neighbor list methods)

Example — adding a custom potential::

    @register_potential("my_potential")
    class MyPotential(PotentialWrapper):
        def __init__(self, species, **kwargs):
            ...
        def _build_energy_fn(self):
            def energy_fn(positions, **kwargs):
                return ...  # scalar energy in eV
            return energy_fn
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# ============================================================================
# Base Potential Wrapper
# ============================================================================


class PotentialWrapper(ABC):
    """Base class for potential energy surface wrappers.

    Subclasses must implement ``_build_energy_fn`` which returns a
    JAX-differentiable function with signature::

        energy_fn(positions, **kwargs) -> scalar

    Potentials that support partial charges should override
    ``build_energy_fn`` so that ``build_energy_fn(with_charges=True)``
    returns ``(energy_fn, charges_fn)``.  The default implementation
    issues a warning and returns ``(energy_fn, None)``.

    Potentials that require neighbor lists should also override the
    ``uses_neighbor_lists`` property and the ``allocate_neighbors``,
    ``update_neighbors``, and ``get_neighbor_kwargs`` methods.
    """

    def initialize(self, positions: jnp.ndarray):
        """Initialize with reference positions (e.g. allocate neighbor lists).

        Called once after construction with the starting geometry.
        """
        pass

    @abstractmethod
    def _build_energy_fn(self) -> Callable:
        """Return a JAX-differentiable energy function.

        Returns:
            Callable ``(positions, **kwargs) -> scalar_energy``
        """
        ...

    def build_energy_fn(self, with_charges: bool = False):
        """Return energy function, optionally with a charges function.

        When *with_charges* is ``True`` and the potential supports partial
        charges, returns ``(energy_fn, charges_fn)``.  Otherwise a warning
        is issued and ``(energy_fn, None)`` is returned.

        Args:
            with_charges: If True, also return a charges function.

        Returns:
            ``energy_fn`` when *with_charges* is False, or
            ``(energy_fn, charges_fn)`` when True (charges_fn may be None).
        """
        energy_fn = self._build_energy_fn()
        if with_charges:
            warnings.warn(
                f"{type(self).__name__} does not support partial charges. "
                "IR intensities cannot be computed.",
                UserWarning,
            )
            return energy_fn, None
        return energy_fn

    # ------------------------------------------------------------------
    # Neighbor list interface (override for potentials that need them)
    # ------------------------------------------------------------------

    @property
    def uses_neighbor_lists(self) -> bool:
        """Whether this potential requires neighbor lists."""
        return False

    def allocate_neighbors(self, positions: jnp.ndarray) -> Any:
        """Allocate initial neighbor list state from positions."""
        return None

    def update_neighbors(self, positions: jnp.ndarray, nbr_state: Any) -> Any:
        """Update neighbor lists for new positions. Returns updated state."""
        return nbr_state

    def get_neighbor_kwargs(self, nbr_state: Any) -> Dict[str, Any]:
        """Return kwargs dict to pass to energy_fn for the given neighbor state."""
        return {}

    @property
    def species(self):
        """Atomic numbers this potential was built for, or ``None``.

        Subclasses that are constructed with a ``species`` argument store it
        and expose it here, so callers that only hold a ``PotentialWrapper``
        (notably :func:`mars.optimizer.optimize_single`) can recover the
        elements without threading them through separately.  Toy potentials
        that are species-agnostic return ``None``.

        Returns:
            ``(n_atoms,)`` array-like of atomic numbers, or ``None``.
        """
        return getattr(self, "_species", None)

    @property
    def analytical_hessian(self) -> bool:
        """Whether this potential supports analytical (AD-based) Hessians.

        Potentials that use ``jax.pure_callback`` (PyTorch fallbacks) cannot
        propagate forward-mode (JVP) differentiation, so ``jax.hessian``
        fails.  Override to return ``False`` to trigger automatic FD fallback.
        """
        return True


# ============================================================================
# Potential Registry
# ============================================================================

_POTENTIAL_REGISTRY: Dict[str, type] = {}


def register_potential(name: str):
    """Decorator to register a potential wrapper class.

    Usage::

        @register_potential("my_pot")
        class MyPotential(PotentialWrapper):
            ...
    """

    def decorator(cls):
        _POTENTIAL_REGISTRY[name] = cls
        return cls

    return decorator


def get_potential(name: str, **kwargs) -> PotentialWrapper:
    """Create a potential wrapper by registered name.

    Args:
        name: Registered potential name (e.g. ``"harmonic"``, ``"so3lr"``).
        **kwargs: Forwarded to the wrapper constructor.

    Returns:
        A ``PotentialWrapper`` instance.

    Raises:
        ValueError: If *name* is not registered.
    """
    if name not in _POTENTIAL_REGISTRY:
        available = ", ".join(sorted(_POTENTIAL_REGISTRY.keys()))
        raise ValueError(f"Unknown potential '{name}'. Available: {available}")
    return _POTENTIAL_REGISTRY[name](**kwargs)


def list_potentials() -> List[str]:
    """Return sorted list of registered potential names."""
    return sorted(_POTENTIAL_REGISTRY.keys())


# ============================================================================
# Built-in Test Potentials
# ============================================================================


@register_potential("harmonic")
class HarmonicPotential(PotentialWrapper):
    """Simple harmonic test potential (spring to centroid).

    Does not use neighbor lists (no pairwise interactions), but implements
    the interface for compatibility with the sampling functions.
    """

    def __init__(self, **kwargs):
        self._initialized = False
        self._dummy_nbr = None

    def initialize(self, positions: jnp.ndarray):
        """Dummy initialization for interface compatibility."""
        self._dummy_nbr = jnp.array([0])  # Dummy neighbor state
        self._initialized = True

    def _build_energy_fn(self):
        if not self._initialized:
            raise RuntimeError(
                "Harmonic potential not initialized. Call initialize(positions) first."
            )

        def energy_fn(positions, **kwargs):
            center = jnp.mean(positions, axis=0)
            return jnp.sum((positions - center) ** 2) * 0.1

        return energy_fn

    @property
    def uses_neighbor_lists(self) -> bool:
        return True  # For interface compatibility

    def allocate_neighbors(self, positions: jnp.ndarray):
        """Return dummy neighbor state (harmonic doesn't use pairwise interactions)."""
        return jnp.array([0])

    def update_neighbors(self, positions: jnp.ndarray, nbr_state):
        """Return unchanged dummy neighbor state."""
        return nbr_state

    def get_neighbor_kwargs(self, nbr_state) -> Dict[str, Any]:
        """Return empty kwargs (no neighbor data needed)."""
        return {}


@register_potential("lj")
class LJPotential(PotentialWrapper):
    """Lennard-Jones pairwise test potential with neighbor lists.

    Uses JAX-MD neighbor lists for efficient pairwise interactions.
    """

    def __init__(self, epsilon=0.1, sigma=2.5, cutoff=10.0, capacity_multiplier=1.25, **kwargs):
        """
        Args:
            epsilon: LJ energy parameter in eV
            sigma: LJ length parameter in Angstrom
            cutoff: Cutoff distance for neighbor list in Angstrom
            capacity_multiplier: Buffer multiplier for neighbor list allocation
        """
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff
        self.capacity_multiplier = capacity_multiplier
        self._neighbor_fn = None
        self._nbrs = None
        self._initialized = False
        self._box = None  # gas-phase (free space)

    def initialize(self, positions: jnp.ndarray):
        """Allocate neighbor lists from the starting geometry."""
        from jax_md import partition, space

        positions = jnp.array(positions)
        displacement_fn, shift_fn = space.free()

        # Gas phase: free space, no periodic box.  With ``disable_cell_list=True``
        # the neighbor list is all-pairs and the ``box`` is never used, but
        # ``neighbor_list`` unconditionally runs ``f32(box)`` (rejects ``None``).
        # Use the same large effective-box placeholder as SO3LR / MACE.
        self._neighbor_fn = partition.neighbor_list(
            displacement_fn,
            jnp.array(1e7),  # unused placeholder (free space, cell list disabled)
            r_cutoff=self.cutoff,
            dr_threshold=0.0,  # Update every step
            capacity_multiplier=self.capacity_multiplier,
            disable_cell_list=True,  # Use direct O(N²) for gas phase
        )

        self._nbrs = self._neighbor_fn.allocate(positions, box=self._box)
        self._initialized = True

    def _build_energy_fn(self):
        """Return JAX-differentiable LJ energy function using neighbor lists."""
        if not self._initialized:
            raise RuntimeError("LJ potential not initialized. Call initialize(positions) first.")

        eps, sig = self.epsilon, self.sigma
        nbrs_idx = self._nbrs.idx  # captured at build time (SO3LR/MACE pattern)

        def energy_fn(positions, neighbor=None, **kwargs):
            """Compute LJ energy using neighbor list.

            Args:
                positions: (n_atoms, 3) positions
                neighbor: optional neighbor indices (n_atoms, max_neighbors).
                    When omitted, the indices captured at build time are used.

            Returns:
                Scalar energy in eV
            """
            neighbor = neighbor if neighbor is not None else nbrs_idx

            # neighbor.idx has shape (n_atoms, max_neighbors)
            # where idx[i, k] is the k-th neighbor of atom i, or n_atoms if none
            n_atoms = positions.shape[0]

            # Vectorized computation over all pairs in neighbor list
            # Get atom i positions (n_atoms, max_neighbors, 3)
            i_positions = jnp.expand_dims(positions, axis=1)

            # Get neighbor positions (n_atoms, max_neighbors, 3)
            # Use mod to handle padding (indices == n_atoms wrap to 0)
            j_indices = neighbor % n_atoms
            j_positions = positions[j_indices]

            # Compute distances
            r_vec = i_positions - j_positions
            r_sq = jnp.sum(r_vec**2, axis=-1)
            r = jnp.sqrt(r_sq + 1e-10)

            # LJ potential
            sr6 = (sig / r) ** 6
            lj_pair = 4.0 * eps * (sr6**2 - sr6)

            # Mask out invalid pairs (padding and self-interactions)
            valid_mask = (neighbor < n_atoms) & (r > 1e-6)
            lj_pair = jnp.where(valid_mask, lj_pair, 0.0)

            # Sum over all pairs (divide by 2 since each pair counted twice)
            return 0.5 * jnp.sum(lj_pair)

        return energy_fn

    # ------------------------------------------------------------------
    # Neighbor list interface
    # ------------------------------------------------------------------

    @property
    def uses_neighbor_lists(self) -> bool:
        return True

    def allocate_neighbors(self, positions: jnp.ndarray, extra_capacity: int = 0):
        """Allocate fresh neighbor list."""
        positions = jnp.array(positions)
        nbrs = self._neighbor_fn.allocate(positions, box=self._box, extra_capacity=extra_capacity)
        return nbrs

    def update_neighbors(self, positions: jnp.ndarray, nbr_state):
        """Update neighbor list. Overflow is checked in the outer loop."""
        positions = jnp.array(positions)
        nbrs = nbr_state
        nbrs = nbrs.update(positions, neighbor=nbrs.idx, box=self._box)
        return nbrs

    def get_neighbor_kwargs(self, nbr_state) -> Dict[str, Any]:
        """Return kwargs for the energy function from a neighbor state."""
        return {"neighbor": nbr_state.idx}


# ============================================================================
# Valence Force Field (test potential with real bonded structure)
# ============================================================================


@register_potential("valence")
class ValencePotential(PotentialWrapper):
    """Minimal valence force field: harmonic bonds/angles, cosine torsions, soft LJ.

    Exists because the other two test potentials cannot exercise
    internal-coordinate machinery.  ``harmonic`` is a spring to the centroid,
    so its minimum has every atom collapsed onto one point — every bond length
    is zero and the internal-coordinate set is singular there — and its
    Cartesian Hessian is exactly quadratic and isotropic, which makes *any*
    coordinate transformation a pessimisation.  ``lj`` has no bonded terms at
    all, so covalent-radius bond detection returns nonsense connectivity.

    This potential builds its topology once from the reference geometry with
    :func:`mars.utils.detect_bonds`, so it has a chemically sensible minimum
    near the input structure and is cheap enough for CI.  It is a *test*
    potential, not a parameterised force field: the constants are generic and
    the energies are not physically meaningful.

    Args:
        species: ``(n_atoms,)`` atomic numbers.
        positions: ``(n_atoms, 3)`` reference geometry used to derive the
            topology and the equilibrium bond lengths / angles.
        k_bond: Bond force constant in eV/A^2 (default 20.0).
        k_angle: Angle force constant in eV/rad^2 (default 3.0).
        k_torsion: Torsion barrier in eV (default 0.05).
        epsilon: Non-bonded LJ well depth in eV (default 0.002).
        sigma: Non-bonded LJ length in Angstrom (default 3.0).
        tolerance: Covalent-radius tolerance for bond detection (default 1.3).

    Example:
        >>> from mars.potentials import get_potential
        >>> pot = get_potential("valence", species=numbers, positions=positions)
        >>> energy_fn = pot.build_energy_fn()
    """

    def __init__(
        self,
        species,
        positions=None,
        k_bond: float = 20.0,
        k_angle: float = 3.0,
        k_torsion: float = 0.05,
        epsilon: float = 0.002,
        sigma: float = 3.0,
        tolerance: float = 1.3,
        **kwargs,
    ):
        self._species = np.asarray(species, dtype=int)
        self.k_bond = float(k_bond)
        self.k_angle = float(k_angle)
        self.k_torsion = float(k_torsion)
        self.epsilon = float(epsilon)
        self.sigma = float(sigma)
        self.tolerance = float(tolerance)
        self._initialized = False
        self._topo = None
        if positions is not None:
            self.initialize(jnp.asarray(positions))

    def initialize(self, positions: jnp.ndarray):
        """Derive bonds/angles/torsions and their reference values."""
        from .topology import build_bonded_lists

        pos = np.asarray(positions, dtype=float)
        bonds, angles, torsions = build_bonded_lists(pos, self._species, tolerance=self.tolerance)
        self._topo = {
            "bonds": jnp.asarray(bonds, dtype=jnp.int32).reshape(-1, 2),
            "angles": jnp.asarray(angles, dtype=jnp.int32).reshape(-1, 3),
            "torsions": jnp.asarray(torsions, dtype=jnp.int32).reshape(-1, 4),
        }
        # Reference values from the input geometry.
        b = np.asarray(bonds, dtype=int).reshape(-1, 2)
        a = np.asarray(angles, dtype=int).reshape(-1, 3)
        self._r0 = jnp.asarray(
            np.linalg.norm(pos[b[:, 1]] - pos[b[:, 0]], axis=-1) if len(b) else np.zeros(0)
        )
        if len(a):
            u = pos[a[:, 0]] - pos[a[:, 1]]
            v = pos[a[:, 2]] - pos[a[:, 1]]
            u = u / np.linalg.norm(u, axis=-1, keepdims=True)
            v = v / np.linalg.norm(v, axis=-1, keepdims=True)
            self._theta0 = jnp.asarray(np.arccos(np.clip(np.sum(u * v, axis=-1), -1.0, 1.0)))
        else:
            self._theta0 = jnp.zeros(0)
        # Non-bonded pairs: everything separated by more than 3 bonds.
        n = len(self._species)
        excl = np.zeros((n, n), dtype=bool)
        for i, j in b:
            excl[i, j] = excl[j, i] = True
        for i, j, k in a:
            excl[i, k] = excl[k, i] = True
        for i, j, k, l in np.asarray(torsions, dtype=int).reshape(-1, 4):
            excl[i, l] = excl[l, i] = True
        iu, ju = np.triu_indices(n, k=1)
        keep = ~excl[iu, ju]
        self._nb_pairs = jnp.asarray(np.stack([iu[keep], ju[keep]], axis=-1), dtype=jnp.int32)
        self._initialized = True

    def _build_energy_fn(self) -> Callable:
        if not self._initialized:
            raise RuntimeError(
                "Valence potential not initialized. Pass positions= or call initialize()."
            )
        bonds = self._topo["bonds"]
        angles = self._topo["angles"]
        torsions = self._topo["torsions"]
        nb = self._nb_pairs
        r0, theta0 = self._r0, self._theta0
        kb, ka, kt = self.k_bond, self.k_angle, self.k_torsion
        eps, sig = self.epsilon, self.sigma

        def energy_fn(positions, **kwargs):
            e = jnp.asarray(0.0, dtype=positions.dtype)
            if bonds.shape[0]:
                d = jnp.linalg.norm(positions[bonds[:, 1]] - positions[bonds[:, 0]], axis=-1)
                e = e + kb * jnp.sum((d - r0) ** 2)
            if angles.shape[0]:
                u = positions[angles[:, 0]] - positions[angles[:, 1]]
                v = positions[angles[:, 2]] - positions[angles[:, 1]]
                u = u / jnp.linalg.norm(u, axis=-1, keepdims=True)
                v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
                cos = jnp.clip(jnp.sum(u * v, axis=-1), -1.0 + 1e-10, 1.0 - 1e-10)
                e = e + ka * jnp.sum((jnp.arccos(cos) - theta0) ** 2)
            if torsions.shape[0]:
                b1 = positions[torsions[:, 1]] - positions[torsions[:, 0]]
                b2 = positions[torsions[:, 2]] - positions[torsions[:, 1]]
                b3 = positions[torsions[:, 3]] - positions[torsions[:, 2]]
                n1 = jnp.cross(b1, b2)
                n2 = jnp.cross(b2, b3)
                b2n = b2 / (jnp.linalg.norm(b2, axis=-1, keepdims=True) + 1e-12)
                m = jnp.cross(n1, b2n)
                phi = jnp.arctan2(jnp.sum(m * n2, axis=-1), jnp.sum(n1 * n2, axis=-1))
                e = e + kt * jnp.sum(1.0 + jnp.cos(3.0 * phi))
            if nb.shape[0]:
                d = jnp.linalg.norm(positions[nb[:, 1]] - positions[nb[:, 0]], axis=-1)
                sr6 = (sig / jnp.maximum(d, 0.5)) ** 6
                e = e + 4.0 * eps * jnp.sum(sr6**2 - sr6)
            return e

        return energy_fn


# ============================================================================
# SO3LR Potential
# ============================================================================


@register_potential("so3lr")
class SO3LRPotential(PotentialWrapper):
    """SO3LR equivariant neural network potential.

    A short-range / long-range machine-learning potential for molecular
    simulations.  Uses JAX-MD neighbor lists internally.

    **Two SO3LR releases are supported transparently** — the loader introspects
    the installed package and adapts, so the same code runs in either
    environment:

    * **stable** ``so3lr`` (v0.1.x): a single bundled v1 model. Model selection
      is unavailable; ``--so3lr-model`` is ignored (with a warning) and the v1
      model is always used.
    * **developing** ``so3lr`` (the ``so3lr_dev`` package, >=0.2): adds a v2
      model registry selected through the ``model`` argument:

      * ``"so3lr-1"``   — v1, the legacy bundled model (default).
      * ``"so3lr-2-s"`` — v2 small.
      * ``"so3lr-2-m"`` — v2 medium (recommended for production).
      * ``"so3lr-2-l"`` — v2 large (5 Å short-range cutoff, 256 features).

      ``model`` may also be a filesystem path to a custom / fine-tuned model
      workdir, which is loaded directly.

      .. deprecated::
         The pre-release names ``"so3lr_v1"``, ``"so3lr"``, ``"so3lr-s"``,
         ``"so3lr-m"`` and ``"so3lr-l"`` still work and resolve to the same
         models as above, but emit a :class:`DeprecationWarning`; use the
         ``so3lr-1`` / ``so3lr-2-s/m/l`` names in new code.

    The developing v2 models are still under active development; the stable
    package and the v1 model remain the safe default.

    Reference:
        https://github.com/general-molecular-simulations/so3lr

    Installation::

        # 1. Install JAX matching your driver (GPU: cuda12 or cuda13)
        pip install -U 'jax[cuda13]'
        # ... or CPU-only:
        pip install -U jax

        # 2a. Stable SO3LR (v1 only)
        pip install so3lr
        # 2b. ...or the developing package for the v2 models (so3lr-s/-m/-l)
        pip install /path/to/so3lr_dev    # e.g. ../so3lr_dev-main
    """

    #: Map MARS-facing model names to SO3LR registry names.  Names not present
    #: here (any custom path) pass through unchanged to
    #: :func:`so3lr.model_registry.resolve_model`.
    _MODEL_ALIASES = {
        # current names
        "so3lr-1": "so3lr",
        "so3lr-2-s": "so3lr-s",
        "so3lr-2-m": "so3lr-m",
        "so3lr-2-l": "so3lr-l",
        # deprecated pre-release names, kept working -- see _DEPRECATED_MODEL_NAMES
        "so3lr_v1": "so3lr",
    }
    #: Names accepted for backward compatibility that raise a DeprecationWarning.
    #: "so3lr"/"so3lr-s"/"so3lr-m"/"so3lr-l" are also SO3LR's own registry keys
    #: (not MARS aliases -- they pass through _MODEL_ALIASES.get() unchanged),
    #: so they are listed here rather than in _MODEL_ALIASES.
    _DEPRECATED_MODEL_NAMES = {"so3lr_v1", "so3lr", "so3lr-s", "so3lr-m", "so3lr-l"}
    _DEFAULT_MODEL = "so3lr-1"

    def __init__(
        self,
        species,
        lr_cutoff: float = 1000.0,
        dtype=None,
        model: str = "so3lr-1",
        charge: float = 0.0,
        capacity_multiplier: float = 1.25,
        buffer_size_multiplier_sr: float = 1.25,
        buffer_size_multiplier_lr: float = 1.25,
        **kwargs,
    ):
        """
        Args:
            species: Array-like of atomic numbers, shape ``(n_atoms,)``.
            lr_cutoff: Long-range cutoff in Angstrom.
                Use 1000 for gas-phase (no PBC), 12 for periodic systems.
            dtype: Float precision (default ``jnp.float32``).
            model: Bundled model name (``"so3lr-1"``, ``"so3lr-2-s"``,
                ``"so3lr-2-m"``, ``"so3lr-2-l"``) or a path to a custom /
                fine-tuned model workdir.  ``None`` uses the default
                (``"so3lr-1"``). The pre-release names (``"so3lr_v1"``,
                ``"so3lr"``, ``"so3lr-s"``, ``"so3lr-m"``, ``"so3lr-l"``)
                still work but are deprecated.
            charge: Total system charge.
            capacity_multiplier: Accepted for backward compatibility; unused
                (SO3LR v2 sizes its neighbor lists from the per-list buffer
                multipliers).
            buffer_size_multiplier_sr: Buffer multiplier for SR neighbor list.
            buffer_size_multiplier_lr: Buffer multiplier for LR neighbor list.
        """
        try:
            import so3lr as _so3lr_mod
            from so3lr import So3lrPotential as _So3lrPotential

            # from so3lr import to_jax_md as _to_jax_md
            from so3lr.cli.so3lr_md import to_jax_md_custom as _to_jax_md
        except ImportError as exc:
            raise ImportError(
                "SO3LR is not installed or failed to import. Install with:\n"
                "  pip install -U 'jax[cuda12]'      # or jax[cuda13] / plain jax (CPU)\n"
                "  pip install /path/to/so3lr_dev     # e.g. ../so3lr_dev-main\n"
                "If it is installed, see the chained error above (an outdated "
                "orbax-checkpoint or flax is the usual cause; see docs/install.md)."
            ) from exc

        import jax
        from jax_md import space

        if dtype is None:
            dtype = jnp.float32
        if dtype in (jnp.float64, "float64"):
            jax.config.update("jax_enable_x64", True)

        self._species = jnp.array(species)
        self._lr_cutoff = lr_cutoff
        self._charge = charge
        self._box = None  # gas-phase; extend for PBC later

        patched_kwargs = {}
        if "compute_charges" in kwargs.keys():
            patched_kwargs.update(output_intermediate_quantities=["partial_charges"])

        # ---- resolve model name --------------------------------------------
        # A bundled name (so3lr-1/2-s/2-m/2-l), a deprecated pre-release name,
        # or a filesystem path.  The developing SO3LR package loads all of
        # these through ``model=``.
        if model is None:
            model = self._DEFAULT_MODEL
        elif model in self._DEPRECATED_MODEL_NAMES:
            replacement = self._MODEL_ALIASES.get(model, model)
            new_name = {
                v: k
                for k, v in self._MODEL_ALIASES.items()
                if k not in self._DEPRECATED_MODEL_NAMES
            }.get(replacement, replacement)
            warnings.warn(
                f"SO3LR model name {model!r} is deprecated and will be removed in a "
                f"future release; use {new_name!r} instead (same model).",
                DeprecationWarning,
                stacklevel=2,
            )
        resolved_model = self._MODEL_ALIASES.get(model, model)

        # ---- detect the installed SO3LR API --------------------------------
        # Two SO3LR releases are supported transparently, detected by
        # introspection (not a version pin) so both conda environments work:
        #   * stable ``so3lr`` (v0.1.x): one bundled v1 model — ``So3lrPotential``
        #     takes no ``model=`` argument, and ``to_jax_md_custom`` sizes cells
        #     with ``minimum_cell_size_multiplier_sr``/``_lr``.
        #   * developing ``so3lr`` (``so3lr_dev``, >=0.2): adds the v2 model
        #     registry (``model=``) and a single ``minimum_cell_size_multiplier``.
        import inspect

        so3lr_version = getattr(_so3lr_mod, "__version__", "?")
        pot_params = inspect.signature(_So3lrPotential).parameters
        supports_model = "model" in pot_params or "workdir" in pot_params

        pot_kwargs = dict(lr_cutoff=lr_cutoff, dtype=dtype, **patched_kwargs)
        if supports_model:
            pot_kwargs["model"] = resolved_model
        elif resolved_model != "so3lr":
            # Compare the alias-resolved name (not the raw ``model`` string) so
            # every v1 spelling -- so3lr-1, so3lr_v1, bare so3lr -- is silently
            # a no-op here; only a genuine v2 request is downgraded with a warning.
            warnings.warn(
                f"The installed SO3LR package (v{so3lr_version}) does not support "
                f"model selection; the requested --so3lr-model {model!r} is ignored and "
                "the bundled v1 model is used. Install the developing SO3LR package "
                "(so3lr_dev, >=0.2) to use the so3lr-2-s / so3lr-2-m / so3lr-2-l v2 models.",
                UserWarning,
                stacklevel=2,
            )

        # ---- build potential ------------------------------------------------
        potential = _So3lrPotential(**pot_kwargs)

        # ---- build jax-md interface ----------------------------------------
        displacement_fn, shift_fn = space.free()

        tjm_params = inspect.signature(_to_jax_md).parameters
        tjm_kwargs = dict(
            potential=potential,
            displacement_or_metric=displacement_fn,
            box=[],
            species=self._species,
            buffer_size_multiplier_sr=buffer_size_multiplier_sr,
            buffer_size_multiplier_lr=buffer_size_multiplier_lr,
            disable_cell_list=True,
            fractional_coordinates=False,
            total_charge=self._charge,
        )
        # The minimum-cell-size kwarg was split per neighbor list (sr/lr) in the
        # stable release and merged into one in the developing release.
        if "minimum_cell_size_multiplier" in tjm_params:
            tjm_kwargs["minimum_cell_size_multiplier"] = 1.0
        else:
            if "minimum_cell_size_multiplier_sr" in tjm_params:
                tjm_kwargs["minimum_cell_size_multiplier_sr"] = 1.0
            if "minimum_cell_size_multiplier_lr" in tjm_params:
                tjm_kwargs["minimum_cell_size_multiplier_lr"] = 1.0

        self._neighbor_fn, self._neighbor_fn_lr, self._energy_fn = _to_jax_md(**tjm_kwargs)

        self._nbrs = None
        self._nbrs_lr = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, positions: jnp.ndarray):
        """Allocate neighbor lists from the starting geometry.

        Must be called before ``build_energy_fn``.
        """
        positions = jnp.array(positions)
        self._nbrs = self._neighbor_fn.allocate(positions, box=self._box)
        self._nbrs_lr = self._neighbor_fn_lr.allocate(positions, box=self._box)
        self._initialized = True

    # ------------------------------------------------------------------
    # Energy function
    # ------------------------------------------------------------------

    def _build_energy_fn(self) -> Callable:
        """Return a JAX-differentiable SO3LR energy function (energy only).

        The returned function accepts optional ``neighbor`` and ``neighbor_lr``
        keyword arguments.  When provided, those values are used instead of the
        indices captured at build time.  This allows the optimizer to pass
        dynamically-updated neighbor lists through the energy/force computation.
        """
        if not self._initialized:
            raise RuntimeError(
                "SO3LR potential not initialized. " "Call initialize(positions) first."
            )
        nbrs_idx = self._nbrs.idx
        nbrs_lr_idx = self._nbrs_lr.idx
        box = self._box
        energy_fn = self._energy_fn

        def so3lr_energy(positions, neighbor=None, neighbor_lr=None, **kwargs):
            _nbr = neighbor if neighbor is not None else nbrs_idx
            _nbr_lr = neighbor_lr if neighbor_lr is not None else nbrs_lr_idx
            # total_charge is baked into the energy fn at construction time.
            return energy_fn(
                positions,
                neighbor=_nbr,
                neighbor_lr=_nbr_lr,
                box=box,
            )

        return so3lr_energy

    def build_energy_fn(self, with_charges: bool = False):
        """Return a JAX-differentiable SO3LR energy function.

        The neighbor list indices captured at build time are used as defaults.
        When the optimizer passes ``neighbor`` / ``neighbor_lr`` kwargs, those
        dynamic values are used instead, enabling neighbor list updates during
        optimization.
        """
        if not self._initialized:
            raise RuntimeError(
                "SO3LR potential not initialized. " "Call initialize(positions) first."
            )

        # Capture current state as local variables for the closure
        nbrs_idx = self._nbrs.idx
        nbrs_lr_idx = self._nbrs_lr.idx
        box = self._box

        if with_charges:
            energy_fn = self._energy_fn

            def so3lr_energy(positions, neighbor=None, neighbor_lr=None, **kwargs):
                _nbr = neighbor if neighbor is not None else nbrs_idx
                _nbr_lr = neighbor_lr if neighbor_lr is not None else nbrs_lr_idx
                return energy_fn(
                    positions,
                    neighbor=_nbr,
                    neighbor_lr=_nbr_lr,
                    box=box,
                    has_aux=False,
                )

            def charges_fn(positions, neighbor=None, neighbor_lr=None, **kwargs):
                _nbr = neighbor if neighbor is not None else nbrs_idx
                _nbr_lr = neighbor_lr if neighbor_lr is not None else nbrs_lr_idx
                energy, aux = energy_fn(
                    positions,
                    neighbor=_nbr,
                    neighbor_lr=_nbr_lr,
                    box=box,
                    has_aux=True,
                )
                return aux["partial_charges"]

            return so3lr_energy, charges_fn

        else:
            return self._build_energy_fn()

    # ------------------------------------------------------------------
    # Neighbor list interface (for advanced / periodic usage)
    # ------------------------------------------------------------------

    @property
    def uses_neighbor_lists(self) -> bool:
        return True

    def allocate_neighbors(self, positions: jnp.ndarray, extra_capacity: int = 0):
        """Allocate fresh neighbor lists. Returns ``(nbrs, nbrs_lr)``."""
        positions = jnp.array(positions)
        nbrs = self._neighbor_fn.allocate(positions, box=self._box, extra_capacity=extra_capacity)
        nbrs_lr = self._neighbor_fn_lr.allocate(
            positions, box=self._box, extra_capacity=extra_capacity
        )
        return (nbrs, nbrs_lr)

    def update_neighbors(self, positions: jnp.ndarray, nbr_state):
        """Update neighbor lists. Overflow is checked in the outer loop."""
        positions = jnp.array(positions)
        nbrs, nbrs_lr = nbr_state
        nbrs = nbrs.update(positions, box=self._box)
        nbrs_lr = nbrs_lr.update(positions, box=self._box)
        return (nbrs, nbrs_lr)

    def get_neighbor_kwargs(self, nbr_state) -> Dict[str, Any]:
        """Return kwargs for the energy function from a neighbor state."""
        nbrs, nbrs_lr = nbr_state
        return {
            "neighbor": nbrs.idx,
            "neighbor_lr": nbrs_lr.idx,
        }


# ============================================================================
# MACE Potential
# ============================================================================


@register_potential("mace")
class MACEPotential(PotentialWrapper):
    """MACE potential from pretrained foundation models.

    Two backends are selected automatically by foundation family:

    * **JAX-native** (``off``, ``mp``, ``anicc``) — the pretrained PyTorch
      checkpoint is converted once to a pure JAX/Flax model via ``mace-jax``
      and cached to disk (``config.json`` + ``params.msgpack``).  Subsequent
      runs reload directly in JAX with no PyTorch dependency.  Inference is
      fully differentiable (forces and Hessians via ``jax.grad`` /
      ``jax.hessian``) and uses JAX-MD neighbor lists, as in SO3LR.
    * **PyTorch (ASE)** (``omol``) — MACE-OMol uses NonLinear interaction
      blocks that ``mace-jax`` does not yet convert faithfully, so it runs on
      MACE's native ASE calculator, with forces bridged into JAX via
      ``jax.custom_vjp``.  In this mode analytical Hessians are unavailable
      (IR falls back to finite differences) and ``charge`` / ``spin`` are
      honoured.

    Foundation families (``foundation`` argument):

    * ``'off'``   — MACE-OFF23, organic molecules (default).
    * ``'off24'`` — MACE-OFF24, the 2024 organic-molecule release.  Only the
      ``medium`` size is published upstream, so ``model`` defaults to it and
      is resolved to the checkpoint URL; a local path or URL is passed
      through unchanged.
    * ``'mp'``    — MACE-MP, materials-project potential.
    * ``'anicc'`` — MACE trained on ANI-cc.
    * ``'omol'``  — MACE-OMol, broad chemical space (PyTorch backend).

    Requirements::

        pip install git+https://github.com/ACEsuit/mace      # mace-torch
        pip install git+https://github.com/ACEsuit/mace-jax  # JAX-native backends

    Args:
        species: Array-like of atomic numbers, shape ``(n_atoms,)``.
        foundation: Foundation family, one of ``'mp'``, ``'off'``,
            ``'off24'``, ``'anicc'``, ``'omol'`` (default ``'off'``).
        model: Variant name forwarded to the foundation loader, e.g.
            ``'small'`` / ``'medium'`` / ``'large'`` for ``off``/``mp``,
            ``'medium-mpa-0'`` for ``mp``, ``'extra_large'`` for ``omol``.
            May also be a local checkpoint path or URL.  Defaults to
            ``'small'`` for ``off``/``mp`` and to ``'medium'`` for ``off24``,
            the only size it publishes.
        dtype: Float precision.  Defaults to ``jnp.float32``.
        cache_dir: Directory for converted JAX weights (JAX-native backends).
            ``None`` uses ``~/.cache/mars/mace_jax``.
        charge: Total molecular charge.  Only used by charge-aware models
            (e.g. ``omol``); ignored otherwise.  Default ``0``.
        spin: Spin multiplicity (1 = singlet, 2 = doublet, ...).  Only used by
            spin-aware models (e.g. ``omol``); ignored otherwise.  Default ``1``.
    """

    _VALID_FOUNDATIONS = ("mp", "off", "off24", "anicc", "omol")

    #: MACE-OFF24 checkpoints. Upstream publishes only the medium size, and
    #: mace-torch has no ``mace_off24`` loader, so OFF24 is loaded through the
    #: OFF loader with an explicit checkpoint. Keyed by size so that
    #: ``--mace-foundation off24 --mace-model medium`` reads like the other
    #: families instead of requiring a URL.
    _OFF24_MODELS = {
        "medium": (
            "https://raw.githubusercontent.com/ACEsuit/mace-off/main/"
            "mace_off24/MACE-OFF24_medium.model"
        ),
    }

    @classmethod
    def _resolve_off24(cls, model):
        """Map an OFF24 size name to its checkpoint; pass paths/URLs through."""
        if model is None:
            return cls._OFF24_MODELS["medium"]
        key = str(model).lower()
        if key in cls._OFF24_MODELS:
            return cls._OFF24_MODELS[key]
        if key == "small":
            # "small" is the CLI-wide default for --mace-model, so it reaches
            # here whenever the user simply did not name a size. OFF24 has one
            # size, so fall through to it rather than refusing.
            return cls._OFF24_MODELS["medium"]
        if key == "large":
            raise ValueError(
                "MACE-OFF24 'large' is not published upstream; only 'medium' "
                "exists. Pass a local checkpoint path to use another."
            )
        # Already a path or URL.
        return model

    def __init__(
        self,
        species,
        foundation: str = "off",
        model: str | None = None,
        dtype=None,
        cache_dir: str | None = None,
        charge: float = 0.0,
        spin: float = 1.0,
        **kwargs,
    ):
        foundation = str(foundation).lower()
        if foundation not in self._VALID_FOUNDATIONS:
            raise ValueError(
                f"Unknown MACE foundation '{foundation}'. "
                f"Choose from: {', '.join(self._VALID_FOUNDATIONS)}"
            )

        # Per-family default size. OFF24 publishes only ``medium``, so an
        # unspecified model resolves there rather than to the ``small`` that
        # every other family defaults to; the size name is turned into the
        # checkpoint here, before it reaches the cache key or the OFF loader.
        if foundation == "off24":
            model = self._resolve_off24(model)
        elif model is None and foundation in ("off", "mp"):
            model = "small"

        if dtype is None:
            dtype = jnp.float32
        if dtype in (jnp.float64, "float64"):
            jax.config.update("jax_enable_x64", True)

        self._species = np.array(species, dtype=int)
        self._n_atoms = len(species)
        self._dtype = dtype
        self._charge = float(charge)
        self._spin = float(spin)
        self._initialized = False

        # OMol uses the PyTorch backend (see class docstring); all other
        # families use the JAX-native conversion path.
        if foundation == "omol":
            self._backend = "torch"
            self._torch_init(foundation, model, dtype)
        else:
            self._backend = "jax"
            self._load_or_convert(foundation, model, dtype, cache_dir)

    # ------------------------------------------------------------------
    # PyTorch / ASE backend (MACE-OMol only)
    # ------------------------------------------------------------------

    def _torch_init(self, foundation, model, dtype):
        """Initialise the native PyTorch ASE calculator for MACE-OMol.

        OMol is charge/spin-aware: the calculator reads ``atoms.info['charge']``
        and ``atoms.info['spin']`` (multiplicity), which we populate from the
        ``charge`` / ``spin`` constructor arguments.
        """
        try:
            import torch
            from mace.calculators import foundations_models
        except ImportError as exc:
            raise ImportError(
                "The PyTorch MACE backend (used for --mace-foundation omol) "
                "requires 'mace' (mace-torch) and torch.\n"
                "Install with: pip install git+https://github.com/ACEsuit/mace\n"
                f"(import error: {exc})"
            )

        loader = getattr(foundations_models, f"mace_{foundation}", None)
        if loader is None:
            raise ValueError(f"No PyTorch loader for MACE foundation '{foundation}'.")

        # OMol ships a single 'extra_large' variant.  The generic size default
        # (``model='small'``, meant for off/mp) is not a valid OMol checkpoint,
        # so map the generic size names to OMol's sole variant.  Explicit
        # ``extra_large`` / local paths / URLs pass through unchanged.
        if foundation == "omol" and model in (None, "small", "medium", "large"):
            model = "extra_large"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        default_dtype_str = "float64" if dtype in (jnp.float64, "float64") else "float32"
        loader_kwargs = {"device": device, "default_dtype": default_dtype_str}
        if model is not None:
            loader_kwargs["model"] = model

        self._calculator = loader(**loader_kwargs)

        underlying = getattr(self._calculator, "models", [None])[0]
        if underlying is None:
            underlying = getattr(self._calculator, "model", None)
        self._r_max = float(getattr(underlying, "r_max", 6.0))

    # ------------------------------------------------------------------
    # Model loading / conversion (JAX-native families)
    # ------------------------------------------------------------------

    @staticmethod
    def _default_cache_dir():
        """Default directory for cached converted JAX weights."""
        from pathlib import Path

        return Path("~/.cache/mars/mace_jax").expanduser()

    @staticmethod
    def _patch_cuequivariance():
        """Work around the cuequivariance 0.9.x ``SegmentedPolynomial.__eq__`` bug.

        The ``assert isinstance(value, SegmentedPolynomial)`` in ``__eq__``
        crashes during ``jax.grad`` tracing.  Returning ``NotImplemented`` for
        foreign types follows the Python data model and lets JAX's JVP
        machinery proceed.  No-op if cuequivariance is not installed.
        """
        try:
            from cuequivariance.segmented_polynomials.segmented_polynomial import (
                SegmentedPolynomial,
            )
        except ImportError:
            return

        _orig_eq = SegmentedPolynomial.__eq__

        def _safe_eq(self, value):
            if not isinstance(value, SegmentedPolynomial):
                return NotImplemented
            return _orig_eq(self, value)

        SegmentedPolynomial.__eq__ = _safe_eq

    @staticmethod
    def _model_slug(model) -> str:
        """Filesystem-safe cache token for a model name, path or URL.

        ``model`` may be a bare size (``"medium"``), a local checkpoint path or
        an https URL. Using it verbatim would put separators into the cache
        key and scatter the cache across nested directories, so keep the
        basename and hash the full value to keep distinct checkpoints distinct.
        """
        import hashlib

        if model is None:
            return "default"
        text = str(model)
        if "/" not in text and "\\" not in text:
            return text
        stem = text.rstrip("/").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        stem = "".join(c if c.isalnum() or c in "-._" else "_" for c in stem)[:48]
        digest = hashlib.sha1(text.encode()).hexdigest()[:8]
        return f"{stem}-{digest}" if stem else digest

    def _load_or_convert(self, foundation, model, dtype, cache_dir):
        """Load converted JAX weights from cache, or convert from torch once.

        On a cache hit the model is rebuilt purely in JAX (no torch import).
        On a miss the PyTorch foundation checkpoint is downloaded, converted
        via ``mace-jax``, and the result is serialized to the cache.
        """
        from pathlib import Path

        from flax import serialization

        x64 = bool(jax.config.jax_enable_x64)
        dtype_str = "float64" if x64 else "float32"
        cache_key = f"{foundation}-{self._model_slug(model)}-{'f64' if x64 else 'f32'}"
        base = Path(cache_dir).expanduser() if cache_dir else self._default_cache_dir()
        cache_path = base / cache_key
        config_file = cache_path / "config.json"
        params_file = cache_path / "params.msgpack"

        if config_file.exists() and params_file.exists():
            # ---- Cache hit: torch-free reload ------------------------------
            try:
                from mace_jax.tools.bundle import load_model_bundle
            except ImportError:
                raise ImportError(
                    "mace-jax is required for the JAX-native MACE potential.\n"
                    "Install with:\n"
                    "  pip install git+https://github.com/ACEsuit/mace-jax"
                )
            bundle = load_model_bundle(str(cache_path), dtype_str)
            self._graphdef = bundle.graphdef
            self._state = bundle.params
            config = bundle.config
        else:
            # ---- Cache miss: convert from torch, then cache ----------------
            try:
                import json

                import torch
                from mace.tools.scripts_utils import extract_config_mace_model
                from mace_jax.cli import mace_jax_from_torch as _mjft
                from mace_jax.cli.mace_jax_from_torch import convert_model
                from mace_jax.modules.wrapper_ops import CuEquivarianceConfig
                from mace_jax.nnx_utils import (
                    state_to_pure_dict,
                    state_to_serializable_dict,
                )
                from mace_jax.tools.foundation_models import load_foundation_torch_model

                # ``_serialize_for_json`` references a module-global ``torch``
                # that mace-jax only imports under TYPE_CHECKING; inject it.
                if getattr(_mjft, "torch", None) is None:
                    _mjft.torch = torch
                _serialize_for_json = _mjft._serialize_for_json
            except Exception as exc:
                raise ImportError(
                    "JAX-native MACE requires both 'mace' (mace-torch) and 'mace-jax'.\n"
                    "Install with:\n"
                    "  pip install git+https://github.com/ACEsuit/mace\n"
                    "  pip install git+https://github.com/ACEsuit/mace-jax\n"
                    f"(import error: {exc})"
                )

            self._patch_cuequivariance()

            try:
                # mace-torch ships no ``mace_off24`` loader; OFF24 is the OFF
                # loader pointed at the OFF24 checkpoint (resolved in __init__).
                source = "off" if foundation == "off24" else foundation
                torch_model = load_foundation_torch_model(
                    source=source,
                    model=model,
                    device="cpu",
                    default_dtype=dtype_str,
                )
            except TypeError as exc:
                # Some foundation loaders (e.g. mace_anicc) don't accept the
                # ``default_dtype`` kwarg that load_foundation_torch_model
                # forwards.  Retry without it.
                if "default_dtype" not in str(exc):
                    raise
                torch_model = load_foundation_torch_model(
                    source=source,
                    model=model,
                    device="cpu",
                )
            config = extract_config_mace_model(torch_model)
            if isinstance(config, dict) and "error" in config:
                raise RuntimeError(config["error"])
            config["torch_model_class"] = torch_model.__class__.__name__

            cueq_config = CuEquivarianceConfig(enabled=False)
            graphdef, state, _ = convert_model(
                torch_model,
                config,
                cueq_config=cueq_config,
            )
            self._graphdef = graphdef
            self._state = state_to_pure_dict(state)

            # ---- Write cache (mace-jax bundle format) ----------------------
            # Serialize via the serializable dict, which flattens ConfigDict
            # leaves that msgpack cannot pack directly.
            try:
                cache_path.mkdir(parents=True, exist_ok=True)
                params_file.write_bytes(serialization.to_bytes(state_to_serializable_dict(state)))
                config_file.write_text(json.dumps(_serialize_for_json(config), indent=2))
            except OSError as exc:
                warnings.warn(
                    f"Could not write MACE weight cache to {cache_path}: {exc}",
                    UserWarning,
                )

            del torch_model

        # ---- Species / z_table mapping (from config; both paths) -----------
        self._r_max = float(config["r_max"])
        atomic_numbers = config.get("atomic_numbers", [])
        if hasattr(atomic_numbers, "__len__") and len(atomic_numbers) > 0:
            self._z_table = sorted(int(z) for z in atomic_numbers)
        else:
            self._z_table = sorted(set(int(z) for z in self._species))
        self._num_species = len(self._z_table)
        self._z_to_index = {z: i for i, z in enumerate(self._z_table)}
        self._species_indices = jnp.array(
            [self._z_to_index[int(z)] for z in self._species],
            dtype=jnp.int32,
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, positions: jnp.ndarray):
        """Allocate the JAX-MD neighbor list and prepare backend state.

        Must be called before :meth:`build_energy_fn`.
        """
        from jax_md import partition, space

        positions = jnp.array(positions)
        n = self._n_atoms
        dt = self._dtype

        # Gas-phase neighbor list (SO3LR pattern).  With disable_cell_list the
        # box is unused, but neighbor_list rejects None — pass the same large
        # placeholder as SO3LR; allocate/update keep box=None.
        displacement_fn, _ = space.free()
        self._box = None
        self._neighbor_fn = partition.neighbor_list(
            displacement_fn,
            jnp.array(1e7),
            r_cutoff=self._r_max,
            dr_threshold=0.5,
            capacity_multiplier=1.25,
            disable_cell_list=True,
        )
        self._nbrs = self._neighbor_fn.allocate(positions, box=self._box)

        if self._backend == "torch":
            # ASE Atoms template, tagged with the molecular charge / spin
            # multiplicity that the MACE-OMol calculator reads from atoms.info.
            from ase import Atoms
            from ase.data import chemical_symbols

            symbols = [chemical_symbols[int(z)] for z in self._species]
            self._atoms_template = Atoms(symbols=symbols, positions=np.array(positions))
            self._atoms_template.info["charge"] = self._charge
            self._atoms_template.info["spin"] = self._spin
        else:
            # Static graph arrays for the JAX-native message-passing model.
            self._node_attrs = jax.nn.one_hot(
                self._species_indices,
                self._num_species,
                dtype=dt,
            )
            self._batch = jnp.zeros(n, dtype=jnp.int32)
            self._ptr = jnp.array([0, n], dtype=jnp.int32)
            self._cell = jnp.zeros((1, 3, 3), dtype=dt)

        self._initialized = True

    # ------------------------------------------------------------------
    # Energy function
    # ------------------------------------------------------------------

    def _build_energy_fn(self) -> Callable:
        if not self._initialized:
            raise RuntimeError("MACEPotential not initialized. Call initialize(positions) first.")
        if self._backend == "torch":
            return self._build_energy_fn_torch()
        return self._build_energy_fn_jax()

    def _build_energy_fn_jax(self) -> Callable:
        """Fully JAX-differentiable energy via mace-jax + JAX-MD neighbors.

        Converts JAX-MD neighbor indices ``(n_atoms, max_neighbors)`` into
        MACE's ``edge_index`` format ``(2, n_edges)``.  A ghost atom placed
        far away (> r_max) absorbs padding entries so that MACE's radial
        envelope zeros them out exactly.
        """
        graphdef = self._graphdef
        state = self._state
        node_attrs = self._node_attrs
        species_indices = self._species_indices
        num_species = self._num_species
        n_atoms = self._n_atoms
        r_max = self._r_max
        dt = self._dtype
        batch_real = self._batch

        # Ghost atom: sits at 3*r_max away from origin → always beyond cutoff.
        # Assigned to graph-1 so its energy is not counted.
        ghost_pos = jnp.array([[3.0 * r_max, 3.0 * r_max, 3.0 * r_max]], dtype=dt)
        ghost_species = jnp.array([0], dtype=jnp.int32)
        ghost_attrs = jnp.zeros((1, num_species), dtype=dt)
        ext_batch = jnp.concatenate(
            [batch_real, jnp.array([1], dtype=jnp.int32)]
        )  # (n_atoms+1,)  real→graph0, ghost→graph1
        ext_ptr = jnp.array([0, n_atoms, n_atoms + 1], dtype=jnp.int32)
        ext_cell = jnp.zeros((2, 3, 3), dtype=dt)

        def energy_fn(positions, neighbor=None, **kwargs):
            # Neighbor indices: (n_atoms, max_neighbors); padding == n_atoms
            nbr_idx = neighbor if neighbor is not None else self._nbrs.idx
            max_nbr = nbr_idx.shape[1]

            # Extended positions: real atoms + ghost atom at index n_atoms
            ext_positions = jnp.concatenate([positions, ghost_pos], axis=0)

            # Flatten neighbor list → senders/receivers
            senders = jnp.repeat(
                jnp.arange(n_atoms, dtype=jnp.int32),
                max_nbr,
            )
            receivers = nbr_idx.reshape(-1).astype(jnp.int32)
            # Padding entries already point to n_atoms (the ghost atom)

            edge_index = jnp.stack([senders, receivers], axis=0)
            n_edges = senders.shape[0]

            # Extended node attributes
            ext_node_attrs = jnp.concatenate([node_attrs, ghost_attrs], axis=0)
            ext_species = jnp.concatenate([species_indices, ghost_species])

            # Non-periodic: zero shifts
            shifts = jnp.zeros((n_edges, 3), dtype=dt)
            unit_shifts = jnp.zeros((n_edges, 3), dtype=dt)

            data = {
                "positions": ext_positions,
                "node_attrs": ext_node_attrs,
                "node_attrs_index": ext_species,
                "edge_index": edge_index,
                "shifts": shifts,
                "unit_shifts": unit_shifts,
                "cell": ext_cell,
                "batch": ext_batch,
                "ptr": ext_ptr,
            }
            out, _ = graphdef.apply(state)(
                data,
                compute_force=False,
                compute_stress=False,
            )
            # out["energy"] shape (2,): [real_molecule, ghost_graph]
            return out["energy"][0]

        return energy_fn

    def _build_energy_fn_torch(self) -> Callable:
        """PyTorch (ASE) backend for MACE-OMol: custom_vjp + pure_callback.

        Runs the native MACE-OMol calculator (which reads charge/spin from
        ``atoms.info``) and injects its autograd forces as the JAX gradient.
        """
        calc = self._calculator
        atoms_template = self._atoms_template
        n_atoms = self._n_atoms
        jnp_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        np_dtype = np.float64 if jax.config.jax_enable_x64 else np.float32

        def _compute(positions):
            pos_np = np.asarray(positions, dtype=np_dtype)
            atoms = atoms_template.copy()
            atoms.positions = pos_np
            atoms.calc = calc
            energy = np.array(float(atoms.get_potential_energy()), dtype=np_dtype)
            forces = np.array(atoms.get_forces(), dtype=np_dtype)
            return energy, forces

        result_shapes = (
            jax.ShapeDtypeStruct((), jnp_dtype),
            jax.ShapeDtypeStruct((n_atoms, 3), jnp_dtype),
        )

        @jax.custom_vjp
        def _energy(positions):
            e, _ = jax.pure_callback(_compute, result_shapes, positions, vmap_method="sequential")
            return e

        def _energy_fwd(positions):
            e, f = jax.pure_callback(_compute, result_shapes, positions, vmap_method="sequential")
            return e, f

        def _energy_bwd(f, g):
            # dE/dr = -forces
            return (-g * f,)

        _energy.defvjp(_energy_fwd, _energy_bwd)

        def energy_fn(positions, **kwargs):
            return _energy(positions)

        return energy_fn

    # ------------------------------------------------------------------
    # Neighbor list interface (JAX-MD)
    # ------------------------------------------------------------------

    @property
    def analytical_hessian(self) -> bool:
        # The JAX backend is fully differentiable; the PyTorch backend (OMol)
        # uses pure_callback, which cannot propagate JVP for analytical Hessians.
        return self._backend != "torch"

    @property
    def uses_neighbor_lists(self) -> bool:
        return True

    def allocate_neighbors(self, positions: jnp.ndarray, extra_capacity: int = 0):
        """Allocate a JAX-MD neighbor list."""
        positions = jnp.array(positions)
        return self._neighbor_fn.allocate(
            positions,
            box=self._box,
            extra_capacity=extra_capacity,
        )

    def update_neighbors(self, positions: jnp.ndarray, nbr_state):
        """Update the JAX-MD neighbor list."""
        positions = jnp.array(positions)
        return nbr_state.update(positions, box=self._box)

    def get_neighbor_kwargs(self, nbr_state) -> Dict[str, Any]:
        """Return neighbor indices for the energy function."""
        return {"neighbor": nbr_state.idx}


# ============================================================================
# dxtb Potential (GFN1-xTB / GFN2-xTB via PyTorch)
# ============================================================================

from jax_md import dataclasses as jax_md_dataclasses


@jax_md_dataclasses.dataclass
class _NullNbrs:
    """Dummy neighbor state for potentials that don't need neighbor lists.

    Uses jax_md.dataclasses so it survives jax.lax.fori_loop round-trips
    and exposes ``did_buffer_overflow`` as a property (same interface as
    jax_md.partition.NeighborList).
    """

    _overflow: jnp.ndarray  # always False

    @property
    def did_buffer_overflow(self):
        return self._overflow


@register_potential("dxtb")
class DXTBPotential(PotentialWrapper):
    """Fully differentiable extended tight-binding potential via dxtb.

    Wraps the ``dxtb`` PyTorch library (GFN1-xTB or GFN2-xTB) and bridges
    it into JAX via ``jax.custom_vjp`` + ``jax.pure_callback``, following
    the same pattern as the MACE PyTorch fallback.

    Units:
        * Positions in Angstrom (MARS convention) → converted to Bohr internally.
        * Energies in Hartree → converted to eV on output.

    Requirements::

        pip install dxtb[libcint]   # libcint highly recommended (Linux only)
        # or:
        pip install dxtb

    Reference:
        https://github.com/grimme-lab/dxtb

    Args:
        species: Array-like of atomic numbers, shape ``(n_atoms,)``.
        method: ``'gfn1'`` (default) or ``'gfn2'``.
        charge: Total molecular charge (default 0).
        dtype: Float precision.  Defaults to ``jnp.float32``.
    """

    # Unit conversion constants
    _ANGSTROM_TO_BOHR = 1.8897259886  # 1 Å = 1/a0 Bohr
    _HARTREE_TO_EV = 27.211386245988  # 1 Eh in eV

    def __init__(
        self,
        species,
        method: str = "gfn1",
        charge: float = 0.0,
        dtype=None,
        **kwargs,
    ):
        try:
            import dxtb
            import torch
        except ImportError:
            raise ImportError(
                "dxtb is not installed. Install with:\n"
                "  pip install dxtb[libcint]   # Linux (recommended)\n"
                "  pip install dxtb             # other platforms"
            )

        if method.lower() not in ("gfn1", "gfn2"):
            raise ValueError(f"method must be 'gfn1' or 'gfn2', got '{method}'")

        if dtype is None:
            dtype = jnp.float32

        # Convert species to numpy immediately to avoid issues with JAX arrays
        species_np = np.asarray(species, dtype=np.int32)
        self._n_atoms = len(species_np)
        self._dtype = dtype
        self._method = method.lower()
        self._charge = charge
        self._initialized = False

        # Determine torch dtype and device
        # dxtb is a physics-based method that benefits from double precision;
        # default to float64 unless the user explicitly requests float32.
        import torch

        self._torch_dtype = (
            torch.float64 if dtype not in (jnp.float32, "float32") else torch.float32
        )
        if torch.cuda.is_available():
            self._device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self._device = torch.device("cpu")

        # Store atomic numbers as torch tensor (fixed for this molecule)
        self._numbers = torch.from_numpy(species_np).to(dtype=torch.int, device=self._device)
        self._charge_tensor = torch.tensor(
            float(charge), dtype=self._torch_dtype, device=self._device
        )
        opts = {"verbosity": 0}

        # Build the dxtb calculator
        dd = {"dtype": self._torch_dtype, "device": self._device}
        if self._method == "gfn1":
            self._calc = dxtb.calculators.GFN1Calculator(self._numbers, opts=opts, **dd)
        else:
            self._calc = dxtb.calculators.GFN2Calculator(self._numbers, opts=opts, **dd)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, positions: jnp.ndarray):  # noqa: ARG002
        """No-op: dxtb is stateless (no neighbor lists needed)."""
        self._initialized = True

    # ------------------------------------------------------------------
    # Energy function
    # ------------------------------------------------------------------

    def _build_energy_fn(self) -> Callable:
        if not self._initialized:
            raise RuntimeError(
                "DXTBPotential not initialized. " "Call initialize(positions) first."
            )
        return self._build_energy_fn_only()

    def build_energy_fn(self, with_charges: bool = False):
        """Return energy function, optionally with Mulliken partial charges.

        When *with_charges* is ``True``, returns ``(energy_fn, charges_fn)``
        where ``charges_fn(positions, **kw) -> (n_atoms,)`` gives per-atom
        Mulliken charges from the dxtb SCF.
        """
        if not self._initialized:
            raise RuntimeError(
                "DXTBPotential not initialized. " "Call initialize(positions) first."
            )
        if with_charges:
            return self._build_energy_and_charges_fns()
        return self._build_energy_fn_only()

    def _build_energy_fn_only(self):
        """Build energy function without computing charges (faster path)."""
        import torch

        calc = self._calc
        charge = self._charge_tensor
        ang2bohr = self._ANGSTROM_TO_BOHR
        ha2ev = self._HARTREE_TO_EV
        torch_dtype = self._torch_dtype
        device = self._device
        n_atoms = self._n_atoms
        jnp_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        np_dtype = np.float64 if jax.config.jax_enable_x64 else np.float32

        def _compute(positions_ang):
            pos_np = np.asarray(positions_ang, dtype=np.float64)
            pos_bohr = torch.tensor(
                pos_np * ang2bohr,
                dtype=torch_dtype,
                device=device,
                requires_grad=True,
            )
            calc.reset()
            energy_ha = calc.get_energy(pos_bohr, chrg=charge)
            forces_ha_per_bohr = calc.get_forces(pos_bohr, energy=energy_ha)
            e_ev = np.array(float(energy_ha.detach().cpu()) * ha2ev, dtype=np_dtype)
            f_ev_ang = np.array(
                forces_ha_per_bohr.detach().cpu().numpy() * (ha2ev * ang2bohr),
                dtype=np_dtype,
            )
            return e_ev, f_ev_ang

        result_shapes = (
            jax.ShapeDtypeStruct((), jnp_dtype),
            jax.ShapeDtypeStruct((n_atoms, 3), jnp_dtype),
        )

        @jax.custom_vjp
        def _energy(positions):
            e, _ = jax.pure_callback(_compute, result_shapes, positions, vmap_method="sequential")
            return e

        def _energy_fwd(positions):
            e, f = jax.pure_callback(_compute, result_shapes, positions, vmap_method="sequential")
            return e, f

        def _energy_bwd(f, g):
            return (-g * f,)

        _energy.defvjp(_energy_fwd, _energy_bwd)

        def energy_fn(positions, **kwargs):
            return _energy(positions)

        return energy_fn

    def _build_energy_and_charges_fns(self):
        """Build energy and charges functions sharing a single callback."""
        import torch

        calc = self._calc
        charge = self._charge_tensor
        ang2bohr = self._ANGSTROM_TO_BOHR
        ha2ev = self._HARTREE_TO_EV
        torch_dtype = self._torch_dtype
        device = self._device
        n_atoms = self._n_atoms
        jnp_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        np_dtype = np.float64 if jax.config.jax_enable_x64 else np.float32

        def _compute(positions_ang):
            """Call dxtb and return (energy_eV, forces_eV_per_ang, charges)."""
            pos_np = np.asarray(positions_ang, dtype=np.float64)
            pos_bohr = torch.tensor(
                pos_np * ang2bohr,
                dtype=torch_dtype,
                device=device,
                requires_grad=True,
            )
            calc.reset()
            energy_ha = calc.get_energy(pos_bohr, chrg=charge)
            forces_ha_per_bohr = calc.get_forces(pos_bohr, energy=energy_ha)

            # Per-atom Mulliken charges (orbital → atom reduction)
            charges_orb = calc.get_charges(pos_bohr, chrg=charge)
            charges_atom = calc.ihelp.reduce_orbital_to_atom(charges_orb)
            q_np = np.array(
                charges_atom.detach().cpu().numpy(),
                dtype=np_dtype,
            )

            # Convert units
            e_ev = np.array(float(energy_ha.detach().cpu()) * ha2ev, dtype=np_dtype)
            # forces = -dE/dr; dxtb gives -dE/dr_bohr [Ha/Bohr]
            # dE_eV/dr_ang = dE_ha/dr_bohr * ha2ev * ang2bohr
            # (chain rule: dr_bohr/dr_ang = ang2bohr)
            f_ev_ang = np.array(
                forces_ha_per_bohr.detach().cpu().numpy() * (ha2ev * ang2bohr),
                dtype=np_dtype,
            )
            return e_ev, f_ev_ang, q_np

        result_shapes = (
            jax.ShapeDtypeStruct((), jnp_dtype),
            jax.ShapeDtypeStruct((n_atoms, 3), jnp_dtype),
            jax.ShapeDtypeStruct((n_atoms,), jnp_dtype),
        )

        @jax.custom_vjp
        def _energy(positions):
            e, _, _ = jax.pure_callback(
                _compute, result_shapes, positions, vmap_method="sequential"
            )
            return e

        def _energy_fwd(positions):
            e, f, _ = jax.pure_callback(
                _compute, result_shapes, positions, vmap_method="sequential"
            )
            return e, f

        def _energy_bwd(f, g):
            # gradient of energy w.r.t. positions = -forces
            return (-g * f,)

        _energy.defvjp(_energy_fwd, _energy_bwd)

        def energy_fn(positions, **kwargs):
            return _energy(positions)

        def charges_fn(positions, **kwargs):
            _, _, q = jax.pure_callback(
                _compute, result_shapes, positions, vmap_method="sequential"
            )
            return q

        return energy_fn, charges_fn

    # ------------------------------------------------------------------
    # Neighbor list interface — dummy state for sampling/IR loop compat
    # ------------------------------------------------------------------

    @property
    def analytical_hessian(self) -> bool:
        return False  # pure_callback cannot propagate JVP for 2nd derivatives

    @property
    def uses_neighbor_lists(self) -> bool:
        return True

    def allocate_neighbors(self, _positions: jnp.ndarray, _extra_capacity: int = 0) -> _NullNbrs:
        return _NullNbrs(_overflow=jnp.array(False))

    def update_neighbors(self, _positions: jnp.ndarray, nbr_state: _NullNbrs) -> _NullNbrs:
        return nbr_state

    def get_neighbor_kwargs(self, _nbr_state: _NullNbrs) -> Dict[str, Any]:
        return {}


# ============================================================================
# Confinement Potential (spherical / ellipsoidal)
# ============================================================================


@register_potential("confinement")
class ConfinementPotential(PotentialWrapper):
    """Confines every atom inside a sphere or ellipsoid centred in space.

    Two modes
    ---------
    ``'harmonic'`` *(elastic)*
        A quadratic penalty kicks in as soon as an atom crosses the surface::

            E_i = k * max(0, d_i - 1)²

    ``'wall'`` *(inelastic / hard wall)*
        A steep polynomial repulsion — atoms effectively bounce off the wall::

            E_i = k * max(0, d_i - 1)^n   (default n = 12)

    In both cases *d_i* is the **scaled distance** of atom *i* from the
    centre.  For a sphere of radius *R*, ``d_i = |r_i - c| / R``.  For an
    ellipsoid with semi-axes *(rx, ry, rz)*::

        d_i = sqrt( ((x-cx)/rx)² + ((y-cy)/ry)² + ((z-cz)/rz)² )

    so the atom is inside the confining volume whenever ``d_i ≤ 1``.

    Args:
        radii: Confinement radius (Å).  Pass a scalar for a **sphere** or a
            length-3 sequence *(rx, ry, rz)* for an **ellipsoid**.
        center: Centre of the confining volume (Å), shape ``(3,)``.
            ``None`` (default) uses the centroid of the initial positions
            computed during :meth:`initialize`.
        k: Force constant (eV / Å²).
        mode: ``'harmonic'`` or ``'wall'``.
        wall_exponent: Exponent *n* used in ``'wall'`` mode (default 12).
    """

    def __init__(
        self,
        radii: float | Tuple[float, float, float] = 10.0,
        center=None,
        k: float = 10.0,
        mode: str = "harmonic",
        wall_exponent: int = 12,
        **kwargs,
    ):
        if mode not in ("harmonic", "wall"):
            raise ValueError(f"mode must be 'harmonic' or 'wall', got '{mode}'")

        radii_arr = jnp.array(radii, dtype=jnp.float32)
        if radii_arr.ndim == 0:
            radii_arr = jnp.broadcast_to(radii_arr, (3,))
        if radii_arr.shape != (3,):
            raise ValueError("radii must be a scalar or a length-3 sequence")

        self._radii = radii_arr
        self._center_init = None if center is None else jnp.array(center, dtype=jnp.float32)
        self._center = self._center_init
        self._k = float(k)
        self._mode = mode
        self._wall_exponent = int(wall_exponent)
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, positions: jnp.ndarray):
        positions = jnp.array(positions)
        if self._center_init is None:
            self._center = jnp.mean(positions, axis=0)
        self._initialized = True

    # ------------------------------------------------------------------
    # Energy function
    # ------------------------------------------------------------------

    def _build_energy_fn(self) -> Callable:
        if not self._initialized:
            raise RuntimeError(
                "ConfinementPotential not initialized. " "Call initialize(positions) first."
            )

        radii = self._radii
        center = self._center
        k = self._k
        mode = self._mode
        n = self._wall_exponent

        def energy_fn(positions, **kwargs):
            # Scaled distance from center for every atom  (n_atoms,)
            dr = positions - center  # (n_atoms, 3)
            scaled = dr / radii  # broadcast over (3,)
            d = jnp.sqrt(jnp.sum(scaled**2, axis=-1) + 1e-12)  # (n_atoms,)

            # Penetration depth (zero inside the confining volume)
            penetration = jnp.maximum(0.0, d - 1.0)  # (n_atoms,)

            if mode == "harmonic":
                contributions = k * penetration**2
            else:  # "wall"
                contributions = k * penetration**n

            return jnp.sum(contributions)

        return energy_fn

    # ------------------------------------------------------------------
    # Neighbor list interface (no pairwise interactions needed)
    # ------------------------------------------------------------------

    @property
    def uses_neighbor_lists(self) -> bool:
        return False
