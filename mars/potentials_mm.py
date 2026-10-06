"""
Classical force-field potentials (AMBER, CHARMM, OPLS-AA, AMOEBA).

Thin :class:`~mars.potentials.PotentialWrapper` around
:class:`mars.forcefields.MMSystem`.  The parameters always come from the
user's files; see :mod:`mars.forcefields.io` for what each family reads.

Registered names:

* ``mm``     — any :class:`MMSystem` (pass ``system=``) or any family via
  ``family=``.
* ``amber``, ``charmm``, ``opls``, ``amoeba`` — the same, with the family
  fixed, so the CLI can say ``--potential amber --prmtop x.prmtop``.

Energies are returned in eV (positions in Å).  All nonbonded sums are
all-pairs in free space, so no neighbor list is needed.
"""

from typing import Any, Callable, Dict, Optional

import jax.numpy as jnp
import numpy as np

from .potentials import ChargesFunction, PotentialWrapper, _NullNbrs, register_potential

_FILE_KEYS = (
    "prmtop",
    "coordinates",
    "psf",
    "params",
    "top",
    "include_dir",
    "xml",
    "pdb",
    "polarization",
    "tinker_xyz",
    "tinker_key",
    "system_xml",
)


@register_potential("mm")
class MMPotential(PotentialWrapper):
    """Classical force field from user-supplied parameter files.

    Args:
        system: A ready :class:`~mars.forcefields.MMSystem`.  If omitted, it
            is read from the files given as keyword arguments.
        family: ``'amber'``, ``'charmm'``, ``'opls'``, ``'amoeba'`` or
            ``'openmm'`` (fixed by the family-specific aliases).
        species: Atomic numbers of the MARS input structure; checked against
            the force-field topology (the atom order must be identical).
        charge: Expected net charge; checked against the force-field charges
            (the charge itself always comes from the parameters).
        spin: Spin multiplicity; must be 1 (force fields are closed-shell).
        prmtop, coordinates, psf, params, top, include_dir, xml, pdb,
        polarization, tinker_xyz, tinker_key, system_xml: Input files /
            options forwarded to :func:`mars.forcefields.load_system`.
    """

    FAMILY: Optional[str] = None

    def __init__(
        self,
        system=None,
        family: Optional[str] = None,
        species=None,
        charge: Optional[float] = None,
        spin: float = 1.0,
        **kwargs,
    ):
        from .forcefields import load_system

        if system is None:
            family = family or self.FAMILY
            if family is None:
                raise ValueError(
                    "The 'mm' potential needs either system=MMSystem or family=... with its "
                    "parameter files."
                )
            files = {k: kwargs[k] for k in _FILE_KEYS if kwargs.get(k) is not None}
            system = load_system(family, **files)
        if species is not None:
            system.check_matches(species)
        if float(spin) != 1.0:
            raise ValueError(
                "Classical force fields are closed-shell: the spin multiplicity must be 1 "
                f"(got {spin:g})."
            )
        self._requested_charge = charge
        self._system = system
        self._species = np.asarray(system.numbers)
        self._initialized = False

    @property
    def system(self):
        """The underlying :class:`~mars.forcefields.MMSystem`."""
        return self._system

    def initialize(self, positions: jnp.ndarray):
        positions = np.asarray(positions, dtype=float)
        from .forcefields.checks import check_geometry

        # Structure checks first (atom count, NaN, overlaps, units, atom order),
        # so a bad structure is reported clearly instead of failing in JAX.
        check_geometry(self._system.openmm_system, self._system.topology, positions)
        self._terms_fn, self._charges = self._system.energy_terms(positions)
        # Then the initial energy / forces (finite, not absurdly large).
        check_geometry(self._system.openmm_system, self._system.topology, positions, self._terms_fn)
        if self._requested_charge is not None:
            # The net charge is fixed by the parameters; catch a --charge that
            # contradicts them (AMOEBA: sum of the permanent monopoles).
            net = 0.0 if self._charges is None else float(np.sum(self._charges))
            if abs(net - float(self._requested_charge)) > 1e-3:
                raise ValueError(
                    f"--charge {self._requested_charge:g} does not match the force-field "
                    f"topology, whose net charge is {net:.3f}."
                )
        self._initialized = True

    def _require_init(self):
        if not self._initialized:
            raise RuntimeError(f"{type(self).__name__} not initialized. Call initialize() first.")

    def energy_components(self, positions) -> Dict[str, float]:
        """Per-term energies in eV (for logging and validation)."""
        from .forcefields import KCAL_TO_EV

        self._require_init()
        terms = self._terms_fn(jnp.asarray(positions))
        return {k: float(v) * KCAL_TO_EV for k, v in terms.items()}

    def _build_energy_fn(self) -> Callable:
        from .forcefields import KCAL_TO_EV

        self._require_init()
        terms_fn = self._terms_fn

        def energy_fn(positions, **kwargs):
            terms = terms_fn(positions)
            return KCAL_TO_EV * sum(jnp.asarray(v) for v in terms.values())

        return energy_fn

    def build_energy_fn(self, with_charges: bool = False):
        energy_fn = self._build_energy_fn()
        if not with_charges:
            return energy_fn
        if self._charges is None:
            return super().build_energy_fn(with_charges=True)
        charges = jnp.asarray(self._charges)

        def charges_fn(positions, **kwargs):
            # Fixed point charges (AMOEBA: the permanent monopoles).
            return charges

        dipoles = None
        if any(
            type(f).__name__ == "AmoebaMultipoleForce"
            for f in self._system.openmm_system.getForces()
        ):
            from .forcefields.amoeba import atomic_dipoles

            # AMOEBA: the molecular dipole also has the permanent atomic dipoles
            # and the induced dipoles, which IR intensities must include.
            amoeba_dipoles = atomic_dipoles(self._system.openmm_system, self._system.polarization)

            def dipoles(positions, **kwargs):
                return amoeba_dipoles(positions)

        return energy_fn, ChargesFunction(charges_fn, dipoles)

    # No neighbor lists: nonbonded terms are all-pairs in free space. The
    # sampling loops still allocate / check a state, so hand them a null one.
    @property
    def uses_neighbor_lists(self) -> bool:
        return False

    def allocate_neighbors(self, positions, extra_capacity: int = 0) -> _NullNbrs:
        return _NullNbrs(_overflow=jnp.array(False))

    def update_neighbors(self, positions, nbr_state: _NullNbrs) -> _NullNbrs:
        return nbr_state

    def get_neighbor_kwargs(self, nbr_state: Any) -> Dict[str, Any]:
        return {}


@register_potential("amber")
class AmberPotential(MMPotential):
    """AMBER / GAFF from ``prmtop`` (+ coordinates). See :class:`MMPotential`."""

    FAMILY = "amber"


@register_potential("charmm")
class CharmmPotential(MMPotential):
    """CHARMM from PSF + parameter files. See :class:`MMPotential`."""

    FAMILY = "charmm"


@register_potential("opls")
class OPLSPotential(MMPotential):
    """OPLS-AA from a GROMACS topology or OpenMM XML. See :class:`MMPotential`."""

    FAMILY = "opls"


@register_potential("amoeba")
class AmoebaPotential(MMPotential):
    """AMOEBA polarizable force field. See :class:`MMPotential`."""

    FAMILY = "amoeba"
