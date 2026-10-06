"""
:class:`MMSystem` — a classical force-field system ready for MARS.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np

#: kcal/mol -> eV
KCAL_TO_EV = 0.0433641043


@dataclass
class MMSystem:
    """A parameterised molecular system, independent of how it was read.

    Attributes:
        family: Force-field family label (``'amber'``, ``'charmm'``,
            ``'opls'``, ``'amoeba'``, ``'openmm'`` ...).
        numbers: Atomic numbers, ``(n_atoms,)``.
        symbols: Element symbols.
        positions: Reference positions in Å, ``(n_atoms, 3)``, or ``None``
            when the parameter files carried no coordinates.
        masses: Atomic masses in amu.
        openmm_system: The ``openmm.System`` the parameters came from (kept
            for reference calculations; MARS itself only uses JAX).
        topology: The matching ``openmm.app.Topology``.
        polarization: AMOEBA polarization mode, if applicable.
        source: The input files, for logging.
    """

    family: str
    numbers: np.ndarray
    symbols: list
    positions: Optional[np.ndarray]
    masses: np.ndarray
    openmm_system: Any
    topology: Any
    polarization: Optional[str] = None
    source: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return len(self.numbers)

    @classmethod
    def from_openmm(
        cls,
        system,
        topology,
        positions_ang=None,
        family: str = "openmm",
        source=None,
        polarization: Optional[str] = None,
    ) -> "MMSystem":
        """Wrap an existing OpenMM ``System`` + ``Topology``."""
        from openmm import unit

        numbers, symbols = [], []
        for atom in topology.atoms():
            if atom.element is None:
                raise NotImplementedError(
                    f"Atom {atom.name!r} in residue {atom.residue.name!r} has no element "
                    "(virtual site / extra point); not supported in MARS."
                )
            numbers.append(int(atom.element.atomic_number))
            symbols.append(atom.element.symbol)
        if len(numbers) != system.getNumParticles():
            raise ValueError(
                f"Topology has {len(numbers)} atoms but the System has "
                f"{system.getNumParticles()} particles."
            )
        masses = np.array(
            [system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(len(numbers))]
        )
        from .checks import check_parameters

        check_parameters(system, topology)
        return cls(
            family=family,
            numbers=np.asarray(numbers, dtype=int),
            symbols=symbols,
            positions=None if positions_ang is None else np.asarray(positions_ang, dtype=float),
            masses=masses,
            openmm_system=system,
            topology=topology,
            polarization=polarization,
            source=dict(source or {}),
        )

    def to_structure(self) -> Dict[str, Any]:
        """MARS structure dict (``positions``, ``symbols``, ``numbers``)."""
        if self.positions is None:
            raise ValueError("This MMSystem has no coordinates; supply a structure file.")
        return {
            "positions": np.array(self.positions),
            "symbols": list(self.symbols),
            "numbers": np.array(self.numbers),
        }

    def check_matches(self, numbers) -> None:
        """Raise if *numbers* (e.g. from an XYZ file) differ from the topology."""
        numbers = np.asarray(numbers, dtype=int)
        if numbers.shape != self.numbers.shape:
            raise ValueError(
                f"The structure has {len(numbers)} atoms but the force-field topology has "
                f"{self.n_atoms}. Use the coordinates that match the parameter files."
            )
        bad = np.nonzero(numbers != self.numbers)[0]
        if bad.size:
            i = int(bad[0])
            raise ValueError(
                f"Atom order differs from the force-field topology: atom {i + 1} is "
                f"Z={numbers[i]} in the structure but Z={self.numbers[i]} ({self.symbols[i]}) in "
                f"the topology ({bad.size} mismatches). The atom order must match exactly."
            )

    def energy_terms(self, positions_ang=None) -> tuple:
        """``(terms_fn, charges)``; see :func:`mars.forcefields.convert.build_energy_terms`."""
        from .convert import build_energy_terms

        ref = positions_ang if positions_ang is not None else self.positions
        return build_energy_terms(
            self.openmm_system, self.topology, ref, polarization=self.polarization
        )

    def energy_fn(self, positions_ang=None) -> Callable:
        """Scalar ``energy(positions_ang) -> eV`` (sum of all terms)."""
        import jax.numpy as jnp

        terms_fn, _ = self.energy_terms(positions_ang)

        def energy(positions):
            terms = terms_fn(positions)
            return KCAL_TO_EV * jnp.sum(jnp.stack([jnp.asarray(v) for v in terms.values()]))

        return energy
