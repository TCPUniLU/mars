"""
Native-JAX AMOEBA nonbonded and torsion-torsion terms.

Converts the three native AMOEBA forces of an OpenMM ``System`` (built from
e.g. ``amoeba2018.xml`` with ``nonbondedMethod=NoCutoff``) into pure JAX
energy functions of the positions (Å in, kcal/mol out):

* ``openmm.AmoebaVdwForce``            -> ``'amoeba_vdw'`` (:mod:`.vdw`)
* ``openmm.AmoebaMultipoleForce``      -> ``'amoeba_multipole'``, permanent
  multipoles plus mutual / direct / extrapolated (OPT) polarization
  (:mod:`.multipole`)
* ``openmm.AmoebaTorsionTorsionForce`` -> ``'amoeba_torsion_torsion'``
  (:mod:`.torsion_torsion`)

The AMOEBA valence terms are ``Custom*Force`` objects in OpenMM and are handled
by the generic Lepton compiler of :mod:`mars.forcefields`.  Conventions follow
the OpenMM Reference platform, against which the energies and forces are
validated.

References
----------
J. W. Ponder et al., J. Phys. Chem. B 114, 2549 (2010);
B. T. Thole, Chem. Phys. 59, 341 (1981);
A. C. Simmonett et al., J. Chem. Phys. 143, 074115 (2015).
"""

from .convert import (
    MultipoleParams,
    TorsionTorsionParams,
    VdwParams,
    multipole_from_openmm,
    scale_matrices,
    torsion_torsion_from_openmm,
    torsion_torsion_params,
    vdw_from_openmm,
)
from .energy import (
    atomic_dipoles,
    build_amoeba_term,
    build_amoeba_terms,
    induced_dipoles,
    is_amoeba_force,
    multipole_components,
)

__all__ = [
    "MultipoleParams",
    "atomic_dipoles",
    "TorsionTorsionParams",
    "VdwParams",
    "build_amoeba_term",
    "build_amoeba_terms",
    "induced_dipoles",
    "is_amoeba_force",
    "multipole_components",
    "multipole_from_openmm",
    "scale_matrices",
    "torsion_torsion_from_openmm",
    "torsion_torsion_params",
    "vdw_from_openmm",
]
