"""
Classical force fields for MARS: AMBER/GAFF, CHARMM, OPLS-AA and AMOEBA.

MARS provides the functional forms, not the parameters: the user supplies
their own parameter / topology files, which are read through OpenMM and
ParmEd (:mod:`mars.forcefields.io`) and evaluated in JAX
(:mod:`mars.forcefields.convert`):

* standard terms (harmonic bonds/angles, periodic torsions, CMAP,
  Lennard-Jones + Coulomb with 1-4 exceptions) by jax-md's
  ``mm_forcefields`` (jax-md PRs #319 by @alecbetancourt and #350 by
  @DRosen285);
* every OpenMM ``Custom*Force`` (CHARMM impropers / NBFIX, GROMACS-style
  terms, AMOEBA valence terms) by a Lepton-expression compiler
  (:mod:`mars.forcefields.lepton`);
* AMOEBA vdW, multipoles + polarization and torsion-torsion natively
  (:mod:`mars.forcefields.amoeba`).

Everything is evaluated in free space with all-pairs nonbonded sums.  Use the
``mm`` / ``amber`` / ``charmm`` / ``opls`` / ``amoeba`` potentials
(:class:`mars.potentials.MMPotential`) to run MARS workflows with them.

Example::

    from mars.forcefields import load_amber
    from mars.potentials import get_potential

    system = load_amber("ala2.prmtop", "ala2.inpcrd")
    pot = get_potential("mm", system=system)
    pot.initialize(system.positions)
    energy_fn = pot.build_energy_fn()   # eV, positions in Å
"""

from .io import (
    load_amber,
    load_amoeba,
    load_charmm,
    load_gromacs,
    load_openmm_system,
    load_openmm_xml,
)
from .system import KCAL_TO_EV, MMSystem


def load_opls(top=None, coordinates=None, xml=None, pdb=None, include_dir=None) -> MMSystem:
    """OPLS-AA from a GROMACS topology or from OpenMM ForceField XML.

    Args:
        top: GROMACS ``.top`` (route 1), with optional ``.gro``/PDB
            ``coordinates`` and ``include_dir`` for ``#include`` files.
        xml: OpenMM XML file(s), e.g. from LigParGen (route 2), with ``pdb``.
    """
    if top is not None:
        return load_gromacs(top, coordinates, include_dir=include_dir, family="opls")
    if xml is not None:
        return load_openmm_xml(xml, pdb, family="opls")
    raise ValueError(
        "OPLS-AA needs either a GROMACS topology (--top [--gro]) or OpenMM XML + PDB "
        "(--ff-xml ... --pdb)."
    )


def load_system(family: str, **files) -> MMSystem:
    """Dispatch to the reader for *family* (used by the CLI)."""
    family = family.lower()
    if family == "amber":
        return load_amber(files.get("prmtop"), files.get("coordinates"))
    if family == "charmm":
        return load_charmm(files.get("psf"), files.get("params") or [], files.get("coordinates"))
    if family == "opls":
        return load_opls(
            top=files.get("top"),
            coordinates=files.get("coordinates"),
            xml=files.get("xml"),
            pdb=files.get("pdb"),
            include_dir=files.get("include_dir"),
        )
    if family == "amoeba":
        return load_amoeba(
            pdb=files.get("pdb"),
            xml=files.get("xml") or ("amoeba2018.xml",),
            polarization=files.get("polarization") or "mutual",
            tinker_xyz=files.get("tinker_xyz"),
            tinker_key=files.get("tinker_key"),
        )
    if family in ("openmm", "mm"):
        if files.get("system_xml"):
            return load_openmm_system(files["system_xml"], files.get("pdb"))
        return load_openmm_xml(files.get("xml") or [], files.get("pdb"))
    raise ValueError(f"Unknown force-field family '{family}'.")


__all__ = [
    "MMSystem",
    "KCAL_TO_EV",
    "load_amber",
    "load_charmm",
    "load_gromacs",
    "load_opls",
    "load_amoeba",
    "load_openmm_xml",
    "load_openmm_system",
    "load_system",
]
