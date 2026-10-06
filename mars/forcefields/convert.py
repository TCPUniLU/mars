"""
OpenMM ``System`` -> JAX energy terms.

Every force in the system is routed to exactly one evaluator:

* **jax-md** (``jax_md.mm_forcefields.amber``, PRs #319/#350):
  ``HarmonicBondForce``, ``PeriodicTorsionForce``,
  ``CMAPTorsionForce`` and ``NonbondedForce`` (Lennard-Jones, Coulomb and the
  1-4 / exception pairs).  These cover AMBER/GAFF, CHARMM (Urey-Bradley is a
  second ``HarmonicBondForce``) and OPLS from OpenMM XML.
* **MARS Lepton compiler** (:mod:`mars.forcefields.custom`): every
  ``Custom*Force`` — CHARMM harmonic impropers and NBFIX tables, GROMACS
  combination rule 3 / quartic bonds / cosine angles, all AMOEBA valence
  terms — plus ``RBTorsionForce`` and ``HarmonicAngleForce`` (jax-md's angle
  helper loses accuracy near 180 degrees; see :func:`custom.harmonic_angle`).
* **MARS AMOEBA** (:mod:`mars.forcefields.amoeba`): ``AmoebaVdwForce``,
  ``AmoebaMultipoleForce`` and ``AmoebaTorsionTorsionForce``.

``CMMotionRemover`` is ignored (it has no energy).  Any other force raises,
so a term is never silently dropped.  Everything runs in free space: the
system must be built with ``nonbondedMethod=NoCutoff`` and without
constraints or virtual sites.

All terms take positions in Å and return kcal/mol.
"""

from typing import Callable, Dict, Optional

import numpy as np

_JAXMD_FORCES = (
    "HarmonicBondForce",
    "PeriodicTorsionForce",
    "CMAPTorsionForce",
    "NonbondedForce",
)
_IGNORED_FORCES = ("CMMotionRemover",)

#: Human-readable names of the per-term energies returned by jax-md.
_JAXMD_TERMS = {
    "bond_pot": "bond",
    "torsion_pot": "torsion",
    "improper_pot": "improper",
    "cmap_pot": "cmap",
    "nb_pot": "nonbonded",
}


def _import_openmm():
    try:
        import openmm
    except ImportError as exc:
        raise ImportError(
            "Classical force fields in MARS read their parameters through OpenMM.\n"
            "Install with: pip install 'mars[mm]'   (openmm, parmed)\n"
            f"(import error: {exc})"
        )
    return openmm


def _check_free_space(system):
    openmm = _import_openmm()
    if system.getNumConstraints() > 0:
        raise ValueError(
            f"The OpenMM System has {system.getNumConstraints()} constraints. MARS integrators "
            "do not apply constraints: rebuild it with constraints=None, rigidWater=False."
        )
    for i in range(system.getNumParticles()):
        if system.isVirtualSite(i):
            raise NotImplementedError(
                "Virtual sites (e.g. TIP4P/OPC water, lone pairs) are not supported in MARS yet."
            )
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            if force.getNonbondedMethod() != openmm.NonbondedForce.NoCutoff:
                raise NotImplementedError(
                    "MARS evaluates force fields in free space: rebuild the System with "
                    "nonbondedMethod=app.NoCutoff."
                )
            if force.getNumParticleParameterOffsets() or force.getNumExceptionParameterOffsets():
                raise NotImplementedError(
                    "NonbondedForce parameter offsets (alchemical setups) are not supported."
                )


def _merged_standard_system(system):
    """Copy of *system* with only the jax-md forces, one force per type.

    jax-md's converter keeps only the last ``PeriodicTorsionForce`` it sees, so duplicates are merged here, and an
    all-zero ``NonbondedForce`` is added when there is none (its builder
    needs per-particle arrays).
    """
    openmm = _import_openmm()
    sub = openmm.System()
    for i in range(system.getNumParticles()):
        sub.addParticle(system.getParticleMass(i))

    merged: Dict[str, object] = {}
    for force in system.getForces():
        name = type(force).__name__
        if name not in _JAXMD_FORCES:
            continue
        if name == "CMAPTorsionForce" and force.getNumTorsions() == 0:
            # OpenMM's CHARMM reader adds an empty CMAP force for molecules
            # without CMAP terms; jax-md's converter reads map 0 unconditionally.
            continue
        if name not in merged:
            merged[name] = openmm.XmlSerializer.clone(force)
            continue
        target = merged[name]
        if name == "HarmonicBondForce":
            for b in range(force.getNumBonds()):
                target.addBond(*force.getBondParameters(b))
        elif name == "PeriodicTorsionForce":
            for t in range(force.getNumTorsions()):
                target.addTorsion(*force.getTorsionParameters(t))
        else:
            raise NotImplementedError(f"More than one {name} in the System is not supported.")

    if "NonbondedForce" not in merged:
        nb = openmm.NonbondedForce()
        nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
        for _ in range(system.getNumParticles()):
            nb.addParticle(0.0, 0.1, 0.0)
        merged["NonbondedForce"] = nb
    for force in merged.values():
        sub.addForce(force)
    return sub


def _jaxmd_terms(system, topology, positions_nm):
    """Build the jax-md AMBER-style evaluator for the standard forces."""
    import jax.numpy as jnp
    from openmm import unit

    try:
        from jax_md.mm_forcefields.amber.energy import energy as amber_energy
        from jax_md.mm_forcefields.io.openmm import convert_openmm_system
        from jax_md.mm_forcefields.nonbonded import electrostatics
    except ImportError as exc:
        raise ImportError(
            "Classical force fields need jax-md's mm_forcefields (jax-md main >= a41c7d1):\n"
            "  pip install 'git+https://github.com/jax-md/jax-md.git'\n"
            f"(import error: {exc})"
        )

    sub = _merged_standard_system(system)
    quantity = unit.Quantity(np.asarray(positions_nm, dtype=float), unit.nanometer)
    mm = convert_openmm_system(sub, topology, quantity, None)
    energy_fn, neighbor_fn, _, _ = amber_energy(
        mm.params,
        mm.topology,
        None,
        coulomb_options=electrostatics.CutoffCoulomb(r_cut=None),
        nb_options=mm.nb_options,
    )
    # No cutoff: the neighbor list holds every pair and never changes, so it
    # is allocated once and closed over.
    nbrs = neighbor_fn.allocate(mm.positions)
    params = mm.params

    def terms(positions):
        out = energy_fn(jnp.asarray(positions), nbrs, params=params)
        return {label: out[key] for key, label in _JAXMD_TERMS.items()}

    return terms, np.asarray(mm.params.nonbonded.charges)


def build_energy_terms(
    system, topology=None, positions_ang=None, polarization: Optional[str] = None
):
    """Compile an OpenMM ``System`` into a JAX per-term energy function.

    Args:
        system: ``openmm.System`` built with ``nonbondedMethod=NoCutoff``.
        topology: The matching ``openmm.app.Topology`` (needed by jax-md).
        positions_ang: Reference positions in Å, ``(n_atoms, 3)``.  Only
            used to allocate the (all-pairs) neighbor list.
        polarization: AMOEBA polarization override (``'mutual'``,
            ``'direct'``, ``'extrapolated'``); ``None`` keeps the System's.

    Returns:
        ``(terms_fn, charges)`` where ``terms_fn(positions_ang)`` returns a
        dict ``{term: energy_kcal_mol}`` and ``charges`` are the fixed point
        charges (AMOEBA: monopoles), or ``None`` if the system has none.
    """
    from .custom import custom_force_energy, harmonic_angle, is_custom_force, rb_torsion

    openmm = _import_openmm()
    _check_free_space(system)
    n_atoms = system.getNumParticles()
    if positions_ang is None:
        positions_ang = np.zeros((n_atoms, 3))
    positions_ang = np.asarray(positions_ang, dtype=float)

    extra: Dict[str, Callable] = {}
    has_standard = False
    has_amoeba = False
    counts: Dict[str, int] = {}

    def key_for(force):
        base = force.getName() or type(force).__name__
        counts[base] = counts.get(base, 0) + 1
        return base if counts[base] == 1 else f"{base}_{counts[base]}"

    for force in system.getForces():
        name = type(force).__name__
        if name in _IGNORED_FORCES:
            continue
        if name in _JAXMD_FORCES:
            has_standard = True
        elif is_custom_force(force):
            extra[key_for(force)] = custom_force_energy(force)
        elif isinstance(force, openmm.RBTorsionForce):
            extra[key_for(force)] = rb_torsion(force)
        elif isinstance(force, openmm.HarmonicAngleForce):
            extra[key_for(force)] = harmonic_angle(force)
        elif name.startswith("Amoeba"):
            has_amoeba = True
        else:
            raise NotImplementedError(
                f"OpenMM force {name} ({force.getName()!r}) is not supported by MARS. "
                "Supported: standard bonded/nonbonded forces, all Custom*Forces, "
                "RBTorsionForce and the AMOEBA vdW / multipole / torsion-torsion forces."
            )

    std_terms = None
    charges = None
    if has_standard:
        std_terms, charges = _jaxmd_terms(system, topology, positions_ang * 0.1)

    amoeba_terms: Dict[str, Callable] = {}
    if has_amoeba:
        from .amoeba import build_amoeba_terms

        amoeba_terms = build_amoeba_terms(system, polarization=polarization)
        mp = [f for f in system.getForces() if isinstance(f, openmm.AmoebaMultipoleForce)]
        if mp:
            from openmm import unit

            charges = np.array(
                [
                    mp[0].getMultipoleParameters(i)[0].value_in_unit(unit.elementary_charge)
                    for i in range(n_atoms)
                ],
                dtype=float,
            )
    if charges is not None and not np.any(charges):
        charges = None

    def terms_fn(positions):
        out = {}
        if std_terms is not None:
            out.update(std_terms(positions))
        for k, fn in extra.items():
            out[k] = fn(positions)
        for k, fn in amoeba_terms.items():
            out[k] = fn(positions)
        return out

    return terms_fn, charges
