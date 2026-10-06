"""
Assemble MARS energy terms (Å in, kcal/mol out) from native AMOEBA forces.

The kernels in :mod:`.vdw`, :mod:`.multipole` and :mod:`.torsion_torsion`
work in OpenMM units; the closures built here convert positions from Å to nm
and energies from kJ/mol to kcal/mol, so the returned callables plug directly
into the MARS potential machinery.  They are pure, jit-able functions of the
positions only; forces and Hessians follow from ``jax.grad`` / ``jax.hessian``.
"""

from typing import Callable, Dict, Optional, Tuple

import jax.numpy as jnp

from .convert import (
    MultipoleParams,
    TorsionTorsionParams,
    VdwParams,
    multipole_from_openmm,
    torsion_torsion_from_openmm,
    vdw_from_openmm,
)
from .multipole import multipole_energy, multipole_energy_components
from .torsion_torsion import torsion_torsion_energy
from .vdw import vdw_energy

ANG_TO_NM = 0.1
KJ_TO_KCAL = 1.0 / 4.184

_NATIVE = ("AmoebaVdwForce", "AmoebaMultipoleForce", "AmoebaTorsionTorsionForce")
_UNSUPPORTED = ("AmoebaGeneralizedKirkwoodForce", "AmoebaWcaDispersionForce")
_KEYS = {
    "AmoebaVdwForce": "amoeba_vdw",
    "AmoebaMultipoleForce": "amoeba_multipole",
    "AmoebaTorsionTorsionForce": "amoeba_torsion_torsion",
}


def is_amoeba_force(force) -> bool:
    """``True`` for the three native AMOEBA forces handled by this package."""
    return type(force).__name__ in _NATIVE


def _nm(positions_ang):
    return jnp.asarray(positions_ang) * ANG_TO_NM


def vdw_term(params: VdwParams) -> Callable:
    """Energy closure (Å -> kcal/mol) for :class:`VdwParams`."""

    def energy(positions_ang):
        return vdw_energy(_nm(positions_ang), params) * KJ_TO_KCAL

    return energy


def multipole_term(params: MultipoleParams) -> Callable:
    """Energy closure (Å -> kcal/mol) for :class:`MultipoleParams` (permanent + polarization)."""

    def energy(positions_ang):
        return multipole_energy(_nm(positions_ang), params) * KJ_TO_KCAL

    return energy


def torsion_torsion_term(params: TorsionTorsionParams) -> Callable:
    """Energy closure (Å -> kcal/mol) for :class:`TorsionTorsionParams`."""

    def energy(positions_ang):
        return torsion_torsion_energy(_nm(positions_ang), params) * KJ_TO_KCAL

    return energy


def build_amoeba_term(
    force, polarization: Optional[str] = None, vdw_lambda: float = 1.0
) -> Tuple[str, Callable]:
    """Convert one native AMOEBA force into ``(key, energy_fn)``.

    Parameters
    ----------
    force : openmm.AmoebaVdwForce | openmm.AmoebaMultipoleForce | openmm.AmoebaTorsionTorsionForce
    polarization : {None, 'mutual', 'direct', 'extrapolated'}
        Multipole polarization override; ``None`` keeps ``force.getPolarizationType()``.
    vdw_lambda : float
        ``AmoebaVdwLambda`` value; only 1 is supported.

    Returns
    -------
    key : str
        ``'amoeba_vdw'``, ``'amoeba_multipole'`` or ``'amoeba_torsion_torsion'``.
    energy_fn : callable
        ``energy_fn(positions_ang) -> kcal/mol``.
    """
    name = type(force).__name__
    if name == "AmoebaVdwForce":
        return _KEYS[name], vdw_term(vdw_from_openmm(force, vdw_lambda=vdw_lambda))
    if name == "AmoebaMultipoleForce":
        return _KEYS[name], multipole_term(multipole_from_openmm(force, polarization))
    if name == "AmoebaTorsionTorsionForce":
        return _KEYS[name], torsion_torsion_term(torsion_torsion_from_openmm(force))
    raise TypeError(f"{name} is not a native AMOEBA force handled by MARS.")


def _check_system(system):
    for force in system.getForces():
        name = type(force).__name__
        if name in _UNSUPPORTED:
            raise NotImplementedError(
                f"{name} (implicit solvent) is not supported; build the System from "
                "amoeba2018.xml without the *_gk.xml implicit-solvent file."
            )


def build_amoeba_terms(
    system, polarization: Optional[str] = None, vdw_lambda: float = 1.0
) -> Dict[str, Callable]:
    """Energy closures for every native AMOEBA force of an OpenMM ``System``.

    Parameters
    ----------
    system : openmm.System
        Built with ``nonbondedMethod=NoCutoff``.
    polarization : {None, 'mutual', 'direct', 'extrapolated'}
        Override for ``AmoebaMultipoleForce.getPolarizationType()``.
    vdw_lambda : float
        ``AmoebaVdwLambda`` value; only 1 is supported.

    Returns
    -------
    dict
        ``{'amoeba_vdw', 'amoeba_multipole', 'amoeba_torsion_torsion'}`` subset
        (one entry per force present) mapping to ``fn(positions_ang) -> kcal/mol``.
        Repeated force types get ``_2``, ``_3`` ... suffixes.
    """
    _check_system(system)
    terms: Dict[str, Callable] = {}
    for force in system.getForces():
        if not is_amoeba_force(force):
            continue
        key, fn = build_amoeba_term(force, polarization=polarization, vdw_lambda=vdw_lambda)
        k, n = key, 1
        while k in terms:
            n += 1
            k = f"{key}_{n}"
        terms[k] = fn
    return terms


def _multipole_params(system, polarization):
    _check_system(system)
    forces = [f for f in system.getForces() if type(f).__name__ == "AmoebaMultipoleForce"]
    if len(forces) != 1:
        raise ValueError(f"Expected exactly one AmoebaMultipoleForce, found {len(forces)}.")
    return multipole_from_openmm(forces[0], polarization)


def multipole_components(system, polarization: Optional[str] = None) -> Callable:
    """Debug helper: ``fn(positions_ang) -> {'permanent', 'polarization'}`` in kcal/mol."""
    params = _multipole_params(system, polarization)

    def components(positions_ang):
        e_perm, e_pol, _ = multipole_energy_components(_nm(positions_ang), params)
        return {"permanent": e_perm * KJ_TO_KCAL, "polarization": e_pol * KJ_TO_KCAL}

    return components


def induced_dipoles(system, polarization: Optional[str] = None) -> Callable:
    """``fn(positions_ang) -> (N, 3)`` induced dipoles in e·Å.

    These are the dipoles induced by the direct-scaled permanent field, the
    quantity returned by OpenMM's ``AmoebaMultipoleForce.getInducedDipoles``
    (which reports e·nm).
    """
    params = _multipole_params(system, polarization)

    def dipoles(positions_ang):
        _, _, mu_d = multipole_energy_components(_nm(positions_ang), params)
        return mu_d / ANG_TO_NM

    return dipoles


def atomic_dipoles(system, polarization: Optional[str] = None) -> Callable:
    """``fn(positions_ang) -> (N, 3)`` per-atom contributions to the molecular dipole.

    ``mu_i = q_i r_i + d_i + mu_ind_i`` (e·Å): the permanent monopole times the
    position, the permanent dipole rotated into the lab frame, and the induced
    dipole.  Their sum is the AMOEBA molecular dipole moment (OpenMM's
    ``getSystemMultipoleMoments`` dipole), so dipole derivatives — and IR
    intensities — include the multipoles and the polarization response.
    Quadrupoles do not contribute to the dipole moment.
    """
    from .multipole import lab_frame_multipoles

    params = _multipole_params(system, polarization)

    def dipoles(positions_ang):
        x = _nm(positions_ang)
        lab = lab_frame_multipoles(x, params)
        _, _, mu_d = multipole_energy_components(x, params)
        return lab.charge[:, None] * positions_ang + (lab.dipole + mu_d) / ANG_TO_NM

    return dipoles
