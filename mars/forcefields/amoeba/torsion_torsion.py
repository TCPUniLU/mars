"""
AMOEBA torsion-torsion coupling (pure JAX).

Each term couples the two consecutive dihedrals ``a-b-c-d`` and ``b-c-d-e``
(typically the backbone phi/psi pair) through a 2D energy grid that stores
the energy and its first and mixed derivatives; the energy between grid
points follows from bicubic interpolation (Numerical Recipes, Sec. 3.6)
[Ponder2010]_.  When a chirality-check atom is given and the signed volume
``(chiral - c) . ((b - c) x (d - c))`` is negative, both angles change sign,
as in ``AmoebaReferenceTorsionTorsionForce.cpp``.

Dihedrals are evaluated with ``atan2`` (same value and sign convention as
OpenMM's ``acos``-plus-sign construction, but with finite derivatives at
0 and 180 degrees).  Cell lookup and the bicubic coefficients are precomputed
in :func:`~mars.forcefields.amoeba.convert.torsion_torsion_from_openmm`.

References
----------
.. [Ponder2010] J. W. Ponder et al., J. Phys. Chem. B 114, 2549 (2010).
"""

import jax.numpy as jnp

from .convert import TorsionTorsionParams

_RAD2DEG = 180.0 / jnp.pi


def _dihedral_deg(p0, p1, p2, p3):
    """OpenMM-convention dihedral angle (degrees) of four ``(T, 3)`` point sets."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    t = jnp.cross(b1, b2)
    u = jnp.cross(b2, b3)
    nb2 = jnp.sqrt(jnp.sum(b2 * b2, axis=-1))
    y = nb2 * jnp.sum(b1 * u, axis=-1)
    x = jnp.sum(t * u, axis=-1)
    return _RAD2DEG * jnp.arctan2(y, x)


def torsion_torsion_energy(pos_nm, params: TorsionTorsionParams):
    """Torsion-torsion energy (kJ/mol) for positions in nm."""
    if params.atoms.shape[0] == 0:
        return jnp.zeros((), dtype=pos_nm.dtype)
    atoms = jnp.asarray(params.atoms)
    pa, pb, pc, pd, pe = (pos_nm[atoms[:, k]] for k in range(5))
    angle1 = _dihedral_deg(pa, pb, pc, pd)
    angle2 = _dihedral_deg(pb, pc, pd, pe)

    chiral = jnp.asarray(params.chiral)
    has_chiral = chiral >= 0
    pch = pos_nm[jnp.where(has_chiral, chiral, atoms[:, 2])]
    vol = jnp.sum((pch - pc) * jnp.cross(pb - pc, pd - pc), axis=-1)
    sign = jnp.where(has_chiral & (vol < 0.0), -1.0, 1.0)
    angle1 = sign * angle1
    angle2 = sign * angle2

    g = jnp.asarray(params.grid)
    start = jnp.asarray(params.start)[g]
    sinv = jnp.asarray(params.spacing_inv)[g]
    ncell = jnp.asarray(params.n_cells)[g]
    ix = jnp.floor((angle1 - start[:, 0]) * sinv + 1.0e-6).astype(jnp.int32)
    iy = jnp.floor((angle2 - start[:, 1]) * sinv + 1.0e-6).astype(jnp.int32)
    ix = jnp.clip(ix, 0, ncell[:, 0] - 1)
    iy = jnp.clip(iy, 0, ncell[:, 1] - 1)

    lower = jnp.asarray(params.lower)[g, ix, iy]
    width = jnp.asarray(params.width)[g, ix, iy]
    c = jnp.asarray(params.coeff)[g, ix, iy]  # (T, 4, 4)
    t = (angle1 - lower[:, 0]) / width[:, 0]
    u = (angle2 - lower[:, 1]) / width[:, 1]
    one = jnp.ones_like(t)
    tp = jnp.stack([one, t, t * t, t * t * t], axis=-1)
    up = jnp.stack([one, u, u * u, u * u * u], axis=-1)
    return jnp.sum(jnp.einsum("ti,tij,tj->t", tp, c, up))
