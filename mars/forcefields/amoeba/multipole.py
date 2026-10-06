"""
AMOEBA permanent multipoles and induced-dipole polarization (pure JAX).

Implements the electrostatics of ``openmm.AmoebaMultipoleForce`` for
non-periodic systems [Ponder2010]_:

* **Local frames.**  Charges, dipoles and (1/3-scaled, traceless)
  quadrupoles are given in a molecular frame defined by up to three
  neighbours (``ZThenX``, ``Bisector``, ``ZBisect``, ``ThreeFold``,
  ``ZOnly``, ``NoAxisType``) and rotated to the lab frame here.  ``ZThenX``
  sites with a y-atom flip the y-components of their moments when the
  chirality volume is negative, exactly as OpenMM does.  Because the
  rotation is written in JAX, frame torques reach the forces through
  automatic differentiation.
* **Permanent energy.**  All pairs, undamped, scaled by the covalent
  ``m`` factors (0, 0, 0.4, 0.8 for 1-2 .. 1-5).
* **Polarization.**  Thole-damped [Thole1981]_ permanent fields ``E_d``
  (direct scale) and ``E_p`` (polar scale) induce dipoles through the
  isotropic polarizabilities; the damping uses ``u = r / (pdamp_i pdamp_j)``
  and ``a = min(thole_i, thole_j)``.  With ``M`` the response matrix of the
  chosen scheme, ``mu_d = M E_d`` and the polarization energy is
  ``-1/2 mu_d . E_p`` (equal to OpenMM's ``-1/4 (mu_d . E_p + mu_p . E_d)``
  since ``M`` is symmetric):

  - ``mutual``: ``M = (1 - alpha T)^-1 alpha`` solved exactly with a dense
    ``3N x 3N`` linear solve (OpenMM iterates DIIS to a tolerance);
  - ``direct``: ``M = alpha``;
  - ``extrapolated``: OPT [Simmonett2015]_, ``mu = sum_k c'_k mu_k`` with
    ``mu_0 = alpha E`` and ``mu_k = alpha T mu_{k-1}`` (field of the induced
    dipoles only), ``c'_k = sum_{j >= k} c_j``.

  Every scheme yields an energy whose exact derivative is the force OpenMM
  computes, so ``jax.grad`` reproduces OpenMM forces (including the OPT
  ``mu T mu`` gradient terms) and ``jax.hessian`` is available.

All quantities are in OpenMM units (nm, e, kJ/mol); ``ONE_4PI_EPS0`` converts
``e^2/nm`` to kJ/mol.  Ground truth is ``AmoebaReferenceMultipoleForce.cpp``.

References
----------
.. [Ponder2010] J. W. Ponder et al., J. Phys. Chem. B 114, 2549 (2010).
.. [Thole1981] B. T. Thole, Chem. Phys. 59, 341 (1981).
.. [Simmonett2015] A. C. Simmonett, F. C. Pickard IV, Y. Shao, T. E. Cheatham III,
   B. R. Brooks, J. Chem. Phys. 143, 074115 (2015).
"""

from typing import NamedTuple

import jax.numpy as jnp

from .convert import (
    BISECTOR,
    NO_AXIS_TYPE,
    ONE_4PI_EPS0,
    THREE_FOLD,
    Z_BISECT,
    Z_ONLY,
    Z_THEN_X,
    MultipoleParams,
)


class LabMultipoles(NamedTuple):
    """Lab-frame permanent moments (e, e nm, e nm^2)."""

    charge: jnp.ndarray
    dipole: jnp.ndarray
    quadrupole: jnp.ndarray


# ── local frames ─────────────────────────────────────────────────────────────


def _safe_unit(v):
    n2 = jnp.sum(v * v, axis=-1, keepdims=True)
    n2 = jnp.where(n2 > 0.0, n2, 1.0)
    return v / jnp.sqrt(n2)


def frame_vectors(pos, params: MultipoleParams):
    """Rows ``(X, Y, Z)`` of each particle's local frame, ``(N, 3, 3)``.

    Mirrors ``applyRotationMatrixToParticle``.  Missing frame atoms are
    replaced by fixed dummy directions so that unused branches stay finite
    (no NaN leaks into gradients through ``jnp.where``).
    """
    n = pos.shape[0]
    idx = jnp.arange(n)
    atom_z = jnp.asarray(params.atom_z)
    atom_x = jnp.asarray(params.atom_x)
    atom_y = jnp.asarray(params.atom_y)
    axis = jnp.asarray(params.axis_type)[:, None]
    dtype = pos.dtype

    def rel(atoms, dummy):
        has = (atoms >= 0)[:, None]
        other = pos[jnp.where(atoms >= 0, atoms, idx)]
        return jnp.where(has, other - pos, jnp.asarray(dummy, dtype=dtype))

    uz = _safe_unit(rel(atom_z, [0.0, 0.0, 1.0]))
    ux = _safe_unit(rel(atom_x, [1.0, 0.0, 0.0]))
    uy = _safe_unit(rel(atom_y, [0.0, 1.0, 0.0]))

    ex = jnp.asarray([1.0, 0.0, 0.0], dtype=dtype)
    ey = jnp.asarray([0.0, 1.0, 0.0], dtype=dtype)
    z_only_x = jnp.where(jnp.abs(uz[:, :1]) < 0.866, ex, ey)

    z = uz
    z = jnp.where(axis == BISECTOR, _safe_unit(uz + ux), z)
    z = jnp.where(axis == THREE_FOLD, _safe_unit(uz + ux + uy), z)
    x = ux
    x = jnp.where(axis == Z_BISECT, _safe_unit(ux + uy), x)
    x = jnp.where(axis == Z_ONLY, z_only_x, x)

    no_frame = (axis == NO_AXIS_TYPE) | (atom_z < 0)[:, None]
    z = jnp.where(no_frame, jnp.asarray([0.0, 0.0, 1.0], dtype=dtype), z)
    x = jnp.where(no_frame, ex, x)

    x = _safe_unit(x - z * jnp.sum(z * x, axis=-1, keepdims=True))
    y = jnp.cross(z, x)
    return jnp.stack([x, y, z], axis=1)


def chirality_sign(pos, params: MultipoleParams):
    """``-1`` where OpenMM inverts the local y-axis moments (``checkChiral``), else ``+1``."""
    n = pos.shape[0]
    idx = jnp.arange(n)
    atom_y = jnp.asarray(params.atom_y)
    use = (jnp.asarray(params.axis_type) == Z_THEN_X) & (atom_y >= 0)
    py = pos[jnp.where(atom_y >= 0, atom_y, idx)]
    pz = pos[jnp.where(use, jnp.asarray(params.atom_z), idx)]
    px = pos[jnp.where(use, jnp.asarray(params.atom_x), idx)]
    vol = jnp.sum(jnp.cross(pz - py, px - py) * (pos - py), axis=-1)
    return jnp.where(use & (vol < 0.0), -1.0, 1.0)


def lab_frame_multipoles(pos, params: MultipoleParams) -> LabMultipoles:
    """Rotate local-frame multipoles to the lab frame (positions in nm)."""
    s = chirality_sign(pos, params)
    # y-flip: d_y, Q_xy, Q_yz change sign (and their symmetric partners)
    flip_d = jnp.stack([jnp.ones_like(s), s, jnp.ones_like(s)], axis=-1)
    one = jnp.ones_like(s)
    flip_q = jnp.stack(
        [
            jnp.stack([one, s, one], axis=-1),
            jnp.stack([s, one, s], axis=-1),
            jnp.stack([one, s, one], axis=-1),
        ],
        axis=-2,
    )
    d_loc = jnp.asarray(params.dipole) * flip_d
    q_loc = jnp.asarray(params.quadrupole) * flip_q
    rot = frame_vectors(pos, params)  # rows are the local axes in lab coordinates
    dipole = jnp.einsum("nk,nki->ni", d_loc, rot)
    quad = jnp.einsum("nki,nkl,nlj->nij", rot, q_loc, rot)
    return LabMultipoles(jnp.asarray(params.charge), dipole, quad)


# ── pair geometry and damping ────────────────────────────────────────────────


def _pair_geometry(pos):
    """``R[i, j] = x_j - x_i``, safe ``r`` and the off-diagonal mask."""
    n = pos.shape[0]
    R = pos[None, :, :] - pos[:, None, :]
    off = ~jnp.eye(n, dtype=bool)
    r2 = jnp.where(off, jnp.sum(R * R, axis=-1), 1.0)
    return R, jnp.sqrt(r2), off


def thole_factors(r, params: MultipoleParams):
    """Thole damping ``(l3, l5, l7)`` of ``getAndScaleInverseRs`` for all pairs."""
    pdamp = jnp.asarray(params.pdamp)
    thole = jnp.asarray(params.thole)
    damp = pdamp[:, None] * pdamp[None, :]
    gamma = jnp.minimum(thole[:, None], thole[None, :])
    has = damp != 0.0
    u = r / jnp.where(has, damp, 1.0)
    au3 = gamma * u * u * u
    active = has & (au3 < 50.0)
    expo = jnp.where(active, jnp.exp(-jnp.where(active, au3, 0.0)), 0.0)
    au3 = jnp.where(active, au3, 0.0)
    l3 = 1.0 - expo
    l5 = 1.0 - (1.0 + au3) * expo
    l7 = 1.0 - (1.0 + au3 + 0.6 * au3 * au3) * expo
    return l3, l5, l7


# ── permanent multipole energy ───────────────────────────────────────────────


def permanent_energy(pos, params: MultipoleParams, lab: LabMultipoles = None):
    """Permanent multipole-multipole energy (kJ/mol), m-scaled, undamped."""
    if lab is None:
        lab = lab_frame_multipoles(pos, params)
    R, r, off = _pair_geometry(pos)
    c, d, Q = lab.charge, lab.dipole, lab.quadrupole
    rr1 = 1.0 / r
    rr2 = rr1 * rr1
    rr3 = rr1 * rr2
    rr5 = 3.0 * rr3 * rr2
    rr7 = 5.0 * rr5 * rr2
    rr9 = 7.0 * rr7 * rr2

    dir_ = jnp.einsum("ia,ija->ij", d, R)
    dkr = jnp.einsum("ja,ija->ij", d, R)
    qi_r = jnp.einsum("iab,ijb->ija", Q, R)
    qk_r = jnp.einsum("jab,ijb->ija", Q, R)
    qir = jnp.sum(qi_r * R, axis=-1)
    qkr = jnp.sum(qk_r * R, axis=-1)
    dik = d @ d.T
    qik = jnp.sum(qi_r * qk_r, axis=-1)
    diqk = jnp.einsum("ia,ija->ij", d, qk_r)
    dkqi = jnp.einsum("ja,ija->ij", d, qi_r)
    qiqk = jnp.einsum("iab,jab->ij", Q, Q)
    ci, ck = c[:, None], c[None, :]

    term1 = ci * ck
    term2 = ck * dir_ - ci * dkr + dik
    term3 = ci * qkr + ck * qir - dir_ * dkr + 2.0 * (dkqi - diqk + qiqk)
    term4 = dir_ * qkr - dkr * qir - 4.0 * qik
    term5 = qir * qkr
    e = term1 * rr1 + term2 * rr3 + term3 * rr5 + term4 * rr7 + term5 * rr9
    scale = jnp.where(off, jnp.asarray(params.m_scale), 0.0)
    return 0.5 * ONE_4PI_EPS0 * jnp.sum(scale * e)


# ── fields and polarization ──────────────────────────────────────────────────


def permanent_fields(pos, params: MultipoleParams, lab: LabMultipoles = None):
    """Thole-damped permanent fields ``(E_d, E_p)`` at every site (e/nm^2).

    ``E_d`` uses the direct (``d``) scale factors and ``E_p`` the polar
    (``p``) ones, as ``calculateFixedMultipoleField`` in OpenMM.
    """
    if lab is None:
        lab = lab_frame_multipoles(pos, params)
    R, r, off = _pair_geometry(pos)
    Rf = -R  # vector from source j to field point i
    l3, l5, l7 = thole_factors(r, params)
    rr1 = 1.0 / r
    rr2 = rr1 * rr1
    rr3 = l3 * rr1 * rr2
    rr5 = 3.0 * l5 * rr1 * rr2 * rr2
    rr7 = 15.0 * l7 * rr1 * rr2 * rr2 * rr2
    c, d, Q = lab.charge, lab.dipole, lab.quadrupole
    dR = jnp.einsum("ja,ija->ij", d, Rf)
    QR = jnp.einsum("jab,ijb->ija", Q, Rf)
    RQR = jnp.sum(QR * Rf, axis=-1)
    radial = rr3 * c[None, :] + rr5 * dR + rr7 * RQR
    field = Rf * radial[..., None] - rr3[..., None] * d[None, :, :] - 2.0 * rr5[..., None] * QR
    dsc = jnp.where(off, jnp.asarray(params.d_scale), 0.0)
    psc = jnp.where(off, jnp.asarray(params.p_scale), 0.0)
    e_d = jnp.einsum("ij,ija->ia", dsc, field)
    e_p = jnp.einsum("ij,ija->ia", psc, field)
    return e_d, e_p


def dipole_tensor(pos, params: MultipoleParams):
    """Damped dipole-dipole field tensor ``T`` with ``E_i = sum_j T_ij mu_j``, ``(N, 3, N, 3)``."""
    R, r, off = _pair_geometry(pos)
    l3, l5, _ = thole_factors(r, params)
    rr1 = 1.0 / r
    rr2 = rr1 * rr1
    rr3 = l3 * rr1 * rr2
    rr5 = 3.0 * l5 * rr1 * rr2 * rr2
    scale = jnp.where(off, jnp.asarray(params.u_scale), 0.0)
    eye = jnp.eye(3, dtype=pos.dtype)
    t = rr5[..., None, None] * R[..., :, None] * R[..., None, :] - rr3[..., None, None] * eye
    t = t * scale[..., None, None]
    return jnp.transpose(t, (0, 2, 1, 3))


def induced_dipoles(pos, params: MultipoleParams, field):
    """Induced dipoles (e nm) responding to ``field`` ``(N, 3)`` for ``params.polarization``."""
    alpha = jnp.asarray(params.polarity)
    mu0 = alpha[:, None] * field
    if params.polarization == "direct":
        return mu0
    n = pos.shape[0]
    T = dipole_tensor(pos, params).reshape(3 * n, 3 * n)
    a3 = jnp.repeat(alpha, 3)
    if params.polarization == "mutual":
        A = jnp.eye(3 * n, dtype=pos.dtype) - a3[:, None] * T
        return jnp.linalg.solve(A, mu0.reshape(-1)).reshape(n, 3)
    # extrapolated (OPT): mu = sum_k c'_k (alpha T)^k alpha E
    ext = [float(c) for c in params.extrapolation]
    mu_k = mu0.reshape(-1)
    total = ext[0] * mu_k
    for c in ext[1:]:
        mu_k = a3 * (T @ mu_k)
        total = total + c * mu_k
    return total.reshape(n, 3)


def multipole_energy_components(pos, params: MultipoleParams):
    """Permanent and polarization energies (kJ/mol) and the induced dipoles.

    Returns
    -------
    e_perm, e_pol : scalar
    mu_d : (N, 3)
        Induced dipoles from the direct-scaled field (OpenMM's ``getInducedDipoles``).
    """
    lab = lab_frame_multipoles(pos, params)
    e_perm = permanent_energy(pos, params, lab)
    e_d, e_p = permanent_fields(pos, params, lab)
    mu_d = induced_dipoles(pos, params, e_d)
    e_pol = -0.5 * ONE_4PI_EPS0 * jnp.sum(mu_d * e_p)
    return e_perm, e_pol, mu_d


def multipole_energy(pos, params: MultipoleParams):
    """Total ``AmoebaMultipoleForce`` energy (kJ/mol) for positions in nm."""
    e_perm, e_pol, _ = multipole_energy_components(pos, params)
    return e_perm + e_pol
