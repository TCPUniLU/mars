"""
OpenMM AMOEBA forces -> frozen NumPy parameter records.

Each ``*_from_openmm`` function reads one native AMOEBA force of an OpenMM
``System`` and returns an immutable :class:`typing.NamedTuple` of NumPy arrays
that the pure-JAX energy kernels (:mod:`.vdw`, :mod:`.multipole`,
:mod:`.torsion_torsion`) consume.  All pair lists, scale-factor matrices and
interpolation coefficients are precomputed here, so the energy functions
contain no Python control flow on traced values.

Parameters are kept in OpenMM units (nm, kJ/mol, e, e nm, e nm^2, nm^3); the
energy wrappers convert positions (Å -> nm) and energies (kJ/mol -> kcal/mol)
at the boundary.

The conventions mirror the OpenMM Reference platform
(``plugins/amoeba/platforms/reference``):

* vdW: per-type sigma/epsilon matrices with the force's combining rules and
  type-pair overrides (``AmoebaVdwForceImpl::createParameterMatrix``); pairs
  listed in ``getParticleExclusions`` are removed entirely (AMOEBA has no
  partial vdW scaling in OpenMM), every other pair is scaled by the per-particle
  ``scaleFactor`` product.
* Multipoles: the m/p/d/u scale factors of
  ``AmoebaReferenceMultipoleForce::setupScaleMaps`` (m = 0, 0, 0.4, 0.8 for
  1-2..1-5; p = 0, 0, 1, 1 with 1-4 pairs inside the same polarization group
  scaled by 0.5; d = 0 within the polarization group (``PolarizationCovalent11``)
  and 1 otherwise; u = 1).
"""

from typing import NamedTuple, Optional, Tuple

import numpy as np

#: OpenMM's ``ONE_4PI_EPS0`` (kJ mol^-1 nm e^-2), from CODATA 2018 constants.
ONE_4PI_EPS0 = 138.93545764438198

#: Axis-type codes, identical to ``openmm.AmoebaMultipoleForce`` enums.
Z_THEN_X, BISECTOR, Z_BISECT, THREE_FOLD, Z_ONLY, NO_AXIS_TYPE = range(6)

#: Polarization-type names indexed by ``AmoebaMultipoleForce.getPolarizationType()``.
POLARIZATION_TYPES = ("mutual", "direct", "extrapolated")

# Fixed scale factors of AmoebaReferenceMultipoleForce::initialize().  Index 0
# is the 1-1 slot, 1..4 are 1-2..1-5 (m, p) or p11..p14 shifted by one (d, u).
_M_SCALE = (0.0, 0.0, 0.0, 0.4, 0.8)
_P_SCALE = (0.0, 0.0, 0.0, 1.0, 1.0)
_D_SCALE = (0.0, 1.0, 1.0, 1.0, 1.0)
_U_SCALE = (1.0, 1.0, 1.0, 1.0, 1.0)
_P14_INTRA_FACTOR = 0.5

# Bicubic weight table of AmoebaReferenceTorsionTorsionForce (Numerical
# Recipes, Sec. 3.6), stored transposed as in OpenMM.
_BICUBIC_WEIGHTS = np.array(
    [
        [1, 0, -3, 2, 0, 0, 0, 0, -3, 0, 9, -6, 2, 0, -6, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, -9, 6, -2, 0, 6, -4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, -6, 0, 0, -6, 4],
        [0, 0, 3, -2, 0, 0, 0, 0, 0, 0, -9, 6, 0, 0, 6, -4],
        [0, 0, 0, 0, 1, 0, -3, 2, -2, 0, 6, -4, 1, 0, -3, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 3, -2, 1, 0, -3, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 2, 0, 0, 3, -2],
        [0, 0, 0, 0, 0, 0, 3, -2, 0, 0, -6, 4, 0, 0, 3, -2],
        [0, 1, -2, 1, 0, 0, 0, 0, 0, -3, 6, -3, 0, 2, -4, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -6, 3, 0, -2, 4, -2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, 2, -2],
        [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 3, -3, 0, 0, -2, 2],
        [0, 0, 0, 0, 0, 1, -2, 1, 0, -2, 4, -2, 0, 1, -2, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 1, -2, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, -1, 1],
        [0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 2, -2, 0, 0, -1, 1],
    ],
    dtype=float,
)


class VdwParams(NamedTuple):
    """Buffered 14-7 (or LJ) parameters of an ``AmoebaVdwForce``.

    Attributes
    ----------
    parent : (N,) int
        Reduction parent of each particle (itself for heavy atoms).
    reduction : (N,) float
        Hydrogen reduction factor ``f``; the interaction site is
        ``parent + f * (x - parent)`` when ``f != 0``, else ``x``.
    pair_i, pair_j : (P,) int
        Interacting pairs ``i < j`` (exclusions removed).
    sigma, epsilon : (P,) float
        Combined pair parameters (nm, kJ/mol), per-particle scale factors folded in.
    lennard_jones : bool
        ``True`` for ``AmoebaVdwForce.LennardJones``, else buffered 14-7.
    """

    parent: np.ndarray
    reduction: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray
    sigma: np.ndarray
    epsilon: np.ndarray
    lennard_jones: bool


class MultipoleParams(NamedTuple):
    """Permanent multipoles, polarizabilities and scale factors of an ``AmoebaMultipoleForce``.

    Attributes
    ----------
    charge : (N,) float
        Monopoles (e).
    dipole : (N, 3) float
        Local-frame dipoles (e nm).
    quadrupole : (N, 3, 3) float
        Local-frame quadrupoles (e nm^2), already carrying OpenMM's (Tinker's)
        factor 1/3, i.e. the potential of a quadrupole is ``3 r.Q.r / r^5``.
    axis_type : (N,) int
        Local-frame definition (see the module constants).
    atom_z, atom_x, atom_y : (N,) int
        Frame-defining atoms, ``-1`` when absent.
    thole, pdamp, polarity : (N,) float
        Thole parameter, damping factor ``alpha^(1/6)`` (nm^(1/2)) and isotropic
        polarizability (nm^3).
    m_scale, p_scale, d_scale, u_scale : (N, N) float
        Symmetric pair scale factors (diagonal unused).
    polarization : str
        ``'mutual'``, ``'direct'`` or ``'extrapolated'``.
    extrapolation : (K,) float
        Partial sums ``sum_{j >= k} c_j`` of the OPT coefficients (only used
        for ``'extrapolated'``).
    """

    charge: np.ndarray
    dipole: np.ndarray
    quadrupole: np.ndarray
    axis_type: np.ndarray
    atom_z: np.ndarray
    atom_x: np.ndarray
    atom_y: np.ndarray
    thole: np.ndarray
    pdamp: np.ndarray
    polarity: np.ndarray
    m_scale: np.ndarray
    p_scale: np.ndarray
    d_scale: np.ndarray
    u_scale: np.ndarray
    polarization: str
    extrapolation: np.ndarray


class TorsionTorsionParams(NamedTuple):
    """Bicubic torsion-torsion (CMAP-like) terms of an ``AmoebaTorsionTorsionForce``.

    Attributes
    ----------
    atoms : (T, 5) int
        The five atoms a-b-c-d-e defining the two dihedrals a-b-c-d and b-c-d-e.
    chiral : (T,) int
        Chirality-check atom (``-1`` for none).
    grid : (T,) int
        Grid index of each term.
    start : (G, 2) float
        First grid angles (degrees) of each grid.
    spacing_inv : (G,) float
        ``(n_x - 1) / 360`` used by OpenMM to locate the enclosing cell.
    n_cells : (G, 2) int
        Number of cells along each axis.
    lower, width : (G, Cx, Cy, 2) float
        Lower-corner angles and cell widths (degrees).
    coeff : (G, Cx, Cy, 4, 4) float
        Bicubic coefficients ``c[i, j]`` of ``sum c_ij t^i u^j`` (kJ/mol).
    """

    atoms: np.ndarray
    chiral: np.ndarray
    grid: np.ndarray
    start: np.ndarray
    spacing_inv: np.ndarray
    n_cells: np.ndarray
    lower: np.ndarray
    width: np.ndarray
    coeff: np.ndarray


# ── helpers ──────────────────────────────────────────────────────────────────


def _val(q):
    """Strip an OpenMM ``Quantity`` (in its own default unit system)."""
    if hasattr(q, "value_in_unit_system"):
        from openmm import unit as u

        return q.value_in_unit_system(u.md_unit_system)
    return q


def _vec(q):
    return np.asarray([float(_val(x)) for x in _val(q)], dtype=float)


# ── vdW ──────────────────────────────────────────────────────────────────────


def _combine(rule_sigma: str, rule_eps: str, si, sj, ei, ej) -> Tuple[float, float]:
    if rule_sigma == "ARITHMETIC":
        sigma = si + sj
    elif rule_sigma == "GEOMETRIC":
        sigma = 2.0 * np.sqrt(si * sj)
    elif rule_sigma == "CUBIC-MEAN":
        s2 = si * si + sj * sj
        sigma = 2.0 * (si**3 + sj**3) / s2 if s2 != 0.0 else 0.0
    else:
        raise ValueError(f"AmoebaVdwForce: unknown sigma combining rule {rule_sigma!r}")

    if rule_eps == "ARITHMETIC":
        eps = 0.5 * (ei + ej)
    elif rule_eps == "GEOMETRIC":
        eps = np.sqrt(ei * ej)
    elif rule_eps == "HARMONIC":
        eps = 2.0 * ei * ej / (ei + ej) if (ei + ej) != 0.0 else 0.0
    elif rule_eps == "W-H":
        s3i, s3j = si**3, sj**3
        es = np.sqrt(ei * ej)
        eps = 0.0 if es == 0.0 else 2.0 * es * s3i * s3j / (s3i * s3i + s3j * s3j)
    elif rule_eps == "HHG":
        d = np.sqrt(ei) + np.sqrt(ej)
        eps = 4.0 * ei * ej / (d * d) if d != 0.0 else 0.0
    else:
        raise ValueError(f"AmoebaVdwForce: unknown epsilon combining rule {rule_eps!r}")
    return sigma, eps


def vdw_from_openmm(force, vdw_lambda: float = 1.0) -> VdwParams:
    """Extract an ``openmm.AmoebaVdwForce`` (NoCutoff only).

    Parameters
    ----------
    force : openmm.AmoebaVdwForce
    vdw_lambda : float
        Value of the ``AmoebaVdwLambda`` context parameter.  Only ``1.0`` (the
        fully coupled state, where softcore terms vanish) is supported.

    Returns
    -------
    VdwParams
    """
    import openmm

    if force.getNonbondedMethod() != openmm.AmoebaVdwForce.NoCutoff:
        raise NotImplementedError("AmoebaVdwForce: only NoCutoff is supported in MARS.")
    # Alchemical particles are harmless at lambda = 1: the softcore offset and
    # the lambda^n epsilon factor both reduce to the plain potential.
    if vdw_lambda != 1.0:
        raise NotImplementedError("AmoebaVdwForce: alchemical lambda != 1 is not supported.")

    n = force.getNumParticles()
    parent = np.zeros(n, dtype=np.int64)
    reduction = np.zeros(n)
    scale = np.ones(n)
    p_sigma = np.zeros(n)
    p_eps = np.zeros(n)
    p_type = np.zeros(n, dtype=np.int64)
    exclusions = []
    for i in range(n):
        iv, sig, eps, red, _alch, typ, sf = force.getParticleParameters(i)
        parent[i] = iv
        reduction[i] = red
        scale[i] = sf
        p_sigma[i] = _val(sig)
        p_eps[i] = _val(eps)
        p_type[i] = typ
        exclusions.append(set(force.getParticleExclusions(i)))

    # type table, exactly as AmoebaVdwForceImpl::createParameterMatrix
    if force.getUseParticleTypes():
        n_types = force.getNumParticleTypes()
        t_sigma = np.zeros(n_types)
        t_eps = np.zeros(n_types)
        for t in range(n_types):
            s, e = force.getParticleTypeParameters(t)
            t_sigma[t], t_eps[t] = _val(s), _val(e)
        types = p_type
    else:
        keys = {}
        types = np.zeros(n, dtype=np.int64)
        for i in range(n):
            key = (p_sigma[i], p_eps[i])
            if key not in keys:
                keys[key] = len(keys)
            types[i] = keys[key]
        n_types = len(keys)
        t_sigma = np.zeros(n_types)
        t_eps = np.zeros(n_types)
        for (s, e), t in keys.items():
            t_sigma[t], t_eps[t] = s, e

    rs, re = force.getSigmaCombiningRule(), force.getEpsilonCombiningRule()
    sig_m = np.zeros((n_types, n_types))
    eps_m = np.zeros((n_types, n_types))
    for a in range(n_types):
        for b in range(a, n_types):
            s, e = _combine(rs, re, t_sigma[a], t_sigma[b], t_eps[a], t_eps[b])
            sig_m[a, b] = sig_m[b, a] = s
            eps_m[a, b] = eps_m[b, a] = e
    if force.getUseParticleTypes():
        for k in range(force.getNumTypePairs()):
            a, b, s, e = force.getTypePairParameters(k)
            sig_m[a, b] = sig_m[b, a] = _val(s)
            eps_m[a, b] = eps_m[b, a] = _val(e)

    ii, jj = np.triu_indices(n, k=1)
    keep = np.array([j not in exclusions[i] for i, j in zip(ii, jj)], dtype=bool)
    ii, jj = ii[keep], jj[keep]
    sigma = sig_m[types[ii], types[jj]]
    epsilon = eps_m[types[ii], types[jj]] * scale[ii] * scale[jj]
    if force.getPotentialFunction() == openmm.AmoebaVdwForce.Buffered147 and np.any(sigma <= 0):
        raise ValueError("AmoebaVdwForce: buffered 14-7 needs sigma > 0 for every pair.")
    return VdwParams(
        parent=parent,
        reduction=reduction,
        pair_i=ii.astype(np.int64),
        pair_j=jj.astype(np.int64),
        sigma=sigma,
        epsilon=epsilon,
        lennard_jones=force.getPotentialFunction() == openmm.AmoebaVdwForce.LennardJones,
    )


# ── multipoles ───────────────────────────────────────────────────────────────


def scale_matrices(covalent_maps) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build dense m/p/d/u scale matrices from ``getCovalentMaps`` output.

    Follows ``AmoebaReferenceMultipoleForce::setupScaleMaps`` literally: for each
    particle ``i`` only partners ``k >= i`` are recorded, later covalent lists
    overwrite earlier ones, and the result for the pair ``(i, k)`` with
    ``i < k`` is the one stored by ``i``.

    Parameters
    ----------
    covalent_maps : sequence of 8 sequences per particle
        ``[c12, c13, c14, c15, p11, p12, p13, p14]`` partner lists.

    Returns
    -------
    m, p, d, u : (N, N) ndarray
        Symmetric scale matrices with a zero diagonal.
    """
    n = len(covalent_maps)
    mats = [np.ones((n, n)) for _ in range(4)]
    m, p, d, u = mats
    for i, info in enumerate(covalent_maps):
        p11 = set(info[4])
        for jj in range(4):
            for k in info[jj]:
                if k < i:
                    continue
                hit = jj == 2 and k in p11
                p[i, k] = _P14_INTRA_FACTOR * _P_SCALE[jj + 1] if hit else _P_SCALE[jj + 1]
                m[i, k] = _M_SCALE[jj + 1]
        for jj in range(4, 8):
            for k in info[jj]:
                if k < i:
                    continue
                d[i, k] = _D_SCALE[jj - 4]
                u[i, k] = _U_SCALE[jj - 4]
    iu = np.triu_indices(n, k=1)
    out = []
    for a in mats:
        s = np.zeros((n, n))
        s[iu] = a[iu]
        out.append(s + s.T)
    return tuple(out)


def multipole_from_openmm(force, polarization: Optional[str] = None) -> MultipoleParams:
    """Extract an ``openmm.AmoebaMultipoleForce`` (NoCutoff only).

    Parameters
    ----------
    force : openmm.AmoebaMultipoleForce
    polarization : {None, 'mutual', 'direct', 'extrapolated'}
        Override of ``force.getPolarizationType()``.

    Returns
    -------
    MultipoleParams
    """
    import openmm

    if force.getNonbondedMethod() != openmm.AmoebaMultipoleForce.NoCutoff:
        raise NotImplementedError("AmoebaMultipoleForce: only NoCutoff is supported (no PME).")
    if polarization is None:
        polarization = POLARIZATION_TYPES[force.getPolarizationType()]
    polarization = polarization.lower()
    if polarization not in POLARIZATION_TYPES:
        raise ValueError(f"polarization must be one of {POLARIZATION_TYPES}, got {polarization!r}")

    n = force.getNumMultipoles()
    charge = np.zeros(n)
    dipole = np.zeros((n, 3))
    quad = np.zeros((n, 3, 3))
    axis = np.zeros(n, dtype=np.int64)
    az, ax, ay = (np.full(n, -1, dtype=np.int64) for _ in range(3))
    thole, pdamp, polarity = np.zeros(n), np.zeros(n), np.zeros(n)
    cov = []
    for i in range(n):
        q, d, qp, at, z, x, y, th, dmp, pol = force.getMultipoleParameters(i)
        charge[i] = _val(q)
        dipole[i] = _vec(d)
        quad[i] = _vec(qp).reshape(3, 3)
        axis[i], az[i], ax[i], ay[i] = at, z, x, y
        thole[i], pdamp[i], polarity[i] = th, _val(dmp), _val(pol)
        cov.append([list(c) for c in force.getCovalentMaps(i)])
    # OpenMM only reads the six unique quadrupole components; symmetrize.
    quad = 0.5 * (quad + np.transpose(quad, (0, 2, 1)))
    m, p, dsc, u = scale_matrices(cov)

    coeffs = np.asarray(force.getExtrapolationCoefficients(), dtype=float)
    ext = np.array([coeffs[k:].sum() for k in range(len(coeffs))]) if coeffs.size else coeffs
    if polarization == "extrapolated" and ext.size == 0:
        raise ValueError("Extrapolated polarization requires extrapolation coefficients.")
    return MultipoleParams(
        charge=charge,
        dipole=dipole,
        quadrupole=quad,
        axis_type=axis,
        atom_z=az,
        atom_x=ax,
        atom_y=ay,
        thole=thole,
        pdamp=pdamp,
        polarity=polarity,
        m_scale=m,
        p_scale=p,
        d_scale=dsc,
        u_scale=u,
        polarization=polarization,
        extrapolation=ext,
    )


# ── torsion-torsion ──────────────────────────────────────────────────────────


def bicubic_coefficients(y, y1, y2, y12, d1, d2) -> np.ndarray:
    """4x4 bicubic coefficient matrix (``getBicubicCoefficientMatrix`` in OpenMM).

    Corner order is (x0, y0), (x1, y0), (x1, y1), (x0, y1).
    """
    x = np.concatenate([y, np.asarray(y1) * d1, np.asarray(y2) * d2, np.asarray(y12) * d1 * d2])
    return (_BICUBIC_WEIGHTS.T @ x).reshape(4, 4)


def _reorder_grid(grid: np.ndarray) -> np.ndarray:
    """Sort a flattened grid so angle1 is the slow index (``reorderGrid``)."""
    flat = grid.reshape(-1, grid.shape[-1])
    order = np.lexsort((flat[:, 1], flat[:, 0]))
    return flat[order].reshape(grid.shape)


def torsion_torsion_from_openmm(force) -> TorsionTorsionParams:
    """Extract an ``openmm.AmoebaTorsionTorsionForce``.

    Returns
    -------
    TorsionTorsionParams
    """
    if force.usesPeriodicBoundaryConditions():
        raise NotImplementedError(
            "AmoebaTorsionTorsionForce: periodic boundaries are not supported."
        )
    nt = force.getNumTorsionTorsions()
    atoms = np.zeros((nt, 5), dtype=np.int64)
    chiral = np.full(nt, -1, dtype=np.int64)
    gidx = np.zeros(nt, dtype=np.int64)
    for t in range(nt):
        a, b, c, d, e, ch, g = force.getTorsionTorsionParameters(t)
        atoms[t] = (a, b, c, d, e)
        chiral[t] = ch
        gidx[t] = g

    grids = [
        np.asarray(force.getTorsionTorsionGrid(g), dtype=float)
        for g in range(force.getNumTorsionTorsionGrids())
    ]
    return torsion_torsion_params(atoms, chiral, gidx, grids)


def torsion_torsion_params(atoms, chiral, grid_index, grids) -> TorsionTorsionParams:
    """Build :class:`TorsionTorsionParams` from raw arrays.

    Parameters
    ----------
    atoms : (T, 5) int
    chiral : (T,) int
        Chirality-check atoms (``-1`` for none).
    grid_index : (T,) int
    grids : list of (nx, ny, 6) arrays
        ``[angle1, angle2, f, df/dangle1, df/dangle2, d2f/dangle1dangle2]`` with
        angles in degrees and energies in kJ/mol (``getTorsionTorsionGrid``).

    Returns
    -------
    TorsionTorsionParams
    """
    atoms = np.asarray(atoms, dtype=np.int64).reshape(-1, 5)
    chiral = np.asarray(chiral, dtype=np.int64).reshape(-1)
    gidx = np.asarray(grid_index, dtype=np.int64).reshape(-1)
    ng = len(grids)
    checked = []
    for grid in grids:
        grid = np.asarray(grid, dtype=float)
        if grid.ndim != 3 or grid.shape[-1] != 6:
            raise ValueError(
                "AmoebaTorsionTorsionForce: grids must provide f, df/dx, df/dy and d2f/dxdy."
            )
        if grid[0, 0, 0] != grid[0, 1, 0]:
            grid = _reorder_grid(grid)
        checked.append(grid)
    grids = checked
    cx = max((gr.shape[0] - 1 for gr in grids), default=1)
    cy = max((gr.shape[1] - 1 for gr in grids), default=1)
    start = np.zeros((max(ng, 1), 2))
    spacing_inv = np.ones(max(ng, 1))
    n_cells = np.ones((max(ng, 1), 2), dtype=np.int64)
    lower = np.zeros((max(ng, 1), cx, cy, 2))
    width = np.ones((max(ng, 1), cx, cy, 2))
    coeff = np.zeros((max(ng, 1), cx, cy, 4, 4))
    for g, grid in enumerate(grids):
        nx, ny = grid.shape[:2]
        start[g] = grid[0, 0, :2]
        spacing_inv[g] = (nx - 1) / 360.0
        n_cells[g] = (nx - 1, ny - 1)
        for ix in range(nx - 1):
            for iy in range(ny - 1):
                x_lo, x_hi = grid[ix, iy, 0], grid[ix + 1, iy, 0]
                y_lo, y_hi = grid[ix, iy, 1], grid[ix + 1, iy + 1, 1]
                corners = [grid[ix, iy], grid[ix + 1, iy], grid[ix + 1, iy + 1], grid[ix, iy + 1]]
                vals = np.array(corners)
                lower[g, ix, iy] = (x_lo, y_lo)
                width[g, ix, iy] = (x_hi - x_lo, y_hi - y_lo)
                coeff[g, ix, iy] = bicubic_coefficients(
                    vals[:, 2], vals[:, 3], vals[:, 4], vals[:, 5], x_hi - x_lo, y_hi - y_lo
                )
    return TorsionTorsionParams(
        atoms=atoms,
        chiral=chiral,
        grid=gidx,
        start=start,
        spacing_inv=spacing_inv,
        n_cells=n_cells,
        lower=lower,
        width=width,
        coeff=coeff,
    )
