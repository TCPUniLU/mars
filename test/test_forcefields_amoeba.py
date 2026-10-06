"""
Tests for the native-JAX AMOEBA terms (mars.forcefields.amoeba).

The first part uses synthetic parameters only (no OpenMM): finite-difference
gradients, rigid-motion invariance, frame/chirality behaviour, mutual vs OPT
polarization and the bicubic torsion-torsion interpolation.  The second part
checks energies and forces against the OpenMM Reference platform and is
skipped when OpenMM is not installed.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mars.forcefields.amoeba import MultipoleParams, VdwParams, scale_matrices
from mars.forcefields.amoeba.convert import torsion_torsion_params
from mars.forcefields.amoeba.multipole import (
    lab_frame_multipoles,
    multipole_energy,
    multipole_energy_components,
)
from mars.forcefields.amoeba.torsion_torsion import torsion_torsion_energy
from mars.forcefields.amoeba.vdw import vdw_energy

jax.config.update("jax_enable_x64", True)


# ── synthetic systems ────────────────────────────────────────────────────────


def _water_cluster(n_mol=3, seed=0):
    """Rigid-ish water molecules (nm) with random orientations."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0, 0], [0.29, 0, 0], [0.14, 0.25, 0.03], [0.15, 0.1, 0.26]])[:n_mol]
    th = np.radians(54.25)
    loc = 0.09572 * np.array([[0, 0, 0], [np.sin(th), np.cos(th), 0], [-np.sin(th), np.cos(th), 0]])
    pos = []
    for c in centers:
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        pos.extend(c + loc @ q.T)
    return np.array(pos)


def _covalent_maps(n_mol):
    maps = []
    for i in range(3 * n_mol):
        base = 3 * (i // 3)
        mol = [base, base + 1, base + 2]
        if i == base:
            c12, c13 = [base + 1, base + 2], []
        else:
            c12, c13 = [base], [j for j in mol[1:] if j != i]
        maps.append([c12, c13, [], [], mol, [], [], []])
    return maps


def _multipole_params(polarization="mutual", n_mol=3, seed=1):
    """Synthetic multipoles exercising every axis type and the chirality flip."""
    rng = np.random.default_rng(seed)
    n = 3 * n_mol
    charge = np.tile([-0.5, 0.25, 0.25], n_mol)
    dipole = rng.normal(scale=4e-3, size=(n, 3))
    quad = rng.normal(scale=2e-4, size=(n, 3, 3))
    quad = 0.5 * (quad + np.transpose(quad, (0, 2, 1)))
    quad -= np.eye(3) * np.trace(quad, axis1=1, axis2=2)[:, None, None] / 3
    # (axis, z, x, y): Bisector, ZThenX(+chiral), ZBisect, ThreeFold, ZOnly, NoAxisType
    frames = [
        (1, 1, 2, -1),
        (0, 0, 2, 3),
        (0, 0, 1, -1),
        (2, 4, 5, 0),
        (0, 3, 5, 6),
        (3, 3, 4, 0),
        (4, 7, -1, -1),
        (0, 6, 8, -1),
        (5, -1, -1, -1),
    ][:n]
    axis, az, ax, ay = (np.array(v) for v in zip(*frames))
    for i in np.where(axis == 4)[0]:  # ZOnly sites must be cylindrically symmetric
        dipole[i, :2] = 0.0
        quad[i] = np.diag([-0.5, -0.5, 1.0]) * quad[i, 2, 2]
    dipole[axis == 5] = 0.0  # NoAxisType: lab-fixed moments would break rotation invariance
    quad[axis == 5] = 0.0
    m, p, d, u = scale_matrices(_covalent_maps(n_mol))
    polarity = np.tile([0.000837, 0.000496, 0.000496], n_mol)
    coeffs = np.array([-0.154, 0.017, 0.658, 0.474])
    return MultipoleParams(
        charge=charge,
        dipole=dipole,
        quadrupole=quad,
        axis_type=axis,
        atom_z=az,
        atom_x=ax,
        atom_y=ay,
        thole=np.full(n, 0.39),
        pdamp=polarity ** (1 / 6),
        polarity=polarity,
        m_scale=m,
        p_scale=p,
        d_scale=d,
        u_scale=u,
        polarization=polarization,
        extrapolation=np.array([coeffs[k:].sum() for k in range(4)]),
    )


def _vdw_params(n=9, lennard_jones=False, seed=2):
    rng = np.random.default_rng(seed)
    parent = np.array([3 * (i // 3) for i in range(n)])
    reduction = np.array([0.0 if i % 3 == 0 else 0.91 for i in range(n)])
    ii, jj = np.triu_indices(n, k=1)
    keep = (ii // 3) != (jj // 3)
    ii, jj = ii[keep], jj[keep]
    return VdwParams(
        parent=parent,
        reduction=reduction,
        pair_i=ii,
        pair_j=jj,
        sigma=0.3 + 0.08 * rng.random(ii.size),
        epsilon=0.2 + 0.3 * rng.random(ii.size),
        lennard_jones=lennard_jones,
    )


def _analytic_grid(n=25):
    """Grid of f = A cos(x) + B sin(y) + C cos(x - y) (degrees, kJ/mol) with exact derivatives."""
    a = np.linspace(-180.0, 180.0, n)
    X, Y = np.meshgrid(a, a, indexing="ij")
    r = np.pi / 180.0
    A, B, C = 3.0, -2.0, 1.5
    f = A * np.cos(r * X) + B * np.sin(r * Y) + C * np.cos(r * (X - Y))
    fx = r * (-A * np.sin(r * X) - C * np.sin(r * (X - Y)))
    fy = r * (B * np.cos(r * Y) + C * np.sin(r * (X - Y)))
    fxy = r * r * (-C * np.cos(r * (X - Y)))
    return np.stack([X, Y, f, fx, fy, fxy], axis=-1), (A, B, C)


def _chain(seed=4):
    rng = np.random.default_rng(seed)
    pos = np.cumsum(rng.normal(size=(7, 3)), axis=0) * 0.12
    return pos


def _fd_grad(fn, x, h=1e-6):
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        xp, xm = x.copy(), x.copy()
        xp[idx] += h
        xm[idx] -= h
        g[idx] = (float(fn(jnp.asarray(xp))) - float(fn(jnp.asarray(xm)))) / (2 * h)
    return g


def _random_rigid(seed=7):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q, rng.normal(size=3)


# ── synthetic tests ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("lj", [False, True])
def test_vdw_fd_gradient(lj):
    params = _vdw_params(lennard_jones=lj)
    pos = _water_cluster()
    fn = jax.jit(lambda x: vdw_energy(x, params))
    g = np.asarray(jax.grad(fn)(jnp.asarray(pos)))
    np.testing.assert_allclose(g, _fd_grad(fn, pos), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("mode", ["mutual", "direct", "extrapolated"])
def test_multipole_fd_gradient(mode):
    params = _multipole_params(mode)
    pos = _water_cluster()
    fn = jax.jit(lambda x: multipole_energy(x, params))
    g = np.asarray(jax.grad(fn)(jnp.asarray(pos)))
    np.testing.assert_allclose(g, _fd_grad(fn, pos), rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize("mode", ["mutual", "direct", "extrapolated"])
def test_multipole_rigid_invariance(mode):
    params = _multipole_params(mode)
    pos = _water_cluster()
    rot, shift = _random_rigid()
    fn = jax.jit(lambda x: multipole_energy(x, params))
    e0 = float(fn(jnp.asarray(pos)))
    e1 = float(fn(jnp.asarray(pos @ rot.T + shift)))
    assert abs(e1 - e0) < 1e-10 * max(1.0, abs(e0))


def test_lab_frame_moments_rotate_covariantly():
    params = _multipole_params()
    pos = _water_cluster()
    rot, shift = _random_rigid(3)
    lab0 = lab_frame_multipoles(jnp.asarray(pos), params)
    lab1 = lab_frame_multipoles(jnp.asarray(pos @ rot.T + shift), params)
    np.testing.assert_allclose(lab1.dipole, np.asarray(lab0.dipole) @ rot.T, atol=1e-14)
    q_rot = np.einsum("ab,nbc,dc->nad", rot, np.asarray(lab0.quadrupole), rot)
    np.testing.assert_allclose(lab1.quadrupole, q_rot, atol=1e-14)


def test_chirality_flip_under_reflection():
    """A ZThenX site with a y-atom keeps its frame-y moments mirror-consistent."""
    params = _multipole_params()
    pos = _water_cluster()
    mirror = pos * np.array([-1.0, 1.0, 1.0])
    lab0 = np.asarray(lab_frame_multipoles(jnp.asarray(pos), params).dipole)
    lab1 = np.asarray(lab_frame_multipoles(jnp.asarray(mirror), params).dipole)
    # site 1 is ZThenX with a chirality atom: its lab dipole is the mirror image
    np.testing.assert_allclose(lab1[1], lab0[1] * np.array([-1.0, 1.0, 1.0]), atol=1e-14)


def test_vdw_rigid_invariance():
    params = _vdw_params()
    pos = _water_cluster()
    rot, shift = _random_rigid()
    e0 = float(vdw_energy(jnp.asarray(pos), params))
    e1 = float(vdw_energy(jnp.asarray(pos @ rot.T + shift), params))
    assert abs(e1 - e0) < 1e-12 * max(1.0, abs(e0))


def test_mutual_vs_extrapolated_close():
    pos = jnp.asarray(_water_cluster())
    _, pol_mut, mu_mut = multipole_energy_components(pos, _multipole_params("mutual"))
    _, pol_opt, mu_opt = multipole_energy_components(pos, _multipole_params("extrapolated"))
    _, pol_dir, _ = multipole_energy_components(pos, _multipole_params("direct"))
    assert abs(float(pol_opt - pol_mut)) < 0.02 * abs(float(pol_mut))
    assert abs(float(pol_opt - pol_mut)) < abs(float(pol_dir - pol_mut))
    assert float(jnp.max(jnp.abs(mu_opt - mu_mut))) < 0.05 * float(jnp.max(jnp.abs(mu_mut)))


def test_mutual_dipoles_are_self_consistent():
    from mars.forcefields.amoeba.multipole import dipole_tensor, permanent_fields

    params = _multipole_params("mutual")
    pos = jnp.asarray(_water_cluster())
    e_d, _ = permanent_fields(pos, params)
    _, _, mu = multipole_energy_components(pos, params)
    T = dipole_tensor(pos, params)
    field = e_d + jnp.einsum("iajb,jb->ia", T, mu)
    np.testing.assert_allclose(mu, params.polarity[:, None] * field, atol=1e-14)


def test_multipole_hessian_symmetric():
    params = _multipole_params("mutual", n_mol=2)
    pos = jnp.asarray(_water_cluster(2))
    H = np.asarray(jax.jit(jax.hessian(lambda x: multipole_energy(x, params)))(pos)).reshape(18, 18)
    assert np.all(np.isfinite(H))
    np.testing.assert_allclose(H, H.T, atol=1e-8 * np.abs(H).max())


def test_scale_matrix_rules():
    # chain 0-1-2-3-4, with atoms 0..3 in one polarization group
    maps = [
        [[1], [2], [3], [4], [0, 1, 2, 3], [4], [], []],
        [[0, 2], [3], [4], [], [0, 1, 2, 3], [4], [], []],
        [[1, 3], [0, 4], [], [], [0, 1, 2, 3], [4], [], []],
        [[2, 4], [1], [0], [], [0, 1, 2, 3], [4], [], []],
        [[3], [2], [1], [0], [4], [0, 1, 2, 3], [], []],
    ]
    m, p, d, u = scale_matrices(maps)
    assert m[0, 1] == 0 and m[0, 2] == 0 and m[0, 3] == 0.4 and m[0, 4] == 0.8
    assert p[0, 3] == 0.5  # 1-4 inside the polarization group
    assert p[1, 4] == 1.0 and p[0, 4] == 1.0  # 1-4 across groups, 1-5
    assert d[0, 3] == 0.0 and d[0, 4] == 1.0
    assert np.all(u[~np.eye(5, dtype=bool)] == 1.0)
    for a in (m, p, d, u):
        np.testing.assert_array_equal(a, a.T)


def test_torsion_torsion_reproduces_grid_nodes():
    grid, (A, B, C) = _analytic_grid()
    params = torsion_torsion_params([[0, 1, 2, 3, 4]], [-1], [0], [grid])
    pos = _chain()
    from mars.forcefields.amoeba.torsion_torsion import _dihedral_deg

    p = jnp.asarray(pos)
    a1 = float(_dihedral_deg(p[None, 0], p[None, 1], p[None, 2], p[None, 3])[0])
    a2 = float(_dihedral_deg(p[None, 1], p[None, 2], p[None, 3], p[None, 4])[0])
    r = np.pi / 180
    exact = A * np.cos(r * a1) + B * np.sin(r * a2) + C * np.cos(r * (a1 - a2))
    e = float(torsion_torsion_energy(p, params))
    assert abs(e - exact) < 0.02  # bicubic interpolation error on a 15-degree grid


def test_torsion_torsion_fd_and_chirality():
    grid, _ = _analytic_grid()
    pos = _chain()
    plain = torsion_torsion_params([[0, 1, 2, 3, 4]], [-1], [0], [grid])
    chiral = torsion_torsion_params([[0, 1, 2, 3, 4]], [5], [0], [grid])
    for params in (plain, chiral):
        fn = jax.jit(lambda x, params=params: torsion_torsion_energy(x, params))
        g = np.asarray(jax.grad(fn)(jnp.asarray(pos)))
        np.testing.assert_allclose(g, _fd_grad(fn, pos), rtol=1e-6, atol=1e-6)
    # with a chirality atom the energy is invariant under reflection
    mirror = pos * np.array([1.0, -1.0, 1.0])
    e = float(torsion_torsion_energy(jnp.asarray(pos), chiral))
    e_m = float(torsion_torsion_energy(jnp.asarray(mirror), chiral))
    assert abs(e - e_m) < 1e-10
    e_plain_m = float(torsion_torsion_energy(jnp.asarray(mirror), plain))
    assert abs(e_plain_m - float(torsion_torsion_energy(jnp.asarray(pos), plain))) > 1e-6


# ── OpenMM parity ────────────────────────────────────────────────────────────

ALANINE_DIPEPTIDE_PDB = """\
REMARK  ACE
ATOM      1 1HH3 ACE     1       2.000   1.000  -0.000
ATOM      2  CH3 ACE     1       2.000   2.090   0.000
ATOM      3 2HH3 ACE     1       1.486   2.454   0.890
ATOM      4 3HH3 ACE     1       1.486   2.454  -0.890
ATOM      5  C   ACE     1       3.427   2.641  -0.000
ATOM      6  O   ACE     1       4.391   1.877  -0.000
ATOM      7  N   ALA     2       3.555   3.970  -0.000
ATOM      8  H   ALA     2       2.733   4.556  -0.000
ATOM      9  CA  ALA     2       4.853   4.614  -0.000
ATOM     10  HA  ALA     2       5.408   4.316   0.890
ATOM     11  CB  ALA     2       5.661   4.221  -1.232
ATOM     12 1HB  ALA     2       5.123   4.521  -2.131
ATOM     13 2HB  ALA     2       6.630   4.719  -1.206
ATOM     14 3HB  ALA     2       5.809   3.141  -1.241
ATOM     15  C   ALA     2       4.713   6.129   0.000
ATOM     16  O   ALA     2       3.601   6.653   0.000
ATOM     17  N   NME     3       5.846   6.835   0.000
ATOM     18  H   NME     3       6.737   6.359  -0.000
ATOM     19  CH3 NME     3       5.846   8.284   0.000
ATOM     20 1HH3 NME     3       4.819   8.648   0.000
ATOM     21 2HH3 NME     3       6.360   8.648   0.890
ATOM     22 3HH3 NME     3       6.360   8.648  -0.890
TER
END
"""


def _water_topology(n_mol):
    from openmm import app

    top = app.Topology()
    chain = top.addChain()
    for _ in range(n_mol):
        res = top.addResidue("HOH", chain)
        o = top.addAtom("O", app.element.oxygen, res)
        for name in ("H1", "H2"):
            top.addBond(o, top.addAtom(name, app.element.hydrogen, res))
    return top


def _openmm_reference(system, pos_nm):
    import openmm as mm
    from openmm import unit

    keys = {
        mm.AmoebaVdwForce: "amoeba_vdw",
        mm.AmoebaMultipoleForce: "amoeba_multipole",
        mm.AmoebaTorsionTorsionForce: "amoeba_torsion_torsion",
    }
    groups = {}
    for i, f in enumerate(system.getForces()):
        f.setForceGroup(i)
        for cls, key in keys.items():
            if isinstance(f, cls):
                groups[key] = i
                if cls is mm.AmoebaMultipoleForce:
                    f.setMutualInducedTargetEpsilon(1e-10)
                    f.setMutualInducedMaxIterations(500)
    ctx = mm.Context(system, mm.VerletIntegrator(1.0), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(pos_nm * unit.nanometer)
    out = {}
    for key, g in groups.items():
        st = ctx.getState(getEnergy=True, getForces=True, groups={g})
        out[key] = (
            st.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole),
            st.getForces(asNumpy=True).value_in_unit(unit.kilocalorie_per_mole / unit.angstrom),
        )
    return out


def _assert_parity(system, pos_nm):
    from mars.forcefields.amoeba import build_amoeba_terms

    ref = _openmm_reference(system, pos_nm)
    terms = build_amoeba_terms(system)
    assert set(terms) == set(ref)
    x = jnp.asarray(pos_nm * 10.0)
    for key, fn in terms.items():
        e, g = jax.jit(jax.value_and_grad(fn))(x)
        e_ref, f_ref = ref[key]
        assert abs(float(e) - e_ref) < 1e-6 * max(1.0, abs(e_ref)), key
        assert np.abs(-np.asarray(g) - np.asarray(f_ref)).max() < 1e-5, key


@pytest.mark.requires_openmm
@pytest.mark.parametrize("mode", ["mutual", "direct", "extrapolated"])
def test_openmm_parity_water_trimer(mode):
    pytest.importorskip("openmm")
    from openmm import app

    ff = app.ForceField("amoeba2018.xml")
    system = ff.createSystem(_water_topology(3), nonbondedMethod=app.NoCutoff, polarization=mode)
    pos = _water_cluster(3, seed=5)
    pos += np.random.default_rng(0).normal(scale=0.005, size=pos.shape)
    _assert_parity(system, pos)


@pytest.mark.requires_openmm
@pytest.mark.parametrize("mode", ["mutual", "extrapolated"])
def test_openmm_parity_alanine_dipeptide(mode, tmp_path):
    pytest.importorskip("openmm")
    from openmm import app

    path = tmp_path / "ala2.pdb"
    path.write_text(ALANINE_DIPEPTIDE_PDB)
    pdb = app.PDBFile(str(path))
    ff = app.ForceField("amoeba2018.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, polarization=mode)
    pos = np.asarray(pdb.getPositions(asNumpy=True)._value)
    pos += np.random.default_rng(1).normal(scale=0.005, size=pos.shape)
    _assert_parity(system, pos)


@pytest.mark.requires_openmm
def test_openmm_parity_torsion_torsion_chiral():
    """Hand-built AmoebaTorsionTorsionForce with a chirality atom vs OpenMM."""
    mm = pytest.importorskip("openmm")

    grid, _ = _analytic_grid()
    pos = _chain()
    system = mm.System()
    for _ in range(pos.shape[0]):
        system.addParticle(12.0)
    tt = mm.AmoebaTorsionTorsionForce()
    tt.setTorsionTorsionGrid(0, grid.tolist())
    tt.addTorsionTorsion(0, 1, 2, 3, 4, 5, 0)
    tt.addTorsionTorsion(1, 2, 3, 4, 5, -1, 0)
    system.addForce(tt)
    for p in (pos, pos * np.array([1.0, -1.0, 1.0])):
        _assert_parity(system, p)


def test_is_amoeba_force_and_unsupported():
    mm = pytest.importorskip("openmm")
    from mars.forcefields.amoeba import build_amoeba_terms, is_amoeba_force

    assert is_amoeba_force(mm.AmoebaVdwForce())
    assert is_amoeba_force(mm.AmoebaMultipoleForce())
    assert is_amoeba_force(mm.AmoebaTorsionTorsionForce())
    assert not is_amoeba_force(mm.HarmonicBondForce())
    system = mm.System()
    system.addParticle(1.0)
    system.addForce(mm.AmoebaGeneralizedKirkwoodForce())
    with pytest.raises(NotImplementedError):
        build_amoeba_terms(system)
    system = mm.System()
    system.addParticle(1.0)
    f = mm.AmoebaMultipoleForce()
    f.setNonbondedMethod(mm.AmoebaMultipoleForce.PME)
    f.addMultipole(0.0, [0, 0, 0], [0] * 9, 5, -1, -1, -1, 0.39, 0.0, 0.0)
    system.addForce(f)
    with pytest.raises(NotImplementedError):
        build_amoeba_terms(system)
