"""Tests for mars.forcefields (Lepton compiler, OpenMM conversion, MM potentials)."""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from mars.forcefields.lepton import (
    CompiledExpression,
    pointangle,
    pointdihedral,
    pointdistance,
    split_expression,
)

# ----------------------------------------------------------------------------
# Lepton expression compiler (no OpenMM needed)
# ----------------------------------------------------------------------------


class TestLepton:
    def test_definitions_and_power(self):
        expr = CompiledExpression("k*(d^2 + -25.5*d^3); d=r-r0")
        r = jnp.array([1.1, 0.9])
        out = expr({"k": 2.0, "r": r, "r0": 1.0})
        d = r - 1.0
        np.testing.assert_allclose(out, 2.0 * (d**2 - 25.5 * d**3))
        assert expr.variables == ["k", "r", "r0"]

    def test_nested_definitions_any_order(self):
        expr = CompiledExpression("a*b; b=c+1; a=2*c")
        assert float(expr({"c": 3.0})) == pytest.approx(6.0 * 4.0)

    def test_negative_base_integer_power_is_finite(self):
        expr = CompiledExpression("x^3")
        g = jax.grad(lambda x: expr({"x": x}))(-2.0)
        assert float(expr({"x": -2.0})) == pytest.approx(-8.0)
        assert float(g) == pytest.approx(12.0)

    @pytest.mark.parametrize(
        "text, env, expected",
        [
            ("step(x)", {"x": -1.0}, 0.0),
            ("step(x)", {"x": 0.0}, 1.0),
            ("select(x, 2, 3)", {"x": 0.0}, 3.0),
            ("min(a, b) + max(a, b)", {"a": 1.0, "b": 4.0}, 5.0),
            ("sqrt(square(x))", {"x": -3.0}, 3.0),
            ("2*pi", {}, 2 * np.pi),
            ("erfc(0)", {}, 1.0),
        ],
    )
    def test_builtins(self, text, env, expected):
        assert float(CompiledExpression(text)(env)) == pytest.approx(expected)

    def test_unbound_variable_raises(self):
        with pytest.raises(KeyError, match="Unbound variable"):
            CompiledExpression("x + y")({"x": 1.0})

    def test_bad_definition_raises(self):
        with pytest.raises(ValueError):
            split_expression("x; nonsense")

    def test_geometry_helpers(self):
        p = [jnp.array(v, dtype=float) for v in ([1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 1])]
        assert float(pointdistance(*p[0], *p[1])) == pytest.approx(1.0)
        assert float(pointangle(*p[0], *p[1], *p[2])) == pytest.approx(np.pi / 2)
        # IUPAC sign convention: this dihedral is -90 deg.
        assert float(pointdihedral(*p[0], *p[1], *p[2], *p[3])) == pytest.approx(-np.pi / 2)


# ----------------------------------------------------------------------------
# OpenMM conversion parity (needs OpenMM)
# ----------------------------------------------------------------------------

KCAL = 0.0433641043


def _reference(system, pos_ang):
    import openmm as mm
    from openmm import unit

    ctx = mm.Context(
        mm.XmlSerializer.clone(system),
        mm.VerletIntegrator(1.0),
        mm.Platform.getPlatformByName("Reference"),
    )
    ctx.setPositions(unit.Quantity(pos_ang * 0.1, unit.nanometer))
    st = ctx.getState(getEnergy=True, getForces=True)
    e = st.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    f = st.getForces(asNumpy=True).value_in_unit(unit.kilocalorie_per_mole / unit.angstrom)
    return e, np.asarray(f)


def _random_molecule_system(custom):
    """Five particles with one force (or several) and random positions."""
    import openmm as mm

    system = mm.System()
    for _ in range(5):
        system.addParticle(12.0)
    for f in custom:
        system.addForce(f)
    rng = np.random.default_rng(1)
    pos = rng.normal(scale=1.5, size=(5, 3))
    return system, pos


def _compare(system, pos, tol_e=1e-8, tol_f=1e-7):
    from mars.forcefields.convert import build_energy_terms

    terms_fn, _ = build_energy_terms(system, None, pos)
    energy = lambda x: sum(terms_fn(x).values())  # noqa: E731
    e = float(energy(jnp.asarray(pos)))
    f = -np.asarray(jax.grad(energy)(jnp.asarray(pos)))
    e_ref, f_ref = _reference(system, pos)
    assert e == pytest.approx(e_ref, rel=1e-7, abs=tol_e)
    np.testing.assert_allclose(f, f_ref, atol=tol_f + 1e-7 * np.abs(f_ref).max())


@pytest.mark.requires_openmm
class TestCustomForces:
    @pytest.fixture(autouse=True)
    def _openmm(self):
        pytest.importorskip("openmm")
        jax.config.update("jax_enable_x64", True)

    def test_custom_bond(self):
        import openmm as mm

        f = mm.CustomBondForce("k*(d^2 + -25.5*d^3 + 379.3125*d^4); d=r-r0")
        f.addPerBondParameter("r0")
        f.addPerBondParameter("k")
        f.addBond(0, 1, [0.15, 1000.0])
        f.addBond(2, 3, [0.12, 500.0])
        _compare(*_random_molecule_system([f]))

    def test_custom_angle_and_torsion(self):
        import openmm as mm

        a = mm.CustomAngleForce("0.5*k*(theta-theta0)^2")
        a.addPerAngleParameter("theta0")
        a.addPerAngleParameter("k")
        a.addAngle(0, 1, 2, [1.9, 300.0])
        t = mm.CustomTorsionForce("k*(1+cos(n*theta-theta0)); n=3")
        t.addPerTorsionParameter("k")
        t.addPerTorsionParameter("theta0")
        t.addTorsion(0, 1, 2, 3, [5.0, 0.3])
        _compare(*_random_molecule_system([a, t]))

    def test_custom_compound_bond_geometry(self):
        import openmm as mm

        f = mm.CustomCompoundBondForce(
            4, "k*(distance(p1,p2) + angle(p1,p2,p3) + sin(dihedral(p1,p2,p3,p4)) + x4^2)"
        )
        f.addGlobalParameter("k", 3.0)
        f.addBond([0, 1, 2, 3], [])
        _compare(*_random_molecule_system([f]))

    def test_custom_nonbonded_exclusions_and_tables(self):
        import openmm as mm

        f = mm.CustomNonbondedForce("a(t1,t2)/r^6 + q1*q2/r")
        f.addPerParticleParameter("q")
        f.addPerParticleParameter("t")
        table = [0.1, 0.2, 0.2, 0.3]
        f.addTabulatedFunction("a", mm.Discrete2DFunction(2, 2, table))
        for i in range(5):
            f.addParticle([0.1 * (i - 2), float(i % 2)])
        f.addExclusion(0, 1)
        _compare(*_random_molecule_system([f]))

    def test_rb_torsion(self):
        import openmm as mm

        f = mm.RBTorsionForce()
        f.addTorsion(0, 1, 2, 3, 1.0, -2.0, 0.5, 0.3, -0.1, 0.05)
        _compare(*_random_molecule_system([f]))

    def test_near_linear_harmonic_angle(self):
        import openmm as mm

        f = mm.HarmonicAngleForce()
        f.addAngle(0, 1, 2, 1.9, 400.0)
        system, pos = _random_molecule_system([f])
        pos[2] = pos[1] + 0.999 * (pos[1] - pos[0])  # ~179.x deg
        pos[2, 0] += 1e-3
        _compare(system, pos)

    def test_unsupported_force_raises(self):
        import openmm as mm

        from mars.forcefields.convert import build_energy_terms

        system, pos = _random_molecule_system([mm.GBSAOBCForce()])
        with pytest.raises(NotImplementedError, match="not supported"):
            build_energy_terms(system, None, pos)

    def test_cutoff_nonbonded_raises(self):
        import openmm as mm

        from mars.forcefields.convert import build_energy_terms

        nb = mm.NonbondedForce()
        nb.setNonbondedMethod(mm.NonbondedForce.CutoffNonPeriodic)
        for _ in range(5):
            nb.addParticle(0.0, 0.3, 0.1)
        system, pos = _random_molecule_system([nb])
        with pytest.raises(NotImplementedError, match="free space"):
            build_energy_terms(system, None, pos)


@pytest.fixture(scope="module")
def ala2(tmp_path_factory):
    pytest.importorskip("openmm")
    pytest.importorskip("jax_md.mm_forcefields")
    jax.config.update("jax_enable_x64", True)
    from openmm import app, unit

    from mars.forcefields import MMSystem

    # Alanine dipeptide built from OpenMM's bundled amber14 templates.
    path = tmp_path_factory.mktemp("ala2") / "ala2.pdb"
    path.write_text(_ALA2_PDB)
    pdb = app.PDBFile(str(path))
    ff = app.ForceField("amber14-all.xml")
    system = ff.createSystem(
        pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False
    )
    pos = np.asarray(pdb.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
    return MMSystem.from_openmm(system, pdb.topology, pos, family="amber")


@pytest.mark.requires_openmm
class TestMMPotential:
    def test_energy_and_forces_match_openmm(self, ala2):
        from mars.potentials import get_potential

        pot = get_potential("mm", system=ala2)
        pot.initialize(ala2.positions)
        f = pot.build_energy_fn()
        pos = jnp.asarray(ala2.positions)
        e = float(f(pos)) / KCAL
        forces = -np.asarray(jax.grad(f)(pos)) / KCAL
        e_ref, f_ref = _reference(ala2.openmm_system, ala2.positions)
        assert e == pytest.approx(e_ref, abs=1e-5)
        np.testing.assert_allclose(forces, f_ref, atol=1e-5)
        comps = pot.energy_components(ala2.positions)
        assert sum(comps.values()) == pytest.approx(float(f(pos)), rel=1e-10)

    def test_hessian_is_symmetric(self, ala2):
        from mars.potentials import get_potential

        pot = get_potential("mm", system=ala2)
        pot.initialize(ala2.positions)
        assert pot.analytical_hessian
        f = pot.build_energy_fn()
        h = np.asarray(
            jax.hessian(lambda x: f(x.reshape(-1, 3)))(jnp.asarray(ala2.positions).ravel())
        )
        np.testing.assert_allclose(h, h.T, atol=1e-8)

    def test_charges_and_null_neighbors(self, ala2):
        from mars.potentials import get_potential

        pot = get_potential("mm", system=ala2)
        pot.initialize(ala2.positions)
        _, charges_fn = pot.build_energy_fn(with_charges=True)
        q = np.asarray(charges_fn(ala2.positions))
        assert q.shape == (ala2.n_atoms,)
        assert abs(q.sum()) < 1e-6
        nb = pot.allocate_neighbors(ala2.positions)
        assert not bool(nb.did_buffer_overflow)
        assert pot.get_neighbor_kwargs(nb) == {}

    def test_atom_order_mismatch_raises(self, ala2):
        from mars.potentials import get_potential

        wrong = np.array(ala2.numbers)[::-1]
        with pytest.raises(ValueError, match="Atom order differs"):
            get_potential("mm", system=ala2, species=wrong)

    def test_missing_files_raise_helpful_errors(self, tmp_path):
        pytest.importorskip("openmm")
        from mars.forcefields import load_amber, load_charmm

        with pytest.raises(FileNotFoundError, match="prmtop"):
            load_amber(str(tmp_path / "missing.prmtop"))
        with pytest.raises(ValueError, match="CHARMM PSF"):
            load_charmm(None, [])


# Alanine dipeptide (ACE-ALA-NME), from OpenMM's test systems.
_ALA2_PDB = """\
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


@pytest.mark.requires_openmm
class TestInputValidation:
    """Wrong / incomplete parameters and bad starting structures are reported clearly."""

    def _pot(self, ala2, positions):
        from mars.potentials import get_potential

        pot = get_potential("mm", system=ala2, species=ala2.numbers)
        pot.initialize(positions)
        return pot

    def test_nm_coordinates(self, ala2):
        from mars.forcefields.checks import ForceFieldError

        with pytest.raises(ForceFieldError, match="look like nm"):
            self._pot(ala2, ala2.positions / 10.0)

    def test_wrong_atom_order(self, ala2):
        from mars.forcefields.checks import ForceFieldError

        pos = ala2.positions.copy()
        pos[[1, 8]] = pos[[8, 1]]
        with pytest.raises(ForceFieldError, match="stretched"):
            self._pot(ala2, pos)

    def test_overlapping_atoms(self, ala2):
        from mars.forcefields.checks import ForceFieldError

        pos = ala2.positions.copy()
        pos[20] = pos[3] + 0.1
        with pytest.raises(ForceFieldError, match="overlap"):
            self._pot(ala2, pos)

    def test_nan_and_missing_atoms(self, ala2):
        from mars.forcefields.checks import ForceFieldError

        pos = ala2.positions.copy()
        pos[0, 0] = np.nan
        with pytest.raises(ForceFieldError, match="NaN"):
            self._pot(ala2, pos)
        with pytest.raises(ForceFieldError, match="Missing hydrogens"):
            self._pot(ala2, ala2.positions[:-1])

    def test_spin_and_charge_checks(self, ala2):
        from mars.potentials import get_potential

        with pytest.raises(ValueError, match="closed-shell"):
            get_potential("mm", system=ala2, spin=2)
        pot = get_potential("mm", system=ala2, charge=1.0)
        with pytest.raises(ValueError, match="does not match the force-field"):
            pot.initialize(ala2.positions)

    def test_incomplete_parameters(self, ala2):
        import openmm as mm

        from mars.forcefields import MMSystem
        from mars.forcefields.checks import ForceFieldError

        system = mm.XmlSerializer.clone(ala2.openmm_system)
        for i in reversed(range(system.getNumForces())):
            if isinstance(system.getForce(i), mm.HarmonicBondForce):
                system.removeForce(i)
        with pytest.raises(ForceFieldError, match="no bond-stretching parameters"):
            MMSystem.from_openmm(system, ala2.topology, ala2.positions)

    def test_negative_force_constant(self, ala2):
        import openmm as mm

        from mars.forcefields import MMSystem
        from mars.forcefields.checks import ForceFieldError

        system = mm.XmlSerializer.clone(ala2.openmm_system)
        bf = next(f for f in system.getForces() if isinstance(f, mm.HarmonicBondForce))
        i, j, r0, k = bf.getBondParameters(0)
        bf.setBondParameters(0, i, j, r0, -k)
        with pytest.warns(UserWarning, match="negative force constants"):
            MMSystem.from_openmm(system, ala2.topology, ala2.positions)
        bf.setBondParameters(0, i, j, r0 * 0.0, k)
        with pytest.raises(ForceFieldError, match="Invalid force-field parameters"):
            MMSystem.from_openmm(system, ala2.topology, ala2.positions)

    def test_reader_errors_are_translated(self, tmp_path):
        pytest.importorskip("openmm")
        from mars.forcefields import load_amber, load_gromacs, load_openmm_xml
        from mars.forcefields.checks import ForceFieldError

        bad = tmp_path / "bad.prmtop"
        bad.write_text("hello\nworld\n")
        with pytest.raises(ForceFieldError, match="expected format"):
            load_amber(str(bad))
        top = tmp_path / "bad.top"
        top.write_text(
            "[ defaults ]\n1 3 yes 0.5 0.5\n[ moleculetype ]\nX 3\n[ atoms ]\n"
            "1 ZZ 1 X C1 1 0.0 12.0\n[ system ]\nx\n[ molecules ]\nX 1\n"
        )
        with pytest.raises(ForceFieldError, match="no parameters for type 'ZZ'"):
            load_gromacs(str(top))
        lig = tmp_path / "lig.pdb"
        lig.write_text(
            "ATOM      1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      2  O1  LIG A   1       1.200   0.000   0.000  1.00  0.00           O\nEND\n"
        )
        with pytest.raises(ForceFieldError, match="No template found"):
            load_openmm_xml(["amber14-all.xml"], str(lig))


@pytest.mark.requires_openmm
def test_empty_cmap_force_is_ignored(ala2):
    """OpenMM's CHARMM reader adds an empty CMAPTorsionForce for molecules without CMAP."""
    import openmm as mm

    from mars.forcefields import MMSystem

    system = mm.XmlSerializer.clone(ala2.openmm_system)
    system.addForce(mm.CMAPTorsionForce())
    msys = MMSystem.from_openmm(system, ala2.topology, ala2.positions)
    terms_fn, _ = msys.energy_terms()
    ref = MMSystem.from_openmm(ala2.openmm_system, ala2.topology, ala2.positions)
    e = sum(float(v) for v in terms_fn(jnp.asarray(ala2.positions)).values())
    e_ref = sum(float(v) for v in ref.energy_terms()[0](jnp.asarray(ala2.positions)).values())
    assert e == pytest.approx(e_ref, abs=1e-9)
