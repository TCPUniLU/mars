"""
OpenMM ``Custom*Force`` objects as JAX energy terms.

Each converter reads one OpenMM force (parameters are copied to NumPy at
conversion time, so the returned closure holds no OpenMM objects) and
returns ``energy(positions_ang) -> kcal/mol``.  The expression itself is
evaluated by :class:`mars.forcefields.lepton.CompiledExpression` in OpenMM
units (nm, kJ/mol); the conversion happens at the boundary.

Supported: ``CustomBondForce``, ``CustomAngleForce``, ``CustomTorsionForce``,
``CustomCompoundBondForce``, ``CustomNonbondedForce`` (no cutoff, with
exclusions, interaction groups and discrete tabulated functions) and
``CustomExternalForce``.  Anything else raises ``NotImplementedError``
instead of being silently dropped.
"""

from typing import Callable, Dict, List

import jax.numpy as jnp
import numpy as np

from .lepton import CompiledExpression, _safe_norm, _vangle, _vdihedral

NM_PER_ANG = 0.1
KCAL_PER_KJ = 1.0 / 4.184


def _globals(force) -> Dict[str, float]:
    return {
        force.getGlobalParameterName(i): float(force.getGlobalParameterDefaultValue(i))
        for i in range(force.getNumGlobalParameters())
    }


def _tabulated_functions(force) -> Dict[str, Callable]:
    """Discrete tabulated functions as JAX lookups (continuous ones raise)."""
    import openmm

    out = {}
    if not hasattr(force, "getNumTabulatedFunctions"):
        return out
    for i in range(force.getNumTabulatedFunctions()):
        name = force.getTabulatedFunctionName(i)
        fn = force.getTabulatedFunction(i)
        if isinstance(fn, openmm.Discrete1DFunction):
            values = jnp.asarray(np.asarray(fn.getFunctionParameters(), dtype=float))
            out[name] = lambda x, v=values: v[jnp.round(x).astype(int)]
        elif isinstance(fn, openmm.Discrete2DFunction):
            nx, ny, values = fn.getFunctionParameters()
            table = jnp.asarray(np.asarray(values, dtype=float).reshape((int(ny), int(nx))))
            out[name] = lambda x, y, t=table: t[jnp.round(y).astype(int), jnp.round(x).astype(int)]
        elif isinstance(fn, openmm.Discrete3DFunction):
            nx, ny, nz, values = fn.getFunctionParameters()
            table = jnp.asarray(
                np.asarray(values, dtype=float).reshape((int(nz), int(ny), int(nx)))
            )
            out[name] = lambda x, y, z, t=table: t[
                jnp.round(z).astype(int), jnp.round(y).astype(int), jnp.round(x).astype(int)
            ]
        else:
            raise NotImplementedError(
                f"Tabulated function {name!r} of type {type(fn).__name__} is not supported yet "
                "(only Discrete1D/2D/3DFunction)."
            )
    return out


def _require_no_cutoff(force):
    import openmm

    method = force.getNonbondedMethod()
    if method != openmm.CustomNonbondedForce.NoCutoff:
        raise NotImplementedError(
            f"{force.getName()}: MARS evaluates force fields in free space; rebuild the "
            "OpenMM System with nonbondedMethod=app.NoCutoff."
        )


def _param_table(get_params, n, n_params):
    table = np.zeros((n, n_params), dtype=float)
    for i in range(n):
        table[i] = np.asarray(get_params(i), dtype=float)
    return table


def _energy_wrapper(compute: Callable) -> Callable:
    def energy(positions):
        return jnp.sum(compute(jnp.asarray(positions) * NM_PER_ANG)) * KCAL_PER_KJ

    return energy


# ----------------------------------------------------------------------------
# Bonded custom forces
# ----------------------------------------------------------------------------


def custom_bond(force) -> Callable:
    n = force.getNumBonds()
    names = [force.getPerBondParameterName(k) for k in range(force.getNumPerBondParameters())]
    idx = np.zeros((n, 2), dtype=np.int32)
    prm = np.zeros((n, len(names)))
    for b in range(n):
        i, j, p = force.getBondParameters(b)
        idx[b] = (i, j)
        prm[b] = p
    expr = CompiledExpression(force.getEnergyFunction(), _tabulated_functions(force))
    glob = _globals(force)
    idx, prm = jnp.asarray(idx), jnp.asarray(prm)

    def compute(x):
        env = dict(glob)
        env.update({nm: prm[:, k] for k, nm in enumerate(names)})
        env["r"] = _safe_norm(x[idx[:, 1]] - x[idx[:, 0]])
        return expr(env)

    return _energy_wrapper(compute)


def custom_angle(force) -> Callable:
    n = force.getNumAngles()
    names = [force.getPerAngleParameterName(k) for k in range(force.getNumPerAngleParameters())]
    idx = np.zeros((n, 3), dtype=np.int32)
    prm = np.zeros((n, len(names)))
    for a in range(n):
        i, j, k, p = force.getAngleParameters(a)
        idx[a] = (i, j, k)
        prm[a] = p
    expr = CompiledExpression(force.getEnergyFunction(), _tabulated_functions(force))
    glob = _globals(force)
    idx, prm = jnp.asarray(idx), jnp.asarray(prm)

    def compute(x):
        env = dict(glob)
        env.update({nm: prm[:, k] for k, nm in enumerate(names)})
        xj = x[idx[:, 1]]
        env["theta"] = _vangle(x[idx[:, 0]] - xj, x[idx[:, 2]] - xj)
        return expr(env)

    return _energy_wrapper(compute)


def custom_torsion(force) -> Callable:
    n = force.getNumTorsions()
    names = [force.getPerTorsionParameterName(k) for k in range(force.getNumPerTorsionParameters())]
    idx = np.zeros((n, 4), dtype=np.int32)
    prm = np.zeros((n, len(names)))
    for t in range(n):
        i, j, k, l, p = force.getTorsionParameters(t)
        idx[t] = (i, j, k, l)
        prm[t] = p
    expr = CompiledExpression(force.getEnergyFunction(), _tabulated_functions(force))
    glob = _globals(force)
    idx, prm = jnp.asarray(idx), jnp.asarray(prm)

    def compute(x):
        env = dict(glob)
        env.update({nm: prm[:, k] for k, nm in enumerate(names)})
        p1, p2, p3, p4 = (x[idx[:, c]] for c in range(4))
        env["theta"] = _vdihedral(p2 - p1, p3 - p2, p4 - p3)
        return expr(env)

    return _energy_wrapper(compute)


def _geometry_functions():
    return {
        "distance": lambda a, b: _safe_norm(a - b),
        "angle": lambda a, b, c: _vangle(a - b, c - b),
        "dihedral": lambda a, b, c, d: _vdihedral(b - a, c - b, d - c),
    }


def custom_compound_bond(force) -> Callable:
    n = force.getNumBonds()
    n_part = force.getNumParticlesPerBond()
    names = [force.getPerBondParameterName(k) for k in range(force.getNumPerBondParameters())]
    idx = np.zeros((n, n_part), dtype=np.int32)
    prm = np.zeros((n, len(names)))
    for b in range(n):
        particles, p = force.getBondParameters(b)
        idx[b] = particles
        prm[b] = p
    funcs = _geometry_functions()
    funcs.update(_tabulated_functions(force))
    expr = CompiledExpression(force.getEnergyFunction(), funcs)
    glob = _globals(force)
    idx, prm = jnp.asarray(idx), jnp.asarray(prm)

    def compute(x):
        env = dict(glob)
        env.update({nm: prm[:, k] for k, nm in enumerate(names)})
        for c in range(n_part):
            pc = x[idx[:, c]]
            env[f"p{c + 1}"] = pc
            env[f"x{c + 1}"], env[f"y{c + 1}"], env[f"z{c + 1}"] = pc[:, 0], pc[:, 1], pc[:, 2]
        return expr(env)

    return _energy_wrapper(compute)


def custom_external(force) -> Callable:
    n = force.getNumParticles()
    names = [
        force.getPerParticleParameterName(k) for k in range(force.getNumPerParticleParameters())
    ]
    idx = np.zeros((n,), dtype=np.int32)
    prm = np.zeros((n, len(names)))
    for a in range(n):
        i, p = force.getParticleParameters(a)
        idx[a] = i
        prm[a] = p
    expr = CompiledExpression(force.getEnergyFunction(), _tabulated_functions(force))
    glob = _globals(force)
    idx, prm = jnp.asarray(idx), jnp.asarray(prm)

    def compute(x):
        env = dict(glob)
        env.update({nm: prm[:, k] for k, nm in enumerate(names)})
        xi = x[idx]
        env["x"], env["y"], env["z"] = xi[:, 0], xi[:, 1], xi[:, 2]
        return expr(env)

    return _energy_wrapper(compute)


# ----------------------------------------------------------------------------
# Custom nonbonded (all pairs, free space)
# ----------------------------------------------------------------------------


def _nonbonded_pairs(force, n_atoms) -> np.ndarray:
    """All interacting pairs i<j after exclusions and interaction groups."""
    excluded = set()
    for e in range(force.getNumExclusions()):
        i, j = force.getExclusionParticles(e)
        excluded.add((min(i, j), max(i, j)))
    if force.getNumInteractionGroups() > 0:
        pairs = set()
        for g in range(force.getNumInteractionGroups()):
            set1, set2 = force.getInteractionGroupParameters(g)
            for i in set1:
                for j in set2:
                    if i != j:
                        pairs.add((min(i, j), max(i, j)))
        pairs = sorted(pairs - excluded)
        return np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
    iu, ju = np.triu_indices(n_atoms, k=1)
    keep = np.ones(iu.shape[0], dtype=bool)
    if excluded:
        ex = np.asarray(sorted(excluded), dtype=np.int64)
        code = iu.astype(np.int64) * n_atoms + ju
        keep = ~np.isin(code, ex[:, 0] * n_atoms + ex[:, 1])
    return np.stack([iu[keep], ju[keep]], axis=1).astype(np.int32)


def custom_nonbonded(force) -> Callable:
    _require_no_cutoff(force)
    if getattr(force, "getNumComputedValues", lambda: 0)() > 0:
        raise NotImplementedError(
            f"{force.getName()}: CustomNonbondedForce computed values are not supported yet."
        )
    # A long-range correction flag is ignored by OpenMM itself without a cutoff.
    n = force.getNumParticles()
    names = [
        force.getPerParticleParameterName(k) for k in range(force.getNumPerParticleParameters())
    ]
    prm = jnp.asarray(_param_table(force.getParticleParameters, n, len(names)))
    pairs = jnp.asarray(_nonbonded_pairs(force, n))
    expr = CompiledExpression(force.getEnergyFunction(), _tabulated_functions(force))
    glob = _globals(force)

    def compute(x):
        i, j = pairs[:, 0], pairs[:, 1]
        env = dict(glob)
        for k, nm in enumerate(names):
            env[f"{nm}1"] = prm[i, k]
            env[f"{nm}2"] = prm[j, k]
        env["r"] = _safe_norm(x[j] - x[i])
        return expr(env)

    return _energy_wrapper(compute)


_CONVERTERS = {
    "CustomBondForce": custom_bond,
    "CustomAngleForce": custom_angle,
    "CustomTorsionForce": custom_torsion,
    "CustomCompoundBondForce": custom_compound_bond,
    "CustomNonbondedForce": custom_nonbonded,
    "CustomExternalForce": custom_external,
}


def is_custom_force(force) -> bool:
    return type(force).__name__ in _CONVERTERS


def custom_force_energy(force) -> Callable:
    """Return ``energy(positions_ang) -> kcal/mol`` for an OpenMM Custom*Force."""
    return _CONVERTERS[type(force).__name__](force)


# ----------------------------------------------------------------------------
# Ryckaert-Bellemans torsions (OPLS via GROMACS topologies)
# ----------------------------------------------------------------------------


def rb_torsion(force) -> Callable:
    """``RBTorsionForce``: E = sum_n C_n cos(psi)^n, psi = phi - 180 deg."""
    n = force.getNumTorsions()
    idx = np.zeros((n, 4), dtype=np.int32)
    c = np.zeros((n, 6))
    for t in range(n):
        i, j, k, l, *cs = force.getTorsionParameters(t)
        idx[t] = (i, j, k, l)
        c[t] = [float(v.value_in_unit_system(_md_unit_system())) for v in cs]
    idx, c = jnp.asarray(idx), jnp.asarray(c)

    def compute(x):
        p1, p2, p3, p4 = (x[idx[:, k]] for k in range(4))
        cpsi = -jnp.cos(_vdihedral(p2 - p1, p3 - p2, p4 - p3))
        powers = jnp.stack([cpsi**k for k in range(6)], axis=-1)
        return jnp.sum(c * powers, axis=-1)

    return _energy_wrapper(compute)


def harmonic_angle(force) -> Callable:
    """``HarmonicAngleForce``: E = k/2 (theta - theta0)^2.

    Evaluated here rather than by jax-md: ``jax_md.mm_forcefields.base.
    compute_angle`` adds 1e-7 to each vector component before taking the norm,
    which biases near-linear angles (errors ~1e-3 kcal/mol/A at 175 deg).
    """
    n = force.getNumAngles()
    idx = np.zeros((n, 3), dtype=np.int32)
    prm = np.zeros((n, 2))
    for a in range(n):
        i, j, k, theta0, kk = force.getAngleParameters(a)
        idx[a] = (i, j, k)
        prm[a] = (
            float(theta0.value_in_unit_system(_md_unit_system())),
            float(kk.value_in_unit_system(_md_unit_system())),
        )
    idx, prm = jnp.asarray(idx), jnp.asarray(prm)

    def compute(x):
        xj = x[idx[:, 1]]
        theta = _vangle(x[idx[:, 0]] - xj, x[idx[:, 2]] - xj)
        return 0.5 * prm[:, 1] * (theta - prm[:, 0]) ** 2

    return _energy_wrapper(compute)


def _md_unit_system():
    from openmm import unit

    return unit.md_unit_system


def list_supported() -> List[str]:
    return sorted([*_CONVERTERS, "RBTorsionForce"])
