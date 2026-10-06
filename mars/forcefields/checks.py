"""
Input validation for classical force fields.

Three layers, each raising :class:`ForceFieldError` (a ``ValueError``) with a
message that says what is wrong and what to do about it:

* :func:`reading_errors` — wraps the OpenMM / ParmEd readers and turns their
  exceptions (``MissingParameter``, ``KeyError: 'CT'``, "No template found",
  ``IndexError`` on a corrupt file, ...) into actionable messages.
* :func:`check_parameters` — sanity checks on the built ``openmm.System``:
  non-finite values, negative force constants, bonds of the topology that no
  force covers (incomplete parameter files), zero masses, non-integer net
  charge.
* :func:`check_geometry` — the starting structure: shape, NaN, overlapping
  atoms, coordinates in nm instead of Å, bonds stretched far beyond their
  equilibrium length (wrong atom order / mismatched coordinates), and a
  non-finite or huge initial energy / force.

Problems that are suspicious but not fatal are reported with
``warnings.warn`` instead.
"""

import contextlib
import math
import warnings
from typing import Dict, Iterable, List, Optional

import numpy as np


class ForceFieldError(ValueError):
    """Invalid or incomplete force-field input."""


# ----------------------------------------------------------------------------
# Reader errors
# ----------------------------------------------------------------------------

_HINTS = (
    (
        ("No template found", "Multiple non-identical matching templates"),
        "The residue / atom names or protonation states in the PDB do not match the "
        "force-field templates. Add hydrogens with the same force field (e.g. "
        "openmm.app.Modeller.addHydrogens), use the template residue names, or load the "
        "missing ligand parameters (an extra XML file).",
    ),
    (
        ("Missing", "MissingParameter", "Could not find", "No parameters", "not found in"),
        "The parameter files are incomplete for this topology: add the parameter / stream "
        "files that define the atom types listed above (e.g. a ligand .str or GAFF frcmod-"
        "derived prmtop).",
    ),
    (
        ("Could not locate file",),
        "Give the full path of the force-field XML, or a name bundled with OpenMM "
        "(e.g. amber14-all.xml, charmm36.xml, amoeba2018.xml).",
    ),
    (
        ("Unrecognized PSF", "PSF", "prmtop", "FLAG", "%VERSION", "list index out of range"),
        "The file is not in the expected format (or is truncated). Check that you passed "
        "the right file to the right option (--prmtop / --psf / --top).",
    ),
)


def _hint(text: str) -> str:
    for keys, hint in _HINTS:
        if any(k.lower() in text.lower() for k in keys):
            return hint
    return ""


@contextlib.contextmanager
def reading_errors(family: str, files: Dict[str, object]):
    """Re-raise reader failures as :class:`ForceFieldError` with a hint."""
    try:
        yield
    except (ForceFieldError, FileNotFoundError, ImportError):
        raise
    except KeyError as exc:
        # GROMACS / ParmEd report an unknown atom or bond type as a bare KeyError.
        raise ForceFieldError(
            f"Could not build the {family} system from {_fmt(files)}: no parameters for type "
            f"{exc}. The parameter files are incomplete or the type name is misspelled; for "
            "GROMACS topologies also check --gmx-include."
        ) from exc
    except Exception as exc:  # noqa: BLE001 — OpenMM/ParmEd raise many unrelated types
        msg = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        hint = _hint(f"{type(exc).__name__} {msg}")
        raise ForceFieldError(
            f"Could not build the {family} system from {_fmt(files)}: "
            f"{type(exc).__name__}: {msg}" + (f"\n  Hint: {hint}" if hint else "")
        ) from exc


def _fmt(files: Dict[str, object]) -> str:
    parts = []
    for k, v in files.items():
        if v is None:
            continue
        parts.append(f"{k}={v}")
    return ", ".join(parts) or "the given files"


# ----------------------------------------------------------------------------
# Parameter checks
# ----------------------------------------------------------------------------


def _atom_label(topology, i: int) -> str:
    if topology is None:
        return f"atom {i + 1}"
    for atom in topology.atoms():
        if atom.index == i:
            return f"atom {i + 1} ({atom.residue.name}{atom.residue.id}:{atom.name})"
    return f"atom {i + 1}"


def _covered_bond_pairs(system) -> set:
    """Atom pairs that carry a bond-stretching term in some force."""
    import openmm

    pairs = set()
    for force in system.getForces():
        if isinstance(force, openmm.HarmonicBondForce):
            for b in range(force.getNumBonds()):
                i, j = force.getBondParameters(b)[:2]
                pairs.add((min(i, j), max(i, j)))
        elif isinstance(force, openmm.CustomBondForce):
            for b in range(force.getNumBonds()):
                i, j = force.getBondParameters(b)[:2]
                pairs.add((min(i, j), max(i, j)))
    return pairs


def _finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(v)) for v in values)


def check_parameters(system, topology=None) -> List[str]:
    """Validate an ``openmm.System`` built from the user's files.

    Raises:
        ForceFieldError: For problems that make the energy meaningless.

    Returns:
        The non-fatal warnings that were issued.
    """
    import openmm
    from openmm import unit

    md = unit.md_unit_system
    errors: List[str] = []
    notes: List[str] = []
    negative_k: List[str] = []
    n = system.getNumParticles()
    if n == 0:
        raise ForceFieldError("The force-field system contains no atoms.")

    zero_mass = [i for i in range(n) if system.getParticleMass(i).value_in_unit(unit.dalton) <= 0]
    if zero_mass:
        errors.append(
            f"{len(zero_mass)} atoms have zero mass (first: {_atom_label(topology, zero_mass[0])}); "
            "MARS cannot integrate massless particles."
        )

    for force in system.getForces():
        name = force.getName() or type(force).__name__
        if isinstance(force, openmm.HarmonicBondForce):
            for b in range(force.getNumBonds()):
                i, j, r0, k = force.getBondParameters(b)
                r0, k = r0.value_in_unit_system(md), k.value_in_unit_system(md)
                if _finite((r0, k)) and r0 > 0 and k < 0:
                    # Legitimate in a few force fields (AMOEBA water's
                    # Urey-Bradley term), so only report it.
                    negative_k.append(
                        f"{_atom_label(topology, i)}-{_atom_label(topology, j)} (k={k:g})"
                    )
                elif not _finite((r0, k)) or r0 <= 0:
                    errors.append(
                        f"{name}: bond {_atom_label(topology, i)}-{_atom_label(topology, j)} has "
                        f"r0={r0:g} nm, k={k:g} kJ/mol/nm^2."
                    )
        elif isinstance(force, openmm.HarmonicAngleForce):
            for a in range(force.getNumAngles()):
                i, j, k_, t0, k = force.getAngleParameters(a)
                t0, k = t0.value_in_unit_system(md), k.value_in_unit_system(md)
                if _finite((t0, k)) and k < 0:
                    negative_k.append(f"angle centred on {_atom_label(topology, j)} (k={k:g})")
                elif not _finite((t0, k)) or not (0.0 <= t0 <= math.pi + 1e-9):
                    errors.append(
                        f"{name}: angle centred on {_atom_label(topology, j)} has theta0={t0:g} rad, "
                        f"k={k:g}."
                    )
        elif isinstance(force, openmm.PeriodicTorsionForce):
            for t in range(force.getNumTorsions()):
                i, j, k_, l, per, phase, k = force.getTorsionParameters(t)
                phase, k = phase.value_in_unit_system(md), k.value_in_unit_system(md)
                if not _finite((phase, k)) or per < 0:
                    errors.append(
                        f"{name}: torsion {i + 1}-{j + 1}-{k_ + 1}-{l + 1} has periodicity {per}, "
                        f"k={k:g}."
                    )
        elif isinstance(force, openmm.NonbondedForce):
            total_q = 0.0
            for i in range(force.getNumParticles()):
                q, sig, eps = force.getParticleParameters(i)
                q = q.value_in_unit(unit.elementary_charge)
                sig, eps = sig.value_in_unit_system(md), eps.value_in_unit_system(md)
                total_q += q
                if not _finite((q, sig, eps)) or eps < 0 or (eps > 0 and sig <= 0):
                    errors.append(
                        f"{name}: {_atom_label(topology, i)} has q={q:g}, sigma={sig:g}, "
                        f"epsilon={eps:g}."
                    )
            if abs(total_q - round(total_q)) > 1e-3:
                notes.append(
                    f"The net charge of the force field is {total_q:.4f} e (not an integer): the "
                    "charges may be incomplete or mixed from different parameter sets."
                )
        elif hasattr(force, "getNumPerBondParameters") and hasattr(force, "getBondParameters"):
            for b in range(force.getNumBonds()):
                prm = force.getBondParameters(b)[-1]
                if not _finite(prm):
                    errors.append(f"{name}: term {b + 1} has non-finite parameters {list(prm)}.")
                    break
        if len(errors) >= 10:
            break

    if topology is not None:
        covered = _covered_bond_pairs(system)
        missing = []
        for bond in topology.bonds():
            i, j = bond[0].index, bond[1].index
            if (min(i, j), max(i, j)) not in covered:
                missing.append((i, j))
        if missing:
            i, j = missing[0]
            errors.append(
                f"{len(missing)} bonds of the topology have no bond-stretching parameters "
                f"(first: {_atom_label(topology, i)}-{_atom_label(topology, j)}). The parameter "
                "files are incomplete, or the system was built with constraints."
            )

    if negative_k:
        notes.append(
            f"{len(negative_k)} bond/angle terms have negative force constants (first: "
            f"{negative_k[0]}). This is intended in some force fields (e.g. AMOEBA water "
            "Urey-Bradley); otherwise check the parameter files."
        )
    if errors:
        more = "" if len(errors) < 10 else "\n  ... (further problems not listed)"
        raise ForceFieldError(
            "Invalid force-field parameters:\n  - " + "\n  - ".join(errors[:10]) + more
        )
    for note in notes:
        warnings.warn(note, UserWarning, stacklevel=3)
    return notes


# ----------------------------------------------------------------------------
# Geometry checks
# ----------------------------------------------------------------------------


def _topology_bonds(system, topology) -> np.ndarray:
    if topology is not None:
        b = [(x[0].index, x[1].index) for x in topology.bonds()]
        if b:
            return np.asarray(b, dtype=int)
    return np.asarray(sorted(_covered_bond_pairs(system)), dtype=int).reshape(-1, 2)


def _bond_r0(system) -> Dict[tuple, float]:
    import openmm
    from openmm import unit

    r0 = {}
    for force in system.getForces():
        if isinstance(force, openmm.HarmonicBondForce):
            for b in range(force.getNumBonds()):
                i, j, r, _ = force.getBondParameters(b)
                r0.setdefault((min(i, j), max(i, j)), r.value_in_unit(unit.angstrom))
    return r0


def check_geometry(
    system, topology, positions_ang, terms_fn=None, max_force: float = 1.0e4
) -> List[str]:
    """Validate the starting structure against the force-field topology.

    Args:
        system: The ``openmm.System``.
        topology: Its ``openmm.app.Topology`` (or ``None``).
        positions_ang: ``(n_atoms, 3)`` positions in Å.
        terms_fn: Optional ``positions -> {term: kcal/mol}`` to check that the
            initial energy and forces are finite.
        max_force: Largest initial force (kcal/mol/Å) accepted without a warning.

    Raises:
        ForceFieldError: For structures that cannot belong to this topology.

    Returns:
        The non-fatal warnings that were issued.
    """
    pos = np.asarray(positions_ang, dtype=float)
    n = system.getNumParticles()
    if pos.shape != (n, 3):
        raise ForceFieldError(
            f"The structure has shape {pos.shape}, but the force field has {n} atoms: expected "
            f"({n}, 3). Missing hydrogens or extra solvent/ions are the usual cause."
        )
    if not np.all(np.isfinite(pos)):
        raise ForceFieldError("The structure contains NaN or infinite coordinates.")

    bonds = _topology_bonds(system, topology)
    if len(bonds):
        d = np.linalg.norm(pos[bonds[:, 0]] - pos[bonds[:, 1]], axis=1)
        med = float(np.median(d))
        if med < 0.3:
            raise ForceFieldError(
                f"Bonded atoms are {med:.3f} apart on average: the coordinates look like nm, "
                "but MARS expects Angstrom. Multiply them by 10."
            )

    # Overlapping atoms (all pairs; fine for the system sizes MARS targets).
    if n <= 20000:
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        np.fill_diagonal(dist, np.inf)
        dmin = float(dist.min())
        if dmin < 0.5:
            i, j = np.unravel_index(int(np.argmin(dist)), dist.shape)
            raise ForceFieldError(
                f"Atoms {_atom_label(topology, int(i))} and {_atom_label(topology, int(j))} "
                f"overlap ({dmin:.3f} A apart). Fix the starting structure."
            )

    notes: List[str] = []
    if len(bonds):
        r0 = _bond_r0(system)
        ref = np.array([r0.get((min(i, j), max(i, j)), np.nan) for i, j in bonds])
        ratio = np.where(np.isfinite(ref) & (ref > 0), d / ref, d / 1.5)
        bad = np.nonzero(ratio > 2.0)[0]
        if bad.size:
            k = int(bad[np.argmax(ratio[bad])])
            i, j = bonds[k]
            raise ForceFieldError(
                f"{bad.size} bonds of the topology are stretched beyond twice their equilibrium "
                f"length (worst: {_atom_label(topology, i)}-{_atom_label(topology, j)} at "
                f"{d[k]:.2f} A). The structure's atom order or coordinates do not match the "
                "force-field topology (e.g. an XYZ with a different atom order than the PSF/"
                "prmtop); use --ff-coords with the coordinates that belong to the parameters."
            )

    if terms_fn is not None:
        import jax
        import jax.numpy as jnp

        x = jnp.asarray(pos)

        def total(p):
            return sum(jnp.asarray(v) for v in terms_fn(p).values())

        terms = {k: float(v) for k, v in terms_fn(x).items()}
        bad_terms = [k for k, v in terms.items() if not math.isfinite(v)]
        if bad_terms:
            raise ForceFieldError(
                f"The initial energy is not finite in term(s) {bad_terms}; check the structure "
                "and the parameters of those terms."
            )
        f = np.asarray(jax.grad(total)(x))
        if not np.all(np.isfinite(f)):
            raise ForceFieldError("The initial forces are not finite; check the structure.")
        fmax = float(np.abs(f).max())
        if fmax > max_force:
            i = int(np.argmax(np.abs(f).max(axis=1)))
            notes.append(
                f"Very large initial force ({fmax:.3g} kcal/mol/A on {_atom_label(topology, i)}): "
                "the structure is strained or clashing; minimise it before running MD."
            )
    for note in notes:
        warnings.warn(note, UserWarning, stacklevel=3)
    return notes


def validate(system, topology, positions_ang: Optional[np.ndarray] = None, terms_fn=None):
    """Run :func:`check_parameters` and, with positions, :func:`check_geometry`."""
    notes = check_parameters(system, topology)
    if positions_ang is not None:
        notes += check_geometry(system, topology, positions_ang, terms_fn)
    return notes
