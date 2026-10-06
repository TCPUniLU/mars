"""
Readers for classical force-field inputs.

MARS ships no force-field parameters: every reader takes the user's own
parameter / topology files and builds an OpenMM ``System`` from them (OpenMM
and ParmEd do the parsing), which :mod:`mars.forcefields.convert` then turns
into JAX energy terms.  All systems are built for free space
(``NoCutoff``), without constraints, so they match how MARS integrates.

==========  ==========================================================
Family      Inputs
==========  ==========================================================
AMBER/GAFF  ``prmtop`` (+ ``inpcrd``/``rst7`` or any coordinate file)
CHARMM      ``psf`` + parameter files (``.prm/.par/.rtf/.top/.str``)
            (+ ``crd``/``pdb``)
OPLS-AA     GROMACS ``top`` (+ ``gro``), or OpenMM ``ForceField`` XML +
            PDB (e.g. an OPLS XML produced by LigParGen / foyer)
AMOEBA      OpenMM ``ForceField`` XML (default ``amoeba2018.xml``) + PDB,
            or Tinker ``xyz`` + ``key``/``prm`` files
generic     serialized OpenMM ``System`` XML + PDB
==========  ==========================================================
"""

import os
from typing import Optional, Sequence

import numpy as np

from .checks import reading_errors
from .system import MMSystem

_CREATE_KW = dict(constraints=None, rigidWater=False, removeCMMotion=False)


def _app():
    try:
        import openmm  # noqa: F401
        from openmm import app, unit
    except ImportError as exc:
        raise ImportError(
            "Reading force-field files needs OpenMM.\n"
            "Install with: pip install 'mars[mm]'   (openmm, parmed)\n"
            f"(import error: {exc})"
        )
    return app, unit


def _need(path, what):
    if path is None:
        raise ValueError(f"Missing input: {what}.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{what} not found: {path}")
    return str(path)


def _read_coordinates(path):
    """Positions (OpenMM Quantity) from an Amber/CHARMM/PDB/GRO/XYZ file."""
    app, unit = _app()
    ext = os.path.splitext(path)[1].lower()
    if ext in (".inpcrd", ".rst7", ".ncrst"):
        return app.AmberInpcrdFile(path).getPositions(asNumpy=True)
    if ext == ".crd":
        # CHARMM card format (Amber uses .inpcrd/.rst7).
        return app.CharmmCrdFile(path).positions
    if ext == ".rst":
        return app.CharmmRstFile(path).positions
    if ext == ".pdb":
        return app.PDBFile(path).getPositions(asNumpy=True)
    if ext in (".cif", ".pdbx"):
        return app.PDBxFile(path).getPositions(asNumpy=True)
    if ext == ".gro":
        return app.GromacsGroFile(path).getPositions(asNumpy=True)
    if ext == ".xyz":
        from ..utils import load_structure

        return unit.Quantity(np.asarray(load_structure(path)["positions"]) * 0.1, unit.nanometer)
    raise ValueError(f"Unsupported coordinate file type: {path}")


def _finish(family, system, topology, positions, source, polarization=None):
    pos = None
    if positions is not None:
        from openmm import unit

        pos = np.asarray(positions.value_in_unit(unit.angstrom), dtype=float)
    return MMSystem.from_openmm(
        system, topology, pos, family=family, source=source, polarization=polarization
    )


def load_amber(prmtop: str, coordinates: Optional[str] = None) -> MMSystem:
    """AMBER / GAFF system from a ``prmtop`` (and optional coordinates).

    Args:
        prmtop: Amber parameter-topology file (from tleap / antechamber).
        coordinates: ``inpcrd``/``rst7`` (or PDB/XYZ) file.  Optional when
            the geometry comes from the MARS input structure instead.
    """
    app, _ = _app()
    prmtop = _need(prmtop, "AMBER prmtop file (--prmtop)")
    with reading_errors("AMBER", {"prmtop": prmtop, "coordinates": coordinates}):
        top = app.AmberPrmtopFile(prmtop)
        system = top.createSystem(nonbondedMethod=app.NoCutoff, **_CREATE_KW)
        pos = _read_coordinates(_need(coordinates, "coordinates")) if coordinates else None
    return _finish("amber", system, top.topology, pos, {"prmtop": prmtop})


def load_charmm(psf: str, params: Sequence[str], coordinates: Optional[str] = None) -> MMSystem:
    """CHARMM system from a PSF and CHARMM parameter / topology / stream files.

    Urey-Bradley, harmonic impropers, CMAP and NBFIX are all supported.

    Args:
        psf: CHARMM/X-PLOR PSF file.
        params: Parameter files (``.prm``, ``.par``, ``.rtf``, ``.top``,
            ``.str``), e.g. ``par_all36m_prot.prm`` + ``top_all36_prot.rtf``.
        coordinates: ``crd`` or PDB file.  Optional (see :func:`load_amber`).
    """
    app, _ = _app()
    psf = _need(psf, "CHARMM PSF file (--psf)")
    if not params:
        raise ValueError(
            "Missing input: CHARMM parameter files (--ff-params par_*.prm top_*.rtf [*.str])."
        )
    params = [_need(p, "CHARMM parameter file") for p in params]
    with reading_errors("CHARMM", {"psf": psf, "params": params, "coordinates": coordinates}):
        psf_file = app.CharmmPsfFile(psf)
        pset = app.CharmmParameterSet(*params)
        system = psf_file.createSystem(pset, nonbondedMethod=app.NoCutoff, **_CREATE_KW)
        pos = _read_coordinates(_need(coordinates, "coordinates")) if coordinates else None
    return _finish("charmm", system, psf_file.topology, pos, {"psf": psf, "params": params})


def load_gromacs(
    top: str,
    coordinates: Optional[str] = None,
    include_dir: Optional[str] = None,
    family: str = "opls",
) -> MMSystem:
    """System from a GROMACS ``.top`` (OPLS-AA, or any GROMACS force field).

    Ryckaert-Bellemans torsions and combination rule 3 (geometric mixing,
    as OPLS uses) are supported.

    Args:
        top: GROMACS topology.  ``#include``-d force-field files are looked up
            in ``include_dir`` (default: ``$GMXDATA/top`` if set).
        coordinates: ``gro`` or PDB file.
        include_dir: Directory with the GROMACS force-field folders.
        family: Label stored on the result (default ``'opls'``).
    """
    app, _ = _app()
    top = _need(top, "GROMACS topology (--top)")
    if include_dir is None and os.environ.get("GMXDATA"):
        include_dir = os.path.join(os.environ["GMXDATA"], "top")
    kw = {"includeDir": include_dir} if include_dir else {}
    with reading_errors("GROMACS", {"top": top, "include_dir": include_dir}):
        gtop = app.GromacsTopFile(top, **kw)
        system = gtop.createSystem(nonbondedMethod=app.NoCutoff, **_CREATE_KW)
        pos = _read_coordinates(_need(coordinates, "coordinates")) if coordinates else None
    return _finish(family, system, gtop.topology, pos, {"top": top})


def load_openmm_xml(
    xml: Sequence[str],
    pdb: str,
    family: str = "openmm",
    polarization: Optional[str] = None,
    **create_kwargs,
) -> MMSystem:
    """System from OpenMM ``ForceField`` XML file(s) and a PDB with residues.

    This is the route for OPLS / AMBER / CHARMM force fields distributed as
    OpenMM XML (bundled with OpenMM, openmmforcefields, LigParGen, foyer...)
    and for AMOEBA.

    Args:
        xml: One or more ForceField XML files (OpenMM resolves bundled names
            such as ``'amber14-all.xml'`` or ``'amoeba2018.xml'``).
        pdb: PDB file whose residue / atom names match the XML templates.
        family: Label stored on the result.
        polarization: AMOEBA only: ``'mutual'``, ``'direct'`` or
            ``'extrapolated'``.
        **create_kwargs: Extra ``ForceField.createSystem`` options.
    """
    app, _ = _app()
    if isinstance(xml, str):
        xml = [xml]
    if not xml:
        raise ValueError("Missing input: OpenMM force-field XML file(s) (--ff-xml).")
    pdb = _need(pdb, "PDB file (--pdb)")
    with reading_errors(family, {"xml": list(xml), "pdb": pdb}):
        pdb_file = app.PDBFile(pdb)
        ff = app.ForceField(*xml)
        kw = dict(_CREATE_KW)
        kw.update(create_kwargs)
        if polarization is not None:
            kw["polarization"] = polarization
            kw.setdefault("mutualInducedTargetEpsilon", 1e-8)
        system = ff.createSystem(pdb_file.topology, nonbondedMethod=app.NoCutoff, **kw)
    return _finish(
        family,
        system,
        pdb_file.topology,
        pdb_file.getPositions(asNumpy=True),
        {"xml": list(xml), "pdb": pdb},
        polarization=polarization,
    )


def load_amoeba(
    pdb: Optional[str] = None,
    xml: Sequence[str] = ("amoeba2018.xml",),
    polarization: str = "mutual",
    tinker_xyz: Optional[str] = None,
    tinker_key: Optional[Sequence[str]] = None,
) -> MMSystem:
    """AMOEBA system from OpenMM XML + PDB, or from Tinker files.

    Args:
        pdb: PDB file (OpenMM XML route).
        xml: AMOEBA ForceField XML (default ``amoeba2018.xml``, bundled with
            OpenMM).
        polarization: ``'mutual'`` (default), ``'direct'`` or
            ``'extrapolated'`` (OPT, Simmonett et al. 2015).
        tinker_xyz: Tinker ``.xyz`` file (Tinker route, OpenMM >= 8.3).
        tinker_key: Tinker ``.key`` / ``.prm`` file(s).
    """
    polarization = str(polarization).lower()
    if polarization not in ("mutual", "direct", "extrapolated"):
        raise ValueError("polarization must be 'mutual', 'direct' or 'extrapolated'.")
    if tinker_xyz is not None:
        app, _ = _app()
        if not hasattr(app, "TinkerFiles"):
            raise ImportError("Reading Tinker files needs OpenMM >= 8.3 (openmm.app.TinkerFiles).")
        tinker_xyz = _need(tinker_xyz, "Tinker xyz file")
        keys = [_need(k, "Tinker key/prm file") for k in (tinker_key or [])]
        if not keys:
            raise ValueError("Missing input: Tinker .key/.prm file(s) (--tinker-key).")
        with reading_errors("Tinker", {"xyz": tinker_xyz, "key": keys}):
            tf = app.TinkerFiles(tinker_xyz, keys)
            system = tf.createSystem(
                nonbondedMethod=app.NoCutoff,
                polarization=polarization,
                mutualInducedTargetEpsilon=1e-8,
            )
        return _finish(
            "amoeba",
            system,
            tf.topology,
            tf.getPositions(asNumpy=True),
            {"tinker_xyz": tinker_xyz, "tinker_key": keys},
            polarization=polarization,
        )
    return load_openmm_xml(xml, pdb, family="amoeba", polarization=polarization)


def load_openmm_system(system_xml: str, pdb: str, family: str = "openmm") -> MMSystem:
    """System from a serialized OpenMM ``System`` XML and a PDB."""
    app, _ = _app()
    import openmm

    system_xml = _need(system_xml, "serialized OpenMM System XML")
    with reading_errors("OpenMM", {"system_xml": system_xml, "pdb": pdb}):
        with open(system_xml) as fh:
            system = openmm.XmlSerializer.deserialize(fh.read())
        pdb_file = app.PDBFile(_need(pdb, "PDB file (--pdb)"))
    return _finish(
        family,
        system,
        pdb_file.topology,
        pdb_file.getPositions(asNumpy=True),
        {"system_xml": system_xml, "pdb": pdb},
    )
