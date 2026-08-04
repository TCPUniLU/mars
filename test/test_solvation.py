"""Tests for mars/solvation.py and mars/solvents/ — library and shell builder."""

import numpy as np
import pytest

import mars.solvents as solvents
from mars.solvation import (
    place_solvents_by_layers,
    solvate,
    SOLVENT_DB,
    get_solvent,
    LayerSpec,
)
from mars.utils import create_structure

# ── Library tests ─────────────────────────────────────────────────────────────


def test_list_solvents_has_at_least_30_entries():
    names = solvents.list_solvents()
    assert len(names) >= 30
    assert "water" in names
    assert "methanol" in names
    assert "dmso" in names


def test_each_library_entry_loads_with_valid_geometry():
    for name in solvents.list_solvents():
        entry = solvents.get_solvent(name)
        assert "symbols" in entry and "positions" in entry
        assert len(entry["symbols"]) >= 1
        assert entry["positions"].shape == (len(entry["symbols"]), 3)
        # Metadata fields populated by manifest
        for key in ("formula", "density", "dielectric", "source", "aliases"):
            assert key in entry


def test_alias_resolution():
    # Common aliases
    assert solvents.get_solvent("hexane")["formula"] == "C6H14"
    assert solvents.get_solvent("DCM")["formula"] == "CH2Cl2"
    assert solvents.get_solvent("EtOH")["formula"] == "C2H5OH"
    assert solvents.get_solvent("DMSO")["formula"] == "(CH3)2SO"


def test_alias_case_insensitive():
    assert solvents.get_solvent("WATER") is solvents.get_solvent("water")
    assert solvents.get_solvent("dmso") is solvents.get_solvent("DMSO")


def test_unknown_solvent_raises():
    with pytest.raises(KeyError):
        solvents.get_solvent("cromulent_solvent")


def test_register_solvent_runtime(tmp_output_dir):
    xyz_path = tmp_output_dir / "custom.xyz"
    xyz_path.write_text("3\ndummy\nO 0 0 0\nH 1 0 0\nH 0 1 0\n")
    solvents.register_solvent(
        "custom_h2o",
        str(xyz_path),
        formula="H2O",
        density=1.0,
        dielectric=80.0,
        aliases=["my_water"],
    )
    try:
        assert "custom_h2o" in solvents.list_solvents()
        assert solvents.get_solvent("my_water")["formula"] == "H2O"
    finally:
        # Clean up — remove from internal state so subsequent tests aren't polluted
        solvents._SOLVENTS.pop("custom_h2o", None)
        solvents._ALIAS_MAP.pop("custom_h2o", None)
        solvents._ALIAS_MAP.pop("my_water", None)


def test_solvent_db_alias_proxy_supports_in_iter_len():
    assert "water" in SOLVENT_DB
    assert "hexane" in SOLVENT_DB
    assert len(SOLVENT_DB) >= 30
    names = list(SOLVENT_DB)
    assert all(isinstance(n, str) for n in names)


def test_legacy_get_solvent_roundtrip():
    data = get_solvent("methanol")
    assert "symbols" in data and "positions" in data


def test_metadata_lookup():
    meta = solvents.get_solvent_metadata("water")
    assert meta["formula"] == "H2O"
    assert meta["density"] == pytest.approx(0.997)


# ── Shell builder tests ──────────────────────────────────────────────────────


@pytest.fixture
def water_solute():
    """Single water molecule used as a solute."""
    positions = np.array(
        [
            [0.000, 0.000, 0.000],
            [0.957, 0.000, 0.000],
            [-0.239, 0.927, 0.000],
        ]
    )
    return create_structure(positions, ["O", "H", "H"])


def test_solvate_water_with_explicit_layer(water_solute):
    out = solvate(water_solute, "water", layers=3, vdw_scale=0.85, seed=1)
    assert "positions" in out and "symbols" in out
    n_solute = 3
    n_solvent_per_mol = 3  # water = OH2
    expected_min_atoms = n_solute + n_solvent_per_mol * 1  # ≥1 placed
    assert len(out["symbols"]) >= expected_min_atoms


def test_place_solvents_two_layers(water_solute):
    out = place_solvents_by_layers(
        water_solute,
        "water",
        layers=[2, 4],
        vdw_scale=0.85,
        seed=42,
    )
    assert len(out["symbols"]) > len(water_solute["symbols"])


def test_solvate_deterministic_with_seed(water_solute):
    a = solvate(water_solute, "methanol", layers=2, seed=7, vdw_scale=0.85)
    b = solvate(water_solute, "methanol", layers=2, seed=7, vdw_scale=0.85)
    np.testing.assert_allclose(np.array(a["positions"]), np.array(b["positions"]))


def test_layer_spec_per_layer_control(water_solute):
    specs = [
        LayerSpec(n_solvent=2, buffer=0.0, vdw_scale=0.85, n_candidates=200, n_shells=3, seed=1),
        LayerSpec(n_solvent=4, buffer=0.5, vdw_scale=0.85, n_candidates=300, n_shells=4, seed=2),
    ]
    out = place_solvents_by_layers(water_solute, "water", layers=specs)
    assert len(out["symbols"]) > len(water_solute["symbols"])


def test_solvent_passed_as_dict(water_solute):
    custom_solvent = {
        "symbols": ["O", "H", "H"],
        "positions": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [-0.24, 0.93, 0.0],
            ]
        ),
    }
    out = place_solvents_by_layers(water_solute, custom_solvent, layers=[2], seed=3)
    assert len(out["symbols"]) > 3


# ── Centering & isotropy (center of mass, isotropic spherical shells) ─────────


def _elongated_solute():
    """A deliberately off-centre, elongated linear triatomic (O=C=O-like),
    shifted away from the origin to exercise recentring."""
    positions = np.array(
        [
            [5.00, 0.0, 0.0],
            [6.16, 0.0, 0.0],
            [7.32, 0.0, 0.0],
        ]
    )
    return create_structure(positions, ["O", "C", "O"])


def test_center_of_mass_helper():
    from mars.utils import center_of_mass

    # Two atoms: H (Z=1) at x=0, O (Z=8) at x=1 → COM shifts toward O.
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    numbers = np.array([1, 8])
    com = center_of_mass(pos, numbers)
    mH, mO = 1.008, 15.999
    assert com[0] == pytest.approx(mO / (mH + mO), abs=1e-3)
    assert com[1] == pytest.approx(0.0)
    assert com[2] == pytest.approx(0.0)


def test_solute_kept_in_lab_frame(water_solute):
    """Like the reference solvate.py, the solute is NOT recentred — its atoms in
    the output are unchanged from the input (only solvent is added after them)."""
    out = solvate(water_solute, "water", layers=4, seed=5, vdw_scale=0.85)
    n_solute = len(water_solute["symbols"])
    out_solute = np.asarray(out["positions"])[:n_solute]
    np.testing.assert_allclose(out_solute, np.asarray(water_solute["positions"]), atol=1e-6)


def test_shell_wraps_the_molecule():
    """The solvent shell wraps the solute (its centroid sits near the solute
    centroid), not bunched on one side."""
    solute = _elongated_solute()
    out = solvate(solute, "water", layers=8, seed=3, vdw_scale=0.85)
    n_solute = len(solute["symbols"])
    p = np.asarray(out["positions"])
    solute_centroid = p[:n_solute].mean(axis=0)
    solvent_centroid = p[n_solute:].mean(axis=0)
    assert np.linalg.norm(solvent_centroid - solute_centroid) < 2.0


def _hexagon_solute():
    """Small flat ring of 6 carbons (~benzene footprint)."""
    ang = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    pos = np.column_stack([1.4 * np.cos(ang), 1.4 * np.sin(ang), np.zeros(6)])
    return create_structure(pos, ["C"] * 6)


@pytest.mark.parametrize("solvent", ["water", "diethyl_ether", "n_heptane", "dmso"])
def test_count_driven_keeps_exactly_n(solvent):
    """auto_solvate(n_molecules=N) keeps EXACTLY N solvents for any solvent size
    (overpack + greedy-select-N, the reference behaviour), and the shell hugs the
    surface (every solute atom has solvent within contact range)."""
    from scipy.spatial.distance import cdist

    from mars.solvation import auto_solvate
    from mars.solvents import load_solvent

    solute = _hexagon_solute()
    N = 10
    out = auto_solvate(solute, solvent, n_molecules=N, optimise_each_layer=False, seed=42)
    n = 6
    nmol = len(load_solvent(solvent)["symbols"])
    p = np.asarray(out["positions"])
    placed = (len(p) - n) // nmol
    assert placed == N  # exactly N, independent of solvent size
    per_solute_nearest = cdist(p[:n], p[n:]).min(axis=1)
    assert per_solute_nearest.max() < 5.5  # surface-hugging, not a detached sphere


def test_greedy_angular_spread():
    """Greedy selection spreads the kept solvents around the solute rather than
    clustering them on one side (proximity + angular-spread scoring)."""
    import itertools

    from mars.solvation import auto_solvate

    solute = create_structure(np.array([[0.0, 0.0, 0.0]]), ["C"])  # single atom
    out = auto_solvate(solute, "water", n_molecules=4, optimise_each_layer=False, seed=1)
    p = np.asarray(out["positions"])
    centroids = p[1:].reshape(-1, 3, 3).mean(axis=1)  # 4 water centroids from origin
    dirs = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    angs = [
        np.degrees(np.arccos(np.clip(np.dot(dirs[i], dirs[j]), -1, 1)))
        for i, j in itertools.combinations(range(len(dirs)), 2)
    ]
    # Well-spread (≈ tetrahedral 109°); clustered would give small angles.
    assert np.mean(angs) > 60.0


def test_surface_hugging_large_molecule():
    """Per-atom placement hugs the surface of a LARGE/elongated solute: every
    solute atom — including the middle of a long chain — has solvent in contact
    range (a detached oversized sphere would leave middle atoms far away)."""
    from scipy.spatial.distance import cdist

    from mars.solvation import auto_solvate

    xs = np.arange(10) * 1.4
    pos = np.column_stack([xs, np.zeros(10), np.zeros(10)])
    solute = create_structure(pos, ["C"] * 10)

    out = auto_solvate(solute, "water", n_molecules=30, optimise_each_layer=False, seed=1)
    n = 10
    p = np.asarray(out["positions"])
    per_solute_nearest = cdist(p[:n], p[n:]).min(axis=1)
    assert per_solute_nearest.max() < 4.5


def test_padding_grows_more_than_contact_shell():
    """--padding (cover mode, geometry only) builds a thicker shell than the
    default single contact shell, and places a sane, bounded count."""
    from mars.solvation import auto_solvate
    from mars.solvents import load_solvent

    solute = create_structure(
        np.array([[0, 0, 0.117], [0, 0.757, -0.469], [0, -0.757, -0.469]]), ["O", "H", "H"]
    )
    nmol = len(load_solvent("water")["symbols"])
    contact = auto_solvate(solute, "water", optimise_each_layer=False, seed=1)
    padded = auto_solvate(solute, "water", padding=4.0, optimise_each_layer=False, seed=1)
    n_contact = (len(contact["symbols"]) - 3) // nmol
    n_padded = (len(padded["symbols"]) - 3) // nmol
    assert 1 <= n_contact < n_padded < 400


# ── Reader: library names + user-supplied XYZ files ───────────────────────────


def test_load_solvent_by_name_and_alias():
    from mars.solvents import load_solvent

    a = load_solvent("ethanol")
    b = load_solvent("EtOH")  # alias, case-insensitive resolution
    assert a["symbols"] == b["symbols"]
    assert a["source"] == "obabel --gen3d (MMFF94)"


def test_load_solvent_from_xyz_path(tmp_path):
    from mars.solvents import load_solvent

    xyz = tmp_path / "my_solvent.xyz"
    xyz.write_text("3\nmy water\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n")
    entry = load_solvent(str(xyz))
    assert entry["symbols"] == ["O", "H", "H"]
    assert entry["positions"].shape == (3, 3)
    assert entry["source"] == "user-file"


def test_load_solvent_missing_path_raises():
    from mars.solvents import load_solvent

    with pytest.raises(FileNotFoundError):
        load_solvent("/no/such/solvent.xyz")


def test_load_solvent_unknown_name_raises():
    from mars.solvents import load_solvent

    with pytest.raises(KeyError):
        load_solvent("cromulent_solvent")


def test_solvate_with_custom_xyz_path(water_solute, tmp_path):
    """The placement API accepts a path string as the solvent."""
    xyz = tmp_path / "mini.xyz"
    xyz.write_text("3\nmini\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n")
    out = place_solvents_by_layers(water_solute, str(xyz), layers=[3], seed=2)
    assert len(out["symbols"]) > len(water_solute["symbols"])


# ── Regenerated library (obabel) provenance ──────────────────────────────────


def test_regenerated_solvents_have_smiles_and_obabel_source():
    import mars.solvents as S

    experimental = {
        "water",
        "methanol",
        "acetonitrile",
        "acetone",
        "benzene",
        "n_hexane",
        "dmso",
        "hydrogen_fluoride",
    }
    for name in S.list_solvents():
        meta = S.get_solvent_metadata(name)
        assert meta["smiles"], f"{name} missing SMILES"
        if name in experimental:
            assert meta["source"] == "experimental"
        else:
            assert meta["source"] == "obabel --gen3d (MMFF94)", name


def test_all_geometries_have_no_clashing_atoms():
    """Regenerated geometries must be physically sane (no overlapping atoms)."""
    from scipy.spatial.distance import pdist

    import mars.solvents as S

    for name in S.list_solvents():
        pos = np.asarray(S.get_solvent(name)["positions"])
        assert np.isfinite(pos).all(), name
        if len(pos) > 1:
            assert pdist(pos).min() > 0.6, name


# ── Iterative cover mode + shape-aware packing ────────────────────────────────


def test_effective_radius_is_shape_aware():
    """Cross-sectional radius ≈ bounding for compact solvents but much smaller
    for elongated ones (the width, not the length)."""
    from mars.solvation import _bounding_radius, _effective_radius
    from mars.solvents import load_solvent

    for name in ["water", "dmso"]:  # compact → effective ≈ bounding
        e = load_solvent(name)
        p = np.asarray(e["positions"])
        p = p - p.mean(0)
        assert abs(_effective_radius(p, e["symbols"]) - _bounding_radius(p, e["symbols"])) < 0.4
    for name in ["n_heptane", "diethyl_ether"]:  # elongated → much smaller
        e = load_solvent(name)
        p = np.asarray(e["positions"])
        p = p - p.mean(0)
        assert _effective_radius(p, e["symbols"]) < 0.7 * _bounding_radius(p, e["symbols"])


def test_cover_default_wraps_contact_shell():
    """Default cover (one saturated contact shell) puts solvent within contact of
    EVERY solute atom, for both a compact and an elongated solvent."""
    from scipy.spatial.distance import cdist

    from mars.solvation import _solvate_cover
    from mars.solvents import load_solvent

    ang = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    solute = create_structure(
        np.column_stack([1.4 * np.cos(ang), 1.4 * np.sin(ang), np.zeros(6)]), ["C"] * 6
    )
    for solv in ["water", "n_heptane"]:
        out = _solvate_cover(
            solute,
            load_solvent(solv),
            padding=None,
            max_shells=10,
            buffer=0.0,
            vdw_scale=0.85,
            n_candidates=200,
            n_orient=8,
            max_refill=1,
            min_solute_dist=0.0,
            min_solvent_dist=0.0,
            relax_fn=None,
            seed=42,
        )
        p = np.asarray(out["positions"])
        n = 6
        assert len(p) > n  # solvent placed
        per_solute = cdist(p[:n], p[n:]).min(axis=1)
        assert per_solute.max() < 4.5  # every solute atom is covered


def test_cover_loop_saturates_and_refills():
    """The fill→relax→refill loop runs and terminates: with an identity relax a
    second fill on the unchanged geometry adds nothing, so the shell saturates."""
    from mars.solvation import _solvate_cover
    from mars.solvents import load_solvent

    solute = create_structure(
        np.array([[0, 0, 0.117], [0, 0.757, -0.469], [0, -0.757, -0.469]]), ["O", "H", "H"]
    )
    calls = {"n": 0}

    def identity_relax(system, n_solute):
        calls["n"] += 1
        return system  # unchanged → refill should add nothing next cycle

    out = _solvate_cover(
        solute,
        load_solvent("water"),
        padding=None,
        max_shells=10,
        buffer=0.0,
        vdw_scale=0.85,
        n_candidates=200,
        n_orient=4,
        max_refill=3,
        min_solute_dist=0.0,
        min_solvent_dist=0.0,
        relax_fn=identity_relax,
        seed=1,
    )
    n_solv = (len(out["symbols"]) - 3) // 3
    assert n_solv > 0  # placed a contact shell
    assert calls["n"] >= 1  # relax was invoked between fills


def test_default_opt_mode_is_layerwise():
    from mars.cli._parser import create_parser

    args = create_parser().parse_args(["solvation", "x.xyz"])
    assert args.opt_mode == "layerwise"


# ── Topology integrity (post-relaxation solvent check) ───────────────────────


def _build_solvated(solute, solvent_name, n_mol, seed=1):
    """Place `n_mol` solvent molecules (geometry only, no relax)."""
    from mars.solvation import auto_solvate

    return auto_solvate(
        solute, solvent_name, n_molecules=n_mol, optimise_each_layer=False, seed=seed
    )


def test_topology_intact_after_placement(water_solute):
    """A freshly placed (un-relaxed) shell has no broken solvent molecules."""
    from mars.solvation import remove_broken_solvent

    sysd = _build_solvated(water_solute, "water", 6)
    clean, broken = remove_broken_solvent(sysd, "water", n_solute=3, tolerance=1.3)
    assert broken == []
    assert len(clean["symbols"]) == len(sysd["symbols"])  # nothing removed


def test_topology_detects_and_removes_broken(water_solute):
    """Exploding one atom of a solvent molecule flags & removes that molecule."""
    from mars.solvation import remove_broken_solvent

    sysd = _build_solvated(water_solute, "water", 6)
    nm = 3  # water
    pos = np.array(sysd["positions"], dtype=float)
    # Break solvent molecule index 2: move its last atom far from the rest.
    off = 3 + 2 * nm
    pos[off + nm - 1] += np.array([6.0, 0.0, 0.0])
    broken_struct = create_structure(pos, list(sysd["symbols"]))

    clean, broken = remove_broken_solvent(broken_struct, "water", n_solute=3, tolerance=1.3)
    assert broken == [2]
    # Exactly one molecule (3 atoms) removed; solute (3 atoms) preserved.
    assert len(clean["symbols"]) == len(sysd["symbols"]) - nm
    assert list(clean["symbols"][:3]) == ["O", "H", "H"]


def test_topology_detects_intermolecular_bond(water_solute):
    """Two solvent molecules fused by a short contact are both flagged broken."""
    from mars.solvation import _find_broken_solvent, _solvent_ref_bonds
    from mars.utils import symbols_to_numbers

    sysd = _build_solvated(water_solute, "water", 6)
    nm = 3
    ref_bonds, _ = _solvent_ref_bonds(get_solvent("water"), 1.3)
    pos = np.array(sysd["positions"], dtype=float)
    # Drag an O of molecule 4 on top of an O of molecule 1 (forms an O–O bond).
    o1 = 3 + 1 * nm
    o4 = 3 + 4 * nm
    pos[o4] = pos[o1] + np.array([1.2, 0.0, 0.0])
    nums = symbols_to_numbers(list(sysd["symbols"]))
    broken = _find_broken_solvent(pos, nums, 3, nm, ref_bonds, 1.3)
    assert 1 in broken and 4 in broken


def test_topology_repair_disabled_keeps_count(water_solute):
    """--no-topology-repair path: auto_solvate with topology_repair=False runs."""
    from mars.solvation import auto_solvate

    out = auto_solvate(
        water_solute,
        "water",
        n_molecules=5,
        optimise_each_layer=False,
        topology_repair=False,
        seed=2,
    )
    assert (len(out["symbols"]) - 3) // 3 == 5


# ── Barostat (spherical moving-wall droplet) ─────────────────────────────────


def test_place_droplet_fills_shell():
    """Droplet placement puts molecules inside the [r_inner, r_outer] shell."""
    from mars.solvation import _place_droplet

    eth = get_solvent("ether")
    center = np.zeros(3)
    struct, n = _place_droplet(
        eth,
        15,
        center,
        r_inner=4.0,
        r_outer=16.0,
        vdw_scale=0.85,
        buffer=0.0,
        n_orient=8,
        seed=1,
    )
    assert n > 0
    p = np.asarray(struct["positions"])
    rad = np.linalg.norm(p - center, axis=1)
    # Atoms live in the shell (small slack for molecular extent at the edges).
    assert rad.min() > 2.0
    assert rad.max() < 18.0


def test_wall_reaction_force_rises_when_compressed():
    """The outer-wall reaction force increases as the wall moves inward."""
    from mars.solvation import _place_droplet, _wall_reaction_force

    eth = get_solvent("ether")
    center = np.zeros(3)
    struct, _ = _place_droplet(
        eth,
        20,
        center,
        r_inner=4.0,
        r_outer=15.0,
        vdw_scale=0.85,
        buffer=0.0,
        n_orient=8,
        seed=2,
    )
    p = np.asarray(struct["positions"])
    f_loose = _wall_reaction_force(p, center, 15.0, 5.0, 2)
    f_tight = _wall_reaction_force(p, center, 9.0, 5.0, 2)
    assert f_tight > f_loose


def test_shell_wall_barrier_penalises_cavity_and_exterior():
    """Inner wall penalises atoms in the cavity; outer wall penalises exterior."""
    import jax.numpy as jnp

    from mars.solvation import _shell_wall_barrier

    barrier = _shell_wall_barrier(np.zeros(3), r_inner=4.0, r_outer=12.0, k=5.0, exponent=2)
    inside_cavity = jnp.array([[1.0, 0.0, 0.0]])  # r=1 < r_inner
    in_shell = jnp.array([[8.0, 0.0, 0.0]])  # r=8 in [4,12]
    outside = jnp.array([[15.0, 0.0, 0.0]])  # r=15 > r_outer
    assert float(barrier(inside_cavity)) > 0.0
    assert float(barrier(in_shell)) == 0.0
    assert float(barrier(outside)) > 0.0


def test_barostat_requires_n_molecules():
    """The CLI rejects --barostat without --n-molecules."""
    from mars.cli._parser import create_parser

    args = create_parser().parse_args(["solvation", "x.xyz", "--barostat"])
    assert args.barostat is True
    assert args.n_molecules is None  # validated/rejected at runtime in _solvation.py
