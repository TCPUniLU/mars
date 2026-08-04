"""
Tests for the `mars viewer` tool: NCI wall geometry, IR result parsing, and
graceful handling when matplotlib is absent.

These tests avoid opening any GUI: parsing/geometry are pure, and matplotlib is
only imported lazily by the drawing functions (exercised via the headless
``--save`` path elsewhere).
"""

import builtins

import numpy as np
import pytest

# ── NCI wall geometry (must mirror the conformer-search solver) ───────────────


class TestNCISphereParams:
    def test_centroid_centred_wall_at_origin(self):
        from mars.viewer import nci_sphere_params

        # Off-centre triangle so centroid != origin.
        positions = np.array([[1.0, 1.0, 1.0], [3.0, 1.0, 1.0], [2.0, 3.0, 1.0]])
        buffer = 3.0
        centered, center, radius, max_dist, min_clearance, far_atom = nci_sphere_params(
            positions, buffer
        )

        # Positions are centred by their centroid.
        assert np.allclose(centered.mean(axis=0), 0.0, atol=1e-10)
        # Wall anchored at the origin.
        assert np.allclose(center, 0.0)
        # Radius = max(|r|) + buffer; clearance == buffer.
        assert radius == pytest.approx(max_dist + buffer)
        assert min_clearance == pytest.approx(buffer)
        # far_atom is the atom closest to the wall (largest |r|).
        dists = np.linalg.norm(centered, axis=1)
        assert far_atom == int(np.argmax(dists))

    def test_matches_solver_formula(self):
        """Independent recomputation of the solver's radius for a known case."""
        from mars.viewer import nci_sphere_params

        positions = np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]])
        _, _, radius, max_dist, _, _ = nci_sphere_params(positions, 3.0)

        centroid = positions.mean(axis=0)
        expected_max = float(np.linalg.norm(positions - centroid, axis=1).max())
        assert max_dist == pytest.approx(expected_max, abs=1e-6)
        assert radius == pytest.approx(expected_max + 3.0, abs=1e-6)

    def test_buffer_slider_updates_radius(self):
        """The interactive buffer must recompute radius = max_dist + buffer."""
        from mars.viewer import _NCIViewer

        frames = [np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]])]
        v = _NCIViewer(
            frames,
            np.array([8, 1, 1]),
            ["O", "H", "H"],
            buffer=3.0,
            atom_scale=1.0,
            bond_width=2.5,
            show_grid=False,
        )
        r0 = v.radius
        v.buffer = 7.0
        assert v.radius == pytest.approx(r0 + 4.0, abs=1e-6)
        assert v.radius == pytest.approx(v.max_dist + 7.0, abs=1e-6)


# ── IR result file parsing ────────────────────────────────────────────────────


@pytest.fixture
def ir_files(tmp_path):
    """Write minimal normal_modes / spectrum / analysis files."""
    modes = tmp_path / "normal_modes.xyz"
    modes.write_text(
        "3\n"
        "Mode 0, Freq: -0.0012 cm⁻¹, Force Const: 1.0e-10 eV/Å²\n"
        "O    0.0 0.0 0.117   0.0 0.0 0.5\n"
        "H    0.0 0.757 -0.469   0.0 0.0 0.5\n"
        "H    0.0 -0.757 -0.469   0.0 0.0 0.5\n"
        "3\n"
        "Mode 1, Freq: 1595.0000 cm⁻¹, Force Const: 5.0e-01 eV/Å²\n"
        "O    0.0 0.0 0.117   0.0 0.0 -0.4\n"
        "H    0.0 0.757 -0.469   0.0 0.3 0.2\n"
        "H    0.0 -0.757 -0.469   0.0 -0.3 0.2\n"
    )
    spectrum = tmp_path / "ir_spectrum.dat"
    spectrum.write_text(
        "# header\n#\n# Frequency    Intensity\n"
        "           -0.00              0.000000\n"
        "         1595.00              0.450000\n"
    )
    # Fixed-width ALL MODES table (matches ir.save_mode_analysis columns).
    analysis = tmp_path / "mode_analysis.txt"
    line0 = f"{0:<6} {(-0.0):>12.1f}   {0.0:>10.4f}   {'trans/rot':<12} {'':<30}\n"
    line1 = (
        f"{1:<6} {1595.0:>12.1f}   {0.45:>10.4f}   {'bending':<12} "
        f"{'H-O-H asymmetric deformation':<30} {'H-O (alcohol)'}\n"
    )
    analysis.write_text(
        "-" * 100 + "\nALL MODES\n" + "-" * 100 + "\n"
        "Mode   Freq\n" + "-" * 100 + "\n" + line0 + line1 + "\n"
        "DETAILED VIBRATIONAL MODE ANALYSIS\n"
    )
    return tmp_path


def test_parse_normal_modes_xyz(ir_files):
    from mars.cli._viewer import _parse_normal_modes_xyz

    symbols, numbers, positions, displacements, freqs, mode_idx = _parse_normal_modes_xyz(
        str(ir_files / "normal_modes.xyz")
    )
    assert symbols == ["O", "H", "H"]
    assert list(numbers) == [8, 1, 1]
    assert positions.shape == (3, 3)
    assert displacements.shape == (2, 3, 3)
    assert freqs[0] == pytest.approx(-0.0012)
    assert freqs[1] == pytest.approx(1595.0)
    assert list(mode_idx) == [0, 1]


def test_parse_ir_spectrum_dat(ir_files):
    from mars.cli._viewer import _parse_ir_spectrum_dat

    freq, inten = _parse_ir_spectrum_dat(str(ir_files / "ir_spectrum.dat"))
    assert len(freq) == 2
    assert inten[1] == pytest.approx(0.45)


def test_parse_mode_analysis_fixed_columns(ir_files):
    from mars.cli._viewer import _parse_mode_analysis

    cls = _parse_mode_analysis(str(ir_files / "mode_analysis.txt"))
    assert cls[0]["type"] == "trans/rot"
    # Label and functional group (which itself contains a space) recovered.
    assert cls[1]["type"] == "bending"
    assert cls[1]["label"] == "H-O-H asymmetric deformation"
    assert cls[1]["functional_group"] == "H-O (alcohol)"


def test_resolve_ir_data_discovery(ir_files):
    import argparse

    from mars.cli._viewer import _resolve_ir_data

    args = argparse.Namespace(dir=str(ir_files), modes=None, spectrum=None, mode_analysis=None)
    data = _resolve_ir_data(args)
    assert data["positions"].shape == (3, 3)
    assert data["displacements"].shape == (2, 3, 3)
    # Intensities aligned by index (same length as modes).
    assert len(data["intensities"]) == 2
    assert data["intensities"][1] == pytest.approx(0.45)
    assert data["classification"][1]["type"] == "bending"


# ── matplotlib-missing path ───────────────────────────────────────────────────


def test_require_matplotlib_raises_without_matplotlib(monkeypatch):
    from mars import viewer

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("matplotlib not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="matplotlib is required"):
        viewer._require_matplotlib()


# ── headless rendering (requires matplotlib) ──────────────────────────────────


@pytest.fixture
def water_structure():
    return {
        "numbers": np.array([8, 1, 1]),
        "symbols": ["O", "H", "H"],
        "positions": np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]]),
    }


def test_view_molecule_save(tmp_path, water_structure):
    pytest.importorskip("matplotlib")
    from mars import viewer

    out = tmp_path / "mol.png"
    viewer.view_molecule(
        water_structure, save=str(out), atom_scale=1.5, bond_width=3.0, show_grid=True
    )
    assert out.exists() and out.stat().st_size > 0


def test_view_nci_save(tmp_path, water_structure):
    pytest.importorskip("matplotlib")
    from mars import viewer

    out = tmp_path / "nci.png"
    viewer.view_nci(water_structure, buffer=4.0, save=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_view_ir_save(tmp_path, ir_files):
    pytest.importorskip("matplotlib")
    import argparse

    from mars import viewer
    from mars.cli._viewer import _resolve_ir_data

    args = argparse.Namespace(dir=str(ir_files), modes=None, spectrum=None, mode_analysis=None)
    data = _resolve_ir_data(args)
    out = tmp_path / "ir.png"
    viewer.view_ir(data, save=str(out))
    assert out.exists() and out.stat().st_size > 0


# ── interactive handlers (requires matplotlib, headless Agg) ──────────────────


def _build_ir_viewer(ir_files):
    import argparse

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mars.cli._viewer import _resolve_ir_data
    from mars.viewer import _IRViewer

    args = argparse.Namespace(dir=str(ir_files), modes=None, spectrum=None, mode_analysis=None)
    data = _resolve_ir_data(args)
    v = _IRViewer(
        data, 1.0, 10.0, (400, 4000), 0.5, atom_scale=1.0, bond_width=2.5, show_grid=False
    )
    fig = plt.figure(figsize=(13, 9))
    v.build(fig)
    return v


def test_ir_viewer_navigation_and_display_handlers(ir_files):
    pytest.importorskip("matplotlib")
    v = _build_ir_viewer(ir_files)

    start = v.current
    v._step(+1)
    assert v.current != start or len(v.nav_indices) == 1
    v._step(-1)
    assert v.current == start

    # Display tuning handlers update state without raising.
    v._on_atom_scale(3.0)
    assert v.atom_scale == 3.0
    v._on_bond_width(5.0)
    assert v.bond_width == 5.0
    v._toggle_grid()
    assert v.show_grid is True
    assert "on" in v.btn_grid.label.get_text()
    v._toggle_grid()
    assert v.show_grid is False


def test_ir_viewer_click_selects_nearest_mode(ir_files):
    pytest.importorskip("matplotlib")
    v = _build_ir_viewer(ir_files)

    class _Event:
        pass

    evt = _Event()
    evt.inaxes = v.ax_spec
    target = float(v.frequencies[v.nav_indices[-1]])
    evt.xdata = target + 30.0
    v._on_click(evt)
    assert v.current == v.nav_indices[-1]


def test_ir_viewer_animation_toggle(ir_files):
    pytest.importorskip("matplotlib")
    v = _build_ir_viewer(ir_files)

    v._toggle_animation()
    assert v._animating is True
    assert v.anim is not None
    v._toggle_animation()
    assert v._animating is False
    assert v.anim is None


def _ir_data(frequencies, intensities):
    """Minimal ir_data dict for the viewer (3-atom water geometry)."""
    pos = np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]])
    n = len(frequencies)
    disp = np.zeros((n, 3, 3))
    disp[:, 0, 2] = 0.5  # give every mode some displacement
    return {
        "positions": pos,
        "numbers": np.array([8, 1, 1]),
        "symbols": ["O", "H", "H"],
        "frequencies": np.asarray(frequencies, dtype=float),
        "intensities": np.asarray(intensities, dtype=float),
        "displacements": disp,
        "classification": {},
    }


def test_ir_viewer_includes_all_modes():
    """Negative/imaginary and low-frequency modes are kept (not filtered)."""
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mars.viewer import _IRViewer

    freqs = [-120.0, -5.0, 0.3, 12.0, 1595.0, 3650.0]
    data = _ir_data(freqs, [0.0, 0.0, 0.0, 0.0, 0.45, 1.0])
    v = _IRViewer(data, 1.0, 10.0, (400, 4000), 0.5)
    # Every mode is navigable, including negatives and low frequencies.
    assert v.nav_indices == list(range(len(freqs)))
    fig = plt.figure()
    v.build(fig)
    # Display range widens to include the negative mode.
    lo, hi = v._display_range()
    assert lo <= -120.0
    assert hi >= 3650.0


def test_ir_viewer_no_intensities_uniform_sticks():
    """With no computed intensities the viewer still shows every mode."""
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mars.viewer import _IRViewer

    freqs = [-10.0, 0.0, 500.0, 1700.0, 3000.0]
    data = _ir_data(freqs, [0.0] * len(freqs))  # no dipoles → all-zero intensities
    v = _IRViewer(data, 1.0, 10.0, (400, 4000), 0.5)
    assert v.has_intensities is False
    fig = plt.figure()
    v.build(fig)
    # Uniform stick height; envelope skipped; marker sits at stick top.
    assert v._envelope is None
    assert v._marker_y(2) == 1.0
    # Navigation still cycles through all modes.
    v._step(+1)
    assert v.current in v.nav_indices


def test_resolve_ir_data_without_spectrum(ir_files):
    """Missing spectrum file → zero intensities (viewer shows uniform sticks)."""
    import argparse

    from mars.cli._viewer import _resolve_ir_data

    # Point only at the modes file; no spectrum/analysis.
    args = argparse.Namespace(
        dir=None,
        modes=str(ir_files / "normal_modes.xyz"),
        spectrum="/does/not/exist.dat",
        mode_analysis="/does/not/exist.txt",
    )
    data = _resolve_ir_data(args)
    assert not np.any(data["intensities"] > 0)


def test_nci_viewer_display_handlers(water_structure):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mars.viewer import _NCIViewer

    v = _NCIViewer(
        [water_structure["positions"]],
        water_structure["numbers"],
        water_structure["symbols"],
        buffer=3.0,
        atom_scale=1.0,
        bond_width=2.5,
        show_grid=False,
    )
    fig = plt.figure(figsize=(9, 8))
    v.build(fig)

    r0 = v.radius
    v._on_buffer(6.0)
    assert v.radius == pytest.approx(r0 + 3.0, abs=1e-6)
    v._on_atom_scale(2.0)
    assert v.atom_scale == 2.0
    v._on_bond_width(4.0)
    assert v.bond_width == 4.0
    v._toggle_grid()
    assert v.show_grid is True


# ── multi-conformer navigation ────────────────────────────────────────────────


@pytest.fixture
def two_conformers():
    """Two frames of water (second is displaced) as structure dicts."""
    p0 = np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]])
    p1 = p0 + np.array([[0.0, 0.0, 0.2], [0.0, 0.1, 0.0], [0.0, -0.1, 0.0]])
    base = {"numbers": np.array([8, 1, 1]), "symbols": ["O", "H", "H"]}
    return [dict(base, positions=p0), dict(base, positions=p1)]


def test_molecule_viewer_conformer_navigation(two_conformers):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Build directly to avoid plt.show() blocking.
    from mars.viewer import _MoleculeViewer

    v = _MoleculeViewer(
        [s["positions"] for s in two_conformers],
        two_conformers[0]["numbers"],
        two_conformers[0]["symbols"],
        atom_scale=1.0,
        bond_width=2.5,
        show_grid=False,
    )
    assert v.n_frames == 2
    fig = plt.figure(figsize=(9, 8))
    v.build(fig)
    assert v.current_frame == 0
    v._step_frame(+1)
    assert v.current_frame == 1
    assert np.allclose(v.positions, two_conformers[1]["positions"])
    v._step_frame(+1)  # wraps
    assert v.current_frame == 0


def test_nci_viewer_conformer_navigation(two_conformers):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mars.viewer import _NCIViewer

    v = _NCIViewer(
        [s["positions"] for s in two_conformers],
        two_conformers[0]["numbers"],
        two_conformers[0]["symbols"],
        buffer=3.0,
        atom_scale=1.0,
        bond_width=2.5,
        show_grid=False,
    )
    assert v.n_frames == 2
    fig = plt.figure(figsize=(9, 8))
    v.build(fig)
    r0 = v.radius
    v._step_frame(+1)
    assert v.current_frame == 1
    # Each frame is recentred and gets its own radius = max|r| + buffer.
    assert v.radius == pytest.approx(v.max_dist + v.buffer, abs=1e-6)


def test_view_molecule_accepts_single_or_list(water_structure, tmp_path):
    pytest.importorskip("matplotlib")
    from mars import viewer

    # Single dict and one-element list both work.
    out1 = tmp_path / "single.png"
    viewer.view_molecule(water_structure, save=str(out1))
    out2 = tmp_path / "list.png"
    viewer.view_molecule([water_structure], save=str(out2))
    assert out1.exists() and out2.exists()
