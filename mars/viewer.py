"""
Interactive visualization for MARS results.

Three entry points, all matplotlib-based (matplotlib is an optional
dependency, imported lazily):

* :func:`view_molecule` — a plain interactive 3D molecule viewer.
* :func:`view_nci` — a molecule wrapped in the spherical confinement wall used
  by ``mars input.xyz --nci`` so the user can judge whether the buffer/padding
  is large enough.
* :func:`view_ir` — an interactive IR explorer: the 3D molecule with per-mode
  displacement arrows (and an optional oscillation animation), an info panel,
  and a clickable stick/envelope IR spectrum with a marker ball on the selected
  mode.

The NCI wall geometry deliberately mirrors what the conformer-search solver
actually enforces (``mars/cli/_conformer_search.py``): positions are centred by
their centroid, the wall is anchored at the origin, and the radius is
``max(|r_i|) + buffer``.  See :func:`nci_sphere_params`.
"""

from typing import Dict, Optional

import numpy as np

from .log import log_info


def _require_matplotlib(save: Optional[str] = None):
    """Lazily import matplotlib and its 3D / widget / animation helpers.

    When *save* is given we select the non-interactive ``Agg`` backend before
    importing pyplot so the viewer works headlessly (servers, CI, plain SSH).

    Raises:
        RuntimeError: if matplotlib is not installed.
    """
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise RuntimeError(
            "matplotlib is required for `mars viewer`. " "Install it with: pip install matplotlib"
        ) from exc

    if save is not None:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt  # noqa: F401
    from matplotlib import (
        animation,  # noqa: F401
        widgets,  # noqa: F401
    )
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    return matplotlib


# ---------------------------------------------------------------------------
# Shared 3D drawing helpers
# ---------------------------------------------------------------------------


def _as_np(positions) -> np.ndarray:
    return np.asarray(positions, dtype=float)


# Default marker size factor (Å radius → scatter points); `atom_scale`
# multiplies the resulting area, so the user can grow/shrink atoms freely.
_ATOM_SIZE_BASE = 46.0


def _atom_sizes(numbers: np.ndarray, atom_scale: float = 1.0) -> np.ndarray:
    """Marker areas (points²) for a scatter plot, scaled by covalent radius.

    ``atom_scale`` linearly scales the marker area so the user can tune the
    apparent atom size.
    """
    from .utils import get_covalent_radius

    radii = np.array([get_covalent_radius(int(z)) or 0.7 for z in numbers])
    # Scatter `s` is a marker area in points²; scale by covalent radius.
    return (radii * _ATOM_SIZE_BASE) ** 2 * float(atom_scale)


def _atom_colors(numbers: np.ndarray):
    from .utils import get_cpk_color

    return [get_cpk_color(int(z), normalized=True) for z in numbers]


def _set_equal_3d(ax, positions: np.ndarray, pad: float = 0.0):
    """Give the 3D axes a cubic aspect centred on the molecule."""
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    center = 0.5 * (mins + maxs)
    half = 0.5 * float((maxs - mins).max()) + pad
    half = max(half, 1.0)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:  # pragma: no cover - very old matplotlib
        pass


def _apply_clean_3d_style(ax, show_grid: bool):
    """Toggle the grid/axes. By default (``show_grid`` False) the entire 3D
    axis — ticks, labels, panes and grid — is hidden for a clean molecule view;
    ``show_grid`` True restores the full grid."""
    if show_grid:
        ax.grid(True)
        ax.set_axis_on()
    else:
        ax.grid(False)
        ax.set_axis_off()


def draw_molecule(
    ax,
    positions: np.ndarray,
    numbers: np.ndarray,
    symbols=None,
    atom_scale: float = 1.0,
    bond_width: float = 2.5,
    show_grid: bool = False,
):
    """Draw atoms (CPK spheres) and bonds (grey sticks) on a 3D axes.

    Args:
        atom_scale: linear multiplier on the marker area (atom size).
        bond_width: bond line width in points.
        show_grid: draw the 3D grid/panes (default off for a clean view).

    Returns:
        (scatter, bond_lines, bonds) — the scatter handle, the list of
        ``Line3D`` bond artists, and the list of ``(i, j)`` bonded pairs, so a
        caller can animate them.
    """
    from .utils import detect_bonds

    positions = _as_np(positions)
    numbers = np.asarray(numbers, dtype=int)

    bonds, _ = detect_bonds(positions, numbers)
    bond_lines = []
    for i, j in bonds:
        (line,) = ax.plot(
            [positions[i, 0], positions[j, 0]],
            [positions[i, 1], positions[j, 1]],
            [positions[i, 2], positions[j, 2]],
            color="#555555",
            linewidth=bond_width,
            zorder=1,
        )
        bond_lines.append(line)

    scatter = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        s=_atom_sizes(numbers, atom_scale),
        c=_atom_colors(numbers),  # Jmol/CPK colours
        edgecolors="black",
        linewidths=0.5,
        depthshade=True,
        zorder=2,
    )

    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    _apply_clean_3d_style(ax, show_grid)
    _set_equal_3d(ax, positions)
    return scatter, bond_lines, bonds


def _update_bond_lines(bond_lines, bonds, positions):
    """Move existing bond ``Line3D`` artists to new atom positions."""
    for line, (i, j) in zip(bond_lines, bonds):
        line.set_data_3d(
            [positions[i, 0], positions[j, 0]],
            [positions[i, 1], positions[j, 1]],
            [positions[i, 2], positions[j, 2]],
        )


class _DisplayControlsMixin:
    """Live handlers shared by every viewer: atom size, bond width, grid toggle.

    Mixed into the viewer classes; expects the host to set ``self.ax3d``,
    ``self.scatter``, ``self.bond_lines``, ``self.numbers``, ``self.atom_scale``,
    ``self.bond_width``, ``self.show_grid`` and ``self.btn_grid``.
    """

    def _on_atom_scale(self, value):
        self.atom_scale = float(value)
        self.scatter.set_sizes(_atom_sizes(self.numbers, self.atom_scale))
        self.ax3d.figure.canvas.draw_idle()

    def _on_bond_width(self, value):
        self.bond_width = float(value)
        for line in self.bond_lines:
            line.set_linewidth(self.bond_width)
        self.ax3d.figure.canvas.draw_idle()

    def _toggle_grid(self):
        self.show_grid = not self.show_grid
        _apply_clean_3d_style(self.ax3d, self.show_grid)
        self.btn_grid.label.set_text("Grid: on" if self.show_grid else "Grid: off")
        self.ax3d.figure.canvas.draw_idle()


# ---------------------------------------------------------------------------
# 1) Plain molecule viewer
# ---------------------------------------------------------------------------


class _MoleculeViewer(_DisplayControlsMixin):
    """Interactive plain-molecule viewer with live atom/bond/grid controls and,
    for multi-frame inputs, Prev/Next conformer navigation."""

    def __init__(self, frames, numbers, symbols, atom_scale, bond_width, show_grid):
        # frames: list of (n_atoms, 3) position arrays (>= 1).
        self.frames = [_as_np(p) for p in frames]
        self.numbers = np.asarray(numbers, dtype=int)
        self.symbols = symbols
        self.n_frames = len(self.frames)
        self.current_frame = 0
        self.positions = self.frames[0]
        self.atom_scale = float(atom_scale)
        self.bond_width = float(bond_width)
        self.show_grid = bool(show_grid)

    def build(self, fig):
        from matplotlib.widgets import Button, Slider

        self.ax3d = fig.add_axes([0.02, 0.20, 0.96, 0.74], projection="3d")
        self.scatter, self.bond_lines, self.bonds = draw_molecule(
            self.ax3d,
            self.positions,
            self.numbers,
            self.symbols,
            atom_scale=self.atom_scale,
            bond_width=self.bond_width,
            show_grid=self.show_grid,
        )
        # Equal aspect over ALL frames so atoms never clip while navigating.
        _set_equal_3d(self.ax3d, np.vstack(self.frames))
        self._update_title()

        # Widget row: grid + atom/bond sliders, plus conformer nav if multi-frame.
        self.ax_grid = fig.add_axes([0.04, 0.05, 0.09, 0.05])
        self.btn_grid = Button(self.ax_grid, "Grid: on" if self.show_grid else "Grid: off")
        self.btn_grid.on_clicked(lambda _evt: self._toggle_grid())

        if self.n_frames > 1:
            self.ax_prev = fig.add_axes([0.15, 0.05, 0.08, 0.05])
            self.ax_next = fig.add_axes([0.24, 0.05, 0.08, 0.05])
            self.btn_prev = Button(self.ax_prev, "◄ Conf")
            self.btn_next = Button(self.ax_next, "Conf ►")
            self.btn_prev.on_clicked(lambda _evt: self._step_frame(-1))
            self.btn_next.on_clicked(lambda _evt: self._step_frame(+1))
            fig.canvas.mpl_connect("key_press_event", self._on_key)
            atom_x = 0.50
        else:
            atom_x = 0.40

        self.ax_atom = fig.add_axes([atom_x, 0.085, 0.42, 0.03])
        self.ax_bond = fig.add_axes([atom_x, 0.035, 0.42, 0.03])
        self.sld_atom = Slider(self.ax_atom, "Atom", 0.2, 4.0, valinit=self.atom_scale)
        self.sld_bond = Slider(self.ax_bond, "Bond", 0.5, 8.0, valinit=self.bond_width)
        self.sld_atom.on_changed(self._on_atom_scale)
        self.sld_bond.on_changed(self._on_bond_width)

    def _update_title(self):
        if self.n_frames > 1:
            self.ax3d.set_title(
                f"{len(self.numbers)} atoms — "
                f"conformer {self.current_frame + 1}/{self.n_frames}"
            )
        else:
            self.ax3d.set_title(f"{len(self.numbers)} atoms")

    def _step_frame(self, direction):
        self.current_frame = (self.current_frame + direction) % self.n_frames
        self.positions = self.frames[self.current_frame]
        self.scatter._offsets3d = (
            self.positions[:, 0],
            self.positions[:, 1],
            self.positions[:, 2],
        )
        _update_bond_lines(self.bond_lines, self.bonds, self.positions)
        self._update_title()
        self.ax3d.figure.canvas.draw_idle()

    def _on_key(self, event):
        if event.key in ("right", "up"):
            self._step_frame(+1)
        elif event.key in ("left", "down"):
            self._step_frame(-1)


def view_molecule(
    structure: Dict,
    save: Optional[str] = None,
    atom_scale: float = 1.0,
    bond_width: float = 2.5,
    show_grid: bool = False,
):
    """Show an interactive 3D view of a molecule (rotate/zoom with the mouse).

    *structure* may be a single structure dict or a list of structure dicts
    (multi-frame input); with more than one frame, Prev/Next buttons and the
    ←/→ keys step through conformers. Atom size, bond width, and the grid/axes
    are adjustable live.
    """
    _require_matplotlib(save)
    import matplotlib.pyplot as plt

    structures = structure if isinstance(structure, (list, tuple)) else [structure]
    frames = [s["positions"] for s in structures]
    numbers = structures[0]["numbers"]
    symbols = structures[0].get("symbols")

    viewer = _MoleculeViewer(frames, numbers, symbols, atom_scale, bond_width, show_grid)
    fig = plt.figure(figsize=(9, 8))
    viewer.build(fig)
    # Keep a reference so widgets are not garbage-collected.
    fig._mars_molecule_viewer = viewer
    if viewer.n_frames > 1:
        log_info(f"Loaded {viewer.n_frames} conformers — use ◄/► or arrow keys.")

    if save is not None:
        fig.savefig(save, dpi=200)
        log_info(f"Saved molecule view to {save}")
        plt.close(fig)
        return viewer
    plt.show()
    return viewer


# ---------------------------------------------------------------------------
# 2) NCI confinement-wall viewer
# ---------------------------------------------------------------------------


def nci_sphere_params(positions, buffer: float):
    """Confinement sphere geometry, mirroring the conformer-search solver.

    Reproduces ``mars/cli/_conformer_search.py``: positions are centred by their
    centroid, the wall is anchored at the origin of that centred frame, and the
    radius is ``max(|r_i|) + buffer``.

    Args:
        positions: (N, 3) atomic positions in Å.
        buffer: Clearance (Å) between the outermost atom and the wall.

    Returns:
        ``(centered_positions, center, radius, max_dist, min_clearance, far_atom)``
        where *center* is the origin, *min_clearance* equals *buffer*, and
        *far_atom* is the index of the atom closest to the wall.
    """
    from .rmsd import center_positions

    positions = _as_np(positions)
    centered = np.asarray(center_positions(positions))
    center = np.zeros(3, dtype=float)
    distances = np.linalg.norm(centered - center, axis=-1)
    max_dist = float(distances.max())
    radius = max_dist + float(buffer)
    min_clearance = float(radius - max_dist)
    far_atom = int(np.argmax(distances))
    return centered, center, radius, max_dist, min_clearance, far_atom


def _sphere_mesh(center, radius, n_u=48, n_v=24):
    """Parametric sphere surface coordinates centred at *center*."""
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    sx = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    sy = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    sz = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return sx, sy, sz


class _NCIViewer(_DisplayControlsMixin):
    """Interactive confinement-wall viewer with a live buffer slider, the shared
    atom-size / bond-width / grid controls, and (for multi-frame inputs)
    Prev/Next conformer navigation. Each conformer is independently
    centroid-centred and gets its own ``radius = max(|r|) + buffer``."""

    def __init__(self, frames, numbers, symbols, buffer, atom_scale, bond_width, show_grid):
        self.raw_frames = [_as_np(p) for p in frames]
        self.n_frames = len(self.raw_frames)
        self.current_frame = 0
        self.numbers = np.asarray(numbers, dtype=int)
        self.symbols = symbols
        self.buffer = float(buffer)
        self.atom_scale = float(atom_scale)
        self.bond_width = float(bond_width)
        self.show_grid = bool(show_grid)
        self.sphere = None
        self.center = np.zeros(3, dtype=float)
        self._recompute_frame()

    def _recompute_frame(self):
        self.centered, self.center, _, self.max_dist, _, self.far_atom = nci_sphere_params(
            self.raw_frames[self.current_frame], self.buffer
        )

    @property
    def radius(self):
        return self.max_dist + self.buffer

    def log_summary(self):
        far_sym = (
            self.symbols[self.far_atom]
            if self.symbols is not None
            else str(self.numbers[self.far_atom])
        )
        log_info("NCI confinement wall (centroid-centred, wall at origin):")
        log_info("  centre        = (0.000, 0.000, 0.000) Å")
        log_info(f"  radius        = {self.radius:.2f} Å")
        log_info(f"  buffer        = {self.buffer:.2f} Å")
        log_info(f"  molecule extent (max |r|) = {self.max_dist:.2f} Å")
        log_info(
            f"  closest atom to wall = {far_sym}({self.far_atom}), "
            f"clearance = {self.buffer:.2f} Å"
        )

    def build(self, fig):
        from matplotlib.widgets import Button, Slider

        self.ax3d = fig.add_axes([0.02, 0.20, 0.96, 0.74], projection="3d")
        self.scatter, self.bond_lines, self.bonds = draw_molecule(
            self.ax3d,
            self.centered,
            self.numbers,
            self.symbols,
            atom_scale=self.atom_scale,
            bond_width=self.bond_width,
            show_grid=self.show_grid,
        )
        self._draw_sphere()
        self._update_view()

        # Widget row.
        self.ax_grid = fig.add_axes([0.04, 0.05, 0.09, 0.05])
        self.btn_grid = Button(self.ax_grid, "Grid: on" if self.show_grid else "Grid: off")
        self.btn_grid.on_clicked(lambda _evt: self._toggle_grid())

        if self.n_frames > 1:
            self.ax_prev = fig.add_axes([0.15, 0.05, 0.08, 0.05])
            self.ax_next = fig.add_axes([0.24, 0.05, 0.08, 0.05])
            self.btn_prev = Button(self.ax_prev, "◄ Conf")
            self.btn_next = Button(self.ax_next, "Conf ►")
            self.btn_prev.on_clicked(lambda _evt: self._step_frame(-1))
            self.btn_next.on_clicked(lambda _evt: self._step_frame(+1))
            fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.ax_buffer = fig.add_axes([0.45, 0.115, 0.42, 0.03])
        self.ax_atom = fig.add_axes([0.45, 0.075, 0.42, 0.03])
        self.ax_bond = fig.add_axes([0.45, 0.035, 0.42, 0.03])
        self.sld_buffer = Slider(self.ax_buffer, "Buffer (Å)", 0.0, 15.0, valinit=self.buffer)
        self.sld_atom = Slider(self.ax_atom, "Atom", 0.2, 4.0, valinit=self.atom_scale)
        self.sld_bond = Slider(self.ax_bond, "Bond", 0.5, 8.0, valinit=self.bond_width)
        self.sld_buffer.on_changed(self._on_buffer)
        self.sld_atom.on_changed(self._on_atom_scale)
        self.sld_bond.on_changed(self._on_bond_width)

    def _draw_sphere(self):
        if self.sphere is not None:
            try:
                self.sphere.remove()
            except Exception:
                pass
        sx, sy, sz = _sphere_mesh(self.center, self.radius)
        self.sphere = self.ax3d.plot_surface(
            sx, sy, sz, color="tab:blue", alpha=0.12, linewidth=0, antialiased=True
        )

    def _update_view(self):
        sphere_box = np.array([self.center - self.radius, self.center + self.radius], dtype=float)
        _set_equal_3d(self.ax3d, np.vstack([self.centered, sphere_box]))
        title = (
            f"NCI wall — radius {self.radius:.2f} Å  "
            f"(buffer {self.buffer:.2f} Å, clearance {self.buffer:.2f} Å)"
        )
        if self.n_frames > 1:
            title += f"  ·  conformer {self.current_frame + 1}/{self.n_frames}"
        self.ax3d.set_title(title)

    def _on_buffer(self, value):
        self.buffer = float(value)
        self._draw_sphere()
        self._update_view()
        self.ax3d.figure.canvas.draw_idle()

    def _step_frame(self, direction):
        self.current_frame = (self.current_frame + direction) % self.n_frames
        self._recompute_frame()
        self.scatter._offsets3d = (
            self.centered[:, 0],
            self.centered[:, 1],
            self.centered[:, 2],
        )
        _update_bond_lines(self.bond_lines, self.bonds, self.centered)
        self._draw_sphere()
        self._update_view()
        self.ax3d.figure.canvas.draw_idle()

    def _on_key(self, event):
        if event.key in ("right", "up"):
            self._step_frame(+1)
        elif event.key in ("left", "down"):
            self._step_frame(-1)


def view_nci(
    structure: Dict,
    buffer: float = 3.0,
    save: Optional[str] = None,
    atom_scale: float = 1.0,
    bond_width: float = 2.5,
    show_grid: bool = False,
):
    """Show the molecule inside its spherical confinement wall.

    Draws a translucent sphere at ``radius = max(|r_i|) + buffer`` (centroid
    centred at the origin, matching ``mars ... --nci``) and reports the centre,
    radius, buffer and closest atom-to-wall clearance so the user can decide
    whether to increase ``--nci-buffer``. The buffer is adjustable live with a
    slider, alongside the shared atom-size / bond-width / grid controls.

    *structure* may be a single structure dict or a list of dicts (multi-frame
    input); with more than one frame, Prev/Next buttons and the ←/→ keys step
    through conformers, each re-centred with its own wall.
    """
    _require_matplotlib(save)
    import matplotlib.pyplot as plt

    structures = structure if isinstance(structure, (list, tuple)) else [structure]
    frames = [s["positions"] for s in structures]
    numbers = structures[0]["numbers"]
    symbols = structures[0].get("symbols")

    viewer = _NCIViewer(frames, numbers, symbols, buffer, atom_scale, bond_width, show_grid)
    viewer.log_summary()
    if viewer.n_frames > 1:
        log_info(f"Loaded {viewer.n_frames} conformers — use ◄/► or arrow keys.")

    fig = plt.figure(figsize=(9, 8))
    viewer.build(fig)
    # Keep a reference so widgets are not garbage-collected.
    fig._mars_nci_viewer = viewer

    if save is not None:
        fig.savefig(save, dpi=200)
        log_info(f"Saved NCI wall view to {save}")
        plt.close(fig)
        return viewer
    plt.show()
    return viewer


# ---------------------------------------------------------------------------
# 3) Interactive IR viewer
# ---------------------------------------------------------------------------


def _lorentzian_envelope(freqs, intensities, freq_range, broadening):
    """Lorentzian-broadened envelope, matching ``ir.save_ir_plot``."""
    axis = np.linspace(freq_range[0], freq_range[1], max(int(freq_range[1] - freq_range[0]), 2))
    spectrum = np.zeros_like(axis)
    g2 = broadening**2
    for f, inten in zip(freqs, intensities):
        if freq_range[0] <= f <= freq_range[1]:
            spectrum += inten * g2 / ((axis - f) ** 2 + g2)
    return axis, spectrum


class _IRViewer(_DisplayControlsMixin):
    """Holds the interactive IR viewer state (current mode, widgets, animation)."""

    def __init__(
        self,
        ir_data,
        arrow_scale,
        broadening,
        freq_range,
        amplitude,
        atom_scale=1.0,
        bond_width=2.5,
        show_grid=False,
    ):
        self.data = ir_data
        self.arrow_scale = float(arrow_scale)
        self.broadening = float(broadening)
        self.freq_range = (float(freq_range[0]), float(freq_range[1]))
        self.amplitude = float(amplitude)
        self.atom_scale = float(atom_scale)
        self.bond_width = float(bond_width)
        self.show_grid = bool(show_grid)

        self.positions = _as_np(ir_data["positions"])
        self.numbers = np.asarray(ir_data["numbers"], dtype=int)
        self.symbols = ir_data.get("symbols")
        self.frequencies = np.asarray(ir_data["frequencies"], dtype=float)
        self.intensities = np.asarray(ir_data["intensities"], dtype=float)
        self.displacements = np.asarray(ir_data["displacements"], dtype=float)
        self.classification = ir_data.get("classification") or {}

        # Whether IR intensities were actually computed. Models without a dipole
        # (no charges) produce frequencies/eigenvectors but no intensities; in
        # that case every stick is drawn at uniform height so all modes show.
        self.has_intensities = bool(np.any(self.intensities > 0))

        # Navigate and plot ALL modes, including negative/imaginary and
        # low-frequency translation/rotation modes — nothing is filtered out.
        self.nav_indices = list(range(len(self.frequencies)))
        self.current = self.nav_indices[0]

        # Base display scale so the largest atom displacement of a mode maps to
        # ~1.5 Å arrows at arrow_scale == 1.0.
        self._mol_span = float(
            np.linalg.norm(self.positions.max(axis=0) - self.positions.min(axis=0))
        )

        self.quiver = None
        self.scatter = None
        self.bond_lines = []
        self.bonds = []
        self.marker = None
        self.anim = None
        self._animating = False

    # -- displacement scaling ------------------------------------------------

    def _mode_disp(self, idx):
        disp = self.displacements[idx]
        max_norm = float(np.linalg.norm(disp, axis=1).max())
        if max_norm < 1e-10:
            return disp, 0.0
        return disp, max_norm

    def _arrow_vectors(self, idx):
        disp, max_norm = self._mode_disp(idx)
        if max_norm == 0.0:
            return np.zeros_like(disp)
        base = 1.5  # Å for the largest arrow at scale 1.0
        return disp * (self.arrow_scale * base / max_norm)

    # -- drawing -------------------------------------------------------------

    def build(self, fig):
        from matplotlib.gridspec import GridSpec
        from matplotlib.widgets import Button, Slider

        # Leave room at the bottom for the widget row.
        gs = GridSpec(
            2,
            2,
            height_ratios=[3, 1],
            width_ratios=[3, 1],
            figure=fig,
            left=0.07,
            right=0.97,
            top=0.95,
            bottom=0.16,
            hspace=0.35,
            wspace=0.25,
        )
        self.ax3d = fig.add_subplot(gs[0, 0], projection="3d")
        self.ax_info = fig.add_subplot(gs[0, 1])
        self.ax_spec = fig.add_subplot(gs[1, :])
        self.ax_info.axis("off")

        self.scatter, self.bond_lines, self.bonds = draw_molecule(
            self.ax3d,
            self.positions,
            self.numbers,
            self.symbols,
            atom_scale=self.atom_scale,
            bond_width=self.bond_width,
            show_grid=self.show_grid,
        )

        self._draw_spectrum()

        # Widget row (fixed figure coordinates, below the spectrum panel).
        # Buttons on the left, two columns of sliders on the right — every
        # display option is adjustable live here.
        self.ax_prev = fig.add_axes([0.05, 0.04, 0.07, 0.05])
        self.ax_next = fig.add_axes([0.13, 0.04, 0.07, 0.05])
        self.ax_anim = fig.add_axes([0.21, 0.04, 0.09, 0.05])
        self.ax_grid = fig.add_axes([0.31, 0.04, 0.08, 0.05])
        self.ax_arrow = fig.add_axes([0.48, 0.075, 0.16, 0.025])
        self.ax_amp = fig.add_axes([0.48, 0.035, 0.16, 0.025])
        self.ax_atom = fig.add_axes([0.80, 0.075, 0.15, 0.025])
        self.ax_bond = fig.add_axes([0.80, 0.035, 0.15, 0.025])

        self.btn_prev = Button(self.ax_prev, "◄ Prev")
        self.btn_next = Button(self.ax_next, "Next ►")
        self.btn_anim = Button(self.ax_anim, "Animate")
        self.btn_grid = Button(self.ax_grid, "Grid: off" if not self.show_grid else "Grid: on")
        self.sld_arrow = Slider(self.ax_arrow, "Arrow", 0.0, 5.0, valinit=self.arrow_scale)
        self.sld_amp = Slider(self.ax_amp, "Amp (Å)", 0.0, 2.0, valinit=self.amplitude)
        self.sld_atom = Slider(self.ax_atom, "Atom", 0.2, 4.0, valinit=self.atom_scale)
        self.sld_bond = Slider(self.ax_bond, "Bond", 0.5, 8.0, valinit=self.bond_width)

        self.btn_prev.on_clicked(lambda _evt: self._step(-1))
        self.btn_next.on_clicked(lambda _evt: self._step(+1))
        self.btn_anim.on_clicked(lambda _evt: self._toggle_animation())
        self.btn_grid.on_clicked(lambda _evt: self._toggle_grid())
        self.sld_arrow.on_changed(self._on_arrow_scale)
        self.sld_amp.on_changed(self._on_amplitude)
        self.sld_atom.on_changed(self._on_atom_scale)
        self.sld_bond.on_changed(self._on_bond_width)

        fig.canvas.mpl_connect("button_press_event", self._on_click)
        fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._update_mode(self.current, redraw=False)

    def _display_range(self):
        """Spectrum x-range: the requested window widened to include every mode
        (so negative / low-frequency modes are always visible)."""
        data_lo = float(self.frequencies.min())
        data_hi = float(self.frequencies.max())
        pad = max(50.0, 0.03 * (data_hi - data_lo))
        lo = min(self.freq_range[0], data_lo - pad)
        hi = max(self.freq_range[1], data_hi + pad)
        return lo, hi

    def _draw_spectrum(self):
        ax = self.ax_spec
        ax.clear()
        fmin, fmax = self._display_range()
        # Uniform stick height when intensities were not computed.
        stick_h = self.intensities if self.has_intensities else np.ones_like(self.frequencies)

        # Sticks for ALL modes, coloured by classification if present.
        for i in self.nav_indices:
            f = self.frequencies[i]
            color = "#1f77b4"
            info = self.classification.get(i)
            if info:
                from .spectro_constants import MODE_TYPE_COLORS

                color = MODE_TYPE_COLORS.get(info.get("type"), "#888888")
            ax.vlines(f, 0, stick_h[i], colors=color, linewidth=1.4)

        # Lorentzian envelope (only when intensities are available).
        if self.has_intensities:
            axis, env = _lorentzian_envelope(
                self.frequencies, self.intensities, (fmin, fmax), self.broadening
            )
            # Scale envelope to match stick height for a readable overlay.
            peak = float(env.max()) if env.size else 0.0
            if peak > 0:
                env = env / peak * float(self.intensities.max())
            ax.plot(axis, env, color="#888888", linewidth=1.0, alpha=0.7)
            ax.fill_between(axis, env, alpha=0.15, color="#888888")
            self._envelope = (axis, env)
        else:
            self._envelope = None

        # Mark the zero-frequency line so negative/imaginary modes are obvious.
        if fmin < 0.0 < fmax:
            ax.axvline(0.0, color="0.7", linewidth=0.8, linestyle="--", zorder=0)

        ax.set_xlim(fmin, fmax)
        ax.set_ylim(0, 1.2 if not self.has_intensities else None)
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Intensity (norm.)" if self.has_intensities else "modes (no intensities)")
        title = "IR spectrum — click a peak or use ◄ ► to select a mode"
        if not self.has_intensities:
            title = "Vibrational modes (no intensities) — click or use ◄ ►"
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Selection marker ball (placed by _update_mode).
        (self.marker,) = ax.plot([], [], "o", color="red", markersize=10, zorder=5)

    def _marker_y(self, idx):
        if not self.has_intensities:
            return 1.0  # uniform stick height
        inten = float(self.intensities[idx])
        if self._envelope is not None:
            axis, env = self._envelope
            j = int(np.argmin(np.abs(axis - self.frequencies[idx])))
            return max(inten, float(env[j]))
        return inten

    def _update_info(self, idx):
        freq = self.frequencies[idx]
        inten_line = (
            f"I = {self.intensities[idx]:.4f} (norm.)"
            if self.has_intensities
            else "I = (not computed)"
        )
        lines = [
            f"Mode {idx}",
            "",
            f"ν = {freq:.1f} cm⁻¹",
            inten_line,
        ]
        info = self.classification.get(idx)
        if info:
            import textwrap

            def _wrap(prefix, value):
                return textwrap.fill(
                    f"{prefix}{value}",
                    width=24,
                    subsequent_indent="  ",
                )

            lines.append("")
            lines.append(f"type: {info.get('type', '')}")
            if info.get("label"):
                lines.append(_wrap("label: ", info["label"]))
            if info.get("functional_group"):
                lines.append(_wrap("group: ", info["functional_group"]))
        lines.append("")
        lines.append("[◄ ►] prev/next mode")
        lines.append("[space] toggle animation")
        lines.append("sliders: arrow/amp,")
        lines.append("  atom size, bond width")
        lines.append("[Grid] toggle grid/axes")
        self.ax_info.clear()
        self.ax_info.axis("off")
        self.ax_info.text(
            0.0,
            1.0,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
            transform=self.ax_info.transAxes,
        )

    def _update_mode(self, idx, redraw=True):
        self.current = idx

        # Redraw arrows (unless animating, where atoms move instead).
        if self.quiver is not None:
            try:
                self.quiver.remove()
            except Exception:
                pass
            self.quiver = None
        if not self._animating:
            vecs = self._arrow_vectors(idx)
            self.quiver = self.ax3d.quiver(
                self.positions[:, 0],
                self.positions[:, 1],
                self.positions[:, 2],
                vecs[:, 0],
                vecs[:, 1],
                vecs[:, 2],
                color="crimson",
                linewidth=1.5,
                arrow_length_ratio=0.3,
                normalize=False,
                zorder=3,
            )

        self._update_info(idx)
        self.marker.set_data([self.frequencies[idx]], [self._marker_y(idx)])

        if redraw:
            self.ax3d.figure.canvas.draw_idle()

    # -- navigation / events -------------------------------------------------

    def _step(self, direction):
        if self.current in self.nav_indices:
            pos = self.nav_indices.index(self.current)
        else:
            pos = 0
        pos = (pos + direction) % len(self.nav_indices)
        self._update_mode(self.nav_indices[pos])

    def _on_click(self, event):
        if event.inaxes is not self.ax_spec or event.xdata is None:
            return
        # Nearest mode to the clicked wavenumber (all modes selectable).
        freqs = np.array([self.frequencies[i] for i in self.nav_indices])
        nearest = self.nav_indices[int(np.argmin(np.abs(freqs - event.xdata)))]
        self._update_mode(nearest)

    def _on_key(self, event):
        if event.key in ("right", "up"):
            self._step(+1)
        elif event.key in ("left", "down"):
            self._step(-1)
        elif event.key == " ":
            self._toggle_animation()

    def _on_arrow_scale(self, value):
        self.arrow_scale = float(value)
        if not self._animating:
            self._update_mode(self.current)

    def _on_amplitude(self, value):
        self.amplitude = float(value)

    # _on_atom_scale / _on_bond_width / _toggle_grid come from
    # _DisplayControlsMixin.

    # -- animation -----------------------------------------------------------

    def _toggle_animation(self):
        if self._animating:
            self._stop_animation()
        else:
            self._start_animation()

    def _start_animation(self):
        from matplotlib import animation

        self._animating = True
        self.btn_anim.label.set_text("Stop")
        # Remove static arrows while animating.
        if self.quiver is not None:
            try:
                self.quiver.remove()
            except Exception:
                pass
            self.quiver = None

        disp, max_norm = self._mode_disp(self.current)
        if max_norm < 1e-10:
            self._stop_animation()
            return
        unit = disp / max_norm  # largest atom moves by `amplitude` Å
        n_frames = 30

        def update(frame):
            factor = self.amplitude * np.sin(2 * np.pi * frame / n_frames)
            pos = self.positions + factor * unit
            self.scatter._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
            _update_bond_lines(self.bond_lines, self.bonds, pos)
            return [self.scatter, *self.bond_lines]

        self.anim = animation.FuncAnimation(
            self.ax3d.figure, update, frames=n_frames, interval=50, blit=False
        )
        self.ax3d.figure.canvas.draw_idle()

    def _stop_animation(self):
        self._animating = False
        self.btn_anim.label.set_text("Animate")
        if self.anim is not None:
            try:
                self.anim.event_source.stop()
            except Exception:
                pass
            self.anim = None
        # Restore equilibrium geometry + arrows.
        self.scatter._offsets3d = (
            self.positions[:, 0],
            self.positions[:, 1],
            self.positions[:, 2],
        )
        _update_bond_lines(self.bond_lines, self.bonds, self.positions)
        self._update_mode(self.current)


def view_ir(
    ir_data: Dict,
    arrow_scale: float = 1.0,
    broadening: float = 10.0,
    freq_range=(400, 4000),
    amplitude: float = 0.5,
    atom_scale: float = 1.0,
    bond_width: float = 2.5,
    show_grid: bool = False,
    save: Optional[str] = None,
):
    """Launch the interactive IR explorer.

    Args:
        ir_data: dict with ``positions`` (N,3), ``numbers`` (N,), ``symbols``,
            ``frequencies`` (M,), ``intensities`` (M,), ``displacements``
            (M, N, 3), and optionally ``classification`` ({mode_index: {...}}).
        arrow_scale: initial displacement-arrow length multiplier.
        broadening: Lorentzian width (cm⁻¹) for the spectrum envelope.
        freq_range: (min, max) wavenumber window for the spectrum panel.
        amplitude: initial animation oscillation amplitude (Å).
        atom_scale: atom marker-size multiplier.
        bond_width: bond line width (points).
        show_grid: draw the 3D grid/panes (default off).
        save: if given, render the initial view to this PNG (headless) instead
            of opening an interactive window.
    """
    _require_matplotlib(save)
    import matplotlib.pyplot as plt

    viewer = _IRViewer(
        ir_data,
        arrow_scale,
        broadening,
        freq_range,
        amplitude,
        atom_scale=atom_scale,
        bond_width=bond_width,
        show_grid=show_grid,
    )
    fig = plt.figure(figsize=(13, 9))
    viewer.build(fig)
    # Keep a reference so widgets/animation are not garbage-collected.
    fig._mars_ir_viewer = viewer

    if save is not None:
        fig.savefig(save, dpi=200)
        log_info(f"Saved IR viewer snapshot to {save}")
        plt.close(fig)
        return viewer
    plt.show()
    return viewer
