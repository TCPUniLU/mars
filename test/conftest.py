"""
Shared fixtures and helpers for the MARS test suite.

All tests use the harmonic potential so no external ML dependencies are needed.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# ── precision ──────────────────────────────────────────────────────────────────
jax.config.update("jax_enable_x64", True)


# ── markers (also declared in pyproject.toml) ──────────────────────────────────


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running tests (skipped by default in CI)")
    config.addinivalue_line("markers", "requires_so3lr: requires the SO3LR potential")
    config.addinivalue_line("markers", "requires_mace: requires the MACE calculator")
    config.addinivalue_line("markers", "requires_dxtb: requires dxtb")
    config.addinivalue_line("markers", "cli: CLI subprocess smoke tests")


# ── energy functions ───────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def harmonic_energy():
    """Harmonic centroid potential: V = k * sum_i |r_i - centroid|^2."""

    def _energy(positions):
        center = jnp.mean(positions, axis=0)
        return jnp.sum((positions - center) ** 2) * 0.1

    return _energy


@pytest.fixture(scope="session")
def bonded_energy():
    """Valence force field over a *given* geometry's topology.

    Needed because neither of the other test potentials can exercise
    internal coordinates: ``harmonic_energy`` is a spring to the centroid, so
    its minimum collapses every atom onto one point where all bond lengths are
    zero and the coordinate set is singular -- and its Cartesian Hessian is
    exactly quadratic and isotropic, which makes any coordinate transformation
    a pessimisation.  ``lj_energy`` has no bonded terms, so covalent-radius
    bond detection on its minimum returns nonsense.

    Returns a factory ``(positions, numbers) -> energy_fn``.
    """
    from mars.potentials import get_potential

    def _make(positions, numbers):
        pot = get_potential("valence", species=numbers, positions=positions)
        return pot.build_energy_fn()

    return _make


@pytest.fixture(scope="session")
def lj_energy():
    """Lennard-Jones potential for testing pair interactions."""

    def _energy(positions, epsilon=0.01, sigma=2.5):
        n = positions.shape[0]
        total = jnp.float64(0.0)
        for i in range(n):
            for j in range(i + 1, n):
                r = jnp.sqrt(jnp.sum((positions[i] - positions[j]) ** 2) + 1e-8)
                sr6 = (sigma / r) ** 6
                total = total + 4.0 * epsilon * (sr6**2 - sr6)
        return total

    return _energy


# ── test structures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def triangle():
    """3-atom equilateral triangle (C3)."""
    from mars.utils import create_structure

    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.75, 1.299, 0.0],
        ]
    )
    return create_structure(positions, ["C", "C", "C"])


@pytest.fixture(scope="session")
def linear_chain():
    """5-atom linear chain (C5)."""
    from mars.utils import create_structure

    positions = jnp.array([[i * 1.5, 0.0, 0.0] for i in range(5)])
    return create_structure(positions, ["C"] * 5)


@pytest.fixture(scope="session")
def small_molecule():
    """8-atom mixed molecule (C4H4) – enough for rotamer MD."""
    from mars.utils import create_structure

    positions = jnp.array(
        [
            [0.00, 0.00, 0.00],
            [1.40, 0.00, 0.00],
            [2.80, 0.00, 0.00],
            [4.20, 0.00, 0.00],
            [-0.50, 1.00, 0.00],
            [1.40, 1.00, 0.00],
            [2.80, 1.00, 0.00],
            [4.70, 1.00, 0.00],
        ]
    )
    return create_structure(positions, ["C", "C", "C", "C", "H", "H", "H", "H"])


@pytest.fixture(scope="session")
def mtd_params_fast():
    """MTD parameters tuned for fast tests."""
    return {"kpush": 0.02, "alpha": 0.5, "cvdump_fs": 10.0}


@pytest.fixture(scope="session")
def water_dimer():
    """Two non-overlapping water molecules — handy for solvation/topology tests."""
    from mars.utils import create_structure

    positions = jnp.array(
        [
            [0.000, 0.000, 0.000],  # O1
            [0.957, 0.000, 0.000],  # H1
            [-0.239, 0.927, 0.000],  # H2
            [3.500, 0.000, 0.000],  # O2
            [4.457, 0.000, 0.000],  # H3
            [3.261, 0.927, 0.000],  # H4
        ]
    )
    return create_structure(positions, ["O", "H", "H", "O", "H", "H"])


@pytest.fixture(scope="session")
def methane():
    """4 H tetrahedrally around C — sturdy structure for IR / Hessian tests."""
    from mars.utils import create_structure

    d = 1.09 / np.sqrt(3.0)
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [d, d, d],
            [d, -d, -d],
            [-d, d, -d],
            [-d, -d, d],
        ]
    )
    return create_structure(positions, ["C", "H", "H", "H", "H"])


@pytest.fixture
def tmp_output_dir():
    """Per-test temp directory cleaned up automatically."""
    path = Path(tempfile.mkdtemp(prefix="mars_test_"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def harmonic_xyz(tmp_output_dir, methane):
    """Write a tiny multi-atom XYZ file the CLI can ingest with --potential harmonic."""
    from mars.utils import save_structure

    path = tmp_output_dir / "input.xyz"
    save_structure(str(path), methane, comment="methane test fixture")
    return path


@pytest.fixture
def mars_cli_cmd():
    """Return a list of args for invoking the MARS CLI through the current Python."""
    return [sys.executable, "-m", "mars.cli"]
