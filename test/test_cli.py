"""End-to-end CLI smoke tests.

Each test invokes ``python -m mars.cli`` as a subprocess so that argument
parsing, dispatcher wiring, and the ``mars.solvents`` package data path are
all exercised. We use ``--potential harmonic`` everywhere so no ML weights
or external dependencies are required.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cli


def _run(cmd, cwd, timeout=300):
    """Run a subprocess capturing stdout/stderr."""
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    return proc


# ── --help on every subcommand ─────────────────────────────────────────────


@pytest.mark.parametrize("subcmd", [[], ["ir"], ["optimize"], ["solvation"]])
def test_cli_help_exit_zero(subcmd, mars_cli_cmd, tmp_output_dir):
    proc = _run(mars_cli_cmd + subcmd + ["--help"], cwd=tmp_output_dir, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "usage" in proc.stdout.lower()


# ── optimize ───────────────────────────────────────────────────────────────


def test_cli_optimize_minimize_harmonic(harmonic_xyz, mars_cli_cmd, tmp_output_dir):
    out = tmp_output_dir / "opt.xyz"
    proc = _run(
        mars_cli_cmd
        + [
            "optimize",
            str(harmonic_xyz),
            "--potential",
            "harmonic",
            "--method",
            "LBFGS",
            "--maxiter",
            "20",
            "--fmax",
            "0.5",
            "-o",
            str(out),
        ],
        cwd=tmp_output_dir,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    assert "Traceback" not in proc.stderr


def test_cli_optimize_task_stationary(harmonic_xyz, mars_cli_cmd, tmp_output_dir):
    """The new --task stationary flag must reach the SP code path."""
    out = tmp_output_dir / "sp.xyz"
    proc = _run(
        mars_cli_cmd
        + [
            "optimize",
            str(harmonic_xyz),
            "--potential",
            "harmonic",
            "--task",
            "stationary",
            "--maxiter",
            "20",
            "-o",
            str(out),
        ],
        cwd=tmp_output_dir,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()


def test_setup_precision_refuses_internal_without_float64():
    """Unit-level guard, so the requirement is covered even where the full CLI
    cannot be imported. --coords internal in float32 must exit rather than run:
    the Pulay back-transformation residual (~1e-4 rad) maps to a spurious force
    of ~4e-3 eV/A, above the tightest convergence threshold, and the failure is
    silent rather than a crash."""
    import types

    from mars.cli._common import setup_precision

    def _args(**kw):
        d = dict(coords="cartesian", coords_coarse="cartesian", float64=False, cpu=False)
        d.update(kw)
        return types.SimpleNamespace(**d)

    with pytest.raises(SystemExit) as exc:
        setup_precision(_args(coords="internal"))
    assert exc.value.code == 1

    # Cartesian in float32 is unaffected, and internal with --float64 is fine.
    setup_precision(_args())


def test_cli_optimize_internal_requires_float64(harmonic_xyz, mars_cli_cmd, tmp_output_dir):
    """--coords internal must refuse to run in float32 rather than silently
    oscillating: the back-transformation residual alone exceeds the tightest
    convergence threshold."""
    proc = _run(
        mars_cli_cmd
        + [
            "optimize",
            str(harmonic_xyz),
            "--potential",
            "harmonic",
            "--coords",
            "internal",
            "--maxiter",
            "5",
        ],
        cwd=tmp_output_dir,
    )
    assert proc.returncode != 0
    out = proc.stdout + proc.stderr
    assert "--float64" in out
    assert "Traceback" not in proc.stderr


# ── ir (Hessian-only, --no-optimize) ───────────────────────────────────────


def test_cli_ir_hessian_no_optimize(harmonic_xyz, mars_cli_cmd, tmp_output_dir):
    out = tmp_output_dir / "ir.dat"
    proc = _run(
        mars_cli_cmd
        + [
            "ir",
            str(harmonic_xyz),
            "--potential",
            "harmonic",
            "--no-optimize",
            "--no-analysis",
            "-o",
            str(out),
        ],
        cwd=tmp_output_dir,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    assert "Traceback" not in proc.stderr


# ── solvation ──────────────────────────────────────────────────────────────


def test_cli_solvation_water_no_opt(harmonic_xyz, mars_cli_cmd, tmp_output_dir):
    out = tmp_output_dir / "solvated.xyz"
    proc = _run(
        mars_cli_cmd
        + [
            "solvation",
            str(harmonic_xyz),
            "--solvent",
            "water",
            "--layers",
            "2",
            "--opt-mode",
            "none",
            "-o",
            str(out),
        ],
        cwd=tmp_output_dir,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()


def test_cli_solvation_aliased_solvent(harmonic_xyz, mars_cli_cmd, tmp_output_dir):
    """Solvent aliases (e.g. 'EtOH') must resolve via the new mars.solvents library."""
    out = tmp_output_dir / "etoh.xyz"
    proc = _run(
        mars_cli_cmd
        + [
            "solvation",
            str(harmonic_xyz),
            "--solvent",
            "EtOH",
            "--layers",
            "2",
            "--opt-mode",
            "none",
            "-o",
            str(out),
        ],
        cwd=tmp_output_dir,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()


def test_cli_solvation_unknown_solvent_errors(harmonic_xyz, mars_cli_cmd, tmp_output_dir):
    proc = _run(
        mars_cli_cmd
        + [
            "solvation",
            str(harmonic_xyz),
            "--solvent",
            "cromulent_solvent",
            "--layers",
            "2",
        ],
        cwd=tmp_output_dir,
        timeout=60,
    )
    assert proc.returncode != 0
    assert "cromulent_solvent" in (proc.stderr + proc.stdout).lower()


# ── --config FILE: end-to-end smoke tests for every subcommand ─────────────


def _write(path, content):
    Path(path).write_text(content)
    return Path(path)


def test_cli_config_ir(harmonic_xyz, mars_cli_cmd, tmp_output_dir):
    cfg = _write(
        tmp_output_dir / "ir.toml",
        "[ir]\nplot = false\nbroadening = 12.0\nno_optimize = true\n"
        "no_analysis = true\nfreq_range = [400, 3500]\n",
    )
    out = tmp_output_dir / "ir.dat"
    proc = _run(
        mars_cli_cmd
        + [
            "ir",
            str(harmonic_xyz),
            "--config",
            str(cfg),
            "--potential",
            "harmonic",
            "-o",
            str(out),
        ],
        cwd=tmp_output_dir,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()


def test_cli_config_optimize_with_constraints(harmonic_xyz, mars_cli_cmd, tmp_output_dir):
    cfg = _write(
        tmp_output_dir / "opt.toml",
        '[optimize]\nmethod = "LBFGS"\nfmax = 0.1\nmaxiter = 15\n\n'
        '[constraints]\natoms = [0]\nk = 50.0\napply_to = "optimization"\n',
    )
    out = tmp_output_dir / "opt.xyz"
    proc = _run(
        mars_cli_cmd
        + [
            "optimize",
            str(harmonic_xyz),
            "--config",
            str(cfg),
            "--potential",
            "harmonic",
            "-o",
            str(out),
        ],
        cwd=tmp_output_dir,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    assert "atoms frozen" in (proc.stdout + proc.stderr)


def test_cli_config_solvation_with_list_layers(harmonic_xyz, mars_cli_cmd, tmp_output_dir):
    """layers = [2, 4] must expand to ``--layers 2 4`` (multi-token argv)."""
    cfg = _write(
        tmp_output_dir / "solv.toml",
        '[solvation]\nsolvent = "water"\nlayers = [2, 4]\n' 'opt_mode = "none"\nseed = 7\n',
    )
    out = tmp_output_dir / "solv.xyz"
    proc = _run(
        mars_cli_cmd
        + [
            "solvation",
            str(harmonic_xyz),
            "--config",
            str(cfg),
            "--potential",
            "harmonic",
            "-o",
            str(out),
        ],
        cwd=tmp_output_dir,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    # 5 (methane) + 3 (water) * (2 + 4) = 23 atoms
    n_atoms = int(out.read_text().splitlines()[0])
    assert n_atoms == 23


def test_cli_config_cli_flag_overrides_config(harmonic_xyz, mars_cli_cmd, tmp_output_dir):
    """An explicit CLI flag wins over the value in the config file."""
    cfg = _write(
        tmp_output_dir / "opt.toml", '[optimize]\nmethod = "LBFGS"\nfmax = 0.001\nmaxiter = 5\n'
    )
    out = tmp_output_dir / "opt.xyz"
    proc = _run(
        mars_cli_cmd
        + [
            "optimize",
            str(harmonic_xyz),
            "--config",
            str(cfg),
            "--potential",
            "harmonic",
            "--fmax",
            "0.5",  # overrides config's 0.001
            "-o",
            str(out),
        ],
        cwd=tmp_output_dir,
    )
    assert proc.returncode == 0, proc.stderr
    combined = proc.stdout + proc.stderr
    # The "starting optimization" line echoes the effective fmax in eV/Å.
    # The CLI --fmax 0.5 must override the config's 0.001.
    assert "fmax=0.5 eV/Å" in combined


def test_cli_config_global_section_applies(harmonic_xyz, mars_cli_cmd, tmp_output_dir):
    """[global] values are merged before the subcommand-specific section."""
    cfg = _write(
        tmp_output_dir / "g.toml",
        '[global]\nlog_level = "DEBUG"\n\n'
        '[optimize]\nmethod = "LBFGS"\nmaxiter = 5\nfmax = 0.5\n',
    )
    out = tmp_output_dir / "opt.xyz"
    proc = _run(
        mars_cli_cmd
        + [
            "optimize",
            str(harmonic_xyz),
            "--config",
            str(cfg),
            "--potential",
            "harmonic",
            "-o",
            str(out),
        ],
        cwd=tmp_output_dir,
    )
    assert proc.returncode == 0, proc.stderr


def test_cli_config_missing_file_errors(harmonic_xyz, mars_cli_cmd, tmp_output_dir):
    proc = _run(
        mars_cli_cmd
        + [
            "optimize",
            str(harmonic_xyz),
            "--config",
            str(tmp_output_dir / "does_not_exist.toml"),
            "--potential",
            "harmonic",
        ],
        cwd=tmp_output_dir,
        timeout=60,
    )
    assert proc.returncode != 0
    assert "config file not found" in (proc.stderr + proc.stdout).lower()


def test_cli_config_equals_syntax(harmonic_xyz, mars_cli_cmd, tmp_output_dir):
    """--config=FILE (single token) is accepted the same as --config FILE."""
    cfg = _write(
        tmp_output_dir / "opt.toml", '[optimize]\nmethod = "LBFGS"\nfmax = 0.5\nmaxiter = 5\n'
    )
    out = tmp_output_dir / "opt.xyz"
    proc = _run(
        mars_cli_cmd
        + [
            "optimize",
            str(harmonic_xyz),
            f"--config={cfg}",
            "--potential",
            "harmonic",
            "-o",
            str(out),
        ],
        cwd=tmp_output_dir,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()


def test_cli_config_input_from_config(harmonic_xyz, mars_cli_cmd, tmp_output_dir):
    """When no input is given on the CLI, the conformer-search default mode
    falls back to the ``input`` key in the [conformer_search] section."""
    cfg = _write(
        tmp_output_dir / "c.toml",
        f'[conformer_search]\ninput = "{harmonic_xyz}"\n'
        "mtd_cycles = 1\nmtd_time = 0.1\nmtd_dt = 0.5\n"
        "rotamer_structures = 1\nrotamer_time = 0.1\n"
        "no_topology = true\nno_trial = true\n",
    )
    proc = _run(
        mars_cli_cmd
        + [
            "--config",
            str(cfg),
            "--potential",
            "harmonic",
        ],
        cwd=tmp_output_dir,
        timeout=120,
    )
    # The harmonic potential breaks deep inside the MTD loop (jaxlib neighbor-
    # list type mismatch — pre-existing, see the deleted test_integration.py).
    # All we need to assert here is that the config wiring fed the input file
    # all the way to the loader before that internal failure.
    assert "Loaded structure with 5 atoms" in (proc.stdout + proc.stderr)
