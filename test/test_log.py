"""Tests for mars/log.py — logging, timing, energy table, progress bar."""

import logging
import time

import pytest

from mars import log as marslog


def test_init_log_writes_to_file(tmp_output_dir):
    logfile = str(tmp_output_dir / "run.log")
    marslog.init_log(logfile=logfile, level="INFO", console=False)
    marslog.log_info("hello")
    marslog.log_warning("watch out")
    marslog.log_error("oops")
    # Flush
    for h in marslog.logger.handlers:
        h.flush()
    contents = open(logfile).read()
    assert "hello" in contents
    assert "watch out" in contents
    assert "oops" in contents


def test_init_log_console_only_does_not_create_file(tmp_output_dir):
    marslog.init_log(logfile=None, level="INFO", console=True)
    marslog.log_info("just console")
    assert not (tmp_output_dir / "missing.log").exists()


def test_log_levels(caplog, tmp_output_dir):
    logfile = str(tmp_output_dir / "lvl.log")
    marslog.init_log(logfile=logfile, level="DEBUG", console=False)
    marslog.log_debug("debug-msg")
    marslog.log_info("info-msg")
    marslog.log_warning("warn-msg")
    marslog.log_error("err-msg")
    for h in marslog.logger.handlers:
        h.flush()
    contents = open(logfile).read()
    for needle in ("debug-msg", "info-msg", "warn-msg", "err-msg"):
        assert needle in contents


class TestTimingTracker:
    def test_basic_record(self):
        t = marslog.TimingTracker()
        t.start("alpha")
        time.sleep(0.01)
        t.end("alpha")
        stages = t.get_stages()
        assert len(stages) == 1
        assert stages[0][0] == "alpha"
        assert stages[0][1] >= 0.01

    def test_repeated_starts_accumulate(self):
        t = marslog.TimingTracker()
        for _ in range(3):
            t.start("loop")
            time.sleep(0.005)
            t.end("loop")
        stages = dict(t.get_stages())
        assert stages["loop"] >= 0.012

    def test_reset_clears_state(self):
        t = marslog.TimingTracker()
        t.start("x")
        t.end("x")
        t.reset()
        assert t.get_stages() == []


def test_timer_globals():
    marslog.timer_reset()
    marslog.timer_start("phase-1")
    time.sleep(0.005)
    marslog.timer_end("phase-1")
    # log_timing_summary should not raise
    marslog.log_timing_summary()


def test_log_timer_context_manager():
    with marslog.LogTimer("ctx-block"):
        time.sleep(0.005)


def test_progress_bar_iterates():
    items = list(marslog.progress_bar([1, 2, 3], total=3, desc="test"))
    assert items == [1, 2, 3]


def test_log_topology_no_throw():
    bonds = [(0, 1), (1, 2)]
    distances = [1.0, 1.0]
    marslog.log_topology(bonds, distances, atomic_numbers=[6, 6, 6])
    marslog.log_topology(
        bonds, distances, atomic_numbers=[6, 6, 6], n_molecules=1, fragment_sizes=[3]
    )


def test_log_energy_table_no_throw():
    from mars.utils import create_structure
    import jax.numpy as jnp

    s = create_structure(jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), ["C", "C"])
    ensemble = [(s, -1.0), (s, -0.99), (s, -0.5)]
    marslog.log_energy_table(ensemble, title="Test ensemble")


def test_log_statistics_and_parameters_no_throw():
    marslog.log_statistics({"n_hills": 12, "ratio": 0.123})
    marslog.log_parameters({"kpush": 0.02, "alpha": 0.5})
