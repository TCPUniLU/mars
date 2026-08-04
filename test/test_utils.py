"""Tests for mars/utils.py — periodic table, structure I/O, bond detection, helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from mars import utils


class TestPeriodicTable:
    def test_array_lengths_match(self):
        n = 119
        assert len(utils.ELEMENT_SYMBOLS) == n
        assert len(utils.ELEMENT_NAMES) == n
        assert len(utils.ATOMIC_MASSES) == n
        assert len(utils.COVALENT_RADII) == n
        assert len(utils.VDW_RADII) == n
        assert len(utils.ELECTRONEGATIVITY_PAULING) == n
        assert len(utils.ELECTRONEGATIVITY_ALLRED_ROCHOW) == n
        assert len(utils.IONIZATION_ENERGY_FIRST) == n
        assert len(utils.ELECTRON_AFFINITY) == n
        assert len(utils.PERIODS) == n
        assert len(utils.GROUPS) == n
        assert len(utils.VALENCE_ELECTRONS) == n
        assert len(utils.CPK_COLORS) == n

    def test_known_masses(self):
        # IUPAC 2021 reference values to 1 decimal
        assert utils.get_atomic_mass("H") == pytest.approx(1.008, abs=0.01)
        assert utils.get_atomic_mass("C") == pytest.approx(12.011, abs=0.01)
        assert utils.get_atomic_mass("N") == pytest.approx(14.007, abs=0.01)
        assert utils.get_atomic_mass("O") == pytest.approx(15.999, abs=0.01)
        assert utils.get_atomic_mass("Fe") == pytest.approx(55.845, abs=0.01)

    def test_symbol_number_roundtrip(self):
        for sym in ["H", "He", "C", "Fe", "Au", "U", "Og"]:
            z = utils.symbol_to_number(sym)
            assert utils.number_to_symbol(z) == sym

    def test_unknown_inputs_are_safe(self):
        assert utils.symbol_to_number("Zz") == 0
        assert utils.number_to_symbol(999) == "X"
        assert utils.get_atomic_mass("Zz") == 0.0
        assert utils.get_period(999) == 0
        assert utils.get_group(999) == 0

    def test_element_names(self):
        assert utils.number_to_name(1) == "Hydrogen"
        assert utils.number_to_name(6) == "Carbon"
        assert utils.number_to_name(118) == "Oganesson"

    def test_radii_known_values(self):
        assert utils.get_covalent_radius("H") == pytest.approx(0.31, abs=0.005)
        assert utils.get_covalent_radius("C") == pytest.approx(0.76, abs=0.005)
        assert utils.get_vdw_radius("H") == pytest.approx(1.20, abs=0.005)
        assert utils.get_vdw_radius("C") == pytest.approx(1.70, abs=0.005)

    def test_vdw_default_for_unknown(self):
        assert utils.get_vdw_radius("Zz") == pytest.approx(2.0)

    def test_electronegativity_pauling(self):
        # Carbon: 2.55, Nitrogen: 3.04, Oxygen: 3.44, Fluorine: 3.98
        assert utils.get_electronegativity("C") == pytest.approx(2.55, abs=0.01)
        assert utils.get_electronegativity("N") == pytest.approx(3.04, abs=0.01)
        assert utils.get_electronegativity("O") == pytest.approx(3.44, abs=0.01)
        assert utils.get_electronegativity("F") == pytest.approx(3.98, abs=0.01)

    def test_electronegativity_allred(self):
        assert utils.get_electronegativity("F", scale="allred-rochow") == pytest.approx(
            4.10, abs=0.05
        )
        assert utils.get_electronegativity("C", scale="allred") == pytest.approx(2.50, abs=0.05)

    def test_electronegativity_invalid_scale(self):
        with pytest.raises(ValueError):
            utils.get_electronegativity("C", scale="bogus")

    def test_ionization_energy(self):
        # H: 13.598 eV, He: 24.587, C: 11.260
        assert utils.get_ionization_energy("H") == pytest.approx(13.598, abs=0.01)
        assert utils.get_ionization_energy("He") == pytest.approx(24.587, abs=0.01)

    def test_electron_affinity(self):
        # Cl has the largest EA among the halogens
        assert utils.get_electron_affinity("Cl") == pytest.approx(3.61, abs=0.05)
        assert np.isnan(utils.get_electron_affinity("He"))

    def test_period_group(self):
        assert utils.get_period("H") == 1
        assert utils.get_period("Li") == 2
        assert utils.get_period("Fe") == 4
        assert utils.get_group("H") == 1
        assert utils.get_group("He") == 18
        assert utils.get_group("Cl") == 17
        assert utils.get_group("Fe") == 8
        # Lanthanides and actinides excluded from standard 18 groups
        assert utils.get_group("Eu") == 0
        assert utils.get_group("U") == 0

    def test_valence_electrons(self):
        assert utils.get_valence_electrons("H") == 1
        assert utils.get_valence_electrons("C") == 4
        assert utils.get_valence_electrons("O") == 6
        assert utils.get_valence_electrons("Ne") == 8

    def test_cpk_color(self):
        c_rgb = utils.get_cpk_color("C")
        assert len(c_rgb) == 3 and all(0 <= v <= 255 for v in c_rgb)
        norm = utils.get_cpk_color("C", normalized=True)
        assert all(0.0 <= v <= 1.0 for v in norm)
        # Hydrogen is white in CPK convention
        assert utils.get_cpk_color("H") == (255, 255, 255)


class TestSymbolsToNumbers:
    def test_basic(self):
        out = utils.symbols_to_numbers(["C", "H", "O", "N"])
        assert out.tolist() == [6, 1, 8, 7]

    def test_unknown_becomes_zero(self):
        out = utils.symbols_to_numbers(["Zz", "C"])
        assert out.tolist() == [0, 6]


class TestBondDetection:
    def test_h2(self):
        positions = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
        bonds, dists = utils.detect_bonds(positions, np.array([1, 1]))
        assert bonds == [(0, 1)]
        assert dists[0] == pytest.approx(0.74, abs=1e-6)

    def test_no_bond_when_far_apart(self):
        positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        bonds, dists = utils.detect_bonds(positions, np.array([6, 6]))
        assert bonds == []

    def test_three_atom_chain(self):
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.10, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ]
        )
        bonds, _ = utils.detect_bonds(positions, np.array([6, 6, 6]))
        assert bonds == [(0, 1)]


class TestStructureRoundTrip:
    def test_save_and_load_xyz(self, methane, tmp_output_dir):
        path = str(tmp_output_dir / "round.xyz")
        utils.save_structure(path, methane, comment="test")
        loaded = utils.load_structure(path)
        assert list(loaded["symbols"]) == list(methane["symbols"])
        np.testing.assert_allclose(
            np.array(loaded["positions"]), np.array(methane["positions"]), atol=1e-5
        )

    def test_create_structure_infers_numbers(self):
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        s = utils.create_structure(positions, ["C", "H"])
        assert list(s["symbols"]) == ["C", "H"]
        assert list(np.asarray(s["numbers"])) == [6, 1]

    def test_save_and_load_ensemble(self, methane, tmp_output_dir):
        ensemble = [(methane, -1.234), (methane, -1.235)]
        path = str(tmp_output_dir / "ens.xyz")
        utils.save_ensemble(path, ensemble)
        loaded = utils.load_ensemble(path)
        assert len(loaded) == 2


class TestCheckCharge:
    def test_even_electron_count_ok(self):
        # Methane: 6 + 4 = 10 electrons (closed shell)
        utils.check_charge({"numbers": np.array([6, 1, 1, 1, 1])}, charge=0)

    def test_odd_electron_count_raises(self):
        # CH3 radical: 6 + 3 = 9 electrons (odd)
        with pytest.raises(ValueError):
            utils.check_charge({"numbers": np.array([6, 1, 1, 1])}, charge=0)

    def test_charge_corrects_count(self):
        # Methyl cation (CH3+): 9 - 1 = 8 → even
        utils.check_charge({"numbers": np.array([6, 1, 1, 1])}, charge=1)


class TestArrayHelpers:
    def test_ensure_jax_array_passthrough(self):
        a = jnp.array([1.0, 2.0, 3.0])
        assert utils.ensure_jax_array(a) is a

    def test_ensure_jax_array_from_numpy(self):
        a = np.array([1.0, 2.0])
        out = utils.ensure_jax_array(a)
        assert isinstance(out, jnp.ndarray)

    def test_to_numpy(self):
        a = jnp.array([1.0, 2.0])
        out = utils.to_numpy(a)
        assert isinstance(out, np.ndarray)
