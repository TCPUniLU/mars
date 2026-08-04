"""Tests for mars/cli/_config.py — TOML config loader and argv merge."""

from __future__ import annotations

import pytest

from mars.cli._config import (
    _parse_value,
    parse_atom_indices,
    load_config,
    config_section_to_argv,
    build_config_argv,
    get_constraints_from_config,
)

# ── _parse_value ────────────────────────────────────────────────────────────


class TestParseValue:
    def test_int(self):
        assert _parse_value("42") == 42
        assert _parse_value("-5") == -5

    def test_float(self):
        assert _parse_value("0.5") == 0.5
        assert _parse_value("-1.25") == -1.25
        assert _parse_value("1e-3") == 1e-3

    def test_bool(self):
        assert _parse_value("true") is True
        assert _parse_value("false") is False

    def test_quoted_string(self):
        assert _parse_value('"hello"') == "hello"
        assert _parse_value("'hi'") == "hi"

    def test_bare_string(self):
        assert _parse_value("thorough") == "thorough"

    def test_list_of_ints(self):
        assert _parse_value("[0, 1, 2]") == [0, 1, 2]

    def test_list_of_floats(self):
        assert _parse_value("[1.0, 2.5]") == [1.0, 2.5]

    def test_empty_list(self):
        assert _parse_value("[]") == []

    def test_list_of_strings(self):
        assert _parse_value('["a", "b"]') == ["a", "b"]


# ── parse_atom_indices ──────────────────────────────────────────────────────


class TestParseAtomIndices:
    def test_list_input(self):
        assert parse_atom_indices([3, 1, 2]) == [1, 2, 3]

    def test_int_input(self):
        assert parse_atom_indices(5) == [5]

    def test_csv_string(self):
        assert parse_atom_indices("0,1,2,5") == [0, 1, 2, 5]

    def test_range_string(self):
        assert parse_atom_indices("0-5") == [0, 1, 2, 3, 4, 5]

    def test_mixed_csv_and_range(self):
        assert parse_atom_indices("0-3,7,10-12") == [0, 1, 2, 3, 7, 10, 11, 12]

    def test_duplicates_collapsed(self):
        assert parse_atom_indices("0,1,1,2,0") == [0, 1, 2]

    def test_sorted_output(self):
        assert parse_atom_indices("5,2,8,1") == [1, 2, 5, 8]


# ── load_config ─────────────────────────────────────────────────────────────


class TestLoadConfig:
    def test_basic_sections(self, tmp_output_dir):
        path = tmp_output_dir / "cfg.toml"
        path.write_text(
            "[conformer_search]\n"
            'mode = "thorough"\n'
            "temperature = 400.0\n"
            "nci = true\n"
            "\n"
            "[ir]\n"
            "plot = true\n"
        )
        s = load_config(str(path))
        assert s["conformer_search"]["mode"] == "thorough"
        assert s["conformer_search"]["temperature"] == 400.0
        assert s["conformer_search"]["nci"] is True
        assert s["ir"]["plot"] is True

    def test_comments_and_blanks_ignored(self, tmp_output_dir):
        path = tmp_output_dir / "cfg.toml"
        path.write_text(
            "# top-level comment\n"
            "\n"
            "[optimize]\n"
            "# inline section comment\n"
            "fmax = 0.001  # the convergence threshold\n"
            'method = "FIRE"\n'
        )
        s = load_config(str(path))
        assert s["optimize"]["fmax"] == 0.001
        assert s["optimize"]["method"] == "FIRE"

    def test_list_values(self, tmp_output_dir):
        path = tmp_output_dir / "cfg.toml"
        path.write_text("[constraints]\n" "atoms = [0, 1, 2]\n" "k = 100.0\n")
        s = load_config(str(path))
        assert s["constraints"]["atoms"] == [0, 1, 2]
        assert s["constraints"]["k"] == 100.0

    def test_key_outside_section_raises(self, tmp_output_dir):
        path = tmp_output_dir / "cfg.toml"
        path.write_text("orphan = 1\n")
        with pytest.raises(ValueError, match="outside any section"):
            load_config(str(path))

    def test_malformed_line_raises(self, tmp_output_dir):
        path = tmp_output_dir / "cfg.toml"
        path.write_text("[bad]\nthis is not a key value line\n")
        with pytest.raises(ValueError, match="cannot parse line"):
            load_config(str(path))

    def test_empty_file_returns_empty_dict(self, tmp_output_dir):
        path = tmp_output_dir / "cfg.toml"
        path.write_text("")
        assert load_config(str(path)) == {}


# ── config_section_to_argv ──────────────────────────────────────────────────


class TestConfigSectionToArgv:
    def test_basic_pairs(self):
        argv = config_section_to_argv({"mode": "thorough", "temperature": 400.0})
        # Order is preserved by dict insertion order
        assert "--mode" in argv
        assert "thorough" in argv
        assert "--temperature" in argv
        assert "400.0" in argv

    def test_bool_true_becomes_bare_flag(self):
        argv = config_section_to_argv({"plot": True, "debug": False})
        assert argv == ["--plot"]

    def test_underscores_become_dashes(self):
        argv = config_section_to_argv({"mtd_time": 5.0, "log_level": "DEBUG"})
        assert "--mtd-time" in argv
        assert "--log-level" in argv
        # And values follow
        assert "5.0" in argv
        assert "DEBUG" in argv

    def test_input_key_is_dropped(self):
        argv = config_section_to_argv({"input": "mol.xyz", "fmax": 0.001})
        assert "mol.xyz" not in argv
        assert "--input" not in argv
        assert "--fmax" in argv

    def test_list_value_expanded_as_multiple_tokens(self):
        """argparse nargs='+' flags (e.g. --layers, --freq-range) need
        each list element as a separate token, not str([2, 4])."""
        argv = config_section_to_argv({"layers": [2, 4, 8]})
        assert argv == ["--layers", "2", "4", "8"]

    def test_two_value_list_for_freq_range(self):
        argv = config_section_to_argv({"freq_range": [400, 3500]})
        assert argv == ["--freq-range", "400", "3500"]

    def test_tuple_value_also_expanded(self):
        argv = config_section_to_argv({"layers": (3, 6)})
        assert argv == ["--layers", "3", "6"]


# ── build_config_argv ───────────────────────────────────────────────────────


class TestBuildConfigArgv:
    def test_returns_command_section_args(self, tmp_output_dir):
        path = tmp_output_dir / "cfg.toml"
        path.write_text("[optimize]\n" "fmax = 0.001\n" 'method = "FIRE"\n')
        input_from_cfg, argv = build_config_argv(str(path), "optimize")
        assert input_from_cfg is None
        assert "--fmax" in argv and "0.001" in argv
        assert "--method" in argv and "FIRE" in argv

    def test_default_command_is_conformer_search(self, tmp_output_dir):
        path = tmp_output_dir / "cfg.toml"
        path.write_text("[conformer_search]\n" 'mode = "quick"\n')
        _, argv = build_config_argv(str(path), None)
        assert "--mode" in argv and "quick" in argv

    def test_input_key_returned_separately(self, tmp_output_dir):
        path = tmp_output_dir / "cfg.toml"
        path.write_text("[conformer_search]\n" 'input = "mol.xyz"\n' 'mode = "normal"\n')
        input_from_cfg, argv = build_config_argv(str(path), None)
        assert input_from_cfg == "mol.xyz"
        assert "mol.xyz" not in argv

    def test_global_section_merged_then_overridden(self, tmp_output_dir):
        path = tmp_output_dir / "cfg.toml"
        path.write_text(
            "[global]\n"
            "temperature = 300.0\n"
            "charge = -1.0\n"
            "\n"
            "[ir]\n"
            "temperature = 400.0\n"  # overrides global
        )
        _, argv = build_config_argv(str(path), "ir")
        # global charge survives
        assert "--charge" in argv and "-1.0" in argv
        # ir temperature overrides global
        idx = argv.index("--temperature")
        assert argv[idx + 1] == "400.0"

    def test_missing_command_section_uses_only_global(self, tmp_output_dir):
        path = tmp_output_dir / "cfg.toml"
        path.write_text("[global]\n" 'log_level = "DEBUG"\n')
        _, argv = build_config_argv(str(path), "ir")
        assert "--log-level" in argv and "DEBUG" in argv

    def test_missing_file_raises(self, tmp_output_dir):
        with pytest.raises(FileNotFoundError):
            build_config_argv(str(tmp_output_dir / "missing.toml"), None)


# ── get_constraints_from_config ─────────────────────────────────────────────


class TestGetConstraintsFromConfig:
    def test_full_constraints_section(self, tmp_output_dir):
        path = tmp_output_dir / "cfg.toml"
        path.write_text(
            "[constraints]\n" "atoms = [0, 1, 5]\n" "k = 50.0\n" 'apply_to = "optimization"\n'
        )
        c = get_constraints_from_config(str(path))
        assert c == {"atoms": [0, 1, 5], "k": 50.0, "apply_to": "optimization"}

    def test_defaults_when_only_atoms_given(self, tmp_output_dir):
        path = tmp_output_dir / "cfg.toml"
        path.write_text("[constraints]\natoms = [2, 3]\n")
        c = get_constraints_from_config(str(path))
        assert c["atoms"] == [2, 3]
        assert c["k"] == 100.0
        assert c["apply_to"] == "all"

    def test_returns_none_without_constraints_section(self, tmp_output_dir):
        path = tmp_output_dir / "cfg.toml"
        path.write_text('[conformer_search]\nmode = "quick"\n')
        assert get_constraints_from_config(str(path)) is None

    def test_returns_none_without_atoms_key(self, tmp_output_dir):
        path = tmp_output_dir / "cfg.toml"
        path.write_text("[constraints]\nk = 100.0\n")
        assert get_constraints_from_config(str(path)) is None

    def test_returns_none_for_missing_path(self):
        assert get_constraints_from_config(None) is None
        # Missing file is swallowed by the helper (returns None, not raises)
        assert get_constraints_from_config("/tmp/does_not_exist_xyzzy.toml") is None

    def test_string_atom_specification(self, tmp_output_dir):
        # parse_atom_indices accepts CSV/range strings too — covered when the
        # user writes atoms as a quoted string instead of a TOML list.
        path = tmp_output_dir / "cfg.toml"
        path.write_text("[constraints]\n" 'atoms = "0-3,7"\n')
        c = get_constraints_from_config(str(path))
        assert c["atoms"] == [0, 1, 2, 3, 7]
