"""GUI option forms (every CLI flag, show_if rules) and `mars --config` run configs."""

from __future__ import annotations

import argparse

import pytest

from mars.cli import _inject_config_defaults
from mars.cli._config import parse_config_text
from mars.cli._parser import create_parser
from mars.gui import jobs
from mars.gui.runconfig import (
    config_to_payload,
    config_to_toml,
    export_config,
    parse_config,
    payload_to_config,
)


def _subparsers():
    main = create_parser()
    subs = {None: main}
    for a in main._actions:
        if isinstance(a, argparse._SubParsersAction):
            subs.update(a.choices)
    return subs


def _argv(payload):
    return jobs.build_command(payload, "in.xyz", python="py")["argv"][3:]


# ── schema ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("task", ["conformer", "optimize", "ir", "solvation"])
def test_every_cli_flag_is_in_the_form(task):
    """Each flag of the subparser is a task field, a potential field or a run setting."""
    t = jobs.TASKS[task]
    schema = jobs.get_schema()["tasks"][task]
    in_form = {f["flag"] for f in schema["fields"]}
    in_form |= {f["flag"] for fs in schema["potential_fields"].values() for f in fs}
    skip = jobs.GLOBAL_FLAGS | (jobs._WORKFLOW_FLAGS if t["subcommand"] is None else set())
    missing = []
    for action in _subparsers()[t["subcommand"]]._actions:
        flag = next((o for o in action.option_strings if o.startswith("--")), None)
        if flag and flag not in skip and action.help != argparse.SUPPRESS and flag not in in_form:
            missing.append(flag)
    assert not missing, f"CLI flags missing from the {task} form: {missing}"


def test_main_options_come_first_and_sections_are_unique():
    for task in jobs.get_schema()["tasks"].values():
        titles = [s["title"] for s in task["sections"]]
        assert titles[0] == "Main options"
        assert len(titles) == len(set(titles))
        flags = [f["flag"] for f in task["fields"]]
        assert len(flags) == len(set(flags))


def test_show_if_rules_name_existing_flags():
    schema = jobs.get_schema()["tasks"]
    for key, task in schema.items():
        flags = {f["flag"] for f in task["fields"]}
        for f in task["fields"]:
            for cond in f["show_if"]:
                assert cond["flag"] in flags, (key, f["flag"], cond)


def test_mtd_and_md_tasks_hide_the_full_search_options():
    schema = jobs.get_schema()["tasks"]
    md = {f["flag"] for f in schema["md"]["fields"]}
    assert "--genetic-crossing" not in md and "--kpush" not in md and "--mtd-only" not in md
    assert "--kpush" in {f["flag"] for f in schema["mtd"]["fields"]}


# ── command building ────────────────────────────────────────────────────────


def test_hidden_options_never_reach_the_command():
    argv = _argv({"task": "ir", "options": {"--nvt-time": 9, "--fd-hessian": True}})
    assert "--nvt-time" not in argv and "--fd-hessian" in argv
    argv = _argv({"task": "ir", "options": {"--md": True, "--nvt-time": 9, "--fd-hessian": True}})
    assert argv[argv.index("--nvt-time") + 1] == "9.0" and "--fd-hessian" not in argv


def test_choice_conditions():
    opts = {"--method": "LBFGS", "--fire-dt-max": 0.3}
    assert "--fire-dt-max" not in _argv({"task": "optimize", "options": opts})
    opts["--method"] = "FIRE"
    assert "--fire-dt-max" in _argv({"task": "optimize", "options": opts})
    opts = {"--opt-mode": "none", "--fmax": 0.2, "--layers": "6 12", "--padding": 5}
    argv = _argv({"task": "solvation", "options": opts})
    assert "--fmax" not in argv and "--padding" not in argv
    assert argv[argv.index("--layers") :][:3] == ["--layers", "6", "12"]


def test_optional_value_flag():
    argv = _argv({"task": "solvation", "options": {"--final-relax": True}})
    assert argv[-1] == "--final-relax"
    argv = _argv({"task": "solvation", "options": {"--final-relax": "0.02"}})
    assert argv[-2:] == ["--final-relax", "0.02"]
    assert "--final-relax" not in _argv({"task": "solvation", "options": {"--final-relax": False}})


def test_every_task_command_parses():
    """A form filled with every non-default value still parses with argparse."""
    parser = create_parser()
    for key, task in jobs.get_schema()["tasks"].items():
        argv = jobs.build_command({"task": key}, "in.xyz", python="py")["argv"][3:]
        ns = parser.parse_args(argv) if jobs.TASKS[key]["subcommand"] else None
        if ns is not None:
            assert ns.input == "in.xyz"


# ── config files ───────────────────────────────────────────────────────────

PAYLOAD = {
    "task": "ir",
    "options": {"--md": True, "--nvt-time": 10, "--freq-range": "400 3500", "--window": "bh"},
    "potential": "mace",
    "potential_options": {"--mace-model": "medium", "--mace-foundation": "off"},
    "charge": -1,
    "device": "cpu",
}


def test_config_round_trip_gives_the_same_command():
    _, text = export_config(PAYLOAD, "mol.xyz", version="1.0.1")
    sections = parse_config(text, "run.toml")
    assert sections["ir"]["input"] == "mol.xyz"
    assert sections["ir"]["freq_range"] == [400.0, 3500.0]
    assert "mace_foundation" not in sections["ir"]  # CLI default left out
    back = config_to_payload(sections)
    assert back["payload"]["task"] == "ir" and back["input"] == "mol.xyz"
    assert _argv(back["payload"]) == _argv(PAYLOAD)


def test_gui_config_runs_through_the_cli(tmp_path):
    """`mars ir --config gui.toml` (no positional) sees the input and every option."""
    _, text = export_config(PAYLOAD, "mol.xyz")
    path = tmp_path / "gui.toml"
    path.write_text(text)
    parser = create_parser()
    _, _, argv = _inject_config_defaults(["ir", "--config", str(path)], "ir", parser)
    ns = parser.parse_args(argv)
    assert ns.input == "mol.xyz" and ns.md and ns.nvt_time == 10.0
    assert ns.freq_range == [400.0, 3500.0] and ns.potential == "mace" and ns.charge == -1.0


def test_list_option_in_config_does_not_swallow_the_input(tmp_path):
    path = tmp_path / "s.toml"
    path.write_text("[solvation]\nlayers = [6, 12]\n")
    parser = create_parser()
    for argv in (
        ["solvation", "--config", str(path), "mol.xyz"],
        ["solvation", "mol.xyz", "--config", str(path), "--seed", "7"],
    ):
        _, _, merged = _inject_config_defaults(argv, "solvation", parser)
        ns = parser.parse_args(merged)
        assert ns.input == "mol.xyz" and ns.layers == [6, 12]
    _, _, merged = _inject_config_defaults(
        ["ir", "--config", str(path), "mol.xyz", "--ff-xml", "a.xml", "b.xml"], "ir", parser
    )
    ns = parser.parse_args(merged)
    assert ns.input == "mol.xyz" and ns.ff_xml == ["a.xml", "b.xml"]


def test_values_the_parser_cannot_read_back_are_refused():
    with pytest.raises(jobs.PayloadError):
        export_config({"task": "optimize", "options": {"--log-file": "a #b.log"}})
    _, text = export_config({"task": "optimize", "name": 'run #2 "x"'})
    assert parse_config(text)["gui"]["name"] == "run 2 'x'"


def test_loading_a_config_reports_what_the_form_cannot_hold():
    sections = parse_config_text(
        "[global]\ncharge = 1\n[conformer_search]\nmtd_only = true\nkpush = 10\n"
        'potential = "amber"\nprmtop = "a.prmtop"\nnot_an_option = 3\n'
    )
    res = config_to_payload(sections)
    p = res["payload"]
    assert p["task"] == "mtd" and p["options"] == {"--kpush": 10}
    assert p["potential"] == "amber" and p["charge"] == 1
    assert res["files"] == ["--prmtop"]
    assert "--not-an-option 3" in p["advanced"]
    assert any("a.prmtop" in w for w in res["warnings"])


def test_config_has_a_gui_section_the_cli_ignores():
    sections = payload_to_config(dict(PAYLOAD, name="test run", advanced="--debug"))
    assert sections["gui"] == {
        "task": "ir",
        "device": "cpu",
        "advanced": "--debug",
        "name": "test run",
    }
    text = config_to_toml(sections)
    assert text.startswith("# MARS run configuration") and "mars ir --config" in text
    assert parse_config(text) == sections
