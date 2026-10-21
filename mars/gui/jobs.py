"""
Job model for the MARS web GUI: task schema, command building, persistence, and execution.

Every GUI run is an ordinary MARS CLI invocation (``python -m mars.cli ...``)
executed as a subprocess inside its own job directory::

    <workdir>/
        jobs.json                 # persistent job index (survives restarts)
        uploads/<upload_id>/      # files uploaded through the GUI
        jobs/<job_id>/
            input.xyz             # copied input structure (+ force-field files)
            job.json              # job record (payload, argv, status, timings)
            command.sh            # the exact, re-runnable shell command
            run.log               # merged stdout / stderr of the run
            ...                   # everything MARS writes (cwd = job dir)

The option forms shown in the browser are generated from the real
:func:`mars.cli._parser.create_parser` (defaults, choices and help strings are
read from argparse), so the GUI never drifts from the CLI.  Every flag of a
task's subparser is offered: a curated set under "Main options", the rest in
sections named after the argparse groups.  ``SHOW_IF`` rules hide options that
do not apply to the choices already made (e.g. FIRE parameters unless
``--method FIRE``), and hidden options never reach the command line.
"""

import argparse
import json
import os
import queue
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import structures

# ---------------------------------------------------------------------------
# Task / potential schema
# ---------------------------------------------------------------------------

#: GUI tasks. ``subcommand`` is the ``mars`` subcommand (``None`` = default
#: conformational-search mode); ``fixed`` flags are always appended; each field
#: is a flag of that subparser, optionally with GUI overrides (label, default).
TASKS: Dict[str, Dict] = {
    "conformer": {
        "label": "Conformer search",
        "subcommand": None,
        "fixed": [],
        "description": "Full MARS conformational sampling (MTD + rotamer MD + refinement).",
        "fields": [
            {"flag": "--mode", "label": "Search mode", "default": "normal"},
            {"flag": "--temperature", "label": "Temperature (K)"},
            {"flag": "--ewin", "label": "Energy window (kcal/mol)"},
            {"flag": "--method", "label": "Minimizer"},
            {"flag": "--fmax", "label": "Final fmax (eV/Å)"},
            {"flag": "--genetic-crossing", "label": "Genetic crossing"},
            {"flag": "--nci", "label": "NCI mode (confinement)"},
            {"flag": "--nci-buffer", "label": "NCI buffer (Å)"},
            {"flag": "--save-trajectory", "label": "Save MD trajectories"},
            {"flag": "--not-parallel", "label": "Disable vmap parallelism"},
        ],
    },
    "optimize": {
        "label": "Geometry optimisation",
        "subcommand": "optimize",
        "fixed": [],
        "description": "Local minimisation or stationary-point search (mars optimize).",
        "fields": [
            {"flag": "--task", "label": "Task"},
            {"flag": "--method", "label": "Minimizer"},
            {"flag": "--fmax", "label": "fmax (eV/Å)"},
            {"flag": "--maxiter", "label": "Max iterations"},
            {"flag": "--max-stepsize", "label": "Max step (Å)"},
            {"flag": "--coords", "label": "Coordinates"},
            {"flag": "--all-conformers", "label": "All conformers (multi-frame input)"},
        ],
    },
    "ir": {
        "label": "IR spectrum",
        "subcommand": "ir",
        "fixed": [],
        "description": "Harmonic (Hessian) or MD-based IR spectrum (mars ir).",
        "fields": [
            {"flag": "--fd-hessian", "label": "Finite-difference Hessian"},
            {"flag": "--md", "label": "MD-based IR (dipole ACF)"},
            {"flag": "--no-optimize", "label": "Skip pre-optimisation"},
            {"flag": "--fmax", "label": "Pre-opt fmax (eV/Å)"},
            {"flag": "--temperature", "label": "Temperature (K)"},
            {"flag": "--broadening", "label": "Lorentzian width (cm⁻¹)"},
            {"flag": "--freq-range", "label": "Frequency range (cm⁻¹)"},
            {"flag": "--mode-xyz", "label": "Write normal modes (for animation)", "default": True},
            {"flag": "--save-structure", "label": "Save optimised structure", "default": True},
            {"flag": "--plot", "label": "Write PNG plot (matplotlib)"},
            {"flag": "--nvt-time", "label": "MD: NVT time (ps)"},
            {"flag": "--nve-time", "label": "MD: NVE time (ps)"},
            {"flag": "--n-replicas", "label": "MD: replicas"},
            {"flag": "--all-conformers", "label": "All conformers (multi-frame input)"},
        ],
    },
    "solvation": {
        "label": "Explicit solvation",
        "subcommand": "solvation",
        "fixed": [],
        "description": "Layer-by-layer solvent shell construction (mars solvation).",
        "fields": [
            {"flag": "--solvent", "label": "Solvent"},
            {"flag": "--padding", "label": "Padding (Å)"},
            {"flag": "--n-molecules", "label": "Number of solvent molecules"},
            {"flag": "--layers", "label": "Molecules per layer (space separated)"},
            {"flag": "--opt-mode", "label": "Optimisation mode"},
            {"flag": "--method", "label": "Minimizer"},
            {"flag": "--fmax", "label": "fmax (eV/Å)"},
            {"flag": "--n-solvations", "label": "Replicas"},
            {"flag": "--barostat", "label": "Barostat droplet compression"},
            {"flag": "--seed", "label": "Random seed"},
        ],
    },
    "mtd": {
        "label": "Metadynamics only",
        "subcommand": None,
        "fixed": ["--mtd-only"],
        "hide_groups": [
            "Search Mode (always auto-configured)",
            "Conformer MTD Options (autoconf only)",
            "Rotamer MD Parameters",
            "Genetic Crossing",
        ],
        "description": "MTD sampling only (no rotamer MD / genetic crossing): mars --mtd-only.",
        "fields": [
            {"flag": "--mtd-cycles", "label": "MTD cycles"},
            {"flag": "--mtd-time", "label": "MTD time per cycle (ps)"},
            {"flag": "--mtd-dt", "label": "Timestep (fs)"},
            {"flag": "--kpush", "label": "Gaussian height (kcal/mol)"},
            {"flag": "--alpha", "label": "Gaussian width"},
            {"flag": "--temperature", "label": "Temperature (K)"},
            {"flag": "--ewin", "label": "Energy window (kcal/mol)"},
            {"flag": "--save-trajectory", "label": "Save MTD trajectories", "default": True},
        ],
    },
    "md": {
        "label": "Molecular dynamics only",
        "subcommand": None,
        "fixed": ["--md-only"],
        "hide_groups": [
            "Search Mode (always auto-configured)",
            "Conformer MTD Options (autoconf only)",
            "Rotamer MD Parameters",
            "Genetic Crossing",
        ],
        "hide_flags": ["--kpush", "--alpha", "--cvdump-fs", "--kscal", "--no-trial"],
        "description": "Plain NVT MD (no bias): mars --md-only; --mtd-time sets the MD length.",
        "fields": [
            {"flag": "--mtd-time", "label": "MD time (ps)"},
            {"flag": "--mtd-dt", "label": "Timestep (fs)"},
            {"flag": "--temperature", "label": "Temperature (K)"},
            {"flag": "--ewin", "label": "Energy window (kcal/mol)"},
            {"flag": "--save-trajectory", "label": "Save MD trajectory", "default": True},
        ],
    },
}

SO3LR_MODELS = ["so3lr-1", "so3lr-2-s", "so3lr-2-m", "so3lr-2-l"]

#: Potentials offered in the GUI.  ``type: "file"/"files"`` fields take
#: upload ids; the files are copied into the job directory.
POTENTIALS: List[Dict] = [
    {
        "name": "so3lr",
        "label": "SO3LR",
        "fields": [
            {"flag": "--so3lr-model", "label": "SO3LR model", "choices": SO3LR_MODELS},
            {"flag": "--lr-cutoff", "label": "Long-range cutoff (Å)"},
            {"flag": "--so3lr-backend", "label": "Backend"},
        ],
    },
    {
        "name": "mace",
        "label": "MACE",
        "fields": [
            {"flag": "--mace-foundation", "label": "Foundation family"},
            {"flag": "--mace-model", "label": "Model (small/medium/large/...)"},
            {"flag": "--mace-backend", "label": "Backend"},
            {"flag": "--mace-cache-dir", "label": "Converted-weights cache dir (path)"},
        ],
    },
    {
        "name": "uma",
        "label": "UMA (FAIR-Chem)",
        "fields": [
            {"flag": "--uma-model", "label": "Checkpoint"},
            {"flag": "--uma-task", "label": "Task head"},
        ],
    },
    {"name": "orb", "label": "ORB", "fields": [{"flag": "--ml-model", "label": "Checkpoint"}]},
    {
        "name": "aimnet2",
        "label": "AIMNet2",
        "fields": [{"flag": "--ml-model", "label": "Checkpoint"}],
    },
    {"name": "aceff", "label": "AceFF", "fields": [{"flag": "--ml-model", "label": "Checkpoint"}]},
    {"name": "ani", "label": "ANI", "fields": [{"flag": "--ml-model", "label": "Checkpoint"}]},
    {
        "name": "dxtb",
        "label": "dxtb (GFN-xTB)",
        "fields": [{"flag": "--dxtb-method", "label": "Method"}],
    },
    {
        "name": "amber",
        "label": "AMBER (prmtop)",
        "fields": [
            {"flag": "--prmtop", "label": "prmtop file", "type": "file"},
            {"flag": "--ff-coords", "label": "Coordinates (inpcrd/rst7, optional)", "type": "file"},
        ],
    },
    {
        "name": "charmm",
        "label": "CHARMM (PSF)",
        "fields": [
            {"flag": "--psf", "label": "PSF file", "type": "file"},
            {"flag": "--ff-params", "label": "Parameter files (.prm/.rtf/.str)", "type": "files"},
            {"flag": "--ff-coords", "label": "Coordinates (crd/pdb, optional)", "type": "file"},
        ],
    },
    {
        "name": "opls",
        "label": "OPLS (GROMACS top)",
        "fields": [
            {"flag": "--top", "label": "GROMACS .top file", "type": "file"},
            {"flag": "--ff-coords", "label": "Coordinates (.gro/.pdb, optional)", "type": "file"},
            {"flag": "--gmx-include", "label": "GROMACS include dir (path)"},
        ],
    },
    {
        "name": "amoeba",
        "label": "AMOEBA",
        "fields": [
            {"flag": "--ff-xml", "label": "Force-field XML (optional)", "type": "files"},
            {"flag": "--ff-pdb", "label": "PDB with residue names (optional)", "type": "file"},
            {"flag": "--amoeba-polarization", "label": "Polarization"},
            {"flag": "--tinker-key", "label": "Tinker .key/.prm (optional)", "type": "files"},
        ],
    },
    {
        "name": "mm",
        "label": "OpenMM XML force field",
        "fields": [
            {"flag": "--ff-xml", "label": "Force-field XML", "type": "files"},
            {"flag": "--ff-pdb", "label": "PDB with residue names (optional)", "type": "file"},
        ],
    },
    {"name": "harmonic", "label": "Harmonic (toy / testing)", "fields": []},
    {"name": "lj", "label": "Lennard-Jones (toy)", "fields": []},
    {"name": "valence", "label": "Valence FF (toy / testing)", "fields": []},
]

_POTENTIALS_BY_NAME = {p["name"]: p for p in POTENTIALS}

#: Flags owned by the "Potential" form: ``{flag: [potentials that use it]}``.
POTENTIAL_FLAGS: Dict[str, List[str]] = {}
for _pot in POTENTIALS:
    for _f in _pot["fields"]:
        POTENTIAL_FLAGS.setdefault(_f["flag"], []).append(_pot["name"])

#: Flags set by the structure card / run settings rather than the task form.
GLOBAL_FLAGS = {
    "--config",
    "--potential",
    "--charge",
    "--spin",
    "--float64",
    "--cpu",
    "-h",
    "--help",
}

#: ``mars --config`` section read by each subcommand (see :mod:`mars.cli._config`).
CONFIG_SECTION = {
    None: "conformer_search",
    "ir": "ir",
    "optimize": "optimize",
    "solvation": "solvation",
}

#: Flags of the default (conformer-search) parser that select a workflow; the
#: GUI exposes those workflows as separate tasks instead.
_WORKFLOW_FLAGS = {"--mtd-only", "--md-only", "--quick", "--thorough"}

#: Section titles for subparsers whose flags have no argparse group (``mars ir``).
SECTION_OF: Dict[Optional[str], Dict[str, str]] = {
    "ir": {
        **dict.fromkeys(["--temperature"], "General"),
        **dict.fromkeys(
            [
                "--no-optimize",
                "--fmax",
                "--fd-hessian",
                "--fd-displacement",
                "--save-hessian",
                "--hessian-output",
                "--save-structure",
                "--structure-output",
                "--mode-xyz",
                "--mode-output",
                "--no-analysis",
                "--analysis-output",
            ],
            "Harmonic IR (Hessian)",
        ),
        **dict.fromkeys(
            [
                "--md",
                "--nvt-time",
                "--nve-time",
                "--md-timestep",
                "--dipole-save-fs",
                "--n-replicas",
                "--chop",
                "--window",
                "--trajectory-file",
                "--not-parallel",
            ],
            "MD-based IR (dipole autocorrelation)",
        ),
        **dict.fromkeys(
            [
                "--output",
                "--broadening",
                "--gaussian-sigma",
                "--freq-range",
                "--plot",
                "--plot-output",
            ],
            "Spectrum & plot",
        ),
        **dict.fromkeys(
            ["--conformer", "--all-conformers", "--output-prefix", "--atoms"],
            "Structure selection",
        ),
        **dict.fromkeys(["--log-file", "--log-level", "--debug"], "Logging"),
    },
}

_FIRE = ["--fire-dt-start", "--fire-dt-max", "--fire-n-min"]
_IR_MD = [
    "--nvt-time",
    "--nve-time",
    "--md-timestep",
    "--dipole-save-fs",
    "--n-replicas",
    "--chop",
    "--window",
    "--trajectory-file",
    "--not-parallel",
    "--gaussian-sigma",
]
_IR_HESSIAN = [
    "--no-optimize",
    "--fd-hessian",
    "--save-hessian",
    "--save-structure",
    "--mode-xyz",
    "--no-analysis",
    "--broadening",
]
_BARO = [
    "--baro-inner-buffer",
    "--baro-radius",
    "--baro-fill",
    "--baro-step",
    "--baro-fine-step",
    "--baro-k",
    "--baro-wall-mode",
    "--baro-max-force",
    "--baro-max-steps",
    "--baro-md",
]


def _is(flag, value=True):
    return {"flag": flag, "is": value}


def _in(flag, *values):
    return {"flag": flag, "in": list(values)}


#: When a flag is shown: ``{subcommand: {flag: [conditions]}}`` (all must
#: hold).  Conditions refer to other flags of the same form:
#: ``is`` (bool / exact value), ``in`` / ``not_in`` (choice) or ``empty``.
#: Hidden fields are left out of the command even if they hold a value.
SHOW_IF: Dict[Optional[str], Dict[str, List[Dict]]] = {
    None: {
        **{f: [_is("--nci")] for f in ("--nci-buffer", "--nci-mode", "--nci-max-dist")},
        **{f: [_in("--method", "FIRE")] for f in _FIRE},
        "--n-children": [_is("--genetic-crossing")],
    },
    "ir": {
        **{f: [_is("--md")] for f in _IR_MD},
        **{f: [_is("--md", False)] for f in _IR_HESSIAN},
        "--fmax": [_is("--md", False), _is("--no-optimize", False)],
        "--fd-displacement": [_is("--md", False), _is("--fd-hessian")],
        "--hessian-output": [_is("--md", False), _is("--save-hessian")],
        "--structure-output": [_is("--md", False), _is("--save-structure")],
        "--mode-output": [_is("--md", False), _is("--mode-xyz")],
        "--analysis-output": [_is("--md", False), _is("--no-analysis", False)],
        "--plot-output": [_is("--plot")],
        "--conformer": [_is("--all-conformers", False)],
        "--output-prefix": [_is("--all-conformers")],
    },
    "optimize": {
        "--method": [_in("--task", "minimize")],
        **{f: [_in("--task", "minimize"), _in("--method", "FIRE")] for f in _FIRE},
        "--max-stepsize": [{"flag": "--method", "not_in": ["FIRE"]}],
        "--interfragment": [_in("--coords", "internal")],
        "--ric-max-atoms": [_in("--coords", "internal")],
        "--ric-backtransform-iter": [_in("--coords", "internal")],
        "--conformer": [_is("--all-conformers", False)],
        "--output-prefix": [_is("--all-conformers")],
    },
    "solvation": {
        **{
            f: [{"flag": "--layers", "empty": True}]
            for f in ("--padding", "--n-molecules", "--max-layers")
        },
        "--freeze-mode": [_in("--opt-mode", "after-all")],
        **{
            f: [{"flag": "--opt-mode", "not_in": ["none"]}]
            for f in ("--method", "--fmax", "--maxiter", "--no-multistage", "--relax-solute")
        },
        **{
            f: [{"flag": "--opt-mode", "not_in": ["none"]}, _in("--method", "FIRE", "HYBRID")]
            for f in _FIRE
        },
        "--max-stepsize": [
            {"flag": "--opt-mode", "not_in": ["none"]},
            {"flag": "--method", "not_in": ["FIRE"]},
        ],
        **{f: [_is("--no-topology-repair", False)] for f in ("--max-repair", "--topo-tolerance")},
        **{f: [_is("--barostat")] for f in _BARO},
        **{
            f: [_is("--barostat"), _is("--baro-md")]
            for f in ("--baro-md-temp", "--baro-md-time", "--baro-md-dt")
        },
        "--no-align": [_is("--solute-swap")],
    },
}


JOB_STATUSES = ("queued", "running", "done", "failed", "cancelled")
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class PayloadError(ValueError):
    """Raised when a job request from the browser is invalid."""


@lru_cache(maxsize=1)
def _subparsers() -> Dict[Optional[str], argparse.ArgumentParser]:
    """Return ``{subcommand: parser}`` built from the real MARS CLI parser."""
    from ..cli._parser import create_parser  # argparse only, JAX-free

    main = create_parser()
    out: Dict[Optional[str], argparse.ArgumentParser] = {None: main}
    for action in main._actions:
        if isinstance(action, argparse._SubParsersAction):
            out.update(action.choices)
    return out


def _actions(subcommand: Optional[str]) -> Dict[str, argparse.Action]:
    parser = _subparsers()[subcommand]
    acts = {}
    for action in parser._actions:
        for opt in action.option_strings:
            acts[opt] = action
    return acts


def supports_flag(subcommand: Optional[str], flag: str) -> bool:
    """Whether ``mars <subcommand>`` accepts *flag*."""
    return flag in _actions(subcommand)


_ACRONYMS = {
    "md": "MD",
    "mtd": "MTD",
    "ir": "IR",
    "nci": "NCI",
    "fd": "FD",
    "nvt": "NVT",
    "nve": "NVE",
    "fire": "FIRE",
    "ric": "RIC",
    "vdw": "vdW",
    "gmx": "GROMACS",
    "xml": "XML",
    "pdb": "PDB",
    "lr": "LR",
    "ml": "ML",
    "uma": "UMA",
    "mace": "MACE",
    "so3lr": "SO3LR",
    "dxtb": "dxtb",
    "fs": "(fs)",
    "ps": "(ps)",
    "dt": "dt",
    "k": "k",
    "baro": "Barostat",
    "ff": "FF",
}


def _auto_label(flag: str) -> str:
    """Readable label from a flag name: ``--md-timestep`` -> ``MD timestep``."""
    words = [_ACRONYMS.get(w, w) for w in flag.lstrip("-").split("-")]
    text = " ".join(words)
    return text[:1].upper() + text[1:] if words[0] == words[0].lower() else text


def _field_from_action(spec: Dict, action: argparse.Action) -> Dict:
    """Merge a curated field spec with what argparse knows about the flag."""
    if spec.get("type") in ("file", "files"):
        ftype = spec["type"]
    elif action.nargs == 0:
        ftype = "bool"
    elif action.nargs == "?":
        ftype = "optional"  # bare flag, or flag + value
    elif spec.get("choices") or action.choices:
        ftype = "choice"
    elif action.nargs in ("+", "*") or isinstance(action.nargs, int):
        ftype = "list"
    elif action.type is int:
        ftype = "int"
    elif action.type is float:
        ftype = "float"
    else:
        ftype = "str"
    choices = spec.get("choices") or (list(action.choices) if action.choices else None)
    help_text = action.help or ""
    if help_text == argparse.SUPPRESS:
        help_text = ""
    try:
        help_text = help_text % {"default": action.default, "prog": "mars"}
    except (KeyError, TypeError, ValueError):
        pass
    cli_default = action.default
    if ftype == "bool":
        cli_default = False
    elif ftype == "optional":
        cli_default = None
    default = spec.get("default", cli_default)
    return {
        "flag": spec["flag"],
        "key": spec["flag"].lstrip("-").replace("-", "_"),
        "label": spec.get("label") or _auto_label(spec["flag"]),
        "type": ftype,
        "choices": choices,
        "default": default,
        "cli_default": cli_default,
        "elem_type": ("int" if action.type is int else "float" if action.type is float else "str"),
        "nargs": action.nargs if isinstance(action.nargs, int) else None,
        "help": " ".join(help_text.split()),
        "basic": bool(spec.get("basic")),
        "show_if": spec.get("show_if") or [],
    }


def _task_sections(task: Dict) -> List[Dict]:
    """Every flag of the task's subparser, grouped into form sections.

    The curated ``fields`` of the task come first ("Main options", shown
    expanded); all other flags follow under their argparse group title (or
    :data:`SECTION_OF` when the subparser has no groups).  Potential flags and
    the run settings in :data:`GLOBAL_FLAGS` are handled elsewhere.
    """
    sub = task["subcommand"]
    parser = _subparsers()[sub]
    acts = _actions(sub)
    show_if = SHOW_IF.get(sub, {})
    hidden = GLOBAL_FLAGS | set(POTENTIAL_FLAGS) | set(task.get("hide_flags", ()))
    if sub is None:
        hidden |= _WORKFLOW_FLAGS
    hide_groups = set(task.get("hide_groups", ()))
    basic = {f["flag"]: f for f in task["fields"]}

    def spec(flag):
        s = dict(basic.get(flag, {"flag": flag}))
        s["basic"] = flag in basic
        s["show_if"] = show_if.get(flag, [])
        return s

    main = [
        _field_from_action(spec(f["flag"]), acts[f["flag"]])
        for f in task["fields"]
        if f["flag"] in acts
    ]
    sections = [{"title": "Main options", "open": True, "fields": main}]
    by_title: Dict[str, Dict] = {}
    for group in parser._action_groups:
        if group.title in hide_groups:
            continue
        for action in group._group_actions:
            if not action.option_strings or isinstance(action, argparse._HelpAction):
                continue
            flag = next((o for o in action.option_strings if o.startswith("--")), None)
            if not flag or flag in hidden or flag in basic or action.help == argparse.SUPPRESS:
                continue
            title = SECTION_OF.get(sub, {}).get(flag)
            if title is None:
                title = (
                    "Other options"
                    if group.title in ("options", "optional arguments")
                    else group.title
                )
            if title not in by_title:
                by_title[title] = {"title": title, "open": False, "fields": []}
                sections.append(by_title[title])
            by_title[title]["fields"].append(_field_from_action(spec(flag), action))
    # Logging / output bookkeeping last; otherwise argparse (or SECTION_OF) order.
    order = list(dict.fromkeys(SECTION_OF.get(sub, {}).values()))
    tail = ("Logging", "Logging Options")
    sections[1:] = sorted(
        sections[1:],
        key=lambda s: (
            s["title"] in tail,
            order.index(s["title"]) if s["title"] in order else len(order),
        ),
    )
    return [s for s in sections if s["fields"]]


@lru_cache(maxsize=1)
def get_schema() -> Dict:
    """Build the task / potential schema served to the browser.

    Every flag of each task's subparser is included (see
    :func:`_task_sections`), with argparse-derived types, defaults, choices and
    help strings, plus the ``show_if`` rules that make the form adapt to the
    options already chosen.

    Returns:
        ``{"tasks": {...}, "potentials": [...], "default_potential": "so3lr"}``.
    """
    tasks = {}
    for key, task in TASKS.items():
        sub = task["subcommand"]
        acts = _actions(sub)
        sections = _task_sections(task)
        pot_fields = {}
        for pot in POTENTIALS:
            pot_fields[pot["name"]] = [
                _field_from_action(dict(f, basic=True), acts[f["flag"]])
                for f in pot["fields"]
                if f["flag"] in acts
            ]
        tasks[key] = {
            "key": key,
            "label": task["label"],
            "description": task["description"],
            "subcommand": sub,
            "fixed": task["fixed"],
            "config_section": CONFIG_SECTION[sub],
            "command": ("mars " + (sub or "") + " " + " ".join(task["fixed"])).replace("  ", " "),
            "sections": sections,
            "fields": [f for s in sections for f in s["fields"]],
            "potential_fields": pot_fields,
            "supports_cpu_flag": "--cpu" in acts,
            "supports_spin": "--spin" in acts,
        }
    pots = [{"name": p["name"], "label": p["label"]} for p in POTENTIALS]
    return {"tasks": tasks, "potentials": pots, "default_potential": "so3lr"}


def condition_holds(cond: Dict, values: Dict, fields: Dict[str, Dict]) -> bool:
    """Evaluate one ``show_if`` condition against the form *values*."""
    flag = cond["flag"]
    field = fields.get(flag)
    value = values.get(flag, field["default"] if field else None)
    if field and field["type"] == "bool":
        value = value in (True, "true", "on", 1, "1")
    if "is" in cond:
        want = cond["is"]
        return bool(value) is want if isinstance(want, bool) else str(value) == str(want)
    if "in" in cond:
        return value is not None and str(value) in [str(v) for v in cond["in"]]
    if "not_in" in cond:
        return value is None or str(value) not in [str(v) for v in cond["not_in"]]
    if "empty" in cond:
        return _is_empty(value) is cond["empty"]
    return True


def field_visible(field: Dict, values: Dict, fields: Dict[str, Dict]) -> bool:
    """Whether *field* is shown (and so passed on) given the other *values*."""
    return all(condition_holds(c, values, fields) for c in field.get("show_if") or [])


def _coerce_scalar(value, kind: str, flag: str):
    if kind == "int":
        try:
            f = float(value)
        except (TypeError, ValueError):
            raise PayloadError(f"{flag}: expected an integer, got {value!r}")
        if f != int(f):
            raise PayloadError(f"{flag}: expected an integer, got {value!r}")
        return int(f)
    if kind == "float":
        try:
            return float(value)
        except (TypeError, ValueError):
            raise PayloadError(f"{flag}: expected a number, got {value!r}")
    text = str(value)
    if "\n" in text or "\r" in text or "\0" in text:
        raise PayloadError(f"{flag}: value may not contain line breaks")
    return text


def _is_empty(value) -> bool:
    return value is None or value == "" or value == [] or value is False


def _field_args(field: Dict, value, resolve_file: Callable[[str], str]) -> List[str]:
    """Translate one form value into CLI tokens (empty list = flag omitted)."""
    flag, ftype = field["flag"], field["type"]
    if ftype == "bool":
        return [flag] if value in (True, "true", "on", 1, "1") else []
    if _is_empty(value):
        return []
    if ftype == "optional":
        if value in (True, "true", "on"):
            return [flag]
        return [flag, str(_coerce_scalar(value, field["elem_type"], flag))]
    if ftype == "file":
        return [flag, resolve_file(str(value))]
    if ftype == "files":
        ids = value if isinstance(value, list) else [value]
        return [flag] + [resolve_file(str(v)) for v in ids if v]
    if ftype == "list":
        items = value if isinstance(value, list) else re.split(r"[,\s]+", str(value).strip())
        items = [_coerce_scalar(v, field["elem_type"], flag) for v in items if str(v) != ""]
        if not items:
            return []
        if field.get("nargs") and len(items) != field["nargs"]:
            raise PayloadError(f"{flag}: expected {field['nargs']} values, got {len(items)}")
        cli_default = field.get("cli_default")
        if isinstance(cli_default, (list, tuple)) and [float(x) for x in items] == [
            float(x) for x in cli_default
        ]:
            return []
        return [flag] + [str(v) for v in items]
    if ftype == "choice":
        text = str(value)
        if field["choices"] and text not in [str(c) for c in field["choices"]]:
            raise PayloadError(f"{flag}: {text!r} is not one of {field['choices']}")
        if field.get("cli_default") is not None and text == str(field["cli_default"]):
            return []
        return [flag, text]
    coerced = _coerce_scalar(value, ftype, flag)
    if field.get("cli_default") is not None and coerced == field["cli_default"]:
        return []
    return [flag, str(coerced)]


def build_command(
    payload: Dict,
    input_name: str,
    resolve_file: Callable[[str], str] = lambda x: x,
    python: Optional[str] = None,
) -> Dict:
    """Build the MARS command line for a GUI job request.

    Args:
        payload: Job request from the browser.  Keys: ``task``, ``options``
            (``{flag: value}``), ``potential`` (name), ``potential_options``
            (``{flag: value}``), ``charge``, ``spin``, ``float64``, ``device``
            (``"auto"``/``"cpu"``/``"gpu"``) and ``advanced`` (free-text args).
        input_name: Input structure filename (relative to the job directory).
        resolve_file: Maps an upload id from a file field to the filename it
            was copied to inside the job directory.
        python: Interpreter to run (default: the server's ``sys.executable``).

    Returns:
        ``{"argv": [...], "env": {...}}`` where *env* holds only the overrides
        (``None`` = remove the variable) applied on top of the server env.

    Raises:
        PayloadError: On an unknown task / potential / flag or a bad value.
    """
    schema = get_schema()
    task_key = payload.get("task", "conformer")
    if task_key not in TASKS:
        raise PayloadError(f"unknown task {task_key!r}")
    task, tschema = TASKS[task_key], schema["tasks"][task_key]
    sub = task["subcommand"]

    argv = [python or sys.executable, "-m", "mars.cli"]
    if sub:
        argv.append(sub)
    argv.append(input_name)
    argv += task["fixed"]

    fields = {f["flag"]: f for f in tschema["fields"]}
    options = payload.get("options") or {}
    for flag, value in options.items():
        if flag not in fields:
            raise PayloadError(f"option {flag!r} is not available for task {task_key!r}")
        if field_visible(fields[flag], options, fields):
            argv += _field_args(fields[flag], value, resolve_file)

    pot = payload.get("potential") or "so3lr"
    if pot not in _POTENTIALS_BY_NAME:
        raise PayloadError(f"unknown potential {pot!r}")
    if pot != "so3lr":
        argv += ["--potential", pot]
    pfields = {f["flag"]: f for f in tschema["potential_fields"][pot]}
    for flag, value in (payload.get("potential_options") or {}).items():
        if flag not in pfields:
            raise PayloadError(f"option {flag!r} is not available for potential {pot!r}")
        argv += _field_args(pfields[flag], value, resolve_file)

    charge = _coerce_scalar(payload.get("charge", 0) or 0, "float", "--charge")
    if charge != 0:
        argv += ["--charge", f"{charge:g}"]
    spin = _coerce_scalar(payload.get("spin", 1) or 1, "float", "--spin")
    if spin < 1:
        raise PayloadError("--spin: multiplicity must be >= 1")
    if spin != 1 and tschema["supports_spin"]:
        argv += ["--spin", f"{spin:g}"]
    if payload.get("float64"):
        argv.append("--float64")

    env: Dict[str, Optional[str]] = {
        "PYTHONUNBUFFERED": "1",
        "FORCE_COLOR": "1",
        "COLORTERM": "truecolor",
    }
    device = payload.get("device", "auto")
    if device == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
        if tschema["supports_cpu_flag"]:
            argv.append("--cpu")
    elif device == "gpu":
        env["JAX_PLATFORMS"] = None  # let JAX pick the accelerator
    elif device != "auto":
        raise PayloadError(f"unknown device {device!r}")

    advanced = (payload.get("advanced") or "").strip()
    if advanced:
        try:
            argv += shlex.split(advanced)
        except ValueError as exc:
            raise PayloadError(f"advanced arguments: {exc}")
    return {"argv": argv, "env": env}


def format_command(argv: List[str], env: Dict[str, Optional[str]]) -> str:
    """Render *argv* + env overrides as a copy-pasteable shell command."""
    parts = []
    for k, v in sorted(env.items()):
        if k in ("PYTHONUNBUFFERED", "FORCE_COLOR", "COLORTERM"):
            continue
        parts.append(f"{k}={shlex.quote(v)}" if v is not None else f"-u {k}")
    prefix = ("env " + " ".join(parts) + " ") if parts else ""
    return prefix + shlex.join(argv)


# ---------------------------------------------------------------------------
# Paths / validation helpers
# ---------------------------------------------------------------------------


def safe_name(name: str, default: str = "file") -> str:
    """Reduce an uploaded filename to a safe basename (no directories)."""
    base = os.path.basename(str(name).replace("\\", "/")).strip()
    base = re.sub(r"[^A-Za-z0-9._+-]", "_", base).lstrip(".")
    return base[:120] or default


def safe_join(base: Path, rel: str) -> Path:
    """Join *rel* under *base*, refusing anything that escapes *base*.

    Args:
        base: Trusted directory.
        rel: Untrusted relative path (e.g. from a URL).

    Returns:
        The resolved path inside *base*.

    Raises:
        PermissionError: If the result would lie outside *base* (``..``,
            absolute paths, symlinks pointing out).
    """
    base = Path(base).resolve()
    if "\0" in rel:
        raise PermissionError("invalid path")
    target = (base / rel.lstrip("/")).resolve()
    if target != base and not target.is_relative_to(base):
        raise PermissionError(f"path {rel!r} escapes the job directory")
    return target


def check_id(value: str, what: str = "id") -> str:
    """Validate a job / upload id (``[A-Za-z0-9_-]``) and return it."""
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise PermissionError(f"invalid {what}")
    return value


def _new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)


def classify_file(path: Path) -> str:
    """Guess what a job output file is, so the browser can pick a viewer.

    Returns one of ``structure``, ``ensemble``, ``trajectory``, ``modes``,
    ``spectrum``, ``analysis``, ``log``, ``image``, ``text``, ``other``.
    """
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix in (".png", ".jpg", ".jpeg", ".svg", ".gif"):
        return "image"
    if name == "run.log" or suffix == ".log":
        return "log"
    if name.startswith("ir_spectrum") and suffix in (".dat", ".txt"):
        return "spectrum"
    if suffix == ".dat" and "spectrum" in name:
        return "spectrum"
    if suffix in (".dat", ".txt"):
        try:
            with open(path, "r", errors="replace") as fh:
                if fh.readline().startswith("# IR Spectrum computed with MARS"):
                    return "spectrum"
        except OSError:
            pass
    if name.startswith("mode_analysis"):
        return "analysis"
    if suffix == ".xyz":
        try:
            with open(path, "r", errors="replace") as fh:
                first = fh.readline()
                comment = fh.readline()
                n = int(first.split()[0])
                n_lines = 2
                for _ in fh:
                    n_lines += 1
                    if n_lines > 4 * (n + 2):
                        break
        except (OSError, ValueError, IndexError):
            return "text"
        multi = n_lines > (n + 2) + 1
        if comment.lstrip().startswith("Mode "):
            return "modes"
        if "trajectory" in name or comment.lstrip().startswith("Frame"):
            return "trajectory" if multi else "structure"
        if multi:
            return "ensemble" if structures.parse_energy(comment) is not None else "trajectory"
        return "structure"
    if suffix in (".txt", ".dat", ".json", ".toml", ".sh", ".csv", ".out"):
        return "text"
    return "other"


# ---------------------------------------------------------------------------
# Persistent job store
# ---------------------------------------------------------------------------


class JobStore:
    """On-disk store of uploads and jobs with a JSON index.

    Args:
        root: Work directory (created if missing).
    """

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.jobs_dir = self.root / "jobs"
        self.uploads_dir = self.root / "uploads"
        self.index_path = self.root / "jobs.json"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: Dict[str, Dict] = {}
        self._load()

    # -- persistence -------------------------------------------------------

    def _load(self):
        if self.index_path.exists():
            try:
                data = json.loads(self.index_path.read_text())
                self._jobs = {j["id"]: j for j in data.get("jobs", [])}
            except (OSError, ValueError, KeyError):
                self._jobs = {}
        # Rebuild entries for job directories missing from the index.
        for d in self.jobs_dir.iterdir():
            rec = d / "job.json"
            if d.is_dir() and d.name not in self._jobs and rec.exists():
                try:
                    self._jobs[d.name] = json.loads(rec.read_text())
                except (OSError, ValueError):
                    pass
        # Drop entries whose directory vanished; a "running" job cannot have
        # survived a server restart (it was our child process).
        for jid in list(self._jobs):
            if not (self.jobs_dir / jid).is_dir():
                del self._jobs[jid]
                continue
            job = self._jobs[jid]
            if job.get("status") == "running":
                job["status"] = "failed"
                job["error"] = "interrupted: the GUI server stopped while the job was running"
                job["finished"] = job.get("finished") or time.time()
        self._save()

    def _save(self, job_id: Optional[str] = None):
        """Atomically rewrite the index (and the ``job.json`` of *job_id*)."""
        with self._lock:
            tmp = self.index_path.with_suffix(".json.tmp")
            jobs = sorted(self._jobs.values(), key=lambda j: j.get("created", 0))
            tmp.write_text(json.dumps({"version": 1, "jobs": jobs}, indent=1))
            os.replace(tmp, self.index_path)
            ids = [job_id] if job_id else list(self._jobs)
            for jid in ids:
                d = self.jobs_dir / jid
                if jid in self._jobs and d.is_dir():
                    (d / "job.json").write_text(json.dumps(self._jobs[jid], indent=1))

    # -- uploads -------------------------------------------------------------

    def save_upload(self, filename: str, data: bytes, kind: str = "auto") -> Dict:
        """Store an uploaded file and, for structures, validate and summarise it.

        Args:
            filename: Original client-side filename.
            data: File bytes.
            kind: ``"structure"``, ``"file"`` (force-field / auxiliary file) or
                ``"auto"`` (structure if the extension is a structure format).

        Returns:
            Upload record (``id``, ``filename``, ``kind``; structures also get
            ``summary`` and ``xyz`` — the parsed geometry as XYZ text).

        Raises:
            structures.StructureError: If a structure upload does not parse.
        """
        name = safe_name(filename, "upload.xyz")
        ext = Path(name).suffix.lower()
        if kind == "auto":
            kind = "structure" if ext in structures.STRUCTURE_EXTENSIONS else "file"
        uid = _new_id()
        d = self.uploads_dir / uid
        d.mkdir(parents=True)
        path = d / name
        path.write_bytes(data)
        rec = {"id": uid, "filename": name, "kind": kind, "size": len(data)}
        if kind == "structure":
            try:
                frames = structures.read_structure_file(path)
            except structures.StructureError:
                shutil.rmtree(d, ignore_errors=True)
                raise
            rec["summary"] = structures.summarize(frames)
            rec["xyz"] = structures.frames_to_xyz(frames)
            # MARS reads XYZ/PDB/SDF through ASE; MOL2 (not ASE-readable) and
            # anything exotic is handed over as converted XYZ.
            rec["run_name"] = (
                name if ext in (".xyz", ".pdb", ".sdf", ".extxyz") else (Path(name).stem + ".xyz")
            )
            if rec["run_name"] != name:
                (d / rec["run_name"]).write_text(rec["xyz"])
        (d / "upload.json").write_text(json.dumps({k: v for k, v in rec.items() if k != "xyz"}))
        return rec

    def upload_path(self, upload_id: str, run_name: bool = False) -> Path:
        """Path of a stored upload (``run_name``: the MARS-readable variant)."""
        d = safe_join(self.uploads_dir, check_id(upload_id, "upload id"))
        meta_path = d / "upload.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"upload {upload_id} not found")
        meta = json.loads(meta_path.read_text())
        return d / (meta.get("run_name") if run_name and meta.get("run_name") else meta["filename"])

    # -- jobs ----------------------------------------------------------------

    def create_job(self, payload: Dict, python: Optional[str] = None) -> Dict:
        """Create a job directory, copy its inputs, and record the command.

        Args:
            payload: Browser job request (see :func:`build_command`); must
                contain ``structure`` (an upload id).
            python: Interpreter override (tests).

        Returns:
            The new job record (status ``queued``).

        Raises:
            PayloadError / FileNotFoundError: On a bad request.
        """
        upload_id = payload.get("structure")
        if not upload_id:
            raise PayloadError("no input structure: upload or paste a structure first")
        src = self.upload_path(upload_id, run_name=True)
        jid = _new_id()
        jdir = self.jobs_dir / jid
        jdir.mkdir(parents=True)
        try:
            input_name = safe_name(src.name, "input.xyz")
            shutil.copy2(src, jdir / input_name)

            def resolve_file(uid: str) -> str:
                p = self.upload_path(uid)
                dest = jdir / safe_name(p.name)
                if not dest.exists():
                    shutil.copy2(p, dest)
                return dest.name

            cmd = build_command(payload, input_name, resolve_file, python=python)
        except Exception:
            shutil.rmtree(jdir, ignore_errors=True)
            raise
        task = payload.get("task", "conformer")
        cmd_str = format_command(cmd["argv"], cmd["env"])
        (jdir / "command.sh").write_text(
            '#!/bin/sh\n# Re-run this MARS GUI job from its directory\ncd "$(dirname "$0")"\n'
            + cmd_str
            + "\n"
        )
        try:  # the same run as a `mars --config` file
            from .. import __version__
            from .runconfig import export_config

            _, cfg = export_config(
                payload, input_name, lambda uid: safe_name(self.upload_path(uid).name), __version__
            )
            (jdir / "config.toml").write_text(cfg)
        except Exception:  # pragma: no cover - never block a job on the extra file
            pass
        job = {
            "id": jid,
            "name": (payload.get("name") or "").strip()[:80] or f"{TASKS[task]['label']}",
            "task": task,
            "task_label": TASKS[task]["label"],
            "potential": payload.get("potential") or "so3lr",
            "input": input_name,
            "status": "queued",
            "created": time.time(),
            "started": None,
            "finished": None,
            "returncode": None,
            "argv": cmd["argv"],
            "env": cmd["env"],
            "command": cmd_str,
            "payload": payload,
            "error": None,
        }
        with self._lock:
            self._jobs[jid] = job
            self._save(jid)
        return dict(job)

    def list_jobs(self) -> List[Dict]:
        """All job records, newest first."""
        with self._lock:
            return sorted(
                (dict(j) for j in self._jobs.values()), key=lambda j: -j.get("created", 0)
            )

    def get(self, job_id: str) -> Dict:
        """Return a copy of a job record (``KeyError`` if unknown)."""
        check_id(job_id, "job id")
        with self._lock:
            return dict(self._jobs[job_id])

    def update(self, job_id: str, **fields) -> Dict:
        """Update fields of a job record and persist the index."""
        with self._lock:
            self._jobs[job_id].update(fields)
            self._save(job_id)
            return dict(self._jobs[job_id])

    def delete(self, job_id: str):
        """Remove a job record and its directory."""
        check_id(job_id, "job id")
        with self._lock:
            self._jobs.pop(job_id)
            self._save(job_id)
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)

    def job_dir(self, job_id: str) -> Path:
        """Directory of a job (validated id)."""
        return safe_join(self.jobs_dir, check_id(job_id, "job id"))

    def job_file(self, job_id: str, rel: str) -> Path:
        """Resolve a file inside a job directory, guarding against traversal."""
        path = safe_join(self.job_dir(job_id), rel)
        if not path.is_file():
            raise FileNotFoundError(rel)
        return path

    def list_files(self, job_id: str) -> List[Dict]:
        """List the files of a job (relative path, size, mtime, kind)."""
        base = self.job_dir(job_id)
        out = []
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.is_symlink():
                continue
            rel = p.relative_to(base).as_posix()
            if rel == "job.json":
                continue
            st = p.stat()
            out.append(
                {"path": rel, "size": st.st_size, "mtime": st.st_mtime, "kind": classify_file(p)}
            )
        return out

    def read_log(self, job_id: str, offset: int = 0, max_bytes: int = 256_000) -> Dict:
        """Read the job log from *offset* (negative = tail of that many bytes).

        Returns:
            ``{"text", "offset", "size"}``; *offset* is where the next poll
            should continue.  Carriage-return progress bars are collapsed.
        """
        path = self.job_dir(job_id) / "run.log"
        if not path.exists():
            return {"text": "", "offset": 0, "size": 0}
        size = path.stat().st_size
        if offset < 0:
            offset = max(0, size + offset)
        offset = min(offset, size)
        with open(path, "rb") as fh:
            fh.seek(offset)
            data = fh.read(max_bytes)
        text = data.decode("utf-8", errors="replace")
        return {"text": text, "offset": offset + len(data), "size": size}


# ---------------------------------------------------------------------------
# Job runner
# ---------------------------------------------------------------------------


def _default_launcher(argv, cwd, env, log_fh):
    return subprocess.Popen(
        argv,
        cwd=str(cwd),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group -> cancel kills children too
    )


def _child_env(overrides: Dict[str, Optional[str]]) -> Dict[str, str]:
    """Server environment + job overrides, with the MARS package importable."""
    env = dict(os.environ)
    for k, v in overrides.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    # Make sure the child imports the same `mars` as the server (dev checkouts).
    pkg_root = str(Path(__file__).resolve().parents[2])
    paths = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    if pkg_root not in paths:
        paths.append(pkg_root)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


class JobRunner:
    """Run queued jobs as subprocesses, at most ``max_parallel`` at a time.

    Args:
        store: The :class:`JobStore`.
        max_parallel: Number of concurrent MARS processes (default 1 — MARS
            usually saturates one GPU).
        launcher: ``launcher(argv, cwd, env, log_fh) -> Popen-like``; tests
            inject a fake one.  The returned object needs ``wait()``,
            ``poll()`` and ``pid``.
    """

    def __init__(self, store: JobStore, max_parallel: int = 1, launcher=None):
        self.store = store
        self.launcher = launcher or _default_launcher
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._procs: Dict[str, object] = {}
        self._cancelled = set()
        self._lock = threading.Lock()
        self._threads = [
            threading.Thread(target=self._worker, daemon=True, name=f"mars-gui-worker-{i}")
            for i in range(max(1, int(max_parallel)))
        ]
        for t in self._threads:
            t.start()
        # Re-queue jobs that were waiting when the server last stopped.
        for job in sorted(store.list_jobs(), key=lambda j: j["created"]):
            if job["status"] == "queued":
                self._queue.put(job["id"])

    def submit(self, job_id: str):
        """Queue a job for execution."""
        self._queue.put(job_id)

    def cancel(self, job_id: str) -> Dict:
        """Cancel a queued or running job (SIGTERM to its process group, then SIGKILL)."""
        job = self.store.get(job_id)
        with self._lock:
            self._cancelled.add(job_id)
            proc = self._procs.get(job_id)
        if job["status"] == "queued":
            return self.store.update(job_id, status="cancelled", finished=time.time())
        if proc is not None and proc.poll() is None:
            _terminate(proc)
        return self.store.get(job_id)

    def is_active(self, job_id: str) -> bool:
        """Whether the job's process is currently running."""
        with self._lock:
            proc = self._procs.get(job_id)
        return proc is not None and proc.poll() is None

    def _worker(self):
        while True:
            job_id = self._queue.get()
            try:
                self._run(job_id)
            except Exception as exc:  # never let a worker thread die
                try:
                    self.store.update(
                        job_id, status="failed", error=f"runner error: {exc}", finished=time.time()
                    )
                except KeyError:
                    pass

    def _run(self, job_id: str):
        try:
            job = self.store.get(job_id)
        except KeyError:
            return  # deleted while queued
        if job["status"] != "queued" or job_id in self._cancelled:
            return
        jdir = self.store.job_dir(job_id)
        env = _child_env(job.get("env") or {})
        with open(jdir / "run.log", "ab", buffering=0) as log_fh:
            log_fh.write(f"$ {job['command']}\n\n".encode())
            try:
                proc = self.launcher(job["argv"], jdir, env, log_fh)
            except OSError as exc:
                self.store.update(
                    job_id, status="failed", error=f"could not start: {exc}", finished=time.time()
                )
                return
            with self._lock:
                self._procs[job_id] = proc
            self.store.update(job_id, status="running", started=time.time(), pid=proc.pid)
            rc = proc.wait()
        with self._lock:
            self._procs.pop(job_id, None)
            cancelled = job_id in self._cancelled
        if cancelled:
            status, err = "cancelled", None
        elif rc == 0:
            status, err = "done", None
        else:
            status, err = "failed", f"MARS exited with code {rc}"
        try:
            self.store.update(
                job_id, status=status, returncode=rc, finished=time.time(), error=err, pid=None
            )
        except KeyError:
            pass  # deleted while running


def _terminate(proc, grace: float = 5.0):
    """Terminate a job's process group, escalating to SIGKILL after *grace* s."""
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, AttributeError, OSError):
        pgid = None

    def _send(sig):
        try:
            if pgid is not None and hasattr(os, "killpg"):
                os.killpg(pgid, sig)
            else:
                proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass

    _send(signal.SIGTERM)

    def _escalate():
        deadline = time.time() + grace
        while time.time() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.1)
        _send(getattr(signal, "SIGKILL", signal.SIGTERM))

    threading.Thread(target=_escalate, daemon=True).start()
