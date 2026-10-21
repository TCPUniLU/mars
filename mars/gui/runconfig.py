"""
GUI run settings <-> ``mars --config`` files (the TOML-style format).

A GUI job request (see :func:`mars.gui.jobs.build_command`) is saved as a
config that the CLI reads directly::

    # MARS run configuration (MARS GUI 1.0.1)
    # Run it with:  mars ir --config ir.toml
    [ir]
    input = "aspirin.xyz"
    potential = "mace"
    mace_model = "medium"
    md = true
    nvt_time = 10.0

    [gui]            # GUI-only settings, ignored by the CLI
    task = "ir"
    device = "auto"

Loading goes the other way: any ``mars --config`` file fills the GUI form, and
keys the form cannot hold are reported.
"""

import re
import shlex
from typing import Callable, Dict, List, Optional, Tuple

from ..cli._config import config_section_to_argv, parse_config_text
from .jobs import (
    _POTENTIALS_BY_NAME,
    TASKS,
    PayloadError,
    _coerce_scalar,
    _field_args,
    field_visible,
    get_schema,
)

#: Section the CLI applies before the command's own section.
GLOBAL_SECTION = "global"
#: GUI-only section (the CLI ignores sections it does not know).
GUI_SECTION = "gui"


def _key(flag: str) -> str:
    return flag.lstrip("-").replace("-", "_")


def _flag(key: str) -> str:
    return "--" + str(key).replace("_", "-")


def _config_value(field: Dict, value, resolve_file: Callable[[str], str]):
    """The config value for one form value, or ``None`` when it equals the CLI default."""
    tokens = _field_args(field, value, resolve_file)
    if not tokens:
        return None
    ftype, args = field["type"], tokens[1:]
    if ftype == "bool" or (ftype == "optional" and not args):
        return True
    if ftype in ("files", "list"):
        conv = {"int": int, "float": float}.get(field.get("elem_type"), str)
        return [conv(a) if ftype == "list" else a for a in args]
    if ftype in ("int", "float", "optional"):
        return _coerce_scalar(args[0], "int" if ftype == "int" else "float", field["flag"])
    return args[0]


def payload_to_config(
    payload: Dict,
    input_name: Optional[str] = None,
    resolve_file: Callable[[str], str] = lambda x: x,
) -> Dict[str, Dict]:
    """Translate a GUI job request into ``mars --config`` sections.

    Args:
        payload: GUI job request (``task``, ``options``, ``potential``, ...).
        input_name: Structure filename written as ``input`` (optional).
        resolve_file: Maps an upload id of a file field to the filename to write.

    Returns:
        ``{section: {key: value}}`` with the command section first and a
        ``gui`` section for GUI-only settings.  Values equal to the CLI
        defaults and options hidden by the form are left out.
    """
    schema = get_schema()
    task_key = payload.get("task", "conformer")
    if task_key not in TASKS:
        raise PayloadError(f"unknown task {task_key!r}")
    tschema = schema["tasks"][task_key]
    body: Dict = {}
    if input_name:
        body["input"] = input_name
    for flag in TASKS[task_key]["fixed"]:
        body[_key(flag)] = True

    pot = payload.get("potential") or "so3lr"
    if pot not in _POTENTIALS_BY_NAME:
        raise PayloadError(f"unknown potential {pot!r}")
    body["potential"] = pot
    pfields = {f["flag"]: f for f in tschema["potential_fields"][pot]}
    for flag, value in (payload.get("potential_options") or {}).items():
        if flag in pfields:
            v = _config_value(pfields[flag], value, resolve_file)
            if v is not None:
                body[_key(flag)] = v

    charge = _coerce_scalar(payload.get("charge", 0) or 0, "float", "--charge")
    if charge != 0:
        body["charge"] = charge
    spin = _coerce_scalar(payload.get("spin", 1) or 1, "float", "--spin")
    if spin != 1 and tschema["supports_spin"]:
        body["spin"] = spin
    if payload.get("float64"):
        body["float64"] = True
    device = payload.get("device", "auto")
    if device == "cpu" and tschema["supports_cpu_flag"]:
        body["cpu"] = True

    fields = {f["flag"]: f for f in tschema["fields"]}
    options = payload.get("options") or {}
    for flag, value in options.items():
        if flag not in fields:
            raise PayloadError(f"option {flag!r} is not available for task {task_key!r}")
        if not field_visible(fields[flag], options, fields):
            continue
        v = _config_value(fields[flag], value, resolve_file)
        if v is not None:
            body[_key(flag)] = v

    gui = {"task": task_key, "device": device}
    for key in ("advanced", "name"):  # free text: keep it parseable
        text = re.sub(r"\s#", " ", str(payload.get(key) or "")).replace('"', "'").strip()
        if text:
            gui[key] = text
    return {tschema["config_section"]: body, GUI_SECTION: gui}


def _toml_value(value) -> str:
    """One value in the TOML-style syntax read by :func:`mars.cli._config.parse_config_text`."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    text = str(value)
    # The parser has no escapes and strips `` #...`` as an inline comment.
    if "\n" in text or ('"' in text and "'" in text) or re.search(r"\s#", text):
        raise PayloadError(f"value {text!r} cannot be written to a config file")
    return f"'{text}'" if '"' in text else f'"{text}"'


def config_to_toml(sections: Dict[str, Dict], version: str = "") -> str:
    """Render config *sections* as a commented ``mars --config`` file."""
    gui = sections.get(GUI_SECTION, {})
    task = TASKS.get(gui.get("task", ""), TASKS["conformer"])
    sub = task["subcommand"]
    cmd = "mars " + (sub + " " if sub else "") + "--config <this file>"
    lines = [
        f"# MARS run configuration{f' (MARS GUI {version})' if version else ''}",
        f"# Run it with:  {cmd}",
        "# Keys are the CLI long options without the leading dashes ('-' -> '_');",
        "# options left out use the CLI defaults. The [gui] section is ignored by the CLI.",
    ]
    advanced = gui.get("advanced")
    if advanced:
        lines.append(f"# Extra arguments set in the GUI (not applied by --config): {advanced}")
    for name, body in sections.items():
        lines += ["", f"[{name}]"]
        lines += [f"{key} = {_toml_value(val)}" for key, val in body.items() if val is not None]
    return "\n".join(lines) + "\n"


def _form_value(field: Dict, value):
    """Turn a config value into what the browser form holds for *field*."""
    if field["type"] == "bool":
        return bool(value)
    if field["type"] == "optional":
        return True if value is True else value
    if field["type"] == "list":
        return list(value) if isinstance(value, (list, tuple)) else shlex.split(str(value))
    return value


def _pick_task(sections: Dict[str, Dict], wanted: Optional[str]) -> str:
    if wanted in TASKS:
        return wanted
    schema = get_schema()
    by_section = {}
    for key, t in schema["tasks"].items():
        by_section.setdefault(t["config_section"], key)
    for section in ("ir", "optimize", "solvation", "conformer_search"):
        if section in sections:
            body = sections[section]
            if section == "conformer_search":
                if body.get("mtd_only"):
                    return "mtd"
                if body.get("md_only"):
                    return "md"
            return by_section[section]
    return "conformer"


def config_to_payload(sections: Dict[str, Dict], task: Optional[str] = None) -> Dict:
    """Fill a GUI job request from ``mars --config`` *sections*.

    Args:
        sections: Parsed config (see :func:`mars.cli._config.parse_config_text`).
        task: GUI task to load into; default: the ``gui.task`` key, else the
            first command section found (``ir``, ``optimize``, ``solvation``,
            ``conformer_search``).

    Returns:
        ``{"payload": {...}, "input": name or None, "warnings": [...],
        "files": [flags that need their files attached again]}``.
    """
    gui = sections.get(GUI_SECTION) or {}
    task_key = _pick_task(sections, task or gui.get("task"))
    tschema = get_schema()["tasks"][task_key]
    merged: Dict = {}
    merged.update(sections.get(GLOBAL_SECTION) or {})
    merged.update(sections.get(tschema["config_section"]) or {})

    warnings: List[str] = []
    other = [
        s
        for s in sections
        if s not in (GLOBAL_SECTION, GUI_SECTION, tschema["config_section"], "constraints")
    ]
    if other:
        warnings.append(
            f"Loaded the [{tschema['config_section']}] settings; ignored section(s): "
            + ", ".join(other)
        )
    if sections.get("constraints"):
        warnings.append(
            "The [constraints] section is applied by the CLI only; "
            "the GUI does not run constrained jobs yet."
        )

    input_name = merged.pop("input", None)
    pot = str(merged.pop("potential", None) or "so3lr")
    if pot not in _POTENTIALS_BY_NAME:
        warnings.append(f"Unknown potential {pot!r}; using so3lr")
        pot = "so3lr"
    charge = merged.pop("charge", 0) or 0
    spin = merged.pop("spin", 1) or 1
    float64 = bool(merged.pop("float64", False))
    cpu = bool(merged.pop("cpu", False))
    device = gui.get("device") or ("cpu" if cpu else "auto")
    if merged.pop("quick", False) and "mode" not in merged:
        merged["mode"] = "quick"
    if merged.pop("thorough", False) and "mode" not in merged:
        merged["mode"] = "thorough"
    for flag in TASKS[task_key]["fixed"]:
        merged.pop(_key(flag), None)
    for key in ("mtd_only", "md_only"):
        if merged.pop(key, False):
            warnings.append(f"{key} is a separate task in the GUI; it was not applied")

    fields = {f["flag"]: f for f in tschema["fields"]}
    pfields = {f["flag"]: f for f in tschema["potential_fields"][pot]}
    all_pot_flags = {f["flag"] for fs in tschema["potential_fields"].values() for f in fs}
    options, pot_options, files, extra = {}, {}, [], []
    for key, value in merged.items():
        flag = _flag(key)
        if value is None:
            continue
        if flag in pfields:
            if pfields[flag]["type"] in ("file", "files"):
                files.append(flag)
            else:
                pot_options[flag] = _form_value(pfields[flag], value)
        elif flag in fields:
            options[flag] = _form_value(fields[flag], value)
        elif flag in all_pot_flags:
            warnings.append(f"{key}: not used by potential {pot!r}; ignored")
        else:
            extra += config_section_to_argv({key: value})
    if files:
        warnings.append(
            "Attach these files again (the config only names them): "
            + ", ".join(f"{f} = {merged[_key(f)]}" for f in files)
        )
    advanced = gui.get("advanced") or ""
    if extra:
        joined = shlex.join(extra)
        advanced = (advanced + " " + joined).strip()
        warnings.append(f"Not a form option of this task; kept as extra arguments: {joined}")
    payload = {
        "task": task_key,
        "options": options,
        "potential": pot,
        "potential_options": pot_options,
        "charge": charge,
        "spin": spin,
        "float64": float64,
        "device": device if device in ("auto", "cpu", "gpu") else "auto",
        "advanced": advanced,
        "name": gui.get("name") or "",
    }
    return {"payload": payload, "input": input_name, "warnings": warnings, "files": files}


def parse_config(text: str, filename: str = "config.toml") -> Dict[str, Dict]:
    """Parse an uploaded ``mars --config`` file."""
    return parse_config_text(text, source=filename)


def export_config(
    payload: Dict,
    input_name: Optional[str] = None,
    resolve_file: Callable[[str], str] = lambda x: x,
    version: str = "",
) -> Tuple[str, str]:
    """Return ``(filename, config_text)`` for a GUI job request."""
    sections = payload_to_config(payload, input_name, resolve_file)
    return f"mars_{payload.get('task', 'conformer')}.toml", config_to_toml(sections, version)
