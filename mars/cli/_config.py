"""
Config file loading for MARS CLI.

Supports a simple TOML-like config file with sections per command:

    [conformer_search]
    mode = "thorough"
    temperature = 400.0
    charge = -1.0
    nci = true

    [ir]
    temperature = 300.0
    plot = true
    atoms = [0, 1, 2]   # optional: partial Hessian / partial dipole over
                        # this subset of atoms (0-based). Also accepts
                        # CSV/range strings like "0-2" or "0-3,7,10-12".
                        # CLI form: --atoms 0,1,2

    [optimize]
    fmax = 0.001
    method = "FIRE"
    fire_dt_start = 0.05
    fire_dt_max = 0.1
    fire_n_min = 2

    [solvation]
    method = "FIRE"
    fire_dt_max = 0.1

    [constraints]
    atoms = [0, 1, 2]   # atom indices to constrain (0-based)
    k = 100.0           # spring constant in eV/Å²  (default: 100.0)
    apply_to = "all"    # "all", "mtd", or "optimization"  (default: "all")

Key names match the long CLI option names with dashes replaced by underscores.
Boolean flags (store_true) are set by writing ``flag = true``.
CLI arguments always override values from the config file.
Constraints are only read from the config file — there are no equivalent CLI flags.
"""

import re

# Map command name to config section name
_COMMAND_SECTION = {
    None: "conformer_search",
    "ir": "ir",
    "optimize": "optimize",
    "solvation": "solvation",
}

# Section applied on top of any command-specific section
_GLOBAL_SECTION = "global"


def _parse_value(raw):
    """Parse a TOML scalar value string into a Python object."""
    s = raw.strip()
    # TOML inline array  [item, item, ...]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(item) for item in inner.split(",")]
    # Quoted strings
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    # Booleans
    if s == "true":
        return True
    if s == "false":
        return False
    # Integer
    try:
        return int(s)
    except ValueError:
        pass
    # Float
    try:
        return float(s)
    except ValueError:
        pass
    # Bare string (no quotes)
    return s


def parse_atom_indices(val):
    """Parse an atom index specification into a sorted list of ints.

    Accepted formats:
    - List/tuple of ints (already parsed by the config loader): ``[0, 1, 2]``
    - Comma-separated string: ``"0,1,2,5"``
    - Range string: ``"0-5"``   →  [0, 1, 2, 3, 4, 5]
    - Mixed: ``"0-3,7,10-12"``  →  [0, 1, 2, 3, 7, 10, 11, 12]
    """
    if isinstance(val, (list, tuple)):
        return sorted(int(x) for x in val)
    if isinstance(val, int):
        return [val]
    indices = []
    for part in str(val).split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            indices.extend(range(int(start.strip()), int(end.strip()) + 1))
        else:
            indices.append(int(part))
    return sorted(set(indices))


def load_config(path):
    """Parse a TOML-style config file.

    Returns a dict mapping section names to dicts of {key: value}.
    Lines starting with ``#`` and blank lines are ignored.
    Inline comments (`` # ...``) are stripped from values.
    """
    sections = {}
    current = None

    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Section header  [name]
            m = re.match(r"^\[([A-Za-z_][A-Za-z0-9_]*)\]$", line)
            if m:
                current = m.group(1)
                sections.setdefault(current, {})
                continue

            # key = value
            if "=" in line:
                if current is None:
                    raise ValueError(f"{path}:{lineno}: key=value outside any section")
                key, _, val = line.partition("=")
                key = key.strip()
                # Strip inline comment
                val = re.sub(r"\s+#.*$", "", val).strip()
                sections[current][key] = _parse_value(val)
                continue

            # Non-empty line that doesn't match anything
            raise ValueError(f"{path}:{lineno}: cannot parse line: {line!r}")

    return sections


def config_section_to_argv(section_dict):
    """Convert a config section dict to a list of CLI-style argument strings.

    Mapping rules:

    - ``bool`` values: ``True`` → bare ``--flag``; ``False`` is skipped
      (CLI flags use ``store_true`` so there is no negation form).
    - ``list``/``tuple`` values: emitted as multiple tokens after the flag,
      e.g. ``layers = [2, 4]`` → ``["--layers", "2", "4"]``.  This matches
      argparse ``nargs="+"`` / ``nargs=2`` arguments such as ``--layers``
      and ``--freq-range``.
    - All other scalars: ``--flag value``.

    Keys use underscores; dashes are inserted automatically.
    The reserved key ``input`` is skipped (it is handled separately as the
    positional input filename by :func:`build_config_argv`).
    """
    argv = []
    for key, val in section_dict.items():
        if key == "input":
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(val, bool):
            if val:
                argv.append(flag)
        elif isinstance(val, (list, tuple)):
            argv.append(flag)
            argv.extend(str(item) for item in val)
        else:
            argv.extend([flag, str(val)])
    return argv


def get_constraints_from_config(config_path):
    """Extract positional-constraint settings from the ``[constraints]`` section.

    Returns a dict with keys:

    - ``atoms``    : list of int — 0-based atom indices to constrain
    - ``k``        : float — spring constant in eV/Å² (default 100.0)
    - ``apply_to`` : str — ``"all"``, ``"mtd"``, or ``"optimization"``

    Returns ``None`` if the config has no ``[constraints]`` section or no
    ``atoms`` key, so callers can do ``if constraints: ...``.
    """
    if config_path is None:
        return None
    try:
        sections = load_config(config_path)
    except Exception:
        return None

    constr = sections.get("constraints", {})
    if not constr or "atoms" not in constr:
        return None

    return {
        "atoms": parse_atom_indices(constr["atoms"]),
        "k": float(constr.get("k", 100.0)),
        "apply_to": str(constr.get("apply_to", "all")),
    }


def build_config_argv(config_path, command):
    """Load *config_path* and return default argv for *command*.

    *command* is ``None`` (conformer search), ``"ir"``, or ``"optimize"``.
    Values from the ``[global]`` section are applied first; the
    command-specific section overrides them.

    Returns a tuple ``(input_from_config, extra_argv)`` where
    *input_from_config* is the value of the ``input`` key in the config
    (or ``None``) and *extra_argv* is a list of ``--flag [value]`` strings.
    """
    try:
        sections = load_config(config_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    merged = {}
    if _GLOBAL_SECTION in sections:
        merged.update(sections[_GLOBAL_SECTION])

    section_name = _COMMAND_SECTION.get(command, "conformer_search")
    if section_name in sections:
        merged.update(sections[section_name])

    input_from_config = merged.pop("input", None)
    extra_argv = config_section_to_argv(merged)
    return input_from_config, extra_argv
