"""
Command-line interface for MARS.

Provides easy access to conformational search workflows from the terminal.
"""

import argparse
import sys

from ._config import build_config_argv, get_constraints_from_config
from ._parser import create_parser

# NOTE: the workflow modules (_conformer_search, _ir, _optimize, _solvation) and
# _common pull in JAX / jax-md / the ML potentials. They are imported lazily
# inside main() per subcommand so that lightweight, JAX-free commands such as
# `mars viewer` run without a working JAX/GPU stack installed.

_SUBCOMMANDS = ("ir", "optimize", "solvation", "viewer")


def _extract_config_path(argv):
    """Return (config_path, argv_without_config) from a raw argv list."""
    config_path = None
    clean = []
    i = 0
    while i < len(argv):
        if argv[i] == "--config" and i + 1 < len(argv):
            config_path = argv[i + 1]
            i += 2
        elif argv[i].startswith("--config="):
            config_path = argv[i].split("=", 1)[1]
            i += 1
        else:
            clean.append(argv[i])
            i += 1
    return config_path, clean


def _detect_command(argv):
    """Return the subcommand name (``"ir"``/``"optimize"``/``"solvation"``)
    if any of those appears as the first non-flag positional in *argv*,
    else ``None``.
    """
    for arg in argv:
        if arg.startswith("-"):
            continue
        return arg if arg in _SUBCOMMANDS else None
    return None


def _inject_config_defaults(argv, command):
    """If *argv* contains ``--config FILE``, load the file and merge its
    values into *argv* so that explicit CLI flags still take precedence.

    For subcommand modes the subcommand name is kept as the leading token so
    argparse routes the config-supplied flags to the correct subparser.

    Returns ``(config_path, input_from_config, effective_argv)``.
    *config_path* is ``None`` when no ``--config`` flag was given.
    """
    config_path, argv_clean = _extract_config_path(argv)
    if config_path is None:
        return None, None, argv

    try:
        input_from_config, config_argv = build_config_argv(config_path, command)
    except FileNotFoundError as exc:
        print(f"mars: error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"mars: error loading config file '{config_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    if command is not None and argv_clean and argv_clean[0] == command:
        # Subcommand mode: keep the subcommand name first, splice config
        # defaults right after it, then the remaining user flags.  Explicit
        # CLI flags appear *after* config defaults so they override.
        merged = [command] + config_argv + argv_clean[1:]
    else:
        # Default conformer-search mode (no subcommand): config defaults
        # first, then explicit CLI flags (which override on duplicates).
        merged = config_argv + argv_clean

    return config_path, input_from_config, merged


def _value_taking_flags(parser):
    """Return the set of option strings whose action consumes a value.

    Used by the conformer-search default-mode input extractor so the value
    after ``--mode``, ``--ewin``, etc. is not mistaken for the input file.
    """
    valued = set()
    for action in parser._actions:
        if not action.option_strings:
            continue
        if isinstance(
            action,
            (
                argparse._StoreTrueAction,
                argparse._StoreFalseAction,
                argparse._HelpAction,
                argparse._CountAction,
            ),
        ):
            continue
        # nargs of 0 means no value; '?' is optional and we treat as no-value
        # for safety in this manual scanner.
        if action.nargs == 0:
            continue
        for opt in action.option_strings:
            valued.add(opt)
    return valued


def _extract_input_and_rest(argv, parser):
    """Pull the first input positional out of *argv* without misidentifying
    the value of a preceding flag as the input.

    Returns ``(input_file, remaining_argv)`` — *input_file* is ``None`` if
    no positional was found.
    """
    valued = _value_taking_flags(parser)
    input_file = None
    remaining = []
    skip_next = False
    for arg in argv:
        if skip_next:
            remaining.append(arg)
            skip_next = False
            continue
        if arg.startswith("-"):
            remaining.append(arg)
            # `--flag value` (two tokens) → the next token is its value.
            # `--flag=value` (single token) → no skip.
            if arg in valued and "=" not in arg:
                skip_next = True
            continue
        if input_file is None:
            input_file = arg
        else:
            remaining.append(arg)
    return input_file, remaining


def _attach_constraints(args, config_path):
    """Load the ``[constraints]`` section from *config_path* and attach its
    values to *args* as ``args.constraint_atoms``, ``args.constraint_k``,
    and ``args.constraint_apply_to``.  Defaults to no constraints."""
    constraints = get_constraints_from_config(config_path)
    if constraints:
        args.constraint_atoms = constraints["atoms"]
        args.constraint_k = constraints["k"]
        args.constraint_apply_to = constraints["apply_to"]
    else:
        args.constraint_atoms = None
        args.constraint_k = 100.0
        args.constraint_apply_to = "all"


def main():
    """Main CLI entry point."""
    parser = create_parser()
    raw_args = sys.argv[1:]

    # Strip --config first so subcommand detection sees the real argv.
    _, args_without_config = _extract_config_path(raw_args)
    command = _detect_command(args_without_config)

    # Now build the effective argv (config defaults merged with explicit CLI flags).
    config_path, input_from_config, effective_args = _inject_config_defaults(raw_args, command)

    if command == "ir":
        from ._common import setup_debug_mode
        from ._ir import run_ir_workflow

        args = parser.parse_args(effective_args)
        _attach_constraints(args, config_path)
        setup_debug_mode(args)
        return run_ir_workflow(args)

    if command == "optimize":
        from ._common import setup_debug_mode
        from ._optimize import run_optimize_workflow

        args = parser.parse_args(effective_args)
        _attach_constraints(args, config_path)
        setup_debug_mode(args)
        return run_optimize_workflow(args)

    if command == "solvation":
        from ._common import setup_debug_mode
        from ._solvation import run_solvation_workflow

        args = parser.parse_args(effective_args)
        setup_debug_mode(args)
        return run_solvation_workflow(args)

    if command == "viewer":
        # Viewer is config-free and JAX-free: no constraints / precision /
        # potential setup, and no jax-md / ML-potential imports.
        from ._viewer import run_viewer_workflow

        args = parser.parse_args(effective_args)
        if getattr(args, "debug", False):
            from .. import enable_debug

            enable_debug()
            args.log_level = "DEBUG"
        return run_viewer_workflow(args)

    # Default: conformational search mode (no subcommand).
    from ._common import setup_debug_mode
    from ._conformer_search import run_conformer_search_workflow

    # Pull the input file out of effective_args without confusing it with the
    # value of a preceding flag (e.g. --mode thorough).
    input_file, remaining_args = _extract_input_and_rest(effective_args, parser)

    # Fall back to input specified in the config file.
    if input_file is None:
        input_file = input_from_config

    args = parser.parse_args(remaining_args)
    args.input = input_file
    args.command = None

    if args.input is None:
        parser.print_help()
        sys.exit(1)

    _attach_constraints(args, config_path)
    setup_debug_mode(args)
    return run_conformer_search_workflow(args)


if __name__ == "__main__":
    sys.exit(main())
