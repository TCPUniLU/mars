"""
``mars gui`` — launch the local web GUI (JAX-free; see :mod:`mars.gui`).
"""

import argparse


def build_gui_subparser(subparsers):
    """Register the ``gui`` subcommand on the MARS CLI."""
    from ..gui.server import add_arguments

    gui_parser = subparsers.add_parser(
        "gui",
        help="Local web GUI: upload a structure, run MARS jobs, view results in the browser",
        description="Start the MARS web GUI on a local port and open it in the default browser.",
        epilog="""
EXAMPLES
  mars gui                              # http://127.0.0.1:8765, jobs in ~/mars-gui-jobs
  mars gui --demo                       # example results only; nothing is run
  mars gui --port 9000 --workdir runs/  # custom port / job directory
  mars gui --no-browser                 # headless (e.g. on a workstation via SSH tunnel)
  mars gui --install-desktop            # add a clickable 'MARS GUI' app-menu launcher
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_arguments(gui_parser)
    return gui_parser


def run_gui_workflow(args) -> int:
    """Run ``mars gui`` with parsed *args*."""
    from ..gui.server import run_from_args

    return run_from_args(args)
