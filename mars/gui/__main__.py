"""``python -m mars.gui`` — start the MARS web GUI."""

import argparse
import sys

from .server import add_arguments, run_from_args


def main(argv=None) -> int:
    """Parse arguments and start the GUI server."""
    parser = argparse.ArgumentParser(
        prog="python -m mars.gui", description="MARS web GUI (local server + browser front end)"
    )
    add_arguments(parser)
    return run_from_args(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
