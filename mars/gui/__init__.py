"""
MARS web GUI — a local, browser-based front end for the MARS CLI.

Launch with ``mars gui`` (or ``python -m mars.gui``).  The server
(:mod:`mars.gui.server`) is pure Python standard library; each run is an
ordinary ``python -m mars.cli ...`` subprocess executed in its own job
directory (:mod:`mars.gui.jobs`).  The front end is a static single-page app
(``mars/gui/static``) that loads 3Dmol.js and Plotly from CDNs.
"""

__all__ = ["serve", "make_server"]


def serve(*args, **kwargs):
    """Start the GUI server (see :func:`mars.gui.server.serve`)."""
    from .server import serve as _serve

    return _serve(*args, **kwargs)


def make_server(*args, **kwargs):
    """Create the GUI server without starting it (see :func:`mars.gui.server.make_server`)."""
    from .server import make_server as _make_server

    return _make_server(*args, **kwargs)
