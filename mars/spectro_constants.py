"""JAX-free spectroscopy constants shared by :mod:`mars.ir` and :mod:`mars.viewer`.

Kept dependency-free (no JAX / numpy) so the lightweight ``mars viewer`` can use
them without importing the ML-potential stack.
"""

# Colors for vibrational mode classification, used both in the static IR plots
# (mars.ir) and in the interactive viewer (mars.viewer).
MODE_TYPE_COLORS = {
    "stretching": "#E63946",
    "bending": "#457B9D",
    "torsion": "#2A9D8F",
    "mixed": "#888888",
}
