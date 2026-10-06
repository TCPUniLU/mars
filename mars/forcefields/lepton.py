"""
Compile OpenMM (Lepton) custom-force expressions into JAX functions.

OpenMM describes every ``Custom*Force`` by an algebraic expression string in
the Lepton language, e.g. ``"k*(d^2 + -25.5*d^3); d=r-r0"``.  Force fields
use these for every term OpenMM has no dedicated class for: all AMOEBA
valence terms, CHARMM harmonic impropers and NBFIX tables, GROMACS
quartic bonds / cosine angles / combination rule 3, and so on.  This module
turns such a string into a pure JAX function of named variables, so MARS can
evaluate (and differentiate) any of them without a hand-written port.

Grammar (as documented in the OpenMM user guide, "Writing Custom
Expressions"): numbers, variables, ``+ - * / ^``, unary minus, parentheses,
function calls, and trailing ``; name = expression`` definitions that may be
used by the main expression and by each other.

Units are those of OpenMM (nm, kJ/mol, radians); conversion to MARS units is
done by the callers in :mod:`mars.forcefields.custom`.
"""

import ast
import re
from typing import Callable, Dict, Mapping

import jax.numpy as jnp

# ----------------------------------------------------------------------------
# Built-in functions
# ----------------------------------------------------------------------------


def _safe_norm(v):
    """Euclidean norm along the last axis with a finite gradient at 0."""
    sq = jnp.sum(v * v, axis=-1)
    safe = jnp.where(sq > 0.0, sq, 1.0)
    return jnp.where(sq > 0.0, jnp.sqrt(safe), 0.0)


def _vangle(a, b):
    """Angle between two vectors in [0, pi], via atan2 for accuracy near 0/pi."""
    return jnp.arctan2(_safe_norm(jnp.cross(a, b)), jnp.sum(a * b, axis=-1))


def _vdihedral(b1, b2, b3):
    """Signed dihedral angle (IUPAC convention, as OpenMM) from bond vectors.

    ``b1 = p2 - p1``, ``b2 = p3 - p2``, ``b3 = p4 - p3``.
    """
    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    x = jnp.sum(n1 * n2, axis=-1)
    y = jnp.sum(jnp.cross(n1, n2) * b2, axis=-1) / jnp.where(
        _safe_norm(b2) > 0.0, _safe_norm(b2), 1.0
    )
    return jnp.arctan2(y, x)


def _pt(*xyz):
    return jnp.stack(xyz, axis=-1)


def pointdistance(x1, y1, z1, x2, y2, z2):
    return _safe_norm(_pt(x1, y1, z1) - _pt(x2, y2, z2))


def pointangle(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    p2 = _pt(x2, y2, z2)
    return _vangle(_pt(x1, y1, z1) - p2, _pt(x3, y3, z3) - p2)


def pointdihedral(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4):
    p1, p2, p3, p4 = _pt(x1, y1, z1), _pt(x2, y2, z2), _pt(x3, y3, z3), _pt(x4, y4, z4)
    return _vdihedral(p2 - p1, p3 - p2, p4 - p3)


def _select(x, y, z):
    return jnp.where(x != 0, y, z)


_FUNCTIONS: Dict[str, Callable] = {
    "sqrt": jnp.sqrt,
    "exp": jnp.exp,
    "log": jnp.log,
    "sin": jnp.sin,
    "cos": jnp.cos,
    "sec": lambda x: 1.0 / jnp.cos(x),
    "csc": lambda x: 1.0 / jnp.sin(x),
    "tan": jnp.tan,
    "cot": lambda x: 1.0 / jnp.tan(x),
    "asin": jnp.arcsin,
    "acos": jnp.arccos,
    "atan": jnp.arctan,
    "atan2": jnp.arctan2,
    "sinh": jnp.sinh,
    "cosh": jnp.cosh,
    "tanh": jnp.tanh,
    "erf": None,  # filled lazily (jax.scipy import)
    "erfc": None,
    "step": lambda x: jnp.where(x >= 0, 1.0, 0.0),
    "delta": lambda x: jnp.where(x == 0, 1.0, 0.0),
    "square": lambda x: x * x,
    "cube": lambda x: x * x * x,
    "recip": lambda x: 1.0 / x,
    "min": jnp.minimum,
    "max": jnp.maximum,
    "abs": jnp.abs,
    "floor": jnp.floor,
    "ceil": jnp.ceil,
    "select": _select,
    "pointdistance": pointdistance,
    "pointangle": pointangle,
    "pointdihedral": pointdihedral,
}


def _builtin(name):
    fn = _FUNCTIONS.get(name)
    if fn is None and name in ("erf", "erfc"):
        from jax.scipy.special import erf, erfc

        _FUNCTIONS["erf"], _FUNCTIONS["erfc"] = erf, erfc
        fn = _FUNCTIONS[name]
    return fn


# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------

_DEF_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)(.*)$", re.S)


def _to_python(expr: str) -> ast.expr:
    """Parse one Lepton expression into a Python AST expression node."""
    # Lepton allows newlines/indentation inside expressions; Python does not.
    text = " ".join(expr.split()).replace("^", "**")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Cannot parse OpenMM expression {expr!r}: {exc}") from exc
    return tree.body


def split_expression(expression: str):
    """Split ``"main; a=...; b=..."`` into ``(main_ast, {name: ast})``."""
    parts = [p for p in expression.split(";") if p.strip()]
    if not parts:
        raise ValueError("Empty OpenMM expression.")
    main = _to_python(parts[0])
    defs = {}
    for part in parts[1:]:
        m = _DEF_RE.match(part)
        if m is None:
            raise ValueError(f"Expected 'name = expression' in {part!r}")
        defs[m.group(1)] = _to_python(m.group(2))
    return main, defs


class CompiledExpression:
    """A Lepton expression compiled to a callable over named JAX arrays.

    Args:
        expression: Lepton expression string.
        functions: Extra functions available to the expression (geometry
            helpers bound to particle positions, tabulated functions, ...).

    Call it with a mapping of variable names to arrays (broadcastable); it
    returns the expression value with the broadcast shape.
    """

    def __init__(self, expression: str, functions: Mapping[str, Callable] | None = None):
        self.expression = expression
        self._main, self._defs = split_expression(expression)
        self._functions = dict(functions or {})
        self.variables = self._free_variables()

    def _free_variables(self):
        names = set()
        called = set()
        for node in [self._main, *self._defs.values()]:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                    called.add(sub.func.id)
                elif isinstance(sub, ast.Name):
                    names.add(sub.id)
        return sorted(names - called - set(self._defs))

    def __call__(self, env: Mapping[str, object]):
        cache: Dict[str, object] = {}
        evaluating = set()

        def lookup(name):
            if name in cache:
                return cache[name]
            if name in self._defs:
                if name in evaluating:
                    raise ValueError(f"Circular definition of {name!r} in {self.expression!r}")
                evaluating.add(name)
                cache[name] = ev(self._defs[name])
                evaluating.discard(name)
                return cache[name]
            if name in env:
                return env[name]
            if name == "pi":
                return jnp.pi
            raise KeyError(f"Unbound variable {name!r} in OpenMM expression {self.expression!r}")

        def ev(node):
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.Name):
                return lookup(node.id)
            if isinstance(node, ast.UnaryOp):
                v = ev(node.operand)
                if isinstance(node.op, ast.USub):
                    return -v
                if isinstance(node.op, ast.UAdd):
                    return v
            if isinstance(node, ast.BinOp):
                a, b = ev(node.left), ev(node.right)
                op = node.op
                if isinstance(op, ast.Add):
                    return a + b
                if isinstance(op, ast.Sub):
                    return a - b
                if isinstance(op, ast.Mult):
                    return a * b
                if isinstance(op, ast.Div):
                    return a / b
                if isinstance(op, ast.Pow):
                    # Integer powers as repeated products: exact and NaN-free
                    # for negative bases (Lepton does the same).
                    if isinstance(b, (int, float)) and float(b).is_integer() and 0 < b <= 8:
                        out = a
                        for _ in range(int(b) - 1):
                            out = out * a
                        return out
                    return a**b
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                name = node.func.id
                fn = self._functions.get(name) or _builtin(name)
                if fn is None:
                    raise ValueError(f"Unsupported function {name!r} in {self.expression!r}")
                return fn(*[ev(a) for a in node.args])
            raise ValueError(
                f"Unsupported syntax {ast.dump(node)} in OpenMM expression {self.expression!r}"
            )

        return ev(self._main)
