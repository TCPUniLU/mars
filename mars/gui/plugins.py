"""Optional GUI sections, found by module name (``KNOWN``) or the ``mars.gui.plugins`` entry point."""

import importlib
import importlib.util
from typing import Dict, List

KNOWN = ("mars.landscape.gui_plugin",)
_CACHE: Dict[str, object] = {}


def discover() -> Dict[str, object]:
    """``{id: plugin}`` of every loadable plugin (cached)."""
    if _CACHE:
        return _CACHE
    found: List[object] = []
    for name in KNOWN:
        try:
            if importlib.util.find_spec(name) is None:
                continue
        except ModuleNotFoundError:
            continue
        try:
            found.append(importlib.import_module(name).PLUGIN)
        except Exception as exc:  # pragma: no cover - a broken plugin must not break the GUI
            print(f"[gui] plugin {name} skipped: {type(exc).__name__}: {exc}")
    try:
        from importlib.metadata import entry_points

        for ep in entry_points(group="mars.gui.plugins"):
            try:
                found.append(ep.load())
            except Exception as exc:  # pragma: no cover
                print(f"[gui] plugin {ep.name} skipped: {type(exc).__name__}: {exc}")
    except Exception:  # pragma: no cover - old importlib.metadata
        pass
    for p in sorted(found, key=lambda p: getattr(p, "order", 100)):
        _CACHE[p.id] = p
    return _CACHE


def listing() -> List[dict]:
    """What the page needs to build the sidebar entries."""
    return [{"id": p.id, "title": p.title} for p in discover().values()]
