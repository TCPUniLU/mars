"""Tests for mars/gui/environments.py and the GUI's environment API."""

import json
import sys

import pytest

from mars.gui import environments as envmod
from mars.gui.jobs import get_schema


def test_every_gui_potential_has_a_family():
    names = [p["name"] for p in get_schema()["potentials"]]
    missing = [n for n in names if envmod.family_of(n) is None]
    assert not missing, f"potentials without an install family: {missing}"


def test_quick_check_of_this_interpreter():
    c = envmod.probe(sys.executable)
    assert "error" not in c and c["families"]["core"]["status"] == "ok"
    assert set(c["families"]) == {f["id"] for f in envmod.families()}
    assert c["python_version"] == sys.version.split()[0]


def test_install_is_validated_and_never_takes_commands(tmp_path, monkeypatch):
    envs = envmod.Environments(tmp_path, sys.executable)
    monkeypatch.setattr(envmod.Environments, "_conda_envs", staticmethod(lambda: []))
    with pytest.raises(ValueError, match="unknown model family"):
        envs.install("rm -rf /", sys.executable)
    with pytest.raises(ValueError, match="unknown environment"):
        envs.install("so3lr", "/usr/bin/false")
    started = []
    monkeypatch.setattr(
        envmod.threading,
        "Thread",
        lambda target, args, daemon: type("T", (), {"start": lambda self: started.append(args)})(),
    )
    rec = envs.install("mm", sys.executable)
    assert started and all(a[0] == sys.executable for a in rec["commands"])
    assert rec["commands"][0][1:4] == ["-m", "pip", "install"]


def test_add_default_and_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(envmod.Environments, "_conda_envs", staticmethod(lambda: []))
    envs = envmod.Environments(tmp_path, sys.executable)
    with pytest.raises(ValueError):
        envs.add(str(tmp_path / "nope"))
    envs.set_default(sys.executable)
    assert envs.default() == sys.executable
    assert json.loads((tmp_path / "gui_settings.json").read_text())["default"] == sys.executable


def test_job_environment_must_be_known(tmp_path, monkeypatch):
    from mars.gui.server import GuiApp

    monkeypatch.setattr(envmod.Environments, "_conda_envs", staticmethod(lambda: []))
    app = GuiApp(tmp_path, launcher=lambda *a, **k: None)
    assert app.job_python({}) is None or app.job_python({}) == app.python
    assert app.job_python({"python": sys.executable}) == sys.executable
    with pytest.raises(ValueError, match="unknown environment"):
        app.job_python({"python": "/bin/sh"})


def test_citations_match_the_docs_and_cover_every_potential():
    from pathlib import Path

    from mars import citations

    doc = Path(__file__).resolve().parents[1] / "docs" / "about" / "citation.md"
    if doc.exists():
        assert citations.entries_from_markdown(doc.read_text()) == citations.BIB
    for p in get_schema()["potentials"]:
        keys = citations.for_job({"task": "conformer", "potential": p["name"]})
        assert keys[0] == "mars2026" and all(k in citations.BIB for k in keys)
    ref = citations.format_ref("kabylda2025so3lr")
    assert ref["doi"] == "10.1021/jacs.5c09558" and "2025" in ref["text"]


def test_demo_is_read_only_and_shows_nothing_local(tmp_path):
    """The demo runs nothing and exposes no local path or interpreter."""
    import os
    import threading
    import urllib.error
    import urllib.request
    from pathlib import Path

    from mars.gui import demo
    from mars.gui.server import make_server

    if not demo.DATA.exists():
        pytest.skip("examples/gui_demo not available")
    server, app = make_server("127.0.0.1", 0, str(tmp_path), demo=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"

    def get(path):
        with urllib.request.urlopen(url + path) as r:
            return r.read().decode()

    def post(path, body):
        req = urllib.request.Request(
            url + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        try:
            urllib.request.urlopen(req)
            return 200
        except urllib.error.HTTPError as e:
            return e.code

    try:
        info, envs, jobs = (
            json.loads(get("/api/info")),
            get("/api/envs"),
            json.loads(get("/api/jobs"))["jobs"],
        )
        assert (
            info["demo"]
            and len(jobs) == len(demo.JOBS)
            and all(j["status"] == "done" for j in jobs)
        )
        assert {e["name"] for e in json.loads(envs)["envs"]} == {"mars", "mars-dxtb", "mars-mm"}
        everything = (
            json.dumps(info)
            + envs
            + json.dumps(jobs)
            + "".join(get(f"/api/jobs/{j['id']}") for j in jobs)
        )
        for private in (str(Path.home()), str(tmp_path), sys.executable, os.getcwd()):
            assert private not in everything, f"demo leaks {private}"
        ir_id = next(j["id"] for j in jobs if j["name"] == "Methanol IR spectrum")
        ir = json.loads(get(f"/api/jobs/{ir_id}"))
        assert {"spectrum", "modes", "structure"} <= {f["kind"] for f in ir["files"]}
        assert post("/api/jobs", {"task": "ir"}) == 403
        assert post("/api/envs/install", {"family": "mm", "python": "x", "confirm": True}) == 403
    finally:
        server.shutdown()
        server.server_close()
