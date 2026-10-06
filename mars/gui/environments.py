"""Python environments and model families for the GUI: discovery, checks and installs."""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import mars

GPU_CHOICES = {"cuda12": 'jax[cuda12]<0.11', "cuda13": 'jax[cuda13]<0.11', "cpu": 'jax<0.11'}


def _mars_spec(extra: str, demo: bool = False) -> str:
    """pip spec of MARS with an extra (editable from this source tree, else from GitHub)."""
    root = Path(mars.__file__).resolve().parents[1]
    if (root / "pyproject.toml").exists() and not demo:
        return f"-e {shlex.quote(str(root))}[{extra}]" if extra else f"-e {shlex.quote(str(root))}"
    url = "git+https://github.com/TCPUniLU/mars.git"
    return f'"mars[{extra}] @ {url}"' if extra else f'"mars @ {url}"'


def families(demo: bool = False) -> List[dict]:
    """The installable model families (commands are pip / python lines run in the environment)."""
    _spec = lambda extra: _mars_spec(extra, demo)  # noqa: E731
    return [
        dict(id="core", label="MARS core (JAX)", potentials=["harmonic", "lj"],
             modules=["mars", "jax", "jax_md"],
             about="MARS itself with JAX. Pick the JAX build for your GPU (CUDA 12 or 13) or CPU.",
             install=['pip install -U "{jax}"', f"pip install {_spec('')}"]),
        dict(id="so3lr", label="SO3LR", potentials=["so3lr"], modules=["so3lr"],
             about="Default potential; also gives the partial charges for IR intensities.",
             install=[f"pip install {_spec('so3lr')}",
                      "pip install git+https://github.com/general-molecular-simulations/so3lr.git"]),
        dict(id="mace", label="MACE", potentials=["mace"], modules=["mace", "mace_jax"],
             about="MACE foundation models, run JAX-native (converted once from PyTorch).",
             install=[f"pip install {_spec('mace')}", "pip install git+https://github.com/ACEsuit/mace-jax"]),
        dict(id="jaxmd_ml", label="jax-md ML models (ORB, AIMNet2, AceFF, ANI)",
             potentials=["orb", "aimnet2", "aceff", "ani"], modules=["jax_md._nn", "equinox"],
             weights=["orb-v3-conservative-omol.eqx", "aimnet2.eqx", "aceff_v2.0.eqx", "ani2x_ensemble.eqx"],
             about="Ports in jax-md main; their weights are downloaded separately.",
             install=['pip install "jax-md @ git+https://github.com/jax-md/jax-md.git"',
                      f"pip install {_spec('jaxmd-ml')}",
                      "python -m jax_md._nn.weights orb", "python -m jax_md._nn.weights aimnet2",
                      "python -m jax_md._nn.weights aceff", "python -m jax_md._nn.weights ani"]),
        dict(id="uma", label="UMA (FAIR-Chem)", potentials=["uma"],
             modules=["jax_md._nn.uma", "torch", "huggingface_hub"],
             about="Meta's UMA via jax-md. The weights are gated behind a licence.",
             install=['pip install "jax-md @ git+https://github.com/jax-md/jax-md.git"', f"pip install {_spec('uma')}"],
             manual=["Accept the licence at https://huggingface.co/facebook/UMA",
                     "Run `huggingface-cli login` once in this environment"]),
        dict(id="dxtb", label="dxtb (GFN1/GFN2-xTB)", potentials=["dxtb"], modules=["dxtb"],
             about="PyTorch tight binding. It conflicts with parts of the JAX stack: "
                   "use a separate environment.",
             install=['pip install "dxtb[libcint]>=0.4.0"']),
        dict(id="mm", label="Classical force fields (AMBER, CHARMM, OPLS-AA, AMOEBA)",
             potentials=["amber", "charmm", "opls", "amoeba", "mm"], modules=["openmm", "parmed"],
             about="Reads your force-field files with OpenMM / ParmEd.",
             install=[f"pip install {_spec('mm')}"]),
    ]


def family_of(potential: str) -> Optional[str]:
    return next((f["id"] for f in families() if potential in f["potentials"]), None)


_PROBE = r"""
import importlib, importlib.util, json, os, sys
spec = json.loads(sys.argv[1]); full = sys.argv[2] == "1"
out = {"python_version": sys.version.split()[0], "executable": sys.executable, "families": {}}
try:
    import importlib.metadata as md
    out["mars"] = md.version("mars")
except Exception:
    out["mars"] = None
cache = os.path.expanduser("~/.cache/jax_md")
for fam in spec:
    missing, errors = [], {}
    for m in fam["modules"]:
        try:
            found = importlib.util.find_spec(m) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(m)
        elif full:
            try:
                importlib.import_module(m)
            except Exception as exc:
                errors[m] = f"{type(exc).__name__}: {exc}"[:300]
    weights = [w for w in fam.get("weights", []) if not any(
        os.path.exists(os.path.join(r, w)) for r, _, fs in os.walk(cache) if w in fs)] if fam.get("weights") else []
    out["families"][fam["id"]] = {"missing": missing, "errors": errors, "missing_weights": weights}
if full:
    try:
        import jax
        out["devices"] = [str(d) for d in jax.devices()]
    except Exception as exc:
        out["devices_error"] = f"{type(exc).__name__}: {exc}"[:300]
print("MARSPROBE" + json.dumps(out))
"""


def probe(python: str, full: bool = False, timeout: float = 120.0) -> dict:
    """Per-family status (ok, missing, broken, no_weights, no_mars) of an interpreter."""
    spec = [{"id": f["id"], "modules": f["modules"], "weights": f.get("weights", [])} for f in families()]
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory() as neutral:
            r = subprocess.run([python, "-c", _PROBE, json.dumps(spec), "1" if full else "0"], capture_output=True,
                               text=True, timeout=timeout, cwd=neutral,
                               env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"python": python, "error": f"{type(exc).__name__}: {exc}"}
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("MARSPROBE")), None)
    if line is None:
        return {"python": python, "error": (r.stderr.strip().splitlines() or ["no output"])[-1][:300]}
    out = json.loads(line[len("MARSPROBE"):])
    core_ok = not out["families"]["core"]["missing"] and not out["families"]["core"]["errors"]
    for fid, fam in out["families"].items():
        fam["status"] = ("missing" if fam["missing"] else "broken" if fam["errors"]
                         else "no_weights" if fam["missing_weights"] else "ok")
        if fid != "core" and fam["status"] == "ok" and not core_ok:
            fam["status"] = "no_mars"
    out.update(python=python, full=full, seconds=round(time.time() - t0, 1), checked=time.time())
    return out


class Environments:
    """Known interpreters, their last checks, the default one, and running installs."""

    def __init__(self, workdir: Path, server_python: Optional[str] = None):
        self.settings_path = Path(workdir) / "gui_settings.json"
        self.server_python = server_python or sys.executable
        self.checks: Dict[str, dict] = {}
        self.installs: Dict[str, dict] = {}
        self.lock = threading.Lock()
        self.install_dir = Path(workdir) / "installs"

    def _settings(self) -> dict:
        try:
            return json.loads(self.settings_path.read_text())
        except (OSError, ValueError):
            return {}

    def _save_settings(self, s: dict) -> None:
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings_path.write_text(json.dumps(s, indent=1))

    @staticmethod
    def _conda_envs() -> List[dict]:
        exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
        if not exe:
            return []
        try:
            r = subprocess.run([exe, "env", "list", "--json"], capture_output=True, text=True, timeout=30)
            prefixes = json.loads(r.stdout).get("envs", [])
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return []
        out = []
        for pre in prefixes:
            py = Path(pre) / ("python.exe" if os.name == "nt" else "bin/python")
            if py.exists():
                name = Path(pre).name if Path(pre).parent.name == "envs" else "base"
                out.append({"name": name, "python": str(py), "source": "conda"})
        return out

    def list(self) -> List[dict]:
        """Every known environment with its last check (if any) and the default flag."""
        s = self._settings()
        envs, seen = [], set()
        candidates = ([{"name": "this GUI's environment", "python": self.server_python, "source": "server"}]
                      + self._conda_envs()
                      + [{"name": Path(p).parents[1].name, "python": p, "source": "added"} for p in s.get("added", [])])
        for e in candidates:
            key = os.path.realpath(e["python"])
            if key in seen:
                continue
            seen.add(key)
            envs.append(dict(e, check=self.checks.get(e["python"]),
                             default=e["python"] == s.get("default", self.server_python)))
        return envs

    def known(self, python: str) -> bool:
        return any(e["python"] == python for e in self.list())

    def add(self, python: str) -> dict:
        p = Path(python).expanduser()
        if p.is_dir():
            p = p / ("python.exe" if os.name == "nt" else "bin/python")
        if not (p.is_file() and os.access(p, os.X_OK)):
            raise ValueError(f"{p} is not a Python interpreter")
        s = self._settings()
        s.setdefault("added", [])
        if str(p) not in s["added"]:
            s["added"].append(str(p))
        self._save_settings(s)
        return self.check(str(p))

    def set_default(self, python: str) -> None:
        if not self.known(python):
            raise ValueError("unknown environment")
        s = self._settings()
        s["default"] = python
        self._save_settings(s)

    def default(self) -> str:
        return self._settings().get("default", self.server_python)

    def check(self, python: str, full: bool = False) -> dict:
        if not self.known(python) and python != self.server_python:
            raise ValueError("unknown environment")
        res = probe(python, full=full)
        with self.lock:
            self.checks[python] = res
        return res

    def pick(self, potential: str) -> Optional[str]:
        """An environment whose last check has the potential's family installed (default first)."""
        fam = family_of(potential)
        order = sorted(self.list(), key=lambda e: not e["default"])
        for e in order:
            c = e.get("check")
            if c and "families" in c and (fam is None or c["families"].get(fam, {}).get("status") == "ok"):
                return e["python"]
        return None

    def install(self, family: str, python: str, jax: str = "cuda12") -> dict:
        """Run a family's install commands in ``python`` (background thread, logged)."""
        fam = next((f for f in families() if f["id"] == family), None)
        if fam is None:
            raise ValueError(f"unknown model family {family!r}")
        if not self.known(python):
            raise ValueError("unknown environment")
        if jax not in GPU_CHOICES:
            raise ValueError(f"jax must be one of {sorted(GPU_CHOICES)}")
        with self.lock:
            if any(i["status"] == "running" for i in self.installs.values()):
                raise ValueError("another installation is running; wait for it to finish")
            iid = f"{time.strftime('%Y%m%d-%H%M%S')}-{family}"
            self.install_dir.mkdir(parents=True, exist_ok=True)
            rec = {"id": iid, "family": family, "python": python, "status": "running",
                   "log": str(self.install_dir / f"{iid}.log"), "commands": []}
            for line in fam["install"]:
                line = line.replace("{jax}", GPU_CHOICES[jax])
                words = shlex.split(line)
                if words[0] == "pip":
                    argv = [python, "-m", "pip", *words[1:]]
                elif words[0] == "python":
                    argv = [python, *words[1:]]
                else:
                    raise ValueError(f"unexpected install command {line!r}")
                rec["commands"].append(argv)
            self.installs[iid] = rec
        threading.Thread(target=self._run_install, args=(rec,), daemon=True).start()
        return {k: v for k, v in rec.items()}

    def _run_install(self, rec: dict) -> None:
        with open(rec["log"], "w") as log:
            for argv in rec["commands"]:
                log.write("$ " + " ".join(shlex.quote(a) for a in argv) + "\n")
                log.flush()
                p = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
                if p.returncode != 0:
                    log.write(f"\n[failed with exit code {p.returncode}]\n")
                    rec["status"] = "failed"
                    return
            log.write("\n[done]\n")
        rec["status"] = "done"
        try:
            self.check(rec["python"])
        except Exception:  # pragma: no cover
            pass

    def install_log(self, iid: str) -> dict:
        rec = self.installs.get(iid)
        if rec is None:
            raise KeyError(iid)
        text = Path(rec["log"]).read_text(errors="replace") if Path(rec["log"]).exists() else ""
        return {"id": iid, "status": rec["status"], "log": text[-20000:]}
