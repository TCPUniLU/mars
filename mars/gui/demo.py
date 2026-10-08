"""Demo mode: finished jobs from published MARS runs, fictional environments, nothing is run."""

import json
import shutil
import time
from pathlib import Path
from typing import List, Optional

from .environments import Environments, families

DATA = Path(__file__).resolve().parents[2] / "examples" / "gui_demo"

JOBS = [
    dict(
        dir="methanol_ir",
        name="Methanol IR spectrum",
        task="ir",
        potential="so3lr",
        input="methanol.xyz",
        argv=["mars", "ir", "methanol.xyz", "--save-structure", "--mode-xyz"],
        minutes=2,
        options={"--save-structure": True, "--mode-xyz": True},
    ),
    dict(
        dir="methanol_ir_md",
        name="Methanol IR spectrum from MD",
        task="ir",
        potential="so3lr",
        input="methanol.xyz",
        argv=["mars", "ir", "methanol.xyz", "--md", "--n-replicas", "10", "-o", "md_ir.dat"],
        minutes=14,
        options={"--md": True, "--n-replicas": 10},
    ),
    dict(
        dir="methoxycyclohexanone_conformers",
        name="2-Methoxycyclohexanone (R) conformers",
        task="conformer",
        potential="so3lr",
        input="R_start.xyz",
        argv=["mars", "R_start.xyz", "--mode", "normal"],
        minutes=41,
        options={"--mode": "normal"},
    ),
    dict(
        dir="ala5_conformers_mace",
        name="Ace-Ala5-NMe conformers (MACE-OFF24)",
        task="conformer",
        potential="mace",
        input="Ace-Ala5-NMe.xyz",
        argv=["mars", "Ace-Ala5-NMe.xyz", "--potential", "mace", "--mace-foundation", "off24"],
        minutes=96,
        options={},
        potential_options={"--mace-foundation": "off24"},
    ),
    dict(
        dir="proline_bcd_nci",
        name="Proline in β-cyclodextrin (NCI mode)",
        task="conformer",
        potential="so3lr",
        input="Pro@bCD.xyz",
        argv=["mars", "Pro@bCD.xyz", "--nci"],
        minutes=188,
        options={"--nci": True},
    ),
    dict(
        dir="alanine_dipeptide_water",
        name="Alanine dipeptide in water",
        task="solvation",
        potential="so3lr",
        input="alanine_dipeptide.xyz",
        argv=["mars", "solvation", "alanine_dipeptide.xyz", "--solvent", "water"],
        minutes=23,
        options={"--solvent": "water"},
    ),
    dict(
        dir="methanol_optimize",
        name="Methanol geometry optimisation",
        task="optimize",
        potential="so3lr",
        input="methanol.xyz",
        argv=["mars", "optimize", "methanol.xyz"],
        minutes=1,
        options={},
    ),
]
TASK_LABELS = {
    "conformer": "Conformer search",
    "optimize": "Geometry optimisation",
    "ir": "IR spectrum",
    "solvation": "Explicit solvation",
    "mtd": "Metadynamics only",
    "md": "Molecular dynamics only",
}


def build(workdir) -> List[str]:
    """Write the demo jobs into ``workdir/jobs`` (finished, with their outputs); returns their ids."""
    jobs_dir = Path(workdir) / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    now, ids = time.time(), []
    for k, j in enumerate(reversed(JOBS)):
        jid = f"demo-{k + 1:02d}-{j['dir'].replace('_', '-')}"
        d = jobs_dir / jid
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(DATA / j["dir"], d)
        finished = now - 3600 * (len(JOBS) - k)
        command = " ".join(j["argv"])
        (d / "command.sh").write_text(f"#!/bin/sh\n{command}\n")
        (d / "run.log").write_text(
            "Demo job: nothing was calculated here.\n"
            f"The outputs in this folder come from a published MARS run ({j['dir']}, see\n"
            "examples/gui_demo/README.md); the command below is how such a run is started.\n\n"
            f"$ {command}\n"
        )
        rec = dict(
            id=jid,
            name=j["name"],
            task=j["task"],
            task_label=TASK_LABELS[j["task"]],
            potential=j["potential"],
            input=j["input"],
            status="done",
            created=finished - 60 * j["minutes"] - 5,
            started=finished - 60 * j["minutes"],
            finished=finished,
            returncode=0,
            argv=j["argv"],
            env={},
            command=command,
            error=None,
            pid=None,
            payload=dict(
                task=j["task"],
                potential=j["potential"],
                options=j.get("options", {}),
                potential_options=j.get("potential_options", {}),
                name=j["name"],
            ),
        )
        (d / "job.json").write_text(json.dumps(rec, indent=1))
        ids.append(jid)
    return ids


_DEMO_ENVS = [
    (
        "mars",
        {
            "core": "ok",
            "so3lr": "ok",
            "mace": "ok",
            "jaxmd_ml": "no_weights",
            "uma": "missing",
            "dxtb": "missing",
            "mm": "missing",
        },
        "1.0.0",
        ["cuda:0"],
    ),
    (
        "mars-dxtb",
        {
            "core": "ok",
            "so3lr": "missing",
            "mace": "missing",
            "jaxmd_ml": "missing",
            "uma": "missing",
            "dxtb": "ok",
            "mm": "missing",
        },
        "1.0.0",
        ["cpu:0"],
    ),
    (
        "mars-mm",
        {
            "core": "ok",
            "so3lr": "ok",
            "mace": "missing",
            "jaxmd_ml": "missing",
            "uma": "missing",
            "dxtb": "missing",
            "mm": "ok",
        },
        "0.1.1",
        ["cuda:0"],
    ),
]


class DemoEnvironments(Environments):
    """Three fictional environments with fixed checks; nothing can be changed or installed."""

    def __init__(self, workdir):
        super().__init__(workdir, server_python="~/miniforge3/envs/mars/bin/python")

    def list(self):
        out = []
        for k, (name, fams, mars, devices) in enumerate(_DEMO_ENVS):
            py = f"~/miniforge3/envs/{name}/bin/python"
            check = {
                "python_version": "3.12.4",
                "mars": mars,
                "devices": devices,
                "full": True,
                "families": {
                    f["id"]: {
                        "status": fams[f["id"]],
                        "missing": [] if fams[f["id"]] != "missing" else f["modules"][:1],
                        "errors": {},
                        "missing_weights": (
                            f.get("weights", [])[:2] if fams[f["id"]] == "no_weights" else []
                        ),
                    }
                    for f in families(demo=True)
                },
            }
            out.append(
                {"name": name, "python": py, "source": "conda", "check": check, "default": k == 0}
            )
        return out

    def default(self):
        return self.list()[0]["python"]

    def check(self, python, full=False):
        env = next((e for e in self.list() if e["python"] == python), None)
        if env is None:
            raise ValueError("unknown environment")
        return env["check"]

    def _refuse(self, *a, **k):
        raise PermissionError("Demo mode: environments cannot be changed and nothing is installed")

    add = set_default = install = _refuse


def demo_workdir(path: Optional[str] = None) -> Path:
    """A fresh directory for a demo session (the given one, or a new temporary one)."""
    import tempfile

    d = Path(path) if path else Path(tempfile.mkdtemp(prefix="mars-gui-demo-"))
    d.mkdir(parents=True, exist_ok=True)
    build(d)
    return d
