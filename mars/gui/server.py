"""
Local HTTP server for the MARS web GUI (Python standard library only).

The server exposes a small JSON API used by the single-page app in
``mars/gui/static/`` and serves that app.  It binds to ``127.0.0.1`` by
default; requests whose ``Host`` header is not a loopback name are refused
when bound to loopback (DNS-rebinding guard), and every state-changing request
must be ``application/json`` (which a cross-site form cannot send without a
CORS preflight that this server never grants).

API (all JSON unless noted)::

    GET    /api/info                      version, workdir, plugins
    GET    /api/envs                      environments (with last checks) + model families
    POST   /api/envs/check                {python, full} -> what that environment provides
    POST   /api/envs/add | default        {python}
    POST   /api/envs/install              {family, python, jax, confirm: true} -> install record
    GET    /api/envs/installs/<id>        install log
    GET    /plugins/<id>/                 an optional section's page (see ``plugins``)
    *      /plugins/<id>/api/...          that section's API
    GET    /api/schema                    task / potential form schema
    POST   /api/upload                    {filename, content | content_b64, kind}
    POST   /api/preview                   job payload -> command (not started)
    GET    /api/jobs                      job list
    POST   /api/jobs                      job payload -> create + queue
    GET    /api/jobs/<id>                 job record + file list
    POST   /api/jobs/<id>/cancel          cancel queued / running job
    POST   /api/jobs/<id>/rerun           clone the job with the same payload
    DELETE /api/jobs/<id>                 delete job and its directory
    GET    /api/jobs/<id>/log?offset=N    incremental log (negative N = tail)
    GET    /api/jobs/<id>/files           output file list
    GET    /api/jobs/<id>/cite[?format=bib] references for the methods and models the job used
    GET    /api/jobs/<id>/files/<path>    raw file (?download=1 for attachment)
    GET    /api/jobs/<id>/zip             whole job directory as a zip
"""

import base64
import binascii
import ipaddress
import json
import mimetypes
import os
import re
import shutil
import socket
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from .. import __version__, citations
from . import plugins, structures
from .environments import GPU_CHOICES, Environments, families
from .jobs import JobRunner, JobStore, PayloadError, build_command, format_command, get_schema

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_WORKDIR = "~/mars-gui-jobs"
DEFAULT_PORT = 8765
MAX_BODY = 200 * 1024 * 1024  # 200 MB upload limit
APP_ID = "mars-gui"

_LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1", "[::1]"}


def _is_loopback(host: str) -> bool:
    if host in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


class GuiApp:
    """State shared by all request handlers: the job store and runner.

    Args:
        workdir: Job / upload directory.
        max_parallel: Concurrent MARS processes.
        launcher: Optional subprocess launcher override (tests).
        python: Interpreter used for MARS jobs (default ``sys.executable``).
        demo: Demo mode (:mod:`mars.gui.demo`): example jobs, fictional
            environments, nothing is run.
    """

    def __init__(self, workdir=DEFAULT_WORKDIR, max_parallel=1, launcher=None, python=None, demo=False):
        self.demo = demo
        if demo:
            from .demo import build

            build(workdir)
        self.store = JobStore(workdir)
        self.runner = JobRunner(self.store, max_parallel=max_parallel, launcher=launcher)
        self.python = python
        if demo:
            from .demo import DemoEnvironments

            self.envs = DemoEnvironments(self.store.root)
        else:
            self.envs = Environments(self.store.root, python or sys.executable)
        if launcher is None and not demo:
            threading.Thread(target=self._check_envs, daemon=True).start()

    def _check_envs(self):
        for e in self.envs.list():
            try:
                self.envs.check(e["python"])
            except Exception:  # pragma: no cover
                pass

    def job_python(self, payload: dict) -> Optional[str]:
        """Interpreter for a job: the environment chosen in the page (must be a known one)."""
        py = payload.get("python")
        if not py:
            return self.python
        if not self.envs.known(py):
            raise ValueError("unknown environment for this job")
        return py

    def job_view(self, job_id: str) -> dict:
        job = self.store.get(job_id)
        job.pop("env", None)
        job["files"] = self.store.list_files(job_id)
        return job


def _make_handler(app: GuiApp, loopback_only: bool):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"MARS-GUI/{__version__}"
        protocol_version = "HTTP/1.1"

        # -- helpers ---------------------------------------------------------

        def log_request(self, code="-", size="-"):  # quiet: only log failed requests
            try:
                failed = int(code) >= 400
            except (TypeError, ValueError):
                failed = False
            if failed:
                self.log_message('"%s" %s', self.requestline, str(code))

        def _send_json(self, obj, status=HTTPStatus.OK):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status, message):
            self._send_json({"error": message}, status)

        def _send_bytes(self, data: bytes, content_type: str, status=HTTPStatus.OK, filename=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            if filename:
                self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{urllib.parse.quote(filename)}")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def _envs(self, method, rest):
            """Models & environments: list, check, add, default, install (+ its log)."""
            if not rest and method == "GET":
                return self._send_json({"envs": app.envs.list(), "families": families(demo=app.demo),
                                        "default": app.envs.default(), "jax_choices": sorted(GPU_CHOICES)})
            if rest[:1] == ["installs"] and len(rest) == 2 and method == "GET":
                return self._send_json(app.envs.install_log(rest[1]))
            if method != "POST" or len(rest) != 1:
                raise FileNotFoundError("/api/envs/" + "/".join(rest))
            body = self._read_json()
            action = rest[0]
            if action == "check":
                return self._send_json(app.envs.check(body["python"], full=bool(body.get("full"))))
            if action == "add":
                return self._send_json(app.envs.add(body["python"]))
            if action == "default":
                app.envs.set_default(body["python"])
                return self._send_json({"default": body["python"]})
            if action == "install":
                if body.get("confirm") is not True:
                    raise ValueError("installation not confirmed")
                return self._send_json(app.envs.install(body["family"], body["python"], body.get("jax", "cuda12")))
            raise FileNotFoundError("/api/envs/" + action)

        def _plugin(self, method, path):
            """``/plugins/<id>/`` (the section page) and ``/plugins/<id>/api/...`` (its API)."""
            parts = [x for x in path[len("/plugins/"):].split("/") if x]
            plugin = plugins.discover().get(parts[0]) if parts else None
            if plugin is None:
                raise FileNotFoundError(path)
            if len(parts) == 1 and method == "GET":
                return self._send_bytes(plugin.page().encode(), "text/html; charset=utf-8")
            if len(parts) >= 2 and parts[1] == "api":
                body = self._read_json() if method == "POST" else None
                status, ctype, data = plugin.handle(method, "/".join(parts[2:]), body)
                if isinstance(data, (dict, list)):
                    return self._send_json(data, status)
                return self._send_bytes(data, ctype, status)
            raise FileNotFoundError(path)

        def _send_file(self, path: Path, download=False, content_type=None):
            ctype = content_type or mimetypes.guess_type(path.name)[0] or "text/plain"
            if ctype.startswith("text/") or ctype in ("application/javascript",):
                ctype += "; charset=utf-8"
            size = path.stat().st_size
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if download:
                fname = urllib.parse.quote(path.name)
                self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{fname}")
            self.end_headers()
            with open(path, "rb") as fh:
                shutil.copyfileobj(fh, self.wfile)

        def _read_json(self):
            ctype = self.headers.get("Content-Type", "")
            if not ctype.startswith("application/json"):
                raise PermissionError("Content-Type must be application/json")
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise ValueError(f"request too large (limit {MAX_BODY // 2**20} MB)")
            raw = self.rfile.read(length) if length else b"{}"
            try:
                return json.loads(raw or b"{}")
            except ValueError:
                raise ValueError("invalid JSON body")

        def _check_host(self) -> bool:
            if not loopback_only:
                return True
            host = (self.headers.get("Host") or "").strip()
            hostname = (
                host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
            )
            if _is_loopback(hostname):
                return True
            self._error(HTTPStatus.FORBIDDEN, "host not allowed")
            return False

        def _dispatch(self, method):
            if not self._check_host():
                return
            url = urllib.parse.urlsplit(self.path)
            path = urllib.parse.unquote(url.path)
            qs = urllib.parse.parse_qs(url.query)
            try:
                self._route(method, path, qs)
            except PermissionError as exc:
                self._error(HTTPStatus.FORBIDDEN, str(exc))
            except (KeyError, FileNotFoundError) as exc:
                self._error(HTTPStatus.NOT_FOUND, f"not found: {exc}")
            except (PayloadError, structures.StructureError, ValueError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:  # pragma: no cover - last-resort 500
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        def do_DELETE(self):
            self._dispatch("DELETE")

        # -- routing ---------------------------------------------------------

        def _route(self, method, path, qs):
            if method == "GET" and path in ("/", "/index.html"):
                return self._send_file(STATIC_DIR / "index.html", content_type="text/html")
            if method == "GET" and path.startswith("/static/"):
                from .jobs import safe_join

                f = safe_join(STATIC_DIR, path[len("/static/") :])
                if not f.is_file():
                    raise FileNotFoundError(path)
                return self._send_file(f)
            if path.startswith("/plugins/"):
                return self._plugin(method, path)
            if not path.startswith("/api/"):
                raise FileNotFoundError(path)
            parts = [p for p in path[len("/api/") :].split("/") if p]
            if not parts:
                raise FileNotFoundError(path)

            head = parts[0]
            if app.demo and head == "jobs" and method in ("POST", "DELETE"):
                raise PermissionError("Demo mode: calculations are not run and jobs cannot be changed")
            if head == "info" and method == "GET":
                return self._send_json(
                    {
                        "app": APP_ID,
                        "version": __version__,
                        "demo": app.demo,
                        "workdir": "~/mars-gui-jobs" if app.demo else str(app.store.root),
                        "python": "python" if app.demo else (app.python or sys.executable),
                        "pid": os.getpid(),
                        "plugins": plugins.listing(),
                    }
                )
            if head == "schema" and method == "GET":
                return self._send_json(get_schema())
            if head == "upload" and method == "POST":
                return self._upload()
            if head == "preview" and method == "POST":
                payload = self._read_json()
                cmd = build_command(payload, "input.xyz", python="python")
                return self._send_json({"command": format_command(cmd["argv"], cmd["env"])})
            if head == "envs":
                return self._envs(method, parts[1:])
            if head != "jobs":
                raise FileNotFoundError(path)

            if len(parts) == 1:
                if method == "GET":
                    jobs = app.store.list_jobs()
                    for j in jobs:
                        j.pop("env", None)
                        j.pop("payload", None)
                    return self._send_json({"jobs": jobs, "now": time.time()})
                if method == "POST":
                    payload = self._read_json()
                    job = app.store.create_job(payload, python=app.job_python(payload))
                    app.runner.submit(job["id"])
                    return self._send_json(app.job_view(job["id"]), HTTPStatus.CREATED)
                raise FileNotFoundError(path)

            job_id = parts[1]
            app.store.get(job_id)  # validates id, KeyError -> 404
            rest = parts[2:]
            if not rest:
                if method == "GET":
                    return self._send_json(app.job_view(job_id))
                if method == "DELETE":
                    if app.runner.is_active(job_id) or app.store.get(job_id)["status"] in (
                        "queued",
                        "running",
                    ):
                        app.runner.cancel(job_id)
                        _wait_inactive(app.runner, job_id, timeout=10)
                    app.store.delete(job_id)
                    return self._send_json({"deleted": job_id})
                raise FileNotFoundError(path)
            action = rest[0]
            if action == "cancel" and method == "POST":
                self._read_json()
                return self._send_json(app.runner.cancel(job_id))
            if action == "rerun" and method == "POST":
                self._read_json()
                payload = app.store.get(job_id).get("payload") or {}
                job = app.store.create_job(payload, python=app.job_python(payload))
                app.runner.submit(job["id"])
                return self._send_json(app.job_view(job["id"]), HTTPStatus.CREATED)
            if action == "log" and method == "GET":
                offset = int((qs.get("offset") or ["0"])[0])
                log = app.store.read_log(job_id, offset)
                log["status"] = app.store.get(job_id)["status"]
                return self._send_json(log)
            if action == "files" and method == "GET":
                if len(rest) == 1:
                    return self._send_json({"files": app.store.list_files(job_id)})
                f = app.store.job_file(job_id, "/".join(rest[1:]))
                if f.suffix == ".log":
                    return self._send_bytes(strip_ansi(f.read_bytes()), "text/plain; charset=utf-8",
                                            filename=f.name if (qs.get("download") or ["0"])[0] == "1" else None)
                return self._send_file(f, download=(qs.get("download") or ["0"])[0] == "1")
            if action == "zip" and method == "GET":
                return self._send_zip(job_id)
            if action == "cite" and method == "GET":
                job_payload = app.store.get(job_id).get("payload") or {}
                keys = citations.for_job(job_payload)
                notes = citations.notes_for_job(job_payload)
                if (qs.get("format") or [""])[0] == "bib":
                    bib = "".join(f"% {n}\n" for n in notes) + citations.bibtex(keys)
                    return self._send_bytes(bib.encode(), "application/x-bibtex; charset=utf-8")
                return self._send_json({"refs": [citations.format_ref(k) for k in keys], "notes": notes})
            raise FileNotFoundError(path)

        # -- endpoints -------------------------------------------------------

        def _upload(self):
            body = self._read_json()
            filename = body.get("filename") or "upload.xyz"
            if body.get("content_b64") is not None:
                try:
                    data = base64.b64decode(body["content_b64"], validate=True)
                except (binascii.Error, ValueError):
                    raise ValueError("content_b64 is not valid base64")
            else:
                data = str(body.get("content") or "").encode()
            if not data.strip():
                raise ValueError("empty file")
            kind = body.get("kind", "auto")
            if kind not in ("auto", "structure", "file"):
                raise ValueError("kind must be auto, structure or file")
            rec = app.store.save_upload(filename, data, kind=kind)
            return self._send_json(rec, HTTPStatus.CREATED)

        def _send_zip(self, job_id):
            base = app.store.job_dir(job_id)
            with tempfile.TemporaryFile() as tmp:
                with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
                    for p in sorted(base.rglob("*")):
                        if p.is_file() and not p.is_symlink():
                            arc = f"{job_id}/{p.relative_to(base).as_posix()}"
                            if p.suffix == ".log":
                                zf.writestr(arc, strip_ansi(p.read_bytes()))
                            else:
                                zf.write(p, arc)
                size = tmp.tell()
                tmp.seek(0)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Length", str(size))
                self.send_header("Content-Disposition", f'attachment; filename="mars-{job_id}.zip"')
                self.end_headers()
                shutil.copyfileobj(tmp, self.wfile)

    return Handler


def strip_ansi(data: bytes) -> bytes:
    """Log bytes without terminal colour codes (for downloads)."""
    return re.sub(rb"\x1b\[[0-9;]*m", b"", data)


def _wait_inactive(runner: JobRunner, job_id: str, timeout: float):
    deadline = time.time() + timeout
    while runner.is_active(job_id) and time.time() < deadline:
        time.sleep(0.1)


def make_server(
    host="127.0.0.1",
    port=DEFAULT_PORT,
    workdir=DEFAULT_WORKDIR,
    max_parallel=1,
    launcher=None,
    python=None,
    demo=False,
):
    """Create (but do not start) the GUI HTTP server.

    Args:
        host: Bind address (default loopback only).
        port: TCP port (``0`` = pick a free one).
        workdir: Job / upload directory.
        max_parallel: Concurrent MARS processes.
        launcher: Subprocess launcher override (tests).
        python: Interpreter for MARS jobs.
        demo: Demo mode (see :class:`GuiApp`).

    Returns:
        ``(server, app)`` — a :class:`ThreadingHTTPServer` and its :class:`GuiApp`.
    """
    app = GuiApp(workdir, max_parallel=max_parallel, launcher=launcher, python=python, demo=demo)
    handler = _make_handler(app, loopback_only=_is_loopback(host))
    server_cls = ThreadingHTTPServer
    if ":" in host:

        class _V6(ThreadingHTTPServer):
            address_family = socket.AF_INET6

        server_cls = _V6
    server = server_cls((host, port), handler)
    server.daemon_threads = True
    return server, app


def _existing_gui(host: str, port: int) -> bool:
    """True if a MARS GUI already answers on host:port."""
    h = "127.0.0.1" if host in ("0.0.0.0", "") else host
    try:
        with urllib.request.urlopen(f"http://{h}:{port}/api/info", timeout=1.5) as resp:
            return json.loads(resp.read()).get("app") == APP_ID
    except Exception:
        return False


def _url(host: str, port: int) -> str:
    h = "127.0.0.1" if host in ("0.0.0.0", "", "::") else host
    if ":" in h and not h.startswith("["):
        h = f"[{h}]"
    return f"http://{h}:{port}/"


def serve(
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    workdir: str = DEFAULT_WORKDIR,
    open_browser: bool = True,
    max_parallel: int = 1,
    demo: bool = False,
) -> int:
    """Start the GUI server and block until interrupted.

    If a MARS GUI is already running on *port* the browser is simply pointed
    at it (so double-clicking the desktop launcher twice is harmless); if the
    port is taken by something else the next free port is used.

    Returns:
        Process exit code.
    """
    server = None
    for p in range(port, port + 20):
        try:
            server, app = make_server(host, p, workdir, max_parallel=max_parallel, demo=demo)
            break
        except OSError:
            if _existing_gui(host, p):
                url = _url(host, p)
                print(f"MARS GUI already running at {url}")
                if open_browser:
                    webbrowser.open(url)
                return 0
            if port == 0:
                raise
    if server is None:
        print(f"mars gui: error: no free port in {port}-{port + 19}", file=sys.stderr)
        return 1
    actual_port = server.server_address[1]
    url = _url(host, actual_port)
    print(f"MARS GUI {__version__} serving at {url}")
    print(f"  work directory: {app.store.root}")
    if not _is_loopback(host):
        print(
            "  WARNING: listening on a non-loopback address — anyone who can reach this "
            "port can run MARS jobs as you.",
            file=sys.stderr,
        )
    print("  press Ctrl+C to stop")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nstopping MARS GUI")
    finally:
        server.server_close()
    return 0


DESKTOP_TEMPLATE = """[Desktop Entry]
Type=Application
Version=1.0
Name=MARS GUI
GenericName=Molecular simulation GUI
Comment=Run MARS (SO3LR, conformer search, IR, solvation) from the web browser
Exec={exec_line}
Icon={icon}
Terminal=false
Categories=Science;Chemistry;Education;
Keywords=MARS;SO3LR;chemistry;conformer;IR;
StartupNotify=false
"""


def desktop_entry(workdir: Optional[str] = None, port: int = DEFAULT_PORT) -> str:
    """Return a ``.desktop`` launcher for the current interpreter / environment."""
    exec_parts = []
    pythonpath = os.environ.get("PYTHONPATH")
    jax_platforms = os.environ.get("JAX_PLATFORMS")
    env_bits = []
    if pythonpath:
        env_bits.append(f"PYTHONPATH={pythonpath}")
    if jax_platforms:
        env_bits.append(f"JAX_PLATFORMS={jax_platforms}")
    if env_bits:
        exec_parts += ["env"] + env_bits
    exec_parts += [sys.executable, "-m", "mars.gui", "--port", str(port)]
    if workdir:
        exec_parts += ["--workdir", str(Path(workdir).expanduser())]
    exec_line = " ".join(_desktop_quote(p) for p in exec_parts)
    icon = STATIC_DIR / "app-icon.svg"
    return DESKTOP_TEMPLATE.format(exec_line=exec_line, icon=icon)


def _desktop_quote(arg: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./=:+-]+", arg):
        return arg
    escaped = arg.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`").replace("$", "\\$")
    return f'"{escaped}"'


def install_desktop(workdir: Optional[str] = None, port: int = DEFAULT_PORT) -> Path:
    """Write ``~/.local/share/applications/mars-gui.desktop`` (Linux app menu entry)."""
    apps = Path(os.environ.get("XDG_DATA_HOME", "~/.local/share")).expanduser() / "applications"
    apps.mkdir(parents=True, exist_ok=True)
    target = apps / "mars-gui.desktop"
    target.write_text(desktop_entry(workdir, port))
    target.chmod(0o755)
    return target


def add_arguments(parser):
    """Add the ``mars gui`` options to an argparse parser."""
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"TCP port (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1 — local only; 0.0.0.0 exposes it to the network)",
    )
    parser.add_argument(
        "--workdir",
        default=os.environ.get("MARS_GUI_WORKDIR", DEFAULT_WORKDIR),
        help=f"Directory for uploads and job folders (default: {DEFAULT_WORKDIR})",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open the web browser automatically"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Number of MARS jobs run concurrently (default: 1)",
    )
    parser.add_argument(
        "--install-desktop",
        action="store_true",
        help="Install a clickable Linux desktop launcher (~/.local/share/applications) and exit",
    )
    parser.add_argument(
        "--print-desktop",
        action="store_true",
        help="Print a .desktop launcher for this environment and exit",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Demo: finished example jobs from published MARS runs, fictional environments; "
        "nothing is run (uses a new temporary job directory unless --workdir is given)",
    )
    return parser


def run_from_args(args) -> int:
    """Entry point shared by ``mars gui`` and ``python -m mars.gui``."""
    workdir = None if args.workdir == DEFAULT_WORKDIR else args.workdir
    if getattr(args, "print_desktop", False):
        print(desktop_entry(workdir, args.port), end="")
        return 0
    if getattr(args, "install_desktop", False):
        target = install_desktop(workdir, args.port)
        print(f"installed desktop launcher: {target}")
        print("It now appears in your application menu as 'MARS GUI'.")
        return 0
    if getattr(args, "demo", False):
        from .demo import demo_workdir

        workdir = str(demo_workdir(workdir))
        print(f"MARS GUI demo: example jobs in {workdir} (nothing is run)")
    else:
        workdir = args.workdir
    return serve(
        host=args.host,
        port=args.port,
        workdir=workdir,
        open_browser=not args.no_browser,
        max_parallel=args.max_parallel,
        demo=getattr(args, "demo", False),
    )
