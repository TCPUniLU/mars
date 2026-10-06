# `mars gui`

Start the MARS graphical interface: a local web server and a page in your
browser. See [Graphical interface](../gui/index.md) for a tour.

```bash
mars gui [OPTIONS]
python -m mars.gui [OPTIONS]      # same thing
```

| Option | Default | Description |
|--------|---------|-------------|
| `--port PORT` | `8765` | TCP port. If it is taken, the next free one is used; if a MARS GUI already runs there, the browser is pointed at it. |
| `--host HOST` | `127.0.0.1` | Bind address. The default is reachable from your own machine only; `0.0.0.0` exposes the GUI (and the right to run jobs as you) to the network. |
| `--workdir DIR` | `~/mars-gui-jobs` | Job and upload directory (also `$MARS_GUI_WORKDIR`). |
| `--no-browser` | off | Do not open the browser (e.g. on a workstation reached through an SSH tunnel). |
| `--max-parallel N` | `1` | Jobs run at the same time; others wait in the queue. |
| `--demo` | off | Finished example jobs from published MARS runs and fictional environments; nothing is calculated or installed. Uses a new temporary job directory unless `--workdir` is given. |
| `--install-desktop` | off | Install a *MARS GUI* launcher in the Linux application menu and exit. |
| `--print-desktop` | off | Print that launcher (`.desktop` file) and exit. |

## Examples

```bash
mars gui                              # http://127.0.0.1:8765, jobs in ~/mars-gui-jobs
mars gui --demo                       # explore the interface with example results
mars gui --port 9000 --workdir runs/  # custom port and job directory
mars gui --no-browser                 # headless; open http://127.0.0.1:8765 through an SSH tunnel
mars gui --install-desktop            # clickable launcher in the application menu
```

## Jobs and environments

Each GUI job is an ordinary `mars` command run in its own folder under
`--workdir/jobs/`, with its input, outputs, `run.log`, the command
(`command.sh`) and a `job.json` record. A job runs with the Python environment
chosen in the page; the environments the GUI knows (its own, every conda
environment and the ones added under *Models & environments*) are listed in
`--workdir/gui_settings.json` together with the default one.
