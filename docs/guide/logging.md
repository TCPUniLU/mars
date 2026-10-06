# Logging

MARS uses a structured logger that writes a clean stream to the console
and a timestamped copy to an optional log file.

## From the CLI

```bash
mars input.xyz --log-file run.log           # tee to file
mars input.xyz --log-level DEBUG            # verbose
mars input.xyz --debug                      # DEBUG + show all warnings
```

`--log-level` accepts `DEBUG`, `INFO` (default), `WARNING`, or `ERROR`.

**Colours.** On a terminal the banner and section titles are drawn in the
red of the MARS logo, warnings in amber and errors in red (true colour where
the terminal supports it, otherwise 256 colours). The *Log* tab of the
[graphical interface](../gui/index.md) shows the same colours. Log files,
pipes and batch-job output are plain text, and so is a log downloaded from
the GUI. Set `NO_COLOR=1` to switch colours off, or `FORCE_COLOR=1` to keep
them when piping into a pager (`less -R`).
`--debug` is a one-shot alias that bumps the level and un-suppresses
Python warnings — useful when investigating numerical issues.

## From Python

```python
from mars.log import (
    init_log, log_info, log_warning, log_error,
    LogTimer, timer_reset, timer_start, timer_end, log_timing_summary,
)

init_log(logfile="run.log", level="INFO")

log_info("Starting analysis")
with LogTimer("Optimization"):
    ...   # whatever you want to time

# Workflow-stage tracker
timer_reset()
timer_start("MTD sampling")
...
timer_end("MTD sampling")
timer_start("Genetic crossing")
...
timer_end("Genetic crossing")
log_timing_summary()    # prints the cumulative table
```

## Features

- **Clean console output** — no timestamps, just structured messages.
- **Timestamped file output** when `--log-file` is set.
- **Progress bars** via `tqdm` for long loops.
- **Energy tables** for ensemble inspection (`log_energy_table`).
- **Topology summary** with bond statistics and fragment detection
  (`log_topology`).
- **Timing tracker** that records cumulative wall-clock time per
  workflow stage, formatted in `s`, `min`, or `h`.

## See also

- API reference: [`mars.log`](../api/log.md).
