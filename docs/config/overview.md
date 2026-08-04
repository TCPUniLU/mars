# Config files (`--config FILE`)

Every MARS subcommand accepts `--config FILE`, where `FILE` is a
TOML-style control file. Values from the file act as **defaults**; any
explicit CLI flag passed on the same command line overrides them.

## Section layout

| Section              | Used by                                       |
|----------------------|-----------------------------------------------|
| `[global]`           | every subcommand (merged in first)            |
| `[conformer_search]` | the default `mars input.xyz` mode             |
| `[ir]`               | `mars ir`                                     |
| `[optimize]`         | `mars optimize`                               |
| `[solvation]`        | `mars solvation`                              |
| `[constraints]`      | frozen-atom restraints (default + `optimize`) |

`[global]` is merged first, then the subcommand-specific section
overrides anything that overlaps.

## Conversion rules

- Keys match the long CLI option names with dashes replaced by
  underscores: `--mtd-time` → `mtd_time`, `--no-optimize` → `no_optimize`.
- Booleans use `true` / `false`. `True` becomes a bare flag
  (`--no-optimize`); `False` is dropped silently.
- Lists use TOML inline arrays. Each element is emitted as a separate
  argv token, so they line up with argparse `nargs="+"` flags:
    - `layers = [4, 8]`        → `--layers 4 8`
    - `freq_range = [400, 3500]` → `--freq-range 400 3500`
- Lines starting with `#` are comments. Inline ` # ...` comments after a
  value are stripped too.

## Override precedence

```
[global] section
  ↓
[<subcommand>] section
  ↓
explicit CLI flags
```

Last writer wins. The CLI flag is always the last thing argparse sees,
so it always overrides the file.

## Tips

- Pass `--debug` to see the merged argv argparse actually receives.
- Use the special `input` key inside any section to embed the positional
  input filename, making the config file fully self-contained:
  ```toml
  [conformer_search]
  input = "mol.xyz"
  mode = "thorough"
  ```
  Then run `mars --config run.toml` (no positional argument needed).
- The `[constraints]` section is **only** read from the config file —
  there is no equivalent CLI flag.

## Per-section reference

- [`[conformer_search]`](conformer_search_section.md)
- [`[ir]`](ir_section.md)
- [`[optimize]`](optimize_section.md)
- [`[solvation]`](solvation_section.md)
- [`[constraints]`](constraints_section.md)
