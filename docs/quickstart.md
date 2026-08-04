# Quickstart

Five minutes from install to a first conformational search.

## 1. Make an input XYZ

Save the following as `methane.xyz`:

```text
5
methane
C  0.0000  0.0000  0.0000
H  0.6294  0.6294  0.6294
H  0.6294 -0.6294 -0.6294
H -0.6294  0.6294 -0.6294
H -0.6294 -0.6294  0.6294
```

## 2. Run the default conformational search

```bash
mars methane.xyz --quick
```

What this does:

1. Auto-detects the topology and flexibility, picks MTD parameters.
2. Runs 1–2 MTD cycles, rotamer MD replicas, and (with
   `--genetic-crossing`) Z-matrix crossing.
3. Prunes by energy + RMSD + topology and writes the unique low-energy
   ensemble.

Output files:

- `auto_final_ensemble.xyz` — multi-frame XYZ of unique conformers,
  sorted by energy.
- Per-stage timings printed to stdout (or to `--log-file run.log`).

!!! tip "Why methane?"
    Methane has only one conformer, so `--quick` finishes in a few
    seconds — perfect for verifying your install. On a real molecule,
    use `mars input.xyz` (the default `--mode normal`) or
    `mars input.xyz --thorough`.

## 3. Try the other tools

The same `mars` command bundles four workflows:

```bash
# IR spectrum from an analytical Hessian, with a quick plot
mars ir methane.xyz --plot

# Tight LBFGS minimization
mars optimize methane.xyz --fmax 0.001

# Solvate methane with two shells of water (placement only, no relax)
mars solvation methane.xyz --solvent water --layers 4 8 --opt-mode none
```

## 4. Drive everything from a config file

Create `run.toml`:

```toml
[conformer_search]
mode = "thorough"
genetic_crossing = true
n_children = 30
ewin = 6.0
output = "ensemble.xyz"
```

Then:

```bash
mars methane.xyz --config run.toml
```

The CLI still wins on anything you pass explicitly, so you can override
one parameter from the command line:

```bash
mars methane.xyz --config run.toml --mode quick
```

See the [Config files overview](config/overview.md) for the full schema.

## Where to next?

- [User guide](guide/index.md) — conceptual overview of each tool.
- [Examples](examples/index.md) — copy-pasteable recipes for every
  subcommand.
- [CLI reference](cli/overview.md) — every flag, organized by subcommand.
