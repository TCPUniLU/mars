# CLI reference

```
mars <input.xyz> [OPTIONS]            # Conformational search (default)
mars ir <input.xyz> [OPTIONS]         # IR spectroscopy
mars optimize <input.xyz> [OPTIONS]   # Structure optimization
mars solvation <solute.xyz> [OPTIONS] # Explicit solvation
mars viewer [input.xyz] [OPTIONS]     # Interactive 3D / IR / NCI viewer
```

Every subcommand prints a complete option reference with `--help`.

## Common options (every subcommand)

| Option | Default | Description |
|--------|---------|-------------|
| `--potential {so3lr,mace,dxtb,nci,harmonic,lj}` | `so3lr` | Potential backend |
| `--mace-foundation {mp,off,off24,anicc,omol}` | `off` | MACE foundation family (with `--potential mace`). `off` is MACE-OFF23; `off24` is MACE-OFF24 (medium only) |
| `--mace-model NAME` | `small` | MACE variant (e.g. small/medium/large, medium-mpa-0) |
| `--mace-cache-dir DIR` | `~/.cache/mars/mace_jax` | Converted JAX weight cache |
| `--dxtb-method {gfn1,gfn2}` | `gfn1` | dxtb method |
| `--so3lr-model PATH` | (pretrained) | Custom SO3LR weights |
| `--lr-cutoff Å` | `1000.0` | SO3LR long-range cutoff |
| `--charge Q` | `0.0` | Total molecular charge |
| `--float64` | off | Use 64-bit precision (IR auto-promotes) |
| `--cpu` | off | Force CPU execution |
| `--not-parallel` | off | Disable vmap batching |
| `--config FILE` | — | Load defaults from a TOML control file |
| `--log-file FILE` | (console only) | Tee log output to a file |
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | `INFO` | Logging verbosity |
| `--debug` | off | Alias for DEBUG + show all warnings |

## Per-subcommand pages

- [`mars` (default)](conformer_search.md) — conformational search
- [`mars ir`](ir.md) — IR spectroscopy
- [`mars optimize`](optimize.md) — structure optimization
- [`mars solvation`](solvation.md) — explicit solvation
- [`mars viewer`](viewer.md) — interactive 3D / IR / NCI viewer
