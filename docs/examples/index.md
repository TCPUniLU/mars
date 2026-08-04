# Examples

Copy-pasteable recipes for every MARS subcommand. The CLI is the
recommended entry point; equivalent Python API snippets follow each
section.

!!! example "Start here: the alanine dipeptide walkthrough"
    The **[end-to-end alanine dipeptide example](alanine_dipeptide.md)** runs a
    single molecule through conformational sampling, microsolvation, and IR
    spectroscopy — with snapshots and a one-command reproduction script. It is
    the best way to see how the pieces fit together.

| Recipe page | Subcommand |
|-------------|------------|
| [End-to-end: alanine dipeptide](alanine_dipeptide.md) | full workflow |
| [Conformational search recipes](conformer_search_recipes.md) | `mars` (default) |
| [IR recipes](ir_recipes.md) | `mars ir` |
| [Optimization recipes](optimize_recipes.md) | `mars optimize` |
| [Solvation recipes](solvation_recipes.md) | `mars solvation` |

!!! tip "Reproducibility"
    Every example uses an `input.xyz` placeholder — substitute your own
    structure. Examples that need SO3LR weights are flagged; the rest
    work with `--potential harmonic` for smoke testing.

For the matching TOML config-file snippets see the
[Config files section](../config/overview.md).
