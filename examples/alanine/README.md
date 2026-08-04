# Alanine dipeptide — end-to-end MARS example

Scripts that reproduce the worked example in
[`docs/examples/alanine_dipeptide.md`](../../docs/examples/alanine_dipeptide.md):
conformational sampling → microsolvation → IR spectroscopy, including every
snapshot shown on the docs page.

## Files

| File | Purpose |
|------|---------|
| `00_make_structure.py` | Build `alanine_dipeptide.xyz` (Ace-Ala-NMe) with RDKit. |
| `lowest_conformer.py`  | Extract the lowest-energy frame of an ensemble. |
| `01_run_workflow.sh`   | Run the full sampling → solvation → IR workflow and render all snapshots. |

## Quick start

```bash
# from this directory, inside the env where MARS is installed
bash 01_run_workflow.sh           # uses SO3LR by default
POT=harmonic bash 01_run_workflow.sh   # fast smoke test, no ML weights
```

Outputs land in `snapshots/` and are copied into `docs/assets/alanine/` so the
documentation page renders them. CPU-only and headless by default
(`JAX_PLATFORMS=cpu`, `MPLBACKEND=Agg`); no GPU required — the 3D viewer
snapshots even run without the JAX stack.
