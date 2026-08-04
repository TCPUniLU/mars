#!/usr/bin/env bash
#
# End-to-end MARS walkthrough on the alanine dipeptide:
#   conformational sampling -> microsolvation -> IR spectroscopy,
# rendering every snapshot used in docs/examples/alanine_dipeptide.md.
#
# Run from this directory:
#     bash 01_run_workflow.sh
#
# Requirements: the `mars` CLI on PATH (or run inside the env where MARS is
# installed). SO3LR weights are needed for the energies/IR; the structure and
# 3D snapshots work with any backend. Snapshots are copied into the docs assets
# folder so the example page renders them directly.

set -euo pipefail


POT="${POT:-so3lr}"                      # set POT=harmonic for a quick smoke test
ASSETS="../../docs/assets/alanine"        # where the docs look for images
WORK="$(pwd)"
mkdir -p snapshots "$ASSETS"

run() { echo; echo ">>> $*"; "$@"; }

# --- 0. build the starting structure ---------------------------------------
run python 00_make_structure.py alanine_dipeptide.xyz
run mars viewer alanine_dipeptide.xyz --save snapshots/alanine_input.png

# --- 1. conformational sampling --------------------------------------------
# 'quick' preset keeps the example fast; use normal/thorough for production.
run mars alanine_dipeptide.xyz --mode quick --potential "$POT" -o conformers.xyz
run python lowest_conformer.py conformers.xyz best.xyz
run mars viewer best.xyz --save snapshots/alanine_best.png

# --- 2. microsolvation ------------------------------------------------------
# Wrap the lowest conformer in a shell of explicit water (geometric builder).
run mars solvation best.xyz --solvent water --padding 4.0 \
    --potential "$POT" --output solvated.xyz
run mars viewer solvated.xyz --save snapshots/alanine_solvated.png

# Barostat (liquid-like) alternative — uncomment to compare:
# run mars solvation best.xyz --solvent water --barostat --n-molecules 20 \
#     --potential "$POT" --output solvated_barostat.xyz

# --- 3. IR spectroscopy -----------------------------------------------------
# Harmonic spectrum of the lowest conformer, with the annotated plot and the
# files the interactive viewer reads back.
run mars ir best.xyz --potential "$POT" \
    --plot --plot-output snapshots/alanine_ir_spectrum.png \
    --mode-xyz --save-structure
run mars viewer --ir --dir . --save snapshots/alanine_ir_viewer.png

# --- collect snapshots for the docs ----------------------------------------
cp -f snapshots/alanine_input.png        "$ASSETS/input.png"
cp -f snapshots/alanine_best.png         "$ASSETS/best_conformer.png"
cp -f snapshots/alanine_solvated.png     "$ASSETS/solvated.png"
cp -f snapshots/alanine_ir_spectrum.png  "$ASSETS/ir_spectrum.png"
cp -f snapshots/alanine_ir_viewer.png    "$ASSETS/ir_viewer.png"

echo
echo "Done. Snapshots in $WORK/snapshots and copied to $ASSETS/"
