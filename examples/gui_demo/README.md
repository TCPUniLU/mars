# GUI demo data

Outputs of published MARS runs (the case studies of the MARS Zenodo package),
used by `mars gui --demo` to show finished jobs without running anything:

| Demo job | Source |
|---|---|
| Methanol IR spectrum (SO3LR) | case study 01, `outputs/` |
| Methanol IR spectrum from MD (SO3LR) | case study 01, `outputs/md_ir.dat` |
| Methanol geometry optimisation (SO3LR) | case study 01, `outputs/optimized_structure.xyz` |
| 2-Methoxycyclohexanone (R) conformers (SO3LR) | case study 03, `ensembles/R_vacuum_so3lr1.xyz` |
| Ace-Ala5-NMe conformers (MACE-OFF24) | case study 04, `mars_ensembles/MACE-OFF24_fp64/` |
| Proline in β-cyclodextrin, NCI mode (SO3LR) | case study 05, `ensembles/Pro@bCD.xyz` |
| Alanine dipeptide in water (SO3LR) | case study 02, `clusters/geometric/water/solvated_rep_1.xyz` |

Files are renamed to the names the `mars` commands write. The run logs of the
demo jobs are not the original logs; they only say where the outputs come from.
