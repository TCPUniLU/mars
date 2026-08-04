# Citation

If you use MARS in published work, please cite this software together
with the underlying methods and data it draws from.

## MARS itself

MARS is described in a preprint of the same title; please cite it:

```bibtex
@article{mars2026,
  title   = {MARS: Machine-Learned Force Field Framework for Automated Conformational Sampling, Vibrational Spectroscopy, and Microsolvation},
  author  = {Su{\'a}rez-Dou, Sergio and Gallegos, Miguel and Tkatchenko, Alexandre},
  year    = {2026},
  note    = {Preprint},
  url     = {https://github.com/TCPUniLU/mars}
}
```

## Methods and algorithms

```bibtex
% MARS's conformational sampling workflow is inspired by the algorithmic
% ideas (multi-step RMSD-MTD + multi-temperature rotamer MD + a
% topology-based flexibility metric) introduced by CREST. MARS provides
% an independent JAX/ML-potential reimplementation; cite CREST for the
% original methodology.
@article{pracht2020crest,
  title   = {Automated exploration of the low-energy chemical space with fast quantum chemical methods},
  author  = {Pracht, Philipp and Bohle, Fabian and Grimme, Stefan},
  journal = {Physical Chemistry Chemical Physics},
  year    = {2020},
  volume  = {22},
  pages   = {7169--7192},
  doi     = {10.1039/C9CP06869D}
}

% AccFG-style functional-group identification used for IR peak
% allocation (mars/functional_groups.py, mars/ir.py)
@article{lin2024accfg,
  author  = {Liu, Xuan and Swaminathan, Sarathkrishna and Zubarev, Dmitry and Ransom, Brandi and Park, Nathaniel and Schmidt, Kristin and Zhao, Huimin},
  title   = {AccFG: Accurate Functional Group Extraction and Molecular Structure Comparison},
  journal = {Journal of Chemical Information and Modeling},
  volume  = {65},
  number  = {16},
  pages   = {8593-8602},
  year    = {2025},
  doi     = {10.1021/acs.jcim.5c01317}
}
```

## ML potentials and engines

```bibtex
% SO3LR: default potential (mars/potentials.py — SO3LRPotential)
@article{kabylda2025so3lr,
  title   = {Molecular Simulations with a Pretrained Neural Network and Universal Pairwise Force Fields},
  author  = {Kabylda, Adil and Frank, J. Thorben and Su{\'a}rez-Dou, Sergio and Khabibrakhmanov, Almaz and Medrano Sandonas, Leonardo and Unke, Oliver T. and Chmiela, Stefan and M{\"u}ller, Klaus-Robert and Tkatchenko, Alexandre},
  journal = {Journal of the American Chemical Society},
  year    = {2025},
  volume  = {147},
  number  = {37},
  pages   = {33723--33734},
  doi     = {10.1021/jacs.5c09558}
}

% MACE: optional potential backend (--potential mace)
@article{Kovacs2025maceoff,
  author  = {Kov{\'a}cs, D{\'a}vid P{\'e}ter and Moore, J. Harry and Browning, Nicholas J. and Batatia, Ilyes and Horton, Joshua T. and Pu, Yixuan and Kapil, Venkat and Witt, William C. and Magd{\'a}u, Ioan-Bogdan and Cole, Daniel J. and Cs{\'a}nyi, G{\'a}bor},
  title   = {{MACE-OFF: Short-Range Transferable Machine Learning Force Fields for Organic Molecules}},
  journal = {J. Am. Chem. Soc.},
  year    = {2025},
  volume  = {147},
  number  = {21},
  pages   = {17598--17611},
  doi     = {10.1021/jacs.4c07099}
}


% dxtb: optional GFN-xTB potential backend (--potential dxtb)
@article{friede2024dxtb,
  title   = {dxtb — An efficient and fully differentiable framework for extended tight-binding},
  author  = {Friede, Marvin and Hölzer, Christian and Ehlert, Sebastian and Grimme, Stefan},
  journal = {The Journal of Chemical Physics},
  year    = {2024},
  volume  = {161},
  pages   = {062501},
  doi     = {10.1063/5.0216715}
}

% JAX: numerical backend
@software{jax2018github,
  title  = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  author = {Bradbury, James and Frostig, Roy and Hawkins, Peter and Johnson, Matthew James and Leary, Chris and Maclaurin, Dougal and Necula, George and Paszke, Adam and VanderPlas, Jake and Wanderman-Milne, Skye and Zhang, Qiao},
  year   = {2018},
  url    = {http://github.com/google/jax}
}

% JAX-MD: simulation primitives used throughout MARS
@inproceedings{schoenholz2020jaxmd,
  title     = {JAX, M.D.: A framework for differentiable physics},
  author    = {Schoenholz, Samuel S. and Cubuk, Ekin Dogus},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2020},
  volume    = {33},
  pages     = {11428--11441}
}
```

## Reference data shipped in `mars.utils`

```bibtex
% Covalent radii (mars.utils.COVALENT_RADII)
@article{pyykko2009covalent,
  title   = {Molecular Single-Bond Covalent Radii for Elements 1–118},
  author  = {Pyykk{\"o}, Pekka and Atsumi, Michiko},
  journal = {Chemistry – A European Journal},
  year    = {2009},
  volume  = {15},
  pages   = {186--197},
  doi     = {10.1002/chem.200800987}
}

% Bondi vdW radii (mars.utils.VDW_RADII)
@article{bondi1964vdw,
  title   = {van der Waals Volumes and Radii},
  author  = {Bondi, A.},
  journal = {The Journal of Physical Chemistry},
  year    = {1964},
  volume  = {68},
  pages   = {441--451},
  doi     = {10.1021/j100785a001}
}

% Extension of Bondi radii to additional main-group elements
@article{mantina2009vdw,
  title   = {Consistent van der Waals Radii for the Whole Main Group},
  author  = {Mantina, Manjeera and Chamberlin, Adam C. and Valero, Rosendo and Cramer, Christopher J. and Truhlar, Donald G.},
  journal = {The Journal of Physical Chemistry A},
  year    = {2009},
  volume  = {113},
  pages   = {5806--5812},
  doi     = {10.1021/jp8111556}
}

% Standard atomic weights, ionization energies, electron affinities — NIST
@misc{nist_asd,
  title  = {NIST Atomic Spectra Database (Ionization Energies Data)},
  author = {{NIST}},
  url    = {https://physics.nist.gov/PhysRefData/ASD/ionEnergy.html}
}
```

## Other scientific software used in MARS

```bibtex
% The atomic simulation environment (ASE)
@article{ase-paper,
  author={Ask Hjorth Larsen and Jens Jørgen Mortensen and Jakob Blomqvist and Ivano E Castelli and Rune Christensen and Marcin
Dułak and Jesper Friis and Michael N Groves and Bjørk Hammer and Cory Hargus and Eric D Hermes and Paul C Jennings and Peter
Bjerre Jensen and James Kermode and John R Kitchin and Esben Leonhard Kolsbjerg and Joseph Kubal and Kristen
Kaasbjerg and Steen Lysgaard and Jón Bergmann Maronsson and Tristan Maxson and Thomas Olsen and Lars Pastewka and Andrew
Peterson and Carsten Rostgaard and Jakob Schiøtz and Ole Schütt and Mikkel Strange and Kristian S Thygesen and Tejs
Vegge and Lasse Vilhelmsen and Michael Walter and Zhenhua Zeng and Karsten W Jacobsen},
  title={The atomic simulation environment—a Python library for working with atoms},
  journal={Journal of Physics: Condensed Matter},
  volume={29},
  number={27},
  pages={273002},
  year={2017},
}

% Open Babel
@article{OBabel,
author={O'Boyle, Noel M. and Banck, Michael and James, Craig A.and Morley, Chris and Vandermeersch, Tim and Hutchison, Geoffrey R.},
title={Open Babel: An open chemical toolbox},
journal={Journal of Cheminformatics},
year={2011},
month={Oct},
day={07},
volume={3},
number={1},
pages={33},
doi={10.1186/1758-2946-3-33},
}


```

Solvent reference geometries shipped under `mars/solvents/library/` are
either experimental (NIST Computational Chemistry Comparison and
Benchmark Database, CCCBDB) or built with `obabel` — the `source` field in `mars/solvents/manifest.toml`
records which one applies to each solvent.

## Funding & acknowledgements

This work was supported by the European Research Council (ERC) under the
European Union's Horizon Europe research and the innovation programme through the
grant **FITMOL** (Field-Theory Approach to Molecular Interactions) (101054629)
and the Luxembourg National Research Fund under grant **FNR-CORE MBD-in-BMD** (18093472).
Miguel Gallegos acknowledges support from the European Union's Horizon Europe Marie
Skłodowska-Curie Actions (MSCA) Postdoctoral Fellowship (101202630). MARS is
developed in the **Theoretical Chemical Physics** group at the **University of
Luxembourg**.

<div class="funding-logos" markdown>
![European Research Council](../assets/logos/erc.png){ .off-glb }
![Luxembourg National Research Fund](../assets/logos/FNR.png){ .off-glb }
![Marie Skłodowska-Curie Actions](../assets/logos/MSCA.png.webp){ .off-glb }
![University of Luxembourg](../assets/logos/uni_luxembourg.png){ .off-glb }
![Theoretical Chemical Physics, University of Luxembourg](../assets/logos/tcp_unilu.png){ .off-glb }
</div>
