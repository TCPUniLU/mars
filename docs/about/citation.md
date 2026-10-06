# Citation

If you use MARS in published work, please cite this software together
with the underlying methods and data it draws from.

In the [graphical interface](../gui/index.md), each job's *Cite this job*
tab lists the references for the methods and models that job used (as text
or BibTeX), taken from this page.

## MARS itself

MARS is described in a preprint of the same title; please cite it:

```bibtex
@article{mars2026,
  title   = {MARS: Machine-Learned Force Field Framework for Automated Conformational Sampling, Vibrational Spectroscopy, and Microsolvation},
  author  = {Su{\'a}rez-Dou, Sergio and Gallegos, Miguel and Tkatchenko, Alexandre},
  journal = {ChemRxiv},
  year    = {2026},
  doi     = {10.26434/chemrxiv.15007087/v1},
  note    = {Preprint}
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

% Initial Hessian model used by --init-hessian lindh (mars/utils.py).
% Cite Lindh et al. if you use the chemically informed initial Hessian.
@article{Lindh1995,
  author  = {Lindh, Roland and Bernhardsson, Anders and Karlstr{\"o}m, Gunnar and Malmqvist, Per-{\AA}ke},
  title   = {On the use of a {H}essian model function in molecular geometry optimizations},
  journal = {Chemical Physics Letters},
  volume  = {241},
  number  = {4},
  pages   = {423--428},
  year    = {1995},
  doi     = {10.1016/0009-2614(95)00646-L}
}

% The two bonded-topology model Hessians that Lindh's is compared against.
% Not implemented in MARS; cited for context in the SI.
@article{Schlegel1984,
  author  = {Schlegel, H. Bernhard},
  title   = {Estimating the {H}essian for gradient-type geometry optimizations},
  journal = {Theoretica Chimica Acta},
  volume  = {66},
  number  = {5},
  pages   = {333--340},
  year    = {1984},
  doi     = {10.1007/BF00554788}
}

@article{Fischer1992,
  author  = {Fischer, Thomas H. and Almlof, Jan},
  title   = {General methods for geometry and wave function optimization},
  journal = {The Journal of Physical Chemistry},
  volume  = {96},
  number  = {24},
  pages   = {9768--9774},
  year    = {1992},
  doi     = {10.1021/j100203a036}
}

% Redundant internal coordinates (--coords internal, mars/internal_coords.py):
% the Pulay-Fogarasi back-transformation, the efficiency literature it rests
% on, and the TRIC rigid-body coordinates used for multi-fragment systems.
@article{Pulay1992,
  author  = {Pulay, Peter and Fogarasi, G{\'e}za},
  title   = {Geometry optimization in redundant internal coordinates},
  journal = {The Journal of Chemical Physics},
  volume  = {96},
  number  = {4},
  pages   = {2856--2860},
  year    = {1992},
  doi     = {10.1063/1.462844}
}

@article{Baker1993,
  author  = {Baker, Jon},
  title   = {Techniques for geometry optimization: A comparison of {C}artesian and natural internal coordinates},
  journal = {Journal of Computational Chemistry},
  volume  = {14},
  number  = {9},
  pages   = {1085--1100},
  year    = {1993},
  doi     = {10.1002/jcc.540140910}
}

@article{Peng1996,
  author  = {Peng, Chunyang and Ayala, Philippe Y. and Schlegel, H. Bernhard and Frisch, Michael J.},
  title   = {Using redundant internal coordinates to optimize equilibrium geometries and transition states},
  journal = {Journal of Computational Chemistry},
  volume  = {17},
  number  = {1},
  pages   = {49--56},
  year    = {1996},
  doi     = {10.1002/(SICI)1096-987X(19960115)17:1<49::AID-JCC5>3.0.CO;2-0}
}

@article{Bakken2002,
  author  = {Bakken, Vebj{\o}rn and Helgaker, Trygve},
  title   = {The efficient optimization of molecular geometries using redundant internal coordinates},
  journal = {The Journal of Chemical Physics},
  volume  = {117},
  number  = {20},
  pages   = {9160--9174},
  year    = {2002},
  doi     = {10.1063/1.1515483}
}

@article{Wang2016,
  author  = {Wang, Lee-Ping and Song, Chenchen},
  title   = {Geometry optimization made simple with translation and rotation coordinates},
  journal = {The Journal of Chemical Physics},
  volume  = {144},
  number  = {21},
  pages   = {214108},
  year    = {2016},
  doi     = {10.1063/1.4952956}
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

% MACE architecture (all --potential mace families)
@inproceedings{batatia2022mace,
  title     = {{MACE}: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
  author    = {Batatia, Ilyes and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Simm, Gregor N. C. and Ortner, Christoph and Cs{\'a}nyi, G{\'a}bor},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2022},
  volume    = {35},
  pages     = {11423--11436}
}

% MACE-MP-0 (--mace-foundation mp)
@article{batatia2023macemp,
  title   = {A foundation model for atomistic materials chemistry},
  author  = {Batatia, Ilyes and Benner, Philipp and Chiang, Yuan and others},
  journal = {arXiv preprint arXiv:2401.00096},
  year    = {2025}
}

% UMA (--potential uma)
@inproceedings{wood2025uma,
  title={{UMA}: A Family of Universal Models for Atoms},
  author={Brandon M Wood and Misko Dzamba and Xiang Fu and Meng Gao and Muhammed Shuaibi and Luis Barroso-Luque and Kareem Abdelmaqsoud and Vahe Gharakhanyan and John R. Kitchin and Daniel S. Levine and Kyle Michel and Anuroop Sriram and Taco Cohen and Abhishek Das and Sushree Jagriti Sahoo and Ammar Rizvi and Zachary Ward Ulissi and C. Lawrence Zitnick},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=SvopaNxYWt}
}

% Orb-v3 (--potential orb)
@article{rhodes2025orbv3,
  title   = {Orb-v3: atomistic simulation at scale},
  author  = {Rhodes, Benjamin and Vandenhaute, Sander and {\v{S}}imkus, Vaidotas and others},
  journal = {arXiv preprint arXiv:2504.06231},
  year    = {2025}
}

% ANI-2x (--potential ani)
@article{devereux2020ani2x,
  title   = {Extending the Applicability of the {ANI} Deep Learning Molecular Potential to Sulfur and Halogens},
  author  = {Devereux, Christian and Smith, Justin S. and Huddleston, Kate K. and Barros, Kipton and Zubatyuk, Roman and Isayev, Olexandr and Roitberg, Adrian E.},
  journal = {Journal of Chemical Theory and Computation},
  year    = {2020},
  volume  = {16},
  pages   = {4192--4202},
  doi     = {10.1021/acs.jctc.0c00121}
}

@article{gao2020torchani,
  title   = {{TorchANI}: A Free and Open Source {PyTorch}-Based Deep Learning Implementation of the {ANI} Neural Network Potentials},
  author  = {Gao, Xiang and Ramezanghorbani, Farhad and Isayev, Olexandr and Smith, Justin S. and Roitberg, Adrian E.},
  journal = {Journal of Chemical Information and Modeling},
  year    = {2020},
  volume  = {60},
  pages   = {3408--3415},
  doi     = {10.1021/acs.jcim.0c00451}
}

% AIMNet2 (--potential aimnet2)
@article{anstine2025aimnet2,
  title   = {{AIMNet2}: a neural network potential to meet your neutral, charged, organic, and elemental-organic needs},
  author  = {Anstine, Dylan M. and Zubatyuk, Roman and Isayev, Olexandr},
  journal = {Chemical Science},
  year    = {2025},
  volume  = {16},
  pages   = {10228--10244},
  doi     = {10.1039/D4SC08572H}
}

% AceFF (--potential aceff) and the TorchMD-Net framework it is built on
@article{farr2026aceff,
  title   = {{AceFF}: A State-of-the-Art Machine Learning Potential for Small Molecules},
  author  = {Stephen E. Farr and Stefan Doerr and Antonio Mirarchi and Francesc Sabanes Zariquiey and Gianni De Fabritiis},
  journal = {arXiv preprint arXiv:2601.00581},
  year    = {2026}
}

@article{pelaez2024torchmdnet2,
  title   = {{TorchMD-Net} 2.0: Fast Neural Network Potentials for Molecular Simulations},
  author  = {Pelaez, Raul P. and Simeon, Guillem and Galvelis, Raimondas and Mirarchi, Antonio and Eastman, Peter and Doerr, Stefan and Thölke, Philipp and Markland, Thomas E. and De Fabritiis, Gianni},
  journal = {Journal of Chemical Theory and Computation},
  year    = {2024},
  volume  = {20},
  pages   = {4076--4087},
  doi     = {10.1021/acs.jctc.4c00253}
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
  title     = {{JAX MD}: A Framework for Differentiable Physics},
  author    = {Schoenholz, Samuel S. and Cubuk, Ekin Dogus},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2020},
  volume    = {33},
  pages     = {11428--11441}
}
```

## Classical force fields

MARS ships no force-field parameters. Cite the force field whose files you
loaded, plus OpenMM, which reads them.

```bibtex
% OpenMM: parameter reading for --potential amber/charmm/opls/amoeba/mm
@article{eastman2024openmm8,
  title   = {{OpenMM} 8: Molecular Dynamics Simulation with Machine Learning Potentials},
  author  = {Eastman, Peter and Galvelis, Raimondas and Pel{\'a}ez, Ra{\'u}l P. and others},
  journal = {The Journal of Physical Chemistry B},
  year    = {2024},
  volume  = {128},
  pages   = {109--116},
  doi     = {10.1021/acs.jpcb.3c06662}
}

% AMBER protein force field (example: ff19SB) and GAFF2
@article{tian2020ff19sb,
  title   = {ff19SB: Amino-Acid-Specific Protein Backbone Parameters Trained against Quantum Mechanics Energy Surfaces in Solution},
  author  = {Tian, Chuan and Kasavajhala, Koushik and Belfon, Kellon A. A. and others},
  journal = {Journal of Chemical Theory and Computation},
  year    = {2020},
  volume  = {16},
  pages   = {528--552},
  doi     = {10.1021/acs.jctc.9b00591}
}

@article{He2020GAFF2,
    author = {He, Xibing and Man, Viet H. and Yang, Wei and Lee, Tai-Sung and Wang, Junmei},
    title = {{A fast and high-quality charge model for the next generation general AMBER force field}},
    journal = {The Journal of Chemical Physics},
    volume = {153},
    number = {11},
    pages = {114502},
    year = {2020},
    month = {09},
    issn = {0021-9606},
    doi = {10.1063/5.0019056},
}


% CHARMM36m
@article{huang2017charmm36m,
  title   = {{CHARMM36m}: an improved force field for folded and intrinsically disordered proteins},
  author  = {Huang, Jing and Rauscher, Sarah and Nawrocki, Grzegorz and others},
  journal = {Nature Methods},
  year    = {2017},
  volume  = {14},
  pages   = {71--73},
  doi     = {10.1038/nmeth.4067}
}

% OPLS-AA
@article{jorgensen1996oplsaa,
  title   = {Development and Testing of the {OPLS} All-Atom Force Field on Conformational Energetics and Properties of Organic Liquids},
  author  = {Jorgensen, William L. and Maxwell, David S. and Tirado-Rives, Julian},
  journal = {Journal of the American Chemical Society},
  year    = {1996},
  volume  = {118},
  pages   = {11225--11236},
  doi     = {10.1021/ja9621760}
}

% AMOEBA (--potential amoeba)
@article{Zhang2018,
   author = {Changsheng Zhang and Chao Lu and Zhifeng Jing and Chuanjie Wu and Jean-Philip Piquemal and Jay W. Ponder and Pengyu Ren},
   doi = {10.1021/acs.jctc.7b01169},
   issn = {1549-9618},
   issue = {4},
   journal = {Journal of Chemical Theory and Computation},
   month = {4},
   pages = {2084-2108},
   pmid = {29438622},
   publisher = {American Chemical Society},
   title = {{AMOEBA} Polarizable Atomic Multipole Force Field for Nucleic Acids},
   volume = {14},
   year = {2018}
}
```

## Implementation credits

Several potentials run on JAX code written by the jax-md community. Please
acknowledge these contributors (GitHub handles) along with the papers above:

| MARS potential | jax-md contribution | Contributors |
|----------------|---------------------|--------------|
| `uma` | UMA port (`jax_md._nn.uma`), PRs #376, #394, #400, #401; neighbor-overflow fix #399 | @ekurtulus, @abhijeetgangan, @teddykoker |
| `orb`, `ani`, `aimnet2`, `aceff`, `--so3lr-backend jaxmd` | bio-mlff ports (`jax_md._nn.{orb,ani,aimnet2,aceff,so3lr}`), PRs #411, #416, #419 | @mitkotak, @abhijeetgangan |
| `--mace-backend jaxmd` | MACE-JAX interface (`jax_md._nn.mace`), PRs #390, #418 | @DRosen285, @abhijeetgangan, @beitom (Tom Ginsberg), @xyf5432 |
| `amber`, `charmm`, `opls` | `mm_forcefields` AMBER energy + OpenMM converter (PR #319), MM refactor + OPLS-AA (PR #350) | @alecbetancourt, @DRosen285, @beitom, @abhijeetgangan |

The original model code that these ports reproduce comes from:

- [FAIR-Chem](https://github.com/FAIR-Chem/fairchem)
- [orb-models](https://github.com/orbital-materials/orb-models)
- [TorchANI](https://github.com/aiqm/torchani)
- [aimnetcentral](https://github.com/isayevlab/aimnetcentral)
- [torchmd-net](https://github.com/torchmd/torchmd-net)
- [mace](https://github.com/ACEsuit/mace) and [mace-jax](https://github.com/ACEsuit/mace-jax)
- [so3lr](https://github.com/general-molecular-simulations/so3lr)
- [bio-mlff](https://github.com/mitkotak/bio-mlff)

The AMOEBA terms, the OpenMM custom-expression compiler and the force-field
readers are MARS code.

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

## Graphical interface

The web GUI (`mars gui`) shows molecules with 3Dmol.js and plots with Plotly.js:

```bibtex
% 3Dmol.js: molecular viewer of the GUI (mars/gui/static)
@article{rego20153dmol,
  title   = {3Dmol.js: molecular visualization with {WebGL}},
  author  = {Rego, Nicholas and Koes, David},
  journal = {Bioinformatics},
  year    = {2015},
  volume  = {31},
  number  = {8},
  pages   = {1322--1324},
  doi     = {10.1093/bioinformatics/btu829}
}

% Plotly.js: plots of the GUI (no journal article; cited as software, MIT licence)
@misc{plotly,
  author    = {{Plotly Technologies Inc.}},
  title     = {Collaborative data science},
  publisher = {Plotly Technologies Inc.},
  address   = {Montr{\'e}al, QC},
  year      = {2015},
  url       = {https://plot.ly}
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
