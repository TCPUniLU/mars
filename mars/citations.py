"""References of MARS and of the methods and models a job used (from ``docs/about/citation.md``)."""

import re
from typing import Dict, List

BIB: Dict[str, str] = {
    'mars2026': "@article{mars2026,\n  title   = {MARS: Machine-Learned Force Field Framework for Automated Conformational Sampling, Vibrational Spectroscopy, and Microsolvation},\n  author  = {Su{\\'a}rez-Dou, Sergio and Gallegos, Miguel and Tkatchenko, Alexandre},\n  journal = {ChemRxiv},\n  year    = {2026},\n  doi     = {10.26434/chemrxiv.15007087/v1},\n  note    = {Preprint}\n}",
    'pracht2020crest': '@article{pracht2020crest,\n  title   = {Automated exploration of the low-energy chemical space with fast quantum chemical methods},\n  author  = {Pracht, Philipp and Bohle, Fabian and Grimme, Stefan},\n  journal = {Physical Chemistry Chemical Physics},\n  year    = {2020},\n  volume  = {22},\n  pages   = {7169--7192},\n  doi     = {10.1039/C9CP06869D}\n}',
    'lin2024accfg': '@article{lin2024accfg,\n  author  = {Liu, Xuan and Swaminathan, Sarathkrishna and Zubarev, Dmitry and Ransom, Brandi and Park, Nathaniel and Schmidt, Kristin and Zhao, Huimin},\n  title   = {AccFG: Accurate Functional Group Extraction and Molecular Structure Comparison},\n  journal = {Journal of Chemical Information and Modeling},\n  volume  = {65},\n  number  = {16},\n  pages   = {8593-8602},\n  year    = {2025},\n  doi     = {10.1021/acs.jcim.5c01317}\n}',
    'Lindh1995': '@article{Lindh1995,\n  author  = {Lindh, Roland and Bernhardsson, Anders and Karlstr{\\"o}m, Gunnar and Malmqvist, Per-{\\AA}ke},\n  title   = {On the use of a {H}essian model function in molecular geometry optimizations},\n  journal = {Chemical Physics Letters},\n  volume  = {241},\n  number  = {4},\n  pages   = {423--428},\n  year    = {1995},\n  doi     = {10.1016/0009-2614(95)00646-L}\n}',
    'Schlegel1984': '@article{Schlegel1984,\n  author  = {Schlegel, H. Bernhard},\n  title   = {Estimating the {H}essian for gradient-type geometry optimizations},\n  journal = {Theoretica Chimica Acta},\n  volume  = {66},\n  number  = {5},\n  pages   = {333--340},\n  year    = {1984},\n  doi     = {10.1007/BF00554788}\n}',
    'Fischer1992': '@article{Fischer1992,\n  author  = {Fischer, Thomas H. and Almlof, Jan},\n  title   = {General methods for geometry and wave function optimization},\n  journal = {The Journal of Physical Chemistry},\n  volume  = {96},\n  number  = {24},\n  pages   = {9768--9774},\n  year    = {1992},\n  doi     = {10.1021/j100203a036}\n}',
    'Pulay1992': "@article{Pulay1992,\n  author  = {Pulay, Peter and Fogarasi, G{\\'e}za},\n  title   = {Geometry optimization in redundant internal coordinates},\n  journal = {The Journal of Chemical Physics},\n  volume  = {96},\n  number  = {4},\n  pages   = {2856--2860},\n  year    = {1992},\n  doi     = {10.1063/1.462844}\n}",
    'Baker1993': '@article{Baker1993,\n  author  = {Baker, Jon},\n  title   = {Techniques for geometry optimization: A comparison of {C}artesian and natural internal coordinates},\n  journal = {Journal of Computational Chemistry},\n  volume  = {14},\n  number  = {9},\n  pages   = {1085--1100},\n  year    = {1993},\n  doi     = {10.1002/jcc.540140910}\n}',
    'Peng1996': '@article{Peng1996,\n  author  = {Peng, Chunyang and Ayala, Philippe Y. and Schlegel, H. Bernhard and Frisch, Michael J.},\n  title   = {Using redundant internal coordinates to optimize equilibrium geometries and transition states},\n  journal = {Journal of Computational Chemistry},\n  volume  = {17},\n  number  = {1},\n  pages   = {49--56},\n  year    = {1996},\n  doi     = {10.1002/(SICI)1096-987X(19960115)17:1<49::AID-JCC5>3.0.CO;2-0}\n}',
    'Bakken2002': '@article{Bakken2002,\n  author  = {Bakken, Vebj{\\o}rn and Helgaker, Trygve},\n  title   = {The efficient optimization of molecular geometries using redundant internal coordinates},\n  journal = {The Journal of Chemical Physics},\n  volume  = {117},\n  number  = {20},\n  pages   = {9160--9174},\n  year    = {2002},\n  doi     = {10.1063/1.1515483}\n}',
    'Wang2016': '@article{Wang2016,\n  author  = {Wang, Lee-Ping and Song, Chenchen},\n  title   = {Geometry optimization made simple with translation and rotation coordinates},\n  journal = {The Journal of Chemical Physics},\n  volume  = {144},\n  number  = {21},\n  pages   = {214108},\n  year    = {2016},\n  doi     = {10.1063/1.4952956}\n}',
    'kabylda2025so3lr': '@article{kabylda2025so3lr,\n  title   = {Molecular Simulations with a Pretrained Neural Network and Universal Pairwise Force Fields},\n  author  = {Kabylda, Adil and Frank, J. Thorben and Su{\\\'a}rez-Dou, Sergio and Khabibrakhmanov, Almaz and Medrano Sandonas, Leonardo and Unke, Oliver T. and Chmiela, Stefan and M{\\"u}ller, Klaus-Robert and Tkatchenko, Alexandre},\n  journal = {Journal of the American Chemical Society},\n  year    = {2025},\n  volume  = {147},\n  number  = {37},\n  pages   = {33723--33734},\n  doi     = {10.1021/jacs.5c09558}\n}',
    'Kovacs2025maceoff': "@article{Kovacs2025maceoff,\n  author  = {Kov{\\'a}cs, D{\\'a}vid P{\\'e}ter and Moore, J. Harry and Browning, Nicholas J. and Batatia, Ilyes and Horton, Joshua T. and Pu, Yixuan and Kapil, Venkat and Witt, William C. and Magd{\\'a}u, Ioan-Bogdan and Cole, Daniel J. and Cs{\\'a}nyi, G{\\'a}bor},\n  title   = {{MACE-OFF: Short-Range Transferable Machine Learning Force Fields for Organic Molecules}},\n  journal = {J. Am. Chem. Soc.},\n  year    = {2025},\n  volume  = {147},\n  number  = {21},\n  pages   = {17598--17611},\n  doi     = {10.1021/jacs.4c07099}\n}",
    'friede2024dxtb': '@article{friede2024dxtb,\n  title   = {dxtb — An efficient and fully differentiable framework for extended tight-binding},\n  author  = {Friede, Marvin and Hölzer, Christian and Ehlert, Sebastian and Grimme, Stefan},\n  journal = {The Journal of Chemical Physics},\n  year    = {2024},\n  volume  = {161},\n  pages   = {062501},\n  doi     = {10.1063/5.0216715}\n}',
    'batatia2022mace': "@inproceedings{batatia2022mace,\n  title     = {{MACE}: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},\n  author    = {Batatia, Ilyes and Kov{\\'a}cs, D{\\'a}vid P{\\'e}ter and Simm, Gregor N. C. and Ortner, Christoph and Cs{\\'a}nyi, G{\\'a}bor},\n  booktitle = {Advances in Neural Information Processing Systems},\n  year      = {2022},\n  volume    = {35},\n  pages     = {11423--11436}\n}",
    'batatia2023macemp': '@article{batatia2023macemp,\n  title   = {A foundation model for atomistic materials chemistry},\n  author  = {Batatia, Ilyes and Benner, Philipp and Chiang, Yuan and others},\n  journal = {arXiv preprint arXiv:2401.00096},\n  year    = {2025}\n}',
    'wood2025uma': '@inproceedings{wood2025uma,\n  title={{UMA}: A Family of Universal Models for Atoms},\n  author={Brandon M Wood and Misko Dzamba and Xiang Fu and Meng Gao and Muhammed Shuaibi and Luis Barroso-Luque and Kareem Abdelmaqsoud and Vahe Gharakhanyan and John R. Kitchin and Daniel S. Levine and Kyle Michel and Anuroop Sriram and Taco Cohen and Abhishek Das and Sushree Jagriti Sahoo and Ammar Rizvi and Zachary Ward Ulissi and C. Lawrence Zitnick},\n  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},\n  year={2025},\n  url={https://openreview.net/forum?id=SvopaNxYWt}\n}',
    'rhodes2025orbv3': '@article{rhodes2025orbv3,\n  title   = {Orb-v3: atomistic simulation at scale},\n  author  = {Rhodes, Benjamin and Vandenhaute, Sander and {\\v{S}}imkus, Vaidotas and others},\n  journal = {arXiv preprint arXiv:2504.06231},\n  year    = {2025}\n}',
    'devereux2020ani2x': '@article{devereux2020ani2x,\n  title   = {Extending the Applicability of the {ANI} Deep Learning Molecular Potential to Sulfur and Halogens},\n  author  = {Devereux, Christian and Smith, Justin S. and Huddleston, Kate K. and Barros, Kipton and Zubatyuk, Roman and Isayev, Olexandr and Roitberg, Adrian E.},\n  journal = {Journal of Chemical Theory and Computation},\n  year    = {2020},\n  volume  = {16},\n  pages   = {4192--4202},\n  doi     = {10.1021/acs.jctc.0c00121}\n}',
    'gao2020torchani': '@article{gao2020torchani,\n  title   = {{TorchANI}: A Free and Open Source {PyTorch}-Based Deep Learning Implementation of the {ANI} Neural Network Potentials},\n  author  = {Gao, Xiang and Ramezanghorbani, Farhad and Isayev, Olexandr and Smith, Justin S. and Roitberg, Adrian E.},\n  journal = {Journal of Chemical Information and Modeling},\n  year    = {2020},\n  volume  = {60},\n  pages   = {3408--3415},\n  doi     = {10.1021/acs.jcim.0c00451}\n}',
    'anstine2025aimnet2': '@article{anstine2025aimnet2,\n  title   = {{AIMNet2}: a neural network potential to meet your neutral, charged, organic, and elemental-organic needs},\n  author  = {Anstine, Dylan M. and Zubatyuk, Roman and Isayev, Olexandr},\n  journal = {Chemical Science},\n  year    = {2025},\n  volume  = {16},\n  pages   = {10228--10244},\n  doi     = {10.1039/D4SC08572H}\n}',
    'farr2026aceff': '@article{farr2026aceff,\n  title   = {{AceFF}: A State-of-the-Art Machine Learning Potential for Small Molecules},\n  author  = {Stephen E. Farr and Stefan Doerr and Antonio Mirarchi and Francesc Sabanes Zariquiey and Gianni De Fabritiis},\n  journal = {arXiv preprint arXiv:2601.00581},\n  year    = {2026}\n}',
    'pelaez2024torchmdnet2': '@article{pelaez2024torchmdnet2,\n  title   = {{TorchMD-Net} 2.0: Fast Neural Network Potentials for Molecular Simulations},\n  author  = {Pelaez, Raul P. and Simeon, Guillem and Galvelis, Raimondas and Mirarchi, Antonio and Eastman, Peter and Doerr, Stefan and Thölke, Philipp and Markland, Thomas E. and De Fabritiis, Gianni},\n  journal = {Journal of Chemical Theory and Computation},\n  year    = {2024},\n  volume  = {20},\n  pages   = {4076--4087},\n  doi     = {10.1021/acs.jctc.4c00253}\n}',
    'jax2018github': '@software{jax2018github,\n  title  = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},\n  author = {Bradbury, James and Frostig, Roy and Hawkins, Peter and Johnson, Matthew James and Leary, Chris and Maclaurin, Dougal and Necula, George and Paszke, Adam and VanderPlas, Jake and Wanderman-Milne, Skye and Zhang, Qiao},\n  year   = {2018},\n  url    = {http://github.com/google/jax}\n}',
    'schoenholz2020jaxmd': '@inproceedings{schoenholz2020jaxmd,\n  title     = {{JAX MD}: A Framework for Differentiable Physics},\n  author    = {Schoenholz, Samuel S. and Cubuk, Ekin Dogus},\n  booktitle = {Advances in Neural Information Processing Systems},\n  year      = {2020},\n  volume    = {33},\n  pages     = {11428--11441}\n}',
    'eastman2024openmm8': "@article{eastman2024openmm8,\n  title   = {{OpenMM} 8: Molecular Dynamics Simulation with Machine Learning Potentials},\n  author  = {Eastman, Peter and Galvelis, Raimondas and Pel{\\'a}ez, Ra{\\'u}l P. and others},\n  journal = {The Journal of Physical Chemistry B},\n  year    = {2024},\n  volume  = {128},\n  pages   = {109--116},\n  doi     = {10.1021/acs.jpcb.3c06662}\n}",
    'tian2020ff19sb': '@article{tian2020ff19sb,\n  title   = {ff19SB: Amino-Acid-Specific Protein Backbone Parameters Trained against Quantum Mechanics Energy Surfaces in Solution},\n  author  = {Tian, Chuan and Kasavajhala, Koushik and Belfon, Kellon A. A. and others},\n  journal = {Journal of Chemical Theory and Computation},\n  year    = {2020},\n  volume  = {16},\n  pages   = {528--552},\n  doi     = {10.1021/acs.jctc.9b00591}\n}',
    'He2020GAFF2': '@article{He2020GAFF2,\n    author = {He, Xibing and Man, Viet H. and Yang, Wei and Lee, Tai-Sung and Wang, Junmei},\n    title = {{A fast and high-quality charge model for the next generation general AMBER force field}},\n    journal = {The Journal of Chemical Physics},\n    volume = {153},\n    number = {11},\n    pages = {114502},\n    year = {2020},\n    month = {09},\n    issn = {0021-9606},\n    doi = {10.1063/5.0019056},\n}',
    'huang2017charmm36m': '@article{huang2017charmm36m,\n  title   = {{CHARMM36m}: an improved force field for folded and intrinsically disordered proteins},\n  author  = {Huang, Jing and Rauscher, Sarah and Nawrocki, Grzegorz and others},\n  journal = {Nature Methods},\n  year    = {2017},\n  volume  = {14},\n  pages   = {71--73},\n  doi     = {10.1038/nmeth.4067}\n}',
    'jorgensen1996oplsaa': '@article{jorgensen1996oplsaa,\n  title   = {Development and Testing of the {OPLS} All-Atom Force Field on Conformational Energetics and Properties of Organic Liquids},\n  author  = {Jorgensen, William L. and Maxwell, David S. and Tirado-Rives, Julian},\n  journal = {Journal of the American Chemical Society},\n  year    = {1996},\n  volume  = {118},\n  pages   = {11225--11236},\n  doi     = {10.1021/ja9621760}\n}',
    'Zhang2018': '@article{Zhang2018,\n   author = {Changsheng Zhang and Chao Lu and Zhifeng Jing and Chuanjie Wu and Jean-Philip Piquemal and Jay W. Ponder and Pengyu Ren},\n   doi = {10.1021/acs.jctc.7b01169},\n   issn = {1549-9618},\n   issue = {4},\n   journal = {Journal of Chemical Theory and Computation},\n   month = {4},\n   pages = {2084-2108},\n   pmid = {29438622},\n   publisher = {American Chemical Society},\n   title = {{AMOEBA} Polarizable Atomic Multipole Force Field for Nucleic Acids},\n   volume = {14},\n   year = {2018}\n}',
    'pyykko2009covalent': '@article{pyykko2009covalent,\n  title   = {Molecular Single-Bond Covalent Radii for Elements 1–118},\n  author  = {Pyykk{\\"o}, Pekka and Atsumi, Michiko},\n  journal = {Chemistry – A European Journal},\n  year    = {2009},\n  volume  = {15},\n  pages   = {186--197},\n  doi     = {10.1002/chem.200800987}\n}',
    'bondi1964vdw': '@article{bondi1964vdw,\n  title   = {van der Waals Volumes and Radii},\n  author  = {Bondi, A.},\n  journal = {The Journal of Physical Chemistry},\n  year    = {1964},\n  volume  = {68},\n  pages   = {441--451},\n  doi     = {10.1021/j100785a001}\n}',
    'mantina2009vdw': '@article{mantina2009vdw,\n  title   = {Consistent van der Waals Radii for the Whole Main Group},\n  author  = {Mantina, Manjeera and Chamberlin, Adam C. and Valero, Rosendo and Cramer, Christopher J. and Truhlar, Donald G.},\n  journal = {The Journal of Physical Chemistry A},\n  year    = {2009},\n  volume  = {113},\n  pages   = {5806--5812},\n  doi     = {10.1021/jp8111556}\n}',
    'nist_asd': '@misc{nist_asd,\n  title  = {NIST Atomic Spectra Database (Ionization Energies Data)},\n  author = {{NIST}},\n  url    = {https://physics.nist.gov/PhysRefData/ASD/ionEnergy.html}\n}',
    'ase-paper': '@article{ase-paper,\n  author={Ask Hjorth Larsen and Jens Jørgen Mortensen and Jakob Blomqvist and Ivano E Castelli and Rune Christensen and Marcin\nDułak and Jesper Friis and Michael N Groves and Bjørk Hammer and Cory Hargus and Eric D Hermes and Paul C Jennings and Peter\nBjerre Jensen and James Kermode and John R Kitchin and Esben Leonhard Kolsbjerg and Joseph Kubal and Kristen\nKaasbjerg and Steen Lysgaard and Jón Bergmann Maronsson and Tristan Maxson and Thomas Olsen and Lars Pastewka and Andrew\nPeterson and Carsten Rostgaard and Jakob Schiøtz and Ole Schütt and Mikkel Strange and Kristian S Thygesen and Tejs\nVegge and Lasse Vilhelmsen and Michael Walter and Zhenhua Zeng and Karsten W Jacobsen},\n  title={The atomic simulation environment—a Python library for working with atoms},\n  journal={Journal of Physics: Condensed Matter},\n  volume={29},\n  number={27},\n  pages={273002},\n  year={2017},\n}',
    'OBabel': "@article{OBabel,\nauthor={O'Boyle, Noel M. and Banck, Michael and James, Craig A.and Morley, Chris and Vandermeersch, Tim and Hutchison, Geoffrey R.},\ntitle={Open Babel: An open chemical toolbox},\njournal={Journal of Cheminformatics},\nyear={2011},\nmonth={Oct},\nday={07},\nvolume={3},\nnumber={1},\npages={33},\ndoi={10.1186/1758-2946-3-33},\n}",
    'rego20153dmol': '@article{rego20153dmol,\n  title   = {3Dmol.js: molecular visualization with {WebGL}},\n  author  = {Rego, Nicholas and Koes, David},\n  journal = {Bioinformatics},\n  year    = {2015},\n  volume  = {31},\n  number  = {8},\n  pages   = {1322--1324},\n  doi     = {10.1093/bioinformatics/btu829}\n}',
    'plotly': "@misc{plotly,\n  author    = {{Plotly Technologies Inc.}},\n  title     = {Collaborative data science},\n  publisher = {Plotly Technologies Inc.},\n  address   = {Montr{\\'e}al, QC},\n  year      = {2015},\n  url       = {https://plot.ly}\n}"
}

ALWAYS = ["mars2026", "jax2018github", "schoenholz2020jaxmd"]
TASKS = {
    "conformer": ["pracht2020crest"], "mtd": ["pracht2020crest"], "md": ["pracht2020crest"],
    "ir": ["lin2024accfg"],
}
POTENTIALS = {
    "so3lr": ["kabylda2025so3lr"],
    "mace": ["batatia2022mace"],
    "uma": ["wood2025uma"],
    "orb": ["rhodes2025orbv3"],
    "ani": ["devereux2020ani2x", "gao2020torchani"],
    "aimnet2": ["anstine2025aimnet2"],
    "aceff": ["farr2026aceff", "pelaez2024torchmdnet2"],
    "dxtb": ["friede2024dxtb"],
    "amber": ["eastman2024openmm8"], "charmm": ["eastman2024openmm8"], "opls": ["eastman2024openmm8"],
    "mm": ["eastman2024openmm8"],
    "amoeba": ["eastman2024openmm8"],
}


def entries_from_markdown(text: str) -> Dict[str, str]:
    """BibTeX entries in a Markdown text (``@type{key, ... }`` blocks), by key."""
    out = {}
    for m in re.finditer(r"^@(\w+)\{([^,\s]+),", text, re.M):
        depth, j = 0, text.index("{", m.start())
        for k in range(j, len(text)):
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            if depth == 0:
                out[m.group(2)] = text[m.start():k + 1]
                break
    return out


def _field(entry: str, name: str) -> str:
    m = re.search(r"\b" + name + r"\s*=\s*\{", entry, re.I)
    if not m:
        return ""
    depth, k = 1, m.end()
    for k in range(m.end(), len(entry)):
        depth += {"{": 1, "}": -1}.get(entry[k], 0)
        if depth == 0:
            break
    return entry[m.end():k]


_TEX = {r"{\'a}": "á", r"{\'e}": "é", r"{\'i}": "í", r"{\'o}": "ó", r"{\'u}": "ú", r'{\"u}': "ü", r'{\"o}': "ö",
        r'{\"a}': "ä", r"{\`e}": "è", r"{\~n}": "ñ", r"{\c c}": "ç"}


def _plain(s: str) -> str:
    for k, v in _TEX.items():
        s = s.replace(k, v)
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)
    return re.sub(r"\s+", " ", s.replace("{", "").replace("}", "")).strip()


def format_ref(key: str) -> dict:
    """``{key, text, doi, url}`` for one entry: Authors, Title, Journal vol, pages (year)."""
    e = BIB[key]
    authors = [a.strip() for a in _plain(_field(e, "author")).split(" and ") if a.strip()]
    names = []
    for a in authors[:3]:
        last, _, first = a.partition(",")
        names.append(f"{(first.strip()[:1] + '. ') if first.strip() else ''}{last.strip()}")
    who = ", ".join(names) + (" et al." if len(authors) > 3 or "others" in authors else "")
    venue = _plain(_field(e, "journal") or _field(e, "booktitle") or _field(e, "publisher") or _field(e, "howpublished"))
    vol, pages, year = _plain(_field(e, "volume")), _plain(_field(e, "pages")).replace("--", "–"), _field(e, "year")
    text = f"{who}, {_plain(_field(e, 'title'))}" + (f", {venue}" if venue else "") + (f" {vol}" if vol else "") + \
        (f", {pages}" if pages else "") + (f" ({year})" if year else "") + "."
    doi = _plain(_field(e, "doi"))
    return {"key": key, "text": text, "doi": doi, "url": f"https://doi.org/{doi}" if doi else _plain(_field(e, "url"))}


def for_job(payload: dict) -> List[str]:
    """Keys of the references a job used (methods, models and software; not training datasets)."""
    keys = list(ALWAYS) + TASKS.get(payload.get("task", ""), []) + POTENTIALS.get(payload.get("potential", ""), [])
    opts = {**(payload.get("options") or {}), **(payload.get("potential_options") or {})}
    pot = payload.get("potential")
    if pot == "mace":
        fam = str(opts.get("--mace-foundation", "off"))
        keys += {"off": ["Kovacs2025maceoff"], "off24": ["Kovacs2025maceoff"], "mp": ["batatia2023macemp"]}.get(fam, [])
    if str(opts.get("--init-hessian", "")) == "lindh":
        keys.append("Lindh1995")
    if str(opts.get("--coords", "")) == "internal":
        keys += ["Pulay1992", "Baker1993", "Peng1996", "Bakken2002", "Wang2016"]
    seen, out = set(), []
    for k in keys:
        if k in BIB and k not in seen:
            seen.add(k)
            out.append(k)
    return out


MM_POTENTIALS = ("amber", "charmm", "opls", "amoeba", "mm")
MM_NOTE = "Remember to also cite the force field whose parameter files you loaded."


def notes_for_job(payload: dict) -> List[str]:
    """Reminders shown with a job's references."""
    return [MM_NOTE] if payload.get("potential") in MM_POTENTIALS else []


def bibtex(keys: List[str]) -> str:
    return "\n\n".join(BIB[k] for k in keys) + "\n"
