"""
JAX utilities for MARS.

Provides conversion functions, I/O utilities, and unit constants for JAX-based implementation.

Note: ``jax.numpy`` is imported lazily inside the few functions that actually
need it, so that simply importing :mod:`mars.utils` (e.g. for structure I/O in
the JAX-free ``mars viewer``) does not pull in JAX. Structure dictionaries store
plain NumPy arrays; the sampling/optimization code converts them to JAX arrays
where needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

if TYPE_CHECKING:  # type-only; never imports JAX at runtime
    import jax.numpy as jnp

# ============================================================================
# Unit Constants (ASE-compatible)
# ============================================================================

# Energy: eV
# Length: Angstrom
# Time: femtoseconds (fs)
# Force: eV/Angstrom
# Temperature: Kelvin

KB_EV_PER_K = 8.617333262145e-5  # Boltzmann constant in eV/K

# ============================================================================
# Periodic Table Data (Z = 0..118, dummy through Oganesson)
#
# ALL atomic data is defined here.  Other modules import from this file.
# Sources:
#   - Symbols, names, group/period:  IUPAC 2024
#   - Atomic masses:                 IUPAC 2021 (CIAAW); most-stable isotope
#                                    mass for radioactive elements (NIST)
#   - Covalent radii:                Pyykkö & Atsumi, Chem. Eur. J. 15 (2009) 186
#   - vdW radii:                     Bondi (1964) + Mantina et al. JPCA 113 (2009)
#   - Electronegativity:             Pauling scale (CRC Handbook); Allred–Rochow
#   - Ionization energy / EA:        NIST ASD
#   - CPK colors:                    Jmol/CPK convention (RGB triples in 0..255)
# Missing values are encoded as ``np.nan`` for numeric arrays, ``""`` for
# strings, and ``(0, 0, 0)`` for colors.
# ============================================================================

# Element symbols indexed by atomic number.
ELEMENT_SYMBOLS = [
    "X",  #  0  dummy
    "H",
    "He",  #  1-2
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",  #  3-10
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",  # 11-18
    "K",
    "Ca",  # 19-20
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",  # 21-30
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",  # 31-36
    "Rb",
    "Sr",  # 37-38
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",  # 39-48
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",  # 49-54
    "Cs",
    "Ba",  # 55-56
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",  # 57-66
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",  # 67-71
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",  # 72-80
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",  # 81-86
    "Fr",
    "Ra",  # 87-88
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",  # 89-98
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",  # 99-103
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",  # 104-112
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",  # 113-118
]

# Element names indexed by atomic number.
ELEMENT_NAMES = [
    "Dummy",
    "Hydrogen",
    "Helium",
    "Lithium",
    "Beryllium",
    "Boron",
    "Carbon",
    "Nitrogen",
    "Oxygen",
    "Fluorine",
    "Neon",
    "Sodium",
    "Magnesium",
    "Aluminium",
    "Silicon",
    "Phosphorus",
    "Sulfur",
    "Chlorine",
    "Argon",
    "Potassium",
    "Calcium",
    "Scandium",
    "Titanium",
    "Vanadium",
    "Chromium",
    "Manganese",
    "Iron",
    "Cobalt",
    "Nickel",
    "Copper",
    "Zinc",
    "Gallium",
    "Germanium",
    "Arsenic",
    "Selenium",
    "Bromine",
    "Krypton",
    "Rubidium",
    "Strontium",
    "Yttrium",
    "Zirconium",
    "Niobium",
    "Molybdenum",
    "Technetium",
    "Ruthenium",
    "Rhodium",
    "Palladium",
    "Silver",
    "Cadmium",
    "Indium",
    "Tin",
    "Antimony",
    "Tellurium",
    "Iodine",
    "Xenon",
    "Caesium",
    "Barium",
    "Lanthanum",
    "Cerium",
    "Praseodymium",
    "Neodymium",
    "Promethium",
    "Samarium",
    "Europium",
    "Gadolinium",
    "Terbium",
    "Dysprosium",
    "Holmium",
    "Erbium",
    "Thulium",
    "Ytterbium",
    "Lutetium",
    "Hafnium",
    "Tantalum",
    "Tungsten",
    "Rhenium",
    "Osmium",
    "Iridium",
    "Platinum",
    "Gold",
    "Mercury",
    "Thallium",
    "Lead",
    "Bismuth",
    "Polonium",
    "Astatine",
    "Radon",
    "Francium",
    "Radium",
    "Actinium",
    "Thorium",
    "Protactinium",
    "Uranium",
    "Neptunium",
    "Plutonium",
    "Americium",
    "Curium",
    "Berkelium",
    "Californium",
    "Einsteinium",
    "Fermium",
    "Mendelevium",
    "Nobelium",
    "Lawrencium",
    "Rutherfordium",
    "Dubnium",
    "Seaborgium",
    "Bohrium",
    "Hassium",
    "Meitnerium",
    "Darmstadtium",
    "Roentgenium",
    "Copernicium",
    "Nihonium",
    "Flerovium",
    "Moscovium",
    "Livermorium",
    "Tennessine",
    "Oganesson",
]

# Reverse map: symbol → atomic number
SYMBOL_TO_NUMBER: Dict[str, int] = {s: z for z, s in enumerate(ELEMENT_SYMBOLS)}

# Reverse map: atomic number → symbol (same as ELEMENT_SYMBOLS but as dict)
NUMBER_TO_SYMBOL: Dict[int, str] = {z: s for z, s in enumerate(ELEMENT_SYMBOLS)}

# Standard atomic masses in amu, indexed by atomic number.
# Source: IUPAC 2021 recommended values.
ATOMIC_MASSES = np.array(
    [
        # fmt: off
    0.000,     #  0  X  (dummy)
    1.008,     #  1  H
    4.003,     #  2  He
    6.941,     #  3  Li
    9.012,     #  4  Be
   10.81,      #  5  B
   12.011,     #  6  C
   14.007,     #  7  N
   15.999,     #  8  O
   18.998,     #  9  F
   20.180,     # 10  Ne
   22.990,     # 11  Na
   24.305,     # 12  Mg
   26.982,     # 13  Al
   28.086,     # 14  Si
   30.974,     # 15  P
   32.06,      # 16  S
   35.45,      # 17  Cl
   39.948,     # 18  Ar
   39.098,     # 19  K
   40.078,     # 20  Ca
   44.956,     # 21  Sc
   47.867,     # 22  Ti
   50.942,     # 23  V
   51.996,     # 24  Cr
   54.938,     # 25  Mn
   55.845,     # 26  Fe
   58.933,     # 27  Co
   58.693,     # 28  Ni
   63.546,     # 29  Cu
   65.38,      # 30  Zn
   69.723,     # 31  Ga
   72.63,      # 32  Ge
   74.922,     # 33  As
   78.971,     # 34  Se
   79.904,     # 35  Br
   83.798,     # 36  Kr
   85.468,     # 37  Rb
   87.62,      # 38  Sr
   88.906,     # 39  Y
   91.224,     # 40  Zr
   92.906,     # 41  Nb
   95.95,      # 42  Mo
   98.0,       # 43  Tc
  101.07,      # 44  Ru
  102.91,      # 45  Rh
  106.42,      # 46  Pd
  107.87,      # 47  Ag
  112.41,      # 48  Cd
  114.82,      # 49  In
  118.71,      # 50  Sn
  121.76,      # 51  Sb
  127.60,      # 52  Te
  126.90,      # 53  I
  131.29,      # 54  Xe
  132.91,      # 55  Cs
  137.33,      # 56  Ba
  138.91,      # 57  La
  140.12,      # 58  Ce
  140.91,      # 59  Pr
  144.24,      # 60  Nd
  145.0,       # 61  Pm
  150.36,      # 62  Sm
  151.96,      # 63  Eu
  157.25,      # 64  Gd
  158.93,      # 65  Tb
  162.50,      # 66  Dy
  164.93,      # 67  Ho
  167.26,      # 68  Er
  168.93,      # 69  Tm
  173.05,      # 70  Yb
  174.97,      # 71  Lu
  178.49,      # 72  Hf
  180.95,      # 73  Ta
  183.84,      # 74  W
  186.21,      # 75  Re
  190.23,      # 76  Os
  192.22,      # 77  Ir
  195.08,      # 78  Pt
  196.97,      # 79  Au
  200.59,      # 80  Hg
  204.38,      # 81  Tl
  207.2,       # 82  Pb
  208.98,      # 83  Bi
  209.0,       # 84  Po
  210.0,       # 85  At
  222.0,       # 86  Rn
  223.0,       # 87  Fr
  226.0,       # 88  Ra
  227.0,       # 89  Ac
  232.038,     # 90  Th
  231.036,     # 91  Pa
  238.029,     # 92  U
  237.0,       # 93  Np
  244.0,       # 94  Pu
  243.0,       # 95  Am
  247.0,       # 96  Cm
  247.0,       # 97  Bk
  251.0,       # 98  Cf
  252.0,       # 99  Es
  257.0,       # 100 Fm
  258.0,       # 101 Md
  259.0,       # 102 No
  266.0,       # 103 Lr
  267.0,       # 104 Rf
  268.0,       # 105 Db
  269.0,       # 106 Sg
  270.0,       # 107 Bh
  277.0,       # 108 Hs
  278.0,       # 109 Mt
  281.0,       # 110 Ds
  282.0,       # 111 Rg
  285.0,       # 112 Cn
  286.0,       # 113 Nh
  289.0,       # 114 Fl
  290.0,       # 115 Mc
  293.0,       # 116 Lv
  294.0,       # 117 Ts
  294.0,
        # 118 Og
        # fmt: on
    ]
)
# Backwards-compat alias used by external code:
STANDARD_ATOMIC_WEIGHTS = ATOMIC_MASSES

# Covalent radii in Å, indexed by atomic number.
# Source: Pyykkö & Atsumi, Chem. Eur. J. 15 (2009) 186.  DOI: 10.1002/chem.200800987
COVALENT_RADII = np.array(
    [
        # fmt: off
    0.00,  #  0  X
    0.31,  #  1  H
    0.28,  #  2  He
    1.28,  #  3  Li
    0.96,  #  4  Be
    0.84,  #  5  B
    0.76,  #  6  C
    0.71,  #  7  N
    0.66,  #  8  O
    0.57,  #  9  F
    0.58,  # 10  Ne
    1.66,  # 11  Na
    1.41,  # 12  Mg
    1.21,  # 13  Al
    1.11,  # 14  Si
    1.07,  # 15  P
    1.05,  # 16  S
    1.02,  # 17  Cl
    1.06,  # 18  Ar
    2.03,  # 19  K
    1.76,  # 20  Ca
    1.70,  # 21  Sc
    1.60,  # 22  Ti
    1.53,  # 23  V
    1.39,  # 24  Cr
    1.39,  # 25  Mn
    1.32,  # 26  Fe
    1.26,  # 27  Co
    1.24,  # 28  Ni
    1.32,  # 29  Cu
    1.22,  # 30  Zn
    1.22,  # 31  Ga
    1.20,  # 32  Ge
    1.19,  # 33  As
    1.20,  # 34  Se
    1.20,  # 35  Br
    1.16,  # 36  Kr
    2.20,  # 37  Rb
    1.95,  # 38  Sr
    1.90,  # 39  Y
    1.75,  # 40  Zr
    1.64,  # 41  Nb
    1.54,  # 42  Mo
    1.47,  # 43  Tc
    1.46,  # 44  Ru
    1.42,  # 45  Rh
    1.39,  # 46  Pd
    1.45,  # 47  Ag
    1.44,  # 48  Cd
    1.42,  # 49  In
    1.39,  # 50  Sn
    1.39,  # 51  Sb
    1.38,  # 52  Te
    1.39,  # 53  I
    1.40,  # 54  Xe
    2.44,  # 55  Cs
    2.15,  # 56  Ba
    2.07,  # 57  La
    2.04,  # 58  Ce
    2.03,  # 59  Pr
    2.01,  # 60  Nd
    1.99,  # 61  Pm
    1.98,  # 62  Sm
    1.98,  # 63  Eu
    1.96,  # 64  Gd
    1.94,  # 65  Tb
    1.92,  # 66  Dy
    1.92,  # 67  Ho
    1.89,  # 68  Er
    1.90,  # 69  Tm
    1.87,  # 70  Yb
    1.87,  # 71  Lu
    1.75,  # 72  Hf
    1.70,  # 73  Ta
    1.62,  # 74  W
    1.51,  # 75  Re
    1.44,  # 76  Os
    1.41,  # 77  Ir
    1.36,  # 78  Pt
    1.36,  # 79  Au
    1.32,  # 80  Hg
    1.45,  # 81  Tl
    1.46,  # 82  Pb
    1.48,  # 83  Bi
    1.40,  # 84  Po
    1.50,  # 85  At
    1.50,  # 86  Rn
    2.60,  # 87  Fr
    2.21,  # 88  Ra
    2.15,  # 89  Ac
    2.06,  # 90  Th
    2.00,  # 91  Pa
    1.96,  # 92  U
    1.90,  # 93  Np
    1.87,  # 94  Pu
    1.80,  # 95  Am
    1.69,  # 96  Cm
    1.68,  # 97  Bk
    1.68,  # 98  Cf
    1.65,  # 99  Es
    1.67,  # 100 Fm
    1.73,  # 101 Md
    1.76,  # 102 No
    1.61,  # 103 Lr
    1.57,  # 104 Rf
    1.49,  # 105 Db
    1.43,  # 106 Sg
    1.41,  # 107 Bh
    1.34,  # 108 Hs
    1.29,  # 109 Mt
    1.28,  # 110 Ds
    1.21,  # 111 Rg
    1.22,  # 112 Cn
    1.36,  # 113 Nh
    1.43,  # 114 Fl
    1.62,  # 115 Mc
    1.75,  # 116 Lv
    1.65,  # 117 Ts
    1.57,
        # 118 Og
        # fmt: on
    ]
)

# Bondi van der Waals radii in Å, indexed by atomic number.
# Source: Bondi, J. Phys. Chem. 68 (1964) 441; Mantina et al., J. Phys. Chem. A 113 (2009) 5806.
# Unknown elements use a default of 2.0 Å.
_DEFAULT_VDW = 2.00
VDW_RADII = np.array(
    [
        # fmt: off
    0.00,  #  0  X
    1.20,  #  1  H
    1.40,  #  2  He
    1.82,  #  3  Li
    1.53,  #  4  Be
    1.92,  #  5  B
    1.70,  #  6  C
    1.55,  #  7  N
    1.52,  #  8  O
    1.47,  #  9  F
    1.54,  # 10  Ne
    2.27,  # 11  Na
    1.73,  # 12  Mg
    1.84,  # 13  Al
    2.10,  # 14  Si
    1.80,  # 15  P
    1.80,  # 16  S
    1.75,  # 17  Cl
    1.88,  # 18  Ar
    2.75,  # 19  K
    2.31,  # 20  Ca
    2.15,  # 21  Sc
    2.00,  # 22  Ti
    2.00,  # 23  V
    2.00,  # 24  Cr
    2.00,  # 25  Mn
    2.05,  # 26  Fe
    2.00,  # 27  Co
    1.63,  # 28  Ni
    1.40,  # 29  Cu
    1.39,  # 30  Zn
    1.87,  # 31  Ga
    2.11,  # 32  Ge
    1.85,  # 33  As
    1.90,  # 34  Se
    1.85,  # 35  Br
    2.02,  # 36  Kr
    3.03,  # 37  Rb
    2.49,  # 38  Sr
    2.00,  # 39  Y
    2.00,  # 40  Zr
    2.00,  # 41  Nb
    2.00,  # 42  Mo
    2.00,  # 43  Tc
    2.00,  # 44  Ru
    2.00,  # 45  Rh
    1.63,  # 46  Pd
    1.72,  # 47  Ag
    1.58,  # 48  Cd
    1.93,  # 49  In
    2.17,  # 50  Sn
    2.06,  # 51  Sb
    2.06,  # 52  Te
    1.98,  # 53  I
    2.16,  # 54  Xe
    3.43,  # 55  Cs
    2.68,  # 56  Ba
    2.00,  # 57  La
    2.00,  # 58  Ce
    2.00,  # 59  Pr
    2.00,  # 60  Nd
    2.00,  # 61  Pm
    2.00,  # 62  Sm
    2.00,  # 63  Eu
    2.00,  # 64  Gd
    2.00,  # 65  Tb
    2.00,  # 66  Dy
    2.00,  # 67  Ho
    2.00,  # 68  Er
    2.00,  # 69  Tm
    2.00,  # 70  Yb
    2.00,  # 71  Lu
    2.00,  # 72  Hf
    2.00,  # 73  Ta
    2.00,  # 74  W
    2.00,  # 75  Re
    2.00,  # 76  Os
    2.00,  # 77  Ir
    1.75,  # 78  Pt
    1.66,  # 79  Au
    1.55,  # 80  Hg
    1.96,  # 81  Tl
    2.02,  # 82  Pb
    2.07,  # 83  Bi
    1.97,  # 84  Po
    2.02,  # 85  At
    2.20,  # 86  Rn
    3.48,  # 87  Fr
    2.83,  # 88  Ra
    2.00,  # 89  Ac
    2.00,  # 90  Th
    2.00,  # 91  Pa
    1.86,  # 92  U
    2.00,  # 93  Np
    2.00,  # 94  Pu
    2.00,  # 95  Am
    2.00,  # 96  Cm
    2.00,  # 97  Bk
    2.00,  # 98  Cf
    2.00,  # 99  Es
    2.00,  # 100 Fm
    2.00,  # 101 Md
    2.00,  # 102 No
    2.00,  # 103 Lr
    2.00,  # 104 Rf
    2.00,  # 105 Db
    2.00,  # 106 Sg
    2.00,  # 107 Bh
    2.00,  # 108 Hs
    2.00,  # 109 Mt
    2.00,  # 110 Ds
    2.00,  # 111 Rg
    2.00,  # 112 Cn
    2.00,  # 113 Nh
    2.00,  # 114 Fl
    2.00,  # 115 Mc
    2.00,  # 116 Lv
    2.00,  # 117 Ts
    2.00,
        # 118 Og
        # fmt: on
    ]
)


# Pauling electronegativity (NaN where undefined). Indexed by atomic number.
ELECTRONEGATIVITY_PAULING = np.array(
    [
        # fmt: off
    np.nan,                       # 0  X
    2.20, np.nan,                  # 1-2
    0.98, 1.57, 2.04, 2.55, 3.04, 3.44, 3.98, np.nan,  # 3-10
    0.93, 1.31, 1.61, 1.90, 2.19, 2.58, 3.16, np.nan,  # 11-18
    0.82, 1.00,                                          # 19-20
    1.36, 1.54, 1.63, 1.66, 1.55, 1.83, 1.88, 1.91, 1.90, 1.65,  # 21-30
    1.81, 2.01, 2.18, 2.55, 2.96, 3.00,                          # 31-36
    0.82, 0.95,                                                   # 37-38
    1.22, 1.33, 1.60, 2.16, 1.90, 2.20, 2.28, 2.20, 1.93, 1.69,  # 39-48
    1.78, 1.96, 2.05, 2.10, 2.66, 2.60,                          # 49-54
    0.79, 0.89,                                                   # 55-56
    1.10, 1.12, 1.13, 1.14, np.nan, 1.17, np.nan, 1.20,
    np.nan, 1.22,                                                 # 57-66
    1.23, 1.24, 1.25, np.nan, 1.27,                               # 67-71
    1.30, 1.50, 2.36, 1.90, 2.20, 2.20, 2.28, 2.54, 2.00,        # 72-80
    1.62, 2.33, 2.02, 2.00, 2.20, 2.20,                           # 81-86
    0.70, 0.90,                                                   # 87-88
    1.10, 1.30, 1.50, 1.38, 1.36, 1.28, 1.30, 1.30, 1.30, 1.30,  # 89-98
    1.30, 1.30, 1.30, 1.30, 1.30,                                 # 99-103
    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        # fmt: on
    ]
)

# Allred–Rochow electronegativity (NaN where undefined).
ELECTRONEGATIVITY_ALLRED_ROCHOW = np.array(
    [
        # fmt: off
    np.nan,
    2.20, np.nan,
    0.97, 1.47, 2.01, 2.50, 3.07, 3.50, 4.10, np.nan,
    1.01, 1.23, 1.47, 1.74, 2.06, 2.44, 2.83, np.nan,
    0.91, 1.04,
    1.20, 1.32, 1.45, 1.56, 1.60, 1.64, 1.70, 1.75, 1.75, 1.66,
    1.82, 2.02, 2.20, 2.48, 2.74, np.nan,
    0.89, 0.99,
    1.11, 1.22, 1.23, 1.30, 1.36, 1.42, 1.45, 1.35, 1.42, 1.46,
    1.49, 1.72, 1.82, 2.01, 2.21, np.nan,
    0.86, 0.97,
    1.08, 1.08, 1.07, 1.07, 1.07, 1.07, 1.01, 1.11, 1.10, 1.10,
    1.10, 1.11, 1.11, 1.06, 1.14,
    1.23, 1.33, 1.40, 1.46, 1.52, 1.55, 1.44, 1.42, 1.44,
    1.44, 1.55, 1.67, 1.76, 1.90, np.nan,
    0.86, 0.97,
    1.00, 1.11, 1.14, 1.22, 1.22, 1.22, 1.20, 1.20, 1.20, 1.20,
    1.20, 1.20, 1.20, 1.20, 1.20,
    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        # fmt: on
    ]
)

# First ionization energy in eV (NaN where unknown).
IONIZATION_ENERGY_FIRST = np.array(
    [
        # fmt: off
    np.nan,
    13.598, 24.587,
    5.392, 9.323, 8.298, 11.260, 14.534, 13.618, 17.423, 21.565,
    5.139, 7.646, 5.986, 8.152, 10.487, 10.360, 12.968, 15.760,
    4.341, 6.113,
    6.561, 6.828, 6.746, 6.767, 7.434, 7.902, 7.881, 7.640, 7.726, 9.394,
    5.999, 7.900, 9.815, 9.752, 11.814, 14.000,
    4.177, 5.695,
    6.217, 6.634, 6.759, 7.092, 7.119, 7.361, 7.459, 8.337, 7.576, 8.994,
    5.786, 7.344, 8.608, 9.010, 10.451, 12.130,
    3.894, 5.212,
    5.577, 5.539, 5.464, 5.525, 5.554, 5.644, 5.670, 6.150, 5.864, 5.939,
    6.022, 6.108, 6.184, 6.254, 5.426,
    6.825, 7.550, 7.864, 7.834, 8.438, 8.967, 8.959, 9.226, 10.437,
    6.108, 7.417, 7.286, 8.414, np.nan, 10.749,
    4.073, 5.279,
    5.170, 6.080, 5.890, 6.194, 6.266, 6.026, 5.974, 5.991, 6.198, 6.282,
    6.420, 6.500, 6.580, 6.626, 4.964,
    6.013, 6.760, 7.860, 7.700, 7.600, 7.250, 7.000, 6.700, 6.600,
    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        # fmt: on
    ]
)

# Electron affinity in eV (NaN where unknown / negative).
ELECTRON_AFFINITY = np.array(
    [
        # fmt: off
    np.nan,
    0.754, np.nan,
    0.618, np.nan, 0.279, 1.262, np.nan, 1.461, 3.401, np.nan,
    0.548, np.nan, 0.432, 1.385, 0.747, 2.077, 3.613, np.nan,
    0.501, 0.024,
    0.188, 0.079, 0.525, 0.666, np.nan, 0.151, 0.662, 1.156, 1.235, np.nan,
    0.41, 1.232, 0.804, 2.020, 3.364, np.nan,
    0.486, 0.052,
    0.307, 0.426, 0.893, 0.748, 0.55, 1.05, 1.137, 0.562, 1.302, np.nan,
    0.404, 1.112, 1.046, 1.971, 3.059, np.nan,
    0.471, 0.144,
    0.47, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5, 0.5,
    0.014, 0.323, 0.815, 0.15, 1.0778, 1.5644, 2.125, 2.309, np.nan,
    0.377, 0.364, 0.946, 1.9, 2.8, np.nan,
    np.nan, np.nan,
    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
    np.nan, np.nan, np.nan, np.nan, np.nan,
    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        # fmt: on
    ]
)


# Period (1..7), 0 for the dummy element.
def _build_period_array():
    p = np.zeros(119, dtype=int)
    p[1:3] = 1
    p[3:11] = 2
    p[11:19] = 3
    p[19:37] = 4
    p[37:55] = 5
    p[55:87] = 6
    p[87:119] = 7
    return p


PERIODS = _build_period_array()


# IUPAC group number (1..18) where defined; 0 for lanthanides, actinides,
# or the dummy element. The lanthanides (57-71) and actinides (89-103)
# are conventionally excluded from the standard 18 groups.
def _build_group_array():
    g = np.zeros(119, dtype=int)
    # Period 1
    g[1] = 1
    g[2] = 18
    # Period 2 & 3 (s + p block)
    g[3] = 1
    g[4] = 2
    g[5:11] = [13, 14, 15, 16, 17, 18]
    g[11] = 1
    g[12] = 2
    g[13:19] = [13, 14, 15, 16, 17, 18]
    # Periods 4 & 5
    g[19] = 1
    g[20] = 2
    g[21:31] = list(range(3, 13))
    g[31:37] = [13, 14, 15, 16, 17, 18]
    g[37] = 1
    g[38] = 2
    g[39:49] = list(range(3, 13))
    g[49:55] = [13, 14, 15, 16, 17, 18]
    # Period 6 (group 0 reserved for La..Lu lanthanides at 57..71)
    g[55] = 1
    g[56] = 2
    g[57:72] = 0
    g[72:81] = list(range(4, 13))
    g[81:87] = [13, 14, 15, 16, 17, 18]
    # Period 7 (group 0 for Ac..Lr at 89..103)
    g[87] = 1
    g[88] = 2
    g[89:104] = 0
    g[104:113] = list(range(4, 13))
    g[113:119] = [13, 14, 15, 16, 17, 18]
    return g


GROUPS = _build_group_array()


# Number of valence electrons (rough guide, by main-block position).
def _build_valence_array():
    v = np.zeros(119, dtype=int)
    main_groups = {1: 1, 2: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8}
    for z in range(1, 119):
        g = GROUPS[z]
        if g in main_groups:
            v[z] = main_groups[g]
        elif 3 <= g <= 12:
            v[z] = g  # transition metals: ns + (n-1)d
        else:
            v[z] = 0  # f-block: leave as 0 (context-dependent)
    return v


VALENCE_ELECTRONS = _build_valence_array()

# Jmol/CPK colors (RGB triples in 0..255). Used for plotting / rendering.
# Source: Jmol documentation (http://jmol.sourceforge.net/jscolors/).
CPK_COLORS = np.array(
    [
        # fmt: off
    (255, 255, 255),  #  0  X
    (255, 255, 255),  #  1  H
    (217, 255, 255),  #  2  He
    (204, 128, 255),  #  3  Li
    (194, 255,   0),  #  4  Be
    (255, 181, 181),  #  5  B
    ( 80,  80,  80),  #  6  C  (default Jmol grey; black is 0,0,0)
    ( 48,  80, 248),  #  7  N
    (255,  13,  13),  #  8  O
    (144, 224,  80),  #  9  F
    (179, 227, 245),  # 10  Ne
    (171,  92, 242),  # 11  Na
    (138, 255,   0),  # 12  Mg
    (191, 166, 166),  # 13  Al
    (240, 200, 160),  # 14  Si
    (255, 128,   0),  # 15  P
    (255, 255,  48),  # 16  S
    ( 31, 240,  31),  # 17  Cl
    (128, 209, 227),  # 18  Ar
    (143,  64, 212),  # 19  K
    ( 61, 255,   0),  # 20  Ca
    (230, 230, 230),  # 21  Sc
    (191, 194, 199),  # 22  Ti
    (166, 166, 171),  # 23  V
    (138, 153, 199),  # 24  Cr
    (156, 122, 199),  # 25  Mn
    (224, 102,  51),  # 26  Fe
    (240, 144, 160),  # 27  Co
    ( 80, 208,  80),  # 28  Ni
    (200, 128,  51),  # 29  Cu
    (125, 128, 176),  # 30  Zn
    (194, 143, 143),  # 31  Ga
    (102, 143, 143),  # 32  Ge
    (189, 128, 227),  # 33  As
    (255, 161,   0),  # 34  Se
    (166,  41,  41),  # 35  Br
    ( 92, 184, 209),  # 36  Kr
    (112,  46, 176),  # 37  Rb
    (  0, 255,   0),  # 38  Sr
    (148, 255, 255),  # 39  Y
    (148, 224, 224),  # 40  Zr
    (115, 194, 201),  # 41  Nb
    ( 84, 181, 181),  # 42  Mo
    ( 59, 158, 158),  # 43  Tc
    ( 36, 143, 143),  # 44  Ru
    ( 10, 125, 140),  # 45  Rh
    (  0, 105, 133),  # 46  Pd
    (192, 192, 192),  # 47  Ag
    (255, 217, 143),  # 48  Cd
    (166, 117, 115),  # 49  In
    (102, 128, 128),  # 50  Sn
    (158,  99, 181),  # 51  Sb
    (212, 122,   0),  # 52  Te
    (148,   0, 148),  # 53  I
    ( 66, 158, 176),  # 54  Xe
    ( 87,  23, 143),  # 55  Cs
    (  0, 201,   0),  # 56  Ba
    (112, 212, 255),  # 57  La
    (255, 255, 199),  # 58  Ce
    (217, 255, 199),  # 59  Pr
    (199, 255, 199),  # 60  Nd
    (163, 255, 199),  # 61  Pm
    (143, 255, 199),  # 62  Sm
    ( 97, 255, 199),  # 63  Eu
    ( 69, 255, 199),  # 64  Gd
    ( 48, 255, 199),  # 65  Tb
    ( 31, 255, 199),  # 66  Dy
    (  0, 255, 156),  # 67  Ho
    (  0, 230, 117),  # 68  Er
    (  0, 212,  82),  # 69  Tm
    (  0, 191,  56),  # 70  Yb
    (  0, 171,  36),  # 71  Lu
    ( 77, 194, 255),  # 72  Hf
    ( 77, 166, 255),  # 73  Ta
    ( 33, 148, 214),  # 74  W
    ( 38, 125, 171),  # 75  Re
    ( 38, 102, 150),  # 76  Os
    ( 23,  84, 135),  # 77  Ir
    (208, 208, 224),  # 78  Pt
    (255, 209,  35),  # 79  Au
    (184, 184, 208),  # 80  Hg
    (166,  84,  77),  # 81  Tl
    ( 87,  89,  97),  # 82  Pb
    (158,  79, 181),  # 83  Bi
    (171,  92,   0),  # 84  Po
    (117,  79,  69),  # 85  At
    ( 66, 130, 150),  # 86  Rn
    ( 66,   0, 102),  # 87  Fr
    (  0, 125,   0),  # 88  Ra
    (112, 171, 250),  # 89  Ac
    (  0, 186, 255),  # 90  Th
    (  0, 161, 255),  # 91  Pa
    (  0, 143, 255),  # 92  U
    (  0, 128, 255),  # 93  Np
    (  0, 107, 255),  # 94  Pu
    ( 84,  92, 242),  # 95  Am
    (120,  92, 227),  # 96  Cm
    (138,  79, 227),  # 97  Bk
    (161,  54, 212),  # 98  Cf
    (179,  31, 212),  # 99  Es
    (179,  31, 186),  # 100 Fm
    (179,  13, 166),  # 101 Md
    (189,  13, 135),  # 102 No
    (199,   0, 102),  # 103 Lr
    (204,   0,  89),  # 104 Rf
    (209,   0,  79),  # 105 Db
    (217,   0,  69),  # 106 Sg
    (224,   0,  56),  # 107 Bh
    (230,   0,  46),  # 108 Hs
    (235,   0,  38),  # 109 Mt
    (235,   0,  38),  # 110 Ds
    (235,   0,  38),  # 111 Rg
    (235,   0,  38),  # 112 Cn
    (235,   0,  38),  # 113 Nh
    (235,   0,  38),  # 114 Fl
    (235,   0,  38),  # 115 Mc
    (235,   0,  38),  # 116 Lv
    (235,   0,  38),  # 117 Ts
    (235,   0,  38),
        # 118 Og
        # fmt: on
    ],
    dtype=np.uint8,
)


# ============================================================================
# Lookup helpers
# ============================================================================


def _normalize_z(z_or_sym) -> int:
    """Coerce an atomic-number int or element-symbol string to ``Z``.

    Returns 0 for unknown inputs.
    """
    if isinstance(z_or_sym, str):
        return SYMBOL_TO_NUMBER.get(z_or_sym, 0)
    try:
        return int(z_or_sym)
    except (TypeError, ValueError):
        return 0


def get_atomic_masses(numbers: jnp.ndarray) -> jnp.ndarray:
    """Look up per-atom masses from atomic numbers.

    Args:
        numbers: (n_atoms,) array of atomic numbers

    Returns:
        masses: (n_atoms,) array of atomic masses in amu
    """
    import jax.numpy as jnp

    numbers_np = np.asarray(numbers, dtype=int)
    masses_np = ATOMIC_MASSES[np.clip(numbers_np, 0, len(ATOMIC_MASSES) - 1)]
    return jnp.array(masses_np)


def get_atomic_mass(z_or_sym) -> float:
    """Return the atomic mass (amu) for an atomic number or element symbol."""
    z = _normalize_z(z_or_sym)
    if 0 <= z < len(ATOMIC_MASSES):
        return float(ATOMIC_MASSES[z])
    return 0.0


def center_of_mass(positions, numbers) -> np.ndarray:
    """Return the mass-weighted center of mass.

    Args:
        positions: (n_atoms, 3) atomic positions.
        numbers: (n_atoms,) atomic numbers (used to look up masses).

    Returns:
        (3,) numpy array — the center of mass. Falls back to the unweighted
        centroid if the total mass is zero (e.g. all-dummy atoms).
    """
    positions = np.asarray(positions, dtype=float)
    masses = np.asarray(get_atomic_masses(numbers), dtype=float)
    total = float(masses.sum())
    if total <= 0.0:
        return positions.mean(axis=0)
    return (masses[:, None] * positions).sum(axis=0) / total


def get_vdw_radius(z_or_sym) -> float:
    """Return the vdW radius (Å) for an atomic number or element symbol.

    Falls back to 2.0 Å for elements outside the table.
    """
    z = _normalize_z(z_or_sym)
    if 0 < z < len(VDW_RADII):
        return float(VDW_RADII[z])
    return _DEFAULT_VDW


def get_covalent_radius(z_or_sym) -> float:
    """Return the covalent radius (Å) for an atomic number or element symbol."""
    z = _normalize_z(z_or_sym)
    if 0 < z < len(COVALENT_RADII):
        return float(COVALENT_RADII[z])
    return 0.0


def get_electronegativity(z_or_sym, scale: str = "pauling") -> float:
    """Return the electronegativity on the requested scale (NaN if undefined).

    Args:
        z_or_sym: Atomic number or element symbol.
        scale: ``"pauling"`` (default) or ``"allred"``/``"allred-rochow"``.
    """
    z = _normalize_z(z_or_sym)
    s = scale.lower().replace("-", "").replace("_", "")
    if s in ("pauling",):
        table = ELECTRONEGATIVITY_PAULING
    elif s in ("allred", "allredrochow"):
        table = ELECTRONEGATIVITY_ALLRED_ROCHOW
    else:
        raise ValueError(f"Unknown electronegativity scale '{scale}'.")
    if 0 <= z < len(table):
        return float(table[z])
    return float("nan")


def get_ionization_energy(z_or_sym) -> float:
    """First ionization energy in eV (NaN if unknown)."""
    z = _normalize_z(z_or_sym)
    if 0 <= z < len(IONIZATION_ENERGY_FIRST):
        return float(IONIZATION_ENERGY_FIRST[z])
    return float("nan")


def get_electron_affinity(z_or_sym) -> float:
    """Electron affinity in eV (NaN if unknown / negative)."""
    z = _normalize_z(z_or_sym)
    if 0 <= z < len(ELECTRON_AFFINITY):
        return float(ELECTRON_AFFINITY[z])
    return float("nan")


def get_period(z_or_sym) -> int:
    """Period (1..7) of the element. Returns 0 for the dummy."""
    z = _normalize_z(z_or_sym)
    if 0 <= z < len(PERIODS):
        return int(PERIODS[z])
    return 0


def get_group(z_or_sym) -> int:
    """IUPAC group (1..18) of the element.

    Returns 0 for lanthanides/actinides (which are conventionally taken out
    of the standard 18 groups) and for the dummy element.
    """
    z = _normalize_z(z_or_sym)
    if 0 <= z < len(GROUPS):
        return int(GROUPS[z])
    return 0


def get_valence_electrons(z_or_sym) -> int:
    """Approximate number of valence electrons for the element."""
    z = _normalize_z(z_or_sym)
    if 0 <= z < len(VALENCE_ELECTRONS):
        return int(VALENCE_ELECTRONS[z])
    return 0


def get_cpk_color(z_or_sym, normalized: bool = False):
    """Return the Jmol/CPK color for an element as an ``(R, G, B)`` triple.

    Args:
        z_or_sym: Atomic number or element symbol.
        normalized: If True, return floats in [0, 1]; else uint8 in [0, 255].
    """
    z = _normalize_z(z_or_sym)
    if 0 <= z < len(CPK_COLORS):
        rgb = CPK_COLORS[z]
    else:
        rgb = CPK_COLORS[0]
    if normalized:
        return (float(rgb[0]) / 255.0, float(rgb[1]) / 255.0, float(rgb[2]) / 255.0)
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]))


def symbol_to_number(sym: str) -> int:
    """Convert element symbol to atomic number.  Returns 0 for unknown."""
    return SYMBOL_TO_NUMBER.get(sym, 0)


def number_to_symbol(z: int) -> str:
    """Convert atomic number to element symbol.  Returns 'X' for unknown."""
    if 0 <= z < len(ELEMENT_SYMBOLS):
        return ELEMENT_SYMBOLS[z]
    return "X"


def number_to_name(z_or_sym) -> str:
    """Return the IUPAC element name for *z_or_sym*. Returns 'Dummy' for unknowns."""
    z = _normalize_z(z_or_sym)
    if 0 <= z < len(ELEMENT_NAMES):
        return ELEMENT_NAMES[z]
    return "Dummy"


def symbols_to_numbers(symbols: List[str]) -> np.ndarray:
    """Convert a list of element symbols to an array of atomic numbers."""
    return np.array([SYMBOL_TO_NUMBER.get(s, 0) for s in symbols], dtype=np.int32)


# ============================================================================
# BOND utils
# ============================================================================


class BondTopology(NamedTuple):
    """Initial molecular topology captured as bonded pairs + reference lengths."""

    bonds: List[Tuple[int, int]]  # list of (i, j) atom index pairs
    distances: jnp.ndarray  # reference bond lengths (Å), shape (nbonds,)


def detect_bonds(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    tolerance: float = 1.3,
    cell: Optional[np.ndarray] = None,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Detect covalent bonds based on IUPAC covalent radii.

    Two atoms i and j are considered bonded if their distance is less than
    tolerance * (r_cov[i] + r_cov[j]), where r_cov are covalent radii.

    Fully vectorised — no Python loops over atom pairs.

    Args:
        positions: (n_atoms, 3) atomic positions in Angstrom
        atomic_numbers: (n_atoms,) atomic numbers
        tolerance: Multiplicative tolerance factor (default: 1.3)
            1.3 allows for stretched bonds while avoiding false positives
        cell: Optional (3, 3) lattice matrix (rows = lattice vectors) in
            Angstrom.  When provided, pairwise distances are computed using
            the minimum image convention (MIC) so that bonds across periodic
            boundaries are detected correctly.  Pass ``None`` (default) for
            non-periodic (gas-phase / cluster) calculations.

    Returns:
        bonds: List of (i, j) tuples representing bonded atom pairs (i < j)
        distances: (n_bonds,) array of reference bond distances

    Example:
        >>> positions = np.array([[0, 0, 0], [1.1, 0, 0]])  # Two atoms
        >>> numbers = np.array([6, 6])  # Both carbon
        >>> bonds, dists = detect_bonds(positions, numbers)
        >>> print(bonds)  # [(0, 1)]
        >>> print(dists)  # [1.1]
    """
    # Pure-NumPy implementation: bond detection is a CPU graph operation and
    # carries no autodiff, so it does not require (or import) JAX.
    positions = np.asarray(positions, dtype=float)
    atomic_numbers = np.asarray(atomic_numbers, dtype=int)
    n_atoms = int(positions.shape[0])

    # Covalent radii for all atoms
    radii = np.asarray(COVALENT_RADII)[np.clip(atomic_numbers, 0, len(COVALENT_RADII) - 1)]

    # Vectorised pairwise displacement vectors  (N, N, 3)
    diff = positions[:, None, :] - positions[None, :, :]

    if cell is not None:
        # Minimum image convention in fractional coordinates.
        # cell rows are lattice vectors: cart = frac @ cell
        cell_np = np.asarray(cell, dtype=float)
        inv_cell = np.linalg.inv(cell_np)  # (3, 3)
        frac_diff = diff @ inv_cell  # (N, N, 3)
        frac_diff = frac_diff - np.round(frac_diff)  # MIC wrap
        diff = frac_diff @ cell_np  # back to Cartesian

    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    # Per-pair bond threshold  (N, N)
    thresh_matrix = tolerance * (radii[:, None] + radii[None, :])

    # Extract upper-triangle indices (i < j)
    i_idx, j_idx = np.triu_indices(n_atoms, k=1)
    pair_dists = dist_matrix[i_idx, j_idx]
    pair_thresh = thresh_matrix[i_idx, j_idx]

    bond_mask = pair_dists < pair_thresh
    i_np = i_idx[bond_mask]
    j_np = j_idx[bond_mask]

    bonds = list(zip(i_np.tolist(), j_np.tolist()))
    distances = pair_dists[bond_mask]

    return bonds, distances


# ============================================================================
# Structure Representation
# ============================================================================


def check_charge(structure: Dict, charge: int = 0) -> None:
    """Check whether the molecule's electron count is consistent with a closed-shell system.

    The total number of electrons is ``sum(atomic_numbers) - charge``.  If this
    is odd the molecule would be a radical (unpaired electron), which is not
    supported by most potentials.

    Args:
        structure: Structure dictionary with key ``'numbers'``.
        charge: Total molecular charge (default 0).

    Raises:
        ValueError: If the electron count is odd.
    """
    total_z = int(np.asarray(structure["numbers"]).sum())
    n_electrons = total_z - int(charge)
    if n_electrons % 2 != 0:
        raise ValueError(
            f"Odd number of electrons ({n_electrons}) for charge={charge}. "
            f"Sum of atomic numbers is {total_z}. "
            f"Please specify the correct molecular charge with --charge."
        )


def create_structure(
    positions: jnp.ndarray, symbols: List[str], numbers: Optional[jnp.ndarray] = None
) -> Dict:
    """Create a structure dictionary from positions and atomic symbols.

    Args:
        positions: (N, 3) array of atomic positions in Angstrom
        symbols: List of N element symbols (e.g., ['C', 'H', 'H', ...])
        numbers: Optional (N,) array of atomic numbers. If None, inferred from symbols.

    Returns:
        structure: Dictionary with keys 'positions', 'symbols', 'numbers'
    """
    if numbers is None:
        numbers = np.asarray(symbols_to_numbers(symbols))

    return {"positions": positions, "symbols": symbols, "numbers": numbers}


def structure_to_arrays(structure: Dict) -> Tuple[jnp.ndarray, List[str], jnp.ndarray]:
    """Extract arrays from structure dictionary.

    Args:
        structure: Structure dictionary

    Returns:
        positions: (N, 3) JAX array
        symbols: List of element symbols
        numbers: (N,) JAX array of atomic numbers
    """
    return structure["positions"], structure["symbols"], structure["numbers"]


# ============================================================================
# Conversion Functions (ASE ↔ JAX)
# ============================================================================


def ase_atoms_to_structure(atoms) -> Dict:
    """Convert ASE Atoms object to structure dictionary.

    Args:
        atoms: ASE Atoms object

    Returns:
        structure: Structure dictionary compatible with JAX functions
    """
    return create_structure(
        positions=np.asarray(atoms.get_positions()),
        symbols=list(atoms.get_chemical_symbols()),
        numbers=np.asarray(atoms.get_atomic_numbers()),
    )


def structure_to_ase_atoms(structure: Dict):
    """Convert structure dictionary to ASE Atoms object.

    Note: Requires ASE to be installed.

    Args:
        structure: Structure dictionary

    Returns:
        atoms: ASE Atoms object
    """
    from ase import Atoms

    positions_np = np.array(structure["positions"])
    return Atoms(symbols=structure["symbols"], positions=positions_np)


# ============================================================================
# XYZ File I/O (Pure NumPy)
# ============================================================================


def load_structure(filename: str) -> Dict:
    """Load structure from XYZ file.

    Uses ASE for reading if available, otherwise implements simple XYZ parser.

    Args:
        filename: Path to XYZ file

    Returns:
        structure: Structure dictionary
    """
    try:
        # Try using ASE (more robust)
        from ase import io

        atoms = io.read(filename)
        return ase_atoms_to_structure(atoms)
    except ImportError:
        # Fallback to simple XYZ parser
        return _load_xyz_simple(filename)


def _load_xyz_simple(filename: str) -> Dict:
    """Simple XYZ file parser (no ASE dependency).

    Format:
        N
        comment line
        Symbol x y z
        Symbol x y z
        ...
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    n_atoms = int(lines[0].strip())
    # Skip comment line (line 1)

    symbols = []
    positions = []

    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        symbols.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return create_structure(positions=np.asarray(positions, dtype=float), symbols=symbols)


def save_structure(filename: str, structure: Dict, comment: str = ""):
    """Save structure to XYZ file.

    Args:
        filename: Path to output XYZ file
        structure: Structure dictionary
        comment: Optional comment line (default: empty)
    """
    positions = np.array(structure["positions"])
    symbols = structure["symbols"]
    n_atoms = len(symbols)

    with open(filename, "w") as f:
        f.write(f"{n_atoms}\n")
        f.write(f"{comment}\n")
        for sym, pos in zip(symbols, positions):
            f.write(f"{sym} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}\n")


def save_ensemble(filename: str, ensemble: List[Tuple[Dict, float]]):
    """Save ensemble of structures to multi-frame XYZ file.

    Args:
        filename: Path to output XYZ file
        ensemble: List of (structure, energy) tuples
    """
    with open(filename, "w") as f:
        for i, (structure, energy) in enumerate(ensemble):
            positions = np.array(structure["positions"])
            symbols = structure["symbols"]
            n_atoms = len(symbols)

            f.write(f"{n_atoms}\n")
            from .auto_config import EV_TO_KCALMOL

            f.write(f"Structure {i+1}, Energy = {energy * EV_TO_KCALMOL:.6f} kcal/mol\n")
            for sym, pos in zip(symbols, positions):
                f.write(f"{sym} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}\n")


def save_trajectory_xyz(filename: str, trajectory: List, symbols: List[str]):
    """Save raw MD trajectory to multi-frame XYZ file.

    Args:
        filename: Path to output XYZ file
        trajectory: List of (n_atoms, 3) position arrays
        symbols: List of element symbols
    """
    n_atoms = len(symbols)
    with open(filename, "w") as f:
        for i, positions in enumerate(trajectory):
            pos = np.array(positions)
            f.write(f"{n_atoms}\n")
            f.write(f"Frame {i+1}\n")
            for sym, p in zip(symbols, pos):
                f.write(f"{sym} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")


def load_ensemble(filename: str) -> List[Tuple[Dict, float]]:
    """Load ensemble from multi-frame XYZ file.

    Args:
        filename: Path to XYZ file

    Returns:
        ensemble: List of (structure, energy) tuples
    """
    try:
        # Try using ASE (more robust)
        from ase import io

        atoms_list = io.read(filename, index=":")
        ensemble = []
        for atoms in atoms_list:
            structure = ase_atoms_to_structure(atoms)
            energy = atoms.info.get("energy", 0.0)
            ensemble.append((structure, energy))
        return ensemble
    except ImportError:
        # Fallback to simple parser
        return _load_ensemble_simple(filename)


def _load_ensemble_simple(filename: str) -> List[Tuple[Dict, float]]:
    """Simple multi-frame XYZ parser (no ASE dependency)."""
    ensemble = []

    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue

        n_atoms = int(lines[i].strip())
        comment = lines[i + 1].strip()

        # Try to extract energy from comment
        energy = 0.0
        if "Energy" in comment or "energy" in comment:
            try:
                energy = float(comment.split("=")[-1].split()[0])
            except:
                pass

        symbols = []
        positions = []
        for j in range(n_atoms):
            parts = lines[i + 2 + j].split()
            symbols.append(parts[0])
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

        structure = create_structure(positions=np.asarray(positions, dtype=float), symbols=symbols)
        ensemble.append((structure, energy))

        i += 2 + n_atoms

    return ensemble


# ============================================================================
# Array Utilities
# ============================================================================


def ensure_jax_array(array) -> jnp.ndarray:
    """Convert numpy array or list to JAX array if needed.

    Args:
        array: NumPy array, list, or JAX array

    Returns:
        jax_array: JAX array
    """
    import jax.numpy as jnp

    if isinstance(array, jnp.ndarray):
        return array
    return jnp.array(array)


def to_numpy(jax_array: jnp.ndarray) -> np.ndarray:
    """Convert JAX array to NumPy array.

    Args:
        jax_array: JAX array

    Returns:
        numpy_array: NumPy array
    """
    return np.array(jax_array)


# ============================================================================
# Precision Control
# ============================================================================


def enable_float64():
    """Enable 64-bit floating point precision in JAX.

    By default, JAX uses 32-bit floats. Call this at module initialization
    for better numerical precision (matches ASE/NumPy default).
    """
    import jax

    jax.config.update("jax_enable_x64", True)


# ============================================================================
# GPU Utilities
# ============================================================================


def get_device_info() -> Dict:
    """Get information about available JAX devices.

    Returns:
        info: Dictionary with device information
    """
    import jax

    devices = jax.devices()

    info = {
        "devices": devices,
        "default_device": jax.devices()[0],
        "device_count": len(devices),
        "has_gpu": any(d.platform == "gpu" for d in devices),
    }

    return info


def to_device(array: jnp.ndarray, device_type: str = "gpu") -> jnp.ndarray:
    """Move array to specified device.

    Args:
        array: JAX array
        device_type: 'gpu' or 'cpu'

    Returns:
        array_on_device: Array on specified device
    """
    import jax

    devices = [d for d in jax.devices() if d.platform == device_type]
    if devices:
        return jax.device_put(array, devices[0])
    return array


def _is_jax_oom(exc: Exception) -> bool:
    """Detect JAX/XLA out-of-memory errors."""
    msg = str(exc).upper()
    return "RESOURCE_EXHAUSTED" in msg or "OUT OF MEMORY" in msg or isinstance(exc, MemoryError)
